from .string_methods import contains_symbol as _contains_symbol


class _Unset:
    """Sentinel for "argument not supplied", distinct from any real value."""

    def __repr__(self):
        return "<unset>"


_UNSET = _Unset()


class ShiftInvertError(RuntimeError):
    """
    Raised when a shift-invert solve returns an eigenpair that fails its
    residual check.

    Drivers that are able to fall back to a full spectral solve should catch
    this exception; it is deliberately *not* a silent failure, because an
    unverified shift-invert result can look converged while being wrong.

    Attributes
    ----------
    sigma : complex or None
        The rejected eigenvalue.
    residual : float or None
        The relative residual that triggered the rejection.
    """

    def __init__(self, message, sigma=None, residual=None):
        super().__init__(message)
        self.sigma = sigma
        self.residual = residual


def _rel_residual(A, B, σ, v):
    """
    Relative residual of the eigenpair (σ, v) for the pencil (A, B):

        ‖Av - σBv‖ / (‖Av‖ + |σ|‖Bv‖)

    B may be None, meaning the identity.  Returns np.inf if the normalisation
    vanishes, so that a degenerate result never passes a tolerance test.
    """
    import numpy as np

    Av = A @ v
    Bv = v if B is None else B @ v

    denom = np.linalg.norm(Av) + abs(σ) * np.linalg.norm(Bv)
    if not np.isfinite(denom) or denom == 0.0:
        return np.inf

    return np.linalg.norm(Av - σ * Bv) / denom


class Solver:
    """
    Assemble and solve the (generalized) eigenvalue problem defined by a
    System on a Grid.

        M₁ v = σ M₂ v

    M₁ is built from the right-hand sides of the system's equations and M₂
    from their left-hand sides.  When M₂ is the identity -- which is the case
    when no boundary conditions are imposed and the eigenvalue appears alone
    on each left-hand side -- the problem reduces to a standard EVP and is
    solved as such.

    Parameters
    ----------
    grid : Grid
        The grid the problem is discretized on.
    system : System
        Holds the linearized equations, the background state and the
        parameters.
    do_gen_evp : bool (default False)
        Force the generalized formulation even when a standard EVP would do.

    Main entry points
    -----------------
    solve                   Full dense solve, sorted, one mode returned.
    solve_full              Full dense solve, unsorted, no side effects.
    solve_mode              Shift-invert solve for a single mode near a guess.
    iterate_solver          Resolution sweep for a single mode.
    iterate_solve_multimode Resolution sweep tracking several modes.
    """

    def __init__(self, grid, system, do_gen_evp=False):
        import numpy as np

        # Grid object
        self.grid = grid

        # System object with linearized equations, parameters and equilibrium.
        self.system = system

        # do_gen_evp, if True, do the full generalized evp even though
        # an evp might be sufficient (default False)
        self.do_gen_evp = do_gen_evp

        # Check that variable names are unique, i.e., that variables
        # are not a substring of another variable
        msg = """eigenmode variable names are not allowed to be substrings
                 of other eigenmode variables names"""
        for var1 in system.variables:
            tmp = np.sum([var.find(var1) for var in system.variables])
            assert tmp == 1 - system.dim, msg

        # Code below ensures backwards compatibility with old way of simply setting
        # True/False in boundary flag.
        if not hasattr(system, 'extra_binfo'):
            extra_binfo = []
            for boundary in system.boundaries:
                if boundary:
                    extra_binfo.append(['Dirichlet', 'Dirichlet'])
                else:
                    extra_binfo.append([None, None])
            system.extra_binfo = extra_binfo


        # Boundary conditions address the first and last grid node. On a
        # periodic grid those are interior points that happen to sit at the
        # ends of the array, so imposing conditions there is meaningless.
        # This used to be accepted silently and quietly deleted a grid point,
        # yielding eigenvalues for a different, meaningless discretization.
        if getattr(grid, 'periodic', False) and any(system.boundaries):
            names = [var for var, has_bc
                     in zip(system.variables, system.boundaries) if has_bc]
            raise ValueError(
                "Boundary conditions were set on {} but {} is periodic, so "
                "its first and last nodes are not domain boundaries. Either "
                "drop the boundary conditions or use a non-periodic grid "
                "such as ChebyshevExtremaGrid."
                .format(names, type(grid).__name__)
            )

        # Check if we need to solve a generalized evp
        self.check_if_evp_or_gevp(verbose=False)


    def check_if_evp_or_gevp(self, verbose=False):
        """
        This function determines whether we need to solve a generalized
        evp, or whether we can make do with a standard evp
        """
        if not self.do_gen_evp:
            # In the current implementation, we always have to solve
            # the generalized evp unless all boundary conditions are Dirichlet
            # or not set
            for info in self.system.extra_binfo:
                for bound in info:
                    if bound is not None and bound != 'Dirichlet':
                        self.do_gen_evp = True
                        if verbose:
                            print('solve generalized evp due to binfo')
                        return

        if not self.do_gen_evp:
            # Boundaries are not all True, and not all False
            if not all(self.system.boundaries) and any(self.system.boundaries):
                self.do_gen_evp = True
                if verbose:
                    print('solve generalized evp due to system boundaries')

        if not self.do_gen_evp:
            # If mat2 is not the identity matrix, then we have to solve a generalized evp
            from scipy import sparse
            self.get_matrix1()
            self.get_matrix2()
            mat2_is_identity = (self.mat2 - sparse.eye(self.mat1.shape[0])).count_nonzero() == 0
            if not mat2_is_identity:
                self.do_gen_evp = True
                if verbose:
                    print('solve generalized evp due to non-identity in mat2')
                diag = (self.mat2 - sparse.diags(self.mat2.diagonal())).count_nonzero() == 0
                single_val = self.mat2.diagonal().max() == self.mat2.diagonal().min()
                if diag and single_val:
                    msg = """Psecas will solve a generalized EVP but it appears that rewriting the
                    LHS of your equations could reduce the calculation to a standard EVP."""
                    print(msg)
        return


    def solve_full(self):
        """
        Construct matrices and solve the full EVP/GEVP.

        Returns
        -------
        E : np.ndarray
            All eigenvalues (unsorted).
        V : np.ndarray
            Eigenvectors as columns (unsorted, aligned with E).

        Notes
        -----
        This function intentionally performs *no* sorting/filtering and has no
        side-effects (does not call keep_result and does not write self.E/self.v).
        """
        from scipy.linalg import eig

        self.get_matrix1()

        # Solve a generalized EVP
        if self.do_gen_evp:
            self.get_matrix2()
            Σ, V = eig(self.mat1.toarray(), self.mat2.toarray())
        # Solve a standard EVP
        else:
            Σ, V = eig(self.mat1.toarray())

        return Σ, V


    def solve_mode(self, guess, v0=None, useOPinv=True, verbose=False,
                   refine=True, residual_tol=1e-6):
        """
        Find the single eigenpair of the (generalized) eigenvalue problem

            M₁ v = σ M₂ v

        whose eigenvalue lies closest to ``guess``, using shift-invert
        iteration.

        Here σ is the eigenvalue and v is the eigenmode.  When no boundary
        conditions are set M₂ is the identity and the problem reduces to

            M₁ v = σ v

        Returns
        -------
        σ : complex
            The eigenvalue closest to ``guess``.
        v : ndarray
            The corresponding eigenvector.

        Parameters
        ----------
        guess : complex
            Shift for the shift-invert iteration.  The eigenvalue nearest to
            this value is returned.

        v0 : ndarray or None
            Optional starting vector for the Arnoldi iteration.

        useOPinv : bool (default True)
            Standard EVP only: if True, factorize (M₁ - σI) explicitly instead
            of letting ``eigs`` do it.  Ignored for the generalized EVP, which
            always uses an explicitly constructed shift-invert operator (see
            the implementation notes below).

        verbose : bool (default False)
            Print the eigenvalue and its relative residual.

        refine : bool (default True)
            Refine the returned eigenvector with fixed-σ inverse iteration.

        residual_tol : float or None (default 1e-6)
            Reject the result if the relative residual

                ‖M₁v - σM₂v‖ / (‖M₁v‖ + |σ|‖M₂v‖)

            exceeds this value, by raising :class:`ShiftInvertError`.  Pass
            None to skip the check.  Callers that can fall back to a full
            solve should catch that exception rather than trusting an
            unverified eigenvalue.

        Raises
        ------
        ShiftInvertError
            If the residual check fails.  The offending eigenvalue and
            residual are attached to the exception.
        """
        import numpy as np
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigs, splu, LinearOperator

        def refine_eigenvector(A, σ, v, B=None, nsteps=10, rtol=1e-3):
            """
            Fixed-sigma inverse iteration refinement.

            generalized: (A - σ B) w = B v
            standard:    (A - σ I) w = v   (i.e., B=I)

            Factors (A - σ B) once since sigma is fixed.
            """
            n = A.shape[0]
            if B is None:
                B = sp.eye(n, format="csc", dtype=A.dtype)

            lu = splu((A - σ * B).tocsc())

            ε = _rel_residual(A, B, σ, v)
            for _ in range(nsteps):
                w  = lu.solve(B @ v)
                w /= np.linalg.norm(w)
                ε_new = _rel_residual(A, B, σ, w)
                # Keep the refined vector only while it actually improves the
                # residual; inverse iteration can stagnate or drift once the
                # attainable accuracy has been reached.
                if ε_new >= ε:
                    break
                v, ε = w, ε_new
                if ε < rtol:
                    break

            return v, ε

        sigma0 = complex(guess)

        self.get_matrix1()
        A = self.mat1.tocsc()
        if self.do_gen_evp:
            self.get_matrix2()
            B = self.mat2.tocsc()
        else:
            B = None

        if self.do_gen_evp:
            # Shift-invert for the matrix pencil (A, B).
            #
            # scipy's eigs(A, M=B, sigma=...) interface requires B to be
            # symmetric positive definite.  Psecas' M₂ never satisfies this
            # once boundary conditions are present: get_matrix2() zeroes whole
            # rows so that the boundary equations do not depend on the
            # eigenvalue, which makes B singular, and physical terms on the
            # left-hand side can make it indefinite or non-symmetric.  Handing
            # such a B to ARPACK invalidates its convergence test: it returns
            # after essentially one iteration with an "eigenvalue" that merely
            # tracks the shift, so a driver that measures convergence by
            # comparing successive eigenvalues sees false convergence.
            #
            # We therefore construct the shift-invert operator ourselves and
            # pass it to eigs() in *regular* mode, where no assumption about B
            # is made:
            #
            #     OP = (A - σ₀B)⁻¹B ,   OP v = 1/(σ - σ₀) v
            #
            # The eigenvalue of OP with largest magnitude is the one belonging
            # to the σ closest to σ₀, and σ = σ₀ + 1/θ recovers it.
            n  = A.shape[0]
            lu = splu((A - sigma0 * B).tocsc())

            OP = LinearOperator((n, n), matvec=lambda x: lu.solve(B @ x),
                                dtype=np.complex128)

            Θ, V = eigs(OP, k=1, v0=v0, which='LM')
            Σ = sigma0 + 1.0 / Θ

        else:
            if useOPinv:
                n = A.shape[0]

                K = (A - sigma0 * sp.eye(n, format="csc", dtype=A.dtype))
                lu = splu(K)

                OPinv = LinearOperator((n, n), matvec=lu.solve, dtype=A.dtype)

                Σ, V = eigs(A, sigma=sigma0, v0=v0, k=1, OPinv=OPinv)
            else:
                Σ, V = eigs(A, sigma=sigma0, v0=v0, k=1)

        if refine:
            for m in range(Σ.size):
                v, r = refine_eigenvector(A, Σ[m], V[:, m], B=B)
                V[:, m] = v

        σ = Σ[0]
        v = V[:, 0]

        # Verify the result.  Shift-invert can fail silently -- this is the
        # only thing standing between a bad factorization and a wrong
        # scientific result, so it is checked by default.
        residual = _rel_residual(A, B, σ, v)
        self.residual = residual

        if verbose:
            print("N: {}, sigma: {}, residual: {:.3e}".format(
                self.grid.N, σ, residual))

        if residual_tol is not None and not (residual <= residual_tol):
            raise ShiftInvertError(
                "solve_mode(): shift-invert did not converge. Relative "
                "residual {:.3e} exceeds residual_tol={:.3e} for the "
                "eigenvalue {} found near the guess {}. The result has been "
                "rejected rather than returned; retry with a better guess, "
                "relax residual_tol, or use solve_full()."
                .format(residual, residual_tol, σ, sigma0),
                sigma=σ, residual=residual,
            )

        return σ, v


    def filter_modes(self, Σ, V, *, re_range=None, im_range=None, require_re_positive=True):
        """
        Filter eigenvalues/eigenvectors using simple, explicit criteria.

        Parameters
        ----------
        Σ : array-like (complex)
            Eigenvalues.
        V : array-like
            Eigenvectors as columns aligned with E. May be None.
        re_range : (re_min, re_max) or None
            Optional bounds for the real part. Use None for an open bound.
        im_range : (im_min, im_max) or None
            Optional bounds for the imaginary part. Use None for an open bound.
        require_re_positive : bool
            If True, require Re(E) > 0 regardless of re_range.

        Returns
        -------
        Σ_f : np.ndarray
            Filtered eigenvalues.
        V_f : np.ndarray or None
            Filtered eigenvectors (columns), or None if V was None.
        """
        import numpy as np

        Σ = np.asarray(Σ)
        if Σ.ndim != 1:
            Σ = Σ.reshape(-1)

        # Always reject NaN/Inf eigenvalues to keep matching/ranking stable.
        mask = np.isfinite(Σ.real) & np.isfinite(Σ.imag)

        if require_re_positive:
            mask &= (Σ.real > 0)

        if re_range is not None:
            re_min, re_max = re_range
            if re_min is not None:
                mask &= (Σ.real >= re_min)
            if re_max is not None:
                mask &= (Σ.real <= re_max)

        if im_range is not None:
            im_min, im_max = im_range
            if im_min is not None:
                mask &= (Σ.imag >= im_min)
            if im_max is not None:
                mask &= (Σ.imag <= im_max)

        Σ_f = Σ[mask]

        if Σ_f.size == 0:
            # No modes survived filtering: signal a hard failure to the caller
            Σ_f = None
            V_f = None
            raise ValueError(
                "filter_modes(): no eigenmodes left after filtering; "
                "relax filtering criteria or check problem setup."
            )

        if V is None:
            V_f = None
        else:
            V = np.asarray(V)
            # Expect eigenvectors as columns: shape (ndof, neigs)
            # If V is 1D (single eigenvector), treat as one column.
            if V.ndim == 1:
                V = V.reshape(-1, 1)
            V_f = V[:, mask]

        return Σ_f, V_f


    def eigenvector_to_fields(self, vec, grid):
        """
        Unpack eigenvector into dict(var -> field profile on grid nodes).
        Returns profiles with length grid.NN.
        """
        import numpy as np

        NN = grid.NN

        trimmed = all(self.system.boundaries) and (not self.do_gen_evp)

        fields = {}
        for j, var in enumerate(self.system.variables):
            if trimmed:
                n_int = NN - 2
                interior = vec[j*n_int:(j+1)*n_int]
                f = np.hstack([0.0, interior, 0.0])
            else:
                f = vec[j*NN:(j+1)*NN]
            fields[var] = f

        return fields


    def fields_to_eigenvector(self, fields, grid):
        """
        Pack dict(var -> field profile on grid nodes) into eigenvector format
        appropriate for the solver on this grid.
        """
        import numpy as np

        NN = grid.NN

        trimmed = all(self.system.boundaries) and (not self.do_gen_evp)

        chunks = []
        for var in self.system.variables:
            f = np.asarray(fields[var])
            if f.shape[0] != NN:
                raise ValueError(f"{var}: expected length {NN}, got {f.shape[0]}")

            if trimmed:
                # drop boundary nodes; solver expects interior only
                chunks.append(f[1:-1])
            else:
                chunks.append(f)

        return np.concatenate(chunks)


    def prolongate_eigenvector(self, V_old, grid_old):
        """
        Interpolate an eigenvector from a coarser grid onto self.grid, to be
        used as a starting guess for a solve at the new resolution.

        Unpacks the vector into per-variable profiles, interpolates each one
        onto the new nodes, and repacks it in the new grid's layout.
        """
        import numpy as np

        # 1) unpack old vector into old-grid profiles (length grid_old.NN)
        fields_old = self.eigenvector_to_fields(V_old, grid_old)

        # 2) interpolate each variable profile onto the new nodes.
        #
        # Interpolate at *every* new node, not the interior ones only. The
        # previous version interpolated at self.grid.zg[1:-1] and filled the
        # two ends with the old grid's end values, which is right only when
        # those values are zero (all-Dirichlet) or when the end nodes sit at
        # fixed physical positions. On a periodic or infinite grid the node
        # positions move with N, and the ends came out wrong - for a Fourier
        # grid refined 32 -> 64 the interior was accurate to 7e-16 while both
        # endpoints were off by a factor of two.
        #
        # Query points are clipped into the old grid's range because the
        # extent of an infinite grid (rational Chebyshev, sinc, Hermite,
        # Laguerre) grows with N, so the new outermost nodes can lie beyond
        # anything the old grid can interpolate.
        z_new = np.asarray(self.grid.zg)
        z_query = np.clip(z_new.real, grid_old.zmin, grid_old.zmax)

        fields_new = {}
        for var, f_old in fields_old.items():
            f_old = np.asarray(f_old)
            if np.iscomplexobj(f_old):
                # Interpolators are real-valued; do the parts separately.
                f_new = (grid_old.interpolate(z_query, f_old.real)
                         + 1j * grid_old.interpolate(z_query, f_old.imag))
            else:
                f_new = grid_old.interpolate(z_query, f_old)
            fields_new[var] = np.asarray(f_new).reshape(-1)

        # 3) repack into a vector consistent with the *new* solver packing
        V_new = self.fields_to_eigenvector(fields_new, self.grid)

        return V_new


    def iterate_solve_multimode(self, Ns, maxmode=None, allmodes=False,
                       rtol=1e-6, atol=1e-14, gtol=1e-2,
                       orderby='tolerance', metric="complex",
                       re_range=None, im_range=None, require_re_positive=True,
                       useOPinv=True, useEVguess=True, verbose=False,
                       residual_tol=1e-6):
        """
        Iteratively solve the eigenvalue problem over a sequence of
        increasing grid resolutions using a multimode, hybrid strategy.

        At the lowest resolution, a full-spectrum solve is performed.
        At higher resolutions, the solver dynamically switches between
        full solves and single-mode shift-invert solves, depending on
        the estimated convergence error of each tracked mode.

        Eigenmodes are filtered using explicit physical and numerical
        criteria, tracked across resolutions using a configurable
        error metric, and iterated until convergence.

        Parameters
        ----------
        Ns : sequence of int
            Grid resolutions to iterate over, in increasing order.

        maxmode : int or None
            Index of the mode to be returned after ordering and filtering.
            If None, the dominant mode according to the ordering criterion
            is selected.

        allmodes : bool
            If False (default), return only a single eigenmode selected
            by maxmode (or the dominant mode if maxmode is None).

            If True, return all eigenmodes from index 0 up to and including
            maxmode, after ordering and filtering. If maxmode is None,
            all surviving eigenmodes are returned.

        rtol : float
            Relative tolerance for eigenvalue convergence between
            successive grid resolutions.

        atol : float
            Absolute tolerance used in convergence tests and as a lower
            bound for relative error normalization.

        gtol : float
            Relative error threshold controlling the solver strategy.
            The relative error is computed using a denominator given by
            max(atol, |σ|), where σ is the eigenvalue magnitude.
            If the estimated relative error is below gtol, the solver
            may switch from a full-spectrum solve to a single-mode
            shift-invert solve using the previous eigenvalue as a guess.

        orderby : str
            Criterion used to order eigenmodes after filtering.

        metric : str
            Error metric used to compare eigenvalues across resolutions,
            e.g. comparison of real parts or full complex values.

        re_range : (float, float) or None
            Optional bounds on the real part of the eigenvalues used
            for filtering. Use None for open bounds.

        im_range : (float, float) or None
            Optional bounds on the imaginary part of the eigenvalues
            used for filtering. Use None for open bounds.

        require_re_positive : bool
            If True (the default), discard every eigenvalue with
            Re(σ) <= 0, i.e. track growing modes only. Set to False to
            follow damped or purely oscillatory modes as well; note that
            'tolerance' ordering is then the meaningful choice, since the
            growth-rate weighting used for the strategy switch assumes a
            positive real part.

        useOPinv : bool
            If True, use an explicit shift-invert operator when performing
            single-mode solves.

        useEVguess : bool
            If True, pass an initial eigenvector guess to solve_mode(),
            interpolated from a lower-resolution grid.

        verbose : bool
            If True, print detailed information about solver progress,
            convergence status, and strategy switching.

        residual_tol : float or None
            Relative-residual tolerance applied to every single-mode
            shift-invert solve. A mode that fails the check is discarded
            and the whole spectrum is recomputed with a full solve for
            that resolution. Pass None to disable the check (not
            recommended).

        Returns
        -------
        sigma : complex
            The converged eigenvalue of the selected mode.

        v : ndarray
            The corresponding eigenvector.

        error : float
            Final convergence error estimate for the selected mode.
        """
        import numpy as np
        import copy

        def _print_modes(Σ, N, errors=None, case=None, delta=None, error=None):
            n = Σ.size
            fmt = f" {n:2d}" if n < 100 else ">99"
            print(f"N: {N:4d}, {fmt} eigenvalue{'s' if n > 1 else ' '}: ", end='')
            m = min(3, n)
            if case is None:
                for i in range(m):
                    print(f" {Σ[i]:.4e}", end='')
                print(" ..." if n > m else '', ' '*10)
            else:
                if errors is None:
                    for i in range(m):
                        print(f" {Σ[i]:.4e}", end='')
                else:
                    for i in range(m):
                        print(f" {Σ[i]:.4e} ({errors[i]:.2e})", end='')
                print(" ..." if n > m else '', end='')
                if delta is not None:
                    print(f", Δσ/σ : {delta:.2e}", end='')
                if error is not None:
                    print(f", error : {error:.2e}", end='')
                print(f"{case}", ' '*10)

        def _errors(Σ_new, Σ_old, rtol=1e-5, atol=1e-10, metric='complex', orderby='tolerance'):
            errors = []
            deltas = []
            if metric == 'real':
                for i in range(Σ_new.size):
                    ΔΣ  = np.abs(Σ_old.real - Σ_new[i].real)
                    j   = np.argsort(ΔΣ)[0]
                    err = ΔΣ[j] / (atol + rtol * max(np.abs(Σ_new[i].real), np.abs(Σ_old[j].real)))
                    errors.append(err)
            elif metric == 'imag':
                for i in range(Σ_new.size):
                    ΔΣ  = np.abs(Σ_old.imag - Σ_new[i].imag)
                    j   = np.argsort(ΔΣ)[0]
                    err = ΔΣ[j] / (atol + rtol * max(np.abs(Σ_new[i].imag), np.abs(Σ_old[j].imag)))
                    errors.append(err)
            else:
                for i in range(Σ_new.size):
                    ΔΣ  = np.abs(Σ_old - Σ_new[i])
                    j   = np.argsort(ΔΣ)[0]
                    err = ΔΣ[j] / (atol + rtol * max(np.abs(Σ_new[i]), np.abs(Σ_old[j])))
                    errors.append(err)
            for i in range(Σ_new.size):
                ΔΣ  = np.abs(Σ_old - Σ_new[i])
                j   = np.argsort(ΔΣ)[0]
                fac = 1.0 + np.abs(Σ_new[i].imag) / max(atol, Σ_new[i].real)
                dlt = ΔΣ[j] / max(atol, np.abs(Σ_new[i])) * fac
                deltas.append(dlt)
            deltas = np.array(deltas)
            errors = np.array(errors)
            if orderby in ['amplitude', 'magnitude']:
                index = np.argsort(np.abs(Σ_new))[::-1]
            elif orderby in ['real_part', 'real']:
                index = np.argsort(Σ_new.real)[::-1]
            elif orderby in ['imag_part', 'imag', 'imaginary']:
                index = np.argsort(Σ_new.imag)[::-1]
            else:
                index = np.argsort(errors)
            return errors[index], deltas[index], index

        # Helper: choose selected index and how many modes to return
        def _select(Nmodes, maxmode, allmodes):
            if Nmodes <= 0:
                return 0, 0
            if maxmode is None:
                # Default: select mode 0; if allmodes, return all
                m = 0
                k = Nmodes if allmodes else 1
                return m, k

            # maxmode is an index; clamp to [0, Nmodes-1]
            m = max(0, min(int(maxmode), Nmodes - 1))

            # if allmodes: return 0..sel inclusive -> k = sel+1
            k = (m + 1) if allmodes else 1
            return m, k


        Ns = list(Ns)
        if len(Ns) < 1:
            raise ValueError(
                "iterate_solve_multimode() needs at least one resolution in "
                "Ns, got an empty sequence."
            )

        self.grid.N = Ns[0]
        Σ, V = self.solve_full()
        Σ_old, V_old = self.filter_modes(
            Σ, V, re_range=re_range, im_range=im_range,
            require_re_positive=require_re_positive)
        grid_old = copy.deepcopy(self.grid)
        if verbose:
            if orderby in ['real_part', 'real']:
                index = np.argsort(Σ_old.real)[::-1]
            elif orderby in ['imag_part', 'imag', 'imaginary']:
                index = np.argsort(Σ_old.imag)[::-1]
            else:
                index = np.argsort(np.abs(Σ_old))[::-1]
            _print_modes(Σ_old[index], self.grid.N)

        mode, modes = _select(Σ_old.size, maxmode, allmodes)

        error = np.inf
        delta = np.inf
        # With a single resolution there is nothing to compare against, so no
        # error can be estimated. Initialising here rather than inside the
        # loop keeps the return statements below well defined; leaving it
        # unbound raised UnboundLocalError for len(Ns) == 1.
        errors = np.full(Σ_old.size, np.inf)

        for N in Ns[1:]:
            self.grid.N = N
            if delta > gtol:
                case = ''
                Σ, V = self.solve_full()
            else:
                case = ' [with guess]'
                try:
                    Σ = []
                    V = []
                    for i in range(modes):
                        σ0 = Σ_old[i]
                        if useEVguess:
                            v0 = self.prolongate_eigenvector(V_old[:,i], grid_old)
                        else:
                            v0 = None
                        σ, v = self.solve_mode(σ0, v0=v0, useOPinv=useOPinv,
                                                verbose=verbose,
                                                residual_tol=residual_tol)
                        Σ.append(σ)
                        V.append(v)
                    Σ = np.array(Σ)
                    V = np.array(V).T
                except ShiftInvertError:
                    # A shift-invert solve produced an eigenpair that failed
                    # its residual check. Never accept it: recover the whole
                    # spectrum with a full solve instead. Doing otherwise
                    # would let the driver "converge" on a wrong eigenvalue,
                    # since the rejected result tends to sit near the guess
                    # it was given.
                    case = ' [guess rejected → full]'
                    Σ, V = self.solve_full()

            try:
                Σ_new, V_new = self.filter_modes(
                    Σ, V, re_range=re_range, im_range=im_range,
                    require_re_positive=require_re_positive)
            except ValueError:
                # Fast solver found no eigenmodes in the specified range.
                # Fall back to full solve to recover the spectrum.
                case = ' [guess failed → full]'
                Σ, V = self.solve_full()
                Σ_new, V_new = self.filter_modes(
                    Σ, V, re_range=re_range, im_range=im_range,
                    require_re_positive=require_re_positive)

            errors, deltas, index = _errors(Σ_new, Σ_old, rtol=rtol, atol=atol, metric=metric, orderby=orderby)

            Σ_new = Σ_new[index]
            V_new = V_new[:,index]

            mode, modes = _select(Σ_new.size, maxmode, allmodes)

            error = errors[mode]
            delta = deltas[mode]

            if verbose:
                _print_modes(Σ_new, self.grid.N, errors=errors, case=case, delta=delta, error=error)

            if error <= 1.0:
                self.keep_result(Σ_new[mode], V_new[:,mode], mode)
                self.system.result.update({"converged": True})
                self.system.result.update({"error": error})
                self.system.result.update({"grid": self.grid.zg})
                if allmodes:
                    return Σ_new[:modes], V_new[:, :modes], errors[:modes]
                return Σ_new[mode], V_new[:,mode], errors[mode]

            Σ_old = Σ_new.copy()
            V_old = V_new.copy()
            grid_old = copy.deepcopy(self.grid)

        self.keep_result(Σ_old[mode], V_old[:,mode], mode)
        self.system.result.update({"converged": False})
        self.system.result.update({"error": error})
        self.system.result.update({"grid": self.grid.zg})

        if allmodes:
            return Σ_old[:modes], V_old[:, :modes], errors[:modes]
        return Σ_old[mode], V_old[:,mode], errors[mode]


    def solve(self, useOPinv=_UNSET, verbose=False, mode=0, saveall=False):
        """
        Construct and solve the (generalized) eigenvalue problem (EVP)

            M₁ v = σ M₂ v

        generated with the grid and parameters contained in the system object.

        Here σ is the eigenvalue and v is the eigenmode.
        Note that M₂ is a diagonal matrix if no boundary conditions are set.
        In that case the EVP is simply

            M₁ v = σ v

        This method stores a dictionary with the result of the calculation
        in self.system.result.

        Returns: One eigenvalue and its eigenvector.

        Optional parameters

        useOPinv: ignored, and deprecated. This method always performs a
        full dense solve, where there is no shift-invert operator to build;
        the parameter never had an effect. Use solve_mode() for a
        shift-invert solve, which is where useOPinv is meaningful.

        verbose (default False): print out information about the calculation.

        mode (default 0): mode=0 is the fastest growing, mode=1 the second
        fastest and so on.

        saveall (default False): also store the full sorted spectrum in
        self.E and the corresponding eigenvectors in self.v.
        """
        import warnings
        from scipy.linalg import eig

        if useOPinv is not _UNSET:
            warnings.warn(
                "Solver.solve() ignores useOPinv and always performs a full "
                "dense solve; the parameter has never had an effect. It will "
                "be removed in a future release. Use solve_mode() for a "
                "shift-invert solve.",
                DeprecationWarning, stacklevel=2,
            )

        # Calculate right-hand matrix
        self.get_matrix1()

        # Solve a generalized EVP
        if self.do_gen_evp:
            self.get_matrix2()
            E, V = eig(self.mat1.toarray(), self.mat2.toarray())
        # Solve a standard EVP
        else:
            E, V = eig(self.mat1.toarray())

        # Sort the eigenvalues
        E, index = self.sorting_strategy(E)

        # Choose the eigenvalue mode value only
        sigma = E[index[mode]]
        v = V[:, index[mode]]

        # Save all eigenvalues and eigenvectors here
        if saveall:
            self.E = E[index]
            self.v = V[:, index]
        if verbose:
            print("N: {}, all eigenvalues: {}".format(self.grid.N, sigma))

        self.keep_result(sigma, v, mode)

        return (sigma, v)

    def solve_with_guess(self, guess, useOPinv=True, verbose=False, mode=0,
                         residual_tol=1e-6):
        """
        Construct and solve the (generalized) eigenvalue problem (EVP)

            M₁ v = σ M₂ v

        generated with the grid and parameters contained in the system object.

        Here σ is the eigenvalue and v is the eigenmode.
        Note that M₂ is a diagonal matrix if no boundary conditions are set.
        In that case the EVP is simply

            M₁ v = σ v

        This method stores a dictionary with the result of the calculation
        in self.system.result.

        Returns: One eigenvalue and its eigenvector.

        guess: Scipy's eigs method is used to find a
        single eigenvalue in the proximity of the guess.

        Optional parameters

        useOPinv (default True): If true, manually calculate OPinv instead of
        letting eigs do it.

        verbose (default False): print out information about the calculation.

        mode (default 0): mode=0 is the fastest growing, mode=1 the second
        fastest and so on.

        residual_tol (default 1e-6): relative-residual tolerance for the
        returned eigenpair. Raises ShiftInvertError if exceeded. Pass None
        to disable the check.
        """
        import numpy as np
        from scipy.sparse.linalg import eigs

        # Calculate right-hand matrix
        self.get_matrix1()

        # Solve a generalized EVP
        if self.do_gen_evp:
            self.get_matrix2()
            # Delegate to solve_mode(), which builds the correct shift-invert
            # operator (A - sigma*B)^-1 B for the pencil and verifies the
            # residual of what it returns.
            #
            # The previous implementation here passed OPinv = (A - sigma*B)^-1
            # to eigs() *without* M=self.mat2, so an operator built for the
            # pencil was applied to the standard problem and a different mode
            # came back; and even with M= supplied, eigs() requires M to be
            # positive definite, which mat2 is not once boundary conditions
            # zero out rows. See the implementation notes in solve_mode().
            sigma, v = self.solve_mode(guess, useOPinv=useOPinv,
                                       residual_tol=residual_tol)
        else:
            if useOPinv:
                from numpy.linalg import inv
                OPinv = inv(self.mat1 - guess * np.eye(self.mat1.shape[0]))
                sigma, v = eigs(self.mat1, k=1, sigma=guess, OPinv=OPinv)
            else:
                sigma, v = eigs(self.mat1, k=1, sigma=guess)

            # Convert result from eigs to have same format as result from eig
            sigma = sigma[0]
            v = np.squeeze(v)

        if verbose:
            print("N:{}, only 1 eigenvalue:{}".format(self.grid.N, sigma))

        self.keep_result(sigma, v, mode)

        return (sigma, v)

    def iterate_solver(
        self, Ns, mode=0, tol=1e-6, atol=1e-16, verbose=False, guess_tol=0.01,
        useOPinv=True, residual_tol=1e-6
    ):
        """
        Iteratively call the solve method with increasing grid resolution, N.
        Returns when the relative difference in the eigenvalue is less than
        the tolerance, tol.

        Ns: list of resolutions to try, e.g. Ns = arange(32)*10

        mode: the index in the list of eigenvalues returned from solve

        tol: the target precision of the eigenvalue

        verbose (default False): print out information about the calculation.

        guess_tol: Increasing the resolution will inevitably lead to a more
        expensive computation. A speedup can however be achieved when
        searching for a single eigenvalue. This method can in this
        case use the eigenvalue from the previous calculation as a guess for
        the result of the new calculation. The parameter guess_tol makes sure
        that the guess used is a good guess. If guess_tol=0.1 the method will
        start using guesses when the relative difference to the previous
        iteration is 10 %.

        residual_tol: relative-residual tolerance for guess-based solves. A
        mode that fails the check is discarded and that resolution is redone
        with a full solve. Pass None to disable the check (not recommended).
        """
        import numpy as np

        Ns = list(Ns)
        if len(Ns) < 2:
            raise ValueError(
                "iterate_solver() compares consecutive resolutions and so "
                "needs at least two entries in Ns, got {}. Use solve() for a "
                "single resolution.".format(Ns)
            )

        self.grid.N = Ns[0]
        (sigma_old, v) = self.solve(mode=mode, verbose=verbose)
        self.grid.N = Ns[1]
        (sigma_new, v) = self.solve(mode=mode, verbose=verbose)
        a_err = np.abs(sigma_old - sigma_new)
        r_err = a_err / np.abs(sigma_old)

        for i in range(2, len(Ns)):
            self.grid.N = Ns[i]
            # Not a good guess yet
            if r_err > guess_tol:
                (sigma_new, v) = self.solve(mode=mode, verbose=verbose)
            # Use guess from previous iteration
            else:
                try:
                    (sigma_new, v) = self.solve_with_guess(
                        sigma_old, mode=mode, verbose=verbose,
                        useOPinv=useOPinv, residual_tol=residual_tol
                    )
                except ShiftInvertError:
                    # The cheap guess-based solve produced an eigenpair that
                    # failed its residual check. Fall back to the full solve
                    # rather than accepting it: a rejected shift-invert result
                    # sits near the guess, which would look like convergence.
                    (sigma_new, v) = self.solve(mode=mode, verbose=verbose)

            a_err = np.abs(sigma_old - sigma_new)
            r_err = a_err / np.abs(sigma_old)
            # Converged
            if r_err < tol or a_err < atol:
                self.system.result.update({"converged": True})
                self.system.result.update({"r_err": r_err, "a_err": a_err})
                return (sigma_new, v, r_err)
            # Overwrite old with new
            sigma_old = np.copy(sigma_new)

        self.system.result.update({"converged": False})
        self.system.result.update({"r_err": r_err, "a_err": a_err})
        return (sigma_new, v, r_err)

        # raise RuntimeError("Did not converge!")

    def sorting_strategy(self, E):
        """
        A default sorting strategy.

        "Large" real and imaginary eigenvalues are removed and the eigenvalues
        are sorted from largest to smallest
        """
        import numpy as np

        E[np.abs(E.real) > 10.0] = 0
        E[np.abs(E.imag) > 10.0] = 0
        # Sort from largest to smallest eigenvalue
        index = np.argsort(np.real(E))[::-1]
        return (E, index)

    def keep_result(self, sigma, vec, mode):
        import numpy as np

        # Store result
        if all(self.system.boundaries) and not self.do_gen_evp:
            # The boundary nodes were trimmed out of the matrices, so the
            # eigenvector holds NN - 2 interior values per variable. Put the
            # zeros back to get a profile defined on every grid node.
            interior = self.grid.NN - 2
            self.system.result = {
                var: np.hstack(
                    [0.0, vec[j * interior:(j + 1) * interior], 0.0]
                )
                for j, var in enumerate(self.system.variables)
            }
        else:
            self.system.result = {
                var: vec[j * self.grid.NN : (j + 1) * self.grid.NN]
                for j, var in enumerate(self.system.variables)
            }
        self.system.result.update(
            {self.system.eigenvalue: sigma, "mode": mode}
        )

    def get_matrix1(self, verbose=False):
        """
        Calculate the matrix M₁ neded in the solve method.
        """
        from scipy import sparse
        import numpy as np
        from .string_methods import var_replace

        dim = self.system.dim
        # Index of the last grid node. Grids disagree on whether NN is N or
        # N + 1 (FourierGrid and HermiteGrid use NN = N, everything else
        # N + 1), so all index arithmetic below is expressed through NN.
        last = self.grid.NN - 1
        equations = self.system.equations
        boundaries = self.system.boundaries
        extra_binfo = self.system.extra_binfo

        # Construct all submatrices as sparse matrices
        rows = []
        for j, equation in enumerate(equations):
            equation = equation.split("=")[1]
            mats = self._find_submatrices(equation, verbose)
            rows.append(mats)

        # Modify according to boundary conditions
        for j in range(dim):
            for i in range(dim):
                if all((boundaries)) and not self.do_gen_evp:
                    rows[j][i] = rows[j][i][1:last, 1:last]
                elif any(boundaries):
                    rows[j][i] = self._modify_submatrix(rows[j][i],
                                                        j + 1, i + 1,
                                                        boundaries[j], 
                                                        extra_binfo[j], 
                                                        verbose)

        # Assemble everything
        self.mat1 = sparse.bmat(rows, format='csr')

    def get_matrix2(self, verbose=False):
        """
        Calculate the matrix M₂ neded in the solve method.
        """
        from scipy import sparse
        import numpy as np
        from .string_methods import var_replace

        dim = self.system.dim
        last = self.grid.NN - 1      # see the note in get_matrix1()
        sys = self.system
        equations = sys.equations
        variables = sys.variables
        boundaries = sys.boundaries
        extra_binfo = sys.extra_binfo

        # Evaluate LHS of equation
        rows = []
        for j, equation in enumerate(equations):
            equation = equation.split("=")[0]
            equation = var_replace(equation, sys.eigenvalue, "1.0")
            mats = self._find_submatrices(equation, verbose)
            rows.append(mats)

        # Modify according to boundary conditions
        for j in range(dim):
            for i in range(dim):
                if all((boundaries)) and not self.do_gen_evp:
                    rows[j][i] = rows[j][i][1:last, 1:last]
                elif any(boundaries):
                    # In generalized EVP mode, boundary conditions are imposed by
                    # row-replacement in mat1 (A). To keep BC equations independent
                    # of the eigenvalue, we must zero the corresponding rows in
                    # mat2 (B), i.e. enforce: (BC row) -> 0 = lambda * 0.
                    #
                    # We zero entire boundary rows across *all* block columns i.
                    # This is stronger and correct; the previous implementation
                    # only zeroed a single diagonal entry, which can leave
                    # eigenvalue-coupled residual terms in BC rows.
                    if self.do_gen_evp and boundaries[j]:
                        # Keep index convention consistent with
                        # _modify_submatrix(): boundary nodes are the first
                        # and last grid nodes, 0 and NN - 1.
                        if extra_binfo[j][0] is not None:
                            rows[j][i][0, :] = 0
                        if extra_binfo[j][1] is not None:
                            rows[j][i][last, :] = 0
                    else:
                        # Backward-compatible behavior for non-generalized EVP:
                        # preserve existing "diagonal-entry zeroing" logic.
                        if extra_binfo[j][0] is not None:
                            rows[j][i][0, 0] = 0
                        if extra_binfo[j][1] is not None:
                            rows[j][i][last, last] = 0

        # Assemble everything
        self.mat2 = sparse.bmat(rows, format='csr')


    def _rewrite_derivatives(self, expr, grid, var,
                              d0_repl="grid.D(0).T",
                              d1_repl="grid.D(1).T",
                              d2_repl="grid.D(2).T",
                              dn_repl=None,
                              z_repl="grid.zg"):
        """
        Rewrite derivative syntax in an equation string.

        This helper currently preserves the existing behavior:
          d{z}(var)           -> d1_repl
          d{z}(d{z}(var))     -> d2_repl
          d{z}(var, n)        -> dn_repl(n)  (defaults to grid.D(n).T)
          var                 -> d0_repl
          {z}                 -> z_repl

        It is introduced as a refactoring hook; subsequent commits extend it
        to support higher-order derivatives and dz(var, n) syntax.
        """
        import re
        from .string_methods import var_replace

        der = "d" + grid.z + "("

        # Handle explicit-order derivative: d{z}(var,n)
        # Strict syntax: no whitespace is permitted.
        if dn_repl is None:
            dn_repl = lambda n: "grid.D({}).T".format(n)

        der_func = "d" + grid.z  # e.g. "dz"
        pat = r"{func}\({var},(\d+)\)".format(
            func=re.escape(der_func), var=re.escape(var)
        )
        expr = re.sub(pat, lambda m: dn_repl(int(m.group(1))), expr)

        expr = expr.replace(der + der + var + "))", d2_repl)
        expr = expr.replace(der + var + ")", d1_repl)
        expr = var_replace(expr, var, d0_repl)
        if z_repl is not None:
            expr = var_replace(expr, grid.z, z_repl)
        return expr


    def _expand_substitutions(self, expr, verbose=False):
        """
        Apply the system's textual substitutions to expr, repeatedly, until
        the text stops changing.

        Substitutions are plain text replacement (see System.add_substitution)
        and may legitimately be defined in terms of one another, so a single
        pass is not always enough. A pass limit guards against a substitution
        that refers to itself.
        """
        from .string_methods import var_replace

        substitutions = getattr(self.system, "substitutions", None)
        if not substitutions:
            return expr

        max_passes = len(substitutions) + 1
        for _ in range(max_passes):
            previous = expr
            for substitution in substitutions:
                name, _, value = substitution.partition("=")
                expr = var_replace(expr, name.strip(), value)
                if verbose:
                    print(expr)
            if expr == previous:
                return expr

        raise ValueError(
            "Could not expand the substitutions in\n\n{}\n\n"
            "after {} passes. This usually means a substitution is defined "
            "in terms of itself, directly or through another substitution."
            .format(expr, max_passes)
        )


    def _find_submatrices(self, eq, verbose=False):
        import builtins
        import numpy as np
        from scipy import sparse
        from .string_methods import var_replace

        grid = self.system.grid

        env = dict(self.system.__dict__)
        env["grid"] = grid

        NN = self.grid.NN
        mats = []

        if verbose:
            print("\nParsing equation:", eq)

        # Expand substitutions ONCE, before anything inspects the equation.
        #
        # This must happen before the per-variable fast path below: a variable
        # can enter an equation only through a substitution (for example
        # "G = dz(dz(f))" used in "sigma*f = q*G"), in which case the raw
        # equation text does not mention it. Testing the unexpanded text meant
        # such a variable was replaced by a zero block, silently producing a
        # wrong eigenvalue with no error or warning.
        #
        # Expanding once rather than once per variable is also cheaper.
        eq = self._expand_substitutions(eq, verbose)

        for i, var in enumerate(self.system.variables):
            # Fast path: variable absent -> sparse zero (no dense zeros)
            if not _contains_symbol(eq, var):
                mats.append(sparse.lil_matrix((NN, NN), dtype=np.complex128))
                continue

            variables_t = list(np.copy(self.system.variables))
            eq_t = self._rewrite_derivatives(eq, grid, var)

            variables_t.remove(var)
            for var2 in variables_t:
                eq_t = self._rewrite_derivatives(
                    eq_t, grid, var2,
                    d0_repl="0.0",
                    d1_repl="0.0",
                    d2_repl="0.0",
                    dn_repl=lambda n: "0.0",
                    z_repl=None,
                )

            if verbose:
                print("\nEvaluating expression:", eq_t)

            try:
                err_msg1 = (
                    "During the parsing of:\n\n{}\n\n"
                    "Psecas tried to evaluate\n\n{}\n\n"
                    "while attempting to evaluate the terms with: {}"
                    "\nThis caused the following error to occur:\n\n"
                )
                # Evaluate the expression in a restricted environment.
                submat = eval(eq_t, {"__builtins__": {"__import__": builtins.__import__}}, env)

            except NameError as e:
                strerror, = e.args
                err_msg2 = (
                    "\n\nThis is likely because the missing variable has"
                    "\nnot been defined in your systems class or its\n"
                    "make_background method."
                )
                raise NameError(err_msg1.format(eq, eq_t, var) + strerror + err_msg2)
            except Exception as e:
                raise Exception(err_msg1.format(eq, eq_t, var) + str(e))

            # Transpose (works for both dense and sparse)
            submat = submat.T

            # Keep sparse as sparse; only densify if truly dense
            if sparse.issparse(submat):
                # Enforce dtype without copying if possible, then LIL for later row edits
                if submat.dtype != np.complex128:
                    submat = submat.astype(np.complex128, copy=False)
                # Optional: enforce shape early (helps catch subtle eval/template issues)
                if submat.shape != (NN, NN):
                    raise ValueError(f"Submatrix has shape {submat.shape}, expected {(NN, NN)}")
                mats.append(submat.tolil())
            else:
                # Dense path (only when eval produced dense)
                submat = np.asarray(submat, dtype=np.complex128)
                if submat.shape != (NN, NN):
                    raise ValueError(f"Submatrix has shape {submat.shape}, expected {(NN, NN)}")

                # Convert dense -> sparse LIL
                mats.append(sparse.lil_matrix(submat))

        return mats


    def _modify_submatrix(self, submat, eq_n, var_n, boundary, binfo, verbose=False):
        """
        This modifies the submatrix to incorporate boundary conditions.
    
        Dirichlet is value set to zero at boundary.
        Neumann is derivative set to zero at boundary.

        Finally, one can set a string such as

        'r**2*dr(dr(Aphi)) + r*dr(Aphi) - Aphi = 0'

        The Boundary condition on a variable cannot depend on the other independent variables.
        """
        import builtins
        import numpy as np
        from .string_methods import var_replace

        grid = self.system.grid

        env = dict(self.system.__dict__)
        env["grid"] = grid

        last = self.grid.NN - 1      # see the note in get_matrix1()
        if boundary:
            for index, bound in zip([0, last], binfo):
                if bound is not None:
                    submat[index, :] = 0
                    if eq_n == var_n:
                        if bound == 'Dirichlet':
                            submat[index, index] = 1
                        elif bound == 'Neumann':
                            submat[index, :] = grid.D(1)[index, :]
                        else:
                            # A custom boundary expression, e.g.
                            # 'r**2*dr(dr(Aphi)) + r*dr(Aphi) - Aphi = 0'.
                            #
                            # These checks used bare asserts, which python -O
                            # strips - turning a validation failure into
                            # silent miscomputation - and parsed the RHS with
                            # int(), so the natural spelling 'dz(f) = 0.0'
                            # died with "invalid literal for int() with base
                            # 10: ' 0.0'" instead of the intended message.
                            if '=' not in bound:
                                raise ValueError(
                                    "The boundary condition\n\n  {}\n\nhas no "
                                    "equal sign. Write it as an expression "
                                    "equal to zero, e.g. 'dz(f) = 0', or use "
                                    "the keywords 'Dirichlet' or 'Neumann'."
                                    .format(bound)
                                )

                            rhs = bound.split("=", 1)[1].strip()
                            try:
                                rhs_is_zero = float(rhs) == 0.0
                            except ValueError:
                                rhs_is_zero = False
                            if not rhs_is_zero:
                                raise ValueError(
                                    "The right-hand side of a boundary "
                                    "expression must be zero, but\n\n  {}\n\n"
                                    "has '{}'. Move every term to the left, "
                                    "e.g. write 'dz(f) - g = 0' rather than "
                                    "'dz(f) = g'.".format(bound, rhs)
                                )

                            var = self.system.variables[var_n-1]
                            bound_t = bound.split("=")[0]

                            # Apply equation substitutions (same expansion
                            # rules as for the equations themselves)
                            bound_t = self._expand_substitutions(bound_t)

                            mask = np.zeros(self.grid.NN)
                            mask[index] = 1
                            env["mask"] = mask
                            bound_t = self._rewrite_derivatives(
                                bound_t, grid, var,
                                d0_repl="mask",
                                d1_repl="grid.D(1)[{}, :]".format(index),
                                d2_repl="grid.D(2)[{}, :]".format(index),
                                dn_repl=lambda n: "grid.D({})[{}, :]".format(n, index),
                                z_repl="grid.zg[{}]".format(index),
                            )
                            if verbose:
                                print("\nEvaluating expression:", bound_t)
                            try:
                                err_msg1 = (
                                    "During the parsing of:\n\n{}\n\n"
                                    "Psecas tried to evaluate\n\n{}\n\n"
                                    "while attempting to evaluate the boundary on: {}"
                                    "\nThis caused the following error to occur:\n\n"
                                )
                                # Evaluate the expression in a restricted environment.
                                submat[index, :] = eval(bound_t, {"__builtins__": {"__import__": builtins.__import__}}, env)

                            except NameError as e:
                                strerror, = e.args
                                err_msg2 = (
                                    "\n\nThis is likely because the missing variable has"
                                    "\nnot been defined in your systems class or its\n"
                                    "make_background method."
                                )
                                raise NameError(
                                    err_msg1.format(bound, bound_t, var) + strerror + err_msg2
                                )
                            except Exception as e:
                                raise Exception(err_msg1.format(bound, bound_t, var) + str(e))

        return submat
