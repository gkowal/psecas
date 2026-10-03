from .string_methods import contains_symbol as _contains_symbol
from . import dense_eig
from numpy.linalg import inv
from scipy import sparse
from scipy.linalg import eig, lu_factor, lu_solve
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigs, splu, LinearOperator
import builtins
import copy
import numpy as np
import re
import scipy.sparse as sp
import warnings


def _make_eval_globals():
    """
    The global namespace that equation and boundary strings are evaluated in.

    This is a convenience namespace, NOT a security boundary.

    The previous version passed {"__builtins__": {"__import__": ...}} under a
    comment describing it as "a restricted environment". It was not: exposing
    __import__ alone is enough to reach anything at all, as

        eval("__import__('os').system(...)", that_namespace)

    demonstrates. It was also actively harmful. Removing the rest of the
    builtins breaks any expression that emits a warning, because the warning
    machinery imports through the evaluating frame's builtins:

        ChebyshevExtremaGrid(500, zmin=0, zmax=1, z='r')
        eval("-2/grid.zg*grid.D(1).T", ...)
        -> zg contains 0 -> RuntimeWarning -> KeyError: '__import__'

    which is how tests/test_bessel_solutions.py fails. So the pretence is
    dropped rather than patched up. Equations come from whoever runs the
    code; eval on input you do not trust cannot be made safe by trimming
    builtins, and pretending otherwise is worse than not trying, because it
    invites someone to feed it untrusted input.

    What this namespace is for is convenience: making numpy available under
    the names people expect when writing an equation, so that sqrt(), tanh()
    and np.where() work in an equation string or a boundary expression
    instead of raising NameError.
    """


    names = {
        "__builtins__": builtins,
        "np": np,
        "numpy": np,
        "pi": np.pi,
        "e": np.e,
        "inf": np.inf,
    }

    # Elementwise functions people reasonably expect to be able to use in an
    # equation or a boundary expression.
    for name in (
        "sqrt", "exp", "log", "log10", "abs",
        "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2",
        "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",
        "sign", "real", "imag", "conj", "where", "heaviside",
        "minimum", "maximum", "hstack", "zeros", "ones", "eye", "diag",
    ):
        names[name] = getattr(np, name)

    return names


_EVAL_GLOBALS = _make_eval_globals()


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

    Av = A @ v
    Bv = v if B is None else B @ v

    denom = np.linalg.norm(Av) + abs(σ) * np.linalg.norm(Bv)
    if not np.isfinite(denom) or denom == 0.0:
        return np.inf

    return np.linalg.norm(Av - σ * Bv) / denom


def _rayleigh_floor(A, B, σ, x, y):
    """
    Rounding bound of the two-sided Rayleigh quotient yᴴAx / yᴴBx
    evaluated in double precision at the eigenvalue σ:

        eps · |y|ᵀ(|A| + |σ||B|)|x| / |yᴴBx|,

    with |·| elementwise; B may be None, meaning the identity, for which
    the second term is |σ|·|y|ᵀ|x|. The polish in Solver.solve() takes a
    move of the eigenvalue within it as no evidence of an improvement.
    """
    ax, ay = np.abs(x), np.abs(y)
    Bx = x if B is None else B @ x
    num = ay @ (np.abs(A) @ ax)
    num += abs(σ) * (ay @ ax if B is None else ay @ (np.abs(B) @ ax))
    return np.finfo(float).eps * num / abs(np.vdot(y, Bx))


# Reasons for which a polish refusal has already been warned about.
_polish_warned = set()


def _warn_polish(reason):
    # Keyed on the message without its parenthesized details, so that a
    # resolution sweep warns once per kind of failure, not once per solve
    # (as dense_eig._warn_fallback does).
    key = reason.split(" (")[0]
    if key not in _polish_warned:
        _polish_warned.add(key)
        warnings.warn("Solver.solve() kept the unpolished eigenpair: "
                      + reason, RuntimeWarning, stacklevel=4)


def _eig_shift_invert(A, B, shift, residual_tol):
    """
    All eigenpairs of the dense pencil (A, B) through the equivalent standard
    problem

        C w = θ w,    C = (A - sB)⁻¹ B,    σ = s + 1/θ,

    which has the same eigenvectors.  scipy's QZ (zggev) is serial and several
    times slower than the standard solver (zgeev) that runs on C, which is
    also threaded by BLAS.

    θ = 0 belongs to an infinite eigenvalue (B singular, e.g. through zeroed
    boundary rows).  Rounding leaves such θ at about eps‖C‖ rather than at
    zero, so everything below n·eps‖C‖ is reported as infinite, which is how
    QZ reports them too.

    zgeev balances C by diagonal scaling before reducing it, and QZ does not.
    When the entries of B span many decades -- a background that decays like
    exp(-z²/2) on a semi-infinite grid does this -- the balanced problem can
    return correct eigenvalues with meaningless eigenvectors.  Every finite
    eigenpair is therefore checked against the original pencil through its
    normwise backward error.

    Returns (Σ, V), or None if A - sB is singular or any finite eigenpair
    has a backward error above residual_tol.  The caller is expected to
    fall back to QZ in that case.
    """

    n = A.shape[0]
    lu, piv = lu_factor(A - shift * B, check_finite=False)
    if np.any(np.diag(lu) == 0):
        return None

    C = lu_solve((lu, piv), B, check_finite=False)
    tiny = n * np.finfo(float).eps * np.linalg.norm(C, 1)
    Θ, V = eig(C, overwrite_a=True, check_finite=False)

    finite = np.abs(Θ) > tiny
    Σ = np.full(n, np.inf, dtype=complex)
    Σ[finite] = shift + 1 / Θ[finite]

    # Normwise backward error of every finite eigenpair,
    #
    #     ‖Av - σBv‖ / ((‖A‖ + |σ|‖B‖) ‖v‖),
    #
    # which QZ keeps near machine precision.  _rel_residual() normalises by
    # ‖Av‖ + |σ|‖Bv‖ instead, which suits one mode near a guess but not a
    # whole spectrum: QZ's own spurious modes reach O(1) by that measure.
    Vf = V[:, finite]
    R = A @ Vf - (B @ Vf) * Σ[finite]
    scale = (np.linalg.norm(A) + np.abs(Σ[finite]) * np.linalg.norm(B)) \
        * np.linalg.norm(Vf, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        backward_errors = np.linalg.norm(R, axis=0) / scale
    if not np.all(backward_errors <= residual_tol):
        return None

    return Σ, V


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
    gevp_method : {'qz', 'shift-invert'} (default 'qz')
        How the full dense solves (solve, solve_full) treat a generalized
        EVP.  'qz' calls scipy's QZ algorithm directly.  'shift-invert'
        solves the equivalent standard EVP for (M₁ - sM₂)⁻¹M₂ instead,
        which is several times faster and, unlike QZ, uses every BLAS
        thread, so it suits a single solve on an otherwise idle machine.
        Its eigenpairs are checked against the original pencil and the
        solve falls back to QZ, with a RuntimeWarning, if any fails.  A
        standard EVP is unaffected.
    gevp_shift : complex (default 0.1·exp(0.7i))
        The shift s used by gevp_method='shift-invert'.  Eigenvalues near s
        are the most accurate, so it should lie in the region of interest
        but not on an eigenvalue; the default avoids the real and imaginary
        axes, where marginal and purely growing modes sit.
    backend : {"auto", "cupy", "scipy"} (default "auto")
        Dense eigensolver used by solve() and solve_full(). "auto" uses
        CuPy on the GPU when it is available and the matrix has at least
        ``gpu_min_size`` rows, and SciPy otherwise; the environment
        variable PSECAS_EIG_BACKEND overrides it. See psecas.dense_eig.
        Stored as ``self.backend`` and may be changed afterwards.

    Main entry points
    -----------------
    solve                   Full dense solve, sorted, one mode returned.
    solve_full              Full dense solve, unsorted, no side effects.
    solve_mode              Shift-invert solve for a single mode near a guess.
    iterate_solver          Resolution sweep for a single mode.
    iterate_solve_multimode Resolution sweep tracking several modes.
    """

    #: Accepted values of the gevp_method argument.
    GEVP_METHODS = ('qz', 'shift-invert')

    #: Default shift for gevp_method='shift-invert'.
    GEVP_SHIFT = 0.1 * np.exp(0.7j)

    #: Largest normwise backward error a shift-invert eigenpair may have
    #: before the solve falls back to QZ.  QZ stays below about 1e-14 on the
    #: systems in the test suite and shift-invert below about 1e-11 where it
    #: works; where zgeev's balancing breaks it, the error is 1e-3 or more.
    #: It is also the floor of the polish in solve(): a polished pair whose
    #: normwise backward error exceeds it is refused.
    GEVP_RESIDUAL_TOL = 1e-9

    def __init__(self, grid, system, do_gen_evp=False, gevp_method='qz',
                 gevp_shift=None, backend="auto"):

        if gevp_method not in self.GEVP_METHODS:
            raise ValueError(
                "gevp_method must be one of {}, not {!r}"
                .format(self.GEVP_METHODS, gevp_method)
            )
        self.gevp_method = gevp_method
        self.gevp_shift = self.GEVP_SHIFT if gevp_shift is None else gevp_shift

        # Grid object
        self.grid = grid

        # System object with linearized equations, parameters and equilibrium.
        self.system = system

        # do_gen_evp, if True, do the full generalized evp even though
        # an evp might be sufficient (default False)
        self.do_gen_evp = do_gen_evp

        # Dense eigensolver backend. Validated here so a typo fails at
        # construction; the PSECAS_EIG_BACKEND override is applied at solve
        # time, so changing the environment or self.backend later works.
        dense_eig.resolve_backend(backend)
        self.backend = backend
        # Smallest matrix dimension "auto" sends to the GPU.
        self.gpu_min_size = dense_eig.GPU_MIN_SIZE
        # Backend that produced the last dense solve ("cupy" or "scipy").
        self.last_backend = None

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


        # Reject names that would collide with the solver's own before
        # anything tries to parse or store them.
        self._check_reserved_names()

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


    #: Names that cannot be used for a variable or the eigenvalue.
    #:
    #: "mode", "converged", "error", "r_err" and "a_err" are metadata keys
    #: that keep_result() and the iterative drivers write into system.result
    #: alongside the eigenmode profiles, so a variable of that name would
    #: have its profile silently overwritten. "grid" is additionally the name
    #: the Grid object is bound to in the namespace equations are evaluated
    #: in, so a variable called "grid" shadows it and the equation fails to
    #: parse with an unrelated-looking AttributeError.
    RESERVED_NAMES = frozenset(
        {"mode", "converged", "error", "grid", "r_err", "a_err"}
    )

    def _check_reserved_names(self):
        """Reject variable and eigenvalue names that collide with our own."""
        clashes = sorted(self.RESERVED_NAMES.intersection(
            self.system.variables))
        if self.system.eigenvalue in self.RESERVED_NAMES:
            clashes.append(self.system.eigenvalue)

        if clashes:
            raise ValueError(
                "The name(s) {} cannot be used for a variable or for the "
                "eigenvalue. Psecas stores eigenmode profiles in "
                "system.result keyed by variable name, next to the metadata "
                "keys {}, and binds the grid to the name 'grid' when "
                "evaluating equations. Please rename."
                .format(clashes, sorted(self.RESERVED_NAMES))
            )

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

        The eigensolver is chosen by self.backend (see psecas.dense_eig);
        self.last_backend records which one ran. The eigenvalue order is
        backend-specific, and the GPU path returns +inf with a zero
        eigenvector for each infinite eigenvalue of a generalized EVP.
        gevp_method='shift-invert' always runs on SciPy, whatever
        self.backend says.
        """

        self.get_matrix1()
        if self.do_gen_evp:
            self.get_matrix2()

        return self._eig_dense()

    def _eig_dense(self):
        """
        Every eigenpair of the assembled problem, as a full dense solve.

        mat1 (and mat2 for a generalized EVP) must be up to date.  A
        generalized EVP is solved as gevp_method says.
        """

        A = self.mat1.toarray()
        if not self.do_gen_evp:
            Σ, V, self.last_backend = dense_eig.eig(
                A, backend=self.backend, min_size=self.gpu_min_size,
                return_backend=True)
            return Σ, V

        B = self.mat2.toarray()
        if self.gevp_method == 'shift-invert':
            # Shift-invert always runs on SciPy, whatever self.backend says.
            result = _eig_shift_invert(A, B, self.gevp_shift,
                                       self.GEVP_RESIDUAL_TOL)
            if result is not None:
                self.last_backend = "scipy"
                return result
            warnings.warn(
                "The shift-invert dense solve (gevp_shift={}) failed its "
                "residual check or hit a singular shift; falling back to QZ."
                .format(self.gevp_shift),
                RuntimeWarning, stacklevel=3,
            )
            # The QZ fallback stays on SciPy too: the pencils that fail
            # shift-invert are the badly scaled ones.
            Σ, V, self.last_backend = dense_eig.eig(
                A, B, backend="scipy", return_backend=True)
            return Σ, V

        Σ, V, self.last_backend = dense_eig.eig(
            A, B, backend=self.backend, min_size=self.gpu_min_size,
            return_backend=True)
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


    def plot_eigenmodes(self, sigma, errors=None, filename=None, **kwargs):
        """
        Plot the eigenvalues of the current problem in the complex plane.

        A thin wrapper around psecas.plotting.plot_eigenvalues that fills in
        a title from the system's wavenumber and the current resolution. Pass
        filename to save instead of showing; any further keyword arguments go
        through to plot_eigenvalues (xlim, ylim, logx, num, title).
        """
        from .plotting import plot_eigenvalues

        if 'title' not in kwargs:
            kx = getattr(self.system, 'kx', None)
            if kx is None:
                kwargs['title'] = 'Eigenmodes (N={})'.format(self.grid.N)
            else:
                kwargs['title'] = 'Eigenmodes (k={:.2e}, N={})'.format(
                    kx, self.grid.N)

        return plot_eigenvalues(sigma, errors=errors, filename=filename,
                                **kwargs)

    def iterate_solve_multimode(self, Ns, maxmode=None, allmodes=False,
                       rtol=1e-6, atol=1e-14, gtol=1e-2,
                       orderby='tolerance', metric="complex",
                       re_range=None, im_range=None, require_re_positive=True,
                       useOPinv=True, useEVguess=True, verbose=False,
                       residual_tol=1e-6, allgrids=False,
                       plots=False, plot_dir=None):
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

        allgrids : bool
            If False (default), return as soon as the selected mode meets
            the convergence criterion. If True, work through every
            resolution in Ns regardless, and report the result from the
            last one. Useful for convergence studies, and for making the
            cost of a sweep independent of the problem.

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

        def _plot(Σ, errors=None):
            """Save a spectrum snapshot for this resolution."""
            import os

            kx = getattr(self.system, 'kx', 0.0)
            name = 'eigenmodes_k{:.6e}N{:04d}.png'.format(kx, self.grid.N)
            if plot_dir is not None:
                os.makedirs(plot_dir, exist_ok=True)
                name = os.path.join(plot_dir, name)
            self.plot_eigenmodes(Σ, errors=errors, filename=name)

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

        if plots:
            _plot(Σ_old)

        mode, modes = _select(Σ_old.size, maxmode, allmodes)

        error = np.inf
        delta = np.inf
        converged = False
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

            if plots:
                _plot(Σ_new, errors=errors)

            mode, modes = _select(Σ_new.size, maxmode, allmodes)

            error = errors[mode]
            delta = deltas[mode]

            if verbose:
                _print_modes(Σ_new, self.grid.N, errors=errors, case=case, delta=delta, error=error)

            if error <= 1.0:
                converged = True
                if not allgrids:
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
        self.system.result.update({"converged": converged})
        self.system.result.update({"error": error})
        self.system.result.update({"grid": self.grid.zg})

        if allmodes:
            return Σ_old[:modes], V_old[:, :modes], errors[:modes]
        return Σ_old[mode], V_old[:,mode], errors[mode]


    def _polish_pair(self, A, B, σ, v, steps, others=None):
        """
        Refine the eigenpair (σ, v) of the dense pencil (A, B) by ``steps``
        steps of inverse iteration with a two-sided Rayleigh quotient; B is
        None for the identity. ``others`` holds the other finite
        eigenvalues of the spectrum, for the same-mode test.

        Returns the polished σ', or the input σ itself when the move to σ'
        lies within max(POLISH_JITTER_FACTOR·|σ₂ - σ₁|, floor) (the
        no-degrade rule; see solve()), with the vector, of the input v and
        the polished one, that has the smaller relative residual at the
        returned σ. The polished vector is returned with unit 2-norm and
        phased so that v_oldᴴv is real and positive; the input v is
        returned as it is. When a guard refuses the result the input
        (σ, v) comes back unchanged, with a one-time RuntimeWarning; keeping
        σ by the no-degrade rule is not a refusal and does not warn. See
        solve() for the guards.
        """
        from scipy.linalg import LinAlgWarning

        σ_old, v_old = σ, v
        n = A.shape[0]

        def B_dot(x):
            return x if B is None else B @ x

        def BH_dot(x):
            return x if B is None else B.conj().T @ x

        def step(s, x, y):
            """One step from (s, x, y): the new triple, or (None, reason)."""
            M = A - (s * np.eye(n) if B is None else s * B)
            try:
                lu = lu_factor(M, overwrite_a=True, check_finite=False)
            except (np.linalg.LinAlgError, ValueError) as exc:
                return None, ("the LU factorisation of A - σB failed ({})"
                              .format(exc))
            w = lu_solve(lu, B_dot(x), check_finite=False)
            z = lu_solve(lu, BH_dot(x if y is None else y), trans=2,
                         check_finite=False)
            nw, nz = np.linalg.norm(w), np.linalg.norm(z)
            if not (np.isfinite(nw) and np.isfinite(nz) and nw > 0
                    and nz > 0):
                return None, ("the inverse iteration gave a non-finite or "
                              "zero vector")
            x, y = w / nw, z / nz
            Bx = B_dot(x)
            num, den = np.vdot(y, A @ x), np.vdot(y, Bx)
            if not (np.isfinite(num) and np.isfinite(den)) or \
                    abs(den) <= np.finfo(float).eps * np.linalg.norm(Bx):
                return None, ("the Rayleigh quotient is not finite or its "
                              "denominator vanishes")
            s = num / den
            # Finite num and a nonzero den can still overflow.
            if not np.isfinite(s):
                return None, "the Rayleigh quotient overflowed"
            return (s, x, y), None

        s = complex(σ)
        x = v / np.linalg.norm(v)
        y = None
        # The Rayleigh quotient of every step that succeeded.
        quotients = []
        # inf * 0 and the like are expected for the results refused below.
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            # A - sB is nearly singular by construction.
            warnings.simplefilter("ignore", LinAlgWarning)
            for k in range(steps):
                result, reason = step(s, x, y)
                if result is None:
                    if k == 0:
                        _warn_polish(reason)
                        return σ_old, v_old
                    # A later step failing (typically A - sB singular to
                    # working precision) means s has converged: keep the
                    # previous step's pair, which still faces the guards.
                    break
                s, x, y = result
                quotients.append(s)

            # Same mode: the vector must not have turned to another mode...
            c = np.vdot(v_old, x)
            overlap = abs(c) / np.linalg.norm(v_old)
            if not overlap >= self.POLISH_MIN_OVERLAP:
                _warn_polish("the eigenvector overlap fell below "
                             "POLISH_MIN_OVERLAP={} (overlap {:.3e}, "
                             "sigma {})".format(self.POLISH_MIN_OVERLAP,
                                                overlap, σ_old))
                return σ_old, v_old
            # ... nor the eigenvalue moved nearer to another one.
            # Ties within a few ulp are accepted: a real eigenvalue that
            # the dense solve split into a pair a ± iε is otherwise refused
            # for one member and not the other, by rounding.
            eps = np.finfo(float).eps
            if others is not None and len(others) and \
                    np.min(np.abs(np.asarray(others) - s)) * (1 + 4 * eps) \
                    < abs(s - σ_old):
                _warn_polish("the polished eigenvalue is nearer to another "
                             "eigenvalue (sigma {} -> {})".format(σ_old, s))
                return σ_old, v_old

            # No degrade: a move within the rounding floor of the last
            # Rayleigh quotient, or within the step-to-step jitter, is no
            # evidence that σ' is better than σ₀, which is then kept as
            # it came. A success, not a refusal: no warning.
            floor = _rayleigh_floor(A, B, s, x, y)
            jitter = (abs(quotients[-1] - quotients[-2])
                      if len(quotients) >= 2 else 0.0)
            keep = abs(s - σ_old) <= max(self.POLISH_JITTER_FACTOR * jitter,
                                         floor)
            σ_new = σ_old if keep else s

            # Return σ_new with whichever vector has the smaller residual
            # at σ_new: on ill-conditioned pencils the inverse-iteration
            # vector can be less accurate than the dense solver's, although
            # σ' is more accurate than σ.
            if _rel_residual(A, B, σ_new, x) < \
                    _rel_residual(A, B, σ_new, v_old):
                v_new = x * (np.conj(c) / abs(c))
                v_new = v_new / np.linalg.norm(v_new)
            else:
                v_new = v_old

            # Catastrophe floor, not an accuracy certificate. Not applied
            # to the unpolished pair coming back unchanged: the polish did
            # not produce it, and keeping it is no refusal to warn about.
            if keep and v_new is v_old:
                return σ_new, v_new
            be = dense_eig._backward_errors(A, B, np.array([σ_new]),
                                            v_new[:, None])[0]
            if not be <= self.GEVP_RESIDUAL_TOL:
                _warn_polish("the normwise backward error exceeds "
                             "GEVP_RESIDUAL_TOL={} (error {:.1e}, sigma {})"
                             .format(self.GEVP_RESIDUAL_TOL, be, σ_old))
                return σ_old, v_old

        return σ_new, v_new

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
        self.E and the corresponding eigenvectors in self.v. Only the slot
        of the returned mode, self.E[mode] and self.v[:, mode], holds the
        polished pair (see below); every other slot is the unpolished
        result of the dense solve.

        Polish

        At large N the dense solve loses accuracy on ill-conditioned
        pencils (fourth-derivative boundary rows give entries growing like
        N⁸): on the tearing problem at N=512 the dominant eigenvalue is off
        by 2e-7 relative with QZ and 7e-6 with the GPU reduction. The
        selected eigenvalue is therefore refined on the original pencil by
        self.polish_steps steps (default 2) of inverse iteration,

            w = (M₁ - σM₂)⁻¹ M₂ v,   v = w / ‖w‖,

        each followed by a two-sided Rayleigh quotient σ = (yᴴM₁v)/(yᴴM₂v),
        with the left vector y from the same LU factorisation. This brings
        the tearing eigenvalues to about 1e-9 relative or better for O(n²)
        work per step after one O(n³) LU, small next to the eigensolve.
        polish_steps = 0 turns the polish off and returns exactly the
        unpolished pair.

        The polish has an accuracy floor of its own, reached from any
        start: on the wells and on tearing at N=128 it is the rounding of
        the Rayleigh quotient evaluated in double precision, at tearing
        N=512 the error of the LU vectors. A start more accurate than the
        floor would come back degraded (the GPU's eigenvalue of the
        standard well at N=256, 3.2e-14 off, came back 4.3e-13 off). So
        the unpolished eigenvalue σ₀ is
        kept, bit for bit and without a warning, when

            |σ' - σ₀| <= max(self.POLISH_JITTER_FACTOR·|σ₂ - σ₁|, floor),
            floor = eps·|y|ᵀ(|M₁| + |σ'||M₂|)|x| / |yᴴM₂x|,

        with x, y the unit right and left vectors of the last step that
        succeeded, σ' its Rayleigh quotient, σ₁ and σ₂ the quotients of the
        first two steps (of the last two, should polish_steps exceed 2; the
        jitter term is 0 unless two steps succeeded) and |·| elementwise;
        M₂ = I for the standard EVP. Otherwise σ' is
        returned. Either way max(4|σ₂ - σ₁|, floor) estimates the error of
        the polished value. Measured on the study of this rule, the error
        of the returned eigenvalue was at most 1.275·max(4|σ₂ - σ₁|, floor)
        in 1628 calls on 11 pencils (wells, standard and generalized, at
        N=64-512, and tearing at N=128-512, from dense-solve, exact and
        perturbed starts). That bound is empirical, not a guarantee. The
        rule never degraded a start there, at a price: within the
        threshold it trades forgone improvement for never degrading,
        keeping starts that the polish would have improved (236 of the 1628
        calls, by up to 74x: tearing at N=256, a start 1e-10 off kept where
        the polish reached 1.4e-12); the returned error still satisfies the
        bound above. All of this was measured with two successful steps:
        with polish_steps = 1, or when the second step fails, the jitter
        term is 0 and the test rests on the floor alone, which in the study
        let wrong polishes through at tearing N=512.

        The returned eigenvalue σ (σ' or σ₀) comes with whichever of the
        unpolished vector v₀ and the polished vector v' has the smaller
        relative residual ‖M₁v - σM₂v‖ / (‖M₁v‖ + |σ|‖M₂v‖). On
        ill-conditioned pencils such as tearing at N >= 256 that is usually
        v₀: inverse iteration on the unbalanced pencil improves σ but not
        the vector. v' is returned with unit 2-norm and rotated by a unit
        phase so that v₀ᴴv' is real and positive; v₀ is returned as the
        dense solve gave it. The returned pair's residual can be larger
        than the unpolished pair's even though σ' is more accurate: on
        ill-conditioned pencils the residual is dominated by the vector's
        error amplified by ‖M₁ - σM₂‖, so a residual check against the
        unpolished pair is not evidence that the polish made things worse.

        Any such failure in a later step, typically an LU singular to
        working precision because σ has converged, keeps the pair of the
        previous step, which goes on to the guards below without a warning.

        A mode whose eigenvalue the sort changed (zeroed by
        sorting_cutoff) or that is not finite is never polished. The
        polish is refused, keeping the unpolished pair with a
        RuntimeWarning issued once per kind of failure, when the first
        step fails: the LU raises, the vector is non-finite or zero, or the
        Rayleigh quotient is not finite or its denominator vanishes; when
        the overlap |v₀ᴴv'| / (‖v₀‖‖v'‖) falls below
        self.POLISH_MIN_OVERLAP, meaning the iteration moved to another
        mode (this is the same-mode protection); when σ' lies nearer to
        another finite eigenvalue of the dense solve than to the unpolished
        one; or when the normwise backward error of the pair to be
        returned (σ' or the kept σ₀, with the vector chosen above),
        ‖M₁v - σM₂v‖ / ((‖M₁‖_F + |σ|‖M₂‖_F)‖v‖), exceeds
        self.GEVP_RESIDUAL_TOL. That last test is a floor against a
        catastrophic failure, not a certificate of accuracy, and is skipped
        when the pair to be returned is the unpolished one (σ₀ kept and v₀
        chosen), which the polish did not produce. The record of
        warnings already issued is process-wide, as in psecas.dense_eig:
        each kind of refusal warns once per Python process, whichever
        solver triggers it.
        """

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
        if self.do_gen_evp:
            self.get_matrix2()

        E_raw, V = self._eig_dense()

        # Sort the eigenvalues. A copy, because an overriding
        # sorting_strategy may change its argument in place, and E_raw must
        # stay the dense solve's own values for the polish test below.
        E, index = self.sorting_strategy(E_raw.copy())

        # Choose the eigenvalue mode value only
        i = index[mode]
        sigma = E[i]
        v = V[:, i]

        # Polish the selected pair on the original pencil, unless the sort
        # changed its value (a cutoff zero, say) or it is not finite.
        polished = False
        if (self.polish_steps > 0 and np.isfinite(E_raw[i])
                and E[i] == E_raw[i]):
            A = self.mat1.toarray()
            B = self.mat2.toarray() if self.do_gen_evp else None
            others = np.delete(E_raw, i)
            others = others[np.isfinite(others)]
            sigma, v = self._polish_pair(A, B, E_raw[i], V[:, i],
                                         self.polish_steps, others=others)
            polished = True

        # Save all eigenvalues and eigenvectors here
        if saveall:
            self.E = E[index]
            self.v = V[:, index]
            if polished:
                # Only the selected mode's slot holds the polished pair.
                self.E = self.E.astype(np.result_type(self.E, sigma),
                                       copy=False)
                self.E[mode] = sigma
                self.v[:, mode] = v
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

    #: Eigenvalues with |Re| or |Im| above this are zeroed by the default
    #: sorting_strategy, to push the spurious modes that a spectral
    #: discretization always produces to the bottom of the ordering. The
    #: right value depends entirely on the problem: growth rates in this
    #: package range from O(1e-4) for the tearing instability to O(1e2) for
    #: the channel modes. Set it on the solver, or override
    #: sorting_strategy() outright, when the default does not fit.
    sorting_cutoff = 10.0

    #: Relative tolerance for recognising a complex-conjugate pair in the
    #: default sorting_strategy: i and j are a pair when
    #: |E[i] - conj(E[j])| <= sorting_tie_rtol * max(|E[i]|, |E[j]|).
    #: The value is set by the dominant tearing pair at N=512, which matches
    #: its conjugate only to 2e-8 relative: 1e-6 leaves a factor of 50.
    #: Its limit: zggev does not enforce conjugate symmetry on the complex
    #: matrices psecas builds, and at N=512 only 75-78 % of the tearing
    #: eigenvalues have a numerical conjugate within 1e-6. Those that do
    #: not stay in plain real-part order. The second tearing pair at N=512
    #: (plain form) matches to only 7e-7, a margin of 1.4, so mode=2/3
    #: there can still depend on the backend. Must lie in [0, 1).
    sorting_tie_rtol = 1e-6

    #: Steps of inverse iteration with a two-sided Rayleigh quotient that
    #: solve() spends polishing the selected eigenpair on the original
    #: pencil (see solve()). Two steps take the dominant tearing eigenvalue
    #: from 2e-7 (QZ) or 7e-6 (GPU reduction) relative error at N=512 to
    #: 1e-9 or better, for about 0.5 s at n=2565 (1.3 s on one thread,
    #: against 118 s for QZ). 0 turns the polish off: solve() then returns
    #: exactly the unpolished pair of the dense solve.
    polish_steps = 2

    #: Smallest overlap |v_oldᴴv_new| / (‖v_old‖‖v_new‖) between the
    #: unpolished and the polished eigenvector for which solve() accepts
    #: the polish; below it the iteration is taken to have moved to another
    #: mode, and the unpolished pair is kept. On tearing at N=128-512 and
    #: every test system 1 - overlap is at most 1.1e-7; a polish started
    #: between two well-separated modes lands on the other one with an
    #: overlap far below 0.99 (0.29 in tests/test_polish.py).
    POLISH_MIN_OVERLAP = 0.99

    #: Factor c on the step-to-step jitter |σ₂ - σ₁| in the no-degrade rule
    #: of the polish: solve() keeps the unpolished eigenvalue σ₀ when the
    #: polish moves it by at most max(c|σ₂ - σ₁|, floor) (see solve()).
    #: c = 4 is set by tearing at N=512, where the jitter is largest: in the
    #: study of the rule (1628 calls), c = 2 polished one start there that
    #: should have been kept (mode 2, 1e-13 off, returned 4.7e-10 off) and
    #: c = 4 none. The QZ starts there, 2e-7 off, are polished with either.
    #: The price of never degrading: within the threshold the rule also
    #: keeps starts the polish would have improved, by up to 74x in that
    #: study (236 of 1628 calls), with the returned error still at most
    #: 1.275·max(4|σ₂ - σ₁|, floor).
    POLISH_JITTER_FACTOR = 4

    def sorting_strategy(self, E):
        """
        A default sorting strategy.

        Eigenvalues whose real or imaginary part exceeds self.sorting_cutoff
        in magnitude are zeroed, and the result is sorted from largest to
        smallest real part.

        The two members of a complex-conjugate pair have real parts that
        agree only to rounding, so plain real-part order would put either
        one first depending on the LAPACK build or backend. Each eigenvalue
        is therefore paired, greedily in sorted order, with its nearest
        unpaired numerical conjugate, |E[i] - conj(E[j])| <=
        self.sorting_tie_rtol * max(|E[i]|, |E[j]|), and the member with
        Im > 0 takes the earlier of the two slots the pair holds. Nothing
        else moves. Two essentially real eigenvalues (|Im| <= rtol * |E|
        for both, which includes the zeroed entries) are never paired.

        One nondeterminism remains: a third eigenvalue whose real part falls
        inside a pair's real-part gap (measured at 1e-16 to 6e-9) can still
        swap slots with a member of the pair by rounding. For a well
        separated growing mode that is rare. It is not rare when the real
        parts themselves sit at rounding level, as in oscillatory or neutral
        spectra, or in Hall MRI, where the cutoff zeros lie inside the gap:
        each pair still comes out Im > 0 first, but which pair sits at a
        given mode index stays rounding-dependent.

        Exactly tied eigenvalues, such as the cutoff zeros, keep their input
        order (the sort is stable); the old argsort()[::-1] reversed it.

        Returns (E, index) where E is a *copy* with the large values zeroed,
        and index orders it. Override this method for problems whose
        eigenvalues do not sit near unity.
        """

        # Copy first. This used to modify the caller's array in place, so
        # merely asking how the solver would sort a spectrum destroyed it.
        E = np.array(E, copy=True)

        cutoff = self.sorting_cutoff
        E[np.abs(E.real) > cutoff] = 0
        E[np.abs(E.imag) > cutoff] = 0

        # Plain real-part order, largest first. A stable sort keeps exact
        # ties (the zeroed entries, say) in input order.
        index = np.argsort(-E.real, kind="stable")

        rtol = self.sorting_tie_rtol
        if not 0 <= rtol < 1:
            raise ValueError(
                "sorting_tie_rtol must lie in [0, 1), got {!r}.".format(rtol)
            )
        mod = np.abs(E)
        real_like = np.abs(E.imag) <= rtol * mod
        # -Re in slot order, ascending, for the candidate window.
        key = -E.real[index]
        paired = np.zeros(index.size, dtype=bool)   # by slot

        for p in range(index.size):
            if paired[p]:
                continue
            i = index[p]
            # A zero (a cutoff entry) has tolerance zero and can never pair.
            if not np.isfinite(E[i]) or mod[i] == 0:
                continue
            # |dRe| <= |E_i - conj(E_j)| <= rtol * max(|E_i|, |E_j|), and
            # |E_j| <= |E_i| / (1 - rtol), which bounds the window.
            w = rtol * mod[i] / (1.0 - rtol)
            lo = np.searchsorted(key, key[p] - w, side="left")
            hi = np.searchsorted(key, key[p] + w, side="right")
            cand = index[lo:hi]
            d = np.abs(E[i] - np.conj(E[cand]))
            ok = ~paired[lo:hi]
            ok[p - lo] = False
            if real_like[i]:
                ok &= ~real_like[cand]
            ok &= d <= rtol * np.maximum(mod[i], mod[cand])
            if not ok.any():
                continue
            # Nearest partner; argmin takes the lowest slot on a tie.
            best = lo + int(np.argmin(np.where(ok, d, np.inf)))
            paired[p] = paired[best] = True
            first, second = min(p, best), max(p, best)
            if E.imag[index[second]] > E.imag[index[first]]:
                index[first], index[second] = index[second], index[first]

        return (E, index)

    def keep_result(self, sigma, vec, mode):

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
                # See _make_eval_globals(): a convenience namespace, not a
                # sandbox.
                submat = eval(eq_t, dict(_EVAL_GLOBALS), env)

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
                                # See _make_eval_globals() on this namespace.
                                submat[index, :] = eval(
                                    bound_t, dict(_EVAL_GLOBALS), env)

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
