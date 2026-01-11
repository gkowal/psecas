class Solver:
    """docstring for Solver"""

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

    def solve(self, useOPinv=True, verbose=False, mode=0, saveall=False):
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

        useOPinv (default True): If true, manually calculate OPinv instead of
        letting eigs do it.

        verbose (default False): print out information about the calculation.

        mode (default 0): mode=0 is the fastest growing, mode=1 the second
        fastest and so on.
        """
        from scipy.linalg import eig

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

    def solve_with_guess(self, guess, useOPinv=True, verbose=False, mode=0):
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
        """
        import numpy as np
        from scipy.sparse.linalg import eigs

        # Calculate right-hand matrix
        self.get_matrix1()

        # Solve a generalized EVP
        if self.do_gen_evp:
            self.get_matrix2()
            if useOPinv:
                from numpy.linalg import inv
                OPinv = inv((self.mat1 - guess * self.mat2).toarray())
                sigma, v = eigs(self.mat1, k=1, sigma=guess, OPinv=OPinv)
            else:    
                sigma, v = eigs(self.mat1, M=self.mat2, k=1, sigma=guess)
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
        useOPinv=True
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
        """
        import numpy as np

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
                (sigma_new, v) = self.solve_with_guess(
                    sigma_old, mode=mode, verbose=verbose, useOPinv=useOPinv
                )

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
            # Add zeros at both ends of the solution
            self.system.result = {
                var: np.hstack(
                    [
                        0.0,
                        vec[
                            j
                            * (self.grid.N - 1) : (j + 1)
                            * (self.grid.N - 1)
                        ],
                        0.0,
                    ]
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
        N = self.grid.N
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
                    rows[j][i] = rows[j][i][1:N, 1:N]
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
        N = self.grid.N
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
                    rows[j][i] = rows[j][i][1:N, 1:N]
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
                        # Keep index convention consistent with _modify_submatrix():
                        # boundary nodes are 0 and N (inclusive grid).
                        if extra_binfo[j][0] is not None:
                            rows[j][i][0, :] = 0
                        if extra_binfo[j][1] is not None:
                            rows[j][i][N, :] = 0
                    else:
                        # Backward-compatible behavior for non-generalized EVP:
                        # preserve existing "diagonal-entry zeroing" logic.
                        if extra_binfo[j][0] is not None:
                            rows[j][i][0, 0] = 0
                        if extra_binfo[j][1] is not None:
                            rows[j][i][N, N] = 0

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


    def _find_submatrices(self, eq, verbose=False):
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

        for i, var in enumerate(self.system.variables):
            if var in eq:
                variables_t = list(np.copy(self.system.variables))
                eq_t = eq
                # Apply equation substitutions
                if hasattr(self.system, 'substitutions'):
                    for substitution in self.system.substitutions:
                        sub_split = substitution.split('=')
                        eq_t = var_replace(eq_t, sub_split[0].strip(), sub_split[1])
                        print(eq_t)
                eq_t = self._rewrite_derivatives(eq_t, grid, var)

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
                    submat = eval(eq_t, {"__builtins__": {}}, env).T

                except NameError as e:
                    strerror, = e.args
                    err_msg2 = (
                        "\n\nThis is likely because the missing variable has"
                        "\nnot been defined in your systems class or its\n"
                        "make_background method."
                    )
                    raise NameError(
                        err_msg1.format(eq, eq_t, var) + strerror + err_msg2
                    )
                except Exception as e:
                    raise Exception(err_msg1.format(eq, eq_t, var) + str(e))
                submat = np.array(submat, dtype="complex128")
            else:
                submat = np.zeros((NN, NN), dtype=np.complex128)

            # Prevent sparse.lil_matrix from changing the shape of
            # a numpy array which is all zeros.
            if np.count_nonzero(submat) == 0:
                submat = np.zeros((NN, NN), dtype=np.complex128)

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
        import numpy as np
        from .string_methods import var_replace

        grid = self.system.grid

        env = dict(self.system.__dict__)
        env["grid"] = grid

        N = self.grid.N
        if boundary:
            for index, bound in zip([0, N], binfo):
                if bound is not None:
                    submat[index, :] = 0
                    if eq_n == var_n:
                        if bound == 'Dirichlet':
                            submat[index, index] = 1
                        elif bound == 'Neumann':
                            submat[index, :] = grid.D(1)[index, :]
                        else:
                            assert '=' in bound, 'equal sign missing in boundary expression'
                            assert int(bound.split("=")[1]) == 0, 'rhs of boundary expressions must be zero'
                            var = self.system.variables[var_n-1]
                            bound_t = bound.split("=")[0]

                            # Apply equation substitutions
                            if hasattr(self.system, 'substitutions'):
                                for substitution in self.system.substitutions:
                                    sub_split = substitution.split('=')
                                    bound_t = var_replace(bound_t, sub_split[0].strip(), sub_split[1])

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
                                submat[index, :] = eval(bound_t, {"__builtins__": {}}, env)

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
