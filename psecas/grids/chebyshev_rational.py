from psecas.grids.grid import Grid


class ChebyshevRationalGrid(Grid):
    """
        This grid uses Rational Chebyshev functions on z ∈ [-∞, ∞],
        the TBn(z) functions, to dicretize the system (Boyd page 356 and
        Table E.5 on page 556).

        N: The number of grid points
        C: A scaling parameter which regulates the extent of the grid

        Optional:
        z: a string which can be set to e.g. 'x' if x is used as the
           coordinate in your linearized equations.

        The domain is in theory [-∞, ∞] but in practice the minimum and
        maximum values of the grid depend on both N and C.
    """

    def __init__(self, N, C=1, z="z", max_derivative_order=2):
        self._observers = []

        self._N = N
        self._C = C
        self._max_derivative_order = int(max_derivative_order)
        self._d = []
        self.make_grid()

        # Grid variable name
        self.z = z

    def bind_to(self, callback):
        self._observers.append(callback)

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, value):
        self._N = value
        self.make_grid()

    @property
    def zmin(self):
        return self.zg.min()

    @property
    def zmax(self):
        return self.zg.max()

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        self._C = value
        self.make_grid()

    def cheb_gauss_nodes_and_Dx(self, N):
        import numpy as np

        j = np.arange(1, N+1)
        φ = (2*j - 1 - N) * np.pi / (2*N)   # Gauss angles (symmetric)
        x = np.sin(φ)                       # Chebyshev-Gauss nodes
        s = np.cos(φ)                       # as Q = sqrt(1 - x**2),
                                            # does not suffer cancellation for |x| -> 1
        λ = ((-1)**(j-1)) * np.cos(φ)       # barycentric weights, the most
                                            # stable way to form an explicit
                                            # first-derivative matrix with;
                                            # any common scale on 𝜆 cancels
        X  = x[:, None]
        dX = X - X.T
        np.fill_diagonal(dX, 1.0)
        Dx = (λ[None, :] / λ[:, None]) / dX
        np.fill_diagonal(Dx, 0.0)

        # Diagonal = negative row sum
        Dx[np.diag_indices(N)] = -Dx.sum(axis=1)

        return s, x, λ, Dx

    def make_grid(self):
        import numpy as np

        C = self.C
        self.NN = self.N + 1
        N = self.NN

        # Improved grid generation considering floating point arithmetic
        s, x, λ, Dx = self.cheb_gauss_nodes_and_Dx(N)

        # nodes on TB grid
        z = C * x / s

        A = np.diag((s**3)/C)

        Dz = [ np.eye(N) ]
        # D^(1) = A @ Dx
        Dprev = A @ Dx
        Dz.append(Dprev.copy(order='C'))

        # Higher orders: D^(m) = A @ (Dx @ D^(m-1)), keep this exact order
        for m in range(2, self._max_derivative_order+1):
            Dprev = A @ (Dx @ Dprev)
            Dz.append(Dprev.copy(order='C'))

        self.zg = z
        self._d  = Dz

        self.finalize_derivatives()

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def to_coefficients(self, f):
        from numpy.polynomial.chebyshev import chebfit
        import numpy as np

        # Convert infinite grid to xg = [-1, 1]
        xg = self.zg / np.sqrt(self.C ** 2 + self.zg ** 2)

        # Get coefficients for standard Chebyshev polynomials
        c, res = chebfit(xg, f, deg=self.N, full=True)

        return c

    def interpolate(self, z, f):
        """See equations 17.37 and 17.38 in Boyd"""
        from numpy.polynomial.chebyshev import chebval
        import numpy as np

        msg = "Can't interpolate outside grid domain"
        assert np.array([z]).min() >= self.zmin, msg
        assert np.array([z]).max() <= self.zmax, msg

        # Get coefficients for standard Chebyshev polynomials
        c = self.to_coefficients(f)

        # Convert infinite grid to xg = [-1, 1]
        x = z / np.sqrt(self.C ** 2 + z ** 2)

        # Evaluate the Chebyshev polynomial
        return chebval(x, c)
