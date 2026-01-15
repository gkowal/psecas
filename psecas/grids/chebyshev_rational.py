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

        # Store corresponding finite-domain Chebyshev-Gauss nodes (in x-space)
        # and barycentric weights for Chebyshev-Gauss nodes for interpolation.
        self._xg = x.copy()
        self._bw = λ.copy()

        self.finalize_derivatives()

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def interpolate(self, z, f):
        """
        Robust interpolation using barycentric formula on Chebyshev–Gauss nodes.

        Parameters
        ----------
        z : float or array-like
            Points in physical (infinite) coordinate where to interpolate.
        f : array-like
            Function values sampled on self.zg (length self.NN).

        Returns
        -------
        p : float or ndarray
            Interpolated values at z.
        """
        import numpy as np

        z = np.asarray(z, dtype=float)
        f = np.asarray(f)

        if f.shape[0] != self.NN:
            raise ValueError("f must have shape (self.NN,)")

        msg = "Can't interpolate outside grid domain"
        if z.min() < self.zmin or z.max() > self.zmax:
            raise ValueError(msg)

        # Map query points to x in [-1, 1]
        C = float(self.C)
        x = z / np.sqrt(C * C + z * z)

        # Nodes and weights in x-space
        xg = self._xg
        w  = self._bw

        # Vectorized barycentric interpolation
        # Handle exact/near-exact node hits robustly to avoid division by zero.
        x_flat = x.ravel()
        out = np.empty_like(x_flat, dtype=np.result_type(f, x_flat))

        # Tolerance for "hit a node" in x-space; scale with machine precision
        tol = 50 * np.finfo(float).eps

        for k, xv in enumerate(x_flat):
            diff = xv - xg
            jhit = np.where(np.abs(diff) <= tol)[0]
            if jhit.size:
                out[k] = f[jhit[0]]
            else:
                tmp = w / diff
                out[k] = (tmp @ f) / tmp.sum()

        return out.reshape(x.shape)
