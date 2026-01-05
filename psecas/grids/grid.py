class Grid:
    """
    Base class for grids.
    """

    def __init__(self, N, zmin, zmax, z='z', max_derivative_order=2):
        self._observers = []

        assert zmax > zmin

        self._N = N
        self._zmin = zmin
        self._zmax = zmax
        self._max_derivative_order = int(max_derivative_order)
        self._d = []
        self.make_grid()

        # Grid variable name
        self.z = z

    @property
    def L(self):
        return self.zmax - self.zmin

    def bind_to(self, callback):
        self._observers.append(callback)

    @property
    def N(self):
        return self._N

    @property
    def zmin(self):
        return self._zmin

    @property
    def zmax(self):
        return self._zmax

    @property
    def d(self):
        if not self._d:
            self.build_derivatives(2)
        return self._d

    @property
    def d0(self):
        return self.D(0)

    @property
    def d1(self):
        return self.D(1)

    @property
    def d2(self):
        return self.D(2)

    @N.setter
    def N(self, value):
        self._N = value
        self.make_grid()

    @zmin.setter
    def zmin(self, value):
        self._zmin = value
        self.make_grid()

    @zmax.setter
    def zmax(self, value):
        self._zmax = value
        self.make_grid()

    def D(self, k):
        self.ensure_derivatives(k)
        return self._d[k]

    def ensure_derivatives(self, k):
        if not self._d:
            self.build_derivatives(min(2, k))
        while len(self._d) <= k:
            self._d.append(self._d[1] @ self._d[-1])

    def build_derivatives(self, max_order=2):
        self._d = []
        self._d.append(self._identity())
        if max_order >= 1:
            self._d.append(self._build_d1())
        if max_order >= 2:
            self._d.append(self._build_d2() if hasattr(self, "_build_d2")
                           else self._d[1] @ self._d[1])

    def _identity(self):
        import numpy as np
        return np.eye(self.N + 1)

    def _build_d1(self):
        raise NotImplementedError

    def finalize_derivatives(self, max_derivative_order=None):
        """
        Ensure the grid has differentiation matrices up to max_derivative_order.

        Concrete grid classes should call this at the end of make_grid(),
        after constructing the baseline operators (typically up to 2nd order).
        """
        if max_derivative_order is None:
            max_derivative_order = self._max_derivative_order
        max_derivative_order = int(max_derivative_order)
        if max_derivative_order < 0:
            raise ValueError("max_derivative_order must be >= 0")

        # Commit 4 assumes ensure_derivatives() exists (from earlier commits).
        # Keeping this guard makes the commit safer during incremental development.
        if hasattr(self, "ensure_derivatives"):
            self.ensure_derivatives(max_derivative_order)

    def der(self, vec):
        """First derivative of vec defined at zg"""
        import numpy as np

        assert type(vec) is np.ndarray
        assert vec.shape[0] == self.NN
        return np.matmul(self.d1, vec)

    def dder(self, vec):
        """Second derivative of vec defined at zg"""
        import numpy as np

        assert type(vec) is np.ndarray
        assert vec.shape[0] == self.NN
        return np.matmul(self.d2, vec)
