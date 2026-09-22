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

    def __getstate__(self):
        """
        Exclude potentially large differentiation matrices from serialized state.

        This replaces older IO-side logic that deleted d0/d1/d2 before pickling.
        With this method, pickle/MPI transfer automatically omits these matrices,
        and they can be rebuilt later by calling make_grid() or via lazy access.
        """
        state = self.__dict__.copy()

        # Newer layout (post-refactor): store all operators in a single list
        # (e.g. self._d or self.d). Drop/clear it for serialization.
        if "_d" in state:
            state["_d"] = []
        if "d" in state and isinstance(state["d"], list):
            state["d"] = []

        # Backward compatibility: older grids store d0/d1/d2 directly.
        # Remove them if present.
        for key in ("d0", "d1", "d2"):
            if key in state:
                try:
                    del state[key]
                except Exception:
                    # Fallback: if deletion fails for any reason, just clear
                    state[key] = None

        return state

    def __setstate__(self, state):
        """
        Restore state from pickle and rebuild the differentiation matrices.

        __getstate__ drops the matrices to keep pickles small, so they have to
        be regenerated here. make_grid() is the only thing that can do it:
        every concrete grid builds self._d directly inside make_grid, using
        formulas specific to its own basis. Leaving the rebuild to a later
        "lazy" access did not work -- there was no working lazy path, and
        grid.D(1) on a restored grid raised NotImplementedError.
        """
        self.__dict__.update(state)

        if "_d" not in self.__dict__:
            self.__dict__["_d"] = []

        if "_observers" not in self.__dict__:
            self.__dict__["_observers"] = []

        # Rebuild with the observers muted.
        #
        # _observers holds bound methods of the objects that depend on this
        # grid, typically System.make_background. During unpickling those
        # objects may themselves be only half-restored -- pickle can hand the
        # grid back before it has set system.grid -- so calling them here
        # raises AttributeError. Muting them is also correct rather than
        # merely expedient: the background arrays they would recompute were
        # pickled alongside everything else and are already restored.
        observers = self.__dict__["_observers"]
        self.__dict__["_observers"] = []
        try:
            self.make_grid()
        finally:
            self.__dict__["_observers"] = observers

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
        """The list of differentiation matrices, [D0, D1, D2, ...]."""
        if not self._d:
            self.make_grid()
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
        """
        Make sure differentiation matrices up to order k exist.

        Orders beyond those the grid builds itself are formed by composition,
        D(n) = D(1) @ D(n-1). That is exact in exact arithmetic but loses
        roughly a digit of accuracy per order on spectral matrices, so grids
        that can build a high order directly should do so in make_grid().
        """
        if k < 0:
            raise ValueError("derivative order must be >= 0, got {}".format(k))

        if not self._d:
            self.make_grid()

        while len(self._d) <= k:
            self._d.append(self._d[1] @ self._d[-1])

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

    def derivative(self, vec, n):
        """N-th order derivative of vec defined at zg"""
        import numpy as np

        assert type(vec) is np.ndarray
        assert vec.shape[0] == self.NN
        return np.matmul(self.D(n), vec)
