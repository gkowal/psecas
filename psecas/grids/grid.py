class InfiniteGrid:
    """
    Mixin for grids on an infinite or semi-infinite domain.

    These grids have no user-chosen endpoints. Their extent follows from the
    resolution N and a scaling parameter C, so zmin and zmax are read off the
    nodes instead of being stored, and change when either is changed.

    This collects the constructor, the C property and the zmin/zmax
    properties that ChebyshevRationalGrid, SincGrid, HermiteGrid,
    LaguerreGrid and ChebyshevTLnGrid each carried their own identical copy
    of - about forty duplicated lines apiece, which is how Grid.__getstate__
    came to be written against a layout none of them actually shared.
    """

    def __init__(self, N, C=1, z="z", max_derivative_order=2):
        self._observers = []

        self._validate_N(N)

        self._N = N
        self._C = C
        self._max_derivative_order = int(max_derivative_order)
        self._d = []

        # Grid variable name; set before make_grid() for the reason given
        # in Grid.__init__.
        self.z = z

        self.make_grid()

    @property
    def zmin(self):
        return self.zg.min()

    @property
    def zmax(self):
        return self.zg.max()

    @property
    def C(self):
        """Scaling parameter controlling how far the grid reaches."""
        return self._C

    @C.setter
    def C(self, value):
        self._C = value
        self.make_grid()


def _barycentric_weights(x):
    """
    Barycentric weights w_j = 1 / prod_{k != j} (x_j - x_k).

    Computed in log space, because the product underflows for the clustered
    nodes of a Gauss-Lobatto grid well before N gets interesting. Only ratios
    of weights are ever used, so the overall normalisation is free.
    """
    import numpy as np

    diff = x[:, None] - x[None, :]
    np.fill_diagonal(diff, 1.0)

    logw = -np.sum(np.log(np.abs(diff)), axis=1)
    sign = np.prod(np.sign(diff), axis=1)
    logw -= logw.max()

    return sign * np.exp(logw)


def _polynomial_derivative_matrices(x, max_order):
    """
    Differentiation matrices up to max_order for Lagrange interpolation at
    the distinct nodes x, via the Welfert recursion:

        D^(k)_ij = k/(x_i - x_j) * ( (w_j/w_i) D^(k-1)_ii - D^(k-1)_ij )
        D^(k)_ii = - sum_{j != i} D^(k)_ij          (negative sum trick)

    This is markedly more accurate at high order than forming D^(k) as a
    product of k copies of D^(1), which loses roughly a digit per order. For
    a degree-7 polynomial on 49 nodes, relative error in D(4):

        grid                composition   recursion
        LegendreExtrema       6.8e-05      2.8e-08
        ChebyshevExtrema      1.8e-06      1.3e-08
        ChebyshevRoots        4.3e-07      2.2e-08

    Reference: Welfert, SIAM J. Numer. Anal. 34 (1997) 1640; see also
    Berrut & Trefethen, SIAM Review 46 (2004) 501.
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    n = x.size

    w = _barycentric_weights(x)

    dX = x[:, None] - x[None, :]
    np.fill_diagonal(dX, 1.0)
    ratio = w[None, :] / w[:, None]

    matrices = [np.eye(n)]

    D = ratio / dX
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(D, -D.sum(axis=1))
    matrices.append(D)

    for k in range(2, max_order + 1):
        prev = matrices[-1]
        D = k * (ratio * np.diag(prev)[:, None] - prev) / dX
        np.fill_diagonal(D, 0.0)
        np.fill_diagonal(D, -D.sum(axis=1))
        matrices.append(D)

    return matrices


class Grid:
    """
    Base class for grids.
    """

    def __init__(self, N, zmin, zmax, z='z', max_derivative_order=2):
        self._observers = []

        if zmax <= zmin:
            raise ValueError(
                "zmax must be greater than zmin, got zmin={}, zmax={}"
                .format(zmin, zmax)
            )
        self._validate_N(N)

        self._N = N
        self._zmin = zmin
        self._zmax = zmax
        self._max_derivative_order = int(max_derivative_order)
        self._d = []

        # Grid variable name. Set before make_grid(), which notifies the
        # objects bound to this grid and so can reach code that reads it.
        self.z = z

        self.make_grid()

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

    #: Whether the first and last grid nodes are domain boundaries on which
    #: boundary conditions can be imposed. False for periodic grids, whose
    #: nodes wrap around and which therefore have no boundary to speak of.
    periodic = False

    #: Largest N this grid can be built at, or None for no limit. The
    #: dmsuite-backed grids have one because the underlying routines lose
    #: accuracy or overflow beyond it.
    maxN = None

    #: Whether the grid interpolates with polynomials through distinct nodes
    #: on a finite domain. Such grids can build high-order differentiation
    #: matrices with the barycentric recursion instead of by composition.
    polynomial = False

    @property
    def L(self):
        return self.zmax - self.zmin

    def _validate_N(self, value):
        """Reject a resolution the grid cannot be built at."""
        if self.maxN is not None and value > self.maxN:
            raise ValueError(
                "N = {} requested for {}, but the maximum it supports is {}."
                .format(value, type(self).__name__, self.maxN)
            )

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
        self._validate_N(value)
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

        Grids based on polynomial interpolation at distinct nodes
        (polynomial = True) build orders above 2 with the Welfert barycentric
        recursion, which is far more accurate than repeated multiplication.
        Everything else falls back to composition, D(n) = D(1) @ D(n-1),
        which is exact in exact arithmetic but loses roughly a digit of
        accuracy per order.
        """
        import numpy as np

        if k < 0:
            raise ValueError("derivative order must be >= 0, got {}".format(k))

        if not self._d:
            self.make_grid()

        if len(self._d) > k:
            return

        if self.polynomial:
            # Work on the standard interval: the nodes of a Gauss-Lobatto
            # grid on a wide domain are far enough apart that the weight
            # products lose precision, and the chain rule for an affine map
            # is just a constant factor per order.
            zg = np.asarray(self.zg, dtype=float)
            scale = 2.0 / self.L
            xg = (zg - self.zmin) * scale - 1.0

            matrices = _polynomial_derivative_matrices(xg, k)
            # Keep the grid's own D(0..2): those come from validated closed
            # forms and are what the existing results were obtained with.
            for order in range(len(self._d), k + 1):
                self._d.append(matrices[order] * scale ** order)
            return

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
