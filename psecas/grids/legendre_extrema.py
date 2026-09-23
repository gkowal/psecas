from psecas.grids.grid import Grid
from numpy.polynomial.legendre import legder, legroots, legval
from numpy.polynomial.legendre import legfit
from numpy.polynomial.legendre import legval
import numpy as np


class LegendreExtremaGrid(Grid):
    """
    This grid uses the Legendre extrema and endpoints grid on
    z ∈ [zmin, zmax] to discretize the system. This grid is also known as the
    Gauss-Lobatto grid. Implementation follows Boyd Appendix F.10 on page 572.

    N: The number of grid points
    zmin: The z value at the lower boundary
    zmax: The z value at the upper boundary

    Optional:
    z: a string which can be set to e.g. 'x' if x is used as the
       coordinate in your linearized equations.

    """

    # Lagrange interpolation through distinct nodes: high-order
    # derivatives come from the barycentric recursion, not composition.
    polynomial = True

    def __init__(self, N, zmin, zmax, z="z", max_derivative_order=2):
        super().__init__(N, zmin, zmax, z=z, max_derivative_order=max_derivative_order)

    def make_grid(self):

        N = self._N
        self.NN = N + 1
        L = self.L

        factor = L / 2

        d1 = np.zeros((N + 1, N + 1))

        cp = legder([0] * N + [1])
        # legroots returns a complex array even when, as here, every root is
        # real. Left as-is it made zg - and hence every differentiation
        # matrix, every background array evaluated on zg, and every matrix
        # the solver assembles - complex128, doubling memory and arithmetic
        # throughout and raising ComplexWarning downstream.
        zg = np.hstack([-1.0, np.real(legroots(cp)), 1.0])

        P_N = legval(zg, [0] * N + [1])

        # The diagonal is 0/0; it is overwritten immediately below. Suppress
        # 'invalid' as well as 'divide', since 0/0 raises the former and the
        # warning was still being printed.
        with np.errstate(divide='ignore', invalid='ignore'):
            d1 = P_N[:, None] / (P_N[None, :] * (zg[:, None] - zg[None, :]))

        d1[np.diag_indices(N+1)] = 0.0
        d1[0, 0] = -N * (N + 1) / 4
        d1[N, N] = +N * (N + 1) / 4

        d2 = np.dot(d1, d1)

        self.zg = (zg + 1) * L / 2 + self.zmin
        self._d = [ np.eye(self.NN), d1 / factor, d2 / factor ** 2 ]

        self.finalize_derivatives()

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def _to_standard(self, z):
        """Map z from [zmin, zmax] onto the standard interval [-1, 1]."""
        return (z - self.zmin) / self.L * 2.0 - 1.0

    def to_coefficients(self, f):
        """Coefficients of f in the standard Legendre basis on [-1, 1]."""

        # Fit on the mapped grid, matching what the Chebyshev grids do.
        # Fitting on the physical grid instead made the accuracy depend on
        # where the domain happens to sit: for a Gaussian on [0, 10] the
        # interpolation error was 1.3e-03 against 1.1e-05 for the same
        # function on [-1, 1], because the Legendre basis is only well
        # conditioned on its own interval.
        c, res = legfit(self._to_standard(self.zg), f, deg=self.N, full=True)

        return c

    def interpolate(self, z, f):

        msg = "Can't interpolate outside grid domain"
        assert np.array([z]).min() >= self.zmin, msg
        assert np.array([z]).max() <= self.zmax, msg

        # Get coefficients for Legendre polynomials
        c = self.to_coefficients(f)

        return legval(self._to_standard(z), c)
