from psecas.grids.grid import Grid, InfiniteGrid
from scipy.linalg import toeplitz
import numpy as np


class SincGrid(InfiniteGrid, Grid):
    """
        This grid uses Whittaker Cardinal or “Sinc” functions on z ∈ [-∞, ∞]
        to dicretize the system. See Boyd Appendix F.7 page 569.

        N: The number of grid points
        C: A scaling parameter which regulates the extent of the grid

        Optional:
        z: a string which can be set to e.g. 'x' if x is used as the
           coordinate in your linearized equations.

        The domain is in theory [-∞, ∞] but in practice the minimum and
        maximum values of the grid depend on both N and C.
    """

    @property
    def dz(self):

        return self.C / np.sqrt(self.N)

    def make_grid(self):
        """Build the nodes zg and the differentiation matrices, then notify
        any objects bound to this grid."""

        self.NN = self.N + 1
        N = self.NN

        zg = self.dz * (0.5 - N / 2 + np.arange(N))
        n = np.arange(N)
        row = -np.hstack([0.0, (-1) ** (n[1:] + 1) / n[1:]])
        col = np.hstack([0.0, (-1) ** (n[1:] + 1) / n[1:]])
        col[0] = row[0]
        d1 = toeplitz(row, col)
        d1 /= self.dz

        row = np.hstack([-np.pi ** 2 / 3, -2 * (-1) ** (n[1:]) / n[1:] ** 2])
        d2 = toeplitz(row)
        d2 /= self.dz ** 2

        self.zg = zg
        self._d = [ np.eye(N), d1, d2 ]

        self.finalize_derivatives()

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def interpolate(self, z, f):
        """
        Interpolate f located at self.zg onto the grid z.

        This function uses Lagrange interpolation (eq. 4.6 in Boyd) with
        the sinc Cardinal functions (eq F.34 in Boyd)
        """

        msg = "Can't interpolate outside solution domain"
        assert np.array([z]).min() >= self.zmin, msg
        assert np.array([z]).max() <= self.zmax, msg

        assert len(f) == self.NN

        def to_grid(z):
            return np.sum(f * np.sinc((z - self.zg) / self.dz))

        to_grid_v = np.vectorize(to_grid)

        return to_grid_v(z)
