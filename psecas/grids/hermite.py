from psecas.grids.grid import Grid, InfiniteGrid
from scipy.interpolate import barycentric_interpolate
import numpy as np


class HermiteGrid(InfiniteGrid, Grid):
    """
        This grid uses Hermite polynomials on z ∈ [-∞, ∞] to dicretize the
        system. dmsuite is used for the setup of the grid.

        N: The number of grid points
        C: A scaling parameter which regulates the extent of the grid

        Optional:
        z: a string which can be set to e.g. 'x' if x is used as the
           coordinate in your linearized equations.

        The domain is in theory [-∞, ∞] but in practice the minimum and
        maximum values of the grid depend on both N and C.
    """

    # dmsuite cannot build this grid beyond here.
    maxN = 245

    def make_grid(self):

        # from numpy.polynomial import Hermite as H
        self.NN = self.N

        # Imported here, not at module level: dmsuite is an optional
        # dependency (pyproject.toml extra 'dmsuite') that only this grid
        # and LaguerreGrid need, so `import psecas` must not require it.
        from dmsuite import herdif

        # Ask dmsuite for every order we need. It builds each one directly
        # rather than by repeated multiplication, which is more accurate at
        # high order than the composition fallback in Grid.
        order = max(2, self._max_derivative_order)
        zg, D = herdif(self.NN, order, 1 / self.C)

        self.zg = zg
        self._d = [np.eye(self.NN)] + [D[i] for i in range(order)]

        self.finalize_derivatives()

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def interpolate(self, z, f):
        """"""
        # from numpy.polynomial.hermite import hermfit, hermval
        # c, res = hermfit(self.zg, f, deg=self.N, full=True)
        # return hermval(z, c)

        msg = "Can't interpolate outside grid domain"
        assert np.array([z]).min() >= self.zmin, msg
        assert np.array([z]).max() <= self.zmax, msg

        return barycentric_interpolate(self.zg, f, z)
