from psecas.grids.grid import Grid, InfiniteGrid
from scipy.interpolate import barycentric_interpolate
import numpy as np


class LaguerreGrid(InfiniteGrid, Grid):
    """
        This grid uses Laguerre plynomials on y ∈ [0, ∞] to dicretize the
        system. dmsuite is used for the setup of the grid.

        N: The number of grid points
        C: A scaling parameter which regulates the extent of the grid

        Optional:
        z: a string which can be set to e.g. 'x' if x is used as the
           coordinate in your linearized equations.

        The domain is in theory [0, ∞] but in practice the minimum and
        maximum values of the grid depend on both N and C.
    """

    # dmsuite cannot build this grid beyond here.
    maxN = 120

    def make_grid(self):

        # from numpy.polynomial import Laguerre as H
        self.NN = self.N + 1

        # Optional dependency; see the note in hermite.py.
        from dmsuite import lagdif

        # Ask dmsuite for every order we need; see the note in hermite.py.
        order = max(2, self._max_derivative_order)
        zg, D = lagdif(self.NN, order, 1 / self.C)

        self.zg = zg
        self._d = [np.eye(self.NN)] + [D[i] for i in range(order)]

        self.finalize_derivatives()

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def interpolate(self, z, f):
        """"""

        msg = "Can't interpolate outside grid domain"
        assert np.array([z]).min() >= self.zmin, msg
        assert np.array([z]).max() <= self.zmax, msg

        return barycentric_interpolate(self.zg, f, z)
