import pytest
def test_legendre_differentation(show=False):
    """Test the differentation routine of LegendreExtremaGrid"""
    from psecas import LegendreExtremaGrid
    import numpy as np

    N = 20
    zmin = -1
    zmax = 1
    grid = LegendreExtremaGrid(N, zmin, zmax)

    assert grid.zg[0] < grid.zg[-1]

    z = grid.zg
    y = np.exp(z) * np.sin(5 * z)
    yp_exac = np.exp(z) * (np.sin(5 * z) + 5 * np.cos(5 * z))
    yp_num = np.matmul(grid.d1, y)

    if show:
        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.clf()
        plt.title("Differentation with matrix (LegendreExtremaGrid)")
        plt.plot(z, yp_exac, "-")
        plt.plot(z, yp_num, "--")
        plt.show()

    np.testing.assert_allclose(yp_num, yp_exac, atol=1e-16)


def test_legendre_interpolation(show=False):
    """Test the inperpolation routine of LegendreExtremaGrid"""
    from psecas import LegendreExtremaGrid
    import numpy as np

    def psi(x, c):
        from numpy.polynomial.hermite import hermval

        return hermval(x, c) * np.exp(-x ** 2 / 2)

    N = 40
    zmin = -1.1
    zmax = 1.5
    grid = LegendreExtremaGrid(N, zmin, zmax)

    grid_fine = LegendreExtremaGrid(N * 4, zmin, zmax)
    z = grid_fine.zg

    y = np.exp(grid.zg) * np.sin(5 * grid.zg)
    y_fine = np.exp(z) * np.sin(5 * z)
    y_interpolated = grid.interpolate(z, y)

    if show:
        import matplotlib.pyplot as plt

        plt.figure(2)
        plt.clf()
        plt.title("Interpolation with Legendre")
        plt.plot(z, y_fine, "-")
        plt.plot(z, y_interpolated, "--")
        plt.plot(grid.zg, y, "+")
        plt.show()

    np.testing.assert_allclose(y_fine, y_interpolated, atol=1e-12)


if __name__ == "__main__":
    test_legendre_differentation(show=True)
    test_legendre_interpolation(show=True)


def test_legendre_grid_is_real():
    """
    legroots returns complex128 even when every root is real. Left as-is it
    made zg, every differentiation matrix, every background array evaluated
    on zg and every assembled solver matrix complex.
    """
    import numpy as np
    from psecas import LegendreExtremaGrid

    grid = LegendreExtremaGrid(24, -1, 1)

    assert grid.zg.dtype == np.float64
    assert grid.D(1).dtype == np.float64
    assert grid.D(2).dtype == np.float64


def test_legendre_grid_construction_is_warning_free():
    import warnings

    from psecas import LegendreExtremaGrid

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        LegendreExtremaGrid(24, -1, 1)


def test_legendre_interpolation_accuracy_is_domain_independent():
    """
    Fitting on the physical grid rather than the standard interval made
    accuracy depend on where the domain sat: the same Gaussian interpolated
    100x worse on [0, 10] than on [-1, 1].
    """
    import numpy as np
    from psecas import LegendreExtremaGrid

    def error_on(zmin, zmax):
        grid = LegendreExtremaGrid(32, zmin, zmax)
        centre, width = (zmin + zmax) / 2, 0.1 * (zmax - zmin)
        f = np.exp(-((grid.zg - centre) / width) ** 2)

        z = np.linspace(zmin, zmax, 501)
        exact = np.exp(-((z - centre) / width) ** 2)
        return np.abs(grid.interpolate(z, f) - exact).max()

    reference = error_on(-1, 1)
    for zmin, zmax in [(0, 1), (0, 10), (-5, 5)]:
        assert error_on(zmin, zmax) == pytest.approx(reference, rel=1e-6)
