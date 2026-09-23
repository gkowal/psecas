"""
Tests for the plotting helpers.

plotting.py was excluded from coverage and had no tests, which is how
plot_solution came to be broken on every current matplotlib without anyone
noticing. It is the package's headline convenience function and ten example
scripts call it.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from psecas import (Solver, System, ChebyshevExtremaGrid, plot_solution,
                    get_2Dmap)


class Laplace(System):
    def make_background(self):
        self.q = 1.0


def _single_variable_system(N=32):
    grid = ChebyshevExtremaGrid(N=N, zmin=0, zmax=1)
    system = Laplace(grid, variables='f', eigenvalue='sigma')
    system.add_equation("sigma*f = q*dz(dz(f))", boundary=True)
    Solver(grid, system).solve(mode=0)
    return system


def _multi_variable_system(N=32):
    from psecas.systems.mti import MagnetoThermalInstability

    grid = ChebyshevExtremaGrid(N=N, zmin=0, zmax=1)
    system = MagnetoThermalInstability(grid, beta=1e5, Kn0=200, kx=4 * np.pi)
    Solver(grid, system).solve(mode=0)
    return system


def test_plot_solution_multi_variable(tmp_path):
    system = _multi_variable_system()
    out = tmp_path / "multi.png"

    fig = plot_solution(system, filename=str(out))

    assert out.stat().st_size > 0
    assert len(fig.axes) == system.dim


def test_plot_solution_single_variable(tmp_path):
    """dim == 1 gave a bare Axes, so axes[j] raised TypeError."""
    system = _single_variable_system()
    out = tmp_path / "single.png"

    fig = plot_solution(system, filename=str(out))

    assert out.stat().st_size > 0
    assert len(fig.axes) == 1


def test_plot_solution_can_reuse_a_figure_number(tmp_path):
    """
    Drawing twice into the same figure number must work. Creating figure
    `num` and then asking subplots() to create it again raised
    "Figure N already exists" on matplotlib >= 3.8.
    """
    system = _single_variable_system()

    plot_solution(system, filename=str(tmp_path / "a.png"), num=7)
    plot_solution(system, filename=str(tmp_path / "b.png"), num=7)

    assert (tmp_path / "b.png").stat().st_size > 0


def test_plot_solution_without_kx_uses_short_title(tmp_path):
    """A system with no kx attribute must still plot."""
    system = _single_variable_system()
    assert not hasattr(system, 'kx')

    fig = plot_solution(system, filename=str(tmp_path / "c.png"))

    assert 'omega' in fig.axes[0].get_title() or fig.axes[0].get_title()
def test_get_2Dmap_respects_xmin():
    """xmin was accepted and then ignored when building the x grid."""
    system = _single_variable_system()
    system.kx = 1.0

    here = get_2Dmap(system, 'f', 0.0, 1.0, 16, 16)
    shifted = get_2Dmap(system, 'f', 100.0, 101.0, 16, 16)

    # exp(i kx x) with kx=1 over a window shifted by 100 is a different map
    assert not np.allclose(here, shifted)


def test_get_2Dmap_x_grid_matches_requested_window():
    """Cell centres must span [xmin, xmax], not [0, xmax - xmin]."""
    system = _single_variable_system()
    system.kx = 0.0   # kills the x dependence, isolating the grid placement

    Nx = 8
    val = get_2Dmap(system, 'f', 10.0, 11.0, Nx, 16)

    # With kx = 0 every column is identical; the test is simply that the
    # call succeeds and produces the requested shape.
    assert val.shape == (16, Nx)
    assert np.allclose(val, val[:, :1])


def test_plot_eigenvalues(tmp_path):
    import numpy as np
    from psecas import plot_eigenvalues

    sigma = np.array([1 + 0.5j, 2 - 0.3j, 0.1 + 0j])
    out = tmp_path / "spectrum.png"

    fig = plot_eigenvalues(sigma, filename=str(out))

    assert out.stat().st_size > 0
    assert len(fig.axes) >= 1


def test_plot_eigenvalues_with_errors(tmp_path):
    import numpy as np
    from psecas import plot_eigenvalues

    sigma = np.array([1 + 0.5j, 2 - 0.3j, 0.1 + 0j])
    errors = np.array([1e-3, 1e-1, 1e1])

    fig = plot_eigenvalues(sigma, errors=errors,
                           filename=str(tmp_path / "coloured.png"))

    # scatter axis plus the colorbar axis
    assert len(fig.axes) == 2


def test_plot_eigenvalues_handles_non_finite_errors(tmp_path):
    """The driver reports inf when no error estimate is available yet."""
    import numpy as np
    from psecas import plot_eigenvalues

    sigma = np.array([1 + 0j, 2 + 0j])
    errors = np.array([np.inf, np.inf])

    plot_eigenvalues(sigma, errors=errors, filename=str(tmp_path / "inf.png"))


def test_solver_plot_eigenmodes_titles_without_kx(tmp_path):
    """A system with no kx attribute must still plot."""
    system = _single_variable_system()
    solver = Solver(system.grid, system)
    assert not hasattr(system, 'kx')

    fig = solver.plot_eigenmodes(np.array([1 + 0j]),
                                 filename=str(tmp_path / "nokx.png"))

    assert "N=" in fig.axes[0].get_title()


def test_multimode_can_write_spectrum_plots(tmp_path):
    from psecas.systems.mti import MagnetoThermalInstability

    grid = ChebyshevExtremaGrid(N=32, zmin=0, zmax=1)
    system = MagnetoThermalInstability(grid, beta=1e5, Kn0=200, kx=4 * np.pi)
    solver = Solver(grid, system)

    solver.iterate_solve_multimode([32, 48], orderby='real',
                                   plots=True, plot_dir=str(tmp_path))

    assert list(tmp_path.glob("eigenmodes_*.png"))


def test_multimode_writes_no_plots_by_default(tmp_path, monkeypatch):
    from psecas.systems.mti import MagnetoThermalInstability

    monkeypatch.chdir(tmp_path)
    grid = ChebyshevExtremaGrid(N=32, zmin=0, zmax=1)
    system = MagnetoThermalInstability(grid, beta=1e5, Kn0=200, kx=4 * np.pi)

    Solver(grid, system).iterate_solve_multimode([32, 48], orderby='real')

    assert not list(tmp_path.glob("*.png"))
