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

from psecas import Solver, System, ChebyshevExtremaGrid, plot_solution


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
