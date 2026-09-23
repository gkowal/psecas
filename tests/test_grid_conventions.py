"""
Tests for the NN / N grid-size convention in the solver.

Grids disagree on whether NN is N or N + 1: FourierGrid and HermiteGrid use
NN = N, every other grid uses NN = N + 1. The solver assumed N + 1 throughout
- it trimmed with [1:N, 1:N], addressed boundary nodes as 0 and N, and sliced
results in blocks of N - 1 - so on the two dissenting grids it silently built
matrices for a different discretization.
"""
import numpy as np
import pytest

from psecas import (Solver, System, ChebyshevExtremaGrid, LegendreExtremaGrid,
                    HermiteGrid, LaguerreGrid, FourierGrid, ChebyshevRootsGrid)


class Laplace(System):
    def make_background(self):
        self.q = 1.0


def _solver_with_dirichlet(grid):
    system = Laplace(grid, variables='f', eigenvalue='sigma')
    system.add_equation("sigma*f = q*dz(dz(f))", boundary=True)
    return Solver(grid, system), system


@pytest.mark.parametrize("make_grid", [
    pytest.param(lambda: ChebyshevExtremaGrid(32, 0, 1), id="ChebyshevExtrema"),
    pytest.param(lambda: LegendreExtremaGrid(32, 0, 1), id="LegendreExtrema"),
    pytest.param(lambda: ChebyshevRootsGrid(32, 0, 1), id="ChebyshevRoots"),
    pytest.param(lambda: HermiteGrid(32), id="Hermite"),
    pytest.param(lambda: LaguerreGrid(32), id="Laguerre"),
])
def test_trimmed_matrix_size_follows_NN(make_grid):
    """All-Dirichlet trimming removes exactly the two boundary nodes."""
    grid = make_grid()
    solver, _ = _solver_with_dirichlet(grid)
    solver.get_matrix1()

    assert solver.mat1.shape == (grid.NN - 2, grid.NN - 2)


@pytest.mark.parametrize("make_grid", [
    pytest.param(lambda: ChebyshevExtremaGrid(32, 0, 1), id="ChebyshevExtrema"),
    pytest.param(lambda: HermiteGrid(32), id="Hermite"),
    pytest.param(lambda: LaguerreGrid(32), id="Laguerre"),
])
def test_result_profile_spans_every_grid_node(make_grid):
    """keep_result must pad back to exactly NN values per variable."""
    grid = make_grid()
    solver, system = _solver_with_dirichlet(grid)
    solver.solve(mode=0)

    assert len(system.result['f']) == grid.NN
    assert system.result['f'][0] == 0.0
    assert system.result['f'][-1] == 0.0


def test_boundary_conditions_on_a_periodic_grid_are_refused():
    """
    This used to be accepted and quietly deleted a grid point: for N = 32 the
    trim [1:N, 1:N] turned a 32x32 block into 31x31, so the eigenvalues were
    for a different, meaningless discretization.
    """
    grid = FourierGrid(N=32, zmin=0, zmax=1)
    system = Laplace(grid, variables='f', eigenvalue='sigma')
    system.add_equation("sigma*f = q*dz(dz(f))", boundary=True)

    with pytest.raises(ValueError, match="periodic"):
        Solver(grid, system)


def test_periodic_grid_without_boundaries_still_works():
    """The guard must not block the legitimate periodic use of a Fourier grid."""
    grid = FourierGrid(N=64, zmin=0, zmax=2 * np.pi)
    system = Laplace(grid, variables='f', eigenvalue='sigma')
    system.add_equation("sigma*f = q*dz(dz(f))")

    solver = Solver(grid, system)
    E, _ = solver.solve_full()

    # d^2/dz^2 on a 2*pi-periodic domain has eigenvalues -k^2, k = 0, 1, 2 ...
    E = np.sort(E[np.isfinite(E)].real)[::-1][:5]
    np.testing.assert_allclose(E, [0, -1, -1, -4, -4], atol=1e-8)


def test_dirichlet_spectrum_is_correct_on_a_grid_with_NN_equal_N():
    """
    HermiteGrid has NN = N. Check an actual eigenvalue, not just a shape:
    the decaying solutions of f'' = sigma f on the Hermite grid are resolved
    well enough to be recognisable.
    """
    grid = HermiteGrid(64, C=1)
    solver, _ = _solver_with_dirichlet(grid)
    E, _ = solver.solve_full()

    assert np.isfinite(E).any()
    assert solver.mat1.shape == (grid.NN - 2, grid.NN - 2)


@pytest.mark.parametrize("make_grid", [
    pytest.param(lambda: __import__('psecas').ChebyshevRationalGrid(32, C=2, z='x'),
                 id="ChebyshevRational"),
    pytest.param(lambda: __import__('psecas').SincGrid(32, C=2, z='x'), id="Sinc"),
    pytest.param(lambda: __import__('psecas').HermiteGrid(32, C=2, z='x'), id="Hermite"),
    pytest.param(lambda: __import__('psecas').LaguerreGrid(32, C=2, z='x'), id="Laguerre"),
    pytest.param(lambda: __import__('psecas').ChebyshevTLnGrid(32, C=2, z='x'),
                 id="ChebyshevTLn"),
])
def test_infinite_grids_share_one_implementation(make_grid):
    """
    The five infinite grids used to carry their own identical copies of the
    constructor and the C / zmin / zmax properties. They now inherit them.
    """
    from psecas.grids.grid import InfiniteGrid

    grid = make_grid()

    assert isinstance(grid, InfiniteGrid)
    assert grid.z == 'x'
    assert grid.C == 2
    assert grid.zmin == grid.zg.min()
    assert grid.zmax == grid.zg.max()


@pytest.mark.parametrize("make_grid", [
    pytest.param(lambda: __import__('psecas').ChebyshevRationalGrid(32), id="ChebyshevRational"),
    pytest.param(lambda: __import__('psecas').SincGrid(32), id="Sinc"),
    pytest.param(lambda: __import__('psecas').HermiteGrid(32), id="Hermite"),
    pytest.param(lambda: __import__('psecas').LaguerreGrid(32), id="Laguerre"),
    pytest.param(lambda: __import__('psecas').ChebyshevTLnGrid(32), id="ChebyshevTLn"),
])
def test_changing_C_rebuilds_the_grid(make_grid):
    grid = make_grid()
    before = np.array(grid.zg)

    grid.C = 4

    assert not np.allclose(np.array(grid.zg), before)


@pytest.mark.parametrize("grid_cls, maxN", [("HermiteGrid", 245), ("LaguerreGrid", 120)])
def test_dmsuite_grids_reject_too_large_N(grid_cls, maxN):
    """
    The limit used to be enforced with an assert, which python -O strips.
    """
    import psecas

    cls = getattr(psecas, grid_cls)

    with pytest.raises(ValueError, match="maximum"):
        cls(maxN + 1)

    grid = cls(16)
    with pytest.raises(ValueError, match="maximum"):
        grid.N = maxN + 1


def test_grid_rejects_an_inverted_domain():
    """Also an assert previously, and so also removed under python -O."""
    with pytest.raises(ValueError, match="zmax must be greater"):
        ChebyshevExtremaGrid(16, 1.0, 0.0)
