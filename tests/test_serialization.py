"""
Round-trip tests for pickling grids and systems.

Grid.__getstate__ drops the differentiation matrices to keep pickles small,
on the stated promise that they can be rebuilt afterwards. That promise was
not kept: the rebuild path called an abstract _build_d1() that no grid
implements, so grid.D(1) on a restored grid raised NotImplementedError and
every save_system/load_system workflow broke on first use.
"""
import pickle

import numpy as np
import pytest

from psecas import (Solver, System, save_system, load_system,
                    ChebyshevExtremaGrid, ChebyshevRootsGrid, FourierGrid,
                    LegendreExtremaGrid, ChebyshevRationalGrid, SincGrid,
                    HermiteGrid, LaguerreGrid, ChebyshevTLnGrid)


GRIDS = [
    pytest.param(lambda: ChebyshevExtremaGrid(16, -1, 1), id="ChebyshevExtrema"),
    pytest.param(lambda: ChebyshevRootsGrid(16, -1, 1), id="ChebyshevRoots"),
    pytest.param(lambda: FourierGrid(16, 0, 1), id="Fourier"),
    pytest.param(lambda: LegendreExtremaGrid(16, -1, 1), id="LegendreExtrema"),
    pytest.param(lambda: ChebyshevRationalGrid(16), id="ChebyshevRational"),
    pytest.param(lambda: SincGrid(16), id="Sinc"),
    pytest.param(lambda: HermiteGrid(16), id="Hermite"),
    pytest.param(lambda: LaguerreGrid(16), id="Laguerre"),
    pytest.param(lambda: ChebyshevTLnGrid(16), id="ChebyshevTLn"),
]


@pytest.mark.parametrize("make_grid", GRIDS)
def test_grid_survives_a_pickle_round_trip(make_grid):
    grid = make_grid()
    zg = np.array(grid.zg)
    d1 = np.array(grid.D(1))
    d2 = np.array(grid.D(2))

    restored = pickle.loads(pickle.dumps(grid))

    np.testing.assert_allclose(np.array(restored.zg), zg)
    np.testing.assert_allclose(np.array(restored.D(1)), d1)
    np.testing.assert_allclose(np.array(restored.D(2)), d2)


@pytest.mark.parametrize("make_grid", GRIDS)
def test_pickle_does_not_carry_the_matrices(make_grid):
    """The size saving __getstate__ exists for must still hold."""
    grid = make_grid()
    grid.D(2)

    state = grid.__getstate__()

    assert state["_d"] == []


def test_grid_still_differentiates_after_a_round_trip():
    grid = ChebyshevExtremaGrid(32, 0, 1)
    restored = pickle.loads(pickle.dumps(grid))

    f = np.sin(2 * np.pi * restored.zg)
    expected = 2 * np.pi * np.cos(2 * np.pi * restored.zg)

    np.testing.assert_allclose(restored.der(f), expected, atol=1e-10)


def test_solved_system_survives_save_and_load(tmp_path):
    """The workflow save_system/load_system exists for."""
    from psecas.systems.mti import MagnetoThermalInstability

    grid = ChebyshevExtremaGrid(N=32, zmin=0, zmax=1)
    system = MagnetoThermalInstability(grid, beta=1e5, Kn0=200, kx=4 * np.pi)
    sigma, _ = Solver(grid, system).solve(mode=0)

    path = tmp_path / "system.p"
    save_system(system, str(path))
    restored = load_system(str(path))

    assert restored.result[restored.eigenvalue] == pytest.approx(sigma)
    # The restored grid must be usable, not merely present.
    np.testing.assert_allclose(np.array(restored.grid.zg), np.array(grid.zg))
    assert restored.grid.D(1).shape == grid.D(1).shape
    restored.grid.interpolate(np.linspace(0, 1, 10), restored.result['dvz'].real)
