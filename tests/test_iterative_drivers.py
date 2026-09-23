"""Argument handling and control flow of the iterative resolution sweeps."""
import numpy as np
import pytest

from psecas import Solver, ChebyshevExtremaGrid


def _mti_solver(N=32):
    from psecas.systems.mti import MagnetoThermalInstability

    grid = ChebyshevExtremaGrid(N=N, zmin=0, zmax=1)
    system = MagnetoThermalInstability(grid, beta=1e5, Kn0=200, kx=4 * np.pi)
    return Solver(grid, system)


def test_multimode_accepts_a_single_resolution():
    """
    One resolution is a legal request - solve once, report that no error
    estimate is available. This raised UnboundLocalError on `errors`.
    """
    sigma, v, error = _mti_solver().iterate_solve_multimode([32], orderby='real')

    assert np.isfinite(sigma)
    assert error == np.inf     # nothing to compare against


def test_multimode_rejects_an_empty_resolution_list():
    with pytest.raises(ValueError, match="at least one resolution"):
        _mti_solver().iterate_solve_multimode([])


@pytest.mark.parametrize("Ns", [[], [32]])
def test_iterate_solver_requires_two_resolutions(Ns):
    """It compares consecutive resolutions, so one is not enough."""
    with pytest.raises(ValueError, match="at least two"):
        _mti_solver().iterate_solver(Ns)


def test_multimode_runs_over_several_resolutions():
    sigma, v, error = _mti_solver().iterate_solve_multimode(
        [32, 48, 64], orderby='real'
    )

    assert np.isfinite(sigma)
    assert np.isfinite(error)


def test_multimode_can_track_damped_modes():
    """
    require_re_positive used to be hard-coded True, so modes with
    Re(sigma) <= 0 could not be found at all.
    """
    from psecas import System

    class Laplace(System):
        def make_background(self):
            self.q = 1.0

    grid = ChebyshevExtremaGrid(N=32, zmin=0, zmax=1)
    system = Laplace(grid, variables='f', eigenvalue='sigma')
    system.add_equation("sigma*f = q*dz(dz(f))", boundary=True)
    solver = Solver(grid, system)

    # Every eigenvalue of this problem is negative (-(n*pi)^2), so the
    # default filter leaves nothing.
    with pytest.raises(ValueError, match="no eigenmodes"):
        solver.iterate_solve_multimode([32, 48])

    sigma, _, _ = solver.iterate_solve_multimode(
        [32, 48], require_re_positive=False, orderby='real'
    )
    assert sigma.real < 0


def test_allgrids_runs_every_resolution():
    """
    With allgrids=True the sweep works through all of Ns even after the
    convergence criterion is met, instead of returning early.
    """
    Ns = [32, 48, 64, 96, 128]

    early = _mti_solver()
    early.iterate_solve_multimode(Ns, rtol=1e-3)

    full = _mti_solver()
    full.iterate_solve_multimode(Ns, rtol=1e-3, allgrids=True)

    assert early.grid.N < Ns[-1]          # stopped early
    assert full.grid.N == Ns[-1]          # ran to the end
    assert early.system.result["converged"] is True
    assert full.system.result["converged"] is True


def test_allgrids_reports_the_same_eigenvalue():
    Ns = [32, 48, 64, 96]

    early = _mti_solver().iterate_solve_multimode(Ns, rtol=1e-3)
    full = _mti_solver().iterate_solve_multimode(Ns, rtol=1e-3, allgrids=True)

    np.testing.assert_allclose(early[0], full[0], rtol=1e-6)


def test_allgrids_still_reports_failure_to_converge():
    """A sweep that never converges must not be marked converged."""
    solver = _mti_solver()

    solver.iterate_solve_multimode([32, 48], rtol=1e-14, atol=1e-30,
                                   allgrids=True)

    assert solver.system.result["converged"] is False
