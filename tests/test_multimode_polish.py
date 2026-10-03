"""
The eigenpair polish of Solver.solve() (see tests/test_polish.py), carried
over to the shift-invert solve_mode().

solve_mode() checks the residual of the pair it found by shift-invert,
and polishes a pair that passes on the dense pencil, with no spectrum for
the nearer-eigenvalue guard. polish_steps = 0 must reproduce the
unpolished path bit for bit.
"""
import numpy as np
import pytest

from psecas import ChebyshevExtremaGrid, Solver, ShiftInvertError, System
from psecas import solver as solver_mod
from psecas.solver import _rel_residual

from test_polish import TEARING_N256_MODE0, _tearing_solver


@pytest.fixture(autouse=True)
def _clean_warnings(monkeypatch):
    monkeypatch.setattr(solver_mod, "_polish_warned", set())


def _well_evp(N=64):
    """Infinite well as a standard EVP (boundaries trimmed), E = -n²π²/2."""
    grid = ChebyshevExtremaGrid(N=N, zmin=0, zmax=1, z='x')
    system = System(grid, variables='phi', eigenvalue='E')
    system.add_equation("E*phi = 1/2*dx(dx(phi))", boundary=True)
    solver = Solver(grid, system, do_gen_evp=False)
    assert not solver.do_gen_evp
    return solver


def _start(solver, seed=0):
    """A fixed ARPACK start vector, so that repeated solves agree bit for
    bit."""
    solver.get_matrix1()
    rng = np.random.default_rng(seed)
    n = solver.mat1.shape[0]
    return rng.standard_normal(n) + 1j * rng.standard_normal(n)


def _spy(monkeypatch, record):
    """Record (grid N, others, B is None, returned σ, error estimate) of
    every _polish_pair call."""
    original = Solver._polish_pair

    def spy(self, A, B, σ, v, steps, others=None):
        out = original(self, A, B, σ, v, steps, others=others)
        record.append((self.grid.N, others, B is None, out[0],
                       self._polish_error_estimate))
        return out

    monkeypatch.setattr(Solver, "_polish_pair", spy)


def test_solve_mode_polish_repairs_the_tearing_mode_at_n256():
    """Shift-invert at N=256, guessed 1e-6 off, is 7e-10 to 5e-9 off the
    reference (with the BLAS threading); the polish takes it to 4e-12 to
    2e-11. Both figures move with the guess and the start vector (the
    polish's own error reaches 1e-10, and shift-invert can land at 5e-11
    from a guess 1e-3 off), so both are fixed here. self.residual is that
    of the certified, unpolished pair, the same with the polish on or off;
    self.polished_residual is that of the returned, polished pair."""
    solver = _tearing_solver(256)
    ref = TEARING_N256_MODE0
    guess = ref * (1 + 1e-6)
    v_start = _start(solver)

    σ, v = solver.solve_mode(guess, v0=v_start.copy())
    assert abs(σ - ref) <= 1e-9 * abs(ref)
    T = solver.last_error_estimate
    assert np.isfinite(T)
    A, B = solver.mat1.tocsc(), solver.mat2.tocsc()
    assert solver.polished_residual == _rel_residual(A, B, σ, v)
    r = solver.residual

    solver.polish_steps = 0
    σ0, v0 = solver.solve_mode(guess, v0=v_start.copy())
    assert abs(σ - ref) < abs(σ0 - ref) / 10
    assert np.isnan(solver.last_error_estimate)
    assert np.isnan(solver.polished_residual)
    assert solver.residual == _rel_residual(A, B, σ0, v0)
    assert r == solver.residual
    # The two residuals differ, so the checks above tell them apart.
    assert _rel_residual(A, B, σ, v) != _rel_residual(A, B, σ0, v0)


@pytest.mark.parametrize("make, guess", [
    (lambda: _tearing_solver(48), TEARING_N256_MODE0),
    (_well_evp, -np.pi ** 2 / 2 * (1 + 1e-3)),
], ids=["tearing-gevp", "well-evp"])
def test_solve_mode_zero_steps_is_the_unpolished_path(monkeypatch, make,
                                                      guess):
    """polish_steps = 0 never calls the polish, and returns what a polish
    that leaves the pair alone returns: the unpolished path, bit for bit.
    """
    solver = make()
    v0 = _start(solver)

    record = []
    _spy(monkeypatch, record)
    solver.polish_steps = 0
    σ0, x0 = solver.solve_mode(guess, v0=v0.copy())
    r0 = solver.residual
    assert record == []
    assert np.isnan(solver.last_error_estimate)

    monkeypatch.setattr(Solver, "_polish_pair",
                        lambda self, A, B, σ, v, steps, others=None: (σ, v))
    solver.polish_steps = 2
    σ1, x1 = solver.solve_mode(guess, v0=v0.copy())
    assert σ0 == σ1
    np.testing.assert_array_equal(x0, x1)
    assert solver.residual == r0


def test_solve_mode_polishes_the_standard_evp_with_identity_b(monkeypatch):
    """The standard EVP branch polishes with B = None, and no spectrum."""
    solver = _well_evp()
    record = []
    _spy(monkeypatch, record)
    σ, _ = solver.solve_mode(-np.pi ** 2 / 2 * (1 + 1e-3))
    assert len(record) == 1
    _, others, b_is_none, σ_rec, T = record[0]
    assert others is None and b_is_none
    assert σ == σ_rec
    assert solver.last_error_estimate == T
    assert abs(σ + np.pi ** 2 / 2) <= 1e-8 * np.pi ** 2 / 2


def test_rejected_solve_mode_leaves_no_error_estimate():
    """A pair the residual check rejects carries no estimate, even after a
    polished solve that set one."""
    solver = _tearing_solver(48)
    solver.solve_mode(TEARING_N256_MODE0)
    assert np.isfinite(solver.last_error_estimate)
    with pytest.raises(ShiftInvertError):
        solver.solve_mode(TEARING_N256_MODE0, residual_tol=1e-300)
    assert np.isnan(solver.last_error_estimate)


def test_polished_residual_above_tolerance_does_not_reject(monkeypatch):
    """The verdict is that of the unpolished pair: a pair that passes is
    returned polished, even when the polished pair's residual exceeds
    residual_tol (as at tearing N=512, where it reached 1.01e-6 against
    1.5e-8 unpolished). Constructed here with a polish that returns a
    poor pair."""
    solver = _tearing_solver(48)
    v_start = _start(solver)
    rng = np.random.default_rng(1)
    junk = rng.standard_normal(v_start.size) + 0j

    def poor(self, A, B, σ, v, steps, others=None):
        self._polish_error_estimate = 1e-12
        return σ * (1 + 1e-3), junk

    monkeypatch.setattr(Solver, "_polish_pair", poor)
    σ, v = solver.solve_mode(TEARING_N256_MODE0, v0=v_start.copy())
    assert solver.residual <= 1e-6
    assert solver.polished_residual > 1e-6
    np.testing.assert_array_equal(v, junk)
    assert solver.last_error_estimate == 1e-12


def test_unconverged_pair_is_rejected_before_any_polish(monkeypatch):
    """A pair that fails the check raises, and the polish, which could
    bring it below the tolerance (as at tearing N=512 from a guess far from
    any eigenvalue: 1.06e-5 unpolished), never sees it. Constructed here
    with a residual_tol below the pair's residual and a polish that would
    return an exact pair."""
    solver = _tearing_solver(48)
    v_start = _start(solver)
    solver.polish_steps = 0
    solver.solve_mode(TEARING_N256_MODE0, v0=v_start.copy())
    r = solver.residual

    calls = []

    def exact(self, A, B, σ, v, steps, others=None):
        calls.append(σ)
        return σ, v

    monkeypatch.setattr(Solver, "_polish_pair", exact)
    solver.polish_steps = 2
    with pytest.raises(ShiftInvertError) as excinfo:
        solver.solve_mode(TEARING_N256_MODE0, v0=v_start.copy(),
                          residual_tol=r / 2)
    assert calls == []
    assert excinfo.value.residual == r
    assert solver.residual == r
    assert np.isnan(solver.last_error_estimate)


def _classical_solver(N=32):
    """As in tests/test_shift_invert.py."""
    from psecas.systems.tearing_instability import TearingClassicalMHD
    grid = ChebyshevExtremaGrid(N=N, zmin=-10, zmax=10)
    system = TearingClassicalMHD(grid, kx=0.5, a=1, S=1e4, periodic=False)
    return Solver(grid, system)


def test_solve_with_guess_reports_the_error_estimate():
    """The generalized EVP goes through solve_mode(), whose estimate
    reaches system.result as after solve(); the standard EVP is not
    polished and gets none."""
    solver = _tearing_solver(48)
    solver.solve_with_guess(TEARING_N256_MODE0)
    T = solver.last_error_estimate
    assert np.isfinite(T)
    assert solver.system.result["error_estimate"] == T

    solver.polish_steps = 0
    solver.solve_with_guess(TEARING_N256_MODE0)
    assert np.isnan(solver.system.result["error_estimate"])

    well = _well_evp()
    well.solve_with_guess(-np.pi ** 2 / 2 * (1 + 1e-3))
    assert "error_estimate" not in well.system.result


def test_iterate_solver_reports_the_error_estimate_on_the_guess_path(
        monkeypatch):
    """With guess_tol=10 the last resolution goes through
    solve_with_guess(), and the result carries its estimate."""
    calls = []
    original = Solver.solve_with_guess

    def spy(self, *args, **kwargs):
        calls.append(self.grid.N)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Solver, "solve_with_guess", spy)
    solver = _classical_solver()
    solver.iterate_solver([32, 48, 64], guess_tol=10.0)
    assert calls == [64]
    T = solver.last_error_estimate
    assert np.isfinite(T)
    assert solver.system.result["error_estimate"] == T


def test_solve_with_guess_standard_evp_leaves_no_stale_estimate():
    """A standard-EVP solve_with_guess() after a polished solve() leaves
    NaN, not the estimate of that solve()."""
    solver = _well_evp()
    # Lowest |E| first, as in tests/test_polish.py: the default sort
    # zeroes these eigenvalues, and a zeroed mode is not polished.
    solver.sorting_strategy = lambda E: (E, np.argsort(np.abs(E)))
    solver.solve(mode=0)
    assert np.isfinite(solver.last_error_estimate)
    solver.solve_with_guess(-np.pi ** 2 / 2 * (1 + 1e-3))
    assert np.isnan(solver.last_error_estimate)


def _refuse_every_polish(monkeypatch):
    """An overlap of at least 1.5 is impossible: every polish is refused,
    with a warning."""
    monkeypatch.setattr(Solver, "POLISH_MIN_OVERLAP", 1.5)


@pytest.mark.parametrize("call", ["solve", "solve_mode"])
def test_refusal_warning_names_the_caller(monkeypatch, call):
    """The warning names no method, and points at the line that called
    solve() or solve_mode(), here in this file."""
    _refuse_every_polish(monkeypatch)
    solver = _tearing_solver(48)
    with pytest.warns(RuntimeWarning,
                      match="^kept the unpolished eigenpair: ") as rec:
        if call == "solve":
            solver.solve(mode=0)
        else:
            solver.solve_mode(TEARING_N256_MODE0)
    assert [w.filename for w in rec] == [__file__]
    assert np.isnan(solver.last_error_estimate)

