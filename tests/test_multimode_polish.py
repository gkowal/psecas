"""
The eigenpair polish of Solver.solve() (see tests/test_polish.py), carried
over to the shift-invert solve_mode() and to iterate_solve_multimode().

solve_mode() checks the residual of the pair it found by shift-invert,
and polishes a pair that passes on the dense pencil, with no spectrum for
the nearer-eigenvalue guard. iterate_solve_multimode() polishes its
tracked modes after every full solve past Ns[0]; on the guess path
solve_mode() does it. polish_steps = 0 must reproduce the unpolished paths
bit for bit.
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


# iterate_solve_multimode() ------------------------------------------------


def _mti_solver(N=32):
    """As in tests/test_iterative_drivers.py."""
    from psecas.systems.mti import MagnetoThermalInstability
    grid = ChebyshevExtremaGrid(N=N, zmin=0, zmax=1)
    system = MagnetoThermalInstability(grid, beta=1e5, Kn0=200, kx=4 * np.pi)
    return Solver(grid, system)


#: The driver settings of tests/test_iterative_drivers.py,
#: tests/test_shift_invert.py (gtol=10, every resolution past the second on
#: the guess path) and the gyrotropic tearing system of tests/test_polish.py.
_DRIVER_CASES = {
    "mti": (_mti_solver, [32, 48, 64], dict(orderby='real')),
    "mti-allgrids": (_mti_solver, [32, 48, 64, 96, 128],
                     dict(rtol=1e-3, allgrids=True)),
    "classical-guess": (_classical_solver, [32, 48, 64],
                        dict(rtol=1e-6, gtol=10.0, orderby='real')),
    "gyrotropic": (_tearing_solver, [32, 48, 64],
                   dict(maxmode=0, allmodes=True, orderby='amplitude')),
}


def _spy_driver(monkeypatch, record):
    """Record (grid N, input σ, others, returned σ, error estimate) of
    every _polish_pair call."""
    original = Solver._polish_pair

    def spy(self, A, B, σ, v, steps, others=None):
        out = original(self, A, B, σ, v, steps, others=others)
        record.append((self.grid.N, σ, others, out[0],
                       self._polish_error_estimate))
        return out

    monkeypatch.setattr(Solver, "_polish_pair", spy)


@pytest.mark.parametrize("case", sorted(_DRIVER_CASES))
def test_multimode_zero_steps_is_the_unpolished_driver(monkeypatch, case):
    """polish_steps = 0 never calls the polish, and the driver returns what
    it returns with a polish that leaves every pair alone: the unpolished
    driver, bit for bit."""
    make, Ns, kw = _DRIVER_CASES[case]

    record = []
    _spy_driver(monkeypatch, record)
    solver = make()
    solver.polish_steps = 0
    σ0, v0, e0 = solver.iterate_solve_multimode(Ns, **kw)
    r0 = dict(solver.system.result)
    assert record == []
    assert np.isnan(solver.last_error_estimate)
    assert np.isnan(r0["error_estimate"])

    monkeypatch.setattr(Solver, "_polish_pair",
                        lambda self, A, B, σ, v, steps, others=None: (σ, v))
    solver = make()
    σ1, v1, e1 = solver.iterate_solve_multimode(Ns, **kw)
    np.testing.assert_array_equal(σ0, σ1)
    np.testing.assert_array_equal(v0, v1)
    np.testing.assert_array_equal(e0, e1)
    assert solver.system.result["converged"] == r0["converged"]
    assert solver.system.result["error"] == r0["error"]


def test_multimode_polishes_the_tracked_modes_after_each_full_solve(
        monkeypatch):
    """No polish at Ns[0]; after every later full solve, at least one call
    per tracked mode (maxmode=2 with allmodes tracks three), each with the
    finite rest of the spectrum for the nearer-eigenvalue guard; no call
    starts from a value an earlier call at the same resolution returned
    (a mode polished twice would), and every mode tracked in the end has
    been polished."""
    record = []
    _spy_driver(monkeypatch, record)
    tracked = {}
    solver = _mti_solver()
    solver.plot_eigenmodes = lambda Σ, errors=None, **kw: tracked.update(
        {solver.grid.N: np.array(Σ[:3])})
    Ns = [32, 48, 64]
    solver.iterate_solve_multimode(Ns, gtol=0.0, maxmode=2, allmodes=True,
                                   orderby='real', allgrids=True,
                                   plots=True)
    assert {r[0] for r in record} == {48, 64}
    for N in (48, 64):
        inputs = [r[1] for r in record if r[0] == N]
        outputs = [r[3] for r in record if r[0] == N]
        assert len(inputs) >= 3
        assert len(set(inputs)) == len(inputs)
        for k, σ in enumerate(inputs):
            assert σ not in outputs[:k]
        assert all(σ in outputs for σ in tracked[N])
    for _, σ, others, _, _ in record:
        assert others is not None and others.size > 0
        assert np.all(np.isfinite(others))
        assert σ not in others


def test_multimode_polishes_a_mode_the_polish_brought_into_the_tracked_set(
        monkeypatch):
    """Gyrotropic tearing at N=64, orderby='real': polishing the tracked
    member of the dominant conjugate pair lowers its real part below that
    of its unpolished partner, which then sorts first. The partner is
    polished in turn, and the mode returned is a polished one. That
    reorder is a property of QZ's starts, so the backend is pinned: the
    GPU's starts at N=64 do not produce it. Even with QZ it rests on the
    sign of a move of about 1e-13 in the real part, which another BLAS
    build can reverse; the deterministic coverage of the same loop is
    test_multimode_polish_loop_follows_a_forced_reorder."""
    record = []
    _spy_driver(monkeypatch, record)
    solver = _tearing_solver(32)
    solver.backend = "scipy"
    σ, _, _ = solver.iterate_solve_multimode([32, 64], gtol=1e-2,
                                             atol=1e-10, maxmode=0,
                                             allmodes=True, orderby='real')
    assert solver.grid.N == 64
    at64 = [r for r in record if r[0] == 64]
    assert len(at64) == 2
    # The second call is the conjugate partner of the first.
    assert abs(at64[1][1] - np.conj(at64[0][1])) <= 1e-10 * abs(at64[0][1])
    assert σ[0] in [r[3] for r in at64]
    T = solver.last_error_estimate
    assert np.isfinite(T)
    assert solver.system.result["error_estimate"] == T


def _fake_polish(monkeypatch, record, move):
    """Replace the polish by σ -> move(σ, k) for the k-th call (k = 1,
    2, ...), with error estimate k·1e-12, and record (grid N, input σ,
    others, returned σ, estimate)."""
    def fake(self, A, B, σ, v, steps, others=None):
        k = len(record) + 1
        σ_new = move(σ, k)
        self._polish_error_estimate = k * 1e-12
        record.append((self.grid.N, σ, others, σ_new,
                       self._polish_error_estimate))
        return σ_new, v

    monkeypatch.setattr(Solver, "_polish_pair", fake)


def test_multimode_polish_loop_follows_a_forced_reorder(monkeypatch):
    """MTI at N=48, orderby='real': the dominant modes are a conjugate
    pair whose real parts agree to far better than 1e-9. A polish that
    lowers the real part by 1e-9|σ| sorts the tracked member below its
    partner, whatever the dense solver's last bits; the partner, now
    tracked, is polished in turn, and the returned mode is a polish
    output with its own estimate."""
    record = []
    _fake_polish(monkeypatch, record,
                 lambda σ, k: σ - 1e-9 * abs(σ))
    solver = _mti_solver()
    σ, _, _ = solver.iterate_solve_multimode([32, 48], gtol=0.0, maxmode=0,
                                             allmodes=True, orderby='real')
    assert solver.grid.N == 48
    assert [r[0] for r in record] == [48, 48]
    # The second call is the conjugate partner of the first.
    assert abs(record[1][1] - np.conj(record[0][1])) \
        <= 1e-10 * abs(record[0][1])
    T = {r[3]: r[4] for r in record}
    assert σ[0] in T
    assert solver.last_error_estimate == T[σ[0]]
    assert solver.system.result["error_estimate"] == T[σ[0]]


def test_multimode_estimates_follow_their_modes_through_a_reorder(
        monkeypatch):
    """Three tracked modes (maxmode=2 with allmodes); a polish that raises
    the real part of the k-th call by k·200, more than the spread of the
    three, reverses their order. The modes come back in that order, and
    the estimate reported for the selected mode (index 2, the first one
    polished) is that of its own polish."""
    record = []
    _fake_polish(monkeypatch, record, lambda σ, k: σ + 200.0 * k)
    solver = _mti_solver()
    Σ, _, _ = solver.iterate_solve_multimode([32, 48], gtol=0.0, maxmode=2,
                                             allmodes=True, orderby='real')
    assert solver.grid.N == 48
    assert len(record) == 3
    np.testing.assert_array_equal(Σ, [r[3] for r in record[::-1]])
    assert solver.last_error_estimate == record[0][4]
    assert solver.system.result["error_estimate"] == record[0][4]


def _guess_path_falls_back(monkeypatch, fake_solve_mode):
    """Run the gtol=10 classical tearing case with solve_mode() replaced,
    and return the solver and the polish record."""
    record = []
    _spy_driver(monkeypatch, record)
    monkeypatch.setattr(Solver, "solve_mode", fake_solve_mode)
    make, Ns, kw = _DRIVER_CASES["classical-guess"]
    solver = make()
    σ, _, _ = solver.iterate_solve_multimode(Ns, **kw)
    return solver, σ, record


@pytest.mark.parametrize("failure", ["rejected", "filtered"])
def test_multimode_fallback_full_solve_is_polished(monkeypatch, failure):
    """The guess path at N=64 fails, by a ShiftInvertError or by a guess
    result the filter removes (Re σ < 0); the full solve that replaces it
    is polished by the driver, with a spectrum, and the returned mode
    carries its estimate."""
    def rejected(self, guess, *args, **kwargs):
        raise ShiftInvertError("forced", sigma=guess, residual=1.0)

    def filtered(self, guess, *args, **kwargs):
        self.get_matrix1()
        return -abs(complex(guess)), np.ones(self.mat1.shape[0], complex)

    solver, σ, record = _guess_path_falls_back(
        monkeypatch, rejected if failure == "rejected" else filtered)
    assert solver.grid.N == 64
    at64 = [r for r in record if r[0] == 64]
    assert len(at64) >= 1
    assert all(r[2] is not None for r in at64)
    T = solver.last_error_estimate
    assert np.isfinite(T)
    assert [r[4] for r in at64 if r[3] == σ] == [T]
    assert solver.system.result["error_estimate"] == T


def test_multimode_refusal_warning_names_the_caller(monkeypatch):
    """The driver calls the polish itself on a full solve, so the warning
    points at the line that called the driver."""
    _refuse_every_polish(monkeypatch)
    solver = _mti_solver()
    with pytest.warns(RuntimeWarning,
                      match="^kept the unpolished eigenpair: ") as rec:
        solver.iterate_solve_multimode([32, 48], gtol=0.0, orderby='real')
    assert [w.filename for w in rec] == [__file__]
    assert np.isnan(solver.last_error_estimate)


def test_multimode_guess_path_is_polished_by_solve_mode(monkeypatch):
    """With gtol=10 the second resolution is a full solve, polished by the
    driver (with a spectrum), and the third a guess solve, polished by
    solve_mode() (without one)."""
    record = []
    _spy_driver(monkeypatch, record)
    make, Ns, kw = _DRIVER_CASES["classical-guess"]
    make().iterate_solve_multimode(Ns, **kw)
    assert [(r[0], r[2] is None) for r in record] == [(48, False),
                                                      (64, True)]


@pytest.mark.parametrize("case, N", [("mti-early", 48),
                                     ("classical-guess", 64)])
def test_multimode_reports_the_error_estimate_of_the_returned_mode(
        monkeypatch, case, N):
    """The estimate comes from the polish of the returned eigenvalue at
    the returned resolution: a full solve (MTI, converged early at 48) or a
    guess solve (tearing, at 64)."""
    if case == "mti-early":
        make, Ns, kw = _mti_solver, [32, 48, 64, 96, 128], dict(rtol=1e-3)
    else:
        make, Ns, kw = _DRIVER_CASES[case]
    record = []
    _spy_driver(monkeypatch, record)
    solver = make()
    σ, _, _ = solver.iterate_solve_multimode(Ns, **kw)
    assert solver.grid.N == N
    T = solver.last_error_estimate
    assert np.isfinite(T)
    assert solver.system.result["error_estimate"] == T
    matches = [r[4] for r in record if r[0] == N and r[3] == σ]
    assert matches == [T]


def test_multimode_single_resolution_has_no_error_estimate():
    """Ns[0] is never polished."""
    solver = _mti_solver()
    solver.iterate_solve_multimode([32], orderby='real')
    assert np.isnan(solver.last_error_estimate)
    assert np.isnan(solver.system.result["error_estimate"])
