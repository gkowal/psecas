"""
The polish of the selected eigenpair in Solver.solve().

solve() refines the pair that sorting_strategy selects with polish_steps
steps of inverse iteration and a two-sided Rayleigh quotient on the
original pencil, and returns the polished eigenvalue with whichever of the
unpolished and polished vectors has the smaller residual. polish_steps = 0
must reproduce the unpolished path bit for bit; a mode whose value the sort
changed is never polished; every guard keeps the unpolished pair with a
single warning per kind of failure; and the no-degrade rule keeps the
unpolished eigenvalue, bit for bit and without a warning, when the polish
moves it by no more than its own floor or step-to-step jitter.
"""
import warnings

import numpy as np
import pytest

from psecas import ChebyshevExtremaGrid, Solver, System
from psecas import solver as solver_mod


@pytest.fixture(autouse=True)
def _clean_warnings(monkeypatch):
    monkeypatch.setattr(solver_mod, "_polish_warned", set())


def _well_solver(N, do_gen_evp=True, backend="auto"):
    """Infinite well, E_n = -n²π²/2. With do_gen_evp=True Psecas solves a
    generalized EVP whose B has zero rows at the two Dirichlet boundaries;
    with False it trims the boundaries and solves a standard EVP. Tests
    whose premise is QZ's error pass backend="scipy": the GPU path can
    start closer to the exact value than the polish's own floor."""
    grid = ChebyshevExtremaGrid(N=N, zmin=0, zmax=1, z='x')
    system = System(grid, variables='phi', eigenvalue='E')
    if do_gen_evp:
        system.add_equation("-E*phi = -1/2*dx(dx(phi))", boundary=True)
    else:
        system.add_equation("E*phi = 1/2*dx(dx(phi))", boundary=True)
    solver = Solver(grid, system, do_gen_evp=do_gen_evp, backend=backend)
    assert solver.do_gen_evp == do_gen_evp
    solver.sorting_strategy = lambda E: (E, np.argsort(np.abs(E)))
    return solver


def _tearing_solver(N=48):
    from psecas.systems.tearing_instability import TearingGyrotropicMHD
    grid = ChebyshevExtremaGrid(N=N, zmin=-10, zmax=10)
    system = TearingGyrotropicMHD(grid, normalized=True, kx=0.5, a=1, S=1e4,
                                  Pr=0.1, β=1.0, Δβ=0.2, ɣpar=3, ɣper=2,
                                  periodic=False)
    return Solver(grid, system, do_gen_evp=True)


def _unpolished(solver, mode):
    """The old solve() path, by hand: dense solve and sort."""
    solver.get_matrix1()
    if solver.do_gen_evp:
        solver.get_matrix2()
    E, V = solver._eig_dense()
    E, index = solver.sorting_strategy(E)
    return E[index[mode]], V[:, index[mode]], E[index], V[:, index]


def _pencil(solver):
    A = solver.mat1.toarray()
    B = solver.mat2.toarray() if solver.do_gen_evp else None
    return A, B


def _perturbed(σ, v, δ, seed):
    """(σ(1 + δ), v') with v' the unit vector at angle δ from v, tilted in a
    random direction orthogonal to v drawn with the given seed."""
    rng = np.random.default_rng(seed)
    x = v / np.linalg.norm(v)
    u = rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)
    u = u - np.vdot(x, u) * x
    u = u / np.linalg.norm(u)
    return σ * (1 + δ), np.cos(δ) * x + np.sin(δ) * u


def test_polish_reaches_the_exact_well_eigenvalue():
    """QZ is 4e-10 off -π²/2 at N=256; two polish steps reach 4e-13."""
    exact = -np.pi ** 2 / 2
    solver = _well_solver(256, backend="scipy")
    σ, _ = solver.solve(mode=0)
    assert abs(σ - exact) <= 1e-11 * abs(exact)

    solver.polish_steps = 0
    σ0, _ = solver.solve(mode=0)
    assert abs(σ0 - exact) > 1e-10 * abs(exact)


@pytest.mark.parametrize("make", [lambda: _well_solver(64),
                                  lambda: _well_solver(64, do_gen_evp=False),
                                  _tearing_solver],
                         ids=["well-gevp", "well-evp", "tearing"])
def test_zero_steps_is_the_unpolished_path_exactly(make):
    solver = make()
    solver.polish_steps = 0
    for mode in range(3):
        σ, v = solver.solve(mode=mode, saveall=True)
        σ_ref, v_ref, E_ref, V_ref = _unpolished(solver, mode)
        assert σ == σ_ref
        np.testing.assert_array_equal(v, v_ref)
        np.testing.assert_array_equal(solver.E, E_ref)
        np.testing.assert_array_equal(solver.v, V_ref)
        assert solver.system.result[solver.system.eigenvalue] == σ_ref


def test_saveall_polishes_only_the_selected_slot(monkeypatch):
    """saveall stores what the polish returns in the selected mode's slot
    and the unpolished sorted spectrum everywhere else. The polish is
    replaced by a sentinel (σ + 1, 2v), so that the test does not depend
    on whether the real polish moves σ (on the GPU it keeps it)."""
    solver = _tearing_solver()
    mode = 1
    monkeypatch.setattr(Solver, "_polish_pair",
                        lambda self, A, B, σ, v, steps, others=None:
                        (σ + 1, 2 * v))
    σ, v = solver.solve(mode=mode, saveall=True)
    σ_ref, v_ref, E_ref, V_ref = _unpolished(solver, mode)
    assert σ == σ_ref + 1
    np.testing.assert_array_equal(v, 2 * v_ref)
    assert solver.E[mode] == σ_ref + 1
    np.testing.assert_array_equal(solver.v[:, mode], 2 * v_ref)
    others = np.arange(E_ref.size) != mode
    np.testing.assert_array_equal(solver.E[others], E_ref[others])
    np.testing.assert_array_equal(solver.v[:, others], V_ref[:, others])
    assert solver.system.result[solver.system.eigenvalue] == σ


def test_zeroed_mode_is_not_polished(monkeypatch):
    solver = _well_solver(64)
    solver.sorting_strategy = Solver.sorting_strategy.__get__(solver)
    solver.sorting_cutoff = 1.0      # |E_n| >= π²/2: zeroes every one
    called = []
    monkeypatch.setattr(Solver, "_polish_pair",
                        lambda self, *a, **k: called.append(1))
    σ, v = solver.solve(mode=0)
    σ_ref, v_ref, _, _ = _unpolished(solver, 0)
    assert σ == 0 and σ_ref == 0
    np.testing.assert_array_equal(v, v_ref)
    assert not called


def test_mode_changed_in_place_by_the_sort_is_not_polished():
    """An overriding sorting_strategy may change its argument in place
    (tests/test_laguerre_solutions.py flips signs); such a mode keeps the
    sorted value and is not polished back to the raw eigenvalue."""
    solver = _well_solver(64)

    def flip(E):
        E[:] = -E
        return E, np.argsort(np.abs(E))

    solver.sorting_strategy = flip
    σ, _ = solver.solve(mode=0)
    assert σ.real > 0
    solver.polish_steps = 0
    assert σ == solver.solve(mode=0)[0]


@pytest.mark.parametrize("make, modes, polished",
                         [(_tearing_solver, range(4), None),
                          (lambda: _well_solver(256, backend="scipy"),
                           range(3), True)],
                         ids=["tearing", "well"])
def test_returned_vector_follows_the_phase_rule(make, modes, polished):
    """Whichever vector comes back follows the phase rule: the unpolished
    v₀ as the dense solve gave it, or the polished v' with unit norm and
    v₀ᴴv' real and positive, whether or not σ moved. On tearing at N=48
    the moves are at the rounding level, so whether σ is kept or polished
    depends on the backend and is not asserted; on the generalized well
    at N=256 QZ is 5e-11 to 4e-10 off, every mode is polished and the
    polished vectors have the smaller residual, so the rotation branch
    runs."""
    solver = make()
    chose_polished = []
    for mode in modes:
        σ, v = solver.solve(mode=mode)
        σ0, v0, _, _ = _unpolished(solver, mode)
        if polished:
            assert σ != σ0
        if np.array_equal(v, v0):
            chose_polished.append(False)
            continue
        chose_polished.append(True)
        c = np.vdot(v0, v)
        assert c.real > 0
        assert abs(c.imag) <= 1e-14 * abs(c)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-14)
    if polished:
        assert all(chose_polished)


def test_returned_vector_has_the_smaller_residual():
    solver = _tearing_solver()
    for mode in range(4):
        σ, v = solver.solve(mode=mode)
        _, v0, _, _ = _unpolished(solver, mode)
        A, B = _pencil(solver)
        assert solver_mod._rel_residual(A, B, σ, v) <= \
            solver_mod._rel_residual(A, B, σ, v0)


def test_standard_evp_is_polished_with_identity_b(monkeypatch):
    """The standard EVP (boundaries trimmed, B = None). At N=256 QZ is
    2.4e-12 off -π²/2, within the polish's own floor (7e-12), so solve()
    keeps it; B = None is exercised from a start 1e-8 off instead, which
    two steps bring to about 3e-13."""
    exact = -np.pi ** 2 / 2
    solver = _well_solver(256, do_gen_evp=False, backend="scipy")
    seen = []
    polish = Solver._polish_pair

    def spy(self, A, B, *args, **kwargs):
        seen.append(B)
        return polish(self, A, B, *args, **kwargs)

    monkeypatch.setattr(Solver, "_polish_pair", spy)
    σ, v = solver.solve(mode=0)
    assert len(seen) == 1 and seen[0] is None
    # Kept or polished, the result is accurate.
    assert abs(σ - exact) <= 1e-11 * abs(exact)
    A, _ = _pencil(solver)
    assert solver_mod._rel_residual(A, None, σ, v) < 1e-10

    σ0, v0, _, _ = _unpolished(solver, 0)
    σs, vs = _perturbed(σ0, v0, 1e-8, seed=20261004)
    σp, vp = polish(solver, A, None, σs, vs, 2)
    assert abs(σp - exact) <= abs(σs - exact) / 100
    assert abs(σp - exact) <= 1e-11 * abs(exact)


#: Dominant eigenvalue of TearingGyrotropicMHD (normalized=True, kx=0.5,
#: a=1, S=1e4, Pr=0.1, β=1.0, Δβ=0.2, ɣpar=3, ɣper=2, periodic=False) on
#: ChebyshevExtremaGrid(N=256, zmin=-10, zmax=10), from the pencil psecas
#: assembles at commit 12d936c. Computed by Newton refinement of the
#: polished pair with residuals in 80-bit long double (eps 1.1e-19), last
#: correction 1.2e-15 relative; not with code in this repository.
TEARING_N256_MODE0 = 2.760376084293426 - 9.033818091031001e-18j


def _memoised(solver):
    """One dense solve serves every later solve() of this solver."""
    solver.get_matrix1()
    solver.get_matrix2()
    E, V = solver._eig_dense()
    solver._eig_dense = lambda: (E.copy(), V.copy())
    return solver


def test_polish_repairs_the_tearing_mode_at_n256():
    """The motivating case: QZ is 2.6e-9 off the reference, the polish
    3.3e-11. One QZ solve, about 15 s. Started from the reference itself,
    with QZ's vector, the polish keeps it bit for bit (no-degrade rule)."""
    solver = _memoised(_tearing_solver(256))
    ref = TEARING_N256_MODE0
    σ, _ = solver.solve(mode=0)
    assert abs(σ - ref) <= 1e-9 * abs(ref)
    # Relative, so that a LAPACK build with a better QZ does not fail it.
    solver.polish_steps = 0
    σ0, v0 = solver.solve(mode=0)
    assert abs(σ - ref) < abs(σ0 - ref) / 10

    A, B = _pencil(solver)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        σk, _ = solver._polish_pair(A, B, ref, v0, 2)
    assert σk == ref


def test_polished_conjugate_pair_stays_paired():
    """Tearing at N=96: modes 0 and 1 are a conjugate pair, Im > 0 first,
    and stay conjugates after each is polished on its own."""
    solver = _tearing_solver(96)
    σ0, _ = solver.solve(mode=0)
    σ1, _ = solver.solve(mode=1)
    assert σ0.imag > 0
    assert abs(σ0 - np.conj(σ1)) <= 1e-8 * abs(σ0)


def test_singular_second_step_keeps_the_first_step(monkeypatch):
    """A later step that fails keeps the previous step's pair, without a
    warning. From test_infinite_well, where one step converges mode 8 of
    the ChebyshevRoots well so far that A - σB is exactly singular in the
    second; lu_solve is forced to fail from its third call (the second
    step) on, so the test does not depend on that rounding. The polish
    starts 1e-8 off the dense solve's pair, so that the one-step move
    exceeds the floor and the no-degrade rule (jitter 0 with one step)
    does not keep the start."""
    from psecas import ChebyshevRootsGrid

    grid = ChebyshevRootsGrid(64, 0, 1, z='x')
    system = System(grid, variables='phi', eigenvalue='E')
    system.hbar = 1
    system.m = 1
    system.add_equation("-E*phi = -hbar/(2*m)*dx(dx(phi))", boundary=True)
    solver = Solver(grid, system)
    solver.sorting_strategy = lambda E: (E, np.argsort(np.abs(E)))
    mode = 8
    σ0, v0, _, _ = _unpolished(solver, mode)
    A, B = _pencil(solver)
    σs, vs = _perturbed(σ0, v0, 1e-8, seed=8)

    σ1, v1 = solver._polish_pair(A, B, σs, vs, 1)

    real_lu_solve = solver_mod.lu_solve
    calls = []

    def failing_from_the_second_step(*args, **kwargs):
        calls.append(1)
        out = real_lu_solve(*args, **kwargs)
        return out if len(calls) <= 2 else np.full_like(out, np.nan)

    monkeypatch.setattr(solver_mod, "lu_solve", failing_from_the_second_step)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        σ, v = solver._polish_pair(A, B, σs, vs, 2)
    assert len(calls) == 4
    assert σ == σ1
    np.testing.assert_array_equal(v, v1)
    # Improved: σ0, the dense solve's value, is far nearer the pencil's
    # eigenvalue than the 1e-8 start.
    assert abs(σ - σ0) <= abs(σs - σ0) / 100


def test_polished_value_is_kept_when_polished_again():
    """Idempotence: a polished pair fed back into the polish is kept, bit
    for bit. Generalized well at N=256, mode 2, the mode whose second
    polish moves least against its floor (0.5 % of it, against 6-7 % for
    modes 0 and 1), so that the widest margin guards the test."""
    solver = _well_solver(256, backend="scipy")
    mode = 2
    σ, v = solver.solve(mode=mode)
    σ0, _, _, _ = _unpolished(solver, mode)
    assert σ != σ0
    A, B = _pencil(solver)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        σ2, _ = solver._polish_pair(A, B, σ, v, 2)
    assert σ2 == σ


def test_gpu_like_start_is_kept():
    """A start 3.2e-14 off -π²/2 (relative), as the GPU's geev gives on
    the standard well at N=256 (and which the polish used to return
    4.3e-13 off), is kept bit for bit. The start is the exact value
    perturbed by 3.2e-14 at a fixed phase, with the dense solve's
    vector."""
    exact = -np.pi ** 2 / 2
    solver = _well_solver(256, do_gen_evp=False, backend="scipy")
    _, v, _, _ = _unpolished(solver, 0)
    A, _ = _pencil(solver)
    σg = exact * (1 + 3.2e-14 * np.exp(1.1j))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        σk, _ = solver._polish_pair(A, None, σg, v, 2)
    assert σk == σg


def test_one_step_rule_rests_on_the_floor_alone(monkeypatch):
    """With polish_steps = 1 there is no jitter term: the floor is
    evaluated, and the decision follows it alone. From the value solve()
    keeps on the standard well at N=256, one step polishes when the floor
    is forced to 0, keeps when it is forced to inf, and with the real
    floor keeps, the one-step move lying within it."""
    solver = _well_solver(256, do_gen_evp=False, backend="scipy")
    σ, v = solver.solve(mode=0)
    A, _ = _pencil(solver)
    real_floor = solver_mod._rayleigh_floor
    floors = []

    def spy(value):
        def floor(*args):
            floors.append(real_floor(*args))
            return floors[-1] if value is None else value
        return floor

    monkeypatch.setattr(solver_mod, "_rayleigh_floor", spy(0.0))
    σp, _ = solver._polish_pair(A, None, σ, v, 1)
    assert len(floors) == 1
    assert σp != σ

    monkeypatch.setattr(solver_mod, "_rayleigh_floor", spy(np.inf))
    σk, _ = solver._polish_pair(A, None, σ, v, 1)
    assert len(floors) == 2
    assert σk == σ

    monkeypatch.setattr(solver_mod, "_rayleigh_floor", spy(None))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        σk, _ = solver._polish_pair(A, None, σ, v, 1)
    assert len(floors) == 3
    assert σk == σ
    assert abs(σp - σ) <= floors[-1]


#: A start decided by the jitter term, in exact arithmetic. On A =
#: diag(1, 2, 3) from σ₀ = 1.75 with v₀ = (1, 0.19, 0) the two steps give
#: σ₁ = 1.2452 and σ₂ = 1.0332: the move 0.717 is 0.85 of 4|σ₂ - σ₁| =
#: 0.848, so the rule keeps σ₀ (a wrong keep, chosen to show the rule's
#: mechanics; the overlap is 0.99999). From v₀ = (1, 0.01, 0) the steps
#: give 1.0009 and 1.0000, and the move is about 200 times 4|σ₂ - σ₁|.
_JITTER_A = np.diag([1.0, 2.0, 3.0]).astype(complex)
_JITTER_KEEP = (1.75, np.array([1.0, 0.19, 0], dtype=complex))
_JITTER_POLISH = (1.75, np.array([1.0, 0.01, 0], dtype=complex))


def _jitter_solver():
    solver = _well_solver(8)
    # The kept σ₀ = 1.75 is far from an eigenvalue: lift the backward-error
    # floor so that only the keep decision is under test.
    solver.GEVP_RESIDUAL_TOL = np.inf
    return solver


def test_jitter_term_decides_with_the_floor_at_zero(monkeypatch):
    monkeypatch.setattr(solver_mod, "_rayleigh_floor", lambda *a: 0.0)
    solver = _jitter_solver()
    σ0, v0 = _JITTER_KEEP
    σ, _ = solver._polish_pair(_JITTER_A, None, σ0, v0, 2)
    assert σ == σ0
    σ0, v0 = _JITTER_POLISH
    σ, _ = solver._polish_pair(_JITTER_A, None, σ0, v0, 2)
    assert abs(σ - 1) < 1e-6


def test_jitter_factor_changes_the_decision():
    solver = _jitter_solver()
    σ0, v0 = _JITTER_KEEP
    assert solver._polish_pair(_JITTER_A, None, σ0, v0, 2)[0] == σ0
    solver.POLISH_JITTER_FACTOR = 0
    assert abs(solver._polish_pair(_JITTER_A, None, σ0, v0, 2)[0] - 1) < 0.1
    σ0, v0 = _JITTER_POLISH
    solver.POLISH_JITTER_FACTOR = Solver.POLISH_JITTER_FACTOR
    assert abs(solver._polish_pair(_JITTER_A, None, σ0, v0, 2)[0] - 1) < 1e-6
    solver.POLISH_JITTER_FACTOR = np.inf
    assert solver._polish_pair(_JITTER_A, None, σ0, v0, 2)[0] == σ0


def test_rayleigh_floor_by_hand():
    """eps·|y|ᵀ(|A| + |σ||B|)|x| / |yᴴBx| on a 2×2 pencil worked by hand:
    with x = (1, 1) and y = (i, 1), |A||x| = (3, 7) and |B||x| = (1, 2),
    so |y|ᵀ(|A| + |σ||B|)|x| = 10 + 2·3 = 16 at |σ| = 2; Bx = (1, -2) and
    yᴴBx = -2 - i, so the floor is 16/√5 eps."""
    eps = np.finfo(float).eps
    A = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=complex)
    B = np.array([[1.0, 0.0], [0.0, -2.0]], dtype=complex)
    x = np.array([1.0, 1.0], dtype=complex)
    y = np.array([1.0j, 1.0], dtype=complex)
    floor = solver_mod._rayleigh_floor(A, B, 2.0j, x, y)
    assert floor / eps == pytest.approx(16 / np.sqrt(5), rel=1e-15, abs=0)
    # B = None is the identity.
    rng = np.random.default_rng(5)
    n = 6
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    y = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    σ = 0.3 - 1.7j
    assert solver_mod._rayleigh_floor(A, None, σ, x, y) == \
        solver_mod._rayleigh_floor(A, np.eye(n), σ, x, y)


def test_mode_left_infinite_by_the_sort_is_not_polished(monkeypatch):
    solver = _well_solver(64)
    solver.sorting_strategy = lambda E: (E, np.argsort(-np.abs(E)))
    called = []
    monkeypatch.setattr(Solver, "_polish_pair",
                        lambda self, *a, **k: called.append(1))
    σ, _ = solver.solve(mode=0)
    assert np.isinf(σ)
    assert not called


# -- guards ------------------------------------------------------------------

def _refused_twice(call, match):
    """Run call() twice: each keeps the input, and only the first warns."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = call()
        second = call()
    polish = [w for w in caught if "unpolished eigenpair" in str(w.message)]
    assert len(polish) == 1, [str(w.message) for w in caught]
    assert issubclass(polish[0].category, RuntimeWarning)
    assert match in str(polish[0].message)
    return first, second


def test_lu_failure_keeps_the_pair(monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("forced")

    monkeypatch.setattr(solver_mod, "lu_factor", fail)
    solver = _tearing_solver()
    results = _refused_twice(lambda: solver.solve(mode=0), "LU")
    σ_ref, v_ref, _, _ = _unpolished(solver, 0)
    for σ, v in results:
        assert σ == σ_ref
        np.testing.assert_array_equal(v, v_ref)


def test_mode_jump_keeps_the_pair():
    """Start at σ = 1.9 with a vector mostly along the σ = 1 mode: inverse
    iteration goes to the σ = 2 mode, whose vector has overlap 0.29."""
    A = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
    v = np.array([1.0, 0.3, 0, 0], dtype=complex)
    solver = _well_solver(8)
    for B in (None, np.eye(4)):
        solver_mod._polish_warned.clear()
        results = _refused_twice(
            lambda: solver._polish_pair(A, B, 1.9, v, 2), "overlap")
        for σ, w in results:
            assert σ == 1.9 and w is v


def test_nearer_to_another_eigenvalue_keeps_the_pair():
    A = np.diag([1.0, 2.0, 3.0]).astype(complex)
    v = np.array([0, 1.0, 1e-3], dtype=complex)
    solver = _well_solver(8)
    # The polish moves 2.01 to 2; a (fictitious) neighbour at 2 + 1e-6 is
    # nearer to that than the start.
    σ, w = solver._polish_pair(A, None, 2.01, v, 2)
    assert abs(σ - 2) < 1e-12
    results = _refused_twice(
        lambda: solver._polish_pair(A, None, 2.01, v, 2,
                                    others=np.array([1.0, 2 + 1e-6, 3.0])),
        "nearer")
    for σ, w in results:
        assert σ == 2.01 and w is v


def test_backward_error_floor_keeps_the_pair():
    """Started 1e-8 off the dense solve's pair, so that σ' is returned
    rather than the kept σ₀ (whose unchanged pair is not checked)."""
    solver = _tearing_solver()
    solver.GEVP_RESIDUAL_TOL = 0.0      # no pair can pass
    σ_ref, v_ref, _, _ = _unpolished(solver, 0)
    A, B = _pencil(solver)
    σs, vs = _perturbed(σ_ref, v_ref, 1e-8, seed=0)
    results = _refused_twice(lambda: solver._polish_pair(A, B, σs, vs, 2),
                             "backward error")
    for σ, v in results:
        assert σ == σs and v is vs


def test_singular_first_step_keeps_the_pair():
    """σ exactly an eigenvalue: A - σI is exactly singular, which
    lu_factor only warns about, and the first step's vector is not
    finite."""
    A = np.diag([1.0, 2.0, 3.0]).astype(complex)
    v = np.array([0, 1.0, 1e-3], dtype=complex)
    solver = _well_solver(8)
    results = _refused_twice(
        lambda: solver._polish_pair(A, None, 2.0, v, 2), "non-finite")
    for σ, w in results:
        assert σ == 2.0 and w is v


def test_vanishing_rayleigh_denominator_keeps_the_pair():
    """B nilpotent: the first step gives x = e1 with Bx = 0, while the
    left vector, from Bᴴv = (0, 1), is not zero."""
    A = np.eye(2, dtype=complex)
    B = np.array([[0, 1], [0, 0]], dtype=complex)
    v = np.array([1.0, 1.0], dtype=complex)
    solver = _well_solver(8)
    results = _refused_twice(
        lambda: solver._polish_pair(A, B, 0.5, v, 2), "denominator")
    for σ, w in results:
        assert σ == 0.5 and w is v
