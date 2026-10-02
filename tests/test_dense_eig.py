"""
Tests for psecas.dense_eig, the backend-selecting dense eigensolver.

The backend selection, the fallback rules and the zero-row deflation run
everywhere; the deflation is exercised on the CPU by handing it
scipy.linalg.eig for the standard solve. Comparisons with the GPU are
skipped when CuPy or a GPU is missing.
"""
import warnings

import numpy as np
import pytest
import scipy.linalg

from psecas import ChebyshevExtremaGrid, Solver, System, dense_eig

gpu = pytest.mark.skipif(not dense_eig.cupy_available(),
                         reason="needs CuPy with a CUDA GPU")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(dense_eig.ENV_VAR, raising=False)
    monkeypatch.setattr(dense_eig, "_warned", set())


def _pair_error(w, w_ref):
    """Largest distance between each w and a distinct nearest w_ref,
    relative to the spectral radius."""
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(np.abs(w[:, None] - w_ref[None, :]))
    return np.max(np.abs(w[rows] - w_ref[cols])) / max(1, np.abs(w_ref).max())


def _max_residual(A, B, E, V):
    """Largest ||A v - σ B v|| / (||A|| + |σ| ||B||) over the finite pairs."""
    finite = np.isfinite(E)
    E, V = E[finite], V[:, finite]
    r = np.linalg.norm(A @ V - (B @ V) * E, axis=0)
    scale = np.linalg.norm(A, 2) + np.abs(E) * np.linalg.norm(B, 2)
    return np.max(r / (scale * np.linalg.norm(V, axis=0)))


def _constrained_pencil(n=60, k=4, seed=0, complex_=True):
    """A pencil shaped like psecas assembles it: B is the identity except
    for k all-zero rows, which carry constraints in A only."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    if complex_:
        A = A + 1j * rng.standard_normal((n, n))
    B = np.eye(n, dtype=A.dtype)
    rows = rng.choice(n, size=k, replace=False)
    B[rows, :] = 0
    return A, B


def _well_solver(N, backend):
    """Infinite well written so that Psecas solves a generalized EVP whose
    B has zero rows at the two Dirichlet boundaries."""
    grid = ChebyshevExtremaGrid(N=N, zmin=0, zmax=1, z='x')
    system = System(grid, variables='phi', eigenvalue='E')
    system.add_equation("-E*phi = -1/2*dx(dx(phi))", boundary=True)
    return Solver(grid, system, do_gen_evp=True, backend=backend)


# -- backend selection -------------------------------------------------------

def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError):
        dense_eig.resolve_backend("gpu")
    with pytest.raises(ValueError):
        _well_solver(16, backend="cuda")


def test_environment_overrides_auto_only(monkeypatch):
    monkeypatch.setenv(dense_eig.ENV_VAR, "SciPy")
    assert dense_eig.resolve_backend("auto") == "scipy"
    assert dense_eig.resolve_backend("cupy") == "cupy"
    monkeypatch.setenv(dense_eig.ENV_VAR, "cuda")
    with pytest.raises(ValueError):
        dense_eig.resolve_backend("auto")


def test_scipy_backend_is_scipy_exactly():
    A, B = _constrained_pencil(n=30)
    for b in (None, B):
        E, V, used = dense_eig.eig(A, b, backend="scipy", return_backend=True)
        E_ref, V_ref = scipy.linalg.eig(A, b)
        assert used == "scipy"
        np.testing.assert_array_equal(E, E_ref)
        np.testing.assert_array_equal(V, V_ref)


def test_auto_without_cupy_uses_scipy_silently(monkeypatch):
    monkeypatch.setattr(dense_eig, "_probe_cupy",
                        lambda: (False, "CuPy is not installed"))
    A, _ = _constrained_pencil(n=20)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, used = dense_eig.eig(A, backend="auto", min_size=0,
                                   return_backend=True)
    assert used == "scipy"


def test_explicit_cupy_without_cupy_raises(monkeypatch):
    monkeypatch.setattr(dense_eig, "_probe_cupy",
                        lambda: (False, "CuPy is not installed"))
    A, _ = _constrained_pencil(n=20)
    with pytest.raises(RuntimeError, match="not installed"):
        dense_eig.eig(A, backend="cupy")


def _fake_gpu(monkeypatch, solve):
    monkeypatch.setattr(dense_eig, "_probe_cupy", lambda: (True, None))
    monkeypatch.setattr(dense_eig, "_cupy_errors", lambda: (MemoryError,))
    monkeypatch.setattr(dense_eig, "_cupy_solve", solve)


def test_auto_keeps_small_problems_on_the_cpu(monkeypatch):
    def solve(a, b):
        raise AssertionError("small problem sent to the GPU")
    _fake_gpu(monkeypatch, solve)
    A, _ = _constrained_pencil(n=20)
    _, _, used = dense_eig.eig(A, backend="auto", min_size=21,
                               return_backend=True)
    assert used == "scipy"


def test_auto_falls_back_and_warns_once_per_reason(monkeypatch):
    def solve(a, b):
        raise np.linalg.LinAlgError(
            "B is singular (rcond={:.2e})".format(np.random.rand()))
    _fake_gpu(monkeypatch, solve)
    A, B = _constrained_pencil(n=20)
    with pytest.warns(RuntimeWarning, match="falling back to SciPy"):
        E, _, used = dense_eig.eig(A, B, backend="auto", min_size=0,
                                   return_backend=True)
    assert used == "scipy"
    np.testing.assert_array_equal(E, scipy.linalg.eig(A, B)[0])
    # Same failure, different details: no second warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dense_eig.eig(A, B, backend="auto", min_size=0)


def test_auto_falls_back_on_gpu_errors(monkeypatch):
    def solve(a, b):
        raise MemoryError("out of memory")
    _fake_gpu(monkeypatch, solve)
    A, _ = _constrained_pencil(n=20)
    with pytest.warns(RuntimeWarning, match="MemoryError"):
        _, _, used = dense_eig.eig(A, backend="auto", min_size=0,
                                   return_backend=True)
    assert used == "scipy"


def test_auto_does_not_hide_other_errors(monkeypatch):
    def solve(a, b):
        raise TypeError("a bug")
    _fake_gpu(monkeypatch, solve)
    A, _ = _constrained_pencil(n=20)
    with pytest.raises(TypeError):
        dense_eig.eig(A, backend="auto", min_size=0)


def test_solver_records_the_backend(monkeypatch):
    monkeypatch.setattr(dense_eig, "_probe_cupy",
                        lambda: (False, "CuPy is not installed"))
    solver = _well_solver(16, backend="auto")
    solver.solve_full()
    assert solver.last_backend == "scipy"


def test_shift_invert_falls_back_to_scipy_qz(monkeypatch):
    """The QZ fallback of gevp_method='shift-invert' ignores the backend.

    The Channel pencil of tests/test_channel_solver.py (B spans many decades
    through h = exp(-z²/2)) fails the backward-error check of the
    shift-invert solve from N = 34 up; N = 48 leaves a margin.
    """
    from psecas import ChebyshevRationalGrid

    monkeypatch.setattr(dense_eig, "_probe_cupy",
                        lambda: (False, "CuPy is not installed"))

    class Channel(System):
        def make_background(self):
            self.h = np.exp(-self.grid.zg ** 2 / 2)

    grid = ChebyshevRationalGrid(N=48, z='z')
    system = Channel(grid, variables='G', eigenvalue='K2')
    system.add_equation("-h*K2*G = dz(dz(G)) +z*dz(G)", boundary=True)
    solver = Solver(grid, system, do_gen_evp=True,
                    gevp_method='shift-invert', backend='cupy')
    with pytest.warns(RuntimeWarning, match="falling back to QZ"):
        E, _ = solver.solve_full()
    assert solver.last_backend == "scipy"
    E_ref = scipy.linalg.eig(solver.mat1.toarray(), solver.mat2.toarray())[0]
    np.testing.assert_array_equal(E, E_ref)


# -- zero-row deflation (CPU) ------------------------------------------------

@pytest.mark.parametrize("complex_", [True, False])
@pytest.mark.parametrize("k", [0, 1, 5])
def test_deflation_matches_qz(complex_, k):
    A, B = _constrained_pencil(n=50, k=k, complex_=complex_)
    E, V = dense_eig._reduced_geig(A, B, scipy.linalg.eig)
    E_qz = scipy.linalg.eig(A, B, right=False)

    finite = np.isfinite(E)
    assert np.count_nonzero(~finite) == k
    assert np.all(np.isposinf(E[~finite].real))
    assert np.all(V[:, ~finite] == 0)
    np.testing.assert_allclose(np.linalg.norm(V[:, finite], axis=0), 1)

    # QZ reports the infinite eigenvalues as inf or as huge finite numbers.
    E_qz = E_qz[np.abs(E_qz) < 1e8]
    assert E_qz.size == A.shape[0] - k
    assert _pair_error(E[finite], E_qz) < 1e-10
    assert _max_residual(A, B, E, V) < 1e-12


def test_deflation_rejects_rank_deficient_constraints():
    A, B = _constrained_pencil(n=20, k=0)
    B[[3, 7], :] = 0
    A[7, :] = 2 * A[3, :]
    with pytest.raises(np.linalg.LinAlgError, match="rank-deficient"):
        dense_eig._reduced_geig(A, B, scipy.linalg.eig)


def test_deflation_rejects_ill_conditioned_b():
    # Like B = diag(exp(-z²/2)) on a semi-infinite grid: B⁻¹A would swamp
    # the eigenvalues of interest, so the pencil is left to QZ.
    A, B = _constrained_pencil(n=20, k=2)
    B[B == 1] = np.exp(-np.linspace(0, 30, 18) ** 2 / 2)
    with pytest.raises(np.linalg.LinAlgError, match="ill-conditioned"):
        dense_eig._reduced_geig(A, B, scipy.linalg.eig)


# -- backward-error check (CPU, SciPy standing in for the GPU) --------------

def _graded_pencil(L, n=60, k=4):
    """_constrained_pencil with B' = diag(exp(-z²/2)) on [0, L]: B' passes
    the condition limit up to L = 7, but by L = 7 the reduction B'⁻¹A'
    loses most of its digits."""
    A, B = _constrained_pencil(n=n, k=k)
    B[B == 1] = np.exp(-np.linspace(0, L, n - k) ** 2 / 2)
    return A, B


def _fake_gpu_eig(monkeypatch, standard_eig):
    """Fake only the GPU's standard solve, so the reduction and the check
    run for real."""
    monkeypatch.setattr(dense_eig, "_probe_cupy", lambda: (True, None))
    monkeypatch.setattr(dense_eig, "_cupy_errors", lambda: (MemoryError,))
    monkeypatch.setattr(dense_eig, "_cupy_eig", standard_eig)


def test_check_rejects_inaccurate_reduction():
    A, B = _graded_pencil(L=7)
    with pytest.raises(np.linalg.LinAlgError,
                       match="failed the backward-error check"):
        dense_eig._cupy_solve(A, B, scipy.linalg.eig)


def test_check_accepts_accurate_reduction_unchanged():
    A, B = _graded_pencil(L=4)
    E, V = dense_eig._cupy_solve(A, B, scipy.linalg.eig)
    E_ref, V_ref = dense_eig._reduced_geig(A, B, scipy.linalg.eig)
    np.testing.assert_array_equal(E, E_ref)
    np.testing.assert_array_equal(V, V_ref)
    assert _max_residual(A, B, E, V) < 1e-12


def test_auto_falls_back_when_the_check_fails(monkeypatch):
    _fake_gpu_eig(monkeypatch, scipy.linalg.eig)
    A, B = _graded_pencil(L=7)
    with pytest.warns(RuntimeWarning, match="backward-error check"):
        E, V, used = dense_eig.eig(A, B, backend="auto", min_size=0,
                                   return_backend=True)
    assert used == "scipy"
    E_ref, V_ref = scipy.linalg.eig(A, B)
    np.testing.assert_array_equal(E, E_ref)
    np.testing.assert_array_equal(V, V_ref)
    with pytest.raises(np.linalg.LinAlgError, match="backward-error check"):
        dense_eig.eig(A, B, backend="cupy")


def test_auto_keeps_a_good_gpu_result(monkeypatch):
    _fake_gpu_eig(monkeypatch, scipy.linalg.eig)
    A, B = _graded_pencil(L=4)
    for b in (None, B):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, _, used = dense_eig.eig(A, b, backend="auto", min_size=0,
                                       return_backend=True)
        assert used == "cupy"


def test_check_rejects_corrupted_standard_eigenvectors(monkeypatch):
    def corrupted_eig(M):
        w, v = scipy.linalg.eig(M)
        rng = np.random.default_rng(1)
        return w, v + 1e-6 * rng.standard_normal(v.shape)
    _fake_gpu_eig(monkeypatch, corrupted_eig)
    A, _ = _constrained_pencil(n=40)
    with pytest.raises(np.linalg.LinAlgError, match="backward-error check"):
        dense_eig.eig(A, backend="cupy")
    with pytest.warns(RuntimeWarning, match="backward-error check"):
        E, V, used = dense_eig.eig(A, backend="auto", min_size=0,
                                   return_backend=True)
    assert used == "scipy"
    E_ref, V_ref = scipy.linalg.eig(A)
    np.testing.assert_array_equal(E, E_ref)
    np.testing.assert_array_equal(V, V_ref)


def test_check_rejects_zero_and_nan_pairs():
    A, _ = _constrained_pencil(n=10)
    w, v = scipy.linalg.eig(A)
    for bad in (0.0, np.nan):
        v_bad = v.copy()
        v_bad[:, 3] = bad
        with pytest.raises(np.linalg.LinAlgError, match="backward-error"):
            dense_eig._verify(A, None, w, v_bad)


def test_check_refuses_infinite_pairs_on_the_standard_path(monkeypatch):
    # The standard problem has no infinite eigenvalues: (+inf, 0) there is
    # garbage, not a deflated row.
    A, _ = _constrained_pencil(n=10)
    w, v = scipy.linalg.eig(A)
    w[3], v[:, 3] = np.inf, 0
    with pytest.raises(np.linalg.LinAlgError, match="backward-error check"):
        dense_eig._verify(A, None, w, v)

    def inf_eig(M):
        w, v = scipy.linalg.eig(M)
        w[3], v[:, 3] = np.inf, 0
        return w, v
    _fake_gpu_eig(monkeypatch, inf_eig)
    with pytest.raises(np.linalg.LinAlgError, match="backward-error check"):
        dense_eig.eig(A, backend="cupy")


def test_check_refuses_infinite_pairs_without_zero_rows_in_b(monkeypatch):
    def inf_eig(M):
        w, v = scipy.linalg.eig(M)
        w[3], v[:, 3] = np.inf, 0
        return w, v
    _fake_gpu_eig(monkeypatch, inf_eig)
    A, B = _constrained_pencil(n=20, k=0)
    with pytest.raises(np.linalg.LinAlgError, match="backward-error check"):
        dense_eig.eig(A, B, backend="cupy")


def test_check_refuses_inf_with_nonzero_vector():
    A, _ = _constrained_pencil(n=10)
    w, v = scipy.linalg.eig(A)
    w[3] = np.inf
    with pytest.raises(np.linalg.LinAlgError, match="backward-error check"):
        dense_eig._verify(A, None, w, v)


def test_check_refuses_misplaced_or_extra_infinite_pairs():
    A, B = _graded_pencil(L=4)
    E, V = dense_eig._reduced_geig(A, B, scipy.linalg.eig)
    k = np.count_nonzero(~B.any(axis=1))
    dense_eig._verify(A, B, E, V, n_inf=k)
    # One more (+inf, 0) among the finite pairs.
    E_bad, V_bad = E.copy(), V.copy()
    E_bad[0], V_bad[:, 0] = np.inf, 0
    with pytest.raises(np.linalg.LinAlgError, match="backward-error check"):
        dense_eig._verify(A, B, E_bad, V_bad, n_inf=k)
    # A deflated slot that is not (+inf, 0).
    E_bad, V_bad = E.copy(), V.copy()
    V_bad[0, -1] = 1
    with pytest.raises(np.linalg.LinAlgError, match="backward-error check"):
        dense_eig._verify(A, B, E_bad, V_bad, n_inf=k)


def test_check_refuses_nan_pairs_on_the_generalized_path(monkeypatch):
    def nan_eig(M):
        w, v = scipy.linalg.eig(M)
        v[:, 2] = np.nan
        return w, v
    _fake_gpu_eig(monkeypatch, nan_eig)
    A, B = _graded_pencil(L=4)
    with pytest.raises(np.linalg.LinAlgError, match="backward-error check"):
        dense_eig.eig(A, B, backend="cupy")
    with pytest.warns(RuntimeWarning, match="backward-error check"):
        _, _, used = dense_eig.eig(A, B, backend="auto", min_size=0,
                                   return_backend=True)
    assert used == "scipy"


def test_tolerance_matches_the_shift_invert_check():
    assert dense_eig.RESIDUAL_TOL == Solver.GEVP_RESIDUAL_TOL


def test_tolerance_is_pinned():
    # n=60, L=6 gives a backward error of about 1.3e-8: refused at 1e-9,
    # accepted if the tolerance were loosened to 1e-6.
    A, B = _graded_pencil(L=6)
    E, V = dense_eig._reduced_geig(A, B, scipy.linalg.eig)
    finite = np.isfinite(E)
    err = dense_eig._backward_errors(A, B, E[finite], V[:, finite]).max()
    assert 1e-9 < err < 1e-7
    with pytest.raises(np.linalg.LinAlgError, match="backward-error check"):
        dense_eig._cupy_solve(A, B, scipy.linalg.eig)


# -- GPU ---------------------------------------------------------------------

@gpu
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_gpu_standard_matches_scipy(dtype):
    A, _ = _constrained_pencil(n=300, complex_=np.dtype(dtype).kind == "c")
    E, V, used = dense_eig.eig(A, backend="cupy", return_backend=True)
    assert used == "cupy"
    assert E.dtype == np.complex128 and V.dtype == np.complex128
    assert _pair_error(E, scipy.linalg.eig(A, right=False)) < 1e-12
    assert _max_residual(A, np.eye(len(A)), E, V) < 1e-13


@gpu
def test_gpu_generalized_matches_qz():
    A, B = _constrained_pencil(n=300, k=6)
    E, V = dense_eig.eig(A, B, backend="cupy")
    E_qz = scipy.linalg.eig(A, B, right=False)
    E_qz = E_qz[np.abs(E_qz) < 1e8]
    finite = np.isfinite(E)
    assert _pair_error(E[finite], E_qz) < 1e-10
    assert _max_residual(A, B, E, V) < 1e-12


@gpu
def test_gpu_rejects_nan_like_scipy():
    A, _ = _constrained_pencil(n=10)
    A[2, 3] = np.nan
    with pytest.raises(ValueError, match="infs or NaNs"):
        dense_eig.eig(A, backend="cupy")


@gpu
def test_gpu_explicit_backend_raises_on_unsupported_pencil():
    A, B = _constrained_pencil(n=20, k=2)
    B[B == 1] = np.exp(-np.linspace(0, 30, 18) ** 2 / 2)
    with pytest.raises(np.linalg.LinAlgError):
        dense_eig.eig(A, B, backend="cupy")
    with pytest.warns(RuntimeWarning):
        _, _, used = dense_eig.eig(A, B, backend="auto", min_size=0,
                                   return_backend=True)
    assert used == "scipy"


@gpu
def test_gpu_solver_generalized_evp_matches_scipy():
    """Infinite well, E_n = -n²π²/2: GPU and QZ give the same modes."""
    results = {}
    for backend in ("cupy", "scipy"):
        solver = _well_solver(64, backend=backend)
        solver.sorting_strategy = lambda E: (E, np.argsort(np.abs(E)))
        sigmas = [solver.solve(mode=m)[0] for m in range(4)]
        assert solver.last_backend == backend
        results[backend] = np.array(sigmas)
    exact = -(np.arange(1, 5) * np.pi) ** 2 / 2
    np.testing.assert_allclose(results["cupy"].real, exact, rtol=1e-8)
    np.testing.assert_allclose(results["cupy"], results["scipy"], rtol=1e-9)


@gpu
def test_gpu_auto_is_used_above_the_threshold():
    solver = _well_solver(64, backend="auto")
    solver.gpu_min_size = 0
    solver.solve_full()
    assert solver.last_backend == "cupy"
    solver.gpu_min_size = 10 ** 6
    solver.solve_full()
    assert solver.last_backend == "scipy"


@gpu
def test_gpu_refuses_inaccurate_reduction():
    A, B = _graded_pencil(L=7)
    with pytest.raises(np.linalg.LinAlgError, match="backward-error check"):
        dense_eig.eig(A, B, backend="cupy")
