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

from psecas import dense_eig

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



# -- backend selection -------------------------------------------------------

def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError):
        dense_eig.resolve_backend("gpu")


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
