"""
Dense eigensolvers for the full (non-shift-invert) solves.

Two backends are available:

scipy
    ``scipy.linalg.eig``: LAPACK geev for the standard problem, QZ for the
    generalized one. Always available.
cupy
    ``cupy.linalg.eig`` (cuSOLVER Xgeev, CuPy >= 14) on the GPU. CuPy has no
    generalized solver, so ``A v = σ B v`` is first reduced to a standard
    problem on the CPU: the all-zero rows of B -- the boundary-condition
    rows psecas assembles into M₁ only -- are deflated exactly, and the
    remaining nonsingular pencil is turned into ``B'⁻¹ A'`` by LU. Pencils
    outside that class (rank-deficient or ill-conditioned constraints, a
    singular or ill-conditioned B') raise ``LinAlgError``. Every GPU result
    is then checked against the original problem: if any finite eigenpair
    has a normwise backward error above ``RESIDUAL_TOL``, the result is
    discarded and ``LinAlgError`` is raised.

``backend="auto"`` uses CuPy when it is installed and a GPU is present and
the matrix has at least ``min_size`` rows (the GPU does not pay off for
smaller dense problems), and SciPy otherwise. It falls back to SciPy, with a
warning, when the GPU path cannot handle a problem or its result fails the
check. The environment variable ``PSECAS_EIG_BACKEND`` overrides "auto"
without touching caller code.
"""
import functools
import os
import warnings

import numpy as np
import scipy.linalg

#: Accepted values for the ``backend`` argument.
BACKENDS = ("auto", "cupy", "scipy")

#: Environment variable that overrides ``backend="auto"``.
ENV_VAR = "PSECAS_EIG_BACKEND"

#: Smallest matrix dimension "auto" sends to the GPU. Measured on a Quadro
#: P6000 against 10-core OpenBLAS, double precision: 0.6-1.3x at n <= 256,
#: 3-4x from n = 512.
GPU_MIN_SIZE = 400

#: Largest normwise backward error, ‖Av - σBv‖ / ((‖A‖ + |σ|‖B‖) ‖v‖), that a
#: finite GPU eigenpair may have before the result is rejected. Matches
#: Solver.GEVP_RESIDUAL_TOL (psecas.solver is not imported here: it imports
#: this module at load time). QZ stays near 1e-15 by this measure; the
#: reduction B'⁻¹A' reaches 1e-5 and more when B' is badly conditioned.
RESIDUAL_TOL = 1e-9

# Reasons for which a fallback warning has already been issued.
_warned = set()


def resolve_backend(backend):
    """
    Validate ``backend`` and apply the ``PSECAS_EIG_BACKEND`` override.

    Returns "auto", "cupy" or "scipy". The override only replaces "auto",
    so a backend chosen explicitly in code always wins.
    """
    if backend not in BACKENDS:
        raise ValueError(
            "backend must be one of {}, got {!r}".format(BACKENDS, backend))
    if backend == "auto":
        env = os.environ.get(ENV_VAR, "").strip().lower()
        if env:
            if env not in BACKENDS:
                raise ValueError(
                    "{} must be one of {}, got {!r}".format(
                        ENV_VAR, BACKENDS, env))
            backend = env
    return backend


@functools.lru_cache(maxsize=None)
def _probe_cupy():
    """
    Return (True, None) when CuPy can solve on a GPU, else (False, reason).

    Cached: the probe creates the CUDA context once, on the first solve that
    would use the GPU, and never again.
    """
    try:
        import cupy
    except ImportError as exc:
        return False, "CuPy is not installed ({})".format(exc)
    try:
        if not cupy.cuda.is_available():
            return False, "CuPy found no usable CUDA device"
        if not hasattr(cupy.linalg, "eig"):
            return False, ("CuPy {} has no cupy.linalg.eig (needs CuPy >= 14)"
                           .format(cupy.__version__))
        # A tiny solve catches broken driver/toolkit setups here rather
        # than in the middle of a calculation.
        cupy.linalg.eig(cupy.asarray([[0.0, 1.0], [-1.0, 0.0]]))
        cupy.cuda.Device().synchronize()
    except Exception as exc:
        return False, "CuPy GPU check failed ({}: {})".format(
            type(exc).__name__, exc)
    return True, None


def cupy_available():
    """True when CuPy is installed and can run eigensolves on a GPU."""
    return _probe_cupy()[0]


def _cupy_errors():
    """Exceptions from the CuPy/CUDA stack that "auto" recovers from."""
    import cupy
    from cupy_backends.cuda.api.runtime import CUDARuntimeError
    from cupy_backends.cuda.libs.cublas import CUBLASError
    from cupy_backends.cuda.libs.cusolver import CUSOLVERError
    return (cupy.cuda.memory.OutOfMemoryError, CUDARuntimeError,
            CUBLASError, CUSOLVERError)


def _check_finite(*arrays):
    # CuPy does not check its input, and cuSOLVER reports NaN/inf only as an
    # opaque internal error. Raise what scipy.linalg.eig would.
    for arr in arrays:
        if not np.all(np.isfinite(arr)):
            raise ValueError("array must not contain infs or NaNs")


def _work_dtype(*arrays):
    dtype = np.result_type(*arrays)
    if dtype.kind not in "fc":
        dtype = np.result_type(dtype, np.float64)
    return dtype


def _cupy_eig(a):
    """Standard EVP on the GPU. Returns host arrays (w, v)."""
    import cupy
    import cupyx

    # By default CuPy ignores non-convergence and returns garbage.
    with cupyx.errstate(linalg="raise"):
        w, v = cupy.linalg.eig(cupy.asarray(a))
    return w.get(), v.get()


def _cond_limit(dtype):
    """
    Largest condition number accepted for the constraint block and for B'.
    Keeps at least about 4 significant digits in the working precision.
    """
    return 1e-4 / np.finfo(dtype).eps


def _deflate_constraints(A, B, C):
    """
    Exact deflation of the all-zero rows ``C`` of B.

    Those rows are constraints ``A[C, :] v = 0``. A pivoted QR of the
    constraint block picks |C| pivot columns ``Jb``, which are eliminated as
    ``v[Jb] = T @ v[Ji]``. The remaining rows of the pencil give the reduced
    problem ``Ap v' = σ Bp v'`` on the columns ``Ji``.

    Raises LinAlgError when the constraints are rank-deficient or
    ill-conditioned.
    """
    n = A.shape[0]
    k = C.size
    Ac = A[C, :]
    R_fac, piv = scipy.linalg.qr(Ac, mode="r", pivoting=True)
    d = np.abs(np.diag(R_fac))
    if d[0] == 0 or d[k - 1] <= np.finfo(A.dtype).eps * n * d[0]:
        raise np.linalg.LinAlgError("constraint rows are rank-deficient")
    Jb = piv[:k]
    Ji = np.setdiff1d(np.arange(n), Jb)
    R = np.setdiff1d(np.arange(n), C)
    Cb = Ac[:, Jb]
    if not np.linalg.cond(Cb) <= _cond_limit(A.dtype):
        raise np.linalg.LinAlgError("constraint block is ill-conditioned")
    T = scipy.linalg.solve(Cb, -Ac[:, Ji])
    AR, BR = A[R, :], B[R, :]
    Ap = AR[:, Ji] + AR[:, Jb] @ T
    Bp = BR[:, Ji] + BR[:, Jb] @ T
    return Ji, Jb, T, Ap, Bp


def _solve_nonsingular(B, A):
    """
    Return ``B⁻¹ A`` via LU. Raises LinAlgError if B is singular or too
    ill-conditioned for the eigenvalues of ``B⁻¹ A`` to be meaningful.
    """
    from scipy.linalg import LinAlgWarning
    from scipy.linalg.lapack import get_lapack_funcs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinAlgWarning)
        lu, piv = scipy.linalg.lu_factor(B, check_finite=False)
    gecon, = get_lapack_funcs(("gecon",), (lu,))
    rcond, info = gecon(lu, np.linalg.norm(B, 1), norm="1")
    if info != 0 or not rcond * _cond_limit(B.dtype) >= 1:
        raise np.linalg.LinAlgError(
            "B (after deflating its zero rows) is singular or "
            "ill-conditioned (rcond={:.2e})".format(rcond))
    return scipy.linalg.lu_solve((lu, piv), A, check_finite=False)


def _reduced_geig(A, B, standard_eig):
    """
    Generalized EVP ``A v = σ B v`` via exact deflation and ``B'⁻¹ A'``.

    ``standard_eig(M)`` solves the resulting standard problem and returns
    (w, v); it is a parameter so the reduction can be tested without a GPU.

    Returns (E, V) like ``scipy.linalg.eig(A, B)``, except that the order
    differs: the finite eigenvalues come first, followed by ``+inf`` for each
    deflated row (the infinite eigenvalues, as QZ reports them). Columns of
    V for finite eigenvalues have unit 2-norm; those for the infinite ones
    are zero.

    The condition limits on the constraint block and on B' only refuse the
    hopeless cases cheaply; B' can pass them and still give eigenpairs with
    a large backward error. The result is not checked here: ``_cupy_solve``
    verifies it against the original pencil and raises LinAlgError if it
    fails.
    """
    n = A.shape[0]
    C = np.flatnonzero(~B.any(axis=1))
    if C.size == 0:
        return standard_eig(_solve_nonsingular(B, A))

    Ji, Jb, T, Ap, Bp = _deflate_constraints(A, B, C)
    m = Ji.size
    cdtype = np.result_type(A.dtype, np.complex64)
    E = np.full(n, np.inf, dtype=cdtype)
    V = np.zeros((n, n), dtype=cdtype)
    if m:
        w, v = standard_eig(_solve_nonsingular(Bp, Ap))
        E[:m] = w
        V[Ji, :m] = v
        V[Jb, :m] = T @ v
        V[:, :m] /= np.linalg.norm(V[:, :m], axis=0)
    return E, V


def _backward_errors(A, B, E, V):
    """
    Normwise backward error of each eigenpair (E[j], V[:, j]) of the pencil
    (A, B), as in solver._eig_shift_invert:

        ‖Av - σBv‖ / ((‖A‖ + |σ|‖B‖) ‖v‖),

    with Frobenius matrix norms; B = None means the identity. A zero vector
    or a non-finite pair gives NaN or inf, which fails any tolerance test.

    Computed on the host. It costs one or two matrix products, small next to
    the O(n³) eigensolve with its much larger constant.
    """
    AV = A @ V
    if B is None:
        R = AV - V * E
        norm_B = np.sqrt(A.shape[0])
    else:
        R = AV - (B @ V) * E
        norm_B = np.linalg.norm(B)
    scale = (np.linalg.norm(A) + np.abs(E) * norm_B) \
        * np.linalg.norm(V, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.linalg.norm(R, axis=0) / scale


def _verify(A, B, E, V, n_inf=0):
    """
    Raise LinAlgError unless every eigenpair other than the expected
    infinite ones has a backward error of at most RESIDUAL_TOL.

    ``n_inf`` is the number of infinite eigenvalues the result may contain:
    the deflated zero rows of B on the reduction path, 0 for the standard
    problem and for a B without zero rows. Exactly the last ``n_inf`` pairs
    must be ``+inf`` with a zero vector, as ``_reduced_geig`` returns them;
    they are not checked further. Every other pair is checked, so an
    infinite or NaN eigenvalue anywhere else fails.
    """
    m = E.size - n_inf
    tail_E, tail_V = E[m:], V[:, m:]
    if not (np.all(np.isposinf(tail_E.real) & (tail_E.imag == 0))
            and not tail_V.any()):
        raise np.linalg.LinAlgError(
            "GPU eigenpairs failed the backward-error check "
            "(infinite eigenvalues not where the deflation puts them)")
    if m == 0:
        return
    errors = _backward_errors(A, B, E[:m], V[:, :m])
    if not np.all(errors <= RESIDUAL_TOL):
        raise np.linalg.LinAlgError(
            "GPU eigenpairs failed the backward-error check "
            "(max {:.1e} > {:.0e})".format(np.max(errors), RESIDUAL_TOL))


def _cupy_solve(a, b, standard_eig=None):
    """
    Solve on the GPU path and verify the result against ``(a, b)``.

    ``standard_eig(M)`` solves the standard problems and returns host arrays
    (w, v); it defaults to ``_cupy_eig`` and is a parameter so the whole path,
    check included, can be tested without a GPU.
    """
    if standard_eig is None:
        standard_eig = _cupy_eig
    if b is None:
        _check_finite(a)
        A = np.asarray(a, dtype=_work_dtype(a))
        E, V = standard_eig(A)
        _verify(A, None, E, V)
        return E, V
    _check_finite(a, b)
    dtype = _work_dtype(a, b)
    A = np.asarray(a, dtype=dtype)
    B = np.asarray(b, dtype=dtype)
    E, V = _reduced_geig(A, B, standard_eig)
    # _reduced_geig puts one +inf per all-zero row of B at the end.
    _verify(A, B, E, V, n_inf=np.count_nonzero(~B.any(axis=1)))
    return E, V


def _warn_fallback(reason):
    # Keyed on the message without its parenthesized details (rcond, sizes),
    # so a resolution sweep warns once per kind of failure, not per solve.
    key = reason.split(" (")[0]
    if key not in _warned:
        _warned.add(key)
        warnings.warn("GPU eigensolver not used, falling back to SciPy: "
                      + reason, RuntimeWarning, stacklevel=3)


def eig(a, b=None, backend="auto", min_size=None, return_backend=False):
    """
    Dense eigenvalues and right eigenvectors of ``a`` or of the pencil
    ``(a, b)``, returned as (E, V) like ``scipy.linalg.eig``.

    Parameters
    ----------
    a, b : (n, n) array_like
        The matrices; ``b=None`` solves the standard problem.
    backend : {"auto", "cupy", "scipy"}
        "scipy" always uses SciPy. "cupy" always uses the GPU and raises if
        it cannot (CuPy missing, unsupported pencil, GPU error, a result
        that fails the check below). "auto" uses the GPU when CuPy is
        available and ``n >= min_size``, and falls back to SciPy with a
        one-time warning when the GPU path fails.
        ``PSECAS_EIG_BACKEND`` overrides "auto".
    min_size : int, optional
        Smallest n that "auto" sends to the GPU (default GPU_MIN_SIZE).
        Ignored by the explicit backends.
    return_backend : bool
        Also return the backend that produced the result.

    Notes
    -----
    The eigenvalue order is backend-specific; sort or pair them rather than
    comparing positions. For a generalized problem the GPU path returns
    ``+inf`` with a zero eigenvector for each infinite eigenvalue.

    Every GPU result is verified against ``(a, b)`` before it is returned:
    each finite eigenpair must have a normwise backward error
    ``‖Av - σBv‖ / ((‖A‖ + |σ|‖B‖) ‖v‖)`` of at most ``RESIDUAL_TOL``
    (B = I for the standard problem). A result that fails raises
    ``LinAlgError``, so "cupy" raises and "auto" falls back to SciPy.
    The tolerance assumes double precision: float32 and complex64 input
    stays in single precision on the GPU path and always fails the check,
    so "cupy" raises for it and "auto" falls back to SciPy.
    """
    backend = resolve_backend(backend)
    if min_size is None:
        min_size = GPU_MIN_SIZE
    n = np.shape(a)[0]

    used = "scipy"
    if backend == "cupy":
        ok, reason = _probe_cupy()
        if not ok:
            raise RuntimeError("backend='cupy' requested but " + reason)
        E, V = _cupy_solve(a, b)
        used = "cupy"
    elif backend == "auto" and n >= min_size and cupy_available():
        try:
            E, V = _cupy_solve(a, b)
            used = "cupy"
        except np.linalg.LinAlgError as exc:
            _warn_fallback(str(exc))
        except _cupy_errors() as exc:
            _warn_fallback("{}: {}".format(type(exc).__name__, exc))

    if used == "scipy":
        E, V = scipy.linalg.eig(a, b)

    if return_backend:
        return E, V, used
    return E, V
