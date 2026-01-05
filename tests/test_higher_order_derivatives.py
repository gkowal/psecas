import numpy as np

from psecas.grids.chebyshev_extrema import ChebyshevExtremaGrid

def _rel_err(a, b, eps=1e-14):
    denom = np.linalg.norm(b) + eps
    return np.linalg.norm(a - b) / denom


def test_eager_build_max_derivative_order():
    """
    If max_derivative_order is passed, the grid should have those operators
    readily available after construction.
    """
    N = 48
    g = ChebyshevExtremaGrid(N, -1.0, 1.0, max_derivative_order=5)

    # Ensure D(5) can be obtained and that the canonical list is long enough.
    D5 = g.D(5)
    assert D5.shape == (N + 1, N + 1)

    # If you expose g.d as the list, assert it contains at least 6 entries (0..5).
    assert hasattr(g, "d")
    assert len(g.d) >= 6


def test_lazy_extension_via_D_accessor():
    """
    Even if the grid is constructed with default max order, requesting a higher
    derivative should extend it.
    """
    N = 32
    g = ChebyshevExtremaGrid(N, -1.0, 1.0)  # default max_derivative_order=2

    # Should extend on demand
    D4 = g.D(4)
    assert D4.shape == (N + 1, N + 1)
    assert len(g.d) >= 5  # 0..4


def test_higher_order_composition_consistency():
    """
    For higher orders, extension is defined by composition. Validate that
    D(k) ~= D(1) @ D(k-1) for a few k.
    """
    N = 40
    g = ChebyshevExtremaGrid(N, -1.0, 1.0)

    # Force extension
    g.D(5)

    D1 = g.D(1)
    for k in (3, 4, 5):
        lhs = g.D(k)
        rhs = D1 @ g.D(k - 1)

        # Tolerance: matrix norms can be large; use relative norm check.
        # This should be very tight if constructed by exact multiplication.
        err = _rel_err(lhs, rhs)
        assert err < 1e-12


def test_compatibility_d1_d2_properties():
    """
    Legacy accessors should remain consistent with the new API.
    """
    N = 24
    g = ChebyshevExtremaGrid(N, -1.0, 1.0)

    # d1/d2 should exist and match D(1)/D(2)
    assert _rel_err(g.d1, g.D(1)) < 1e-14
    assert _rel_err(g.d2, g.D(2)) < 1e-14


def test_functional_third_derivative_on_polynomial():
    """
    Functional correctness check on a polynomial.

    Note: High-order differentiation matrices on Chebyshev extrema grids become
    ill-conditioned as N grows; roundoff amplification can dominate. This test
    therefore uses a tolerance scaled to machine precision and an estimate of
    operator amplification.
    """
    import numpy as np
    from psecas.grids.chebyshev_extrema import ChebyshevExtremaGrid

    N = 48
    g = ChebyshevExtremaGrid(N, -1.0, 1.0, max_derivative_order=3)

    z = g.zg
    f = z**6 - 2*z**4 + 0.5*z**2 + 3.0
    f3_exact = 120*z**3 - 48*z

    D3 = g.D(3)
    f3_num = D3 @ f

    # Ignore endpoints (largest amplification typically occurs near boundaries)
    sl = slice(2, -2)

    resid = f3_num[sl] - f3_exact[sl]
    resid_norm = np.linalg.norm(resid, ord=2)

    # Scale by an estimate of numerical amplification: ||D3|| * ||f||
    # Use inf-norm as a cheap upper bound proxy.
    eps = np.finfo(float).eps
    amp = np.linalg.norm(D3, ord=np.inf) * np.linalg.norm(f, ord=np.inf)

    # Allow a generous factor for accumulation and conditioning effects.
    # This is not a convergence test; it is a sanity check against gross errors.
    assert resid_norm <= 1e5 * eps * amp + 1e-12
