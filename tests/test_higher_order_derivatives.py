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


def test_higher_order_derivatives_against_an_analytic_function():
    """
    D(k) must differentiate, for k above the orders a grid builds directly.

    This replaces a test that asserted D(k) == D(1) @ D(k-1) to 1e-12. That
    was the definition of the old implementation restated as an assertion:
    it could not fail, and it measured nothing about accuracy. It is also
    insensitive to the very thing that matters here - the recursion and
    repeated multiplication agree to ~1e-13 in relative matrix norm, because
    the norm is dominated by the large entries, while differing by three
    orders of magnitude in their action on a function.
    """
    import numpy as np

    N = 48
    g = ChebyshevExtremaGrid(N, -1.0, 1.0)
    z = g.zg

    # A polynomial of degree 7, so every derivative up to 5 is exact.
    f = z**7 - 2*z**5 + 0.5*z**2 + 3.0
    exact = {
        3: 210*z**4 - 120*z**2 + 0.0,
        4: 840*z**3 - 240*z,
        5: 2520*z**2 - 240.0,
    }

    # Interior only: amplification is largest at the boundaries.
    sl = slice(2, -2)
    for k, ref in exact.items():
        got = g.D(k) @ f
        err = np.linalg.norm(got[sl] - ref[sl]) / np.linalg.norm(ref[sl])
        assert err < 1e-6, f"D({k}) relative error {err:.3e}"


def test_higher_order_derivatives_beat_repeated_multiplication():
    """
    Pin the accuracy gain from the barycentric recursion, so a regression to
    composition is caught rather than silently tolerated.
    """
    import numpy as np

    N = 48
    g = ChebyshevExtremaGrid(N, -1.0, 1.0)
    z = g.zg

    f = z**7 - 2*z**5 + 0.5*z**2 + 3.0
    exact = 840*z**3 - 240*z

    def rel(a):
        return np.linalg.norm(a - exact) / np.linalg.norm(exact)

    recursion = rel(g.D(4) @ f)
    composition = rel(np.linalg.matrix_power(np.asarray(g.D(1)), 4) @ f)

    assert recursion < composition


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
