# psecas/tests/test_grids_generic.py

from __future__ import annotations
from numpy.polynomial.hermite import hermval

import numpy as np
import pytest

# Adjust imports to your project structure
from psecas.grids import (
    FourierGrid,
    ChebyshevExtremaGrid,
    ChebyshevRootsGrid,
    LegendreExtremaGrid,
    ChebyshevRationalGrid,
    ChebyshevTLnGrid,
    SincGrid,
    HermiteGrid,
    LaguerreGrid,
)


# --------------------------
# Analytic reference: psi and derivatives up to 4
# --------------------------
def psi(x, c):
    return hermval(x, c) * np.exp(-x**2 / 2)

def _herm_phys_derivative_coeff(a):
    """
    For physicists' Hermite polynomials H_n:
        d/dx H_n(x) = 2n H_{n-1}(x)

    If Q(x) = sum a[n] H_n(x),
    then Q'(x) = sum b[n] H_n(x) with:
        b[n] = 2(n+1) a[n+1]
    """
    a = np.asarray(a, dtype=float)
    b = np.zeros_like(a)
    if a.size >= 2:
        b[:-1] = 2.0 * np.arange(1, a.size) * a[1:]
    return b

def _herm_phys_mul_x_coeff(a):
    """
    For physicists' Hermite polynomials H_n:
        x H_n(x) = (1/2) H_{n+1}(x) + n H_{n-1}(x)

    If Q(x) = sum a[n] H_n(x), then xQ(x) = sum b[n] H_n(x) with:
        b[1:]  += 0.5 * a[:-1]
        b[:-1] += (n+1) * a[n+1]  (i.e., np.arange(1, N) * a[1:])
    """
    a = np.asarray(a, dtype=float)
    b = np.zeros_like(a)
    b[1:] += 0.5 * a[:-1]
    if a.size >= 2:
        b[:-1] += np.arange(1, a.size) * a[1:]
    return b

def _psi_derivatives_up_to_4(x, c):
    """
    Returns [psi, psi', psi'', psi''', psi''''] evaluated on x
    using exact coefficient recurrences consistent with hermval (physicists basis),
    and padding to avoid truncation as degree grows.
    """
    x = np.asarray(x, dtype=float)
    c = np.asarray(c, dtype=float)
    expfac = np.exp(-x**2 / 2)

    Q = c.copy()
    out = []
    for _ in range(5):  # orders 0..4
        out.append(hermval(x, Q) * expfac)

        # IMPORTANT: degree can grow by +1 due to -xQ, so pad coefficients
        Q = np.concatenate([Q, [0.0]])

        Qp = _herm_phys_derivative_coeff(Q)
        xQ = _herm_phys_mul_x_coeff(Q)
        Q = Qp - xQ

    return out


# --------------------------
# Grid registry and helpers
# --------------------------
def _make_grid_specs():
    """
    Each spec:
      (name, factory(N)->grid, symmetric_about_midpoint_bool)
    Mark semi-infinite / one-sided grids as non-symmetric.
    """
    zmin, zmax = -2.0, 2.0

    return [
#        ("FourierGrid", lambda N: FourierGrid(N, -np.pi, np.pi), True),
        ("ChebyshevExtremaGrid", lambda N: ChebyshevExtremaGrid(N, zmin, zmax), True),
        ("ChebyshevRootsGrid", lambda N: ChebyshevRootsGrid(N, zmin, zmax), True),
        ("LegendreExtremaGrid", lambda N: LegendreExtremaGrid(N, zmin, zmax), True),
        ("SincGrid", lambda N: SincGrid(N, C=3.0), True),
#        ("HermiteGrid", lambda N: HermiteGrid(N, C=2.0), True),
        ("ChebyshevRationalGrid", lambda N: ChebyshevRationalGrid(N, C=6.0), True),
#        ("ChebyshevTLnGrid", lambda N: ChebyshevTLnGrid(N, C=6.0), False),  # semi-infinite
#        ("LaguerreGrid", lambda N: LaguerreGrid(N, C=2.5), False),          # semi-infinite
    ]


GRID_SPECS = _make_grid_specs()


def _grid_midpoint(g) -> float:
    return 0.5 * (float(g.zmin) + float(g.zmax))


def _overlap_points(g_src, z_tgt: np.ndarray) -> np.ndarray:
    zmin = float(g_src.zmin)
    zmax = float(g_src.zmax)
    z = np.asarray(z_tgt)
    return z[(z >= zmin) & (z <= zmax)]


# --------------------------
# Tests
# --------------------------
@pytest.mark.parametrize("grid_name, grid_factory, expect_symmetric", GRID_SPECS)
def test_grid_coordinate_symmetry(grid_name, grid_factory, expect_symmetric):
    N = 80 if grid_name == "HermiteGrid" else 64
    g = grid_factory(N)
    z = np.asarray(g.zg, dtype=float)

    if not expect_symmetric:
        # For one-sided grids, just check ordering (typical expectation).
        assert np.all(np.diff(z) > 0), f"{grid_name}: expected strictly increasing coordinates."
        return

    zmid = _grid_midpoint(g)
    zr = z - zmid
    err = np.max(np.abs(zr + zr[::-1]))

    # Tight tolerance relative to extent.
    tol = 1e-12 * max(1.0, float(g.zmax) - float(g.zmin))
    assert err < tol, f"{grid_name}: symmetry failed (err={err:g}, tol={tol:g})."


@pytest.mark.parametrize("grid_name, grid_factory, expect_symmetric", GRID_SPECS)
def test_grid_Dn_up_to_4th_order_against_analytic(grid_name, grid_factory, expect_symmetric):
    # Keep sizes moderate; higher derivatives amplify numerical noise.
    N = 96 if grid_name == "FourierGrid" else (80 if grid_name == "HermiteGrid" else 64)
    g = grid_factory(N)

    x = np.asarray(g.zg, dtype=float)
    NN = int(g.NN)
    assert x.shape[0] == NN, f"{grid_name}: zg length mismatch with NN."

    # Coeffs for Hermite combination (moderate degree)
    c = np.zeros(9, dtype=float)
    c[0] = 1.0
    c[2] = -0.3
    c[3] = 0.15
    c[5] = -0.05
    c[7] = 0.02

    ref = _psi_derivatives_up_to_4(x, c)  # orders 0..4
    f0 = ref[0]

    # Tolerances set from measured errors with roughly an order of magnitude
    # of headroom. Worst case across the grids covered here is 4.5e-08 (D1),
    # 2.2e-08 (D2), 1.4e-07 (D3), 3.4e-07 (D4).
    #
    # D3 and D4 used to need 5e-6 and 6e-5 because orders above 2 were formed
    # by repeated multiplication of D1, which loses roughly a digit per
    # order; LegendreExtremaGrid failed even that at 6.2e-05. They are now
    # built with the barycentric recursion, so these tolerances are tight
    # enough to catch a regression back to composition.
    #
    # One-sided grids are allowed looser tolerances due to domain mapping effects.
    tol = {1: 5e-7, 2: 5e-7, 3: 2e-6, 4: 5e-6}
    if not expect_symmetric:  # proxy for one-sided here
        tol = {k: 10.0 * v for k, v in tol.items()}

    for n in (1, 2, 3, 4):
        if not hasattr(g, "D"):
            pytest.skip(f"{grid_name}: grid has no D(n) operator.")
        Dn = g.D(n)

        approx = Dn @ f0
        exact = ref[n]

        denom = np.linalg.norm(exact) + 1e-30
        rel_err = np.linalg.norm(approx - exact) / denom

        assert rel_err < tol[n], (
            f"{grid_name}: D({n}) failed: rel L2 err={rel_err:.3e}, tol={tol[n]:.3e}"
        )


@pytest.mark.parametrize("grid_name, grid_factory, expect_symmetric", GRID_SPECS)
def test_grid_interpolation_between_resolutions(grid_name, grid_factory, expect_symmetric):
    N0 =  96 if grid_name not in [ "HermiteGrid", "LaguerreGrid" ] else 60
    N1 = 384 if grid_name not in [ "HermiteGrid", "LaguerreGrid" ] else 120

    g0 = grid_factory(N0)
    g1 = grid_factory(N1)

    x0 = np.asarray(g0.zg, dtype=float)
    x1 = np.asarray(g1.zg, dtype=float)

    c = np.zeros(9, dtype=float)
    c[0] = 1.0
    c[2] = -0.3
    c[3] = 0.15
    c[5] = -0.05
    c[7] = 0.02

    f0 = psi(x0, c)

    if not hasattr(g0, "interpolate"):
        pytest.skip(f"{grid_name}: grid has no interpolate().")

    # Restrict fine-grid points to coarse domain to satisfy interpolation guards.
    x_common = _overlap_points(g0, x1)
    if x_common.size < max(10, int(0.2 * x1.size)):
        pytest.skip(f"{grid_name}: insufficient domain overlap for interpolation test.")

    f_interp = g0.interpolate(x_common, f0)
    f_exact = psi(x_common, c)

    denom = np.linalg.norm(f_exact) + 1e-30
    rel_err = np.linalg.norm(f_interp - f_exact) / denom

    interp_tol = 5e-7
    if not expect_symmetric:
        interp_tol *= 10.0

    assert rel_err < interp_tol, (
        f"{grid_name}: interpolation failed: rel L2 err={rel_err:.3e}, tol={interp_tol:.3e}"
    )

