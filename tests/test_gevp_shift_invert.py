"""
Tests for Solver(gevp_method='shift-invert'), the dense generalized solve
that runs the standard eigensolver on (M₁ - sM₂)⁻¹M₂ instead of QZ.

It must reproduce QZ: the same finite eigenvalues, eigenvectors that satisfy
the original pencil, and the same number of infinite eigenvalues when
boundary rows make M₂ singular.  Where zgeev's balancing ruins the
eigenvectors it must fall back to QZ rather than return them.
"""
import warnings

import numpy as np
import pytest
from scipy.linalg import eig

from psecas import (Solver, System, ChebyshevRationalGrid,
                    ChebyshevExtremaGrid)
from psecas.systems.mti import MagnetoThermalInstability
from psecas.systems.tearing_instability import TearingClassicalMHD


class Channel(System):
    """-h(z) K2 G = G'' + z G' , with h = exp(-z^2/2) so that M2 = -diag(h)."""

    def make_background(self):
        self.h = np.exp(-self.grid.zg ** 2 / 2)


def _tearing(**kwargs):
    grid = ChebyshevRationalGrid(N=150, C=0.2)
    return Solver(grid, TearingClassicalMHD(grid, kx=0.5), **kwargs)


def _mti(**kwargs):
    grid = ChebyshevExtremaGrid(N=64, zmin=0, zmax=1)
    system = MagnetoThermalInstability(grid, beta=1e5, Kn0=200, kx=4 * np.pi)
    return Solver(grid, system, **kwargs)


def _channel(**kwargs):
    grid = ChebyshevRationalGrid(N=199, z='z')
    system = Channel(grid, variables='G', eigenvalue='K2')
    system.add_equation("-h*K2*G = dz(dz(G)) +z*dz(G)", boundary=True)
    return Solver(grid, system, do_gen_evp=True, **kwargs)


def _backward_errors(solver, Σ, V):
    A = solver.mat1.toarray()
    B = solver.mat2.toarray()
    f = np.isfinite(Σ)
    R = A @ V[:, f] - (B @ V[:, f]) * Σ[f]
    scale = (np.linalg.norm(A) + np.abs(Σ[f]) * np.linalg.norm(B)) \
        * np.linalg.norm(V[:, f], axis=0)
    return np.linalg.norm(R, axis=0) / scale


def _leading(Σ, count=6):
    Σ = Σ[np.isfinite(Σ) & (np.abs(Σ) < 1e4)]
    return Σ[np.lexsort((Σ.imag, -Σ.real))][:count]


@pytest.mark.parametrize("make", [_tearing, _mti], ids=["tearing", "mti"])
def test_matches_qz(make):
    qz = make()
    si = make(gevp_method='shift-invert')
    assert qz.do_gen_evp and si.do_gen_evp

    Σq, _ = qz.solve_full()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        Σs, Vs = si.solve_full()

    assert Σs.shape == Σq.shape
    assert np.sum(~np.isfinite(Σs)) == np.sum(~np.isfinite(Σq))
    for σ in _leading(Σq):
        assert np.min(np.abs(Σs - σ)) <= 1e-8 * max(abs(σ), 1.0)
    assert np.all(_backward_errors(si, Σs, Vs) <= Solver.GEVP_RESIDUAL_TOL)


def test_singular_m2_gives_infinite_eigenvalues():
    """MTI's boundary rows make M₂ singular; those must come back infinite."""
    solver = _mti(gevp_method='shift-invert')
    Σ, _ = solver.solve_full()
    B = solver.mat2.toarray()
    assert np.linalg.matrix_rank(B) < B.shape[0]
    assert np.sum(~np.isfinite(Σ)) == B.shape[0] - np.linalg.matrix_rank(B)


def test_balancing_failure_falls_back_to_qz():
    """
    The channel's M₂ spans ~150 decades, which breaks zgeev's eigenvectors.
    The result must be QZ's, announced by a RuntimeWarning.
    """
    qz = _channel()
    si = _channel(gevp_method='shift-invert')

    Σq, Vq = qz.solve_full()
    with pytest.warns(RuntimeWarning, match="falling back to QZ"):
        Σs, Vs = si.solve_full()

    np.testing.assert_array_equal(Σs, Σq)
    np.testing.assert_array_equal(Vs, Vq)


def test_solve_uses_the_method():
    qz = _mti()
    si = _mti(gevp_method='shift-invert')
    σq, _ = qz.solve()
    σs, vs = si.solve()
    assert σq.real > 0
    assert abs(σs - σq) <= 1e-8 * max(abs(σq), 1.0)
    A, B = si.mat1.toarray(), si.mat2.toarray()
    assert np.linalg.norm(A @ vs - σs * (B @ vs)) \
        <= 1e-9 * np.linalg.norm(A) * np.linalg.norm(vs)


def test_custom_shift():
    Σq, _ = _tearing().solve_full()
    Σs, _ = _tearing(gevp_method='shift-invert',
                     gevp_shift=0.3 - 0.2j).solve_full()
    for σ in _leading(Σq):
        assert np.min(np.abs(Σs - σ)) <= 1e-8 * max(abs(σ), 1.0)


def test_standard_evp_is_unaffected():
    """With no boundaries and no forced GEVP, both methods run the same eig."""
    grid = ChebyshevExtremaGrid(N=32, zmin=0, zmax=1)
    system = System(grid, variables='f', eigenvalue='sigma')
    system.add_equation("sigma*f = dz(dz(f))")
    Σq, _ = Solver(grid, system).solve_full()
    solver = Solver(grid, system, gevp_method='shift-invert')
    assert not solver.do_gen_evp
    Σs, _ = solver.solve_full()
    np.testing.assert_array_equal(Σs, Σq)


def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="gevp_method"):
        _tearing(gevp_method='cupy')
