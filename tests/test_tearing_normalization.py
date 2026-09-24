"""
The two pressure-anisotropy formulations of TearingGyrotropicMHD.

normalized=False solves for the pressure anisotropy delta-Delta-p directly.
normalized=True solves for the rescaled variable

    delta-pi = delta-Delta-p / sqrt(Gamma_beta)

and divides the anisotropy equation through by sqrt(Gamma_beta), which makes
the coupling to the momentum equations symmetric. It is a change of
variables, so the eigenvalues must be identical.

These two formulations used to live in separate files, the second a 657-line
copy of the first differing in 41 lines.
"""
import numpy as np
import pytest
from scipy.linalg import eig

from psecas import Solver, ChebyshevExtremaGrid
from psecas.systems.tearing_instability import TearingGyrotropicMHD


PARAMS = dict(kx=0.5, a=1, S=1e4, Pr=0.1, β=1.0, Δβ=0.2,
              ɣpar=3, ɣper=2, periodic=False)


def _solver(normalized, N=48):
    grid = ChebyshevExtremaGrid(N=N, zmin=-10, zmax=10)
    system = TearingGyrotropicMHD(grid, normalized=normalized, **PARAMS)
    solver = Solver(grid, system)
    solver.get_matrix1()
    solver.get_matrix2()
    return system, solver


def _dominant(solver):
    E, _ = eig(solver.mat1.toarray(), solver.mat2.toarray())
    E = E[np.isfinite(E)]
    E = E[E.real > 0]
    return E[np.argmax(E.real)]


def test_variable_naming_follows_the_flag():
    plain, _ = _solver(normalized=False)
    scaled, _ = _solver(normalized=True)

    assert plain.variables[-1] == "ddp"
    assert scaled.variables[-1] == "dpi"
    assert r"\Delta p" in plain.labels[-1]
    assert r"\pi" in scaled.labels[-1]


def test_both_formulations_give_the_same_eigenvalue():
    """The whole point: a change of variables cannot change the physics."""
    _, plain = _solver(normalized=False)
    _, scaled = _solver(normalized=True)

    np.testing.assert_allclose(_dominant(plain), _dominant(scaled), rtol=1e-8)


def test_normalization_rescales_the_anisotropy_terms():
    """
    The scaled form must carry sqrt(Gamma_beta) in the momentum coupling and
    divide the anisotropy equation by it, rather than carrying Gamma_beta in
    one direction and 1 in the other.
    """
    plain, _ = _solver(normalized=False)
    scaled, _ = _solver(normalized=True)

    scaled_eqs = " ".join(scaled.equations)
    plain_eqs = " ".join(plain.equations)

    assert "sqrtG0" in scaled_eqs
    assert "sqrtG0" not in plain_eqs
    assert "Γβ" in plain_eqs
    assert scaled.sqrtG0 == pytest.approx(np.sqrt(scaled.Γβ))


def test_lambda_matches_the_closed_form_at_zero_sigma():
    """
    With sigma = 0 the boundary-decay rate reduces to the expression the
    separate file hard-coded, so merging the two did not change it.
    """
    system, _ = _solver(normalized=True)

    β, Δβ, ɣpar, ɣper = PARAMS['β'], PARAMS['Δβ'], PARAMS['ɣpar'], PARAMS['ɣper']
    expected = PARAMS['kx'] * PARAMS['a'] * np.sqrt(
        (2 - Δβ) / (2 + (ɣpar + ɣper - 2) * β + ɣpar * Δβ)
    )

    assert system.λ == pytest.approx(expected)
