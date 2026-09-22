"""
Regression tests for equation substitutions.

A variable may enter an equation only through a substitution, in which case
the raw equation text never mentions it. The parser used to test the
unexpanded text in its per-variable fast path and silently emitted a zero
block for such a variable, producing a wrong eigenvalue with no error.
"""
import numpy as np
import pytest

from psecas import Solver, ChebyshevExtremaGrid, System


class Laplace(System):
    """d^2/dz^2 on [0, 1] with Dirichlet ends: sigma_n = -(n*pi)^2."""

    def make_background(self):
        self.q = 1.0


def _spectrum(equation, substitutions=(), N=48, nvals=4):
    grid = ChebyshevExtremaGrid(N=N, zmin=0, zmax=1)
    system = Laplace(grid, variables='f', eigenvalue='sigma')
    for substitution in substitutions:
        system.add_substitution(substitution)
    system.add_equation(equation, boundary=True)

    E, _ = Solver(grid, system).solve_full()
    E = E[np.isfinite(E)]
    return np.sort(E.real)[::-1][:nvals]


EXACT = -np.array([1, 2, 3, 4], dtype=float) ** 2 * np.pi ** 2


def test_variable_reachable_only_through_substitution():
    """The substituted form must reproduce the analytic spectrum."""
    got = _spectrum("sigma*f = q*G", substitutions=["G = dz(dz(f))"])
    np.testing.assert_allclose(got, EXACT, rtol=1e-10)


def test_substituted_form_matches_direct_form():
    """Substitutions must not change the assembled problem."""
    direct = _spectrum("sigma*f = q*dz(dz(f))")
    substituted = _spectrum("sigma*f = q*G", substitutions=["G = dz(dz(f))"])
    np.testing.assert_allclose(substituted, direct, rtol=1e-12)


def test_substitution_produces_a_nonzero_matrix():
    """The specific failure mode: an all-zero M1 block."""
    grid = ChebyshevExtremaGrid(N=32, zmin=0, zmax=1)
    system = Laplace(grid, variables='f', eigenvalue='sigma')
    system.add_substitution("G = dz(dz(f))")
    system.add_equation("sigma*f = q*G", boundary=True)

    solver = Solver(grid, system)
    solver.get_matrix1()
    assert solver.mat1.count_nonzero() > 0


def test_chained_substitutions_are_expanded():
    """Substitutions defined in terms of one another resolve fully."""
    got = _spectrum("sigma*f = G",
                    substitutions=["D2 = dz(dz(f))", "G = q*D2"])
    np.testing.assert_allclose(got, EXACT, rtol=1e-10)


def test_self_referential_substitution_is_reported():
    """A substitution that cannot terminate raises, rather than hanging."""
    grid = ChebyshevExtremaGrid(N=16, zmin=0, zmax=1)
    system = Laplace(grid, variables='f', eigenvalue='sigma')
    system.add_substitution("G = G + dz(dz(f))")
    system.add_equation("sigma*f = q*G", boundary=True)

    with pytest.raises(ValueError, match="defined in terms of itself"):
        Solver(grid, system)
