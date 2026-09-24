"""
The namespace that equation and boundary strings are evaluated in.

It used to expose only {"__import__": builtins.__import__}, described in a
comment as "a restricted environment". That was no restriction at all, and
it broke any expression that emitted a warning.
"""
import numpy as np
import pytest

from psecas import Solver, System, ChebyshevExtremaGrid


class Laplace(System):
    def make_background(self):
        self.q = 1.0


def _spectrum(equation, n=2):
    grid = ChebyshevExtremaGrid(N=48, zmin=0, zmax=1)
    system = Laplace(grid, variables='f', eigenvalue='sigma')
    system.add_equation(equation, boundary=True)

    E, _ = Solver(grid, system).solve_full()
    E = E[np.isfinite(E)]
    return np.sort(E.real)[::-1][:n]


EXACT = -np.array([1.0, 2.0]) ** 2 * np.pi ** 2


@pytest.mark.parametrize("factor", [
    "sqrt(4.0)/2",
    "exp(0.0)",
    "cos(0.0)",
    "abs(-1.0)",
    "np.sqrt(1.0)",
    "numpy.float64(1.0)",
])
def test_numpy_names_are_available_in_equations(factor):
    """sqrt(), exp() and friends should just work in an equation string."""
    got = _spectrum("sigma*f = {}*q*dz(dz(f))".format(factor))
    np.testing.assert_allclose(got, EXACT, rtol=1e-10)


def test_pi_is_available():
    got = _spectrum("sigma*f = pi/pi*q*dz(dz(f))")
    np.testing.assert_allclose(got, EXACT, rtol=1e-10)


def test_an_expression_that_warns_still_evaluates():
    """
    Dividing by a grid that contains zero emits a RuntimeWarning, and the
    warning machinery imports through the evaluating frame's builtins. With
    those trimmed away it failed with KeyError: '__import__' - which is how
    the spherical Bessel test broke.
    """
    from psecas.solver import _EVAL_GLOBALS

    grid = ChebyshevExtremaGrid(64, 0, 1.0, z='r')
    assert np.any(np.asarray(grid.zg) == 0.0)      # the premise

    with np.errstate(divide='ignore', invalid='ignore'):
        result = eval("-2/grid.zg*grid.D(1).T",
                      dict(_EVAL_GLOBALS), {"grid": grid})

    assert result.shape == (grid.NN, grid.NN)


def test_unknown_name_still_raises_NameError():
    """Error reporting for a genuine typo must be unaffected."""
    grid = ChebyshevExtremaGrid(N=16, zmin=0, zmax=1)
    system = Laplace(grid, variables='f', eigenvalue='sigma')
    system.add_equation("sigma*f = notdefined*dz(dz(f))")

    with pytest.raises(NameError, match="notdefined"):
        Solver(grid, system)
