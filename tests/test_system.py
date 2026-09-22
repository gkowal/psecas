"""Tests for System's equation and boundary bookkeeping."""
import pytest

from psecas import System, ChebyshevExtremaGrid


@pytest.fixture
def grid():
    return ChebyshevExtremaGrid(N=16, zmin=0, zmax=1)


def test_equation_without_a_variable_on_the_lhs_is_rejected(grid):
    """
    Silently dropping the equation left an empty slot, and the failure then
    surfaced far away as an IndexError inside get_matrix1().
    """
    system = System(grid, variables='f', eigenvalue='sigma')

    with pytest.raises(ValueError, match="left-hand side"):
        system.add_equation("sigma*G = -q*G")


def test_equation_without_an_equal_sign_is_rejected(grid):
    system = System(grid, variables='f', eigenvalue='sigma')

    with pytest.raises(ValueError, match="equal sign"):
        system.add_equation("sigma*f - dz(dz(f))")


def test_two_variables_on_the_lhs_are_rejected(grid):
    system = System(grid, variables=['f', 'g'], eigenvalue='sigma')

    with pytest.raises(RuntimeError, match="[Oo]nly one variable"):
        system.add_equation("sigma*f + g = 0")


def test_lhs_matching_uses_whole_identifiers(grid):
    """'vx' must not be matched inside 'dvx' and claim the wrong slot."""
    system = System(grid, variables=['vx', 'dvx'], eigenvalue='sigma')

    system.add_equation("sigma*dvx = vx")

    assert system.equations[0] == ''
    assert system.equations[1] == "sigma*dvx = vx"


def test_add_equation_records_boundary_flags(grid):
    system = System(grid, variables='f', eigenvalue='sigma')

    system.add_equation("sigma*f = dz(dz(f))", boundary=True)

    assert system.boundaries[0] is True
    assert system.extra_binfo[0] == ['Dirichlet', 'Dirichlet']


def test_add_boundary_rejects_an_unknown_variable(grid):
    system = System(grid, variables='f', eigenvalue='sigma')

    with pytest.raises(Exception):
        system.add_boundary('nosuchvar', 'Dirichlet', 'Dirichlet')
