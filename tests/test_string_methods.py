"""
Tests for var_replace, the identifier-aware text substitution that the
equation parser is built on.
"""
import pytest

from psecas.string_methods import var_replace


@pytest.mark.parametrize("eq, var, new, expected", [
    # The documented behaviour: do not replace inside a longer identifier.
    ("-1j*kx*v*drho -drhodz*dvz -1.0*dz(dvz) - drho", "drho", "foo",
     "-1j*kx*v*foo -drhodz*dvz -1.0*dz(dvz) - foo"),
    # Digits are identifier characters: 'beta' must not match inside 'beta2'.
    ("beta2*x", "beta", "B", "beta2*x"),
    ("rho1*rho", "rho", "R", "rho1*R"),
    ("2*beta + beta2", "beta", "B", "2*B + beta2"),
    # Underscores are identifier characters: 'v' must not match inside 'v_x'.
    ("v_x + v", "v", "W", "W" .join(["v_x + ", ""])),
    ("dv_dz*v", "v", "W", "dv_dz*W"),
    # Standalone occurrences next to operators and brackets are replaced.
    ("z*dz(f)", "z", "grid.zg", "grid.zg*dz(f)"),
    ("(f)+f", "f", "X", "(X)+X"),
    # Absent identifier leaves the string untouched.
    ("a*b", "c", "X", "a*b"),
])
def test_var_replace(eq, var, new, expected):
    assert var_replace(eq, var, new) == expected


def test_replacement_is_not_rescanned():
    """
    A replacement containing the identifier it replaces must terminate.

    The previous implementation did not advance past a completed replacement,
    so this looped forever.
    """
    assert var_replace("G + dz(dz(f))", "G", "G + q") == "G + q + dz(dz(f))"


def test_replacement_text_is_literal():
    r"""Backslashes and \1-style group references must not be interpreted."""
    assert var_replace("a", "a", r"\1") == r"\1"
    assert var_replace("a", "a", r"c:\n") == r"c:\n"


def test_empty_identifier_is_a_no_op():
    assert var_replace("a*b", "", "X") == "a*b"
