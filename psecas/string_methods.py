import re

# An identifier character for the purposes of equation parsing. Python names
# may contain letters, digits and underscores, and Psecas equations routinely
# use names such as 'beta2', 'v_x' or 'rho1'.
_IDENT = r"A-Za-z0-9_"

# Cache compiled patterns: var_replace is called once per variable per
# equation per submatrix, so the same handful of patterns are rebuilt
# thousands of times during a resolution sweep.
_PATTERN_CACHE = {}


def _pattern(var):
    """Compiled pattern matching var only as a standalone identifier."""
    pattern = _PATTERN_CACHE.get(var)
    if pattern is None:
        pattern = re.compile(
            r"(?<![" + _IDENT + r"])" + re.escape(var) + r"(?![" + _IDENT + r"])"
        )
        _PATTERN_CACHE[var] = pattern
    return pattern


def var_replace(eq, var, new):
    """
    Replace every standalone occurrence of the identifier var in eq with new.

    This differs from str.replace in that it will not replace var when it is
    part of a longer identifier. Letters, digits and underscores all count as
    identifier characters, so 'beta' is not matched inside 'beta2', and 'v' is
    not matched inside 'v_x'.

    The replacement text is inserted literally; it is never rescanned, so a
    replacement that itself contains var terminates normally.

    Example:
    eq = "-1j*kx*v*drho -drhodz*dvz -1.0*dz(dvz) - drho"
    var_replace(eq, 'drho', 'foo')
    returns '-1j*kx*v*foo -drhodz*dvz -1.0*dz(dvz) - foo'
    where drhodz has not been replaced.
    """
    if not var:
        return eq

    # re.sub with a function avoids interpreting backslashes or group
    # references in `new`, which may be an arbitrary equation fragment.
    return _pattern(var).sub(lambda _: new, eq)
