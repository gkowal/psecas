from psecas import Solver


class _DummyGrid:
    # The solver rewrite logic uses grid.z to form "d{z}(" (e.g., "dz(")
    z = "z"


def _rewrite(expr, var="u", **kwargs):
    # Avoid constructing a full Solver instance; we only need the method.
    s = Solver.__new__(Solver)
    return s._rewrite_derivatives(expr, _DummyGrid(), var, **kwargs)


def test_rewrite_dz_first_order():
    out = _rewrite("dz(u)")
    assert out == "grid.D(1).T"


def test_rewrite_dz_second_order_nested():
    out = _rewrite("dz(dz(u))")
    assert out == "grid.D(2).T"


def test_rewrite_dz_explicit_order_strict():
    out = _rewrite("dz(u,3)")
    assert out == "grid.D(3).T"


def test_rewrite_dz_explicit_order_reject_spaces():
    # Strict grammar: spaces are not allowed, so dz(u, 3) must NOT be rewritten
    # to grid.D(3).T. However, normal variable replacement still applies.
    out = _rewrite("dz(u, 3)")
    assert out == "dz(grid.D(0).T, 3)"


def test_rewrite_dz_explicit_order_boundary_form():
    # Boundary context should map dz(u,n) to a row slice grid.D(n)[i, :].
    out = _rewrite(
        "dz(u,4)",
        d0_repl="mask",
        d1_repl="grid.D(1)[5, :]",
        d2_repl="grid.D(2)[5, :]",
        dn_repl=lambda n: f"grid.D({n})[5, :]",
        z_repl="grid.zg[5]",
    )
    assert out == "grid.D(4)[5, :]"
