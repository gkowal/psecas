"""
The default Solver.sorting_strategy and its conjugate-pair swap.

Eigenvalues are ordered by decreasing real part. The two members of a
complex-conjugate pair have real parts that agree only to rounding, so
without the swap which one solve(mode=0) returned depended on the LAPACK
build or the backend. A recognised pair keeps its two slots and puts the
member with Im > 0 in the earlier one; nothing else moves.
"""
import numpy as np
import pytest

from psecas import Solver, System, ChebyshevExtremaGrid


def _solver():
    grid = ChebyshevExtremaGrid(N=8, zmin=0, zmax=1)
    system = System(grid, variables='f', eigenvalue='sigma')
    system.add_equation("sigma*f = dz(dz(f))")
    return Solver(grid, system)


def _ordered(E):
    returned, index = _solver().sorting_strategy(E)
    assert sorted(index) == list(range(len(E)))
    return returned[index]


@pytest.mark.parametrize("perturb_upper", [True, False],
                         ids=["upper-member-larger", "lower-member-larger"])
def test_conjugate_pair_one_ulp_apart_puts_positive_imag_first(perturb_upper):
    re = 0.04505098
    re_up = np.nextafter(re, 1.0)
    σ_upper = complex(re_up if perturb_upper else re, 0.54218353)
    σ_lower = complex(re if perturb_upper else re_up, -0.54218353)
    others = [0.01 + 0.5j, 0.01 - 0.5j, -0.3 + 0j, 0.002 + 0j, -1.0 + 2.0j]
    E = np.array([σ_upper, σ_lower] + others)

    rng = np.random.default_rng(1234)
    for _ in range(50):
        ordered = _ordered(E[rng.permutation(E.size)])
        assert ordered[0] == σ_upper
        assert ordered[1] == σ_lower
        assert ordered[2] == 0.01 + 0.5j


def test_real_parts_beyond_the_tolerance_keep_real_part_order():
    solver = _solver()
    gap = 100 * solver.sorting_tie_rtol
    E = np.array([1.0 + 1.0j, (1.0 + gap) - 1.0j, (1.0 - gap) + 2.0j])

    returned, index = solver.sorting_strategy(E)

    np.testing.assert_array_equal(returned[index],
                                  [E[1], E[0], E[2]])


def test_cutoff_entries_are_zeroed_on_a_copy_and_still_ordered():
    solver = _solver()
    E = np.array([1.0 + 0j, 20.0 + 0j, -2.0 + 0j, 0.5 + 30.0j, 3.0 - 1.0j])
    before = E.copy()

    returned, index = solver.sorting_strategy(E)

    np.testing.assert_array_equal(E, before)
    assert returned is not E
    assert returned[1] == 0 and returned[3] == 0
    # 3 and 1 first, then the two zeroed entries in input order (an exact
    # tie, kept stable), then the negative one.
    np.testing.assert_array_equal(index, [4, 0, 1, 3, 2])


def test_tolerance_is_configurable():
    solver = _solver()
    E = np.array([1.0 - 1.0j, (1.0 - 1e-4) + 1.0j])

    assert list(solver.sorting_strategy(E)[1]) == [0, 1]

    solver.sorting_tie_rtol = 1e-3
    assert list(solver.sorting_strategy(E)[1]) == [1, 0]


def test_empty_spectrum():
    returned, index = _solver().sorting_strategy(np.array([], dtype=complex))

    assert returned.size == 0
    assert index.size == 0


def test_pair_swaps_across_intervening_elements_which_stay_put():
    """
    The two members sit in slots 0 and 4; between them are another
    eigenvalue whose real part lies inside the pair's gap and two cutoff
    zeros at Re = 0 (the pair's real parts straddle zero).
    """
    solver = _solver()
    lower = 1e-10 - 1.5j
    upper = -1e-10 + 1.5j
    between = 0.0 + 0.3j
    E = np.array([upper, 20.0 + 0j, between, lower, 0.5 + 30.0j, -2.0 + 0j])

    returned, index = solver.sorting_strategy(E)

    # Plain real-part order would be [3, 1, 2, 4, 0, 5]: lower, zero,
    # between, zero, upper, -2.
    np.testing.assert_array_equal(index, [0, 1, 2, 4, 3, 5])
    assert returned[index[0]] == upper


@pytest.mark.parametrize("signs", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
def test_near_equal_real_eigenvalues_are_never_swapped(signs):
    solver = _solver()
    E = np.array([(1.0 + 1e-7) + signs[0] * 1e-15j,
                  1.0 + signs[1] * 1e-15j])

    assert list(solver.sorting_strategy(E)[1]) == [0, 1]
    assert list(solver.sorting_strategy(E[::-1])[1]) == [1, 0]


def test_cutoff_zeros_are_never_paired_and_never_move():
    solver = _solver()
    E = np.array([0.5 + 0j, 11.0 + 0j, 0.0 + 12.0j, -11.0 + 0j, -0.5 + 0j])

    returned, index = solver.sorting_strategy(E)

    np.testing.assert_array_equal(returned[1:4], 0)
    np.testing.assert_array_equal(index, [0, 1, 2, 3, 4])


def test_a_false_pair_is_not_swapped():
    """Equal real parts, but 0.5 and -0.501 are not conjugates."""
    solver = _solver()
    E = np.array([0.2 - 0.5j, 0.2 + 0.501j])

    assert list(solver.sorting_strategy(E)[1]) == [0, 1]


def test_the_nearest_of_two_candidates_is_the_partner():
    """
    Modelled on TearingClassicalMHD at N=300: two pairs whose real parts are
    2.8e-8 apart relative to |E|. For a, both conj(a) and conj(b) lie in the
    window and inside the tolerance (|a - b| is about 2e-8), so only the
    nearest-partner rule keeps each pair together.
    """
    a = -1.25708656e-05 + 0.499999996j
    b = -1.25848262e-05 + 0.500000011j
    E = np.array([a, b, np.conj(a), np.conj(b)])
    assert abs(a - b) <= 1e-6 * abs(a)     # the premise of this test

    rng = np.random.default_rng(7)
    for _ in range(20):
        ordered = _ordered(E[rng.permutation(E.size)])
        np.testing.assert_array_equal(ordered, [a, np.conj(a), b, np.conj(b)])


def test_nearer_candidate_wins_inside_the_window():
    """
    i = 1-1j comes first and has two numerical conjugates in the window,
    both inside the tolerance; it must pair with the nearer one.
    """
    solver = _solver()
    i = 1.0 - 1.0j
    near = 1.0 + (1.0 + 2e-7) * 1j
    far = 1.0 + (1.0 + 8e-7) * 1j
    assert abs(i - np.conj(far)) <= solver.sorting_tie_rtol * abs(far)
    # All three real parts are exactly 1, so plain order is input order.
    E = np.array([i, far, near])

    returned, index = solver.sorting_strategy(E)

    # Paired with near (slot 2): the two swap and far stays in slot 1.
    # Pairing with far would have given [1, 0, 2].
    np.testing.assert_array_equal(index, [2, 1, 0])


def test_mixed_pair_one_real_like_member_is_paired():
    """
    1+0.9e-6j is real-like at rtol 1e-6 (|Im| <= rtol*|E|), 1-1.1e-6j is
    not. The rule only refuses pairs where *both* are real-like, and
    |E_i - conj(E_j)| = 2e-7 is inside the tolerance, so they pair and the
    larger Im comes first.
    """
    solver = _solver()
    E = np.array([1.0 - 1.1e-6j, 1.0 + 0.9e-6j])

    assert list(solver.sorting_strategy(E)[1]) == [1, 0]
    assert list(solver.sorting_strategy(E[::-1])[1]) == [0, 1]


def test_nan_never_pairs_and_does_not_crash():
    solver = _solver()
    nan = complex(np.nan, np.nan)
    E = np.array([0.5 - 1.0j, nan, 0.5 + 1.0j, complex(np.nan, 1.0)])

    returned, index = solver.sorting_strategy(E)

    assert sorted(index) == [0, 1, 2, 3]
    # The pair swaps; the NaNs sort last and keep input order.
    np.testing.assert_array_equal(index, [2, 0, 1, 3])


def test_real_dtype_input():
    solver = _solver()
    E = np.array([0.5, 20.0, 3.0, -1.0, 3.0])

    returned, index = solver.sorting_strategy(E)

    assert returned.dtype == np.float64
    assert returned[1] == 0
    np.testing.assert_array_equal(index, [2, 4, 0, 1, 3])


@pytest.mark.parametrize("rtol", [-1e-6, 1.0, 2.0, np.nan])
def test_tolerance_outside_zero_one_is_refused(rtol):
    solver = _solver()
    solver.sorting_tie_rtol = rtol

    with pytest.raises(ValueError, match="sorting_tie_rtol"):
        solver.sorting_strategy(np.array([1.0 + 1.0j, 1.0 - 1.0j]))
