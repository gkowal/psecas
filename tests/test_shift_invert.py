"""
Regression tests for the shift-invert solvers on *generalized* eigenvalue
problems whose M2 is genuinely not the identity.

The rest of the suite cannot catch a broken generalized shift-invert: every
system it exercises has mat2 within a few percent of the identity (the MTI
system measures ||mat2 - I||/sqrt(n) = 0.078), so an operator built for the
wrong problem still lands near the right answer.

The systems below are chosen so that mat2 is singular, indefinite and - for
the tearing case - non-symmetric, which is what Psecas actually produces once
boundary conditions zero out rows. scipy's eigs(..., M=B, sigma=...) requires
B positive definite; violating that made ARPACK return an "eigenvalue" that
merely tracked the shift it was given.
"""
import numpy as np
import pytest
from scipy.linalg import eig

from psecas import (Solver, System, ShiftInvertError,
                    ChebyshevRationalGrid, ChebyshevExtremaGrid)


class Channel(System):
    """-h(z) K2 G = G'' + z G' , with h = exp(-z^2/2) so that M2 = -diag(h)."""

    def make_background(self):
        self.h = np.exp(-self.grid.zg ** 2 / 2)


def _channel(N=99):
    grid = ChebyshevRationalGrid(N=N, z='z')
    system = Channel(grid, variables='G', eigenvalue='K2')
    system.add_equation("-h*K2*G = dz(dz(G)) +z*dz(G)", boundary=True)
    return Solver(grid, system, do_gen_evp=True)


def _dense_reference(solver):
    """Every finite eigenvalue of the pencil, via a full dense solve."""
    solver.get_matrix1()
    solver.get_matrix2()
    E, _ = eig(solver.mat1.toarray(), solver.mat2.toarray())
    return E[np.isfinite(E)]


def test_mat2_is_not_positive_definite():
    """
    Guard the premise of these tests: if mat2 ever becomes well behaved,
    they stop exercising the path they were written for.
    """
    solver = _channel()
    solver.get_matrix2()
    B = solver.mat2.toarray()

    assert np.linalg.matrix_rank(B) < B.shape[0], "mat2 is expected to be singular"
    assert np.linalg.eigvals(B).real.min() <= 0.0


@pytest.mark.parametrize("guess", [85.0, 69.0, 55.0])
def test_solve_mode_matches_dense_solve(guess):
    """solve_mode must find the eigenvalue a dense solve puts nearest the guess."""
    solver = _channel()
    E = _dense_reference(solver)
    exact = E[np.argmin(np.abs(E - guess))]

    sigma, _ = solver.solve_mode(complex(guess))

    np.testing.assert_allclose(sigma, exact, rtol=1e-8)


@pytest.mark.parametrize("guess", [85.0, 69.0, 55.0])
def test_solve_with_guess_matches_dense_solve(guess):
    """Same requirement for the legacy entry point used by iterate_solver."""
    solver = _channel()
    E = _dense_reference(solver)
    exact = E[np.argmin(np.abs(E - guess))]

    sigma, _ = solver.solve_with_guess(complex(guess))

    np.testing.assert_allclose(sigma, exact, rtol=1e-8)


def test_solve_mode_converges_from_a_distant_guess():
    """
    A shift-invert solve must converge on the nearby eigenvalue, not return
    the shift. This is the precise failure that made the iterative driver
    report false convergence.
    """
    solver = _channel()
    E = _dense_reference(solver)
    exact = E[np.argmin(np.abs(E - 85.0))]

    # Deliberately 10% off; the returned value must be the eigenvalue, and
    # must not sit near the guess.
    guess = complex(exact.real * 1.10)
    sigma, _ = solver.solve_mode(guess)

    np.testing.assert_allclose(sigma, exact, rtol=1e-8)
    assert abs(sigma - guess) > abs(sigma - exact)


def test_returned_eigenpair_satisfies_its_residual():
    solver = _channel()
    sigma, v = solver.solve_mode(85.0 + 0j)

    A = solver.mat1.tocsc()
    B = solver.mat2.tocsc()
    residual = (np.linalg.norm(A @ v - sigma * (B @ v))
                / (np.linalg.norm(A @ v) + abs(sigma) * np.linalg.norm(B @ v)))

    assert residual < 1e-8
    assert solver.residual == pytest.approx(residual)


def test_unconverged_result_is_rejected_not_returned():
    """An impossible residual_tol must raise rather than hand back a mode."""
    solver = _channel()

    with pytest.raises(ShiftInvertError) as excinfo:
        solver.solve_mode(85.0 + 0j, residual_tol=1e-300)

    assert excinfo.value.sigma is not None
    assert excinfo.value.residual > 1e-300


def test_iterate_solve_multimode_matches_dense_solve_on_the_guess_path():
    """
    End-to-end guard. gtol is set high on purpose so that the shift-invert
    path is taken at every resolution after the first: this configuration
    used to return a value 82% away from the truth while reporting a
    shrinking error.
    """
    from psecas.systems.tearing_instability import TearingClassicalMHD

    Ns = [32, 48, 64]

    def dense_dominant(N):
        grid = ChebyshevExtremaGrid(N=N, zmin=-10, zmax=10)
        system = TearingClassicalMHD(grid, kx=0.5, a=1, S=1e4, periodic=False)
        E = _dense_reference(Solver(grid, system))
        E = E[E.real > 0]
        return E[np.argmax(E.real)]

    grid = ChebyshevExtremaGrid(N=Ns[0], zmin=-10, zmax=10)
    system = TearingClassicalMHD(grid, kx=0.5, a=1, S=1e4, periodic=False)
    solver = Solver(grid, system)

    sigma, _, _ = solver.iterate_solve_multimode(
        Ns, rtol=1e-6, gtol=10.0, orderby='real'
    )

    np.testing.assert_allclose(sigma.real, dense_dominant(Ns[-1]).real, rtol=1e-6)
