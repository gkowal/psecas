def test_solver_methods(verbose=False):
    """Show how the solver class methods can be called directly,
       using the MTI as an example.
       This can be useful when setting up a new problem.
    """
    import numpy as np
    from psecas import Solver, ChebyshevExtremaGrid
    from psecas.systems.mti import MagnetoThermalInstability
    from scipy.linalg import eig

    grid = ChebyshevExtremaGrid(N=64, zmin=0, zmax=1)

    system = MagnetoThermalInstability(grid, beta=1e5, Kn0=200, kx=4 * np.pi)

    solver = Solver(grid, system)

    solver.get_matrix1(verbose=verbose)

    solver.get_matrix2(verbose=verbose)

    E, V = eig(solver.mat1.toarray(), solver.mat2.toarray())

    # Sort the eigenvalues
    E, index = solver.sorting_strategy(E)

    mode = 0

    # Choose the eigenvalue mode value only
    sigma = E[index[mode]]
    v = V[:, index[mode]]

    np.testing.assert_allclose(1.7814514515967603, sigma, atol=1e-8)

    # Compare against the same result obtained through solve(). Note that
    # solve() always performs a full dense solve: it used to accept a
    # useOPinv argument, but never acted on it.
    solver.solve(saveall=True)
    np.testing.assert_allclose(solver.E[mode], sigma, atol=1e-8)


def test_solve_warns_about_the_ignored_useOPinv_argument():
    import numpy as np
    import pytest
    from psecas import Solver, ChebyshevExtremaGrid
    from psecas.systems.mti import MagnetoThermalInstability

    grid = ChebyshevExtremaGrid(N=32, zmin=0, zmax=1)
    system = MagnetoThermalInstability(grid, beta=1e5, Kn0=200, kx=4 * np.pi)
    solver = Solver(grid, system)

    with pytest.deprecated_call():
        solver.solve(useOPinv=False)



if __name__ == '__main__':
    test_solver_methods(True)
