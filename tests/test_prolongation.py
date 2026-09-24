"""
Tests for prolongate_eigenvector, which carries an eigenvector from one
resolution to the next as a starting guess for the iterative drivers.

It used to interpolate only the interior nodes and fill the two ends with
the *old* grid's end values. That is correct only when those values are zero
(all-Dirichlet) or when the end nodes sit at fixed physical positions; on a
periodic or infinite grid the node positions move with N.
"""
import copy

import numpy as np
import pytest

from psecas import (Solver, System, FourierGrid, ChebyshevExtremaGrid,
                    ChebyshevRationalGrid)


class Laplace(System):
    def make_background(self):
        self.q = 1.0


def _solver(grid, boundary=False):
    system = Laplace(grid, variables='f', eigenvalue='sigma')
    system.add_equation("sigma*f = q*dz(dz(f))", boundary=boundary)
    return Solver(grid, system)


def test_endpoints_are_interpolated_on_a_periodic_grid():
    """
    FourierGrid node positions shift with N, so copying the old end values
    put them at the wrong place: refining 32 -> 64 left the interior accurate
    to 7e-16 while both endpoints were off by a factor of two.
    """
    grid = FourierGrid(N=32, zmin=0, zmax=1)
    solver = _solver(grid)
    grid_old = copy.deepcopy(grid)

    f_old = np.sin(2 * np.pi * grid_old.zg).astype(complex)
    solver.grid.N = 64
    f_new = solver.prolongate_eigenvector(f_old, grid_old)

    exact = np.sin(2 * np.pi * solver.grid.zg)
    np.testing.assert_allclose(f_new.real, exact, atol=1e-12)


def test_prolongation_is_accurate_with_dirichlet_packing():
    """The trimmed (interior-only) packing must round-trip too."""
    grid = ChebyshevExtremaGrid(N=32, zmin=0, zmax=1)
    solver = _solver(grid, boundary=True)
    grid_old = copy.deepcopy(grid)

    profile = np.sin(np.pi * grid_old.zg).astype(complex)
    vec_old = solver.fields_to_eigenvector({'f': profile}, grid_old)

    solver.grid.N = 64
    vec_new = solver.prolongate_eigenvector(vec_old, grid_old)
    f_new = solver.eigenvector_to_fields(vec_new, solver.grid)['f']

    np.testing.assert_allclose(f_new.real,
                               np.sin(np.pi * solver.grid.zg), atol=1e-12)


def test_prolongation_at_the_same_resolution_is_the_identity():
    grid = ChebyshevExtremaGrid(N=32, zmin=0, zmax=1)
    solver = _solver(grid)
    grid_old = copy.deepcopy(grid)

    f_old = np.cos(np.pi * grid_old.zg).astype(complex)
    f_new = solver.prolongate_eigenvector(f_old, grid_old)

    np.testing.assert_allclose(f_new, f_old, atol=1e-12)


def test_prolongation_onto_a_wider_infinite_grid():
    """
    The extent of an infinite grid grows with N, so the new outermost nodes
    can lie beyond anything the old grid can interpolate. That must not raise.
    """
    grid = ChebyshevRationalGrid(N=49)
    solver = _solver(grid)
    grid_old = copy.deepcopy(grid)

    f_old = np.exp(-grid_old.zg ** 2 / 4).astype(complex)
    solver.grid.N = 99
    assert solver.grid.zmax > grid_old.zmax   # the premise of this test

    f_new = solver.prolongate_eigenvector(f_old, grid_old)

    assert len(f_new) == solver.grid.NN
    assert np.isfinite(f_new).all()


def test_complex_profiles_keep_their_imaginary_part():
    grid = ChebyshevExtremaGrid(N=32, zmin=0, zmax=1)
    solver = _solver(grid)
    grid_old = copy.deepcopy(grid)

    f_old = (np.sin(np.pi * grid_old.zg)
             + 1j * np.cos(np.pi * grid_old.zg))
    solver.grid.N = 48
    f_new = solver.prolongate_eigenvector(f_old, grid_old)

    np.testing.assert_allclose(f_new.real,
                               np.sin(np.pi * solver.grid.zg), atol=1e-12)
    np.testing.assert_allclose(f_new.imag,
                               np.cos(np.pi * solver.grid.zg), atol=1e-12)
