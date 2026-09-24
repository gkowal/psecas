"""
Tests for the MPI IO helper, run on a single rank.

mpi4py works without mpirun (comm.size == 1), which is enough to exercise
the filesystem and logging paths that used to shell out.
"""
import os

import numpy as np
import pytest

pytest.importorskip("mpi4py")

from psecas import IO, Solver, ChebyshevExtremaGrid


@pytest.fixture
def solved_system():
    from psecas.systems.mti import MagnetoThermalInstability

    grid = ChebyshevExtremaGrid(N=32, zmin=0, zmax=1)
    system = MagnetoThermalInstability(grid, beta=1e5, Kn0=200, kx=4 * np.pi)
    Solver(grid, system).solve(mode=0)
    return system


@pytest.fixture
def experiment(tmp_path):
    """A script path with a directory prefix and a space in the name."""
    script = tmp_path / "sub dir" / "my experiment.py"
    script.parent.mkdir()
    script.write_text("# experiment\n")
    return str(script)


def test_paths_with_spaces_are_handled(solved_system, experiment, tmp_path):
    """
    The folder and script names were interpolated unquoted into shell
    commands, so anything with a space in it misbehaved.
    """
    folder = str(tmp_path / "data with space")

    io = IO(solved_system, folder, experiment, steps=4, tag="test")

    assert os.path.isdir(io.data_folder)
    # The copy used to be built as data_folder + experiment, which breaks
    # as soon as the script has a directory prefix.
    assert os.path.exists(os.path.join(io.data_folder, "my experiment.py"))


def test_data_folder_may_lack_a_trailing_separator(solved_system, experiment,
                                                   tmp_path):
    """This used to be an assert, which python -O strips."""
    io = IO(solved_system, str(tmp_path / "nosep"), experiment, steps=1)

    assert os.path.isdir(io.data_folder)


def test_existing_data_folder_is_reused(solved_system, experiment, tmp_path):
    """mkdir without -p failed on an existing directory, status discarded."""
    folder = str(tmp_path / "data")

    IO(solved_system, folder, experiment, steps=1)
    IO(solved_system, folder, experiment, steps=1)    # must not raise


def test_logging_and_saving(solved_system, experiment, tmp_path):
    io = IO(solved_system, str(tmp_path / "run"), experiment, steps=2)

    io.log(0, 1.23, "N=32")
    io.rank_log("a message\n")
    io.save_system(0)
    io.finished()

    contents = os.listdir(io.data_folder)
    assert "psecas.log" in contents
    assert any(name.startswith("globalid-") for name in contents)

    log = open(io.logfile).read()
    assert "hostname" in log
    assert "Time elapsed" in log


def test_log_survives_a_rank_with_no_work(solved_system, experiment, tmp_path):
    """steps_local is 0 when there are more ranks than problems."""
    io = IO(solved_system, str(tmp_path / "empty"), experiment, steps=0)

    io.log(0, 1.0, "N=32")     # must not raise ZeroDivisionError
