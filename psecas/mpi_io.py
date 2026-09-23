class IO:
    def __init__(self, system, data_folder, experiment, steps, tag=""):
        """
        Initialisation creates the output directory and saves the path to the
        io object.
        It also copies the experiment script to the data directory and
        saves a dictionary with important information.
        """
        from mpi4py.MPI import COMM_WORLD as comm
        from mpi4py.MPI import Wtime
        from numpy import arange

        self.system = system

        self.steps = steps
        self.index_global = arange(steps)
        self.index_local = self.index_global[comm.rank :: comm.size]
        self.steps_local = len(self.index_local)

        # Folder where data is stored. Accept it with or without a trailing
        # separator rather than asserting on one; asserts are stripped by
        # python -O, and os.path.join does the right thing either way.
        import os

        self.data_folder = os.path.join(data_folder, "")
        self.logfile = os.path.join(self.data_folder, "psecas.log")

        if comm.rank == 0:
            import os
            import shutil
            import socket
            import subprocess
            from datetime import datetime
            from numpy import float64

            # Create datafolder.
            #
            # This used to be subprocess.call("mkdir " + folder, shell=True):
            # the name was interpolated unquoted into a shell command, so a
            # path containing a space, a semicolon or a $ misbehaved or ran
            # as code, and mkdir without -p failed on an existing directory
            # with the non-zero status discarded.
            os.makedirs(self.data_folder, exist_ok=True)

            # Copy the experiment script next to the data. The old version
            # built its destination as data_folder + experiment, which broke
            # whenever `experiment` carried a directory prefix.
            shutil.copy(experiment, os.path.join(self.data_folder,
                                                 os.path.basename(experiment)))
            info = {"experiment": experiment}

            # Save git commit number. Not being in a repository, or not
            # having git installed, is not an error worth stopping a run for.
            try:
                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=os.path.dirname(os.path.abspath(experiment)) or ".",
                    stderr=subprocess.DEVNULL,
                )
                info.update({"git_commit": git_commit.strip().decode("utf-8")})
            except (OSError, subprocess.SubprocessError):
                # A bare except here also swallowed KeyboardInterrupt.
                pass

            # Save the hostname
            info.update({"hostname": socket.gethostname()})

            # Save the time at the start of simulation
            i = datetime.now()
            simulation_start = i.strftime("%d/%m/%Y at %H:%M:%S")
            info.update({"simulation_start": simulation_start})

            # Add tag which can be used to group simulations together
            info.update({"tag": tag})

            # Collect all variables in local namespace that are int, float or
            # float64. These will in general be the values set by us.
            # for key in local_vars.keys():
            #     if type(local_vars[key]) in (float, float64, int):
            #         info.update({key: local_vars[key]})

            # Save the number of MPI processes used
            info.update({"MPI": comm.size})

            # Save the dictionary info to the file info.p
            # pickle.dump(info, open(data_folder+'info.p', 'wb'))

            # Start a log file
            with open(self.logfile, "w") as f:
                f.write("Log file for EVP run\n")
                f.write(
                    "Solving {} evp problems on {} processors\n\n".format(
                        self.steps, comm.size
                    )
                )
                # Write start date and time
                f.write("Calculation started on " + simulation_start + "\n\n")
                # Write the contents of info.p for convenience
                f.write("Contents of info.p is printed below \n")
                for key in info.keys():
                    f.write(key + " = {} \n".format(info[key]))

                f.write("\nContents of system is printed below \n")
                for key in self.system.__dict__.keys():
                    if type(self.system.__dict__[key]) in (float, int, list):
                        f.write(key + " = {}\n".format(
                            self.system.__dict__[key]))

                grid = self.system.grid
                f.write("\nUsing {} with\n".format(type(grid)))
                for key in grid.__dict__.keys():
                    if type(grid.__dict__[key]) in (float, float64, int):
                        f.write(key + " = {} \n".format(grid.__dict__[key]))

                f.write("\n\nEntering main calculation loop \n")

        # Used for computing total runtime
        self.wt = Wtime()

        # Wait until all processors are done
        comm.barrier()

    def log(self, i, time, custom_str):
        from mpi4py.MPI import COMM_WORLD as comm

        # steps_local is zero on ranks that were handed no work, which
        # happens whenever there are more processes than problems.
        percent = 100.0 * (i + 1) / self.steps_local if self.steps_local else 100.0

        msg = (
            "Solved EVP with "
            + custom_str
            + " in {:1.2f} seconds. \
               Rank {} is {:2.0f}% done.\n"
        )
        with open(self.logfile, "a") as f:
            f.write(msg.format(time, comm.rank, percent))

    def rank_log(self, string):
        from mpi4py.MPI import COMM_WORLD as comm

        with open(self.logfile, "a") as f:
            f.write("Rank {}:".format(comm.rank) + string)

    def save_system(self, i):
        import os
        import pickle

        path = os.path.join(
            self.data_folder,
            "globalid-{:04d}.p".format(self.index_local[i]),
        )
        with open(path, "wb") as f:
            pickle.dump(self.system, f)

    def finished(self):
        """Write elapsed time to log file and move the log file to the data
        directory"""
        from mpi4py.MPI import COMM_WORLD as comm
        from mpi4py.MPI import Wtime

        # Wait until all processors are done
        comm.barrier()

        seconds = Wtime() - self.wt
        if comm.rank == 0:
            # import subprocess
            from datetime import datetime

            # Time at end of simulation
            i = datetime.now()
            endtime = i.strftime("%d/%m/%Y at %H:%M:%S")

            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            d, h = divmod(h, 24)

            msg = (
                "Time elapsed was {} days {} hours {} minutes {:1.4} seconds"
            )
            with open(self.logfile, "a") as f:
                f.write("\nCalculation ended on " + endtime + "\n")
                f.write(msg.format(d, h, m, s))
