def plot_solution(system, filename=None, num=1, smooth=True, limits=None):
    import numpy as np
    from evp import setup
    pylab = setup('ps')
    import matplotlib.pyplot as plt

    sol = system.result
    grid = system.grid

    title = r'$\omega = {:1.4f}, k_x = {:1.2f}, m={}$'
    plt.figure(num)
    plt.clf()
    fig, axes = plt.subplots(num=num, nrows=system.dim, sharex=True)
    for j, var in enumerate(system.variables):
        if smooth:
            if limits is None:
                z = np.linspace(grid.zmin, grid.zmax, 2000)
            else:
                z = np.linspace(limits[0], limits[1], 2000)
            axes[j].plot(z, grid.interpolate(z, sol[var].real),
                         'C0', label='Real')
            axes[j].plot(z, grid.interpolate(z, sol[var].imag),
                         'C1', label='Imag')
        axes[j].plot(grid.zg, sol[var].real, 'C0.', label='Real')
        axes[j].plot(grid.zg, sol[var].imag, 'C1.', label='Imag')
        axes[j].set_ylabel(system.labels[j])
    axes[system.dim-1].set_xlabel(r"$z$")
    axes[0].set_title(title.format(sol[system.eigenvalue], system.kx, sol['mode']))
    axes[0].legend(frameon=False)

    if not pylab and filename is not None:
        fig.savefig('../figures/' + filename + '.eps')
    else:
        plt.show()


def get_2Dmap(system, var, xmin, xmax, Nx, Nz, zmin=None, zmax=None, time=0):
    import numpy as np
    dx = (xmax-xmin)/Nx
    xg = (0.5 + np.arange(Nx))*dx

    if zmin is None or zmax is None:
        zmin = system.grid.zmin
        zmax = system.grid.zmax
    dz = (zmax - zmin) / Nz
    zg = (0.5 + np.arange(Nz)) * dz + zmin
    xx, zz = np.meshgrid(xg, zg)

    # Wavenumber
    kx = system.kx

    val = np.zeros((Nz, Nx))

    def return_real_ampl(f, x):
        return (f.real*2*np.cos(kx*x) - f.imag*2*np.sin(kx*x))

    def return_real_ampl(f, x):
        """Hardcode to the sigma notation..."""
        return (2*f*np.exp(1j*kx*x + system.result['sigma']*time)).real

    # Interpolate onto z-grid
    if type(var) is str:
        yr = system.grid.interpolate(zg, system.result[var].real)
        yi = system.grid.interpolate(zg, system.result[var].imag)
    else:
        yr = system.grid.interpolate(zg, var.real)
        yi = system.grid.interpolate(zg, var.imag)
    y = yr + 1j*yi
    for i in range(Nx):
        val[:, i] = return_real_ampl(y, xg[i])

    return val


def load_system(filename):
    """Load object containing solution.
    Input: filename, eg 'system.p'
    Output: system object
    """
    import pickle
    system = pickle.load(open(filename, 'rb'))
    return system


def save_system(system, filename):
    import pickle
    # Delete d0, d1 and d2 for storage effieciency
    del system.grid.d0
    del system.grid.d1
    del system.grid.d2
    pickle.dump(system, open(filename, 'wb'))


def write_athena(system, Nz, Lz, path=None, name=None):
    """
    Interpolate theory onto grid in Athena
    """
    import numpy as np

    # Grid points where Athena is defined (improve this!)
    dz = Lz/Nz
    z = np.arange(dz/2, Nz*dz, dz)
    znodes = np.arange(0., (Nz+1)*dz, dz)

    grid = system.grid
    result = system.result

    if path is None:
        path = './'

    if name is None:
        name = 'Pertubations'

    # Calculate and store imaginary part
    perturb = []
    for key in system.variables:
        if key != 'dA':
            y = np.hstack([grid.interpolate(z, result[key].imag), 0.0])
            perturb.append(y)

    if 'dA' in system.variables:
        znodes = np.arange(0., (Nz+1)*dz, dz)
        perturb.append(grid.interpolate(znodes, result['dA'].imag))
    else:
        perturb.append(np.zeros_like(znodes))

    perturb = np.transpose(perturb)
    np.savetxt(path + 'imag' + name + '{}.txt'.format(Nz), perturb,
               delimiter="\t", newline="\n", fmt="%1.16e")

    # Calculate and store real part
    perturb = []
    for key in system.variables:
        if key != 'dA':
            y = np.hstack([grid.interpolate(z, result[key].real), 0.0])
            perturb.append(y)

    if 'dA' in system.variables:
        perturb.append(grid.interpolate(znodes, result['dA'].real))
    else:
        perturb.append(np.zeros_like(znodes))

    perturb = np.transpose(perturb)
    np.savetxt(path + 'real' + name + '{}.txt'.format(Nz), perturb,
               delimiter="\t", newline="\n", fmt="%1.16e")