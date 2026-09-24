import numpy as np
import pickle

def load_system(filename):
    """Load object containing solution.
    Input: filename, eg 'system.p'
    Output: system object

    Note that this unpickles the file, which executes code contained in it.
    Only load files you produced yourself or otherwise trust.
    """

    with open(filename, 'rb') as fh:
        return pickle.load(fh)


def save_system(system, filename):
    """Save psecas system object as pickle

    example:

    save_system(system, 'system.p')

    """

    with open(filename, 'wb') as fh:
        pickle.dump(system, fh)


def write_athena(system, Nz, Lz, path=None, name=None):
    """
    Interpolate theory onto grid in Athena
    """

    # Grid points where Athena is defined (improve this!)
    dz = Lz / Nz
    z = np.arange(dz / 2, Nz * dz, dz)
    znodes = np.arange(0.0, (Nz + 1) * dz, dz)

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
        znodes = np.arange(0.0, (Nz + 1) * dz, dz)
        perturb.append(grid.interpolate(znodes, result['dA'].imag))
    else:
        perturb.append(np.zeros_like(znodes))

    perturb = np.transpose(perturb)
    np.savetxt(
        path + 'imag' + name + '{}.txt'.format(Nz),
        perturb,
        delimiter="\t",
        newline="\n",
        fmt="%1.16e",
    )

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
    np.savetxt(
        path + 'real' + name + '{}.txt'.format(Nz),
        perturb,
        delimiter="\t",
        newline="\n",
        fmt="%1.16e",
    )
