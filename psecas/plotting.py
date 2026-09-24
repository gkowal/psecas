import numpy as np

# matplotlib is imported inside the functions that need it rather than at
# module level. psecas/__init__.py imports this module, so a module-level
# `import matplotlib.pyplot` would pull in a plotting backend on every
# `import psecas` - slow, and it fixes the backend before a caller has had
# a chance to choose one.


def plot_solution(system, filename=None, num=1, smooth=True, limits=None):
    """
    Quickly plot the 1D eigenmodes stored in the system object.

    system:   a system whose .result holds a solved eigenmode
    filename: if given, save the figure there instead of showing it
    num:      matplotlib figure number to draw into (reused if it exists)
    smooth:   also draw the spectrally interpolated profile, not just nodes
    limits:   (zmin, zmax) to restrict the interpolated curve to

    Returns the matplotlib Figure, so callers can adjust it further.
    """
    import matplotlib.pyplot as plt

    sol = system.result
    grid = system.grid

    title = r'$\omega = {:1.4f}, k_x = {:1.2f}, m={}$'

    # clear=True resets a pre-existing figure `num` in place. Calling
    # plt.figure(num) first and then plt.subplots(num=num, ...) asks
    # matplotlib to create the same figure twice with a different geometry,
    # which raises "Figure N already exists" on matplotlib >= 3.8.
    #
    # squeeze=False keeps `axes` a 2D array even for a single-variable
    # system, where matplotlib would otherwise hand back a bare Axes and
    # make axes[j] a TypeError.
    fig, axes = plt.subplots(num=num, nrows=system.dim, sharex=True,
                             squeeze=False, clear=True)
    axes = axes[:, 0]

    for j, var in enumerate(system.variables):
        if smooth:
            if limits is None:
                z = np.linspace(grid.zmin, grid.zmax, 2000)
            else:
                z = np.linspace(limits[0], limits[1], 2000)
            axes[j].plot(
                z, grid.interpolate(z, sol[var].real), 'C0', label='Real'
            )
            axes[j].plot(
                z, grid.interpolate(z, sol[var].imag), 'C1', label='Imag'
            )
        axes[j].plot(grid.zg, sol[var].real, 'C0.', label='Real')
        axes[j].plot(grid.zg, sol[var].imag, 'C1.', label='Imag')
        axes[j].set_ylabel(system.labels[j])

    axes[system.dim - 1].set_xlabel(r"$z$")

    # Systems without kx or a stored mode index get the short title. Catching
    # only the lookups that can legitimately be missing avoids swallowing
    # KeyboardInterrupt and real formatting errors, as a bare except did.
    try:
        axes[0].set_title(
            title.format(sol[system.eigenvalue], system.kx, sol['mode'])
        )
    except (AttributeError, KeyError):
        axes[0].set_title(
            r'$\omega$ = {:1.6f}'.format(sol[system.eigenvalue])
        )
    axes[0].legend(frameon=False)

    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()

    return fig


def plot_eigenvalues(sigma, errors=None, filename=None, num=1, title=None,
                     xlim=None, ylim=None, logx=False):
    """
    Scatter the eigenvalues in the complex plane, optionally coloured by a
    per-mode error estimate.

    Useful while developing a new problem: it shows at a glance where the
    physical modes sit relative to the spurious ones a spectral
    discretization always produces.

    sigma:    array of eigenvalues
    errors:   optional per-mode errors, used to colour the points (log scale)
    filename: save here instead of showing
    num:      matplotlib figure number to draw into
    title:    figure title
    xlim/ylim: axis limits; autoscaled when not given
    logx:     use a logarithmic real axis, for growth rates spanning decades

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    sigma = np.asarray(sigma).reshape(-1)

    fig, ax = plt.subplots(num=num, figsize=(6, 6), clear=True)

    if errors is not None:
        errors = np.asarray(errors).reshape(-1)
        finite = errors[np.isfinite(errors) & (errors > 0)]
        if finite.size:
            norm = LogNorm(finite.min(), finite.max())
        else:
            norm = None
        sc = ax.scatter(sigma.real, sigma.imag, c=errors, cmap='viridis',
                        marker='o', norm=norm)
        fig.colorbar(sc, ax=ax, label='error')
    else:
        ax.scatter(sigma.real, sigma.imag, marker='o')

    ax.set_xlabel('Real part')
    ax.set_ylabel('Imaginary part')
    if logx:
        ax.set_xscale('log')
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title is not None:
        ax.set_title(title)

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

    return fig


def get_2Dmap(system, var, xmin, xmax, Nx, Nz, zmin=None, zmax=None, time=0):
    """Create a 2D map of the eigenmode var stored in system.result[var].
       This function assumes that the eigenmodes have the form
       f(z)*exp(i kx x).
    """

    dx = (xmax - xmin) / Nx
    # + xmin: the cell centres must span [xmin, xmax]. Without it the map was
    # always built over [0, xmax - xmin] and xmin was silently ignored.
    xg = (0.5 + np.arange(Nx)) * dx + xmin

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
        """"""
        return (
            2*f*np.exp(1j*kx*x + system.result[system.eigenvalue]*time)
        ).real

    # Interpolate onto z-grid
    if type(var) is str:
        yr = system.grid.interpolate(zg, system.result[var].real)
        yi = system.grid.interpolate(zg, system.result[var].imag)
    else:
        yr = system.grid.interpolate(zg, var.real)
        yi = system.grid.interpolate(zg, var.imag)
    y = yr + 1j * yi
    for i in range(Nx):
        val[:, i] = return_real_ampl(y, xg[i])

    return val


def get_2D_cylindrical_map_in_cylindrical_coords(
    system, var, phimin, phimax, Nphi, Nr, rmin=None, rmax=None, time=0, z=0
):
    """Create a 2D map of the eigenmode var stored in system.result[var].
       This function assumes that the eigenmodes have the form
       f(r)*exp(i kz z + i m phi). It returns a map in the r-phi plane at
       a fixed value of z (default 0)
    """

    # Create linear grid in phi
    dphi = (phimax - phimin) / (Nphi - 1)
    # + phimin, for the same reason as xmin in get_2Dmap()
    phig = np.arange(Nphi) * dphi + phimin

    # Create linear grid in r
    if rmin is None:
        rmin = system.grid.zmin
    if rmax is None:
        rmax = system.grid.zmax
    dr = (rmax - rmin) / Nr
    rg = (0.5 + np.arange(Nr)) * dr + rmin

    # Contruct meshgrids
    rr, phiphi = np.meshgrid(rg, phig)

    # Azimuthal mode number
    m = system.m

    # Wavenumber
    kz = system.kz

    val = np.zeros((Nphi, Nr))

    def return_real_ampl(f, phi, z):
        """"""
        return (
            2
            * f
            * np.exp(
                1j*kz*z + 1j*m*phi + system.result[system.eigenvalue]*time
            )
        ).real

    # Interpolate onto r-grid
    if type(var) is str:
        yr = system.grid.interpolate(rg, system.result[var].real)
        yi = system.grid.interpolate(rg, system.result[var].imag)
    else:
        yr = system.grid.interpolate(rg, var.real)
        yi = system.grid.interpolate(rg, var.imag)
    y = yr + 1j * yi
    for i in range(Nphi):
        val[i, :] = return_real_ampl(y, phig[i], z)

    # This is how you would plot the map
    # xx = rr * np.cos(phiphi)
    # yy = rr * np.sin(phiphi)

    # plt.pcolormesh(xx, yy, val)
    # plt.axis('equal')
    # plt.show()

    return (rr, phiphi, val)


def get_2D_cylindrical_map(
    system, var, xmin, xmax, ymin, ymax, Nx, Ny, time=0, z=0
):
    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    phiphi = np.arctan2(yy, xx)

    # Azimuthal mode number
    m = system.m

    # Wavenumber
    kz = system.kz

    # Interpolate onto r-grid
    rg = rr.flatten()
    if type(var) is str:
        yr = system.grid.interpolate(rg, system.result[var].real)
        yi = system.grid.interpolate(rg, system.result[var].imag)
    else:
        yr = system.grid.interpolate(rg, var.real)
        yi = system.grid.interpolate(rg, var.imag)
    y = yr + 1j * yi

    # np.meshgrid(x, y) with len(x)=Nx, len(y)=Ny produces arrays of shape
    # (Ny, Nx), so the interpolated values must be folded back to rr.shape.
    # The previous np.resize(y, (Nx, Ny)) transposed the map and, being
    # resize rather than reshape, would have tiled or truncated silently
    # instead of failing whenever Nx != Ny.
    val = y.reshape(rr.shape)
    val = (2*val*np.exp(1j*kz*z + 1j*m*phiphi
                        + system.result[system.eigenvalue]*time)).real

    return (xx, yy, val)
