#!/usr/bin/env python3
#
import numpy as np
import os, sys

from functools import lru_cache
from pathlib import Path


class DeltaError(ValueError):
    """Raised when Δ' <= 0 or imaginary (stable mode at the given α) so the tearing eigenmode is not excited."""
    pass

class ConvergenceError(ValueError):
    """Raised when N cannot be increased enough (up to Nmax) to satisfy the outer/inner constraints."""
    pass


def estimate_max(**params):
    """
    Determine the maximum growth rate and corresponding wavenumber.

    Parameters (kwargs)
    -------------------
    CGL          : bool  , enable CGL anisotropy (default False)
    S            : float , Lundquist number (default 1e4)
    Pr           : float , magnetic Prandtl number (used softly here;) (default 0)
    β, Δβ        : floats, gyrotropic plasma-β and anisotropy (defaults β=0, Δβ=0)
    ɣpar, ɣper   : floats, aidabatic indices (γ∥=3, γ⟂=2)

    Returns
    -------
    α_max : float
    Δ_max : float
        The wavenumber corresponding to maximum eigenmode.
    """
    CGL          = params.get('CGL'    , False  )
    S            = params.get('S'      ,    1e4 )
    Pr           = params.get('Pr'     ,    0   )
    β            = params.get('β'      ,    0   )
    Δβ           = params.get('Δβ'     ,    0   )
    ɣpar         = params.get('ɣpar'   ,    3   )
    ɣper         = params.get('ɣper'   ,    2   )

    # --- Δ'(α); raise if stable at α
    if CGL and (Δβ <= - (2 + (ɣpar + ɣper - 2) * β) / ɣpar or Δβ >= 2):
        raise DeltaError(f"Δ' purely imaginary for Δβ = {Δβ:+.3e} => stable eigenmode for any α.")

    # --- CGL anisotropy factor and scaled wavenumbers
    μ = np.sqrt((2 + (ɣpar + ɣper - 2) * β + ɣpar * Δβ) / (2 - Δβ)) if CGL else 1

    # α_m scaling
    αm = 1.3583e+00 * (S / (S + 400))**(1/4) * S**(-1/4) * μ**(-3/4) * ((0.05*Pr**2+0.7*Pr+1)/(12*Pr+1))**(1/8)
    Xm = αm * μ
    Δm = 2 * (1 / Xm - Xm)

    return αm, Δm


def select_NC(**params):
    """
    Determine the Chebyshev–TB grid resolution N and scaling factor C for the linear tearing
    instability eigenproblem under the mapping z = C * tan(theta).

    This routine enforces BOTH of the following constraints for a given set of physical
    parameters (S, Pr, β, etc.) and numerical requirements:
      (1) OUTER decay extent: The outermost collocation |z|max must reach the location where the
          eigenmode amplitude has decayed by 'decay_efolds' e-foldings:
              z_req = decay_efolds / λ,   where λ = α / μ and μ is the CGL anisotropy factor.
          For TB grids, |z|max(N, C) = C * cot(π / (2*(N+1))). This yields a LOWER bound on C:
              C_out(N) = z_req / cot(π / (2*(N+1))) = z_req * tan(π / (2*(N+1))).
      (2) INNER resolution: At least n_inner collocation points must lie within the smallest
          inner-layer width δ_min = min(a, δ_FKR, δ_Cop). Using the m = (n_inner-1)/2 node near
          the origin on the TB grid, |z_m| = C * tan(mπ/(N+1)) ≤ δ_min gives an UPPER bound:
              C_in(N) = δ_min / tan(mπ / (N+1)).

    With your convention (smaller C packs more points near z=0), feasibility requires:
        C_out(N) ≤ C ≤ C_in(N).
    The algorithm searches N = Nmin, Nmin+Ninc, ... up to Nmax for the first N satisfying
    C_out(N) ≤ C_in(N). It returns:
        N*  = smallest feasible resolution,
        C*  = C_out(N*), i.e., the tight lower bound (maximizes inner packing at that N).

    Notes
    -----
    • Keeps your scaling for α_m (needed for c_FKR and c_Cop prefactors).
    • Handles CGL via μ = sqrt((1 + ((γ∥+γ⟂−2)/2)*β + Δβ/2) / (1 − Δβ/2)); set μ=1 if CGL=False.
    • Enforces odd n_inner (centered grid has a node at z=0).
    • Raises DeltaError if Δ' ≤ 0 (no tearing at that α).
    • Raises ConvergenceError if no feasible N is found up to Nmax.

    Parameters (kwargs)
    -------------------
    Nmin         : int   , starting resolution (default 128)
    Nmax         : int   , maximum resolution to try (default 1024)
    Ninc         : int   , resolution increment per iteration (default 32)
    n_inner      : int   , minimum collocation points inside δ_min (must be odd; default 5)
    decay_efolds : float , number of e-foldings to resolve at |z|max (default 3.51 ~ 3% amplitude)
    CGL          : bool  , enable CGL anisotropy (default False)
    S            : float , Lundquist number (default 1e4)
    Pr           : float , magnetic Prandtl number (used softly here;) (default 0)
    β, Δβ        : floats, gyrotropic plasma-β and anisotropy (defaults β=0, Δβ=0)
    ɣpar, ɣper   : floats, aidabatic indices (γ∥=3, γ⟂=2)
    a            : float , current-sheet half-width (default 1)
    α            : float , dimensionless wavenumber (k * a) at which to set the grid (default 0.1)
    verbose      : bool  , print per-iteration diagnostics (default False)

    Returns
    -------
    N_opt : int
        Smallest resolution satisfying both constraints.
    C_opt : float
        Scaling factor chosen at the outer lower bound, C_out(N_opt).

    Raises
    ------
    DeltaError
    ConvergenceError
    """
    Nmin         = params.get('Nmin'        ,  128   )
    Nmax         = params.get('Nmax'        , 1024   )
    Ninc         = params.get('Ninc'        ,   32   )
    n_inner      = params.get('n_inner'     ,    5   )
    decay_efolds = params.get('decay_efolds',    3.51)
    CGL          = params.get('CGL'         , False  )
    S            = params.get('S'           ,    1e4 )
    Pr           = params.get('Pr'          ,    0   )
    β            = params.get('β'           ,    0   )
    Δβ           = params.get('Δβ'          ,    0   )
    ɣpar         = params.get('ɣpar'        ,    3   )
    ɣper         = params.get('ɣper'        ,    2   )
    a            = params.get('a'           ,    1   )
    α            = params.get('α'           ,    0.1 )
    θ            = params.get('θ'           ,    8.0 )
    verbose      = params.get('verbose'     , False  )

    # Enforce odd n_inner (so m is integer and z=0 is a collocation point)
    if n_inner % 2 == 0:
        n_inner += 1
    m = (n_inner - 1) / 2

    # --- Δ'(α); raise if stable at α
    if CGL and (Δβ <= - (2 + (ɣpar + ɣper - 2) * β) / ɣpar or Δβ >= 2):
        raise DeltaError(f"Δ' purely imaginary for Δβ = {Δβ:+.3e} => stable eigenmode for α = {α:.3e}.")

    # --- CGL anisotropy factor and scaled wavenumbers
    μ = np.sqrt((2 + (ɣpar + ɣper - 2) * β + ɣpar * Δβ) / (2 - Δβ)) if CGL else 1
    λ = α / μ
    X = α * μ

    # --- Δ'(α); raise if stable at α
    Δ = 2 * (1 / X - X)
    if Δ <= 0.0:
        raise DeltaError(f"Δ' <= 0 (Δ = {Δ:.3e}) ⇒ stable eigenmode for α = {α:.3e}.")

    # α_m scaling and corresponding Δ'(α_m)
    #αm, Δm = estimate_max(**params)

    # --- Prefactors and inner-layer widths (soft Pr factors kept)
    #cFKR = (αm**-2 * Δm * S**-2 * (1 + 1e-4 * Pr) * C**-1)**(-1/5) / (Δm * a)
    #cCop = a * (αm * S)**(1/3) * (1 + Pr)**(-1/6) * C**(1/6) / Δm

    #δFKR = cFKR * a * (α**-2 * Δ * S**-2 * (1 + 1e-4 * Pr) * C**-1)**(1/5)
    #δCop = cCop * a * (α * S)**(-1/3) * C**(-1/6) * (1 + Pr)**(1/6)
    #δmin = min(a, δFKR, δCop)

    #cCop = 0.875
    #cFKR = 0.900
    #δFKR = cFKR * a * ((S * α)**-2 * a * Δ)**(1.0/5.0)
    #δCop = cCop * a * (α * S)**(-1.0/3.0)

    fS   = (0.038288 / (0.047443 + S**-0.46355) + 0.079649)
    gS   = 0.91451 - 2.0654 / (S**0.37651 + 0.88448)
    fPr  = (1.0777 + ((Pr * 0.71554) * (5.765 + Pr)))**0.079176
    gPr  = 2.4379 * ((Pr + 0.0045991)**0.16186)

    δCop = fS * fPr * a * (α * S)**(-1.0/3.0)
    δFKR = gS * gPr * a * ((S * α)**-2 * a * Δ)**(1.0/5.0)

    # --- Smooth minimum to avoid sharp transitions    
    δmin = δCop * δFKR / (δCop**θ + δFKR**θ)**(1.0/θ)
    if verbose:
        print(f"[inner scales for α={α:.4e}] a={a:.4e}, δCop={δCop:.4e}, δFKR={δFKR:.4e}, δmin={δmin:.4e}")

    # --- Outer requirement: amplitude reduced by e^{-decay_efolds} at |z|max
    #     z_req = decay_efolds / λ; TB outermost node: |z|max(N,C) = C * cot(π/(2(N+1)))
    zreq = decay_efolds / λ
    πh   = np.pi / 2
    πm   = np.pi * m
    Ntop = Nmax - 3 * Ninc

    # Iterate N upward until C_out(N) <= C_in(N); then set C = C_out(N).
    N  = Nmin
    while True:
        if N > Ntop:
            raise ConvergenceError(
                f"Insufficient N up to Nmax={Nmax}: cannot satisfy C_out(N) <= C_in(N). "
                f"Try increasing Nmax or relaxing n_inner/decay_efolds."
            )
        Np   = N + 1
        Cout = zreq * np.tan(πh / Np)
        Cin  = δmin / np.tan(πm / Np)

        if Cout <= Cin:
            if verbose:
                print(f"N={N:4d}  C_out={Cout:.4e}  C_in={Cin:.4e}  (feasible? {Cout <= Cin})")
            break
        N += Ninc

    return N, Cout, Cin, δmin


def inner_layer_thickness(system, δtol=1e-3, maxiter=20):
    """
    Calculate the inner layer thickness δ for the tearing instability eigenmode.
    Parameters
    ----------
    system : psecas.systems.tearing.Tearing
        The tearing instability system object.
    α : float
        The dimensionless wavenumber (k * a).
    σ : complex
        The eigenvalue (growth rate) of the mode.
    Returns
    -------
    δ : float
        The inner layer thickness.
    """
    grid = system.grid
    sol  = system.result

    a = system.a
    S = system.S
    α = system.kx * a
    u = sol['duz']
    b = sol['dbz']

    Ti = np.abs(1j * α * u * system.Bx)
    Tn = np.abs(system.grid.dder(b) - α**2 * b) / S
    Td = Tn - Ti

    I   = np.where(np.abs(grid.zg) <= 2 * a)
    zin = grid.zg[I]
    Tin = Tn[I] - Ti[I]

    # check if Td changes sign in the interval
    if Tin.min() <= 0.0 and Tin.max() >= 0.0:    
        # start from z=0 and search at which distance Td becomes negative
        idx = np.abs(zin).argmin()
        while Tin[idx] > 0.0 and idx < (Tin.size - 1):
            idx += 1
        idx -= 1
        if idx < (Tin.size - 1):
            # endpoints of the bracket on the sliced arrays
            zl, zh = float(zin[idx]), float(zin[idx + 1])
            tl, th = float(Tin[idx]), float(Tin[idx + 1])

            # If bracket does not change sign, fall back to choosing the best of endpoints/midpoint
            if tl * th > 0.0:
                zm = 0.5 * (zl + zh)
                tm = grid.interpolate(zm, Td)
                # pick the point with smallest absolute value
                best = min(((abs(tl), zl), (abs(tm), zm), (abs(th), zh)), key=lambda x: x[0])[1]
                δ = float(best)
            else:
                it = 0
                while 2.0 * (zh - zl) > δtol * (zh + zl) and it < maxiter:
                    zm = 0.5 * (zl + zh)
                    tm = grid.interpolate(zm, Td)
                    if tm == 0.0:
                        zl = zh = zm
                        break
                    if tl * tm > 0:
                        zl, tl = zm, tm
                    else:
                        zh, th = zm, tm
                    it += 1
                δ = float(0.5 * (zl + zh))
    else:    
        # no region where Td >= 0: set δ to zero (no inner layer detected)
        δ = 0.0

    return δ


def eigenmodes(**params):
    """
    Calculates tearing instability eigenmodes for a given set of parameters.

    Parameters:
    -----------
    **params : dict
        Keyword arguments for physical and numerical parameters.
        Keys and default values:
        'CGL'      ( bool, default=False): Flag for Gyrotropic (True) or Stadard MHD (False).
        'Nmin'     (  int, default=128  ): Minimum number of collocation points.
        'Nmax'     (  int, default=1024 ): Maximum number of collocation points.
        'Ninc'     (  int, default=32   ): Increment in collocation points for iteration.
        'C'        (float, default=None ): Scaling factor for the rational Chebyshev grid.
                                           If None, it is auto-determined.
        'a'        (float, default=1.0  ): Half-width of the equilibrium current profile.
        'w'        (float, default=0.0  ): Half-width of the velocity shear region.
        'α'        (float, default=0.1  ): Wave number k multiplied by the half-width a.
        'ζ'        (float, default=1.0  ): Parameter controlling equilibrium magnetic field profile.
        'S'        (float, default=1e4  ): Lundquist number.
        'Pr'       (float, default=0.0  ): Prandtl number (or similar ratio).
        'ξ'        (float, default=0.0  ): The strength of the transversal magnetic field component.
        'ϵ'        (float, default=0.0  ): The strength of the Hall term.
        'β'        (float, default=1.0  ): Plasma beta.
        'Δβ'       (float, default=0.0  ): Plasma beta difference between parallel and perpendicular directions.
        'ɣpar'     (float, default=3.0  ): Parallel adiabatic index.
        'ɣper'     (float, default=2.0  ): Perpendicular adiabatic index.
        'σlower'   (float, default=1e-5 ): Lower bound for real part of eigenvalue search.
        'σupper'   (float, default=1.0  ): Upper bound for real part of eigenvalue search.
        'mode'     (  int, default=0    ): Index of the mode to track (0 is the fastest growing).
        'atol'     (float, default=1e-10): Absolute tolerance for convergence.
        'rtol'     (float, default=1e-5 ): Relative tolerance for convergence.
        'orderby'  (  str, default='amp'): Criterion to order eigenvalues: amplitude or tolerance.
        'allgrids' ( bool, default=False): Flag to iterate over all grids even if the convergence was reached.
        'verbose'  ( bool, default=False): Flag to enable verbose output.

    Returns:
    --------
    tuple: (σ, v, e, C, N_final, success)
        σ       (complex array or None): Eigenvalues (growth rates).
        v       (complex array or None): Eigenvectors (eigenmodes).
        e       (object or None)       : Error estimates.
        C       (float or None)        : Final scaling factor used in the grid.
        N       (int or None): Final number of collocation points.
        success (bool): True if the solver was successful, False otherwise.

    Raises:
    -------
    ValueError: If a required parameter is not positive or within a specified range.
    """
    '''
        Calculates tearing instability eigenmodes for a given set of parameters
    '''
    α        = params.get('α'       ,   0.1)
    verbose  = params.get('verbose' , False)

    try:
        from psecas import Solver, ChebyshevRationalGrid
        from psecas.systems.tearing import Tearing
        import numpy as np

        Nmin         = params.get('Nmin'    ,  128   )
        Nmax         = params.get('Nmax'    , 1024   )
        Ninc         = params.get('Ninc'    ,   32   )
        n_inner      = params.get('n_inner' ,    5   )
        decay_efolds = params.get('decay_efolds' , -np.log(0.03))
        C        = params.get('C'       , None   )
        a        = params.get('a'       ,   1    )
        w        = params.get('w'       ,   0    )
        ζ        = params.get('ζ'       ,   1    )
        S        = params.get('S'       ,   1e4  )
        Pr       = params.get('Pr'      ,   0    )
        CGL      = params.get('CGL'     , False  )
        ξ        = params.get('ξ'       ,   0    )
        ϵ        = params.get('ϵ'       ,   0    )
        β        = params.get('β'       ,   1    )
        Δβ       = params.get('Δβ'      ,   0    )
        ɣpar     = params.get('ɣpar'    ,   3    )
        ɣper     = params.get('ɣper'    ,   2    )
        σlower   = params.get('σlower'  ,   1e-5 )
        σupper   = params.get('σupper'  ,   1    )
        mode     = params.get('mode '   ,   0    )
        atol     = params.get('atol'    ,   1e-10)
        rtol     = params.get('rtol'    ,   1e-5 )
        gtol     = params.get('gtol'    ,   1e-2 )
        δtol     = params.get('δtol'    ,   1e-3 )
        orderby  = params.get('orderby' , 'amplitude')
        allgrids = params.get('allgrids', False  )

        if α <= 0:
            raise ValueError("α must be positive.")
        if a <= 0:
            raise ValueError("a must be positive.")
        if not 0 <= ζ <= 1:
            raise ValueError("ζ must be between 0 and 1.")
        if S <= 0:
            raise ValueError("S must be positive.")
        if Pr < 0:
            raise ValueError("Pr cannot be negative.")
        if β < 0:
            raise ValueError("β cannot be negative.")
        if ɣpar <= 0:
            raise ValueError("ɣpar must be positive.")
        if ɣper <= 0:
            raise ValueError("ɣper must be positive.")
        if ϵ < 0:
            raise ValueError("ϵ cannot be negative.")

        if C == None:
            Nlow, _, C, δin = select_NC(**params)
        else:
            _, _, _, δin = select_NC(**params)
            Nlow = Nmin

        if C <= 0:
            raise ValueError("C must be positive.")

        order = 4 if Pr > 0 else 2
        Ns    = np.arange(Nlow, Nmax+Ninc, Ninc)

        grid    = ChebyshevRationalGrid(N=Ns[0], order=order, C=C)
        system  = Tearing(grid, kx=α, a=a, w=w, ζ=ζ, CGL=CGL, S=S, Pr=Pr, \
                     ξ=ξ, ϵ=ϵ, β=β, Δβ=Δβ, ɣpar=ɣpar, ɣper=ɣper, \
                     periodic=False)
        solver  = Solver(grid, system)
        σ, v, e = solver.iterate_solve_multimode(Ns, maxmode=mode, useOPinv=False, \
                     gtol=gtol, atol=atol, rtol=rtol, \
                     allmodes=True, allgrids=allgrids, orderby=orderby, verbose=verbose)
        N = solver.grid.N

        δin = inner_layer_thickness(system, δtol=δtol)

        I   = np.where(np.abs(system.grid.zg) <= δin)
        nin = I[0].size
        if verbose:
            print(f'Calculation done for α = {α:.4e} with C = {C:.3e} ({nin} points over the interval |z| < δin):')
            print(f'  σ₀ = {σ[0].real:.4e}{σ[0].imag:+.4e}j (error = {e[0]:.3e}, N = {N})')

        return σ, v, e, δin, C, nin, N, True

    except DeltaError as ex:
        if verbose:
            print(f"Stable eigenmode: {ex}")
        return None, None, None, None, None, None, None, False

    except ConvergenceError as ex:
        if verbose:
            print(f"Insufficient resolution: {ex}")
        return None, None, None, None, None, None, None, False

    except ValueError as ex:
        if verbose:
            print(f"Wrong parameter: {ex}")
        return None, None, None, None, None, None, None, False

    except Exception as ex:
        if verbose:
            print(f"[WARNING] Solver failed for α = {α:.3e}: {ex}")
        return None, None, None, None, None, None, None, False


@lru_cache(maxsize=1024)
def f_cached(α, **params):
    import numpy as np

    if α <= 0:
        return 0

    params['α'] = α

    σ, _, _, _, _, _, _, status = eigenmodes(**params)

    if status:
        return float(-σ[0].real)
    else:
        return 0


def f_scalar(α, **params):
    return f_cached(round(float(α), 12), **params)


def task_new(k, params):
    import numpy as np
    import os

    global counter

    ntasks  = params.get('ntasks' , 1)
    verbose = params.get('verbose', False)
    force   = params.get('force'  , False)

    end      = '\n' if verbose else ''
    status   = True

    α = k * a

    sname  = os.path.join(params.get('data_path', './'), f'state_α{α:.6e}.npz')

    if os.path.exists(sname) and not force:
        state = np.load(sname)

        α   = state['wavenumber']
        σ   = state['eigenvalues']
        e   = state['tolerances']
        δin = state['inner_scale']
        C   = state['scaling_factor']
        N   = state['resolution']
        nin = state['n_inner']
    else:
        try:
            params['α'] = α

            σ, v, e, δin, C, nin, N, status = eigenmodes(**params)

            if status:
                np.savez_compressed(sname, wavenumber=α, eigenvalues=σ, \
                            eigenvectors=v, tolerances=e, inner_scale=δin, \
                            scaling_factor=C, n_inner=nin, resolution=N)

        except Exception as ex:
            status = False
            if verbose:
                print(f"\n[WARNING] Could not find eigenmode for α = {α:.3e}: {ex}")

    with counter.get_lock():
        counter.value += 1
        n = counter.value

    fmt    = r'[{:' + str(len(str(ntasks))) + 'd}/' + str(ntasks) + ']'
    output = f"{fmt.format(n)}  α = {α:.3e}: "

    if status:
        print('\r{:<150s}'.format(output \
             + f'{σ.size:3d} eigenmode{'s' if σ.size > 1 else ' '}, ' \
             + f'σ₀ = {σ[0].real:.3e}{σ[0].imag:+.3e}j (tol = {e[0]:.3e}, ' \
             + f'C = {C:.3e}, N = {N}) {'Did not converge!' if e[0] > 1 else ''}'), \
                 end=end, flush=True)
        return α, σ[0].real, σ[0].imag, e[0], C, N
    else:
        print('\r{:<150s}'.format(output + '   NO eigenmodes!'), end=end, flush=True)
        return α, None, None, None, None, None


def task(dpath, dependence, value, ntask, ntasks, k, params):

    from scipy.optimize import bracket, minimize_scalar

    global counter

    status = True

    end='\n' if verbose else ''

    UP = '\033[F'
    info = f"  {dependence} = {value:+.3e}: "
    progress_line = ''
    result_line = ''

    params[dependence] = float(value)

    rtol   = params.get('rtol' , 1e-4)
    wtol   = params.get('wtol' , 1e-4)
    force  = params.get('force', False)

    α = k * a

    sname = os.path.join(params.get('data_path', './'), f'state_{dependence}{value:+.6e}.npz')

    if os.path.exists(sname) and not force:
        state = np.load(sname)

        α   = state['wavenumber']
        σ   = state['eigenvalues']
        e   = state['tolerances']
        δin = state['inner_scale']
        C   = state['scaling_factor']
        N   = state['resolution']
        nin = state['n_inner']
    else:
        try:
            params['α'] = α

            σ, v, e, δin, C, nin, N, status = eigenmodes(**params)

            if status:
                np.savez_compressed(sname, value=value, wavenumber=α, eigenvalues=σ, \
                            eigenvectors=v, tolerances=e, inner_scale=δin, \
                            scaling_factor=C, n_inner=nin, resolution=N)

        except Exception as ex:
            status = False
            if verbose:
                print(f"\n[WARNING] Could not find eigenmode for α = {α:.3e}: {ex}")

    with counter.get_lock():
        counter.value += 1
        n = counter.value

    progress   = n / ntasks
    percentage = int(progress * 100)

    fmt = r'[{:0' + str(len(str(ntasks))) + 'd}/' + str(ntasks) + ']'
    progress_line = f"Progress {percentage}% complete {'█' * (percentage // 2)}{' ' * (50 - (percentage // 2))} {fmt.format(n)}"

    if status:
        result_line = info + f"α={α:.4e}  σ={σ[0].real:.4e}  δin={δin:.3e}  C={C:.3e}  n={nin}  N={N}" + ' '*6
        if verbose:
            print(f"{result_line}", flush=True)
        else:
            print(f"\r{result_line}\n\n{progress_line}{UP}{UP}", end='', flush=True)

        return value, α, σ, δin, C, nin, N
    else:
        result_line  = info + "could not find any eigenvalue!" + ' '*80
        if verbose:
            print(f"{result_line}", flush=True)
        else:
            print(f"\r{result_line}\n\n{progress_line}{UP}{UP}", end='', flush=True)
        return value, None, None, None, None, None, None, None, None


def load_data(path: str, pattern: str = "*.npz"):
    p = Path(path)
    files = sorted(p.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found in {path}")
    rows = []
    for f in files:
        state = np.load(f)
        rows.append([state['value'], state['wavenumber'], state['eigenvalues'][0], state['tolerances'][0], state['inner_scale'], state['scaling_factor'], state['n_inner'], state['resolution']])

    rows.sort(key = lambda r: r[0])

    v = np.array([x[0] for x in rows])
    α = np.array([x[1] for x in rows])
    σ = np.array([x[2] for x in rows])
    e = np.array([x[3] for x in rows])
    δ = np.array([x[4] for x in rows])
    c = np.array([x[5] for x in rows])
    n = np.array([x[6] for x in rows])
    N = np.array([x[7] for x in rows])

    return v, α, σ, e, δ, c, n, N


if __name__ == "__main__":
    '''
        Given provided options, calculates eigenmodes of the equilibrium field with magnetic and velocity shear.
    '''
    import argparse, multiprocessing, os, sys, time
    import numpy as np

    sys.path.append('/home/gkowal/Research/Software/Codes/psecas/')
    sys.path.append('/home/gkowal/Shared/Work/Tearing Instability/psecas')

    parser = argparse.ArgumentParser(description="Calculates eigenmodes of the equilibrium field with magnetic and velocity shear.")
    parser.add_argument(
        "--CGL", "-CGL", action='store_true', default=False,
        help="Gyrotropic MHD"
    )
    parser.add_argument(
        "--dependence", "-d", default='S',
        help="the dependence of to calculate: S, Pr, β, ξ, ϵ, w, or a"
    )
    parser.add_argument(
        "--range", "-R", default=[0, 1, 0.1], type=float, nargs=3,
        help="the range of the dependent parameter to evaluate followed by the increment"
    )
    parser.add_argument(
        "--logarithmic", "-log", action='store_true', default=False,
        help="the dependence scale is logarithmic"
    )
    parser.add_argument(
        "--Lundquist-number", "-S", type=float, default=1e4,
        help="the Lundquist number"
    )
    parser.add_argument(
        "--Prandtl-number", "-Pr", type=float, default=0,
        help="the Prandtl number"
    )
    parser.add_argument(
        "--plasma-beta", "-β", type=float, default=0,
        help="the perpendicular plasma-β"
    )
    parser.add_argument(
        '--eos',
        choices=['adiabatic', 'polytropic', 'isothermal', 'custom'],
        default='adiabatic',
        help="Equation of state type. If 'custom' is selected, "
             "'--gamma-parallel' and '--gamma-perpendicular' are required."
    )
    parser.add_argument(
        "--gamma-parallel", "-ɣpar", type=float, default=3,
        help="the parallel adiabatic index"
    )
    parser.add_argument(
        "--gamma-perpendicular", "-ɣper", type=float, default=2,
        help="the perpendicular adiabatic index"
    )
    parser.add_argument(
        "--beta-difference", "-Δβ", type=float, default=0,
        help="the parallel to perpendicular plasma-β difference"
    )
    parser.add_argument(
        "--magnetic-transverse-field", "-ξ", type=float, default=0,
        help="the transverse magnetic strength"
    )
    parser.add_argument(
        "--hall", "-ϵ", type=float, default=0,
        help="the Hall current term strength"
    )
    parser.add_argument(
        "--thickness", "-a", type=float, default=1,
        help="the thickness of the current sheet"
    )
    parser.add_argument(
        "--width", "-w", type=float, default=0,
        help="the width of the current sheet"
    )
    parser.add_argument(
        "--resolution-range", "-N", type=int, nargs=3, default=[64, 2048, 32],
        help="the range of the grid resolutions followed by the increment"
    )
    parser.add_argument(
        "--n-inner", "-nin", type=int, default=3,
        help="Minimum number of collocation points required to resolve the smallest inner-layer width."
    )
    parser.add_argument(
        "--amp-fraction-outer", "--f-outer", type=float, default=0.01,
        help=("Fraction of the eigenmode amplitude that must be resolved "
              "at the outermost collocation point z_max. "
              "For example, 0.03 means the eigenmode amplitude should drop "
              "to 3%% of its maximum at z_max (equivalent to q = -ln(0.03) ≈ 3.51).")
    )
    parser.add_argument(
        "--scaling-factor", "-C", type=float, default=None,
        help="the scaling factor for the grid"
    )
    parser.add_argument(
        "--wavenumber", "-k", type=float, default=0.01,
        help="the wavenumber")
    parser.add_argument(
        "--growth-rate-range", "-E", type=float, nargs=2, default=[1e-5, 1],
        help="the range for the growth rate to consider"
    )
    parser.add_argument(
        "--absolute-tolerance", "-atol", type=float, default=1e-10,
        help="the absolute tolerance for the growth rate"
    )
    parser.add_argument(
        "--relative-tolerance", "-rtol", type=float, default=1e-5,
        help="the relative tolerance for the growth rate"
    )
    parser.add_argument(
        "--wavenumber-tolerance", "-ktol", type=float, default=1e-3,
        help="the relative tolerance for the maximum wavenumber"
    )
    parser.add_argument(
        "--guess-tolerance", "-gtol", type=float, default=1e-2,
        help="the guess tolerance for switching to iterative solver"
    )
    parser.add_argument(
        "--thickness-tolerance", "-δtol", type=float, default=1e-3,
        help="the inner-layer thickness estimation tolerance"
    )
    parser.add_argument(
        "--orderby", "-O", default="tolerance",
        help="the ordering of wavenumber: amplitude or tolerance"
    )
    parser.add_argument(
        "--modes", "-m", type=int, default=1,
        help="the eigenmode number"
    )
    parser.add_argument(
        "--suffix", "-s", default="",
        help="the suffix added to the output file"
    )
    parser.add_argument(
        "--force", "-f", action='store_true', default=False,
        help="force recalculations even though the eigenmodes are already stored in cache directory"
    )
    parser.add_argument(
        "--verbose", "-v", action='store_true', default=False,
        help="be verbose"
    )

    args = parser.parse_args()

    print("\nCalculation of the maximum growth rate dependence on several parameters for the selected eigenmode.")
    print("Use option '-h' to show all possible arguments.\n")

    omp_threads = os.getenv("OMP_NUM_THREADS")

    if 'SLURM_CPUS_PER_TASK' in os.environ:
        number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        number_of_cores = multiprocessing.cpu_count()

    S          = None if args.dependence == 'S'  else args.Lundquist_number
    Pr         = None if args.dependence == 'Pr' else args.Prandtl_number
    β          = None if args.dependence == 'β'  else args.plasma_beta
    Δβ         = None if args.dependence == 'Δβ' else args.beta_difference
    ξ          = None if args.dependence == 'ξ'  else args.magnetic_transverse_field
    ϵ          = None if args.dependence == 'ϵ'  else args.hall
    a          = None if args.dependence == 'a'  else args.thickness
    w          = None if args.dependence == 'w'  else args.width
    C          = args.scaling_factor
    Nmin       = max(args.resolution_range[0], args.resolution_range[2])
    Nmax       = args.resolution_range[1]
    Ninc       = args.resolution_range[2]
    modes      = args.modes
    verbose    = args.verbose
    log        = args.logarithmic
    if args.eos == 'adiabatic':
        ɣpar = 3
        ɣper = 2
    elif args.eos == 'polytropic':
        ɣpar = 0.5
        ɣper = 2
    elif args.eos == 'isothermal':
        ɣpar = 1
        ɣper = 1
    else:
        ɣpar = args.gamma_parallel
        ɣper = args.gamma_perpendicular

    if args.CGL:
        Dstring = { 'a': 'current sheet thickness', 'w': 'current sheet width', 'S': 'Lundquist number', 'Pr': 'Prandtl number', 'β': 'plasma-β', 'Δβ': 'plasma-β difference', 'ξ': 'magnetic transverse field', 'ϵ': 'Hall current term'}
    else:
        Dstring = { 'a': 'current sheet thickness', 'w': 'current sheet width', 'S': 'Lundquist number', 'Pr': 'Prandtl number', 'ξ': 'magnetic transverse field', 'ϵ': 'Hall current term'}

    print("Plasma parameters:")
    print(f"  Equations                          =  {'Gyrotropic' if args.CGL else 'Classical'} MHD")
    if args.dependence != 'S':
        print(f"  Lundquist number (S)               =  {S:.3e}")
    if args.dependence != 'Pr':
        print(f"  Prandtl number (Pr)                =  {Pr:.3e}")
    if args.CGL:
        if args.dependence != 'β':
            print(f"  Plasma-β (β)                       =  {β:.3e}")
        if args.dependence != 'Δβ':
            print(f"  Plasma-β difference (Δβ)           = {Δβ:+.3f}")
    if args.dependence != 'ξ':
        print(f"  Magnetic transverse field (ξ)      =  {ξ:.3e}")
    if args.dependence != 'ϵ':
        print(f"  Hall current term strength (ϵ)     =  {ϵ:.3e}")
    if args.CGL:
        print(f"  Parallel adiabatic index           =  {ɣpar}")
        print(f"  Perpendicular adiabatic index      =  {ɣper}")
    print("Equilibrium parameters:")
    if args.dependence != 'a':
        print(f"  Current sheet thickness (a)        =  {a:.3e}")
    if args.dependence != 'w':
        print(f"  Current sheet width (w)            =  {w:.3e}")
    print("Dependence parameters:")
    print(f"  Calculating dependence on          =  {Dstring[args.dependence]}")
    print(f"  Range of values for {args.dependence} (R) {' '*(9-len(args.dependence))} =  {'10^' if log else ''}{[args.range[0], args.range[1]]} with the increment of {args.range[2]}")
    print("Geometry/convergence parameters:")
    print(f"  Resolution range (N)               =  {[Nmin, Nmax]} with the increment of {Ninc}")
    if C != None:
        print(f"  Scaling factor (C)                 =  {C:.3e}")
    print(f"  Wavenumber                         =  {args.wavenumber:.3e}")
    print(f"  Growth rate range                  =  {args.growth_rate_range}")
    print(f"  Growth rate absolute tolerance     =  {args.absolute_tolerance:.3e}")
    print(f"  Growth rate relative tolerance     =  {args.relative_tolerance:.3e}")
    print(f"  Growth rate guess tolerance        =  {args.guess_tolerance:.3e}")
    print(f"  Wavenumber  absolute tolerance     =  {args.wavenumber_tolerance:.3e}")
    print(f"  Eigenmode order                    =  {args.orderby}")
    print(f"  Converge the number of modes       =  {args.modes}")
    print(f"  Inner-layer thickness tolerance    =  {args.thickness_tolerance:.3e}")
    print(f"  Number of inner collocation points =  {args.n_inner}")
    print(f"  Amplitude fraction at zmax         =  {args.amp_fraction_outer}")
    if len(args.suffix) > 0:
        print(f"Suffix                         =  {args.suffix}")

    if not args.CGL and args.dependence == 'Δβ':
        print("\nThe Δβ dependence works only with CGL-MHD!")
        sys.exit(1)
    if args.CGL:
        if args.dependence != 'β' and args.dependence != 'Δβ':
            if β + Δβ <= 0:
                print("\nThe choice of β and Δβ results in negative parallel pressure!")
                sys.exit(1)

    if omp_threads != "1":
        print("\n\033[1mPlease set OMP_NUM_THREADS=1 to ensure optimal performance!\033[0m")

    if args.range[1] > args.range[0]:
        if log:
            nn = int((args.range[1] - args.range[0]) / args.range[2]) + 1
            Rs = np.logspace(args.range[0], args.range[1], nn)
        else:
            n = round((args.range[1] - args.range[0]) / args.range[2]) + 1
            Rs = np.linspace(args.range[0], args.range[1], n, dtype='float64')
    else:
        Rs = np.array([ args.range[0] ], dtype='float64')
        n = 1

    number_of_cores = min(Rs.size, number_of_cores)
    number_of_tasks = len(Rs)

    params = {'Nmin'         : Nmin, \
              'Nmax'         : Nmax, \
              'Ninc'         : Ninc, \
              'CGL'          : args.CGL, \
              'n_inner'      : args.n_inner, \
              'decay_efolds' : -float(np.log(args.amp_fraction_outer)), \
              'C'        : None, \
              'a'        : a, \
              'w'        : w, \
              'S'        : S, \
              'Pr'       : Pr, \
              'ξ'        : ξ, \
              'ϵ'        : ϵ, \
              'β'        : β, \
              'Δβ'       : Δβ,
              'ɣpar'     : ɣpar, \
              'ɣper'     : ɣper, \
              'mode'     : max(0, args.modes-1), \
              'σlower'   : args.growth_rate_range[0], \
              'σupper'   : args.growth_rate_range[1], \
              'orderby'  : args.orderby, \
              'atol'     : args.absolute_tolerance, \
              'rtol'     : args.relative_tolerance, \
              'wtol'     : args.wavenumber_tolerance, \
              'gtol'     : args.guess_tolerance, \
              'δtol'         : args.thickness_tolerance, \
              'ntasks'   : number_of_tasks, \
              'force'    : args.force, \
              'verbose'  : args.verbose}

    if args.CGL:
        dpath  = './RESULTS/CGL-MHD'
    else:
        dpath  = './RESULTS/MHD'
    if not os.path.exists(dpath):
        os.makedirs(dpath)

    fname = os.path.join(dpath, '')
    for item in Dstring.keys():
        if item != args.dependence:
            if item in ['Δβ']:
                fname += f'{item}{params[item]:+.2f}'
            else:
                fname += f'{item}{params[item]:.3e}'
    fname += f"k{args.wavenumber:.3e}"
    fname += f"{args.suffix}"
    dpath  = fname + '-cache'
    fname += '.dat'

    params['data_path'] = dpath

    if not os.path.exists(dpath):
        os.makedirs(dpath)

    print(f"\nCalculation initiated with {number_of_cores} process{'es' if number_of_cores > 1 else ''} for {Rs.size} values.\n")

    counter = multiprocessing.Value('i', 0)

    start_time = time.time()

    with multiprocessing.Pool(number_of_cores) as pool:
        async_results = [ pool.apply_async(task, args=(dpath, args.dependence, r, ntask, number_of_tasks, args.wavenumber, params)) for ntask, r in enumerate(Rs) ]
        results = [ r.get() for r in async_results ]

    if os.path.exists(dpath):
        v, α, σ, e, δ, c, n, N = load_data(dpath)

        with open(fname, 'w') as io:
            io.write(f"#\n# Tearing Instability - mode {modes}\n#\n")
            io.write(f"# equations                            =   {'Gyrotropic' if args.CGL else 'Classical'} MHD\n")
            if args.dependence != 'S':
                io.write(f"# Lundquist number (S)                 =  {S:10.3e}\n")
            if args.dependence != 'Pr':
                io.write(f"# Prandtl number (Pr)                  =  {Pr:10.3e}\n")
            if args.CGL:
                if args.dependence != 'β':
                    io.write(f"# plasma-β (β)                         =  {β:10.3e}\n")
                if args.dependence != 'Δβ':
                    io.write(f"# plasma-β difference (Δβ)             =  {Δβ:10.3e}\n")
                    if Δβ != 0:
                        io.write(f"# parallel pressure                =  {(β+Δβ)/2:10.3e}\n")
                        io.write(f"# perpendicular pressure           =  {β/2:10.3e}\n")
            io.write(f"# magnetic transverse field (ξ)        =  {ξ:10.3e}\n")
            if args.dependence != 'ϵ':
                io.write(f"# Hall current term strength (ϵ)       =  {ϵ:10.3e}\n")
            if args.CGL:
                io.write(f"# parallel adiabatic index (ɣpar)      =  {ɣpar:10.3e}\n")
                io.write(f"# perpendicular adiabatic index (ɣper) =  {ɣper:10.3e}\n")
            if args.dependence != 'a':
                io.write(f"# current sheet thickness (a)          =  {a:10.3e}\n")
            if args.dependence != 'w':
                io.write(f"# current sheet half-width (w)         =  {w:10.3e}\n")
            io.write(f"# wavenumber (k)                       =  {args.wavenumber:.3f}\n")
            io.write(f"# resolution range (N)                 =  {[Nmin, Nmax]} with the increment of {Ninc}\n")
            io.write(f"# growth rate range                    =  {args.growth_rate_range}\n")
            io.write(f"# eigenvalue absolute tolerance        =  {args.absolute_tolerance:10.3e}\n")
            io.write(f"# eigenvalue relative tolerance        =  {args.relative_tolerance:10.3e}\n")
            io.write(f"# wavenumber relative tolerance        =  {args.wavenumber_tolerance:10.3e}\n")
            io.write(f"# selection order                      =   {args.orderby}\n")
            io.write(f"# inner-layer thickness tolerance      =  {args.thickness_tolerance:.3e}\n")
            io.write(f"# converge the number of modes         =   {args.modes}\n")
            io.write(f"# number of inner collocation points   =   {args.n_inner}\n")
            io.write(f"# amplitude fraction at zmax           =   {args.amp_fraction_outer}\n")
            io.write("#\n#")
            io.write(f"     {args.dependence:<2s}              α               σ               δ_in            tolerance       C              n_in      N\n")
            io.write("#  --------------  --------------  --------------  --------------  --------------  --------------  --------  --------\n")

            for i in range(v.size):
                io.write(f"  {v[i]:14.6e}  {α[i]:14.6e}  {σ[i].real:14.6e}  {δ[i]:14.6e}  {e[i]:14.6e}  {c[i]:14.6e}    {n[i]:>6d}    {N[i]:>6d}\n")
            io.flush()

    print(f"\n\n\n\n\nCalculations done in {time.time() - start_time:.2f} seconds.\n")
