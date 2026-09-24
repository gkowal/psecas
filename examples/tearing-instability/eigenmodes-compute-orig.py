#!/usr/bin/env python3
#
from tearing_eigenmodes import *

class DeltaError(ValueError):
    """Raised when Δ' <= 0 (stable mode at the given α) so the tearing eigenmode is not excited."""
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
    C = 1 - Δβ/2 if CGL else 1
    μ = np.sqrt((2 + (ɣpar + ɣper - 2) * β + ɣpar * Δβ) / (2 - Δβ)) if CGL else 1

    # α_m scaling
    #αm = (1.3621 + (-4.4446 / ((1.5284**np.log(S)) + 2.856))) * S**-0.25 * C**(-1/8) * μ**(-3/4) * ((0.05*Pr**2+0.7*Pr+1)/(12*Pr+1))**(1/8)
    αm = 0.37217 / (((S - 1.8454)**-0.43765) + 0.27338) * S**-0.25 * C**(-1/8) * μ**(-3/4) * ((0.05*Pr**2+0.7*Pr+1)/(12*Pr+1))**(1/8)
    #αm = 1.3583e+00 * (S / (S + 400))**(1/4) * S**(-1/4) * C**(-1/8) * μ**(-3/4) * ((0.05*Pr**2+0.7*Pr+1)/(12*Pr+1))**(1/8)
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
    C = 1.0 - Δβ/2 if CGL else 1
    μ = np.sqrt((2 + (ɣpar + ɣper - 2) * β + ɣpar * Δβ) / (2 - Δβ)) if CGL else 1
    λ = α / μ
    X = α * μ

    # --- Δ'(α); raise if stable at α
    Δ = 2.0 * (1.0 / X - X)
    if Δ <= 0.0 or np.isclose(Δ, 0.0):
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
    #δFKR = (0.9221 - 1.3493*S**-0.31361) * a * ((S * α)**-2 * a * Δ)**(1.0/5.0)

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
    α       = params.get('α'       ,   0.1)
    verbose = params.get('verbose' , False)

    try:
        from psecas import Solver, ChebyshevRationalGrid
        from psecas.systems.tearing import Tearing
        import numpy as np

        Nmin         = params.get('Nmin'    ,  128   )
        Nmax         = params.get('Nmax'    , 1024   )
        Ninc         = params.get('Ninc'    ,   32   )
        n_inner      = params.get('n_inner' ,    5   )
        decay_efolds = params.get('decay_efolds' , -np.log(0.001))
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
        reσlo    = params.get('σlower'  ,   1e-6 )
        reσup    = params.get('σupper'  ,   1    )
        imσamp   = params.get('imσamp'  ,   1e-8 )
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
                     periodic=False, Harris=False)
        solver  = Solver(grid, system, re_range=[reσlo, reσup], im_range=[-imσamp,imσamp])
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


def task(k, params):
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
             + f'σ₀ = {σ[0].real:.3e}{σ[0].imag:+.3e}j (δin = {δin:.3e}, ' \
             + f'tol = {e[0]:.3e}, C = {C:.3e}, N = {N}) ' \
             + f'{'Did not converge!' if e[0] > 1 else ''}'), \
                 end=end, flush=True)
        return α, σ[0].real, σ[0].imag, δin, e[0], C, N
    else:
        print('\r{:<150s}'.format(output + '   NO eigenmodes!'), end=end, flush=True)
        return α, None, None, None, None, None, None


if __name__ == "__main__":
    '''
        Given options, calculates the dispersion relation under the equilibrium field with magnetic and velocity shear.
    '''
    import argparse, multiprocessing, os, sys, time
    import numpy as np

    parser = argparse.ArgumentParser(description="Calculates eigenmodes of the equilibrium field with magnetic and velocity shear.")
    parser.add_argument(
        "--CGL", "-CGL", action='store_true', default=False,
        help="Gyrotropic MHD"
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
        required=True,
        choices=['adiabatic', 'polytropic', 'isothermal', 'custom'],
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
        "--Hall-parameter", "-ϵ", type=float, default=0,
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
        "--wavenumber-range", "-K", type=float, nargs=3, default=[0, 1, 0.01],
        help="the bounds and step for the wavenumber")
    parser.add_argument(
        "--resolution-range", "-N", type=int, nargs=3, default=[64, 2048, 32],
        help="the range of the grid resolutions followed by the increment"
    )
    parser.add_argument(
        "--scaling-factor", "-C", type=float, default=None,
        help="the scaling factor for the grid"
    )
    parser.add_argument(
        "--growth-rate-range", "-E", type=float, nargs=2, default=[1e-6, 1],
        help="the range for the growth rate to consider"
    )
    parser.add_argument(
        "--imaginary-amplitude-limit", "-I", type=float, default=1e-8,
        help="the maximum magnitude of the eigenvalue's imaginary part"
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
        "--guess-tolerance", "-gtol", type=float, default=1e-2,
        help="the guess tolerance for switching to iterative solver"
    )
    parser.add_argument(
        "--thickness-tolerance", "-δtol", type=float, default=1e-3,
        help="the inner-layer thickness estimation tolerance"
    )
    parser.add_argument(
        "--orderby", "-O", default="amplitude",
        help="the ordering of wavenumber: amplitude or tolerance"
    )
    parser.add_argument(
        "--modes", "-m", type=int, default=1,
        help="the eigenmode number"
    )
    parser.add_argument(
        "--logarithmic", "-log", action='store_true', default=False,
        help="the wave number scale is logarithmic"
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
    parser.add_argument(
        "--n-inner", "-nin", type=int, default=5,
        help="Minimum number of collocation points required to resolve the smallest inner-layer width."
    )
    parser.add_argument(
        "--amp-fraction-outer", "--f-outer", type=float, default=0.001,
        help=("Fraction of the eigenmode amplitude that must be resolved "
              "at the outermost collocation point z_max. "
              "For example, 0.01 means the eigenmode amplitude should drop "
              "to 1%% of its maximum at z_max (equivalent to q = -ln(0.01) ≈ 4.605).")
    )

    args = parser.parse_args()

    counter = None

    omp_threads = os.getenv("OMP_NUM_THREADS")

    start_time = time.time()

    if 'SLURM_CPUS_PER_TASK' in os.environ:
        number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        number_of_cores = multiprocessing.cpu_count()
    if omp_threads != "1":
        number_of_cores = 1

    C          = args.scaling_factor
    a          = args.thickness
    w          = args.width
    S          = args.Lundquist_number
    Pr         = args.Prandtl_number
    ξ          = args.magnetic_transverse_field
    ϵ          = args.Hall_parameter
    β          = args.plasma_beta
    Δβ         = args.beta_difference
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
    if args.logarithmic:
        kmin       = args.wavenumber_range[0]
    else:
        kmin       = max(args.wavenumber_range[0], args.wavenumber_range[2])
    kmax       = args.wavenumber_range[1]
    kinc       = args.wavenumber_range[2]
    Nmin       = max(args.resolution_range[0], args.resolution_range[2])
    Nmax       = args.resolution_range[1]
    Ninc       = args.resolution_range[2]
    re_range   = args.growth_rate_range
    abstol     = args.absolute_tolerance
    reltol     = args.relative_tolerance

    if args.logarithmic:
        ks     = 10**np.arange(kmin, kmax+kinc, kinc, dtype='float64')
    else:
        ks     = np.arange(round(kmin/kinc), round((kmax+kinc)/kinc)) * kinc
    ks /= a

    number_of_tasks = ks.size
    number_of_cores = min(ks.size, number_of_cores)

    if args.CGL:
        dpath  = './RESULTS/CGL-MHD/S{:.1e}_Pr{:.1e}_β{:.1e}_Δβ{:+.2f}_ξ{:.1e}_ϵ{:.1e}_a{:.1e}_w{:.3f}'.format(S, Pr, β, Δβ, ξ, ϵ, a, w)
    else:
        dpath  = './RESULTS/MHD/S{:.1e}_Pr{:.1e}_ξ{:.1e}_ϵ{:.1e}_a{:.1e}_w{:.3f}'.format(S, Pr, ξ, ϵ, a, w)
    if C != None:
        dpath += f'-C{C:.3f}'
    if args.logarithmic:
        dpath += '_log'
    dpath += f'{args.suffix}'
    if not os.path.exists(dpath):
        os.makedirs(dpath)
    prefix = os.path.join(dpath, '{}')

    print('\nCalculation of the eigenmodes for an equlibrium with the velocity and magnetic field shear.')
    print("Use option '-h' to show all possible arguments.\n")

    print("Plasma parameters:")
    print(f"  Equations                          =  {'Gyrotropic' if args.CGL else 'Classical'} MHD")
    print(f"  Lundquist number (S)               =  {S:.3e}")
    print(f"  Prandtl number (Pr)                =  {Pr:.3e}")
    if args.CGL:
        print(f"  Plasma-β (β)                       =  {β:.3e}")
        print(f"  Plasma-β difference (Δβ)           = {Δβ:+.3f}")
    print(f"  Magnetic transverse field (ξ)      =  {ξ:.3e}")
    print(f"  Hall current term strength (ϵ)     =  {ϵ:.3e}")
    if args.CGL:
        print(f"  Parallel adiabatic index           =  {ɣpar}")
        print(f"  Perpendicular adiabatic index      =  {ɣper}")
    print("Equilibrium parameters:")
    print(f"  Current sheet thickness (a)        =  {a:.3e}")
    print(f"  Current sheet width (w)            =  {w:.3e}")
    print("Wavenumbers:")
    print(f"  Wavenumber range (k)               =  {'10^' if args.logarithmic else ''}{[kmin, kmax]} with the increment of {kinc}")
    print("Geometry/convergence parameters:")
    print(f"  Resolution range (N)               =  {[Nmin, Nmax]} with the increment of {Ninc}")
    if C != None:
        print(f"  Scaling factor (C)                 =  {C:.3e}")
    print(f"  Growth rate range                  =  {args.growth_rate_range}")
    print(f"  Growth rate absolute tolerance     =  {args.absolute_tolerance:.3e}")
    print(f"  Growth rate relative tolerance     =  {args.relative_tolerance:.3e}")
    print(f"  Growth rate guess tolerance        =  {args.guess_tolerance:.3e}")
    print(f"  Imaginary amplitude limit          =  {args.imaginary_amplitude_limit:.3e}")
    print(f"  Eigenmode order                    =  {args.orderby}")
    print(f"  Converge the number of modes       =  {args.modes}")
    print(f"  Inner-layer thickness tolerance    =  {args.thickness_tolerance:.3e}")
    print(f"  Number of inner collocation points =  {args.n_inner}")
    print(f"  Amplitude fraction at zmax         =  {args.amp_fraction_outer}")
    if len(args.suffix) > 0:
        print(f"Suffix                               =  {args.suffix}")

    if β + Δβ < 0:
        print("\nThe choice of β and Δβ results in negative parallel pressure!")
        sys.exit(1)

    if ks.size > 1 and omp_threads != "1":
        print("\n\033[1mPlease set OMP_NUM_THREADS=1 to ensure optimal performance!\033[0m")

    params = {'data_path'    : dpath, \
              'Nmin'         : Nmin, \
              'Nmax'         : Nmax, \
              'Ninc'         : Ninc, \
              'n_inner'      : args.n_inner, \
              'decay_efolds' : -np.log(args.amp_fraction_outer), \
              'CGL'          : args.CGL, \
              'C'            : None, \
              'a'            : args.thickness, \
              'w'            : args.width, \
              'α'            : None, \
              'S'            : S, \
              'Pr'           : Pr, \
              'β'            : β, \
              'Δβ'           : Δβ,
              'ɣpar'         : ɣpar, \
              'ɣper'         : ɣper, \
              'modes'        : args.modes, \
              'σlower'       : args.growth_rate_range[0], \
              'σupper'       : args.growth_rate_range[1], \
              'imσamp'       : args.imaginary_amplitude_limit, \
              'orderby'      : args.orderby, \
              'atol'         : args.absolute_tolerance, \
              'rtol'         : args.relative_tolerance, \
              'gtol'         : args.guess_tolerance, \
              'δtol'         : args.thickness_tolerance, \
              'ntasks'       : number_of_tasks, \
              'force'        : args.force, \
              'verbose'      : args.verbose}

    with open(os.path.join(dpath, 'info.txt'), 'w') as io:
        io.write("Plasma parameters:\n")
        io.write(f"  Equations                          =  {'Gyrotropic' if args.CGL else 'Classical'} MHD\n")
        io.write(f"  Lundquist number (S)               =  {S:.3e}\n")
        io.write(f"  Prandtl number (Pr)                =  {Pr:.3e}\n")
        if args.CGL:
            io.write(f"  Plasma-β (β)                       =  {β:.3e}\n")
            io.write(f"  Plasma-β difference (Δβ)           = {Δβ:+.3f}\n")
        io.write(f"  Magnetic transverse field (ξ)      =  {ξ:.3e}\n")
        io.write(f"  Hall current term strength (ϵ)     =  {ϵ:.3e}\n")
        if args.CGL:
            io.write(f"  Parallel adiabatic index           =  {ɣpar}\n")
            io.write(f"  Perpendicular adiabatic index      =  {ɣper}\n")
            if Δβ != 0:
                io.write(f"  Parallel pressure                  =  {(β+Δβ)/2:10.3e}\n")
                io.write(f"  Perpendicular pressure             =  {β/2:10.3e}\n")
        io.write("Equilibrium parameters:\n")
        io.write(f"  Current sheet thickness (a)        =  {a:.3e}\n")
        io.write(f"  Current sheet width (w)            =  {w:.3e}\n")
        io.write("Wavenumbers:\n")
        io.write(f"  wavenumber scale                   =  {'logarithmic' if args.logarithmic else 'linear'}\n")
        io.write(f"  Wavenumber range (k)               =  {'10^' if args.logarithmic else ''}{[kmin, kmax]} with the increment of {kinc}\n")
        io.write("Geometry/convergence parameters:\n")
        io.write(f"  Resolution range (N)               =  {[Nmin, Nmax]} with the increment of {Ninc}\n")
        if C != None:
            io.write(f"  Scaling factor (C)                 =  {C:.3e}\n")
        io.write(f"  Growth rate range                  =  {args.growth_rate_range}\n")
        io.write(f"  Growth rate absolute tolerance     =  {args.absolute_tolerance:.3e}\n")
        io.write(f"  Growth rate relative tolerance     =  {args.relative_tolerance:.3e}\n")
        io.write(f"  Growth rate guess tolerance        =  {args.guess_tolerance:.3e}\n")
        io.write(f"  Imaginary amplitude limit          =  {args.imaginary_amplitude_limit:.3e}\n")
        io.write(f"  Eigenmode order                    =  {args.orderby}\n")
        io.write(f"  Converge the number of modes       =  {args.modes}\n")
        io.write(f"  Inner-layer thickness tolerance    =  {args.thickness_tolerance:.3e}\n")
        io.write(f"  Number of inner collocation points =  {args.n_inner}\n")
        io.write(f"  Amplitude fraction at zmax         =  {args.amp_fraction_outer}\n")

    fname = dpath + '.dat'

    print('\nCalculation initiated with {} process{} for {} values.'.format(number_of_cores, 'es' if number_of_cores > 1 else '', ks.size))
    counter = multiprocessing.Value('i', 0)

    with multiprocessing.Pool(number_of_cores) as pool:
        async_results = [ pool.apply_async(task, args=(k, params)) for k in ks ]
        results = [ r.get() for r in async_results ]

        with open(fname, 'w') as io:
            io.write("#\n# Tearing Instability\n#\n")
            io.write(f"# Lundquist number (S)                 =  {S:.3e}\n")
            io.write(f"# Prandtl number (Pr)                  =  {Pr:.3e}\n")
            if args.CGL:
                io.write(f"# Plasma-β (β)                         =  {β:.3e}\n")
                io.write(f"# Plasma-β difference (Δβ)             = {Δβ:+.3f}\n")
                io.write(f"# Parallel adiabatic index (ɣpar)      =  {ɣpar}\n")
                io.write(f"# Perpendicular adiabatic index (ɣper) =  {ɣper}\n")
            io.write(f"# Magnetic transverse field (ξ)        = {ξ:.3e}\n")
            io.write(f"# Hall current term strength (ϵ)       = {ϵ:.3e}\n")
            io.write(f"# Current sheet thickness (a)          = {a:.3e}\n")
            io.write(f"# Current sheet half-width (w)         = {w:.3e}\n")
            io.write("#\n")
            io.write("#    α                Re(σ)            Im(σ)            δ_in             tol              C                N\n")
            io.write("# ---------------- ---------------- ---------------- ---------------- ---------------- ----------------  -------\n#\n")
            io.flush()

            for r in results:
                if all(x is not None for x in r):
                    io.write("  {:15.8e}  {:15.8e}  {:15.8e}  {:15.8e}  {:15.8e}  {:15.8e}  {:6d}\n".format(*r))
            io.flush()

    sys.stdout.write('\nCalculation done in {:.2f} seconds.\n\n'.format(time.time() - start_time))

    with open(os.path.join(dpath, 'info.txt'), 'a') as io:
        io.write("\nCalculation done in {:.2f} seconds.\n".format(time.time() - start_time))
