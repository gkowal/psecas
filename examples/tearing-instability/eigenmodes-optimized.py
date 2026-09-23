#!/usr/bin/env python3
#
import numpy as np
import os, sys

from functools import lru_cache
from pathlib import Path


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
    if CGL and ((2 * ɣpar - 1) * Δβ <= - (2.0 + (ɣpar + ɣper - 2) * β) or Δβ >= 2.0):
        raise DeltaError(f"Δ' purely imaginary for Δβ = {Δβ:+.3e} => stable eigenmode for any α.")

    # --- CGL anisotropy factor and scaled wavenumbers
    μ = np.sqrt((1 + 0.5 * (ɣpar + ɣper - 2) * β + (ɣpar - 0.5) * Δβ) / (1 - Δβ / 2)) if CGL else 1

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
    verbose      = params.get('verbose'     , False  )

    # Enforce odd n_inner (so m is integer and z=0 is a collocation point)
    if n_inner % 2 == 0:
        n_inner += 1
    m = (n_inner - 1) / 2

    # --- Δ'(α); raise if stable at α
    if CGL and ((2 * ɣpar - 1) * Δβ <= - (2.0 + (ɣpar + ɣper - 2) * β) or Δβ >= 2.0):
        raise DeltaError(f"Δ' purely imaginary for Δβ = {Δβ:+.3e} => stable eigenmode for α = {α:.3e}.")

    # --- CGL anisotropy factor and scaled wavenumbers
    μ = np.sqrt((1 + 0.5 * (ɣpar + ɣper - 2) * β + (ɣpar - 0.5) * Δβ) / (1 - Δβ / 2)) if CGL else 1
    λ = α / μ
    X = α * μ

    # --- Δ'(α); raise if stable at α
    Δ = 2 * (1 / X - X)
    if Δ <= 0.0:
        raise DeltaError(f"Δ' <= 0 (Δ = {Δ:.3e}) ⇒ stable eigenmode for α = {α:.3e}.")

    # α_m scaling and corresponding Δ'(α_m)
    αm, Δm = estimate_max(**params)

    # --- Prefactors and inner-layer widths (soft Pr factors kept)
    cFKR = ((1 + 1e-4 * Pr) / S**2 * Δm / αm)**(-1/5) / (Δm * a)
    cCop = (a * (S * αm)**(1/3) * (1 + Pr)**(-1/6)) / Δm

    δFKR = cFKR * a * ((1 + 1e-4 * Pr) / S**2 * Δ / α)**(1/5)
    δCop = cCop * a * (S * α)**(-1/3) * (1 + Pr)**(1/6)
    δmin = min(a, δFKR, δCop)
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
                     gtol=gtol, atol=atol, rtol=rtol, re_range=[σlower, σupper], \
                     allmodes=True, allgrids=allgrids, orderby=orderby, verbose=verbose)
        N = solver.grid.N

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


def eigenmodes_old(**params):
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
        'abstol'   (float, default=1e-10): Absolute tolerance for convergence.
        'reltol'   (float, default=1e-5 ): Relative tolerance for convergence.
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
        abstol   = params.get('abstol'  ,   1e-10)
        reltol   = params.get('reltol'  ,   1e-5 )
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
            Nlow, Cout, Cin = select_NC(**params)
            C = (Cout + Cin) / 2
        else:
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
        σ, v, e = solver.iterate_solve_multimode(Ns, maxmode=mode, \
                     atol=abstol, rtol=reltol, re_range=[σlower, σupper], \
                     allmodes=True, allgrids=allgrids, orderby=orderby, verbose=verbose)
        N = solver.grid.N

        if verbose:
            I = np.where(np.abs(system.grid.zg) <= a)
            n = I[0].size
            print(f'Calculation done for α = {α:.4e} with C = {C:.3e} ({n} points over the interval |z| < a):')
            print(f'  σ₀ = {σ[0].real:.4e}{σ[0].imag:+.4e}j (error = {e[0]:.3e}, N = {N})')

        return σ, v, e, C, N, True

    except DeltaError as ex:
        if verbose:
            print(f"Stable eigenmode: {ex}")
        return None, None, None, None, None, False

    except ConvergenceError as ex:
        if verbose:
            print(f"Insufficient resolution: {ex}")
        return None, None, None, None, None, False

    except ValueError as ex:
        if verbose:
            print(f"Wrong parameter: {ex}")
        return None, None, None, None, None, False

    except Exception as ex:
        if verbose:
            print(f"[WARNING] Solver failed for α = {α:.3e}: {ex}")
        return None, None, None, None, None, False


@lru_cache(maxsize=1024)
def f_cached(x, **params):
    # Clo = params.get('Clower', 1e-01)
    # Cup = params.get('Cupper', 3e+01)
    #
    # if x < Clo or x > Cup:
    #     return np.inf

    params['C'] = x
    _, _, e, _, _, _, _, status = eigenmodes(**params)
    if status:
        return e[0]
    else:
        return np.inf


def f_scalar(x, **params):
    return f_cached(round(float(x), 12), **params)


def task(α, params):
    from scipy.optimize import bracket, minimize_scalar
    import os

    global counter

    verbose  = params.get('verbose' , False)
    ntasks   = params.get('ntasks'  , 1)
    S        = params.get('S'       , 1e4)
    Pr       = params.get('Pr'      , 0)
    factol   = params.get('factol'  , 1e-3)
    CGL      = params.get('CGL'     , False  )
    if CGL:
        ɣpar = params.get('ɣpar'    ,   3    )
        ɣper = params.get('ɣper'    ,   2    )

    end = '\n' if verbose else ''

    passed = False

    Cref = 4.623566167 * α**(-2/3) * S**(-1/4)

    prefix = os.path.join(params.get('data_path', './'), '{}' + f'_α{α:.6e}.npz')

    sname = prefix.format('state')

    status = True

    UP = '\033[F'
    info = f"  α = {α:.3e}: "
    progress_line = ''
    bracket_line = ''
    result_line = ''

    if os.path.exists(sname):
        state = np.load(sname)
        σ = state['eigenvalue']
        v = state['eigenvector']
        e = state['error']
        C = state['scaling_factor']
        N = state['resolution']
        i = state['iterations']
    else:
        try:
            params['α'] = α

            Nlow, Clo, Cup, _ = select_NC(**params)

            Nmin = params.get('Nmin',  128)
            Nmax = params.get('Nmax', 1024)
            Ninc = params.get('Ninc',   32)
            params['Clower']   = Clo
            params['Cupper']   = Cup
            params['Nmin']     = Nlow
            params['Nmax']     = Nlow + 3 * Ninc
            params['allgrids'] = True

            Cl, Cu = Clo, (Cup + 1.618 * Clo) / 2.618
            xa, xb, xc, fa, fb, fc, fn = bracket(lambda x: f_scalar(x, **params), xa=Cl, xb=Cu)
            if xa > xc:
                xa, xc, fa, fc = xc, xa, fc, fa
            bracket_line = info + f"bracket for C = ({xa:.2e}, {xb:.2e}, {xc:.2e}), Errors = ({fa:.2e}, {fb:.2e}, {fc:.2e}) after {fn} function calls"
            if verbose:
                print(f"{bracket_line}", flush=True)
            else:
                print(f"\r{bracket_line}\n{result_line}\n\n{progress_line}{UP}{UP}{UP}", end='', flush=True)
            bracket_line = ''

            try:
                res = minimize_scalar(lambda x: f_scalar(x, **params), bracket=(xa, xb, xc), tol=factol, method='brent')

                i = res.nit
                C = res.x
                params['C']        = C
                params['Nmax']     = Nmax
                params['allgrids'] = False

                if verbose:
                    print(res)
                    print(f"Clower = {Clo:.4e}, C = {C:.4e}, Cupper = {Cup:.4e}")
                    print('Final refinement:')

                σ, v, e, _, N, status = eigenmodes(**params)

                if status:
                    if σ.size > 0:
                        σ = σ[0]
                        v = v[0]
                        e = e[0]

                    np.savez_compressed(sname, wavenumber=α, eigenvalue=σ, eigenvector=v, error=e, scaling_factor=res.x, resolution=N, iterations=i)

            except RuntimeError as ex:
                status = False
                if verbose:
                    σlower = params.get('σlower'  ,   1e-6)
                    σupper = params.get('σupper'  ,   1   )
                    print(f"\nThe brackets found for α = {α:.3e}, however, could not find the minimum!")
                    print(f"Initial brackets for the scaling factor: {Clo:.4e} ... {Cup:.4e}.")
                    print(f"Eigenvalue brackets: {σlower:.4e} ... {σupper:.4e}.")
                    print(f"{ex}")

        except DeltaError as ex:
            status = False
            if verbose:
                print(f"Stable eigenmode: {ex}")

        except ValueError as ex:
            status = False
            if verbose:
                print(f"Wrong parameter: {ex}")

        except Exception as ex:
            status = False
            if verbose:
                print(f"\nCould not find brackets for α = {α:.3e}: {ex}")

    with counter.get_lock():
        counter.value += 1
        n = counter.value

    progress   = n / ntasks
    percentage = int(progress * 100)

    fmt = r'[{:0' + str(len(str(ntasks))) + 'd}/' + str(ntasks) + ']'
    progress_line = f"Progress {percentage}% complete {'█' * (percentage // 2)}{' ' * (50 - (percentage // 2))} {fmt.format(n)}"

    if status:
        result_line = info + f'σ = {σ.real:.3e} (tolerance = {e:.3e}, C = {C:.3e}, N = {N}, {i} iterations) {'Did not converge!' if e > 1 else ''}'
        if verbose:
            print(f"{result_line}", flush=True)
        else:
            print(f"\r{bracket_line}\n{result_line}\n\n{progress_line}{UP}{UP}{UP}", end='', flush=True)

        return α, σ.real, σ.imag, e, C, N
    else:
        bracket_line = info + 'could not find any bracket!'
        result_line  = info + 'could not find any minimum!'
        if verbose:
            print(f"{result_line}", flush=True)
        else:
            print(f"\r{bracket_line}\n{result_line}\n\n{progress_line}{UP}{UP}{UP}", end='', flush=True)
        return α, None, None, None, None, None


if __name__ == "__main__":
    '''
        Given provided options, calculates eigenmodes of the equilibrium field with magnetic and velocity shear.
    '''
    import argparse, multiprocessing, os, sys, time
    import numpy as np

    sys.path.append('/home/gkowal/Research/Software/Codes/psecas/')
    sys.path.append('/cluster/home/gkowal/Codes/psecas/')

    counter = None

    parser = argparse.ArgumentParser(description='Calculates eigenmodes of the equilibrium field with magnetic and velocity shear.')
    parser.add_argument(
        "--CGL", "-CGL", action='store_true', default=False,
        help='Gyrotropic MHD'
    )
    parser.add_argument(
        "--n-inner", "-nin", type=int, default=5,
        help="Minimum number of collocation points required to resolve the smallest inner-layer width."
    )
    parser.add_argument(
        "--amp-fraction-outer", "--f-outer", type=float, default=0.03,
        help=("Fraction of the eigenmode amplitude that must be resolved "
              "at the outermost collocation point z_max. "
              "For example, 0.03 means the eigenmode amplitude should drop "
              "to 3%% of its maximum at z_max (equivalent to q = -ln(0.03) ≈ 3.51).")
    )
    parser.add_argument('--range', '-R', default=[0, 1, 0.1], type=float, nargs=3, help='the range of the dependent parameter to evaluate followed by the increment')
    parser.add_argument('--resolution-range', '-N', default=[128, 1024, 32], type=int, nargs=3, help='the range of the grid resolutions followed by the increment for the final growth rate estimation')
    parser.add_argument('--Lundquist-number', '-S', default=1e4, type=float, help='the Lundquist number')
    parser.add_argument('--Prandtl-number', '-Pr', default=0, type=float, help='the Prandtl number')
    parser.add_argument('--plasma-beta', '-β', default=1, type=float, help='the perpendicular plasma-β')
    parser.add_argument(
        '--eos',
        required=True,
        choices=['adiabatic', 'polytropic', 'isothermal', 'custom'],
        help="Equation of state type. If 'custom' is selected, "
             "'--gamma-parallel' and '--gamma-perpendicular' are required."
    )
    parser.add_argument('--gamma-parallel', '-ɣpar', default=3, type=float, help='the parallel adiabatic index')
    parser.add_argument('--gamma-perpendicular', '-ɣper', default=2, type=float, help='the perpendicular adiabatic index')
    parser.add_argument('--beta-difference', '-Δβ', default=0, type=float, help='the parallel to perpendicular plasma-β difference')
    parser.add_argument('--magnetic-transverse-field', '-ξ', default=0, type=float, help='the transverse magnetic strength')
    parser.add_argument('--hall', '-ϵ', default=0, type=float, help='the Hall current term strength')
    parser.add_argument('--thickness', '-a', default=1, type=float, help='the thickness of the current sheet')
    parser.add_argument('--width', '-w', default=0, type=float, help='the width of the current sheet')
    parser.add_argument('--wavenumber-range', '-K', default=[0, 1, 0.01], type=float, nargs=3, help='the bounds and step for the wavenumber')
    parser.add_argument('--logarithmic', '-log', default=False, action='store_true', help='the wavenumber scale is logarithmic')
    parser.add_argument('--eigenvalue-brackets', '-E', default=[1e-5, 1], type=float, nargs=2, help='the brackets for the real parts of the eigenvalues')
    parser.add_argument('--absolute-tolerance', '-atol', default=1e-8, type=float, help='the absolute tolerance for the growth rate')
    parser.add_argument('--relative-tolerance', '-rtol', default=1e-4, type=float, help='the relative tolerance for the growth rate')
    parser.add_argument('--factor-tolerance', '-tol', default=1e-3, type=float, help='the absolute tolerance for the scaling factor')
    parser.add_argument('--orderby', '-O', default='amplitude', help='the ordering of wavenumber: amplitude or tolerance')
    parser.add_argument('--mode', '-m', default=0, type=int, help='the eigenmode number')
    parser.add_argument('--suffix', '-s', default='', help='the suffix added to the output file')
    parser.add_argument('--verbose', '-v', default=False, action='store_true', help='be verbose')

    args = parser.parse_args()

    omp_threads = os.getenv("OMP_NUM_THREADS")

    if 'SLURM_CPUS_PER_TASK' in os.environ:
        number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        number_of_cores = multiprocessing.cpu_count()

    a          = args.thickness
    w          = args.width
    S          = args.Lundquist_number
    Pr         = args.Prandtl_number
    β          = args.plasma_beta
    Δβ         = args.beta_difference
    ξ          = args.magnetic_transverse_field
    ϵ          = args.hall
    if args.logarithmic:
        kmin       = args.wavenumber_range[0]
    else:
        kmin       = max(args.wavenumber_range[0], args.wavenumber_range[2])
    kmax       = args.wavenumber_range[1]
    kinc       = args.wavenumber_range[2]
    Nmin       = max(args.resolution_range[0], args.resolution_range[2])
    Nmax       = args.resolution_range[1]
    Ninc       = args.resolution_range[2]
    brackets   = args.eigenvalue_brackets
    abstol     = args.absolute_tolerance
    reltol     = args.relative_tolerance
    factol     = args.factor_tolerance
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

    print('\nCalculation of the maximum growth rate dependence on several parameters for the selected eigenmode.')
    print("Use option '-h' to show all possible arguments.\n")

    if a <= 0:
        raise ValueError("The current sheet thickness must be positive.")
    if not log and kmin <  0:
        raise ValueError('The minimum wavenumber cannot be negative.')
    if kmax <  kmin:
        raise ValueError('Wrong order for wavenumber limits.')
    if kinc <= 0:
        raise ValueError('The wavenumber increment must be positive.')

    print("Plasma parameters:")
    print(f"  Equations                          = {'Gyrotropic' if args.CGL else 'Classical'} MHD")
    print(f"  Lundquist number (S)               = {S:.3e}")
    print(f"  Prandtl number (Pr)                = {Pr:.3e}")
    if args.CGL:
        print(f"  Plasma-β (β)                       = {β:.3e}")
        print(f"  Plasma-β difference (Δβ)           = {Δβ:.3f}")
    print(f"  Magnetic transverse field (ξ)      = {ξ:.3e}")
    print(f"  Hall current term strength (ϵ)     = {ϵ:.3e}")
    if args.CGL:
        print(f"  Parallel adiabatic index           = {ɣpar}")
        print(f"  Perpendicular adiabatic index      = {ɣper}")
    print("Equilibrium parameters:")
    print(f"  Current sheet thickness (a)        = {a:.3e}")
    print(f"  Current sheet width (w)            = {w:.3e}")
    print("Wavenumber parameters:")
    if args.logarithmic:
        print(f"  wavenumber range (k)               = 10^{[kmin, kmax]} with the increment of {kinc}")
    else:
        print(f"  wavenumber range (k)               = {[kmin, kmax]} with the increment of {kinc}")
    print("Geometry/convergence parameters:")
    print(f"  Resolution range (N)               = {[Nmin, Nmax]} with the increment of {Ninc}")
    print(f"  Eigenvalue brackets                = {brackets}")
    print(f"  Growth rate absolute tolerance     = {abstol:.3e}")
    print(f"  Growth rate relative tolerance     = {reltol:.3e}")
    print(f"  Factor absolute tolerance          = {factol:.3e}")
    print(f"  Eigenmode order                    = {args.orderby}")
    print(f"  Converge the number of modes       = {args.mode}")
    print(f"  Number of inner collocation points = {args.n_inner}")
    print(f"  Amplitude fraction at zmax         = {args.amp_fraction_outer}")
    if len(args.suffix) > 0:
        print(f"Suffix                         = {args.suffix}")

    if omp_threads != "1":
        print("\n\033[1mPlease set OMP_NUM_THREADS=1 to ensure optimal performance!\033[0m")

    if args.logarithmic:
        ks     = 10**np.arange(kmin, kmax+kinc, kinc, dtype='float64')
    else:
        ks     = np.arange(round(kmin/kinc), round((kmax+kinc)/kinc)) * kinc
    ks /= a

    number_of_tasks = ks.size
    number_of_cores = min(number_of_tasks, number_of_cores)

    if args.CGL:
        dpath  = './RESULTS/CGL-MHD/' \
               + f'S{S:.1e}_Pr{Pr:.1e}_β{β:.1e}_Δβ{Δβ:+.2f}_ξ{ξ:.1e}_ϵ{ϵ:.1e}_a{a:.1e}_w{w:.3f}'
    else:
        dpath  = './RESULTS/MHD/' \
               + f'S{S:.1e}_Pr{Pr:.1e}_ξ{ξ:.1e}_ϵ{ϵ:.1e}_a{a:.1e}_w{w:.3f}'
    if args.logarithmic:
        dpath += '_log'
    dpath += f'{args.suffix}'
    if not os.path.exists(dpath):
        os.makedirs(dpath)
    fpath = dpath + '.dat'

    params = {'data_path'    : dpath, \
              'Nmin'         : Nmin, \
              'Nmax'         : Nmax, \
              'Ninc'         : Ninc, \
              'CGL'          : args.CGL, \
              'C'            : None, \
              'n_inner'      : args.n_inner, \
              'decay_efolds' : -float(np.log(args.amp_fraction_outer)), \
              'α'            : None, \
              'S'            : S, \
              'Pr'           : Pr, \
              'β'            : β, \
              'Δβ'           : Δβ,
              'ɣpar'         : ɣpar, \
              'ɣper'         : ɣper, \
              'mode'         : args.mode, \
              'σlower'       : brackets[0], \
              'σupper'       : brackets[1], \
              'orderby'      : args.orderby, \
              'abstol'       : abstol, \
              'reltol'       : reltol, \
              'factol'       : factol, \
              'ntasks'       : number_of_tasks, \
              'verbose'      : args.verbose}

    with open(os.path.join(dpath, 'info.txt'), 'w') as io:
        io.write("Plasma parameters:\n")
        io.write(f"  Equations                          = {'Gyrotropic' if args.CGL else 'Classical'} MHD\n")
        io.write(f"  Lundquist number (S)               = {S:.3e}\n")
        io.write(f"  Prandtl number (Pr)                = {Pr:.3e}\n")
        if args.CGL:
            io.write(f"  Plasma-β (β)                       = {β:.3e}\n")
            io.write(f"  Plasma-β difference (Δβ)           = {Δβ:+.3f}\n")
        io.write(f"  Magnetic transverse field (ξ)      = {ξ:.3e}\n")
        io.write(f"  Hall current term strength (ϵ)     = {ϵ:.3e}\n")
        if args.CGL:
            io.write(f"  Parallel adiabatic index           = {ɣpar}\n")
            io.write(f"  Perpendicular adiabatic index      = {ɣper}\n")
        io.write("Equilibrium parameters:\n")
        io.write(f"  Current sheet thickness (a)        = {a:.3e}\n")
        io.write(f"  Current sheet width (w)            = {w:.3e}\n")
        io.write("Wavenumber parameters:\n")
        io.write(f"  wavenumber scale                   =  {'logarithmic' if args.logarithmic else 'linear'}\n")
        if args.logarithmic:
            io.write(f"  wavenumber range (k)               = 10^{[kmin, kmax]} with the increment of {kinc}\n")
        else:
            io.write(f"  wavenumber range (k)               = {[kmin, kmax]} with the increment of {kinc}\n")
        io.write("Geometry/convergence parameters:\n")
        io.write(f"  Resolution range (N)               = {[Nmin, Nmax]} with the increment of {Ninc}\n")
        io.write(f"  Eigenvalue brackets                = {brackets}\n")
        io.write(f"  Growth rate absolute tolerance     = {abstol:.3e}\n")
        io.write(f"  Growth rate relative tolerance     = {reltol:.3e}\n")
        io.write(f"  Factor absolute tolerance          = {factol:.3e}\n")
        io.write(f"  Eigenmode order                    = {args.orderby}\n")
        io.write(f"  Converge the number of modes       = {args.mode}\n")
        io.write(f"  Number of inner collocation points = {args.n_inner}\n")
        io.write(f"  Amplitude fraction at zmax         = {args.amp_fraction_outer}\n")

    print(f"\nCalculation initiated with {number_of_cores} process{'es' if number_of_cores > 1 else ''} for {number_of_tasks} values.\n")

    counter = multiprocessing.Value('i', 0)

    start_time = time.time()

    with multiprocessing.Pool(number_of_cores) as pool:
        async_results = [ pool.apply_async(task, args=(k, params)) for k in ks ]
        results = [ r.get() for r in async_results ]

        with open(fpath, 'w') as io:
            io.write("#\n# Tearing Instability - Scaling Factors\n#\n")
            io.write(f"# Lundquist number (S)                 = {S:10.3e}\n")
            io.write(f"# Prandtl number (Pr)                  = {Pr:10.3e}\n")
            if args.CGL:
                io.write(f"# Plasma-β (β)                         = {β:10.3e}\n")
                io.write(f"# Plasma-β difference (Δβ)             = {Δβ:+6.3f}\n")
                io.write(f'# Parallel adiabatic index (ɣpar)      = {ɣpar:10.3e}\n')
                io.write(f'# Perpendicular adiabatic index (ɣper) = {ɣper:10.3e}\n')
            io.write(f"# Magnetic transverse field (ξ)        = {ξ:10.3e}\n")
            io.write(f"# Hall current term strength (ϵ)       = {ϵ:10.3e}\n")
            io.write(f"# Current sheet thickness (a)          = {a:10.3e}\n")
            io.write(f"# Current sheet half-width (w)         = {w:10.3e}\n")
            io.write("#\n")
            io.write("#   α                Re(σ)            Im(σ)            tol              C                 N\n")
            io.write("# ---------------- ---------------- ---------------- ---------------- ----------------  --------\n#\n")
            for r in results:
                if all(x is not None for x in r):
                    io.write("  {:15.8e}  {:15.8e}  {:15.8e}  {:15.8e}  {:15.8e}  {:8d}\n".format(*r))
            io.flush()

    print(f"\n\n\n\n\nCalculations done in {time.time() - start_time:.2f} seconds.\n")
