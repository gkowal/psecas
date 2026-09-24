#!/usr/bin/env python
#
import numpy as np

from tearing_eigenmodes import load_eigenmodes

def sciformat(x):
    if x == 1:
        return '1'
    elif x > 0:
        return '10^{' + '{:d}'.format(int(np.log10(x))) + '}'
    else:
        return '0'

def find_non_monotonic(arr):
    """
    Returns the boolean mask and the indices of points
    that do not lie between their neighbors.
    """
    if len(arr) < 3:
        return np.array([], dtype=bool), np.array([], dtype=int)

    diffs = np.diff(arr)

    # Identify where the direction of the slope changes
    # mask[1:-1] corresponds to the internal elements of the original array
    is_non_monotonic = np.zeros_like(arr, dtype=bool)
    is_non_monotonic[1:-1] = diffs[:-1] * diffs[1:] <= 0

    # Extract indices where the mask is True
    indices = np.where(is_non_monotonic)[0]

    return is_non_monotonic, indices

def Δp_func(α, β=0, Δβ=0, ɣpar=3, ɣper=2):
    fe = 2 + (ɣpar + ɣper - 2) * β + ɣpar * Δβ
    fd = 2 - Δβ
    if fe < 0.0:
        fe = np.nan
    if fd < 0.0:
        fd = np.nan
    fc = α * np.sqrt(fe / fd)
    return 2 * (1/fc - fc)

def ɣFKR_func(α, S=1e4, β=0, Δβ=0, ɣpar=3, ɣper=2):
    C  = 1 - Δβ/2
    Δp = Δp_func(α, β=β, Δβ=Δβ, ɣpar=ɣpar, ɣper=ɣper)
    return (C * S**-3 * α**2 * Δp**4)**(1/5)

def ɣCop_func(α, S=1e4, Δβ=0):
    C  = 1 - Δβ/2
    return (C * S**-1 * α**2)**(1/3)

def f(α, S=1e4, β=0, Δβ=0, ɣpar=3, ɣper=2):
    ɣFKR = ɣFKR_func(α, S=S, β=β, Δβ=Δβ, ɣpar=ɣpar, ɣper=ɣper)
    ɣCop = ɣCop_func(α, S=S, Δβ=Δβ)
    return ɣFKR - ɣCop

def λfunc(Δβ, a, β, c, ɣpar=3, ɣper=2):
    fe = 2 - Δβ
    fd = 2 + (ɣpar + ɣper - 2) * β + ɣpar * Δβ
    fd[np.where(fd < 0.0)] = np.nan
    return a * (fe / fd)**c

def αmax(S=1e4, β=0, Δβ=0, ɣpar=3, ɣper=2):
    C  = 1 - Δβ/2
    Q  = 1 + (ɣpar + ɣper - 2) * β / 2 + ɣpar * Δβ / 2
    return 1.358 * S**-0.25 * C**0.25 * Q**-0.375 * 0.85

def σmax(S=1e4, β=0, Δβ=0, ɣpar=3, ɣper=2):
    C  = 1 - Δβ/2
    Q  = 1 + (ɣpar + ɣper - 2) * β / 2 + ɣpar * Δβ / 2
    return 0.623 * S**-0.5 * C**0.5 * Q**-0.25


if __name__ == "__main__":
    '''
        Plot dependence on plasma-Δβ.
    '''
    import argparse
    import matplotlib.pyplot as plt
    import os, sys

    from scipy.optimize import curve_fit, fsolve

    parser = argparse.ArgumentParser(description="Plots dependence on Δβ.")
    parser.add_argument(
        "--thickness", "-a", type=float, default=1,
        help="the thickness of the current sheet"
    )
    parser.add_argument(
        "--width", "-w", type=float, default=0,
        help="the width of the current sheet"
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
        "--magnetic-transverse-field", "-ξ", type=float, default=0,
        help="the transverse magnetic strength"
    )
    parser.add_argument(
        "--hall", "-ϵ", type=float, default=0,
        help="the Hall current term strength"
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
        '--output-format',
        default='pdf',
        choices=['pdf', 'png'],
        help="the output format of the plot"
    )
    parser.add_argument(
        "--fit", "-f", action='store_true', default=False,
        help="perform fitting"
    )

    args = parser.parse_args()

    a  = args.thickness
    w  = args.width
    Pr = args.Prandtl_number
    β  = args.plasma_beta
    ξ  = args.magnetic_transverse_field
    ϵ  = args.hall
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

    Ss = [ 1e4, 1e5, 1e6 ]

    markers = { 1e4: '*', 1e5: 'x', 1e6: '.', 1e7: '+' }
    sizes   = { 1e4:   1, 1e5:   1, 1e6:   2, 1e7:   2 }

    Ls = { 1e4: 'solid', 1e5: 'dashed', 1e6: 'dashdot' }

    fig, axs = plt.subplots(ncols=3, figsize=(15,2.5), dpi=300)

    αMHD = { 0: [ 1.279302e-01, 7.488290e-02, 4.269671e-02 ], 1: [ 1.010932e-01, 5.844173e-02, 3.316118e-02 ] }
    σMHD = { 0: [ 6.040923e-03, 1.951885e-03, 6.214884e-04 ], 1: [ 4.810950e-03, 1.560925e-03, 4.975909e-04 ] }
    δMHD = { 0: [ 7.461615e-02, 4.200549e-02, 2.362308e-02 ], 1: [ 1.034597e-01, 5.880539e-02, 3.314098e-02 ] }

    if β < 2:
        axs[0].axvspan(-2.0, -β, color='grey', alpha=0.25, lw=0)
        axs[1].axvspan(-2.0, -β, color='grey', alpha=0.25, lw=0)
        axs[2].axvspan(-2.0, -β, color='grey', alpha=0.25, lw=0)
        axs[0].text(-1 - β / 2, 0.001, 'unphysical', fontsize=7, ha='center')
        axs[1].text(-1 - β / 2, 0.030, 'unphysical', fontsize=7, ha='center')
        axs[2].text(-1 - β / 2, 0.030, 'unphysical', fontsize=7, ha='center')

    for i, S in enumerate(Ss):
        c = 'C{}'.format(i)

        dpath = f"./RESULTS/S{S:.3e}Pr{Pr:.3e}β{β:.3e}ξ{ξ:.3e}ϵ{ϵ:.3e}a{a:.3e}w{w:.3e}"
        if os.path.exists(dpath):
            Δβ, α, σ, _, δ, _, _, _, _ = load_eigenmodes(dpath)

            mask, idx = find_non_monotonic(α)
            if np.any(mask):
                print(f"S={S:.1e} : ", Δβ[idx])

            I = np.where(σ.real > 0.0)

            sc = axs[0].plot(Δβ[I], σ[I].real, c=c, label=r'$S_a = {}$'.format(sciformat(S)), lw=0.8)
            sc = axs[1].plot(Δβ[I], α[I]     , c=c, label=r'$S_a = {}$'.format(sciformat(S)), lw=0.8)
            sc = axs[2].plot(Δβ[I], δ[I]     , c=c, label=r'$S_a = {}$'.format(sciformat(S)), lw=0.8)

            #x = np.linspace(-1.0, 1.0, 101)
            #bt = 0.5 * ((ɣpar + ɣper - 2) * β + (ɣpar - 1) * x)
            #A  = 1 - 0.5 * x
            #R0 = 1 + bt + 0.5 * x
            #mu = bt + x
            #CD = 4 * np.sqrt(A) * (2 * A - mu) / (np.sqrt(R0) * (4 * A + mu))
            #y = 0.5*CD**0.5 * A**0.25 * S**-0.5
            #I = np.where(y > 0.0)
            #axs[0].plot(x[I], y[I], c='grey', lw=0.6)
            #y = 0.5*CD**0.75 * A**-0.125 * S**-0.25
            #I = np.where(y > 0.0)
            #axs[1].plot(x[I], y[I], c='grey', lw=0.6)
            #y = 0.5*CD**-0.25 * A**-0.125 * S**-0.25
            #I = np.where(y > 0.0)
            #axs[2].plot(x[I], y[I], c='grey', lw=0.6)

            axs[0].plot([-2,2], [σMHD[Pr][i]]*2, '--', c='grey', lw=0.5)
            axs[1].plot([-2,2], [αMHD[Pr][i]]*2, '--', c='grey', lw=0.5)
            axs[2].plot([-2,2], [δMHD[Pr][i]]*2, '--', c='grey', lw=0.5)
            axs[0].scatter(0, σMHD[Pr][i], c=c, s=1)
            axs[1].scatter(0, αMHD[Pr][i], c=c, s=1)
            axs[2].scatter(0, δMHD[Pr][i], c=c, s=1)

    axs[0].set_title(r'Maximum growth rate ($Pr_m={}, \beta_0={}$)'.format(sciformat(Pr), β), fontsize=9)
    axs[0].set_ylabel(r"$\sigma_\mathrm{max} \tau_a$", fontsize=9)
    axs[0].set_ylim(1e-4, 6e-2)

    axs[1].set_title(r'Maximum wavenumber ($Pr_m={}, \beta_0={}$)'.format(sciformat(Pr), β), fontsize=9)
    axs[1].set_ylabel(r"$k_\mathrm{max} a$", fontsize=9)
    axs[1].set_ylim(1e-2, 1e+0)

    axs[2].set_title(r'Inner-layer thickness ($Pr_m={}, \beta_0={}$)'.format(sciformat(Pr), β), fontsize=9)
    axs[2].set_ylabel(r"$\delta_\mathrm{in} / a$", fontsize=9)
    axs[2].set_ylim(4e-3, 4e-1)

    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.set_xlabel(r"$\Delta \beta_0$", fontsize=9)
        ax.set_xlim(-2, 2)
        ax.legend(fontsize=8)
        ax.set_yscale('log')

    fname = f"Pr{Pr:.1e}-β{β:.1e}-max_growth-Δβ_dep."
    plt.savefig(fname + args.output_format, bbox_inches='tight')
