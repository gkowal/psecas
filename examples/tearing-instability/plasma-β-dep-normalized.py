#!/usr/bin/env python3
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

if __name__ == "__main__":
    '''
        Plot dependence on plasma-β.
    '''
    import argparse
    import matplotlib.pyplot as plt
    import os, sys

    from scipy.optimize import curve_fit, fsolve

    parser = argparse.ArgumentParser(description="Plots dependence on plasma-Δβ.")
    parser.add_argument(
        "--thickness", "-a", type=float, default=1,
        help="the thickness of the current sheet"
    )
    parser.add_argument(
        "--width", "-w", type=float, default=0,
        help="the width of the current sheet"
    )
    parser.add_argument(
        "--Lundquist-number", "-S", type=float, default=1e6,
        help="the Lundquist number"
    )
    parser.add_argument(
        "--Prandtl-number", "-Pr", type=float, default=0,
        help="the Prandtl number"
    )
    parser.add_argument(
        "--plasma-beta-difference", "-Δβ", type=float, default=0,
        help="the plasma-Δβ"
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
    S  = args.Lundquist_number
    Pr = args.Prandtl_number
    Δβ = args.plasma_beta_difference
    ξ  = args.magnetic_transverse_field
    ϵ  = args.hall
    ɣpar = { 'adiabatic': 3.0, 'polytropic': 0.5, 'isothermal': 1.0 }
    ɣper = { 'adiabatic': 2.0, 'polytropic': 2.0, 'isothermal': 1.0 }

    EOSs = [ 'adiabatic', 'polytropic', 'isothermal' ]

    Ss = [ 1e4, 1e5, 1e6 ]

    markers = { 1e4: '*', 1e5: 'x', 1e6: '.', 1e7: '+' }
    sizes   = { 1e4:   1, 1e5:   1, 1e6:   2, 1e7:   2 }

    Ls = { 1e4: 'solid', 1e5: 'dashed', 1e6: 'dashdot' }

    fig, axs = plt.subplots(ncols=3, figsize=(15,2.5), dpi=300)

    αMHD = { 0: { 1e4: 1.279302e-01, 1e5: 7.488290e-02, 1e6: 4.269671e-02 }, 1: { 1e4: 1.010932e-01, 1e5: 5.844173e-02, 1e6: 3.316118e-02 } }
    σMHD = { 0: { 1e4: 6.040923e-03, 1e5: 1.951885e-03, 1e6: 6.214884e-04 }, 1: { 1e4: 4.810950e-03, 1e5: 1.560925e-03, 1e6: 4.975909e-04 } }
    δMHD = { 0: { 1e4: 7.461615e-02, 1e5: 4.200549e-02, 1e6: 2.362308e-02 }, 1: { 1e4: 1.034597e-01, 1e5: 5.880539e-02, 1e6: 3.314098e-02 } }

    for i, eos in enumerate(EOSs):
        dpath = f"./double-{eos}/RESULTS/S{S:.3e}Pr{Pr:.3e}Δβ{Δβ:+.2f}ξ{ξ:.3e}ϵ{ϵ:.3e}a{a:.3e}w{w:.3e}"
        if os.path.exists(dpath):
            β, α, σ, _, δ, _, _, _, _ = load_eigenmodes(dpath)

            I = np.where(σ.real > 0.0)

            sc = axs[0].plot(β[I], σ[I].real / σMHD[Pr][S], label=fr'$\gamma_\parallel = {ɣpar[eos]:.1f}, \gamma_\perp = {ɣper[eos]:.1f}$')
            sc = axs[1].plot(β[I], α[I]      / αMHD[Pr][S], label=fr'$\gamma_\parallel = {ɣpar[eos]:.1f}, \gamma_\perp = {ɣper[eos]:.1f}$')
            sc = axs[2].plot(β[I], δ[I]      / δMHD[Pr][S], label=fr'$\gamma_\parallel = {ɣpar[eos]:.1f}, \gamma_\perp = {ɣper[eos]:.1f}$')

    axs[0].set_title(r'Maximum growth rate $\gamma_\max \equiv {\rm Re}(\sigma)$', fontsize=9)
    axs[0].set_ylabel(r"$\gamma_\max / \gamma_\max^{\rm MHD}$", fontsize=9)
    axs[0].set_ylim(0.0, 1.05)

    axs[1].set_title(r'Maximum wavenumber $k_\max$', fontsize=9)
    axs[1].set_ylabel(r"$k_\max / k_\max^{\rm MHD}$", fontsize=9)
    axs[1].set_ylim(0.0, 1.05)

    axs[2].set_title(r'Inner-layer thickness $\delta_{\rm in}$', fontsize=9)
    axs[2].set_ylabel(r"$\delta_{\rm in} / \delta_{\rm in}^{\rm MHD}$", fontsize=9)

    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.set_xlabel(r"$\beta_0$", fontsize=9)
        ax.set_xlim(1e-2, 1e+2)
        ax.set_xscale('log')
        ax.legend(fontsize=8)

    fname = f"S{S:.1e}Pr{Pr:.1e}-Δβ{Δβ:+.2f}-max_growth-β_dep_norm."
    plt.savefig(fname + args.output_format, bbox_inches='tight')
