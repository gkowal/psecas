#!/usr/bin/env python3
#
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.optimize import curve_fit, fsolve
from tearing_eigenmodes import load_eigenmodes

def g(x, S):
    return 0.4 * (x**2 / S)**(1.0/3.0)

def f(x, S, β, Δβ):
        λ  = x * np.sqrt((1 - Δβ/2) / (1 + β + Δβ/2))
        a2 = -0.5 * (β + Δβ) * λ**2
        a1 = ((1 - Δβ/2) / (1 + β + Δβ/2) + a2) / λ - 0.5 * λ * (β + Δβ) / (1 - Δβ/2)
        Δ  = 2 * (a1 - λ)
        return 0.2 * x**0.4 * Δ**0.8 * S**-0.6 - g(x,S)

def fit(x, a, b, c):
    return a*(1 + (x/b))**c

def sciformat(x):
    if x == 1:
        return '1'
    elif x > 0:
        return '10^{' + '{:d}'.format(int(np.log10(x))) + '}'
    else:
        return '0'

a  = 1
w  = 0
Pr = 0
ξ  = 0
Δβ = 0
ϵ  = 0

Ss = [ 1e4, 1e5, 1e6 ]
ξs = [ 0 ]

markers = { 1e4: '*', 1e5: 'x', 1e6: '.', 1e7: '+' }
sizes   = { 1e4:   1, 1e5:   1, 1e6:   2, 1e7:   2 }

Ls = { 1e4: 'solid', 1e5: 'dashed', 1e6: 'dashdot' }

fig, axs = plt.subplots(ncols=2, figsize=(10,2.5), dpi=300)

dname = f"./RESULTS/a{a:.3e}w{w:.3e}Pr{Pr:.3e}ξ{ξ:.3e}ϵ{ϵ:.3e}-MHD.dat"
if os.path.exists(dname):
    D = np.loadtxt(dname)
    S = D[:,0]
    α = D[:,1]
    σ = D[:,2]

    for i in range(S.size):
        c = f"C{i}"
        if S[i] in Ss:
            axs[0].axhline(σ[i], linewidth=0.5, c=c, ls='dashed')
            axs[1].axhline(α[i], linewidth=0.5, c=c, ls='dashed')

for i, S in enumerate(Ss):
    c = f"C{i}"

    dpath = f"./RESULTS/S{S:.3e}Pr{Pr:.3e}β{β:.3e}ξ{ξ:.3e}ϵ{ϵ:.3e}a{a:.3e}w{w:.3e}-cache"
    if os.path.exists(dpath):
        S, α, σ, _, δ, _, _, _ = load_eigenmodes(dpath)
    # dname = f"./RESULTS/CGL-MHD/a{a:.3e}w{w:.3e}S{S:.3e}Pr{Pr:.3e}Δβ{Δβ:+.2f}ξ{ξ:.3e}ϵ{ϵ:.3e}.dat"
    # if os.path.exists(dname):
        # D = np.loadtxt(dname)
        # β = D[:,0]
        # α = D[:,1]
        # σ = D[:,2]

#        σopt, σcov = curve_fit(fit, β, σ, p0=[0.001,0.5,-0.25])
#        αopt, αcov = curve_fit(fit, β, α, p0=[1.000,0.5,-0.38])
#        print(f'S={S:.3e} -> ', '{:.3e} {:.3e} {:.3e} '.format(σopt[0]*S**0.5, σopt[1], σopt[2]), '{:.3e} {:.3e} {:.3e} '.format(αopt[0]*S**0.25, αopt[1], αopt[2]))

        axs[0].plot(β, σ, linewidth=0.75, c=c, label=r'$S_{\rm a} = ' + f'{sciformat(S)}' + '$')
        axs[1].plot(β, α, linewidth=0.75, c=c, label=r'$S_{\rm a} = ' + f'{sciformat(S)}' + '$')

#        axs[0].plot(β,fit(β,*σopt), lw=0.5, c='grey')
#        axs[0].plot(β,fit(β,0.50*S**-0.50,0.25*S**(1/15),-0.25), lw=0.5, c='grey')
#        axs[1].plot(β,g(β,1.06*S**-0.25,0.25*S**(1/15),0.385), lw=0.5, c='grey')

#        x = []
#        y = []
#        z = []
#        for i, b in enumerate(β):
#            r = fsolve(f, α[i], args=(S, 1.5*b, 0.0))
#            print(b, α[i], r[0], g(r[0], S))
#            x.append(b)
#            y.append(r[0])
#            z.append(g(r[0], S))
#        axs[0].plot(x, z, lw=0.5, c='grey')
#        axs[1].plot(x, y, lw=0.5, c='grey')


axs[0].set_title(r"Maximum growth rate ($Pr_{\rm m}=" + f"{sciformat(Pr)}" + r"$)", fontsize=9)
axs[0].set_ylabel(r"$\sigma_\mathrm{max} \tau_{\rm a}$", fontsize=9)
axs[0].set_ylim(1e-4, 1e-2)
#axs[0].set_ylim(0, 6e-3)

#x = 10**np.arange(-2,2.1,0.1)
#axs[0].plot(x, 0.45*1e4**-0.5/(1+(x/2)**0.3), lw=0.5, c='black')

x = 10**np.arange(1,2,0.1)
axs[0].plot(x, 1.2e-3*x**-0.25, lw=0.5, c='black')
axs[0].text(14,3.5e-4,r'$\sim \beta_0^{-1/4}$',fontsize=8)

axs[1].plot(x, 4.2e-2*x**-0.375, lw=0.5, c='black')
axs[1].text(14,8.5e-3,r'$\sim \beta_0^{-3/8}$',fontsize=8)

axs[1].set_title(r"Maximum wavenumber ($Pr_{\rm m}=" + f"{sciformat(Pr)}" + r"$)", fontsize=9)
axs[1].set_ylabel(r"$k_\mathrm{max} a$", fontsize=9)
axs[1].set_ylim(6e-3, 2e-1)
axs[1].set_xlabel(r"$\beta_0$", fontsize=9)

for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.set_xlabel(r"$\beta_0$", fontsize=9)
    ax.set_xlim(1e-2, 1e2)
    ax.set_xscale('log')
    ax.legend(fontsize=8)

axs[0].set_yscale('log')
axs[1].set_yscale('log')

fname = 'Pr{:.1e}-max_growth-β_dep'.format(Pr)
plt.savefig(fname + '.pdf', bbox_inches='tight')
plt.savefig(fname + '.png', bbox_inches='tight')
