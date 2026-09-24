#!/usr/bin/env python3
#
import os, sys, time
import numpy as np

from collections import deque
from functools import lru_cache

from psecas import ChebyshevRationalGrid
from tearing_eigenmodes import build_params, build_dpath, \
                             print_info, refine_wavenumber_bracket, refine_inner_scale, \
                             eigenmodes, find_peak_location, write_results, DeltaError, estimate_max


class Extrapolator:
    """
    Tracks a history of (x, y) pairs and extrapolates y at a new x
    using polynomial fitting (degree 1=linear, 2=quadratic, 3=cubic).
    Falls back gracefully when insufficient history is available.
    Direction-agnostic: works for both increasing and decreasing x sweeps.
    """
    def __init__(self, maxdeg=2, minpoints=2, maxhistory=6, ymin=None):
        self.maxdeg    = maxdeg
        self.minpoints = minpoints
        self.ymin      = ymin        # optional lower clamp on predicted value
        self.xs        = deque(maxlen=maxhistory)
        self.ys        = deque(maxlen=maxhistory)

    def add(self, x, y):
        self.xs.append(float(x))
        self.ys.append(y)

    def predict(self, x_new):
        n = len(self.xs)
        if n < self.minpoints:
            return None                          # not enough history yet
        deg = min(self.maxdeg, n - 1)           # can't exceed n-1
        xs  = np.array(self.xs)
        ys  = np.array(self.ys)
        # centre & scale for numerical stability; ptp() is direction-agnostic
        x0     = xs.mean()
        xscale = (xs.max() - xs.min()) or 1.0
        coeffs = np.polyfit((xs - x0) / xscale, ys, deg)
        y_pred = float(np.polyval(coeffs, (x_new - x0) / xscale))
        if self.ymin is not None:
            y_pred = max(y_pred, self.ymin)
        return y_pred

    def __len__(self):
        return len(self.xs)

def make_objective(params_base):
    # capture a shallow copy once; treat as immutable thereafter
    params_fixed = dict(params_base)

    @lru_cache(maxsize=1024)
    def objective(αq: float) -> float:
        if αq <= 0:
            return 0.0
        p = dict(params_fixed)
        p["alpha"] = αq
        σ, _, _, _, _, _, _, _, _, status = eigenmodes(p)
        return float(-σ.real) if status else 0.0

    def f(α: float) -> float:
        return objective(round(float(α), 12))

    return f

def task(n, value, αbracket, sigma, δinner, params):
    from scipy.optimize import bracket, minimize_scalar

    status = False

    params_base = dict(params)

    ntasks     = params_base.get('ntasks'  , 1)
    dependence = params_base.get('dependence'  , 'S')
    verbose    = params_base.get('verbose', False)
    force      = params_base.get('force'  , False)
    Nmax       = params_base.get('Nmax'   , 2048)
    rtol       = params_base.get('rtol'   , 1e-5)
    wtol       = params_base.get('wtol'   , 1e-3)
    w          = params_base.get('w'       , 0.0)
    a          = params_base.get('a'       , 1.0)
    n_inner    = params_base.get('n_inner', 3)

    end='\n' if verbose else ''

    UP = '\033[F'
    info = f"  {dependence} = {value:+.3e}: "
    progress_line = ''
    bracket_line = ''
    result_line = ''

    dep_map = {
        'a':  'a',
        'w':  'w',
        'S':  'S',
        'Pr': 'Pr',
        'ξ':  'xi',
        'ϵ':  'Hall',
        'β':  'plasma_beta',
        'Δβ': 'plasma_beta_difference'
    }
    params_base[dep_map[dependence]] = float(value)

    sname = os.path.join(params_base.get('data_path', './'), f'state_{dependence}{value:+.6e}.npz')

    if os.path.exists(sname):
        with np.load(sname) as state:

            αm  = state['wavenumber']
            σm  = state['growth_rate']
            e   = state['tolerance']
            δin = state['inner_scale']
            nin = state['n_inner']
            if 'n_wa' in state:
                nwa = state['n_wa']
            else:
                z = state['grid']
                I = np.where(np.abs(z) <= (w + a))
                nwa = I[0].size
            C   = state['scaling_factor']
            N   = state['resolution']
            if 'niter' in state.keys():
                nit = state['niter']
            else:
                nit = 1
            if 'wavenumber_error' in state.keys():
                Δα = state['wavenumber_error']
            else:
                Δα = wtol * αm
            if 'eigenvalue_error' in state.keys():
                Δσ = state['eigenvalue_error']
            else:
                Δσ = rtol * σm.real * e

            status = not force and not (e > 1.0 and N < Nmax and nin < n_inner)

    if not status:
        status = True
        if αbracket is None or len(αbracket) != 2:
            try:
                αm, _ = estimate_max(params_base)
                αlo = αm * (1 - wtol)
                αup = αm * (1 + wtol)
            except DeltaError as ex:
                status = False
                if verbose:
                    print(f"Stable eigenmode: {ex}")
        else:
            αlo, αup = αbracket
        if status:
            if verbose:
                print(f"Initial bracket for {dependence}={value:+.3e}: α = {αlo:.4e} … {αup:.4e}")
            αu = (αup + 1.618 * αlo) / 2.618

            try:
                params_base['sigma']   = sigma
                params_base['delta']   = δinner

                f = make_objective(params_base)

                αa, αb, αc, σa, σb, σc, fn = bracket(f, xa=αlo, xb=αu)
                if αa > αc:
                    αa, αc, σa, σc = αc, αa, σc, σa
                if αa <= 0:
                    αa = 1e-3
                bracket_line = info + f"α-bracket = [ {αa:.3e}, {αb:.3e}, {αc:.3e} ],  σ-values = [ {-σa:.3e}, {-σb:.3e}, {-    σc:.3e} ]  after {fn} function calls" + ' '*4
                if verbose:
                    print(f"{bracket_line}", flush=True)
                else:
                    print(f"\r{bracket_line}\n{result_line}\n\n{progress_line}{UP}{UP}{UP}", end='', flush=True)
                bracket_line = ''

                Δα = wtol * αa
                res = minimize_scalar(f, bracket=(αa, αb, αc), method='brent', options={'xtol': Δα})

                if verbose:
                    print(res)
                    print('Final refinement:')
                αm  = res.x
                params_final = dict(params_base)
                params_final["alpha"] = αm
                σ, s, e, δin, nin, nwa, C, N, z, status = eigenmodes(params_final)
                if status:
                    σm  = σ
                    Δσ  = rtol * σm.real * e
                    nit = res.nfev

                    np.savez_compressed(sname, value=value, wavenumber=αm, wavenumber_error=Δα, \
                                    growth_rate=σm, growth_rate_error=Δσ, tolerance=e, \
                                    inner_scale=δin, scaling_factor=C, n_inner=nin, n_wa=nwa, \
                                    niter=nit, resolution=N, grid=z, **s)

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
                    print(f"\nCould not find brackets for {dependence} = {value:+.3e}: {ex}")


    progress   = n / ntasks
    percentage = int(progress * 100)

    fmt = r'[{:0' + str(len(str(ntasks))) + 'd}/' + str(ntasks) + ']'
    progress_line = f"Progress {percentage}% complete {'█' * (percentage // 2)}{' ' * (50 - (percentage // 2))} {fmt.format(n)}"

    if status:
        result_line = info + f"α={αm:.4e}±{Δα:.1e}  σ={σm.real:.4e}±{Δσ:.1e}  δin={δin:.3e}  nin={nin}  nwa={nwa}  C={C:.3e}  N={N} after {nit} function calls" + ' '*6
        if verbose:
            print(f"{result_line}", flush=True)
        else:
            print(f"\r{bracket_line}\n{result_line}\n\n{progress_line}{UP}{UP}{UP}", end='', flush=True)

        return αm, σm, δin, N, status
    else:
        bracket_line = info + "could not find any bracket!" + ' '*80
        result_line  = info + "could not find any maximum!" + ' '*80
        if verbose:
            print(f"{result_line}", flush=True)
        else:
            print(f"\r{bracket_line}\n{result_line}\n\n{progress_line}{UP}{UP}{UP}", end='', flush=True)

        return None, None, None, None, status


def main():
    '''
    Given provided options, calculates eigenmodes of the equilibrium field
    with magnetic and velocity shear.
    '''
    params = build_params(parser_type='maximum')
    dpath  = build_dpath(params)

    vl, vu, dv = params['vmin'], params['vmax'], params['vinc']
    vs = np.linspace(vl, vu, int(np.ceil((vu - vl + 0.5 * dv) / dv)))
    if params['logarithmic']:
        vs = 10**vs

    ntasks = vs.size
    nprocs = 1

    params['data_path'] = dpath
    params['ntasks']    = ntasks

    if not os.path.exists(dpath):
        os.makedirs(dpath)

    print("\nCalculation of the maximum growth rate dependence on several parameters for the selected eigenmode.")
    print("Use option '-h' to show all possible arguments.\n")

    print_info(params)

    if ntasks > 1 and os.getenv("OMP_NUM_THREADS") != "1":
        print("\n\033[1mPlease set OMP_NUM_THREADS=1 to ensure optimal performance!\033[0m")
        nprocs = 1

    plural = 'es' if nprocs > 1 else ''
    print(f'\nCalculation initiated with {nprocs} process{plural}'
          f' for {ntasks} values.\n')

    delta_time = -time.time()

    kbracket     = params.get('kbracket', None)
    dm           = params.get('delta', None)
    gm           = params.get('sigma', None)
    extrap_deg   = params.get('extrap_deg',   2)     # 1=linear, 2=quadratic, 3=cubic
    extrap_guard = params.get('extrap_guard', 0.01)  # max fractional deviation allowed

    k_extrap = Extrapolator(maxdeg=extrap_deg, ymin=1e-6)
    d_extrap = Extrapolator(maxdeg=extrap_deg, ymin=1e-10)
    g_extrap = Extrapolator(maxdeg=extrap_deg, ymin=1e-10)

    if kbracket is None:
        kl, ku = 0.1, 0.12
    else:
        kl, ku = kbracket[0], kbracket[1]

    n = 0
    while n < len(vs):
        value = vs[n]

        n += 1

        # ── extrapolate wavenumber bracket ────────────────────────────────────
        k_pred = k_extrap.predict(value)
        if k_pred is not None:
            guard    = extrap_guard * k_pred
            kl       = max(k_pred - guard, 1e-6)   # clamp: k must be positive
            ku       = k_pred + guard
            kbracket = [kl, ku]
            if params.get('verbose'):
                print(f"  k extrapolated: {k_pred:.4e}  →  bracket [{kl:.4e}, {ku:.4e}]")

        # # ── extrapolate inner-layer thickness ─────────────────────────────────
        # d_pred = d_extrap.predict(value)
        # if d_pred is not None:
            # dm = d_pred                             # ymin clamp applied inside predict()
            # if params.get('verbose'):
                # print(f"  δ extrapolated: {dm:.4e}")

        # ── extrapolate growth rate ─────────────────────────────────
        # g_pred = g_extrap.predict(value)
        # if g_pred is not None:
            # gm = g_pred                             # ymin clamp applied inside predict()
            # if params.get('verbose'):
                # print(f"  g extrapolated: {gm:.4e}")

        km, gm, dm, N, status = task(n, value, kbracket, gm, dm, params)

        if not status:
            break
        if gm.real < 1e-6:
            break

        # ── record for next extrapolation ─────────────────────────────────────
        k_extrap.add(value, km)
        if dm is not None:
            d_extrap.add(value, dm)
        if gm is not None:
            g_extrap.add(value, gm)

    delta_time += time.time()

    write_results(params, delta_time)

    if not params['verbose']:
        sys.stdout.write("\n\n\n\n")
    sys.stdout.write(f"\nCalculation done in {delta_time:.2f} seconds.\n")


if __name__ == "__main__":
    main()
