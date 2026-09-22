#!/usr/bin/env python3
#
import os, sys, time
import multiprocessing as mp
import numpy as np

from functools import lru_cache
from psecas import ChebyshevRationalGrid
from tearing_eigenmodes import build_params, build_dpath, \
                             print_info, refine_wavenumber_bracket, \
                             refine_thickness, refine_inner_scale, refine_growth_rate, \
                             eigenmodes, find_peak_location, write_results, DeltaError, estimate_max

counter = None

def init_worker(shared_counter):
    """Assign the shared object to the global variable in this worker."""
    global counter
    counter = shared_counter

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

def task(value, αbracket, sigma, δinner, params):
    from scipy.optimize import bracket, minimize_scalar

    global counter

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
            N   = state['resolution']
            C   = state['scaling_factor']
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

            status = not force and not (e > 1.0 and N < Nmax)

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


    n = 0
    with counter.get_lock():
        counter.value += 1
        n = counter.value

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

    else:
        bracket_line = info + "could not find any bracket!" + ' '*80
        result_line  = info + "could not find any maximum!" + ' '*80
        if verbose:
            print(f"{result_line}", flush=True)
        else:
            print(f"\r{bracket_line}\n{result_line}\n\n{progress_line}{UP}{UP}{UP}", end='', flush=True)

def main():
    '''
        Given provided options, calculates eigenmodes of the equilibrium field with magnetic and velocity shear.
    '''
    params = build_params(parser_type='maximum')

    dpath = build_dpath(params)

    vl, vu, dv = params['vmin'], params['vmax'], params['vinc']
    vs = np.linspace(vl, vu, int(np.ceil((vu - vl + 0.5 * dv) / dv)))
    if params['logarithmic']:
        vs = 10**vs

    ntasks = vs.size
    nprocs = int(os.getenv('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    nprocs = min(ntasks, nprocs)

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

    k = refine_wavenumber_bracket(vs, params)
    d = refine_thickness(vs, params)
    g = refine_growth_rate(vs, params)

    plural = 'es' if nprocs > 1 else ''
    print(f'\nCalculation initiated with {nprocs} process{plural}'
          f' for {ntasks} values.\n')

    delta_time = -time.time()

    shared_counter = mp.Value('i', 0)

    with mp.Pool(
        processes=nprocs,
        initializer=init_worker,
        initargs=(shared_counter,)
    ) as pool:
        pool.starmap(task, [(v, k[n], g[n], d[n], params) for n, v in enumerate(vs)])

    delta_time += time.time()

    write_results(params, delta_time)

    if not params['verbose']:
        sys.stdout.write("\n\n\n\n")
    sys.stdout.write(f"\nCalculation done in {delta_time:.2f} seconds.\n")


if __name__ == "__main__":
    main()
