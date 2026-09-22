#!/usr/bin/env python3
#
from functools import lru_cache
from scipy.interpolate import make_interp_spline
from tearing_eigenmodes import DeltaError, eigenmodes, estimate_max, load_eigenmodes
import numpy as np
import os, sys

# This global 'counter' only exists inside the worker process's memory
# It will hold a reference to the shared Value object
counter = None
delta_hat = None
delta_points = []


def init_worker(shared_counter):
    """Assign the shared object to the global variable in this worker."""
    global counter
    counter = shared_counter


def _freeze_params(params):
    items = []
    for k, v in params.items():
        if k in ("α", "δ"):      # exclude varying keys from signature
            continue
        if isinstance(v, (list, tuple)):
            v = tuple(v)
        elif isinstance(v, dict):
            v = tuple(sorted(v.items()))
        items.append((k, v))
    return tuple(sorted(items))


@lru_cache(maxsize=1024)
def _f_cached_core(αq, δq, frozen_items):
    params_local = dict(frozen_items)
    params_local["α"] = αq
    params_local["δ"] = δq

    σ, _, _, δ_est, _, _, _, _, status = eigenmodes(params_local)

    if status:
        return float(-σ.real), float(δ_est), True
    else:
        return 0.0, None, False


def f_cached(α, δ_guess, params):
    αq = round(float(α), 12)
    if αq <= 0:
        return 0.0, None, False
    δq = round(float(δ_guess), 12)
    frozen = _freeze_params(params)
    return _f_cached_core(αq, δq, frozen)


def f_scalar_bracket(α, params):
    αq = round(float(α), 12)

    # Build params for this call
    params_local = dict(params)
    params_local["α"] = αq

    σ, _, _, δ_est, _, _, _, _, status = eigenmodes(params_local)

    if status:
        delta_points.append((αq, δ_est))
        return -float(σ.real)
    else:
        return 0.0


def f_scalar_min(α, params):
    if delta_hat is None:
        δ_guess = params["δ"]
    else:
        δ_guess = delta_hat(α)

    val, _, _ = f_cached(α, δ_guess, params)
    return val


def build_delta_interpolant(delta_points, bracket_triple, clamp=None):
    """
    Return delta_hat(alpha) built from up to 3 points spanning the bracket.
    Uses linear if only 2 points, constant if 1 point.
    """
    αa, αb, αc = bracket_triple
    lo, hi = min(αa, αc), max(αa, αc)

    # Filter to within bracket, unique by α (keep last)
    uniq = {}
    for a, d in delta_points:
        if lo <= a <= hi:
            uniq[a] = d
    pts = sorted(uniq.items())
    if len(pts) == 0:
        # fall back: no points in bracket; caller should handle
        return None, []

    # Choose up to 3 representative points: left, mid, right
    if len(pts) > 3:
        left = pts[0]
        right = pts[-1]
        amid = 0.5 * (left[0] + right[0])
        mid = min(pts[1:-1], key=lambda p: abs(p[0] - amid))
        pts3 = [left, mid, right]
    else:
        pts3 = pts

    xs = np.array([p[0] for p in pts3], float)
    ys = np.array([p[1] for p in pts3], float)

    # Build spline/interpolant
    if len(xs) == 1:
        def delta_hat(a):
            d = float(ys[0])
            if clamp is not None:
                d = max(clamp[0], min(clamp[1], d))
            return d

    elif len(xs) == 2:
        def delta_hat(a):
            d = float(np.interp(float(a), xs, ys))
            if clamp is not None:
                d = max(clamp[0], min(clamp[1], d))
            return d

    else:
        # quadratic spline through 3 points
        spline = make_interp_spline(xs, ys, k=2)
        def delta_hat(a):
            d = float(spline(float(a)))
            if clamp is not None:
                d = max(clamp[0], min(clamp[1], d))
            return d

    return delta_hat, pts3


def task(dpath, dependence, value, ntask, ntasks, αbracket, δinner, params):

    from scipy.optimize import bracket, minimize_scalar

    global counter
    global delta_points, delta_hat
    
    _f_cached_core.cache_clear()

    status = False

    params_base = dict(params)

    verbose = params_base.get('verbose', False)
    force   = params_base.get('force'  , False)
    Nmax    = params_base.get('Nmax'   , 2048)
    rtol    = params_base.get('rtol'   , 1e-5)
    wtol    = params_base.get('wtol'   , 1e-3)

    end='\n' if verbose else ''

    UP = '\033[F'
    info = f"  {dependence} = {value:+.3e}: "
    progress_line = ''
    bracket_line = ''
    result_line = ''

    params_base[dependence] = float(value)

    sname = os.path.join(params.get('data_path', './'), f'state_{dependence}{value:+.6e}.npz')

    if os.path.exists(sname):
        state = np.load(sname)

        αm  = state['wavenumber']
        σm  = state['growth_rate']
        e   = state['tolerance']
        δin = state['inner_scale']
        nin = state['n_inner']
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
                delta_points = []
                delta_hat = None
                params_base['δ'] = δinner

                #αa, αb, αc, σa, σb, σc, fn = bracket(
                #    f_scalar_bracket, xa=αlo, xb=αu, args=(params_base,)
                #)
                αa, αb, αc, σa, σb, σc, fn = bracket(
                    f_scalar_bracket,
                    xa=αlo,
                    xb=αu,
                    args=(params_base,)
                )
                if verbose:
                    print("Collected delta points:", delta_points)
                #αa, αb, αc, σa, σb, σc, fn = bracket(f_scalar, xa=αlo, xb=αu, args=(params_base,))
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

                delta_hat, pts3 = build_delta_interpolant(delta_points, (αa, αb, αc))

                if verbose and pts3:
                    print("δ interpolant points:", pts3)

                Δα = wtol * αa
                #res = minimize_scalar(f_scalar, args=(params_base,), bracket=(αa, αb, αc), method='brent', options={'xtol': Δα})
                res = minimize_scalar(
                    f_scalar_min, args=(params_base,),
                    bracket=(αa, αb, αc),
                    method='brent',
                    options={'xtol': Δα}
                )

                if verbose:
                    print(res)
                    print('Final refinement:')
                αm  = res.x
                params_final = dict(params_base)
                params_final["α"] = αm
                σ, s, e, δin, nin, C, N, z, status = eigenmodes(params_final)
                if status:
                    σm  = σ
                    Δσ  = rtol * σm.real * e
                    nit = res.nfev

                    np.savez_compressed(sname, value=value, wavenumber=αm, wavenumber_error=Δα, \
                                    growth_rate=σm, growth_rate_error=Δσ, tolerance=e, \
                                    inner_scale=δin, scaling_factor=C, n_inner=nin, niter=nit, \
                                    resolution=N, grid=z, **s)

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
        result_line = info + f"α={αm:.4e}±{Δα:.1e}  σ={σm.real:.4e}±{Δσ:.1e}  δin={δin:.3e}  C={C:.3e}  n={nin}  N={N} after {nit} function calls" + ' '*6
        if verbose:
            print(f"{result_line}", flush=True)
        else:
            print(f"\r{bracket_line}\n{result_line}\n\n{progress_line}{UP}{UP}{UP}", end='', flush=True)

        return value, αm, σm, Δα, Δσ, δin, C, nin, N
    else:
        bracket_line = info + "could not find any bracket!" + ' '*80
        result_line  = info + "could not find any maximum!" + ' '*80
        if verbose:
            print(f"{result_line}", flush=True)
        else:
            print(f"\r{bracket_line}\n{result_line}\n\n{progress_line}{UP}{UP}{UP}", end='', flush=True)
        return value, None, None, None, None, None, None, None, None


def refine_wavenumber_brackets_thickness(dpath, Rs, αbracket, δinner, args, log=False):
    """Update wavenumber brackets using cached eigenmodes if available."""
    import numpy as np
    from scipy.interpolate import interp1d

    try:
        v, α, _, _, δ, *_ = load_eigenmodes(dpath)
    except FileNotFoundError:
        return αbracket, δinner

    if v.size < 2:
        return αbracket, δinner

    print("\nImproved wavenumber bounds:")

    def _alpha_at_last_leq(x):
        idx = np.searchsorted(v, x, side="right") - 1
        return α[idx] if idx >= 0 else None

    def _alpha_at_first_geq(x):
        idx = np.searchsorted(v, x, side="left")
        return α[idx] if idx < v.size else None

    # Determine interpolation kind based on points
    if v.size >= 4:
        kind = 'cubic'
    elif v.size == 3:
        kind = 'quadratic'
    else:
        kind = 'linear'

    # Configure x-axis based on log setting
    I = np.where(v > 0.0)
    x_input = np.log10(v[I])  if log else v
    x_eval  = np.log10(Rs) if log else Rs

    # Initialize interpolator
    # fill_value=(δ[0], δ[-1]) ensures the last available values are used outside the range
    f = interp1d(x_input, δ, kind=kind, bounds_error=False, fill_value=(δ[0], δ[-1]))
    δinner = f(x_eval)

    # Fallback to nearest-neighbor (k=0) if negative values are encountered
    if δinner.min() <= 1.0e-4:
        f_fallback = interp1d(x_input, δ, kind='nearest', bounds_error=False, fill_value=(δ[0], δ[-1]))
        δinner = f_fallback(x_eval)
    
    vmn, vmx = v.min(), v.max()

    #δinner = δinner.tolist()
    #for i, x in enumerate(Rs):
    #    if x < vmn or x > vmx:
    #        δinner[i] = None

    for n, x in enumerate(Rs):
        kl, ku = args.wavenumber_bracket or (None, None)

        if vmn <= x <= vmx: # we can interpolate or determine bracket between already known points
            αl = _alpha_at_last_leq(x)
            αu = _alpha_at_first_geq(x)

            if αl is not None and αu is not None:
                if kl is None:
                   kl = min(αl, αu)
                else:
                   kl = max(kl, min(αl, αu))
                if ku is None:
                   ku = max(αl, αu)
                else:
                   ku = min(ku, max(αl, αu))

        if kl is None or ku is None:
            αbracket[n] = None
            continue

        todo = not np.isclose(kl, ku)
        if not todo:
            kl *= max(0.5, 1.0 - 5.0 * args.wavenumber_tolerance)
            ku *= min(1.5, 1.0 + 5.0 * args.wavenumber_tolerance)

        αbracket[n] = [float(kl), float(ku)]
        if todo or args.force:
            print(
                f"\t{args.dependence} = {x:+.3e}: "
                f"α = {αbracket[n][0]:.4e} ... {αbracket[n][1]:.4e}"
            )

    return αbracket, δinner


if __name__ == "__main__":
    '''
        Given provided options, calculates eigenmodes of the equilibrium field with magnetic and velocity shear.
    '''
    import argparse, multiprocessing, os, sys, time
    import numpy as np

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
        "--resolution-range", "-N", type=int, nargs=3, default=[128, 2048, 32],
        help="the range of the grid resolutions followed by the increment"
    )
    parser.add_argument(
        "--n-inner", "-nin", type=int, default=5,
        help="Minimum number of collocation points (default 3) required to resolve the smallest inner-layer width."
    )
    parser.add_argument(
        "--amp-fraction-outer", "--f-outer", type=float, default=0.001,
        help=("Fraction of the eigenmode amplitude that must be resolved "
              "at the outermost collocation point z_max. "
              "For example, 0.01 means the eigenmode amplitude should drop "
              "to 1%% of its maximum at z_max (equivalent to q = -ln(0.01) ≈ 4.605).")
    )
    parser.add_argument(
        "--scaling-factor", "-C", type=float, default=None,
        help="the scaling factor for the grid"
    )
    parser.add_argument(
        "--wavenumber-bracket", "-K", type=float, nargs=2, default=None,
        help="the bracket for the wavenumber")
    parser.add_argument(
        "--growth-rate-range", "-E", type=float, nargs=2, default=[1e-6, 1],
        help="the range for the growth rate to consider"
    )
    parser.add_argument(
        "--imaginary-amplitude-limit", "-I", type=float, default=None,
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
        "--orderby", "-O", default="real",
        help="the ordering of wavenumber: magnitude/amplitude, real and imaginary parts, or tolerance/errors"
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

    Dstring = { 'S': 'Lundquist number', 'Pr': 'Prandtl number' }
    if args.CGL:
        Dstring['β']  = 'plasma-β'
        Dstring['Δβ'] = 'plasma-β difference'
    Dstring['ξ'] = 'magnetic transverse field'
    Dstring['ϵ'] = 'Hall current term'
    Dstring['a'] = 'current sheet thickness'
    Dstring['w'] = 'current sheet width'

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
    if C is not None:
        print(f"  Scaling factor (C)                 =  {C:.3e}")
    if args.wavenumber_bracket is not None:
        print(f"  Wavenumber bracket                 =  {args.wavenumber_bracket}")
    print(f"  Growth rate range                  =  {args.growth_rate_range}")
    print(f"  Growth rate absolute tolerance     =  {args.absolute_tolerance:.3e}")
    print(f"  Growth rate relative tolerance     =  {args.relative_tolerance:.3e}")
    print(f"  Growth rate guess tolerance        =  {args.guess_tolerance:.3e}")
    if args.imaginary_amplitude_limit is not None:
        print(f"  Imaginary amplitude limit          =  {args.imaginary_amplitude_limit:.3e}")
    print(f"  Wavenumber relative tolerance      =  {args.wavenumber_tolerance:.3e}")
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
    if args.CGL and args.dependence not in [ 'β', 'Δβ' ]:
        if β + Δβ < 0:
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

    params = {'Nmin'            : Nmin, \
              'Nmax'            : Nmax, \
              'Ninc'            : Ninc, \
              'CGL'             : args.CGL, \
              'n_inner'         : args.n_inner, \
              'decay_efolds'    : -float(np.log(args.amp_fraction_outer)), \
              'C'               : args.scaling_factor, \
              'a'               : a, \
              'w'               : w, \
              'S'               : S, \
              'Pr'              : Pr, \
              'ξ'               : ξ, \
              'ϵ'               : ϵ, \
              'β'               : β, \
              'Δβ'              : Δβ,
              'ɣpar'            : ɣpar, \
              'ɣper'            : ɣper, \
              'mode'            : max(0, args.modes-1), \
              'σlower'          : args.growth_rate_range[0], \
              'σupper'          : args.growth_rate_range[1], \
              'imσamp'          : args.imaginary_amplitude_limit, \
              'orderby'         : args.orderby, \
              'atol'            : args.absolute_tolerance, \
              'rtol'            : args.relative_tolerance, \
              'wtol'            : args.wavenumber_tolerance, \
              'gtol'            : args.guess_tolerance, \
              'δtol'            : args.thickness_tolerance, \
              'ntasks'          : number_of_tasks, \
              'force'           : args.force, \
              'verbose'         : args.verbose}

    dpath  = './RESULTS'
    if not os.path.exists(dpath):
        os.makedirs(dpath)

    fname = os.path.join(dpath, '')
    for item in Dstring.keys():
        if item != args.dependence:
            if item in ['Δβ']:
                fname += f'{item}{params[item]:+.2f}'
            else:
                fname += f'{item}{params[item]:.3e}'
    fname += f"{args.suffix}"
    dpath  = fname + '-cache'
    fname += '.dat'

    params['data_path'] = dpath

    if args.wavenumber_bracket is not None:
        αbracket = [ args.wavenumber_bracket ]*Rs.size
    else:
        αbracket = [ None ]*Rs.size
    δinner = [ None ]*Rs.size

    if os.path.exists(dpath):
        αbracket, δinner = refine_wavenumber_brackets_thickness(dpath, Rs, αbracket, δinner, args, log=log)
    else:
        os.makedirs(dpath)

    print(f"\nCalculation initiated with {number_of_cores} process{'es' if number_of_cores > 1 else ''} for {Rs.size} values.\n")

    counter = multiprocessing.Value('i', 0)
    shared_counter = multiprocessing.Value('i', 0)

    start_time = time.time()

    with multiprocessing.Pool(
        processes=number_of_cores,
        initializer=init_worker,
        initargs=(shared_counter,)
    ) as pool:
        async_results = [ pool.apply_async(task, args=(dpath, args.dependence, r, ntask, number_of_tasks, αbracket[ntask], δinner[ntask], params)) for ntask, r in enumerate(Rs) ]
        results = [ r.get() for r in async_results ]

    if os.path.exists(dpath):
        v, α, σ, e, δ, c, n, N = load_eigenmodes(dpath)

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
            io.write(f"# wavenumber range (k)                 =  {args.wavenumber_bracket}\n")
            io.write(f"# resolution range (N)                 =  {[Nmin, Nmax]} with the increment of {Ninc}\n")
            io.write(f"# growth rate range                    =  {args.growth_rate_range}\n")
            if args.imaginary_amplitude_limit is not None:
                io.write(f"# imaginary amplitude limit            =  {args.imaginary_amplitude_limit:.3e}\n")
            io.write(f"# eigenvalue absolute tolerance        =  {args.absolute_tolerance:10.3e}\n")
            io.write(f"# eigenvalue relative tolerance        =  {args.relative_tolerance:10.3e}\n")
            io.write(f"# wavenumber relative tolerance        =  {args.wavenumber_tolerance:10.3e}\n")
            io.write(f"# selection order                      =   {args.orderby}\n")
            io.write(f"# converge the number of modes         =   {args.modes}\n")
            io.write(f"# inner-layer thickness tolerance      =  {args.thickness_tolerance:10.3e}\n")
            io.write(f"# number of inner collocation points   =   {args.n_inner}\n")
            io.write(f"# amplitude fraction at zmax           =   {args.amp_fraction_outer}\n")
            io.write("#\n#")
            io.write(f"     {args.dependence:<2s}              α_max           Re(σ_max)       Im(σ_max)       δ_in            tolerance       C              n_in      N\n")
            io.write("#  --------------  --------------  --------------  --------------  --------------  --------------  --------------  --------  --------\n")

            for i in range(v.size):
                io.write(f"  {v[i]:14.6e}  {α[i]:14.6e}  {σ[i].real:14.6e}  {σ[i].imag:14.6e}  {δ[i]:14.6e}  {e[i]:14.6e}  {c[i]:14.6e}    {n[i]:>6d}    {N[i]:>6d}\n")
            io.flush()

    print(f"\n\n\n\n\nCalculations done in {time.time() - start_time:.2f} seconds.\n")
