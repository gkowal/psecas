# Psecas — Code Audit

**Date:** 2026-09-22
**Revision audited:** branch `solver/multimode-iterative-solver` @ `91c0864`, plus uncommitted/untracked
working-tree files (`psecas/systems/tearing_instability.py`, `psecas/systems/tearing_instability-dpi.py`,
`tests/test_grids_generic.py`, `examples/tearing-instability/`).
**Environment:** Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1, SymPy 1.14.0, Matplotlib 3.11.2.

---

## 1. Scope and method

The whole repository was reviewed: the `psecas` package (4 978 LOC), the test suite (1 907 LOC,
24 files) and the example scripts (7 234 LOC). Review combined line-by-line reading with empirical
verification — every finding marked **Confirmed** below was reproduced by running code against this
tree. Numerical claims were checked against dense `scipy.linalg.eig` references.

Test suite status on this environment: **54 passed, 1 failed** (`test_grids_generic.py::
test_grid_Dn_up_to_4th_order_against_analytic[LegendreExtremaGrid]`, rel. L2 error 6.203e-05 vs
tolerance 6.000e-05).

---

## 2. Executive summary

Psecas is a well-conceived package: the Grid/System/Solver separation is clean, the spectral
discretisations are textbook-correct, and the equation-string parser is an elegant piece of design.
The finite-domain, Dirichlet-boundary, full-spectrum path — which is what the published Berlok &
Pfrommer results rest on — is sound, and the regression tests pinning those eigenvalues pass.

The problems are concentrated in the parts that have grown since: the shift-invert "fast" solvers,
the parser's interaction with substitutions, and the plotting layer.

The single most consequential finding is **C1**: every shift-invert solve of a *generalised* EVP
(`solve_mode`, `solve_with_guess`, and therefore `iterate_solver` and `iterate_solve_multimode`)
returns approximately the guess it was given rather than a converged eigenvalue. Because the
iterative drivers measure convergence by comparing successive eigenvalues, this produces **false
convergence**: the solver reports a small error while sitting on a wrong answer. On the tearing
instability system this was measured at **82 % relative error with a reported Δσ/σ of 3 %**.

| # | Severity | Finding | Status |
|---|---|---|---|
| C1 | **Critical** | Generalised shift-invert returns the shift, not the eigenvalue → silent false convergence | Confirmed |
| C2 | **Critical** | `solve_with_guess` omits `M=mat2` in the generalised path | Confirmed |
| C3 | **Critical** | Parser drops variables that appear only inside a substitution | Confirmed |
| H1 | High | `var_replace` mangles identifiers containing digits/underscores | Confirmed |
| H2 | High | `plot_solution` crashes on current Matplotlib | Confirmed |
| H3 | High | `add_equation` silently discards equations with no variable on the LHS | Confirmed |
| H4 | High | Grid pickle round-trip destroys the differentiation matrices irrecoverably | Confirmed |
| H5 | High | `prolongate_eigenvector` corrupts endpoint values on non-Dirichlet grids | Confirmed |
| H6 | High | Boundary conditions on `FourierGrid`/`HermiteGrid` silently produce wrong matrices | Confirmed |
| M1–M12 | Medium | See §5 | Mixed |
| L1–L10 | Low | See §6 | — |

---

## 3. Critical findings

### C1 — Generalised shift-invert does not solve the eigenvalue problem

**Files:** [psecas/solver.py:195-218](psecas/solver.py#L195-L218) (`solve_mode`),
[psecas/solver.py:735-749](psecas/solver.py#L735-L749) (`solve_with_guess`)

`scipy.sparse.linalg.eigs(A, M=B, sigma=…)` requires `M` to be **symmetric positive definite**.
Psecas's `mat2` never satisfies this once boundary conditions are present: `get_matrix2` zeroes whole
rows to decouple the BC equations from the eigenvalue, which makes `mat2` singular, and physical
LHS terms make it indefinite or non-symmetric.

Measured on the systems in this tree:

| System | `mat2` symmetric | rank deficiency | eigenvalue range |
|---|---|---|---|
| `Channel` (test_channel_solver) | yes | 8 | [-1, 0] |
| `TearingClassicalMHD` (non-periodic) | **no** | 4 | [-7.96e+03, 1] |

ARPACK's convergence test is then meaningless and it returns after essentially one iteration. The
returned value tracks the shift:

```
TearingClassicalMHD(kx=0.5, a=1, S=1e4, periodic=False), N=64
dense reference (scipy.linalg.eig):  sigma = 0.00033106

  guess = 0.00049659  ->  solve_mode = 0.00049715   rel.err 5.02e-01
  guess = 0.00039727  ->  solve_mode = 0.00038234   rel.err 1.57e-01
  guess = 0.00034762  ->  solve_mode = 0.00034922   rel.err 5.49e-02
  guess = 0.00033106  ->  solve_mode = 0.00033106   rel.err 5.77e-12
```

The answer is only correct when the guess was already correct.

**Why this is critical rather than merely inaccurate.** `iterate_solve_multimode` switches to
`solve_mode` once `delta <= gtol`, then declares convergence by comparing Σ_new to Σ_old. Since
`solve_mode` *returns* Σ_old, the comparison always succeeds. The convergence criterion validates
itself. Forcing the guess path on the tearing system:

```
N:   32,  sigma = 3.2888e-05
N:   48,  sigma = 1.8102e-04                           Δσ/σ = 8.18e-01
N:   64,  sigma = 1.7539e-04   [with guess]            Δσ/σ = 3.64e-02
N:   96,  sigma = 1.7090e-04   [with guess]            Δσ/σ = 2.90e-02
N:  128,  sigma = 1.6649e-04   [with guess]            Δσ/σ = 2.91e-02

returned: 1.6649e-04     truth at N=128 (dense): 9.3675e-04     ->  82.2 % error
```

The eigenvalue freezes near where it was when the switch happened, while Δσ/σ falls to ~3 % and keeps
shrinking — exactly the signature of convergence. With a looser `rtol` this run would have set
`result["converged"] = True`.

**Note:** the *standard* EVP path (`do_gen_evp=False`, `B=None`) is unaffected — `OPinv = (A-σI)⁻¹`
with `M=None` is the mathematically correct operator there.

**Recommended fix.** Do not delegate the generalised shift-invert to ARPACK's `M=` interface. Either

- drive the iteration yourself with the correct operator `OP = (A - σB)⁻¹B` supplied as a
  `LinearOperator` to a *standard* `eigs` call (no `M=`), reconstructing `λ = σ + 1/θ`; or
- keep `eigs` but **verify every returned mode** with the relative residual
  `‖Av - σBv‖ / (‖Av‖ + |σ|‖Bv‖)` and fall back to `solve_full` when it exceeds tolerance.

`solve_mode` already computes exactly this residual inside `refine_eigenvector`
([solver.py:151-153](psecas/solver.py#L151-L153)) — and then **discards it**
([solver.py:222](psecas/solver.py#L222) assigns `v` but drops `r`). Returning and checking that
residual is the cheapest possible guard and would have caught this.

### C2 — `solve_with_guess` omits `M=mat2` in the generalised branch

**File:** [psecas/solver.py:737-742](psecas/solver.py#L737-L742)

```python
if useOPinv:
    OPinv = inv((self.mat1 - guess * self.mat2).toarray())
    sigma, v = eigs(self.mat1, k=1, sigma=guess, OPinv=OPinv)   # M=self.mat2 missing
else:
    sigma, v = eigs(self.mat1, M=self.mat2, k=1, sigma=guess)
```

`OPinv` is built for the pencil `(A, B)` but the operator is then applied to the *standard* problem.
This is a distinct defect from C1 and compounds it. On a clean, well-conditioned generalised problem
where C1 does not apply (SPD `B`, no boundary rows):

```
true nearest generalized eigenvalue : 2.8565474 + 0.6458806j
solve_with_guess (M= omitted)       : 6.1418976 + 1.6797993j      <- wrong mode entirely
correct (M=B)                       : 2.8565474 + 0.6458806j
```

`useOPinv=True` is the **default**, and `iterate_solver` passes it through, so this is the path most
example scripts take.

Note the asymmetry this creates: `useOPinv=False` is currently *more* correct than the default.
Tests `test_khi_solver.py` parametrise over both and pass only because the KHI system's `mat2` is
close to the identity.

**Fix:** add `M=self.mat2` to the `useOPinv` call — then address C1, which the fix alone does not
cure.

### C3 — Variables reachable only through a substitution are silently dropped

**File:** [psecas/solver.py:1023-1026](psecas/solver.py#L1023-L1026)

```python
for i, var in enumerate(self.system.variables):
    # Fast path: variable absent -> sparse zero (no dense zeros)
    if var not in eq:
        mats.append(sparse.lil_matrix((NN, NN), dtype=np.complex128))
        continue
    ...
    # substitutions are applied 10 lines LATER, at line 1033
```

The early-out tests the **raw** equation text, but substitutions are expanded afterwards. A variable
that enters the equation only via a substitution is therefore replaced by a zero block. No error, no
warning — the returned eigenvalue is simply wrong.

Reproduction:

```python
system.add_substitution('G = dz(dz(f))')
system.add_equation("sigma*f = q*G", boundary=True)   # 'f' on RHS only through G

solver.get_matrix1()
# mat1 nonzeros: 0
# solve() -> sigma = 0j        (expected -pi**2 = -9.8696)
```

`add_substitution` is public API and documented in `System`; nothing warns against this usage.

**Fix:** expand substitutions once per equation *before* the per-variable loop, and test the expanded
text in the early-out. This is also faster — substitutions currently re-expand for every variable.

---

## 4. High-severity findings

### H1 — `var_replace` treats digits and underscores as word boundaries

**File:** [psecas/string_methods.py:20-26](psecas/string_methods.py#L20-L26)

The guard checks only `str.isalpha()` on the neighbouring characters, so a name is considered
"standalone" when it is followed by a digit or underscore:

```
var_replace('beta2*x',  'beta', 'B') -> 'B2*x'        # should be unchanged
var_replace('v_x + v',  'v',    'W') -> 'W_x + W'     # should be 'v_x + W'
var_replace('rho1*rho', 'rho',  'R') -> 'R1*R'        # should be 'rho1*R'
```

Any system using `beta`/`beta2`, `v`/`v_x`, `rho`/`rho1`-style names silently gets a corrupted
equation. This affects substitutions, the eigenvalue symbol, the coordinate symbol and every variable
name. The `Solver.__init__` substring check
([solver.py:21-23](psecas/solver.py#L21-L23)) catches variable-vs-variable collisions but not
variable-vs-parameter ones, which is where this bites.

**Fix:** replace the hand-rolled scan with `re.sub(r'(?<![A-Za-z0-9_])' + re.escape(var) + r'(?![A-Za-z0-9_])', ...)`.

Related latent hazard in the same function: after a successful substitution `pos` is not advanced, so
a replacement string containing `var` as a standalone token would loop forever. Today's call sites
escape only because `"grid.zg"` happens to embed its `z` and `r` next to alphabetic characters — luck,
not design.

### H2 — `plot_solution` is broken on current Matplotlib

**File:** [psecas/plotting.py:10-12](psecas/plotting.py#L10-L12)

```python
plt.figure(num)
plt.clf()
fig, axes = plt.subplots(num=num, nrows=system.dim, sharex=True)
```

`plt.figure(num)` creates figure `num`; `plt.subplots(num=num, …)` then attempts to create it again
with a different geometry. On Matplotlib ≥ 3.8 this raises:

```
ValueError: Figure 1 already exists. Use plt.figure(1) to get it or plt.close(1) to close it.
Alternatively, pass 'clear=True' to subplots().
```

Verified for `dim=1` and `dim=5`. This is the package's headline convenience function; **10 example
scripts call it** and all of them fail on a current install. It is invisible to CI because
`.coveragerc` omits `plotting.py` and no test exercises it.

**Fix:** drop the two `plt` lines and use `plt.subplots(num=num, nrows=…, sharex=True, clear=True)`.

A second, independent bug lurks behind it: when `system.dim == 1`, `plt.subplots` returns a bare
`Axes`, so `axes[j]` raises `TypeError`. Pass `squeeze=False` or wrap with `np.atleast_1d`.

### H3 — `add_equation` silently discards equations

**File:** [psecas/system.py:37-51](psecas/system.py#L37-L51)

If no variable name is found in `eq.split('=')[0]`, the method falls through the loop and returns
having stored nothing. The slot in `self.equations` stays `''`, and the failure surfaces much later
and far away:

```
IndexError: list index out of range
  at psecas/solver.py:886  ->  equation = equation.split("=")[1]
```

Two of my own probe scripts hit this before I recognised the pattern; a user writing
`sigma*G = -q*G` (LHS expressed through a substitution) gets an `IndexError` in the matrix builder
with no indication that their equation was never registered.

The matching detection is also substring-based, so an LHS like `sigma*dvx` matches both `vx` and
`dvx` if both are variables.

**Fix:** `raise ValueError` naming the equation when `found` is False; use `var_replace`-style
word-boundary matching for the LHS scan.

### H4 — Grid pickle round-trip is irreversible

**File:** [psecas/grids/grid.py:21-60](psecas/grids/grid.py#L21-L60)

`__getstate__` strips the differentiation matrices with the comment *"they can be rebuilt later by
calling make_grid() or via lazy access"*. Lazy access does not work:

```python
g2 = pickle.loads(pickle.dumps(ChebyshevExtremaGrid(16, -1, 1)))
g2._d       # []
g2.D(1)     # NotImplementedError
```

`D(k)` → `ensure_derivatives` → `build_derivatives` → `self._build_d1()`, and `_build_d1` is an
abstract stub ([grid.py:66-67](psecas/grids/grid.py#L66-L67)) that **no grid subclass implements** —
every subclass populates `self._d` directly inside `make_grid`. `Grid._identity` is likewise wrong for
`FourierGrid` and `HermiteGrid`, which use `NN = N` rather than `N + 1`.

This breaks `load_system` and the MPI `IO.save_system` workflow: any post-processing that calls
`grid.der`, `grid.interpolate` or `plot_solution` on a reloaded system fails.

**Fix:** call `self.make_grid()` at the end of `__setstate__`, and delete the unreachable
`_build_d1`/`build_derivatives` path (or implement it properly per subclass).

### H5 — `prolongate_eigenvector` corrupts endpoint values

**File:** [psecas/solver.py:369-371](psecas/solver.py#L369-L371)

```python
f_new = grid_old.interpolate(self.grid.zg[1:-1], f_old)
fields_new[var] = np.pad(f_new, pad_width=1, mode='constant',
                         constant_values=(f_old[0], f_old[-1]))
```

Interior nodes are interpolated properly, but the two endpoints are filled with the *old grid's*
endpoint values. That is correct only when those values are zero (all-Dirichlet) or when the endpoints
sit at fixed physical positions. Neither holds for periodic or infinite grids, whose node positions
move with `N`:

```
FourierGrid, f = sin(2*pi*z), N: 32 -> 64
  interior  max error: 6.7e-16
  endpoints max error: 4.9e-02      V[0] = 0.098017  vs exact 0.049068   (factor 2 off)
```

Since this vector is the `v0` starting guess handed to `solve_mode`, a poisoned guess degrades
convergence exactly where the multimode solver is trying to save work.

A related latent hazard: for `ChebyshevRationalGrid`, `SincGrid`, `HermiteGrid` and `LaguerreGrid` the
domain *extent* grows with `N`, so `self.grid.zg[1:-1]` can fall outside `grid_old`'s range and trip
the `"Can't interpolate outside grid domain"` assertion. A 49→99 refinement survives only because the
outermost rational-Chebyshev node is far outside the rest (`zg[0] = -63.66`, `zg[1] = -21.20`).

**Fix:** interpolate the full new node set (`self.grid.zg`, not `[1:-1]`) and re-impose boundary values
explicitly only when the packing is actually trimmed; clamp or extrapolate for infinite grids.

### H6 — Boundary conditions on `FourierGrid` / `HermiteGrid` produce wrong matrices without error

**Files:** [psecas/solver.py:891-900](psecas/solver.py#L891-L900),
[psecas/solver.py:1124-1132](psecas/solver.py#L1124-L1132),
[psecas/solver.py:844-864](psecas/solver.py#L844-L864)

The solver assumes `NN == N + 1` throughout: it trims with `[1:N, 1:N]`, addresses boundary nodes as
`0` and `N`, and `keep_result` slices in blocks of `N-1`. But the grids disagree on that convention —
`FourierGrid` and `HermiteGrid` set `NN = N`, everything else sets `NN = N + 1`.

```python
grid = FourierGrid(N=32, zmin=0, zmax=1)   # NN = 32
system.add_equation("sigma*f = q*dz(dz(f))", boundary=True)
solver.get_matrix1()
# mat1.shape == (31, 31)      <- a grid point was silently deleted
```

No exception is raised; the eigenvalues are simply for a different (meaningless) discretisation, and
`keep_result` then pads back to length 33 ≠ 32. Combining boundaries with a periodic grid is
physically odd, but nothing stops a user doing it, and the same index arithmetic makes
`_modify_submatrix` write to row `N` — out of range for those grids.

**Fix:** derive the trim and boundary indices from `grid.NN` rather than `grid.N`, and raise a clear
error when boundary conditions are requested on a grid with no boundary nodes.

---

## 5. Medium-severity findings

**M1 — `filter_modes` hard-codes `require_re_positive=True` in the multimode driver.**
[solver.py:554](psecas/solver.py#L554), [solver.py:592](psecas/solver.py#L592) call `filter_modes`
without the flag, so the default `True` applies and `iterate_solve_multimode` **cannot find damped or
oscillatory modes** (Re σ ≤ 0). The parameter exists on `filter_modes` but is not plumbed through.
For a package whose stated purpose is linear stability analysis, silently discarding the stable half
of the spectrum is a significant restriction and is undocumented in the method's long docstring.
Related: `_errors` computes `fac = 1 + |Im σ| / max(atol, Re σ)`
([solver.py:519](psecas/solver.py#L519)) which is meaningless for negative `Re σ`, so relaxing M1
requires fixing that too.

**M2 — `iterate_solve_multimode(Ns)` crashes for a single resolution.**
`UnboundLocalError: cannot access local variable 'errors'` at
[solver.py:632-633](psecas/solver.py#L632-L633) — `errors` is only bound inside the `for N in Ns[1:]`
loop. Confirmed. `iterate_solver` has the mirror-image problem: it indexes `Ns[1]` unconditionally
([solver.py:793](psecas/solver.py#L793)) and raises `IndexError` for `len(Ns) < 2`. Neither
requirement is documented.

**M3 — Boundary-expression validation rejects legal input and uses `assert`.**
[solver.py:1134-1135](psecas/solver.py#L1134-L1135):

```python
assert int(bound.split("=")[1]) == 0, 'rhs of boundary expressions must be zero'
```

`add_boundary('f', 'dz(f) = 0.0', …)` — a natural spelling — raises
`ValueError: invalid literal for int() with base 10: ' 0.0'` instead of the intended message.
Confirmed. Both assertions here are also stripped under `python -O`, turning a validation failure into
silent miscomputation. Use `float(...)` and raise `ValueError`.

**M4 — `LegendreExtremaGrid` produces a complex-valued grid.**
`legroots` returns `complex128`, so `self.zg` and every differentiation matrix are complex
([legendre_extrema.py:36](psecas/grids/legendre_extrema.py#L36)). The imaginary parts are exactly
zero, but the cost is doubled memory and arithmetic throughout the solver, plus `ComplexWarning`s in
downstream code. `np.errstate(divide='ignore')` at line 40 also fails to suppress the accompanying
`RuntimeWarning: invalid value encountered in divide`. Wrap with `np.real(...)` and add `invalid='ignore'`.

**M5 — The one failing test is a real accuracy signal.**
`LegendreExtremaGrid: D(4) rel L2 err = 6.203e-05, tol = 6.000e-05`. Orders above 2 are built by
repeated multiplication (`self._d[1] @ self._d[-1]`,
[grid.py:128-129](psecas/grids/grid.py#L128-L129)), which loses roughly one digit per order on
spectral matrices. `ChebyshevTLnGrid` already carries a `TODO` acknowledging the same issue for `d2`.
Either compute `D(n)` from explicit formulas or document the accuracy ceiling and loosen the test —
but do not simply bump the tolerance without recording why.

**M6 — `get_2Dmap` ignores its `xmin` argument.**
[plotting.py:59-60](psecas/plotting.py#L59-L60): `xg = (0.5 + np.arange(Nx)) * dx`, missing the
`+ xmin` that the adjacent `zg` line has. Confirmed: maps for `xmin=0` and `xmin=100` are bit-identical.

**M7 — `get_2D_cylindrical_map` mis-shapes its output.**
[plotting.py:183](psecas/plotting.py#L183): `np.resize(y, (Nx, Ny))` while `meshgrid` produced
`(Ny, Nx)`. The subsequent multiply by `phiphi` broadcasts incorrectly, or raises, unless `Nx == Ny`.
`get_2D_cylindrical_map_in_cylindrical_coords` similarly ignores `phimin`
([plotting.py:106](psecas/plotting.py#L106)) and is not exported from `psecas/__init__.py`.

**M8 — Unused and misleading parameters.**
`Solver.solve(useOPinv=True, …)` documents the parameter and never uses it
([solver.py:636](psecas/solver.py#L636)); `solve_mode(verbose=…)` likewise. `solve` always does a full
dense `eig`, so a caller passing `useOPinv` believes they selected a sparse path they did not get.

**M9 — `sorting_strategy` mutates its argument and hard-codes thresholds.**
[solver.py:834-835](psecas/solver.py#L834-L835) sets `E[np.abs(E.real) > 10.0] = 0` **in place** on the
caller's array, and the magic value `10.0` is arbitrary for a problem whose eigenvalues may be
O(10⁻⁴) (tearing) or O(10²) (channel). It is documented as overridable — `test_channel_solver` does
exactly that — but the default silently zeroes legitimate eigenvalues. Copy the input and make the
cutoff a parameter.

**M10 — `keep_result` can collide with its own metadata keys.**
[solver.py:865-867](psecas/solver.py#L865-L867) writes `mode`, and the iterative drivers add
`converged`, `error`, `grid`, `r_err`, `a_err` into the same dict that holds the eigenmode profiles
keyed by variable name. A system with a variable called `grid` or `mode` loses data silently. Use a
nested `result['fields'][var]` or prefix the metadata.

**M11 — `mpi_io.IO` shells out for filesystem operations.**
[mpi_io.py:30-37](psecas/mpi_io.py#L30-L37):

```python
subprocess.call("mkdir " + self.data_folder, shell=True)
subprocess.call("cp " + experiment + " " + data_folder + experiment, shell=True)
```

Unquoted interpolation into `shell=True` — a folder or script name containing a space, `;` or `$`
misbehaves or executes. `mkdir` without `-p` fails on an existing directory and the non-zero status is
discarded. The `cp` destination concatenates `data_folder + experiment`, which breaks whenever
`experiment` carries a directory prefix. Use `os.makedirs(..., exist_ok=True)` and `shutil.copy`.
The bare `except: pass` at [mpi_io.py:45](psecas/mpi_io.py#L45) swallows `KeyboardInterrupt` too.
`log()` divides by `self.steps_local`, which is zero when there are more ranks than steps.

**M12 — `eval` runs in an environment described as restricted but is not.**
[solver.py:1064](psecas/solver.py#L1064) and [solver.py:1166](psecas/solver.py#L1166) pass
`{"__builtins__": {"__import__": builtins.__import__}}` under the comment *"Evaluate the expression in
a restricted environment."* Verified: `eval("__import__('os').getcwd()", …)` succeeds — exposing
`__import__` alone defeats the sandbox entirely. Equations are normally author-supplied, so this is not
an exploit path in ordinary use, but it becomes one in combination with `serial_io.load_system`, which
`pickle.load`s arbitrary files ([serial_io.py:8](psecas/serial_io.py#L8)) — unpickling untrusted data
is arbitrary code execution regardless. Either drop the misleading comment or remove `__import__` and
inject the handful of names (`np`, etc.) the equations actually need.

---

## 6. Low-severity findings and code quality

**L1 — No packaging metadata.** There is no `setup.py`, `pyproject.toml` or `setup.cfg`. `pip install
psecas` is impossible; the README instructs `pip install -r requirements.txt` and relies on the user
running from the repository root. The workaround is visible in the tree: `tests/psecas` is a
**symlink to `psecas/`**, present solely to make imports resolve. A minimal `pyproject.toml` removes
that hack and lets CI test an installed package.

**L2 — Unpinned dependency on a personal Git fork.** `requirements.txt` ends with
`git+https://github.com/tberlok/dmsuite.git` — no tag, no commit hash. `HermiteGrid` and `LaguerreGrid`
break if that fork moves or disappears. Pin a commit, or vendor the ~100 lines of `herdif`/`lagdif`
actually used.

**L3 — CI targets an unsupported Python and a retired image.** `.circleci/config.yml` uses
`circleci/python:3.6.1`; Python 3.6 reached end of life in 2021 and the `circleci/*` image namespace is
deprecated. The code is developed on 3.14 (f-strings, walrus-free but `Δ`/`σ` identifiers, `@` matmul).
CI is therefore testing a configuration nobody uses, and the badge in the README may be stale.

**L4 — 17 test functions `return` instead of `assert`.** Every grid test file ends with
`return ...`, raising `PytestReturnNotNoneWarning` on pytest 7+ and scheduled to become an **error**
in pytest 8. Worst offenders: `test_grids_generic.py` (8), `test_sinc.py` (6),
`test_rational_chebyshev.py` (6), `test_hermite.py` (6).

**L5 — One test is tautological.** `test_higher_order_derivatives.py::
test_higher_order_composition_consistency` asserts `D(k) ≈ D(1) @ D(k-1)` to `1e-12` — which is the
literal implementation of `ensure_derivatives`. It cannot fail and validates nothing about accuracy.
`test_grid_Dn_up_to_4th_order_against_analytic` is the test that does real work (and is the one
failing, per M5).

**L6 — Substantial duplication.**
- `psecas/systems/tearing_instability-dpi.py` is a 657-line near-copy of `tearing_instability.py`;
  `diff` reports **41 changed lines**. The hyphen in the filename also makes it un-importable
  (`import …tearing_instability-dpi` is a syntax error), so it can only be loaded via `importlib`.
  Fold the variant into the main class behind a flag.
- Five grid classes (`chebyshev_rational`, `sinc`, `hermite`, `laguerre`, `chebyshev_semi_infinite`)
  each re-implement `__init__`, `bind_to`, and the `N`/`zmin`/`zmax`/`C` properties instead of
  inheriting them — ~40 duplicated lines each. They subclass `Grid` but bypass `Grid.__init__`
  entirely. An intermediate `InfiniteGrid` base would remove ~200 lines and fix H4 in one place.
- `examples/tearing-instability/` holds five near-variants of the same driver
  (`eigenmodes-compute.py`, `-orig.py`, `-optimized.py`, `-maxima*.py`), ~5 000 lines total.

**L7 — Inconsistent indentation.** Both `tearing_instability*.py` files are **tab-indented** (556/557
tab-led lines); the rest of the package uses 4 spaces. Mixed styles in one package are a merge-conflict
generator.

**L8 — Imports inside function bodies throughout.** Nearly every method begins with
`import numpy as np`. It is harmless at runtime (module cache) but defeats static analysis, hides the
dependency surface, and makes the 1 181-line `solver.py` harder to read. Move to module scope.

**L9 — Documentation gaps.** 148 of 216 public classes/functions (**69 %**) have no docstring.
`Solver` itself is documented as `"""docstring for Solver"""`
([solver.py:2](psecas/solver.py#L2)). The Sphinx setup under `docs/` exists but is minimal and its
copyright line still reads 2020. The README does not mention `iterate_solve_multimode`,
`solve_mode`, `filter_modes` or the `max_derivative_order` / `dz(var, n)` features — none of the work
on this branch is user-visible in the docs.

**L10 — Working tree hygiene.** Uncommitted at the time of audit: two patch files
(`fixes.patch`, `eigenmode_visualization.patch`) whose content is either already applied
(`fixes.patch` matches [solver.py:591-598](psecas/solver.py#L591-L598) verbatim) or never applied
(`plot_eigenmodes`); a `tests/psecas` symlink; an `.agents/` directory; and the two tearing system
modules, which examples already import. The `.gitignore` entry `*.txt` (no trailing newline) will
silently exclude any future text file added to the repo.

---

## 7. What is working well

Worth recording, since an audit naturally skews negative:

- **Architecture.** The Grid / System / Solver split is genuinely good, and the observer pattern
  (`grid.bind_to(system.make_background)`) that re-evaluates the equilibrium when `N` changes is an
  elegant solution to a real problem in resolution studies.
- **The equation parser.** Writing linearised equations as strings and having the solver assemble the
  block matrix is the package's main contribution, and `_rewrite_derivatives` is a clean refactor of
  what must once have been inline string surgery. The `dz(var, n)` higher-order syntax fits naturally.
- **Numerical cores.** The spectral discretisations match their Boyd/Trefethen references. The
  barycentric rewrite in `ChebyshevRationalGrid.cheb_gauss_nodes_and_Dx`
  ([chebyshev_rational.py:62-83](psecas/grids/chebyshev_rational.py#L62-L83)) — using
  `s = cos(φ)` to avoid cancellation as `|x| → 1`, and the negative-row-sum trick for the diagonal —
  is careful, well-commented numerical work. `sech_stable` in the tearing system is the same instinct.
- **Regression tests.** Pinning published eigenvalues to `atol=1e-8` (`test_mti_solver`,
  `test_mri_solution`, `test_khi_solver`) is the right way to protect a scientific code, and the
  analytic-solution tests (Bessel, Hermite, Laguerre, infinite well) validate the grids independently
  of the physics.
- **Error messages in the parser.** The `err_msg1` templates in `_find_submatrices` — showing the
  original equation, the transformed expression, and which variable was being processed — are more
  helpful than most codes of this size manage, and `test_error_messages.py` protects them.

---

## 8. Recommended order of work

**Before any further physics results are produced from this branch:**

1. **C1** — add a residual check to every shift-invert solve and fall back to `solve_full` when it
   fails. `refine_eigenvector` already computes the residual; return it and act on it. Until this is
   done, treat any eigenvalue from `iterate_solve_multimode`'s `[with guess]` path as unverified.
2. **C2** — one-line fix (`M=self.mat2`), then re-run `test_channel_solver` and `test_khi_solver`.
3. **C3** — move substitution expansion ahead of the per-variable early-out.
4. Add a regression test that compares each iterative driver against a dense `solve_full` reference on
   a generalised problem with a non-identity `mat2`. The current suite cannot catch C1 or C2 because
   every system it exercises has `mat2 ≈ I`.

**Next:**

5. **H1** (regex word boundaries), **H3** (raise on unmatched equation), **H2** (`clear=True` +
   `squeeze=False`) — each is a small, self-contained fix with clear blast radius.
6. **H4**, **H5**, **H6** — the `NN` vs `N + 1` convention should be settled once and applied
   everywhere; H6 and part of H5 both stem from it.
7. **M2**, **M3**, **M6**, **M7** — small correctness fixes with obvious tests.

**Then, as housekeeping:**

8. Add `pyproject.toml`, delete the `tests/psecas` symlink, pin `dmsuite`, modernise CI to Python
   3.11+ (L1–L3).
9. Convert `return` to `assert` in the test suite before pytest 8 makes it an error (L4).
10. Merge `tearing_instability-dpi.py` into `tearing_instability.py`; extract an `InfiniteGrid` base
    class (L6).
11. Plumb `require_re_positive` through `iterate_solve_multimode` so stable modes are reachable (M1).

---

*Findings marked "Confirmed" were reproduced against this working tree. Reproduction snippets are
embedded inline above and can be run from the repository root.*
