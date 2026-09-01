# PROJECT_STATUS

Persistent state for this repository. Update after **every** milestone.

**Current position:** Milestone 5 complete. Next: Milestone 6 (convergence and computational efficiency).

---

## Milestone 1 — European baseline + binomial tree ✅

### Completed
- `src/amopt/black_scholes.py` — closed-form European call/put and the full
  first/second-order Greek set (delta, gamma, vega, theta, rho), fully broadcast
  over all six parameters, with correct `T → 0` and `σ → 0` limits.
- `src/amopt/binomial.py` — CRR lattice from first principles: European and
  American exercise, lattice Greeks, and early-exercise-boundary extraction.
- `src/amopt/config.py` — ten canonical parameter regimes shared by all experiments.
- `src/amopt/plotting.py` — one figure style; fixed method→colour mapping.
- `tests/` — 78 tests passing.
- `experiments/m1_european_baseline.py` — convergence study.

### Important decisions
- **CRR parameterisation** `u = e^{σ√Δt}`, `d = 1/u`, `p = (e^{(r−q)Δt} − d)/(u − d)`.
  Node prices are computed as `S0·exp((2j − i)σ√Δt)` rather than `u^j d^{i−j}` to
  avoid floating-point overflow at large `N`.
- **The lattice raises on `p ∉ [0,1]`** instead of clamping. Clamping would hide
  an arbitrageable lattice and silently bias prices.
- **Lattice Greeks** exploit `ud = 1`: the step-2 nodes contain `S0` exactly, so
  delta/gamma/theta come from the lattice itself at zero extra cost.
- The **averaged lattice** `½(V_N + V_{N+1})` is available but is used only where
  explicitly stated, never silently.

### Numerical results
See `RESULTS.md` §1.

### Known limitations
- Lattice boundary extraction returns `nan` near `t = 0` because the lattice cone
  rooted at `S0` does not reach the exercise region. Structural, not a bug —
  documented in `crr()`. The PDE solver will not share this limitation.
- CRR is `O(N^{-1})` and oscillates with period two in `N`; single-lattice values
  at moderate `N` are not accurate enough to serve as the final reference solution.
  A high-resolution reference is built in Milestone 6.

### Unresolved
- None.

### Next
Superseded — see Milestone 2 below.

---

## Milestone 2 — Mathematical formulation ✅

### Completed
- `docs/01_formulation.md` — risk-neutral valuation, the optimal-stopping
  representation, the Snell envelope and why it forces `𝓛V ≤ 0`, exercise vs
  continuation regions, the variational inequality and its complementarity form,
  terminal and boundary conditions (including why `V(0,t) = K` for the American
  put but `Ke^{−rτ}` for the European), value matching, smooth pasting with an
  argument for *why* `V_S = −1`, and the perpetual closed form.
- `docs/02_crank_nicolson.md` — the full discretisation: grid construction with
  the strike pinned to a node, the tridiagonal coefficients with every factor of
  `ΔS` cancelled, the θ-scheme, the boundary rows at both time levels, the
  discrete LCP, the derivation of PSOR from Gauss–Seidel → SOR → projection, the
  M-matrix / cell-Péclet analysis, red–black ordering, Rannacher start-up, the
  Brennan–Schwartz cross-check, and boundary extraction by interpolating the
  early-exercise gap.
- `src/amopt/perpetual.py` + 12 tests.
- `experiments/m2_analytic_anchors.py`.

### Important decisions
- **Solve in time-to-maturity** `τ = T − t`, so the terminal condition becomes an
  initial condition and the scheme marches forward.
- **Uniform grid in `S`, not in `log S`.** Every power of `ΔS` cancels out of the
  coefficients, `S = 0` and the boundary region are both in the domain, and the
  strike can be pinned exactly to a node (which preserves second-order accuracy
  through the payoff kink).
- **The M-matrix condition is checked, not assumed.** `a_i ≥ 0` requires
  `i ≥ (r−q)/σ²`; for the base case that is `1.25`, so exactly one row (`i = 1`)
  fails. The solver will count violating rows and offer upwinding; the effect is
  to be *measured* in Milestone 3, not asserted negligible.
- **Two exact identities** (`𝓐ₕ1 = −r`, `𝓐ₕS = −qS`) are asserted at assembly.
  Central differences are exact on constants and linears, so these catch sign and
  indexing errors in the coefficients.

### Numerical results
See `RESULTS.md` §2.

### Known limitations
- The maturity-limit study hits the CRR noise floor beyond `T ≈ 50`; the
  perpetual limit cannot be verified more tightly with the lattice alone.

### Unresolved
- None. Two issues found during Milestone 2 were diagnosed and fixed: the `r = 0`
  perpetual degeneracy, and the apparent non-monotonicity of the gap to the
  perpetual value (lattice error, now quantified and reported).

### Next
Superseded — see Milestone 3 below.

---

## Milestone 3 — Crank–Nicolson + PSOR ✅

### Completed
- `src/amopt/crank_nicolson.py`: uniform price grid with the strike pinned to a
  node, θ-scheme time stepping with Rannacher start-up, the tridiagonal operator
  with assembly-time identity assertions, three LCP solvers (vectorised red–black
  PSOR, reference lexicographic PSOR, exact Brennan–Schwartz), configurable
  `omega`/`tol`/`max_iter`, selective and full upwinding, and exercise-boundary
  extraction with a smooth-pasting sub-grid refinement.
- `tests/test_crank_nicolson.py` — 74 tests. Total suite: 164 passing.
- `experiments/m3_crank_nicolson.py` — six result tables, four figures.

### Important decisions
- **Red–black ordering for PSOR.** A tridiagonal matrix is consistently ordered
  in Young's sense, so red–black SOR has the same spectral radius and the same
  optimal `ω` as the lexicographic sweep — but each half sweep is one vectorised
  NumPy expression. Measured 19× faster at `M = 400`, identical output.
- **Brennan–Schwartz kept as an exact cross-check**, not as the production path.
  PSOR is what the brief asks for; having a tolerance-free solver to compare
  against is what makes the PSOR result trustworthy.
- **Selective, not global, upwinding.** Measured: global upwinding is first order
  and costs `1.16 × 10⁻²`; selective upwinding costs `< 10⁻⁸`.
- **The exercise set is read from the projection, not from a threshold on
  `v − g`.** See `RESULTS.md` §3.7 — the threshold version was wrong at `r = 0`.
- **Convergence orders are measured by self-convergence per axis.** Comparing
  against an external benchmark while refining one axis measures the other
  axis's error floor. See `RESULTS.md` §3.2.

### Numerical results
See `RESULTS.md` §3.

### Known limitations
- **Temporal order is `1.26`, not `2`.** Inherent to applying Crank–Nicolson to a
  free-boundary problem, not a bug — confirmed by the fully implicit scheme
  measuring `1.01` on the same setup.
- PSOR iteration counts grow with grid size (`18.7` sweeps/step at `M = N = 2000`
  in the base case, `159` in the `high_vol` regime), because SOR's spectral
  radius approaches 1 under refinement. A fixed `ω` is therefore not optimal
  across the sweep; Milestone 6 quantifies the cost.
- At `r = 0` the boundary is only resolvable to floating-point noise
  (`S*(0) = 0.6` on a strike of 100), since the American and European values
  coincide to machine precision deep in the money.

### Unresolved
- None. Four issues found during Milestone 3 (grid-nudge bound, Brennan–Schwartz
  test premise, global-vs-selective upwinding, threshold-based boundary
  extraction) were diagnosed and fixed.

### Next
Superseded — see Milestone 4 below.

---

## Milestone 4 — Longstaff–Schwartz ✅

### Completed
- `src/amopt/lsm.py`: exact-in-distribution GBM simulation (no Euler error),
  three configurable regression bases (monomial, weighted Laguerre, Chebyshev),
  in-the-money path filtering, backward-induction policy fitting, forward policy
  evaluation, in-sample and out-of-sample estimators, and pair-aware standard
  errors for antithetic sampling.
- `bermudan_dates` added to `amopt.binomial.crr` — the exact `n`-date Bermudan
  value, which is the benchmark LSM should actually be measured against.
- `tests/test_lsm.py` — 32 tests. Total suite: 196 passing.
- `experiments/m4_longstaff_schwartz.py` — five result tables, three figures.

### Important decisions
- **Benchmark against the exact Bermudan value, not the American value.** With
  50 dates the exercise-date bias (`+0.0117`) is the same size as the whole
  regression error (`−0.0101`) and they nearly cancel. Reporting only the
  American deviation would have shown `−0.0017` and hidden both.
- **Both estimators are reported.** In-sample is the classic Longstaff–Schwartz
  and is biased high; out-of-sample fits the policy on one sample and values it
  on an independent one, and is a valid lower bound. Together they bracket.
- **Antithetic standard errors are computed over pair means**, not over paths.
  The pair is the unit of independence; treating `2n` dependent paths as `2n`
  observations is the standard way to report a spuriously tight interval.
- **Spurious NumPy warnings are suppressed narrowly, not globally.** NumPy 2.0.2's
  BLAS `matmul` raises FP flags from SIMD tail lanes on finite inputs; verified
  against `einsum` (8.5e-14) and an explicit row sum (bitwise identical). The
  suppression is wrapped in an explicit finiteness check so a genuine
  ill-conditioning failure still raises.

### Numerical results
See `RESULTS.md` §4.

### Known limitations
- LSM stores every path at every exercise date, so memory is `O(n_paths × n_dates)`.
  200,000 × 50 doubles is ~80 MB; a million paths at 200 dates would not fit
  comfortably. This caps the path counts used in Milestone 6.
- The out-of-sample estimator costs 2× the simulation budget.
- Only the *lower* bound (a fixed policy) is implemented. A dual/upper bound
  (Andersen–Broadie) is not, so LSM results are one-sided.

### Unresolved
- None. Two issues found during Milestone 4 (a wrong expected ratio in the
  Bermudan-gap test — the date grid is not a pure doubling — and a null result
  for the foresight bias at a single operating point) were fixed by correcting
  the test and by redesigning the study as a sweep over paths and basis size.

### Next
Superseded — see Milestone 5 below.

---

## Milestone 5 — Variance reduction ✅

### Completed
- `src/amopt/variance_reduction.py`: naive, antithetic, European-put control
  variate, and the combination; pair-aware statistics; `paths_for_target_se`;
  `naive_path_level_se` retained so the *wrong* statistic can be measured rather
  than only described.
- `tests/test_variance_reduction.py` — 19 tests including empirical coverage of
  the 95% interval for all four methods. Total suite: 215 passing.
- `experiments/m5_variance_reduction.py` — five result tables, three figures.

### Important decisions
- **Variance-reduction factors are computed per path, not per sampling unit.**
  An antithetic unit consumes two paths; the per-unit comparison would report
  double the true gain.
- **The work-normalised gain (variance × time) is reported alongside the raw
  VRF**, since a technique that halves variance and doubles cost gains nothing.
- **`b` is estimated on the training sample**, which makes the control-variate
  estimator exactly unbiased at no extra cost. The conventional same-sample
  choice is implemented too and measured to differ by `< 4 × 10⁻⁴`.
- **The antithetic error bar is a pair statistic.** Demonstrated in both
  directions on exact payoffs: conservative for a monotone payoff (98.9%
  coverage), and genuinely broken for a non-monotone one (90.6% coverage).

### Numerical results
See `RESULTS.md` §5.

### Known limitations
- The best combined variance reduction is `2.97×` per path. That is a constant
  factor; the `O(N^{-1/2})` exponent is unchanged (measured `−0.505`), so it
  cannot close the gap to a PDE solver at high precision.
- **Variance reduction converts a variance problem into a bias problem.** Once
  the interval narrows, coverage of the true value drops (`0.950 → 0.910` for
  antithetic) because the fixed-policy bias stops being negligible.
- No dual/upper bound (Andersen–Broadie) is implemented, so the Monte Carlo
  results remain one-sided.

### Unresolved
- None. One issue found during Milestone 5: the first "path-level standard
  error" comparison actually computed `correct_SE / √2` rather than the real
  naive formula, because the marginal path variance was not retained. Fixed by
  adding `path_variance` and `pair_correlation` to `VRResult`, which also
  reversed the reported direction of the error (it over-covers, not under-covers,
  for this payoff) and motivated the butterfly experiment that demonstrates the
  dangerous direction.

### Next
Milestone 6: high-resolution reference solution, full convergence and
error-vs-runtime study across both methods.
