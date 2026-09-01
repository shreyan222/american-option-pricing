# RESULTS

Every number in this file was produced by a script in `experiments/` and is
backed by a CSV in `results/`. Nothing here is quoted from the literature or
estimated by hand.

Base contract throughout: `S₀ = 100, K = 100, T = 1, r = 5%, σ = 20%, q = 0`.

---

## 1. European baseline and the CRR lattice

Source: `experiments/m1_european_baseline.py` →
`results/m1_bs_reference.csv`, `results/m1_binomial_convergence.csv`,
`results/m1_american_convergence.csv`, `results/m1_lattice_parity.csv`.

| Quantity | Value |
|---|---|
| European put, closed form | `5.57352602` |
| European put, CRR `N = 12800` | `5.57336980` |
| Absolute error at `N = 12800` | `1.56 × 10⁻⁴` |
| Fitted log–log convergence slope (put) | `−0.990` |
| Fitted log–log convergence slope (call) | `−0.990` |
| **American put, CRR `N = 12800`** | **`6.09031197`** |
| Early-exercise premium | `0.516786` |

**Convergence order.** The measured slope of `log|error|` against `log N` over
`N ∈ [25, 12800]` is `−0.990` for both the put and the call, consistent with the
first-order convergence expected of CRR.

**Put–call parity holds on the lattice to machine precision.** The residual
`|C − P − (S₀e^{−qT} − Ke^{−rT})|` is at most `3.15 × 10⁻¹¹` over
`N ∈ {50, 200, 800, 3200}`. This is why the European put and call error curves
coincide: the lattice error is carried entirely by the terms parity cancels.

**Odd/even oscillation.** Over `N ∈ [60, 140]` the signed CRR European error
alternates with period two in `N`. Averaging adjacent lattices,
`½(V_N + V_{N+1})`, reduces the mean absolute error by a factor of **15.6×**.

Figures: `figures/m1_binomial_convergence.png`, `figures/m1_american_convergence.png`.

---

## 2. Analytic anchors from the free-boundary formulation

Source: `experiments/m2_analytic_anchors.py` →
`results/m2_perpetual_anchors.csv`, `results/m2_maturity_limit.csv`.

The perpetual American put has a closed form (derivation: `docs/01_formulation.md`
§1.6), which gives an *exact* upper bound on the finite-maturity value and an
exact lower bound on the exercise boundary. Both are used as validation.

| Quantity (base case) | Value |
|---|---|
| `β₋` (negative root of the characteristic quadratic) | `−2.500000` |
| Perpetual boundary `S∞ = K·γ/(1+γ)`, `γ = 2r/σ² = 2.5` | `71.428571` |
| Perpetual put value at `S₀ = 100` | `12.320033` |
| CRR American put, `T = 1` | `6.089990` |
| CRR American put, `T = 200` | `12.318306` (`99.99%` of the perpetual value) |

**Dominance holds in all 10 parameter regimes:** `V_amer(T) ≤ V_perp` for every
regime in `amopt.config.REGIMES`.

**The `r = 0` case is a genuine degeneracy, not a bug.** With `r = 0` we get
`β₋ = 0` and `S∞ = 0`, so the exercise region is empty. But under `Q` with
`r = 0`, `S_t → 0` almost surely, so `(K − S_t)⁺ → K` and the supremum over
stopping times is `K = 100` — approached, never attained. An earlier version
returned `0` here ("never exercise, so worthless"), which broke the dominance
check against the `r = 0` American put of `7.965070`. The limit is now returned
correctly and asserted in `tests/test_perpetual.py`.

**The monotone approach to the perpetual value is only resolvable above the
lattice noise floor.** `V_perp − V_amer(T)` decreases monotonically for
`T ≤ 50`. At `T = 100` and `T = 200` the gap is `1.69 × 10⁻³` and `1.73 × 10⁻³`
while the lattice's own discretisation error, measured as `|V_N − V_{N/2}|`, is
`4.3 × 10⁻⁴` to `1.5 × 10⁻³`. The apparent non-monotonicity at those two
maturities is CRR discretisation error, not a violation of the theory — 2 of 10
maturities are below the noise floor. This is reported rather than trimmed.

Figure: `figures/m2_perpetual_limit.png`.

---

## 3. Crank–Nicolson + PSOR

Source: `experiments/m3_crank_nicolson.py` → `results/m3_*.csv`.

### 3.1 Accuracy against two independent references

| Quantity (base case) | Value |
|---|---|
| CRR benchmark, `N = 40,000` | `6.09035196` |
| CN-PSOR, `M = N = 3200` | `6.09030490` |
| Absolute deviation | `4.71 × 10⁻⁵` |
| `S*(0)` at `M = N = 3200` | `80.87563` |

Across all **10 parameter regimes** (`results/m3_regime_table.csv`, `M = N = 2000`
vs CRR `N = 20,000`) the largest CN-vs-CRR disagreement is
**`2.45 × 10⁻⁴`** (`low_vol`). Two solvers sharing no code agree to that level.

### 3.2 Convergence order — measured, and not what the textbook order says

Refining `M = N` together gives a fitted order of `2.00` (European) and `1.98`
(American), but that measurement is **spatially dominated** and says nothing
about the time discretisation. Refining each axis against a reference computed
on the *same* grid in the other axis (so the other axis's error cancels exactly):

| Axis / scheme | Measured order |
|---|---|
| Space, `ΔS` (CN) | **`2.11`** |
| Time, `Δτ` (Crank–Nicolson, 2 Rannacher steps) | **`1.26`** |
| Time, `Δτ` (fully implicit) | **`1.01`** |

**The American constraint costs Crank–Nicolson its second-order time accuracy.**
The scheme is second order in space and formally second order in time, but the
free boundary is only located to `O(Δτ)`, and the measured temporal order for the
American put is `1.26`, not `2`. Crank–Nicolson still beats fully implicit both in
order (`1.26` vs `1.01`) and in constant (`1.6 × 10⁻³` vs `1.4 × 10⁻²` at `N = 100`).

This correction matters: an earlier version of this study measured the temporal
order against the external CRR benchmark at `M = 4000` and reported `1.18`. That
number was the *spatial* error floor of `≈ 2.5 × 10⁻⁵`, not a convergence rate.

### 3.3 The three LCP solvers agree

`results/m3_solver_agreement.csv`. Maximum deviation between vectorised red–black
PSOR, the literal lexicographic PSOR loop, and the exact `O(M)` Brennan–Schwartz
elimination: **`3.78 × 10⁻¹²`**.

Runtime at `M = N = 400`: PSOR `0.066 s`, Brennan–Schwartz `0.089 s`,
lexicographic PSOR `1.23 s` — red–black ordering buys a **19×** speedup over the
scalar sweep at identical output.

### 3.4 PSOR relaxation parameter

`results/m3_omega_study.csv`. The price is invariant to `ω`: across all `ω` and
all grid sizes the spread is `1.35 × 10⁻⁷`. Only the cost changes, and the
optimum drifts upward with refinement:

| `M = N` | best `ω` |
|---|---|
| 400 | `1.1` |
| 800 | `1.2` |
| 1600 | `1.3` |
| 3200 | `1.4` |

The optimum sits well below 2 because Crank–Nicolson's matrix
`A = I − θΔτ𝓐ₕ` has diagonal `1 + θΔτ(σ²i² + r)`, which is strongly diagonally
dominant at large `i`, so the Jacobi spectral radius is small.

### 3.5 Rannacher start-up

`results/m3_rannacher.csv`. Crank–Nicolson is A-stable but not L-stable, so the
payoff kink rings. At `N = 25` (`M = 800`), two fully implicit start-up steps cut

- the total variation of gamma within ±15 of the strike from `0.46201` to `0.01257` — a **37× reduction**;
- the price error from `3.63 × 10⁻²` to `3.11 × 10⁻³` — a **12× reduction**.

At `N ≥ 100` the effect vanishes and start-up costs a little accuracy
(`6.16 × 10⁻⁴` → `6.28 × 10⁻⁴` at `N = 400`), because the two implicit steps are
themselves only first order. Both directions are in the CSV.

### 3.6 The cell-Péclet violation is real but harmless — when repaired selectively

`results/m3_upwind.csv`. For the base case exactly **one** row (`i = 1`,
threshold `(r−q)/σ² = 1.25`) violates the M-matrix condition under central
differencing.

| Scheme | Price (`M = N = 1600`) | non-M-matrix rows |
|---|---|---|
| Central differences | `6.09014363` | 1 |
| Selective upwinding (violating rows only) | `6.09014363` | 0 |
| Full upwinding (every row) | `6.10179234` | 0 |

Selective upwinding restores the M-matrix property and leaves the price
**bit-identical** to the central-difference value. **Upwinding every row costs
`1.16 × 10⁻²`** — a measured error 250× larger than the central-difference grid
error — and its measured convergence order over `M ∈ [200, 1600]` is `0.943`,
i.e. first order rather than second. This is why the solver upwinds selectively
rather than globally.

### 3.7 The exercise-region test must read the projection, not a threshold

Locating the discrete exercise set by thresholding the early-exercise gap
(`v − g ≤ 10⁻⁸`) reported a spurious boundary of **`S* = 31.8`** in the `r = 0`
regime, where theory says the exercise region is *empty*. The gap is genuinely
`O(10⁻⁹)` deep in the money there because the American and European puts coincide
— not because the constraint binds. The solver now reads the exercise set from
whether the projection actually selected the obstacle (a bitwise `v == g` test,
no tolerance), which returns `S*(0) = 0.6` at `r = 0` — floating-point noise on a
strike of 100, and the correct answer to the precision available.

Figures: `figures/m3_cn_convergence.png`, `figures/m3_psor_efficiency.png`,
`figures/m3_rannacher.png`, `figures/m3_value_and_boundary.png`.

---

## 4. Longstaff–Schwartz least-squares Monte Carlo

Source: `experiments/m4_longstaff_schwartz.py` → `results/m4_*.csv`.

### 4.1 Benchmarking against the *right* target

LSM with `n` exercise dates estimates the `n`-date **Bermudan** value, not the
continuous-exercise American value. The lattice computes the Bermudan value
exactly (`crr(..., bermudan_dates=n)`), which lets the deviation be split:

```
American − LSM  =  (American − Bermudan_n)  +  (Bermudan_n − LSM)
                    exercise-date bias         regression + sampling error
```

Base case, 200,000 paths, 50 exercise dates, cubic polynomial basis, seed `20240901`:

| | value |
|---|---|
| American, CRR `N = 40,000` | `6.090352` |
| American, CN-PSOR `M = N = 3200` | `6.090305` |
| **Bermudan, 50 dates, exact lattice** | **`6.078622`** |
| LSM in-sample | `6.069081 ± 0.016027`, 95% CI `[6.037669, 6.100493]` |
| LSM out-of-sample | `6.088680 ± 0.016028`, 95% CI `[6.057266, 6.120094]` |

Deviation of the out-of-sample estimate from the exact Bermudan value:
`+0.010058` (`+0.63` standard errors) — inside the confidence interval.
Deviation from the American value: `−0.001672`.

**At 50 exercise dates the exercise-date bias (`+0.011730`) is the same size as
the entire regression-plus-sampling error (`−0.010058`).** Quoting only
"deviation from the American benchmark" would have reported `−0.0017` and hidden
two errors of opposite sign that nearly cancel.

### 4.2 The exercise-date bias is `O(1/n)`

`results/m4_bias_decomposition.csv`. Gap from the exact `n`-date Bermudan value
to the American value:

| dates `n` | Bermudan | gap to American |
|---|---|---|
| 5 | `5.981142` | `0.109210` |
| 10 | `6.033621` | `0.056731` |
| 25 | `6.067113` | `0.023239` |
| 50 | `6.078622` | `0.011730` |
| 100 | `6.084454` | `0.005898` |
| 200 | `6.087399` | `0.002953` |

The gap shrinks by exactly the factor by which the date count grows (fitted
log–log slope `−1`, asserted in
`tests/test_lsm.py::test_bermudan_value_increases_with_exercise_dates_towards_the_american`).

### 4.3 Look-ahead bias: measured, not asserted

The classic estimator fits the exercise policy on the same paths it values, so
the exercise decisions carry hindsight. Fitting on one sample and valuing on an
**independent** sample removes it. Sweeping paths and basis size (25 exercise
dates, polynomial basis, exact Bermudan target `6.067113`, 40–400 independent
repetitions per cell):

| paths | degree | in-sample | out-of-sample | foresight bias (in − out) |
|---|---|---|---|---|
| 500 | 3 | `6.32929` | `5.95746` | `+0.37183 ± 0.02330` |
| 500 | 10 | `6.61295` | `5.85629` | **`+0.75666 ± 0.02324`** |
| 2,000 | 3 | `6.13379` | `6.00775` | `+0.12604 ± 0.01250` |
| 2,000 | 10 | `6.22896` | `5.97274` | `+0.25622 ± 0.01260` |
| 10,000 | 3 | `6.08147` | `6.06079` | `+0.02069 ± 0.00825` |
| 10,000 | 10 | `6.10236` | `6.04970` | `+0.05266 ± 0.00780` |
| 50,000 | 3 | `6.06814` | `6.06187` | `+0.00626 ± 0.00447` |
| 50,000 | 10 | `6.07460` | `6.06169` | `+0.01291 ± 0.00437` |
| 200,000 | 3 | `6.06342` | `6.06080` | `+0.00262 ± 0.00397` |
| 200,000 | 10 | `6.06736` | `6.06313` | `+0.00423 ± 0.00386` |

Three findings:

1. **The bias is real and large when the sample is small relative to the basis.**
   At 500 paths with an 11-function basis the in-sample estimate is `6.613`, i.e.
   **9.0% above** the true Bermudan value — and above the *American* value
   `6.0904`, so it is not even a valid upper-bound-free estimate.
2. **The two estimators bracket the truth — with one instructive exception.**
   Out-of-sample is below the exact Bermudan value at every path count, as a
   fixed-policy lower bound must be. In-sample is above it in **9 of the 10
   cells**; at 200,000 paths with a cubic basis it falls `0.0037` *below*,
   because by then the foresight bias (`+0.0026`) has decayed past the
   policy-suboptimality bias pushing the other way. In-sample is biased high by
   hindsight, but that is not the only bias acting on it, and at large samples
   it stops being the dominant one.
3. **The bias decays roughly like `1/n_paths` and grows with the number of basis
   functions**, consistent with an overfitting bias of order (parameters/paths).
   It is statistically undetectable at 200,000 paths with a cubic basis
   (`+0.0026 ± 0.0040`), which is the operating point used elsewhere in this
   repository.

**The reported standard errors are honest.** Comparing each run's self-reported
standard error with the realised standard deviation across repetitions, the ratio
lies in `[0.98, 1.18]` across all ten cells for both estimators. The in-sample
problem is bias, not an understated error bar.

### 4.4 In-the-money path filtering is not a refinement — it is essential

`results/m4_itm_filter.csv`. Deviation from the exact 50-date Bermudan value,
200,000 paths, out-of-sample:

| degree | regress on ITM paths only | regress on all paths |
|---|---|---|
| 2 | `−0.014296` | `−0.338926` |
| 3 | `+0.010058` | `−0.155557` |
| 4 | `+0.010321` | `−0.160226` |
| 6 | `+0.008582` | `−0.069535` |

Standard errors are `≈ 0.016` throughout, so the all-paths deviations are
`4–21` standard errors from the benchmark while the ITM-only deviations are
under one. Regressing on all paths forces the polynomial to fit the flat
out-of-the-money region where the exercise decision is never live, and it buys
that fit by degrading the continuation estimate exactly where the decision
matters. **Filtering to in-the-money paths is worth up to `0.34` in price — 33×
the sampling error.**

### 4.5 Basis family and degree

`results/m4_basis_study.csv`. Accuracy saturates at degree 3: beyond it every
basis sits within one standard error (`≈ 0.016`) of the Bermudan benchmark, and
the best cell (`poly`, degree 6, `|dev| = 8.58 × 10⁻³`) is not distinguishable
from the others. Conditioning does *not* saturate — condition number of the
in-the-money design matrix at `t = T/2`, degree 10:

| basis | condition number |
|---|---|
| Chebyshev | `6.2 × 10¹⁰` |
| monomial (`poly`) | `4.7 × 10¹²` |
| weighted Laguerre | `3.2 × 10¹⁶` |

Laguerre at degree 10 is at the edge of double precision. The practical
recommendation from these numbers is a **low-degree basis** — the accuracy is
already there at degree 3 — and Chebyshev if a high degree is unavoidable.

Figures: `figures/m4_bias_decomposition.png`,
`figures/m4_in_vs_out_of_sample.png`, `figures/m4_basis_study.png`.

---

## 5. Variance reduction

Source: `experiments/m5_variance_reduction.py` → `results/m5_*.csv`.
All four estimators price the same object — the 50-date Bermudan put, exact value
`6.078622` — under a policy fitted on an independent training sample, so
differences in interval width are attributable to the technique alone.

### 5.1 Headline comparison, 200,000 valuation paths

| method | price | SE | 95% CI width | variance per path | VRF | work-normalised gain | paths for the naive SE |
|---|---|---|---|---|---|---|---|
| naive | `6.08868` | `0.016028` | `0.06283` | `51.3772` | `1.00` | `1.0` | `200,000` |
| antithetic | `6.08613` | `0.009551` | `0.03744` | `18.2447` | `2.82` | `3.1` | `71,022` |
| control variate | `6.09755` | `0.010660` | `0.04179` | `22.7259` | `2.26` | `2.0` | `88,467` |
| antithetic + control | `6.08255` | `0.009294` | `0.03643` | `17.2774` | **`2.97`** | `2.6` | `67,257` |

The variance-reduction factor is computed **per path**, not per sampling unit —
an antithetic unit consumes two paths, and comparing per unit would double-count
the gain. The work-normalised gain multiplies by the runtime ratio: antithetic
sampling scores *above* its raw VRF (`3.1` against `2.82`) because it draws half
as many normals, while the control variate scores below its own (`2.0` against
`2.26`) because computing and regressing the control costs real time.

> **On timings.** All wall-clock figures are single-threaded, single-machine, single-run measurements. Re-running them moves the absolute times by 5–25%; the *ratios and the fitted power-law slopes are the stable quantities* and are reproducible to two decimal places. Timings are quoted to two significant figures for that reason.
> The VRFs, standard errors and CI widths above are deterministic and reproduce
> bit-identically.

**Antithetic sampling reaches the naive method's accuracy on 71,022 paths
instead of 200,000 — a 2.8× reduction in simulation budget.** Combining both
techniques reaches it on 67,257.

### 5.2 The control variate's power is exactly `1/(1 − ρ²)`

`results/m5_regimes.csv`. Measured VRF against the theoretical value across all
ten regimes, 100,000 paths each:

| regime | `ρ` | VRF measured | `1/(1−ρ²)` | antithetic VRF |
|---|---|---|---|---|
| base | `0.7481` | `2.27` | `2.27` | `2.81` |
| itm | `0.4628` | `1.27` | `1.27` | `1.47` |
| otm | `0.8279` | `3.18` | `3.18` | `1.25` |
| low_vol | `0.6199` | `1.62` | `1.62` | `2.15` |
| high_vol | `0.8275` | `3.17` | `3.17` | `4.42` |
| short_maturity | `0.8058` | `2.85` | `2.85` | `2.73` |
| long_maturity | `0.6665` | `1.80` | `1.80` | `2.73` |
| high_rate | `0.6285` | `1.65` | `1.65` | `2.31` |
| zero_rate | `0.9021` | `5.37` | `5.37` | `3.31` |
| dividend | `0.8365` | `3.33` | `3.33` | `3.55` |

The measured factor matches theory to two decimal places in **every** regime —
the dashed curve in `figures/m5_efficiency.png` is theory, not a fit.

The two techniques are complementary rather than redundant, and which one wins
depends on the regime: out-of-the-money the control variate gives `3.18` and
antithetic only `1.25`; at high volatility antithetic gives `4.42` against
`3.17`. `ρ` falls as the early-exercise premium grows (lowest at `itm`, `0.463`,
where the American cash flow most often stops early and stops depending on `S_T`
at all), which is exactly when the European control has least to say.

### 5.3 Monte Carlo error is empirically `O(N^{-1/2})`

`results/m5_scaling.csv`, `N` from 5,000 to 400,000. Fitted log–log slopes of the
standard error against path count:

| method | slope |
|---|---|
| naive | `−0.5012` |
| antithetic | `−0.5062` |
| control variate | `−0.5047` |
| antithetic + control | `−0.5051` |

All four are within `0.7%` of the theoretical `−1/2`. Variance reduction moves
the *intercept*, never the exponent — which is precisely why it cannot rescue
Monte Carlo from being the wrong tool when high precision is needed (Milestone 6).

### 5.4 Antithetic pairs are dependent, and the standard error must say so

The correct standard error uses **pair means** as the unit of independence. Two
exact-payoff experiments (3,000 repetitions, 4,000 paths each, no regression, so
the true value is known independently) show the mistake runs in both directions:

| payoff | monotone in `S_T`? | within-pair `ρ` | naive/correct SE | coverage, pair SE | coverage, path-level SE |
|---|---|---|---|---|---|
| European put | yes | `−0.4145` | `1.307` | `0.9510` | `0.9890` |
| butterfly | no | `+0.4282` | `0.837` | `0.9523` | **`0.9060`** |

- For a payoff **monotone** in the driving noise — the case antithetic sampling
  is designed for — the pair correlation is negative and the naive path-level
  formula is `1.31×` too wide. It over-covers at `98.9%`, and in doing so throws
  away the whole variance reduction.
- For a **non-monotone** payoff both legs of a pair are small together whenever
  `|Z|` is large, the pair correlation turns *positive*, and the naive interval
  is `0.84×` too narrow and covers only **`90.6%`** instead of 95%.

In the LSM antithetic runs the within-pair correlation is `−0.639`, the naive
path-level SE is `0.0508` against the correct `0.0305`, and its interval covers
`99.7%` of the time.

### 5.5 Coverage of the LSM intervals

`results/m5_coverage.csv`, 300 repetitions at 20,000 paths:

| method | covers its own mean | covers the exact Bermudan value | bias |
|---|---|---|---|
| naive | `0.950` | `0.950` | `−0.00650` |
| antithetic | `0.943` | `0.910` | `−0.00978` |
| control variate | `0.933` | `0.927` | `−0.00891` |
| antithetic + control | `0.940` | `0.923` | `−0.01026` |

Coverage of the estimator's own mean is within the binomial noise band of 95% for
every method: the standard errors are sound. Coverage of the *true* value is
lower for the variance-reduced methods — not because their intervals are wrong,
but because **once the interval narrows, the residual bias of the fixed exercise
policy stops being negligible relative to it.** Variance reduction converts a
variance problem into a bias problem; further gains require a better exercise
policy or a dual upper bound, not more paths.

### 5.6 Where the control coefficient is estimated barely matters

`results/m5_beta_source.csv`. Estimating `b` on the training sample (exactly
unbiased) versus on the valued sample (the conventional choice, biased at
`O(1/n)`):

| paths | `b` from training | `b` from sample | deviation, training | deviation, sample |
|---|---|---|---|---|
| 2,000 | `0.6216` | `0.6273` | `+0.07369` | `+0.07378` |
| 10,000 | `0.6216` | `0.6247` | `+0.01498` | `+0.01534` |
| 50,000 | `0.6216` | `0.6230` | `−0.01872` | `−0.01867` |
| 200,000 | `0.6216` | `0.6233` | `+0.01075` | `+0.01076` |

The difference is under `4 × 10⁻⁴` even at 2,000 paths — far below the sampling
error. Using the suboptimal `b = 1` instead of `b* ≈ 0.62` is the choice that
actually costs: deviation `+0.07968` versus `+0.07369` at 2,000 paths, with a
materially larger variance.

Figures: `figures/m5_error_vs_paths.png`, `figures/m5_efficiency.png`,
`figures/m5_coverage.png`.

---

## 6. Convergence and computational efficiency

Source: `experiments/m6_convergence.py` → `results/m6_*.csv`.

### 6.1 The reference solution

Neither solver is accurate enough at any affordable resolution to benchmark
itself, so the reference is built by Richardson-extrapolating **two methods that
share no code** and quoting their disagreement as the uncertainty:

| construction | value |
|---|---|
| CRR lattice, `2V_{2N} − V_N` at `N = 40,000/80,000` | `6.090370659` |
| CN, second-order space Richardson + a time correction at the measured order `p = 1.265` | `6.090370566` |
| **Reference (mean)** | **`6.090370613`** |
| **Uncertainty (their spread)** | **`9.36 × 10⁻⁸`** |

Every error in this section is measured against `6.090370613`.

### 6.2 Crank–Nicolson: grid resolution

`results/m6_cn_grid.csv`. Measured orders, fitting only points more than `10×`
above the other axis's irreducible error floor:

| axis | order | points used / dropped into the floor |
|---|---|---|
| space `ΔS` | **`1.995`** | 4 / 2 |
| time `Δτ` | **`1.223`** | 3 / 3 |

The floors are `5.35 × 10⁻⁵` (time axis fixed at `N = 6400`) and
`2.05 × 10⁻⁵` (space axis fixed at `M = 6400`). Fitting through them instead
gave `1.957` and `1.121`, i.e. an understated order in both cases — the same trap
identified in §3.2, now handled explicitly. The temporal order of `1.223` is
consistent with the `1.265` measured independently by self-convergence in
Milestone 3.

### 6.3 PSOR tolerance: tighter is not better past a point

`results/m6_cn_tolerance.csv`, `ω = 1.4`:

| grid | discretisation error floor | tolerance at which the floor is reached | cost of tightening to `10⁻¹²` |
|---|---|---|---|
| `M = N = 800` | `5.37 × 10⁻⁴` | `10⁻⁵` | `3.0×` the PSOR sweeps |
| `M = N = 3200` | `4.61 × 10⁻⁵` | `10⁻⁶` | `3.1×` the PSOR sweeps |

The linear-complementarity solve only has to be accurate relative to the
discretisation error. Beyond `tol ≈ 10⁻⁶` the price does not move and the sweeps
triple. This is a free `3×` speedup that a default of `tol = 10⁻¹²` throws away.

### 6.4 Domain truncation is negligible beyond `S_max = 2K`

`results/m6_cn_domain.csv`, with `M` scaled so that `ΔS = 0.125` stays fixed:

| `S_max` | `1.5K` | `2K` | `2.5K` | `3K` | `4K` | `6K` | `8K` | `12K` |
|---|---|---|---|---|---|---|---|---|
| error | `1.621e−04` | `6.540e−05` | `6.540e−05` | `6.540e−05` | `6.540e−05` | `6.540e−05` | `6.540e−05` | `6.540e−05` |

Truncation is only visible at `S_max = 1.5K`; from `2K` onward the error is
identical to six digits and is pure discretisation error. The default `4K` is
safely inside the flat region.

**This measurement is only meaningful because `ΔS` was held fixed.** Holding `M`
fixed instead — the obvious thing to do — makes a larger domain mean a coarser
grid, and the measured error then *rises* monotonically from `3.3 × 10⁻⁵` at `2K`
to `4.2 × 10⁻⁴` at `12K`, which says nothing about truncation at all. The first
version of this study made exactly that mistake.

### 6.5 Monte Carlo: `O(N^{-1/2})`, confirmed — and a bias floor

`results/m6_lsm_paths.csv`, RMSE over `8` seeds per point, 50 exercise dates:

| method | fitted slope of the seed-to-seed s.d. against paths |
|---|---|
| naive | `−0.4806` |
| antithetic + control | `−0.4948` |

Both are consistent with the theoretical `−1/2`. But the **RMSE against the
reference does not follow the sampling error down**: at 400,000 paths the
sampling s.d. has fallen to well under `10⁻²` while the RMSE is `0.01474`,
because the bias is `−0.01361` and does not move.

`results/m6_lsm_dates.csv` shows why adding exercise dates does not fix it
(100,000 paths):

| dates | RMSE | exercise-date bias | regression error | runtime |
|---|---|---|---|---|
| 5 | `0.10696` | `+0.10923` | `−0.00352` | `0.03 s` |
| 10 | `0.06143` | `+0.05675` | `+0.00385` | `0.05 s` |
| 25 | `0.02841` | `+0.02326` | `+0.00346` | `0.13 s` |
| 50 | `0.02286` | `+0.01175` | `+0.00869` | `0.27 s` |
| 100 | `0.02010` | `+0.00592` | `+0.00791` | `0.53 s` |
| 200 | **`0.01739`** | `+0.00297` | `+0.01216` | `1.06 s` |
| 400 | `0.02372` | `+0.00149` | `+0.01748` | `2.29 s` |

**The two error terms trade off against each other.** More exercise dates shrink
the Bermudan bias (`0.109 → 0.0015`, cleanly `O(1/n)`) but the accumulated
regression/policy error *grows* with the number of decisions (`−0.004 → +0.017`).
The total is minimised around 200 dates and gets worse after that. This is the
central limitation of the method, and it is not a sampling problem.

Pushing to the memory ceiling (`results/m6_lsm_large.csv`) confirms it:

| configuration | bias | s.d. | RMSE | runtime |
|---|---|---|---|---|
| 200,000 paths × 200 dates | `−0.01496` | `0.00514` | `0.01561` | `1.47 s` |
| 200,000 paths × 400 dates | `−0.01456` | `0.00792` | `0.01609` | `2.51 s` |
| 400,000 paths × 200 dates | `−0.01601` | `0.00395` | `0.01637` | `1.85 s` |

The sampling s.d. falls with paths exactly as it should; the bias sits at
`≈ −0.015` and refuses to move. **The best RMSE anywhere in the entire sweep is
`1.44 × 10⁻²`** (80,000 paths × 100 dates), about `0.24%` of the price.

### 6.6 Monomial and Chebyshev bases are the same space — and the data proves it

`results/m6_lsm_basis.csv`. Monomials and Chebyshev polynomials of the same
degree span the *same* function space, so the least-squares fit, every exercise
decision, and the price must be identical in exact arithmetic. Measured
`|price_poly − price_chebyshev|`:

| degree | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| difference | `0` | `0` | `0` | `0` | `3.4e−05` | `2.1e−04` | `2.1e−03` | `2.0e−03` |

Bitwise identical to degree 4, then diverging — the divergence is **pure
conditioning**, nothing else. Weighted Laguerre carries an `e^{-x/2}` factor and
so spans a genuinely different space; it differs at every degree.

RMSE is minimised around degree 3–4 for every basis and gets slowly worse
afterwards, so a low-degree basis is the right default.

### 6.7 Error against wall-clock cost — the headline comparison

`results/m6_frontier.csv`. Fitted power laws for error against runtime:

| method | error scaling | time to `10⁻³` | time to `10⁻⁴` | best achieved |
|---|---|---|---|---|
| CRR lattice | `t^{−0.55}` | `0.016 s` | `0.25 s` | `9.3 × 10⁻⁶` at `13 s` |
| CN + PSOR | `t^{−1.22}` | `0.21 s` | `1.8 s` | `2.1 × 10⁻⁵` at `10 s` |
| LSM (antithetic + control) | floors | **never** | **never** | `1.4 × 10⁻²` at `0.44 s` |

> **On timings.** All wall-clock figures are single-threaded, single-machine, single-run measurements. Re-running them moves the absolute times by 5–25%; the **ratios and the fitted power-law slopes are the stable quantities** and are reproducible to two decimal places. Timings are quoted to two significant figures for that reason.

Three findings:

1. **Monte Carlo is not competitive for this problem, by three orders of
   magnitude.** LSM never reaches `10⁻²` at any configuration tested, while
   Crank–Nicolson reaches it in `0.078 s` and the lattice in `0.015 s`. The
   `O(N^{-1/2})` sampling law is not the binding constraint — the policy bias is.
2. **The lattice beats the PDE solver throughout the tested range.** CRR's
   `t^{−0.55}` follows directly from `O(1/N)` accuracy at `O(N²)` cost. CN's
   error falls faster with cost (`t^{−1.22}`), so the fitted power laws cross at
   `t ≈ 11 s`, `error ≈ 9.5 × 10⁻⁶` — right at the edge of the measured range.
   For a single-asset vanilla American put, the lattice is the efficient choice
   at any accuracy a practitioner would ask for.
3. **What the PDE buys is not speed.** It returns the whole value surface, the
   free boundary at every time level (the lattice cone misses it near `t = 0`,
   §1), stable Greeks from the same grid, and a genuinely second-order spatial
   discretisation. Those are the reasons to pay `10×`.

Figures: `figures/m6_convergence.png`, `figures/m6_error_vs_runtime.png`,
`figures/m6_psor_tolerance.png`.

---

## 7. The early-exercise boundary

Source: `experiments/m7_exercise_boundary.py` → `results/m7_*.csv`.
Predictions **P1–P7** were written down with their reasons in
[`docs/03_exercise_boundary.md`](docs/03_exercise_boundary.md) *before* this study
was run, together with a table of what each failure would mean. Every one is
tested in `tests/test_boundary.py`.

Base case: `S*(0) = 80.87563`, between the perpetual floor `S∞ = 71.42857` and
the strike `100`.

### P1 — Monotone in `t` ✅

The largest decrease anywhere along `S*(t)` is `−4.33 × 10⁻²`, i.e. `0.35` of a
grid cell (`ΔS = 0.125`). Within grid noise; no systematic violation.

### P2 — Terminal boundary `min(K, rK/q)` ✅

The sharpest prediction here: with `K = 100`, `r = 5%`, `q = 10%` the boundary
just before maturity should be `50`, not `100`.

| `q` | predicted `min(K, rK/q)` | measured just before `T` | relative deviation |
|---|---|---|---|
| `0.00` | `100.000` | `98.741` | `1.26%` |
| `0.02` | `100.000` | `98.618` | `1.38%` |
| `0.04` | `100.000` | `98.361` | `1.64%` |
| `0.06` | `83.333` | `83.097` | **`0.28%`** |
| `0.10` | `50.000` | `49.830` | **`0.34%`** |
| `0.15` | `33.333` | `33.186` | **`0.44%`** |

**The two cases converge at different rates, and P3 says why.** When `q > r` the
limit `rK/q` sits away from the payoff kink and is hit to within `0.5%`. When
`q ≤ r` the limit is `K` and the boundary approaches it along the square-root-log
law, so at the last resolved time level the gap is `≈ 1.3` on a strike of 100 —
which is exactly `Kσ√(Δτ ln(1/Δτ))`. The test asserts the predicted *scale* in
that case rather than demanding an agreement the asymptotics forbid.

**The boundary is genuinely discontinuous at maturity when `q > r`:** the solver
reports `S*(T) = 100` (at `τ = 0` every in-the-money put is exercised) and
`S*(T⁻) = 49.83` for `q = 10%`. That jump is a real feature of the problem.

### P3 — The square-root-log law `K − S* ~ Kσ√(τ ln(1/τ))` ✅

| `τ` | `K − S*` | predicted | ratio | expected band `1 ± 1/ln(1/τ)` |
|---|---|---|---|---|
| `3.1e−4` | `1.15302` | `1.00442` | `1.1479` | `± 0.124` |
| `9.4e−4` | `1.55342` | `1.61698` | `0.9607` | `± 0.143` |
| `3.0e−3` | `2.55916` | `2.62884` | `0.9735` | `± 0.172` |
| `1.0e−2` | `4.20549` | `4.29193` | `0.9799` | `± 0.217` |
| `3.0e−2` | `6.42445` | `6.48681` | `0.9904` | `± 0.285` |
| `1.0e−1` | `9.83177` | `9.59705` | `1.0245` | `± 0.434` |

Regressing `K − S*` on `√(τ ln(1/τ))` through the origin gives a slope of
**`21.578`** against the predicted `Kσ = 20` — `7.9%` high, inside the `15%`
band the documentation committed to in advance. This tests the *functional form*:
a plain `√τ` law would not fit.

The single point at `τ = 3.1 × 10⁻⁴` sits just outside its band (`1.148` vs
`1.124`). It is two time steps from maturity at `N = 6400`, where the boundary is
moving fastest and is least resolved in time; it is reported rather than dropped.

The law implies **infinite slope at maturity**. Testing that directly: the
last-step slope of `S*` grows monotonically with time refinement
(`N = 400 → 3200`) rather than converging, which a finite terminal slope would
not do.

### P4 — Volatility ✅ and P5 — Interest rate ✅

| `σ` | `10%` | `15%` | `20%` | `30%` | `40%` | `60%` |
|---|---|---|---|---|---|---|
| `S*(0)` | `92.754` | `86.956` | `80.876` | `69.125` | `58.531` | `41.477` |

| `r` | `0%` | `1%` | `2%` | `5%` | `8%` | `12%` |
|---|---|---|---|---|---|---|
| `S*(0)` | `0.496` | `69.730` | `74.232` | `80.876` | `84.547` | `87.701` |

Both strictly monotone in the predicted direction, and in both figures the
measured `S*(0)` tracks the closed-form perpetual boundary
`S∞ = Kγ/(1+γ)`, `γ = 2r/σ²`, staying strictly above it everywhere.

**The financial reading.** Volatility is the raw material of optionality: the
right to wait is worth more when the stock can travel further, so the holder
demands a deeper in-the-money price before surrendering it. The interest rate is
the entire reason to exercise a put early — exercising converts the option into
cash `K` earning `r`. At `r = 0` that incentive vanishes completely and the
measured `S*(0)` collapses to `0.496` on a strike of `100`, i.e. **no exercise
region at all**, which is the theoretical answer to within floating-point noise
(§3.7).

### P6 — Maturity ✅

| `T` | `0.25` | `0.5` | `1` | `2` | `5` | `10` | `25` |
|---|---|---|---|---|---|---|---|
| `S*(0)` | `86.838` | `83.939` | `80.876` | `77.893` | `74.526` | `72.763` | `71.672` |

Strictly decreasing, and at `T = 25` the boundary is `1.0034 ×` the perpetual
limit `71.42857` — converging to the closed form from above, as required.

### P7 — Scale invariance ✅

`S*(t)/K` is unchanged across `K ∈ {1, 50, 100, 500}` to `2 × 10⁻³` relative.
On an identical grid the boundary is **bitwise identical** for `S₀ = 60` and
`S₀ = 140`, confirming that `S₀` enters nowhere in the free-boundary problem.
With the default domain `S_max = 4·max(K, S₀)` the mesh moves with `S₀` and the
boundary shifts by `0.058` — less than one grid spacing, i.e. the cell-alignment
noise measured next.

### Boundary accuracy: better than the grid, but not monotone in it

`results/m7_boundary_accuracy.csv`, refining `M` at `N = 6400`, deviations from
the `M = 12800` value:

| `M` | `ΔS` | `S*(0)` refined | deviation | raw estimate | raw deviation |
|---|---|---|---|---|---|
| 400 | `1.0000` | `80.88469` | `9.5e−03` | `81.0000` | `1.25e−01` |
| 800 | `0.5000` | `80.89040` | `1.5e−02` | `81.0000` | `1.25e−01` |
| 1600 | `0.2500` | `80.97198` | **`9.7e−02`** | `80.7500` | `1.25e−01` |
| 3200 | `0.1250` | `80.87560` | `4.4e−04` | `80.8750` | `1.6e−04` |
| 6400 | `0.0625` | `80.87527` | `1.0e−04` | `80.8750` | `1.6e−04` |

The smooth-pasting refinement usually buys about an order of magnitude over the
raw staircase, **but the boundary error is not monotone in `ΔS`**: at `M = 1600`
it is `9.7 × 10⁻²`, worse than at `M = 400`. Where `S*` happens to fall inside a
cell matters, because the discrete exercise set overshoots the true boundary by
up to one cell and the local quadratic model then extrapolates from an unusually
small gap. This is reported rather than smoothed away; it is why the tests that
compare boundaries across grids use a tolerance of one grid spacing.

### PDE versus lattice

`results/m7_pde_vs_lattice.csv`. The lattice boundary is **undefined at 58 of
3001 time levels** (`t ≤ 0.019`), because the cone `S₀e^{±iσ√Δt}` rooted at a
single spot does not reach the exercise region near `t = 0`. Where both are
defined they agree to a mean absolute difference of `0.204`, about half the
lattice node spacing at `N = 3000`. The PDE grid is fixed in `S` and resolves
`S*` over the whole time axis — a structural advantage of the PDE approach that
has nothing to do with accuracy or speed.

Figures: `figures/m7_boundary_families.png`,
`figures/m7_boundary_sensitivity.png`, `figures/m7_boundary_asymptotics.png`.
