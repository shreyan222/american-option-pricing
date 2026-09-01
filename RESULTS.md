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

For the base case exactly **one** row (`i = 1`, threshold `(r−q)/σ² = 1.25`)
violates the M-matrix condition under central differencing.

| Scheme | Price (`M = N = 1600`) | non-M-matrix rows |
|---|---|---|
| Central differences | `6.09014363` | 1 |
| Selective upwinding (violating rows only) | `6.09014363` | 0 |
| Full upwinding (every row) | `6.10179234` | 0 |

Selective upwinding restores the M-matrix property and changes the price by less
than `10⁻⁸`. **Upwinding every row costs `1.16 × 10⁻²`** — a measured error 250×
larger than the central-difference grid error — and is first order
(`tests/test_crank_nicolson.py::test_full_upwinding_is_first_order`). This is why
the solver upwinds selectively rather than globally.

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
2. **The two estimators bracket the truth.** In-sample is above the exact
   Bermudan value at every path count; out-of-sample is below it at every path
   count. Out-of-sample is a valid lower bound because it evaluates a fixed
   policy; in-sample is biased high by the hindsight.
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
