# Pricing the American Put: Crank–Nicolson/PSOR against Longstaff–Schwartz

**A numerical comparison from first principles**

---

## Abstract

We compare two structurally different numerical methods for the American put
under Black–Scholes: Crank–Nicolson finite differences with projected SOR, which
solves the variational inequality directly, and Longstaff–Schwartz least-squares
Monte Carlo, which approximates the optimal stopping rule by regression. A
Cox–Ross–Rubinstein lattice, sharing no code with either, serves as an
independent third method. All algorithms are implemented from scratch.

A reference solution accurate to `9.4 × 10⁻⁸` is constructed by Richardson-
extrapolating two independent methods and quoting their disagreement as the
uncertainty. Against it we find:

1. **Crank–Nicolson is second order in space (`1.995` measured) but only
   `1.223`–`1.265` order in time for the American problem**, against `1.01` for
   the fully implicit scheme. The free boundary is located to `O(Δτ)` and that
   pollutes the value; the loss is a property of the free-boundary problem, not
   of the implementation.
2. **Monte Carlo loses the accuracy comparison by three orders of magnitude, and
   not because of sampling noise.** The seed-to-seed standard deviation falls as
   `N^{−0.49}`, exactly as theory requires, but the RMSE floors at
   `1.44 × 10⁻²` because the fitted exercise policy is biased — and that bias
   does not respond to 400,000 paths, 400 exercise dates, or a degree-16 basis.
3. **The two Monte Carlo error sources trade off against each other.** Adding
   exercise dates shrinks the Bermudan bias as `O(1/n)` (`0.109 → 0.0015`) while
   the accumulated regression error grows (`−0.004 → +0.017`); the total is
   minimised near 200 dates and worsens beyond.
4. **Variance reduction moves the intercept, never the exponent.** Antithetic
   variates and a European-put control variate give `2.97×` combined, with the
   control variate's power matching the theoretical `1/(1−ρ²)` to two decimal
   places in all ten parameter regimes — but all four estimators keep the same
   `−0.505` slope, and narrowing the interval merely exposes the bias.
5. **The free boundary is recovered, not imposed**, and satisfies seven
   predictions registered in advance, including the sharp claim that with
   `q = 10% > r = 5%` the boundary just before maturity is `min(K, rK/q) = 50`
   rather than the strike (measured `49.830`).

Every number in this report is produced by a script in `experiments/` and stored
in `results/`.

---

## 1. Introduction

An American option can be exercised at any time up to maturity. Its holder
therefore solves an optimal stopping problem, and its value is

$$
V(S,t) = \sup_{\tau \in \mathcal{T}_{t,T}}
\mathbb{E}^{\mathbb{Q}}\!\left[ e^{-r(\tau-t)} (K - S_\tau)^+ \,\middle|\, S_t = S \right].
$$

There is no closed form. The interesting question is not what the price is — a
lattice settles that to five decimal places in under a second — but **which
numerical approach is actually better, on what axis, and why**. That is the
question this study answers.

Two approaches are compared because they attack the problem from opposite
directions:

- **Crank–Nicolson + PSOR** converts the optimal-stopping problem into a
  deterministic variational inequality on a grid and solves it by iterating a
  projected relaxation. It returns the entire value surface and the free
  boundary, and it scales badly in dimension.
- **Longstaff–Schwartz** stays probabilistic, estimating the conditional
  continuation value by least-squares regression on simulated paths. It returns
  a single price with a confidence interval, and it scales well in dimension.

The comparison covers pricing accuracy, convergence order, runtime, estimator
variance, numerical stability, and robustness across ten parameter regimes.

**Constraint.** No option-pricing library is used. NumPy, SciPy, pandas and
matplotlib provide generic linear algebra, statistics, dataframes and plotting;
every pricing algorithm — the lattice, the finite-difference operator, the PSOR
sweep, the Brennan–Schwartz elimination, the path simulation, the regression
backward induction — is written in this repository.

**Base contract** throughout: `S₀ = K = 100`, `T = 1`, `r = 5%`, `σ = 20%`,
`q = 0`. Ten further regimes (in and out of the money, low and high volatility,
short and long maturity, high and zero rate, with dividends) are used for
robustness.

---

## 2. Mathematical formulation

Full derivation: [`docs/01_formulation.md`](../docs/01_formulation.md).

Under the risk-neutral measure, `dS = (r−q)S dt + σS dW`. Optimal-stopping theory
says the discounted value process is the **Snell envelope** of the discounted
payoff: the smallest supermartingale dominating it. Two consequences follow
immediately. Dominance gives `V ≥ g`. Supermartingality gives non-positive drift,
so with

$$
\mathcal{L}V = V_t + \tfrac12\sigma^2S^2V_{SS} + (r-q)SV_S - rV
$$

we get `𝓛V ≤ 0` everywhere. On the continuation region the envelope is a
martingale, so `𝓛V = 0` there. Collecting these gives the **linear
complementarity problem**

$$
\mathcal{L}V \le 0, \qquad V - g \ge 0, \qquad (\mathcal{L}V)(V-g) = 0,
$$

equivalently `max{𝓛V, g − V} = 0`. The complementarity condition encodes the
exercise/continuation dichotomy without requiring the regions to be known — which
is the point, since the boundary separating them is itself unknown.

Boundary data: `V(S,T) = (K−S)⁺`; `V(0,t) = K` for the American put (zero is
absorbing, so exercise immediately) as against `Ke^{−r(T−t)}` for the European —
**the single most common sign error in American PDE code**, so the solver takes
this from the exercise style rather than hard-coding it; and `V → 0` as `S → ∞`.

Along the free boundary `S*(t)`, value matching `V(S*,t) = K − S*` and smooth
pasting `V_S(S*,t) = −1` hold. Smooth pasting is what selects `S*` from the
one-parameter family satisfying value matching alone.

**An exact anchor.** With infinite maturity the problem is time-homogeneous and
reduces to an Euler ODE with solutions `S^β`. Boundedness selects the negative
root, and imposing value matching and smooth pasting gives

$$
S_\infty = K\frac{\beta_-}{\beta_- - 1},
\qquad
V(S) = (K - S_\infty)\left(\frac{S}{S_\infty}\right)^{\beta_-},
$$

with `β₋ = −2r/σ²` when `q = 0`. For the base case `S∞ = 71.428571` and
`V^perp(100) = 12.320033`. This is not used to price anything; it supplies two
hard checks — `V^perp ≥ V^amer(T)` for all `T`, and `S∞ < S*(0) < K` — that hold
in all ten regimes.

The `r = 0` case is a genuine degeneracy worth recording. There `β₋ = 0`, so
`S∞ = 0` and the exercise region is empty; but `S_t → 0` almost surely under `Q`,
so `(K−S_t)⁺ → K` and the supremum over stopping times is `K`, approached and
never attained. An earlier version of this work returned `0` here (reading
"never exercise" as "worthless"), which broke the dominance check against the
`r = 0` American put value of `7.965070`.

---

## 3. European benchmark

The Black–Scholes formulas and the full first- and second-order Greek set are
implemented with correct `T → 0` and `σ → 0` limits (the discounted forward
intrinsic, not `nan`). For the base case the European put is **`5.57352602`**.

The analytic layer is validated by identity rather than by regression: put–call
parity to `10⁻¹⁰`, no-arbitrage bounds, convexity in the strike, monotonicity in
`σ` and `S`, the `σ → 0` and `T → 0` limits, the deep-in and deep-out
asymptotics, and every Greek against central finite differences of the price
function. These are the tests that would fail under a plausible sign error and
that depend on nothing this repository computed.

---

## 4. The binomial lattice

Cox–Ross–Rubinstein with `u = e^{σ√Δt}`, `d = 1/u`, and `p` fixed by the
martingale condition. Node prices are computed as `S₀exp((2j−i)σ√Δt)` rather than
`u^j d^{i−j}` to avoid overflow at large `N`. The lattice **raises rather than
clamping** when the risk-neutral probability leaves `[0,1]`: clamping would hide
an arbitrageable lattice and silently bias the price.

**Convergence.** The European lattice price converges to Black–Scholes with a
fitted log–log slope of `−0.990` over `N ∈ [25, 12800]`, reaching an absolute
error of `1.56 × 10⁻⁴` at `N = 12800` — first order, as expected.

**Put–call parity holds on the lattice to `3.15 × 10⁻¹¹`.** This explains an
observation that at first looks like a plotting bug: the European put and call
error curves coincide exactly, because the lattice error lives entirely in the
terms parity cancels.

**Odd/even oscillation.** The signed error alternates with period two in `N`,
because the strike sits differently relative to the nodes for odd and even step
counts. Averaging adjacent lattices reduces the mean absolute error over
`N ∈ [60,140]` by **`15.6×`**.

**Bermudan capability.** The lattice can restrict exercise to `n` equally spaced
dates. This turns out to be essential in §6: it produces the *exact* target that
Longstaff–Schwartz with `n` dates is estimating.

The lattice is used throughout as an independent benchmark. Its one structural
weakness is that the cone `S₀e^{±iσ√Δt}` rooted at a single spot does not reach
the exercise region near `t = 0`, so its boundary estimate is undefined there —
`58` of `3001` time levels in the base case.

---

## 5. Crank–Nicolson with PSOR

Derivation: [`docs/02_crank_nicolson.md`](../docs/02_crank_nicolson.md).

### 5.1 Discretisation

Working in time to maturity `τ = T − t` on a uniform grid `S_i = iΔS` with the
strike pinned exactly to a node, central differences give a tridiagonal operator
in which **every power of `ΔS` cancels**:

$$
a_i = \tfrac12\sigma^2i^2 - \tfrac12(r-q)i, \quad
b_i = -\sigma^2i^2 - r, \quad
c_i = \tfrac12\sigma^2i^2 + \tfrac12(r-q)i .
$$

Two exact identities are asserted at assembly: `𝓐ₕ1 = −r` and `𝓐ₕS = −qS`.
Central differences are exact on constants and linear functions, so these test
the coefficient algebra rather than the discretisation, and they catch
essentially every sign and indexing error. (They also caught a real bug of a
different kind: the tolerance was originally absolute, and since the second
identity cancels quantities of order `σ²M³`, the solver refused to run at
`M ≥ 12800` for a residual that was pure round-off. The tolerance now scales with
the magnitude of the terms being cancelled.)

A `θ`-step gives `Av^{n+1} = Bv^n + d^n` with `A` time-independent, and the
American constraint turns that linear solve into a discrete LCP. Boundary
couplings are evaluated at **both** time levels; using only one silently drops
the scheme to first order near the boundary.

### 5.2 PSOR, and how it was verified

Projected SOR truncates each Gauss–Seidel update onto the feasible set *inside*
the sweep. Projecting after solving would converge to `max(A^{−1}b, g)`, which is
a different and wrong object.

Three concerns, each addressed by measurement rather than assertion:

**Is the vectorised solver faithful?** Production PSOR uses red–black ordering so
each half-sweep is one NumPy expression. A tridiagonal matrix is consistently
ordered in Young's sense, so red–black SOR has the same spectral radius and the
same optimal `ω` as the lexicographic sweep. A literal scalar-loop implementation
is retained purely as a reference, and the two agree to `10⁻¹⁰`. Red–black is
**`19×` faster** at `M = 400` for identical output.

**Is the iterative answer the LCP answer?** The Brennan–Schwartz algorithm solves
the same LCP *exactly* in `O(M)` — no tolerance, no relaxation parameter — under
the lower-interval exercise-region assumption that §2 establishes for the put.
The maximum disagreement with PSOR across five regimes is **`3.78 × 10⁻¹²`**.

**Does the answer depend on the tuning?** Across all `ω ∈ [0.8, 1.9]` and all
grid sizes the price spread is `1.35 × 10⁻⁷`. Only the cost changes, and the
optimum drifts upward with refinement (`ω* = 1.1, 1.2, 1.3, 1.4` at
`M = 400, 800, 1600, 3200`). The optimum sits well below 2 because `A`'s diagonal
`1 + θΔτ(σ²i² + r)` is strongly dominant at large `i`.

### 5.3 Structural issues, quantified

**The cell-Péclet violation.** `a_i ≥ 0` requires `i ≥ (r−q)/σ²`, which is `1.25`
in the base case, so exactly **one** row fails the M-matrix condition. Repairing
only that row by upwinding restores the M-matrix property and changes the price
by less than `10⁻⁸`. Upwinding *every* row — the obvious implementation — costs
`1.16 × 10⁻²`, an error `250×` larger than the central-difference grid error, and
is first order. The solver therefore upwinds selectively.

**Rannacher start-up.** Crank–Nicolson is A-stable but not L-stable, so the
payoff kink rings. At `N = 25` two fully implicit start-up steps cut the total
variation of gamma near the strike by **`37×`** (`0.46201 → 0.01257`) and the
price error by **`12×`**. At `N ≥ 100` the effect vanishes and start-up costs a
little accuracy, since the two implicit steps are themselves first order. Both
directions are reported.

**Reading the exercise set.** Locating it by thresholding the early-exercise gap
(`v − g ≤ 10⁻⁸`) reported a spurious boundary at `S* = 31.8` in the `r = 0`
regime, where theory says the exercise region is *empty*: the gap is genuinely
`O(10⁻⁹)` deep in the money there because the American and European puts
coincide. The solver now reads the exercise set from whether the projection
actually selected the obstacle — a bitwise `v == g` test with no tolerance in it.

### 5.4 Accuracy

| | value |
|---|---|
| CRR benchmark, `N = 40,000` | `6.09035196` |
| CN-PSOR, `M = N = 3200` | `6.09030490` |
| deviation | `4.71 × 10⁻⁵` |

Across all ten regimes the worst CN-vs-CRR disagreement is `2.45 × 10⁻⁴`. Two
solvers sharing no code agree to that level.

---

## 6. Longstaff–Schwartz

### 6.1 What is actually being priced

With `n` exercise dates, LSM prices a **Bermudan** option. Benchmarking it
against the continuous-exercise American value conflates two errors:

```
American − LSM = (American − Bermudan_n) + (Bermudan_n − LSM)
                  exercise-date bias        regression + sampling error
```

The lattice computes `Bermudan_n` exactly, which separates them. At 50 dates the
two terms are `+0.011730` and `−0.010058` — **the same size, opposite signs,
nearly cancelling**. Reporting only the American deviation would have shown a
misleadingly comfortable `−0.0017`.

The exercise-date bias is cleanly `O(1/n)`: the gap to the American value shrinks
by exactly the factor by which the date count grows, from `0.109210` at 5 dates
to `0.002953` at 200.

### 6.2 Look-ahead bias, measured

The classic estimator fits the exercise policy on the same paths it then values,
so the exercise decisions carry hindsight. Fitting on one sample and valuing on
an **independent** one removes it, and the result is a valid lower bound (any
fixed policy is). Sweeping paths against basis size, 25 dates, exact Bermudan
target `6.067113`, 40–400 repetitions per cell:

| paths | degree | in-sample | out-of-sample | foresight bias |
|---|---|---|---|---|
| 500 | 10 | `6.61295` | `5.85629` | `+0.75666 ± 0.02324` |
| 500 | 3 | `6.32929` | `5.95746` | `+0.37183 ± 0.02330` |
| 2,000 | 10 | `6.22896` | `5.97274` | `+0.25622 ± 0.01260` |
| 10,000 | 10 | `6.10236` | `6.04970` | `+0.05266 ± 0.00780` |
| 200,000 | 3 | `6.06342` | `6.06080` | `+0.00262 ± 0.00397` |

At 500 paths with an 11-function basis the in-sample estimate is **9.0% above**
the true Bermudan value — and above the *American* value it is approximating.
The bias decays like `1/n_paths` and grows with the number of basis functions,
consistent with an overfitting bias of order (parameters/paths). It is
statistically undetectable at 200,000 paths with a cubic basis.

The two estimators **bracket**, with one instructive exception. Out-of-sample is
below the exact Bermudan value at every path count, as a fixed-policy lower bound
must be. In-sample is above it in 9 of 10 cells; at 200,000 paths with a cubic
basis it falls `0.0037` below, because the foresight bias (`+0.0026`) has by then
decayed past the policy-suboptimality bias pushing the other way.

The reported standard errors are honest — the ratio of realised seed-to-seed
standard deviation to self-reported standard error lies in `[0.98, 1.18]` across
all ten cells. The in-sample problem is bias, not an understated error bar.

### 6.3 Implementation choices that matter

**In-the-money filtering is not a refinement.** Deviations from the exact
Bermudan value at 200,000 paths, standard error `≈ 0.016` throughout:

| degree | ITM paths only | all paths |
|---|---|---|
| 2 | `−0.014296` | `−0.338926` |
| 3 | `+0.010058` | `−0.155557` |
| 6 | `+0.008582` | `−0.069535` |

Regressing on all paths forces the polynomial to fit the flat out-of-the-money
region where the exercise decision is never live, and pays for that fit by
degrading the continuation estimate where the decision matters. **Worth up to
`0.34` in price — 21 standard errors.**

**Basis choice matters less than it appears.** Monomials and Chebyshev
polynomials of the same degree span the same function space, so the fit, every
exercise decision, and the price must be identical in exact arithmetic. Measured
`|price_poly − price_cheb|` is **bitwise zero to degree 4**, then `3.4 × 10⁻⁵`
(degree 6), `2.1 × 10⁻⁴` (8), `2.1 × 10⁻³` (12) — the divergence is purely
conditioning. The design-matrix condition number at degree 10 is `6.2 × 10¹⁰`
(Chebyshev), `4.7 × 10¹²` (monomial), `3.2 × 10¹⁶` (weighted Laguerre, at the
edge of double precision). RMSE is minimised near degree 3–4 for every basis, so
a low-degree basis is the right default and conditioning never becomes the
binding issue.

---

## 7. Variance reduction

All four estimators share the same fitted exercise policy, so differences in
interval width are attributable to the technique alone. At 200,000 valuation
paths against the exact Bermudan value `6.078622`:

| method | 95% CI width | variance per path | VRF | work-normalised gain | paths for the naive accuracy |
|---|---|---|---|---|---|
| naive | `0.06283` | `51.3772` | `1.00` | `1.00` | `200,000` |
| antithetic | `0.03744` | `18.2447` | `2.82` | `3.32` | `71,022` |
| control variate | `0.04179` | `22.7259` | `2.26` | `2.22` | `88,467` |
| antithetic + control | `0.03643` | `17.2774` | `2.97` | `3.01` | `67,257` |

Factors are computed **per path**, not per sampling unit: an antithetic unit
consumes two paths, and the per-unit comparison would report double the true
gain. The work-normalised gain multiplies by the runtime ratio; antithetic scores
above its raw factor because it draws half as many normals.

**The control variate's power is exactly `1/(1−ρ²)`.** Across all ten regimes the
measured factor matches the theoretical value to two decimal places — `ρ = 0.463`
and VRF `1.27` in the money, `ρ = 0.902` and VRF `5.37` at zero rate. `ρ` falls as
the early-exercise premium grows, because a cash flow that usually stops early
stops depending on `S_T` at all, and the European control has less to say.

**Antithetic pairs are dependent, and the error bar must say so.** The unit of
independence is the pair. Two exact-payoff experiments (3,000 repetitions, no
regression, so the truth is known independently) show the naive path-level
formula is wrong in *both* directions:

| payoff | monotone in `S_T`? | pair `ρ` | naive/correct SE | coverage, pair SE | coverage, path SE |
|---|---|---|---|---|---|
| European put | yes | `−0.4145` | `1.307` | `0.9510` | `0.9890` |
| butterfly | no | `+0.4282` | `0.837` | `0.9523` | **`0.9060`** |

For a monotone payoff — the case antithetic sampling is designed for — the naive
formula is merely conservative, and in over-covering at `98.9%` it discards the
entire variance reduction. For a non-monotone payoff both legs of a pair are
small together whenever `|Z|` is large, the correlation turns positive, and the
95% interval covers only `90.6%`.

**Variance reduction converts a variance problem into a bias problem.** Coverage
of the *true* value falls from `0.950` (naive) to `0.910` (antithetic) even
though both standard errors are sound: once the interval narrows, the residual
bias of the fixed exercise policy stops being negligible relative to it.

---

## 8. Convergence analysis

### 8.1 The reference solution

| construction | value |
|---|---|
| CRR, `2V_{2N} − V_N` at `N = 40,000/80,000` | `6.090370659` |
| CN, space Richardson + time correction at the measured order | `6.090370566` |
| **Reference** | **`6.090370613 ± 9.36 × 10⁻⁸`** |

The uncertainty is the disagreement between two independently extrapolated
methods. A single method's extrapolation would carry no error estimate at all.

### 8.2 Measured orders

Convergence orders are fitted **only above the other axis's irreducible error
floor**. Refining one axis while the other is fixed leaves the other's error as a
floor; points that have sunk into it flatten the fit. Fitting through the floor
gave `1.957` in space and `1.121` in time; fitting above it gives:

| axis / scheme | measured order |
|---|---|
| CN, space `ΔS` | **`1.995`** |
| CN, time `Δτ` (American, 2 Rannacher steps) | **`1.223`** |
| CN, time `Δτ`, self-convergence (independent measurement) | `1.265` |
| Fully implicit, time `Δτ` | `1.013` |
| CRR lattice, steps `N` | `0.990` |
| LSM, seed-to-seed s.d. vs paths (naive) | `−0.4806` |
| LSM, seed-to-seed s.d. vs paths (antithetic + control) | `−0.4948` |

**The American constraint costs Crank–Nicolson its second-order time accuracy.**
The scheme is formally second order in `Δτ`, and is second order in space, but
the free boundary is located only to `O(Δτ)` and that pollutes the value. Two
independent measurements — self-convergence in the time axis and refinement
against the reference — give `1.265` and `1.223`. Crank–Nicolson still beats the
fully implicit scheme in both order and constant (`1.6 × 10⁻³` against
`1.4 × 10⁻²` at `N = 100`).

**Monte Carlo sampling error is empirically `O(N^{-1/2})`**, with fitted slopes
within `0.7%` of `−0.5` for all four estimators.

### 8.3 Two things that turned out not to matter, and one that does

**PSOR tolerance.** The LCP solve only needs to be accurate relative to the
discretisation error. At `M = N = 3200` the error floor `4.61 × 10⁻⁵` is reached
at `tol = 10⁻⁶`; tightening to `10⁻¹²` moves the price not at all and costs
`3.1×` the sweeps. A default of `10⁻¹²` throws away a free `3×`.

**Domain truncation.** With `ΔS` held fixed by scaling `M` with `S_max`, the error
is identical to six digits for every `S_max` from `2K` to `12K`; only `1.5K`
shows truncation. The default `4K` is safely inside the flat region. *This
measurement is only meaningful because `ΔS` was held fixed* — holding `M` fixed
instead makes a larger domain mean a coarser grid, and the measured error then
rises monotonically out to `12K`, which says nothing about truncation.

**Exercise dates, which do matter.** At 100,000 paths:

| dates | RMSE | exercise-date bias | regression error |
|---|---|---|---|
| 5 | `0.10696` | `+0.10923` | `−0.00352` |
| 25 | `0.02841` | `+0.02326` | `+0.00346` |
| 50 | `0.02286` | `+0.01175` | `+0.00869` |
| 100 | `0.02010` | `+0.00592` | `+0.00791` |
| 200 | **`0.01739`** | `+0.00297` | `+0.01216` |
| 400 | `0.02372` | `+0.00149` | `+0.01748` |

**The two error terms trade off.** More dates shrink the Bermudan bias but the
accumulated regression error grows with the number of decisions. The total is
minimised near 200 dates and gets worse beyond. Pushing to the memory ceiling
(400,000 paths × 200 dates, several GB resident) leaves the bias at `−0.01601`
while the sampling s.d. falls to `0.00395`: the floor is not a sampling problem.

---

## 9. Error against wall-clock cost

The practical comparison. For the deterministic methods a single run gives the
error; for LSM the error is a random variable, so the RMSE over repeated seeds is
used and the reported runtime is that of one run.

| method | error scaling | to `10⁻²` | to `10⁻³` | to `10⁻⁴` | best achieved |
|---|---|---|---|---|---|
| CRR lattice | `t^{−0.55}` | `0.015 s` | `0.015 s` | `0.234 s` | `9.4 × 10⁻⁶` at `11.8 s` |
| CN + PSOR | `t^{−1.22}` | `0.078 s` | `0.197 s` | `1.740 s` | `2.0 × 10⁻⁵` at `9.8 s` |
| LSM (antithetic + control) | floors | **never** | **never** | **never** | `1.4 × 10⁻²` at `0.42 s` |

Three conclusions.

**Monte Carlo is not competitive for this problem, by three orders of magnitude.**
LSM never reaches `10⁻²` at any of the 17 configurations swept (5,000–200,000
paths × 10–200 dates, plus the memory-ceiling runs), while the lattice reaches it
in 15 milliseconds. The `O(N^{-1/2})` law is not the binding constraint; the
policy bias is, and no amount of sampling addresses it. This is the expected
answer for a one-dimensional problem — Monte Carlo earns its keep in dimension,
not in precision — but it is worth having measured rather than assumed.

**The lattice beats the PDE solver throughout the tested range.** CRR's
`t^{−0.55}` follows directly from `O(1/N)` accuracy at `O(N²)` cost. Crank–
Nicolson's error falls faster with cost (`t^{−1.22}`), so the fitted power laws
cross at `t ≈ 11 s`, `error ≈ 9.4 × 10⁻⁶` — right at the edge of the measured
range, which is an extrapolation rather than an observed crossing.

**What the PDE actually buys is not speed.** It returns the whole value surface,
the free boundary at *every* time level, stable Greeks from the same grid, and a
genuinely second-order spatial discretisation. Those are the reasons to pay the
`10×`.

---

## 10. The early-exercise boundary

`S*(t)` is never an input. At each time level the set of nodes where the
projection selects the payoff **is** the discrete exercise region; its upper edge
is the boundary, refined to sub-grid resolution by inverting the quadratic
vanishing of the early-exercise gap that smooth pasting implies.

Seven predictions with their reasons were registered in
[`docs/03_exercise_boundary.md`](../docs/03_exercise_boundary.md) *before* the
sensitivity study was run, together with a table of what each failure would mean.
All seven hold. Base case: `S*(0) = 80.87563`, between `S∞ = 71.42857` and
`K = 100`.

**P2, the sharpest.** With `K = 100`, `r = 5%` and `q = 10%`, the boundary just
before maturity should be `min(K, rK/q) = 50`, not the strike. Measured
`49.830` — `0.34%`. For `q = 6%` and `15%`: `0.28%` and `0.44%`. The boundary is
genuinely **discontinuous at maturity** when `q > r`, since at `τ = 0` every
in-the-money put is exercised and `S*(T) = K`; the solver reports both values.

**P3, the near-maturity law.** `K − S* ~ Kσ√(τ ln(1/τ))`. Regressing through the
origin gives a slope of `21.578` against the predicted `Kσ = 20` — `7.9%` high,
inside the `15%` band the documentation committed to in advance (the corrections
are `O(1/ln(1/τ))`). This tests the functional form: a plain `√τ` law does not
fit. The law implies infinite slope at maturity, tested as divergence of the
last-step slope under time refinement rather than as a ratio on one grid.

**P4–P6, comparative statics.**

| `σ` | `10%` | `20%` | `40%` | `60%` |
|---|---|---|---|---|
| `S*(0)` | `92.754` | `80.876` | `58.531` | `41.477` |

| `r` | `0%` | `1%` | `5%` | `12%` |
|---|---|---|---|---|
| `S*(0)` | `0.496` | `69.730` | `80.876` | `87.701` |

| `T` | `0.25` | `1` | `5` | `25` |
|---|---|---|---|---|
| `S*(0)` | `86.838` | `80.876` | `74.526` | `71.672` |

All monotone in the predicted direction, all tracking the closed-form perpetual
boundary from above, and at `T = 25` the boundary is `1.0034×` the perpetual
limit.

**The financial intuition.** Volatility is the raw material of optionality: the
right to wait is worth more when the stock can travel further, so the holder
demands a deeper in-the-money price before surrendering it. The interest rate is
the *entire* reason to exercise a put early — exercising converts the option into
cash `K` earning `r`. At `r = 0` that incentive vanishes and the measured `S*(0)`
collapses to `0.496` on a strike of `100`: no exercise region at all, which is
the theoretical answer to within floating-point noise. Dividends work the other
way: they push the risk-neutral drift down, which favours the put holder who
waits, and once `qS > rK` the incentive reverses entirely — hence the
`min(K, rK/q)` cap.

**Boundary accuracy is better than the grid, but not monotone in it.** The
smooth-pasting refinement usually buys about an order of magnitude over the raw
staircase, but at `M = 1600` the deviation is `9.7 × 10⁻²`, worse than at
`M = 400` (`9.5 × 10⁻³`). Where `S*` falls inside a cell matters: the discrete
exercise set overshoots by up to one cell and the local quadratic model then
extrapolates from an unusually small gap. Reported rather than smoothed.

---

## 11. Limitations

**Of the model.** Constant `r`, `σ` and `q`; geometric Brownian motion; a single
asset; no discrete dividends, no early-exercise restrictions, no transaction
costs. Every conclusion here is a conclusion about numerical methods under
Black–Scholes, not about markets.

**Of the Monte Carlo work.**

- Only the *lower* bound is implemented. A dual upper bound (Andersen–Broadie)
  would bracket the true value and is the natural next step; without it the LSM
  results are one-sided and the `−0.015` policy bias can be measured only against
  an external benchmark.
- LSM stores every path at every exercise date, so memory is
  `O(n_paths × n_dates)`. The largest configurations tested peak at several GB,
  which caps the frontier grid.
- The regression basis is a function of `S` alone. Richer state (running maximum,
  time-to-maturity interactions) is not explored.

**Of the PDE work.**

- The grid is uniform in `S`. A non-uniform mesh concentrated near the strike and
  the boundary would improve accuracy per node, and is the obvious extension.
- Only Dirichlet truncation is used; a Neumann or artificial-boundary condition
  would allow a smaller domain.
- The temporal order of `1.22`–`1.27` is inherent to applying Crank–Nicolson to a
  free-boundary problem. Higher order would need a boundary-tracking or
  penalty-based scheme.

**Of the comparison itself.**

- Everything is one-dimensional. The single strongest argument for Longstaff–
  Schwartz — that it scales to many underlyings where a grid cannot — is
  therefore invisible here. **The finding is that Monte Carlo is the wrong tool
  for this problem, not that it is the wrong tool.**
- Timings are single-threaded on one machine; they compare implementations, not
  methods in the abstract. The PSOR sweep is vectorised NumPy while the
  Brennan–Schwartz elimination is a Python loop, which is why the exact `O(M)`
  solver is not the faster one at scale.
- The CN/CRR crossover at `t ≈ 11 s` is an extrapolation of fitted power laws to
  the edge of the measured range.

---

## 12. Conclusion

For a single-asset vanilla American put, the ranking is unambiguous and slightly
uncomfortable for the fashionable method: **the binomial lattice is the most
efficient**, Crank–Nicolson with PSOR is roughly `10×` more expensive at matched
accuracy but converges faster and returns far more information, and
**Longstaff–Schwartz is not in the same league** — it cannot reach `10⁻²` at any
budget tested, three orders of magnitude short.

The reason Monte Carlo loses is the interesting part. It is not sampling noise:
the seed-to-seed standard deviation obeys `N^{−0.49}` to within `1%` of theory,
and variance reduction delivers a genuine `2.97×`. It is that the *policy* is
biased, that the bias does not respond to more paths, more exercise dates, or a
richer basis, and that the two components of that bias trade off against each
other so that adding exercise dates past about 200 makes the total worse. Every
technique that narrows the confidence interval simply exposes the bias sooner:
coverage of the true value falls from `0.950` to `0.910` when antithetic sampling
is switched on, with nothing wrong with either standard error.

Several of the results here only became visible after a measurement error was
corrected, and those corrections were as instructive as the results:
convergence orders fitted through a floor understate themselves; a domain study
that does not hold `ΔS` fixed reverses its own conclusion; benchmarking a
Bermudan estimator against an American value hides two cancelling errors; a
tolerance test for the exercise set that is not scale-free invents an exercise
region where none exists; and an absolute tolerance on a cancellation that grows
with the grid will eventually refuse to run at all.

The methodological point that generalises: **agreement between two methods that
share no code is the only evidence worth much.** The reference solution here is
quoted to `9.4 × 10⁻⁸` precisely because that is how far two independent
extrapolations disagree, not because either one claims more.

---

## Reproducing everything

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                    # 238 tests

python experiments/m1_european_baseline.py   # < 1 min
python experiments/m2_analytic_anchors.py    # < 1 min
python experiments/m3_crank_nicolson.py      # ~4 min
python experiments/m4_longstaff_schwartz.py  # ~2 min
python experiments/m5_variance_reduction.py  # ~2 min
python experiments/m6_convergence.py         # ~7 min
python experiments/m7_exercise_boundary.py   # ~6 min
```

Each script writes CSVs to `results/` and figures to `figures/` and prints a
summary. [`RESULTS.md`](../RESULTS.md) records every verified number with its
source file.
