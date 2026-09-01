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
