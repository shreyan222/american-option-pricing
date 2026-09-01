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
