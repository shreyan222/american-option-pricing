"""Milestone 5: naive vs antithetic vs control-variate Monte Carlo.

Produces
--------
results/m5_headline.csv        Variance, SE, CI width, runtime, VRF, work-normalised gain.
results/m5_coverage.csv        Empirical 95% coverage, with and without the pair correction.
results/m5_scaling.csv         SE vs path count for each method, with fitted slopes.
results/m5_beta_source.csv     Control coefficient from the training sample vs the valued sample.
results/m5_regimes.csv         Control-variate correlation and VRF across parameter regimes.
figures/m5_error_vs_paths.png
figures/m5_efficiency.png
figures/m5_coverage.png

Run:  python experiments/m5_variance_reduction.py     (~5 min)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amopt import plotting  # noqa: E402
from amopt.binomial import crr  # noqa: E402
from amopt.config import REGIME_NOTES, REGIMES  # noqa: E402
from amopt.lsm import Z_975, simulate_gbm  # noqa: E402
from amopt.variance_reduction import (  # noqa: E402
    METHODS,
    lsm_with_variance_reduction,
    variance_reduction_factor,
)

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
P = REGIMES["base"]
N_DATES = 50
N_PATHS = 200_000
SEED = 20240901

PRETTY = {
    "naive": "naive",
    "antithetic": "antithetic",
    "control": "control variate",
    "antithetic_control": "antithetic + control",
}
COLOR = {
    "naive": plotting.color("lsm"),
    "antithetic": plotting.color("lsm_antithetic"),
    "control": plotting.color("lsm_control"),
    "antithetic_control": plotting.PALETTE["yellow"],
}


def bermudan(n_dates=N_DATES, p=P) -> float:
    N = (40_000 // n_dates) * n_dates
    return crr(p.S0, p.K, p.T, p.r, p.sigma, N, p.q, "put", "american",
               bermudan_dates=n_dates).price


def headline(berm: float) -> pd.DataFrame:
    runs = {
        m: lsm_with_variance_reduction(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put",
                                       n_paths=N_PATHS, n_steps=N_DATES, method=m, seed=SEED)
        for m in METHODS
    }
    base = runs["naive"]
    target_se = base.std_error
    rows = []
    for m, res in runs.items():
        vrf = variance_reduction_factor(base, res)
        # Work-normalised efficiency: variance x time, the standard measure, since
        # a technique that halves the variance but doubles the cost gains nothing.
        gain = vrf * (base.runtime_s / res.runtime_s)
        rows.append(
            {
                "method": m, "price": res.price, "deviation_from_bermudan": res.price - berm,
                "unit_variance": res.unit_variance, "n_units": res.n_units,
                "n_paths": res.n_paths,
                "variance_per_path": res.unit_variance * (2.0 if "antithetic" in m else 1.0),
                "std_error": res.std_error, "ci_low": res.ci_low, "ci_high": res.ci_high,
                "ci_width": res.ci_width, "runtime_s": res.runtime_s,
                "variance_reduction_factor": vrf,
                "work_normalised_gain": gain,
                "paths_for_naive_se": res.paths_for_target_se(target_se),
                "correlation": res.correlation, "beta": res.beta,
            }
        )
    return pd.DataFrame(rows)


def coverage_study(berm: float, n_rep: int = 300, n_paths: int = 20_000) -> pd.DataFrame:
    r"""Empirical coverage of the nominal 95% interval.

    Two targets are used deliberately:

    * the estimator's own mean across repetitions, which isolates the quality of
      the *standard error*; and
    * the exact Bermudan value, which additionally exposes the low bias of a
      fixed (suboptimal) exercise policy.

    For antithetic methods a third interval is reported, built from the naive
    path-level standard error :math:`s/\sqrt{2n}` that ignores the dependence
    within a pair.  That is the mistake this repository is testing for.
    """
    rows = []
    store: dict[str, list] = {m: [] for m in METHODS}
    for m in METHODS:
        for i in range(n_rep):
            store[m].append(
                lsm_with_variance_reduction(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put",
                                            n_paths=n_paths, n_steps=N_DATES, method=m,
                                            seed=90_000 + i)
            )
    for m, runs in store.items():
        prices = np.array([r.price for r in runs])
        lo = np.array([r.ci_low for r in runs])
        hi = np.array([r.ci_high for r in runs])
        own = prices.mean()
        row = {
            "method": m, "n_rep": n_rep, "n_paths": n_paths,
            "mean_price": own, "realised_sd": prices.std(ddof=1),
            "mean_reported_se": float(np.mean([r.std_error for r in runs])),
            "coverage_of_own_mean": float(np.mean((lo <= own) & (own <= hi))),
            "coverage_of_bermudan": float(np.mean((lo <= berm) & (berm <= hi))),
            "bias_vs_bermudan": own - berm,
        }
        if "antithetic" in m:
            # The genuine naive path-level formula, s/sqrt(2n) over the raw path
            # payoffs, ignoring that paths 2k and 2k+1 are a dependent pair.
            naive_se = np.array([r.naive_path_level_se() for r in runs])
            lo_n, hi_n = prices - Z_975 * naive_se, prices + Z_975 * naive_se
            row["naive_se_coverage_of_own_mean"] = float(np.mean((lo_n <= own) & (own <= hi_n)))
            row["mean_naive_se"] = float(naive_se.mean())
            row["mean_pair_correlation"] = float(np.mean([r.pair_correlation for r in runs]))
        rows.append(row)
    return pd.DataFrame(rows)


def antithetic_pairing_demonstration(n_paths: int = 4_000, n_rep: int = 3_000) -> pd.DataFrame:
    r"""Both directions of the antithetic standard-error mistake, on exact payoffs.

    The American put is monotone in the terminal price, so its antithetic pairs
    are *negatively* correlated and the naive path-level standard error is merely
    conservative -- it over-covers, and throws away the whole variance reduction.
    A **butterfly** payoff is not monotone: both legs of a pair are small
    together whenever :math:`|Z|` is large, the pair correlation turns
    *positive*, and the naive interval becomes too narrow and under-covers.

    Terminal payoffs are used directly (no LSM, no regression), so the
    expectation is known to arbitrary precision from an independent sample and
    the coverage measurement is exact rather than contaminated by policy bias.
    """
    r_, sig, T_, K_ = P.r, P.sigma, P.T, P.K
    disc = np.exp(-r_ * T_)

    def european_put(S):
        return disc * np.maximum(K_ - S, 0.0)

    def butterfly(S):
        return disc * (np.maximum(S - 90.0, 0.0) - 2 * np.maximum(S - 100.0, 0.0)
                       + np.maximum(S - 110.0, 0.0))

    rows = []
    for name, fn, monotone in (("european_put", european_put, True),
                               ("butterfly", butterfly, False)):
        big = simulate_gbm_terminal(4_000_000, seed=999)
        truth = float(fn(big).mean())
        rng = np.random.default_rng(0)
        cov_pair = cov_naive = 0
        corrs, se_pairs, se_naives = [], [], []
        for _ in range(n_rep):
            S = simulate_gbm(P.S0, r_, sig, T_, n_paths, 1, P.q, rng=rng, antithetic=True)[:, -1]
            X = fn(S)
            pm = 0.5 * (X[0::2] + X[1::2])
            se_pair = pm.std(ddof=1) / np.sqrt(pm.size)
            se_naive = X.std(ddof=1) / np.sqrt(n_paths)
            m = pm.mean()
            corrs.append(np.corrcoef(X[0::2], X[1::2])[0, 1])
            se_pairs.append(se_pair); se_naives.append(se_naive)
            cov_pair += abs(m - truth) <= Z_975 * se_pair
            cov_naive += abs(m - truth) <= Z_975 * se_naive
        rows.append(
            {
                "payoff": name, "monotone_in_S_T": monotone, "truth": truth,
                "n_paths": n_paths, "n_rep": n_rep,
                "mean_pair_correlation": float(np.mean(corrs)),
                "mean_pair_se": float(np.mean(se_pairs)),
                "mean_naive_path_se": float(np.mean(se_naives)),
                "naive_over_pair_se_ratio": float(np.mean(se_naives) / np.mean(se_pairs)),
                "coverage_pair_se": cov_pair / n_rep,
                "coverage_naive_se": cov_naive / n_rep,
            }
        )
    return pd.DataFrame(rows)


def simulate_gbm_terminal(n, seed):
    return simulate_gbm(P.S0, P.r, P.sigma, P.T, n, 1, P.q, seed=seed)[:, -1]


def scaling_study() -> pd.DataFrame:
    rows = []
    for m in METHODS:
        for n in (5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 400_000):
            res = lsm_with_variance_reduction(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put",
                                              n_paths=n, n_steps=N_DATES, method=m,
                                              seed=SEED, n_train=50_000)
            rows.append({"method": m, "n_paths": n, "price": res.price,
                         "std_error": res.std_error, "ci_width": res.ci_width,
                         "runtime_s": res.runtime_s})
    df = pd.DataFrame(rows)
    slopes = {
        m: float(np.polyfit(np.log(g["n_paths"]), np.log(g["std_error"]), 1)[0])
        for m, g in df.groupby("method")
    }
    df["fitted_se_slope"] = df["method"].map(slopes)
    return df


def beta_source_study(berm: float) -> pd.DataFrame:
    rows = []
    for src in ("training", "sample", "none"):
        for n in (2_000, 10_000, 50_000, 200_000):
            res = lsm_with_variance_reduction(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put",
                                              n_paths=n, n_steps=N_DATES, method="control",
                                              beta_source=src, seed=SEED, n_train=50_000)
            rows.append({"beta_source": src, "n_paths": n, "price": res.price,
                         "beta": res.beta, "std_error": res.std_error,
                         "deviation_from_bermudan": res.price - berm})
    return pd.DataFrame(rows)


def regime_study() -> pd.DataFrame:
    rows = []
    for name, p in REGIMES.items():
        b = bermudan(N_DATES, p)
        base = lsm_with_variance_reduction(p.S0, p.K, p.T, p.r, p.sigma, p.q, "put",
                                           n_paths=100_000, n_steps=N_DATES,
                                           method="naive", seed=SEED)
        out = {"regime": name, "note": REGIME_NOTES[name], **p.as_dict(),
               "bermudan": b, "naive_price": base.price, "naive_se": base.std_error}
        for m in ("antithetic", "control", "antithetic_control"):
            res = lsm_with_variance_reduction(p.S0, p.K, p.T, p.r, p.sigma, p.q, "put",
                                              n_paths=100_000, n_steps=N_DATES,
                                              method=m, seed=SEED)
            out[f"vrf_{m}"] = variance_reduction_factor(base, res)
            if m == "control":
                out["control_correlation"] = res.correlation
                out["control_beta"] = res.beta
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    plotting.use_style()
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    berm = bermudan()

    head = headline(berm); head.to_csv(RESULTS / "m5_headline.csv", index=False)
    cov = coverage_study(berm); cov.to_csv(RESULTS / "m5_coverage.csv", index=False)
    sca = scaling_study(); sca.to_csv(RESULTS / "m5_scaling.csv", index=False)
    bet = beta_source_study(berm); bet.to_csv(RESULTS / "m5_beta_source.csv", index=False)
    pair = antithetic_pairing_demonstration()
    pair.to_csv(RESULTS / "m5_antithetic_pairing.csv", index=False)
    reg = regime_study(); reg.to_csv(RESULTS / "m5_regimes.csv", index=False)

    # ---- Figure: error vs paths ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    for m in METHODS:
        d = sca[sca["method"] == m]
        ax.plot(d["n_paths"], d["std_error"], marker="o", ms=5, color=COLOR[m],
                label=f"{PRETTY[m]} (slope {d['fitted_se_slope'].iloc[0]:.3f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("valuation paths $N$"); ax.set_ylabel("standard error")
    ax.set_title(r"Monte Carlo error follows $O(N^{-1/2})$")
    d0 = sca[sca["method"] == "naive"]
    plotting.reference_slope(ax, d0["n_paths"].iloc[0], d0["std_error"].iloc[0], -0.5,
                             r"slope $-1/2$", offset=1.6)
    ax.legend(loc="lower left", fontsize=7.5)

    ax = axes[1]
    for m in METHODS:
        d = sca[sca["method"] == m]
        ax.plot(d["runtime_s"], d["std_error"], marker="o", ms=5, color=COLOR[m],
                label=PRETTY[m])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("runtime (s)"); ax.set_ylabel("standard error")
    ax.set_title("Error against wall-clock cost")
    ax.legend(loc="lower left", fontsize=7.5)
    plotting.save(fig, FIGURES / "m5_error_vs_paths.png",
                  caption=f"{P.label()}; {N_DATES} exercise dates, cubic basis, "
                          f"policy fitted on 50,000 independent training paths, seed {SEED}.")

    # ---- Figure: efficiency -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    x = np.arange(len(METHODS))
    vrf = [head[head["method"] == m]["variance_reduction_factor"].iloc[0] for m in METHODS]
    gain = [head[head["method"] == m]["work_normalised_gain"].iloc[0] for m in METHODS]
    ax.bar(x - 0.19, vrf, width=0.34, color=plotting.PALETTE["blue"],
           label="variance reduction factor")
    ax.bar(x + 0.19, gain, width=0.34, color=plotting.PALETTE["orange"],
           label=r"work-normalised gain (variance $\times$ time)")
    for xi, (a, b) in enumerate(zip(vrf, gain)):
        plotting.annotate(ax, xi - 0.19, a, f"{a:.2f}", dx=0, dy=3, ha="center")
        plotting.annotate(ax, xi + 0.19, b, f"{b:.2f}", dx=0, dy=3, ha="center")
    ax.axhline(1.0, color=plotting.INK_MUTED, lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[m] for m in METHODS], fontsize=7.5)
    ax.set_ylabel("factor vs naive"); ax.set_ylim(0, max(vrf + gain) * 1.35)
    ax.set_title(f"Variance reduction at {N_PATHS:,} paths")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[1]
    d = reg.sort_values("control_correlation")
    ax.scatter(d["control_correlation"] ** 2, d["vrf_control"], s=45,
               color=plotting.color("lsm_control"), zorder=3, label="measured, by regime")
    rho2 = np.linspace(d["control_correlation"].min() ** 2 * 0.9,
                       d["control_correlation"].max() ** 2 * 1.02, 100)
    ax.plot(rho2, 1.0 / (1.0 - rho2), lw=1.6, ls=(0, (4, 3)), color=plotting.INK_MUTED,
            label=r"theory: $1/(1-\rho^2)$")
    for _, row in d.iterrows():
        plotting.annotate(ax, row["control_correlation"] ** 2, row["vrf_control"],
                          row["regime"], dx=5, dy=-3)
    ax.set_xlabel(r"$\rho^2$, squared correlation with the European payoff")
    ax.set_ylabel("variance reduction factor")
    ax.set_title("The control variate's power is exactly $1/(1-\\rho^2)$")
    ax.legend(loc="upper left", fontsize=7.5)
    plotting.save(fig, FIGURES / "m5_efficiency.png",
                  caption="Left: base case. Right: one point per parameter regime, "
                          "100,000 paths each; the dashed curve is theory, not a fit.")

    # ---- Figure: coverage ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    ax = axes[0]
    x = np.arange(len(METHODS))
    c_own = [cov[cov["method"] == m]["coverage_of_own_mean"].iloc[0] for m in METHODS]
    c_ber = [cov[cov["method"] == m]["coverage_of_bermudan"].iloc[0] for m in METHODS]
    ax.bar(x - 0.22, c_own, width=0.30, color=plotting.PALETTE["blue"],
           label="covers the estimator's own mean (tests the SE)")
    ax.bar(x + 0.10, c_ber, width=0.30, color=plotting.PALETTE["orange"],
           label="covers the exact Bermudan value (SE + policy bias)")
    naive_x, naive_y = [], []
    for i, m in enumerate(METHODS):
        v = cov[cov["method"] == m].get("naive_se_coverage_of_own_mean")
        if v is not None and not pd.isna(v.iloc[0]):
            naive_x.append(i + 0.40); naive_y.append(float(v.iloc[0]))
    if naive_x:
        ax.bar(naive_x, naive_y, width=0.24, color=plotting.PALETTE["red"],
               label="same, but with the path-level SE that ignores pairing")
    ax.axhline(0.95, color=plotting.INK, lw=1.4, ls=(0, (4, 3)))
    plotting.annotate(ax, len(METHODS) - 0.5, 0.95, "nominal 95%", dx=0, dy=4, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[m] for m in METHODS], fontsize=8)
    ax.set_ylabel("empirical coverage"); ax.set_ylim(0.80, 1.02)
    ax.set_title("Do the 95% intervals actually cover?")
    ax.legend(loc="lower left", fontsize=7)

    ax = axes[1]
    x2 = np.arange(len(pair))
    ax.bar(x2 - 0.19, pair["coverage_pair_se"], width=0.34, color=plotting.PALETTE["blue"],
           label="pair standard error (correct)")
    ax.bar(x2 + 0.19, pair["coverage_naive_se"], width=0.34, color=plotting.PALETTE["red"],
           label="path-level standard error (ignores pairing)")
    for i, row in pair.reset_index().iterrows():
        plotting.annotate(ax, i - 0.19, row["coverage_pair_se"],
                          f"{row['coverage_pair_se']:.3f}", dx=0, dy=3, ha="center")
        plotting.annotate(ax, i + 0.19, row["coverage_naive_se"],
                          f"{row['coverage_naive_se']:.3f}", dx=0, dy=3, ha="center")
    ax.axhline(0.95, color=plotting.INK, lw=1.4, ls=(0, (4, 3)))
    ax.set_xticks(x2)
    ax.set_xticklabels(
        [rf"European put (monotone in $S_T$)" "\n" rf"pair $\rho$ = "
         f"{pair['mean_pair_correlation'].iloc[0]:+.2f}",
         rf"butterfly (not monotone)" "\n" rf"pair $\rho$ = "
         f"{pair['mean_pair_correlation'].iloc[1]:+.2f}"], fontsize=8)
    ax.set_ylabel("empirical coverage"); ax.set_ylim(0.84, 1.0)
    ax.set_title("Why the pair is the unit of independence")
    ax.legend(loc="lower left", fontsize=7)
    plotting.save(fig, FIGURES / "m5_coverage.png",
                  caption=f"Left: {cov['n_rep'].iloc[0]} LSM repetitions at "
                          f"{cov['n_paths'].iloc[0]:,} paths. Right: {pair['n_rep'].iloc[0]:,} "
                          f"repetitions on exact terminal payoffs, {pair['n_paths'].iloc[0]:,} "
                          f"paths each. Binomial 2-sigma band around 0.95 is "
                          f"+-{2*np.sqrt(0.95*0.05/pair['n_rep'].iloc[0]):.3f} (right panel).")

    # ---- Console -------------------------------------------------------
    print("== Milestone 5 ==")
    print(f"exact Bermudan ({N_DATES} dates): {berm:.6f}")
    print(f"{'method':>21} {'price':>9} {'SE':>9} {'CI width':>9} {'var/path':>9} "
          f"{'VRF':>6} {'work gain':>9} {'paths for naive SE':>19}")
    for _, row in head.iterrows():
        print(f"{PRETTY[row['method']]:>21} {row['price']:9.5f} {row['std_error']:9.6f} "
              f"{row['ci_width']:9.5f} {row['variance_per_path']:9.4f} "
              f"{row['variance_reduction_factor']:6.2f} {row['work_normalised_gain']:9.2f} "
              f"{row['paths_for_naive_se']:19,.0f}")
    print(f"control-variate correlation rho = "
          f"{head[head['method']=='control']['correlation'].iloc[0]:.4f}, "
          f"beta = {head[head['method']=='control']['beta'].iloc[0]:.4f}; "
          f"theory 1/(1-rho^2) = "
          f"{1/(1-head[head['method']=='control']['correlation'].iloc[0]**2):.2f}")
    print("coverage of the 95% interval:")
    for _, row in cov.iterrows():
        extra = ""
        if not pd.isna(row.get("naive_se_coverage_of_own_mean", np.nan)):
            extra = (f"   [path-level SE: coverage {row['naive_se_coverage_of_own_mean']:.3f}, "
                     f"SE {row['mean_naive_se']:.6f} vs correct {row['mean_reported_se']:.6f}, "
                     f"pair rho {row['mean_pair_correlation']:+.3f}]")
        print(f"   {PRETTY[row['method']]:>21}: own mean {row['coverage_of_own_mean']:.3f}, "
              f"Bermudan {row['coverage_of_bermudan']:.3f}, "
              f"bias {row['bias_vs_bermudan']:+.5f}{extra}")
    print("fitted SE-vs-paths slopes (theory -0.5):")
    for m in METHODS:
        print(f"   {PRETTY[m]:>21}: {sca[sca['method']==m]['fitted_se_slope'].iloc[0]:+.4f}")
    print("control-variate rho^2 and VRF by regime:")
    for _, row in reg.iterrows():
        print(f"   {row['regime']:>15}: rho={row['control_correlation']:.4f} "
              f"VRF={row['vrf_control']:5.2f} (theory {1/(1-row['control_correlation']**2):5.2f}), "
              f"antithetic VRF={row['vrf_antithetic']:5.2f}")
    print("antithetic pairing, exact payoffs:")
    for _, row in pair.iterrows():
        print(f"   {row['payoff']:>13} (monotone={bool(row['monotone_in_S_T'])}): "
              f"pair rho {row['mean_pair_correlation']:+.4f}, "
              f"naive/pair SE ratio {row['naive_over_pair_se_ratio']:.3f}, "
              f"coverage pair {row['coverage_pair_se']:.4f} vs naive "
              f"{row['coverage_naive_se']:.4f}")
    print("control coefficient source:")
    for src, g in bet.groupby("beta_source"):
        print(f"   {src:>9}: " + ", ".join(
            f"n={int(rr['n_paths']):,}: dev {rr['deviation_from_bermudan']:+.5f} (b={rr['beta']:.4f})"
            for _, rr in g.iterrows()))


if __name__ == "__main__":
    main()
