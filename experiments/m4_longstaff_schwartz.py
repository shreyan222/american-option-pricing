"""Milestone 4: Longstaff-Schwartz, validated against the right benchmark.

The central methodological point: LSM with ``n`` exercise dates estimates the
``n``-date **Bermudan** value, not the continuous-exercise American value.  The
lattice computes the Bermudan value exactly, so the two error sources can be
separated:

    American - LSM  =  (American - Bermudan_n)  +  (Bermudan_n - LSM)
                        ^ exercise-date bias      ^ regression / sampling error

Reporting only the first quantity, as a single "deviation from benchmark", hides
which part of the method is actually inaccurate.

Produces
--------
results/m4_headline.csv          Headline estimate vs three benchmarks, with CI.
results/m4_bias_decomposition.csv  Exercise-date bias vs regression error by n_dates.
results/m4_in_vs_out_of_sample.csv Foresight bias and standard-error honesty, 40 seeds.
results/m4_basis_study.csv       Basis family x polynomial degree.
results/m4_itm_filter.csv        In-the-money path filtering, on and off.
figures/m4_bias_decomposition.png
figures/m4_in_vs_out_of_sample.png
figures/m4_basis_study.png

Run:  python experiments/m4_longstaff_schwartz.py     (~3 min)
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
from amopt.binomial import crr, crr_price  # noqa: E402
from amopt.config import REGIMES  # noqa: E402
from amopt.crank_nicolson import solve_pde  # noqa: E402
from amopt.lsm import _apply_policy, _fit_policy, basis_matrix, longstaff_schwartz, simulate_gbm  # noqa: E402

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
P = REGIMES["base"]
CRR_STEPS = 40_000
N_PATHS = 200_000
N_DATES = 50
#: Fewer dates for the bias sweep: it is repeated thousands of times.
N_DATES_BIAS = 25
SEED = 20240901


def bermudan_benchmark(n_dates: int) -> float:
    """Exact n-date Bermudan value from the lattice (N chosen as a multiple)."""
    N = (CRR_STEPS // n_dates) * n_dates
    return crr(P.S0, P.K, P.T, P.r, P.sigma, N, P.q, "put", "american",
               bermudan_dates=n_dates).price


def headline(am_ref: float, cn_ref: float) -> pd.DataFrame:
    berm = bermudan_benchmark(N_DATES)
    rows = []
    for oos in (False, True):
        res = longstaff_schwartz(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put",
                                 n_paths=N_PATHS, n_steps=N_DATES, degree=3,
                                 basis="poly", seed=SEED, out_of_sample=oos)
        for name, ref in (("american_crr", am_ref), ("american_cn_psor", cn_ref),
                          ("bermudan_50_crr", berm)):
            d = res.deviation_from(ref)
            rows.append(
                {
                    "estimator": "lsm_out_of_sample" if oos else "lsm_in_sample",
                    "price": res.price, "std_error": res.std_error,
                    "ci_low": res.ci_low, "ci_high": res.ci_high,
                    "ci_width": res.ci_high - res.ci_low,
                    "n_paths": res.n_paths, "n_steps": res.n_steps,
                    "runtime_s": res.runtime_s,
                    "early_exercise_fraction": res.early_exercise_fraction,
                    "mean_stopping_time": res.mean_stopping_time,
                    "benchmark_name": name, **d,
                }
            )
    return pd.DataFrame(rows)


def bias_decomposition(am_ref: float) -> pd.DataFrame:
    rows = []
    for n_dates in (5, 10, 25, 50, 100, 200):
        berm = bermudan_benchmark(n_dates)
        res = longstaff_schwartz(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put",
                                 n_paths=N_PATHS, n_steps=n_dates, degree=3,
                                 basis="poly", seed=SEED, out_of_sample=True)
        rows.append(
            {
                "n_dates": n_dates,
                "american_crr": am_ref,
                "bermudan_crr": berm,
                "lsm_price": res.price,
                "lsm_std_error": res.std_error,
                "exercise_date_bias": am_ref - berm,
                "regression_and_sampling_error": berm - res.price,
                "total_deviation": am_ref - res.price,
                "regression_error_in_se": (berm - res.price) / res.std_error,
                "runtime_s": res.runtime_s,
            }
        )
    return pd.DataFrame(rows)


def in_vs_out_of_sample() -> pd.DataFrame:
    """Sweep the sample size and basis size to expose the in-sample foresight bias.

    The classic estimator fits the exercise policy on the same paths it then
    values, so the fitted continuation values partly explain the noise of those
    particular paths and the exercise decisions carry a sliver of hindsight.  The
    size of that bias should scale with the ratio of fitted parameters to paths,
    so it is invisible at 200,000 paths with a cubic basis and enormous at 500
    paths with a degree-10 basis.  Sweeping both makes the mechanism visible
    rather than leaving it as a null result at one operating point.

    For each cell the whole estimator is repeated over independent seeds, so the
    *realised* standard deviation across repetitions can be compared with the
    standard error each run reports about itself.
    """
    berm = bermudan_benchmark(N_DATES_BIAS)
    plan = [
        (500, 3, 400), (500, 10, 400),
        (2_000, 3, 300), (2_000, 10, 300),
        (10_000, 3, 200), (10_000, 10, 200),
        (50_000, 3, 100), (50_000, 10, 100),
        (200_000, 3, 40), (200_000, 10, 40),
    ]
    rows = []
    for n_paths, degree, n_seeds in plan:
        ins, oos, se_in, se_out = [], [], [], []
        for s in range(n_seeds):
            rng = np.random.default_rng(7_000 + s)
            S = simulate_gbm(P.S0, P.r, P.sigma, P.T, n_paths, N_DATES_BIAS, P.q, rng=rng)
            coef, cash, _, _ = _fit_policy(
                S, P.K, P.r, P.T, N_DATES_BIAS, degree, "poly", True, "put"
            )
            S2 = simulate_gbm(P.S0, P.r, P.sigma, P.T, n_paths, N_DATES_BIAS, P.q, rng=rng)
            cash2, _ = _apply_policy(
                S2, P.K, P.r, P.T, N_DATES_BIAS, degree, "poly", True, "put", coef
            )
            ins.append(cash.mean()); oos.append(cash2.mean())
            se_in.append(cash.std(ddof=1) / np.sqrt(n_paths))
            se_out.append(cash2.std(ddof=1) / np.sqrt(n_paths))
        ins, oos = np.array(ins), np.array(oos)
        d = ins - oos
        rows.append(
            {
                "n_paths": n_paths, "degree": degree, "n_basis": degree + 1,
                "n_seeds": n_seeds, "bermudan_benchmark": berm,
                "in_sample_mean": ins.mean(),
                "out_of_sample_mean": oos.mean(),
                "in_sample_bias": ins.mean() - berm,
                "out_of_sample_bias": oos.mean() - berm,
                "foresight_bias": d.mean(),
                "foresight_bias_se": d.std(ddof=1) / np.sqrt(n_seeds),
                "in_sample_reported_se": float(np.mean(se_in)),
                "in_sample_realised_sd": ins.std(ddof=1),
                "out_of_sample_reported_se": float(np.mean(se_out)),
                "out_of_sample_realised_sd": oos.std(ddof=1),
                "params_per_path": (degree + 1) / n_paths,
            }
        )
    return pd.DataFrame(rows)


def basis_study() -> pd.DataFrame:
    berm = bermudan_benchmark(N_DATES)
    S = simulate_gbm(P.S0, P.r, P.sigma, P.T, 50_000, N_DATES, P.q, seed=99)
    mid = S[:, N_DATES // 2]
    mid = mid[mid < P.K]
    rows = []
    for basis in ("poly", "laguerre", "chebyshev"):
        for degree in (1, 2, 3, 4, 5, 6, 8, 10):
            res = longstaff_schwartz(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put",
                                     n_paths=N_PATHS, n_steps=N_DATES, degree=degree,
                                     basis=basis, seed=SEED, out_of_sample=True)
            rows.append(
                {
                    "basis": basis, "degree": degree, "price": res.price,
                    "std_error": res.std_error,
                    "deviation_from_bermudan": res.price - berm,
                    "abs_deviation": abs(res.price - berm),
                    "design_condition_number": float(
                        np.linalg.cond(basis_matrix(mid, P.K, degree, basis))
                    ),
                    "runtime_s": res.runtime_s,
                    "n_skipped_regressions": res.n_skipped_regressions,
                }
            )
    return pd.DataFrame(rows)


def itm_filter_study() -> pd.DataFrame:
    berm = bermudan_benchmark(N_DATES)
    rows = []
    for itm_only in (True, False):
        for degree in (2, 3, 4, 6):
            res = longstaff_schwartz(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put",
                                     n_paths=N_PATHS, n_steps=N_DATES, degree=degree,
                                     basis="poly", itm_only=itm_only, seed=SEED,
                                     out_of_sample=True)
            rows.append({"itm_only": itm_only, "degree": degree, "price": res.price,
                         "std_error": res.std_error,
                         "deviation_from_bermudan": res.price - berm,
                         "runtime_s": res.runtime_s})
    return pd.DataFrame(rows)


def main() -> None:
    plotting.use_style()
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)

    am_ref = crr_price(P.S0, P.K, P.T, P.r, P.sigma, CRR_STEPS, P.q, "put", "american")
    cn_ref = solve_pde(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "american",
                       M=3200, N=3200).price
    berm = bermudan_benchmark(N_DATES)

    head = headline(am_ref, cn_ref); head.to_csv(RESULTS / "m4_headline.csv", index=False)
    bias = bias_decomposition(am_ref); bias.to_csv(RESULTS / "m4_bias_decomposition.csv", index=False)
    io = in_vs_out_of_sample(); io.to_csv(RESULTS / "m4_in_vs_out_of_sample.csv", index=False)
    bs_ = basis_study(); bs_.to_csv(RESULTS / "m4_basis_study.csv", index=False)
    itm = itm_filter_study(); itm.to_csv(RESULTS / "m4_itm_filter.csv", index=False)

    # ---- Figure: bias decomposition -----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    ax.axhline(am_ref, color=plotting.color("binomial"), lw=1.6, ls=(0, (4, 3)),
               label=f"American (CRR $N$={CRR_STEPS:,})")
    ax.plot(bias["n_dates"], bias["bermudan_crr"], marker="s", ms=5,
            color=plotting.PALETTE["violet"], label="Bermudan (exact, lattice)")
    ax.errorbar(bias["n_dates"], bias["lsm_price"],
                yerr=1.96 * bias["lsm_std_error"], marker="o", ms=5, lw=2.0,
                capsize=3, color=plotting.color("lsm"), label="LSM (out-of-sample, 95% CI)")
    ax.set_xscale("log")
    ax.set_xlabel("exercise dates $n$"); ax.set_ylabel("price")
    ax.set_title("LSM tracks the Bermudan value, not the American value")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(bias["n_dates"], bias["exercise_date_bias"], marker="s", ms=5,
            color=plotting.PALETTE["violet"], label="exercise-date bias (American - Bermudan)")
    ax.plot(bias["n_dates"], bias["regression_and_sampling_error"].abs(), marker="o", ms=5,
            color=plotting.color("lsm"), label="|regression + sampling error|")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("exercise dates $n$"); ax.set_ylabel("absolute contribution to error")
    ax.set_title("Which term dominates, and where they cross")
    plotting.reference_slope(ax, bias["n_dates"].iloc[0], bias["exercise_date_bias"].iloc[0],
                             -1.0, "slope $-1$", offset=2.2)
    ax.legend(loc="lower left")
    plotting.save(fig, FIGURES / "m4_bias_decomposition.png",
                  caption=f"{P.label()}; {N_PATHS:,} paths, cubic polynomial basis, "
                          f"out-of-sample policy evaluation, seed {SEED}.")

    # ---- Figure: in-sample vs out-of-sample ---------------------------
    berm_bias = io["bermudan_benchmark"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    for i, deg in enumerate((3, 10)):
        d = io[io["degree"] == deg].sort_values("n_paths")
        ax.plot(d["n_paths"], d["in_sample_mean"], marker="o", ms=5,
                color=plotting.SLOTS[i], label=f"in-sample, degree {deg}")
        ax.plot(d["n_paths"], d["out_of_sample_mean"], marker="s", ms=5, ls=(0, (3, 2)),
                color=plotting.SLOTS[i], label=f"out-of-sample, degree {deg}")
    ax.axhline(berm_bias, color=plotting.INK, lw=1.6, ls=(0, (4, 3)))
    plotting.annotate(ax, io["n_paths"].max(), berm_bias,
                      f"exact Bermudan {berm_bias:.4f}", dx=-4, dy=6, ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("paths"); ax.set_ylabel("mean estimate over seeds")
    ax.set_title("In-sample fitting biases the price upward")
    ax.legend(loc="upper right", fontsize=7.5)

    ax = axes[1]
    for i, deg in enumerate((3, 10)):
        d = io[io["degree"] == deg].sort_values("n_paths")
        ax.errorbar(d["n_paths"], d["foresight_bias"], yerr=1.96 * d["foresight_bias_se"],
                    marker="o", ms=5, lw=2.0, capsize=3, color=plotting.SLOTS[i],
                    label=f"degree {deg} ({deg+1} basis functions)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("paths"); ax.set_ylabel("in-sample minus out-of-sample")
    ax.set_title("Foresight bias scales with parameters per path")
    plotting.reference_slope(ax, io["n_paths"].min(),
                             io[io["degree"] == 10]["foresight_bias"].max(),
                             -1.0, "slope $-1$", offset=2.0)
    ax.legend(loc="lower left")
    worst = io.loc[io["foresight_bias"].idxmax()]
    plotting.save(fig, FIGURES / "m4_in_vs_out_of_sample.png",
                  caption=f"{P.label()}; {N_DATES_BIAS} exercise dates, polynomial basis, "
                          f"40-400 independent repetitions per point. Worst case "
                          f"({int(worst['n_paths'])} paths, degree {int(worst['degree'])}): "
                          f"in-sample overstates by {worst['foresight_bias']:.4f}, "
                          f"{100*worst['in_sample_bias']/berm_bias:.1f}% above the true value.")

    # ---- Figure: basis study ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    for i, b in enumerate(("poly", "laguerre", "chebyshev")):
        d = bs_[bs_["basis"] == b]
        ax.plot(d["degree"], d["abs_deviation"], marker="o", ms=5,
                color=plotting.SLOTS[i], label=b)
    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree"); ax.set_ylabel("|LSM - exact Bermudan|")
    ax.set_title("Accuracy saturates quickly in the basis degree")
    ax.legend(loc="upper left")

    ax = axes[1]
    for i, b in enumerate(("poly", "laguerre", "chebyshev")):
        d = bs_[bs_["basis"] == b]
        ax.plot(d["degree"], d["design_condition_number"], marker="o", ms=5,
                color=plotting.SLOTS[i], label=b)
    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree"); ax.set_ylabel("condition number of the design matrix")
    ax.set_title("...but conditioning does not")
    ax.legend(loc="upper left")
    plotting.save(fig, FIGURES / "m4_basis_study.png",
                  caption=f"{P.label()}; {N_PATHS:,} paths, {N_DATES} exercise dates, "
                          f"out-of-sample. Condition numbers computed on the "
                          f"in-the-money cross-section at t = T/2.")

    # ---- Console -------------------------------------------------------
    h_in = head[(head["estimator"] == "lsm_in_sample") & (head["benchmark_name"] == "bermudan_50_crr")].iloc[0]
    h_out = head[(head["estimator"] == "lsm_out_of_sample") & (head["benchmark_name"] == "bermudan_50_crr")].iloc[0]
    h_out_am = head[(head["estimator"] == "lsm_out_of_sample") & (head["benchmark_name"] == "american_crr")].iloc[0]
    print("== Milestone 4 ==")
    print(f"American CRR N={CRR_STEPS:,}          : {am_ref:.6f}")
    print(f"American CN-PSOR M=N=3200         : {cn_ref:.6f}")
    print(f"Bermudan (50 dates, exact lattice): {berm:.6f}")
    print(f"LSM in-sample   : {h_in['price']:.6f} +- {h_in['std_error']:.6f} "
          f"CI [{h_in['ci_low']:.6f}, {h_in['ci_high']:.6f}]  dev vs Bermudan "
          f"{h_in['deviation']:+.6f} ({h_in['deviation_in_se']:+.2f} SE)")
    print(f"LSM out-of-sample: {h_out['price']:.6f} +- {h_out['std_error']:.6f} "
          f"CI [{h_out['ci_low']:.6f}, {h_out['ci_high']:.6f}]  dev vs Bermudan "
          f"{h_out['deviation']:+.6f} ({h_out['deviation_in_se']:+.2f} SE)")
    print(f"   ... vs the American value      : {h_out_am['deviation']:+.6f}")
    print(f"bias split at n=50: exercise dates {bias.loc[bias['n_dates']==50,'exercise_date_bias'].iloc[0]:+.6f}, "
          f"regression {bias.loc[bias['n_dates']==50,'regression_and_sampling_error'].iloc[0]:+.6f}")
    print("in-sample vs out-of-sample (25 dates, polynomial basis):")
    print(f"   exact Bermudan target: {io['bermudan_benchmark'].iloc[0]:.6f}")
    print(f"   {'paths':>8} {'deg':>4} {'in-sample':>10} {'out-sample':>11} "
          f"{'foresight':>10} {'+-':>8} {'SE honesty in/out':>20}")
    for _, row in io.iterrows():
        print(f"   {int(row['n_paths']):8d} {int(row['degree']):4d} "
              f"{row['in_sample_mean']:10.5f} {row['out_of_sample_mean']:11.5f} "
              f"{row['foresight_bias']:+10.5f} {row['foresight_bias_se']:8.5f} "
              f"{row['in_sample_realised_sd']/row['in_sample_reported_se']:9.2f} "
              f"{row['out_of_sample_realised_sd']/row['out_of_sample_reported_se']:9.2f}")

    best = bs_.loc[bs_["abs_deviation"].idxmin()]
    print(f"best basis/degree : {best['basis']} deg {int(best['degree'])} "
          f"(|dev| {best['abs_deviation']:.2e}); cond at deg 10: "
          + ", ".join(f"{b}={bs_[(bs_['basis']==b)&(bs_['degree']==10)]['design_condition_number'].iloc[0]:.1e}"
                      for b in ('poly', 'laguerre', 'chebyshev')))
    print("ITM filter (deviation from Bermudan):")
    for _, row in itm.iterrows():
        print(f"   itm_only={str(row['itm_only']):5s} degree={int(row['degree'])}: "
              f"{row['deviation_from_bermudan']:+.6f} (SE {row['std_error']:.6f})")


if __name__ == "__main__":
    main()
