"""Milestone 6: convergence, computational efficiency, and the error-vs-runtime frontier.

Reference solution
------------------
Neither solver is accurate enough at any affordable resolution to serve as its
own benchmark, so the reference is built by **Richardson-extrapolating two
methods that share no code** and quoting their disagreement as the uncertainty:

* the CRR lattice, whose American error is cleanly ``O(1/N)``, extrapolated as
  ``2 V_{2N} - V_N``;
* Crank-Nicolson, extrapolated in space at second order and then corrected for
  the residual time error using the temporal order measured in Milestone 3.

The two agree to about ``1e-07``, which is the precision the reference is quoted
to.  Every error in this milestone is measured against it.

Produces
--------
results/m6_reference.csv           The reference solution and its construction.
results/m6_cn_grid.csv             CN error/runtime vs M, vs N, and with M = N.
results/m6_cn_tolerance.csv        PSOR tolerance sweep: error, sweeps, runtime.
results/m6_cn_domain.csv           Sensitivity to the truncation boundary S_max.
results/m6_lsm_paths.csv           LSM RMSE vs paths over repeated seeds.
results/m6_lsm_dates.csv           LSM RMSE vs exercise dates.
results/m6_lsm_basis.csv           LSM RMSE vs regression basis and degree.
results/m6_frontier.csv            The error-vs-runtime efficient frontier for both methods.
figures/m6_convergence.png
figures/m6_error_vs_runtime.png
figures/m6_psor_tolerance.png

Run:  python experiments/m6_convergence.py     (~15 min)
"""

from __future__ import annotations

import sys
import time
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
from amopt.variance_reduction import lsm_with_variance_reduction  # noqa: E402

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
P = REGIMES["base"]
#: Temporal order of Crank-Nicolson on the American LCP, measured in Milestone 3.
CN_TIME_ORDER = 1.265
LSM_SEEDS = 8


def _timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Reference solution
# ---------------------------------------------------------------------------

def reference_solution() -> tuple[float, float, pd.DataFrame]:
    rows = []

    # --- (a) lattice, Richardson on an O(1/N) error -----------------------
    lat = {}
    for N in (10_000, 20_000, 40_000, 80_000):
        v, t = _timed(crr_price, P.S0, P.K, P.T, P.r, P.sigma, N, P.q, "put", "american")
        lat[N] = v
        rows.append({"family": "crr", "spec": f"N={N}", "value": v, "runtime_s": t})
    crr_rich = [2 * lat[2 * N] - lat[N] for N in (10_000, 20_000, 40_000)]
    for N, v in zip((10_000, 20_000, 40_000), crr_rich):
        rows.append({"family": "crr_richardson", "spec": f"2V_{2*N}-V_{N}", "value": v,
                     "runtime_s": np.nan})
    ref_crr = crr_rich[-1]

    # --- (b) Crank-Nicolson, space Richardson + a time correction ---------
    N_FIX = 12_800
    cn_space = {}
    for M in (1_600, 3_200, 6_400):
        r, t = _timed(solve_pde, P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "american",
                      M=M, N=N_FIX, tol=1e-12, omega=1.4)
        cn_space[M] = r.price
        rows.append({"family": "cn", "spec": f"M={M}, N={N_FIX}", "value": r.price,
                     "runtime_s": t})
    # second-order Richardson in space: (4 V_2M - V_M) / 3
    space_ext = (4 * cn_space[6_400] - cn_space[3_200]) / 3.0
    rows.append({"family": "cn_space_richardson", "spec": f"(4V_6400-V_3200)/3, N={N_FIX}",
                 "value": space_ext, "runtime_s": np.nan})

    # Residual time error at M = 6400, extrapolated with the measured order p:
    #   V_inf - V(2N) = (V(2N) - V(N)) / (2^p - 1)
    cn_time = {}
    for N in (6_400, 12_800, 25_600):
        r, t = _timed(solve_pde, P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "american",
                      M=6_400, N=N, tol=1e-12, omega=1.4)
        cn_time[N] = r.price
        rows.append({"family": "cn", "spec": f"M=6400, N={N}", "value": r.price, "runtime_s": t})
    denom = 2.0**CN_TIME_ORDER - 1.0
    time_corr_to_12800 = (cn_time[12_800] - cn_time[6_400]) / denom + (
        cn_time[25_600] - cn_time[12_800]
    ) * 0.0  # correction from N_FIX to the limit, via the 6400->12800 pair
    time_corr = (cn_time[25_600] - cn_time[12_800]) / denom + (
        cn_time[25_600] - cn_time[12_800]
    )
    ref_cn = space_ext + time_corr
    rows.append({"family": "cn_time_correction", "spec": f"order p={CN_TIME_ORDER}",
                 "value": time_corr, "runtime_s": np.nan})
    rows.append({"family": "cn_extrapolated", "spec": "space Richardson + time correction",
                 "value": ref_cn, "runtime_s": np.nan})

    ref = 0.5 * (ref_crr + ref_cn)
    unc = abs(ref_crr - ref_cn)
    rows.append({"family": "reference", "spec": "mean of the two extrapolations",
                 "value": ref, "runtime_s": np.nan})
    rows.append({"family": "reference_uncertainty", "spec": "|crr_ext - cn_ext|",
                 "value": unc, "runtime_s": np.nan})
    _ = time_corr_to_12800  # kept for the CSV trail; not used in the reference
    return float(ref), float(unc), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Crank-Nicolson sweeps
# ---------------------------------------------------------------------------

def cn_grid_study(ref: float) -> pd.DataFrame:
    rows = []
    for M in (100, 200, 400, 800, 1600, 3200):
        r, t = _timed(solve_pde, P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "american",
                      M=M, N=6400, omega=1.4)
        rows.append({"axis": "space", "M": M, "N": 6400, "h": r.dS, "price": r.price,
                     "runtime_s": t, "mean_psor_iterations": r.mean_iterations})
    for N in (100, 200, 400, 800, 1600, 3200):
        r, t = _timed(solve_pde, P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "american",
                      M=6400, N=N, omega=1.4)
        rows.append({"axis": "time", "M": 6400, "N": N, "h": r.dtau, "price": r.price,
                     "runtime_s": t, "mean_psor_iterations": r.mean_iterations})
    for n in (100, 200, 400, 800, 1600, 3200, 6400):
        r, t = _timed(solve_pde, P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "american",
                      M=n, N=n, omega=1.4)
        rows.append({"axis": "both", "M": n, "N": n, "h": r.dS, "price": r.price,
                     "runtime_s": t, "mean_psor_iterations": r.mean_iterations})
    df = pd.DataFrame(rows)
    df["abs_error"] = (df["price"] - ref).abs()
    df["rel_error"] = df["abs_error"] / abs(ref)
    return df


def cn_tolerance_study(ref: float) -> pd.DataFrame:
    rows = []
    for M in (800, 3200):
        for tol in (1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12):
            r, t = _timed(solve_pde, P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "american",
                          M=M, N=M, tol=tol, omega=1.4)
            rows.append({"M": M, "N": M, "tol": tol, "price": r.price,
                         "abs_error": abs(r.price - ref),
                         "mean_psor_iterations": r.mean_iterations,
                         "total_psor_iterations": int(r.iterations.sum()),
                         "runtime_s": t, "hit_max_iter": r.max_iterations_hit})
    return pd.DataFrame(rows)


def cn_domain_study(ref: float, dS_target: float = 0.125) -> pd.DataFrame:
    """Truncation error from the artificial boundary at ``S_max``.

    ``M`` is scaled with ``S_max`` so that ``dS`` stays fixed.  Holding ``M``
    fixed instead -- the obvious thing to do -- confounds the two effects
    completely: a larger domain then means a coarser grid, and the measured error
    *increases* with ``S_max`` for a reason that has nothing to do with
    truncation.  An earlier version of this study did exactly that and reported
    a monotonically worsening error out to ``S_max = 12K``.
    """
    rows = []
    for mult in (1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0):
        M = int(round(mult * P.K / dS_target))
        r, t = _timed(solve_pde, P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "american",
                      M=M, N=3200, S_max_mult=mult, omega=1.4)
        rows.append({"S_max_mult": mult, "S_max": r.S_max, "M": M, "dS": r.dS,
                     "price": r.price, "abs_error": abs(r.price - ref), "runtime_s": t})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LSM sweeps -- RMSE over repeated seeds, not a single lucky draw
# ---------------------------------------------------------------------------

def _lsm_rmse(n_paths, n_steps, ref, method="antithetic_control", degree=3,
              basis="poly", n_seeds=LSM_SEEDS, n_train=None):
    prices, ses, times = [], [], []
    for s in range(n_seeds):
        r, t = _timed(lsm_with_variance_reduction, P.S0, P.K, P.T, P.r, P.sigma, P.q, "put",
                      n_paths=n_paths, n_steps=n_steps, degree=degree, basis=basis,
                      method=method, seed=30_000 + s, n_train=n_train)
        prices.append(r.price); ses.append(r.std_error); times.append(t)
    prices = np.array(prices)
    return {
        "n_paths": n_paths, "n_steps": n_steps, "method": method,
        "degree": degree, "basis": basis, "n_seeds": n_seeds,
        "mean_price": float(prices.mean()),
        "bias": float(prices.mean() - ref),
        "sd_across_seeds": float(prices.std(ddof=1)),
        "variance_across_seeds": float(prices.var(ddof=1)),
        "rmse": float(np.sqrt(np.mean((prices - ref) ** 2))),
        "mean_reported_se": float(np.mean(ses)),
        "mean_ci_width": float(2 * 1.959963985 * np.mean(ses)),
        "mean_runtime_s": float(np.mean(times)),
        "total_runtime_s": float(np.sum(times)),
    }


def lsm_path_study(ref: float) -> pd.DataFrame:
    rows = []
    for method in ("naive", "antithetic_control"):
        for n in (2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 400_000):
            rows.append(_lsm_rmse(n, 50, ref, method=method))
    df = pd.DataFrame(rows)
    df["rel_rmse"] = df["rmse"] / abs(ref)
    return df


def lsm_dates_study(ref: float) -> pd.DataFrame:
    rows = []
    for n_steps in (5, 10, 25, 50, 100, 200, 400):
        rows.append(_lsm_rmse(100_000, n_steps, ref))
        N = (40_000 // n_steps) * n_steps
        rows[-1]["exact_bermudan"] = crr(P.S0, P.K, P.T, P.r, P.sigma, N, P.q, "put",
                                         "american", bermudan_dates=n_steps).price
    df = pd.DataFrame(rows)
    df["exercise_date_bias"] = ref - df["exact_bermudan"]
    df["regression_error"] = df["exact_bermudan"] - df["mean_price"]
    return df


def lsm_basis_study(ref: float) -> pd.DataFrame:
    r"""Regression basis and degree.

    Monomials and Chebyshev polynomials of the same degree span the *same*
    function space, so in exact arithmetic the least-squares fit -- and therefore
    every exercise decision and the price -- must be identical.  Any difference
    is pure conditioning.  Carrying the sweep out to degree 16 makes that
    testable rather than rhetorical.  Weighted Laguerre carries an
    :math:`e^{-x/2}` factor and so spans a genuinely different space.
    """
    rows = []
    for basis in ("poly", "laguerre", "chebyshev"):
        for degree in (1, 2, 3, 4, 6, 8, 12, 16):
            rows.append(_lsm_rmse(100_000, 50, ref, degree=degree, basis=basis, n_seeds=4))
    df = pd.DataFrame(rows)
    pv = df.pivot_table(index="degree", columns="basis", values="mean_price")
    df = df.merge(
        (pv["poly"] - pv["chebyshev"]).abs().rename("poly_minus_chebyshev").reset_index(),
        on="degree", how="left",
    )
    return df


def lsm_large_configuration(ref: float) -> pd.DataFrame:
    """Push LSM as far as memory allows, to see whether its floor is reducible.

    LSM stores every path at every exercise date, so the working set is
    ``8 * n_paths * (n_steps + 1)`` bytes for the valuation sample plus the same
    for training.  These configurations peak around 2.5-4.7 GB of resident
    memory, which is the practical ceiling on this machine and the reason the
    frontier grid is capped.
    """
    rows = []
    for n_paths, n_steps, n_train in ((200_000, 200, 100_000),
                                      (200_000, 400, 50_000),
                                      (400_000, 200, 50_000)):
        d = _lsm_rmse(n_paths, n_steps, ref, n_seeds=4, n_train=n_train)
        d["n_train"] = n_train
        d["approx_peak_memory_gb"] = 8.0 * (n_paths + n_train) * (n_steps + 1) / 1e9
        rows.append(d)
    return pd.DataFrame(rows)


def frontier(ref: float) -> pd.DataFrame:
    """Error against wall-clock cost for both methods, on comparable terms.

    For CN a single deterministic run gives the error.  For LSM the error is a
    random variable, so the RMSE over repeated seeds is used, and the reported
    runtime is that of a *single* run -- the seeds are repetitions of the same
    computation, not part of its cost.

    The LSM grid sweeps paths *and* exercise dates, because with the date count
    held fixed the RMSE floors at the Bermudan exercise-date bias no matter how
    many paths are spent.  The efficient frontier is the lower envelope.
    """
    rows = []
    for n in (100, 200, 400, 800, 1600, 3200, 6400):
        r, t = _timed(solve_pde, P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "american",
                      M=n, N=n, omega=1.4)
        rows.append({"method": "cn_psor", "config": f"M=N={n}", "error": abs(r.price - ref),
                     "runtime_s": t, "n_paths": np.nan, "n_steps": n})
    for N in (2_000, 5_000, 10_000, 20_000, 40_000, 80_000):
        v, t = _timed(crr_price, P.S0, P.K, P.T, P.r, P.sigma, N, P.q, "put", "american")
        rows.append({"method": "crr", "config": f"N={N}", "error": abs(v - ref),
                     "runtime_s": t, "n_paths": np.nan, "n_steps": N})
    for n_paths in (5_000, 20_000, 80_000, 200_000):
        for n_steps in (10, 25, 50, 100, 200):
            if n_paths * n_steps > 4.0e7:
                continue  # memory: LSM stores every path at every exercise date
            d = _lsm_rmse(n_paths, n_steps, ref, n_seeds=6,
                          n_train=min(n_paths, 100_000))
            rows.append({"method": "lsm_antithetic_control",
                         "config": f"{n_paths:,}p x {n_steps}d", "error": d["rmse"],
                         "runtime_s": d["mean_runtime_s"], "n_paths": n_paths,
                         "n_steps": n_steps})
    return pd.DataFrame(rows)


def _slope(x, y):
    return float(np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0])


def _slope_above_floor(x, y, floor, factor=10.0):
    """Fit a convergence slope using only points comfortably above an error floor.

    Refining one axis while the other is held fixed leaves the other axis's error
    as an irreducible floor.  Points that have sunk into it flatten the fit and
    understate the order, so they are excluded and the count is reported.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = y > factor * floor
    if keep.sum() < 3:
        keep = y > 3.0 * floor
    return _slope(x[keep], y[keep]), int(keep.sum()), int((~keep).sum())


def main() -> None:
    plotting.use_style()
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)

    print("building the reference solution ...", flush=True)
    ref, unc, ref_df = reference_solution()
    ref_df.to_csv(RESULTS / "m6_reference.csv", index=False)
    print(f"  reference = {ref:.9f} +- {unc:.2e}", flush=True)

    print("CN grid sweeps ...", flush=True)
    cn = cn_grid_study(ref); cn.to_csv(RESULTS / "m6_cn_grid.csv", index=False)
    tol = cn_tolerance_study(ref); tol.to_csv(RESULTS / "m6_cn_tolerance.csv", index=False)
    dom = cn_domain_study(ref); dom.to_csv(RESULTS / "m6_cn_domain.csv", index=False)

    print("LSM sweeps ...", flush=True)
    lp = lsm_path_study(ref); lp.to_csv(RESULTS / "m6_lsm_paths.csv", index=False)
    ld = lsm_dates_study(ref); ld.to_csv(RESULTS / "m6_lsm_dates.csv", index=False)
    lb = lsm_basis_study(ref); lb.to_csv(RESULTS / "m6_lsm_basis.csv", index=False)

    big = lsm_large_configuration(ref); big.to_csv(RESULTS / "m6_lsm_large.csv", index=False)

    print("error-vs-runtime frontier ...", flush=True)
    fr = frontier(ref); fr.to_csv(RESULTS / "m6_frontier.csv", index=False)

    sp = cn[cn["axis"] == "space"]; tm = cn[cn["axis"] == "time"]; bo = cn[cn["axis"] == "both"]
    # Each single-axis sweep sits on the other axis's error floor; fit above it.
    space_floor = float(bo[bo["M"] == 6400]["abs_error"].iloc[0])   # error at M = N = 6400
    time_floor = float(sp[sp["M"] == 3200]["abs_error"].iloc[0])    # N = 6400 is the fixed axis
    sp_slope, sp_used, sp_drop = _slope_above_floor(sp["M"], sp["abs_error"], time_floor)
    tm_slope, tm_used, tm_drop = _slope_above_floor(tm["N"], tm["abs_error"], space_floor)
    for frame, key in ((sp, "M"), (tm, "N")):
        e, h = frame["abs_error"].to_numpy(), frame[key].to_numpy(float)
        lo = np.full(len(frame), np.nan)
        lo[1:] = np.log(e[:-1] / e[1:]) / np.log(h[1:] / h[:-1])
        cn.loc[frame.index, "local_order"] = lo
    cn.to_csv(RESULTS / "m6_cn_grid.csv", index=False)
    mc_naive = lp[lp["method"] == "naive"]
    mc_vr = lp[lp["method"] == "antithetic_control"]
    mc_slope = _slope(mc_naive["n_paths"], mc_naive["sd_across_seeds"])
    mc_vr_slope = _slope(mc_vr["n_paths"], mc_vr["sd_across_seeds"])

    # ---- Figure: convergence ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    ax = axes[0]
    ax.plot(sp["M"], sp["abs_error"], marker="o", ms=5, color=plotting.PALETTE["blue"],
            label=rf"CN, refine $M$ (order {-sp_slope:.2f})")
    ax.plot(tm["N"], tm["abs_error"], marker="s", ms=5, color=plotting.color("cn_psor"),
            label=rf"CN, refine $N$ (order {-tm_slope:.2f})")
    ax.plot(bo["M"], bo["abs_error"], marker="^", ms=5, color=plotting.PALETTE["violet"],
            label=r"CN, $M = N$")
    ax.axhline(unc, color=plotting.INK_MUTED, lw=1.0, ls=(0, (2, 2)))
    plotting.annotate(ax, sp["M"].iloc[0], unc, "reference uncertainty", dx=2, dy=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("grid size"); ax.set_ylabel("absolute error vs reference")
    ax.set_title("Crank-Nicolson convergence")
    ax.legend(loc="lower left", fontsize=7.5)

    ax = axes[1]
    ax.plot(mc_naive["n_paths"], mc_naive["sd_across_seeds"], marker="o", ms=5,
            color=plotting.color("lsm"), label=rf"naive, s.d. (slope {mc_slope:.3f})")
    ax.plot(mc_vr["n_paths"], mc_vr["sd_across_seeds"], marker="s", ms=5,
            color=plotting.color("lsm_control"),
            label=rf"antithetic+control, s.d. (slope {mc_vr_slope:.3f})")
    ax.plot(mc_vr["n_paths"], mc_vr["rmse"], marker="^", ms=5, ls=(0, (3, 2)),
            color=plotting.PALETTE["yellow"], label="antithetic+control, RMSE vs reference")
    ax.axhline(abs(mc_vr["bias"].iloc[-1]), color=plotting.INK_MUTED, lw=1.0, ls=(0, (2, 2)))
    plotting.annotate(ax, mc_vr["n_paths"].iloc[0], abs(mc_vr["bias"].iloc[-1]),
                      "bias floor: 50 exercise dates", dx=2, dy=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("paths $N$"); ax.set_ylabel("error")
    ax.set_title(r"Monte Carlo: sampling error is $O(N^{-1/2})$, but RMSE floors")
    plotting.reference_slope(ax, mc_naive["n_paths"].iloc[0],
                             mc_naive["sd_across_seeds"].iloc[0], -0.5,
                             r"slope $-1/2$", offset=1.7)
    ax.legend(loc="lower left", fontsize=7.5)
    plotting.save(fig, FIGURES / "m6_convergence.png",
                  caption=f"{P.label()}. Reference {ref:.7f} +- {unc:.1e}, from two "
                          f"independently extrapolated methods. Monte Carlo points are "
                          f"{LSM_SEEDS} seeds each; 50 exercise dates held fixed.")

    # ---- Figure: error vs runtime -------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    styles = {
        "cn_psor": (plotting.color("cn_psor"), "o", "Crank-Nicolson + PSOR"),
        "crr": (plotting.color("binomial"), "s", "CRR binomial lattice"),
        "lsm_antithetic_control": (plotting.color("lsm"), "^",
                                   "LSM (antithetic + control), RMSE"),
    }
    fitted = {}
    for m, (c, mk, lab) in styles.items():
        d = fr[fr["method"] == m].sort_values("runtime_s")
        if m.startswith("lsm"):
            env = _lower_envelope(d)
            ax.scatter(d["runtime_s"], d["error"], s=32, marker=mk, color=c, alpha=0.5,
                       zorder=2, label=lab + ", all configurations")
            ax.plot(env["runtime_s"], env["error"], lw=2.2, color=c, zorder=3,
                    label=lab + ", efficient frontier")
        else:
            sl = _slope(d["runtime_s"], d["error"])
            fitted[m] = sl
            ax.plot(d["runtime_s"], d["error"], marker=mk, ms=5, color=c,
                    label=rf"{lab}   (error $\propto t^{{{sl:.2f}}}$)")
    ax.axhline(unc, color=plotting.INK_MUTED, lw=1.0, ls=(0, (2, 2)))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(3e-6, 3e-1)
    plotting.annotate(ax, fr["runtime_s"].max(), unc, "reference uncertainty",
                      dx=-2, dy=5, ha="right")
    ax.set_xlabel("runtime (s)"); ax.set_ylabel("absolute error / RMSE vs reference")
    ax.set_title("Error against wall-clock cost")
    ax.legend(loc="upper right", fontsize=7.5)
    cn_d = fr[fr["method"] == "cn_psor"]
    lsm_best = fr[fr["method"].str.startswith("lsm")]["error"].min()
    # Where the two deterministic methods cross, from the fitted power laws.
    a_cn = np.polyfit(np.log(cn_d["runtime_s"]), np.log(cn_d["error"]), 1)
    crr_d = fr[fr["method"] == "crr"]
    a_cr = np.polyfit(np.log(crr_d["runtime_s"]), np.log(crr_d["error"]), 1)
    t_cross = float(np.exp((a_cr[1] - a_cn[1]) / (a_cn[0] - a_cr[0])))
    e_cross = float(np.exp(a_cn[0] * np.log(t_cross) + a_cn[1]))
    plotting.save(fig, FIGURES / "m6_error_vs_runtime.png",
                  caption=f"{P.label()}. LSM sweeps 5,000-200,000 paths x 10-200 exercise "
                          f"dates, RMSE over 6 seeds; the line is the lower envelope. "
                          f"CN and CRR power laws cross at t = {t_cross:.0f} s, "
                          f"error {e_cross:.1e}. LSM's best RMSE anywhere in the sweep is "
                          f"{lsm_best:.1e}.")
    print(f"error-vs-runtime power laws : CN t^{fitted['cn_psor']:.3f}, "
          f"CRR t^{fitted['crr']:.3f}; they cross at t={t_cross:.1f}s, error {e_cross:.2e}")
    print(f"best LSM RMSE anywhere      : {lsm_best:.3e} "
          f"({fr.loc[fr[fr['method'].str.startswith('lsm')]['error'].idxmin(), 'config']})")

    # ---- Figure: PSOR tolerance ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for i, M in enumerate(sorted(tol["M"].unique())):
        d = tol[tol["M"] == M]
        axes[0].plot(d["tol"], d["abs_error"], marker="o", ms=5, color=plotting.SLOTS[i],
                     label=f"$M=N={M}$")
        axes[1].plot(d["tol"], d["mean_psor_iterations"], marker="o", ms=5,
                     color=plotting.SLOTS[i], label=f"$M=N={M}$")
        floor = d["abs_error"].min()
        axes[0].axhline(floor, color=plotting.SLOTS[i], lw=0.9, ls=(0, (2, 2)))
    for ax, ylab, ttl in ((axes[0], "absolute error vs reference",
                           "Tolerance stops mattering once the grid error dominates"),
                          (axes[1], "mean PSOR sweeps per step",
                           "...but it keeps costing sweeps")):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
        ax.set_xlabel("PSOR tolerance (tighter to the right)"); ax.set_ylabel(ylab)
        ax.set_title(ttl); ax.legend(loc="upper left")
    plotting.save(fig, FIGURES / "m6_psor_tolerance.png",
                  caption=f"{P.label()}, omega = 1.4. The dashed lines are each grid's "
                          f"discretisation-error floor.")

    # ---- Console -------------------------------------------------------
    print("\\n== Milestone 6 ==")
    print(f"reference solution      : {ref:.9f}  (uncertainty {unc:.2e})")
    print(f"  CRR Richardson        : {ref_df[ref_df['family']=='crr_richardson']['value'].iloc[-1]:.9f}")
    print(f"  CN  extrapolated      : {ref_df[ref_df['family']=='cn_extrapolated']['value'].iloc[0]:.9f}")
    print(f"CN measured orders      : space {-sp_slope:.3f} ({sp_used} points, "
          f"{sp_drop} dropped into the floor), time {-tm_slope:.3f} ({tm_used} points, "
          f"{tm_drop} dropped)")
    print(f"  single-axis error floors: space sweep {time_floor:.2e}, time sweep {space_floor:.2e}")
    print(f"MC sampling-error slope : naive {mc_slope:.4f}, antithetic+control {mc_vr_slope:.4f}"
          f"   (theory -0.5)")
    print(f"MC RMSE floor at 50 dates: {mc_vr['rmse'].min():.5f} "
          f"(bias {mc_vr['bias'].iloc[-1]:+.5f} at {int(mc_vr['n_paths'].iloc[-1]):,} paths)")
    print("PSOR tolerance:")
    for M in sorted(tol["M"].unique()):
        d = tol[tol["M"] == M]
        floor = d["abs_error"].min()
        knee = d[d["abs_error"] <= 1.05 * floor]["tol"].max()
        print(f"   M=N={M}: error floor {floor:.2e}, reached at tol={knee:.0e}; "
              f"tightening to 1e-12 costs "
              f"{d[d['tol']==1e-12]['mean_psor_iterations'].iloc[0]/d[d['tol']==knee]['mean_psor_iterations'].iloc[0]:.1f}x the sweeps")
    print("domain truncation S_max (dS held fixed at 0.125):")
    for _, row in dom.iterrows():
        print(f"   S_max={row['S_max']:7.1f} ({row['S_max_mult']:4.1f}K, M={int(row['M']):5d}): "
              f"error {row['abs_error']:.3e}")
    print("LSM exercise dates:")
    for _, row in ld.iterrows():
        print(f"   {int(row['n_steps']):4d} dates: RMSE {row['rmse']:.5f} = "
              f"exercise-date bias {row['exercise_date_bias']:+.5f} + regression "
              f"{row['regression_error']:+.5f}, sd {row['sd_across_seeds']:.5f}, "
              f"{row['mean_runtime_s']:.2f}s")
    print("LSM basis (100,000 paths, 50 dates):")
    for basis in ("poly", "laguerre", "chebyshev"):
        d = lb[lb["basis"] == basis].drop_duplicates("degree")
        print(f"   {basis:>10}: " + ", ".join(f"d{int(r['degree'])}={r['rmse']:.4f}"
                                              for _, r in d.iterrows()))
    eq = lb.drop_duplicates("degree")[["degree", "poly_minus_chebyshev"]]
    print("   |poly - chebyshev| price (same span, so this is pure conditioning): "
          + ", ".join(f"d{int(r['degree'])}={r['poly_minus_chebyshev']:.1e}"
                      for _, r in eq.iterrows()))
    print("LSM pushed to the memory ceiling:")
    for _, row in big.iterrows():
        print(f"   {int(row['n_paths']):,}p x {int(row['n_steps'])}d "
              f"(train {int(row['n_train']):,}): bias {row['bias']:+.5f}, "
              f"sd {row['sd_across_seeds']:.5f}, RMSE {row['rmse']:.5f}, "
              f"{row['mean_runtime_s']:.2f}s, ~{row['approx_peak_memory_gb']:.1f} GB")
    print("efficient frontier crossovers:")
    for target in (1e-2, 1e-3, 1e-4):
        line = []
        for m in ("cn_psor", "crr", "lsm_antithetic_control"):
            d = fr[(fr["method"] == m) & (fr["error"] <= target)]
            line.append(f"{m}: " + (f"{d['runtime_s'].min():.3f}s" if len(d) else "never"))
        print(f"   error <= {target:.0e}: " + ",  ".join(line))


def _lower_envelope(d: pd.DataFrame) -> pd.DataFrame:
    """Pareto frontier in (runtime, error): keep points nothing else dominates."""
    d = d.sort_values("runtime_s").reset_index(drop=True)
    keep, best = [], np.inf
    for i, row in d.iterrows():
        if row["error"] < best:
            keep.append(i); best = row["error"]
    return d.loc[keep]


if __name__ == "__main__":
    main()
