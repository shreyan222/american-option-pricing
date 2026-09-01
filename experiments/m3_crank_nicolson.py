"""Milestone 3: validation and characterisation of the Crank-Nicolson/PSOR solver.

Produces
--------
results/m3_cn_convergence.csv     Error vs M = N, European (vs analytic) and American (vs CRR).
results/m3_space_time_split.csv   Spatial and temporal order measured separately.
results/m3_solver_agreement.csv   Red-black PSOR vs lexicographic PSOR vs Brennan-Schwartz.
results/m3_omega_study.csv        PSOR iteration count and runtime vs the relaxation parameter.
results/m3_rannacher.csv          Effect of implicit start-up on gamma oscillation and error.
results/m3_regime_table.csv       Black-Scholes / CRR / CN-PSOR across all ten regimes.
figures/m3_cn_convergence.png
figures/m3_psor_efficiency.png
figures/m3_rannacher.png
figures/m3_value_and_boundary.png

Run:  python experiments/m3_crank_nicolson.py
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
from amopt.binomial import crr_price  # noqa: E402
from amopt.black_scholes import bs_put  # noqa: E402
from amopt.config import REGIME_NOTES, REGIMES  # noqa: E402
from amopt.crank_nicolson import solve_pde  # noqa: E402
from amopt.perpetual import perpetual_put_boundary  # noqa: E402

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
P = REGIMES["base"]

#: Independent lattice benchmark for the American put in the base case.
CRR_STEPS = 40_000


def _slope(x, y):
    return float(np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0])


def cn_convergence(crr_ref: float) -> pd.DataFrame:
    exact_eu = float(bs_put(P.S0, P.K, P.T, P.r, P.sigma, P.q))
    rows = []
    for n in (100, 200, 400, 800, 1600, 3200):
        eu = solve_pde(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "european",
                       M=n, N=n, solver="direct")
        am = solve_pde(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "american", M=n, N=n)
        rows.append(
            {
                "M": n, "N": n, "dS": eu.dS, "dtau": eu.dtau,
                "european_price": eu.price, "european_abs_error": abs(eu.price - exact_eu),
                "european_runtime_s": eu.runtime_s,
                "american_price": am.price, "american_abs_error": abs(am.price - crr_ref),
                "american_runtime_s": am.runtime_s,
                "mean_psor_iterations": am.mean_iterations,
                "boundary_S_star_0": am.boundary_S[-1],
            }
        )
    return pd.DataFrame(rows)


def space_time_split() -> pd.DataFrame:
    """Measure the spatial and temporal orders separately, by *self*-convergence.

    Comparing against an external benchmark does not work here.  Refining only
    the time axis at ``M = 4000`` leaves a fixed spatial error of about
    ``2.5e-05``; the temporal error falls through that floor, and a naive fit
    against the CRR benchmark then reports a meaningless order.  (An earlier
    version of this study did exactly that and reported a temporal order of
    1.18 which was really the floor.)

    Instead each axis is refined against a reference computed on the *same*
    grid in the other axis, so the other axis's error cancels exactly:

    * temporal: ``|V(M=4000, N) - V(M=4000, N=12800)|``
    * spatial:  ``|V(M, N=4000) - V(M=6400, N=4000)|``
    """
    rows = []

    for label, kw in (
        ("cn_rannacher2", dict(theta=0.5, rannacher_steps=2)),
        ("cn_no_rannacher", dict(theta=0.5, rannacher_steps=0)),
        ("fully_implicit", dict(theta=1.0, rannacher_steps=0)),
    ):
        ref = solve_pde(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "american",
                        M=4000, N=12800, **kw).price
        for N in (100, 200, 400, 800, 1600):
            am = solve_pde(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "american",
                           M=4000, N=N, **kw)
            rows.append({"axis": "time", "scheme": label, "M": 4000, "N": N,
                         "h": am.dtau, "price": am.price,
                         "abs_error": abs(am.price - ref), "reference": ref,
                         "runtime_s": am.runtime_s})

    ref = solve_pde(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "american",
                    M=6400, N=4000).price
    for M in (200, 400, 800, 1600, 3200):
        am = solve_pde(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "american", M=M, N=4000)
        rows.append({"axis": "space", "scheme": "cn_rannacher2", "M": M, "N": 4000,
                     "h": am.dS, "price": am.price,
                     "abs_error": abs(am.price - ref), "reference": ref,
                     "runtime_s": am.runtime_s})

    df = pd.DataFrame(rows)
    df["local_order"] = np.nan
    for key, g in df.groupby(["axis", "scheme"]):
        g = g.sort_values("h", ascending=False)
        e, h = g["abs_error"].to_numpy(), g["h"].to_numpy()
        lo = np.full(len(g), np.nan)
        lo[1:] = np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])
        df.loc[g.index, "local_order"] = lo
    return df


def solver_agreement() -> pd.DataFrame:
    """All three LCP solvers on the same grid; the lexicographic one is slow by design."""
    rows = []
    for M in (100, 200, 400):
        for solver in ("psor", "psor_lex", "brennan_schwartz"):
            res = solve_pde(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "american",
                            M=M, N=M, solver=solver, tol=1e-12)
            rows.append({"M": M, "N": M, "solver": solver, "price": res.price,
                         "runtime_s": res.runtime_s,
                         "mean_iterations": res.mean_iterations})
    df = pd.DataFrame(rows)
    ref = df[df["solver"] == "brennan_schwartz"].set_index("M")["price"]
    df["abs_dev_from_exact_lcp"] = df.apply(lambda r: abs(r["price"] - ref[r["M"]]), axis=1)
    return df


def omega_study() -> pd.DataFrame:
    rows = []
    for M in (400, 800, 1600, 3200):
        for w in (0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 1.9):
            res = solve_pde(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "american",
                            M=M, N=M, omega=w, tol=1e-10)
            rows.append({"M": M, "omega": w, "price": res.price,
                         "mean_iterations": res.mean_iterations,
                         "total_iterations": int(res.iterations.sum()),
                         "runtime_s": res.runtime_s})
    return pd.DataFrame(rows)


def _tv_gamma(res, K, window=15.0) -> float:
    """Total variation of the numerical gamma within +-`window` of the strike.

    A clean gamma is smooth and unimodal there, so its total variation is small;
    Crank-Nicolson ringing off the payoff kink inflates it.
    """
    mask = np.abs(res.S - K) <= window
    gamma = np.gradient(np.gradient(res.values, res.dS), res.dS)
    return float(np.sum(np.abs(np.diff(gamma[mask]))))


def rannacher_study() -> pd.DataFrame:
    exact_eu = float(bs_put(P.S0, P.K, P.T, P.r, P.sigma, P.q))
    rows = []
    for N in (25, 50, 100, 200, 400, 800):
        for rs in (0, 2, 4):
            res = solve_pde(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "european",
                            M=800, N=N, rannacher_steps=rs, solver="direct")
            rows.append({"N": N, "rannacher_steps": rs, "price": res.price,
                         "abs_error": abs(res.price - exact_eu),
                         "tv_gamma_near_strike": _tv_gamma(res, P.K)})
    return pd.DataFrame(rows)


def regime_table() -> pd.DataFrame:
    rows = []
    for name, p in REGIMES.items():
        t0 = time.perf_counter()
        cn = solve_pde(p.S0, p.K, p.T, p.r, p.sigma, p.q, "put", "american", M=2000, N=2000)
        cn_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        lat = crr_price(p.S0, p.K, p.T, p.r, p.sigma, 20_000, p.q, "put", "american")
        lat_t = time.perf_counter() - t0
        eu = float(bs_put(p.S0, p.K, p.T, p.r, p.sigma, p.q))
        rows.append(
            {
                "regime": name, "note": REGIME_NOTES[name], **p.as_dict(),
                "european_bs": eu,
                "american_crr_20k": lat, "crr_runtime_s": lat_t,
                "american_cn_psor": cn.price, "cn_runtime_s": cn_t,
                "abs_diff_cn_vs_crr": abs(cn.price - lat),
                "early_exercise_premium": cn.price - eu,
                "S_star_0": cn.boundary_S[-1],
                "mean_psor_iterations": cn.mean_iterations,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    plotting.use_style()
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)

    crr_ref = crr_price(P.S0, P.K, P.T, P.r, P.sigma, CRR_STEPS, P.q, "put", "american")

    conv = cn_convergence(crr_ref); conv.to_csv(RESULTS / "m3_cn_convergence.csv", index=False)
    split = space_time_split(); split.to_csv(RESULTS / "m3_space_time_split.csv", index=False)
    agree = solver_agreement(); agree.to_csv(RESULTS / "m3_solver_agreement.csv", index=False)
    om = omega_study(); om.to_csv(RESULTS / "m3_omega_study.csv", index=False)
    ran = rannacher_study(); ran.to_csv(RESULTS / "m3_rannacher.csv", index=False)
    reg = regime_table(); reg.to_csv(RESULTS / "m3_regime_table.csv", index=False)

    eu_slope = _slope(conv["M"], conv["european_abs_error"])
    am_slope = _slope(conv["M"], conv["american_abs_error"])
    sp = split[split["axis"] == "space"]
    tm = split[(split["axis"] == "time") & (split["scheme"] == "cn_rannacher2")]
    imp = split[(split["axis"] == "time") & (split["scheme"] == "fully_implicit")]
    sp_slope = _slope(sp["M"], sp["abs_error"])
    tm_slope = _slope(tm["N"], tm["abs_error"])
    imp_slope = _slope(imp["N"], imp["abs_error"])

    # ---- Figure: convergence ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    ax.plot(conv["M"], conv["european_abs_error"], marker="o", ms=5,
            color=plotting.PALETTE["blue"], label="European (vs Black-Scholes)")
    ax.plot(conv["M"], conv["american_abs_error"], marker="s", ms=5,
            color=plotting.color("cn_psor"), label=f"American (vs CRR $N$={CRR_STEPS:,})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("grid size $M = N$"); ax.set_ylabel("absolute error")
    ax.set_title("Crank-Nicolson error decay")
    plotting.reference_slope(ax, conv["M"].iloc[0], conv["european_abs_error"].iloc[0], -2.0,
                             "slope $-2$")
    ax.legend(loc="lower left")

    ax = axes[1]
    ax.plot(sp["M"], sp["abs_error"], marker="o", ms=5, color=plotting.PALETTE["blue"],
            label=rf"space, $\Delta S$ (order {-sp_slope:.2f})")
    ax.plot(tm["N"], tm["abs_error"], marker="s", ms=5, color=plotting.color("cn_psor"),
            label=rf"time, CN (order {-tm_slope:.2f})")
    ax.plot(imp["N"], imp["abs_error"], marker="^", ms=5, color=plotting.PALETTE["aqua"],
            label=rf"time, fully implicit (order {-imp_slope:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("refined axis size ($M$ or $N$)")
    ax.set_ylabel("self-convergence error")
    ax.set_title("The American constraint costs CN its second-order time accuracy")
    ax.legend(loc="lower left")
    plotting.save(fig, FIGURES / "m3_cn_convergence.png",
                  caption=f"{P.label()}. Left: error vs a fixed benchmark with M = N. "
                          f"Right: each axis refined against a reference on the same grid in "
                          f"the other axis, so the other axis's error cancels. Measured orders: "
                          f"space {-sp_slope:.2f}, time (CN) {-tm_slope:.2f}, "
                          f"time (implicit) {-imp_slope:.2f}.")

    # ---- Figure: PSOR efficiency --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    for i, M in enumerate(sorted(om["M"].unique())):
        d = om[om["M"] == M]
        ax.plot(d["omega"], d["mean_iterations"], marker="o", ms=4,
                color=plotting.SLOTS[i], label=f"$M=N={M}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"relaxation parameter $\omega$")
    ax.set_ylabel("mean PSOR sweeps per time step")
    ax.set_title(r"Optimal $\omega$ drifts upward as the grid refines")
    ax.legend(loc="upper left")

    ax = axes[1]
    best = om.loc[om.groupby("M")["mean_iterations"].idxmin()]
    ax.plot(conv["M"], conv["mean_psor_iterations"], marker="o", ms=4,
            color=plotting.color("cn_psor"), label=r"$\omega = 1.2$ (default)")
    ax.plot(best["M"], best["mean_iterations"], marker="s", ms=4,
            color=plotting.PALETTE["violet"], label=r"best $\omega$ on the sweep")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("grid size $M = N$"); ax.set_ylabel("mean PSOR sweeps per time step")
    ax.set_title("Iteration count grows with refinement")
    ax.legend(loc="upper left")
    bestrow = best.set_index("M")["omega"].to_dict()
    plotting.save(fig, FIGURES / "m3_psor_efficiency.png",
                  caption="Best omega by grid size: "
                          + ", ".join(f"M={k}: {v:g}" for k, v in sorted(bestrow.items()))
                          + ". Prices agree across all omega to "
                          f"{om.groupby('M')['price'].apply(lambda s: s.max()-s.min()).max():.1e}.")

    # ---- Figure: Rannacher --------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax, col, ttl, ylab in (
        (axes[0], "tv_gamma_near_strike", "Gamma ringing off the payoff kink",
         r"total variation of $\Gamma$ within $\pm 15$ of $K$"),
        (axes[1], "abs_error", "Price error", "absolute error vs Black-Scholes"),
    ):
        for i, rs in enumerate((0, 2, 4)):
            d = ran[ran["rannacher_steps"] == rs]
            ax.plot(d["N"], d[col], marker="o", ms=4, color=plotting.SLOTS[i],
                    label=f"{rs} implicit start-up steps")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("time steps $N$ ($M$ = 800)"); ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.legend(loc="upper right")
    r0 = ran[(ran["N"] == 25) & (ran["rannacher_steps"] == 0)].iloc[0]
    r2 = ran[(ran["N"] == 25) & (ran["rannacher_steps"] == 2)].iloc[0]
    plotting.save(fig, FIGURES / "m3_rannacher.png",
                  caption=f"European put, {P.label()}. At N=25 two implicit start-up steps cut "
                          f"gamma total variation by {r0['tv_gamma_near_strike']/r2['tv_gamma_near_strike']:.0f}x "
                          f"and the price error by {r0['abs_error']/r2['abs_error']:.0f}x.")

    # ---- Figure: value function and boundary ---------------------------
    res = solve_pde(P.S0, P.K, P.T, P.r, P.sigma, P.q, "put", "american", M=3200, N=3200)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    m = res.S <= 200.0
    ax.plot(res.S[m], np.maximum(P.K - res.S[m], 0.0), lw=1.6, ls=(0, (4, 3)),
            color=plotting.INK_MUTED, label="payoff $(K-S)^+$")
    ax.plot(res.S[m], bs_put(res.S[m], P.K, P.T, P.r, P.sigma, P.q), lw=1.8,
            color=plotting.PALETTE["blue"], label="European put (analytic)")
    ax.plot(res.S[m], res.values[m], lw=2.0, color=plotting.color("cn_psor"),
            label="American put (CN-PSOR)")
    S_star = res.boundary_S[-1]
    ax.axvline(S_star, color=plotting.PALETTE["violet"], lw=1.2, ls=(0, (2, 2)))
    plotting.annotate(ax, S_star, 45.0, f"$S^*(0)={S_star:.2f}$", dx=6, dy=0)
    ax.set_xlabel("$S$"); ax.set_ylabel("value")
    ax.set_title("Value functions and the exercise boundary at $t=0$")
    ax.legend(loc="upper right")

    ax = axes[1]
    t = res.boundary_t
    ax.plot(t, res.boundary_S, lw=2.0, color=plotting.color("cn_psor"),
            label=r"$S^*(t)$, CN-PSOR")
    S_inf = perpetual_put_boundary(P.K, P.r, P.sigma, P.q)
    ax.axhline(S_inf, color=plotting.color("black_scholes"), lw=1.4, ls=(0, (4, 3)),
               label=rf"perpetual limit $S_\infty={S_inf:.2f}$")
    ax.axhline(P.K, color=plotting.INK_MUTED, lw=1.0, label=f"strike $K={P.K:g}$")
    ax.set_xlabel("$t$ (years)"); ax.set_ylabel(r"$S^*(t)$")
    ax.set_ylim(68.0, 103.0)
    ax.set_title("Early-exercise boundary, recovered not imposed")
    ax.legend(loc="lower right")

    # Inset: the staircase is one cell tall (dS = 0.125 here), invisible on the
    # full axis, so zoom in to show what the smooth-pasting refinement removes.
    axin = ax.inset_axes([0.10, 0.44, 0.40, 0.40])
    zoom = t <= 0.12
    axin.step(t[zoom], res.boundary_S_raw[zoom], where="mid", lw=1.2,
              color=plotting.INK_MUTED, label="raw")
    axin.plot(t[zoom], res.boundary_S[zoom], lw=1.6, color=plotting.color("cn_psor"),
              label="refined")
    axin.set_title(r"zoom: $\Delta S$ staircase", fontsize=7.5, pad=3)
    axin.tick_params(labelsize=6.5)
    axin.grid(True, lw=0.5)
    axin.legend(fontsize=6.5, loc="lower right")
    plotting.save(fig, FIGURES / "m3_value_and_boundary.png",
                  caption=f"{P.label()}, M = N = 3200. "
                          f"S*(0) = {S_star:.4f}, S*(T) = {res.boundary_S[0]:.4f} = K.")

    # ---- Console summary -----------------------------------------------
    print("== Milestone 3 ==")
    print(f"CRR benchmark (N={CRR_STEPS:,})       : {crr_ref:.8f}")
    print(f"CN-PSOR M=N=3200                  : {conv['american_price'].iloc[-1]:.8f} "
          f"(abs dev {conv['american_abs_error'].iloc[-1]:.2e})")
    print(f"M=N sweep order: European {-eu_slope:.3f}, American {-am_slope:.3f} "
          f"(both spatially dominated)")
    print(f"self-convergence: spatial {-sp_slope:.3f}, temporal CN {-tm_slope:.3f}, "
          f"temporal fully-implicit {-imp_slope:.3f}")
    print(f"max |PSOR - Brennan-Schwartz|     : {agree['abs_dev_from_exact_lcp'].max():.2e}")
    print(f"max price spread over all omega   : "
          f"{om.groupby('M')['price'].apply(lambda s: s.max()-s.min()).max():.2e}")
    print(f"best omega by M                   : {bestrow}")
    print(f"Rannacher at N=25: TV(gamma) {r0['tv_gamma_near_strike']:.5f} -> "
          f"{r2['tv_gamma_near_strike']:.5f}, error {r0['abs_error']:.2e} -> {r2['abs_error']:.2e}")
    print(f"max |CN - CRR| across 10 regimes  : {reg['abs_diff_cn_vs_crr'].max():.2e} "
          f"({reg.loc[reg['abs_diff_cn_vs_crr'].idxmax(), 'regime']})")
    print(f"S*(0) at M=N=3200                 : {S_star:.5f} "
          f"(perpetual lower bound {S_inf:.5f}, strike {P.K:g})")
    print(f"PSOR/BS runtime at M=N=400        : "
          f"{agree[(agree['M']==400)].set_index('solver')['runtime_s'].to_dict()}")


if __name__ == "__main__":
    main()
