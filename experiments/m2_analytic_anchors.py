"""Milestone 2: analytic anchors implied by the free-boundary formulation.

The formulation in `docs/01_formulation.md` predicts several exact statements
that any American solver must satisfy.  This script measures them.

Produces
--------
results/m2_perpetual_anchors.csv    Perpetual boundary/price across regimes.
results/m2_maturity_limit.csv       American put value vs T against the perpetual limit.
figures/m2_perpetual_limit.png      Convergence of the finite-maturity value to the perpetual.

Run:  python experiments/m2_analytic_anchors.py
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
from amopt.binomial import crr_price  # noqa: E402
from amopt.black_scholes import bs_put  # noqa: E402
from amopt.config import REGIMES  # noqa: E402
from amopt.perpetual import beta_minus, perpetual_put_boundary, perpetual_put_price  # noqa: E402

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
MATURITIES = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0])


def perpetual_table() -> pd.DataFrame:
    rows = []
    for name, p in REGIMES.items():
        if p.r <= 0.0:
            b_inf, v_inf, beta = 0.0, float(perpetual_put_price(p.S0, p.K, p.r, p.sigma, p.q)), np.nan
        else:
            beta = beta_minus(p.r, p.sigma, p.q)
            b_inf = perpetual_put_boundary(p.K, p.r, p.sigma, p.q)
            v_inf = float(perpetual_put_price(p.S0, p.K, p.r, p.sigma, p.q))
        rows.append(
            {
                "regime": name, **p.as_dict(),
                "beta_minus": beta,
                "S_inf": b_inf,
                "S_inf_over_K": b_inf / p.K,
                "perpetual_put": v_inf,
                "american_put_T": crr_price(p.S0, p.K, p.T, p.r, p.sigma, 4000, p.q, "put"),
                "european_put_T": float(bs_put(p.S0, p.K, p.T, p.r, p.sigma, p.q)),
            }
        )
    df = pd.DataFrame(rows)
    df["dominance_ok"] = df["american_put_T"] <= df["perpetual_put"] + 1e-8
    return df


def maturity_limit_table(p) -> pd.DataFrame:
    perp = float(perpetual_put_price(p.S0, p.K, p.r, p.sigma, p.q))
    rows = []
    for T in MATURITIES:
        # Keep the per-year step density roughly fixed so accuracy is comparable.
        N = int(np.clip(2000 * T, 2000, 40000))
        am = crr_price(p.S0, p.K, float(T), p.r, p.sigma, N, p.q, "put", "american")
        # |V_N - V_{N/2}| is a first-order proxy for the lattice discretisation
        # error at this maturity.  Beyond T ~ 50 the true gap to the perpetual
        # value falls below this floor, so gap comparisons there are noise.
        am_half = crr_price(p.S0, p.K, float(T), p.r, p.sigma, N // 2, p.q, "put", "american")
        eu = float(bs_put(p.S0, p.K, float(T), p.r, p.sigma, p.q))
        rows.append(
            {
                "T": float(T), "N_steps": N,
                "american_put": am, "european_put": eu,
                "early_exercise_premium": am - eu,
                "perpetual_put": perp,
                "gap_to_perpetual": perp - am,
                "fraction_of_perpetual": am / perp,
                "lattice_noise_floor": abs(am - am_half),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    plotting.use_style()
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    p = REGIMES["base"]

    perp = perpetual_table()
    perp.to_csv(RESULTS / "m2_perpetual_anchors.csv", index=False)
    mat = maturity_limit_table(p)
    mat.to_csv(RESULTS / "m2_maturity_limit.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    ax.plot(mat["T"], mat["american_put"], marker="o", ms=4,
            color=plotting.color("binomial"), label="American put (CRR)")
    ax.plot(mat["T"], mat["european_put"], marker="s", ms=4,
            color=plotting.PALETTE["orange"], label="European put (analytic)")
    ax.axhline(mat["perpetual_put"].iloc[0], color=plotting.color("black_scholes"),
               lw=1.6, ls=(0, (4, 3)), label="perpetual put (closed form)")
    ax.set_xscale("log")
    ax.set_xlabel("maturity $T$ (years)"); ax.set_ylabel("price")
    ax.set_title("American put rises to the perpetual value; European does not")
    ax.legend(loc="center left")

    ax = axes[1]
    ax.plot(mat["T"], mat["gap_to_perpetual"], marker="o", ms=4,
            color=plotting.color("binomial"), label=r"$V^{\rm perp} - V^{\rm amer}(T)$")
    ax.plot(mat["T"], mat["lattice_noise_floor"], marker="s", ms=4,
            color=plotting.PALETTE["orange"], label=r"lattice noise floor $|V_N - V_{N/2}|$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("maturity $T$ (years)"); ax.set_ylabel("gap to perpetual value")
    ax.set_title("Gap closes monotonically until it hits the lattice noise floor")
    ax.legend(loc="lower left")
    S_inf = perpetual_put_boundary(p.K, p.r, p.sigma, p.q)
    plotting.save(
        fig, FIGURES / "m2_perpetual_limit.png",
        caption=f"S0={p.S0:g}, K={p.K:g}, r={p.r:.0%}, sigma={p.sigma:.0%}, q={p.q:.0%}; "
                f"T varies. Perpetual boundary S_inf = {S_inf:.4f} "
                f"(= K*gamma/(1+gamma), gamma = 2r/sigma^2 = {2*p.r/p.sigma**2:.3f}); "
                f"perpetual value {mat['perpetual_put'].iloc[0]:.6f}.",
    )

    print("== Milestone 2 ==")
    print(f"beta_-                        : {beta_minus(p.r, p.sigma, p.q):.6f}")
    print(f"perpetual boundary S_inf      : {S_inf:.6f}")
    print(f"perpetual put value at S0=100 : {mat['perpetual_put'].iloc[0]:.6f}")
    am_1y = float(mat.loc[mat["T"] == 1.0, "american_put"].iloc[0])
    print(f"American put T=1  / T=200     : {am_1y:.6f} / {mat['american_put'].iloc[-1]:.6f}")
    print(f"fraction of perpetual at T=200: {mat['fraction_of_perpetual'].iloc[-1]:.4f}")
    print(f"dominance V_amer <= V_perp    : "
          f"{'holds in all 10 regimes' if perp['dominance_ok'].all() else 'VIOLATED'}")

    # The gap to the perpetual value must decrease in T -- but only while it is
    # resolvable above the lattice's own discretisation error.
    resolvable = mat["gap_to_perpetual"] > 10.0 * mat["lattice_noise_floor"]
    sub = mat[resolvable]
    mono = bool(np.all(np.diff(sub["gap_to_perpetual"]) < 0))
    print(f"gap decreasing where resolvable: {mono} "
          f"(T <= {sub['T'].max():g}; {int((~resolvable).sum())} of {len(mat)} maturities "
          f"are below the lattice noise floor)")
    unres = mat[~resolvable]
    if len(unres):
        print(f"  unresolvable maturities      : T = {list(unres['T'])}, "
              f"gap {unres['gap_to_perpetual'].min():.2e}-{unres['gap_to_perpetual'].max():.2e} "
              f"vs noise floor {unres['lattice_noise_floor'].min():.2e}-"
              f"{unres['lattice_noise_floor'].max():.2e}")


if __name__ == "__main__":
    main()
