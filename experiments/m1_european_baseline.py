"""Milestone 1: European analytic baseline and CRR lattice validation.

Produces
--------
results/m1_bs_reference.csv        Analytic European prices and Greeks per regime.
results/m1_binomial_convergence.csv  CRR European error vs step count (+ observed order).
results/m1_american_convergence.csv  CRR American put value vs step count.
figures/m1_binomial_convergence.png  Log-log error decay with an O(1/N) guide.
figures/m1_american_convergence.png  American put value and early-exercise premium.

Run:  python experiments/m1_european_baseline.py
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
from amopt.binomial import crr_price, crr_price_averaged  # noqa: E402
from amopt.black_scholes import bs_greeks, bs_price  # noqa: E402
from amopt.config import REGIME_NOTES, REGIMES  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

N_GRID = np.array([25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800])


def analytic_reference_table() -> pd.DataFrame:
    """Closed-form European prices and Greeks for every parameter regime."""
    rows = []
    for name, p in REGIMES.items():
        for kind in ("call", "put"):
            g = bs_greeks(p.S0, p.K, p.T, p.r, p.sigma, p.q, kind)
            rows.append(
                {
                    "regime": name,
                    "note": REGIME_NOTES[name],
                    "kind": kind,
                    **p.as_dict(),
                    **{k: float(v) for k, v in g.items()},
                }
            )
    return pd.DataFrame(rows)


def binomial_convergence(p, kind="put") -> pd.DataFrame:
    """European CRR error against the analytic price, with wall-clock cost."""
    exact = float(bs_price(p.S0, p.K, p.T, p.r, p.sigma, p.q, kind))
    rows = []
    for N in N_GRID:
        t0 = time.perf_counter()
        v = crr_price(p.S0, p.K, p.T, p.r, p.sigma, int(N), p.q, kind, "european")
        dt = time.perf_counter() - t0
        rows.append(
            {
                "N": int(N),
                "price": v,
                "exact": exact,
                "abs_error": abs(v - exact),
                "rel_error": abs(v - exact) / exact,
                "runtime_s": dt,
            }
        )
    df = pd.DataFrame(rows)
    # Observed local order between consecutive refinements: log(e_i/e_{i+1})/log(N_{i+1}/N_i)
    df["observed_order"] = np.nan
    e, N = df["abs_error"].to_numpy(), df["N"].to_numpy(float)
    df.loc[1:, "observed_order"] = np.log(e[:-1] / e[1:]) / np.log(N[1:] / N[:-1])
    return df


def american_convergence(p) -> pd.DataFrame:
    """American put value vs step count, plus the early-exercise premium."""
    euro_exact = float(bs_price(p.S0, p.K, p.T, p.r, p.sigma, p.q, "put"))
    rows = []
    for N in N_GRID:
        t0 = time.perf_counter()
        v = crr_price(p.S0, p.K, p.T, p.r, p.sigma, int(N), p.q, "put", "american")
        dt = time.perf_counter() - t0
        rows.append(
            {
                "N": int(N),
                "american_put": v,
                "european_put_exact": euro_exact,
                "early_exercise_premium": v - euro_exact,
                "runtime_s": dt,
            }
        )
    df = pd.DataFrame(rows)
    df["delta_vs_previous"] = df["american_put"].diff().abs()
    return df


def main() -> None:
    plotting.use_style()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    p = REGIMES["base"]

    ref = analytic_reference_table()
    ref.to_csv(RESULTS / "m1_bs_reference.csv", index=False)

    conv_put = binomial_convergence(p, "put")
    conv_call = binomial_convergence(p, "call")
    conv = pd.concat(
        [conv_put.assign(kind="put"), conv_call.assign(kind="call")], ignore_index=True
    )
    conv.to_csv(RESULTS / "m1_binomial_convergence.csv", index=False)

    amer = american_convergence(p)
    amer.to_csv(RESULTS / "m1_american_convergence.csv", index=False)

    # ---- Lattice put-call parity residual (an exact invariant) ----------
    parity = []
    for N in (50, 200, 800, 3200):
        c = crr_price(p.S0, p.K, p.T, p.r, p.sigma, N, p.q, "call", "european")
        pu = crr_price(p.S0, p.K, p.T, p.r, p.sigma, N, p.q, "put", "european")
        lhs = c - pu
        rhs = p.S0 * np.exp(-p.q * p.T) - p.K * np.exp(-p.r * p.T)
        parity.append({"N": N, "residual": abs(lhs - rhs)})
    parity = pd.DataFrame(parity)
    parity.to_csv(RESULTS / "m1_lattice_parity.csv", index=False)

    # ---- Figure 1: European convergence and the odd/even oscillation ----
    # The put and call error curves coincide to machine precision because the
    # lattice satisfies put-call parity exactly, so only the put is plotted.
    N_dense = np.arange(60, 141)
    exact_put = float(bs_price(p.S0, p.K, p.T, p.r, p.sigma, p.q, "put"))
    signed = np.array(
        [crr_price(p.S0, p.K, p.T, p.r, p.sigma, int(N), p.q, "put", "european") - exact_put
         for N in N_dense]
    )
    averaged = np.array(
        [crr_price_averaged(p.S0, p.K, p.T, p.r, p.sigma, int(N), p.q, "put", "european") - exact_put
         for N in N_dense]
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    d = conv[conv["kind"] == "put"]
    ax.plot(d["N"], d["abs_error"], marker="o", ms=5,
            color=plotting.color("binomial"), label="CRR European put")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("time steps $N$"); ax.set_ylabel("absolute error vs Black-Scholes")
    ax.set_title("CRR European price converges at $O(N^{-1})$")
    plotting.reference_slope(ax, d["N"].iloc[0], d["abs_error"].iloc[0], -1.0, "slope $-1$")
    ax.legend(loc="lower left")

    ax = axes[1]
    ax.axhline(0.0, color=plotting.GRID, lw=1.0)
    ax.plot(N_dense, signed, lw=1.4, color=plotting.color("binomial"),
            label="single lattice, $N$")
    ax.plot(N_dense, averaged, lw=1.8, color=plotting.PALETTE["orange"],
            label=r"averaged, $(V_N + V_{N+1})/2$")
    ax.set_xlabel("time steps $N$"); ax.set_ylabel("signed error vs Black-Scholes")
    ax.set_title("Odd/even oscillation and its cancellation")
    ax.legend(loc="upper right")
    plotting.save(
        fig, FIGURES / "m1_binomial_convergence.png",
        caption=f"Base case: {p.label()}. Fitted log-log slope (put) "
                f"{np.polyfit(np.log(d['N']), np.log(d['abs_error']), 1)[0]:.3f}; "
                f"averaging cuts the mean |error| over N in [60,140] by "
                f"{np.mean(np.abs(signed)) / np.mean(np.abs(averaged)):.1f}x.",
    )
    slope_put = float(np.polyfit(np.log(d["N"]), np.log(d["abs_error"]), 1)[0])
    slope_call = float(np.polyfit(np.log(conv[conv["kind"] == "call"]["N"]),
                                  np.log(conv[conv["kind"] == "call"]["abs_error"]), 1)[0])
    osc_gain = float(np.mean(np.abs(signed)) / np.mean(np.abs(averaged)))

    # ---- Figure 2: American put convergence ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    ax.plot(amer["N"], amer["american_put"], marker="o", ms=4,
            color=plotting.color("binomial"), label="CRR American put")
    ax.axhline(amer["european_put_exact"].iloc[0], color=plotting.color("black_scholes"),
               lw=1.6, ls=(0, (4, 3)), label="European put (analytic)")
    ax.set_xscale("log")
    ax.set_xlabel("time steps $N$"); ax.set_ylabel("price")
    ax.set_title("American put value stabilises with lattice refinement")
    ax.legend(loc="center right")
    plotting.annotate(ax, amer["N"].iloc[-1], amer["american_put"].iloc[-1],
                      f"{amer['american_put'].iloc[-1]:.4f}", dx=-6, dy=-14, ha="right")

    ax = axes[1]
    ax.plot(amer["N"][1:], amer["delta_vs_previous"][1:], marker="o", ms=4,
            color=plotting.color("binomial"), label="|price change vs previous $N$|")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("time steps $N$"); ax.set_ylabel("successive difference")
    ax.set_title("Cauchy self-consistency of the lattice")
    ax.legend(loc="lower left")
    plotting.save(
        fig, FIGURES / "m1_american_convergence.png",
        caption=f"Base case: {p.label()}. Early-exercise premium at N=12800: "
                f"{amer['early_exercise_premium'].iloc[-1]:.4f}.",
    )

    # ---- Console summary ------------------------------------------------
    print("== Milestone 1 ==")
    print(f"base case: {p.label()}")
    print(f"European put (analytic)      : {conv_put['exact'].iloc[0]:.8f}")
    print(f"European put CRR N=12800     : {conv_put['price'].iloc[-1]:.8f} "
          f"(abs err {conv_put['abs_error'].iloc[-1]:.2e})")
    print(f"fitted log-log slope put/call: {slope_put:.3f} / {slope_call:.3f}")
    print(f"max lattice put-call parity residual: {parity['residual'].max():.2e}")
    print(f"odd/even averaging error reduction  : {osc_gain:.1f}x")
    print(f"American put CRR N=12800     : {amer['american_put'].iloc[-1]:.8f}")
    print(f"early-exercise premium       : {amer['early_exercise_premium'].iloc[-1]:.6f}")
    print(f"wrote {RESULTS}/m1_*.csv and {FIGURES}/m1_*.png")


if __name__ == "__main__":
    main()
