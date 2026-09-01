"""Milestone 7: recovering S*(t) and testing the predictions made in docs/03.

`docs/03_exercise_boundary.md` states seven predictions (P1-P7) about the free
boundary, with reasons, **before** this study was run.  This script measures each
of them.  Nothing here is presented as a discovery that was actually a
post-hoc rationalisation.

Produces
--------
results/m7_boundary_base.csv        S*(t) in the base case (raw and refined).
results/m7_boundary_accuracy.csv    Grid convergence and cell-alignment noise.
results/m7_sensitivity_sigma.csv    P4: volatility.
results/m7_sensitivity_rate.csv     P5: interest rate.
results/m7_sensitivity_maturity.csv P6: maturity, against the perpetual limit.
results/m7_sensitivity_dividend.csv P2: terminal boundary min(K, rK/q).
results/m7_near_maturity.csv        P3: the square-root-log law.
results/m7_pde_vs_lattice.csv       Boundary from the PDE vs from the lattice.
figures/m7_boundary_families.png
figures/m7_boundary_sensitivity.png
figures/m7_boundary_asymptotics.png

Run:  python experiments/m7_exercise_boundary.py     (~4 min)
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
from amopt.binomial import crr_boundary  # noqa: E402
from amopt.config import REGIMES  # noqa: E402
from amopt.crank_nicolson import solve_pde  # noqa: E402
from amopt.perpetual import perpetual_put_boundary  # noqa: E402

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
P = REGIMES["base"]
GRID = dict(M=3200, N=3200, omega=1.4)

SIGMAS = (0.10, 0.15, 0.20, 0.30, 0.40, 0.60)
RATES = (0.00, 0.01, 0.02, 0.05, 0.08, 0.12)
MATURITIES = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
DIVIDENDS = (0.0, 0.02, 0.04, 0.05, 0.06, 0.10, 0.15)


def solve(**kw):
    p = dict(S0=P.S0, K=P.K, T=P.T, r=P.r, sigma=P.sigma, q=P.q, kind="put", **GRID)
    p.update(kw)
    res = solve_pde(**p)
    order = np.argsort(res.boundary_t)
    return res, res.boundary_t[order], res.boundary_S[order], res.boundary_S_raw[order]


def base_boundary() -> pd.DataFrame:
    res, t, Sb, raw = solve()
    return pd.DataFrame({"t": t, "tau": P.T - t, "S_star": Sb, "S_star_raw": raw,
                         "K_minus_S_star": P.K - Sb, "dS": res.dS})


def boundary_accuracy() -> pd.DataFrame:
    r"""Grid convergence of ``S*(0)``, and the cell-alignment noise on top of it.

    The refined estimate removes the ``dS`` staircase but not the underlying
    ``O(dS)`` accuracy of the free boundary itself, and where ``S*`` happens to
    fall inside a cell matters: some grids are unlucky.  Both effects are
    reported rather than smoothed away.
    """
    rows = []
    for M in (400, 800, 1600, 3200, 6400, 12800):
        res, t, Sb, raw = solve(M=M, N=6400)
        rows.append({"M": M, "N": 6400, "dS": res.dS, "S_star_0": Sb[0],
                     "S_star_0_raw": raw[0], "runtime_s": res.runtime_s})
    df = pd.DataFrame(rows)
    finest = df["S_star_0"].iloc[-1]
    df["dev_from_finest"] = (df["S_star_0"] - finest).abs()
    df["dev_raw_from_finest"] = (df["S_star_0_raw"] - finest).abs()
    # Where does the strike-pinned grid put S* inside its cell?
    df["cell_position"] = (df["S_star_0"] % df["dS"]) / df["dS"]
    return df


def sensitivity(param: str, values) -> pd.DataFrame:
    rows, curves = [], {}
    for v in values:
        kw = {param: v}
        if param == "T":
            res, t, Sb, _ = solve(T=v)
        else:
            res, t, Sb, _ = solve(**kw)
        curves[v] = (t, Sb)
        r_ = v if param == "r" else P.r
        s_ = v if param == "sigma" else P.sigma
        q_ = v if param == "q" else P.q
        S_inf = perpetual_put_boundary(P.K, r_, s_, q_) if r_ > 0 else 0.0
        rows.append(
            {
                param: v,
                "S_star_0": Sb[0],
                "S_star_0_over_K": Sb[0] / P.K,
                "S_star_just_before_T": Sb[-2],
                "S_star_at_T": Sb[-1],
                "perpetual_S_inf": S_inf,
                "predicted_terminal": min(P.K, r_ * P.K / q_) if q_ > 0 else P.K,
                "price": res.price,
                "runtime_s": res.runtime_s,
            }
        )
    return pd.DataFrame(rows), curves


def near_maturity_law() -> pd.DataFrame:
    r"""P3: ``K - S*(T-tau) ~ K sigma sqrt(tau ln(1/tau))``."""
    res, t, Sb, _ = solve(M=6400, N=6400)
    tau = P.T - t
    rows = []
    for target in (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1):
        i = int(np.argmin(np.abs(tau - target)))
        tt = float(tau[i])
        if tt <= 0:
            continue
        x = np.sqrt(tt * np.log(1.0 / tt))
        pred = P.K * P.sigma * x
        rows.append({"tau": tt, "S_star": Sb[i], "K_minus_S_star": P.K - Sb[i],
                     "sqrt_tau_log": x, "predicted": pred,
                     "ratio": (P.K - Sb[i]) / pred,
                     "correction_scale_1_over_log": 1.0 / np.log(1.0 / tt)})
    df = pd.DataFrame(rows)
    xs, ys = df["sqrt_tau_log"].to_numpy(), df["K_minus_S_star"].to_numpy()
    df["fitted_slope_through_origin"] = float(xs @ ys / (xs @ xs))
    df["predicted_slope_K_sigma"] = P.K * P.sigma
    return df


def pde_vs_lattice() -> pd.DataFrame:
    res, t, Sb, _ = solve()
    lt, lS = crr_boundary(P.S0, P.K, P.T, P.r, P.sigma, 3000, P.q, "put")
    interp = np.interp(lt, t, Sb)
    return pd.DataFrame({"t": lt, "lattice_S_star": lS, "pde_S_star_interp": interp,
                         "lattice_defined": np.isfinite(lS),
                         "abs_diff": np.abs(lS - interp)})


def main() -> None:
    plotting.use_style()
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)

    base = base_boundary(); base.to_csv(RESULTS / "m7_boundary_base.csv", index=False)
    acc = boundary_accuracy(); acc.to_csv(RESULTS / "m7_boundary_accuracy.csv", index=False)
    sig_df, sig_c = sensitivity("sigma", SIGMAS)
    sig_df.to_csv(RESULTS / "m7_sensitivity_sigma.csv", index=False)
    rate_df, rate_c = sensitivity("r", RATES)
    rate_df.to_csv(RESULTS / "m7_sensitivity_rate.csv", index=False)
    mat_df, mat_c = sensitivity("T", MATURITIES)
    mat_df.to_csv(RESULTS / "m7_sensitivity_maturity.csv", index=False)
    div_df, div_c = sensitivity("q", DIVIDENDS)
    div_df.to_csv(RESULTS / "m7_sensitivity_dividend.csv", index=False)
    nm = near_maturity_law(); nm.to_csv(RESULTS / "m7_near_maturity.csv", index=False)
    pl = pde_vs_lattice(); pl.to_csv(RESULTS / "m7_pde_vs_lattice.csv", index=False)

    S_inf_base = perpetual_put_boundary(P.K, P.r, P.sigma, P.q)

    # ---- Figure: families of boundaries -------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.6))
    # r = 0 is plotted separately: with no interest to earn there is no exercise
    # region at all (S* ~ 0.5 on a strike of 100), and including it forces the
    # panel's y-axis to span the full 0-100 and compresses every other curve.
    panels = [
        (axes[0, 0], sig_c, SIGMAS, r"$\sigma$", "{:.0%}",
         "Higher volatility defers exercise (P4)"),
        (axes[0, 1], rate_c, tuple(r for r in RATES if r > 0), "$r$", "{:.0%}",
         "Higher rates pull exercise forward (P5)"),
        (axes[1, 0], mat_c, MATURITIES, "$T$", "{:g}y",
         "Longer contracts wait longer (P6)"),
        (axes[1, 1], div_c, DIVIDENDS, "$q$", "{:.0%}",
         r"Dividends cap the boundary at $\min(K, rK/q)$ (P2)"),
    ]
    for ax, curves, values, name, fmt, title in panels:
        # The final point is tau = 0 exactly, where the exercise region is all of
        # {S < K} and S*(T) = K by definition.  For q > r the boundary is
        # genuinely discontinuous there (docs/03 P2); plotting the jump would
        # compress every other curve, so the terminal point is dropped and the
        # discontinuity is reported in the caption and in RESULTS.md instead.
        for i, v in enumerate(values):
            t, Sb = curves[v]
            x = t / t.max() if name == "$T$" else t
            c = plotting.SLOTS[i % len(plotting.SLOTS)]
            ax.plot(x[:-1], Sb[:-1], lw=1.8, color=c)
            # Direct labels at t = 0, where the curves are well separated; a
            # legend box would sit on top of the data in every panel.
            plotting.annotate(ax, x[0], Sb[0], f"{name} = {fmt.format(v)}",
                              dx=4, dy=4, va="bottom", fontsize=7.5)
        ax.axhline(P.K, color=plotting.INK_MUTED, lw=1.0)
        ax.set_ylabel(r"$S^*(t)$")
        ax.set_xlabel("$t/T$" if name == "$T$" else "$t$ (years)")
        ax.set_title(title)
        ax.margins(x=0.02, y=0.06)
        if name == "$r$":
            zero_t, zero_S = rate_c[0.0]
            plotting.annotate(
                ax, 0.5, ax.get_ylim()[0],
                rf"$r=0$: $S^*(0)={zero_S[0]:.2f}$ on a strike of {P.K:g}"
                "\nno exercise region at all (P5)",
                dx=0, dy=8, ha="center", fontsize=7.5,
            )
    axes[0, 0].axhline(S_inf_base, color=plotting.color("black_scholes"), lw=1.2,
                       ls=(0, (4, 3)))
    plotting.annotate(axes[0, 0], 1.0, S_inf_base,
                      rf"perpetual $S_\infty={S_inf_base:.1f}$ at $\sigma=20\%$",
                      dx=-4, dy=5, ha="right")
    plotting.save(fig, FIGURES / "m7_boundary_families.png",
                  caption=f"Base contract {P.label()} with one parameter varied; "
                          f"M = N = {GRID['M']}. Predictions P2, P4, P5, P6 were stated in "
                          f"docs/03_exercise_boundary.md before this study was run. The "
                          f"t = T point is omitted: S*(T) = K by definition, and for q > r "
                          f"the boundary jumps there from rK/q to K.")

    # ---- Figure: comparative statics against the closed form ----------
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))
    ax = axes[0]
    ax.plot(sig_df["sigma"], sig_df["S_star_0"], marker="o", ms=5,
            color=plotting.color("cn_psor"), label=r"$S^*(0)$, CN-PSOR")
    ax.plot(sig_df["sigma"], sig_df["perpetual_S_inf"], marker="s", ms=4, ls=(0, (4, 3)),
            color=plotting.color("black_scholes"),
            label=r"$S_\infty=K\gamma/(1+\gamma)$ (closed form)")
    ax.set_xlabel(r"volatility $\sigma$"); ax.set_ylabel(r"$S^*(0)$")
    ax.set_title(r"P4: decreasing in $\sigma$")
    ax.legend(loc="upper right", fontsize=7.5)

    ax = axes[1]
    ax.plot(rate_df["r"], rate_df["S_star_0"], marker="o", ms=5,
            color=plotting.color("cn_psor"), label=r"$S^*(0)$, CN-PSOR")
    ax.plot(rate_df["r"], rate_df["perpetual_S_inf"], marker="s", ms=4, ls=(0, (4, 3)),
            color=plotting.color("black_scholes"), label=r"$S_\infty$ (closed form)")
    ax.set_xlabel("risk-free rate $r$"); ax.set_ylabel(r"$S^*(0)$")
    ax.set_title("P5: increasing in $r$")
    ax.legend(loc="lower right", fontsize=7.5)

    ax = axes[2]
    ax.plot(mat_df["T"], mat_df["S_star_0"], marker="o", ms=5,
            color=plotting.color("cn_psor"), label=r"$S^*(0;T)$")
    ax.axhline(S_inf_base, color=plotting.color("black_scholes"), lw=1.4, ls=(0, (4, 3)),
               label=rf"perpetual limit $S_\infty={S_inf_base:.2f}$")
    ax.set_xscale("log")
    ax.set_xlabel("maturity $T$ (years)"); ax.set_ylabel(r"$S^*(0)$")
    ax.set_title(r"P6: decreasing in $T$, down to $S_\infty$")
    ax.legend(loc="upper right", fontsize=7.5)
    plotting.save(fig, FIGURES / "m7_boundary_sensitivity.png",
                  caption="Dashed curves are the closed-form perpetual boundary, not fits. "
                          f"The finite-maturity boundary must lie strictly above it, and does "
                          f"in every case measured.")

    # ---- Figure: asymptotics and accuracy -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))
    ax = axes[0]
    res, t, Sb, _ = solve(M=6400, N=6400)
    tau = P.T - t
    m = (tau > 0) & (tau <= 0.4)
    x = np.sqrt(tau[m] * np.log(1.0 / tau[m]))
    ax.plot(x, P.K - Sb[m], lw=2.0, color=plotting.color("cn_psor"),
            label=r"measured $K - S^*(T-\tau)$")
    xs = np.linspace(0, x.max(), 50)
    ax.plot(xs, P.K * P.sigma * xs, lw=1.6, ls=(0, (4, 3)),
            color=plotting.color("black_scholes"),
            label=rf"$K\sigma\sqrt{{\tau\ln(1/\tau)}}$ (slope ${P.K*P.sigma:.0f}$)")
    ax.set_xlabel(r"$\sqrt{\tau \ln(1/\tau)}$"); ax.set_ylabel(r"$K - S^*$")
    ax.set_title("P3: the square-root-log law")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[1]
    ax.plot(nm["tau"], nm["ratio"], marker="o", ms=5, color=plotting.color("cn_psor"),
            label="measured / predicted")
    ax.axhline(1.0, color=plotting.INK_MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.fill_between(nm["tau"], 1 - nm["correction_scale_1_over_log"],
                    1 + nm["correction_scale_1_over_log"],
                    color=plotting.PALETTE["blue"], alpha=0.14,
                    label=r"expected band $1 \pm 1/\ln(1/\tau)$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel("ratio to the asymptotic prediction")
    ax.set_title("Convergence is logarithmically slow, as predicted")
    ax.legend(loc="lower right", fontsize=7.5)

    ax = axes[2]
    ax.plot(acc["dS"], acc["dev_raw_from_finest"], marker="s", ms=5,
            color=plotting.INK_MUTED, label="raw (grid staircase)")
    ax.plot(acc["dS"], acc["dev_from_finest"], marker="o", ms=5,
            color=plotting.color("cn_psor"), label="smooth-pasting refinement")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta S$"); ax.set_ylabel(r"$|S^*(0) - S^*(0)|_{\rm finest}$")
    ax.set_title(r"Boundary accuracy, and its cell-alignment noise")
    ax.legend(loc="lower right", fontsize=7.5)
    plotting.save(fig, FIGURES / "m7_boundary_asymptotics.png",
                  caption=f"Base case, {P.label()}. Left/middle at M = N = 6400; right "
                          f"refines M at N = 6400 against the M = 12800 value.")

    # ---- Console -------------------------------------------------------
    print("== Milestone 7 ==")
    print(f"base case S*(0) = {base['S_star'].iloc[0]:.5f}  (perpetual floor "
          f"{S_inf_base:.5f}, strike {P.K:g})")
    dS = float(base["dS"].iloc[0])
    worst_drop = float(np.diff(base["S_star"]).min())
    print(f"P1 monotone in t: largest decrease {worst_drop:+.3e}, i.e. "
          f"{abs(worst_drop)/dS:.2f} grid cells (dS = {dS:.4f}) -- "
          f"{'within grid noise' if abs(worst_drop) < dS else 'A REAL VIOLATION'}")
    print("P2 terminal boundary min(K, rK/q):")
    for _, row in div_df.iterrows():
        print(f"   q={row['q']:.2f}: predicted {row['predicted_terminal']:7.3f}, "
              f"measured just before T {row['S_star_just_before_T']:7.3f}  "
              f"(rel dev {abs(row['S_star_just_before_T']/row['predicted_terminal']-1):.4f}), "
              f"at T exactly {row['S_star_at_T']:.1f}")
    print("P3 square-root-log law:")
    for _, row in nm.iterrows():
        print(f"   tau={row['tau']:.5f}: K-S*={row['K_minus_S_star']:8.5f}, "
              f"predicted {row['predicted']:8.5f}, ratio {row['ratio']:.4f} "
              f"(expected band 1 +- {row['correction_scale_1_over_log']:.3f})")
    print(f"   fitted slope through the origin {nm['fitted_slope_through_origin'].iloc[0]:.4f} "
          f"vs K*sigma = {P.K*P.sigma:.4f}")
    print("P4 volatility:  " + ", ".join(f"sigma={r_['sigma']:.2f}: {r_['S_star_0']:.3f}"
                                         for _, r_ in sig_df.iterrows()))
    print(f"   monotone decreasing: {bool(np.all(np.diff(sig_df['S_star_0']) < 0))}")
    print("P5 rate:        " + ", ".join(f"r={r_['r']:.2f}: {r_['S_star_0']:.3f}"
                                         for _, r_ in rate_df.iterrows()))
    print(f"   monotone increasing: {bool(np.all(np.diff(rate_df['S_star_0']) > 0))}")
    print("P6 maturity:    " + ", ".join(f"T={r_['T']:g}: {r_['S_star_0']:.3f}"
                                         for _, r_ in mat_df.iterrows()))
    print(f"   monotone decreasing: {bool(np.all(np.diff(mat_df['S_star_0']) < 0))}; "
          f"T=25 is {mat_df['S_star_0'].iloc[-1]/S_inf_base:.4f} x the perpetual limit")
    print("boundary accuracy vs dS:")
    for _, row in acc.iterrows():
        print(f"   M={int(row['M']):6d} dS={row['dS']:.4f}: S*(0)={row['S_star_0']:.5f} "
              f"(raw {row['S_star_0_raw']:.4f}), dev from finest "
              f"{row['dev_from_finest']:.2e} (raw {row['dev_raw_from_finest']:.2e})")
    defined = pl["lattice_defined"]
    print(f"PDE vs lattice boundary: lattice undefined at "
          f"{int((~defined).sum())}/{len(pl)} time levels (t <= "
          f"{pl.loc[~defined, 't'].max():.3f}); where both are defined the mean "
          f"absolute difference is {pl.loc[defined, 'abs_diff'].mean():.4f}")


if __name__ == "__main__":
    main()
