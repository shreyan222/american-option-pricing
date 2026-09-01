"""Every headline number in README.md, RESULTS.md and paper/ must be in results/.

This is the mechanical enforcement of the repository's no-fabrication rule.  Each
claim below names the document that quotes it, the CSV that produced it, and how
to look it up.  If an experiment is rerun and a number moves, this test fails and
the documents have to be updated -- which is the point.

The tests are skipped (not failed) when `results/` has not been generated, so a
fresh clone can run the unit suite before running the experiments.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = {
    "README.md": ROOT / "README.md",
    "RESULTS.md": ROOT / "RESULTS.md",
    "paper": ROOT / "paper" / "american_put_pde_vs_monte_carlo.md",
}


def load(name):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not generated; run the experiments first")
    return pd.read_csv(path)


def quoted(text: str, *docs) -> bool:
    """Is this string present in the named documents?

    The documents use the typographic minus sign U+2212, so both spellings are
    accepted; nothing else is normalised, because the point is to check the
    digits exactly as printed.
    """
    variants = {text, text.replace("-", "\u2212"), text.replace("\u2212", "-")}
    return all(any(v in DOCS[d].read_text() for v in variants) for d in docs)


# ---------------------------------------------------------------------------
# Milestone 1-2
# ---------------------------------------------------------------------------

def test_european_put_and_binomial_convergence():
    d = load("m1_binomial_convergence.csv")
    put = d[d["kind"] == "put"].sort_values("N")
    assert f"{put['exact'].iloc[0]:.8f}" == "5.57352602"
    assert f"{put['price'].iloc[-1]:.8f}" == "5.57336980"
    slope = np.polyfit(np.log(put["N"]), np.log(put["abs_error"]), 1)[0]
    assert f"{slope:.3f}" == "-0.990"
    assert quoted("5.57352602", "RESULTS.md", "README.md")
    assert quoted("-0.990", "RESULTS.md")


def test_lattice_parity_residual():
    d = load("m1_lattice_parity.csv")
    assert f"{d['residual'].max():.2e}" == "3.15e-11"
    assert quoted("3.15 × 10⁻¹¹", "RESULTS.md")


def test_perpetual_anchors():
    d = load("m2_perpetual_anchors.csv")
    base = d[d["regime"] == "base"].iloc[0]
    assert f"{base['beta_minus']:.6f}" == "-2.500000"
    assert f"{base['S_inf']:.6f}" == "71.428571"
    assert f"{base['perpetual_put']:.6f}" == "12.320033"
    assert bool(d["dominance_ok"].all())
    assert quoted("71.428571", "RESULTS.md")
    assert quoted("12.320033", "RESULTS.md", "paper")


# ---------------------------------------------------------------------------
# Milestone 3
# ---------------------------------------------------------------------------

def test_cn_matches_the_lattice():
    d = load("m3_cn_convergence.csv").sort_values("M")
    assert f"{d['american_price'].iloc[-1]:.8f}" == "6.09030490"
    assert quoted("6.09030490", "RESULTS.md", "paper")


def test_cn_agrees_with_crr_across_all_regimes():
    d = load("m3_regime_table.csv")
    assert len(d) == 10
    assert f"{d['abs_diff_cn_vs_crr'].max():.2e}" == "2.45e-04"
    assert quoted("2.45 × 10⁻⁴", "RESULTS.md")


def test_three_lcp_solvers_agree():
    d = load("m3_solver_agreement.csv")
    assert set(d["solver"]) == {"psor", "psor_lex", "brennan_schwartz"}
    assert f"{d['abs_dev_from_exact_lcp'].max():.2e}" == "3.78e-12"
    assert quoted("3.78 × 10⁻¹²", "RESULTS.md", "paper")


def test_price_is_invariant_to_omega():
    d = load("m5_headline.csv")  # sanity that results/ is populated
    o = load("m3_omega_study.csv")
    spread = o.groupby("M")["price"].apply(lambda s: s.max() - s.min()).max()
    assert f"{spread:.2e}" == "1.35e-07"
    assert quoted("1.35 × 10⁻⁷", "RESULTS.md", "paper")
    assert len(d) == 4


def test_rannacher_effect():
    d = load("m3_rannacher.csv")
    r0 = d[(d["N"] == 25) & (d["rannacher_steps"] == 0)].iloc[0]
    r2 = d[(d["N"] == 25) & (d["rannacher_steps"] == 2)].iloc[0]
    assert f"{r0['tv_gamma_near_strike']:.5f}" == "0.46201"
    assert f"{r2['tv_gamma_near_strike']:.5f}" == "0.01257"
    assert round(r0["tv_gamma_near_strike"] / r2["tv_gamma_near_strike"]) == 37
    assert round(r0["abs_error"] / r2["abs_error"]) == 12
    assert quoted("0.46201", "RESULTS.md", "paper")


# ---------------------------------------------------------------------------
# Milestone 4
# ---------------------------------------------------------------------------

def test_bias_decomposition_at_fifty_dates():
    d = load("m4_bias_decomposition.csv")
    row = d[d["n_dates"] == 50].iloc[0]
    assert f"{row['bermudan_crr']:.6f}" == "6.078622"
    assert f"{row['exercise_date_bias']:+.6f}" == "+0.011730"
    assert f"{row['regression_and_sampling_error']:+.6f}" == "-0.010058"
    assert quoted("6.078622", "RESULTS.md", "paper")
    assert quoted("+0.011730", "RESULTS.md")


def test_bermudan_gap_is_first_order():
    d = load("m4_bias_decomposition.csv").sort_values("n_dates")
    slope = np.polyfit(np.log(d["n_dates"]), np.log(d["exercise_date_bias"]), 1)[0]
    assert -1.05 < slope < -0.95
    assert f"{d['exercise_date_bias'].iloc[0]:.6f}" == "0.109210"


def test_foresight_bias_worst_case():
    d = load("m4_in_vs_out_of_sample.csv")
    w = d[(d["n_paths"] == 500) & (d["degree"] == 10)].iloc[0]
    assert f"{w['in_sample_mean']:.5f}" == "6.61295"
    assert f"{w['out_of_sample_mean']:.5f}" == "5.85629"
    assert f"{w['foresight_bias']:+.5f}" == "+0.75666"
    pct = 100 * w["in_sample_bias"] / w["bermudan_benchmark"]
    assert f"{pct:.1f}" == "9.0"
    assert quoted("6.61295", "RESULTS.md", "paper")
    assert quoted("9.0%", "RESULTS.md", "README.md")


def test_in_and_out_of_sample_bracket_the_truth():
    r"""In-sample sits above the exact Bermudan value and out-of-sample below --
    *except* at the largest sample, where the foresight bias has decayed below
    the policy-suboptimality bias and the in-sample estimate falls through.

    This asymmetry was found by this test contradicting an earlier, stronger
    claim in the write-up; the documents now state the exception explicitly.
    """
    d = load("m4_in_vs_out_of_sample.csv")
    b = d["bermudan_benchmark"].iloc[0]
    assert np.all(d["out_of_sample_mean"] < b), "out-of-sample must be a lower bound"
    above = d["in_sample_mean"] > b
    assert above.sum() == len(d) - 1, d[["n_paths", "degree", "in_sample_mean"]]
    exception = d[~above].iloc[0]
    assert int(exception["n_paths"]) == 200_000 and int(exception["degree"]) == 3
    assert f"{b - exception['in_sample_mean']:.4f}" == "0.0037"
    assert quoted("0.0037", "RESULTS.md", "paper")


def test_reported_standard_errors_are_honest():
    d = load("m4_in_vs_out_of_sample.csv")
    ratios = np.concatenate([
        (d["in_sample_realised_sd"] / d["in_sample_reported_se"]).to_numpy(),
        (d["out_of_sample_realised_sd"] / d["out_of_sample_reported_se"]).to_numpy(),
    ])
    assert f"{ratios.min():.2f}" == "0.98"
    assert f"{ratios.max():.2f}" == "1.18"
    assert quoted("[0.98, 1.18]", "RESULTS.md", "paper")


def test_itm_filtering_matters():
    d = load("m4_itm_filter.csv")
    worst_all = d[~d["itm_only"]]["deviation_from_bermudan"].abs().max()
    assert f"{worst_all:.6f}" == "0.338926"
    assert quoted("0.338926", "RESULTS.md", "paper")
    assert d[d["itm_only"]]["deviation_from_bermudan"].abs().max() < 0.02


# ---------------------------------------------------------------------------
# Milestone 5
# ---------------------------------------------------------------------------

def test_variance_reduction_headline():
    d = load("m5_headline.csv").set_index("method")
    for m, ciw, vrf in (("naive", "0.06283", "1.00"),
                        ("antithetic", "0.03744", "2.82"),
                        ("control", "0.04179", "2.26"),
                        ("antithetic_control", "0.03643", "2.97")):
        assert f"{d.loc[m, 'ci_width']:.5f}" == ciw, m
        assert f"{d.loc[m, 'variance_reduction_factor']:.2f}" == vrf, m
        assert quoted(ciw, "RESULTS.md", "paper")
    assert f"{d.loc['antithetic', 'paths_for_naive_se']:,.0f}" == "71,022"
    assert quoted("71,022", "RESULTS.md", "README.md", "paper")


def test_control_variate_matches_one_minus_rho_squared_in_every_regime():
    d = load("m5_regimes.csv")
    assert len(d) == 10
    theory = 1.0 / (1.0 - d["control_correlation"] ** 2)
    assert np.max(np.abs(theory - d["vrf_control"])) < 0.005, "VRF must match 1/(1-rho^2)"
    assert quoted("2.27", "RESULTS.md") and quoted("5.37", "RESULTS.md")


def test_monte_carlo_error_is_root_n():
    d = load("m5_scaling.csv")
    for m, slope in (("naive", "-0.5012"), ("antithetic", "-0.5062"),
                     ("control", "-0.5047"), ("antithetic_control", "-0.5051")):
        got = d[d["method"] == m]["fitted_se_slope"].iloc[0]
        assert f"{got:.4f}" == slope, m
        assert quoted(slope, "RESULTS.md")


def test_antithetic_pairing_coverage_both_directions():
    d = load("m5_antithetic_pairing.csv").set_index("payoff")
    assert f"{d.loc['european_put', 'mean_pair_correlation']:+.4f}" == "-0.4145"
    assert f"{d.loc['european_put', 'coverage_naive_se']:.4f}" == "0.9890"
    assert f"{d.loc['butterfly', 'mean_pair_correlation']:+.4f}" == "+0.4282"
    assert f"{d.loc['butterfly', 'coverage_naive_se']:.4f}" == "0.9060"
    assert f"{d.loc['butterfly', 'coverage_pair_se']:.4f}" == "0.9523"
    assert quoted("0.9060", "RESULTS.md", "paper")
    assert quoted("90.6%", "README.md")


def test_coverage_falls_when_the_interval_narrows():
    d = load("m5_coverage.csv").set_index("method")
    assert f"{d.loc['naive', 'coverage_of_bermudan']:.3f}" == "0.950"
    assert f"{d.loc['antithetic', 'coverage_of_bermudan']:.3f}" == "0.910"
    assert quoted("0.910", "RESULTS.md", "paper")


# ---------------------------------------------------------------------------
# Milestone 6
# ---------------------------------------------------------------------------

def test_reference_solution():
    d = load("m6_reference.csv")
    ref = d[d["family"] == "reference"]["value"].iloc[0]
    unc = d[d["family"] == "reference_uncertainty"]["value"].iloc[0]
    crr = d[d["family"] == "crr_richardson"]["value"].iloc[-1]
    cn = d[d["family"] == "cn_extrapolated"]["value"].iloc[0]
    assert f"{ref:.9f}" == "6.090370613"
    assert f"{unc:.2e}" == "9.36e-08"
    assert f"{crr:.9f}" == "6.090370659"
    assert f"{cn:.9f}" == "6.090370566"
    assert quoted("6.090370613", "RESULTS.md", "README.md", "paper")
    assert quoted("9.36 × 10⁻⁸", "RESULTS.md")


def test_measured_convergence_orders():
    d = load("m6_cn_grid.csv")
    sp = d[d["axis"] == "space"]
    tm = d[d["axis"] == "time"]
    both = d[d["axis"] == "both"]
    space_floor = float(both[both["M"] == 6400]["abs_error"].iloc[0])
    time_floor = float(sp[sp["M"] == 3200]["abs_error"].iloc[0])

    def fit(frame, key, floor):
        k = frame["abs_error"] > 10 * floor
        return np.polyfit(np.log(frame[key][k]), np.log(frame["abs_error"][k]), 1)[0]

    assert f"{-fit(sp, 'M', time_floor):.3f}" == "1.995"
    assert f"{-fit(tm, 'N', space_floor):.3f}" == "1.223"
    assert quoted("1.995", "RESULTS.md", "README.md", "paper")
    assert quoted("1.223", "RESULTS.md", "paper")


def test_lsm_rmse_floor():
    d = load("m6_lsm_paths.csv")
    vr = d[d["method"] == "antithetic_control"].sort_values("n_paths")
    assert f"{vr['rmse'].min():.5f}" == "0.01474"
    assert f"{vr['bias'].iloc[-1]:+.5f}" == "-0.01361"
    fr = load("m6_frontier.csv")
    lsm = fr[fr["method"].str.startswith("lsm")]
    assert f"{lsm['error'].min():.2e}" == "1.44e-02"
    assert lsm["error"].min() > 1e-2, "LSM must never reach 1e-2, as documented"
    assert quoted("1.44 × 10⁻²", "RESULTS.md", "paper")


def test_error_vs_runtime_power_laws():
    d = load("m6_frontier.csv")
    for m, expected in (("cn_psor", "-1.22"), ("crr", "-0.55")):
        f = d[d["method"] == m]
        slope = np.polyfit(np.log(f["runtime_s"]), np.log(f["error"]), 1)[0]
        assert f"{slope:.2f}" == expected, m
        assert quoted(expected, "RESULTS.md", "README.md", "paper")


def test_domain_truncation_is_flat_beyond_two_strikes():
    d = load("m6_cn_domain.csv")
    flat = d[d["S_max_mult"] >= 2.0]["abs_error"]
    # Identical to six significant digits: the residual spread is 1.6e-10, which
    # is round-off in the extra far-out-of-the-money nodes, not truncation.
    assert (flat.max() - flat.min()) / flat.mean() < 1e-5
    assert f"{flat.iloc[0]:.3e}" == "6.540e-05"
    assert f"{d[d['S_max_mult'] == 1.5]['abs_error'].iloc[0]:.3e}" == "1.621e-04"


def test_poly_and_chebyshev_span_the_same_space():
    d = load("m6_lsm_basis.csv").drop_duplicates("degree").set_index("degree")
    assert d.loc[[1, 2, 3, 4], "poly_minus_chebyshev"].max() == 0.0
    assert d.loc[12, "poly_minus_chebyshev"] > 1e-3
    assert quoted("bitwise zero to degree 4", "paper")


# ---------------------------------------------------------------------------
# Milestone 7
# ---------------------------------------------------------------------------

def test_boundary_base_value():
    d = load("m7_boundary_base.csv").sort_values("t")
    assert f"{d['S_star'].iloc[0]:.5f}" == "80.87563"
    assert quoted("80.87563", "RESULTS.md", "paper")


def test_terminal_boundary_min_K_rK_over_q():
    d = load("m7_sensitivity_dividend.csv").set_index("q")
    assert f"{d.loc[0.10, 'S_star_just_before_T']:.3f}" == "49.830"
    assert f"{d.loc[0.10, 'predicted_terminal']:.3f}" == "50.000"
    for q in (0.06, 0.10, 0.15):
        rel = abs(d.loc[q, "S_star_just_before_T"] / d.loc[q, "predicted_terminal"] - 1)
        assert rel < 0.005, q
    assert quoted("49.830", "RESULTS.md", "README.md", "paper")


def test_near_maturity_law_slope():
    d = load("m7_near_maturity.csv")
    assert f"{d['fitted_slope_through_origin'].iloc[0]:.3f}" == "21.578"
    assert f"{d['predicted_slope_K_sigma'].iloc[0]:.1f}" == "20.0"
    assert quoted("21.578", "RESULTS.md", "README.md", "paper")


def test_boundary_comparative_statics_are_monotone_and_quoted():
    sig = load("m7_sensitivity_sigma.csv").sort_values("sigma")
    rate = load("m7_sensitivity_rate.csv").sort_values("r")
    mat = load("m7_sensitivity_maturity.csv").sort_values("T")
    assert np.all(np.diff(sig["S_star_0"]) < 0)
    assert np.all(np.diff(rate["S_star_0"]) > 0)
    assert np.all(np.diff(mat["S_star_0"]) < 0)
    assert f"{sig['S_star_0'].iloc[0]:.3f}" == "92.754"
    assert f"{rate['S_star_0'].iloc[0]:.3f}" == "0.496"
    assert f"{mat['S_star_0'].iloc[-1]:.3f}" == "71.672"
    assert quoted("92.754", "RESULTS.md", "paper")
    assert quoted("0.496", "RESULTS.md", "paper")


def test_boundary_accuracy_is_not_monotone_in_dS():
    d = load("m7_boundary_accuracy.csv").set_index("M")
    assert f"{d.loc[1600, 'dev_from_finest']:.1e}" == "9.7e-02"
    assert f"{d.loc[400, 'dev_from_finest']:.1e}" == "9.5e-03"
    assert d.loc[1600, "dev_from_finest"] > d.loc[400, "dev_from_finest"]
    assert quoted("not monotone", "RESULTS.md", "paper")


def test_lattice_boundary_is_undefined_near_t_zero():
    d = load("m7_pde_vs_lattice.csv")
    undefined = (~d["lattice_defined"]).sum()
    assert undefined == 58
    assert len(d) == 3001
    assert quoted("58", "RESULTS.md", "paper")


# ---------------------------------------------------------------------------
# Global guard
# ---------------------------------------------------------------------------

def test_every_experiment_produced_its_results():
    if not RESULTS.exists() or not any(RESULTS.glob("*.csv")):
        pytest.skip("results/ not generated")
    expected_prefixes = {"m1", "m2", "m3", "m4", "m5", "m6", "m7"}
    found = {p.name.split("_")[0] for p in RESULTS.glob("*.csv")}
    assert expected_prefixes <= found, expected_prefixes - found


def test_no_document_cites_a_missing_results_file():
    """Every `results/*.csv` named in the documents must exist."""
    import re

    for name, path in DOCS.items():
        for match in re.findall(r"results/([A-Za-z0-9_]+\.csv)", path.read_text()):
            assert (RESULTS / match).exists(), f"{name} cites missing {match}"
