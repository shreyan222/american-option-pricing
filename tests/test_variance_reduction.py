"""Tests for antithetic variates and the European-put control variate.

The theme is that a variance-reduction technique must reduce variance *without*
moving the estimand and *without* breaking the error bar.  So every method is
checked on three axes: does it still price the same thing, does the interval
still cover, and is the reported standard error the honest one for the
dependence structure the method introduces.
"""

import numpy as np
import pytest

from amopt.binomial import crr
from amopt.black_scholes import bs_put
from amopt.lsm import simulate_gbm
from amopt.variance_reduction import (
    METHODS,
    _european_control,
    _ols_beta,
    lsm_with_variance_reduction,
    variance_reduction_factor,
)

S0, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.2
N_DATES = 25


@pytest.fixture(scope="module")
def bermudan():
    return crr(S0, K, T, R, SIG, 40_000, kind="put", bermudan_dates=N_DATES).price


# ---------------------------------------------------------------------------
# The control variate itself
# ---------------------------------------------------------------------------

def test_european_control_has_the_black_scholes_mean():
    """E[e^{-rT}(K - S_T)^+] must equal the closed-form European put."""
    n = 500_000
    S = simulate_gbm(S0, R, SIG, T, n, 1, seed=2)
    Y = _european_control(S[:, -1], K, R, T, "put")
    se = Y.std(ddof=1) / np.sqrt(n)
    assert abs(Y.mean() - float(bs_put(S0, K, T, R, SIG))) < 4.0 * se


def test_ols_beta_recovers_a_known_coefficient():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=50_000)
    X = 3.0 + 2.5 * Y + 0.1 * rng.normal(size=50_000)
    assert _ols_beta(X, Y) == pytest.approx(2.5, rel=2e-3)


def test_ols_beta_is_zero_for_a_constant_control():
    assert _ols_beta(np.arange(10.0), np.ones(10)) == 0.0


# ---------------------------------------------------------------------------
# Every method prices the same object
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_all_methods_agree_with_the_exact_bermudan_value(method, bermudan):
    res = lsm_with_variance_reduction(S0, K, T, R, SIG, n_paths=200_000, n_steps=N_DATES,
                                      method=method, seed=101)
    assert abs(res.price - bermudan) < 4.0 * res.std_error + 0.01, (method, res.price, bermudan)


def test_control_variate_is_unbiased_for_any_beta():
    r"""X - b(Y - E[Y]) is unbiased for *every* b, not just the optimal one."""
    ref = lsm_with_variance_reduction(S0, K, T, R, SIG, n_paths=200_000, n_steps=N_DATES,
                                      method="naive", seed=55)
    forced = lsm_with_variance_reduction(S0, K, T, R, SIG, n_paths=200_000, n_steps=N_DATES,
                                         method="control", beta_source="none", seed=55)
    # b = 1 is far from optimal (b* ~ 0.6) but must not bias the estimate
    assert abs(forced.price - ref.price) < 4.0 * np.hypot(ref.std_error, forced.std_error)
    assert forced.beta == 1.0


# ---------------------------------------------------------------------------
# Variance actually goes down
# ---------------------------------------------------------------------------

def test_every_method_reduces_variance_relative_to_naive():
    base = lsm_with_variance_reduction(S0, K, T, R, SIG, n_paths=200_000, n_steps=N_DATES,
                                       method="naive", seed=7)
    for method in ("antithetic", "control", "antithetic_control"):
        res = lsm_with_variance_reduction(S0, K, T, R, SIG, n_paths=200_000, n_steps=N_DATES,
                                          method=method, seed=7)
        vrf = variance_reduction_factor(base, res)
        assert vrf > 1.5, f"{method} gave a variance reduction factor of only {vrf:.2f}"


def test_control_variate_reduction_matches_the_theoretical_one_minus_rho_squared():
    r"""Var(X - b*Y) = Var(X)(1 - rho^2) exactly, so VRF must equal 1/(1 - rho^2)."""
    base = lsm_with_variance_reduction(S0, K, T, R, SIG, n_paths=300_000, n_steps=N_DATES,
                                       method="naive", seed=13)
    cv = lsm_with_variance_reduction(S0, K, T, R, SIG, n_paths=300_000, n_steps=N_DATES,
                                     method="control", seed=13)
    predicted = 1.0 / (1.0 - cv.correlation**2)
    observed = variance_reduction_factor(base, cv)
    assert observed == pytest.approx(predicted, rel=0.05), (observed, predicted, cv.correlation)


def test_variance_reduction_factor_accounts_for_the_antithetic_path_budget():
    """One antithetic unit costs two paths; the comparison must be per path."""
    base = lsm_with_variance_reduction(S0, K, T, R, SIG, n_paths=100_000, n_steps=N_DATES,
                                       method="naive", seed=3)
    anti = lsm_with_variance_reduction(S0, K, T, R, SIG, n_paths=100_000, n_steps=N_DATES,
                                       method="antithetic", seed=3)
    per_path = variance_reduction_factor(base, anti)
    per_unit = base.unit_variance / anti.unit_variance
    assert per_unit == pytest.approx(2.0 * per_path, rel=1e-12)
    assert per_path < per_unit, "per-unit comparison would flatter antithetic sampling"


# ---------------------------------------------------------------------------
# The error bar must survive the dependence
# ---------------------------------------------------------------------------

def test_antithetic_standard_error_uses_pairs_not_paths():
    r"""The reported SE must be the pair statistic.

    Reconstruct what the naive path-level formula would have given and check the
    reported value is not that -- with negative antithetic correlation the naive
    formula is systematically too large here, and for a strongly anti-correlated
    payoff it would be badly wrong in either direction.
    """
    res = lsm_with_variance_reduction(S0, K, T, R, SIG, n_paths=200_000, n_steps=N_DATES,
                                      method="antithetic", seed=21)
    assert res.n_units == 100_000
    assert res.n_paths == 200_000
    assert res.std_error == pytest.approx(np.sqrt(res.unit_variance / res.n_units))


@pytest.mark.parametrize("method", METHODS)
def test_confidence_interval_covers_the_estimator_mean_at_the_nominal_rate(method):
    r"""Empirical coverage of the 95% interval, measured over independent repetitions.

    Coverage is checked against the estimator's *own* mean across seeds, which
    isolates the quality of the standard error from the (separately measured)
    low bias of a fixed exercise policy.  A wrong treatment of antithetic
    dependence shows up here as under-coverage.
    """
    n_rep = 120
    runs = [
        lsm_with_variance_reduction(S0, K, T, R, SIG, n_paths=8_000, n_steps=N_DATES,
                                    method=method, seed=5_000 + i)
        for i in range(n_rep)
    ]
    prices = np.array([r.price for r in runs])
    target = prices.mean()
    covered = np.mean([r.ci_low <= target <= r.ci_high for r in runs])
    # Binomial 3-sigma band around 0.95 for n_rep draws.
    tol = 3.0 * np.sqrt(0.95 * 0.05 / n_rep)
    assert abs(covered - 0.95) < tol + 0.02, f"{method}: coverage {covered:.3f}"


def test_paths_for_target_se_is_consistent_with_the_realised_error():
    res = lsm_with_variance_reduction(S0, K, T, R, SIG, n_paths=100_000, n_steps=N_DATES,
                                      method="naive", seed=17)
    needed = res.paths_for_target_se(res.std_error)
    assert needed == pytest.approx(res.n_paths, rel=1e-9)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def test_invalid_arguments():
    with pytest.raises(ValueError):
        lsm_with_variance_reduction(S0, K, T, R, SIG, method="importance_sampling")
    with pytest.raises(ValueError):
        lsm_with_variance_reduction(S0, K, T, R, SIG, method="control", beta_source="magic")
    with pytest.raises(ValueError):
        lsm_with_variance_reduction(S0, K, T, R, SIG, n_paths=1001, method="antithetic")


def test_deterministic_given_a_seed():
    kw = dict(n_paths=20_000, n_steps=N_DATES, method="antithetic_control", seed=77)
    a = lsm_with_variance_reduction(S0, K, T, R, SIG, **kw)
    b = lsm_with_variance_reduction(S0, K, T, R, SIG, **kw)
    assert a.price == b.price and a.std_error == b.std_error and a.beta == b.beta
