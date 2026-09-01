"""Tests for Longstaff-Schwartz least-squares Monte Carlo.

Monte Carlo needs a different testing discipline from a deterministic solver.
Every test here is either

* **distributional** -- an exact property of the simulated paths (martingale
  identity, lognormal moments, antithetic structure),
* **a limiting case** -- ``n_steps = 1`` must reproduce the European price
  exactly, since there is then no early exercise to estimate,
* **a benchmark comparison with a confidence interval** -- LSM with ``n`` exercise
  dates estimates the ``n``-date *Bermudan* value, which the lattice computes
  exactly, so the comparison is against the right target rather than against the
  continuous-exercise American value, or
* **a mutation test** -- deliberately leaking future information must move the
  price, which is what makes the absence of leakage in the real estimator
  meaningful.

All randomness is seeded.
"""

import numpy as np
import pytest

from amopt.binomial import crr, crr_price
from amopt.black_scholes import bs_put
from amopt.lsm import (
    Z_975,
    _mean_and_se,
    basis_matrix,
    longstaff_schwartz,
    simulate_gbm,
)
from amopt.perpetual import perpetual_put_price

S0, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.2


# ---------------------------------------------------------------------------
# Path simulation
# ---------------------------------------------------------------------------

def test_paths_start_at_S0_and_have_the_right_shape():
    S = simulate_gbm(S0, R, SIG, T, 1000, 20, seed=0)
    assert S.shape == (1000, 21)
    assert np.all(S[:, 0] == S0)
    assert np.all(S > 0.0)


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_discounted_stock_is_a_martingale(q):
    r"""E[e^{-(r-q)T} S_T] = S_0 -- the defining property of the risk-neutral measure."""
    n = 400_000
    S = simulate_gbm(S0, R, SIG, T, n, 12, q, seed=3)
    disc = np.exp(-(R - q) * T) * S[:, -1]
    se = disc.std(ddof=1) / np.sqrt(n)
    assert abs(disc.mean() - S0) < 4.0 * se


def test_terminal_log_moments_are_exact():
    r"""log(S_T/S_0) ~ N((r - q - sigma^2/2)T, sigma^2 T) exactly, for any n_steps."""
    n = 500_000
    for n_steps in (1, 7, 50):
        S = simulate_gbm(S0, R, SIG, T, n, n_steps, seed=11)
        x = np.log(S[:, -1] / S0)
        mu, var = (R - 0.5 * SIG**2) * T, SIG**2 * T
        se_mu = np.sqrt(var / n)
        assert abs(x.mean() - mu) < 4.0 * se_mu, n_steps
        # variance of the sample variance of a normal is 2*sigma^4/n
        se_var = np.sqrt(2.0) * var / np.sqrt(n)
        assert abs(x.var(ddof=1) - var) < 4.0 * se_var, n_steps


def test_antithetic_pairs_have_the_exact_reflection_structure():
    r"""Path 2k uses Z, path 2k+1 uses -Z, so their log-returns sum to 2*drift."""
    n_steps = 10
    S = simulate_gbm(S0, R, SIG, T, 1000, n_steps, seed=5, antithetic=True)
    dt = T / n_steps
    drift = (R - 0.5 * SIG**2) * dt
    la = np.diff(np.log(S[0::2]), axis=1)
    lb = np.diff(np.log(S[1::2]), axis=1)
    assert np.allclose(la + lb, 2.0 * drift, atol=1e-10)


def test_antithetic_requires_even_path_count():
    with pytest.raises(ValueError):
        simulate_gbm(S0, R, SIG, T, 101, 10, antithetic=True, seed=0)


def test_simulation_is_reproducible_and_seed_sensitive():
    a = simulate_gbm(S0, R, SIG, T, 100, 5, seed=42)
    b = simulate_gbm(S0, R, SIG, T, 100, 5, seed=42)
    c = simulate_gbm(S0, R, SIG, T, 100, 5, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------------------
# Regression bases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["poly", "laguerre", "chebyshev"])
def test_basis_shapes(kind):
    S = np.linspace(50.0, 150.0, 37)
    A = basis_matrix(S, K, 4, kind)
    assert A.shape == (37, 5)
    assert np.isfinite(A).all()


def test_poly_basis_is_the_monomials_in_S_over_K():
    S = np.array([50.0, 100.0, 150.0])
    A = basis_matrix(S, K, 3, "poly")
    x = S / K
    assert np.allclose(A, np.column_stack([np.ones_like(x), x, x**2, x**3]))


def test_chebyshev_basis_satisfies_its_recurrence():
    S = np.linspace(1.0, 199.0, 51)
    A = basis_matrix(S, K, 5, "chebyshev")
    u = np.clip(S / K - 1.0, -1.0, 1.0)
    assert np.allclose(A[:, 0], 1.0)
    assert np.allclose(A[:, 1], u)
    for k in range(2, 6):
        assert np.allclose(A[:, k], 2 * u * A[:, k - 1] - A[:, k - 2])


def test_laguerre_basis_matches_the_first_polynomials():
    S = np.linspace(10.0, 200.0, 23)
    x = S / K
    A = basis_matrix(S, K, 2, "laguerre")
    w = np.exp(-x / 2)
    assert np.allclose(A[:, 0], w)
    assert np.allclose(A[:, 1], w * (1 - x))
    assert np.allclose(A[:, 2], w * (1 - 2 * x + x**2 / 2))


def test_chebyshev_is_better_conditioned_than_monomials():
    """Why the basis is configurable: conditioning differs by orders of magnitude."""
    S = simulate_gbm(S0, R, SIG, T, 20_000, 10, seed=1)[:, 5]
    S = S[S < K]
    cond_poly = np.linalg.cond(basis_matrix(S, K, 6, "poly"))
    cond_cheb = np.linalg.cond(basis_matrix(S, K, 6, "chebyshev"))
    assert cond_cheb < cond_poly / 10.0


def test_invalid_basis_and_degree():
    with pytest.raises(ValueError):
        basis_matrix(np.array([100.0]), K, 3, "hermite")
    with pytest.raises(ValueError):
        basis_matrix(np.array([100.0]), K, 0, "poly")


# ---------------------------------------------------------------------------
# Standard error with dependent antithetic pairs
# ---------------------------------------------------------------------------

def test_antithetic_standard_error_is_computed_over_pairs_not_paths():
    r"""Antithetic paths are negatively dependent; treating them as 2n independent
    draws would understate the variance of the mean.  The pair is the unit of
    independence."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal(20_000)
    cash = np.empty(40_000)
    cash[0::2] = 1.0 + z
    cash[1::2] = 1.0 - z  # perfectly anti-correlated: every pair mean is exactly 1
    mean_pair, se_pair = _mean_and_se(cash, antithetic=True)
    mean_iid, se_iid = _mean_and_se(cash, antithetic=False)
    assert mean_pair == pytest.approx(mean_iid, abs=1e-12)
    assert mean_pair == pytest.approx(1.0, abs=1e-12)
    # The pair means carry no variance at all here; the naive i.i.d. formula
    # would report a large spurious standard error.
    assert se_pair < 1e-12
    assert se_iid > 1e-3


# ---------------------------------------------------------------------------
# Limiting case: a single exercise date is a European option
# ---------------------------------------------------------------------------

def test_one_exercise_date_reproduces_the_european_price():
    res = longstaff_schwartz(S0, K, T, R, SIG, n_paths=400_000, n_steps=1, seed=7)
    exact = float(bs_put(S0, K, T, R, SIG))
    assert abs(res.price - exact) < 4.0 * res.std_error
    assert res.early_exercise_fraction == 0.0


# ---------------------------------------------------------------------------
# Against the exact Bermudan target
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_dates", [10, 50])
def test_lsm_matches_the_exact_bermudan_lattice_value(n_dates):
    """LSM with n exercise dates estimates the n-date Bermudan value, which the
    lattice gives exactly.  Comparing it to the continuous-exercise American
    value would conflate the regression error with the exercise-date bias."""
    berm = crr(S0, K, T, R, SIG, 40_000, kind="put", bermudan_dates=n_dates).price
    res = longstaff_schwartz(S0, K, T, R, SIG, n_paths=300_000, n_steps=n_dates,
                             degree=3, seed=17, out_of_sample=True)
    # A fixed suboptimal policy is a lower bound, so allow a one-sided gap, but
    # it must not exceed the benchmark by more than sampling noise.
    assert res.price < berm + 3.0 * res.std_error
    assert res.price > berm - 0.02, f"LSM {res.price} far below Bermudan {berm}"


def test_bermudan_value_increases_with_exercise_dates_towards_the_american():
    am = crr_price(S0, K, T, R, SIG, 40_000, kind="put")
    dates = np.array([5, 10, 25, 50, 100, 200])
    vals = np.array([crr(S0, K, T, R, SIG, 40_000, kind="put", bermudan_dates=int(d)).price
                     for d in dates])
    assert np.all(np.diff(vals) > 0), "Bermudan value must increase with exercise dates"
    assert np.all(vals < am), "Bermudan value cannot exceed the American value"

    # The gap to the American value is O(1/n_dates), so the gap should shrink by
    # the same factor as the date count grows.  (Note the date grid is not a pure
    # doubling: 10 -> 25 is a factor 2.5, and the gap ratio must track that.)
    gaps = am - vals
    observed = gaps[:-1] / gaps[1:]
    expected = dates[1:] / dates[:-1]
    assert np.allclose(observed, expected, rtol=0.10), (observed, expected)

    slope = np.polyfit(np.log(dates), np.log(gaps), 1)[0]
    assert -1.1 < slope < -0.9, f"Bermudan gap order {slope:.3f}, expected ~ -1"


def test_lsm_is_close_to_the_american_value_with_many_dates():
    am = crr_price(S0, K, T, R, SIG, 40_000, kind="put")
    res = longstaff_schwartz(S0, K, T, R, SIG, n_paths=300_000, n_steps=200,
                             degree=4, basis="chebyshev", seed=23)
    assert abs(res.price - am) < 0.03


# ---------------------------------------------------------------------------
# Bounds, invariance, robustness
# ---------------------------------------------------------------------------

def test_price_respects_intrinsic_and_perpetual_bounds():
    res = longstaff_schwartz(S0, K, T, R, SIG, n_paths=100_000, n_steps=50, seed=1)
    assert res.price >= max(K - S0, 0.0)
    assert res.price <= float(perpetual_put_price(S0, K, R, SIG))


def test_deep_in_the_money_exercises_immediately():
    """At S0 = 20 with K = 100 the payoff is 80 and waiting cannot beat it."""
    res = longstaff_schwartz(20.0, K, T, R, SIG, n_paths=50_000, n_steps=50, seed=2)
    assert res.exercised_at_zero
    assert res.price == pytest.approx(80.0)


@pytest.mark.parametrize("basis", ["poly", "laguerre", "chebyshev"])
def test_bases_agree_within_sampling_error(basis):
    berm = crr(S0, K, T, R, SIG, 40_000, kind="put", bermudan_dates=50).price
    res = longstaff_schwartz(S0, K, T, R, SIG, n_paths=300_000, n_steps=50,
                             degree=4, basis=basis, seed=31)
    assert abs(res.price - berm) < 0.03, basis


def test_confidence_interval_is_consistent_with_the_standard_error():
    res = longstaff_schwartz(S0, K, T, R, SIG, n_paths=50_000, n_steps=25, seed=4)
    assert res.ci_low == pytest.approx(res.price - Z_975 * res.std_error)
    assert res.ci_high == pytest.approx(res.price + Z_975 * res.std_error)
    assert Z_975 == pytest.approx(1.959963985, abs=1e-8)


def test_deterministic_given_a_seed():
    a = longstaff_schwartz(S0, K, T, R, SIG, n_paths=20_000, n_steps=20, seed=9)
    b = longstaff_schwartz(S0, K, T, R, SIG, n_paths=20_000, n_steps=20, seed=9)
    assert a.price == b.price and a.std_error == b.std_error


def test_out_of_sample_uses_independent_paths():
    """The out-of-sample estimator must not reuse the fitting paths."""
    a = longstaff_schwartz(S0, K, T, R, SIG, n_paths=50_000, n_steps=25, seed=6,
                           out_of_sample=False)
    b = longstaff_schwartz(S0, K, T, R, SIG, n_paths=50_000, n_steps=25, seed=6,
                           out_of_sample=True)
    assert a.price != b.price


def test_too_few_in_the_money_paths_skips_the_regression_rather_than_failing():
    """Far out of the money with few paths, some dates have no usable cross-section."""
    res = longstaff_schwartz(180.0, K, T, R, SIG, n_paths=300, n_steps=50,
                             degree=6, seed=8)
    assert res.n_skipped_regressions > 0
    assert np.isfinite(res.price) and res.price >= 0.0


def test_invalid_arguments():
    with pytest.raises(ValueError):
        longstaff_schwartz(S0, K, T, R, SIG, kind="straddle")
    with pytest.raises(ValueError):
        longstaff_schwartz(S0, K, T, R, SIG, n_steps=0)
    with pytest.raises(ValueError):
        crr(S0, K, T, R, SIG, 100, kind="put", bermudan_dates=7)  # 100 % 7 != 0
    with pytest.raises(ValueError):
        crr(S0, K, T, R, SIG, 100, kind="put", exercise="european", bermudan_dates=10)


# ---------------------------------------------------------------------------
# Mutation test: leaking the future must change the answer
# ---------------------------------------------------------------------------

def test_perfect_foresight_policy_is_materially_higher_than_lsm():
    r"""Sanity check on the whole look-ahead argument.

    If the exercise decision is allowed to see the realised future -- exercise at
    the date maximising the actual discounted pathwise payoff -- the estimate is
    the *perfect-foresight* upper bound, which strictly exceeds the true American
    value.  This test does not validate the estimator; it validates that the test
    suite could detect leakage if it were introduced, by showing how large the
    effect would be.
    """
    n_steps = 50
    S = simulate_gbm(S0, R, SIG, T, 100_000, n_steps, seed=13)
    t = np.arange(n_steps + 1) * (T / n_steps)
    disc_payoff = np.maximum(K - S, 0.0) * np.exp(-R * t)[None, :]
    foresight = disc_payoff[:, 1:].max(axis=1).mean()

    lsm = longstaff_schwartz(S0, K, T, R, SIG, n_paths=100_000, n_steps=n_steps, seed=13)
    am = crr_price(S0, K, T, R, SIG, 40_000, kind="put")
    assert foresight > am + 1.0, f"foresight bound {foresight} not above American {am}"
    assert foresight > lsm.price + 1.0
