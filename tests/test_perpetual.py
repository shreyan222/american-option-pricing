"""Tests for the perpetual-put closed form, and the limits it pins down."""

import numpy as np
import pytest

from amopt.binomial import crr_price
from amopt.perpetual import beta_minus, perpetual_put_boundary, perpetual_put_price


def test_beta_root_solves_the_characteristic_quadratic():
    for r, sigma, q in [(0.05, 0.2, 0.0), (0.1, 0.35, 0.03), (0.02, 0.6, 0.07)]:
        b = beta_minus(r, sigma, q)
        residual = 0.5 * sigma**2 * b * (b - 1.0) + (r - q) * b - r
        assert residual == pytest.approx(0.0, abs=1e-12)
        assert b < 0.0


def test_zero_dividend_closed_form():
    r"""With q = 0 the root is exactly -2r/sigma^2 and S_inf = K*gamma/(1+gamma)."""
    r, sigma, K = 0.05, 0.2, 100.0
    gamma = 2.0 * r / sigma**2
    assert beta_minus(r, sigma, 0.0) == pytest.approx(-gamma, rel=1e-12)
    assert perpetual_put_boundary(K, r, sigma) == pytest.approx(K * gamma / (1 + gamma), rel=1e-12)


def test_value_matching_and_smooth_pasting_hold():
    K, r, sigma, q = 100.0, 0.06, 0.3, 0.01
    S_inf = perpetual_put_boundary(K, r, sigma, q)
    assert perpetual_put_price(S_inf, K, r, sigma, q) == pytest.approx(K - S_inf, rel=1e-12)
    h = 1e-5
    slope = (
        perpetual_put_price(S_inf + h, K, r, sigma, q)
        - perpetual_put_price(S_inf - h, K, r, sigma, q)
    ) / (2 * h)
    # one-sided from the continuation side, since below S_inf the payoff is linear
    slope_up = (
        perpetual_put_price(S_inf + 2 * h, K, r, sigma, q)
        - perpetual_put_price(S_inf, K, r, sigma, q)
    ) / (2 * h)
    assert slope == pytest.approx(-1.0, abs=1e-4)
    assert slope_up == pytest.approx(-1.0, abs=1e-4)


def test_perpetual_solves_the_ode_in_the_continuation_region():
    K, r, sigma, q = 100.0, 0.05, 0.25, 0.02
    S = np.array([90.0, 110.0, 150.0])
    h = 1e-3
    V = lambda x: perpetual_put_price(x, K, r, sigma, q)  # noqa: E731
    d1 = (V(S + h) - V(S - h)) / (2 * h)
    d2 = (V(S + h) - 2 * V(S) + V(S - h)) / h**2
    residual = 0.5 * sigma**2 * S**2 * d2 + (r - q) * S * d1 - r * V(S)
    assert np.max(np.abs(residual)) < 1e-5


def test_boundary_is_below_the_strike():
    for r, sigma in [(0.01, 0.2), (0.05, 0.2), (0.2, 0.15), (0.05, 0.8)]:
        assert 0.0 < perpetual_put_boundary(100.0, r, sigma) < 100.0


def test_boundary_decreases_with_volatility():
    """More volatility means more option value in waiting: exercise later, lower S*."""
    b = [perpetual_put_boundary(100.0, 0.05, s) for s in (0.1, 0.2, 0.4, 0.8)]
    assert np.all(np.diff(b) < 0)


def test_boundary_increases_with_rate():
    """More interest on the exercise proceeds means exercise sooner: higher S*."""
    b = [perpetual_put_boundary(100.0, r, 0.25) for r in (0.01, 0.03, 0.08, 0.15)]
    assert np.all(np.diff(b) > 0)


def test_zero_rate_never_exercises_but_the_supremum_is_the_strike():
    r"""With r = 0 the exercise region is empty (S_inf = 0), yet S_t -> 0 a.s.
    under Q, so sup_tau E[(K - S_tau)^+] = K -- approached, never attained."""
    assert perpetual_put_boundary(100.0, 0.0, 0.2) == pytest.approx(0.0, abs=1e-12)
    assert float(perpetual_put_price(100.0, 100.0, 0.0, 0.2)) == pytest.approx(100.0)
    # dominance must still hold against a finite-maturity r = 0 American put
    assert crr_price(100.0, 100.0, 5.0, 0.0, 0.2, 2000, kind="put") < 100.0


def test_dominance_holds_in_every_configured_regime():
    from amopt.config import REGIMES

    for name, p in REGIMES.items():
        perp = float(perpetual_put_price(p.S0, p.K, p.r, p.sigma, p.q))
        am = crr_price(p.S0, p.K, p.T, p.r, p.sigma, 2000, p.q, "put", "american")
        assert am <= perp + 1e-8, f"{name}: American {am} exceeded perpetual {perp}"


def test_perpetual_dominates_finite_maturity_and_is_approached_from_below():
    r"""V^perp >= V^amer(T) for every T, with V^amer(T) increasing in T."""
    S0, K, r, sigma = 100.0, 100.0, 0.05, 0.2
    perp = float(perpetual_put_price(S0, K, r, sigma))
    vals = [crr_price(S0, K, T, r, sigma, 3000, kind="put") for T in (1.0, 5.0, 20.0, 50.0)]
    assert np.all(np.diff(vals) > 0), f"American put not increasing in T: {vals}"
    assert all(v < perp + 1e-8 for v in vals), f"finite-T value exceeded perpetual: {vals}, {perp}"
    assert vals[-1] > 0.85 * perp, "T=50 should already be close to the perpetual value"


def test_price_above_intrinsic_and_below_strike():
    S = np.linspace(1.0, 400.0, 200)
    v = perpetual_put_price(S, 100.0, 0.05, 0.2)
    assert np.all(v >= np.maximum(100.0 - S, 0.0) - 1e-12)
    assert np.all(v <= 100.0 + 1e-12)
    assert np.all(np.diff(v) < 1e-12)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        beta_minus(0.05, 0.0)
    with pytest.raises(ValueError):
        beta_minus(-0.01, 0.2)
