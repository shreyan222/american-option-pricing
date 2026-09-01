"""Analytic-property tests for the closed-form Black-Scholes layer.

These are *identity* tests rather than regression tests: each one would fail
under a plausible sign or indexing error, and none of them depend on a value
produced by this repository.
"""

import numpy as np
import pytest

from amopt.black_scholes import bs_call, bs_greeks, bs_price, bs_put, d1_d2

BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, q=0.0)


def test_reference_values_from_literature():
    """Hull, *Options, Futures and Other Derivatives*: S=K=100, T=1, r=5%, sigma=20%."""
    assert bs_call(100, 100, 1.0, 0.05, 0.2) == pytest.approx(10.450583572, abs=1e-8)
    assert bs_put(100, 100, 1.0, 0.05, 0.2) == pytest.approx(5.573526022, abs=1e-8)


@pytest.mark.parametrize("S", [50.0, 90.0, 100.0, 110.0, 200.0])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.6])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(S, sigma, q):
    r"""C - P = S e^{-qT} - K e^{-rT} must hold exactly, not approximately."""
    K, T, r = 100.0, 1.5, 0.04
    c = bs_call(S, K, T, r, sigma, q)
    p = bs_put(S, K, T, r, sigma, q)
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_no_arbitrage_bounds(kind):
    S, K, T, r, sigma = 100.0, 95.0, 2.0, 0.03, 0.35
    v = bs_price(S, K, T, r, sigma, kind=kind)
    if kind == "call":
        assert max(S - K * np.exp(-r * T), 0.0) <= v <= S
    else:
        assert max(K * np.exp(-r * T) - S, 0.0) <= v <= K * np.exp(-r * T)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_monotone_in_volatility(kind):
    """Vanilla prices are strictly increasing in sigma (vega > 0)."""
    sig = np.linspace(0.01, 1.5, 60)
    v = bs_price(100.0, 100.0, 1.0, 0.05, sig, kind=kind)
    assert np.all(np.diff(v) > 0)


def test_call_monotone_in_spot_put_decreasing():
    S = np.linspace(1.0, 300.0, 200)
    assert np.all(np.diff(bs_call(S, 100.0, 1.0, 0.05, 0.2)) > 0)
    assert np.all(np.diff(bs_put(S, 100.0, 1.0, 0.05, 0.2)) < 0)


def test_convexity_in_strike():
    """Butterfly spreads have non-negative value => price is convex in K."""
    K = np.linspace(50.0, 150.0, 201)
    c = bs_call(100.0, K, 1.0, 0.05, 0.25)
    assert np.all(np.diff(c, 2) > -1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_zero_volatility_limit(kind):
    r"""sigma -> 0 collapses to the discounted forward intrinsic."""
    S, K, T, r = 110.0, 100.0, 1.0, 0.05
    limit = bs_price(S, K, T, r, 1e-12, kind=kind)
    tiny = bs_price(S, K, T, r, 1e-6, kind=kind)
    assert limit == pytest.approx(tiny, abs=1e-6)
    fwd = S * np.exp(r * T)
    expected = np.exp(-r * T) * max(fwd - K, 0.0) if kind == "call" else np.exp(-r * T) * max(K - fwd, 0.0)
    assert limit == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S", [80.0, 100.0, 120.0])
def test_expiry_limit_is_payoff(kind, S):
    K = 100.0
    v = bs_price(S, K, 0.0, 0.05, 0.2, kind=kind)
    expected = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    assert v == pytest.approx(expected, abs=1e-12)


def test_deep_itm_call_is_discounted_forward():
    """As S/K -> infinity the call approaches S e^{-qT} - K e^{-rT}."""
    S, K, T, r, q = 1e6, 100.0, 1.0, 0.05, 0.02
    v = bs_call(S, K, T, r, 0.2, q)
    assert v == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), rel=1e-12)


def test_deep_otm_put_is_worthless():
    assert bs_put(1e6, 100.0, 1.0, 0.05, 0.2) == pytest.approx(0.0, abs=1e-12)


def test_d1_d2_relation():
    d1, d2 = d1_d2(**BASE)
    assert d1 - d2 == pytest.approx(BASE["sigma"] * np.sqrt(BASE["T"]), abs=1e-12)


# --------------------------------------------------------------------------
# Greeks: validated against central finite differences of the price function.
# --------------------------------------------------------------------------

def _fd(fn, x, h):
    return (fn(x + h) - fn(x - h)) / (2.0 * h)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S", [80.0, 100.0, 125.0])
def test_greeks_match_finite_differences(kind, S):
    K, T, r, sigma, q = 100.0, 1.0, 0.05, 0.25, 0.01
    g = bs_greeks(S, K, T, r, sigma, q, kind)

    delta_fd = _fd(lambda s: bs_price(s, K, T, r, sigma, q, kind), S, 1e-4)
    assert g["delta"] == pytest.approx(delta_fd, rel=1e-6)

    h = 1e-3
    gamma_fd = (
        bs_price(S + h, K, T, r, sigma, q, kind)
        - 2 * bs_price(S, K, T, r, sigma, q, kind)
        + bs_price(S - h, K, T, r, sigma, q, kind)
    ) / h**2
    assert g["gamma"] == pytest.approx(gamma_fd, rel=1e-4)

    vega_fd = _fd(lambda v: bs_price(S, K, T, r, v, q, kind), sigma, 1e-5)
    assert g["vega"] == pytest.approx(vega_fd, rel=1e-6)

    rho_fd = _fd(lambda rr: bs_price(S, K, T, rr, sigma, q, kind), r, 1e-6)
    assert g["rho"] == pytest.approx(rho_fd, rel=1e-5)

    # theta = dV/dt with calendar time; increasing t decreases time-to-maturity.
    theta_fd = -_fd(lambda tt: bs_price(S, K, tt, r, sigma, q, kind), T, 1e-5)
    assert g["theta"] == pytest.approx(theta_fd, rel=1e-5)


def test_gamma_and_vega_are_kind_independent():
    """Put and call differ by a forward, which has zero gamma and vega."""
    c = bs_greeks(105.0, 100.0, 0.7, 0.03, 0.3, 0.01, "call")
    p = bs_greeks(105.0, 100.0, 0.7, 0.03, 0.3, 0.01, "put")
    assert c["gamma"] == pytest.approx(p["gamma"], rel=1e-12)
    assert c["vega"] == pytest.approx(p["vega"], rel=1e-12)
    assert c["delta"] - p["delta"] == pytest.approx(np.exp(-0.01 * 0.7), rel=1e-12)


def test_broadcasting_shapes():
    S = np.linspace(80, 120, 7)
    T = np.array([0.5, 1.0])[:, None]
    v = bs_put(S, 100.0, T, 0.05, 0.2)
    assert v.shape == (2, 7)


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        bs_price(100, 100, 1, 0.05, 0.2, kind="straddle")
