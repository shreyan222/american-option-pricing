"""Tests for the CRR lattice.

The lattice is the independent benchmark used later against the PDE and Monte
Carlo solvers, so it is validated here against the *analytic* European price and
against exercise-policy theorems that hold independently of any numerics.
"""

import numpy as np
import pytest

from amopt.binomial import crr, crr_boundary, crr_price, crr_price_averaged
from amopt.black_scholes import bs_call, bs_greeks, bs_put

S0, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.2


@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_lattice_converges_to_black_scholes(kind):
    exact = bs_call(S0, K, T, R, SIG) if kind == "call" else bs_put(S0, K, T, R, SIG)
    err = [
        abs(crr_price(S0, K, T, R, SIG, N, kind=kind, exercise="european") - exact)
        for N in (200, 400, 800, 1600, 3200)
    ]
    assert err[-1] < 1e-3
    assert np.all(np.diff(err) < 0), f"errors not monotonically decreasing: {err}"


@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_convergence_order_is_first_order(kind):
    """CRR is O(1/N); estimate the slope from consecutive step-count doublings."""
    exact = bs_call(S0, K, T, R, SIG) if kind == "call" else bs_put(S0, K, T, R, SIG)
    Ns = np.array([200, 400, 800, 1600, 3200])
    err = np.array(
        [abs(crr_price(S0, K, T, R, SIG, int(N), kind=kind, exercise="european") - exact) for N in Ns]
    )
    slope = np.polyfit(np.log(Ns), np.log(err), 1)[0]
    assert -1.25 < slope < -0.75, f"observed convergence slope {slope:.3f}, expected ~ -1"


@pytest.mark.parametrize("S", [70.0, 100.0, 130.0])
def test_american_put_dominates_european_and_intrinsic(S):
    am = crr_price(S, K, T, R, SIG, 1000, kind="put", exercise="american")
    eu = crr_price(S, K, T, R, SIG, 1000, kind="put", exercise="european")
    assert am >= eu - 1e-12
    assert am >= max(K - S, 0.0) - 1e-12


def test_early_exercise_premium_is_strictly_positive():
    """With r > 0 the American put must be worth strictly more than the European."""
    am = crr_price(S0, K, T, R, SIG, 2000, kind="put", exercise="american")
    eu = crr_price(S0, K, T, R, SIG, 2000, kind="put", exercise="european")
    assert am - eu > 0.4, f"premium {am - eu:.4f} implausibly small"


def test_american_call_equals_european_when_no_dividends():
    """Merton: it is never optimal to exercise an American call early if q = 0."""
    am = crr_price(S0, K, T, R, SIG, 800, q=0.0, kind="call", exercise="american")
    eu = crr_price(S0, K, T, R, SIG, 800, q=0.0, kind="call", exercise="european")
    assert am == pytest.approx(eu, abs=1e-12)


def test_american_call_exceeds_european_with_dividends():
    am = crr_price(S0, K, T, R, SIG, 800, q=0.08, kind="call", exercise="american")
    eu = crr_price(S0, K, T, R, SIG, 800, q=0.08, kind="call", exercise="european")
    assert am > eu + 1e-4


def test_american_put_equals_european_when_rate_is_zero():
    r"""With r = q = 0 the discounted payoff is a submartingale (Jensen), so
    waiting is always weakly better and there is no early-exercise premium."""
    am = crr_price(S0, K, T, 0.0, SIG, 800, kind="put", exercise="american")
    eu = crr_price(S0, K, T, 0.0, SIG, 800, kind="put", exercise="european")
    assert am == pytest.approx(eu, abs=1e-12)


def test_american_put_call_parity_bounds():
    r"""For q = 0: S_0 - K <= C_am - P_am <= S_0 - K e^{-rT}."""
    c = crr_price(S0, K, T, R, SIG, 1000, kind="call", exercise="american")
    p = crr_price(S0, K, T, R, SIG, 1000, kind="put", exercise="american")
    assert S0 - K - 1e-9 <= c - p <= S0 - K * np.exp(-R * T) + 1e-9


def test_price_is_cauchy_in_step_count():
    """Successive refinements agree; self-consistency without a hard-coded value."""
    vals = [crr_price(S0, K, T, R, SIG, N, kind="put") for N in (1000, 2000, 4000)]
    assert abs(vals[1] - vals[0]) < 5e-3
    assert abs(vals[2] - vals[1]) < abs(vals[1] - vals[0]) + 1e-12


def test_averaged_lattice_beats_single_lattice():
    """Averaging N and N+1 cancels the odd/even oscillation."""
    exact = bs_put(S0, K, T, R, SIG)
    single = abs(crr_price(S0, K, T, R, SIG, 501, kind="put", exercise="european") - exact)
    avg = abs(crr_price_averaged(S0, K, T, R, SIG, 501, kind="put", exercise="european") - exact)
    assert avg < single


@pytest.mark.parametrize("kind", ["call", "put"])
def test_lattice_greeks_match_analytic_european(kind):
    res = crr(S0, K, T, R, SIG, 4000, kind=kind, exercise="european", greeks=True)
    g = bs_greeks(S0, K, T, R, SIG, 0.0, kind)
    assert res.delta == pytest.approx(g["delta"], abs=2e-3)
    assert res.gamma == pytest.approx(g["gamma"], abs=2e-4)
    assert res.theta == pytest.approx(g["theta"], abs=2e-2)


def test_arbitrageable_lattice_raises():
    """Huge drift with tiny vol and one step breaks 0 <= p <= 1; we refuse to clamp."""
    with pytest.raises(ValueError, match="outside"):
        crr_price(100.0, 100.0, 1.0, 0.5, 0.01, 1, kind="put")


def test_zero_maturity_returns_payoff():
    assert crr_price(90.0, 100.0, 0.0, 0.05, 0.2, 10, kind="put") == pytest.approx(10.0)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        crr_price(100, 100, 1, 0.05, 0.2, 0)
    with pytest.raises(ValueError):
        crr(100, 100, 1, 0.05, 0.2, 10, exercise="bermudan")
    with pytest.raises(ValueError):
        crr(100, 100, 1, 0.05, 0.0, 10)


def test_exercise_boundary_shape_and_monotonicity():
    r"""S*(t) for a put is non-decreasing in t and bounded above by
    min(K, rK/q) = K when q = 0; at maturity it equals K."""
    t, Sb = crr_boundary(S0, K, T, R, SIG, 1500, kind="put")
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(T)

    finite = np.isfinite(Sb)
    # The undefined region must be a contiguous *prefix*: the lattice cone
    # S0*exp(+-i*sigma*sqrt(dt)) is too narrow to reach S*(t) for small t.
    assert not finite[0], "expected the cone to miss the boundary at t=0"
    assert finite[-1]
    assert np.all(np.diff(finite.astype(int)) >= 0), "nan gaps in the middle of the boundary"

    assert np.all(Sb[finite] <= K + 1e-9), "put boundary must not exceed the strike"
    # Terminal boundary -> K, up to one node spacing K*(u-1).
    node_gap = K * (np.exp(SIG * np.sqrt(T / 1500)) - 1.0)
    assert Sb[-1] == pytest.approx(K, abs=2 * node_gap)
    # Allow the lattice staircase, but require a positive overall trend.
    assert np.polyfit(t[finite], Sb[finite], 1)[0] > 0.0


def test_deterministic():
    a = crr_price(S0, K, T, R, SIG, 700, kind="put")
    b = crr_price(S0, K, T, R, SIG, 700, kind="put")
    assert a == b
