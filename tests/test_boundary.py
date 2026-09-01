"""Falsification tests for the early-exercise boundary.

Each test corresponds to a numbered prediction in `docs/03_exercise_boundary.md`,
which was written before the sensitivity study was run.  The table in §3.3 of
that document lists what each failure would mean.
"""

import numpy as np
import pytest

from amopt.crank_nicolson import solve_pde
from amopt.perpetual import perpetual_put_boundary

BASE = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
FINE = dict(M=1600, N=1600, omega=1.4)

_CACHE: dict = {}


def boundary(**kw):
    """Solve and return ``(result, t, S*(t))`` sorted by ``t``.  Memoised: these
    tests reuse the same solves many times and the PDE is deterministic."""
    p = {**BASE, **FINE, "kind": "put"}
    p.update(kw)
    key = tuple(sorted(p.items()))
    if key not in _CACHE:
        res = solve_pde(**p)
        order = np.argsort(res.boundary_t)
        _CACHE[key] = (res, res.boundary_t[order], res.boundary_S[order])
    return _CACHE[key]


# --- P1: monotone in t -----------------------------------------------------

def test_p1_boundary_is_non_decreasing_in_time():
    res, t, Sb = boundary()
    assert np.all(np.diff(Sb) > -2.0 * res.dS), "boundary decreases in t beyond grid noise"
    assert np.polyfit(t, Sb, 1)[0] > 0.0


# --- P2: upper bound and terminal value ------------------------------------

@pytest.mark.parametrize("q,expected", [(0.0, 100.0), (0.02, 100.0), (0.06, 100.0 * 0.05 / 0.06),
                                        (0.10, 50.0), (0.15, 100.0 * 0.05 / 0.15)])
def test_p2_terminal_boundary_is_min_K_and_rK_over_q(q, expected):
    r"""lim_{tau->0+} S* = min(K, rK/q).  With q = 10% and r = 5% this predicts 50,
    not 100 -- a sharp, easily falsified claim.

    The tolerance differs by case *because the two cases approach their limits at
    different rates*, which P3 predicts:

    * When ``q <= r`` the limit is ``K`` and the boundary approaches it along the
      square-root-log law, ``K - S* ~ K sigma sqrt(tau ln(1/tau))``.  At the last
      resolved time level that gap is of order 1 on a strike of 100, so
      demanding 1% agreement there would be demanding the numerics beat the
      asymptotics.  The test asserts the gap matches the predicted scale.
    * When ``q > r`` the limit ``rK/q`` sits away from the payoff kink and is
      approached quickly; 1% is then the right bar.
    """
    res, t, Sb = boundary(q=q, N=1600)
    just_before = Sb[-2]  # the last entry is tau = 0 exactly; see the next test
    assert np.all(Sb[:-1] <= expected + 1e-6), "boundary exceeded min(K, rK/q)"

    if expected >= 100.0 - 1e-9:  # limit is K: approach governed by P3
        tau = t[-1] - t[-2]
        scale = 100.0 * 0.2 * np.sqrt(tau * np.log(1.0 / tau))
        assert (expected - just_before) == pytest.approx(scale, rel=0.35), (
            q, just_before, expected, scale
        )
    else:
        assert just_before == pytest.approx(expected, rel=0.01), (q, just_before, expected)


def test_p2_boundary_is_discontinuous_at_maturity_when_q_exceeds_r():
    """At tau = 0 exactly every in-the-money put is exercised, so S*(T) = K, while
    the limit from tau > 0 is rK/q.  The jump is real, not a numerical artefact."""
    res, t, Sb = boundary(q=0.10, N=1600)
    assert Sb[-1] == pytest.approx(100.0, abs=1e-9)
    assert Sb[-2] == pytest.approx(50.0, rel=0.01)


# --- P3: near-maturity asymptotics -----------------------------------------

def test_p3_near_maturity_square_root_log_law():
    r"""K - S*(T-tau) ~ K sigma sqrt(tau ln(1/tau)).

    Two checks.  Pointwise, the ratio to the predicted scale must lie within the
    15% band that `docs/03` §P3 committed to in advance (the corrections are
    O(1/ln(1/tau)), which is 0.14 at tau = 1e-3, so equality is not available).
    And structurally, regressing ``K - S*`` on ``sqrt(tau ln(1/tau))`` through the
    origin must recover a slope of ``K*sigma`` -- this tests the *functional form*
    rather than one point, and would fail for a plain ``sqrt(tau)`` law.
    """
    res, t, Sb = boundary(M=3200, N=3200)
    tau = 1.0 - t
    K, sigma = 100.0, 0.2

    ratios, xs, ys = [], [], []
    for target in (1e-3, 3e-3, 1e-2, 3e-2):
        i = int(np.argmin(np.abs(tau - target)))
        tt = tau[i]
        x = np.sqrt(tt * np.log(1.0 / tt))
        ratios.append((K - Sb[i]) / (K * sigma * x))
        xs.append(x); ys.append(K - Sb[i])
    ratios = np.array(ratios)
    assert np.all(np.abs(ratios - 1.0) < 0.15), ratios

    xs, ys = np.array(xs), np.array(ys)
    slope = float(xs @ ys / (xs @ xs))  # least squares through the origin
    assert slope == pytest.approx(K * sigma, rel=0.15), (slope, K * sigma)


def test_p3_slope_at_maturity_diverges_under_time_refinement():
    r"""P3 implies ``dS*/dt -> infinity`` as ``t -> T``.

    A finite-difference slope on a fixed time grid can never *be* infinite, so
    the falsifiable statement is that the last-step slope **grows without bound
    as the time grid is refined**.  A boundary meeting the strike with a finite
    slope would give a slope that converges instead.  A single ratio against the
    mid-life slope on one grid would only have tested that the boundary is
    steeper near maturity, which is much weaker.
    """
    slopes = []
    for N in (400, 800, 1600, 3200):
        _, t, Sb = boundary(M=3200, N=N)
        slopes.append((Sb[-1] - Sb[-2]) / (t[-1] - t[-2]))
    slopes = np.array(slopes)
    assert np.all(np.diff(slopes) > 0), f"last-step slope did not diverge: {slopes}"
    assert slopes[-1] > 3.0 * slopes[0], slopes

    # ... and it is far steeper near maturity than at mid-life on any one grid.
    _, t, Sb = boundary(M=3200, N=3200)
    d = np.diff(Sb) / np.diff(t)
    assert d[-1] > 20.0 * d[len(d) // 2], (d[-1], d[len(d) // 2])


# --- P4/P5: comparative statics --------------------------------------------

@pytest.mark.parametrize("t_idx", [0, -50])
def test_p4_boundary_decreases_with_volatility(t_idx):
    vals = [boundary(sigma=s)[2][t_idx] for s in (0.10, 0.20, 0.30, 0.45)]
    assert np.all(np.diff(vals) < 0), vals


@pytest.mark.parametrize("t_idx", [0, -50])
def test_p5_boundary_increases_with_rate(t_idx):
    vals = [boundary(r=r)[2][t_idx] for r in (0.01, 0.03, 0.06, 0.12)]
    assert np.all(np.diff(vals) > 0), vals


# --- P6: maturity ----------------------------------------------------------

def test_p6_boundary_at_zero_decreases_with_maturity_towards_the_perpetual_limit():
    S_inf = perpetual_put_boundary(100.0, 0.05, 0.2)
    vals = [boundary(T=T)[2][0] for T in (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)]
    assert np.all(np.diff(vals) < 0), vals
    assert np.all(np.array(vals) > S_inf), (vals, S_inf)
    assert vals[-1] < S_inf * 1.10, f"T=10 boundary {vals[-1]} not close to S_inf {S_inf}"


def test_p6_perpetual_limit_is_a_hard_floor_in_every_regime():
    from amopt.config import REGIMES

    for name, p in REGIMES.items():
        if p.r <= 0.0:
            continue
        S_inf = perpetual_put_boundary(p.K, p.r, p.sigma, p.q)
        _, _, Sb = boundary(S0=p.S0, K=p.K, T=p.T, r=p.r, sigma=p.sigma, q=p.q)
        assert Sb[0] > S_inf, f"{name}: S*(0)={Sb[0]} fell below the perpetual floor {S_inf}"


# --- P7: scale invariance --------------------------------------------------

@pytest.mark.parametrize("K", [1.0, 50.0, 100.0, 500.0])
def test_p7_boundary_scales_with_the_strike(K):
    _, _, ref = boundary(K=100.0, S0=100.0)
    _, _, got = boundary(K=K, S0=K)
    assert got[0] / K == pytest.approx(ref[0] / 100.0, rel=2e-3)


def test_p7_boundary_does_not_depend_on_the_query_spot():
    r"""``S0`` enters the free-boundary problem nowhere -- only the query point.

    On an *identical* grid the boundary must therefore be bitwise identical.  The
    grid is pinned with an explicit ``S_max`` because the default domain is
    ``S_max_mult * max(K, S0)``, so leaving it to the default changes the mesh
    along with ``S0`` and the comparison would be testing the grid, not the
    physics.
    """
    a = boundary(S0=60.0, S_max=400.0)[2]
    b = boundary(S0=140.0, S_max=400.0)[2]
    assert np.array_equal(a, b)


def test_p7_default_domain_depends_on_the_spot_but_only_to_grid_accuracy():
    r"""With the default domain ``S_max = 4 max(K, S0)``, the mesh moves with
    ``S0``, so the extracted boundary shifts by the cell-alignment noise that
    `RESULTS.md` §7 measures.  The tolerance is one grid spacing -- the accuracy
    the boundary actually has -- not an arbitrary constant.
    """
    ra, _, Sa = boundary(S0=60.0)
    rb, _, Sb_ = boundary(S0=140.0)
    assert abs(Sa[0] - Sb_[0]) < max(ra.dS, rb.dS), (Sa[0], Sb_[0], ra.dS, rb.dS)


# --- the boundary must converge under refinement ---------------------------

def test_boundary_converges_under_grid_refinement():
    vals = [boundary(M=M, N=3200)[2][0] for M in (800, 3200, 6400)]
    assert abs(vals[-1] - vals[-2]) < 5e-3, vals
    assert all(80.0 < v < 82.0 for v in vals), vals


def test_operator_identity_tolerance_scales_with_the_grid():
    """Regression guard: an absolute 1e-9 bound on the cancellation residual made
    the solver refuse to run at M >= 12800 for pure round-off."""
    res = solve_pde(**BASE, kind="put", M=12800, N=200, omega=1.4)
    assert np.isfinite(res.price)
