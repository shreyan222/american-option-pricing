"""Tests for the Crank-Nicolson / PSOR solver.

Validation strategy, in order of strength:

1. **Analytic** -- the European limit must reproduce Black-Scholes.
2. **Independent numerics** -- the American price must match the CRR lattice,
   which shares no code with the PDE solver.
3. **Internal cross-checks** -- three different LCP solvers (vectorised red-black
   PSOR, a literal lexicographic PSOR loop, and the direct Brennan-Schwartz
   elimination) must agree.
4. **Exact identities** -- the discrete operator applied to constants and to the
   identity function, and the M-matrix row count against its closed-form
   threshold.
5. **Invariance** -- the answer must not depend on `omega`, which only affects
   cost.
"""

import numpy as np
import pytest

from amopt.binomial import crr_price
from amopt.black_scholes import bs_price, bs_put
from amopt.config import REGIMES
from amopt.crank_nicolson import (
    brennan_schwartz,
    build_grid,
    operator_coefficients,
    psor_lexicographic,
    psor_redblack,
    solve_pde,
)
from amopt.perpetual import perpetual_put_boundary

BASE = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


# ---------------------------------------------------------------------------
# Grid and operator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M", [50, 101, 400, 777])
@pytest.mark.parametrize("mult", [2.0, 4.0, 7.5])
def test_strike_lands_exactly_on_a_grid_node(M, mult):
    K = 100.0
    g = build_grid(K, M, mult * K)
    assert g.S[g.i_K] == pytest.approx(K, abs=1e-12)
    assert g.S.size == M + 1
    assert np.allclose(np.diff(g.S), g.dS)
    # Pinning the strike rescales the grid by at most half a cell measured at
    # the strike: |S_max/target - 1| <= 1/(2 i_K).
    assert abs(g.S_max / (mult * K) - 1.0) <= 0.5 / g.i_K + 1e-12


@pytest.mark.parametrize("upwind", [False, True])
@pytest.mark.parametrize("r,sigma,q", [(0.05, 0.2, 0.0), (0.1, 0.35, 0.04), (0.0, 0.25, 0.06)])
def test_operator_identities(upwind, r, sigma, q):
    r"""A_h applied to 1 gives -r; applied to S gives -qS. Both are exact for
    central differences, so they test the algebra rather than the discretisation."""
    M = 200
    a, b, c = operator_coefficients(M, r, sigma, q, upwind=upwind)
    i = np.arange(1, M, dtype=float)
    assert np.allclose(a + b + c, -r, atol=1e-10)
    assert np.allclose(a * (i - 1) + b * i + c * (i + 1), -q * i, atol=1e-8)


def test_m_matrix_row_count_matches_the_cell_peclet_threshold():
    r"""a_i < 0 exactly when i < (r-q)/sigma^2; the solver must report that count."""
    for r, sigma, q in [(0.05, 0.2, 0.0), (0.15, 0.2, 0.0), (0.05, 0.1, 0.0), (0.03, 0.3, 0.0)]:
        a, _, c = operator_coefficients(400, r, sigma, q)
        expected = int(np.ceil((r - q) / sigma**2)) - 1
        assert int(np.sum(a < 0)) == max(expected, 0)
        assert int(np.sum(c < 0)) == 0
        res = solve_pde(100.0, 100.0, 1.0, r, sigma, q, M=400, N=200)
        assert res.n_non_mmatrix_rows == max(expected, 0)


def test_upwinding_restores_the_m_matrix_property():
    a, _, c = operator_coefficients(400, 0.15, 0.2, 0.0, upwind=True)
    assert np.all(a >= 0) and np.all(c >= 0)


# ---------------------------------------------------------------------------
# European limit against Black-Scholes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["put", "call"])
def test_european_matches_black_scholes(kind):
    res = solve_pde(**BASE, kind=kind, exercise="european", M=1600, N=1600, solver="direct")
    exact = float(bs_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, kind))
    assert res.price == pytest.approx(exact, abs=5e-4)


@pytest.mark.parametrize("name", ["base", "itm", "otm", "low_vol", "high_vol", "high_rate", "dividend"])
def test_european_put_across_regimes(name):
    p = REGIMES[name]
    res = solve_pde(p.S0, p.K, p.T, p.r, p.sigma, p.q, "put", "european",
                    M=1200, N=1200, solver="direct")
    exact = float(bs_put(p.S0, p.K, p.T, p.r, p.sigma, p.q))
    assert res.price == pytest.approx(exact, abs=2e-3), name


def test_european_convergence_is_second_order():
    """Measured log-log slope of the error against M = N."""
    exact = float(bs_put(100.0, 100.0, 1.0, 0.05, 0.2))
    Ms = np.array([100, 200, 400, 800, 1600])
    err = np.array(
        [abs(solve_pde(**BASE, kind="put", exercise="european", M=int(m), N=int(m),
                       solver="direct").price - exact) for m in Ms]
    )
    slope = np.polyfit(np.log(Ms), np.log(err), 1)[0]
    assert -2.15 < slope < -1.85, f"observed order {-slope:.3f}, expected ~2"


# ---------------------------------------------------------------------------
# The three LCP solvers must agree
# ---------------------------------------------------------------------------

def test_redblack_psor_matches_lexicographic_psor_on_a_random_lcp():
    """The vectorised solver is a faithful transcription of the scalar loop."""
    rng = np.random.default_rng(0)
    m = 60
    off = -rng.uniform(0.1, 1.0, m)
    dg = -(np.roll(off, -1) + off) + rng.uniform(1.0, 2.0, m)  # strictly dominant
    lo = off.copy(); lo[0] = 0.0
    up = off.copy(); up[-1] = 0.0
    rhs = rng.normal(size=m)
    g = rng.normal(size=m) * 0.3
    v0 = np.maximum(g, 0.0)
    a, ka, _ = psor_redblack(dg, lo, up, rhs, g, v0, 1.3, 1e-13, 100_000)
    b, kb, _ = psor_lexicographic(dg, lo, up, rhs, g, v0, 1.3, 1e-13, 100_000)
    assert np.max(np.abs(a - b)) < 1e-10


def test_psor_solution_satisfies_the_complementarity_conditions():
    """Av >= b, v >= g, and (Av - b)(v - g) = 0 componentwise."""
    rng = np.random.default_rng(7)
    m = 80
    off = -rng.uniform(0.2, 1.0, m)
    dg = -(np.roll(off, -1) + off) + rng.uniform(0.5, 1.5, m)
    lo = off.copy(); lo[0] = 0.0
    up = off.copy(); up[-1] = 0.0
    rhs = rng.normal(size=m)
    g = rng.normal(size=m)
    v, _, hit = psor_redblack(dg, lo, up, rhs, g, np.maximum(g, 0.0), 1.2, 1e-14, 200_000)
    assert not hit
    Av = dg * v
    Av[1:] += lo[1:] * v[:-1]
    Av[:-1] += up[:-1] * v[1:]
    slack, gap = Av - rhs, v - g
    assert np.all(gap >= -1e-10), "primal feasibility v >= g violated"
    assert np.all(slack >= -1e-8), "dual feasibility Av >= b violated"
    assert np.max(np.abs(slack * gap)) < 1e-7, "complementarity violated"


def test_psor_matches_brennan_schwartz_exact_solve():
    """PSOR is iterative; Brennan-Schwartz is exact. They must agree."""
    for name in ("base", "itm", "otm", "high_vol", "long_maturity"):
        p = REGIMES[name]
        kw = dict(M=600, N=600, kind="put", exercise="american")
        a = solve_pde(p.S0, p.K, p.T, p.r, p.sigma, p.q, **kw, solver="psor", tol=1e-12).price
        b = solve_pde(p.S0, p.K, p.T, p.r, p.sigma, p.q, **kw, solver="brennan_schwartz").price
        assert a == pytest.approx(b, abs=1e-8), name


def test_unconstrained_psor_matches_the_direct_banded_solve():
    a = solve_pde(**BASE, kind="put", exercise="european", M=400, N=400,
                  solver="psor", tol=1e-13).price
    b = solve_pde(**BASE, kind="put", exercise="european", M=400, N=400, solver="direct").price
    assert a == pytest.approx(b, abs=1e-9)


def test_brennan_schwartz_solves_the_actual_put_lcp_exactly():
    r"""Brennan-Schwartz is only valid when the active set is a *lower interval*.

    An arbitrary random LCP does not satisfy that, so this test builds the real
    discrete American-put system from `operator_coefficients` -- where §1.6
    guarantees the exercise region is ``{S <= S*}`` -- and checks the exact
    elimination against a very tightly converged PSOR.
    """
    M, K, r, sigma, q, th, dtau = 300, 100.0, 0.05, 0.2, 0.0, 0.5, 1.0 / 300
    a, b, c = operator_coefficients(M, r, sigma, q)
    dg = 1.0 - th * dtau * b
    lo = -th * dtau * a; lo[0] = 0.0
    up = -th * dtau * c; up[-1] = 0.0

    S = build_grid(K, M, 4 * K).S
    g = np.maximum(K - S[1:M], 0.0)
    # One Crank-Nicolson step away from the payoff: rhs = B g + boundary terms.
    rhs = (1.0 + (1 - th) * dtau * b) * g
    rhs[1:] += (1 - th) * dtau * a[1:] * g[:-1]
    rhs[:-1] += (1 - th) * dtau * c[:-1] * g[1:]
    rhs[0] += dtau * a[0] * K

    v = brennan_schwartz(dg, lo, up, rhs, g)
    ref, _, hit = psor_redblack(dg, lo, up, rhs, g, g.copy(), 1.0, 1e-15, 500_000)
    assert not hit
    assert np.max(np.abs(v - ref)) < 1e-10

    # And the active set really is a prefix, as the algorithm assumes.  The check
    # is restricted to S < K: far out of the money both v and g are ~0, so the
    # constraint looks "active" there for reasons that have nothing to do with
    # early exercise.  Theory (§1.6) puts S* below K, so that is the only region
    # where the lower-interval assumption has content.
    i_K = np.argmin(np.abs(S - K))
    below = slice(0, i_K - 1)
    active = np.flatnonzero(v[below] <= g[below] + 1e-12)
    assert np.array_equal(active, np.arange(active.size)), "active set is not a lower interval"
    assert 0 < active.size < i_K - 1, "expected a non-trivial exercise region below the strike"


# ---------------------------------------------------------------------------
# American pricing against the independent lattice
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name", ["base", "itm", "otm", "low_vol", "high_vol", "short_maturity",
             "long_maturity", "high_rate", "zero_rate", "dividend"]
)
def test_american_put_matches_the_binomial_lattice(name):
    p = REGIMES[name]
    pde = solve_pde(p.S0, p.K, p.T, p.r, p.sigma, p.q, "put", "american", M=1600, N=1600)
    lat = crr_price(p.S0, p.K, p.T, p.r, p.sigma, 12000, p.q, "put", "american")
    assert pde.price == pytest.approx(lat, abs=2e-3), f"{name}: PDE {pde.price} vs CRR {lat}"


def test_american_put_dominates_european_and_intrinsic():
    am = solve_pde(**BASE, kind="put", exercise="american", M=800, N=800).price
    eu = solve_pde(**BASE, kind="put", exercise="european", M=800, N=800, solver="direct").price
    assert am > eu
    assert am >= max(100.0 - 100.0, 0.0)


def test_american_call_equals_european_when_no_dividends():
    kw = dict(M=800, N=800, kind="call")
    am = solve_pde(**BASE, **kw, exercise="american").price
    eu = solve_pde(**BASE, **kw, exercise="european", solver="direct").price
    assert am == pytest.approx(eu, abs=1e-6)


def test_american_put_equals_european_when_rate_is_zero():
    kw = dict(S0=100.0, K=100.0, T=1.0, r=0.0, sigma=0.2, M=800, N=800, kind="put")
    am = solve_pde(**kw, exercise="american").price
    eu = solve_pde(**kw, exercise="european", solver="direct").price
    assert am == pytest.approx(eu, abs=1e-6)


# ---------------------------------------------------------------------------
# Invariance and robustness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("omega", [0.8, 1.0, 1.2, 1.5, 1.9])
def test_price_is_independent_of_the_relaxation_parameter(omega):
    """omega changes only the iteration count, never the fixed point."""
    ref = solve_pde(**BASE, kind="put", M=500, N=500, omega=1.2, tol=1e-12).price
    got = solve_pde(**BASE, kind="put", M=500, N=500, omega=omega, tol=1e-12).price
    assert got == pytest.approx(ref, abs=1e-8)


def test_tightening_tolerance_converges_monotonically_towards_the_exact_lcp():
    exact = solve_pde(**BASE, kind="put", M=500, N=500, solver="brennan_schwartz").price
    errs = [
        abs(solve_pde(**BASE, kind="put", M=500, N=500, tol=t).price - exact)
        for t in (1e-4, 1e-6, 1e-8, 1e-10)
    ]
    assert errs[-1] < errs[0]
    assert errs[-1] < 1e-8


def test_selective_upwinding_leaves_the_price_unchanged_and_fixes_the_m_matrix():
    """Exactly one row violates the cell-Peclet condition; repairing it should
    restore the M-matrix property at negligible cost in accuracy."""
    central = solve_pde(**BASE, kind="put", M=1600, N=1600, upwind=False)
    sel = solve_pde(**BASE, kind="put", M=1600, N=1600, upwind=True)
    assert central.n_peclet_violations == 1
    assert central.n_non_mmatrix_rows == 1
    assert sel.n_non_mmatrix_rows == 0
    assert abs(central.price - sel.price) < 1e-7, (
        f"selective upwinding moved the price by {abs(central.price - sel.price):.2e}"
    )


def test_full_upwinding_costs_accuracy():
    """Upwinding *every* row is first-order and measurably worse -- which is why
    the solver upwinds selectively rather than globally."""
    ref = crr_price(100.0, 100.0, 1.0, 0.05, 0.2, 12000, kind="put")
    central = solve_pde(**BASE, kind="put", M=1600, N=1600, upwind=False).price
    full = solve_pde(**BASE, kind="put", M=1600, N=1600, upwind="full").price
    assert abs(full - ref) > 20 * abs(central - ref)
    assert abs(full - central) > 1e-3


def test_full_upwinding_is_first_order():
    ref = crr_price(100.0, 100.0, 1.0, 0.05, 0.2, 12000, kind="put")
    Ms = np.array([200, 400, 800, 1600])
    err = np.array(
        [abs(solve_pde(**BASE, kind="put", M=int(m), N=int(m), upwind="full").price - ref)
         for m in Ms]
    )
    slope = np.polyfit(np.log(Ms), np.log(err), 1)[0]
    assert -1.3 < slope < -0.7, f"full upwinding order {-slope:.2f}, expected ~1"


def test_invalid_upwind_mode_raises():
    with pytest.raises(ValueError):
        operator_coefficients(100, 0.05, 0.2, 0.0, upwind="sideways")


def test_solution_shape_is_financially_sensible():
    res = solve_pde(**BASE, kind="put", M=800, N=800)
    V, S = res.values, res.S
    assert np.all(np.diff(V) <= 1e-9), "put value must be non-increasing in S"
    assert np.all(np.diff(V, 2) >= -1e-6), "put value must be convex in S"
    assert np.all(V >= np.maximum(100.0 - S, 0.0) - 1e-9), "value fell below intrinsic"
    assert np.all(V <= 100.0 + 1e-9)
    assert V[0] == pytest.approx(100.0)


def test_greeks_are_sensible():
    res = solve_pde(**BASE, kind="put", M=1600, N=1600)
    assert -1.0 < res.delta < 0.0
    assert res.gamma > 0.0


def test_deterministic():
    a = solve_pde(**BASE, kind="put", M=400, N=400).price
    b = solve_pde(**BASE, kind="put", M=400, N=400).price
    assert a == b


# ---------------------------------------------------------------------------
# Free boundary
# ---------------------------------------------------------------------------

def test_boundary_is_bounded_monotone_and_above_the_perpetual_limit():
    res = solve_pde(**BASE, kind="put", M=1600, N=1600)
    t, Sb = res.boundary_t, res.boundary_S
    order = np.argsort(t)
    t, Sb = t[order], Sb[order]
    assert np.all(np.isfinite(Sb)), "PDE boundary must be defined at every time level"
    assert np.all(Sb <= 100.0 + 1e-9)
    S_inf = perpetual_put_boundary(100.0, 0.05, 0.2)
    assert np.all(Sb > S_inf), f"boundary fell below the perpetual limit {S_inf}"
    assert Sb[-1] == pytest.approx(100.0, abs=1e-9), "S*(T) must equal K when q = 0"
    # non-decreasing in t up to one cell of discretisation noise
    assert np.all(np.diff(Sb) > -2.0 * res.dS)
    assert np.polyfit(t, Sb, 1)[0] > 0.0


def test_zero_rate_has_essentially_no_exercise_region():
    r"""With r = 0 it is never optimal to exercise a put early, so S*(t) ~ 0.

    It is not *exactly* zero: deep in the money at r = 0 the European value and
    the payoff agree to machine precision, so whether the projection binds there
    is decided by floating-point noise, not by economics.  The assertion is
    therefore that the boundary is a negligible fraction of the strike.
    """
    res = solve_pde(S0=100.0, K=100.0, T=1.0, r=0.0, sigma=0.2, kind="put", M=2000, N=2000)
    S_star_0 = res.boundary_S[-1]
    assert S_star_0 < 0.02 * 100.0, f"spurious exercise region at r=0: S*(0)={S_star_0}"


def test_boundary_uses_the_projection_not_a_threshold_on_the_gap():
    """Regression guard: an absolute tolerance on v - g reported S* = 31.8 at
    r = 0, where the true exercise region is empty."""
    res = solve_pde(S0=100.0, K=100.0, T=1.0, r=0.0, sigma=0.2, kind="put", M=2000, N=2000)
    assert res.boundary_S[-1] < 5.0


def test_boundary_covers_the_whole_time_axis_unlike_the_lattice():
    """The PDE grid is fixed in S, so S*(t) is resolved right down to t = 0."""
    from amopt.binomial import crr_boundary

    res = solve_pde(**BASE, kind="put", M=800, N=800)
    _, lat = crr_boundary(100.0, 100.0, 1.0, 0.05, 0.2, 800, kind="put")
    assert np.isfinite(res.boundary_S).all()
    assert np.isnan(lat).any(), "expected the lattice cone to miss the boundary near t=0"


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kw",
    [
        dict(kind="straddle"),
        dict(exercise="bermudan"),
        dict(solver="nonesuch"),
        dict(solver="direct", exercise="american"),
        dict(solver="brennan_schwartz", kind="call"),
        dict(omega=0.0),
        dict(omega=2.0),
        dict(N=0),
    ],
)
def test_invalid_arguments_raise(kw):
    args = dict(BASE, M=100, N=100)
    args.update(kw)
    with pytest.raises(ValueError):
        solve_pde(**args)


def test_grid_too_coarse_for_the_strike_raises():
    with pytest.raises(ValueError):
        build_grid(100.0, 4, 1e6)
