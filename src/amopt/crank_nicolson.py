r"""Crank-Nicolson finite differences with PSOR for the American-option LCP.

Implements `docs/02_crank_nicolson.md` from first principles.  The only external
numerical routine used is :func:`scipy.linalg.solve_banded`, and that only for
the *European* (unconstrained) problem, which is a plain tridiagonal solve; the
American constraint is handled entirely by code in this module.

Summary of the scheme
---------------------
Work in time to maturity :math:`\tau = T - t` on a uniform grid
:math:`S_i = i\,\Delta S`, :math:`i = 0..M`, with the strike pinned exactly to a
node.  The discrete spatial operator is tridiagonal with

.. math::
    a_i = \tfrac12\sigma^2 i^2 - \tfrac12 (r-q) i,\quad
    b_i = -\sigma^2 i^2 - r,\quad
    c_i = \tfrac12\sigma^2 i^2 + \tfrac12 (r-q) i,

every power of :math:`\Delta S` having cancelled.  A :math:`\theta`-step gives
:math:`A v^{n+1} = B v^n + d^n`, and the American constraint turns that linear
solve into the LCP

.. math::
    Av^{n+1} \ge b^n,\quad v^{n+1} \ge g,\quad (Av^{n+1}-b^n)^\top(v^{n+1}-g)=0,

solved by projected SOR.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded

__all__ = [
    "CNResult",
    "GridSpec",
    "build_grid",
    "operator_coefficients",
    "solve_pde",
    "psor_redblack",
    "psor_lexicographic",
    "brennan_schwartz",
]


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridSpec:
    """A uniform price grid with the strike pinned to a node.

    ``S_max`` is nudged so that ``K`` is exactly ``S[i_K]``.  Leaving the kink
    between nodes smears the payoff and degrades the observed spatial order from
    2 towards 1.

    The nudge is *relatively* small but not absolutely small: since
    ``dS = K/i_K`` and ``i_K = round(K M / S_max_target)``,

        ``|S_max / S_max_target - 1| <= 1 / (2 i_K)``,

    i.e. half a cell measured at the strike, which the coarse-grid/large-domain
    combination then amplifies by ``M / i_K``.  With the default
    ``S_max = 4K`` and ``M >= 100`` this is under 2%.
    """

    S: np.ndarray
    dS: float
    M: int
    i_K: int
    S_max: float


def build_grid(K: float, M: int, S_max_target: float) -> GridSpec:
    """Build the price grid, adjusting ``S_max`` so ``K`` lands on a node."""
    if M < 4:
        raise ValueError("M must be >= 4")
    if S_max_target <= K:
        raise ValueError("S_max_target must exceed the strike")
    dS0 = S_max_target / M
    i_K = int(round(K / dS0))
    if i_K < 1:
        raise ValueError("grid too coarse to place the strike on a node; increase M")
    dS = K / i_K
    S = np.arange(M + 1, dtype=float) * dS
    return GridSpec(S=S, dS=dS, M=M, i_K=i_K, S_max=float(S[-1]))


# ---------------------------------------------------------------------------
# Spatial operator
# ---------------------------------------------------------------------------

def operator_coefficients(M: int, r: float, sigma: float, q: float = 0.0, upwind=False):
    r"""Tridiagonal coefficients of :math:`\mathcal{A}_h` on interior nodes ``1..M-1``.

    Returns ``(a, b, c)`` each of length ``M-1`` (index ``j`` is node ``i=j+1``).

    Parameters
    ----------
    upwind : {False, True, 'selective', 'full'}
        ``False`` -- central differences everywhere: second-order accurate, but
        :math:`a_i < 0` whenever :math:`i < (r-q)/\sigma^2` (the cell-Peclet
        condition), which breaks the M-matrix structure in those rows.

        ``True`` / ``'selective'`` -- central differences everywhere **except**
        the rows that violate the condition, which are upwinded.  This restores
        :math:`a_i, c_i \ge 0` globally while keeping second-order accuracy on
        all but a handful of rows near :math:`S = 0`.

        ``'full'`` -- upwind every row.  Unconditionally an M-matrix, but only
        first-order accurate, and measurably so: see
        ``tests/test_crank_nicolson.py::test_full_upwinding_costs_accuracy``.

    Both variants satisfy two exact identities, asserted by the caller:
    ``a+b+c = -r`` (the operator applied to a constant) and
    ``a(i-1)+bi+c(i+1) = -q i`` (applied to the identity function).  Upwinding
    changes ``a`` and ``c`` by equal and opposite amounts relative to ``b``, so
    both identities survive it -- which is why they cannot detect the accuracy
    loss, and a price comparison is needed instead.
    """
    if upwind not in (False, True, "selective", "full"):
        raise ValueError(f"upwind must be False, True, 'selective' or 'full', got {upwind!r}")
    i = np.arange(1, M, dtype=float)
    drift = r - q

    a_c = 0.5 * sigma**2 * i**2 - 0.5 * drift * i
    b_c = -(sigma**2) * i**2 - r
    c_c = 0.5 * sigma**2 * i**2 + 0.5 * drift * i
    if upwind is False:
        return a_c, b_c, c_c

    if drift >= 0.0:  # forward difference for v_S: (v_{i+1} - v_i)/dS
        a_u = 0.5 * sigma**2 * i**2
        b_u = -(sigma**2) * i**2 - drift * i - r
        c_u = 0.5 * sigma**2 * i**2 + drift * i
    else:  # backward difference: (v_i - v_{i-1})/dS
        a_u = 0.5 * sigma**2 * i**2 - drift * i
        b_u = -(sigma**2) * i**2 + drift * i - r
        c_u = 0.5 * sigma**2 * i**2

    if upwind == "full":
        return a_u, b_u, c_u
    bad = (a_c < 0.0) | (c_c < 0.0)
    return np.where(bad, a_u, a_c), np.where(bad, b_u, b_c), np.where(bad, c_u, c_c)


def _check_operator_identities(a, b, c, r, q, atol=1e-9):
    """Assert the two exact identities from `docs/02` §2.2."""
    i = np.arange(1, a.size + 1, dtype=float)
    const = a + b + c
    if not np.allclose(const, -r, atol=atol):
        raise AssertionError(
            f"operator identity A_h*1 = -r violated (max dev "
            f"{np.max(np.abs(const + r)):.3e}) -- coefficient algebra is wrong"
        )
    linear = a * (i - 1.0) + b * i + c * (i + 1.0)
    if not np.allclose(linear, -q * i, atol=atol * max(1.0, i[-1])):
        raise AssertionError(
            f"operator identity A_h*S = -qS violated (max dev "
            f"{np.max(np.abs(linear + q * i)):.3e}) -- indexing or sign error"
        )


# ---------------------------------------------------------------------------
# Payoff and boundary data
# ---------------------------------------------------------------------------

def _payoff(S, K, kind):
    return np.maximum(K - S, 0.0) if kind == "put" else np.maximum(S - K, 0.0)


def _boundary_values(tau, K, r, q, S_max, kind, exercise):
    r"""Dirichlet data :math:`(\ell(\tau), u(\tau))` at ``S=0`` and ``S=S_max``.

    The put's lower boundary is the one that differs by exercise style:
    ``V(0,t) = K`` for the American put (zero is absorbing, so exercise now),
    versus ``K e^{-r\tau}`` for the European.  Using the European value for an
    American solve is the classic silent bug.
    """
    tau = np.asarray(tau, dtype=float)
    if kind == "put":
        lower = np.full_like(tau, K) if exercise == "american" else K * np.exp(-r * tau)
        upper = np.zeros_like(tau)
    else:
        lower = np.zeros_like(tau)
        eu = S_max * np.exp(-q * tau) - K * np.exp(-r * tau)
        upper = np.maximum(eu, S_max - K) if exercise == "american" else eu
    return lower, upper


# ---------------------------------------------------------------------------
# LCP solvers
# ---------------------------------------------------------------------------

def psor_redblack(dg, lo, up, rhs, g, v0, omega, tol, max_iter):
    r"""Projected SOR with red-black (odd-even) ordering -- the production solver.

    ``dg``, ``lo``, ``up`` are the diagonal, sub- and super-diagonals of ``A``
    with ``lo[0] = up[-1] = 0`` (the boundary couplings are already folded into
    ``rhs``).  ``g`` is the obstacle; pass ``-inf`` to recover plain SOR.

    Because ``A`` is tridiagonal, odd-indexed nodes couple only to even-indexed
    nodes.  Updating all odds, then all evens, is therefore an exact Gauss-Seidel
    sweep in a different ordering -- and a tridiagonal matrix is *consistently
    ordered* in Young's sense, so the spectral radius and the optimal ``omega``
    are unchanged.  The payoff is that each half sweep is one vectorised NumPy
    expression instead of a Python loop over ``M`` nodes.

    Returns ``(v, iterations, hit_max_iter)``.
    """
    v = np.asarray(v0, dtype=float).copy()
    m = v.size
    red = np.arange(0, m, 2)
    black = np.arange(1, m, 2)
    vm = np.zeros(m)
    vp = np.zeros(m)
    for k in range(max_iter):
        v_prev = v.copy()
        for idx in (red, black):
            vm[0] = 0.0
            vm[1:] = v[:-1]
            vp[-1] = 0.0
            vp[:-1] = v[1:]
            gs = (rhs[idx] - lo[idx] * vm[idx] - up[idx] * vp[idx]) / dg[idx]
            v[idx] = np.maximum(g[idx], v[idx] + omega * (gs - v[idx]))
        if np.max(np.abs(v - v_prev)) < tol:
            return v, k + 1, False
    return v, max_iter, True


def psor_lexicographic(dg, lo, up, rhs, g, v0, omega, tol, max_iter):
    """Reference PSOR in natural (lexicographic) order, as a literal Python loop.

    Deliberately slow and transparent.  Its only job is to be obviously a
    faithful transcription of equation (2.5) so that the vectorised red-black
    solver can be validated against it in the test suite.
    """
    v = np.asarray(v0, dtype=float).copy()
    m = v.size
    for k in range(max_iter):
        err = 0.0
        for j in range(m):
            left = v[j - 1] if j > 0 else 0.0
            right = v[j + 1] if j < m - 1 else 0.0
            gs = (rhs[j] - lo[j] * left - up[j] * right) / dg[j]
            new = max(g[j], v[j] + omega * (gs - v[j]))
            err = max(err, abs(new - v[j]))
            v[j] = new
        if err < tol:
            return v, k + 1, False
    return v, max_iter, True


def brennan_schwartz(dg, lo, up, rhs, g):
    r"""Exact :math:`O(M)` LCP solve for a *lower*-interval exercise region.

    A UL elimination ordered from the continuation side (high ``S``) followed by
    a forward substitution that projects onto the obstacle.  Because the
    constrained nodes are visited last, the projection never invalidates a row
    that has already been eliminated, and the result solves the LCP exactly --
    no tolerance, no relaxation parameter.

    Valid for the American **put**, whose exercise region is
    :math:`\{S \le S^*\}` (see `docs/01` §1.6).  Used as an independent check on
    PSOR, not as the production path.
    """
    m = dg.size
    d = np.empty(m)
    y = np.empty(m)
    d[-1] = dg[-1]
    y[-1] = rhs[-1]
    for j in range(m - 2, -1, -1):
        mult = up[j] / d[j + 1]
        d[j] = dg[j] - mult * lo[j + 1]
        y[j] = rhs[j] - mult * y[j + 1]
    v = np.empty(m)
    v[0] = max(y[0] / d[0], g[0])
    for j in range(1, m):
        v[j] = max((y[j] - lo[j] * v[j - 1]) / d[j], g[j])
    return v


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CNResult:
    """Everything a Crank-Nicolson solve produces, including its diagnostics."""

    price: float
    delta: float
    gamma: float
    S: np.ndarray
    values: np.ndarray
    tau: np.ndarray
    boundary_t: np.ndarray | None = None
    boundary_S: np.ndarray | None = None
    boundary_S_raw: np.ndarray | None = None
    iterations: np.ndarray | None = None
    mean_iterations: float = 0.0
    max_iterations_hit: bool = False
    n_non_mmatrix_rows: int = 0
    n_peclet_violations: int = 0
    M: int = 0
    N: int = 0
    dS: float = 0.0
    dtau: float = 0.0
    S_max: float = 0.0
    i_K: int = 0
    S0_on_node: bool = False
    solver: str = "psor"
    theta: float = 0.5
    omega: float = 1.2
    tol: float = 0.0
    rannacher_steps: int = 0
    upwind: object = False
    runtime_s: float = 0.0
    meta: dict = field(default_factory=dict)


def _interp_value(S, V, S0):
    """Local cubic Lagrange interpolation, to avoid degrading O(dS^2) accuracy."""
    j = int(np.searchsorted(S, S0)) - 1
    j = int(np.clip(j - 1, 0, S.size - 4))
    xs, ys = S[j : j + 4], V[j : j + 4]
    total = 0.0
    for k in range(4):
        term = ys[k]
        for l in range(4):
            if l != k:
                term *= (S0 - xs[l]) / (xs[k] - xs[l])
        total += term
    return float(total)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def solve_pde(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: str = "put",
    exercise: str = "american",
    M: int = 800,
    N: int = 800,
    S_max_mult: float = 4.0,
    S_max: float | None = None,
    theta: float = 0.5,
    omega: float = 1.2,
    tol: float = 1e-9,
    max_iter: int = 20_000,
    rannacher_steps: int = 2,
    upwind: bool = False,
    solver: str = "psor",
    track_boundary: bool = True,
    return_surface: bool = False,
) -> CNResult:
    r"""Price a vanilla option by Crank-Nicolson finite differences.

    Parameters
    ----------
    M, N : int
        Spatial and temporal step counts.
    S_max_mult, S_max :
        Domain truncation.  ``S_max`` overrides ``S_max_mult * max(K, S0)``.
        The truncation error decays like the probability of reaching the
        boundary; it is measured in Milestone 6, not assumed.
    theta : float
        ``0.5`` Crank-Nicolson, ``1.0`` fully implicit, ``0.0`` explicit.
    omega, tol, max_iter :
        PSOR relaxation parameter, convergence tolerance on the max-norm change
        between sweeps, and iteration cap.
    rannacher_steps : int
        Number of initial fully-implicit steps.  Crank-Nicolson is A-stable but
        not L-stable, so the payoff kink excites undamped high-frequency modes;
        two implicit steps restore clean second-order behaviour.
    upwind : {False, True, 'selective', 'full'}
        ``True``/``'selective'`` upwinds only the rows violating the cell-Peclet
        condition; ``'full'`` upwinds every row and is first-order accurate.
    solver : {'psor', 'psor_lex', 'brennan_schwartz', 'direct'}
        ``'direct'`` is a plain banded solve and is only valid for European
        exercise (it ignores the constraint).

    Returns
    -------
    CNResult
    """
    t_start = time.perf_counter()
    if kind not in ("put", "call"):
        raise ValueError(f"kind must be 'put' or 'call', got {kind!r}")
    if exercise not in ("european", "american"):
        raise ValueError(f"exercise must be 'european' or 'american', got {exercise!r}")
    if solver not in ("psor", "psor_lex", "brennan_schwartz", "direct"):
        raise ValueError(f"unknown solver {solver!r}")
    if solver == "direct" and exercise == "american":
        raise ValueError("solver='direct' ignores the constraint; it cannot price American options")
    if solver == "brennan_schwartz" and kind != "put":
        raise ValueError("brennan_schwartz assumes a lower-interval exercise region (put only)")
    if not (0.0 < omega < 2.0):
        raise ValueError("omega must lie in (0, 2) for SOR to converge")
    if T <= 0.0 or sigma <= 0.0 or N < 1:
        raise ValueError("require T > 0, sigma > 0, N >= 1")

    target = S_max if S_max is not None else S_max_mult * max(K, S0)
    grid = build_grid(K, M, target)
    S = grid.S
    dtau = T / N
    tau_levels = np.arange(N + 1, dtype=float) * dtau

    a, b, c = operator_coefficients(M, r, sigma, q, upwind=upwind)
    _check_operator_identities(a, b, c, r, q)
    n_bad = int(np.sum(a < 0.0) + np.sum(c < 0.0))
    a_c, _, c_c = operator_coefficients(M, r, sigma, q, upwind=False)
    n_peclet = int(np.sum(a_c < 0.0) + np.sum(c_c < 0.0))

    g_full = _payoff(S, K, kind)
    g = g_full[1:M]
    obstacle = g if exercise == "american" else np.full(M - 1, -np.inf)

    v = g_full.copy()  # tau = 0 is the payoff

    # Assemble A and B for each theta actually used (at most two).
    def assemble(th):
        A_lo = np.empty(M - 1)
        A_up = np.empty(M - 1)
        A_dg = 1.0 - th * dtau * b
        A_lo[0] = 0.0
        A_lo[1:] = -th * dtau * a[1:]
        A_up[-1] = 0.0
        A_up[:-1] = -th * dtau * c[:-1]
        B_lo = (1.0 - th) * dtau * a
        B_dg = 1.0 + (1.0 - th) * dtau * b
        B_up = (1.0 - th) * dtau * c
        banded = np.zeros((3, M - 1))
        banded[0, 1:] = A_up[:-1]
        banded[1, :] = A_dg
        banded[2, :-1] = A_lo[1:]
        return dict(A_dg=A_dg, A_lo=A_lo, A_up=A_up, B_dg=B_dg, B_lo=B_lo, B_up=B_up, banded=banded)

    thetas = {1.0: assemble(1.0)} if rannacher_steps > 0 else {}
    thetas.setdefault(theta, assemble(theta))

    iters = np.zeros(N, dtype=int)
    hit_max = False
    bnd_raw = np.full(N + 1, np.nan)
    bnd_interp = np.full(N + 1, np.nan)
    surface = np.zeros((N + 1, M + 1)) if return_surface else None
    if return_surface:
        surface[0] = v
    if track_boundary and exercise == "american" and kind == "put":
        bnd_raw[0] = K
        bnd_interp[0] = K

    for n in range(N):
        th = 1.0 if n < rannacher_steps else theta
        mats = thetas[th]
        lo_n, up_n = _boundary_values(tau_levels[n], K, r, q, grid.S_max, kind, exercise)
        lo_n1, up_n1 = _boundary_values(tau_levels[n + 1], K, r, q, grid.S_max, kind, exercise)

        vi = v[1:M]
        rhs = mats["B_dg"] * vi
        rhs[1:] += mats["B_lo"][1:] * vi[:-1]
        rhs[:-1] += mats["B_up"][:-1] * vi[1:]
        # boundary couplings, evaluated at BOTH time levels (docs/02 sec 2.4)
        rhs[0] += (1.0 - th) * dtau * a[0] * float(lo_n) + th * dtau * a[0] * float(lo_n1)
        rhs[-1] += (1.0 - th) * dtau * c[-1] * float(up_n) + th * dtau * c[-1] * float(up_n1)

        if solver == "direct":
            v_new = solve_banded((1, 1), mats["banded"], rhs)
            it, hm = 0, False
        elif solver == "brennan_schwartz":
            v_new = brennan_schwartz(mats["A_dg"], mats["A_lo"], mats["A_up"], rhs, obstacle)
            it, hm = 0, False
        else:
            fn = psor_redblack if solver == "psor" else psor_lexicographic
            v_new, it, hm = fn(
                mats["A_dg"], mats["A_lo"], mats["A_up"], rhs, obstacle, vi, omega, tol, max_iter
            )
        iters[n] = it
        hit_max |= hm

        # The projection writes `g` bitwise when it binds, so equality is an
        # exact test for membership of the discrete exercise set.
        binding = np.empty(M + 1, dtype=bool)
        binding[0] = exercise == "american"
        binding[M] = False
        binding[1:M] = v_new == obstacle

        v = np.empty(M + 1)
        v[0] = float(lo_n1)
        v[M] = float(up_n1)
        v[1:M] = v_new
        if return_surface:
            surface[n + 1] = v

        if track_boundary and exercise == "american" and kind == "put":
            bnd_raw[n + 1], bnd_interp[n + 1] = _extract_boundary(S, v, g_full, binding, grid.i_K)

    i0 = int(np.searchsorted(S, S0))
    on_node = bool(abs(S[min(i0, M)] - S0) < 1e-12 or (i0 > 0 and abs(S[i0 - 1] - S0) < 1e-12))
    price = float(v[i0]) if (on_node and abs(S[min(i0, M)] - S0) < 1e-12) else _interp_value(S, v, S0)
    j = int(np.clip(np.argmin(np.abs(S - S0)), 1, M - 1))
    delta = float((v[j + 1] - v[j - 1]) / (2.0 * grid.dS))
    gamma = float((v[j + 1] - 2.0 * v[j] + v[j - 1]) / grid.dS**2)

    res = CNResult(
        price=price, delta=delta, gamma=gamma, S=S, values=v, tau=tau_levels,
        iterations=iters, mean_iterations=float(iters.mean()), max_iterations_hit=hit_max,
        n_non_mmatrix_rows=n_bad, n_peclet_violations=n_peclet, M=M, N=N, dS=grid.dS, dtau=dtau, S_max=grid.S_max,
        i_K=grid.i_K, S0_on_node=on_node, solver=solver, theta=theta, omega=omega,
        tol=tol, rannacher_steps=rannacher_steps, upwind=upwind,
        runtime_s=time.perf_counter() - t_start,
    )
    if track_boundary and exercise == "american" and kind == "put":
        res.boundary_t = T - tau_levels
        res.boundary_S_raw = bnd_raw
        res.boundary_S = bnd_interp
    if return_surface:
        res.meta["surface"] = surface
    return res


def _extract_boundary(S, v, g_full, binding, i_K=None):
    r"""Locate :math:`S^*` as the edge of the discrete exercise set.

    The raw estimate is the last node at which the constraint is active, which is
    only accurate to :math:`\Delta S`.

    The refinement uses smooth pasting.  Just above the boundary,

    .. math::
        \phi(S) := v(S) - g(S)
        = \underbrace{v(S^*) - g(S^*)}_{=0\ \text{(value matching)}}
        + \underbrace{(v_S(S^*)+1)}_{=0\ \text{(smooth pasting)}}(S-S^*)
        + \tfrac12 v_{SS}(S^*)(S-S^*)^2 + \dots

    so the early-exercise gap vanishes *quadratically*, and
    :math:`\sqrt{\phi}` is locally **linear** in :math:`S`.  Linearly
    extrapolating :math:`\sqrt{\phi}` from the first two continuation nodes back
    to zero therefore gives a sub-grid estimate of :math:`S^*`.

    Interpolating :math:`\phi` itself would be useless: :math:`\phi` is exactly
    zero at every exercised node, so a secant through the last exercised node has
    its root at that node and returns the raw estimate unchanged.

    Three practical points, all learned from the data rather than assumed:

    * **The exercise set is read from the projection, not from a threshold on**
      :math:`\phi`.  ``binding[i]`` is true exactly when the PSOR (or
      Brennan-Schwartz) update selected the obstacle, which is a bitwise
      ``v == g`` test with no tolerance in it.  An absolute threshold such as
      :math:`\phi \le 10^{-8}` is not scale free and fails outright at
      :math:`r = 0`, where the American put equals the European put and
      :math:`\phi` is genuinely of order :math:`10^{-9}` deep in the money
      without the constraint ever binding -- that version reported a spurious
      boundary at :math:`S^* = 31.8` for a contract that should never be
      exercised at all.
    * The search is restricted to :math:`S \le K`, which theory (§1.6)
      guarantees, removing any residual far-out-of-the-money false positives.
    * The refined root may legitimately fall *below* the last exercised node.
      The discrete exercise set overshoots the true boundary by up to one cell,
      because a node is pinned to the payoff as soon as the discrete LCP says so.
      The bracket is therefore ``[S[k] - 2 dS, S[k+2]]``, not ``[S[k], S[k+1]]``.

    The refinement does not beat the underlying :math:`O(\Delta S)` accuracy of
    the free boundary itself -- it removes the staircase, not the error.  See
    `RESULTS.md` for the measured convergence of :math:`S^*(0)` in
    :math:`\Delta S`.
    """
    phi = v - g_full
    active = np.asarray(binding, dtype=bool).copy()
    if i_K is not None:
        active[i_K + 1 :] = False  # S* <= K, so never look above the strike
    if not active[0]:
        # The constraint does not bind even at S = 0: the exercise region is
        # empty (r = 0 does this), so there is no boundary to report.
        return 0.0, 0.0
    k = 0
    while k + 1 < active.size and active[k + 1]:
        k += 1
    S_raw = float(S[k])
    if k + 2 >= phi.size:
        return S_raw, S_raw
    p1, p2 = phi[k + 1], phi[k + 2]
    if not (p2 > p1 > 0.0):
        return S_raw, S_raw
    r1, r2 = np.sqrt(p1), np.sqrt(p2)
    dS = S[k + 2] - S[k + 1]
    S_star = float(S[k + 1] - r1 * dS / (r2 - r1))
    if not (S[k] - 2.0 * dS <= S_star <= S[k + 2]):
        return S_raw, S_raw
    return S_raw, S_star
