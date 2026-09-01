r"""Cox-Ross-Rubinstein binomial lattice, implemented from first principles.

Construction
------------
Partition :math:`[0,T]` into :math:`N` steps of length :math:`\Delta t = T/N`.
CRR chooses the multiplicative moves

.. math::
    u = e^{\sigma\sqrt{\Delta t}}, \qquad d = 1/u,

which matches the variance of the log-return to :math:`\sigma^2\Delta t` to
:math:`O(\Delta t^2)`.  The risk-neutral probability follows from requiring the
discounted asset to be a martingale, :math:`\mathbb{E}^{\mathbb{Q}}[S_{t+\Delta t}] = S_t e^{(r-q)\Delta t}`:

.. math::
    p = \frac{e^{(r-q)\Delta t} - d}{u - d}.

Because :math:`d = 1/u`, an up-move followed by a down-move returns to the same
node, so the lattice recombines and node :math:`(i, j)` (time step :math:`i`,
:math:`j` up-moves) carries

.. math::
    S_{i,j} = S_0 u^{j} d^{i-j} = S_0 e^{(2j - i)\sigma\sqrt{\Delta t}}.

The exponential form is used in code to avoid overflow of ``u**j`` for large
:math:`N`.

Backward induction
------------------
Terminal values are the payoff.  Rolling back one step,

.. math::
    C_{i,j} = e^{-r\Delta t}\bigl[p\, V_{i+1,j+1} + (1-p) V_{i+1,j}\bigr]

is the *continuation* value.  For a European claim :math:`V_{i,j} = C_{i,j}`;
for an American claim the holder takes the better of exercising now and waiting,

.. math::
    V_{i,j} = \max\bigl\{ g(S_{i,j}),\; C_{i,j} \bigr\},

which is the discrete-time Snell envelope of the discounted payoff and is the
lattice analogue of the linear complementarity problem solved in
:mod:`amopt.crank_nicolson`.

This module is deliberately independent of the PDE and Monte Carlo solvers so
that it can act as a genuinely independent benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BinomialResult", "crr", "crr_price", "crr_price_averaged", "crr_boundary"]


def _payoff(S, K, kind):
    if kind == "put":
        return np.maximum(K - S, 0.0)
    if kind == "call":
        return np.maximum(S - K, 0.0)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


@dataclass
class BinomialResult:
    """Container for a lattice valuation.

    Attributes
    ----------
    price : float
        Value at the root node.
    delta, gamma, theta : float or None
        Lattice Greeks, extracted from the nodes at steps 1 and 2 (see
        :func:`crr`).  ``None`` when ``greeks=False`` or ``N < 2``.
    boundary_t, boundary_S : np.ndarray or None
        Sampled early-exercise boundary :math:`S^*(t)`, ``None`` unless
        ``boundary=True``.
    N : int
        Number of time steps used.
    """

    price: float
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    boundary_t: np.ndarray | None = None
    boundary_S: np.ndarray | None = None
    N: int = 0


def crr(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int,
    q: float = 0.0,
    kind: str = "put",
    exercise: str = "american",
    greeks: bool = False,
    boundary: bool = False,
) -> BinomialResult:
    r"""Price a vanilla option on a CRR lattice with ``N`` time steps.

    Parameters
    ----------
    exercise : {'european', 'american'}
        ``'american'`` applies the early-exercise projection at every step.
    greeks : bool
        Extract delta/gamma/theta from the lattice.  Because :math:`ud = 1`,
        the step-2 nodes contain :math:`S_0` itself, giving a centred
        second difference at no extra cost.
    boundary : bool
        Record, for each time step, the largest asset price at which immediate
        exercise is optimal (a put) -- a discrete estimate of :math:`S^*(t)`.

        .. warning::
           A lattice rooted at ``S0`` only spans the cone
           :math:`S_0 e^{\pm i\sigma\sqrt{\Delta t}}`.  Near :math:`t = 0`
           that cone can be entirely inside the continuation region, in which
           case no node is exercised and ``nan`` is recorded.  This is a real
           structural limitation of lattice boundary extraction; the PDE solver
           in :mod:`amopt.crank_nicolson` resolves :math:`S^*(t)` on the whole
           time axis because its grid is fixed in :math:`S` rather than
           expanding from a single root.

    Raises
    ------
    ValueError
        If the risk-neutral probability falls outside ``[0, 1]``, which means
        the lattice admits arbitrage.  This happens when
        :math:`\\sigma\\sqrt{\\Delta t} < |(r-q)\\Delta t|`, i.e. ``N`` is far
        too small for the drift; the fix is more steps, not clamping ``p``.
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    if T <= 0.0:
        return BinomialResult(price=float(_payoff(np.asarray(S0, float), K, kind)), N=N)
    if exercise not in ("european", "american"):
        raise ValueError(f"exercise must be 'european' or 'american', got {exercise!r}")

    dt = T / N
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0 for a CRR lattice")
    vol_step = sigma * np.sqrt(dt)
    u = np.exp(vol_step)
    d = 1.0 / u
    growth = np.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"risk-neutral probability p={p:.6f} outside [0,1]: the lattice is "
            f"arbitrageable at N={N}. Increase N so that sigma*sqrt(dt) > |(r-q)*dt|."
        )
    disc = np.exp(-r * dt)

    j = np.arange(N + 1)
    S_T = S0 * np.exp((2 * j - N) * vol_step)
    V = _payoff(S_T, K, kind)

    keep = {}
    bnd_t, bnd_S = [], []
    if boundary and exercise == "american":
        # At maturity the exercise region is {g > 0}; record the extreme node in it.
        ex_T = np.flatnonzero(_payoff(S_T, K, kind) > 0.0)
        bnd_t.append(T)
        bnd_S.append(float(S_T[ex_T.max() if kind == "put" else ex_T.min()]) if ex_T.size else np.nan)

    for i in range(N - 1, -1, -1):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        if exercise == "american" or boundary:
            S_i = S0 * np.exp((2 * np.arange(i + 1) - i) * vol_step)
            intrinsic = _payoff(S_i, K, kind)
        if exercise == "american":
            exercise_now = intrinsic > V
            V = np.where(exercise_now, intrinsic, V)
            if boundary:
                if np.any(exercise_now):
                    # Put: exercise region is {S <= S*}; take the largest such node.
                    idx = np.max(np.flatnonzero(exercise_now)) if kind == "put" else np.min(
                        np.flatnonzero(exercise_now)
                    )
                    bnd_t.append(i * dt)
                    bnd_S.append(float(S_i[idx]))
                else:
                    bnd_t.append(i * dt)
                    bnd_S.append(np.nan)
        if greeks and i in (1, 2):
            keep[i] = (V.copy(), S0 * np.exp((2 * np.arange(i + 1) - i) * vol_step))

    price = float(V[0])
    res = BinomialResult(price=price, N=N)

    if greeks and N >= 2 and 1 in keep and 2 in keep:
        V1, S1 = keep[1]
        V2, S2 = keep[2]
        # Step-1 nodes straddle S0: central difference in S.
        res.delta = float((V1[1] - V1[0]) / (S1[1] - S1[0]))
        # Step-2 nodes are (S0 d^2, S0, S0 u^2): non-uniform central 2nd difference.
        h_up, h_dn = S2[2] - S2[1], S2[1] - S2[0]
        res.gamma = float(
            2.0 * (h_dn * (V2[2] - V2[1]) - h_up * (V2[1] - V2[0])) / (h_up * h_dn * (h_up + h_dn))
        )
        # V2[1] sits at (t=2*dt, S=S0); differencing against the root gives theta.
        res.theta = float((V2[1] - price) / (2.0 * dt))

    if boundary and exercise == "american":
        order = np.argsort(bnd_t)
        res.boundary_t = np.asarray(bnd_t, float)[order]
        res.boundary_S = np.asarray(bnd_S, float)[order]
    return res


def crr_price(S0, K, T, r, sigma, N, q=0.0, kind="put", exercise="american") -> float:
    """Scalar convenience wrapper returning only the root value."""
    return crr(S0, K, T, r, sigma, N, q, kind, exercise).price


def crr_price_averaged(S0, K, T, r, sigma, N, q=0.0, kind="put", exercise="american") -> float:
    r"""Average of the ``N`` and ``N+1`` step lattices.

    CRR prices oscillate around the true value with period two in ``N`` because
    the strike sits between lattice nodes differently for odd and even ``N``.
    Averaging adjacent step counts cancels the leading oscillatory term and is a
    standard, cheap accuracy improvement; we use it only where explicitly noted.
    """
    return 0.5 * (
        crr_price(S0, K, T, r, sigma, N, q, kind, exercise)
        + crr_price(S0, K, T, r, sigma, N + 1, q, kind, exercise)
    )


def crr_boundary(S0, K, T, r, sigma, N, q=0.0, kind="put"):
    """Return ``(t, S*(t))`` sampled on the lattice time grid.

    Entries are ``nan`` at times where the lattice cone does not reach the
    exercise region -- see the warning in :func:`crr`.
    """
    res = crr(S0, K, T, r, sigma, N, q, kind, "american", boundary=True)
    return res.boundary_t, res.boundary_S
