r"""Closed-form perpetual American put -- an exact analytic anchor.

With no maturity the free-boundary problem of :doc:`docs/01_formulation` becomes
time homogeneous, and the continuation-region equation collapses to the Euler ODE

.. math::
    \tfrac12\sigma^2 S^2 V'' + (r-q) S V' - r V = 0,

with solutions :math:`S^{\beta}` where

.. math::
    \tfrac12\sigma^2\beta(\beta-1) + (r-q)\beta - r = 0 .

Boundedness as :math:`S\to\infty` selects the negative root :math:`\beta_-`.
Imposing *value matching* and *smooth pasting* at the (constant) boundary
:math:`S_\infty` pins both the boundary and the multiplicative constant:

.. math::
    S_\infty = K\frac{\beta_-}{\beta_- - 1},
    \qquad
    V(S) = (K - S_\infty)\left(\frac{S}{S_\infty}\right)^{\beta_-},\; S > S_\infty,
    \qquad V(S) = K - S,\; S \le S_\infty .

For :math:`q = 0` the quadratic factors as :math:`(\beta + \gamma)(\beta - 1)`
with :math:`\gamma = 2r/\sigma^2`, so :math:`\beta_- = -\gamma` and
:math:`S_\infty = K\gamma/(1+\gamma)`.

This is *not* used to price anything in the study.  It exists because it gives
two exact statements that any finite-maturity American solver must respect:

* :math:`V^{\text{perp}}(S) \ge V^{\text{amer}}(S, 0; T)` for every :math:`T`,
  with convergence from below as :math:`T \to \infty`;
* the extracted boundary must satisfy :math:`S_\infty < S^*(0) < K` and decrease
  towards :math:`S_\infty` as :math:`T` grows.
"""

from __future__ import annotations

import numpy as np

__all__ = ["beta_minus", "perpetual_put_boundary", "perpetual_put_price"]


def beta_minus(r: float, sigma: float, q: float = 0.0) -> float:
    r"""Negative root of :math:`\tfrac12\sigma^2\beta(\beta-1) + (r-q)\beta - r = 0`."""
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    if r < 0.0:
        raise ValueError("the perpetual put is only well posed for r >= 0")
    m = r - q - 0.5 * sigma**2
    disc = m**2 + 2.0 * sigma**2 * r
    return (-m - np.sqrt(disc)) / sigma**2


def perpetual_put_boundary(K: float, r: float, sigma: float, q: float = 0.0) -> float:
    r"""The constant exercise boundary :math:`S_\infty = K\beta_-/(\beta_- - 1)`.

    For :math:`r = 0` this returns :math:`0`: with no interest to earn there is
    never a reason to exercise a perpetual put (§1.3), so the exercise region is
    empty and the boundary degenerates to the origin.
    """
    b = beta_minus(r, sigma, q)
    return float(K * b / (b - 1.0))


def perpetual_put_price(S, K: float, r: float, sigma: float, q: float = 0.0):
    r"""Value of the perpetual American put, vectorised over ``S``.

    The :math:`r = 0` case is a genuine degeneracy rather than a division by
    zero.  With :math:`r = 0` we get :math:`\beta_- = 0` and :math:`S_\infty = 0`,
    so it is never optimal to stop.  But under :math:`\mathbb{Q}` with
    :math:`r = 0`, :math:`S_t = S_0\exp(-\tfrac12\sigma^2 t + \sigma W_t) \to 0`
    almost surely, so :math:`(K - S_t)^+ \to K` a.s. and the *supremum* over
    stopping times is :math:`K` -- approached, never attained.  Returning
    :math:`0` here (the "never exercise, so worthless" reading) is wrong and
    would break the dominance check
    :math:`V^{\mathrm{perp}} \ge V^{\mathrm{amer}}(T)`.
    """
    S = np.asarray(S, dtype=float)
    b = beta_minus(r, sigma, q)
    if b == 0.0:  # r == 0
        return np.full_like(S, float(K))
    S_inf = K * b / (b - 1.0)
    cont = (K - S_inf) * np.power(np.maximum(S, 1e-300) / S_inf, b)
    return np.where(S <= S_inf, K - S, cont)
