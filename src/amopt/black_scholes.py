r"""Closed-form Black-Scholes-Merton prices and Greeks for European options.

Under the risk-neutral measure :math:`\mathbb{Q}` the underlying follows

.. math::
    dS_t = (r - q) S_t\, dt + \sigma S_t\, dW_t^{\mathbb{Q}},

so :math:`S_T = S_0 \exp[(r - q - \tfrac12\sigma^2)T + \sigma\sqrt{T} Z]`
with :math:`Z \sim N(0,1)`.  Discounted expected payoffs give the standard
formulas implemented below.  These serve three purposes in this repository:

1. an exact benchmark for the *European* limit of every numerical scheme,
2. the control variate used in :mod:`amopt.variance_reduction`,
3. a lower bound for the American put (early exercise has non-negative value).

All functions are vectorised over ``S``, ``K``, ``T``, ``r``, ``q`` and
``sigma`` via NumPy broadcasting, and handle the degenerate limits
:math:`T \to 0` and :math:`\sigma \to 0` by returning the discounted
forward intrinsic value rather than ``nan``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "d1_d2",
    "bs_price",
    "bs_call",
    "bs_put",
    "bs_greeks",
    "forward_intrinsic",
]

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _phi(x):
    """Standard normal pdf (local implementation avoids a scipy call in hot loops)."""
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / _SQRT_2PI


def _N(x):
    """Standard normal cdf."""
    return norm.cdf(x)


def forward_intrinsic(S, K, T, r, q=0.0, kind="call"):
    r"""Discounted intrinsic value of the *forward*, i.e. the :math:`\sigma \to 0` price.

    With zero volatility :math:`S_T = S e^{(r-q)T}` deterministically, so the
    option is worth :math:`e^{-rT}(S e^{(r-q)T} - K)^+` for a call.
    """
    S, K, T, r, q = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (S, K, T, r, q)))
    fwd = S * np.exp((r - q) * T)
    disc = np.exp(-r * T)
    if kind == "call":
        return disc * np.maximum(fwd - K, 0.0)
    if kind == "put":
        return disc * np.maximum(K - fwd, 0.0)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def d1_d2(S, K, T, r, sigma, q=0.0):
    r"""Return the Black-Scholes :math:`d_1, d_2`.

    .. math::
        d_1 = \frac{\ln(S/K) + (r - q + \tfrac12\sigma^2)T}{\sigma\sqrt{T}},
        \qquad d_2 = d_1 - \sigma\sqrt{T}.

    Where :math:`\sigma\sqrt{T} = 0` the ratio is undefined; we return
    :math:`\pm\infty` according to the sign of the log-moneyness so that the
    normal cdf collapses to the correct 0/1 indicator.
    """
    S, K, T, r, sigma, q = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (S, K, T, r, sigma, q))
    )
    vol = sigma * np.sqrt(np.maximum(T, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.log(S / K) + (r - q + 0.5 * sigma**2) * T
        d1 = np.where(vol > 0.0, num / np.where(vol > 0.0, vol, 1.0), np.sign(num) * np.inf)
        d2 = np.where(vol > 0.0, d1 - vol, d1)
    return d1, d2


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    r"""European Black-Scholes-Merton price.

    .. math::
        C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad
        P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1).
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    S, K, T, r, sigma, q = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (S, K, T, r, sigma, q))
    )
    degenerate = (T <= 0.0) | (sigma <= 0.0)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    if kind == "call":
        val = S * dq * _N(d1) - K * dr * _N(d2)
    else:
        val = K * dr * _N(-d2) - S * dq * _N(-d1)
    return np.where(degenerate, forward_intrinsic(S, K, np.maximum(T, 0.0), r, q, kind), val)


def bs_call(S, K, T, r, sigma, q=0.0):
    """European call price (thin wrapper on :func:`bs_price`)."""
    return bs_price(S, K, T, r, sigma, q, kind="call")


def bs_put(S, K, T, r, sigma, q=0.0):
    """European put price (thin wrapper on :func:`bs_price`)."""
    return bs_price(S, K, T, r, sigma, q, kind="put")


def bs_greeks(S, K, T, r, sigma, q=0.0, kind="call"):
    r"""Analytic first- and second-order Greeks.

    Returns a dict with ``price, delta, gamma, vega, theta, rho``.

    Conventions
    -----------
    * ``vega``  is :math:`\partial V/\partial\sigma` per **unit** volatility
      (multiply by 0.01 for "per volatility point").
    * ``theta`` is :math:`\partial V/\partial t` per **year**, i.e. the
      calendar-time decay, negative for a long vanilla in most regimes.
    * ``rho``   is :math:`\partial V/\partial r` per unit rate.
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    S, K, T, r, sigma, q = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (S, K, T, r, sigma, q))
    )
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    sqrtT = np.sqrt(np.maximum(T, 0.0))
    pdf = _phi(d1)
    safe = (T > 0.0) & (sigma > 0.0)

    gamma = np.where(safe, dq * pdf / np.where(safe, S * sigma * sqrtT, 1.0), 0.0)
    vega = np.where(safe, S * dq * pdf * sqrtT, 0.0)
    common_theta = np.where(safe, -S * dq * pdf * sigma / (2.0 * np.where(safe, sqrtT, 1.0)), 0.0)

    if kind == "call":
        delta = dq * _N(d1)
        theta = common_theta - r * K * dr * _N(d2) + q * S * dq * _N(d1)
        rho = K * T * dr * _N(d2)
    else:
        delta = -dq * _N(-d1)
        theta = common_theta + r * K * dr * _N(-d2) - q * S * dq * _N(-d1)
        rho = -K * T * dr * _N(-d2)

    return {
        "price": bs_price(S, K, T, r, sigma, q, kind),
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
