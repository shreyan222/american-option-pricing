r"""Longstaff-Schwartz least-squares Monte Carlo for the American put.

The PDE solver attacks the variational inequality; LSM stays with the
probabilistic formulation (1.1) and approximates the *optimal stopping rule*
directly.  At each exercise date the holder compares the immediate payoff with
the **continuation value**

.. math::
    C(S_{t_n}) = \mathbb{E}^{\mathbb{Q}}\!\left[
        e^{-r(\tau - t_n)} g(S_\tau) \mid \mathcal{F}_{t_n} \right],

which is a conditional expectation and therefore a function of the state.  LSM
estimates that function by **least-squares regression of realised future
discounted cash flows on a basis of functions of the current state**, path by
path, working backwards from maturity.

What is actually being priced
-----------------------------
With :math:`n` exercise dates this prices a **Bermudan** option, not an American
one.  Restricting exercise to a finite set of dates can only lose value, so the
Bermudan price is a *lower* bound on the American price and the gap closes as
:math:`n \to \infty`.  That systematic downward bias is measured in Milestone 6,
not assumed negligible.

Bias, look-ahead and data leakage
---------------------------------
Two biases act in opposite directions.

* **Low bias from a suboptimal policy.** Any fixed exercise rule gives a lower
  bound on the option value.  A regression-based rule is not the optimal rule, so
  it loses value.
* **High bias from in-sample foresight.** The classic estimator fits the
  regression on the *same* paths it then values.  The fitted continuation values
  are partly explaining the noise of those particular paths, so the exercise
  decisions are made with a sliver of hindsight and the estimate is biased
  upward.  The reported standard error is also optimistic, because the payoffs
  are no longer independent of the policy.

This module implements both.  :func:`longstaff_schwartz` with
``out_of_sample=False`` is the classic in-sample estimator.  With
``out_of_sample=True`` the regression coefficients are fitted on one set of paths
and the resulting policy is then applied to an **independent** set, which
removes the foresight bias entirely: the result is a genuine lower bound on the
Bermudan value with an honest i.i.d. standard error.  The difference between the
two is measured in `RESULTS.md` §4 rather than asserted small.

What is *not* leakage
---------------------
Using realised future cash flows along a path as the regression target is not
look-ahead bias.  The regression is a projection onto functions of
:math:`\mathcal{F}_{t_n}`-measurable state; the future cash flow is only the
noisy observation of the conditional expectation being estimated.  What would be
leakage is letting a *future* state variable into the basis, or letting the
regression see the exercise decision it is about to inform.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

__all__ = [
    "LSMResult",
    "simulate_gbm",
    "basis_matrix",
    "longstaff_schwartz",
]

Z_975 = float(norm.ppf(0.975))


# ---------------------------------------------------------------------------
# Path simulation
# ---------------------------------------------------------------------------

def simulate_gbm(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    q: float = 0.0,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    antithetic: bool = False,
) -> np.ndarray:
    r"""Simulate risk-neutral GBM paths on a uniform time grid.

    Uses the **exact** solution of the SDE,

    .. math::
        S_{t+\Delta t} = S_t \exp\!\left[(r - q - \tfrac12\sigma^2)\Delta t
                                        + \sigma\sqrt{\Delta t}\, Z\right],
        \qquad Z \sim N(0,1),

    so there is no Euler discretisation error: the simulated marginals are exact
    for any ``n_steps``.  The only effect of ``n_steps`` is on the *exercise*
    dates available to the Bermudan approximation.

    With ``antithetic=True`` the ``n_paths`` returned paths consist of
    ``n_paths // 2`` independent draws paired with their sign-flipped
    counterparts, arranged so that path ``2k`` and path ``2k+1`` form a pair.
    ``n_paths`` must then be even.  Downstream estimators must account for the
    dependence within a pair -- see :mod:`amopt.variance_reduction`.

    Returns
    -------
    ndarray of shape ``(n_paths, n_steps + 1)`` with ``S[:, 0] = S0``.
    """
    if n_paths < 1 or n_steps < 1:
        raise ValueError("n_paths and n_steps must be >= 1")
    if antithetic and n_paths % 2 != 0:
        raise ValueError("antithetic sampling requires an even n_paths")
    if rng is None:
        rng = np.random.default_rng(seed)

    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal((half, n_steps))
        Zfull = np.empty((n_paths, n_steps))
        Zfull[0::2] = Z
        Zfull[1::2] = -Z
    else:
        Zfull = rng.standard_normal((n_paths, n_steps))

    log_increments = drift + vol * Zfull
    log_paths = np.cumsum(log_increments, axis=1)
    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    np.exp(log_paths, out=log_paths)
    S[:, 1:] = S0 * log_paths
    return S


# ---------------------------------------------------------------------------
# Regression bases
# ---------------------------------------------------------------------------

def basis_matrix(S: np.ndarray, K: float, degree: int, kind: str = "poly") -> np.ndarray:
    r"""Design matrix of basis functions evaluated at ``S``.

    The state is always normalised as :math:`x = S/K` before the basis is
    applied.  Without normalisation the monomial design matrix at :math:`S \sim
    100` has a condition number of order :math:`100^{2d}`, and the fitted
    coefficients become numerically meaningless well before ``degree = 5``.

    Parameters
    ----------
    kind : {'poly', 'laguerre', 'chebyshev'}
        ``'poly'`` -- monomials :math:`1, x, \dots, x^d`.
        ``'laguerre'`` -- weighted Laguerre functions
        :math:`e^{-x/2} L_k(x)`, the basis used in the original
        Longstaff-Schwartz paper.
        ``'chebyshev'`` -- Chebyshev polynomials of the first kind on
        :math:`x \in [0, 2]` mapped to :math:`[-1, 1]`, the best-conditioned of
        the three.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")
    x = np.asarray(S, dtype=float) / K
    n = x.size
    cols = degree + 1
    A = np.empty((n, cols))
    if kind == "poly":
        A[:, 0] = 1.0
        for k in range(1, cols):
            A[:, k] = A[:, k - 1] * x
    elif kind == "laguerre":
        w = np.exp(-0.5 * x)
        L_prev = np.ones(n)
        L_cur = 1.0 - x
        A[:, 0] = w * L_prev
        if cols > 1:
            A[:, 1] = w * L_cur
        for k in range(2, cols):
            L_next = ((2 * k - 1 - x) * L_cur - (k - 1) * L_prev) / k
            A[:, k] = w * L_next
            L_prev, L_cur = L_cur, L_next
    elif kind == "chebyshev":
        u = np.clip(x - 1.0, -1.0, 1.0)  # map x in [0,2] -> [-1,1]
        A[:, 0] = 1.0
        if cols > 1:
            A[:, 1] = u
        for k in range(2, cols):
            A[:, k] = 2.0 * u * A[:, k - 1] - A[:, k - 2]
    else:
        raise ValueError(f"unknown basis {kind!r}")
    return A


def _predict(A: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """``A @ beta`` with a guard around a spurious NumPy floating-point warning.

    On NumPy 2.0.2 the BLAS-backed ``matmul`` raises divide-by-zero / overflow /
    invalid flags from its SIMD tail lanes even when every input and every output
    is finite.  Verified against ``np.einsum`` (agreement to 8.5e-14) and against
    an explicit ``(A * beta).sum(axis=1)`` (bitwise identical).  The flags are
    suppressed and replaced with an explicit finiteness check, so a *genuine*
    numerical failure -- an ill-conditioned design matrix producing infinite
    coefficients, say -- still raises rather than passing silently.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = A @ beta
    if not np.isfinite(out).all():
        raise FloatingPointError(
            "non-finite continuation values: the regression design matrix is "
            f"ill-conditioned (cond = {np.linalg.cond(A):.3e}). Lower `degree` or "
            "use basis='chebyshev'."
        )
    return out


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class LSMResult:
    """A Longstaff-Schwartz estimate and everything needed to judge it."""

    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int
    n_steps: int
    degree: int
    basis: str
    itm_only: bool
    out_of_sample: bool
    antithetic: bool
    seed: int | None
    runtime_s: float
    #: Fraction of paths exercised strictly before maturity, under the fitted policy.
    early_exercise_fraction: float = 0.0
    #: Mean stopping time in years.
    mean_stopping_time: float = 0.0
    #: Per-date count of regressions skipped for having too few in-the-money paths.
    n_skipped_regressions: int = 0
    #: Immediate exercise at t=0 was optimal (deep in the money).
    exercised_at_zero: bool = False
    payoffs: np.ndarray | None = field(default=None, repr=False)
    coefficients: list | None = field(default=None, repr=False)

    def deviation_from(self, benchmark: float) -> dict:
        """Signed deviation from a benchmark, in absolute and standard-error units."""
        d = self.price - benchmark
        return {
            "benchmark": benchmark,
            "deviation": d,
            "abs_deviation": abs(d),
            "deviation_in_se": d / self.std_error if self.std_error > 0 else np.nan,
            "benchmark_in_ci": self.ci_low <= benchmark <= self.ci_high,
        }


# ---------------------------------------------------------------------------
# The algorithm
# ---------------------------------------------------------------------------

def _payoff(S, K, kind):
    return np.maximum(K - S, 0.0) if kind == "put" else np.maximum(S - K, 0.0)


def _fit_policy(S, K, r, T, n_steps, degree, basis, itm_only, kind):
    r"""Backward induction on the training paths.

    Returns ``(coefficients, cash, stop_index, n_skipped)`` where
    ``coefficients[n]`` are the regression coefficients for exercise date ``n``
    (``None`` if no regression was possible there), ``cash`` is each path's
    realised cash flow discounted to :math:`t_0`, and ``stop_index`` is the
    exercise date index per path (``n_steps`` meaning "held to maturity").
    """
    n_paths = S.shape[0]
    dt = T / n_steps
    disc = np.exp(-r * dt)

    cash = _payoff(S[:, n_steps], K, kind)
    stop_index = np.full(n_paths, n_steps, dtype=np.int64)
    coefficients: list = [None] * (n_steps + 1)
    n_skipped = 0

    for n in range(n_steps - 1, 0, -1):
        cash *= disc  # roll the realised cash flow back to t_n
        ex = _payoff(S[:, n], K, kind)
        sel = ex > 0.0 if itm_only else np.ones(n_paths, dtype=bool)
        n_sel = int(sel.sum())
        if n_sel <= degree + 1:
            # Not enough information to fit; hold everywhere at this date.
            n_skipped += 1
            continue
        A = basis_matrix(S[sel, n], K, degree, basis)
        beta, *_ = np.linalg.lstsq(A, cash[sel], rcond=None)
        coefficients[n] = beta
        continuation = _predict(A, beta)
        do_ex = ex[sel] > continuation
        idx = np.flatnonzero(sel)[do_ex]
        cash[idx] = ex[idx]
        stop_index[idx] = n

    cash *= disc  # from t_1 to t_0
    return coefficients, cash, stop_index, n_skipped


def _apply_policy(S, K, r, T, n_steps, degree, basis, itm_only, kind, coefficients):
    """Value a *fixed* policy on fresh paths -- forward, one decision per date.

    This is what makes the out-of-sample estimator a valid lower bound: the
    exercise rule is decided entirely by coefficients fitted elsewhere, so no
    information from these paths influences when they stop.
    """
    n_paths = S.shape[0]
    dt = T / n_steps
    alive = np.ones(n_paths, dtype=bool)
    cash = np.zeros(n_paths)
    stop_index = np.full(n_paths, n_steps, dtype=np.int64)

    for n in range(1, n_steps):
        beta = coefficients[n]
        if beta is None:
            continue
        ex = _payoff(S[:, n], K, kind)
        sel = alive & ((ex > 0.0) if itm_only else np.ones(n_paths, dtype=bool))
        if not sel.any():
            continue
        A = basis_matrix(S[sel, n], K, degree, basis)
        do_ex = ex[sel] > _predict(A, beta)
        idx = np.flatnonzero(sel)[do_ex]
        cash[idx] = ex[idx] * np.exp(-r * n * dt)
        stop_index[idx] = n
        alive[idx] = False

    term = _payoff(S[alive, n_steps], K, kind)
    cash[alive] = term * np.exp(-r * T)
    return cash, stop_index


def longstaff_schwartz(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: str = "put",
    n_paths: int = 100_000,
    n_steps: int = 50,
    degree: int = 3,
    basis: str = "poly",
    itm_only: bool = True,
    seed: int | None = 0,
    antithetic: bool = False,
    out_of_sample: bool = False,
    paths: np.ndarray | None = None,
    return_payoffs: bool = False,
) -> LSMResult:
    r"""Price an American (Bermudan) option by least-squares Monte Carlo.

    Parameters
    ----------
    n_paths, n_steps :
        Number of simulated paths and exercise dates.  With ``out_of_sample``
        the *total* simulation budget is ``2 * n_paths``: ``n_paths`` to fit the
        policy and ``n_paths`` independent paths to value it.
    degree, basis, itm_only :
        Regression specification.  ``itm_only=True`` is the Longstaff-Schwartz
        recommendation: regress only on paths that are in the money, where the
        exercise decision is actually live.
    out_of_sample :
        Fit the policy on one sample and value it on an independent one.
        Removes the in-sample foresight bias and makes the standard error and
        confidence interval honest.  The result is a valid *lower* bound on the
        Bermudan value.
    antithetic :
        Use antithetic pairs.  Note the pairs are dependent, so the naive i.i.d.
        standard error would be wrong; it is computed over **pair means** here.
        See :mod:`amopt.variance_reduction`.

    Returns
    -------
    LSMResult
    """
    t0 = time.perf_counter()
    if kind not in ("put", "call"):
        raise ValueError(f"kind must be 'put' or 'call', got {kind!r}")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    rng = np.random.default_rng(seed)

    if paths is not None:
        S = paths
        n_paths, n_steps = S.shape[0], S.shape[1] - 1
    else:
        S = simulate_gbm(S0, r, sigma, T, n_paths, n_steps, q, rng=rng, antithetic=antithetic)

    coefficients, cash, stop_index, n_skipped = _fit_policy(
        S, K, r, T, n_steps, degree, basis, itm_only, kind
    )

    if out_of_sample:
        S_test = simulate_gbm(S0, r, sigma, T, n_paths, n_steps, q, rng=rng, antithetic=antithetic)
        cash, stop_index = _apply_policy(
            S_test, K, r, T, n_steps, degree, basis, itm_only, kind, coefficients
        )

    price, se = _mean_and_se(cash, antithetic)

    # Exercising immediately at t=0 is always available and is not part of the
    # regression (there is only one state there, so no cross-section to fit).
    immediate = float(_payoff(np.array([S0]), K, kind)[0])
    exercised_at_zero = immediate > price
    if exercised_at_zero:
        price, se = immediate, 0.0

    dt = T / n_steps
    res = LSMResult(
        price=float(price),
        std_error=float(se),
        ci_low=float(price - Z_975 * se),
        ci_high=float(price + Z_975 * se),
        n_paths=int(n_paths), n_steps=int(n_steps), degree=degree, basis=basis,
        itm_only=itm_only, out_of_sample=out_of_sample, antithetic=antithetic,
        seed=seed, runtime_s=time.perf_counter() - t0,
        early_exercise_fraction=float(np.mean(stop_index < n_steps)),
        mean_stopping_time=float(np.mean(stop_index) * dt),
        n_skipped_regressions=n_skipped,
        exercised_at_zero=bool(exercised_at_zero),
        payoffs=cash if return_payoffs else None,
        coefficients=coefficients,
    )
    return res


def _mean_and_se(cash: np.ndarray, antithetic: bool):
    r"""Sample mean and standard error of the discounted cash flows.

    With antithetic sampling paths ``2k`` and ``2k+1`` are *negatively dependent*
    by construction, so treating them as ``n`` independent observations
    understates the variance of the mean.  The correct unit of independence is
    the **pair**: averaging within each pair gives ``n/2`` i.i.d. observations,
    and the standard error of their mean is the honest one.  Getting this wrong
    is the standard way to report a spuriously tight antithetic confidence
    interval.
    """
    cash = np.asarray(cash, dtype=float)
    if antithetic:
        pair_means = 0.5 * (cash[0::2] + cash[1::2])
        m = pair_means.size
        return float(pair_means.mean()), float(pair_means.std(ddof=1) / np.sqrt(m))
    n = cash.size
    return float(cash.mean()), float(cash.std(ddof=1) / np.sqrt(n))
