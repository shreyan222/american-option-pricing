r"""Variance reduction for Longstaff-Schwartz: antithetic variates and a control variate.

All four estimators here price the same object -- the :math:`n`-date Bermudan put
-- under a policy fitted on an independent training sample.  They differ only in
how the valuation sample is drawn and post-processed, so any difference in the
*width* of the confidence interval is attributable to the variance-reduction
technique and not to a change of target.

Antithetic variates
-------------------
Draw :math:`Z` and use both :math:`Z` and :math:`-Z`.  The two resulting payoffs
are negatively correlated whenever the payoff is a monotone function of the
driving noise, so their average has lower variance than two independent draws.

The statistical trap is the error bar.  Paths :math:`2k` and :math:`2k+1` are
*dependent by construction*, so the :math:`2n` path payoffs are not :math:`2n`
observations.  The unit of independence is the **pair**: average within each
pair, and the :math:`n` pair means are i.i.d.  Writing
:math:`\operatorname{Var}(X)` for the marginal path variance and :math:`c` for
the within-pair covariance,

.. math::
    \operatorname{Var}(\bar X_{\text{pairs}})
      = \frac{\operatorname{Var}(X) + c}{2n},
    \qquad\text{versus the naive}\qquad
    \frac{\operatorname{Var}(X)}{2n}.

The naive path-level formula is therefore wrong by the factor
:math:`\sqrt{1 + c/\operatorname{Var}(X)}` in *either* direction.  When the
payoff is monotone in the driving noise -- the case antithetic sampling is
designed for -- :math:`c < 0` and the naive formula is merely conservative.
When it is not monotone, :math:`c > 0`, and the naive interval is too narrow and
under-covers.  Both directions are measured in `RESULTS.md` §5.  Everything here
computes the pair statistic, and :attr:`VRResult.path_variance` is retained so
the naive alternative can be reconstructed and compared.

Control variate
---------------
The European put on the same paths has a payoff
:math:`Y = e^{-rT}(K - S_T)^+` whose expectation is known in closed form.  Since
:math:`Y` is strongly correlated with the American cash flow :math:`X`, the
adjusted estimator

.. math::
    X^{\mathrm{cv}} = X - b\,\bigl(Y - \mathbb{E}[Y]\bigr)

is unbiased for any constant :math:`b`, and the variance-minimising choice is

.. math::
    b^\star = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(Y)},
    \qquad
    \operatorname{Var}(X^{\mathrm{cv}}) = \operatorname{Var}(X)\,(1 - \rho^2).

The variance reduction is therefore governed entirely by :math:`\rho^2`, the
squared correlation between the American and European payoffs.

:math:`b^\star` has to be estimated.  Estimating it on the *same* sample being
valued makes the estimator only asymptotically unbiased, with a bias of order
:math:`1/n`.  This module estimates :math:`b` on the **training** sample (the
one already used to fit the exercise policy), which costs nothing extra and
leaves the valuation sample untouched -- so the estimator is exactly unbiased.
``beta_source='sample'`` reproduces the conventional same-sample choice, and the
difference is measured in `RESULTS.md` §5 rather than assumed negligible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .black_scholes import bs_price
from .lsm import Z_975, _apply_policy, _fit_policy, simulate_gbm

__all__ = ["VRResult", "lsm_with_variance_reduction", "METHODS"]

METHODS = ("naive", "antithetic", "control", "antithetic_control")


@dataclass
class VRResult:
    """One variance-reduced LSM estimate with fully specified error statistics."""

    method: str
    price: float
    std_error: float
    ci_low: float
    ci_high: float
    ci_width: float
    #: Variance of a single independent sampling *unit* (a path, or an antithetic pair).
    unit_variance: float
    #: Marginal variance of an individual path payoff, before any pairing.
    #: Retained so the (incorrect) naive path-level standard error can be
    #: reconstructed and its coverage measured.
    path_variance: float
    #: Within-pair correlation of the path payoffs (``nan`` when not antithetic).
    pair_correlation: float
    #: Number of independent units (``n_paths`` or ``n_paths // 2``).
    n_units: int
    #: Simulated valuation paths (twice the unit count under antithetic sampling).
    n_paths: int
    n_steps: int
    runtime_s: float
    #: Control-variate coefficient actually used (``nan`` when no control variate).
    beta: float = float("nan")
    #: Correlation between the American cash flow and the European control.
    correlation: float = float("nan")
    beta_source: str = ""
    seed: int | None = None

    def naive_path_level_se(self) -> float:
        r"""The *incorrect* standard error obtained by ignoring antithetic pairing.

        Provided so that the size and direction of the mistake can be measured
        rather than described.  Equals :attr:`std_error` when no pairing is used.
        """
        return float(np.sqrt(self.path_variance / self.n_paths))

    def paths_for_target_se(self, target_se: float) -> float:
        r"""Independent units needed to reach ``target_se``, from :math:`\sigma/\sqrt{n}`.

        Reported in *paths*, so antithetic results are directly comparable on
        simulation cost: one antithetic unit consumes two paths.
        """
        units = self.unit_variance / target_se**2
        return units * (2.0 if "antithetic" in self.method else 1.0)


def lsm_with_variance_reduction(
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
    method: str = "naive",
    beta_source: str = "training",
    seed: int | None = 0,
    n_train: int | None = None,
) -> VRResult:
    """Price a Bermudan option by LSM with the requested variance-reduction scheme.

    The exercise policy is always fitted on an independent training sample, so
    the estimate is a valid lower bound and the four methods are directly
    comparable.  ``n_paths`` counts *valuation* paths.

    Parameters
    ----------
    method : {'naive', 'antithetic', 'control', 'antithetic_control'}
    beta_source : {'training', 'sample', 'none'}
        Where the control-variate coefficient comes from.  ``'training'`` keeps
        the estimator exactly unbiased; ``'sample'`` is the conventional choice
        and is unbiased only as ``n -> inf``.
    n_train : int, optional
        Training path count; defaults to ``n_paths``.
    """
    t0 = time.perf_counter()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if beta_source not in ("training", "sample", "none"):
        raise ValueError(f"beta_source must be 'training', 'sample' or 'none', got {beta_source!r}")
    antithetic = "antithetic" in method
    use_cv = "control" in method
    if antithetic and n_paths % 2 != 0:
        raise ValueError("antithetic sampling requires an even n_paths")

    rng = np.random.default_rng(seed)
    n_train = n_train or n_paths

    # ---- 1. fit the exercise policy on an independent training sample ----
    S_tr = simulate_gbm(S0, r, sigma, T, n_train, n_steps, q, rng=rng)
    coefficients, _, _, _ = _fit_policy(S_tr, K, r, T, n_steps, degree, basis, itm_only, kind)

    # ---- 2. control-variate coefficient, also from the training sample ----
    beta = float("nan")
    if use_cv and beta_source == "training":
        X_tr, _ = _apply_policy(S_tr, K, r, T, n_steps, degree, basis, itm_only, kind, coefficients)
        Y_tr = _european_control(S_tr[:, -1], K, r, T, kind)
        beta = _ols_beta(X_tr, Y_tr)

    # ---- 3. valuation on fresh paths ------------------------------------
    S = simulate_gbm(S0, r, sigma, T, n_paths, n_steps, q, rng=rng, antithetic=antithetic)
    X, _ = _apply_policy(S, K, r, T, n_steps, degree, basis, itm_only, kind, coefficients)

    corr = float("nan")
    if use_cv:
        Y = _european_control(S[:, -1], K, r, T, kind)
        EY = float(bs_price(S0, K, T, r, sigma, q, kind))
        if beta_source == "sample":
            beta = _ols_beta(X, Y)
        elif beta_source == "none":
            beta = 1.0
        corr = float(np.corrcoef(X, Y)[0, 1])
        X = X - beta * (Y - EY)

    # ---- 4. statistics on the correct unit of independence ---------------
    path_var = float(X.var(ddof=1))
    if antithetic:
        units = 0.5 * (X[0::2] + X[1::2])
        a, b = X[0::2], X[1::2]
        pair_corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    else:
        units = X
        pair_corr = float("nan")
    n_units = units.size
    price = float(units.mean())
    var = float(units.var(ddof=1))
    se = float(np.sqrt(var / n_units))

    return VRResult(
        method=method, price=price, std_error=se,
        ci_low=price - Z_975 * se, ci_high=price + Z_975 * se, ci_width=2 * Z_975 * se,
        unit_variance=var, path_variance=path_var, pair_correlation=pair_corr,
        n_units=n_units, n_paths=n_paths, n_steps=n_steps,
        runtime_s=time.perf_counter() - t0,
        beta=beta, correlation=corr, beta_source=beta_source if use_cv else "",
        seed=seed,
    )


def _european_control(S_T: np.ndarray, K: float, r: float, T: float, kind: str) -> np.ndarray:
    """Discounted European payoff on the same terminal prices."""
    payoff = np.maximum(K - S_T, 0.0) if kind == "put" else np.maximum(S_T - K, 0.0)
    return np.exp(-r * T) * payoff


def _ols_beta(X: np.ndarray, Y: np.ndarray) -> float:
    r"""Variance-minimising control coefficient :math:`\operatorname{Cov}(X,Y)/\operatorname{Var}(Y)`.

    The ``errstate`` guard is for the same spurious NumPy 2.0.2 ``matmul``
    floating-point flags documented in :func:`amopt.lsm._predict`; the result is
    checked for finiteness rather than trusted blindly.
    """
    Yc = Y - Y.mean()
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        denom = float(Yc @ Yc)
        num = float(Yc @ (X - X.mean()))
    if not (np.isfinite(denom) and np.isfinite(num)):
        raise FloatingPointError("non-finite control-variate regression; inputs are degenerate")
    if denom <= 0.0:
        return 0.0
    return num / denom


def variance_reduction_factor(baseline: VRResult, other: VRResult) -> float:
    r"""Ratio of squared standard errors **at equal path count**.

    Comparing raw ``unit_variance`` would flatter antithetic sampling, whose unit
    consumes two paths.  Rescaling to a common path budget is the honest
    comparison, and is what
    :meth:`VRResult.paths_for_target_se` also reports.
    """
    def var_per_path(res: VRResult) -> float:
        return res.unit_variance * (2.0 if "antithetic" in res.method else 1.0)

    return var_per_path(baseline) / var_per_path(other)
