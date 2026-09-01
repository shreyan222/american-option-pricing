"""Canonical parameter sets used across every experiment in this repository.

Fixing the test cases in one place means the convergence study, the variance
reduction study and the boundary study all speak about the same contracts, so
their numbers can be compared directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = ["MarketParams", "BASE", "REGIMES", "REGIME_NOTES"]


@dataclass(frozen=True)
class MarketParams:
    """A Black-Scholes market plus contract terms for a single vanilla option."""

    S0: float = 100.0
    K: float = 100.0
    T: float = 1.0
    r: float = 0.05
    sigma: float = 0.20
    q: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)

    def replace(self, **kw) -> "MarketParams":
        return MarketParams(**{**asdict(self), **kw})

    def label(self) -> str:
        return (
            f"S0={self.S0:g}, K={self.K:g}, T={self.T:g}, "
            f"r={self.r:.0%}, sigma={self.sigma:.0%}, q={self.q:.0%}"
        )


#: The headline contract: at-the-money one-year American put.
BASE = MarketParams()

#: Parameter regimes that stress different parts of the numerics.
REGIMES: dict[str, MarketParams] = {
    "base": BASE,
    "itm": BASE.replace(S0=85.0),
    "otm": BASE.replace(S0=115.0),
    "low_vol": BASE.replace(sigma=0.10),
    "high_vol": BASE.replace(sigma=0.40),
    "short_maturity": BASE.replace(T=0.25),
    "long_maturity": BASE.replace(T=3.0),
    "high_rate": BASE.replace(r=0.10),
    "zero_rate": BASE.replace(r=0.0),
    "dividend": BASE.replace(q=0.04),
}

REGIME_NOTES: dict[str, str] = {
    "base": "at-the-money reference contract",
    "itm": "in-the-money put: large early-exercise region",
    "otm": "out-of-the-money put: few in-the-money Monte Carlo paths",
    "low_vol": "low volatility: exercise boundary close to the strike, stiff PDE",
    "high_vol": "high volatility: wide grid needed, large continuation value",
    "short_maturity": "short maturity: few exercise dates, steep payoff kink",
    "long_maturity": "long maturity: large early-exercise premium",
    "high_rate": "high rate: strong incentive to exercise early",
    "zero_rate": "zero rate: American put must coincide with the European put",
    "dividend": "continuous dividend yield: early exercise for calls becomes possible",
}
