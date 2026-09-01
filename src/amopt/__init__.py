"""American option pricing from first principles.

Sub-modules
-----------
black_scholes    Closed-form European prices and Greeks under Black-Scholes.
binomial         Cox-Ross-Rubinstein lattice (European and American exercise).
crank_nicolson   Crank-Nicolson finite differences with PSOR for the LCP.
lsm              Longstaff-Schwartz least-squares Monte Carlo.
variance_reduction  Antithetic and control-variate LSM estimators.
util             Timing helpers and shared dataclasses.
"""

__version__ = "0.1.0"
