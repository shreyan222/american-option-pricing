# 3. The early-exercise boundary: predictions before measurement

This note states what the free boundary $S^*(t)$ *should* do, and why, **before**
`experiments/m7_exercise_boundary.py` measures it. Each prediction is labelled
and each is falsifiable; the measured outcomes are in `RESULTS.md` §7. Writing
the predictions down first is the point — a sensitivity study that only reports
what it found can rationalise any result.

Throughout, $\tau = T - t$ is time to maturity, $g(S) = (K-S)^+$, and the
exercise region is $\mathcal{E} = \{S \le S^*(t)\}$ (§1.6).

---

## 3.1 How the boundary is recovered

The PDE solver never receives $S^*$ as an input. At each time level it solves the
discrete linear complementarity problem, and the set of nodes where the
projection selects the payoff **is** the discrete exercise region. Its upper edge
is the raw boundary estimate, accurate to one grid spacing; the sub-grid
refinement inverts the quadratic vanishing of the early-exercise gap implied by
smooth pasting (§2.8):

$$
\phi(S) = v(S) - g(S) = \tfrac12 v_{SS}(S^*)\,(S - S^*)^2 + O\!\left((S-S^*)^3\right)
\;\Longrightarrow\;
\sqrt{\phi}\ \text{is locally linear in } S .
$$

The lattice can also report a boundary, but only inside the cone
$S_0 e^{\pm i\sigma\sqrt{\Delta t}}$ spanned from its single root, so it returns
`nan` near $t=0$. The PDE grid is fixed in $S$ and covers $[0, S_{\max}]$ at
every time level, so it resolves $S^*$ over the whole time axis. That asymmetry
is a genuine advantage of the PDE approach and is asserted in
`tests/test_crank_nicolson.py`.

---

## 3.2 The predictions

### P1 — Monotonicity in time

$S^*(t)$ is **non-decreasing in $t$**.

*Why.* The value of waiting is the value of the remaining optionality, which can
only shrink as maturity approaches. With less optionality to give up, the holder
is willing to exercise at prices closer to the strike. Formally, $V(S,t)$ is
non-increasing in $t$ for fixed $S$ while $g$ is time-independent, so the set
where $V = g$ can only grow.

### P2 — The upper bound and the terminal value

$$
S^*(t) \le \min\left\{K,\ \frac{rK}{q}\right\}, \qquad
\lim_{\tau \to 0^+} S^*(T - \tau) = \min\left\{K,\ \frac{rK}{q}\right\}.
$$

*Why.* Exercising is only ever optimal where holding the exercised position beats
holding the option locally, which by §1.3 requires $\mathcal{L}g < 0$, i.e.
$-rK + qS < 0$, i.e. $S < rK/q$. Combined with $S^* \le K$ (a put is not
exercised out of the money), the bound follows. As $\tau \to 0$ there is no time
value left, so the bound is attained.

*Financial reading.* Exercising converts the put into cash $K$, which earns $rK$
per unit time; the cost is forgoing the dividend $qS$ that a stock position would
not have earned anyway — but more importantly, once $qS > rK$ the drift of the
stock under $\mathbb{Q}$ is so negative that waiting is worth more than the
interest. With $q = 0$ this is simply $S^*(T^-) = K$.

**This prediction is sharp and easy to falsify:** with $K = 100$, $r = 5\%$ and
$q = 10\%$ it says the boundary just before maturity is $50$, not $100$.

*A genuine discontinuity.* At $\tau = 0$ exactly, every in-the-money put is
exercised, so the exercise region is all of $\{S < K\}$ and the boundary is $K$.
For any $\tau > 0$ it is $\min\{K, rK/q\}$. When $q > r$ the boundary is therefore
**discontinuous at maturity**. This is a real feature of the problem, not a
numerical artefact, and the solver reports both values.

### P3 — Near-maturity asymptotics

$$
K - S^*(T-\tau) \;\sim\; K\sigma\sqrt{\tau \ln(1/\tau)}
\qquad \text{as } \tau \to 0^+ \quad (q = 0,\ r > 0).
$$

*Why.* This is the classical result of Barles–Burdeau–Romano–Samsœn and
Kuske–Keller. Heuristically: exercise becomes optimal when the interest earned
over the remaining life, $\approx rK\tau$, outweighs the probability-weighted
value of the stock recovering above the boundary, and matching those two through
the Gaussian tail produces the $\sqrt{\tau\ln(1/\tau)}$ scale rather than a plain
$\sqrt{\tau}$.

*Consequence.* $dS^*/dt \to +\infty$ as $t \to T$: the boundary meets the strike
with **infinite slope**. Any picture of $S^*$ that approaches $K$ with a finite
slope is wrong.

*Caveat stated in advance.* The correction terms are of relative order
$1/\ln(1/\tau)$, so convergence to the limit is logarithmically slow. At
$\tau = 10^{-3}$, $1/\ln(1/\tau) \approx 0.14$, so the measured ratio should be
within roughly 15% of $1$ and **should not** be expected to be much closer.

### P4 — Volatility: $\partial S^*/\partial\sigma < 0$

Higher volatility **lowers** the boundary (exercise is deferred).

*Why.* Volatility is the raw material of optionality. The right to wait is worth
more when the stock might move a long way, so the holder demands a deeper
in-the-money price before giving that right up. The perpetual closed form makes
this exact: with $q = 0$, $S_\infty = K\gamma/(1+\gamma)$ with
$\gamma = 2r/\sigma^2$, which is strictly decreasing in $\sigma$.

### P5 — Interest rate: $\partial S^*/\partial r > 0$

Higher rates **raise** the boundary (exercise sooner).

*Why.* The entire reason to exercise a put early is to get the cash $K$ working
at rate $r$. Raise $r$ and that incentive strengthens, so exercise becomes
optimal at prices further from the money. Again exact in the perpetual limit:
$\gamma = 2r/\sigma^2$ is increasing in $r$, and $S_\infty = K\gamma/(1+\gamma)$
is increasing in $\gamma$. At $r = 0$ the incentive vanishes entirely and
$S_\infty = 0$: **never exercise** (§1.3).

### P6 — Maturity: $\partial S^*(0;T)/\partial T < 0$, converging to $S_\infty$

$S^*(0)$ is decreasing in $T$ and $S^*(0;T) \downarrow S_\infty$ as
$T \to \infty$, with $S^*(0;T) > S_\infty$ for every finite $T$.

*Why.* $S^*(0;T)$ is the same object as $S^*(T - \tau)$ at $\tau = T$, so P1
applied to a longer contract says it must be lower. The perpetual boundary is the
infimum over all maturities.

### P7 — Scale invariance

$S^*(t)/K$ depends on $(r, \sigma, q, T-t)$ but not on $K$ or $S_0$.

*Why.* Black–Scholes is homogeneous of degree one in $(S, K)$: $V(\lambda S,
\lambda K) = \lambda V(S, K)$. The boundary must scale with the strike. It cannot
depend on $S_0$ at all, since $S_0$ is not part of the free-boundary problem —
only of the query point.

---

## 3.3 What would falsify the implementation

| Observation | What it would mean |
|---|---|
| $S^*$ decreasing in $t$ anywhere beyond one grid cell | the projection or the time direction is wrong |
| $S^*(T^-) \ne \min(K, rK/q)$ | the boundary condition or the dividend term is wrong |
| $S^*(0) \le S_\infty$ | the solver is over-exercising; the perpetual value is a hard floor |
| $S^*$ approaching $K$ with finite slope | the near-maturity behaviour is being smeared by the time grid |
| $S^*/K$ depending on $K$ | the grid is not scaling with the contract |
| $S^*$ increasing in $\sigma$, or decreasing in $r$ | signs reversed in the operator |

Each of these is checked in `tests/test_boundary.py`.
