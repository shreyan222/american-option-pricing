# 1. The American put as an optimal-stopping and free-boundary problem

This note derives, from the risk-neutral valuation principle, the variational
inequality that `src/amopt/crank_nicolson.py` discretises. It is written to be
checkable: every sign, every inequality direction and every boundary condition is
justified rather than asserted.

Notation: $S$ is the underlying price, $t \in [0,T]$ calendar time,
$\tau = T - t$ time to maturity, $K$ the strike, $r$ the risk-free rate, $q$ a
continuous dividend yield, $\sigma$ the volatility. The put payoff is
$g(S) = (K-S)^+$.

---

## 1.1 Risk-neutral valuation

The market is Black–Scholes: a bank account $dB_t = r B_t\,dt$ and a stock

$$
dS_t = (\mu - q) S_t\,dt + \sigma S_t\,dW_t .
$$

The market is complete and free of arbitrage, so there is a unique equivalent
martingale measure $\mathbb{Q}$ under which the *total-return* discounted stock
$e^{-(r-q)t} S_t$ is a martingale, i.e.

$$
dS_t = (r-q) S_t\,dt + \sigma S_t\,dW_t^{\mathbb{Q}},
\qquad
S_T = S_t \exp\!\Big[\big(r - q - \tfrac{1}{2}\sigma^2\big)(T-t) + \sigma \sqrt{T-t}\, Z\Big].
$$

For a **European** claim with payoff $h(S_T)$ the price is
$e^{-r(T-t)}\mathbb{E}^{\mathbb{Q}}[h(S_T)\mid \mathcal{F}_t]$, which for
$h = g$ integrates to the Black–Scholes put formula implemented in
`amopt.black_scholes`.

An **American** claim can be exercised at any stopping time. Its holder chooses
the exercise rule, so the value is the supremum over admissible stopping times:

$$
\boxed{\;
V(S,t) \;=\; \sup_{\tau \in \mathcal{T}_{t,T}}
\mathbb{E}^{\mathbb{Q}}\!\left[\, e^{-r(\tau - t)}\, g(S_\tau) \;\middle|\; S_t = S \right]
\;}
\tag{1.1}
$$

where $\mathcal{T}_{t,T}$ is the set of stopping times with values in $[t,T]$.
This is the object every method in this repository approximates. The three
solvers differ only in *how* they attack (1.1):

| Method | Attack |
|---|---|
| CRR lattice | discretise the state space and time, solve (1.1) by backward induction on a tree |
| Crank–Nicolson + PSOR | convert (1.1) into a PDE variational inequality and solve it on a grid |
| Longstaff–Schwartz | keep (1.1) probabilistic; estimate the *continuation value* by regression on simulated paths |

---

## 1.2 The Snell envelope and why the value is a supermartingale

Write $Y_t = e^{-rt} g(S_t)$ for the discounted payoff process and
$U_t = e^{-rt} V(S_t, t)$ for the discounted value. Standard optimal-stopping
theory says $U$ is the **Snell envelope** of $Y$: the smallest càdlàg
supermartingale dominating $Y$. Two consequences drive everything below.

**(a) $U$ dominates $Y$.** Taking $\tau = t$ in (1.1),

$$
V(S,t) \;\ge\; g(S) \qquad \text{for all } (S,t).
\tag{1.2}
$$

The American value never falls below the intrinsic value — otherwise one would
buy the option and exercise immediately for a riskless profit.

**(b) $U$ is a supermartingale.** Waiting cannot be worth more than the current
value: $\mathbb{E}^{\mathbb{Q}}[U_{t'} \mid \mathcal{F}_t] \le U_t$ for $t' \ge t$.

Assume $V$ is smooth enough to apply Itô. With the Black–Scholes generator

$$
\mathcal{L}V \;=\;
\frac{\partial V}{\partial t}
+ \tfrac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
+ (r - q) S \frac{\partial V}{\partial S}
- r V ,
\tag{1.3}
$$

Itô's lemma gives $dU_t = e^{-rt}\big(\mathcal{L}V\big)\,dt + e^{-rt}\sigma S_t V_S\,dW_t^{\mathbb{Q}}$.
A supermartingale has non-positive drift, so

$$
\mathcal{L}V \;\le\; 0 \qquad \text{for all } (S,t).
\tag{1.4}
$$

(With $q = 0$, (1.3) is exactly the operator in the project brief,
$V_t + \tfrac12\sigma^2S^2V_{SS} + rSV_S - rV$.)

---

## 1.3 Exercise and continuation regions

Partition the domain by whether the constraint (1.2) is tight:

$$
\mathcal{E} = \{(S,t) : V(S,t) = g(S)\} \quad\text{(exercise region)},
\qquad
\mathcal{C} = \{(S,t) : V(S,t) > g(S)\} \quad\text{(continuation region)} .
$$

The optimal stopping time is the first entry into the exercise region,
$\tau^* = \inf\{u \ge t : (S_u, u) \in \mathcal{E}\}$.

On $\mathcal{C}$ it is strictly suboptimal to exercise, so the Snell envelope is
a *martingale* there, not merely a supermartingale, and the drift vanishes:

$$
\mathcal{L}V = 0 \quad \text{on } \mathcal{C}.
\tag{1.5}
$$

This is the Black–Scholes PDE, holding only where the option is alive.
On $\mathcal{E}$, $V = g$ and in general $\mathcal{L}V < 0$: for the put, on
$\{S < K\}$ where $g = K - S$,

$$
\mathcal{L}g = 0 + 0 + (r-q)S(-1) - r(K - S) = -rK + qS \;<\; 0
\quad\text{whenever } S < \frac{rK}{q}
$$

(and always, when $q = 0$, since then $\mathcal{L}g = -rK < 0$ for $r>0$). The
economic reading is direct: holding an exercised put means holding cash $K$ that
earns $r$; not exercising forgoes the interest $rK$ and saves the dividend $qS$.
**When $r = 0$ and $q = 0$, $\mathcal{L}g = 0$ and there is no incentive to
exercise early at all** — the American put then coincides with the European put,
which is asserted as a test in `tests/test_binomial.py`.

---

## 1.4 The variational inequality and the linear complementarity problem

Collecting (1.2), (1.4), (1.5): everywhere on $(0,\infty) \times [0,T)$,

$$
\boxed{\;
\mathcal{L}V \le 0,
\qquad
V - g \ge 0,
\qquad
\big(\mathcal{L}V\big)\big(V - g\big) = 0 .
\;}
\tag{1.6}
$$

The third line is the **complementarity condition**: at every point at least one
of the two inequalities is an equality. It encodes the dichotomy of §1.3 without
requiring us to know where the regions are. Equivalently,

$$
\max\big\{ \mathcal{L}V,\; g - V \big\} = 0 ,
\tag{1.7}
$$

which is the form most convenient to discretise: it is checked in the exercise
region ($\mathcal{L}V \le 0$, $g - V = 0$) and in the continuation region
($\mathcal{L}V = 0$, $g - V < 0$) alike.

This is a **free-boundary problem**: the region on which the PDE holds is itself
unknown. The numerical significance is that (1.6) is a *linear complementarity
problem* (LCP) once discretised, and LCPs are solved by projected iterative
methods — hence PSOR.

---

## 1.5 Terminal and boundary conditions

**Terminal condition.** At maturity the option is worth its payoff:

$$
V(S,T) = (K - S)^+ .
\tag{1.8}
$$

**Lower boundary, $S \to 0$.** Zero is an absorbing state of geometric Brownian
motion: if $S_t = 0$ then $S_u = 0$ for all $u \ge t$. The payoff is then $K$ with
certainty and it is optimal to take it immediately, so

$$
V(0,t) = K .
\tag{1.9}
$$

This differs from the **European** condition $V^{\text{eu}}(0,t) = Ke^{-r(T-t)}$,
and the difference is exactly the early-exercise premium at $S=0$. Getting this
wrong is the single most common sign error in American PDE code, so the solver
takes the boundary condition from the exercise style rather than hard-coding it.

**Upper boundary, $S \to \infty$.** A deep out-of-the-money put is worthless and
so is the right to exercise it:

$$
\lim_{S \to \infty} V(S,t) = 0 .
\tag{1.10}
$$

Numerically the domain is truncated at a finite $S_{\max}$ and (1.10) is imposed
there as a Dirichlet condition. The truncation error decays like the probability
of reaching $S_{\max}$, i.e. exponentially in $\log(S_{\max}/S_0)/(\sigma\sqrt{T})$;
$S_{\max} = 4K$ is used by default and the sensitivity is measured in Milestone 6
rather than assumed.

---

## 1.6 The early-exercise boundary, value matching and smooth pasting

For the put, $V(\cdot, t) - g$ is non-decreasing in $S$ where it is positive, and
the exercise region at each time is a *lower* interval. Hence there is a single
critical price

$$
S^*(t) = \sup\{ S > 0 : V(S,t) = K - S \},
$$

with $\mathcal{E} = \{S \le S^*(t)\}$ and $\mathcal{C} = \{S > S^*(t)\}$. The
curve $t \mapsto S^*(t)$ is the **early-exercise boundary**. It is *not* given in
advance; recovering it is the content of Milestone 7.

Two conditions hold along the boundary.

**Value matching.** $V$ is continuous across the boundary:

$$
V\big(S^*(t), t\big) = K - S^*(t).
\tag{1.11}
$$

**Smooth pasting (high contact).** The delta is also continuous:

$$
\frac{\partial V}{\partial S}\Big(S^*(t), t\Big) = -1 .
\tag{1.12}
$$

*Why (1.12) must hold.* Suppose $V_S(S^{*+}) > -1$, so the value function meets
the payoff line with a kink opening upward. Consider exercising instead at
$S^* - \epsilon$. Because $V \ge g$ with equality at $S^*$, a first-order
expansion shows the holder can strictly improve by shifting the boundary, so
$S^*$ was not optimal. Conversely $V_S(S^{*+}) < -1$ would violate $V \ge g$ just
to the right of $S^*$. Only $V_S = -1$ is consistent with optimality. Smooth
pasting is what selects the boundary out of the one-parameter family of curves
satisfying value matching alone.

**Properties of $S^*(t)$ used later as validation.**

1. $S^*(t) \le K$ for all $t$, and more precisely $S^*(t) \le \min\{K,\, rK/q\}$
   (from §1.3: exercise requires $\mathcal{L}g < 0$).
2. $S^*$ is non-decreasing in $t$: less remaining optionality makes exercise more
   attractive.
3. $S^*(T^-) = \min\{K,\, rK/q\}$, which is $K$ when $q = 0$.
4. As $T - t \to \infty$, $S^*(t)$ decreases to the **perpetual** boundary
   $S_\infty$ derived next.

### The perpetual American put — an exact limiting case

With infinite maturity the problem is time-homogeneous and (1.5) becomes an ODE,
$\tfrac12\sigma^2S^2V'' + (r-q)SV' - rV = 0$, whose solutions are $S^\beta$ with

$$
\tfrac12 \sigma^2 \beta(\beta - 1) + (r-q)\beta - r = 0
\;\Longrightarrow\;
\beta_\pm = \frac{-\left(r - q - \tfrac12\sigma^2\right) \pm \sqrt{\left(r-q-\tfrac12\sigma^2\right)^2 + 2\sigma^2 r}}{\sigma^2}.
$$

Boundedness as $S \to \infty$ forces the negative root $\beta_-$, so
$V(S) = A S^{\beta_-}$ on $S > S_\infty$. Imposing value matching (1.11) and
smooth pasting (1.12) at $S_\infty$:

$$
A S_\infty^{\beta_-} = K - S_\infty, \qquad \beta_- A S_\infty^{\beta_- - 1} = -1
\;\Longrightarrow\;
\boxed{\;S_\infty = K\,\frac{\beta_-}{\beta_- - 1},\qquad
V(S) = \big(K - S_\infty\big)\left(\frac{S}{S_\infty}\right)^{\beta_-}\;}
\tag{1.13}
$$

for $S > S_\infty$, and $V(S) = K - S$ below. For $q=0$ the quadratic factors as
$(\beta + \gamma)(\beta - 1) = 0$ with $\gamma = 2r/\sigma^2$, giving
$\beta_- = -\gamma$ and $S_\infty = K\gamma/(1+\gamma)$.

This closed form is **not** a substitute for the finite-maturity solver; it is an
independent analytic anchor. It supplies two hard checks used in the test suite:

- $V^{\text{perp}}(S) \ge V^{\text{amer}}(S,0;T)$ for every $T$, with convergence
  from below as $T \to \infty$;
- the numerically extracted $S^*(0)$ must lie in $(S_\infty, K)$ and must decrease
  towards $S_\infty$ as $T$ grows.

It is implemented in `src/amopt/perpetual.py`.

---

## 1.7 Change of variable used by the solver

The solver integrates forward in **time to maturity** $\tau = T - t$, because the
data (1.8) is given at $t = T$. Since $\partial_t = -\partial_\tau$, writing
$v(S,\tau) = V(S, T-\tau)$ turns (1.7) into an initial-value complementarity
problem:

$$
\max\left\{\;
-\frac{\partial v}{\partial \tau}
+ \tfrac{1}{2}\sigma^2 S^2 \frac{\partial^2 v}{\partial S^2}
+ (r-q) S \frac{\partial v}{\partial S}
- r v ,
\;\; g - v \;\right\} = 0,
\qquad v(S,0) = (K-S)^+ .
\tag{1.14}
$$

Equivalently, defining the **spatial** operator

$$
\mathcal{A}v \;=\; \tfrac{1}{2}\sigma^2 S^2 v_{SS} + (r-q) S v_S - r v ,
\tag{1.15}
$$

the continuation-region equation is the forward parabolic equation
$v_\tau = \mathcal{A}v$, subject to $v \ge g$ and the complementarity condition.
Note the sign: marching *forward* in $\tau$ is marching *backward* in $t$, which
is the direction in which the terminal condition propagates. Section 2 discretises
(1.14)–(1.15).
