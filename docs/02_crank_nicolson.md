# 2. Crank–Nicolson discretisation of the American-put LCP

This note derives the linear algebra that `src/amopt/crank_nicolson.py`
implements. It carries through the indexing explicitly, because an off-by-one in
the boundary rows is silent — the code still runs, and the price is merely wrong.

We discretise (1.14): $v_\tau = \mathcal{A}v$ in the continuation region,
$v \ge g$ everywhere, $v(S,0) = g(S)$.

---

## 2.1 Grids

**Space.** Truncate to $[0, S_{\max}]$ and use a uniform grid

$$
S_i = i\,\Delta S, \qquad i = 0,1,\dots,M, \qquad \Delta S = S_{\max}/M .
$$

$S_{\max}$ is chosen as a multiple of $K$ (default $4K$) and $M$ is adjusted so
that **the strike lands exactly on a grid node**. This matters: the payoff has a
kink at $S=K$, and a kink that falls between nodes is smeared by interpolation,
degrading the observed convergence order from 2 towards 1. The solver enforces
this and reports the node index of $K$.

**Time.** $\tau_n = n\,\Delta\tau$, $n = 0,\dots,N$, $\Delta\tau = T/N$, marching
from $\tau_0 = 0$ (maturity) to $\tau_N = T$ (today).

Unknowns are the interior values $v^n_i$, $i = 1,\dots,M-1$; $v^n_0$ and $v^n_M$
are fixed by the boundary conditions.

---

## 2.2 The discrete spatial operator

Replace derivatives at the interior node $i$ by central differences:

$$
v_S \approx \frac{v_{i+1} - v_{i-1}}{2\Delta S} + O(\Delta S^2),
\qquad
v_{SS} \approx \frac{v_{i+1} - 2v_i + v_{i-1}}{\Delta S^2} + O(\Delta S^2).
$$

Substituting into (1.15) with $S_i = i\Delta S$:

$$
(\mathcal{A}_h v)_i
= \tfrac12 \sigma^2 (i\Delta S)^2 \frac{v_{i+1} - 2v_i + v_{i-1}}{\Delta S^2}
+ (r-q)(i\Delta S) \frac{v_{i+1} - v_{i-1}}{2\Delta S}
- r v_i .
$$

**Every factor of $\Delta S$ cancels.** Collecting terms,

$$
\boxed{\;
(\mathcal{A}_h v)_i = a_i v_{i-1} + b_i v_i + c_i v_{i+1},
\qquad
\begin{aligned}
a_i &= \tfrac12\sigma^2 i^2 - \tfrac12 (r-q)\, i, \\
b_i &= -\sigma^2 i^2 - r, \\
c_i &= \tfrac12\sigma^2 i^2 + \tfrac12 (r-q)\, i .
\end{aligned}
\;}
\tag{2.1}
$$

Two structural checks worth performing on any implementation of (2.1):

* **Row-sum identity.** $a_i + b_i + c_i = -r$. Applying $\mathcal{A}_h$ to the
  constant vector $\mathbf{1}$ must return $-r\mathbf{1}$, which is exactly
  $\mathcal{A}\,1 = -r$. The solver asserts this.
* **Linear-function identity.** $a_i (i-1) + b_i i + c_i (i+1) = (r-q)i - r i$,
  i.e. $\mathcal{A}_h S = (r-q)S - rS = -qS$, matching $\mathcal{A}S = -qS$
  exactly. Central differences are exact on linear functions, so this identity is
  a genuine test of the coefficient algebra rather than a discretisation
  approximation. The solver asserts this too.

These two identities catch essentially every sign and indexing error in (2.1).

---

## 2.3 The $\theta$-scheme and Crank–Nicolson

Averaging the explicit and implicit Euler steps with weight $\theta$:

$$
\frac{v^{n+1}_i - v^n_i}{\Delta \tau}
= \theta\,(\mathcal{A}_h v^{n+1})_i + (1-\theta)\,(\mathcal{A}_h v^{n})_i .
$$

$\theta = 0$ is explicit (conditionally stable, needs $\Delta\tau = O(\Delta S^2)$),
$\theta = 1$ fully implicit (unconditionally stable, first order in $\Delta\tau$),
and $\theta = \tfrac12$ is **Crank–Nicolson**: unconditionally stable and
*second* order in $\Delta\tau$, because the scheme is symmetric about
$\tau_{n+1/2}$ and the leading odd truncation term cancels. The truncation error
is $O(\Delta\tau^2 + \Delta S^2)$.

Rearranging with all $\tau_{n+1}$ terms on the left, and writing
$\alpha_i = \theta\Delta\tau\, a_i$, $\beta_i = \theta \Delta\tau\, b_i$,
$\gamma_i = \theta\Delta\tau\, c_i$ and
$\tilde\alpha_i = (1-\theta)\Delta\tau\,a_i$, etc.:

$$
-\alpha_i v^{n+1}_{i-1} + (1-\beta_i) v^{n+1}_i - \gamma_i v^{n+1}_{i+1}
\;=\;
\tilde\alpha_i v^{n}_{i-1} + (1+\tilde\beta_i) v^{n}_i + \tilde\gamma_i v^{n}_{i+1} .
\tag{2.2}
$$

In matrix form, with $A = I - \theta \Delta\tau\, \mathcal{A}_h$ and
$B = I + (1-\theta)\Delta\tau\, \mathcal{A}_h$ restricted to interior rows,

$$
A\,v^{n+1} = B\,v^{n} + d^{n} ,
\tag{2.3}
$$

$A$ and $B$ tridiagonal of size $(M-1)$, **time-independent** — so $A$ is
assembled once, not once per step.

---

## 2.4 Boundary rows — where the indexing bites

Row $i=1$ of (2.2) references $v_0$, and row $i = M-1$ references $v_M$. Both are
known, so they move to the right-hand side. Writing $\ell(\tau)$ and $u(\tau)$
for the Dirichlet data at $S=0$ and $S=S_{\max}$:

$$
d^n_1 = \tilde\alpha_1\, \ell(\tau_n) + \alpha_1\, \ell(\tau_{n+1}),
\qquad
d^n_{M-1} = \tilde\gamma_{M-1}\, u(\tau_n) + \gamma_{M-1}\, u(\tau_{n+1}),
$$

and $d^n_i = 0$ otherwise. Note that **both** time levels appear: the $\theta$-scheme
evaluates the boundary at $\tau_n$ *and* $\tau_{n+1}$. Using only one level is a
common bug; it silently drops the scheme to first order near the boundary.

The Dirichlet data depends on the exercise style (§1.5):

| | $\ell(\tau) = v(0,\tau)$ | $u(\tau) = v(S_{\max},\tau)$ |
|---|---|---|
| European put | $K e^{-r\tau}$ | $0$ |
| American put | $K$ | $0$ |

---

## 2.5 The discrete LCP and PSOR

Applying (1.6) at the discrete level, the step from $\tau_n$ to $\tau_{n+1}$ is
not a linear solve but a **linear complementarity problem**: find $v^{n+1}$ with

$$
\boxed{\;
A v^{n+1} \ge b^n, \qquad
v^{n+1} \ge g, \qquad
\big(A v^{n+1} - b^n\big)^{\!\top}\big(v^{n+1} - g\big) = 0,
\;}
\qquad b^n := B v^n + d^n .
\tag{2.4}
$$

### Deriving PSOR

Start from the linear system. The Gauss–Seidel sweep for $Av = b$ updates node
$i$ using already-updated values below it and old values above:

$$
v_i^{\text{GS}} = \frac{1}{A_{ii}}\Big( b_i - A_{i,i-1} v^{(k+1)}_{i-1} - A_{i,i+1} v^{(k)}_{i+1} \Big).
$$

**Successive over-relaxation** extrapolates past the Gauss–Seidel value by a
factor $\omega$,
$v^{(k+1)}_i = v^{(k)}_i + \omega\big(v_i^{\text{GS}} - v^{(k)}_i\big)$, which for
$1 < \omega < 2$ accelerates convergence markedly. **Projected** SOR then imposes
the constraint by truncating each update onto the feasible set:

$$
\boxed{\;
v^{(k+1)}_i = \max\Big\{\, g_i,\;\;
v^{(k)}_i + \omega\Big[\tfrac{1}{A_{ii}}\big( b_i - A_{i,i-1} v^{(k+1)}_{i-1} - A_{i,i+1} v^{(k)}_{i+1}\big) - v^{(k)}_i \Big] \Big\}
\;}
\tag{2.5}
$$

Iterate until $\max_i |v^{(k+1)}_i - v^{(k)}_i| < \varepsilon$. Projecting *inside*
the sweep — rather than solving the linear system and clipping afterwards — is
what makes (2.5) converge to the LCP solution rather than to
$\max(A^{-1}b, g)$, which is a different and wrong object.

The projection is what makes the free boundary emerge: the set of nodes where the
$\max$ is attained by $g_i$ *is* the discrete exercise region, and its upper edge
is the discrete $S^*(\tau_{n+1})$. Nothing about the boundary was specified in
advance.

### When is convergence guaranteed?

Cryer's theorem gives convergence of PSOR for $\omega \in (0,2)$ when $A$ is
symmetric positive definite. Our $A$ is **not symmetric** — the convection term
$(r-q)Sv_S$ is not self-adjoint. The relevant sufficient condition here is that
$A$ be an **M-matrix**: $A_{ii} > 0$, $A_{ij} \le 0$ for $i \ne j$, and $A$
irreducibly diagonally dominant. From (2.2), $A_{i,i-1} = -\theta\Delta\tau\,a_i$
and $A_{i,i+1} = -\theta\Delta\tau\,c_i$, so we need $a_i \ge 0$ and $c_i \ge 0$:

$$
a_i \ge 0 \iff i \ge \frac{r-q}{\sigma^2},
\qquad
c_i \ge 0 \iff i \ge -\frac{r-q}{\sigma^2} .
\tag{2.6}
$$

$c_i \ge 0$ always holds for $r \ge q$. The condition on $a_i$ is a **cell Péclet
condition**: central differencing of a convection term is only monotone when the
cell Péclet number $\text{Pe}_i = (r-q)S_i \Delta S / (\sigma^2 S_i^2) = (r-q)/(\sigma^2 i)$
is at most $1$. For the base case $r=0.05$, $q=0$, $\sigma=0.2$ the threshold is
$(r-q)/\sigma^2 = 1.25$, so **exactly one row, $i=1$, violates it**.

We do not sweep this under the rug. The solver:

1. **counts** the violating rows and exposes the count as
   `CNResult.n_non_mmatrix_rows`;
2. offers `upwind=True`, which replaces the central difference by an upwind
   difference in exactly those rows, restoring $a_i = \tfrac12\sigma^2 i^2 \ge 0$
   at the cost of first-order accuracy *there only*:
   $$a_i = \tfrac12\sigma^2 i^2, \quad b_i = -\sigma^2 i^2 - (r-q) i - r, \quad c_i = \tfrac12\sigma^2 i^2 + (r-q) i \qquad \text{(for } r > q);$$
3. and `tests/test_crank_nicolson.py` measures the price difference between the
   two variants rather than assuming it is negligible.

The affected node sits at $S = \Delta S$, deep inside the exercise region where
$v = K - S$ is pinned by the constraint, which is why the effect is expected to
be small — but "expected" is not "measured".

### Red–black ordering

The sweep (2.5) is sequential: $v^{(k+1)}_{i-1}$ must be known before node $i$.
A Python-level loop over $M$ nodes $\times$ iterations $\times$ time steps is
prohibitively slow. We therefore use **red–black (odd–even) ordering**: because
$A$ is tridiagonal, an odd-indexed node's neighbours are all even and vice versa,
so all odd nodes can be updated simultaneously, then all even nodes. Each half
sweep is a single vectorised NumPy expression.

This is not an approximation. A tridiagonal matrix is *consistently ordered* in
Young's sense, and red–black is a consistent ordering, so red–black SOR has the
same spectral radius and the same optimal $\omega$ as the lexicographic sweep.
The solver also ships a slow, transparent lexicographic implementation
(`psor_lexicographic`) and the test suite asserts the two agree to solver
tolerance — which is the check that the vectorised version is faithful.

---

## 2.6 Crank–Nicolson and the payoff kink: Rannacher start-up

Crank–Nicolson is A-stable but **not L-stable**: its amplification factor
$(1 + \tfrac12 \Delta\tau \lambda)/(1 - \tfrac12\Delta\tau\lambda)$ tends to
$-1$, not $0$, as $\lambda \to -\infty$. High-frequency components of the initial
data are therefore not damped, only sign-flipped each step. The put payoff has a
kink at $S = K$, which is exactly such high-frequency data, and the result is
spurious oscillation in the numerical gamma near the strike that decays only
slowly.

The standard fix is **Rannacher start-up**: run the first few steps fully
implicitly ($\theta = 1$, which *is* L-stable and annihilates high frequencies)
and switch to $\theta = \tfrac12$ thereafter. Two fully implicit steps suffice to
restore clean second-order behaviour. The solver takes `rannacher_steps`
(default 2), and Milestone 6 measures the observed convergence order with and
without it rather than taking the textbook claim on trust.

---

## 2.7 Brennan–Schwartz: an independent exact LCP solve

PSOR is iterative, so its answer depends on `omega`, `tol` and `max_iter`. To
check it we also implement the **Brennan–Schwartz** algorithm, which solves (2.4)
*exactly* in $O(M)$ operations — no iteration, no tolerance — under the
assumption that the exercise region is a single interval $\{i \le k\}$, which
§1.6 establishes for the put.

The algorithm is a UL factorisation ordered *from the continuation side*:
eliminate the super-diagonal from $i = M-1$ downward,

$$
d_{M-1} = A_{M-1,M-1},\quad y_{M-1} = b_{M-1}, \qquad
\begin{cases}
m_i = A_{i,i+1}/d_{i+1} \\
d_i = A_{ii} - m_i A_{i+1,i} \\
y_i = b_i - m_i y_{i+1}
\end{cases}
\quad i = M-2,\dots,1,
$$

then substitute forward from $i=1$, projecting as we go:

$$
v_1 = \max\{ y_1/d_1,\; g_1\}, \qquad
v_i = \max\Big\{ \frac{y_i - A_{i,i-1} v_{i-1}}{d_i},\; g_i \Big\},\quad i = 2,\dots,M-1 .
$$

Because the elimination is ordered so that the constrained nodes are visited
*last*, the projection never invalidates an earlier row. $A$ does not change with
time, so $d_i$ and $m_i$ are factorised once.

If PSOR and Brennan–Schwartz disagree by more than the PSOR tolerance, one of
them is wrong. That comparison is a test, not a comment.

---

## 2.8 Extracting the free boundary

At each time level the solver records the discrete exercise set
$\mathcal{E}_h^n = \{ i : v^n_i \le g_i + \delta \}$ for a small $\delta$ tied to
the solver tolerance. For the put this set is $\{0,\dots,k_n\}$ and the raw
boundary estimate is $S^*_h(\tau_n) = S_{k_n}$, accurate to one grid spacing.

$O(\Delta S)$ resolution is coarse. We refine it by **linear interpolation of the
early-exercise gap** $\phi_i = v^n_i - g_i$, which is $0$ on the exercise side and
grows smoothly on the continuation side: the boundary is located at the root of
$\phi$, estimated as

$$
S^*(\tau_n) \approx S_{k_n} + \Delta S \cdot \frac{\phi_{k_n}}{\phi_{k_n} - \phi_{k_n+1}} ,
$$

Both the raw and the interpolated boundary are returned so the improvement can be
measured. Because the grid is fixed in $S$ and covers $[0, S_{\max}]$ at every
time level, the PDE solver resolves $S^*(\tau)$ over the **whole** time axis —
unlike the lattice, whose cone rooted at $S_0$ misses the boundary near $t=0$
(see `crr()`).
