# PROJECT_STATUS

Persistent state for this repository. Update after **every** milestone.

**Current position:** Milestone 2 complete. Next: Milestone 3 (Crank–Nicolson + PSOR).

---

## Milestone 1 — European baseline + binomial tree ✅

### Completed
- `src/amopt/black_scholes.py` — closed-form European call/put and the full
  first/second-order Greek set (delta, gamma, vega, theta, rho), fully broadcast
  over all six parameters, with correct `T → 0` and `σ → 0` limits.
- `src/amopt/binomial.py` — CRR lattice from first principles: European and
  American exercise, lattice Greeks, and early-exercise-boundary extraction.
- `src/amopt/config.py` — ten canonical parameter regimes shared by all experiments.
- `src/amopt/plotting.py` — one figure style; fixed method→colour mapping.
- `tests/` — 78 tests passing.
- `experiments/m1_european_baseline.py` — convergence study.

### Important decisions
- **CRR parameterisation** `u = e^{σ√Δt}`, `d = 1/u`, `p = (e^{(r−q)Δt} − d)/(u − d)`.
  Node prices are computed as `S0·exp((2j − i)σ√Δt)` rather than `u^j d^{i−j}` to
  avoid floating-point overflow at large `N`.
- **The lattice raises on `p ∉ [0,1]`** instead of clamping. Clamping would hide
  an arbitrageable lattice and silently bias prices.
- **Lattice Greeks** exploit `ud = 1`: the step-2 nodes contain `S0` exactly, so
  delta/gamma/theta come from the lattice itself at zero extra cost.
- The **averaged lattice** `½(V_N + V_{N+1})` is available but is used only where
  explicitly stated, never silently.

### Numerical results
See `RESULTS.md` §1.

### Known limitations
- Lattice boundary extraction returns `nan` near `t = 0` because the lattice cone
  rooted at `S0` does not reach the exercise region. Structural, not a bug —
  documented in `crr()`. The PDE solver will not share this limitation.
- CRR is `O(N^{-1})` and oscillates with period two in `N`; single-lattice values
  at moderate `N` are not accurate enough to serve as the final reference solution.
  A high-resolution reference is built in Milestone 6.

### Unresolved
- None.

### Next
Superseded — see Milestone 2 below.

---

## Milestone 2 — Mathematical formulation ✅

### Completed
- `docs/01_formulation.md` — risk-neutral valuation, the optimal-stopping
  representation, the Snell envelope and why it forces `𝓛V ≤ 0`, exercise vs
  continuation regions, the variational inequality and its complementarity form,
  terminal and boundary conditions (including why `V(0,t) = K` for the American
  put but `Ke^{−rτ}` for the European), value matching, smooth pasting with an
  argument for *why* `V_S = −1`, and the perpetual closed form.
- `docs/02_crank_nicolson.md` — the full discretisation: grid construction with
  the strike pinned to a node, the tridiagonal coefficients with every factor of
  `ΔS` cancelled, the θ-scheme, the boundary rows at both time levels, the
  discrete LCP, the derivation of PSOR from Gauss–Seidel → SOR → projection, the
  M-matrix / cell-Péclet analysis, red–black ordering, Rannacher start-up, the
  Brennan–Schwartz cross-check, and boundary extraction by interpolating the
  early-exercise gap.
- `src/amopt/perpetual.py` + 12 tests.
- `experiments/m2_analytic_anchors.py`.

### Important decisions
- **Solve in time-to-maturity** `τ = T − t`, so the terminal condition becomes an
  initial condition and the scheme marches forward.
- **Uniform grid in `S`, not in `log S`.** Every power of `ΔS` cancels out of the
  coefficients, `S = 0` and the boundary region are both in the domain, and the
  strike can be pinned exactly to a node (which preserves second-order accuracy
  through the payoff kink).
- **The M-matrix condition is checked, not assumed.** `a_i ≥ 0` requires
  `i ≥ (r−q)/σ²`; for the base case that is `1.25`, so exactly one row (`i = 1`)
  fails. The solver will count violating rows and offer upwinding; the effect is
  to be *measured* in Milestone 3, not asserted negligible.
- **Two exact identities** (`𝓐ₕ1 = −r`, `𝓐ₕS = −qS`) are asserted at assembly.
  Central differences are exact on constants and linears, so these catch sign and
  indexing errors in the coefficients.

### Numerical results
See `RESULTS.md` §2.

### Known limitations
- The maturity-limit study hits the CRR noise floor beyond `T ≈ 50`; the
  perpetual limit cannot be verified more tightly with the lattice alone.

### Unresolved
- None. Two issues found during Milestone 2 were diagnosed and fixed: the `r = 0`
  perpetual degeneracy, and the apparent non-monotonicity of the gap to the
  perpetual value (lattice error, now quantified and reported).

### Next
Milestone 3: implement `src/amopt/crank_nicolson.py` per `docs/02`.
