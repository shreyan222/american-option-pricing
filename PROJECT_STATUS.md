# PROJECT_STATUS

Persistent state for this repository. Update after **every** milestone.

**Current position:** Milestone 1 complete. Next: Milestone 2 (mathematical formulation).

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
Milestone 2: write the optimal-stopping / variational-inequality derivation and
the Crank–Nicolson discretisation in `docs/`.
