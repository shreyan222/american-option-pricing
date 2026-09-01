# CLAUDE.md — working notes for this repository

## What this project is

A from-first-principles numerical study of the **American put** under Black–Scholes,
comparing two structurally different solvers:

1. **Crank–Nicolson finite differences + PSOR** — solves the variational
   inequality / linear complementarity problem directly on a fixed grid.
2. **Longstaff–Schwartz least-squares Monte Carlo** — approximates the optimal
   stopping rule by regressing continuation values on simulated paths.

A **CRR binomial lattice** is the independent third method used as a cross-check.

The deliverable is a *comparison* — accuracy, convergence order, runtime,
estimator variance, stability, robustness — not a single price.

## Hard rules

- **No option-pricing libraries.** NumPy/SciPy/pandas/matplotlib are used for
  generic linear algebra, statistics, dataframes and plotting only. QuantLib and
  equivalents are forbidden. Every pricing algorithm is written here.
- **No fabricated numbers.** Every figure in `README.md`, `RESULTS.md` and
  `paper/` must be traceable to a CSV in `results/` produced by a script in
  `experiments/`. If a number is not in `results/`, it does not go in the docs.
  Two mechanisms enforce this, and both must pass:
  - `pytest tests/test_documented_claims.py` — pins each headline number to a
    named CSV column, to the digits as printed.
  - `python experiments/audit_documents.py` — sweeps *every* quoted number in
    the docs and reports any with no source. Exit code 1 if anything is
    unsupported.
- **Wall-clock numbers are quoted to two significant figures, with the caveat.**
  Re-running moves absolute timings by 5–25%. Ratios and fitted power-law slopes
  are stable and are what the conclusions rest on.
- **Do not weaken a failing test to make it pass.** Diagnose it. If a test was
  genuinely wrong, fix the test and say why in the commit message.
- **Do not delete inconvenient experiments.** Unexpected results get
  investigated and documented, not hidden.
- **Never force-push.** `git push origin main` only.

## Layout

```
src/amopt/        library code (the algorithms)
tests/            pytest suite; identity/limit/convergence tests, not snapshots
experiments/      scripts that produce results/ and figures/; each is runnable standalone
results/          CSV outputs — the single source of truth for every claim
figures/          PNG outputs
paper/            the research report
docs/             mathematical derivations
PROJECT_STATUS.md persistent state across sessions — read this first
RESULTS.md        experimentally verified numbers only
```

## Conventions

- Every experiment script is `experiments/mN_*.py`, runnable as
  `python experiments/mN_*.py`, writes `results/mN_*.csv` and `figures/mN_*.png`,
  and prints a short console summary.
- Monte Carlo work takes an explicit `seed`; results in `results/` are reproducible.
- Plot styling comes from `amopt.plotting.use_style()`. Method colours are fixed
  in `amopt.plotting.METHOD_COLORS` — the same method is the same colour in every
  figure.
- Parameter regimes live in `amopt.config.REGIMES` so all experiments price the
  same contracts.

## After changing an experiment

1. Rerun the affected `experiments/mN_*.py`.
2. Run `pytest -q` — `test_documented_claims.py` will fail if a documented
   number moved.
3. Update `RESULTS.md`, `README.md` and `paper/` to the new values.
4. Run `python experiments/audit_documents.py` and confirm it exits 0.
5. Regenerate any figure whose *plotting code* changed, not just its data —
   the final audit found a figure that had been stale for four milestones.

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Numerical conventions worth remembering

- `theta` is ∂V/∂t (calendar time), per year. `vega` is per unit vol, not per point.
- The CRR lattice raises rather than clamping when the risk-neutral probability
  leaves `[0, 1]`; an arbitrageable lattice is a bug in the caller's `N`, not
  something to paper over.
- The lattice can only resolve the exercise boundary inside its cone
  `S0·exp(±i·σ√Δt)`; near `t = 0` it returns `nan`. The PDE solver does not have
  this limitation — that asymmetry is itself a reportable finding.
