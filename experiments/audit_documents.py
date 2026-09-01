"""Audit: every number quoted in the documents must be supported by `results/`.

`tests/test_documented_claims.py` checks a hand-picked list of headline numbers
against named CSV columns.  This script is the complementary *sweep*: it extracts
**every** backtick-quoted numeric literal from the documentation and checks that
it matches, to its own printed precision, either

* a value stored in some `results/*.csv` column,
* a ratio of two values from the same column (variance-reduction factors,
  speed-ups, "37x" style claims), or
* a difference of two values from the same column (biases, gaps, deviations).

That covers the derived quantities the targeted tests do not name individually.
It is deliberately a script rather than a test: the "ratio or difference of the
same column" rule is a heuristic, and a heuristic that fails a build on a
false positive is worse than one that prints a list to triage.

Numbers deliberately exempt: 0.5, 1, 2 (used in formulas), integers up to 500
without a decimal point (grid sizes, counts, section numbers), and the random
seed.

**Known limitation -- false negatives.** The pool includes every within-column
ratio and difference, which is a large set, so a *wrong* number can occasionally
match something by coincidence.  During the final audit this sweep passed while
three work-normalised-gain figures were stale by up to 25%, because the pool
happened to contain matching ratios.  The sweep is therefore a net to catch
numbers with no source at all; the targeted assertions in
`tests/test_documented_claims.py` are what actually pin each headline value to
its own column, and timing-derived values are checked there with an explicit
tolerance band because they are not reproducible to three significant figures.

Run:  python experiments/audit_documents.py
Exit code 1 if anything is unsupported.
"""

from __future__ import annotations

import glob
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = [
    "README.md",
    "RESULTS.md",
    "paper/american_put_pde_vs_monte_carlo.md",
    "docs/01_formulation.md",
    "docs/02_crank_nicolson.md",
    "docs/03_exercise_boundary.md",
]
EXEMPT = {0.5, 1.0, 2.0, 20240901.0}
SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
NUMBER = re.compile(
    r"`([−\-]?\d[\d,]*\.?\d*(?:\s*[×x]\s*10⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+|[eE][+\-]?\d+)?)`"
)


def build_pool() -> np.ndarray:
    """Stored values, plus within-column ratios and differences."""
    columns: dict[str, np.ndarray] = {}
    for f in sorted(glob.glob(str(ROOT / "results" / "*.csv"))):
        d = pd.read_csv(f)
        for c in d.columns:
            if d[c].dtype.kind in "fci":
                v = d[c].to_numpy(float)
                columns[f"{pathlib.Path(f).name}:{c}"] = v[np.isfinite(v)]
    if not columns:
        raise SystemExit("results/ is empty -- run the experiments first")

    parts = [np.concatenate(list(columns.values()))]
    for v in columns.values():
        if 1 < v.size <= 60:  # small columns only: the outer products are O(n^2)
            nz = v[np.abs(v) > 1e-12]
            if nz.size > 1:
                r = np.abs(np.outer(nz, 1.0 / nz)).ravel()
                parts.append(r[np.isfinite(r)])
            diff = np.abs(v[:, None] - v[None, :]).ravel()
            parts.append(diff[np.isfinite(diff)])
    pool = np.abs(np.concatenate(parts))
    return pool[np.isfinite(pool)]


def parse(raw: str):
    """Return (magnitude, significant digits) for a quoted literal, or None."""
    s = raw.replace("−", "-").replace(",", "").replace(" ", "")
    s = s.replace("×10⁻", "e-").replace("x10⁻", "e-").replace("×10", "e")
    s = s.translate(SUPERSCRIPTS)
    try:
        x = abs(float(s))
    except ValueError:
        return None
    mantissa = re.split(r"[eE×x]", s)[0]
    if "." not in mantissa and "e" not in s.lower() and x == int(x) and x <= 500:
        return None  # a count, a grid size, or a section number
    sig = len(mantissa.replace("-", "").replace(".", "").lstrip("0")) or 1
    return x, sig


def main() -> int:
    pool = build_pool()
    unmatched: dict[str, set] = {}
    checked = 0
    for doc in DOCS:
        path = ROOT / doc
        if not path.exists():
            continue
        for m in NUMBER.finditer(path.read_text()):
            parsed = parse(m.group(1))
            if parsed is None:
                continue
            x, sig = parsed
            if x in EXEMPT or x < 1e-14:
                continue
            checked += 1
            # half a unit in the last printed significant digit
            tol = 0.5 * x * 10.0 ** (-(sig - 1)) * 1.0001
            if not np.any(np.abs(pool - x) <= tol):
                unmatched.setdefault(doc, set()).add(m.group(1))

    print(f"checked {checked} quoted numbers across {len(DOCS)} documents")
    if not unmatched:
        print("\nUNSUPPORTED CLAIMS: none.")
        print("Every quoted number matches a stored value, a ratio of stored values,")
        print("or a difference of stored values, to its own printed precision.")
        return 0
    print("\nUNSUPPORTED CLAIMS -- each needs a source in results/ or removal:")
    for doc, vals in unmatched.items():
        print(f"\n  {doc}:")
        for v in sorted(vals):
            print(f"      {v}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
