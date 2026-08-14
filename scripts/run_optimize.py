"""Constrained parameter search, and the strictness frontier.

    python scripts/run_optimize.py --panel L --trials 512

The scan stores each trial's worst rolling-window excess, so the frontier across
different tolerance levels is derived from a single scan rather than re-run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from lam.data import panels
from lam.opt import search
from lam.report import tables

SHOW = [
    "cagr", "max_dd", "sharpe", "calmar", "avg_leverage", "worst_excess",
    "sigma_target", "max_leverage", "dd_budget_margin", "vol_halflife",
    "trend_fast_weight", "growth_budget",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="L", choices=["L", "X", "M"])
    ap.add_argument("--trials", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tolerance", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    panel = panels.build(args.panel)
    print(panel, "\n")

    results = search.scan(
        panel, n=args.trials, seed=args.seed,
        tolerance=args.tolerance, constraint_step=21,
    )

    out_path = Path(args.out or f"output/scan_panel{args.panel}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out_path)
    print(f"\n{len(results)} trials -> {out_path}")

    print("\nSTRICTNESS FRONTIER (rolling-window tolerance relaxed; full-sample stays strict)")
    print(tables.fmt(tables.frontier_table(results)))

    best, frontier = search.select(results)
    if best is None:
        print("\nNo feasible parameter set at the strict tolerance.")
        return 0

    print(f"\nTOP 10 FEASIBLE (tolerance {args.tolerance:.1%})")
    print(tables.fmt(frontier[SHOW].head(10)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
