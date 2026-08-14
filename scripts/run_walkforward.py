"""Out-of-sample validation: walk-forward, permutation, cost sensitivity.

    python scripts/run_walkforward.py --panel L --scan output/scan_panelL.parquet

The stitched out-of-sample curve is the headline result of the project. An
in-sample optimum that satisfies a drawdown constraint proves nothing -- the
degenerate solution is zero leverage, and any search with enough parameters will
find something that dodges the particular crashes in the sample.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from lam.alloc.stack import StackParams
from lam.data import panels
from lam.metrics.constraint import evaluate
from lam.metrics.core import summary
from lam.opt import search, validate
from lam.report import plots, tables
from lam import pipeline

PARAM_FIELDS = [
    "sigma_target", "max_leverage", "vol_halflife", "trend_fast_weight",
    "growth_budget", "dd_budget_margin", "leverage_ratchet_up", "lev_band",
]


def candidates_from_scan(path: Path, top: int = 24) -> list[StackParams]:
    """Take a spread of scanned points as the walk-forward's candidate set."""
    df = pd.read_parquet(path)
    df = df.dropna(subset=["cagr"]).sort_values("cagr", ascending=False)
    feasible = df[df["feasible"]] if "feasible" in df else df
    pool = feasible if len(feasible) >= top else df
    return [StackParams(**{k: float(r[k]) for k in PARAM_FIELDS}) for _, r in pool.head(top).iterrows()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="L", choices=["L", "X", "M"])
    ap.add_argument("--scan", default=None)
    ap.add_argument("--tolerance", type=float, default=0.0)
    ap.add_argument("--permutations", type=int, default=0)
    args = ap.parse_args()

    panel = panels.build(args.panel)
    print(panel, "\n")

    scan_path = Path(args.scan or f"output/scan_panel{args.panel}.parquet")
    if scan_path.exists():
        candidates = candidates_from_scan(scan_path)
        print(f"{len(candidates)} candidates from {scan_path}")
    else:
        candidates = search.sample_params(48, seed=3)
        print(f"no scan found; using {len(candidates)} fresh Sobol draws")

    print("\nWALK-FORWARD (anchored, 20y initial train, 2y steps)")
    oos, folds = validate.walk_forward(
        panel, candidates, tolerance=args.tolerance, constraint_step=21
    )
    if oos.empty:
        print("no out-of-sample segments produced")
        return 1
    print(tables.fmt(folds))

    oos_stats = summary(oos, benchmark=panel.benchmark, rf_daily=panel.rf / 252.0)
    oos_constraint = evaluate(oos, panel.benchmark.reindex(oos.index).dropna(), step=5)
    print(f"\nSTITCHED OUT-OF-SAMPLE  {oos.index[0].date()} -> {oos.index[-1].date()}")
    print(f"  CAGR {oos_stats['cagr']:.2%} | vol {oos_stats['vol']:.1%} | "
          f"Sharpe {oos_stats['sharpe']:.2f} | maxDD {oos_stats['max_dd']:.2%}")
    print(f"  {oos_constraint}")

    bench = panel.benchmark.reindex(oos.index).dropna()
    print(f"  S&P 500 over the same dates: CAGR "
          f"{summary(bench)['cagr']:.2%}, maxDD {summary(bench)['max_dd']:.2%}")

    best = candidates[0]
    print("\nCOST SENSITIVITY (best in-sample candidate)")
    print(tables.fmt(validate.cost_sensitivity(panel, best)))

    if args.permutations:
        print(f"\nPERMUTATION TEST ({args.permutations} shuffles)")
        perm = validate.permutation_test(panel, best, n=args.permutations)
        print(f"  observed CAGR {perm['observed_cagr']:.2%} | null mean "
              f"{perm['null_mean']:.2%} | null p95 {perm['null_p95']:.2%} | "
              f"percentile {perm['percentile']:.1%}")

    dsr = validate.deflated_sharpe(oos_stats["sharpe"], len(candidates) * 20, len(oos))
    print(f"\nDEFLATED SHARPE: observed {dsr['observed']:.2f} vs expected max under "
          f"null {dsr['expected_max_null']:.2f} (excess {dsr['excess_over_null']:+.2f})")

    path = f"output/walkforward_panel{args.panel}.png"
    plots.tearsheet(
        oos, bench, panel=panel,
        title=f"Out-of-sample (walk-forward) - Panel {args.panel}",
        subtitle="anchored 20y train, 2y steps; parameters chosen on training data only",
        path=path,
    )
    print(f"\ntearsheet -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
