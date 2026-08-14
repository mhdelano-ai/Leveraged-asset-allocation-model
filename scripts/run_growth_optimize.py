"""Search for the return-maximising configuration on a concentrated panel.

    python scripts/run_growth_optimize.py --panel N --trials 2048

One scan answers every drawdown tolerance: the frontier table is derived from the
stored trials rather than re-run, so the exchange rate between drawdown and
compound return comes out of a single pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from lam.alloc.growth import GrowthParams
from lam.data import panels
from lam.opt import growth as og
from lam.report import tables

SHOW = [
    "cagr", "excess_cagr", "max_dd", "sharpe", "calmar", "avg_leverage",
    "half_sample_cagr", "sigma_target", "max_leverage", "trend_floor",
    "stress_cap", "dd_budget", "leverage_ratchet_up", "vol_halflife",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="N", choices=["N", "S", "C", "M"])
    ap.add_argument("--trials", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dd-ceiling", type=float, default=0.50,
                    help="drawdown tolerance for the headline pick, as a positive depth")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-params", default=None)
    args = ap.parse_args()

    panel = panels.build(args.panel)
    print(panel)
    print(f"  {panel.description}\n")

    results = og.scan(panel, n=args.trials, seed=args.seed)

    out_path = Path(args.out or f"output/growth_scan_panel{args.panel}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out_path)
    print(f"\n{len(results)} trials -> {out_path}")
    print(f"ruined: {int(results['ruined'].sum())}")

    print("\nRETURN / DRAWDOWN FRONTIER (best CAGR at each drawdown tolerance)")
    print(tables.fmt(og.frontier(results)))

    best, pool = og.select(results, dd_ceiling=args.dd_ceiling)
    if best is None:
        print(f"\nNo configuration survives a {args.dd_ceiling:.0%} drawdown ceiling.")
        return 0

    print(f"\nTOP 10 WITHIN A {args.dd_ceiling:.0%} DRAWDOWN CEILING")
    print(tables.fmt(pool[SHOW].head(10)))

    chosen = GrowthParams(
        **{k: float(best[k]) for k in GrowthParams.__dataclass_fields__ if k in best}
    )
    print("\nCHOSEN")
    for k, v in asdict(chosen).items():
        print(f"  {k:22s} {v:.4f}")

    if args.save_params:
        Path(args.save_params).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_params).write_text(json.dumps(asdict(chosen), indent=2))
        print(f"\nparameters -> {args.save_params}")

    print("\nLAYER ABLATION")
    print(tables.fmt(og.ablation(panel, chosen)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
