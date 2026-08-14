"""What the model holds right now.

    python scripts/show_allocation.py --panel M
    python scripts/show_allocation.py --panel L --params docs/fullsample_params.json
    python scripts/show_allocation.py --panel L --history decade

Defaults to the tradable panel and the strict parameter set, which is the
combination someone would actually implement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from lam.alloc.stack import StackParams
from lam.data import panels
from lam.report import allocation, tables
from lam import pipeline

PARAM_FIELDS = [
    "sigma_target", "max_leverage", "vol_halflife", "trend_fast_weight",
    "growth_budget", "dd_budget_margin", "leverage_ratchet_up", "lev_band",
]


def load_params(path: str | None) -> StackParams:
    if not path:
        return StackParams()
    raw = json.loads(Path(path).read_text())
    return StackParams(**{k: float(raw[k]) for k in PARAM_FIELDS if k in raw})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="M", choices=["L", "X", "M"])
    ap.add_argument("--params", default="docs/robust_params.json")
    ap.add_argument("--date", default=None, help="as-of date; defaults to the last bar")
    ap.add_argument("--history", default=None, choices=["QE", "YE", "decade"])
    args = ap.parse_args()

    params_path = args.params if Path(args.params).exists() else None
    if args.params and params_path is None:
        print(f"note: {args.params} not found, using defaults\n")
    params = load_params(params_path)

    panel = panels.build(args.panel)
    out = pipeline.execute(panel, params, constraint_step=21)

    print(allocation.describe(out, panel, date=args.date))

    print("\nLONG-RUN MIX (share of invested capital)")
    shares = allocation.risk_book_shares(out)
    for sleeve, v in shares.items():
        print(f"  {sleeve:<14s} {v:6.1%}")

    if args.history:
        print(f"\nHISTORY ({args.history})")
        hist = allocation.allocation_history(out, args.history)
        if args.history != "decade":
            hist = hist.tail(12)
        print((hist * 100).round(1).to_string())

    s = out.stats
    print(f"\nfor context: CAGR {s['cagr']:.2%} | maxDD {s['max_dd']:.2%} | "
          f"Sharpe {s['sharpe']:.2f} | constraint "
          f"{'PASS' if out.constraint.passed else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
