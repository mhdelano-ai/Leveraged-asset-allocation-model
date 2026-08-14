"""Run one backtest and render its tearsheet.

    python scripts/run_backtest.py --panel L --sigma 0.11
"""

from __future__ import annotations

import argparse
import sys

from lam.alloc.stack import StackParams
from lam.data import panels
from lam.metrics.core import cagr
from lam.metrics.drawdown import max_drawdown
from lam.opt import validate
from lam.report import plots, tables
from lam import pipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="L", choices=["L", "X", "M"])
    ap.add_argument("--sigma", type=float, default=0.11)
    ap.add_argument("--max-leverage", type=float, default=3.0)
    ap.add_argument("--dd-margin", type=float, default=0.80)
    ap.add_argument("--params", default=None,
                    help="JSON file of StackParams; overrides the individual flags")
    ap.add_argument("--vehicle", default="none", choices=["none", "letf", "margin", "regt"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    panel = panels.build(args.panel)
    print(panel)
    print(f"  {panel.description}\n")

    vehicle = None
    if args.vehicle == "letf":
        from lam.financing.letf import LeveredETF

        vehicle = LeveredETF()
    elif args.vehicle in {"margin", "regt"}:
        from lam.financing.margin import MarginAccount

        vehicle = MarginAccount(schedule="box_spread" if args.vehicle == "margin" else "reg_t")

    if args.params:
        import json
        from pathlib import Path

        raw = json.loads(Path(args.params).read_text())
        fields = StackParams.__dataclass_fields__
        params = StackParams(**{k: float(v) for k, v in raw.items() if k in fields})
    else:
        params = StackParams(
            sigma_target=args.sigma,
            max_leverage=args.max_leverage,
            dd_budget_margin=args.dd_margin,
        )
    out = pipeline.execute(panel, params, vehicle=vehicle, constraint_step=5)

    print("STRATEGY:", out.headline())
    print("CONSTRAINT:", out.constraint)
    print(f"design drawdown budget: {out.budget:.1%}\n")

    rows = [tables.summary_row("strategy", out.stats, out.constraint)]
    for name, r in pipeline.reference_portfolios(panel).items():
        r = r.dropna()
        rows.append({"strategy": name, "cagr": cagr(r), "max_dd": max_drawdown(r)})
    print(tables.fmt(tables.comparison_table(rows)))

    print("\nSTRESS EPISODES")
    print(tables.fmt(validate.stress_table(out.result.returns, panel.benchmark)))

    print("\nREGIMES")
    print(tables.fmt(validate.regime_table(out.result.returns, panel.benchmark, panel.rf)))

    path = args.out or f"output/tearsheet_panel{args.panel}.png"
    plots.tearsheet(
        out.result.returns,
        panel.benchmark,
        result=out.result,
        references=pipeline.reference_portfolios(panel),
        panel=panel,
        title=f"Leveraged multi-asset strategy - Panel {args.panel}",
        subtitle=(
            f"sigma target {args.sigma:.0%} | max leverage {args.max_leverage:.1f}x | "
            f"vehicle {args.vehicle} | {panel.returns.index[0].date()} to "
            f"{panel.returns.index[-1].date()}"
        ),
        path=path,
    )
    print(f"\ntearsheet -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
