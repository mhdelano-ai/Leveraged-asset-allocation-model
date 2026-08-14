"""Leveraged ETFs versus portfolio margin / box spreads, head to head.

The same allocation logic is run through both vehicles so the comparison
measures the wrapper, not two different strategies.

The interesting question is not which financing spread is cheaper. It is whether
the LETF path's *inability to express the target allocation* -- there is no usable
3x commodity, REIT or IG credit fund -- costs more than the margin path's exposure
to forced liquidation. Both effects are modelled, and the table reports them
separately so the answer is visible rather than asserted.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from lam.alloc.stack import StackParams
from lam.data import panels
from lam.financing.letf import LeveredETF
from lam.financing.margin import MarginAccount
from lam.report import tables
from lam import pipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="M", choices=["L", "X", "M"])
    ap.add_argument("--sigma", type=float, default=0.11)
    ap.add_argument("--max-leverage", type=float, default=3.0)
    args = ap.parse_args()

    panel = panels.build(args.panel)
    print(panel)
    print(f"  {panel.description}\n")

    params = StackParams(sigma_target=args.sigma, max_leverage=args.max_leverage)
    letf = LeveredETF()

    print("LETF PRODUCT COVERAGE (the structural limitation)")
    coverage = letf.expressiveness_report(panel.returns.columns)
    print(tables.fmt(coverage))
    unlevered = coverage[~coverage["levered"]].index.tolist()
    print(f"\n  sleeves with no leveraged product: {', '.join(unlevered) or 'none'}")
    print("  these must be held unlevered, consuming capital 1:1 and crowding out"
          "\n  leverage elsewhere -- the LETF book cannot hold the target mix.\n")

    vehicles = {
        "no vehicle (ideal)": None,
        "LETF": letf,
        "box spread": MarginAccount(schedule="box_spread"),
        "portfolio margin": MarginAccount(schedule="portfolio_margin"),
        "Reg-T 2x": MarginAccount(schedule="reg_t"),
    }

    rows = []
    for name, vehicle in vehicles.items():
        out = pipeline.execute(panel, params, vehicle=vehicle, constraint_step=5)
        s = out.stats
        rows.append(
            {
                "strategy": name,
                "cagr": s["cagr"],
                "vol": s["vol"],
                "sharpe": s["sharpe"],
                "max_dd": s["max_dd"],
                "calmar": s["calmar"],
                "avg_lev": s["avg_leverage"],
                "fin_bps": s["financing_drag_bps"],
                "tc_bps": s["transaction_drag_bps"],
                "calls": s["margin_calls"],
                "ruined": s["ruined"],
                "passes": out.constraint.passed,
            }
        )

    print("HEAD TO HEAD")
    print(tables.fmt(tables.comparison_table(rows)))
    print("\n  calls = forced deleveraging events. An LETF cannot margin-call the")
    print("  holder, which is its one genuine structural advantage.")
    if args.panel != "M":
        print("\n  NOTE: portfolio margin did not exist for retail before 2007, so its")
        print("  pre-2007 results here are counterfactual. Reg-T 2x is the honest")
        print("  pre-2007 assumption.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
