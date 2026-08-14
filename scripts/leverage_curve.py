"""How far brute-force leverage gets you, and where it stops working.

    python scripts/leverage_curve.py --panel N

This is the reference every dynamic rule has to beat. It answers three questions
in order: what constant leverage would have maximised compound return, how
confident anyone could possibly be about that number, and whether the
growth-optimal leverage moves enough with observable state to make a dynamic rule
worth building.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from lam.data import panels
from lam.metrics import growth as mg
from lam.report import tables


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="N", choices=["N", "S", "C", "M"])
    ap.add_argument("--spread", type=float, default=0.0050,
                    help="financing spread over the bill rate")
    args = ap.parse_args()

    panel = panels.build(args.panel)
    asset, rf = panel.benchmark, panel.rf
    cost = float(panel.costs[0])
    print(panel)
    print(f"  {panel.description}\n")

    curve = mg.leverage_curve(
        asset, rf,
        levels=np.round(np.arange(0.5, 5.01, 0.25), 4),
        borrow_spread=args.spread, cost_per_unit=cost,
    )
    print("CONSTANT LEVERAGE, DAILY RESET")
    print(tables.fmt(curve))

    best_lev, best_cagr = mg.optimal_static_leverage(
        asset, rf, borrow_spread=args.spread, cost_per_unit=cost
    )
    one = curve.loc[1.0]
    print(
        f"\noptimal constant leverage {best_lev:.2f}x -> CAGR {best_cagr:.2%}"
        f"   (buy and hold {one['cagr']:.2%} at {one['max_dd']:.1%} drawdown)"
    )

    print("\nKELLY OPTIMUM")
    analytic = mg.kelly_analytic(asset, rf, spread=args.spread)
    boot = mg.kelly_bootstrap(asset, rf, spread=args.spread, n=2000)
    print(f"  from sample moments        {analytic:.2f}x")
    print(f"  from the realised path     {best_lev:.2f}x")
    print(
        f"  block bootstrap            p05 {boot['p05']:.2f}x | median "
        f"{boot['p50']:.2f}x | p95 {boot['p95']:.2f}x"
    )
    print(f"  probability L* < 1         {boot['frac_below_1']:.1%}")
    print(
        "  The interval is the point. L* is a ratio of two noisily estimated\n"
        "  moments, and the numerator is the equity risk premium -- the hardest\n"
        "  number in finance to pin down. Betting the point estimate is betting\n"
        "  the whole interval."
    )

    print("\nGROWTH DECOMPOSITION (bp/yr)")
    rows = [mg.growth_decomposition(asset, rf, lev, borrow_spread=args.spread,
                                    cost_per_unit=cost)
            for lev in (1.0, 2.0, 3.0)]
    print(tables.fmt(pd.DataFrame(rows).set_index("leverage")))

    print("\nGROWTH-OPTIMAL LEVERAGE BY STATE (lagged, in sample)")
    print(tables.fmt(mg.regime_kelly(asset, rf, spread=args.spread)))
    print(
        "\n  The spread across states is the case for a dynamic rule. If L* were\n"
        "  constant there would be nothing for trend or volatility regimes to do,\n"
        "  and the right answer would be a fixed multiple of the index."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
