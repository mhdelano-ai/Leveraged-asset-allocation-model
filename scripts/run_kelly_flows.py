"""The Kelly fraction for an account that is being paid into, or drawn down.

    python scripts/run_kelly_flows.py --paths 250

Kelly's fraction is derived for a closed account, where the problem is
scale-invariant and the optimum therefore does not depend on how much money is in
it. External flows are the case almost everyone is actually in, and they break
that invariance -- but only if they are a fixed size. That distinction, not the
direction of the flow, is what moves the answer.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from lam.kelly import ASSETS, PRETTY, build_panel
from lam.kelly.forward import (
    GLOBAL_EQUITY_WEIGHTS,
    build_cma,
    flow_ladder,
    geometric_frontier,
    human_capital_leverage,
    market_inputs,
)

LEVELS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)

SCENARIOS = [
    ("Closed account, 20y", dict(years=20)),
    ("Saving 10%/yr of the opening balance, 20y", dict(years=20, deposit_rate=0.10)),
    ("Saving 20%/yr, 20y", dict(years=20, deposit_rate=0.20)),
    ("Saving 20%/yr, PROPORTIONAL, 20y",
     dict(years=20, deposit_rate=0.20, flows_are_proportional=True)),
    ("Drawing 4%/yr of the opening balance, 30y", dict(years=30, withdrawal_rate=0.04)),
    ("Drawing 6%/yr, 30y", dict(years=30, withdrawal_rate=0.06)),
    ("Drawing 4%/yr of the CURRENT balance, 30y",
     dict(years=30, withdrawal_rate=0.04, flows_are_proportional=True)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=250)
    ap.add_argument("--spread", type=float, default=0.012)
    args = ap.parse_args()

    panel = build_panel("modern")
    cma = build_cma(panel, inputs=market_inputs())
    half = geometric_frontier(
        cma, fractions=np.array([0.5]), financing_spread=args.spread,
        equity_split=GLOBAL_EQUITY_WEIGHTS,
    ).iloc[0]
    holding = np.array([float(half[a]) for a in ASSETS])
    holding = holding / np.abs(holding).sum()
    print("holding (the frontier's half-Kelly mix, levered as shown):")
    print("   " + "  ".join(
        f"{PRETTY[a]} {holding[i]:.0%}" for i, a in enumerate(ASSETS) if holding[i] > 0.005
    ))

    for name, kwargs in SCENARIOS:
        table = flow_ladder(
            panel, cma, holding, levels=LEVELS, paths=args.paths,
            financing_spread=args.spread, **kwargs,
        )
        show = pd.DataFrame(index=table.index)
        show["median wealth"] = table["median_real_wealth"].map(lambda v: f"{v:7.2f}x")
        show["5th pct"] = table["p5_real_wealth"].map(lambda v: f"{v:6.2f}x")
        show["95th pct"] = table["p95_real_wealth"].map(lambda v: f"{v:7.2f}x")
        show["median IRR"] = table["median_irr"].map(lambda v: f"{v * 100:5.2f}%")
        show["median max DD"] = table["median_max_drawdown"].map(lambda v: f"{v * 100:6.1f}%")
        show["P(call)"] = table["p_margin_call"].map(lambda v: f"{v * 100:5.1f}%")
        show["P(ruin)"] = table["p_ruin"].map(lambda v: f"{v * 100:5.1f}%")
        best_wealth = table["median_real_wealth"].idxmax()
        safest = table["p_ruin"].idxmin()
        print("\n" + "=" * 100)
        print(f"{name}   ({args.paths} paths)")
        print("=" * 100)
        print(show.to_string())
        print(
            f"   median wealth is maximised at {best_wealth:.2f}x"
            + (
                f"; ruin is least likely at {safest:.2f}x"
                if table["p_ruin"].max() > 0
                else "; ruin is impossible at every level"
            )
        )

    print("\n" + "=" * 100)
    print("The lifecycle bound: future deposits are a bond you already own")
    print("=" * 100)
    print(
        "   If the Kelly fraction is the right share of TOTAL economic wealth, and total\n"
        "   wealth is the account plus the present value of what you will still pay in,\n"
        "   then the leverage to run on the account alone is L x (1 + PV/account)."
    )
    rows = []
    for years, rate in ((10, 0.20), (20, 0.20), (30, 0.20), (20, 0.40)):
        out = human_capital_leverage(1.58, deposit_rate=rate, years=years, discount_rate=0.047)
        rows.append({
            "horizon": f"{years}y",
            "deposits": f"{rate:.0%}/yr of opening balance",
            "PV of deposits": f"{out['pv_of_deposits']:.2f}x",
            "implied leverage": f"{out['implied_account_leverage']:.2f}x",
        })
    print(pd.DataFrame(rows).set_index("horizon").to_string())
    print(
        "\n   Reg-T caps you at 2x. For a young saver the binding constraint is the broker,\n"
        "   not Kelly -- and the assumption underneath is that a salary is a Treasury bond,\n"
        "   which it is not. Treat this as an upper bound, never a target."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
