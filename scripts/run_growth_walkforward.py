"""Out-of-sample validation for the return-maximising model.

    python scripts/run_growth_walkforward.py --panel N

The stitched out-of-sample curve is the headline result. The in-sample optimum
never is: compound return as a function of leverage has a flat top and a cliff
just past it, so the in-sample argmax is exactly the point most likely to be on
the wrong side of the cliff next decade.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lam.alloc.growth import GrowthParams
from lam.data import panels
from lam.metrics import growth as mg
from lam.metrics.core import ann_vol, cagr, sharpe
from lam.metrics.drawdown import max_drawdown
from lam.opt import growth as og
from lam.opt import validate
from lam.report import plots, tables


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="N", choices=["N", "S", "C"])
    ap.add_argument("--scan", default=None, help="parquet of scan trials to draw candidates from")
    ap.add_argument("--candidates", type=int, default=40)
    ap.add_argument("--train-years", type=int, default=15)
    ap.add_argument("--step-years", type=int, default=2)
    ap.add_argument("--dd-ceiling", type=float, default=0.50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    panel = panels.build(args.panel)
    print(panel, "\n")

    scan_path = Path(args.scan or f"output/growth_scan_panel{args.panel}.parquet")
    if not scan_path.exists():
        print(f"no scan at {scan_path}; run scripts/run_growth_optimize.py first")
        return 1
    scan = pd.read_parquet(scan_path)

    # Candidates are the best surviving trials by compound return. They embed
    # full-sample information -- see the docstring of walk_forward_growth -- so
    # this measures parameter stability, not discovery.
    _, pool = og.select(scan, dd_ceiling=args.dd_ceiling)
    fields = GrowthParams.__dataclass_fields__
    candidates = [
        GrowthParams(**{k: float(row[k]) for k in fields if k in row})
        for _, row in pool.head(args.candidates).iterrows()
    ]
    print(f"{len(candidates)} candidates from {scan_path} "
          f"(drawdown ceiling {args.dd_ceiling:.0%})\n")

    oos, folds = validate.walk_forward_growth(
        panel, candidates,
        initial_train_years=args.train_years,
        step_years=args.step_years,
        dd_ceiling=args.dd_ceiling,
    )
    if oos.empty:
        print("no out-of-sample segments produced")
        return 1

    print("FOLDS")
    print(tables.fmt(folds.set_index("train_end")))

    asset = panel.benchmark.reindex(oos.index).dropna()
    rf_daily = panel.rf.reindex(oos.index).ffill() / 252.0

    # The comparison that decides whether the timing is worth anything: constant
    # leverage carrying exactly the same realised volatility as the strategy.
    matched = mg.vol_matched_static(
        oos, asset, panel.rf, cost_per_unit=float(panel.costs[0])
    )

    print("\nOUT OF SAMPLE, STITCHED")
    def _row(name, r):
        return {
            "series": name,
            "cagr": cagr(r),
            "vol": ann_vol(r),
            "sharpe": sharpe(r, rf_daily.reindex(r.index).ffill()),
            "max_dd": max_drawdown(r),
            "terminal": float(np.prod(1 + r)),
        }

    comparison = pd.DataFrame(
        [
            _row("strategy (out of sample)", oos),
            _row(f"{panel.tickers[0]} buy and hold", asset),
            _row(f"static {matched['matched_leverage']:.2f}x (vol-matched)", matched["returns"]),
        ]
    ).set_index("series")
    print(tables.fmt(comparison))
    print(
        f"\nagainst constant leverage carrying the same {matched['target_vol']:.1%} volatility:\n"
        f"  {matched['edge_bps']:+.0f}bp/yr of compound return and "
        f"{matched['dd_saved_pp']:+.1f}pp of drawdown.\n"
        "  This is the number that separates timing from simply running more risk."
    )

    print(
        f"\nfolds beating the asset: {int((folds['oos_cagr'] > folds['asset_cagr']).sum())}"
        f" / {len(folds)}"
    )
    print(f"worst fold: {folds['oos_cagr'].min():.2%} "
          f"({folds.loc[folds['oos_cagr'].idxmin(), 'train_end']} onward)")

    print("\nWHERE THE EDGE OVER VOL-MATCHED STATIC LEVERAGE WAS EARNED")
    attribution = validate.edge_attribution(oos, matched["returns"])
    print(tables.fmt(attribution[attribution["days"] > 0]))

    for label, windows in [
        ("2000-02 dotcom", [("2000-03-01", "2002-12-31")]),
        ("2007-09 GFC", [("2007-10-01", "2009-03-31")]),
        ("both", [("2000-03-01", "2002-12-31"), ("2007-10-01", "2009-03-31")]),
    ]:
        ex = validate.edge_excluding(oos, matched["returns"], windows)
        if ex:
            print(
                f"  edge excluding {label:16s} {ex['edge_bps']:+8.0f}bp/yr "
                f"({ex['days_dropped']} days dropped)"
            )

    dsr = validate.deflated_sharpe(sharpe(oos, rf_daily), len(scan), len(oos))
    print(
        f"\ndeflated Sharpe: observed {dsr['observed']:.2f} against an "
        f"expected-maximum-under-the-null of {dsr['expected_max_null']:.2f} "
        f"over {dsr['n_trials']} trials"
    )

    print("\nPARAMETER STABILITY ACROSS FOLDS")
    print(tables.fmt(folds[["sigma_target", "max_leverage", "trend_floor",
                            "stress_cap", "dd_budget"]].describe().T))

    path = args.out or f"output/growth_walkforward_panel{args.panel}.png"
    plots.tearsheet(
        oos, asset,
        title=f"Out-of-sample return-maximising strategy - Panel {args.panel}",
        subtitle=(
            f"anchored walk-forward, {args.train_years}y initial training, "
            f"re-selected every {args.step_years}y, {len(folds)} folds"
        ),
        path=path,
    )
    print(f"\ntearsheet -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
