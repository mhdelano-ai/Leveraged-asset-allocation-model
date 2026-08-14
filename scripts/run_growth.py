"""Run the return-maximising strategy on a concentrated panel and report it.

    python scripts/run_growth.py --panel N --params docs/growth_params_N.json

Everything is reported against the same hurdle: simply owning the asset. A
levered, trend-gated, regime-aware book that trails buy-and-hold has done a great
deal of work to destroy money, and the tables are laid out so that is impossible
to miss.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from lam import pipeline
from lam.alloc import regime
from lam.alloc.growth import GrowthParams
from lam.data import panels
from lam.metrics import growth as mg
from lam.metrics.core import cagr, summary
from lam.metrics.drawdown import max_drawdown
from lam.opt import growth as og
from lam.opt import validate
from lam.report import plots, tables


def _load_params(path: str | None) -> GrowthParams:
    if not path:
        return GrowthParams()
    data = json.loads(Path(path).read_text())
    fields = GrowthParams.__dataclass_fields__
    return GrowthParams(**{k: float(v) for k, v in data.items() if k in fields})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="N", choices=["N", "S", "C", "M"])
    ap.add_argument("--params", default=None, help="JSON file of GrowthParams")
    ap.add_argument("--vehicle", default="none", choices=["none", "letf", "margin", "regt"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    panel = panels.build(args.panel)
    params = _load_params(args.params)
    print(panel)
    print(f"  {panel.description}\n")
    print("PARAMETERS")
    for k, v in asdict(params).items():
        print(f"  {k:22s} {v:.4f}")

    vehicle = None
    if args.vehicle == "letf":
        from lam.financing.letf import LeveredETF

        vehicle = LeveredETF()
    elif args.vehicle in {"margin", "regt"}:
        from lam.financing.margin import MarginAccount

        vehicle = MarginAccount(schedule="box_spread" if args.vehicle == "margin" else "reg_t")

    out = pipeline.execute_growth(panel, params, vehicle=vehicle)
    asset = panel.benchmark

    print("\nSTRATEGY:", out.headline())
    print(f"beats buy-and-hold: {'YES' if out.beats_benchmark else 'NO'}")

    # ---------------------------------------------------------------- headline
    rows = [
        tables.summary_row("strategy", out.stats),
        tables.summary_row(f"{panel.tickers[0]} buy and hold", out.benchmark_stats),
    ]
    rf = panel.rf
    cost = float(panel.costs[0])
    for lev in (2.0, 3.0):
        r = mg.constant_leverage_returns(asset, rf, lev, cost_per_unit=cost)
        rows.append(tables.summary_row(f"static {lev:.0f}x daily reset",
                                       summary(r, benchmark=asset, rf_daily=rf / 252.0)))

    matched = mg.vol_matched_static(out.result.returns, asset, rf, cost_per_unit=cost)
    rows.append(
        tables.summary_row(
            f"static {matched['matched_leverage']:.2f}x (vol-matched)",
            summary(matched["returns"], benchmark=asset, rf_daily=rf / 252.0),
        )
    )
    print("\nVERSUS THE ALTERNATIVES")
    print(tables.fmt(pd.DataFrame(rows).set_index("strategy")))
    print(
        f"\nAgainst constant leverage carrying the same {matched['target_vol']:.1%} "
        f"volatility: {matched['edge_bps']:+.0f}bp/yr and "
        f"{matched['dd_saved_pp']:+.1f}pp of drawdown. Everything else in this "
        "report is context; this line is what the timing rules earned."
    )

    # ------------------------------------------------------------- the layers
    print("\nLAYER ABLATION (what each piece is worth)")
    print(tables.fmt(og.ablation(panel, params)))

    # --------------------------------------------------------------- exposure
    lev_series = out.result.leverage
    labels = regime.regime_labels(out.signals.vol_forecast).reindex(lev_series.index)
    exposure = pd.DataFrame(
        {
            "share of days": labels.value_counts(normalize=True),
            "avg leverage": lev_series.groupby(labels).mean(),
            "asset return": asset.groupby(labels).mean() * 252,
            "strategy return": out.result.returns.groupby(labels).mean() * 252,
        }
    ).dropna()
    print("\nEXPOSURE BY VOLATILITY REGIME")
    print(tables.fmt(exposure))

    print(
        f"\nleverage: mean {lev_series.mean():.2f}x, median {lev_series.median():.2f}x, "
        f"max {lev_series.max():.2f}x | above 1x on {(lev_series > 1).mean():.1%} of days, "
        f"flat on {(lev_series < 0.01).mean():.1%}"
    )
    print(f"turnover {out.stats['ann_turnover']:.1f}x/yr | "
          f"financing {out.stats['financing_drag_bps']:.0f}bp/yr | "
          f"trading {out.stats['transaction_drag_bps']:.0f}bp/yr")

    # ---------------------------------------------------------------- stress
    print("\nSTRESS EPISODES vs THE UNLEVERED INDEX")
    print(tables.fmt(validate.stress_table(out.result.returns, asset)))
    print(
        "  A levered book is deeper than an unlevered index in any short shock, so "
        "read\n  the failures here as a description of the risk being run, not as a "
        "defect."
    )

    print(f"\nSTRESS EPISODES vs STATIC {matched['matched_leverage']:.2f}x AT THE SAME VOLATILITY")
    print(tables.fmt(validate.stress_table(out.result.returns, matched["returns"])))
    print(
        "  This is the like-for-like comparison. Where the rules earn their keep is "
        "slow\n  bear markets, which they see coming; sharp one-week shocks they do not."
    )

    print("\nREGIMES")
    print(tables.fmt(validate.regime_table(out.result.returns, asset, panel.rf)))

    print("\nCOST SENSITIVITY")
    cost_rows = []
    for m in (1.0, 2.0, 3.0):
        alt = pipeline.execute_growth(
            panel, params, cost_per_unit=panel.costs * m, borrow_spread=0.0050 * m
        )
        cost_rows.append({"cost_multiple": m, "cagr": alt.stats["cagr"],
                          "max_dd": alt.stats["max_dd"], "sharpe": alt.stats["sharpe"]})
    print(tables.fmt(pd.DataFrame(cost_rows).set_index("cost_multiple")))

    # --------------------------------------------------------------- tearsheet
    path = args.out or f"output/growth_tearsheet_panel{args.panel}.png"
    references = {
        f"{panel.tickers[0]} buy and hold": asset,
        "static 2x": mg.constant_leverage_returns(asset, rf, 2.0, cost_per_unit=float(panel.costs[0])),
    }
    plots.tearsheet(
        out.result.returns,
        asset,
        result=out.result,
        references=references,
        panel=panel,
        title=f"Return-maximising leveraged strategy - Panel {args.panel}",
        subtitle=(
            f"sigma target {params.sigma_target:.0%} | max leverage "
            f"{params.max_leverage:.1f}x | vehicle {args.vehicle} | "
            f"{panel.returns.index[0].date()} to {panel.returns.index[-1].date()}"
        ),
        path=path,
    )
    print(f"\ntearsheet -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
