"""Cross-checks the in-sample result cannot provide.

    python scripts/run_growth_robustness.py

Four questions, in increasing order of how uncomfortable the answers are:

1. **Does it transfer?** Parameters fitted on the Nasdaq-100 applied to the S&P
   500 and back. A rule that only works on the series it was fitted to is a
   description of that series.
2. **Does it survive an era nobody alive has traded?** 1929-32, as a labelled
   counterfactual.
3. **Does it survive the 1970s?** The Nasdaq Composite reaches 1973-74, fourteen
   years before the Nasdaq-100 begins.
4. **Can it actually be implemented?** Through leveraged ETFs and through a
   margin account, with real financing costs and real margin calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lam import pipeline
from lam.alloc.growth import GrowthParams
from lam.data import panels
from lam.metrics import growth as mg
from lam.opt import validate
from lam.report import tables


def _load(path: str) -> GrowthParams:
    data = json.loads(Path(path).read_text())
    fields = GrowthParams.__dataclass_fields__
    return GrowthParams(**{k: float(v) for k, v in data.items() if k in fields})


def _row(label: str, panel, params) -> dict:
    out = pipeline.execute_growth(panel, params)
    matched = mg.vol_matched_static(
        out.result.returns, panel.benchmark, panel.rf, cost_per_unit=float(panel.costs[0])
    )
    return {
        "case": label,
        "cagr": out.stats["cagr"],
        "asset_cagr": out.benchmark_stats["cagr"],
        "max_dd": out.stats["max_dd"],
        "asset_dd": out.benchmark_stats["max_dd"],
        "sharpe": out.stats["sharpe"],
        "avg_leverage": out.stats["avg_leverage"],
        "vs_matched_bps": matched["edge_bps"],
        "beats_asset": out.beats_benchmark,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-n", default="output/growth_params_N.json")
    ap.add_argument("--params-s", default="output/growth_params_S.json")
    args = ap.parse_args()

    pn = _load(args.params_n)
    ps = _load(args.params_s)
    panel_n = panels.build("N")
    panel_s = panels.build("S")

    # ------------------------------------------------------------- transfer
    print("=" * 78)
    print("1. CROSS-ASSET TRANSFER")
    print("=" * 78)
    rows = [
        _row("Nasdaq params on Nasdaq (fitted)", panel_n, pn),
        _row("S&P params on Nasdaq (transferred)", panel_n, ps),
        _row("S&P params on S&P (fitted)", panel_s, ps),
        _row("Nasdaq params on S&P (transferred)", panel_s, pn),
    ]
    print(tables.fmt(pd.DataFrame(rows).set_index("case")))
    print(
        "\n  Transferred parameters losing only a little to fitted ones is the "
        "single\n  cheapest evidence that the rules describe leveraged equity rather "
        "than one\n  index's history."
    )

    # --------------------------------------------------------------- 1929
    print("\n" + "=" * 78)
    print("2. THE 1929-32 COUNTERFACTUAL")
    print("=" * 78)
    scenario_rows = []
    for label, params in [("Nasdaq params", pn), ("S&P params", ps)]:
        for rate in (0.02, 0.05):
            s = validate.deep_history_scenario(params, assumed_rf=rate)
            scenario_rows.append(
                {
                    "case": f"{label} @ {rate:.0%} financing",
                    "cagr": s["strategy_cagr"],
                    "index_cagr": s["index_cagr"],
                    "max_dd": s["strategy_dd"],
                    "index_dd": s["index_dd"],
                    "avg_leverage": s["avg_leverage"],
                    "ruined": s["ruined"],
                }
            )
    print(tables.fmt(pd.DataFrame(scenario_rows).set_index("case")))
    print(
        "\n  Prices and dividends are real; the financing rate is assumed flat and "
        "the\n  result is a scenario, not a backtest. Read the drawdown and the ruin "
        "flag,\n  not the return."
    )

    # ---------------------------------------------------------- composite
    print("\n" + "=" * 78)
    print("3. THE 1970s, VIA THE NASDAQ COMPOSITE")
    print("=" * 78)
    try:
        panel_c = panels.build("C")
        early = panel_c.returns.index < pd.Timestamp("1985-10-02")
        rows = [_row("Composite, full history 1971+", panel_c, pn)]
        out = pipeline.execute_growth(panel_c, pn)
        seg = out.result.returns[early]
        bench = panel_c.benchmark[early]
        from lam.metrics.core import cagr
        from lam.metrics.drawdown import max_drawdown

        rows.append(
            {
                "case": "Composite, 1971-1985 only",
                "cagr": cagr(seg),
                "asset_cagr": cagr(bench),
                "max_dd": max_drawdown(seg),
                "asset_dd": max_drawdown(bench),
                "sharpe": np.nan,
                "avg_leverage": float(out.result.leverage[early].mean()),
                "vs_matched_bps": np.nan,
                "beats_asset": cagr(seg) > cagr(bench),
            }
        )
        print(tables.fmt(pd.DataFrame(rows).set_index("case")))
        print("\nSTRESS, 1973-74")
        print(tables.fmt(validate.stress_table(out.result.returns, panel_c.benchmark).head(3)))
    except Exception as exc:  # noqa: BLE001 - ONEQ may not be cached
        print(f"  unavailable: {type(exc).__name__}: {exc}")

    # ----------------------------------------------------------- vehicles
    print("\n" + "=" * 78)
    print("4. IMPLEMENTATION VEHICLES (Nasdaq-100)")
    print("=" * 78)
    from lam.financing.letf import LeveredETF
    from lam.financing.margin import MarginAccount

    vehicles = {
        "idealised (no vehicle)": None,
        "leveraged ETF (TQQQ)": LeveredETF(),
        "portfolio margin + box spread": MarginAccount(schedule="box_spread"),
        "Reg-T margin": MarginAccount(schedule="reg_t"),
    }
    rows = []
    for label, vehicle in vehicles.items():
        out = pipeline.execute_growth(panel_n, pn, vehicle=vehicle)
        rows.append(
            {
                "vehicle": label,
                "cagr": out.stats["cagr"],
                "max_dd": out.stats["max_dd"],
                "sharpe": out.stats["sharpe"],
                "avg_leverage": out.stats["avg_leverage"],
                "margin_calls": out.stats["margin_calls"],
                "ruined": out.stats["ruined"],
            }
        )
    print(tables.fmt(pd.DataFrame(rows).set_index("vehicle")))

    print("\nLETF CALIBRATION ON NASDAQ FUNDS")
    from lam.financing import calibrate

    fits = pd.DataFrame([calibrate.fit_spread(f) for f in ("TQQQ", "QLD")]).set_index("fund")
    print(tables.fmt(fits[["multiplier", "implied_spread", "wrapper_cost_ann",
                           "wrapper_vs_ter", "n_days", "start"]], decimals=3))
    checks = pd.DataFrame(
        [calibrate.validate(f, float(fits["implied_spread"].mean())) for f in ("TQQQ", "QLD")]
    ).set_index("fund")
    print(tables.fmt(checks[["spread_used", "corr", "cagr_sim", "cagr_actual",
                             "cagr_diff_bps", "te_ann"]], decimals=3))
    print(
        "\n  TQQQ and QLD price the same index at different multipliers from "
        "separate\n  fits. Agreement between them is evidence the functional form is "
        "right; it is\n  not evidence about the split between the expense ratio and "
        "the swap spread,\n  which the data cannot separate."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
