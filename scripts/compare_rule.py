"""Head-to-head: the four-state leveraged rotation against the fitted model.

    python scripts/compare_rule.py

The four-state rule is a well-known design and a good one. Comparing it fairly
takes more care than running both and reading the CAGR column, because the two
sit at different points on the risk axis: the rule as specified runs 44%
volatility, the fitted model 30%. At those two levels almost any comparison can
be made to come out either way.

So the comparison is made three times: as each is specified, with the rule dialled
down to the model's risk, and with the model dialled up to the rule's. Only the
matched-risk rows are evidence about which set of rules is better.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from lam.alloc.growth import GrowthParams
from lam.alloc.rules import RuleParams, backtest_four_state, state_table, switch_count
from lam.data import panels
from lam.financing.letf import LeveredETF
from lam.metrics.core import ann_vol, cagr, sharpe, worst_rolling
from lam.metrics.drawdown import max_drawdown
from lam.opt import validate
from lam.pipeline import execute_growth
from lam.report import tables


def row(label: str, r: pd.Series, rf: pd.Series) -> dict:
    eq = (1.0 + r).cumprod()
    underwater = float(((eq / eq.cummax() - 1.0) < -0.20).sum() / 252.0)
    return {
        "strategy": label,
        "cagr": cagr(r),
        "vol": ann_vol(r),
        "sharpe": sharpe(r, rf.reindex(r.index).ffill() / 252.0),
        "max_dd": max_drawdown(r),
        "calmar": cagr(r) / abs(max_drawdown(r)) if max_drawdown(r) else np.nan,
        "worst_12m": worst_rolling(r),
        "yrs_20pct_under": underwater,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="N", choices=["N", "C"])
    ap.add_argument("--params", default="docs/growth_params_N.json")
    ap.add_argument("--borrow-spread", type=float, default=0.0175)
    args = ap.parse_args()

    panel = panels.build(args.panel)
    idx, rf, cost = panel.benchmark, panel.rf, float(panel.costs[0])
    raw = json.loads(Path(args.params).read_text())
    fields = GrowthParams.__dataclass_fields__
    base = GrowthParams(**{k: float(v) for k, v in raw.items() if k in fields})

    def rule(**kw):
        return backtest_four_state(idx, rf, RuleParams(**kw), cost_per_unit=cost)[0]

    def model(**kw):
        return execute_growth(panel, replace(base, **kw), borrow_spread=args.borrow_spread).result.returns

    print(panel, "\n")

    # ------------------------------------------------------------ as specified
    print("AS EACH IS SPECIFIED")
    print(tables.fmt(pd.DataFrame([
        row("four-state rule (3x leg)", rule(), rf),
        row("fitted model (Reg-T, 2x cap)", model(max_leverage=2.0), rf),
        row("QQQ buy and hold", idx, rf),
    ]).set_index("strategy")))
    print("  Different risk levels; not yet a comparison.")

    # ----------------------------------------------------------- matched risk
    print("\nAT MATCHED RISK (~30% volatility)")
    print(tables.fmt(pd.DataFrame([
        row("fitted model (Reg-T, 2x cap)", model(max_leverage=2.0), rf),
        row("rule, 2x levered leg", rule(levered_multiple=2.0, up_calm=2.0), rf),
        row("rule improved, 2x leg", rule(levered_multiple=2.0, up_calm=2.0, up_wild=1.0), rf),
    ]).set_index("strategy")))

    print("\nAT MATCHED RISK (~43% volatility)")
    letf_model = execute_growth(
        panel, replace(base, sigma_target=1.10, max_leverage=3.0), vehicle=LeveredETF()
    ).result.returns
    print(tables.fmt(pd.DataFrame([
        row("fitted model via TQQQ, dialled up", letf_model, rf),
        row("four-state rule (3x leg)", rule(), rf),
        row("rule improved (3x leg)", rule(up_wild=1.0), rf),
    ]).set_index("strategy")))

    # ------------------------------------------------------- the vol filter
    print("\nWHAT EACH BRANCH OF THE VOLATILITY FILTER IS WORTH")
    print(tables.fmt(pd.DataFrame([
        row("as specified", rule(), rf),
        row("above SMA + volatile -> 1x, not cash", rule(up_wild=1.0), rf),
        row("above SMA + volatile -> stay 3x", rule(up_wild=3.0), rf),
        row("below SMA + volatile -> 1x, not cash", rule(down_wild=1.0), rf),
    ]).set_index("strategy")))
    print("  The filter earns its keep below the trend and costs money above it.")

    # ------------------------------------------------------------ robustness
    print("\nSENSITIVITY TO THE TWO THRESHOLDS (CAGR % / max drawdown %)")
    grid = []
    for sma in (100, 150, 200, 250, 300):
        line = {"sma_window": sma}
        for vt in (0.20, 0.24, 0.28, 0.32, 0.36):
            r = rule(sma_window=sma, vol_threshold=vt)
            line[f"vol<{vt:.0%}"] = f"{cagr(r) * 100:5.1f} / {max_drawdown(r) * 100:5.0f}"
        grid.append(line)
    print(pd.DataFrame(grid).set_index("sma_window").to_string())

    mid = len(idx) // 2
    print("\nSPLIT SAMPLE - is the threshold choice stable?")
    split = []
    for vt in (0.20, 0.24, 0.28, 0.32, 0.36):
        r = rule(vol_threshold=vt)
        split.append({
            "vol_threshold": vt,
            "first_half_cagr": cagr(r.iloc[:mid]), "first_half_dd": max_drawdown(r.iloc[:mid]),
            "second_half_cagr": cagr(r.iloc[mid:]), "second_half_dd": max_drawdown(r.iloc[mid:]),
        })
    print(tables.fmt(pd.DataFrame(split).set_index("vol_threshold")))

    # --------------------------------------------------------------- crises
    print("\nDRAWDOWN BY EPISODE")
    t = validate.stress_table(rule(), idx)[["strat_dd", "bench_dd"]]
    t = t.rename(columns={"strat_dd": "rule_dd", "bench_dd": "qqq_dd"})
    t["model_dd"] = validate.stress_table(model(max_leverage=2.0), idx)["strat_dd"]
    print(tables.fmt(t))

    _, sig = backtest_four_state(idx, rf, RuleParams(), cost_per_unit=cost)
    print(f"\nstate switches: {switch_count(sig):.1f} a year "
          f"({switch_count(sig) * 2:.0f} trades)")
    print(tables.fmt(state_table(idx, rule(), sig)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
