"""What the return-maximising model says to hold today, in shares.

    python scripts/show_growth_position.py --equity 25000 --account regt
    python scripts/show_growth_position.py --equity 25000 --account regt --letf

The backtest answers "how did it do". This answers "what do I place", which is
the only question that matters once the research is finished, and it is the step
where a strategy usually goes wrong: the model wants a *notional exposure*, and a
brokerage account takes *share counts under a margin rule*.

Two account types are modelled because they are genuinely different instruments:

``regt``
    A standard retail margin account. Leverage is capped at 2x by Regulation T
    and financing is retail-priced. This is what Robinhood Gold, and most retail
    margin, actually is.
``portfolio``
    Portfolio margin, roughly 6:1 on a concentrated equity book. Not available
    at every broker, and generally requiring $125k+ of equity.

The ``--letf`` flag prices the same exposure through a 3x fund instead of
borrowing, which needs no margin agreement at all and cannot be margin-called.
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
from lam.data import panels, yahoo

ACCOUNTS = {
    "regt": {"cap": 2.0, "maintenance": 0.30, "label": "Reg-T margin (Robinhood Gold, most retail)"},
    "portfolio": {"cap": 6.0, "maintenance": 0.15, "label": "portfolio margin"},
    "cash": {"cap": 1.0, "maintenance": 1.00, "label": "cash account, no borrowing"},
}


def _load(path: str | None) -> GrowthParams:
    if not path:
        return GrowthParams()
    raw = json.loads(Path(path).read_text())
    fields = GrowthParams.__dataclass_fields__
    return GrowthParams(**{k: float(v) for k, v in raw.items() if k in fields})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="N", choices=["N", "S", "C"])
    ap.add_argument("--params", default="docs/growth_params_N.json")
    ap.add_argument("--equity", type=float, default=10_000.0)
    ap.add_argument("--account", default="regt", choices=list(ACCOUNTS))
    ap.add_argument("--letf", action="store_true",
                    help="express exposure through a 3x fund instead of borrowing")
    ap.add_argument("--letf-ticker", default="TQQQ")
    ap.add_argument("--base-ticker", default=None)
    args = ap.parse_args()

    panel = panels.build(args.panel)
    params = _load(args.params)
    spec = ACCOUNTS[args.account]

    # The account's leverage cap is a hard constraint on the instrument, not a
    # preference, so it is applied to the parameters rather than to the output.
    capped = min(params.max_leverage, spec["cap"])
    if capped < params.max_leverage:
        params = GrowthParams(**{**params.__dict__, "max_leverage": capped})

    out = pipeline.execute_growth(panel, params)
    sig = out.signals
    when = out.result.leverage.index[-1]

    held = float(out.result.leverage.iloc[-1])
    target = float(sig.target_leverage.iloc[-1])
    gate = float(sig.trend_gate.iloc[-1])
    stress = float(sig.stress_multiplier.iloc[-1])
    vol = float(sig.vol_forecast.iloc[-1])

    base_ticker = args.base_ticker or panel.tickers[0]

    print(f"POSITION - Panel {panel.name} @ {when.date()}")
    print(f"  account:  {spec['label']}")
    print(f"  equity:   ${args.equity:,.0f}\n")

    print("  SIGNAL")
    print(f"    volatility forecast     {vol:6.1%}   (risk target {params.sigma_target:.1%})")
    print(f"    volatility-implied      {params.sigma_target / vol:6.2f}x")
    print(f"    trend gate              {gate:6.2f}   {'ON' if gate > 0.95 else 'REDUCED' if gate > 0.05 else 'OFF'}")
    print(f"    volatility regime       {stress:6.2f}")
    print(f"    -> target leverage      {target:6.2f}x   (account cap {spec['cap']:.1f}x)")
    print(f"    currently held          {held:6.2f}x")

    gap = abs(target - held)
    fire = gap > params.lev_band
    print(f"\n  ACTION: {'TRADE' if fire else 'HOLD'}  "
          f"(|target - held| = {gap:.2f}x, band {params.lev_band:.2f}x)")

    notional = target * args.equity
    print("\n  ORDER")
    try:
        px = float(yahoo.prices(base_ticker if not args.letf else args.letf_ticker,
                                field="adjclose").iloc[-1])
    except Exception:  # noqa: BLE001 - price lookup is a convenience, not the answer
        px = float("nan")

    if args.letf:
        # A 3x fund delivers 3 units of exposure per unit of capital, so the
        # capital required is a third of the notional -- and it cannot exceed
        # the account's equity, which is what caps this route at 3x.
        capital = min(notional / 3.0, args.equity)
        print(f"    {args.letf_ticker:6s}  ${capital:>12,.0f} of capital "
              f"= {capital * 3.0 / args.equity:.2f}x notional")
        if not np.isnan(px):
            print(f"            {capital / px:>12,.2f} shares at ${px:,.2f}")
        print(f"    cash    ${args.equity - capital:>12,.0f}")
        print("\n    No borrowing, no margin agreement, no margin call. The financing")
        print("    lives inside the fund and is charged on 2 units of notional for")
        print("    every 1 unit of capital -- whether or not you wanted that much.")
    else:
        borrowed = max(0.0, notional - args.equity)
        print(f"    {base_ticker:6s}  ${notional:>12,.0f} of exposure = {target:.2f}x")
        if not np.isnan(px):
            print(f"            {notional / px:>12,.2f} shares at ${px:,.2f}")
        print(f"    borrowed ${borrowed:>11,.0f}")
        if borrowed > 0:
            # Distance to a maintenance call, expressed as the drop in the
            # underlying that would trigger it. This is the number that decides
            # whether a position is survivable, not the leverage figure.
            m = spec["maintenance"]
            trigger = 1.0 - (borrowed / (1.0 - m)) / notional if notional else np.nan
            print(f"    a {trigger:.1%} fall in {base_ticker} reaches the "
                  f"{m:.0%} maintenance requirement")

    print("\n  CONTEXT (full-sample backtest of these parameters)")
    s = out.stats
    print(f"    CAGR {s['cagr']:.2%} | maxDD {s['max_dd']:.2%} | "
          f"avg leverage {s['avg_leverage']:.2f}x | "
          f"levered {s['frac_levered']:.0%} of days")
    print(f"    asset {out.benchmark_stats['cagr']:.2%} | "
          f"maxDD {out.benchmark_stats['max_dd']:.2%}")

    # How often this actually asks you to trade -- the operational question.
    lev = out.result.leverage
    trades = int((out.result.turnover > 0).sum())
    years = (lev.index[-1] - lev.index[0]).days / 365.25
    print(f"\n    trades: {trades / years:.0f} a year on average "
          f"({trades} over {years:.0f} years)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
