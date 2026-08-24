"""The Kelly answer, translated into a Reg-T retail brokerage account.

    python scripts/run_kelly_account.py --spread 0.012

``scripts/run_kelly.py`` answers the abstract question -- what maximises expected
log wealth. This one answers the question an account holder actually faces: what
happens when that allocation is held at a retail broker, where leverage is capped
at 2x, the debit balance costs the broker's rate rather than the bill rate, and a
maintenance breach forces a sale at the worst possible moment.

Defaults are set to Robinhood's published terms as of 2026-08: 5.00% on margin
balances up to $50k against a 3.80% 3-month bill, i.e. a spread of ~1.2pp, and a
25% maintenance floor on diversified funds.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from lam.data import rates
from lam.financing.letf import simulate_letf
from lam.kelly import ASSETS, PRETTY, build_panel, empirical_kelly, score
from lam.kelly.account import REG_T_MAX_LEVERAGE, call_threshold, leverage_ladder, simulate

# Candidate allocations, as notional weights summing to 1x before leverage.
CANDIDATES = {
    "100% US equity": np.array([1.0, 0.0, 0.0, 0.0]),
    "80/20 US/intl equity": np.array([0.8, 0.2, 0.0, 0.0]),
    "70/30 US/intl equity": np.array([0.7, 0.3, 0.0, 0.0]),
    "90/10 equity/bonds": np.array([0.72, 0.18, 0.07, 0.03]),
    "60/40 global": np.array([0.36, 0.24, 0.28, 0.12]),
}

LEVELS = (1.0, 1.25, 1.5, 1.75, 2.0)


def _pct(x, dp=2):
    if x is None or (isinstance(float(x) if x is not None else 0.0, float) and pd.isna(x)):
        return "  --  "
    return f"{float(x) * 100:.{dp}f}%"


def _ladder_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["CAGR"] = frame["cagr"].map(lambda v: _pct(v))
    out["vol"] = frame["vol"].map(lambda v: _pct(v, 1))
    out["max drawdown"] = frame["max_drawdown"].map(lambda v: _pct(v, 1))
    out["x growth"] = frame["final_multiple"].map(lambda v: f"{v:,.1f}x")
    out["margin calls"] = frame["margin_calls"]
    out["first call"] = [
        "-" if pd.isna(d) or d is None else f"{d:%Y-%m}" for d in frame["first_call"]
    ]
    out["ruined"] = ["RUINED" if r else "-" for r in frame["ruined"]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="long", choices=["long", "modern"])
    ap.add_argument("--spread", type=float, default=0.012, help="broker rate over bills")
    args = ap.parse_args()

    panel = build_panel(args.panel)
    print(panel)
    bill = float(rates.bill_rate_bey(prefer="fred").iloc[-1])
    print(
        f"financing: bills {bill:.2%} + {args.spread:.2%} spread = {bill + args.spread:.2%} "
        f"today; Reg-T cap {REG_T_MAX_LEVERAGE:.0f}x, 25% maintenance, daily path"
    )

    print("\n" + "=" * 92)
    print("0. How far the holding can fall between rebalances before the broker sells")
    print("=" * 92)
    edges = pd.DataFrame(
        [
            {
                "leverage": f"{lev:g}x",
                "textbook 25% rule": _pct(call_threshold(lev), 1),
                "as simulated": _pct(
                    call_threshold(lev, intraday_stress=1.35, procyclicality=0.5), 1
                ),
            }
            for lev in LEVELS
        ]
    ).set_index("leverage")
    print(edges.to_string())
    print(
        "   Measured from the last rebalance, not from the peak: rebalancing monthly to a\n"
        "   fixed leverage sells on the way down, which is what keeps a slow 55% bear\n"
        "   market from triggering a call that a fast 30% one would."
    )

    print("\n" + "=" * 92)
    print("1. What leverage survives a Reg-T account, holding 100% US equity")
    print("=" * 92)
    ladder = leverage_ladder(
        panel, CANDIDATES["100% US equity"], levels=LEVELS, financing_spread=args.spread
    )
    print(_ladder_table(ladder).to_string())

    print("\n   ... and the same, for an investor who does not re-lever after being sold out:")
    behavioural = leverage_ladder(
        panel,
        CANDIDATES["100% US equity"],
        levels=LEVELS,
        financing_spread=args.spread,
        relever_after_call=False,
    )
    print(_ladder_table(behavioural).to_string())

    print("\n" + "=" * 92)
    print("2. Which allocation to lever -- all at 1.5x, the highest level with no forced sale")
    print("=" * 92)
    rows = []
    for name, weights in CANDIDATES.items():
        for level in (1.0, 1.5):
            result = simulate(
                panel, weights, target_leverage=level, financing_spread=args.spread
            )
            row = {"allocation": name, "leverage": f"{level:g}x"}
            row.update(result.summary)
            rows.append(row)
    table = pd.DataFrame(rows).set_index(["allocation", "leverage"])
    print(_ladder_table(table).to_string())

    print("\n" + "=" * 92)
    print("3. The cost of holding international equity you cannot justify from the data")
    print("=" * 92)
    for name in ("100% US equity", "80/20 US/intl equity", "70/30 US/intl equity"):
        stats = score(CANDIDATES[name], panel, financing_spread=args.spread)
        print(
            f"   {name:24s} log growth {_pct(stats['growth'])}  vol {_pct(stats['vol'], 1)}"
            f"  max DD {_pct(stats['max_drawdown'], 1)}  Sharpe {stats['sharpe']:.2f}"
        )

    print("\n" + "=" * 92)
    print("4. Margin vs a 2x daily-reset ETF (SSO), the only lever available in an IRA")
    print("=" * 92)
    daily = panel.daily["us_equity"]
    rf = (1.0 + panel.daily_cash).pow(365).sub(1.0)  # back to an annual rate
    day_counts = (
        pd.Series(daily.index, index=daily.index).diff().dt.days.fillna(1.0).clip(1, 5).to_numpy()
    )
    letf = simulate_letf(
        daily, rf, multiplier=2.0, expense_ratio=0.0089, swap_spread=0.0083,
        day_counts=day_counts,
    )
    letf_equity = (1.0 + letf).cumprod()
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    margin_2x = simulate(
        panel, CANDIDATES["100% US equity"], target_leverage=2.0, financing_spread=args.spread
    )
    print(
        f"   SSO-style 2x daily reset   CAGR {_pct(letf_equity.iloc[-1] ** (1 / years) - 1)}"
        f"  max DD {_pct((letf_equity / letf_equity.cummax() - 1).min(), 1)}"
        f"  margin calls  none possible"
    )
    m = margin_2x.summary
    print(
        f"   2x on Reg-T margin         CAGR {_pct(m['cagr'])}"
        f"  max DD {_pct(m['max_drawdown'], 1)}"
        f"  margin calls  {m['margin_calls']}"
    )

    print("\n" + "=" * 92)
    print("5. Does the safe leverage level survive a harsher broker?")
    print("=" * 92)
    variants = {
        "base (25% maint, 1.35 intraday stress)": {},
        "30% maintenance": {"maintenance": dict.fromkeys(ASSETS, 0.30)},
        "35% maintenance": {"maintenance": dict.fromkeys(ASSETS, 0.35)},
        "1.50 intraday stress": {"intraday_stress": 1.5},
        "no procyclical hike": {"procyclicality": 0.0},
        "spread 2.5%": {"financing_spread": 0.025},
    }
    rows = []
    for label, kwargs in variants.items():
        spread = kwargs.pop("financing_spread", args.spread)
        table = leverage_ladder(
            panel,
            CANDIDATES["100% US equity"],
            levels=(1.25, 1.5, 1.75, 2.0),
            financing_spread=spread,
            **kwargs,
        )
        row = {"assumption": label}
        for level, stats in table.iterrows():
            row[f"{level:g}x"] = (
                f"{stats['cagr'] * 100:5.1f}%  {int(stats['margin_calls'])} calls"
            )
        rows.append(row)
    print(pd.DataFrame(rows).set_index("assumption").to_string())

    print("\n" + "=" * 92)
    print("6. For reference: the unconstrained Kelly weights at this financing cost")
    print("=" * 92)
    for mode, cap in (("unconstrained", None), ("long_only", 2.0), ("long_only_unlevered", None)):
        w = empirical_kelly(
            panel, mode=mode, max_gross=cap, financing_spread=args.spread
        ).weights
        legs = "  ".join(f"{PRETTY[a][:12]:>12s} {w[a]:6.2f}" for a in ASSETS)
        print(f"   {mode:22s} gross {w.abs().sum():5.2f}x   {legs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
