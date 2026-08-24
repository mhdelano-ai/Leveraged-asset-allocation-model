"""The Kelly question asked forward, from what the assets yield today.

    python scripts/run_kelly_forward.py --spread 0.012 --paths 400

Historical means are the wrong input for this decision and the backtest said so:
33 years cannot identify four risk premia, and the sample contains a 40-year fall
in yields that cannot repeat. This script replaces them with a build-up from
observable yields -- see :mod:`lam.kelly.forward` -- and re-answers every question
the historical run answered: the weights, the leverage, and the account risk.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from lam.kelly import ASSETS, PRETTY, build_panel
from lam.kelly.forward import (
    GLOBAL_EQUITY_WEIGHTS,
    PRESETS,
    Assumptions,
    build_cma,
    bundle_kelly,
    bundle_moments,
    constrained_kelly,
    decompose_growth,
    forward_ladder,
    geometric_frontier,
    growth_needed_for,
    historical_real_eps_growth,
    kelly,
    market_inputs,
    profit_share,
    scaled_frontier,
    rolling_real_eps_growth,
    tilt_sensitivity,
    with_equity_premium,
)


def _pct(v, dp=2):
    return "  --  " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.{dp}f}%"


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _weights(w: pd.Series) -> str:
    return "  ".join(f"{PRETTY[a].split()[0][:5]:>5s} {w[a]:5.2f}" for a in ASSETS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spread", type=float, default=0.012, help="broker rate over bills")
    ap.add_argument("--paths", type=int, default=400)
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--risk-window", type=float, default=5.0)
    args = ap.parse_args()

    panel = build_panel("modern")
    inputs = market_inputs()
    cma = build_cma(
        panel, inputs=inputs, assumptions=Assumptions(risk_window_years=args.risk_window)
    )

    print(inputs)
    print("=" * 100)
    print(f"1. Forward expected returns, built up from those yields (risk from the last "
          f"{args.risk_window:g} years)")
    print("=" * 100)
    table = cma.table()
    print(table.to_string(float_format=lambda v: f"{v * 100:6.2f}%"))
    print(f"\n   cash (3m bill) {cma.cash:.2%} -- the hurdle every excess return above is measured over")
    print("\n   correlations:")
    corr = cma.corr.copy()
    corr.index = corr.columns = [PRETTY[a] for a in ASSETS]
    print(corr.round(2).to_string())

    print("\n" + "=" * 100)
    print("2. Kelly on those inputs")
    print("=" * 100)
    for label, spread in (("no financing cost", 0.0), (f"borrowing at bills + {args.spread:.1%}", args.spread)):
        print(f"   unconstrained, {label:32s} {_weights(kelly(cma, financing_spread=spread))}"
              f"   gross {kelly(cma, financing_spread=spread).abs().sum():.2f}x")
    for mode, cap in (("long_only", None), ("long_only", 2.0), ("long_only_unlevered", None)):
        w = constrained_kelly(cma, mode=mode, max_gross=cap, financing_spread=args.spread)
        label = mode + (f" <= {cap:g}x" if cap else "")
        print(f"   {label:46s} {_weights(w)}   gross {w.abs().sum():.2f}x")

    print("\n" + "=" * 100)
    print("3. The same question over two bundles instead of four sleeves")
    print("=" * 100)
    moments = bundle_moments(cma)
    print(
        f"   global equity at cap weights ({GLOBAL_EQUITY_WEIGHTS[0]:.0%} US / "
        f"{GLOBAL_EQUITY_WEIGHTS[1]:.0%} intl):"
    )
    print(
        f"      expected {_pct(moments['geometric'])} geometric, excess {_pct(moments['excess'])},"
        f" vol {_pct(moments['vol'], 1)}, Sharpe {moments['sharpe']:.2f}"
    )
    print(f"      unlevered Kelly multiple  {moments['kelly']:.2f}x   (if you could borrow at the bill rate)")
    for label, spread in (("at the bill rate", 0.0), (f"at bills + {args.spread:.1%}", args.spread)):
        w = bundle_kelly(cma, financing_spread=spread)
        print(f"      Kelly, {label:22s} {_weights(w)}   gross {w.abs().sum():.2f}x")
    need = moments["vol"] ** 2 * 1.5 + args.spread
    print(
        f"\n   For full Kelly to reach 1.5x, global equity would have to earn "
        f"{_pct(need)} over cash instead of {_pct(moments['excess'])}\n"
        f"   -- about {_pct(need - moments['excess'], 1)}/yr more growth than the yields imply."
    )

    print("\n" + "=" * 100)
    print("4. How much of this is a real preference for international equity?")
    print("=" * 100)
    tilt = tilt_sensitivity(cma, financing_spread=args.spread)
    show = tilt.copy()
    show.index = [f"{v * 100:+.1f}pp" for v in show.index]
    show.index.name = "US expected return minus intl, vs the base case"
    print(show.round(2).to_string())
    print(
        "\n   The split is a knife edge: a 2pp swing in an unobservable growth assumption\n"
        "   moves it from all-international to all-US. That is what an optimiser does with\n"
        "   two assets correlated 0.81, and it is the argument for holding market weights."
    )

    print("\n" + "=" * 100)
    print("5. Where the growth assumption comes from, and what it implies")
    print("=" * 100)
    hist = historical_real_eps_growth()
    show = hist.copy()
    for c in ("raw_endpoints", "smoothed_endpoints"):
        show[c] = show[c].map(lambda v: _pct(v))
    show["years"] = hist["years"].map(lambda v: f"{v:.0f}")
    print("   Realised real earnings PER SHARE growth, S&P 500 (Shiller):")
    print(show.to_string())

    rolling = rolling_real_eps_growth()
    print(f"\n   Every 30-year window since 1871 ({len(rolling)} of them):")
    print(
        "      "
        + "   ".join(
            f"p{q}: {_pct(np.percentile(rolling, q))}" for q in (5, 25, 50, 75, 95)
        )
        + f"   max: {_pct(rolling.max())}"
    )

    implied = cma.implied_eps_growth
    pct = float((rolling.to_numpy() < implied["us_equity"]).mean() * 100)
    print(
        f"\n   The build-up assumes {_pct(cma.build_up.loc['us_equity', 'real growth'])} of"
        f" AGGREGATE real growth plus {_pct(cma.build_up.loc['us_equity', 'buyback'])} of buyback,"
        f"\n   which is {_pct(implied['us_equity'])} PER SHARE -- the {pct:.0f}th percentile of"
        " that distribution.\n   The base case is an above-median growth forecast, not a"
        " pessimistic one."
    )
    for target, name in ((0.08, "8% nominal"), (0.1082, "its own 1993-2026 realised 10.82%")):
        need = growth_needed_for(cma, target)
        print(
            f"\n   For US equities to return {name}, per-share real growth must be"
            f" {_pct(need)}\n      -- against a best-ever 30-year figure of {_pct(rolling.max())}."
        )

    print("\n" + "=" * 100)
    print("5b. The geometric frontier -- fractional Kelly by slope, not by scaling")
    print("=" * 100)
    grid = np.round(np.arange(0.1, 1.51, 0.1), 3)
    slope = geometric_frontier(
        cma, fractions=grid, financing_spread=args.spread,
        equity_split=GLOBAL_EQUITY_WEIGHTS,
    )
    scaled = scaled_frontier(
        cma, fractions=grid, financing_spread=args.spread,
        equity_split=GLOBAL_EQUITY_WEIGHTS,
    )
    table = slope[["gross", "vol", "geometric", *ASSETS, "cash"]].copy()
    table["vs scaling"] = slope["geometric"] - scaled["geometric"]
    display = pd.DataFrame(index=table.index)
    display["gross"] = table["gross"].map(lambda v: f"{v:.2f}x")
    for col in ("vol", "geometric"):
        display[col] = table[col].map(lambda v: _pct(v))
    short = {"us_equity": "US eq", "intl_equity": "Intl eq",
             "us_bonds": "US bd", "intl_bonds": "Intl bd"}
    for a in ASSETS:
        display[short[a]] = table[a].map(lambda v: f"{v:5.2f}")
    display["cash"] = table["cash"].map(lambda v: f"{v:5.2f}")
    display["vs scaling"] = table["vs scaling"].map(lambda v: f"{v * 100:+.2f}pp")
    print(display.to_string())
    peak = slope["geometric"].idxmax()
    print(
        f"\n   Peak growth {_pct(slope.loc[peak, 'geometric'])} at f = {peak:.2f}"
        f" ({slope.loc[peak, 'gross']:.2f}x gross, {_pct(slope.loc[peak, 'vol'], 1)} vol)."
    )
    bonds = slope["us_bonds"] + slope["intl_bonds"]
    print(
        "   Bonds appear only where nothing is borrowed: they compete against the 3.80%\n"
        "   bill rate below 1x gross, not the 5.00% margin rate above it. Scaling the\n"
        f"   full-Kelly weights instead of re-optimising gives up as much as"
        f" {(table['vs scaling'].max() * 100):.2f}pp/yr.\n"
        f"   Largest bond weight on the frontier: {bonds.max():.0%} of capital."
    )

    print("\n" + "=" * 100)
    print("6. Which parts of recent growth can repeat, and which cannot")
    print("=" * 100)
    parts = decompose_growth()
    show = parts.copy()
    for c in show.columns:
        show[c] = parts[c].map(lambda v: f"{v * 100:+6.2f}%")
    print(show.to_string())
    share = profit_share()
    print(
        f"\n   Profit share of GDP: {share.loc[:'2000'].mean():.2%} average to 2000,"
        f" {share.loc['2010':].mean():.2%} since 2010, {share.iloc[-1]:.2%} now."
    )
    print(
        "   Dilution swung from +2.12%/yr against shareholders to -0.84%/yr in their\n"
        "   favour -- a ~3pp/yr structural gain, and the durable half of the modern era's\n"
        "   growth. The other half, +1.75%/yr of profit-share expansion, is a level shift:\n"
        "   the share doubled, and cannot double again."
    )

    print("\n   The same question under each named assumption set:")
    rows = []
    for name, preset in PRESETS.items():
        scenario = build_cma(panel, inputs=inputs, assumptions=preset)
        moments = bundle_moments(scenario)
        w = bundle_kelly(scenario, financing_spread=args.spread)
        equity = w["us_equity"] + w["intl_equity"]
        rows.append(
            {
                "assumptions": name,
                "US per-share growth": _pct(scenario.implied_eps_growth["us_equity"]),
                "global equity return": _pct(moments["geometric"]),
                "over cash": _pct(moments["excess"]),
                "full Kelly": f"{equity:.2f}x",
                "half Kelly": f"{equity / 2:.2f}x",
            }
        )
    print(pd.DataFrame(rows).set_index("assumptions").to_string())

    print("\n" + "=" * 100)
    print("7. If the equity premium is not what the yields imply")
    print("=" * 100)
    rows = []
    for shift in (-0.02, -0.01, 0.0, 0.01, 0.02, 0.03):
        shifted = with_equity_premium(cma, shift)
        m = bundle_moments(shifted)
        w = bundle_kelly(shifted, financing_spread=args.spread)
        rows.append(
            {
                "shift": f"{shift * 100:+.0f}pp",
                "equity geometric": _pct(m["geometric"]),
                "excess over cash": _pct(m["excess"]),
                "full Kelly equity": f"{w['us_equity'] + w['intl_equity']:.2f}x",
                "half Kelly": f"{(w['us_equity'] + w['intl_equity']) / 2:.2f}x",
            }
        )
    print(pd.DataFrame(rows).set_index("shift").to_string())

    print("\n" + "=" * 100)
    print(f"8. Forward account risk: {args.paths} bootstrapped {args.years}-year paths, Reg-T account")
    print("=" * 100)
    equity = np.array([GLOBAL_EQUITY_WEIGHTS[0], GLOBAL_EQUITY_WEIGHTS[1], 0.0, 0.0])
    ladder = forward_ladder(
        panel,
        cma,
        equity,
        levels=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
        years=args.years,
        paths=args.paths,
        financing_spread=args.spread,
    )
    show = pd.DataFrame(index=ladder.index)
    show["median CAGR"] = ladder["median_cagr"].map(lambda v: _pct(v))
    show["5th pct"] = ladder["p5_cagr"].map(lambda v: _pct(v))
    show["95th pct"] = ladder["p95_cagr"].map(lambda v: _pct(v))
    show["median max DD"] = ladder["median_max_drawdown"].map(lambda v: _pct(v, 1))
    show["worst 5% max DD"] = ladder["p5_max_drawdown"].map(lambda v: _pct(v, 1))
    show["P(margin call)"] = ladder["p_margin_call"].map(lambda v: _pct(v, 1))
    show["P(lose to cash)"] = ladder["p_underperform_cash"].map(lambda v: _pct(v, 1))
    show["P(ruin)"] = ladder["p_ruin"].map(lambda v: _pct(v, 1))
    print(show.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
