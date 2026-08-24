"""Kelly-optimal allocation across US/international equities and bonds, plus cash.

    python scripts/run_kelly.py                 # both panels, full report
    python scripts/run_kelly.py --panel long --draws 2000

Writes ``docs/kelly_results.json`` and prints every table in the write-up. The
report is deliberately ordered so the uncomfortable numbers come first: the
unconstrained Kelly portfolio, then what it did to a real account, then the
fractional and constrained versions that are actually holdable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lam.kelly import (
    ASSETS,
    PRETTY,
    PROXY_NOTES,
    bootstrap_weights,
    build_panel,
    constrained_shrunk_kelly,
    fraction_curve,
    empirical_kelly,
    gaussian_kelly,
    score,
    shrunk_kelly,
    summarise,
    walk_forward,
)

MONTHS = 12
REPO_ROOT = Path(__file__).resolve().parents[1]


def _pct(x: float, dp: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  --  "
    if x == -np.inf:
        return " RUIN "
    return f"{x * 100:.{dp}f}%"


def asset_stats(panel) -> pd.DataFrame:
    rets = panel.returns
    years = len(rets) / MONTHS
    vol = rets.std(ddof=1) * np.sqrt(MONTHS)
    out = pd.DataFrame(
        {
            "arith": rets.mean() * MONTHS,
            "cagr": (1.0 + rets).prod() ** (1.0 / years) - 1.0,
            "vol": vol,
            "sharpe": panel.excess.mean() * MONTHS / vol,
            "worst_month": rets.min(),
        }
    )
    out.index = [PRETTY[a] for a in out.index]
    cash_cagr = (1.0 + panel.cash).prod() ** (1.0 / years) - 1.0
    out.loc["Cash (3m T-bill)"] = [
        panel.cash.mean() * MONTHS,
        cash_cagr,
        panel.cash.std(ddof=1) * np.sqrt(MONTHS),
        0.0,
        panel.cash.min(),
    ]
    return out


def weight_row(name: str, weights, panel) -> dict:
    w = pd.Series(np.asarray(weights, dtype=float), index=ASSETS)
    sc = score(w, panel)
    row = {"portfolio": name}
    row.update({PRETTY[a]: float(w[a]) for a in ASSETS})
    row["Cash"] = 1.0 - float(w.sum())
    row.update(sc)
    return row


def solve_all(panel) -> pd.DataFrame:
    rows = [
        weight_row("Gaussian Kelly (Sigma^-1 mu)", gaussian_kelly(panel), panel),
        weight_row("Full Kelly, empirical", empirical_kelly(panel).weights, panel),
    ]
    full = empirical_kelly(panel).weights
    for f in (0.5, 0.25):
        rows.append(weight_row(f"{f:g}x Kelly, empirical", full * f, panel))
    rows.append(
        weight_row("Full Kelly, long-only", empirical_kelly(panel, mode="long_only").weights, panel)
    )
    for cap in (3.0, 2.0, 1.5):
        rows.append(
            weight_row(
                f"Kelly, long-only, gross <= {cap:g}x",
                empirical_kelly(panel, mode="long_only", max_gross=cap).weights,
                panel,
            )
        )
    rows.append(
        weight_row(
            "Kelly, long-only, no leverage",
            empirical_kelly(panel, mode="long_only_unlevered").weights,
            panel,
        )
    )
    rows.append(weight_row("Shrunk Kelly (mu 50%, cov 20%)", shrunk_kelly(panel), panel))
    rows.append(weight_row("Shrunk half-Kelly", shrunk_kelly(panel) * 0.5, panel))
    rows.append(
        weight_row(
            "Shrunk Kelly, long-only, no leverage",
            constrained_shrunk_kelly(panel),
            panel,
        )
    )
    for cap in (2.0, 1.5):
        rows.append(
            weight_row(
                f"Shrunk Kelly, long-only, gross <= {cap:g}x",
                constrained_shrunk_kelly(panel, mode="long_only", max_gross=cap),
                panel,
            )
        )
    rows.append(weight_row("60/40 global cap-weighted", [0.36, 0.24, 0.28, 0.12], panel))
    rows.append(weight_row("100% US equity", [1.0, 0.0, 0.0, 0.0], panel))
    return pd.DataFrame(rows).set_index("portfolio")


def print_weights(frame: pd.DataFrame) -> None:
    cols = [PRETTY[a] for a in ASSETS] + ["Cash"]
    show = frame[cols + ["gross", "growth", "vol", "max_drawdown", "worst_month"]].copy()
    for c in cols:
        show[c] = show[c].map(lambda v: f"{v:6.2f}")
    for c in ("growth", "vol", "max_drawdown", "worst_month"):
        show[c] = show[c].map(lambda v: _pct(v, 1))
    show["gross"] = show["gross"].map(lambda v: f"{v:5.2f}x")
    print(show.to_string())


def run_panel(name: str, *, draws: int, min_months: int) -> dict:
    panel = build_panel(name)
    print("\n" + "=" * 100)
    print(panel)
    print("proxies: " + "; ".join(f"{t} = {PROXY_NOTES[t]}" for t in panel.tickers.values()))
    print("=" * 100)

    stats = asset_stats(panel)
    print("\n-- Asset classes, annualised --")
    print(stats.map(lambda v: _pct(v, 2)).to_string())

    print("\n-- Correlation of monthly returns --")
    corr = panel.returns.corr()
    corr.index = corr.columns = [PRETTY[a] for a in ASSETS]
    print(corr.round(2).to_string())

    solutions = solve_all(panel)
    print("\n-- Kelly solutions (weights as a fraction of capital; Cash = 1 - sum) --")
    print_weights(solutions)

    print(f"\n-- Bootstrap of the weights ({draws} block draws, 12-month blocks) --")
    boots = {}
    for mode in ("unconstrained", "long_only_unlevered"):
        b = bootstrap_weights(panel, mode=mode, draws=draws)
        b.columns = [PRETTY.get(c, c) for c in b.columns]
        q = b.quantile([0.05, 0.5, 0.95]).T
        q.columns = ["p5", "median", "p95"]
        print(f"\n   mode = {mode}")
        print(q.round(2).to_string())
        boots[mode] = q

    print("\n-- Sensitivity: does the answer survive a different financing cost? --")
    spread_rows = []
    for spread in (0.0, 0.006, 0.015):
        for mode, cap in (("unconstrained", None), ("long_only", 2.0)):
            w = empirical_kelly(
                panel, mode=mode, max_gross=cap, financing_spread=spread
            ).weights
            row = {"spread": _pct(spread, 2), "mode": mode + (f" <= {cap:g}x" if cap else "")}
            row.update({PRETTY[a]: round(float(w[a]), 2) for a in ASSETS})
            row["gross"] = round(float(w.abs().sum()), 2)
            spread_rows.append(row)
    spreads = pd.DataFrame(spread_rows).set_index(["spread", "mode"])
    print(spreads.to_string())

    print("\n-- Sensitivity: does the answer survive being asked in a different decade? --")
    era_rows = []
    # Positional 10-year slices from the start of the panel, so no months are
    # dropped at the front and the last (short) block is folded into the one
    # before it rather than being reported on its own.
    step = 120
    starts = list(range(0, panel.months, step))
    if len(starts) > 1 and panel.months - starts[-1] < 36:
        starts.pop()
    for i, lo in enumerate(starts):
        hi = starts[i + 1] if i + 1 < len(starts) else panel.months
        window = panel.returns.iloc[lo:hi]
        sub = type(panel)(
            name=panel.name,
            returns=window,
            cash=panel.cash.iloc[lo:hi],
            tickers=panel.tickers,
        )
        era = f"{window.index[0]:%Y-%m}..{window.index[-1]:%Y-%m}"
        for mode, cap in (("unconstrained", None), ("long_only", 2.0)):
            w = empirical_kelly(sub, mode=mode, max_gross=cap).weights
            row = {"era": era, "mode": mode + (f" <= {cap:g}x" if cap else "")}
            row.update({PRETTY[a]: round(float(w[a]), 2) for a in ASSETS})
            row["gross"] = round(float(w.abs().sum()), 2)
            era_rows.append(row)
    eras = pd.DataFrame(era_rows).set_index(["era", "mode"])
    print(eras.to_string())

    walks = {}
    if panel.months >= min_months + 120:
        print("\n-- Out of sample: fit Kelly on the past only, hold it forward --")
        for mode, cap in (("unconstrained", None), ("long_only", 2.0), ("long_only_unlevered", None)):
            wf = walk_forward(panel, mode=mode, max_gross=cap, min_months=min_months)
            label = mode + (f" (gross <= {cap:g}x)" if cap else "")
            rows = []
            for col in wf.columns:
                summary = summarise(wf[col])
                summary["strategy"] = col
                rows.append(summary)
            table = pd.DataFrame(rows).set_index("strategy")
            print(f"\n   fitted as: {label};  live {wf.index[0]:%Y-%m} -> {wf.index[-1]:%Y-%m}")
            show = table.copy()
            for c in ("cagr", "log_growth", "vol", "max_drawdown", "worst_month"):
                show[c] = show[c].map(lambda v: _pct(v, 2))
            show["ruin"] = [
                f"RUINED {d:%Y-%m}" if r else "-"
                for r, d in zip(table["ruined"], table["ruin_date"])
            ]
            print(show[["cagr", "log_growth", "vol", "max_drawdown", "worst_month", "ruin"]].to_string())
            walks[label] = table

        print("\n-- Log growth against the Kelly fraction, in sample vs out --")
        curve = fraction_curve(
            panel, mode="unconstrained", fractions=np.round(np.arange(0.0, 1.01, 0.05), 3),
            min_months=min_months,
        )
        show = curve.copy()
        for c in ("in_sample_growth", "out_of_sample_growth", "out_of_sample_vol", "out_of_sample_max_drawdown"):
            show[c] = show[c].map(lambda v: _pct(v, 2))
        print(show.to_string())
        best = curve["out_of_sample_growth"].idxmax()
        print(
            f"\n   in-sample peak at 1.00x ({_pct(curve['in_sample_growth'].max(), 2)});"
            f" out-of-sample peak at {best:.2f}x ({_pct(curve['out_of_sample_growth'].max(), 2)});"
            f" ruin from {curve.index[curve['ruined']].min():.2f}x upward"
        )
    else:
        print(
            f"\n-- Out of sample: skipped. {panel.months} months of data and a "
            f"{min_months}-month training window leave "
            f"{max(panel.months - min_months, 0)} live months; under 120 the "
            "comparison says nothing. --"
        )

    return {
        "span": [str(panel.returns.index[0].date()), str(panel.returns.index[-1].date())],
        "months": panel.months,
        "tickers": panel.tickers,
        "asset_stats": stats.to_dict(),
        "correlation": panel.returns.corr().to_dict(),
        "solutions": solutions.drop(columns=["ruined"], errors="ignore").to_dict(orient="index"),
        "bootstrap": {k: v.to_dict() for k, v in boots.items()},
        "financing_sensitivity": spreads.reset_index().to_dict(orient="records"),
        "era_sensitivity": eras.reset_index().to_dict(orient="records"),
        "fraction_curve": (curve.reset_index().to_dict(orient="records") if walks else []),
        "walk_forward": {
            k: v.drop(columns=["ruin_date"]).to_dict(orient="index") for k, v in walks.items()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="both", choices=["modern", "long", "both"])
    ap.add_argument("--draws", type=int, default=400, help="bootstrap draws")
    ap.add_argument("--min-months", type=int, default=120, help="walk-forward training window")
    ap.add_argument("--out", default="docs/kelly_results.json")
    args = ap.parse_args()

    names = ["modern", "long"] if args.panel == "both" else [args.panel]
    results = {
        name: run_panel(name, draws=args.draws, min_months=args.min_months) for name in names
    }

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
