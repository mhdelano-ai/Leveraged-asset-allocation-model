"""What the model actually holds.

The rest of the reporting answers "how did it do". This answers "what do I buy",
which is the question an allocation model exists to answer and the one the
performance tables cannot.

Three things make a bare weight vector misleading here, and each gets its own
column rather than a footnote:

* **Held is not target.** Weights drift between rebalances, so the last row of
  the realised weight path is the *drifted* book. Trading to it is not the same
  as trading to the standing order, so both are shown and a material gap is
  flagged.
* **A zero needs a reason.** Sleeves go to zero because the trend gate switched
  them off, not because the optimiser dislikes them. Without the gate value
  beside the weight, a 0% line looks like an error.
* **Gross exposure is the headline, not the mix.** Under the strict constraint
  this book averages 0.57x gross and is levered only 7.5% of the time -- a
  "leveraged" strategy that is usually de-levered. That belongs in the report,
  not in a reader's inference.

Everything here reads an existing :class:`~lam.pipeline.RunOutput`; nothing is
recomputed, so the report cannot disagree with the backtest it describes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.panels import Panel
from ..engine.accounting import needs_rebalance

TRADING_YEAR = 252


def current_allocation(out, panel: Panel, *, date=None) -> pd.DataFrame:
    """Held and target weights for one date, with the signals that produced them.

    ``date`` defaults to the last row of the backtest.
    """
    from ..pipeline import stack_for

    weights = out.result.weights
    targets = out.result.target_weights
    when = pd.Timestamp(date) if date is not None else weights.index[-1]
    if when not in weights.index:
        when = weights.index[weights.index.get_indexer([when], method="ffill")[0]]

    signals = stack_for(panel, out.params)
    held = weights.loc[when]
    target = targets.loc[when]
    gross = float(held.sum())

    trailing = weights.loc[:when].tail(TRADING_YEAR)
    trend = signals.trend_scaler.reindex(weights.index).ffill().loc[when]
    vol = None
    if hasattr(signals, "base_weights"):
        vol = signals.base_weights.loc[when]

    rows = []
    for i, sleeve in enumerate(weights.columns):
        rows.append(
            {
                "sleeve": sleeve,
                "ticker": panel.tickers[i],
                "held": float(held[sleeve]),
                "target": float(target[sleeve]),
                "share_of_book": float(held[sleeve] / gross) if gross > 1e-9 else 0.0,
                "avg_1y": float(trailing[sleeve].mean()),
                "avg_all": float(weights[sleeve].mean()),
                "trend_gate": float(trend.get(sleeve, np.nan)),
                "composition": float(vol[sleeve]) if vol is not None else np.nan,
                "eligible": bool(panel.eligible.loc[when, sleeve])
                if when in panel.eligible.index
                else True,
            }
        )
    frame = pd.DataFrame(rows).set_index("sleeve")
    frame.attrs["date"] = when
    frame.attrs["gross"] = gross
    frame.attrs["cash"] = 1.0 - gross
    frame.attrs["proxies"] = panel.tickers_are_proxies
    return frame


def needs_trade(out, panel: Panel, *, date=None) -> tuple[bool, float]:
    """Whether the held book is far enough from target to warrant trading.

    Bands come from the same :class:`EngineConfig` the backtest ran with, so the
    report cannot quietly disagree with the engine about what "close enough"
    means. The calendar trigger is deliberately excluded -- this asks whether the
    *drift* justifies trading, not whether it happens to be month-end.
    """
    from ..pipeline import engine_config

    cfg = engine_config(panel, out.params)
    weights, targets = out.result.weights, out.result.target_weights
    when = pd.Timestamp(date) if date is not None else weights.index[-1]
    held = weights.loc[when].to_numpy()
    target = targets.loc[when].to_numpy()
    fired = needs_rebalance(
        held,
        target,
        calendar_trigger=False,
        abs_band=cfg.abs_band,
        rel_band=cfg.rel_band,
        lev_band=cfg.lev_band,
        risk_off_move=0.0,
        risk_off_threshold=1.0,
    )
    return bool(fired), float(np.abs(target - held).sum())


def leverage_profile(out) -> dict:
    """Gross-exposure statistics, including how often leverage is used at all."""
    lev = out.result.leverage
    return {
        "current": float(lev.iloc[-1]),
        "avg_1y": float(lev.tail(TRADING_YEAR).mean()),
        "avg_all": float(lev.mean()),
        "median": float(lev.median()),
        "max": float(lev.max()),
        "frac_above_1x": float((lev > 1.0).mean()),
        "frac_above_2x": float((lev > 2.0).mean()),
        "avg_cash": float(1.0 - lev.mean()),
    }


def allocation_history(out, freq: str = "QE") -> pd.DataFrame:
    """Weights resampled to period ends, with a gross-exposure column.

    ``freq="decade"`` groups by decade and averages instead, which is the view
    that shows the mix shifting across regimes.
    """
    weights = out.result.weights
    if freq == "decade":
        grouped = weights.groupby(weights.index.year // 10 * 10).mean()
        grouped.index = [f"{d}s" for d in grouped.index]
    else:
        grouped = weights.resample(freq).last()
    grouped = grouped.copy()
    grouped["GROSS"] = grouped.sum(axis=1)
    return grouped


def risk_book_shares(out) -> pd.Series:
    """Long-run mix as shares of invested capital, ignoring the cash weight.

    This is the answer to "what is the allocation" in the sense people usually
    mean -- the mix -- separated from "how much of it do you own", which is the
    leverage profile.
    """
    avg = out.result.weights.mean()
    total = float(avg.sum())
    return (avg / total).sort_values(ascending=False) if total > 1e-9 else avg


def describe(out, panel: Panel, *, date=None) -> str:
    """Console block: current book, leverage profile, long-run mix."""
    frame = current_allocation(out, panel, date=date)
    prof = leverage_profile(out)
    fired, drift = needs_trade(out, panel, date=date)
    when = frame.attrs["date"]

    lines = [
        f"ALLOCATION - Panel {panel.name} @ {when.date()}",
        "",
        f"  {'sleeve':<14s} {'ticker':>7s} {'held':>8s} {'target':>8s} "
        f"{'of book':>8s} {'1y avg':>8s} {'trend':>7s}",
    ]
    for sleeve, r in frame.iterrows():
        gate = "-" if np.isnan(r["trend_gate"]) else f"{r['trend_gate']:.2f}"
        lines.append(
            f"  {sleeve:<14s} {r['ticker']:>7s} {r['held']:8.1%} {r['target']:8.1%} "
            f"{r['share_of_book']:8.1%} {r['avg_1y']:8.1%} {gate:>7s}"
        )
    lines += [
        f"  {'':<14s} {'':>7s} {'-'*8}",
        f"  {'gross':<14s} {'':>7s} {frame.attrs['gross']:8.1%}",
        f"  {'cash':<14s} {'':>7s} {frame.attrs['cash']:8.1%}",
        "",
        f"  leverage  now {prof['current']:.2f}x | 1y avg {prof['avg_1y']:.2f}x | "
        f"all {prof['avg_all']:.2f}x | max {prof['max']:.2f}x",
        f"            above 1x {prof['frac_above_1x']:.1%} of days, "
        f"above 2x {prof['frac_above_2x']:.1%} | avg cash {prof['avg_cash']:.1%}",
    ]
    if prof["frac_above_1x"] < 0.25:
        lines.append(
            f"            NOTE: levered only {prof['frac_above_1x']:.1%} of the time -- "
            "under this constraint the book is usually de-levered."
        )
    lines += [
        "",
        f"  rebalance: {'TRADE - outside bands' if fired else 'hold - inside bands'} "
        f"(|target - held| = {drift:.1%})",
    ]
    zero_gated = frame[(frame["held"] < 0.001) & (frame["trend_gate"] < 0.5)]
    if len(zero_gated):
        lines.append(
            f"  at zero because trend is off: {', '.join(zero_gated.index)}"
        )
    if frame.attrs["proxies"]:
        lines += [
            "",
            "  Tickers are modern proxies: this panel backtests synthetic series",
            "  (par-bond rolls, LBMA gold) that predate the ETFs.",
        ]
    return "\n".join(lines)
