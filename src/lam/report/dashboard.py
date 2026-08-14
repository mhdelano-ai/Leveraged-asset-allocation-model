"""Assembles ``docs/dashboard.html``: the JSON payload, and the page around it.

Kept apart from ``scripts/build_dashboard.py`` on purpose: everything here is a
pure function of already-loaded series, so ``tests/test_dashboard.py`` can drive
the real shaping logic -- monthly resampling, the trailing window, the headline
statistics -- against a small synthetic fixture without ever touching the
network or the on-disk parquet cache. :func:`load_and_build_payload` is the thin
I/O-touching wrapper the generator script actually calls; nothing in this module
outside that one function reads a file or opens a socket.

The strategy priced here is the *improved* variant discussed in
``docs/growth_findings.html`` -- ``RuleParams(up_wild=1.0)``, i.e. above the
200-day SMA but too volatile to lever, hold 1x QQQ rather than sit in cash. See
``lam.alloc.rules`` for the ``price_basis`` and ``sma_band`` parameters this
payload relies on: the SMA here is evaluated on raw closes with a 1% hysteresis
band, matching exactly what the page can compute for itself from a live feed
with no dividend stream to reinvest and no way to loop.

The page shell lives beside this module in ``templates/dashboard.html`` rather
than as a string literal inside the generator. It is 46KB of hand-authored
HTML, CSS and JavaScript, and inside a Python string none of that gets syntax
highlighting, formatting or linting from any tool that understands it -- and
every brace and quote becomes something to escape rather than something to
read. As a file it is editable as what it actually is, and :func:`render` is
the only thing that needs to know it is a template at all.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..alloc.rules import (
    CASH_SPREAD,
    QQQ_EXPENSE_RATIO,
    TQQQ_EXPENSE_RATIO,
    TQQQ_SWAP_SPREAD,
    RuleParams,
    backtest_four_state,
    switch_count,
)
from ..data import nasdaq, panels, yahoo
from ..metrics.core import TRADING_DAYS, ann_vol, cagr, calmar, sharpe
from ..metrics.drawdown import drawdown_series, max_drawdown

DASHBOARD_PANEL = "N"
DEFAULT_TRAILING_MONTHS = 18

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "dashboard.html"

# The one token the template has substituted into it. Deliberately something no
# HTML, CSS, JS or JSON output could ever produce on its own, so a plain
# ``str.replace`` is safe: a single substitution needs no templating engine, and
# an engine would only mean fighting the braces in the page's CSS and JS.
PLACEHOLDER = "__LAM_DASHBOARD_PAYLOAD__"


def _round_list(values: pd.Series | np.ndarray, decimals: int = 6) -> list[float]:
    return [round(float(v), decimals) for v in np.asarray(values, dtype=float)]


def _monthly_series(returns: pd.Series) -> tuple[list[str], list[float], list[float]]:
    """Month-end equity (rebased to $1 at ``returns``' first observation) and
    drawdown, sampled at each month's last trading day -- the same shape as
    ``docs/findings.html``'s embedded ``D`` object, so the chart-drawing code
    could be shared almost verbatim if this page ever wants it.
    """
    equity = (1.0 + returns).cumprod()
    dd = drawdown_series(returns)
    m_eq = equity.resample("ME").last()
    m_dd = dd.resample("ME").last()
    dates = [d.strftime("%Y-%m") for d in m_eq.index]
    return dates, _round_list(m_eq), _round_list(m_dd)


def _rebased_equity(returns: pd.Series) -> list[float]:
    """Growth of $1 invested at the start of ``returns``, not the full history.

    Drawdown for this window is deliberately *not* baked alongside it: the page
    may later splice a live tail onto this window and then slice to a shorter
    display range (the trailing-12-month chart out of an 18-month bake), and a
    peak computed over the wrong window is simply wrong. ``dashboard.html``
    recomputes drawdown from whatever equity path ends up on screen via
    ``drawdownFromEquity`` -- see the module docstring in that file.
    """
    equity = (1.0 + returns).cumprod()
    return _round_list(equity / equity.iloc[0])


def _headline_stats(
    returns: pd.Series, rf_daily: pd.Series, *, switches_per_year: float | None = None
) -> dict:
    cal = calmar(returns)
    out = {
        "cagr": round(cagr(returns), 6),
        "vol": round(ann_vol(returns), 6),
        "sharpe": round(sharpe(returns, rf_daily), 4),
        "max_dd": round(max_drawdown(returns), 6),
        # calmar is +inf for a monotone series -- JSON has no Infinity, and a
        # live page has no business rendering one anyway, so it becomes null.
        "calmar": round(float(cal), 4) if np.isfinite(cal) else None,
    }
    if switches_per_year is not None:
        out["switches_per_year"] = round(switches_per_year, 2)
    return out


def _days_in_current_state(state: pd.Series) -> int:
    """Length of the trailing run of days sharing today's state.

    Walked backward from the end so the break on the first mismatch lands at
    the start of the current run rather than the start of the series.
    """
    current = state.iloc[-1]
    days = 0
    for v in reversed(state.tolist()):
        if v != current:
            break
        days += 1
    return days


def build_payload(
    index_tr: pd.Series,
    rf_annual: pd.Series,
    price_basis: pd.Series,
    *,
    params: RuleParams | None = None,
    cost_per_unit: float = 0.0002,
    dividend_yield: float = 0.0,
    cash_rate_annual: float = 0.0,
    trailing_months: int = DEFAULT_TRAILING_MONTHS,
    generated_at: str | None = None,
    panel_name: str = DASHBOARD_PANEL,
) -> dict:
    """Build the dict that gets ``json.dumps``-ed into ``docs/dashboard.html``.

    ``index_tr`` is the daily total-return series the strategy trades (Panel
    N's benchmark); ``price_basis`` is the raw, non-dividend-adjusted price
    level the live page will actually see from its data feed -- see
    ``four_state_signal``'s docstring for why the two must not be conflated.
    ``dividend_yield`` and ``cash_rate_annual`` are not used to shape the baked
    history (that is exactly what ``price_basis`` avoids needing); they are
    carried through so the page can extend the equity curve past this payload's
    last day using the same total-return and cash-accrual definitions the
    backtest used, rather than inventing a different one for the live tail.
    """
    p = params or RuleParams(up_wild=1.0)
    returns, sig = backtest_four_state(
        index_tr, rf_annual, p, cost_per_unit=cost_per_unit, price_basis=price_basis,
    )
    # four_state_signal only ever adds lagged columns onto index_tr's own axis,
    # so all three series share one DatetimeIndex -- a single boolean mask
    # slices the trailing window out of all of them consistently.
    if not (returns.index.equals(index_tr.index) and sig.index.equals(index_tr.index)):
        raise AssertionError("backtest_four_state returned a misaligned index")

    dates, strat_eq, strat_dd = _monthly_series(returns)
    _, qqq_eq, qqq_dd = _monthly_series(index_tr)

    cutoff = index_tr.index[-1] - pd.DateOffset(months=trailing_months)
    window = index_tr.index >= cutoff
    trend_pct = sig["trend_price"] / sig["sma"] - 1.0

    rf_daily = rf_annual.reindex(index_tr.index).ffill().fillna(0.0) / TRADING_DAYS
    switches = switch_count(sig)

    return {
        "generated_at": generated_at or str(index_tr.index[-1].date()),
        "panel": panel_name,
        "params": asdict(p),
        "costs": {
            "cost_per_unit": cost_per_unit,
            "qqq_expense_ratio": QQQ_EXPENSE_RATIO,
            "tqqq_expense_ratio": TQQQ_EXPENSE_RATIO,
            "tqqq_swap_spread": TQQQ_SWAP_SPREAD,
            "cash_spread": CASH_SPREAD,
        },
        "dividend_yield": round(float(dividend_yield), 6),
        "cash_rate_annual": round(float(cash_rate_annual), 6),
        "monthly": {
            "dates": dates,
            "strategy_equity": strat_eq,
            "strategy_drawdown": strat_dd,
            "qqq_equity": qqq_eq,
            "qqq_drawdown": qqq_dd,
        },
        "daily": {
            "dates": [d.strftime("%Y-%m-%d") for d in index_tr.index[window]],
            "strategy_equity": _rebased_equity(returns[window]),
            "qqq_equity": _rebased_equity(index_tr[window]),
            "pct_above_sma": _round_list(trend_pct[window]),
            "vol20": _round_list(sig["vol"][window]),
            "state": sig["state"][window].tolist(),
            "exposure": _round_list(sig["exposure"][window], 4),
            # Unlagged -- see four_state_signal's docstring for trend_held.
            # Needed (alongside pct_above_sma) to say how far *tomorrow's*
            # reading is from flipping, which is a different question from
            # what today's (lagged) state/exposure already answers.
            "trend_held": sig["trend_held"][window].tolist(),
        },
        "headline": {
            "strategy": _headline_stats(returns, rf_daily, switches_per_year=switches),
            "qqq": _headline_stats(index_tr, rf_daily),
        },
        "current": {
            # above_sma/state/exposure are lagged -- this is what an investor
            # following the rule *holds* today, decided from yesterday's close.
            "date": str(sig.index[-1].date()),
            "state": str(sig["state"].iloc[-1]),
            "exposure": float(sig["exposure"].iloc[-1]),
            "days_in_state": _days_in_current_state(sig["state"]),
            # trend_price/sma/trend_held are *not* lagged -- this is today's
            # own close against its SMA, i.e. the reading that will decide
            # tomorrow's state. Showing both together is deliberate: "what you
            # hold" and "what just happened" are different questions.
            "pct_above_sma": round(float(trend_pct.iloc[-1]), 6),
            "vol20": round(float(sig["vol"].iloc[-1]), 6),
            "trend_held": bool(sig["trend_held"].iloc[-1]),
        },
    }


def load_and_build_payload(
    *, refresh: bool = False, params: RuleParams | None = None, trailing_months: int = DEFAULT_TRAILING_MONTHS
) -> dict:
    """Load Panel N and the raw Nasdaq-100 price index, then build the payload.

    This is the only function in the module that touches the network or the
    parquet cache under ``data/cache/`` -- everything it calls is already
    cached from prior runs of the research scripts, so in practice this reads
    local files. ``scripts/build_dashboard.py`` is the only caller; tests use
    :func:`build_payload` directly against a synthetic fixture instead.
    """
    panel = panels.build(DASHBOARD_PANEL)
    idx, rf, cost = panel.benchmark, panel.rf, float(panel.costs[0])

    # The live page fetches QQQ, but the panel's own trend basis is ^NDX (the
    # series the total-return reconstruction is built from -- see
    # lam.data.nasdaq). Using it here rather than QQQ's own history keeps the
    # baked backtest on the same clean series the reconstruction already
    # validated, and sidesteps QQQ's pre-2010 bad prints (see nasdaq.py).
    raw_close = yahoo.prices("^NDX", field="close", refresh=refresh)
    dividend_yield = nasdaq.implied_dividend_yield(refresh=refresh)
    cash_rate_annual = float(rf.iloc[-1]) + CASH_SPREAD

    return build_payload(
        idx,
        rf,
        raw_close,
        params=params or RuleParams(up_wild=1.0),
        cost_per_unit=cost,
        dividend_yield=dividend_yield,
        cash_rate_annual=cash_rate_annual,
        trailing_months=trailing_months,
    )


def load_template() -> str:
    """The page shell, read from ``templates/dashboard.html``."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise RuntimeError(
            f"{TEMPLATE_PATH} is missing its {PLACEHOLDER} placeholder -- the page "
            "would be generated with no data in it"
        )
    return template


def render(payload: dict) -> str:
    """The finished page: the template with ``payload`` spliced into it.

    ``allow_nan=False`` is load-bearing rather than fastidious. A NaN or an
    Infinity reaching here is always a bug upstream -- an un-trimmed warmup row,
    or a Calmar computed against a driftless synthetic series -- and Python's
    default would emit it as a bare ``NaN`` token. That is not valid JSON, but it
    *is* valid JavaScript, so it would parse silently in the page and surface
    much later as a blank chart rather than here as a traceback.
    """
    return load_template().replace(
        PLACEHOLDER, json.dumps(payload, allow_nan=False, separators=(",", ":"))
    )
