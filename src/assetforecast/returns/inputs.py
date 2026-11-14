"""Inputs of the return side, all read from the package Panel.

``load_return_panel``  the panel without tier D (month-average eras), from 1926-07-31
``return_predictors``  the 73 return-side predictors (``rp_`` series) as a wide month-end panel
``excess_returns``     monthly log excess returns through the last complete month
``observed_through``   that last complete month end

Everything works on ``Panel.asof`` views: a view cut at ``t`` exposes only the
``rp_`` rows dated at or before ``t`` (their ``obs_date`` equals their
``available_date``, the month end at which the value was known), and the
metadata dates are recomputed from what the view holds.
"""
from __future__ import annotations

import pandas as pd

from ..data.panel import Panel, load_panel
from ..data.sources.return_predictors import META_COLUMNS, PREFIX, TRACK_META, read_meta, to_wide
from . import config

__all__ = ["load_return_panel", "return_predictors", "excess_returns", "observed_through"]


def load_return_panel(max_date=None) -> Panel:
    """``load_panel(max_date, drop_tiers=RETURN_DROP_TIERS)`` with the monthly
    frames (returns, cash, tiers) starting at ``RETURN_START``.

    Daily data are cut at the first day of ``RETURN_START``'s month (today a
    no-op: the daily panel starts 1926-07-01). Predictors are NOT cut: their
    history before ``RETURN_START`` was known at it, and expanding statistics
    of the feature formulas use it.
    """
    p = load_panel(max_date, drop_tiers=config.RETURN_DROP_TIERS)
    start = pd.Timestamp(config.RETURN_START)
    first_day = start - pd.offsets.MonthBegin(1)
    monthly = p.monthly.loc[start:]
    daily = p.daily.loc[first_day:]
    return Panel(daily=daily, monthly=monthly,
                 cash_daily=p.cash_daily.loc[first_day:], cash_monthly=p.cash_monthly.loc[start:],
                 tiers=p.tiers.loc[start:], predictors=p.predictors, asof_date=p.asof_date)


def observed_through(panel: Panel) -> pd.Timestamp:
    """The last month end whose returns are in the panel (only complete months are).

    On an ``asof`` view that is the last month end at or before ``asof_date``
    with any return in it."""
    m = panel.monthly
    if panel.asof_date is not None:
        m = m.loc[:panel.asof_date]
    m = m.dropna(how="all")
    if m.empty:
        raise ValueError("panel holds no monthly returns")
    return pd.Timestamp(m.index[-1])


def excess_returns(panel: Panel) -> pd.DataFrame:
    """Monthly LOG excess returns, month end x the 12 asset ids, through ``observed_through``."""
    return panel.excess_monthly.loc[:observed_through(panel)]


def return_predictors(panel: Panel, meta_path=TRACK_META) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(wide, meta) for the ``rp_`` series the panel (or view) holds.

    ``wide``: month-end index from the first to the last ``rp_`` row in the
    panel, one float column per series in the sidecar's order, prefix stripped,
    NaN where a series has no value. On the full panel it equals the archived
    assetpred wide file exactly.
    ``meta``: ``META_COLUMNS`` from ``returns_meta.csv``; ``first_date`` and
    ``last_date`` are those of ``wide`` (empty string for a series with no
    value yet), so a view's metadata says nothing about later data.
    """
    meta = read_meta(meta_path)
    p = panel.predictors
    p = p[p["series"].astype(str).str.startswith(PREFIX)]
    if p.empty:
        raise ValueError("the panel has no rp_ series: run `python -m assetforecast.data.sources."
                         "return_predictors --from-committed` and then the assembler")
    present = {s[len(PREFIX):] for s in p["series"].unique()}
    unknown = present - set(meta["name"])
    if unknown:
        raise ValueError(f"rp_ series missing from {meta_path}: {sorted(unknown)}")
    names = list(meta["name"])
    wide = to_wide(p, names=names)
    first = {c: wide[c].first_valid_index() for c in names}
    last = {c: wide[c].last_valid_index() for c in names}
    meta["first_date"] = [first[c].date().isoformat() if first[c] is not None else "" for c in names]
    meta["last_date"] = [last[c].date().isoformat() if last[c] is not None else "" for c in names]
    return wide, meta[META_COLUMNS].reset_index(drop=True)
