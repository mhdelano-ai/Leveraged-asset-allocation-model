"""Yahoo Finance daily bar loader.

Two traps are encoded as runtime guards rather than comments, because both fail
silently and produce a plausible-looking backtest:

1. ``range=max&interval=1d`` returns **quarterly** bars, not daily -- SPY comes
   back as ~168 points spanning 1984+ instead of ~8,400 daily bars, and nothing in
   the response says so. Passing explicit ``period1``/``period2`` epochs returns
   true daily data. This module never sends ``range``, and
   :func:`_assert_daily` re-checks the delivered bar spacing regardless.

2. For **indices** (``^GSPC``, ``^SPGSCI``) Yahoo's ``adjclose`` is identical to
   ``close`` -- it is *not* dividend-adjusted. Treating it as a total-return
   series silently discards ~2-4%/yr of dividends. :func:`prices` therefore
   requires the caller to state which field it wants, and index callers must
   build total return explicitly (see ``synth_equity``).
"""

from __future__ import annotations

import json
from urllib.parse import quote

import pandas as pd

from ..timeaxis import median_day_gap
from . import cache
from .http import get

BASE = "https://query2.finance.yahoo.com/v8/finance/chart/"

# Epoch bounds wide enough for the full ^GSPC history (1927-12-30 onward).
EPOCH_START = -2_208_988_800  # 1900-01-01
EPOCH_END = 1_900_000_000  # 2030-03-17

# Minimum plausible row counts. Yahoo intermittently returns a *valid, parseable*
# response containing only the last few weeks of data -- during this build ^TNX
# and ^FVX both cached with 15 rows instead of ~16,262, and nothing about the
# payload flagged it. Without this guard the truncation surfaces much later as an
# inexplicable backtest result.
MIN_ROWS = {
    "^GSPC": 24_000, "^SP500TR": 9_000, "^SPGSCI": 10_000, "^VIX": 9_000,
    "^IRX": 16_000, "^FVX": 16_000, "^TNX": 16_000, "^TYX": 12_000,
    "SPY": 8_000, "VTI": 6_000, "EFA": 6_000, "EEM": 5_500, "TLT": 6_000,
    "IEF": 6_000, "LQD": 6_000, "HYG": 4_500, "GLD": 5_000, "DBC": 5_000,
    "VNQ": 5_000, "UPRO": 4_000, "SSO": 4_800, "TMF": 4_000, "UGL": 4_200,
    "VTRIX": 10_000, "VUSTX": 9_500, "VFITX": 8_000, "VWESX": 11_000,
    "VWEHX": 11_000, "FRESX": 9_500, "VFINX": 11_000,
}


class NonDailyBarsError(RuntimeError):
    """Raised when Yahoo returns coarser-than-daily bars."""


class TruncatedSeriesError(RuntimeError):
    """Raised when Yahoo returns a valid response with implausibly few rows."""


def _assert_daily(index: pd.DatetimeIndex, symbol: str) -> None:
    if len(index) < 20:
        return
    gap = median_day_gap(index)
    if gap > 4.0:
        raise NonDailyBarsError(
            f"{symbol}: median bar spacing {gap:.1f} days -- not daily data. "
            "This is the `range=max` quarterly-bar trap; use explicit period1/period2."
        )


def _assert_complete(frame: pd.DataFrame, symbol: str) -> None:
    floor = MIN_ROWS.get(symbol)
    if floor is not None and len(frame) < floor:
        raise TruncatedSeriesError(
            f"{symbol}: got {len(frame)} rows, expected >= {floor}. Yahoo returned a "
            "truncated response; retry rather than caching this."
        )


def bars(symbol: str, *, refresh: bool = False) -> pd.DataFrame:
    """Full available daily OHLC + adjclose history for ``symbol``.

    Returns columns ``open, high, low, close, adjclose, volume``.
    """
    key = f"yahoo_{symbol.replace('^', 'IDX_').replace('=', '_')}"
    if cache.has(key) and not refresh:
        return cache.load(key)

    url = f"{BASE}{quote(symbol)}"
    params = {"period1": EPOCH_START, "period2": EPOCH_END, "interval": "1d"}
    raw = get(url, params=params)
    payload = json.loads(raw)

    result = payload.get("chart", {}).get("result")
    if not result:
        err = payload.get("chart", {}).get("error")
        raise RuntimeError(f"{symbol}: no data returned ({err})")
    node = result[0]

    idx = pd.DatetimeIndex(pd.to_datetime(node["timestamp"], unit="s").normalize())
    quote_block = node["indicators"]["quote"][0]
    frame = pd.DataFrame(
        {
            "open": quote_block.get("open"),
            "high": quote_block.get("high"),
            "low": quote_block.get("low"),
            "close": quote_block.get("close"),
            "volume": quote_block.get("volume"),
        },
        index=idx,
    )
    adj = node["indicators"].get("adjclose")
    frame["adjclose"] = adj[0]["adjclose"] if adj else frame["close"]

    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.dropna(subset=["close"])
    _assert_daily(frame.index, symbol)
    _assert_complete(frame, symbol)

    cache.store(key, frame, url=f"{url}?period1={EPOCH_START}&period2={EPOCH_END}", raw=raw)
    return frame


def prices(symbol: str, *, field: str, refresh: bool = False) -> pd.Series:
    """Single price column.

    ``field='adjclose'`` for ETFs and mutual funds (genuinely total return, net of
    fees). ``field='close'`` for indices, where adjclose is a no-op and total
    return must be reconstructed separately.
    """
    if field not in {"close", "adjclose", "low", "high", "open"}:
        raise ValueError(f"unsupported field {field!r}")
    out = bars(symbol, refresh=refresh)[field].dropna()
    out.name = symbol
    return out


def total_return_prices(symbol: str, *, refresh: bool = False) -> pd.Series:
    """Adjusted-close series for a dividend-paying fund or ETF.

    Rejects index symbols, where adjclose == close and this would silently drop
    the dividend stream.
    """
    if symbol.startswith("^"):
        raise ValueError(
            f"{symbol} is an index: Yahoo adjclose == close and carries no dividends. "
            "Build total return explicitly instead."
        )
    return prices(symbol, field="adjclose", refresh=refresh)
