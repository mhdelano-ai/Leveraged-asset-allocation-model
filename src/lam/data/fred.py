"""FRED CSV loader.

No API key needed for the graph CSV endpoint. Two traps are handled here:

* Missing values are the literal string ``"."`` and must become NaN. Zero-filling
  a yield series would silently inject 0% rates on every market holiday.
* Several FRED series are **licence-truncated** to a rolling window while still
  returning HTTP 200 with well-formed CSV. ``SP500`` and ``DJIA`` carry only 10
  years; the ICE BofA credit series (``BAMLH0A0HYM2``, ``BAMLC0A0CM``) carry only
  ~3. A status-code check passes and the data is useless. ``MIN_ROWS`` asserts a
  floor so this fails loudly at build time instead of quietly shortening a panel.
"""

from __future__ import annotations

import io

import pandas as pd

from . import cache
from .http import get

BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Expected minimum row counts. Anything materially shorter means the series was
# truncated upstream (licence change) or renamed.
MIN_ROWS = {
    "DGS2": 12_000,
    "DGS10": 16_000,
    "DGS20": 11_000,
    "DGS30": 12_000,
    "DTB3": 18_000,
    "DFF": 26_000,
    "CPIAUCSL": 900,
    "VIXCLS": 9_000,
    "SOFR": 1_800,
}

# Series that look fine but are licence-truncated. Guard against accidental use.
BLOCKED = {
    "SP500": "licence-truncated to 10 years; use ^GSPC + Shiller dividends instead",
    "DJIA": "licence-truncated to 10 years",
    "BAMLH0A0HYM2": "ICE BofA licence-truncated to ~3 years (795 rows from 2023)",
    "BAMLC0A0CM": "ICE BofA licence-truncated to ~3 years",
    "GOLDAMGBD228NLBM": "discontinued (404); use the LBMA JSON feed instead",
    "GOLDPMGBD228NLBM": "discontinued (404); use the LBMA JSON feed instead",
    "WILL5000IND": "discontinued (404)",
}


class LicenceTruncatedError(RuntimeError):
    pass


def series(series_id: str, *, refresh: bool = False) -> pd.Series:
    """Fetch a FRED series as a float Series indexed by date, NaNs dropped."""
    if series_id in BLOCKED:
        raise LicenceTruncatedError(f"FRED {series_id}: {BLOCKED[series_id]}")

    key = f"fred_{series_id}"
    if cache.has(key) and not refresh:
        return cache.load(key)

    url = f"{BASE}?id={series_id}"
    raw = get(url)
    text = raw.decode("utf-8", "replace")
    if text.lstrip().startswith("<"):
        raise RuntimeError(f"FRED {series_id}: returned HTML, not CSV (series missing?)")

    df = pd.read_csv(io.StringIO(text))
    date_col, value_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    # "." is FRED's missing marker; coerce turns it into NaN.
    values = pd.to_numeric(df[value_col], errors="coerce")
    out = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(df[date_col]), name=series_id)
    out = out.dropna()

    floor = MIN_ROWS.get(series_id)
    if floor is not None and len(out) < floor:
        raise LicenceTruncatedError(
            f"FRED {series_id}: got {len(out)} rows, expected >= {floor}. "
            "Series is likely licence-truncated upstream."
        )

    cache.store(key, out, url=url, raw=raw)
    return out


def discount_to_bey(discount_pct: pd.Series, days: int = 91) -> pd.Series:
    """Convert a T-bill *discount* rate to bond-equivalent yield.

    ``DTB3`` is quoted on a discount basis, not as a yield. Using it directly as a
    financing or cash-return rate understates the true rate -- by ~10bp at 4%, and
    materially more at the 1980s' double-digit levels, which is exactly the regime
    where leverage cost decides the outcome.

        BEY = 365 * d / (360 - d * days)
    """
    d = discount_pct / 100.0
    return 365.0 * d / (360.0 - d * days)
