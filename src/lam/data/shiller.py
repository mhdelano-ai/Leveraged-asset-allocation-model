"""Robert Shiller's long-run S&P 500 dataset (Yale), used for dividends only.

**Critical constraint on how this file may be used.** Shiller's ``SP500`` column
is the *monthly average of daily closes*, not the month-end level. For October
1987 it reads ~280 against an actual intramonth low of 224.84 -- using it as a
price path would erase Black Monday entirely, which is precisely the event a
drawdown-constrained study cannot afford to miss.

So: the price path always comes from daily ``^GSPC``. This module supplies only
the trailing-twelve-month **dividend yield** (``Dividend / SP500``), where taking
the ratio of two monthly averages is a sound estimate of the yield in that month.
"""

from __future__ import annotations

import io

import pandas as pd

from . import cache
from .http import get

URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"


def _decimal_year_to_timestamp(value: float) -> pd.Timestamp:
    """Shiller encodes dates as YYYY.MM with month 10 written as ``.1``."""
    year = int(value)
    frac = round((value - year) * 100)
    month = 10 if frac == 1 else int(frac)
    month = min(max(month, 1), 12)
    return pd.Timestamp(year=year, month=month, day=1)


def dividend_yield(*, refresh: bool = False) -> pd.Series:
    """Monthly trailing-12m S&P 500 dividend yield as a decimal (0.04 == 4%).

    Indexed by month start, from 1871.
    """
    key = "shiller_div_yield"
    if cache.has(key) and not refresh:
        return cache.load(key)

    raw = get(URL, timeout=120)
    sheet = pd.read_excel(io.BytesIO(raw), sheet_name="Data", header=None)

    # Locate the header row: the one whose first cell is literally "Date".
    header_row = None
    for i in range(min(20, len(sheet))):
        if str(sheet.iloc[i, 0]).strip().lower() == "date":
            header_row = i
            break
    if header_row is None:
        raise RuntimeError("Shiller sheet: could not locate the 'Date' header row")

    body = sheet.iloc[header_row + 1 :].copy()
    body.columns = [str(c).strip() for c in sheet.iloc[header_row]]

    dates = pd.to_numeric(body.iloc[:, 0], errors="coerce")
    price = pd.to_numeric(body.iloc[:, 1], errors="coerce")  # "P" / SP500 comp
    dividend = pd.to_numeric(body.iloc[:, 2], errors="coerce")  # "D"

    valid = dates.notna() & price.notna() & dividend.notna() & (price > 0)
    dates, price, dividend = dates[valid], price[valid], dividend[valid]

    idx = pd.DatetimeIndex([_decimal_year_to_timestamp(v) for v in dates])
    out = pd.Series((dividend / price).to_numpy(), index=idx, name="div_yield")
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # Shiller pads the most recent months with literal 0.0 rather than a missing
    # marker; a 0% dividend yield would quietly remove dividends from the newest
    # data. Drop the trailing zero block and let callers forward-fill.
    out = out[out > 0]

    cache.store(key, out, url=URL, raw=raw)
    return out
