"""Managed futures (trend following): two monthly series, kept OUTSIDE the 12-asset panel.

    python -m assetforecast.data.sources.managed_futures [--refresh]

Writes ``data/processed/managed_futures_monthly.parquet`` (index month-end
``date``) and ``managed_futures_meta.json``. Nothing in the panel, the
covariance models or the published forecast reads these files; only the
diversifier study (``assetforecast.diversifiers``) does.

``tsmom_xr``   AQR "Time Series Momentum: Factors, Monthly", the all-asset
               factor (Moskowitz, Ooi and Pedersen 2012: 12-month trend, 1-month
               holding, 58 futures and forwards in equities, bonds, currencies
               and commodities), from 1985-01. A simple EXCESS return (futures
               positions on cash collateral). It is a HYPOTHETICAL portfolio:
               before management fees and before trading costs, built with
               rules published in 2012. Its level overstates what an investor
               earned; ``FEE_DRAG`` in the study is the declared allowance.
``funds_ret``  equal-weight mean of the simple TOTAL returns of the ``FUNDS``
               that have a return that month (``n_funds``), from Yahoo adjusted
               closes: live, after fees and costs, from 2007-03. Per-fund
               returns are in the ``fund_<SYMBOL>`` columns.

The fund list is a declared judgement, fixed before any of their correlations
with stocks was computed: US mutual funds and ETFs whose mandate is systematic
trend following in futures and that are alive today. That is a SURVIVOR list
(funds that closed are missing), so the composite's mean return is biased up;
its co-movement with stocks is much less affected. Yahoo's adjusted close is
used as is (the panel's ETFs get validated distributions; these funds do not),
so each fund's return is checked only against ``MAX_ABS_MONTH``.

A month is kept only when it is complete: the fund has a price in the last
``EDGE_DAYS`` calendar days of that month and of the month before.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from ...paths import PROCESSED
from ..fetch import fetch

AQR_URL = ("https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
           "Time-Series-Momentum-Factors-Monthly.xlsx")
AQR_NAME = "aqr/Time-Series-Momentum-Factors-Monthly.xlsx"
YAHOO_URL = ("https://{host}.finance.yahoo.com/v8/finance/chart/{sym}"
             "?period1=-2208988800&period2={p2}&interval=1d&events=div%2Csplits")
FUNDS = {
    "RYMFX": "Guggenheim Managed Futures Strategy P",
    "AQMIX": "AQR Managed Futures Strategy I",
    "ASFYX": "Virtus AlphaSimplex Managed Futures Strategy I",
    "WTMF": "WisdomTree Managed Futures Strategy Fund",
    "PQTIX": "PIMCO TRENDS Managed Futures Strategy Institutional",
    "ABYIX": "Abbey Capital Futures Strategy I",
    "DBMF": "iMGP DBi Managed Futures Strategy ETF",
    "KMLM": "KraneShares Mount Lucas Managed Futures Index Strategy ETF",
    "CTA": "Simplify Managed Futures Strategy ETF",
}
EDGE_DAYS = 5          # a month-end price must fall in the last 5 calendar days of the month
MAX_ABS_MONTH = 0.25   # a larger monthly fund return is treated as a data error and fails the build
OUT = PROCESSED / "managed_futures_monthly.parquet"
META = PROCESSED / "managed_futures_meta.json"


def read_tsmom(path) -> pd.Series:
    """The all-asset TSMOM factor as a month-end series of simple excess returns."""
    raw = pd.read_excel(path, sheet_name="TSMOM Factors", header=None)
    head = raw.index[raw[1].astype(str).str.strip() == "TSMOM"]
    if len(head) != 1:
        raise ValueError("TSMOM sheet: no single header row with the all-asset column 'TSMOM'")
    col = [1]
    body = raw.loc[head[0] + 1:, [0, col[0]]].dropna()
    s = pd.Series(body[col[0]].astype(float).to_numpy(),
                  index=pd.to_datetime(body[0]) + pd.offsets.MonthEnd(0), name="tsmom_xr")
    if s.index.has_duplicates or not s.index.is_monotonic_increasing:
        raise ValueError("TSMOM sheet: dates are not unique and increasing")
    return s.rename_axis("date")


def month_returns(px: pd.Series, edge_days: int = EDGE_DAYS) -> pd.Series:
    """Simple month-end to month-end returns of a daily price series, complete months only."""
    px = px.dropna()
    last = px.groupby(px.index + pd.offsets.MonthEnd(0)).tail(1)
    me = last.index + pd.offsets.MonthEnd(0)
    ok = (me - last.index).days < edge_days
    lvl = pd.Series(last.to_numpy(), index=me).where(ok)
    lvl = lvl.reindex(pd.date_range(me[0], me[-1], freq="ME"))
    return lvl.pct_change(fill_method=None).dropna().rename_axis("date")


def read_yahoo_adjclose(path) -> pd.Series:
    j = json.loads(open(path).read())["chart"]["result"][0]
    tz = j["meta"].get("exchangeTimezoneName", "America/New_York")
    idx = pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    s = pd.Series(j["indicators"]["adjclose"][0]["adjclose"], index=idx, dtype=float)
    return s[~s.index.duplicated(keep="last")].dropna()


def fetch_fund(sym: str, refresh: bool = False):
    p2 = int(time.time()) + 86400
    last = None
    for k, host in enumerate(["query1", "query2"] * 3):
        try:
            return fetch(YAHOO_URL.format(host=host, sym=sym, p2=p2), f"yahoo_mf/{sym}.json",
                         refresh=refresh, retries=2)
        except RuntimeError as e:      # Yahoo rate-limits bursts
            last = e
            time.sleep(15 * (k + 1))
    raise RuntimeError(f"Yahoo download failed for {sym}: {last}")


def composite(funds: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight mean over the funds with a return each month, and how many there were."""
    return pd.DataFrame({"funds_ret": funds.mean(axis=1), "n_funds": funds.notna().sum(axis=1)})


def build(refresh: bool = False) -> pd.DataFrame:
    tsmom = read_tsmom(fetch(AQR_URL, AQR_NAME, refresh=refresh))
    funds = {}
    for sym in FUNDS:
        r = month_returns(read_yahoo_adjclose(fetch_fund(sym, refresh)))
        if (r.abs() > MAX_ABS_MONTH).any():
            raise ValueError(f"{sym}: monthly return beyond {MAX_ABS_MONTH:.0%}: "
                             f"{r[r.abs() > MAX_ABS_MONTH].round(3).to_dict()}")
        funds[f"fund_{sym}"] = r
        time.sleep(1.0)
    funds = pd.DataFrame(funds)
    out = pd.concat([tsmom, composite(funds), funds], axis=1).sort_index().rename_axis("date")
    out["n_funds"] = out["n_funds"].fillna(0).astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    both = out[["tsmom_xr", "funds_ret"]].dropna()
    meta = {
        "tsmom": {"source": AQR_URL, "start": str(tsmom.index[0].date()), "end": str(tsmom.index[-1].date()),
                  "months": int(len(tsmom)), "kind": "hypothetical, simple excess return, before fees and costs"},
        "funds": {s: {"name": FUNDS[s], "start": str(funds[f"fund_{s}"].dropna().index[0].date()),
                      "end": str(funds[f"fund_{s}"].dropna().index[-1].date())} for s in FUNDS},
        "funds_kind": "live, simple total return after fees, Yahoo adjusted close; survivor list",
        "overlap_months": int(len(both)),
        "overlap_corr_monthly": float(np.corrcoef(both["tsmom_xr"], both["funds_ret"])[0, 1]),
    }
    META.write_text(json.dumps(meta, indent=1))
    return out


def load() -> pd.DataFrame:
    return pd.read_parquet(OUT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m assetforecast.data.sources.managed_futures")
    ap.add_argument("--refresh", action="store_true", help="download again instead of using data/raw")
    a = ap.parse_args(argv)
    out = build(a.refresh)
    print(out[["tsmom_xr", "funds_ret", "n_funds"]].describe().round(4).to_string())
    print(META.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
