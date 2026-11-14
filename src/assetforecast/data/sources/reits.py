"""US equity REITs (``reits``, VNQ-like): longest credible total-return history.

Run ``python -m assetforecast.data.sources.reits`` to (re)build
``data/processed/assets/reits_{daily,monthly}.parquet`` and print the splice
diagnostics. Raw downloads are cached under ``data/raw/reits/``; pass
``--refresh`` to re-download them.

Construction (justified, with overlap diagnostics, in ``reports/data/reits.md``)
-------------------------------------------------------------------------------
monthly
  * 1972-01 .. 1986-11   FTSE Nareit All Equity REITs total return (monthly
                         index, published by Nareit) -> tier C
  * 1986-12 .. last full month   compounded daily series below
daily
  * 1986-11-17 .. 1996-05-31  Fidelity Real Estate Investment Portfolio (FRESX),
                         an actively managed REIT fund, Yahoo adjclose -> tier B.
                         It is an open-end fund NAV, and an active proxy for the
                         asset class (monthly tracking error ~3.9%/yr vs the
                         Nareit index in this era, vs ~1.1% for VGSIX later).
  * 1996-06-03 .. 2004-09-30  Vanguard REIT Index Inv (VGSIX), Yahoo adjclose -> tier B
                         (open-end fund NAV)
  * 2004-10-01 .. today       Vanguard Real Estate ETF (VNQ), close + validated
                         distributions -> tier A

Tiers (project-wide): A = daily price of an exchange-traded fund (market
close); B = daily open-end mutual-fund NAV, index, or reconstruction; C =
monthly index total return; D = monthly model-based reconstruction. The
VGSIX -> VNQ splice is therefore a tier change (B -> A).

VNQ's total return is computed from its close plus Yahoo's distributions, each
validated against the ex-date price move relative to other REIT ETFs
(``distributions.validate_distributions``); Yahoo's adjclose is not trusted for
it. Nothing in this track is fitted on post-2014 data: the fund choices rest on
the 1986-2004 overlaps with the Nareit index.

Every return is a SIMPLE TOTAL return (dividends reinvested) in decimals, not
excess. Nothing is interpolated; the current, incomplete month is left out of
the monthly file. Nareit's spreadsheet is fetched with a non-browser
User-Agent: reit.com serves a JavaScript challenge to browser User-Agents.
"""
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ...paths import ASSET_DIR, RAW
from . import distributions as dist

ASSET = "reits"
CACHE = RAW / "reits"
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
PLAIN_UA = "python-requests/2 (assetforecast data loader)"

YAHOO_URL = "https://{host}.finance.yahoo.com/v8/finance/chart/{sym}"
NAREIT_URL = "https://www.reit.com/sites/default/files/returns/MonthlyHistoricalReturns.xls"
FRENCH_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
              "49_Industry_Portfolios_CSV.zip")

# Daily fund segments: (symbol, first return date used). Each runs until the
# day before the next segment's start.
FUNDS = [("FRESX", "1986-11-17"), ("VGSIX", "1996-06-03"), ("VNQ", "2004-10-01")]
# Other Yahoo series, used only for diagnostics in the report.
DIAG_SYMBOLS = ["CSRSX", "DFREX", "IYR", "ICF", "RWR"]
NAREIT_SERIES = "All Equity REITs"   # excludes mortgage REITs
NAREIT_DIAG = ["Equity REITs", "All REITs", "Mortgage REITs"]

TIER = {"FRESX": "B", "VGSIX": "B", "VNQ": "A", "nareit": "C"}
SOURCE = {
    "FRESX": "yahoo:FRESX adjclose (Fidelity Real Estate Investment, active)",
    "VGSIX": "yahoo:VGSIX adjclose (Vanguard REIT Index Inv)",
    "VNQ": "yahoo:VNQ close + validated distributions (Vanguard Real Estate ETF)",
    "nareit": "nareit:FTSE Nareit All Equity REITs total return",
}
# Bad prints in Yahoo adjclose: the PRICE on these dates is dropped (set to NaN),
# so the next return spans two days. Found by scanning every used segment for
# one-day spike-reversals (|r|>2%, reversed the next day to within 25%) and
# checking peer funds on the same day; only one failed the peer check:
#   FRESX 1993-07-29: close 13.29 between 13.90 and 13.91, no distribution;
#   CSRSX +0.4%, DFREX +1.0% that day.
BAD_PRICES = {"FRESX": ["1993-07-29"]}

# Overlap diagnostics behind each splice (from ``diagnostics()``; see report),
# recorded as the ``notes`` column of segments.csv.
NOTES = {
    "nareit": ("All Equity REITs (excl. mortgage REITs). Next segment FRESX vs this index, "
               "monthly 1986-12..1996-05: corr 0.948, vol ratio 0.985, TE 3.9%/yr. Pre-1990s "
               "REIT market tiny/illiquid: low measured vol, smoothed returns (AC1 0.18 in 1980s)."),
    "FRESX": ("Active fund used as daily proxy. Monthly vs Nareit 1986-12..1996-05: corr 0.948, "
              "vol ratio 0.985, TE 3.9%/yr; daily vs VGSIX 1996-06..2004-09: corr 0.971, "
              "vol ratio 0.981, TE 2.9%/yr."),
    "VGSIX": ("Open-end index fund NAV (MSCI US REIT), tier B. Monthly vs Nareit 1996-06..2004-09: corr 0.997, vol ratio "
              "1.009, TE 1.1%/yr; daily vs VNQ 2004-10..2026-11: corr 0.994, vol ratio 1.022, "
              "TE 3.2%/yr (monthly corr 0.999, TE 0.75%/yr)."),
    "VNQ": ("Monthly vs Nareit 2004-10..2026-10: corr 0.998, vol ratio 1.014, TE 1.5%/yr. Close + Yahoo "
            "distributions, each validated by the ex-date price test against IYR/ICF/RWR total returns "
            "(every distribution passed)."),
}


# --------------------------------------------------------------------- fetch
def _get(url: str, params: dict | None = None, ua: str = BROWSER_UA,
         tries: int = 6, sleep: float = 2.0) -> requests.Response:
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": ua, "Accept": "*/*"},
                             timeout=60)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {url}")
            if r.status_code == 404:
                break
        except requests.RequestException as e:  # network hiccup: retry
            last = e
        time.sleep(sleep * (i + 1))
    raise RuntimeError(f"download failed: {url}: {last}")


def fetch_yahoo(sym: str, refresh: bool = False) -> dict:
    """Raw Yahoo chart JSON (full daily history with dividend/split events), cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"yahoo_{sym.replace('^', '_')}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    params = {"period1": 0, "period2": int(time.time()) + 86400,
              "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"}
    err = None
    for host in ("query1", "query2"):
        try:
            r = _get(YAHOO_URL.format(host=host, sym=sym), params=params)
            j = r.json()
            if j["chart"]["result"]:
                path.write_text(json.dumps(j))
                time.sleep(1.0)  # be polite: other agents share this endpoint
                return j
            err = j["chart"]["error"]
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(2.0)
    raise RuntimeError(f"Yahoo returned no data for {sym}: {err}")


def yahoo_adjclose(sym: str, refresh: bool = False) -> pd.Series:
    """Daily adjusted close (splits + distributions) indexed by exchange-local date."""
    res = fetch_yahoo(sym, refresh)["chart"]["result"][0]
    tz = res["meta"].get("exchangeTimezoneName", "America/New_York")
    idx = (pd.to_datetime(res["timestamp"], unit="s", utc=True)
           .tz_convert(tz).tz_localize(None).normalize())
    s = pd.Series(res["indicators"]["adjclose"][0]["adjclose"], index=idx, dtype=float)
    s = s[~s.index.duplicated(keep="last")].dropna()
    s = s[s > 0]
    s.index.name = "date"
    return s.sort_index().rename(sym)


def fetch_nareit(refresh: bool = False) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "nareit_MonthlyHistoricalReturns.xls"
    if path.exists() and not refresh:
        return path
    r = _get(NAREIT_URL, ua=PLAIN_UA)
    if not r.content.startswith(b"\xd0\xcf\x11\xe0"):  # OLE2 .xls magic
        raise RuntimeError("reit.com returned HTML (JavaScript challenge?) instead of the .xls")
    path.write_bytes(r.content)
    return path


def nareit_monthly(refresh: bool = False) -> pd.DataFrame:
    """FTSE Nareit monthly TOTAL returns (decimals), one column per index, month-end index."""
    raw = pd.read_excel(fetch_nareit(refresh), sheet_name="Index Data", header=None)
    # locate header rows: series names on the row containing "All Equity REITs",
    # "Total" on the next, "Return" on the one after.
    hdr = raw.index[raw.apply(lambda r: r.astype(str).str.strip().eq("All Equity REITs").any(),
                              axis=1)][0]
    names = raw.loc[hdr].ffill()
    kind = raw.loc[hdr + 1].astype(str).str.strip()
    what = raw.loc[hdr + 2].astype(str).str.strip()
    dates = pd.to_datetime(raw[0].where(raw[0].map(lambda v: hasattr(v, "year"))), errors="coerce")
    body = raw[dates.notna()]
    out = {}
    for c in raw.columns:
        if kind[c] == "Total" and what[c] == "Return" and isinstance(names[c], str):
            out[names[c].strip()] = pd.to_numeric(body[c], errors="coerce").values / 100.0
    df = pd.DataFrame(out, index=pd.DatetimeIndex(dates[dates.notna()].values, name="date"))
    df.index = df.index + pd.offsets.MonthEnd(0)  # file uses last business day
    return df.dropna(how="all").sort_index()


def french_real_estate(refresh: bool = False) -> pd.Series:
    """Ken French 49-industry 'RlEst' value-weighted monthly return (diagnostics only)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "french_49_Industry_Portfolios_CSV.zip"
    if not path.exists() or refresh:
        path.write_bytes(_get(FRENCH_URL).content)
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith(",Agric"))
    block = [lines[start]]
    for ln in lines[start + 1:]:
        if not ln.strip() or not ln.strip()[0].isdigit():
            break
        block.append(ln)
    df = pd.read_csv(io.StringIO("\n".join(block)), index_col=0)
    df.columns = [c.strip() for c in df.columns]
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m") + pd.offsets.MonthEnd(0)
    s = df["RlEst"].replace([-99.99, -999], np.nan) / 100.0
    return s.rename("french_rlest")


# --------------------------------------------------------------------- build
# ETFs whose return is close + VALIDATED distributions (peers: other REIT ETFs' total returns)
ETF_PEERS = {"VNQ": ["IYR", "ICF", "RWR"]}


def yahoo_frame(sym: str, refresh: bool = False) -> pd.DataFrame:
    """Daily close (split-adjusted) and listed distributions on their ex-date trading day."""
    res = fetch_yahoo(sym, refresh)["chart"]["result"][0]
    tz = res["meta"].get("exchangeTimezoneName", "America/New_York")
    to_date = lambda t: pd.to_datetime(t, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()  # noqa: E731
    df = pd.DataFrame({"close": res["indicators"]["quote"][0]["close"]}, index=to_date(res["timestamp"]),
                      dtype=float)
    df = df[~df.index.duplicated(keep="last")].dropna()
    df = df[df["close"] > 0].sort_index()
    df.index.name = "date"
    ev = res.get("events", {}).get("dividends", {})
    div = pd.Series({to_date(int(v["date"])): float(v["amount"]) for v in ev.values()}, dtype=float)
    div = div.groupby(level=0).sum() if len(div) else div
    div, _ = dist.dedupe_repeats(div)
    df["div"] = dist.align_dividends(div, df.index)
    return df


def etf_returns(sym: str, refresh: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    """ETF total return from close + validated distributions, and the validation table."""
    df = yahoo_frame(sym, refresh)
    peers = pd.DataFrame({p: yahoo_adjclose(p, refresh).pct_change() for p in ETF_PEERS[sym]})
    div, table = dist.validate_distributions(df["close"], df["div"], peers)
    df, _ = dist.drop_bad_prices(df.assign(div=div), BAD_PRICES.get(sym, []))
    return dist.total_return(df["close"], df["div"]).iloc[1:].dropna().rename(sym), table


def daily_returns(sym: str, refresh: bool = False) -> pd.Series:
    if sym in ETF_PEERS:
        return etf_returns(sym, refresh)[0]
    p = yahoo_adjclose(sym, refresh)
    p = p.drop(pd.DatetimeIndex(BAD_PRICES.get(sym, [])), errors="ignore")
    return p.pct_change().dropna().rename(sym)


def build_daily(refresh: bool = False) -> pd.DataFrame:
    parts = []
    for i, (sym, start) in enumerate(FUNDS):
        r = daily_returns(sym, refresh)
        end = pd.Timestamp(FUNDS[i + 1][1]) - pd.Timedelta(days=1) if i + 1 < len(FUNDS) else None
        r = r.loc[pd.Timestamp(start):end]
        if r.empty:
            raise RuntimeError(f"no data for {sym} in its segment")
        parts.append(pd.DataFrame({"ret": r, "source": SOURCE[sym], "tier": TIER[sym]}))
    d = pd.concat(parts)
    d.index.name = "date"
    assert d.index.is_unique and d.index.is_monotonic_increasing
    return d


def build_monthly(daily: pd.DataFrame, nareit: pd.DataFrame) -> pd.DataFrame:
    first_full = daily.index.min() + pd.offsets.MonthBegin(1)   # first month fully in daily
    first_full_me = first_full + pd.offsets.MonthEnd(0)
    last_me = _last_complete_me(daily.index)
    g = daily.loc[first_full:]
    dm = (1 + g["ret"]).groupby(pd.Grouper(freq="ME")).prod() - 1
    meta = g.groupby(pd.Grouper(freq="ME"))[["source", "tier"]].agg(_mode)
    dm = pd.DataFrame({"ret": dm}).join(meta).loc[first_full_me:last_me]
    nm = nareit[NAREIT_SERIES].dropna().loc[:first_full_me - pd.offsets.MonthEnd(1)]
    nm = pd.DataFrame({"ret": nm, "source": SOURCE["nareit"], "tier": TIER["nareit"]})
    m = pd.concat([nm, dm])
    m.index.name = "date"
    assert m.index.is_unique and m.index.is_monotonic_increasing
    return m


def _mode(x: pd.Series):
    return x.mode().iloc[0] if len(x) else np.nan


def _last_complete_me(idx: pd.DatetimeIndex) -> pd.Timestamp:
    last = idx.max()
    me = last + pd.offsets.MonthEnd(0)
    # complete if no business day remains in the month after the last observation
    if last + pd.offsets.BDay(1) > me:
        return me
    return me - pd.offsets.MonthEnd(1)


# --------------------------------------------------------------- diagnostics
def overlap_stats(a: pd.Series, b: pd.Series, periods: int) -> dict:
    """Overlap stats of a vs b (b = reference): n, vol ratio, corr, tracking error (ann.)."""
    x = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(x) < 10:
        return {"n": len(x)}
    ra, rb = x.iloc[:, 0], x.iloc[:, 1]
    return {
        "start": x.index[0].date().isoformat(), "end": x.index[-1].date().isoformat(),
        "n": len(x),
        "vol_a": ra.std() * np.sqrt(periods), "vol_b": rb.std() * np.sqrt(periods),
        "vol_ratio": ra.std() / rb.std(),
        "corr": ra.corr(rb),
        "te": (ra - rb).std() * np.sqrt(periods),
        "mean_diff_ann": (ra - rb).mean() * periods,
    }


def to_monthly(r: pd.Series) -> pd.Series:
    """Compound daily returns to calendar months; drops the first and last (maybe partial) month."""
    m = (1 + r).groupby(pd.Grouper(freq="ME")).prod() - 1
    return m.iloc[1:-1]


def diagnostics(refresh: bool = False) -> dict:
    nareit = nareit_monthly(refresh)
    syms = [s for s, _ in FUNDS] + DIAG_SYMBOLS
    d = {s: daily_returns(s, refresh) for s in syms}
    m = {s: to_monthly(d[s]) for s in syms}
    out = {"daily": {}, "monthly": {}}
    pairs_d = [("FRESX", "VGSIX"), ("CSRSX", "VGSIX"), ("DFREX", "VGSIX"),
               ("VGSIX", "VNQ"), ("IYR", "VNQ"), ("ICF", "VNQ"), ("RWR", "VNQ"),
               ("FRESX", "CSRSX"), ("FRESX", "DFREX")]
    for a, b in pairs_d:
        out["daily"][f"{a}~{b}"] = overlap_stats(d[a], d[b], 252)
    ref = nareit[NAREIT_SERIES]
    for s in syms:
        out["monthly"][f"{s}~nareit"] = overlap_stats(m[s], ref, 12)
    for a, b in [("FRESX", "VGSIX"), ("VGSIX", "VNQ")]:
        out["monthly"][f"{a}~{b}"] = overlap_stats(m[a], m[b], 12)
    for c in NAREIT_DIAG:
        out["monthly"][f"nareit_{c}~nareit"] = overlap_stats(nareit[c], ref, 12)
    try:
        out["monthly"]["french_rlest~nareit"] = overlap_stats(french_real_estate(refresh), ref, 12)
    except Exception as e:  # noqa: BLE001  (diagnostic only)
        out["monthly"]["french_rlest~nareit"] = {"error": str(e)}
    return out


def vol_by_decade(monthly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    lm = np.log1p(monthly["ret"])
    ld = np.log1p(daily["ret"])
    dec = lambda ix: (ix.year // 10) * 10  # noqa: E731
    out = pd.DataFrame({
        "months": lm.groupby(dec(lm.index)).count(),
        "vol_monthly": lm.groupby(dec(lm.index)).std() * np.sqrt(12),
        "days": ld.groupby(dec(ld.index)).count(),
        "vol_daily": ld.groupby(dec(ld.index)).std() * np.sqrt(252),
    })
    out.index.name = "decade"
    return out


# ---------------------------------------------------------------------- main
def build(refresh: bool = False, write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = build_daily(refresh)
    nareit = nareit_monthly(refresh)
    monthly = build_monthly(daily, nareit)
    if write:
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(ASSET_DIR / f"{ASSET}_daily.parquet")
        monthly.to_parquet(ASSET_DIR / f"{ASSET}_monthly.parquet")
    return daily, monthly


def segments(daily: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Provenance rows in the segments.csv schema (asset, start, end, freq, source, tier, notes)."""
    key = {v: k for k, v in SOURCE.items()}
    rows = []
    for (src, tier), g in daily.groupby(["source", "tier"], sort=False):
        rows.append((ASSET, g.index.min(), g.index.max(), "D", src, tier, NOTES[key[src]]))
    m = monthly[monthly["source"] == SOURCE["nareit"]]
    rows.insert(0, (ASSET, m.index.min(), m.index.max(), "M", SOURCE["nareit"], "C",
                    NOTES["nareit"]))
    df = pd.DataFrame(rows, columns=["asset", "start", "end", "freq", "source", "tier", "notes"])
    return df.sort_values("start").reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-download raw files")
    ap.add_argument("--no-diag", action="store_true", help="skip splice diagnostics")
    a = ap.parse_args(argv)
    daily, monthly = build(a.refresh)
    pd.set_option("display.width", 200)
    print(f"daily   {daily.index.min().date()} .. {daily.index.max().date()}  n={len(daily)}")
    print(f"monthly {monthly.index.min().date()} .. {monthly.index.max().date()}  n={len(monthly)}")
    print(segments(daily, monthly).drop(columns="notes").to_string(index=False))
    print(vol_by_decade(monthly, daily).round(4).to_string())
    if not a.no_diag:
        diag = diagnostics(a.refresh)
        for freq, dd in diag.items():
            print(f"\n{freq} overlap diagnostics (a~b: vol_ratio = vol_a/vol_b, te annualized)")
            print(pd.DataFrame(dd).T.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
