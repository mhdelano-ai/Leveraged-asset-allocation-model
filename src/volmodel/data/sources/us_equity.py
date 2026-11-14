"""US equity total returns: ``us_large`` (SPY-like) and ``us_small`` (IWM-like).

Run ``python -m volmodel.data.sources.us_equity`` to (re)build
``data/processed/assets/us_{large,small}_{daily,monthly}.parquet``.
Raw downloads are cached under ``data/raw/us_equity/``; pass ``--refresh`` to
re-download them.

Construction (justified in ``reports/data/us_equity.md``)
--------------------------------------------------------
us_large
  * 1871-01 .. 1926-06  monthly, Shiller S&P Composite (Cowles) total return
                        from month-AVERAGE prices plus dividends/12 -> tier D
  * 1926-07 .. VFINX-1  daily, Ken French CRSP value-weighted market
                        (Mkt-RF + RF), total return -> tier B
  * VFINX start .. SPY-1 daily, Vanguard 500 Index Inv (VFINX) adjclose -> tier A
  * 1993-01-29 .. today daily, SPDR S&P 500 ETF (SPY) adjclose -> tier A
us_small
  * 1926-07 .. NAESX-index-era-1  daily, Ken French size portfolio chosen by
                        calibration against IWM (see ``SMALL_PORTFOLIO``) -> tier B
  * 1989-12 .. 2000-05-25  daily, Vanguard Small-Cap Index Inv (NAESX), which
                        tracked the Russell 2000 in that era -> tier A
  * 2000-05-26 .. today daily, iShares Russell 2000 ETF (IWM) adjclose -> tier A

Every return is a SIMPLE TOTAL return in decimals. Nothing is interpolated.
"""
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ...paths import ASSET_DIR, RAW

CACHE = RAW / "us_equity"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "ff3_daily": "F-F_Research_Data_Factors_daily_CSV.zip",
    "ff3_monthly": "F-F_Research_Data_Factors_CSV.zip",
    "me_daily": "Portfolios_Formed_on_ME_daily_CSV.zip",
    "me_monthly": "Portfolios_Formed_on_ME_CSV.zip",
}
SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
YAHOO_URL = "https://{host}.finance.yahoo.com/v8/finance/chart/{sym}"

# Fund / ETF segments. ``start`` is the first return date used (None = from the
# first available Yahoo return).
LARGE_FUNDS = [("VFINX", None), ("SPY", "1993-02-01")]
# NAESX became a Russell 2000 index fund on 1989-12-?? (Naess & Thomas Special
# Fund before that): use it only from 1990-01-02.
SMALL_FUNDS = [("NAESX", "1990-01-02"), ("IWM", "2000-05-30")]
# Extra Yahoo series used only for diagnostics.
DIAG_SYMBOLS = ["^SP500TR", "^GSPC", "^RUT"]

# French size portfolio used for us_small before the fund era. Chosen by the
# calibration in ``calibrate()`` (lowest tracking error vs IWM / NAESX with
# vol ratio close to 1); see the report.
SMALL_PORTFOLIO = "Qnt 2"
LARGE_PORTFOLIO = "Mkt"


# --------------------------------------------------------------------- fetch
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "*/*"})
    return s


def _get(url: str, params: dict | None = None, tries: int = 8, sleep: float = 2.0) -> bytes:
    s = _session()
    last = None
    for i in range(tries):
        try:
            r = s.get(url, params=params, timeout=60)
            if r.status_code == 200 and r.content:
                return r.content
            last = f"HTTP {r.status_code}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:  # pragma: no cover - network
            last = repr(e)
        time.sleep(sleep * (2 ** i) + np.random.uniform(0, 1.5))
    raise RuntimeError(f"download failed for {url}: {last}")


def cached(name: str, url: str, refresh: bool = False, params: dict | None = None) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if path.exists() and not refresh:
        return path.read_bytes()
    data = _get(url, params=params)
    path.write_bytes(data)
    return data


def french_zip(key: str, refresh: bool = False) -> str:
    raw = cached(FRENCH_FILES[key], FRENCH_BASE + FRENCH_FILES[key], refresh)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        return z.read(z.namelist()[0]).decode("latin-1")


def yahoo_json(sym: str, refresh: bool = False) -> dict:
    name = f"yahoo_{sym.replace('^', 'IDX_')}.json"
    path = CACHE / name
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    params = {"period1": 0, "period2": int(time.time()) + 86400,
              "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"}
    err = None
    for host in ("query1", "query2"):
        try:
            raw = _get(YAHOO_URL.format(host=host, sym=sym), params=params, tries=6, sleep=3.0)
            d = json.loads(raw)
            if d.get("chart", {}).get("result"):
                CACHE.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                time.sleep(1.0)  # be polite: other agents share this endpoint
                return d
            err = str(d.get("chart", {}).get("error"))
        except RuntimeError as e:
            err = str(e)
        time.sleep(2.0)
    raise RuntimeError(f"Yahoo chart failed for {sym}: {err}")


# --------------------------------------------------------------------- parse
def parse_french(text: str, section: str) -> pd.DataFrame:
    """Return one section of a French CSV as decimals, indexed by date.

    ``section`` is a substring of the section title line ("" = first table).
    Daily tables are indexed by trading date, monthly tables by month-end.
    """
    lines = text.splitlines()
    start = 0
    if section:
        start = next(i for i, l in enumerate(lines) if section in l)
    hdr = next(i for i in range(start, len(lines)) if lines[i].startswith(","))
    cols = [c.strip() for c in lines[hdr].split(",")[1:]]
    rows = []
    for l in lines[hdr + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if not parts[0].isdigit():
            break
        rows.append(parts)
    df = pd.DataFrame([r[1:] for r in rows], index=[r[0] for r in rows], columns=cols).astype(float)
    n = len(df.index[0])
    if n == 8:
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
    elif n == 6:
        df.index = pd.to_datetime(df.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        raise ValueError(f"unexpected date code {df.index[0]!r}")
    df.index.name = "date"
    df = df.mask(df <= -99.99)
    return df / 100.0


def french_daily(refresh: bool = False) -> pd.DataFrame:
    """Daily total returns: 'Mkt' (CRSP VW market) and all VW size portfolios."""
    ff = parse_french(french_zip("ff3_daily", refresh), "")
    me = parse_french(french_zip("me_daily", refresh), "Value Weighted Returns -- Daily")
    me = me.rename(columns=lambda c: c.replace("Dec ", "Dec") if c.startswith("Dec ") else c)
    me = me.rename(columns={f"Dec{i}": f"{i}-Dec" for i in range(2, 10)})
    out = me.drop(columns=["<= 0"], errors="ignore")
    out.insert(0, "Mkt", ff["Mkt-RF"] + ff["RF"])
    out.insert(1, "RF", ff["RF"])
    return out


def french_monthly(refresh: bool = False) -> pd.DataFrame:
    ff = parse_french(french_zip("ff3_monthly", refresh), "")
    me = parse_french(french_zip("me_monthly", refresh), "Value Weight Returns -- Monthly")
    out = me.drop(columns=["<= 0"], errors="ignore")
    out.insert(0, "Mkt", ff["Mkt-RF"] + ff["RF"])
    out.insert(1, "RF", ff["RF"])
    return out


def yahoo_series(sym: str, refresh: bool = False) -> pd.DataFrame:
    """Daily close / adjclose / dividends indexed by exchange-local trading date."""
    d = yahoo_json(sym, refresh)["chart"]["result"][0]
    tz = d["meta"].get("exchangeTimezoneName", "America/New_York")
    ts = pd.to_datetime(d["timestamp"], unit="s", utc=True).tz_convert(tz)
    q = d["indicators"]["quote"][0]
    adj = d["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    df = pd.DataFrame({"close": q["close"], "adjclose": adj},
                      index=pd.DatetimeIndex(ts.tz_localize(None).normalize(), name="date"))
    df = df[~df.index.duplicated(keep="last")].astype(float)
    div = d.get("events", {}).get("dividends", {})
    dv = pd.Series({pd.Timestamp(v["date"], unit="s", tz="UTC").tz_convert(tz).tz_localize(None).normalize():
                    v["amount"] for v in div.values()}, dtype=float)
    df["div"] = dv.reindex(df.index).fillna(0.0) if len(dv) else 0.0
    return df.dropna(subset=["adjclose"])


def yahoo_returns(sym: str, refresh: bool = False) -> pd.Series:
    a = yahoo_series(sym, refresh)["adjclose"]
    r = a.pct_change().iloc[1:]
    r.name = sym
    return r


def shiller_monthly(refresh: bool = False) -> pd.Series:
    """Shiller S&P Composite monthly total return (month-average prices).

    R_t = (P_t + D_t/12) / P_{t-1} - 1, D annualized dividend. Indexed by
    month-end of month t.
    """
    raw = cached("ie_data.xls", SHILLER_URL, refresh)
    x = pd.read_excel(io.BytesIO(raw), sheet_name="Data", header=None, skiprows=8)
    x = x.iloc[:, :3]
    x.columns = ["date", "P", "D"]
    x = x[pd.to_numeric(x["date"], errors="coerce").notna()].copy()
    d = x["date"].astype(float)
    year = np.floor(d).astype(int)
    month = np.round((d - year) * 100).astype(int)
    x.index = pd.to_datetime(dict(year=year, month=month, day=1)) + pd.offsets.MonthEnd(0)
    p = pd.to_numeric(x["P"], errors="coerce")
    dv = pd.to_numeric(x["D"], errors="coerce")
    r = (p + dv / 12.0) / p.shift(1) - 1
    r.index.name = "date"
    return r.dropna().rename("shiller")


# ---------------------------------------------------------------- diagnostics
def overlap_stats(a: pd.Series, b: pd.Series, freq: str = "D") -> dict:
    """Compare candidate ``a`` against reference ``b`` on their common dates."""
    df = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    ann = 252 if freq == "D" else 12
    if freq == "D3":  # non-overlapping 3-day log sums: robust to close timing
        lg = np.log1p(df)
        df = np.expm1(lg.rolling(3).sum().iloc[2::3])
        ann = 252 / 3
    va, vb = df["a"].std() * np.sqrt(ann), df["b"].std() * np.sqrt(ann)
    return dict(start=df.index[0].date(), end=df.index[-1].date(), n=len(df),
                corr=df["a"].corr(df["b"]), vol_a=va, vol_b=vb, vol_ratio=va / vb,
                te=(df["a"] - df["b"]).std() * np.sqrt(ann),
                mean_diff=(df["a"] - df["b"]).mean() * ann)


def fmt_stats(s: dict) -> str:
    return (f"{s['start']}..{s['end']} n={s['n']}: corr {s['corr']:.4f}, "
            f"vol {s['vol_a']:.2%} vs {s['vol_b']:.2%} (ratio {s['vol_ratio']:.3f}), "
            f"TE {s['te']:.2%}, mean diff {s['mean_diff']:+.2%}/yr")


def calibrate(refresh: bool = False) -> pd.DataFrame:
    """Every French VW portfolio vs the fund series, daily, on their overlap."""
    fd = french_daily(refresh)
    refs = {"SPY": yahoo_returns("SPY", refresh), "VFINX": yahoo_returns("VFINX", refresh),
            "IWM": yahoo_returns("IWM", refresh), "NAESX(1990-2000)":
            yahoo_returns("NAESX", refresh).loc["1990-01-02":"2000-05-25"]}
    rows = []
    for ref, rs in refs.items():
        for c in fd.columns.drop("RF"):
            for f in ("D", "D3"):
                s = overlap_stats(fd[c], rs, f)
                rows.append(dict(ref=ref, portfolio=c, freq=f, **s))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ assemble
@dataclass
class Segment:
    asset: str
    start: pd.Timestamp
    end: pd.Timestamp
    freq: str
    source: str
    tier: str
    notes: str = ""


def _splice(pieces: list[tuple[pd.Series, str, str]]) -> pd.DataFrame:
    """Each later piece overrides earlier ones from its first date onwards."""
    out = None
    for r, src, tier in pieces:
        r = r.dropna()
        df = pd.DataFrame({"ret": r.astype(float), "source": src, "tier": tier})
        out = df if out is None else pd.concat([out.loc[: r.index[0] - pd.Timedelta(days=1)], df])
    out.index.name = "date"
    return out


def to_monthly(daily: pd.DataFrame, through: pd.Timestamp) -> pd.DataFrame:
    """Compound daily to calendar month-ends; source/tier = the dominant segment."""
    g = daily.groupby(pd.Grouper(freq="ME"))
    m = pd.DataFrame({
        "ret": g["ret"].apply(lambda x: np.prod(1 + x.values) - 1 if len(x) else np.nan),
        "source": g["source"].agg(lambda x: x.mode().iloc[0] if len(x) else None),
        "tier": g["tier"].agg(lambda x: x.mode().iloc[0] if len(x) else None),
    })
    return m.loc[:through]


def _segments_of(df: pd.DataFrame, asset: str, freq: str) -> list[Segment]:
    key = df["source"].ne(df["source"].shift()).cumsum()
    segs = []
    for _, g in df.groupby(key, sort=False):
        segs.append(Segment(asset, g.index[0], g.index[-1], freq, g["source"].iloc[0], g["tier"].iloc[0]))
    return segs


SRC = {
    "Mkt": "french_mkt_vw",
    "VFINX": "yahoo_VFINX",
    "SPY": "yahoo_SPY",
    "NAESX": "yahoo_NAESX",
    "IWM": "yahoo_IWM",
    "shiller": "shiller_sp_composite_avgprice",
}


def build(refresh: bool = False, write: bool = True) -> dict:
    fd = french_daily(refresh)
    fund = {s: yahoo_returns(s, refresh) for s, _ in LARGE_FUNDS + SMALL_FUNDS}
    out: dict = {"segments": [], "diag": {}}

    for asset, port, funds in (("us_large", LARGE_PORTFOLIO, LARGE_FUNDS),
                               ("us_small", SMALL_PORTFOLIO, SMALL_FUNDS)):
        src0 = SRC["Mkt"] if port == "Mkt" else "french_me_vw_" + port.replace(" ", "").lower()
        pieces = [(fd[port], src0, "B")]
        for sym, start in funds:
            r = fund[sym]
            if start:
                r = r.loc[start:]
            pieces.append((r, SRC[sym], "A"))
        daily = _splice(pieces)
        last_complete = (daily.index[-1] + pd.offsets.MonthEnd(0)
                         if daily.index[-1].is_month_end or _is_last_trading_day(daily.index[-1])
                         else daily.index[-1] - pd.offsets.MonthEnd(1))
        monthly = to_monthly(daily, last_complete)
        if asset == "us_large":
            sh = shiller_monthly(refresh).loc[:monthly.index[0] - pd.Timedelta(days=1)]
            pre = pd.DataFrame({"ret": sh, "source": SRC["shiller"], "tier": "D"})
            monthly = pd.concat([pre, monthly])
        monthly.index.name = "date"
        daily["tier"] = daily["tier"].astype(str)
        monthly["tier"] = monthly["tier"].astype(str)
        out[asset] = (daily, monthly)
        out["segments"] += _segments_of(daily, asset, "D")
        if asset == "us_large":
            out["segments"] = [_segments_of(monthly.loc[monthly.source == SRC["shiller"]], asset, "M")[0]] + out["segments"]
        if write:
            ASSET_DIR.mkdir(parents=True, exist_ok=True)
            daily.to_parquet(ASSET_DIR / f"{asset}_daily.parquet")
            monthly.to_parquet(ASSET_DIR / f"{asset}_monthly.parquet")
    return out


def segments_frame(res: dict) -> pd.DataFrame:
    """Segments in the ``segments.csv`` layout (asset, start, end, freq, source, tier, notes)."""
    return pd.DataFrame([dict(asset=s.asset, start=s.start.date(), end=s.end.date(), freq=s.freq,
                              source=s.source, tier=s.tier, notes=s.notes) for s in res["segments"]])


def _is_last_trading_day(d: pd.Timestamp) -> bool:
    # conservative: a month counts as complete only if its last weekday is in the data
    last_bd = d + pd.offsets.BMonthEnd(0)
    return d == last_bd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-download raw files")
    a = ap.parse_args(argv)
    res = build(refresh=a.refresh)
    for s in res["segments"]:
        print(f"{s.asset:9s} {s.freq} {s.start.date()} .. {s.end.date()} {s.tier} {s.source}")


if __name__ == "__main__":
    main()
