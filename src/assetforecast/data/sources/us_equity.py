"""US equity total returns: ``us_large`` (SPY-like) and ``us_small`` (IWM-like).

Run ``python -m assetforecast.data.sources.us_equity`` to (re)build
``data/processed/assets/us_{large,small}_{daily,monthly}.parquet``.
Raw downloads are cached under ``data/raw/us_equity/``; pass ``--refresh`` to
re-download them.

Construction (justified in ``reports/data/us_equity.md``)
--------------------------------------------------------
us_large
  * 1871-01 .. 1926-06  monthly, Shiller S&P Composite (Cowles) total return
                        from month-AVERAGE prices plus dividends/12 -> tier D
  * 1926-07-01 .. 1993-01-29  daily, Ken French CRSP value-weighted top-30%
                        size portfolio ("Hi 30", NYSE breakpoints) -> tier B
  * 1993-02-01 .. today daily, SPDR S&P 500 ETF (SPY), close + dividends -> tier A
us_small
  * 1926-07-01 .. 2000-05-26  daily, 50/50 daily-rebalanced blend of the French
                        value-weighted size quintiles 2 and 3 ("Qnt 2+3",
                        NYSE 20th-60th percentiles) -> tier B
  * 2000-05-30 .. today daily, iShares Russell 2000 ETF (IWM), close + dividends -> tier A

Tiers (project-wide): A = daily price of an exchange-traded fund (market
close); B = daily open-end mutual-fund NAV, index, or reconstruction; C =
monthly index total return; D = monthly model-based reconstruction.

ETF dividends are validated against the ex-date price move
(``distributions.validate_distributions``); the confirmed bad SPY close of
2001-04-30 is dropped (``BAD_PRICES``). The French portfolios are fixed
choices; the report shows they are also the best-tracking candidates on
overlaps that end in 2014 (no holdout data needed to select them).

Yahoo histories of VFINX (1980-) and NAESX (Russell 2000 fund 1990-2003) were
evaluated and rejected: misdated / stale NAVs and capital-gain distributions
missing from the dividend events (see the report). They are used only as
diagnostics.

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
from . import distributions as dist

CACHE = RAW / "us_equity"
UA = "Mozilla/5.0"  # a full Chrome UA string is rate-limited from this network

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
LARGE_FUNDS = [("SPY", None)]
SMALL_FUNDS = [("IWM", None)]
# Evaluated but rejected (data quality); downloaded for diagnostics only.
REJECTED_FUNDS = ["VFINX", "NAESX"]

# French size portfolios used before the fund era, chosen by the calibration in
# ``diagnostics()`` (tracking error, correlation and vol ratio vs SPY and vs
# IWM / NAESX on the overlap); see the report.
SMALL_PORTFOLIO = "Qnt 2+3"
LARGE_PORTFOLIO = "Hi 30"


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
    # fixed 50/50 blend, rebalanced daily (see SMALL_PORTFOLIO)
    out["Qnt 2+3"] = 0.5 * out["Qnt 2"] + 0.5 * out["Qnt 3"]
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
    df = df.dropna(subset=["adjclose"])
    dv, _ = dist.dedupe_repeats(dv.groupby(level=0).sum()) if len(dv) else (dv, [])
    df["div"] = dist.align_dividends(dv, df.index)
    return df


def yahoo_returns(sym: str, refresh: bool = False) -> pd.Series:
    """Total return from close + cash dividend on the ex-date: (C_t + D_t) / C_{t-1} - 1.

    Yahoo's ``close`` is split-adjusted, not dividend-adjusted. ``adjclose`` is
    checked against this (``adjclose_check``) rather than trusted. For the ETFs
    used (``ETF_PEERS``) every dividend is validated against the ex-date price
    move (``distributions.validate_distributions``), and confirmed bad closes
    (``BAD_PRICES``) are dropped so the next return spans them.
    """
    return yahoo_returns_checked(sym, refresh)[0]


# Confirmed bad closes (price dropped, never interpolated). SPY 2001-04-30: Yahoo 126.66
# (+0.70%) then +0.31% on 05-01; the S&P 500 index (Yahoo ^GSPC) moved -0.29% then
# +1.36%, CRSP Hi-30 -0.08% then +1.41%; markets.ft.com's close (124.91, -0.69%) is
# wrong the other way. All agree on the two-day move (+1.0-1.1%), so only that day's
# price is bad.
BAD_PRICES = {"SPY": ["2001-04-30"]}
# Peers for the dividend test: CRSP market (French) and the other ETF's total return.
ETF_PEERS = {"SPY": ["IWM"], "IWM": ["SPY"]}


def yahoo_returns_checked(sym: str, refresh: bool = False) -> tuple[pd.Series, pd.DataFrame | None]:
    df = yahoo_series(sym, refresh)
    table = None
    if sym in ETF_PEERS:
        peers = {p: yahoo_series(p, refresh)["adjclose"].pct_change() for p in ETF_PEERS[sym]}
        peers["Mkt"] = french_daily(refresh)["Mkt"]
        div, table = dist.validate_distributions(df["close"], df["div"], pd.DataFrame(peers))
        df = df.assign(div=div)
    df, _ = dist.drop_bad_prices(df, BAD_PRICES.get(sym, []))
    r = ((df["close"] + df["div"]) / df["close"].shift(1) - 1).iloc[1:]
    r.name = sym
    return r, table


def adjclose_check(sym: str, refresh: bool = False) -> float:
    """Max |adjclose return - close+div return| (should be a rounding-size number)."""
    df = yahoo_series(sym, refresh)
    a = df["adjclose"].pct_change().iloc[1:]
    return float((a - yahoo_returns(sym, refresh)).abs().max())


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


def last_complete_month(last_day: pd.Timestamp) -> pd.Timestamp:
    """Month-end of the last month whose final weekday is in the data."""
    if last_day >= last_day + pd.offsets.BMonthEnd(0):
        return last_day + pd.offsets.MonthEnd(0)
    return last_day - pd.offsets.MonthEnd(1)


def to_monthly(daily: pd.DataFrame, through: pd.Timestamp) -> pd.DataFrame:
    """Compound daily to calendar month-ends; source/tier of the segment with most days."""
    d = daily.loc[:through]
    g = d.groupby(pd.Grouper(freq="ME"))
    m = pd.DataFrame({
        "ret": g["ret"].apply(lambda x: float(np.prod(1 + x.to_numpy()) - 1)),
        "source": g["source"].agg(lambda x: x.value_counts().index[0]),
        "tier": g["tier"].agg(lambda x: x.value_counts().index[0]),
    })
    m.index.name = "date"
    return m


def _segments_of(df: pd.DataFrame, asset: str, freq: str) -> list[Segment]:
    key = df["source"].ne(df["source"].shift()).cumsum()
    return [Segment(asset, g.index[0], g.index[-1], freq, g["source"].iloc[0], g["tier"].iloc[0])
            for _, g in df.groupby(key, sort=False)]


def _src_name(port: str) -> str:
    if port == "Mkt":
        return "french_mkt_vw"
    if port == "Qnt 2+3":
        return "french_me_vw_qnt2_qnt3_5050"
    return "french_me_vw_" + port.replace(" ", "").replace("-", "").lower()


SHILLER_SRC = "shiller_sp_composite_avgprice"

# asset -> (French portfolio, [(fund symbol, first return date used), ...])
SPECS = {
    "us_large": (LARGE_PORTFOLIO, LARGE_FUNDS),
    "us_small": (SMALL_PORTFOLIO, SMALL_FUNDS),
}


def _short(s: dict) -> str:
    return (f"{s['start']}..{s['end']}: corr {s['corr']:.3f}, vol ratio {s['vol_ratio']:.3f}, "
            f"TE {s['te']:.1%}")


def build(refresh: bool = False, write: bool = True) -> dict:
    fd = french_daily(refresh)
    fm = french_monthly(refresh)
    syms = sorted({s for _, f in SPECS.values() for s, _ in f})
    fund = {s: yahoo_returns(s, refresh) for s in syms}
    res: dict = {"segments": [], "french_daily": fd, "fund": fund}

    for asset, (port, funds) in SPECS.items():
        fr = fd[port]
        pieces = [(fr, _src_name(port), "B")]
        raw = {_src_name(port): fr}
        for sym, start in funds:
            r = fund[sym].loc[start:] if start else fund[sym]
            pieces.append((r, f"yahoo_{sym}", "A"))
            raw[f"yahoo_{sym}"] = fund[sym]
        daily = _splice(pieces)
        monthly = to_monthly(daily, last_complete_month(daily.index[-1]))
        segs = _segments_of(daily, asset, "D")

        # overlap notes ------------------------------------------------
        last_fund = raw[f"yahoo_{funds[-1][0]}"]
        segs[0].notes = (f"CRSP via Ken French ({port}); vs {funds[-1][0]} daily {_short(overlap_stats(fr, last_fund))}; "
                         f"3-day {_short(overlap_stats(fr, last_fund, 'D3'))}")
        for k in range(1, len(segs)):
            cur, prev = raw[segs[k].source], raw[segs[k - 1].source]
            segs[k].notes = (f"vs {segs[k - 1].source} daily {_short(overlap_stats(cur, prev))}; "
                             f"3-day {_short(overlap_stats(cur, prev, 'D3'))}")

        if asset == "us_large":
            sh = shiller_monthly(refresh)
            pre = pd.DataFrame({"ret": sh.loc[:"1926-06-30"], "source": SHILLER_SRC, "tier": "D"})
            st = overlap_stats(sh, fm["Mkt"], "M")
            ann = pd.concat([np.log1p(sh), np.log1p(fm["Mkt"])], axis=1).dropna().resample("YE").sum()
            ann_ratio = ann.iloc[:, 0].std() / ann.iloc[:, 1].std()
            seg0 = _segments_of(pre, asset, "M")[0]
            seg0.notes = (f"month-AVERAGE prices (Cowles pre-1926) -> volatility damped; vs French Mkt monthly "
                          f"{_short(st)}; annual (calendar-year) vol ratio {ann_ratio:.3f}")
            monthly = pd.concat([pre, monthly])
            segs = [seg0] + segs
            res["shiller_overlap"] = dict(monthly=st, annual_ratio=ann_ratio, annual_corr=ann.corr().iloc[0, 1])

        daily["tier"] = daily["tier"].astype(str)
        monthly["tier"] = monthly["tier"].astype(str)
        monthly.index.name = "date"
        res[asset] = (daily, monthly)
        res["segments"] += segs
        if write:
            ASSET_DIR.mkdir(parents=True, exist_ok=True)
            daily.to_parquet(ASSET_DIR / f"{asset}_daily.parquet")
            monthly.to_parquet(ASSET_DIR / f"{asset}_monthly.parquet")
    if write:
        segments_frame(res).to_csv(CACHE / "segments_us_equity.csv", index=False)
    return res


def segments_frame(res: dict) -> pd.DataFrame:
    """Segments in the ``segments.csv`` layout (asset, start, end, freq, source, tier, notes)."""
    return pd.DataFrame([dict(asset=s.asset, start=s.start.date(), end=s.end.date(), freq=s.freq,
                              source=s.source, tier=s.tier, notes=s.notes) for s in res["segments"]])


# -------------------------------------------------------------------- report
KNOWN_DATES = ["1929-10-28", "1929-10-29", "1987-10-19", "1987-10-20", "2008-10-13", "2008-10-15",
               "2020-03-12", "2020-03-16", "2020-03-24"]


def _md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
           "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join([str(i.date() if hasattr(i, "date") else i)] + cells) + " |")
    return "\n".join(out)


def diagnostics(res: dict) -> dict:
    fd = res["french_daily"]
    fund = dict(res["fund"])
    for sym in REJECTED_FUNDS:
        fund[sym] = yahoo_returns(sym)
    spy, iwm = fund["SPY"], fund["IWM"]
    naesx = fund["NAESX"].loc["1990-01-02":"2000-05-25"]
    end = fd.index[-1]
    rows = []
    for c in fd.columns.drop("RF"):
        row = {"portfolio": c}
        for ref, rs in (("SPY", spy), ("IWM", iwm), ("NAESX90-00", naesx)):
            s1 = overlap_stats(fd[c], rs.loc[:end])
            s3 = overlap_stats(fd[c], rs.loc[:end], "D3")
            row |= {f"{ref} corr": s1["corr"], f"{ref} volratio": s1["vol_ratio"],
                    f"{ref} TE": s1["te"], f"{ref} corr3d": s3["corr"], f"{ref} volratio3d": s3["vol_ratio"]}
        rows.append(row)
    calib = pd.DataFrame(rows).set_index("portfolio")

    # sub-period calibration of the chosen portfolios (3-day returns)
    sub = []
    for ref, rs, cands in (("SPY", spy, ["Mkt", "Hi 20", LARGE_PORTFOLIO]),
                           ("IWM", iwm, ["Qnt 2", "Qnt 3", "Lo 30", SMALL_PORTFOLIO])):
        for a_, b_ in (("1993", "1999"), ("2000", "2009"), ("2010", "2019"), ("2020", "2026")):
            for c in cands:
                x = rs.loc[a_:b_]
                if len(x) < 200:
                    continue
                st = overlap_stats(fd[c], x.loc[:end], "D3")
                sub.append({"ref / period / portfolio": f"{ref} {a_}-{b_} {c}", "corr3d": st["corr"],
                            "volratio3d": st["vol_ratio"], "TE3d": st["te"]})
    sub = pd.DataFrame(sub).set_index("ref / period / portfolio")

    # fund data quality: date alignment, stale days, outliers vs French same day
    q = []
    windows = {"SPY": (spy, LARGE_PORTFOLIO, None, None),
               "VFINX": (fund["VFINX"], LARGE_PORTFOLIO, "1980", "1993-01-29"),
               "IWM": (iwm, SMALL_PORTFOLIO, None, None),
               "NAESX": (fund["NAESX"], SMALL_PORTFOLIO, "1990", "2000-05-26")}
    worst = []
    for sym, (r, port, a_, b_) in windows.items():
        r = r.loc[a_:b_].loc[:end]
        d = pd.concat([r, fd[port]], axis=1, keys=["f", "fr"]).dropna()
        cor = {k: pd.concat([d["fr"], d["f"].shift(k)], axis=1).dropna().corr().iloc[0, 1] for k in (-1, 0, 1)}
        diff = d["f"] - d["fr"]
        z = (diff - diff.median()) / (1.4826 * (diff - diff.median()).abs().median())
        q.append({"fund (window)": f"{sym} ({d.index[0].date()}..{d.index[-1].date()})", "vs": port,
                  "corr fund t-1": cor[1], "corr same day": cor[0], "corr fund t+1": cor[-1],
                  "zero-return days": int((d["f"] == 0).sum()), "days robust z>8": int((z.abs() > 8).sum()),
                  "days |diff|>2%": int((diff.abs() > 0.02).sum())})
        for t in diff.abs().sort_values().index[-4:][::-1]:
            worst.append({"date": t, "fund": sym, "fund ret": d.loc[t, "f"], "French": port, "French ret": d.loc[t, "fr"]})
    quality = pd.DataFrame(q).set_index("fund (window)")
    worst = pd.DataFrame(worst).set_index("date")
    return dict(calib=calib, sub=sub, quality=quality, worst=worst)


HOLDOUT_START = "2015-01-01"   # evaluation holdout: selections must not depend on it


def holdout_check(res: dict) -> pd.DataFrame:
    """Selection of the French portfolios re-done on overlaps that END before the
    holdout: 3-day tracking error / vol ratio / correlation vs SPY (1993-2014) and
    IWM (2000-2014) for the candidates. The chosen portfolios must still rank as
    they do on the full overlap."""
    fd, fund = res["french_daily"], res["fund"]
    end = pd.Timestamp(HOLDOUT_START) - pd.Timedelta(days=1)
    rows = []
    for ref, cands in (("SPY", ["Mkt", "Hi 10", "Hi 20", LARGE_PORTFOLIO]),
                       ("IWM", ["Qnt 2", "Qnt 3", "Lo 30", "4-Dec", SMALL_PORTFOLIO])):
        for c in cands:
            for lab, e in (("to 2014", end), ("full", fd.index[-1])):
                st = overlap_stats(fd[c], fund[ref].loc[:e].loc[:fd.index[-1]], "D3")
                rows.append({"ref / portfolio / overlap": f"{ref} {c} {lab}", "TE3d": st["te"],
                             "volratio3d": st["vol_ratio"], "corr3d": st["corr"]})
    return pd.DataFrame(rows).set_index("ref / portfolio / overlap")


def vol_by_decade(res: dict) -> pd.DataFrame:
    rows = {}
    for asset in SPECS:
        d, m = res[asset]
        key_d, key_m = (d.index.year // 10) * 10, (m.index.year // 10) * 10
        dl = d["ret"].groupby(key_d)
        rows[f"{asset} daily-based"] = dl.std() * np.sqrt(252)
        rows[f"{asset} monthly-based"] = m["ret"].groupby(key_m).std() * np.sqrt(12)
        rows[f"{asset} daily AC1"] = dl.apply(lambda x: x.autocorr())
        rows[f"{asset} days/yr"] = dl.size() / d.index.year.to_series().groupby(key_d).nunique().values
    out = pd.DataFrame(rows).sort_index()
    out.index = [f"{i}s" for i in out.index]
    out.index.name = "decade"
    return out


def splice_checks(res: dict) -> pd.DataFrame:
    rows = []
    for asset in SPECS:
        d = res[asset][0]
        vol = d["ret"].rolling(63).std().shift(1)
        for t in d.index[d["source"].ne(d["source"].shift())][1:]:
            prev = d.index[d.index.get_loc(t) - 1]
            rows.append({"asset / first day": f"{asset} {t.date()}", "from": d.loc[prev, "source"],
                         "to": d.loc[t, "source"], "ret first day": d.loc[t, "ret"],
                         "z vs trailing 63d vol": d.loc[t, "ret"] / vol.loc[t]})
    return pd.DataFrame(rows).set_index("asset / first day")


def write_report(res: dict, path: Path | None = None) -> Path:
    from ...paths import REPORTS
    path = path or REPORTS / "data" / "us_equity.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    dg = diagnostics(res)
    seg = segments_frame(res)
    L, S = res["us_large"], res["us_small"]
    kd = pd.DataFrame({
        "us_large": L[0]["ret"].reindex(pd.to_datetime(KNOWN_DATES)),
        "us_large source": L[0]["source"].reindex(pd.to_datetime(KNOWN_DATES)),
        "us_small": S[0]["ret"].reindex(pd.to_datetime(KNOWN_DATES)),
        "us_small source": S[0]["source"].reindex(pd.to_datetime(KNOWN_DATES)),
    })
    kd.index.name = "date"
    mon = pd.DataFrame({"us_large": L[0]["ret"].groupby(pd.Grouper(freq="ME")).apply(lambda x: np.prod(1 + x) - 1),
                        "us_small": S[0]["ret"].groupby(pd.Grouper(freq="ME")).apply(lambda x: np.prod(1 + x) - 1)})
    mon = mon.reindex(pd.to_datetime(["1929-10-31", "1987-10-31", "2008-10-31", "2020-03-31"]))
    mon.index.name = "month"
    vd = vol_by_decade(res)
    sh = res["shiller_overlap"]
    seg_md = ["| asset | start | end | freq | source | tier | overlap notes |", "|---|---|---|---|---|---|---|"]
    for _, r in seg.iterrows():
        seg_md.append(f"| {r.asset} | {r.start} | {r.end} | {r.freq} | {r.source} | {r.tier} | {r.notes} |")
    calib = dg["calib"]
    pct = calib.copy()
    txt = TEMPLATE.format(
        today=pd.Timestamp.today().date(),
        large_d0=L[0].index[0].date(), large_d1=L[0].index[-1].date(),
        large_m0=L[1].index[0].date(), large_m1=L[1].index[-1].date(),
        small_d0=S[0].index[0].date(), small_d1=S[0].index[-1].date(),
        small_m0=S[1].index[0].date(), small_m1=S[1].index[-1].date(),
        french_end=res["french_daily"].index[-1].date(),
        small_port=SMALL_PORTFOLIO, large_port=LARGE_PORTFOLIO,
        segments="\n".join(seg_md),
        calib_large=_md(pct[[c for c in pct.columns if c.startswith("SPY")]], "{:.3f}"),
        calib_small=_md(pct[[c for c in pct.columns if c.startswith(("IWM", "NAESX"))]], "{:.3f}"),
        sub=_md(dg["sub"], "{:.3f}"),
        holdout=_md(holdout_check(res), "{:.3f}"),
        bad=", ".join(f"{s} {d}" for s, v in BAD_PRICES.items() for d in v),
        quality=_md(dg["quality"], "{:.3f}"),
        worst=_md(dg["worst"], "{:+.4f}"),
        splice=_md(splice_checks(res), "{:+.4f}"),
        adj="\n".join(f"* {k}: {adjclose_check(k):.5f}" for k in ["SPY", "IWM"] + REJECTED_FUNDS),
        known=_md(kd, "{:+.4f}"), known_m=_md(mon, "{:+.4f}"),
        vol=_md(vd, "{:.3f}"),
        sh_corr=sh["monthly"]["corr"], sh_ratio=sh["monthly"]["vol_ratio"],
        sh_aratio=sh["annual_ratio"], sh_acorr=sh["annual_corr"],
    )
    path.write_text(txt)
    return path


TEMPLATE = """# US equity data: `us_large`, `us_small`

Generated by `python -m assetforecast.data.sources.us_equity --report` on {today}.
Loader: `src/assetforecast/data/sources/us_equity.py`; raw cache: `data/raw/us_equity/`.
All returns are SIMPLE TOTAL returns (dividends reinvested), in decimals, not in excess of cash.

## Coverage

| asset | daily | monthly |
|---|---|---|
| us_large | {large_d0} .. {large_d1} | {large_m0} .. {large_m1} |
| us_small | {small_d0} .. {small_d1} | {small_m0} .. {small_m1} |

The monthly files contain complete calendar months only, so there is no partial current month. In the daily era each
monthly value is the compounded daily return (a month that spans a splice takes the source and tier of the segment with
the most days in it). French data ends {french_end}. From there the ETF segments carry both series to the latest Yahoo
close.

## Sources

* **Ken French data library** (CRSP-based):
  * `F-F_Research_Data_Factors_daily` gives the CRSP value-weighted market (Mkt-RF + RF). It is used here only for
    calibration.
  * `Portfolios_Formed_on_ME_daily` gives value-weighted size portfolios formed on NYSE market-cap breakpoints and
    rebalanced each June. Stocks are NYSE only before 1962, AMEX is added in 1962 and Nasdaq in 1972.
  * Returns are total returns in percent, rounded to 2 decimals (1 bp).
* **Yahoo chart API** (`query1.finance.yahoo.com/v8/finance/chart`, User-Agent `Mozilla/5.0`) for SPY and IWM. Each
  return is computed as (close_t + dividend_t) / close_(t-1) - 1, where close is split-adjusted and the dividend is
  dated on its ex-date. Timestamps are converted to New York trading dates. Every dividend is validated against
  the ex-date price move relative to the CRSP market and the other ETF (`distributions.validate_distributions`;
  none rejected). Confirmed bad closes are dropped so the next return spans them: {bad} (Yahoo +0.70% then +0.31%;
  the S&P 500 index moved -0.29% then +1.36% and CRSP Hi 30 -0.08% then +1.41%; markets.ft.com's close is wrong
  the other way; all agree on the two-day move). Yahoo's `adjclose` was checked against this calculation and was
  not used directly. The largest absolute daily difference for each fund is (SPY's is the dropped bad close):
{adj}
* **Shiller** `ie_data.xls`: monthly S&P Composite (Cowles Commission before 1926). Its price is the average of daily
  closes in the month, and its dividends are annualized (interpolated from annual data before 1926). The file runs to
  2023-06.

## Construction and why

**us_large (SPY-like).** From 1926-07 until SPY's first full return day (1993-02-01), the series is the French
value-weighted **`{large_port}`** size portfolio: stocks above the NYSE 70th percentile of market cap. It was chosen
over the whole CRSP market (Mkt) because, on the SPY overlap, it tracks SPY better in every sub-period:
* Tracking error is lower: 2.3% vs 3.0% on 3-day returns over 1993-2026, and 0.7% vs 1.5% over 2010-19.
* Since 2010 its vol ratio is closer to 1: 1.00-1.02, against 1.04 for Mkt. Mkt's small-cap tail adds volatility that
  the S&P 500 does not have.

**us_small (IWM-like).** IWM tracks the Russell 2000, roughly the stocks ranked 1001-3000 by market cap, value
weighted. Every French size portfolio, and a few simple blends, were compared with IWM (2000-2026) and with NAESX
during its Russell-2000 years (1990-2000). The chosen series is **`{small_port}`**: a fixed 50/50 blend of the
value-weighted size quintiles 2 and 3 (NYSE 20th-60th percentiles), rebalanced daily. Against IWM it has the highest
correlation (0.990 on 3-day returns), the lowest tracking error (3.2%) and a vol ratio of 0.987. Its vol ratio stays
between 0.96 and 1.01 in every sub-period.
* Quintile 2 alone tracks nearly as well, but it runs 4-5% too volatile against IWM since 2010.
* "Lo 30" (which includes microcaps) has a 5-6% tracking error.

The breakpoints are NYSE-based throughout, so the series is the same kind of exposure in 1926 as in 2026.

**Fund histories rejected.** Both funds were downloaded and tested as bridges to an earlier tier A start (see "Fund
data quality" below). Neither was credible as a daily series:
* **VFINX** (Vanguard 500, 1980-1993):
  * Large negative returns on distribution dates, for example -5.6% on 1985-12-27 (the market was up 1%) and -7.0% on
    1986-12-09. These are capital-gain distributions missing from Yahoo's dividend events.
  * About 7-8 stale (zero-return) days a year, typically followed by a catch-up day (for example 1987-04-21/22).
* **NAESX** (Vanguard Small-Cap Index, which tracked the Russell 2000 in 1990-2003): returns correlate 0.22-0.26 with
  French returns one day earlier and one day later. That means many prices are misdated by a day.

Using either fund would add artificial daily volatility and errors in the lead-lag structure. That costs more than
the gain from upgrading the label from tier B to tier A.

**Splices** join returns, never price levels, so a level jump cannot arise. The return on the first day of each new
segment is checked below and in the tests.

## Segments

{segments}

### Splice-day check

{splice}

## Calibration of French portfolios against the ETFs

Measured on daily returns over the full overlap, up to the end of the French data. `corr`, `volratio`
(French / fund) and `TE` (annualized tracking error) use same-day daily returns. `corr3d` and `volratio3d` use
non-overlapping 3-day returns, which are robust to differences in closing times.

### vs SPY (1993-02 .. {french_end})

{calib_large}

### vs IWM (2000-05 .. {french_end}) and NAESX (1990-2000, Russell 2000 era)

{calib_small}

### Sub-periods (3-day returns)

{sub}

### Holdout independence of the choice

The evaluation holdout starts 2015-01. Re-doing the comparison on overlaps that end in 2014 (3-day returns)
gives the same ranking: for SPY, Hi 20 and Hi 30 tie on tracking error and Hi 30's vol ratio is closer to 1;
for IWM, Qnt 2+3 has the lowest tracking error. So the portfolio choice does not depend on post-2014 data.

{holdout}

## Fund data quality and date alignment

Each fund is compared with the French portfolio on the same day and with the fund shifted one day either way. A peak
at "same day" with near-zero neighbours means both series are dated by the trading date of the return. SPY and IWM
pass this check, while NAESX does not.

{quality}

The largest |fund - French| same-day differences per fund (the SPY and IWM entries are genuine market days with fast
moves; the timing of the closing print and the 1 bp rounding in French data explain them):

{worst}

## Known dates (daily simple total returns)

{known}

Monthly:

{known_m}

## Annualized volatility by decade

* `daily-based` = std of daily returns x sqrt(252).
* `monthly-based` = std of monthly returns x sqrt(12).
* `daily AC1` = first-order autocorrelation of daily returns.
* `days/yr` = trading days per year. Saturday sessions before 1952 raise it to about 290.

{vol}

## Shiller (pre-1926) damping

Over 1926-07..2023 the Shiller monthly return has a correlation of {sh_corr:.3f} with the French market and a vol
ratio of {sh_ratio:.3f}. Because prices are averaged over the month, each "monthly return" is roughly a two-month
moving average of true returns. That cuts monthly volatility by about 15-20% and induces positive autocorrelation. On
calendar-year log returns the vol ratio is {sh_aratio:.3f} (corr {sh_acorr:.3f}), so the bias is mostly an artifact of
monthly measurement.

The segment is tier D. Models should not treat its monthly variance as comparable to tier A/B data without a
correction (for example, scale monthly variance by about 1/{sh_ratio:.2f}^2). No credible pre-1926 small-cap series
exists, so `us_small` starts in 1926-07.

## Caveats (read these if you model volatility)

1. **Daily vol is understated in index data before about 2000, most of all for small caps.** Daily CRSP portfolio
   returns before 2000 are positively autocorrelated because illiquid stocks have stale closing prices.
   * For `us_small`, daily AC1 is 0.25-0.41 from the 1960s to the 1990s, and monthly-based vol is 35-70% above
     daily-based vol in those decades.
   * For `us_large`, AC1 is 0.13-0.26 from the 1940s to the 1970s, and the ratio is 10-25%.
   * After 2000, when the ETFs take over, AC1 is about 0 or slightly negative, and monthly-based vol is below
     daily-based vol (mean reversion within the month).

   Variances from 1-day returns (the `cov_rv` target) are therefore not on the same scale before and after
   2000. A target that uses multi-day (for example 3- or 5-day) returns for variances, or a variance-ratio
   correction, would be more consistent. ARCHITECTURE.md already applies 3-day returns to correlations.
2. **Saturday trading before 1952.** French daily data includes Saturday sessions, so there are about 290 trading
   days a year before 1952. Annualizing with 252 understates annual variance in that era by roughly 290/252, about
   15% in variance.
3. Exchange closures: the 1933 bank holiday (no rows 1933-03-04..03-14; the 1933-03-15 return of +15% spans it) and 2001-09-11..14. The return on the
   reopening day spans the whole closure. The 1914 closure (1914-07-31 to 12-12) falls inside the Shiller monthly data.
4. French portfolio returns before 1962 are NYSE-only, and the size breakpoints are NYSE breakpoints throughout.
   The series are representative portfolios built from the same universe that indexes draw on, not the S&P 500 or
   Russell 2000 themselves. The ETF segments carry expense ratios (0.09%/yr for SPY, about 0.19%/yr for IWM) and IWM
   carries Russell reconstitution effects. Mean differences against French are +0.2%/yr (Hi 30 - SPY) and
   +1.4%/yr (Qnt 2+3 - IWM). These matter for means, not for vol.
5. French daily data is rounded to 1 bp. The added noise is negligible (rounding variance is about 1e-9 per day).
6. The Shiller pre-1926 dividends are interpolated from annual figures, and its prices are monthly averages. That
   segment is tier D.
7. SPY's first returns (1993) include some zero-return days on low early volume. They are kept as recorded.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-download raw files")
    ap.add_argument("--report", action="store_true", help="also write reports/data/us_equity.md")
    a = ap.parse_args(argv)
    res = build(refresh=a.refresh)
    for s in res["segments"]:
        print(f"{s.asset:9s} {s.freq} {s.start.date()} .. {s.end.date()} {s.tier} {s.source}\n    {s.notes}")
    if a.report:
        print("wrote", write_report(res))


if __name__ == "__main__":
    main()
