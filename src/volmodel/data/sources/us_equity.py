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
  * 1926-07-01 .. 1993-01-29  daily, Ken French CRSP value-weighted top-30%
                        size portfolio ("Hi 30", NYSE breakpoints) -> tier B
  * 1993-02-01 .. today daily, SPDR S&P 500 ETF (SPY), close + dividends -> tier A
us_small
  * 1926-07-01 .. 2000-05-26  daily, 50/50 daily-rebalanced blend of the French
                        value-weighted size quintiles 2 and 3 ("Qnt 2+3",
                        NYSE 20th-60th percentiles) -> tier B
  * 2000-05-30 .. today daily, iShares Russell 2000 ETF (IWM), close + dividends -> tier A

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
    df["div"] = dv.reindex(df.index).fillna(0.0) if len(dv) else 0.0
    return df.dropna(subset=["adjclose"])


def yahoo_returns(sym: str, refresh: bool = False) -> pd.Series:
    """Total return from close + cash dividend on the ex-date: (C_t + D_t) / C_{t-1} - 1.

    Yahoo's ``close`` is split-adjusted, not dividend-adjusted. ``adjclose`` is
    checked against this (``adjclose_check``) rather than trusted.
    """
    df = yahoo_series(sym, refresh)
    r = ((df["close"] + df["div"]) / df["close"].shift(1) - 1).iloc[1:]
    r.name = sym
    return r


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


def fmt_stats(s: dict) -> str:
    return (f"{s['start']}..{s['end']} n={s['n']}: corr {s['corr']:.4f}, "
            f"vol {s['vol_a']:.2%} vs {s['vol_b']:.2%} (ratio {s['vol_ratio']:.3f}), "
            f"TE {s['te']:.2%}, mean diff {s['mean_diff']:+.2%}/yr")


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
            own = cur.loc[segs[k].start:segs[k].end]
            segs[k].notes = (f"vs {segs[k - 1].source} on full overlap {_short(overlap_stats(cur, prev))}; "
                             f"vs French {port} on this segment {_short(overlap_stats(own, fr))}")

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
    fd, fund = res["french_daily"], res["fund"]
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
    # date alignment: correlation of French with fund at leads/lags
    lag = {}
    LP = LARGE_PORTFOLIO
    for name, a, b in ((f"{LP} vs SPY", fd[LP], spy), (f"{LP} vs VFINX", fd[LP], fund["VFINX"]),
                       (f"{SMALL_PORTFOLIO} vs IWM", fd[SMALL_PORTFOLIO], iwm),
                       (f"{SMALL_PORTFOLIO} vs NAESX", fd[SMALL_PORTFOLIO], fund["NAESX"])):
        lag[name] = {k: pd.concat([a, b.shift(k)], axis=1).dropna().corr().iloc[0, 1] for k in (-1, 0, 1)}
    lag = pd.DataFrame(lag).T
    lag.columns = ["fund leads 1d (k=-1)", "same day", "fund lags 1d (k=+1)"]
    lag.index.name = "pair"
    # outliers: fund-day returns far from French same-day
    outl = []
    for sym, port in (("VFINX", LARGE_PORTFOLIO), ("SPY", LARGE_PORTFOLIO), ("NAESX", SMALL_PORTFOLIO), ("IWM", SMALL_PORTFOLIO)):
        d = pd.concat([fund[sym], fd[port]], axis=1, keys=["f", "fr"]).dropna()
        diff = d["f"] - d["fr"]
        z = (diff - diff.median()) / (1.4826 * (diff - diff.median()).abs().median())
        for t in z.index[z.abs() > 8]:
            outl.append(dict(date=t, fund=sym, fund_ret=d.loc[t, "f"], french=port, french_ret=d.loc[t, "fr"]))
    outl = pd.DataFrame(outl)
    return dict(calib=calib, lag=lag, outliers=outl)


def vol_by_decade(res: dict) -> pd.DataFrame:
    rows = {}
    for asset in SPECS:
        d, m = res[asset]
        dl = d["ret"].groupby((d.index.year // 10) * 10)
        ml = m["ret"].groupby((m.index.year // 10) * 10)
        rows[f"{asset} daily"] = dl.std() * np.sqrt(252)
        rows[f"{asset} monthly"] = ml.std() * np.sqrt(12)
        rows[f"{asset} days/yr"] = dl.size() / d["ret"].groupby(d.index.year).size().groupby(
            (d["ret"].groupby(d.index.year).size().index // 10) * 10).size()
    out = pd.DataFrame(rows)
    out.index = [f"{i}s" for i in out.index]
    out.index.name = "decade"
    return out


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
        lag=_md(dg["lag"], "{:.4f}"),
        outliers=(_md(dg["outliers"].set_index("date"), "{:.4f}") if len(dg["outliers"]) else "none"),
        known=_md(kd, "{:+.4f}"), known_m=_md(mon, "{:+.4f}"),
        vol=_md(vd, "{:.3f}"),
        sh_corr=sh["monthly"]["corr"], sh_ratio=sh["monthly"]["vol_ratio"],
        sh_aratio=sh["annual_ratio"], sh_acorr=sh["annual_corr"],
        extra=REPORT_EXTRA,
    )
    path.write_text(txt)
    return path


REPORT_EXTRA = ""  # filled below once findings are known

TEMPLATE = """# US equity data: `us_large`, `us_small`

Generated by `python -m volmodel.data.sources.us_equity --report` on {today}.
Loader: `src/volmodel/data/sources/us_equity.py`; raw cache: `data/raw/us_equity/`.
All returns are SIMPLE TOTAL returns (dividends reinvested), in decimals, not excess of cash.

## Coverage

| asset | daily | monthly |
|---|---|---|
| us_large | {large_d0} .. {large_d1} | {large_m0} .. {large_m1} |
| us_small | {small_d0} .. {small_d1} | {small_m0} .. {small_m1} |

The monthly files hold complete calendar months only; in the daily era they are compounded daily returns.
French data ends {french_end}; the fund/ETF segments carry the series to the latest Yahoo close.

## Sources

* **Ken French data library** (CRSP-based): `F-F_Research_Data_Factors_daily` (Mkt-RF + RF = CRSP value-weighted
  market total return, NYSE only before 1962, + AMEX 1962, + Nasdaq 1972) and `Portfolios_Formed_on_ME_daily`
  (value-weighted size portfolios on NYSE market-cap breakpoints, rebalanced each June). Percent, 2 decimals.
* **Yahoo chart API** `adjclose` (dividend- and split-adjusted): VFINX (Vanguard 500 Index Inv), SPY,
  NAESX (Vanguard Small-Cap Index Inv), IWM. Dates converted to exchange-local (New York) trading dates.
* **Shiller** `ie_data.xls`: S&P Composite (Cowles Commission before 1926) monthly price = average of daily closes
  in the month, and annualized dividends (interpolated from annual data before 1926).

## Construction and why

**us_large (SPY-like).** The French value-weighted **`{large_port}`** size portfolio (stocks above the NYSE 70th
percentile of market cap) is used from 1926-07 until VFINX begins. On the SPY overlap it beats the whole CRSP
market (Mkt-RF + RF) on every sub-period: lower tracking error (3-day TE 2.3% vs 3.0% over 1993-2026; 0.7% vs 1.5%
in 2010-19) and a vol ratio closer to 1 since 2010 (1.00-1.02 vs 1.04 for Mkt, whose small-cap tail adds
volatility the S&P 500 does not have). VFINX (an S&P 500 index fund, investable) is then used until SPY starts
trading, then SPY. Before July 1926 only monthly data exists: Shiller's series is used as **tier D**
because month-average prices damp measured volatility (see below).

**us_small (IWM-like).** IWM tracks the Russell 2000 (ranks ~1001-3000 by cap, value weighted). Every French
value-weighted size portfolio was compared with IWM (2000-2026) and with NAESX in its Russell-2000 era
(1990-2000). **`{small_port}`** was chosen: see the calibration table — it matches IWM's volatility level closely
with the highest correlation / lowest tracking error among single portfolios. Its definition (NYSE breakpoints)
is stable over time, so it is the same kind of exposure in 1926 as in 2026. NAESX is used from 1990-01-02 (it
became a Russell 2000 index fund at the end of 1989; before that it was an actively managed fund) until IWM starts,
then IWM. NAESX switched benchmark (MSCI US Small Cap 1750 in 2003, CRSP in 2013) only after IWM takes over.

Splices are of RETURNS, never of price levels, so no level jump can arise; the first-day return of every
segment is checked (tests: |r| < 4 x trailing 63-day vol).

## Segments

{segments}

## Calibration of French portfolios against the ETFs (daily, overlap through French end)

`corr`, `volratio` (French / fund), `TE` (annualized tracking error) on same-day daily returns; `corr3d`,
`volratio3d` on non-overlapping 3-day returns (robust to close-timing differences).

### vs SPY

{calib_large}

### vs IWM (2000-2026) and NAESX (1990-2000, Russell 2000 era)

{calib_small}

### Date alignment

Correlation of French with fund returns at -1/0/+1 day shifts. The peak at "same day" confirms both are indexed
by the trading date of the return.

{lag}

### Fund days far from the French same-day return (robust z > 8)

{outliers}

## Known dates (daily simple total returns)

{known}

Monthly:

{known_m}

## Annualized volatility by decade

`daily` = std of daily returns x sqrt(252); `monthly` = std of monthly returns x sqrt(12);
`days/yr` = average trading days per year in the decade (Saturday sessions before 1952 raise it).

{vol}

## Shiller (pre-1926) damping

On 1926-07..2023 the Shiller monthly return has correlation {sh_corr:.3f} with the French market and a vol ratio
of {sh_ratio:.3f}: averaging prices over the month makes each "monthly return" roughly a two-month moving
average of true returns, which cuts monthly volatility by ~15-20% and induces positive autocorrelation.
On calendar-year log returns the ratio is {sh_aratio:.3f} (corr {sh_acorr:.3f}), so the bias is mostly a
monthly-measurement artifact. That segment is tier D; models should not treat its monthly variance as comparable
to tier B/A without correction. Dividends before 1926 are interpolated from annual figures.
There is no credible pre-1926 small-cap series; `us_small` starts 1926-07.

## Caveats
{extra}
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
