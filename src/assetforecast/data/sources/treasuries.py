"""Treasury total-return histories: ``ust_int`` (IEF-like) and ``ust_long`` (TLT-like).

Run ``python -m assetforecast.data.sources.treasuries`` to download (cached under
``data/raw/treasuries/``) and rebuild

* ``data/processed/assets/ust_{int,long}_{daily,monthly}.parquet``
* ``data/processed/predictors/treasury_curve.parquet``
* ``data/processed/treasuries_segments.csv`` -- this track's rows for ``segments.csv``
* ``reports/data/treasuries_stats.json`` -- diagnostics quoted in reports/data/treasuries.md

Method (details and numbers in reports/data/treasuries.md)
---------------------------------------------------------
Tiers (project-wide): A = daily price of an exchange-traded fund (market
close); B = daily open-end mutual-fund NAV, index, or reconstruction; C =
monthly index total return; D = monthly model-based reconstruction.

Tier A  ETF total return (IEF, TLT) from their 2002-07 inception: Yahoo close
        plus distributions, every distribution validated against the ex-date
        price behaviour (``distributions.validate_distributions``) and against
        Nasdaq's dividend record. Yahoo's adjclose is NOT used: it credits five
        inflated IEF distributions in 2003.
Tier B  Daily reconstruction from the Fed's Gurkaynak-Sack-Wright (GSW)
        fitted Nelson-Siegel-Svensson curve. Every day a par bond of the
        target maturity is bought at the close (coupon = that day's fitted par
        yield, semiannual), held one calendar interval and repriced off the
        next day's fitted discount curve with its maturity shortened by the
        elapsed time. The return therefore contains carry, roll-down and
        price change exactly, and duration moves with the yield level.
        ``ust_int`` is an equal-weight ladder of 7..10y par bonds,
        ``ust_long`` an equal-weight ladder of 20..30y par bonds (the maturity
        buckets of the indices IEF and TLT track). Where the long end of the
        GSW curve was not published (before 1985-11 for 30y) the long ladder
        is priced off H.15 constant-maturity yields instead (see below).
Tier B  Daily reconstruction from H.15 constant-maturity par yields (DBnomics
        mirror of the Fed release): a par bond at yield y_t is repriced at
        y_{t+1} with maturity shortened by the elapsed time (flat local curve).
Tier D  Monthly reconstruction from month-AVERAGE long-bond yields (NBER
        Macrohistory m13033a/b, Fed H.15 monthly averages): returns are
        smoothed (vol damped, positive autocorrelation); quantified in the
        report.

Predictor timing: H.15 constant-maturity yields are published the next
business day (``H15_PUBLICATION_LAG_BDAYS``); anything derived from the GSW
fitted curve is published by the Fed with a lag of about a week, so it is
stamped ``GSW_PUBLICATION_LAG_BDAYS`` = 10 business days after observation
(the same rule as the TIPS track's GSW series).

Nothing here is fitted on the ETF overlap: the ladder maturities are the
index maturity buckets (7-10y, 20-30y), not calibrated, and the overlap
statistics are diagnostics only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ...paths import ASSET_DIR, PREDICTOR_DIR, PROCESSED, RAW
from . import distributions as dist

RAW_DIR = RAW / "treasuries"
# Publication lags of the predictor inputs (business days after the observation).
H15_PUBLICATION_LAG_BDAYS = 1     # H.15 posts a day's CMT yields the next business day
GSW_PUBLICATION_LAG_BDAYS = 10    # GSW curve file: ~1 week lag; same rule as tips.py
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
DBN_URL = "https://api.db.nomics.world/v22/series/FED/H15/{code}?observations=1&format=json"
YAHOO_URL = ("https://query{n}.finance.yahoo.com/v8/finance/chart/{sym}"
             "?period1=0&period2={p2}&interval=1d&events=div%2Csplits")
NBER_URL = "https://data.nber.org/databases/macrohistory/rectdata/13/{code}.dat"

H15_CODES = {  # H.15 Treasury constant maturities, nominal
    "cmt_2y": "RIFLGFCY02_N.B", "cmt_5y": "RIFLGFCY05_N.B", "cmt_7y": "RIFLGFCY07_N.B",
    "cmt_10y": "RIFLGFCY10_N.B", "cmt_20y": "RIFLGFCY20_N.B", "cmt_30y": "RIFLGFCY30_N.B",
    "cmt_10y_m": "RIFLGFCY10_N.M", "cmt_20y_m": "RIFLGFCY20_N.M",
}
YAHOO_SYMBOLS = ["IEF", "TLT", "VUSTX", "VFITX"]

# Definitional breaks in H.15 series: the return across the break is not a
# market move, so that day's return is dropped (a gap, never filled).
# 1973-01-04: the 20y CMT jumps +74bp while the 10y CMT and the GSW curve fall
# 2-3bp (the long-end series was re-based, flower bonds stopped setting it).
H15_BREAKS = {"cmt_20y": ["1973-01-04"]}
# The 20y CMT is a different series before that re-basing (the level jumps 74bp
# and stays): the ``yield_20y`` PREDICTOR starts on the break date, so no level
# break enters a predictor or a spread built on it. (The return reconstruction
# only uses yield CHANGES and drops the break day.)
PREDICTOR_START = {"cmt_20y": "1973-01-04"}

# Peers for the ETF distribution test: daily changes of these CMT yields.
ETF_PEERS = {"IEF": ["cmt_5y", "cmt_7y", "cmt_10y"], "TLT": ["cmt_10y", "cmt_20y", "cmt_30y"]}

# Ladders (years) that mimic the ICE/Barclays maturity buckets the ETFs track.
LADDERS = {
    "ust_int": np.arange(7.0, 10.01, 0.5),    # IEF: 7-10y Treasuries
    "ust_long": np.arange(20.0, 30.01, 1.0),  # TLT: 20y+ Treasuries
}
ETF = {"ust_int": "IEF", "ust_long": "TLT"}
ETF_SOURCE = {s: f"{s} close + validated distributions (Yahoo; Nasdaq dividend record)" for s in ("IEF", "TLT")}


# =================================================================== downloads
def _get(url: str, tries: int = 6, sleep: float = 2.0, timeout: float = 120) -> bytes:
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers={"User-Agent": UA, "Accept": "*/*"}, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
            last = f"HTTP {r.status_code}: {r.text[:100]!r}"
        except requests.RequestException as e:  # pragma: no cover - network
            last = repr(e)
        time.sleep(sleep * (2 ** i))
    raise RuntimeError(f"download failed for {url}: {last}")


def _cached(name: str, url: str, refresh: bool, validate=None) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / name
    if path.exists() and not refresh:
        return path
    content = _get(url)
    if validate is not None:
        validate(content)
    path.write_bytes(content)
    return path


def load_gsw(refresh: bool = False) -> pd.DataFrame:
    """GSW (feds200628) table: NSS parameters and published yields, by date."""
    path = _cached("feds200628.csv", GSW_URL, refresh)
    with open(path) as f:
        skip = next(i for i, line in enumerate(f) if line.startswith("Date,"))
    df = pd.read_csv(path, skiprows=skip, index_col=0, na_values=["NA", "-999.99"])
    df.index = pd.to_datetime(df.index)
    df = df[df["BETA0"].notna()].sort_index()
    return df


def load_h15(key: str, refresh: bool = False) -> pd.Series:
    """One H.15 series (percent) from the DBnomics mirror of the Fed release."""
    code = H15_CODES[key]

    def ok(c):
        json.loads(c)["series"]["docs"][0]["period"]

    path = _cached(f"dbn_H15_{code}.json", DBN_URL.format(code=code), refresh, ok)
    doc = json.loads(path.read_text())["series"]["docs"][0]
    s = pd.Series(pd.to_numeric(pd.Series(doc["value"]), errors="coerce").values,
                  index=pd.to_datetime(doc["period"]), name=key)
    s = s.dropna()
    if key.endswith("_m"):
        s.index = s.index + pd.offsets.MonthEnd(0)
    return s


def load_yahoo(sym: str, refresh: bool = False) -> pd.Series:
    """Adjusted close (dividends reinvested) from the Yahoo chart API (diagnostics only)."""
    return load_yahoo_frame(sym, refresh)["adjclose"].rename(sym)


def load_yahoo_frame(sym: str, refresh: bool = False) -> pd.DataFrame:
    """Daily close (split-adjusted), adjclose and listed cash distributions (on the
    trading day of their ex-date) from the Yahoo chart API."""
    path = RAW_DIR / f"y_{sym}.json"
    if refresh or not path.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        p2 = int(time.time()) + 86400
        content = None
        for i in range(8):
            try:
                url = YAHOO_URL.format(n=1 + i % 2, sym=sym, p2=p2)
                r = requests.get(url, headers={"User-Agent": UA, "Accept": "application/json"},
                                 timeout=60)
                if r.status_code == 200 and r.text.startswith("{"):
                    content = r.content
                    break
            except requests.RequestException:  # pragma: no cover - network
                pass
            time.sleep(5 * (i + 1))
        if content is None:
            raise RuntimeError(f"Yahoo download failed for {sym}")
        path.write_bytes(content)
        time.sleep(1.5)
    res = json.loads(path.read_text())["chart"]["result"][0]
    tz = res["meta"].get("exchangeTimezoneName", "America/New_York")
    to_date = lambda t: pd.to_datetime(t, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()  # noqa: E731
    idx = to_date(res["timestamp"])
    df = pd.DataFrame({"close": res["indicators"]["quote"][0]["close"],
                       "adjclose": res["indicators"]["adjclose"][0]["adjclose"]}, index=idx, dtype=float)
    df = df[~df.index.duplicated(keep="last")].dropna()
    df = df[(df["close"] > 0) & (df["adjclose"] > 0)]
    df.index.name = "date"
    ev = res.get("events", {}).get("dividends", {})
    div = pd.Series({to_date(int(v["date"])): float(v["amount"]) for v in ev.values()}, dtype=float)
    div = div.groupby(level=0).sum() if len(div) else div
    div, _ = dist.dedupe_repeats(div)
    df["div"] = dist.align_dividends(div, df.index)
    return df


def etf_total_return(sym: str, h15: dict[str, pd.Series], refresh: bool = False
                     ) -> tuple[pd.Series, pd.DataFrame]:
    """ETF total return from close + VALIDATED distributions.

    Each Yahoo distribution is tested against the ex-date move of the ETF relative to
    daily CMT yield changes, and against Nasdaq's dividend record (cached as
    ``nasdaq_div_<sym>.json``; Nasdaq lists dividends of Nasdaq-listed ETFs such as
    IEF and TLT). Returns (daily simple returns, validation table)."""
    df = load_yahoo_frame(sym, refresh)
    peers = pd.DataFrame({k: h15[k].diff() / 100.0 for k in ETF_PEERS[sym]})
    second = dist.fetch_nasdaq_dividends(sym, RAW_DIR / f"nasdaq_div_{sym}.json", refresh)
    div, table = dist.validate_distributions(df["close"], df["div"], peers, second=second)
    r = dist.total_return(df["close"], div).iloc[1:].rename(sym)
    return r, table


def load_nber(code: str, refresh: bool = False) -> pd.Series:
    """NBER Macrohistory monthly series (year, month, value); '.' is missing."""
    def ok(c):
        if not c[:4].decode(errors="ignore").strip().isdigit():
            raise RuntimeError(f"unexpected NBER content for {code}")

    path = _cached(f"nber_{code}.dat", NBER_URL.format(code=code), refresh, ok)
    rows = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) >= 3 and p[2] != ".":
            rows.append((pd.Timestamp(int(p[0]), int(p[1]), 1) + pd.offsets.MonthEnd(0), float(p[2])))
    return pd.Series(dict(rows), name=code).sort_index()


# ======================================================== curve and bond math
def nss_zero(params: pd.DataFrame | dict, m) -> np.ndarray:
    """Continuously-compounded zero yield (decimal) at maturities ``m`` (years).

    ``params`` holds BETA0..BETA3, TAU1, TAU2 (GSW units: percent, years).
    Rows broadcast against ``m``: params of shape (n,) and m of shape (n, k).
    A missing/zero BETA3 or TAU2 means the 4-parameter Nelson-Siegel form.
    """
    g = lambda k: np.asarray(params[k], dtype=float)
    b0, b1, b2, b3, t1, t2 = (g(k) for k in ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"])
    m = np.maximum(np.asarray(m, dtype=float), 1e-6)
    if b0.ndim == 1 and m.ndim == 2:
        b0, b1, b2, b3, t1, t2 = (x[:, None] for x in (b0, b1, b2, b3, t1, t2))
    b3 = np.nan_to_num(b3)
    t2 = np.where(np.isfinite(t2) & (t2 > 0), t2, 1.0)
    x1, x2 = m / t1, m / t2
    f1 = (1 - np.exp(-x1)) / x1
    f2 = f1 - np.exp(-x1)
    f3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    return (b0 + b1 * f1 + b2 * f2 + b3 * f3) / 100.0


def nss_discount(params, m) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    return np.exp(-nss_zero(params, m) * np.maximum(m, 0.0))


def par_yield(params, maturity: float) -> np.ndarray:
    """Semiannual coupon-equivalent par yield (decimal) for ``maturity`` years
    (an integer number of half-years)."""
    n = int(round(maturity * 2))
    times = np.arange(1, n + 1) / 2.0
    single = not hasattr(np.asarray(params["BETA0"]), "__len__") or np.ndim(params["BETA0"]) == 0
    t = times[None, :] if not single else times
    df = nss_discount(params, t if single else np.broadcast_to(t, (len(np.asarray(params["BETA0"])), n)))
    df = np.atleast_2d(df)
    c = 2.0 * (1.0 - df[:, -1]) / df.sum(axis=1)
    return c[0] if single else c


def bond_value_from_discount(coupon: np.ndarray, maturity: float, elapsed: np.ndarray,
                             params) -> np.ndarray:
    """Dirty value per 1 face of a bond issued ``elapsed`` years ago with
    original maturity ``maturity`` and semiannual ``coupon`` (decimal),
    valued off the discount curve ``params``. Coupons paid in the interval
    (0, elapsed] are included at face value (received in cash)."""
    n = int(round(maturity * 2))
    times = np.arange(1, n + 1) / 2.0                       # from issue
    elapsed = np.asarray(elapsed, dtype=float)
    rem = times[None, :] - elapsed[:, None]                 # remaining time to each cash flow
    cf = np.broadcast_to(np.asarray(coupon)[:, None] / 2.0, rem.shape).copy()
    cf[:, -1] += 1.0
    df = nss_discount(params, np.maximum(rem, 0.0))
    df = np.where(rem <= 0, 1.0, df)                        # already received
    return (cf * df).sum(axis=1)


def gsw_par_bond_returns(gsw: pd.DataFrame, maturity: float) -> pd.Series:
    """Daily total return of a constant-maturity par bond off the GSW curve.

    Return on day t (close t-1 -> close t): buy at par at t-1 with coupon =
    par yield(t-1, maturity), revalue at t off curve t with time elapsed
    (calendar days / 365.25).
    """
    p = gsw[["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]]
    c = par_yield(p, maturity)
    dt = gsw.index.to_series().diff().dt.days.to_numpy() / 365.25
    prev_c = np.r_[np.nan, c[:-1]]
    v = bond_value_from_discount(np.nan_to_num(prev_c), maturity, np.nan_to_num(dt), p)
    ret = pd.Series(v - 1.0, index=gsw.index)
    ret.iloc[0] = np.nan
    return ret


def par_bond_price(y, coupon, years_to_maturity) -> np.ndarray:
    """Price per 1 face of a semiannual bond at yield ``y`` (bond-equivalent),
    maturity ``years_to_maturity`` (may be fractional), dirty price with
    fractional first period (street convention, compounding at y/2)."""
    y = np.asarray(y, dtype=float)
    T = np.asarray(years_to_maturity, dtype=float)
    n = np.ceil(T * 2 - 1e-9)       # coupons remaining
    w = n - T * 2                   # fraction of the current period already elapsed
    v = 1.0 / (1.0 + y / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        annuity = np.where(np.abs(y) > 1e-12, (1 - v ** n) / (y / 2.0), n)
    # value one full period before the first coupon, rolled forward w periods
    return (coupon / 2.0 * annuity + v ** n) * v ** (-w)


def cmt_par_bond_returns(y_pct: pd.Series, maturity: float) -> pd.Series:
    """Daily total return of a constant-maturity par bond from a par-yield series
    (percent). Buy at par at y_{t-1}; reprice at y_t with maturity M - dt.
    No roll-down (the local curve is taken as flat)."""
    y = y_pct.dropna().sort_index() / 100.0
    dt = y.index.to_series().diff().dt.days.to_numpy() / 365.25
    prev = y.shift(1).to_numpy()
    price = par_bond_price(y.to_numpy(), prev, maturity - np.nan_to_num(dt))
    ret = pd.Series(price - 1.0, index=y.index)
    ret.iloc[0] = np.nan
    return ret


def monthly_par_bond_returns(y_pct: pd.Series, maturity: float) -> pd.Series:
    """Monthly total return from a monthly yield series (percent): buy at par at
    y_{m-1}, sell one month later at y_m with maturity M - 1/12."""
    y = y_pct.dropna().sort_index() / 100.0
    prev = y.shift(1).to_numpy()
    price = par_bond_price(y.to_numpy(), prev, maturity - 1.0 / 12.0)
    ret = pd.Series(price - 1.0, index=y.index)
    ret.iloc[0] = np.nan
    return ret


def par_mod_duration(y, maturity: float) -> np.ndarray:
    """Modified duration (years) of a semiannual par bond at yield y (decimal)."""
    y = np.asarray(y, dtype=float)
    n = maturity * 2
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (1.0 - (1.0 + y / 2.0) ** (-n)) / y
    return np.where(np.abs(y) < 1e-9, maturity, d)


# ================================================================== ladders
def gsw_ladder(gsw: pd.DataFrame, mats) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Equal-weight ladder of constant-maturity par bonds off the GSW curve,
    rebalanced daily. Returns (daily return, modified duration, mean par yield)."""
    p = gsw[["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]]
    rets, durs, ys = [], [], []
    for m in mats:
        rets.append(gsw_par_bond_returns(gsw, m))
        y = par_yield(p, m)
        ys.append(pd.Series(y, index=gsw.index))
        durs.append(pd.Series(par_mod_duration(y, m), index=gsw.index))
    mean = lambda xs: pd.concat(xs, axis=1).mean(axis=1, skipna=False)
    return mean(rets), mean(durs), mean(ys)


def cmt_ladder_yields(h15: dict[str, pd.Series], mats, anchors: list[tuple[float, str]]) -> pd.DataFrame:
    """Par yields (percent) for each ladder maturity, linearly interpolated
    between H.15 CMT anchor maturities and held flat beyond the end anchors.
    Only dates where every anchor is observed are kept."""
    a = pd.concat({m: h15[k] for m, k in anchors}, axis=1, sort=True).dropna()
    am = np.array([m for m, _ in anchors], dtype=float)
    out = {m: np.array([np.interp(m, am, row) for row in a.to_numpy()]) for m in mats}
    return pd.DataFrame(out, index=a.index)


def cmt_ladder(yields: pd.DataFrame, breaks=()) -> tuple[pd.Series, pd.Series]:
    """Daily ladder return and modified duration from par yields per maturity
    (columns = maturity in years, values in percent). Returns on ``breaks``
    dates are dropped."""
    rets = [cmt_par_bond_returns(yields[m], m) for m in yields.columns]
    durs = [pd.Series(par_mod_duration(yields[m].to_numpy() / 100, m), index=yields.index)
            for m in yields.columns]
    mean = lambda xs: pd.concat(xs, axis=1).mean(axis=1, skipna=False)
    ret = mean(rets)
    ret = ret.drop(pd.DatetimeIndex(breaks).intersection(ret.index))
    return ret, mean(durs)


def monthly_ladder(y_pct: pd.Series, mats) -> pd.Series:
    """Monthly ladder return from ONE monthly (average) yield series held flat
    across the ladder maturities."""
    return pd.concat([monthly_par_bond_returns(y_pct, m) for m in mats], axis=1).mean(axis=1, skipna=False)


# ============================================================== diagnostics
def compare(rec: pd.Series, ref: pd.Series, start=None, end=None, freq: str = "D") -> dict:
    """Vol ratio (rec/ref), correlation, annualized tracking error, n."""
    df = pd.concat([rec.rename("r"), ref.rename("e")], axis=1, sort=True).dropna()
    df = df.loc[start:end]
    if freq == "M":
        df = (1 + df).resample("ME").prod(min_count=1) - 1
        df = df.iloc[1:-1] if len(df) > 2 else df     # drop partial first/last months
        k = 12
    else:
        k = 252
    if len(df) < 10:
        return {}
    return {
        "start": str(df.index[0].date()), "end": str(df.index[-1].date()), "n": int(len(df)),
        "vol_rec": float(df.r.std() * np.sqrt(k)), "vol_ref": float(df.e.std() * np.sqrt(k)),
        "vol_ratio": float(df.r.std() / df.e.std()), "corr": float(df.r.corr(df.e)),
        "te": float((df.r - df.e).std() * np.sqrt(k)),
        "ac1_rec": float(df.r.autocorr()), "ac1_ref": float(df.e.autocorr()),
    }


def to_monthly(daily: pd.Series) -> pd.Series:
    """Compound daily simple returns to calendar month-ends (NaN-free months only)."""
    d = daily.dropna()
    return (1 + d).resample("ME").prod(min_count=1) - 1


# =================================================================== build
def _frame(ret: pd.Series, source: str, tier: str) -> pd.DataFrame:
    ret = ret.dropna()
    return pd.DataFrame({"ret": ret.astype(float), "source": source, "tier": tier}, index=ret.index)


def build(refresh: bool = False) -> dict:
    """Download (cached) and construct everything. Returns a dict with the
    daily/monthly frames per asset, the predictor frame, segments and diagnostics."""
    gsw = load_gsw(refresh)
    h15 = {k: load_h15(k, refresh) for k in H15_CODES}
    etf = {s: load_yahoo(s, refresh) for s in YAHOO_SYMBOLS}
    nber = {c: load_nber(c, refresh) for c in ["m13033a", "m13033b"]}
    # VFITX/VUSTX (diagnostics only): adjclose; IEF/TLT: close + validated distributions
    etf_ret = {s: v.pct_change().iloc[1:] for s, v in etf.items()}
    div_tables = {}
    for s in ETF.values():
        etf_ret[s], div_tables[s] = etf_total_return(s, h15, refresh)
        # total-return index of the ETF as used in the panel (base = first close)
        etf[s] = (1 + etf_ret[s]).cumprod().rename(s)

    # first date at which each GSW maturity is published (in-sample)
    pub = {m: gsw[f"SVENY{m:02d}"].first_valid_index() for m in (10, 20, 30)}
    gsw_i_start, gsw_l_start = pub[10], pub[30]

    out: dict = {"segments": [], "diag": {}, "assets": {}}
    lad = {}
    lad["ust_int"] = gsw_ladder(gsw.loc[gsw_i_start:], LADDERS["ust_int"])
    lad["ust_long"] = gsw_ladder(gsw.loc[gsw_l_start:], LADDERS["ust_long"])

    # H.15 CMT ladders for the eras before the GSW curve covers the ladder
    y_int = cmt_ladder_yields(h15, LADDERS["ust_int"], [(10.0, "cmt_10y")])
    cmt_int, cmt_int_dur = cmt_ladder(y_int)
    y_long_20 = cmt_ladder_yields(h15, LADDERS["ust_long"], [(20.0, "cmt_20y")])
    cmt_long20, cmt_long20_dur = cmt_ladder(y_long_20, H15_BREAKS["cmt_20y"])
    y_long_2030 = cmt_ladder_yields(h15, LADDERS["ust_long"], [(20.0, "cmt_20y"), (30.0, "cmt_30y")])
    cmt_long2030, cmt_long2030_dur = cmt_ladder(y_long_2030, H15_BREAKS["cmt_20y"])

    # ------------------------------------------------------------ daily splice
    daily_parts = {
        "ust_int": [
            (cmt_int,
             "H.15 10y CMT par-bond ladder 7-10y (flat curve)", "B"),
            (lad["ust_int"][0],
             "GSW fitted curve, par-bond ladder 7-10y", "B"),
            (etf_ret["IEF"], ETF_SOURCE["IEF"], "A"),
        ],
        "ust_long": [
            (cmt_long20,
             "H.15 20y CMT par-bond ladder 20-30y (flat beyond 20y)", "B"),
            (cmt_long2030,
             "H.15 20y/30y CMT par-bond ladder 20-30y (interpolated)", "B"),
            (lad["ust_long"][0],
             "GSW fitted curve, par-bond ladder 20-30y", "B"),
            (etf_ret["TLT"], ETF_SOURCE["TLT"], "A"),
        ],
    }
    # ---------------------------------------------------------- monthly-only
    int_m = monthly_ladder(h15["cmt_10y_m"], LADDERS["ust_int"])
    long_m20 = monthly_ladder(h15["cmt_20y_m"], LADDERS["ust_long"])
    long_a = monthly_ladder(nber["m13033a"], LADDERS["ust_long"])
    long_b = monthly_ladder(nber["m13033b"], LADDERS["ust_long"])
    monthly_parts = {
        "ust_int": [
            (int_m, "H.15 10y CMT monthly-average yield, par-bond ladder 7-10y", "D"),
        ],
        "ust_long": [
            (long_a.loc[:"1941-10-31"], "NBER m13033a long-term US bond yield (monthly avg), ladder 20-30y", "D"),
            (long_b.loc["1941-11-30":"1953-04-30"],
             "NBER m13033b long-term US bond yield (monthly avg), ladder 20-30y", "D"),
            (long_m20, "H.15 20y CMT monthly-average yield, par-bond ladder 20-30y", "D"),
        ],
    }

    for a in ["ust_int", "ust_long"]:
        # daily: later segments win; each segment strictly after the previous one
        # daily: a later (better) segment owns its whole span; an earlier one
        # fills only the dates before the later segment's first return
        frames, first = [], None
        for ret, src, tier in reversed(daily_parts[a]):
            r = ret.dropna()
            if first is not None:
                r = r.loc[r.index < first]
            if r.empty:
                continue
            frames.append(_frame(r, src, tier))
            first = r.index[0]
        daily = pd.concat(frames).sort_index()
        daily.index.name = "date"
        # monthly from daily: full months only (first daily month is partial)
        first_full = (daily.index[0] + pd.offsets.MonthEnd(0))
        g = daily["ret"].groupby(daily.index + pd.offsets.MonthEnd(0))
        m_ret = g.apply(lambda x: (1 + x).prod() - 1)
        tier_order = {"A": 0, "B": 1, "C": 2, "D": 3}
        m_tier = daily["tier"].groupby(daily.index + pd.offsets.MonthEnd(0)).agg(
            lambda x: max(x, key=lambda t: tier_order[t]))
        m_src = daily.groupby(daily.index + pd.offsets.MonthEnd(0))["source"].agg(
            lambda x: x.value_counts().index[0])
        md = pd.DataFrame({"ret": m_ret, "source": m_src, "tier": m_tier})
        md = md.loc[md.index > first_full]       # drop partial first month
        last_day = daily.index[-1]
        if last_day < last_day + pd.offsets.BMonthEnd(0):   # current month incomplete
            md = md.loc[md.index < last_day + pd.offsets.MonthEnd(0)]
        # monthly-only segments before the daily era
        pre = []
        for ret, src, tier in monthly_parts[a]:
            r = ret.dropna()
            r = r.loc[r.index < md.index[0]]
            if pre:
                r = r.loc[r.index > pre[-1].index[-1]]
            if not r.empty:
                pre.append(_frame(r, src, tier))
        monthly = pd.concat(pre + [md]).sort_index()
        monthly.index.name = "date"
        assert monthly.index.is_unique and daily.index.is_unique
        out["assets"][a] = {"daily": daily, "monthly": monthly}

    out["parts"] = {"daily": daily_parts, "monthly": monthly_parts}
    out["series"] = dict(gsw=gsw, h15=h15, etf=etf, etf_ret=etf_ret, nber=nber, lad=lad,
                         cmt_int=cmt_int, cmt_long20=cmt_long20, cmt_long2030=cmt_long2030,
                         cmt_int_dur=cmt_int_dur, cmt_long20_dur=cmt_long20_dur,
                         cmt_long2030_dur=cmt_long2030_dur, int_m=int_m, long_m20=long_m20,
                         long_a=long_a, long_b=long_b, pub=pub, div_tables=div_tables)
    out["predictors"] = build_predictors(out)
    out["diag"] = diagnostics(out)
    out["segments"] = segments_table(out)
    return out


def _fmt(d: dict, freq: str) -> str:
    if not d:
        return "n/a"
    return (f"{freq} {d['start'][:7]}..{d['end'][:7]}: vol ratio {d['vol_ratio']:.3f}, "
            f"corr {d['corr']:.3f}, TE {100 * d['te']:.1f}%/yr")


def _div_note(D: dict, sym: str) -> str:
    ch = D.get(f"{sym}_distributions_changed", [])
    if not ch:
        return "No Yahoo distribution changed."
    return "Yahoo distributions corrected: " + "; ".join(
        f"{c['date']} {c['amount']:.3f} -> {c['used']:.3f} ({c['action'].split(':')[1]})" for c in ch) + "."


def segment_notes(D: dict) -> dict[str, str]:
    """Overlap checks recorded in segments.csv notes, keyed by source label."""
    vr = lambda a: max(D[f"{a}_daily_vr21"].items(), key=lambda kv: kv[1])
    return {
        "H.15 10y CMT par-bond ladder 7-10y (flat curve)":
            "vs GSW 7-10y ladder on 1971-2026 overlap: " + _fmt(D["ust_int_cmt_vs_gsw_D"], "daily")
            + "; " + _fmt(D["ust_int_cmt_vs_gsw_M"], "monthly")
            + ". Daily returns pre-1987 positively autocorrelated (stale quotes): 21d variance ratio up to "
            f"{vr('ust_int')[1]:.2f} ({vr('ust_int')[0]})",
        "GSW fitted curve, par-bond ladder 7-10y":
            "vs IEF: " + _fmt(D["ust_int_gsw_vs_IEF_D_all"], "daily") + "; "
            + _fmt(D["ust_int_gsw_vs_IEF_M_all"], "monthly")
            + ". Daily corr limited by quote time (curve ~3pm vs ETF 4pm close)",
        ETF_SOURCE["IEF"]: "investable ETF; total return = close + distributions, each validated by the "
                           "ex-date price test and Nasdaq's dividend record. " + _div_note(D, "IEF"),
        "H.15 10y CMT monthly-average yield, par-bond ladder 7-10y":
            "MONTH-AVERAGE yields: vs month-end returns " + _fmt(D["ust_int_monthavg_vs_monthend_all"], "monthly")
            + f"; AC1 {D['ust_int_monthavg_vs_monthend_all']['ac1_rec']:.2f} vs "
            f"{D['ust_int_monthavg_vs_monthend_all']['ac1_ref']:.2f} (vol damped, autocorrelation induced)",
        "H.15 20y CMT par-bond ladder 20-30y (flat beyond 20y)":
            "vs GSW 20-30y ladder 1985-2026: " + _fmt(D["ust_long_cmt20flat_vs_gsw_D"], "daily")
            + "; " + _fmt(D["ust_long_cmt20flat_vs_gsw_M"], "monthly")
            + f". 1960s quotes stale (~50% zero daily changes 1962-66); 21d variance ratio up to "
            f"{vr('ust_long')[1]:.2f} ({vr('ust_long')[0]})",
        "H.15 20y/30y CMT par-bond ladder 20-30y (interpolated)":
            "vs GSW 20-30y ladder 1985-2026: " + _fmt(D["ust_long_cmt2030_vs_gsw_D"], "daily")
            + "; " + _fmt(D["ust_long_cmt2030_vs_gsw_M"], "monthly"),
        "GSW fitted curve, par-bond ladder 20-30y":
            "vs TLT: " + _fmt(D["ust_long_gsw_vs_TLT_D_all"], "daily") + "; "
            + _fmt(D["ust_long_gsw_vs_TLT_M_all"], "monthly")
            + f"; 2020-26 daily vol ratio {D['ust_long_gsw_vs_TLT_D_2020']['vol_ratio']:.3f} "
            "(TLT holds low-coupon discount bonds: higher duration than par bonds)",
        ETF_SOURCE["TLT"]: "investable ETF; total return = close + distributions, each validated by the "
                           "ex-date price test and Nasdaq's dividend record. " + _div_note(D, "TLT"),
        "NBER m13033a long-term US bond yield (monthly avg), ladder 20-30y":
            "MONTH-AVERAGE yields (partially tax-exempt bonds, 12y+). vs m13033b on 1941-44 overlap: "
            + _fmt(D["ust_long_nber_a_vs_nber_b_M"], "monthly"),
        "NBER m13033b long-term US bond yield (monthly avg), ladder 20-30y":
            "MONTH-AVERAGE yields. vs H.15 20y monthly avg on 1953-67 overlap: "
            + _fmt(D["ust_long_nber_b_vs_h15avg_M"], "monthly"),
        "H.15 20y CMT monthly-average yield, par-bond ladder 20-30y":
            "MONTH-AVERAGE yields: vs month-end returns " + _fmt(D["ust_long_monthavg_vs_monthend_all"], "monthly")
            + f"; AC1 {D['ust_long_monthavg_vs_monthend_all']['ac1_rec']:.2f} vs "
            f"{D['ust_long_monthavg_vs_monthend_all']['ac1_ref']:.2f}",
    }


def segments_table(out: dict) -> pd.DataFrame:
    notes = segment_notes(out["diag"])
    rows = []
    for a, d in out["assets"].items():
        for freq, df in [("daily", d["daily"]), ("monthly", d["monthly"])]:
            if freq == "monthly":
                df = df[df["tier"].isin(["C", "D"])]
            key = (df["source"] != df["source"].shift()).cumsum()
            for _, seg in df.groupby(key):
                rows.append({"asset": a, "start": seg.index[0].date(), "end": seg.index[-1].date(),
                             "freq": freq, "source": seg["source"].iloc[0], "tier": seg["tier"].iloc[0],
                             "notes": notes.get(seg["source"].iloc[0], "")})
    return pd.DataFrame(rows)


# =============================================================== predictors
def build_predictors(out: dict) -> pd.DataFrame:
    """Daily yield levels (decimal, H.15 CMT par yields) and modified durations
    (years) of the constructed par-bond ladders; long format.

    ``available_date``: H.15 inputs + ``H15_PUBLICATION_LAG_BDAYS``; GSW-curve inputs
    + ``GSW_PUBLICATION_LAG_BDAYS`` (a series switching between the two inputs takes
    the lag of the input used for each observation)."""
    S = out["series"]
    rows = []
    h15_lag, gsw_lag = pd.offsets.BDay(H15_PUBLICATION_LAG_BDAYS), pd.offsets.BDay(GSW_PUBLICATION_LAG_BDAYS)

    def add(series: pd.Series, name: str, lag=h15_lag, gsw_span=None):
        s = series.dropna()
        avail = s.index + lag
        if gsw_span is not None:    # observations priced off the GSW curve
            on = (s.index >= gsw_span[0]) & (s.index <= gsw_span[1])
            avail = pd.DatetimeIndex(np.where(on, s.index + gsw_lag, s.index + h15_lag))
        rows.append(pd.DataFrame({"obs_date": s.index, "available_date": avail,
                                  "series": name, "value": s.to_numpy(dtype=float)}))

    for k in ["cmt_2y", "cmt_5y", "cmt_7y", "cmt_10y", "cmt_20y", "cmt_30y"]:
        add((S["h15"][k] / 100.0).loc[PREDICTOR_START.get(k):], "yield_" + k.split("_")[1])

    gi, gl = S["lad"]["ust_int"], S["lad"]["ust_long"]
    # durations: consistent with the yields each era's reconstruction used;
    # CMT-based after the last GSW date (GSW is published with a lag)
    d_int = pd.concat([S["cmt_int_dur"].loc[: gi[1].index[0] - pd.Timedelta(days=1)], gi[1],
                       S["cmt_int_dur"].loc[gi[1].index[-1] + pd.Timedelta(days=1):]])
    d_long = pd.concat([S["cmt_long20_dur"].loc[: S["h15"]["cmt_30y"].index[0]],
                        S["cmt_long2030_dur"].loc[S["h15"]["cmt_30y"].index[0] + pd.Timedelta(days=1):
                                                  gl[1].index[0] - pd.Timedelta(days=1)],
                        gl[1],
                        S["cmt_long2030_dur"].loc[gl[1].index[-1] + pd.Timedelta(days=1):]])
    add(d_int, "duration_ust_int", gsw_span=(gi[1].index[0], gi[1].index[-1]))
    add(d_long, "duration_ust_long", gsw_span=(gl[1].index[0], gl[1].index[-1]))
    add(gi[2], "yield_ladder_ust_int", gsw_lag)
    add(gl[2], "yield_ladder_ust_long", gsw_lag)
    # empirical effective duration of the ETFs (rolling regression on the GSW ladder yield)
    for a, (ret, _, y) in [("ust_int", gi), ("ust_long", gl)]:
        add(etf_effective_duration(S["etf"][ETF[a]], y), f"duration_{a}_etf_est", gsw_lag)
    pred = pd.concat(rows, ignore_index=True)
    pred["obs_date"] = pd.to_datetime(pred["obs_date"])
    pred["available_date"] = pd.to_datetime(pred["available_date"])
    return pred.sort_values(["series", "obs_date"]).reset_index(drop=True)


def etf_effective_duration(price: pd.Series, y: pd.Series, window: int = 252, h: int = 5) -> pd.Series:
    """Rolling slope of h-day ETF log returns on minus h-day changes in the
    ladder par yield (decimal). Multi-day changes reduce the attenuation from
    the ETF closing ~30-60 minutes after the curve's quote time. Uses only
    data up to each date."""
    df = pd.concat([np.log(price).rename("p"), y.rename("y")], axis=1, sort=True).dropna()
    r = df["p"].diff(h)
    dy = -df["y"].diff(h)
    cov = (r * dy).rolling(window).mean() - r.rolling(window).mean() * dy.rolling(window).mean()
    var = (dy * dy).rolling(window).mean() - dy.rolling(window).mean() ** 2
    return (cov / var).dropna()


# ============================================================== diagnostics
EPISODES = {
    "volcker_1979_10_1982_12": ("1979-10-01", "1982-12-31"),
    "crash_1987_10": ("1987-10-01", "1987-10-31"),
    "bond_massacre_1994": ("1994-01-01", "1994-12-31"),
    "gfc_2008_q4": ("2008-10-01", "2008-12-31"),
    "taper_2013_05_09": ("2013-05-01", "2013-09-30"),
    "covid_2020_03": ("2020-03-01", "2020-03-31"),
    "inflation_2022": ("2022-01-01", "2022-12-31"),
}


def diagnostics(out: dict) -> dict:
    S = out["series"]
    E = S["etf_ret"]
    D: dict = {}
    for sym, t in S["div_tables"].items():
        ch = dist.changed_distributions(t)
        D[f"{sym}_distributions_changed"] = [
            dict(date=str(d.date()), amount=float(r.amount), used=float(r.used),
                 second=None if not np.isfinite(r.second) else float(r.second),
                 implied=float(r.implied), z=float(r.z), action=r.action) for d, r in ch.iterrows()]
        D[f"{sym}_distributions_summary"] = t["action"].value_counts().to_dict()
    gi, gl = S["lad"]["ust_int"][0], S["lad"]["ust_long"][0]
    periods = [(None, None), ("2002", "2007"), ("2008", "2014"), ("2015", "2019"), ("2020", "2026")]
    # 1. reconstruction vs ETFs
    for a, rec, ref in [("ust_int", gi, "IEF"), ("ust_long", gl, "TLT")]:
        for f in ["D", "M"]:
            for s_, e_ in periods:
                D[f"{a}_gsw_vs_{ref}_{f}_{s_ or 'all'}"] = compare(rec, E[ref], s_, e_, f)
        df = pd.concat([rec.rename("r"), E[ref].rename("e")], axis=1, sort=True).dropna()
        yr = df.groupby(df.index.year)
        D[f"{a}_yearly_vol_ratio_vs_{ref}"] = {int(k): float(v.r.std() / v.e.std()) for k, v in yr}
        D[f"{a}_yearly_corr_vs_{ref}"] = {int(k): float(v.r.corr(v.e)) for k, v in yr}
        D[f"{a}_corr_lead_lag_vs_{ref}"] = {k: float(df.r.corr(df.e.shift(k))) for k in (-1, 0, 1)}
    # 2. funds before the ETFs (informational)
    for a, rec, ref in [("ust_int", gi, "VFITX"), ("ust_long", gl, "VUSTX")]:
        for f in ["D", "M"]:
            D[f"{a}_gsw_vs_{ref}_{f}_pre2002"] = compare(rec, E[ref], None, "2002-07-29", f)
    # 3. CMT method vs GSW (validates the pre-GSW segments' method)
    D["ust_int_cmt_vs_gsw_D"] = compare(S["cmt_int"], gi, None, None, "D")
    D["ust_int_cmt_vs_gsw_M"] = compare(S["cmt_int"], gi, None, None, "M")
    D["ust_long_cmt2030_vs_gsw_D"] = compare(S["cmt_long2030"], gl, None, None, "D")
    D["ust_long_cmt2030_vs_gsw_M"] = compare(S["cmt_long2030"], gl, None, None, "M")
    D["ust_long_cmt20flat_vs_gsw_D"] = compare(S["cmt_long20"], gl, None, None, "D")
    D["ust_long_cmt20flat_vs_gsw_M"] = compare(S["cmt_long20"], gl, None, None, "M")
    D["ust_int_cmt_vs_IEF_D"] = compare(S["cmt_int"], E["IEF"], None, None, "D")
    D["ust_long_cmt2030_vs_TLT_D"] = compare(S["cmt_long2030"], E["TLT"], None, None, "D")
    # 4. damping from month-average yields: avg-based vs month-end (daily compounded)
    for a, mavg, dly in [("ust_int", S["int_m"], S["cmt_int"]), ("ust_long", S["long_m20"], S["cmt_long20"])]:
        me = to_monthly(dly).iloc[1:-1]
        for s_, e_ in [(None, None), ("1962", "1986"), ("1994", "2026")]:
            D[f"{a}_monthavg_vs_monthend_{s_ or 'all'}"] = compare_monthly(mavg, me, s_, e_)
    # NBER m13033b vs H.15 20y (both month averages)
    D["ust_long_nber_b_vs_h15avg_M"] = compare_monthly(S["long_b"], S["long_m20"], None, None)
    D["ust_long_nber_a_vs_nber_b_M"] = compare_monthly(S["long_a"], S["long_b"], None, None)
    # daily serial correlation: variance ratio of 21-day to 1-day log returns
    for a, d in out["assets"].items():
        x = np.log1p(d["daily"]["ret"])
        vr = {}
        for y0 in range(1960, 2030, 5):
            xx = x.loc[str(y0):str(y0 + 4)]
            if len(xx) > 500:
                r21 = xx.rolling(21).sum().iloc[20::21]
                vr[f"{y0}-{y0 + 4}"] = float(r21.var() / (21 * xx.var()))
        D[f"{a}_daily_vr21"] = vr
    # 5. vol by decade and episodes
    for a, d in out["assets"].items():
        m, dl = d["monthly"], d["daily"]
        dec = {}
        for y0 in range(1910, 2030, 10):
            x = m.loc[str(y0):str(y0 + 9)]
            xd = dl.loc[str(y0):str(y0 + 9)]
            if len(x) < 6:
                continue
            dec[f"{y0}s"] = {"n_months": int(len(x)), "vol_monthly": float(x.ret.std() * np.sqrt(12)),
                             "vol_daily": float(xd.ret.std() * np.sqrt(252)) if len(xd) > 100 else None,
                             "ac1_monthly": float(x.ret.autocorr()),
                             "tiers": "".join(sorted(x.tier.unique()))}
        D[f"{a}_by_decade"] = dec
        ep = {}
        for k, (s_, e_) in EPISODES.items():
            x = dl.loc[s_:e_]
            xm = m.loc[s_:e_]
            ep[k] = {"cum_ret": float((1 + xm.ret).prod() - 1) if len(xm) else None,
                     "vol_daily": float(x.ret.std() * np.sqrt(252)) if len(x) > 5 else None,
                     "worst_day": [str(x.ret.idxmin().date()), float(x.ret.min())] if len(x) else None,
                     "best_day": [str(x.ret.idxmax().date()), float(x.ret.max())] if len(x) else None}
        D[f"{a}_episodes"] = ep
    return D


def compare_monthly(rec: pd.Series, ref: pd.Series, start, end) -> dict:
    df = pd.concat([rec.rename("r"), ref.rename("e")], axis=1, sort=True).dropna().loc[start:end]
    if len(df) < 12:
        return {}
    return {"start": str(df.index[0].date()), "end": str(df.index[-1].date()), "n": int(len(df)),
            "vol_rec": float(df.r.std() * np.sqrt(12)), "vol_ref": float(df.e.std() * np.sqrt(12)),
            "vol_ratio": float(df.r.std() / df.e.std()), "corr": float(df.r.corr(df.e)),
            "te": float((df.r - df.e).std() * np.sqrt(12)),
            "ac1_rec": float(df.r.autocorr()), "ac1_ref": float(df.e.autocorr())}


# ===================================================================== main
def save(out: dict) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTOR_DIR.mkdir(parents=True, exist_ok=True)
    for a, d in out["assets"].items():
        for f in ["daily", "monthly"]:
            df = d[f].copy()
            df["source"] = df["source"].astype(str)
            df["tier"] = df["tier"].astype(str)
            df.to_parquet(ASSET_DIR / f"{a}_{f}.parquet")
    out["predictors"].to_parquet(PREDICTOR_DIR / "treasury_curve.parquet", index=False)
    out["segments"].to_csv(RAW_DIR / "segments_treasuries.csv", index=False)
    (PROCESSED / "segments").mkdir(parents=True, exist_ok=True)
    out["segments"].to_csv(PROCESSED / "segments" / "treasuries.csv", index=False)
    pd.concat({s: t for s, t in out["series"]["div_tables"].items()}, names=["etf", "date"]).to_csv(
        RAW_DIR / "distribution_checks.csv")
    (RAW_DIR / "diagnostics.json").write_text(json.dumps(out["diag"], indent=1, default=str))


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-download all sources")
    args = ap.parse_args(argv)
    out = build(refresh=args.refresh)
    save(out)
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    print(out["segments"].to_string(index=False))
    for a, d in out["assets"].items():
        print(a, "daily", d["daily"].index[0].date(), "->", d["daily"].index[-1].date(),
              "| monthly", d["monthly"].index[0].date(), "->", d["monthly"].index[-1].date())


if __name__ == "__main__":
    main()
