"""Credit total returns: ``ig_credit`` (LQD-like) and ``hy_credit`` (HYG-like).

Run ``python -m assetforecast.data.sources.credit`` to (re)build
``data/processed/assets/{ig,hy}_credit_{daily,monthly}.parquet``,
``data/processed/predictors/credit_spreads.parquet`` and ``reports/data/credit.md``.
Raw downloads are cached under ``data/raw/credit/``; ``--refresh`` re-downloads.

Construction (numbers and justification in ``reports/data/credit.md``)
---------------------------------------------------------------------
Tiers (project-wide): A = daily price of an exchange-traded fund (market
close); B = daily open-end mutual-fund NAV, index, or reconstruction; C =
monthly index total return; D = monthly model-based reconstruction.

ig_credit
  * 1919-02 .. 1925-12  monthly, par bond (maturity calibrated to LQD vol) priced off
                        month-AVERAGE Moody's (Aaa+Baa)/2 yields (Fed H.15)    -> D
  * 1926-01 .. 1980-01  monthly, w x Ibbotson LT corporate TR + (1-w) x T-bill
                        (Welch-Goyal ``corpr``/``Rfree``; w = vol match to LQD) -> C
  * 1980-01-03 .. 1984  daily, constant-mix VWESX + FBNDX (fund NAVs)         -> B
  * 1985-01-02 .. 2002-07-30  daily, constant-mix VWESX + VFSTX (fund NAVs)   -> B
  * 2002-07-31 .. today daily, LQD (ETF close + validated distributions)      -> A
hy_credit
  * 1980-02 .. 1984-12  monthly, VWEHX NAV (Yahoo has one NAV per month, on the
                        ex-dividend date, before 1985)                        -> C
  * 1985-01-02 .. 2007-04-11  daily, VWEHX (matrix-priced NAV: stale)         -> B
  * 2007-04-12 .. today daily, HYG (ETF close + validated distributions)      -> A

Every calibration that shapes the pre-ETF history (blend weights, the
Ibbotson/T-bill weight, the tier-D bond maturity) uses LQD data only up to
``CALIBRATION_END`` (2014-12-31): data from 2015 on are the evaluation holdout
and must not influence any series the models train on.

Fund returns are built from the NAV plus distributions, not from Yahoo's
adjclose: VWEHX, VFSTX, FBNDX and VWESX (from 1985) declare income daily and pay
it monthly without an ex-date NAV drop, so each month-end distribution is spread
over its accrual month (``distributions.accrual_total_return``). Confirmed bad
NAV prints are dropped (the next return spans them). ETF distributions are
validated (``distributions.validate_distributions``).

Every return is a SIMPLE TOTAL return in decimals. Nothing is smoothed,
unsmoothed or interpolated; staleness is measured and reported, not corrected.
"""
from __future__ import annotations

import io
import json
import re
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from ...paths import ASSET_DIR, PREDICTOR_DIR, PROCESSED, RAW, REPORTS
from . import distributions as dist

CACHE = RAW / "credit"

# Holdout integrity: the evaluation holdout starts 2015-01-01. Any parameter that is
# FITTED (vol-matched blend weights, the Ibbotson/T-bill weight, the maturity of the
# tier-D par bond) is calibrated on LQD data up to this date only, so that no
# post-2014 observation shapes the pre-2015 history the models are trained on.
CALIBRATION_END = "2014-12-31"

# Funds that declare income daily and pay it monthly (NAV does not drop on the
# month-end payment date), with the first date from which that holds. Before it,
# their distributions are ex-date cash flows (VWESX dropped its NAV on the
# ex-date in 1980-84; VWEHX/VFSTX have daily data only from 1985).
ACCRUAL_FUNDS = {"VWEHX": "1985-01-01", "VFSTX": "1985-01-01", "FBNDX": "1980-01-01",
                 "VWESX": "1985-01-01"}
# Confirmed bad NAV prints: the PRICE on these dates is dropped, so the next return
# spans two days (never interpolated). Evidence in reports/data/credit.md
# ("Bad prints"): each is a one-day move of 1-3.6% that reverses the next day
# while VWESX, the Treasury reconstructions and (for VWEHX) four HY peer funds
# (FHIIX, PRHYX, AHITX, FAGIX) are flat.
BAD_PRICES = {
    "FBNDX": ["1982-10-06", "1983-11-21", "1983-11-28", "1984-02-02", "1984-11-14", "1984-12-06"],
    "VWEHX": ["1991-08-19"],
}
# Misdated distributions: Yahoo books them one day after the NAV dropped. In Jan
# 1987 VWEHX, VWESX and VFSTX all dropped on 01-29 (VFSTX's distribution is dated
# 01-29) while Treasuries and HY peers were flat; VWEHX/VWESX list theirs on 01-30.
# Re-dated distributions are ex-date cash flows on the corrected date.
EXDATE_FIXES = {"VWEHX": {"1987-01-30": "1987-01-29"}, "VWESX": {"1987-01-30": "1987-01-29"}}
# The era in which each fund enters ig_credit / hy_credit (for the report's effect table).
FUND_ERAS = {"VWEHX": ("1985-01-02", "2007-04-11"), "VFSTX": ("1985-01-02", "2002-07-30"),
             "FBNDX": ("1980-01-03", "1984-12-31"), "VWESX": ("1980-01-03", "2002-07-30")}
# Peers for the ETF distribution test (H.15 CMT yield changes; total returns of
# related funds; price returns of daily-accrual funds, whose NAV never drops).
ETF_PEERS = {"LQD": {"cmt": ["RIFLGFCY05_N.B", "RIFLGFCY10_N.B", "RIFLGFCY30_N.B"],
                     "nav": ["VFICX", "VWESX"], "tr": []},
             "HYG": {"cmt": ["RIFLGFCY05_N.B"], "nav": ["VWEHX", "FHIIX"], "tr": ["JNK"]}}
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
YAHOO_URL = "https://{host}.finance.yahoo.com/v8/finance/chart/{sym}"
H15_URL = "https://www.federalreserve.gov/datadownload/Output.aspx?rel=H15&filetype=zip"
EBP_URL = "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv"
# Welch-Goyal predictor file (data up to 2025, GW 2008 + GWZ 2024), linked from
# https://sites.google.com/view/agoyal145 as a Google Sheet.
GOYAL_URL = ("https://docs.google.com/spreadsheets/d/17mw_IpaiLFDrGnrPRQ2o1ugV5nJsZuD1/"
             "export?format=xlsx")


# --------------------------------------------------------------------- fetch
def _get(url: str, params: dict | None = None, tries: int = 6, sleep: float = 2.0) -> bytes:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "*/*"})
    last = None
    for i in range(tries):
        try:
            r = s.get(url, params=params, timeout=90)
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


def yahoo_json(sym: str, refresh: bool = False) -> dict:
    path = CACHE / f"yahoo_{sym}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    params = {"period1": 0, "period2": int(time.time()) + 86400,
              "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"}
    err = None
    for host in ("query1", "query2"):
        try:
            raw = _get(YAHOO_URL.format(host=host, sym=sym), params=params)
            j = json.loads(raw)
            if j.get("chart", {}).get("result"):
                CACHE.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                time.sleep(1.0)  # be polite: other agents share the endpoint
                return j
            err = j.get("chart", {}).get("error")
        except RuntimeError as e:  # pragma: no cover - network
            err = e
    raise RuntimeError(f"yahoo failed for {sym}: {err}")


def yahoo_frame(sym: str, refresh: bool = False) -> pd.DataFrame:
    """Daily close, adjclose and cash distributions (ex-date) from the chart API."""
    j = yahoo_json(sym, refresh)["chart"]["result"][0]
    tz = j["meta"].get("exchangeTimezoneName", "America/New_York")
    idx = (pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert(tz)
           .tz_localize(None).normalize())
    q = j["indicators"]["quote"][0]
    df = pd.DataFrame({"close": q["close"],
                       "adjclose": j["indicators"]["adjclose"][0]["adjclose"]},
                      index=pd.DatetimeIndex(idx, name="date"), dtype=float)
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"])
    ev = {}
    for v in j.get("events", {}).get("dividends", {}).values():
        d = (pd.Timestamp(v["date"], unit="s", tz="UTC").tz_convert(tz)
             .tz_localize(None).normalize())
        ev[d] = ev.get(d, 0.0) + float(v["amount"])
    # exact repeats within a few days are vendor duplicates; an event dated on a
    # non-trading day is booked on the next trading day (never dropped)
    div, _ = dist.dedupe_repeats(pd.Series(ev, dtype=float).sort_index())
    df["div"] = dist.align_dividends(div, df.index)
    return df


def h15_series(names: list[str], refresh: bool = False) -> dict[str, pd.Series]:
    """Series from the Fed H.15 full data package (Moody's yields are in it to 2016-10)."""
    raw = cached("fed_h15.zip", H15_URL, refresh)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        xml = z.read("H15_data.xml").decode("utf-8")
    out = {}
    for name in names:
        i = xml.find(f'SERIES_NAME="{name}"')
        if i < 0:
            raise KeyError(name)
        seg = xml[i:xml.find("</kf:Series>", i)]
        obs = re.findall(r'OBS_STATUS="(\w+)" OBS_VALUE="([^"]+)" TIME_PERIOD="([^"]+)"', seg)
        s = pd.Series({pd.Timestamp(t): float(v) for st, v, t in obs if st == "A"}, dtype=float)
        s.index.name = "date"
        out[name] = s.sort_index()
    return out


def goyal_monthly(refresh: bool = False) -> pd.DataFrame:
    """Welch-Goyal monthly file: AAA, BAA (decimals), corpr (Ibbotson LT corporate TR)."""
    import warnings
    raw = cached("goyal_predictors.xlsx", GOYAL_URL, refresh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = pd.read_excel(io.BytesIO(raw), sheet_name="Monthly")
    d.index = pd.to_datetime(d["yyyymm"].astype(int).astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    d.index.name = "date"
    return d[["AAA", "BAA", "lty", "corpr", "Rfree"]].astype(float)


def par_bond_returns(y: pd.Series, maturity: float, periods_per_year: int = 12) -> pd.Series:
    """Total return of a constant-maturity par bond rolled every period.

    At t-1 buy a par bond (semi-annual coupon = y[t-1], ``maturity`` years); at t
    reprice it at yield y[t] with maturity reduced by one period. Return =
    price change + coupon accrual. ``y`` in decimals. NaN where y is missing.
    """
    y = y.astype(float)
    c = y.shift(1)
    n = 2 * (maturity - 1.0 / periods_per_year)  # remaining semi-annual periods
    h = y / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        annuity = np.where(h.abs() > 1e-12, (1 - (1 + h) ** (-n)) / h, n)
        price = (c / 2) * annuity + (1 + h) ** (-n)
    r = pd.Series(price, index=y.index) - 1 + c / periods_per_year
    return r


def fed_ebp(refresh: bool = False) -> pd.DataFrame:
    """Gilchrist-Zakrajsek credit spread and excess bond premium (monthly, percent)."""
    d = pd.read_csv(io.BytesIO(cached("fed_ebp.csv", EBP_URL, refresh)))
    d.index = pd.to_datetime(d["date"]) + pd.offsets.MonthEnd(0)
    d.index.name = "date"
    return d[["gz_spread", "ebp"]].astype(float)


# ------------------------------------------------------------------ returns
# Before 1985 Yahoo only has one NAV per month (on the ex-dividend date, a few
# days before month-end) for these funds, forward-filled to look daily. Daily
# returns are used only from these dates on.
DAILY_START = {"VWEHX": "1985-01-02", "VFSTX": "1985-01-02",
               "FHIIX": "1985-01-02", "FAGIX": "1985-01-02"}
IG_SYMBOLS = ["LQD", "VWESX", "VFSTX", "FBNDX", "VFICX"]
HY_SYMBOLS = ["HYG", "JNK", "VWEHX", "FHIIX", "PRHYX", "SPHIX", "AHITX"]
ETF_START = {"ig_credit": "2002-07-31", "hy_credit": "2007-04-12"}  # first ETF return used
# monthly-from-daily starts with the first month whose base price is in the
# previous month (VWESX/FBNDX's first Yahoo price is 1980-01-02).
MONTHLY_FROM_DAILY = {"ig_credit": "1980-02", "hy_credit": "1985-01"}


def yahoo_returns(sym: str, refresh: bool = False) -> pd.Series:
    """Daily simple total returns from Yahoo adjclose (distributions reinvested)."""
    df = yahoo_frame(sym, refresh)
    r = df["adjclose"].pct_change().dropna()
    if sym in DAILY_START:
        r = r.loc[DAILY_START[sym]:]
    r.name = sym
    return r


def fund_returns(sym: str, refresh: bool = False) -> tuple[pd.Series, pd.DataFrame, list]:
    """Daily total return of a daily-accrual bond fund from NAV + distributions.

    Misdated distributions are re-dated (``EXDATE_FIXES``), confirmed bad prices are
    dropped (``BAD_PRICES``), and month-end distributions from ``ACCRUAL_FUNDS[sym]``
    on are spread over their accrual month. Returns (returns from ``DAILY_START``,
    distribution classification table, dropped price dates)."""
    df = yahoo_frame(sym, refresh)[["close", "div"]].copy()
    for old, new in EXDATE_FIXES.get(sym, {}).items():
        old, new = pd.Timestamp(old), pd.Timestamp(new)
        if old in df.index and new in df.index and df.at[old, "div"] > 0:
            df.at[new, "div"] += df.at[old, "div"]
            df.at[old, "div"] = 0.0
    df, dropped = dist.drop_bad_prices(df, BAD_PRICES.get(sym, []))
    r, cls = dist.accrual_total_return(df, accrual_start=ACCRUAL_FUNDS.get(sym))
    r = r.dropna()
    if sym in DAILY_START:
        r = r.loc[DAILY_START[sym]:]
    return r.rename(sym), cls, dropped


def etf_returns(sym: str, refresh: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    """ETF total return from close + VALIDATED distributions (see ``ETF_PEERS``)."""
    df = yahoo_frame(sym, refresh)
    spec = ETF_PEERS[sym]
    h = h15_series(spec["cmt"], refresh)
    peers = {k: h[k].diff() / 100.0 for k in spec["cmt"]}
    for s in spec["nav"]:
        peers[s] = yahoo_frame(s, refresh)["close"].pct_change()
    for s in spec["tr"]:
        peers[s] = yahoo_frame(s, refresh)["adjclose"].pct_change()
    div, table = dist.validate_distributions(df["close"], df["div"], pd.DataFrame(peers))
    r = dist.total_return(df["close"], div).iloc[1:].dropna()
    return r.rename(sym), table


def sparse_monthly(sym: str, refresh: bool = False) -> pd.Series:
    """Monthly returns from the pre-daily era (one NAV per month on the ex-div date).

    The NAV used for month m is the last one dated in m (typically the 25th-30th),
    so each return spans ex-date to ex-date, not month-end to month-end. A month
    with no new NAV is a gap: it and the following month are set to NaN.
    """
    df = yahoo_frame(sym, refresh)
    px = df["adjclose"].loc[: pd.Timestamp(DAILY_START[sym]) - pd.Timedelta(days=1)]
    changed = px.diff().ne(0)
    month_has_obs = changed.groupby(pd.Grouper(freq="ME")).any()
    m = px.resample("ME").last()
    r = m.pct_change()
    bad = ~month_has_obs.reindex(r.index, fill_value=False)
    r[bad | bad.shift(1, fill_value=False)] = np.nan
    return r.iloc[1:]


def resample_ret(r: pd.Series | pd.DataFrame, freq: str):
    """Compound simple daily returns to weekly ("W-FRI") or monthly ("ME")."""
    if freq == "D":
        return r
    return np.expm1(np.log1p(r).resample(freq).sum(min_count=1))


def overlap_stats(a: pd.Series, b: pd.Series, freq: str = "M", start=None, end=None) -> dict:
    """Compare candidate ``a`` with target ``b`` on their common dates at ``freq``."""
    ann = {"D": 252, "W": 52, "M": 12}[freq]
    code = {"D": "D", "W": "W-FRI", "M": "ME"}[freq]
    x = pd.concat([resample_ret(a.loc[start:end], code), resample_ret(b.loc[start:end], code)],
                  axis=1, keys=["a", "b"], sort=True).dropna()
    la, lb = np.log1p(x["a"]), np.log1p(x["b"])
    return {"freq": freq, "n": len(x), "start": x.index[0].date(), "end": x.index[-1].date(),
            "corr": la.corr(lb), "vol_a": la.std() * np.sqrt(ann), "vol_b": lb.std() * np.sqrt(ann),
            "vol_ratio": la.std() / lb.std(), "te": (la - lb).std() * np.sqrt(ann)}


def blend_returns(w: float, a: pd.Series, b: pd.Series) -> pd.Series:
    """Daily constant-mix blend w*a + (1-w)*b, rebalanced on the days both funds have a
    price. A day on which one fund has no price (e.g. a dropped bad print) is not a blend
    day: each fund's returns are compounded onto the next common day, so the next blend
    return spans the gap for BOTH funds (nothing is lost, nothing interpolated)."""
    common = a.dropna().index.intersection(b.dropna().index)

    def onto(r: pd.Series) -> pd.Series:
        r = r.dropna()
        r = r.loc[r.index >= common[0]]
        pos = common.searchsorted(r.index, side="left")
        keep = pos < len(common)
        lr = np.log1p(r[keep])
        return np.expm1(lr.groupby(common[pos[keep]]).sum())

    return (w * onto(a) + (1 - w) * onto(b)).dropna()


def vol_match_weight(a: pd.Series, b: pd.Series, target: pd.Series) -> float:
    """w in [0,1] such that the monthly vol of w*a + (1-w)*b equals that of target."""
    x = pd.concat([a, b, target], axis=1, keys=["a", "b", "t"], sort=True).dropna()
    goal = np.log1p(x["t"]).std()
    grid = np.linspace(0, 1, 1001)
    vols = np.array([np.log1p(w * x["a"] + (1 - w) * x["b"]).std() for w in grid])
    return float(grid[np.argmin(np.abs(vols - goal))])


def stale_stats(r: pd.Series, start=None, end=None) -> dict | None:
    """Stale-pricing diagnostics of daily simple returns in [start, end]."""
    r = r.loc[start:end].dropna()
    if len(r) < 200:
        return None
    lr = np.log1p(r)
    w = np.log1p(resample_ret(r, "W-FRI")).dropna()
    m = np.log1p(resample_ret(r, "ME")).dropna()
    vd, vw, vm = lr.std() * np.sqrt(252), w.std() * np.sqrt(52), m.std() * np.sqrt(12)
    out = {"n": len(r), "zero_frac": float((r == 0).mean())}
    for k in range(1, 6):
        out[f"ac{k}"] = lr.autocorr(k)
    out.update(vol_d=vd, vol_w=vw, vol_m=vm, w_over_d=vw / vd, m_over_d=vm / vd)
    return out


def _splice(pieces: list[tuple[pd.Series, str, str]]) -> pd.DataFrame:
    """Each later piece overrides earlier ones from its first date onwards."""
    out = None
    for r, src, tier in pieces:
        r = r.dropna()
        df = pd.DataFrame({"ret": r.astype(float), "source": src, "tier": tier})
        out = df if out is None else pd.concat([out.loc[: r.index[0] - pd.Timedelta(days=1)], df])
    out.index.name = "date"
    out["source"] = out["source"].astype(str)
    out["tier"] = out["tier"].astype(str)
    return out


def to_monthly(daily: pd.DataFrame, first: str, last: pd.Timestamp) -> pd.DataFrame:
    """Compound daily to calendar month-ends; source/tier = the dominant segment."""
    d = daily.loc[first:]
    g = d.groupby(pd.Grouper(freq="ME"))
    m = pd.DataFrame({
        "ret": np.expm1(g["ret"].apply(lambda x: np.log1p(x).sum() if len(x) else np.nan)),
        "source": g["source"].agg(lambda x: x.mode().iloc[0] if len(x) else None),
        "tier": g["tier"].agg(lambda x: x.mode().iloc[0] if len(x) else None),
    })
    return m.loc[:last].dropna(subset=["ret"])


def last_complete_month(daily_index: pd.DatetimeIndex) -> pd.Timestamp:
    d = daily_index[-1]
    if d == d + pd.offsets.BMonthEnd(0):
        return d + pd.offsets.MonthEnd(0)
    return d - pd.offsets.MonthEnd(1)


def segments_of(df: pd.DataFrame, asset: str, freq: str, notes: dict | None = None) -> list[dict]:
    key = df["source"].ne(df["source"].shift()).cumsum()
    out = []
    for _, g in df.groupby(key, sort=False):
        src = g["source"].iloc[0]
        out.append({"asset": asset, "start": g.index[0].date(), "end": g.index[-1].date(),
                    "freq": freq, "source": src, "tier": g["tier"].iloc[0],
                    "notes": (notes or {}).get(src, "")})
    return out


# Source labels (the ``source`` column).
SRC = {
    "LQD": "yahoo_LQD", "HYG": "yahoo_HYG", "VWEHX": "yahoo_VWEHX",
    "VWEHX_m": "yahoo_VWEHX_exdiv_monthly",
    "ig80": "blend_VWESX_FBNDX", "ig85": "blend_VWESX_VFSTX",
    "corp": "ibbotson_ltcorp_rf_blend", "recon": "moodys_aaa_baa_avg_parbond",
}
TIER_D_MATURITIES = list(range(6, 31))


def calibrate_recon_maturity(lqd_m: pd.Series, refresh: bool = False,
                             end: str = CALIBRATION_END) -> tuple[int, pd.DataFrame]:
    """Pick the par-bond maturity whose returns from END-of-month Moody's yields
    (daily H.15, 1986-2016) match LQD's monthly vol on 2002-08..``end`` (the
    calibration cutoff). The tier-D series then uses that maturity with
    month-AVERAGE yields (the only ones before 1986), so its residual vol damping is
    purely the averaging effect."""
    h = h15_series(["RIMLPAAAR_N.B", "RIMLPBAAR_N.B"], refresh)
    yd = ((h["RIMLPAAAR_N.B"] + h["RIMLPBAAR_N.B"]) / 200).dropna()
    yeom = yd.resample("ME").last()
    rows = []
    for n in TIER_D_MATURITIES:
        s = overlap_stats(par_bond_returns(yeom, n), lqd_m, "M", "2002-08", end)
        rows.append({"maturity": n, "vol_ratio_vs_LQD": s["vol_ratio"], "corr": s["corr"]})
    t = pd.DataFrame(rows).set_index("maturity")
    return int((t["vol_ratio_vs_LQD"] - 1).abs().idxmin()), t


def build(refresh: bool = False, write: bool = True) -> dict:
    """Build both assets and the predictor file; return diagnostics for the report."""
    rets = {s: yahoo_returns(s, refresh) for s in IG_SYMBOLS + HY_SYMBOLS}   # adjclose (diagnostics)
    raw_adj = dict(rets)
    fund_cls, dropped, div_tables = {}, {}, {}
    for s in ACCRUAL_FUNDS:                     # funds used in the series: NAV + accrual-spread income
        rets[s], fund_cls[s], dropped[s] = fund_returns(s, refresh)
    for s in ETF_PEERS:                         # ETFs used in the series: close + validated distributions
        rets[s], div_tables[s] = etf_returns(s, refresh)
    g = goyal_monthly(refresh)
    h = h15_series(["RIMLPAAAR_N.M", "RIMLPBAAR_N.M"], refresh)
    diag: dict = {"rets": rets, "raw_adj": raw_adj, "fund_cls": fund_cls, "dropped": dropped,
                  "div_tables": div_tables}
    M = {s: resample_ret(r, "ME") for s, r in rets.items()}
    lqd, hyg = rets["LQD"], rets["HYG"]
    last_m = last_complete_month(lqd.index)
    # calibration window: LQD's first full month .. CALIBRATION_END (never the holdout)
    cal = (ETF_START["ig_credit"][:7], CALIBRATION_END)

    # ---------------- ig_credit, daily: two fund blends, then LQD
    w80 = vol_match_weight(M["VWESX"].loc[cal[0]:cal[1]], M["FBNDX"].loc[cal[0]:cal[1]], M["LQD"].loc[cal[0]:cal[1]])
    w85 = vol_match_weight(M["VWESX"].loc[cal[0]:cal[1]], M["VFSTX"].loc[cal[0]:cal[1]], M["LQD"].loc[cal[0]:cal[1]])
    ig80 = blend_returns(w80, rets["VWESX"], rets["FBNDX"])
    ig85 = blend_returns(w85, rets["VWESX"], rets["VFSTX"])
    ig_d = _splice([(ig80, SRC["ig80"], "B"), (ig85, SRC["ig85"], "B"),
                    (lqd.loc[ETF_START["ig_credit"]:], SRC["LQD"], "A")])
    ig_m = to_monthly(ig_d, MONTHLY_FROM_DAILY["ig_credit"], last_m)

    # ---------------- ig_credit, monthly: Ibbotson LT corp + T-bill blend (C), Moody's recon (D)
    wc = vol_match_weight(g["corpr"].loc[cal[0]:cal[1]], g["Rfree"].loc[cal[0]:cal[1]], M["LQD"].loc[cal[0]:cal[1]])
    corp = (wc * g["corpr"] + (1 - wc) * g["Rfree"]).dropna()
    n_star, mat_table = calibrate_recon_maturity(M["LQD"], refresh, CALIBRATION_END)
    # what the full-sample (holdout-contaminated) calibration would have given, for the report only
    full = (ETF_START["ig_credit"][:7], last_m)
    diag["calib_full_sample"] = dict(
        w80=vol_match_weight(M["VWESX"].loc[full[0]:full[1]], M["FBNDX"].loc[full[0]:full[1]], M["LQD"].loc[full[0]:full[1]]),
        w85=vol_match_weight(M["VWESX"].loc[full[0]:full[1]], M["VFSTX"].loc[full[0]:full[1]], M["LQD"].loc[full[0]:full[1]]),
        wc=vol_match_weight(g["corpr"].loc[full[0]:"2025-12"], g["Rfree"].loc[full[0]:"2025-12"],
                            M["LQD"].loc[full[0]:"2025-12"]),
        n_star=calibrate_recon_maturity(M["LQD"], refresh, "2016-09")[0])
    yavg = ((h["RIMLPAAAR_N.M"] + h["RIMLPBAAR_N.M"]) / 200).dropna()
    recon = par_bond_returns(yavg, n_star).dropna()
    first_daily_m = pd.Timestamp(MONTHLY_FROM_DAILY["ig_credit"]) + pd.offsets.MonthEnd(0)
    pre_c = corp.loc[: first_daily_m - pd.Timedelta(days=1)]
    pre_d = recon.loc[: pre_c.index[0] - pd.Timedelta(days=1)]
    ig_monthly = pd.concat([
        pd.DataFrame({"ret": pre_d, "source": SRC["recon"], "tier": "D"}),
        pd.DataFrame({"ret": pre_c, "source": SRC["corp"], "tier": "C"}),
        ig_m])

    # ---------------- hy_credit: VWEHX (monthly sparse, then daily), then HYG
    hy_d = _splice([(rets["VWEHX"], SRC["VWEHX"], "B"),
                    (hyg.loc[ETF_START["hy_credit"]:], SRC["HYG"], "A")])
    hy_m = to_monthly(hy_d, MONTHLY_FROM_DAILY["hy_credit"], last_m)
    sp = sparse_monthly("VWEHX", refresh).dropna()
    hy_monthly = pd.concat([pd.DataFrame({"ret": sp, "source": SRC["VWEHX_m"], "tier": "C"}), hy_m])

    out = {}
    for asset, d, m in (("ig_credit", ig_d, ig_monthly), ("hy_credit", hy_d, hy_monthly)):
        m.index.name = "date"
        m["source"] = m["source"].astype(str)
        m["tier"] = m["tier"].astype(str)
        out[asset] = (d, m)
        if write:
            ASSET_DIR.mkdir(parents=True, exist_ok=True)
            d.to_parquet(ASSET_DIR / f"{asset}_daily.parquet")
            m.to_parquet(ASSET_DIR / f"{asset}_monthly.parquet")

    pred = build_predictors(refresh)
    if write:
        PREDICTOR_DIR.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(PREDICTOR_DIR / "credit_spreads.parquet", index=False)

    diag.update(out=out, w80=w80, w85=w85, wc=wc, n_star=n_star, mat_table=mat_table,
                ig80=ig80, ig85=ig85, corp=corp, recon=recon, yavg=yavg, goyal=g,
                sparse=sp, last_m=last_m, pred=pred)
    diag["segments"] = segment_table(diag)
    if write:
        diag["segments"].to_csv(CACHE / "segments_credit.csv", index=False)
        (PROCESSED / "segments").mkdir(parents=True, exist_ok=True)
        diag["segments"].to_csv(PROCESSED / "segments" / "credit.csv", index=False)
        pd.concat(div_tables, names=["etf", "date"]).to_csv(CACHE / "distribution_checks.csv")
    return diag


# --------------------------------------------------------------- predictors
def build_predictors(refresh: bool = False) -> pd.DataFrame:
    """Long-format credit-spread predictors (percentage points).

    available_date: Moody's / Treasury yields are published daily with a one
    business-day lag, so a monthly average is known on the first business day
    after month-end and a daily value on the next business day. The GZ spread
    file is a research product updated with a lag and revised: we use one full
    month after the observation month (EBP is additionally estimated on the full
    sample, i.e. it contains look-ahead in its loadings; use with care).
    """
    h = h15_series(["RIMLPAAAR_N.M", "RIMLPBAAR_N.M", "RIMLPAAAR_N.B", "RIMLPBAAR_N.B",
                    "RIFLGFCY10_N.M", "RIFLGFCY10_N.B"], refresh)
    g = goyal_monthly(refresh)
    # monthly Moody's: H.15 to 2016-09, then Welch-Goyal (same Moody's/FRED data, identical on the overlap)
    aaa = h["RIMLPAAAR_N.M"].copy()
    baa = h["RIMLPBAAR_N.M"].copy()
    aaa.index = aaa.index + pd.offsets.MonthEnd(0)
    baa.index = baa.index + pd.offsets.MonthEnd(0)
    aaa = aaa.combine_first(g["AAA"] * 100)
    baa = baa.combine_first(g["BAA"] * 100)
    t10 = h["RIFLGFCY10_N.M"].copy()
    t10.index = t10.index + pd.offsets.MonthEnd(0)
    nb = pd.offsets.BDay(1)
    frames = []

    def add(s: pd.Series, name: str, avail):
        s = s.dropna()
        frames.append(pd.DataFrame({"obs_date": s.index, "available_date": avail(s.index),
                                    "series": name, "value": s.values.astype(float)}))

    monthly_avail = lambda idx: idx + nb  # noqa: E731
    add(aaa, "moodys_aaa_yield_m", monthly_avail)
    add(baa, "moodys_baa_yield_m", monthly_avail)
    add(baa - aaa, "moodys_baa_minus_aaa_m", monthly_avail)
    add((baa - t10).dropna(), "moodys_baa_minus_ust10y_m", monthly_avail)
    daily_avail = lambda idx: idx + nb  # noqa: E731
    da, db, d10 = h["RIMLPAAAR_N.B"], h["RIMLPBAAR_N.B"], h["RIFLGFCY10_N.B"]
    add((db - da).dropna(), "moodys_baa_minus_aaa_d", daily_avail)
    add((db - d10).dropna(), "moodys_baa_minus_ust10y_d", daily_avail)
    e = fed_ebp(refresh)
    lag = lambda idx: idx + pd.offsets.MonthEnd(1)  # noqa: E731
    add(e["gz_spread"], "gz_credit_spread_m", lag)
    add(e["ebp"], "gz_excess_bond_premium_m", lag)
    p = pd.concat(frames, ignore_index=True)
    p["obs_date"] = pd.to_datetime(p["obs_date"]).astype("datetime64[ns]")
    p["available_date"] = pd.to_datetime(p["available_date"]).astype("datetime64[ns]")
    p["series"] = p["series"].astype(str)
    return p.sort_values(["series", "obs_date"]).reset_index(drop=True)


# ---------------------------------------------------------------- diagnostics
def splice_checks(diag: dict) -> dict[str, list[dict]]:
    """Overlap statistics behind every splice (candidate vs target)."""
    r = diag["rets"]
    last = diag["last_m"]
    lqd, hyg = r["LQD"], r["HYG"]
    ov_ig, ov_hy = (ETF_START["ig_credit"], last), (ETF_START["hy_credit"], last)
    chk = {}
    chk["blend_VWESX_FBNDX vs LQD"] = [overlap_stats(diag["ig80"], lqd, f, *ov_ig) for f in "DWM"]
    chk["blend_VWESX_VFSTX vs LQD"] = [overlap_stats(diag["ig85"], lqd, f, *ov_ig) for f in "DWM"]
    chk["VWESX alone vs LQD"] = [overlap_stats(r["VWESX"], lqd, f, *ov_ig) for f in "DWM"]
    chk["blend_VWESX_FBNDX vs blend_VWESX_VFSTX (1985-2002)"] = [
        overlap_stats(diag["ig80"], diag["ig85"], f, "1985-01-02", "2002-07-30") for f in "DWM"]
    ig_daily_m = resample_ret(diag["out"]["ig_credit"][0]["ret"], "ME")
    chk["ibbotson_ltcorp_rf_blend vs LQD"] = [overlap_stats(diag["corp"], resample_ret(lqd, "ME"), "M", "2002-08", "2025-12")]
    chk["ibbotson_ltcorp_rf_blend vs fund blends (1980-2002)"] = [
        overlap_stats(diag["corp"], ig_daily_m, "M", "1980-02", "2002-07")]
    chk["ibbotson corpr (unblended) vs LQD"] = [overlap_stats(diag["goyal"]["corpr"], resample_ret(lqd, "ME"), "M", "2002-08", "2025-12")]
    chk["ibbotson corpr (unblended) vs VWESX"] = [overlap_stats(diag["goyal"]["corpr"], resample_ret(r["VWESX"], "ME"), "M", "1980-02", "2025-12")]
    chk["moodys recon (month-avg yields) vs ibbotson blend (1926-1979)"] = [
        overlap_stats(diag["recon"], diag["corp"], "M", "1926-01", "1979-12")]
    chk["moodys recon (month-avg yields) vs LQD"] = [
        overlap_stats(diag["recon"], resample_ret(lqd, "ME"), "M", "2002-08", "2016-09")]
    chk["moodys recon (month-avg yields) vs fund blends (1980-2002)"] = [
        overlap_stats(diag["recon"], ig_daily_m, "M", "1980-02", "2002-07")]
    chk["VWEHX vs HYG"] = [overlap_stats(r["VWEHX"], hyg, f, *ov_hy) for f in "DWM"]
    for s in ["JNK", "FHIIX", "PRHYX", "SPHIX", "AHITX"]:
        chk[f"{s} vs HYG"] = [overlap_stats(r[s], hyg, f, *ov_hy) for f in "DWM"]
    comp = pd.concat([r["VWEHX"], r["FHIIX"], r["PRHYX"]], axis=1, sort=True).mean(axis=1, skipna=False)
    chk["equal-weight VWEHX/FHIIX/PRHYX vs HYG"] = [overlap_stats(comp, hyg, f, *ov_hy) for f in "DWM"]
    # effect of ex-dividend-date sampling (pre-1985 HY monthly), measured on 1985-2026
    chk["VWEHX ex-div-date-sampled vs month-end (1985-2026)"] = [exdiv_sampling_check()]
    return chk


def horizon_vol_ratio(a: pd.Series, b: pd.Series, k: int, start=None, end=None) -> float:
    """Ratio of annualized vols of overlapping k-day log-return sums (uncentered, as in
    the 21-day target definition) of ``a`` and ``b`` on their common days."""
    x = np.log1p(pd.concat([a, b], axis=1, keys=["a", "b"], sort=True).loc[start:end].dropna())
    s = x.rolling(k).sum().dropna()
    return float(np.sqrt((s["a"] ** 2).mean() / (s["b"] ** 2).mean()))


def vwehx_hyg_horizon(diag: dict) -> dict:
    v, h = diag["rets"]["VWEHX"], diag["rets"]["HYG"]
    a0 = ETF_START["hy_credit"]
    return {"r1": horizon_vol_ratio(v, h, 1, a0), "r21": horizon_vol_ratio(v, h, 21, a0),
            "r21_0709": horizon_vol_ratio(v, h, 21, a0, "2009-12-31"), "r63": horizon_vol_ratio(v, h, 63, a0),
            "r21_dev": horizon_vol_ratio(v, h, 21, a0, CALIBRATION_END)}


def adjclose_check(symbols: list[str], refresh: bool = False) -> pd.DataFrame:
    """Do Yahoo adjclose returns equal (close + distribution) / previous close?"""
    rows = []
    for s in symbols:
        df = yahoo_frame(s, refresh)
        if s in DAILY_START:
            df = df.loc[pd.Timestamp(DAILY_START[s]) - pd.Timedelta(days=7):]
        ra = df["adjclose"].pct_change()
        ro = (df["close"] + df["div"]) / df["close"].shift() - 1
        dd = (ra - ro).abs().dropna()
        n_years = len(dd) / 252
        rows.append({"symbol": s, "first": df.index[0].date(), "n_div": int((df["div"] > 0).sum()),
                     "median_abs_diff": dd.median(), "p99_abs_diff": dd.quantile(0.99),
                     "ann_TR_adjclose": np.expm1(np.log1p(ra.dropna()).sum() / n_years),
                     "ann_TR_close_div": np.expm1(np.log1p(ro.dropna()).sum() / n_years)})
    return pd.DataFrame(rows)


def lead_lag(fund: pd.Series, etf: pd.Series, start, end, lags=range(-1, 4)) -> dict:
    """corr(fund_t, etf_{t-k}): positive k = fund reacts k days after the ETF."""
    x = pd.concat([fund, etf], axis=1, keys=["f", "e"], sort=True).loc[start:end].dropna()
    return {k: x["f"].corr(x["e"].shift(k)) for k in lags}


def exdiv_sampling_check(refresh: bool = False) -> dict:
    df = yahoo_frame("VWEHX", refresh).loc["1985-01-01":]
    px = df["adjclose"]
    exd = df.index[df["div"] > 0]
    s = px.loc[exd]
    s = s.groupby(s.index.to_period("M")).last()
    s.index = s.index.to_timestamp("M")
    a = s.pct_change().dropna()
    b = px.resample("ME").last().pct_change().dropna()
    return overlap_stats_monthly(a, b)


def overlap_stats_monthly(a: pd.Series, b: pd.Series) -> dict:
    x = pd.concat([a, b], axis=1, keys=["a", "b"], sort=True).dropna()
    la, lb = np.log1p(x["a"]), np.log1p(x["b"])
    return {"freq": "M", "n": len(x), "start": x.index[0].date(), "end": x.index[-1].date(),
            "corr": la.corr(lb), "vol_a": la.std() * np.sqrt(12), "vol_b": lb.std() * np.sqrt(12),
            "vol_ratio": la.std() / lb.std(), "te": (la - lb).std() * np.sqrt(12)}


def segment_table(diag: dict) -> pd.DataFrame:
    ch = splice_checks(diag)
    diag["checks"] = ch

    def short(key, i=-1):
        s = ch[key][i]
        return f"{s['freq']} {s['start']}..{s['end']}: corr {s['corr']:.2f}, vol ratio {s['vol_ratio']:.2f}, TE {s['te']:.3f}"

    w80, w85, wc = diag["w80"], diag["w85"], diag["wc"]
    hz = diag["vwehx_hyg_horizon"] = vwehx_hyg_horizon(diag)
    calw = f"vol match to LQD, monthly, {ETF_START['ig_credit'][:7]}..{CALIBRATION_END[:7]} only (holdout excluded)"
    acc = ("fund NAV + distributions, month-end income spread over its accrual month (daily-accrual fund); "
           "open-end NAVs: stale pricing, daily vol understated")

    def div_note(sym):
        ch = dist.changed_distributions(diag["div_tables"][sym])
        if not len(ch):
            return "every Yahoo distribution passed the ex-date price test"
        return "Yahoo distributions corrected: " + "; ".join(
            f"{d.date()} {r.amount:.3f} -> {r.used:.3f} (price test z {r.z:.1f})" for d, r in ch.iterrows())

    def bad(sym):
        return f"; bad NAV prints dropped: {', '.join(diag['dropped'][sym])}" if diag["dropped"].get(sym) else ""
    notes = {
        SRC["ig80"]: (f"daily constant-mix {w80:.2f} VWESX + {1-w80:.2f} FBNDX (weight = {calw}); "
                      f"vs LQD {short('blend_VWESX_FBNDX vs LQD')}; vs VWESX/VFSTX blend {short('blend_VWESX_FBNDX vs blend_VWESX_VFSTX (1985-2002)')}; "
                      f"{acc}{bad('FBNDX')}"),
        SRC["ig85"]: (f"daily constant-mix {w85:.2f} VWESX + {1-w85:.2f} VFSTX (weight = {calw}); "
                      f"vs LQD {short('blend_VWESX_VFSTX vs LQD')}; daily {short('blend_VWESX_VFSTX vs LQD', 0)}; "
                      f"{acc}; VWESX 1987-01 distribution re-dated to its NAV drop (01-29)"),
        SRC["LQD"]: f"iShares iBoxx IG corporate ETF, close + validated distributions; {div_note('LQD')}",
        SRC["corp"]: (f"{wc:.2f} x Ibbotson/SBBI long-term corporate TR + {1-wc:.2f} x T-bill (Welch-Goyal corpr, Rfree), "
                      f"duration-shortened to LQD (weight = {calw}); vs LQD {short('ibbotson_ltcorp_rf_blend vs LQD')}; "
                      f"vs fund blends {short('ibbotson_ltcorp_rf_blend vs fund blends (1980-2002)')}"),
        SRC["recon"]: (f"{diag['n_star']}y constant-maturity par bond priced off month-AVERAGE (Aaa+Baa)/2 Moody's yields (H.15) "
                       f"(maturity = vol match to LQD on {ETF_START['ig_credit'][:7]}..{CALIBRATION_END[:7]}); "
                       f"vs Ibbotson blend {short('moodys recon (month-avg yields) vs ibbotson blend (1926-1979)')}; "
                       f"vs LQD {short('moodys recon (month-avg yields) vs LQD')}; averaging damps vol and blurs timing"),
        SRC["VWEHX"]: (f"Vanguard High-Yield Corporate NAV; vs HYG {short('VWEHX vs HYG')}; daily {short('VWEHX vs HYG', 0)}; "
                       f"TARGET-HORIZON VOL LEVEL: 21-day-horizon vol ratio VWEHX/HYG {hz['r21']:.2f} on the 2007-04..2026 "
                       f"overlap ({hz['r21_0709']:.2f} in 2007-09, {hz['r21_dev']:.2f} on 2007-04..2014), {hz['r63']:.2f} at "
                       f"63 days, {hz['r1']:.2f} at 1 day (independent audit: 0.83 overall, 0.80 in 2007-09, 0.93 at 63 days): "
                       "this era's vol level is ~15-20% below HYG's even at 21 days, so do not pool it with HYG's level "
                       "(tier B vs A marks the break); "
                       f"matrix-priced NAV, positive autocorrelation; {acc.split(';')[0]}; "
                       f"1987-01 distribution re-dated to its NAV drop (01-29){bad('VWEHX')}"),
        SRC["VWEHX_m"]: ("Yahoo has one NAV per month before 1985 (ex-div date, ~25th-30th): returns span ex-date to ex-date; "
                         f"sampling effect measured 1985-2026: {short('VWEHX ex-div-date-sampled vs month-end (1985-2026)')}"),
        SRC["HYG"]: f"iShares iBoxx HY corporate ETF, close + validated distributions; {div_note('HYG')}",
    }
    rows = []
    for asset, (d, m) in diag["out"].items():
        pre = m.loc[: d.index[0] + pd.offsets.MonthEnd(0)]
        pre = pre[~pre["source"].isin(d["source"].unique())]
        rows += segments_of(pre, asset, "M", notes)
        rows += segments_of(d, asset, "D", notes)
    return pd.DataFrame(rows, columns=["asset", "start", "end", "freq", "source", "tier", "notes"])


def stale_table(diag: dict) -> pd.DataFrame:
    r = dict(diag["rets"])
    r["IG blend VWESX/FBNDX"] = diag["ig80"]
    r["IG blend VWESX/VFSTX"] = diag["ig85"]
    eras = [("1980-01", "1989-12"), ("1990-01", "1999-12"), ("2000-01", "2007-03"),
            ("2007-04", "2009-12"), ("2010-01", "2019-12"), ("2020-01", "2026-11")]
    order = ["LQD", "IG blend VWESX/FBNDX", "IG blend VWESX/VFSTX", "VWESX", "VFSTX", "FBNDX", "VFICX",
             "HYG", "JNK", "VWEHX", "FHIIX", "PRHYX", "SPHIX", "AHITX"]
    rows = []
    for s in order:
        for a, b in eras:
            st = stale_stats(r[s], a, b)
            if st:
                rows.append({"source": s, "era": f"{a[:4]}-{b[:4]}" if a[:4] != b[:4] else a[:4], **st})
    return pd.DataFrame(rows)


def decade_vols(diag: dict) -> pd.DataFrame:
    rows = []
    for asset, (d, m) in diag["out"].items():
        ld, lm = np.log1p(d["ret"]), np.log1p(m["ret"])
        for dec in range(1910, 2030, 10):
            a, b = f"{dec}", f"{dec + 9}"
            x, y = ld.loc[a:b], lm.loc[a:b]
            if len(y) < 12:
                continue
            rows.append({"asset": asset, "decade": f"{dec}s",
                         "vol_daily": x.std() * np.sqrt(252) if len(x) > 200 else np.nan,
                         "vol_monthly": y.std() * np.sqrt(12), "n_months": len(y),
                         "tiers": "".join(sorted(m.loc[a:b, "tier"].unique()))})
    return pd.DataFrame(rows)


EPISODES = [("1987-10", "1987-10", "Oct 1987 crash"), ("1990-01", "1990-12", "1990 HY bust (calendar year)"),
            ("1991-01", "1991-12", "1991 HY rebound"), ("1994-01", "1994-12", "1994 bond rout"),
            ("1998-08", "1998-10", "LTCM / Russia (Aug-Oct 1998)"),
            ("2002-05", "2002-07", "WorldCom / 2002 (May-Jul)"), ("2008-09", "2008-11", "Lehman (Sep-Nov 2008)"),
            ("2008-01", "2008-12", "2008 calendar year"), ("2009-01", "2009-12", "2009 calendar year"),
            ("2020-03", "2020-03", "Mar 2020"), ("2022-01", "2022-12", "2022 calendar year")]


def episode_table(diag: dict) -> pd.DataFrame:
    rows = []
    ig = diag["out"]["ig_credit"][1]["ret"]
    hy = diag["out"]["hy_credit"][1]["ret"]
    lqd = resample_ret(diag["rets"]["LQD"], "ME")
    hyg = resample_ret(diag["rets"]["HYG"], "ME")
    corpr = diag["goyal"]["corpr"]

    def cum(s, a, b):
        x = s.loc[a:b]
        n_exp = len(pd.period_range(a, b, freq="M"))
        return np.expm1(np.log1p(x).sum()) if len(x.dropna()) == n_exp else np.nan
    for a, b, lab in EPISODES:
        rows.append({"episode": lab, "ig_credit": cum(ig, a, b), "LQD": cum(lqd, a, b),
                     "ibbotson_ltcorp": cum(corpr, a, b), "hy_credit": cum(hy, a, b), "HYG": cum(hyg, a, b)})
    return pd.DataFrame(rows)




# -------------------------------------------------------------------- report
def distribution_report(diag: dict) -> str:
    """Markdown section: ETF distribution validation, fund accrual treatment, bad prints."""
    L = ["## Distributions and bad prints", "",
         "**ETF distributions (LQD, HYG).** Each Yahoo distribution is tested on its ex-date: the ETF's total "
         "return (close + distribution) is compared with a regression on peers (H.15 CMT yield changes; price "
         "returns of daily-accrual bond funds, whose NAV never drops; total returns of other ETFs) fitted on the "
         "surrounding non-ex days. A paid distribution leaves a residual near 0; one the price never paid leaves "
         "a residual near its own size. A distribution is replaced by the price-implied amount only when the "
         "residual is clear (z > 3) AND its amount is anomalous for the fund (> 1.5x its neighbours), or the "
         "residual alone is overwhelming (z > 8); rule and thresholds in "
         "`assetforecast.data.sources.distributions.validate_distributions`. Changed:", ""]
    rows = []
    for sym, t in diag["div_tables"].items():
        for d, r in dist.changed_distributions(t).iterrows():
            rows.append({"ETF": sym, "ex-date": str(d.date()), "Yahoo amount": r.amount, "used": r.used,
                         "yield": r.yld, "price-test z": r.z, "x neighbours": t.at[d, "ratio"]})
        L.append(f"* {sym}: {len(t)} distributions tested, "
                 f"{ {k: int(v) for k, v in t['action'].value_counts().items()} }.")
    L += ["", _md_table(pd.DataFrame(rows), "{:.4f}") if rows else "(none)", "",
          "LQD 2003-12-16 (1.26% of the price) is not reflected in the price at all (LQD's price rose 0.23% "
          "that day); the independent audit found the same with markets.ft.com closes.", "",
          "**Daily-accrual funds (VWEHX, VFSTX, FBNDX; VWESX from 1985).** They declare income daily and pay it "
          "on the last trading day of the month without an NAV drop, so Yahoo's adjclose books a month of income "
          "on one day. Month-end distributions are spread over their month by calendar days "
          "(`distributions.accrual_total_return`); a month-end distribution more than 2x the fund's regular "
          "income is split (regular part accrued, excess = capital gain paid on the ex-date); distributions on "
          "other days are ex-date cash flows (VWESX dropped its NAV on the ex-date before 1985). Monthly returns "
          "are unchanged up to intra-month compounding.", ""]
    rows = []
    for sym, (a0, a1) in FUND_ERAS.items():
        c = diag["fund_cls"][sym]
        a = diag["rets"][sym].loc[a0:a1]
        b = diag["raw_adj"][sym].loc[a0:a1]
        mo = (np.expm1(np.log1p(a).resample("ME").sum()) - np.expm1(np.log1p(b).resample("ME").sum())).abs()
        mo = mo.iloc[1:]                       # first month may be partial in one of them
        cc = c.loc[a0:a1, "treated_as"].value_counts()
        rows.append({"fund (era used)": f"{sym} {a0[:4]}-{a1[:4]}",
                     "accrued": int(cc.get("daily_accrual", 0) + cc.get("accrual+ex_date_excess", 0)),
                     "of which split": int(cc.get("accrual+ex_date_excess", 0)), "ex-date": int(cc.get("ex_date", 0)),
                     "1-day vol adjclose": b.std() * np.sqrt(252), "1-day vol rebuilt": a.std() * np.sqrt(252),
                     "months diff > 1e-4": int((mo > 1e-4).sum()), "max abs monthly diff": float(mo.max()),
                     "worst month": str(mo.idxmax().date())})
    L += [_md_table(pd.DataFrame(rows), "{:.4f}"), "",
          "Monthly differences above rounding come only from the re-dated January-1987 distributions "
          "(VWEHX, VWESX: moved within the month, so zero) and dropped bad prints (within their month, so zero); "
          "any month listed above differs by intra-month compounding of the accrued income only.", "",
          "**Bad NAV prints dropped** (the PRICE is removed, the next return spans the gap, nothing is "
          "interpolated): " + "; ".join(f"{s} {', '.join(v)}" for s, v in diag["dropped"].items() if v) + ". "
          "Each is a 1-3.6% one-day move that reverses the next day while VWESX and the Treasury reconstructions "
          "(FBNDX) or the HY peer funds FHIIX, PRHYX, AHITX, FAGIX (VWEHX 1991-08-19: peers -0.4% to -0.5%, "
          "VWEHX -1.3% then +1.0%) are flat.", "",
          "**Re-dated distributions:** " + "; ".join(f"{s} {o} -> {n}" for s, m in EXDATE_FIXES.items()
                                                    for o, n in m.items()) + ". In January 1987 the NAVs of "
          "VWEHX, VWESX and VFSTX all dropped on 01-29 by about their distribution while Treasuries and HY peer "
          "funds were flat; Yahoo dates VFSTX's distribution 01-29 but VWEHX's and VWESX's 01-30 (where adjclose "
          "then showed a spurious -1.2%/+1.2% pair). The re-dated distributions are ex-date cash flows.", "",
          "**Not changed:** FBNDX 1983-08-30 (-1.5%, listed by the audit) is a persistent level drop, not a "
          "reversal once the 08-31 distribution is accrued: dropping the price would only move the drop to 08-31. "
          "It may be an unlisted distribution; it is left as recorded.", ""]
    return "\n".join(L)


def averaging_damping(diag: dict, refresh: bool = False) -> dict:
    """Tier-D effect of month-AVERAGE yields: same par bond, average vs end-of-month
    Moody's yields, 1986-2016 (the only period with daily Moody's data)."""
    h = h15_series(["RIMLPAAAR_N.B", "RIMLPBAAR_N.B"], refresh)
    yd = ((h["RIMLPAAAR_N.B"] + h["RIMLPBAAR_N.B"]) / 200).dropna()
    yeom = yd.resample("ME").last()
    n = diag["n_star"]
    a = par_bond_returns(diag["yavg"], n)
    b = par_bond_returns(yeom, n)
    return overlap_stats_monthly(a.loc["1986-02":"2016-09"], b.loc["1986-02":"2016-09"])


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(diag: dict) -> str:
    seg = diag["segments"]
    chk = diag["checks"]
    st = stale_table(diag)
    dv = decade_vols(diag)
    ep = episode_table(diag)
    damp = averaging_damping(diag)
    ig_d, ig_m = diag["out"]["ig_credit"]
    hy_d, hy_m = diag["out"]["hy_credit"]
    pred = diag["pred"]
    w80, w85, wc, n_star = diag["w80"], diag["w85"], diag["wc"], diag["n_star"]

    ck_rows = []
    for k, v in chk.items():
        for s in v:
            ck_rows.append({"comparison (a vs b)": k, "freq": s["freq"], "window": f"{s['start']}..{s['end']}",
                            "n": s["n"], "corr": s["corr"], "vol a": s["vol_a"], "vol b": s["vol_b"],
                            "vol ratio a/b": s["vol_ratio"], "TE (ann.)": s["te"]})
    ck = pd.DataFrame(ck_rows)
    st_fmt = st.copy()
    st_fmt["n"] = st_fmt["n"].astype(str)
    mt = diag["mat_table"].reset_index()
    mt = mt[mt["maturity"].isin([8, 10, 12, 13, 14, 15, 16, 20, 25, 30])]
    pr = (pred.groupby("series")
          .agg(first_obs=("obs_date", "min"), last_obs=("obs_date", "max"), n=("value", "size"),
               median=("value", "median"))
          .reset_index())
    lag = (pred["available_date"] - pred["obs_date"]).dt.days
    pr["median_lag_days"] = lag.groupby(pred["series"]).median().values.astype(float)
    pr["first_obs"] = pr["first_obs"].dt.date
    pr["last_obs"] = pr["last_obs"].dt.date
    pr["n"] = pr["n"].astype(str)
    seg_md = seg.copy()
    for c in ("start", "end"):
        seg_md[c] = seg_md[c].astype(str)

    def g(key, i=-1):
        return chk[key][i]

    vw = g("VWEHX vs HYG", 0), g("VWEHX vs HYG", 1), g("VWEHX vs HYG", 2)
    comp = g("equal-weight VWEHX/FHIIX/PRHYX vs HYG")
    used = ["LQD", "HYG", "VWESX", "FBNDX", "VFSTX", "VWEHX"]
    adj = adjclose_check(used)
    for c in ("first",):
        adj[c] = adj[c].astype(str)
    adj["n_div"] = adj["n_div"].astype(str)
    r = diag["rets"]
    ll_rows = []
    for fund, etf, a, b in [("VWEHX", "HYG", "2007-04-12", "2026-10-31"), ("VWEHX", "HYG", "2010-01-01", "2026-10-31"),
                            ("FHIIX", "HYG", "2010-01-01", "2026-10-31"),
                            ("IG blend VWESX/VFSTX", "LQD", "2002-07-31", "2026-10-31"),
                            ("VWESX", "LQD", "2002-07-31", "2026-10-31")]:
        f = diag["ig85"] if fund.startswith("IG blend") else r[fund]
        ll = lead_lag(f, r[etf], a, b)
        ll_rows.append({"fund_t vs etf_(t-k)": f"{fund} vs {etf}", "window": f"{a[:4]}-{b[:4]}",
                        **{f"k={k}": v for k, v in ll.items()}})
    llt = pd.DataFrame(ll_rows)
    fs = diag["calib_full_sample"]
    calib_tbl = pd.DataFrame([
        {"parameter": "w80: VWESX share of the 1980-84 blend", "used (2002-08..2014-12)": w80,
         "full sample to 2026 (not used)": fs["w80"]},
        {"parameter": "w85: VWESX share of the 1985-2002 blend", "used (2002-08..2014-12)": w85,
         "full sample to 2026 (not used)": fs["w85"]},
        {"parameter": "wc: Ibbotson share of the 1926-79 blend", "used (2002-08..2014-12)": wc,
         "full sample to 2026 (not used)": fs["wc"]},
        {"parameter": "n_star: tier-D par-bond maturity (years)", "used (2002-08..2014-12)": float(n_star),
         "full sample to 2026 (not used)": float(fs["n_star"])}])
    dist_md = distribution_report(diag)

    text = f"""# Credit data: `ig_credit` (LQD-like) and `hy_credit` (HYG-like)

Built by `python -m assetforecast.data.sources.credit` (module `src/assetforecast/data/sources/credit.py`);
raw downloads cached in `data/raw/credit/`. All returns are SIMPLE TOTAL returns
(distributions reinvested), not excess returns. This file is generated by the
module; numbers below are recomputed on every run.

## History achieved

| asset | monthly from | daily from | last daily | last full month |
|---|---|---|---|---|
| ig_credit | {ig_m.index[0].date()} | {ig_d.index[0].date()} | {ig_d.index[-1].date()} | {ig_m.index[-1].date()} |
| hy_credit | {hy_m.index[0].date()} | {hy_d.index[0].date()} | {hy_d.index[-1].date()} | {hy_m.index[-1].date()} |

Monthly files equal compounded daily returns from {MONTHLY_FROM_DAILY['ig_credit']} (ig) and
{MONTHLY_FROM_DAILY['hy_credit']} (hy); before that they hold the monthly segments below. The
first daily month of ig_credit (1980-01) is partial, so 1980-01 in the monthly file comes from the
tier-C segment.

## Sources

* **Yahoo chart API** (`query1.finance.yahoo.com/v8/finance/chart/<sym>`): `close` (split-
  adjusted price or NAV) and dividend events. Returns are built from close + distributions,
  NOT from `adjclose` (see "Distributions and bad prints" below): the ETFs' distributions are
  validated against the ex-date price behaviour, and the daily-accrual funds' month-end
  income is spread over its accrual month. The table below compares Yahoo's adjclose with
  close + Yahoo dividend events (both before any correction); the residual is at the level of
  NAV rounding, so adjclose merely reinvests Yahoo's dividend list (including its errors).
  Calendar-year returns agree with figures recalled from memory (VWEHX 1990 -5.9%,
  1991 +29.4%, 2008 -21.2%; VWESX 2008 +2.3%; LQD 2008 +2.4%, 2022 -17.9%; HYG 2008 -17.6%);
  these were NOT checked against fund documents here.
  * ETFs: LQD (iShares iBoxx $ IG Corp, from 2002-07-30), HYG (iShares iBoxx $ HY Corp,
    from 2007-04-11), JNK (diagnostic only).
  * IG funds: VWESX Vanguard Long-Term Investment-Grade (daily from 1980-01-02),
    VFSTX Vanguard Short-Term Investment-Grade (daily only from 1985-01-02), FBNDX
    Fidelity Investment Grade Bond (daily from 1980-01-02), VFICX Vanguard
    Intermediate-Term IG (1993-, diagnostic only).
  * HY funds: VWEHX Vanguard High-Yield Corporate (used), FHIIX Federated High Income
    Bond, PRHYX T. Rowe Price High Yield, SPHIX Fidelity High Income, AHITX American
    High-Income Trust (diagnostics).
  * **Important data finding**: for VWEHX, FHIIX, VFSTX and FAGIX Yahoo serves only
    ONE NAV PER MONTH before 1985 (on the ex-dividend date, typically the 25th-30th),
    forward-filled so that ~95% of "daily" returns are exactly zero. Those years are
    NOT used as daily data. VWESX and FBNDX are genuinely daily from 1980.

{_md_table(adj, "{:.2e}")}

* **Fed H.15 full data package** (`federalreserve.gov/datadownload/Output.aspx?rel=H15&filetype=zip`):
  Moody's seasoned Aaa and Baa yields, monthly averages 1919-01..2016-09 and daily
  1983/1986..2016-10-07 (the Fed stopped carrying Moody's in Oct 2016); 10-year CMT.
  `fred.stlouisfed.org` times out here, and the DBnomics `FED/H15` dataset no longer
  contains the Moody's series (there is no FRED provider on DBnomics any more).
* **Welch-Goyal predictor file** (Google Sheet linked from `sites.google.com/view/agoyal145`,
  "all data up to 2025"): `corpr` = Ibbotson/SBBI long-term corporate bond total return
  (monthly 1926-01..2025-12), `Rfree` (T-bill), and Moody's `AAA`/`BAA` monthly to 2025-12
  (identical to H.15 on 1919-2016, max abs diff 0).
* **Fed Gilchrist-Zakrajsek** credit spread and excess bond premium
  (`federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv`), monthly 1973-01..2026-10.
* Not obtained: ICE BofA index series (HY OAS, total-return indices). They live on FRED
  (unreachable) and ICE (403); no mirror found on DBnomics.

## Segments

{_md_table(seg_md)}

Construction logic:

* **Holdout integrity.** Every fitted parameter of the pre-LQD history is calibrated on
  LQD's first full month ({ETF_START['ig_credit'][:7]}) to `CALIBRATION_END` = {CALIBRATION_END} only;
  the 2015+ holdout never enters. Full-sample values are shown for comparison only:

{_md_table(calib_tbl, "{:.3f}")}

* **ig_credit daily.** LQD from its first return. Before LQD, a daily constant-mix blend of
  the long-term IG fund VWESX with a shorter fund, because VWESX alone has ~1.3x LQD's
  vol (duration ~13-14y vs LQD ~8-9y; see "VWESX alone vs LQD" below). Weights are chosen
  so the blend's monthly vol equals LQD's on {ETF_START['ig_credit'][:7]}..{CALIBRATION_END[:7]}:
  {w85:.2f} VWESX / {1-w85:.2f} VFSTX (1985-2002, VFSTX keeps the blend all-corporate) and
  {w80:.2f} VWESX / {1-w80:.2f} FBNDX for 1980-1984, when VFSTX has no daily data (FBNDX is a
  core bond fund with Treasuries/MBS, so less credit content; the two blends correlate 0.99
  on 1985-2002). Tier B: open-end fund NAVs (tier A is reserved for exchange-traded prices),
  so the 2002-07 splice to LQD is a visible tier change.
* **ig_credit monthly before 1980.** Tier C: {wc:.2f} x Ibbotson long-term corporate TR +
  {1-wc:.2f} x T-bill. Ibbotson's index is long duration (vol 1.24x LQD on the overlap);
  mixing with T-bills is the monthly analogue of the fund blend (weight again from a vol
  match to LQD on {ETF_START['ig_credit'][:7]}..{CALIBRATION_END[:7]}). Before 1926, tier D: a
  {n_star}-year constant-maturity par bond priced off month-AVERAGE (Aaa+Baa)/2 Moody's
  yields; the maturity is chosen so that the same bond priced off END-of-month daily
  Moody's yields matches LQD's monthly vol on {ETF_START['ig_credit'][:7]}..{CALIBRATION_END[:7]}
  (table below). The Ibbotson series was preferred to the reconstruction from 1926 because
  it is a real index return (corr 0.94 with VWESX on 1980-2025) while the yield
  reconstruction correlates only 0.41 with it on 1926-1979.
* **hy_credit.** HYG from its first return; VWEHX daily 1985-01-02..2007-04-11 (best
  monthly tracking of HYG among long-lived HY funds: corr {vw[2]['corr']:.2f}, vol ratio
  {vw[2]['vol_ratio']:.2f}); VWEHX monthly (one NAV per month on the ex-dividend date) for
  1980-02..1984-12, tier C because it is monthly. Measured on 1985-2026, ex-date sampling
  vs month-end sampling: corr {g('VWEHX ex-div-date-sampled vs month-end (1985-2026)')['corr']:.3f},
  vol ratio {g('VWEHX ex-div-date-sampled vs month-end (1985-2026)')['vol_ratio']:.2f}. No credible HY
  return history before 1980 was found (the modern HY market starts ~1977-78).
  An equal-weight composite of VWEHX/FHIIX/PRHYX was tried and did not track HYG better
  (monthly corr {comp['corr']:.2f}, vol ratio {comp['vol_ratio']:.2f}).

{dist_md}

## Overlap diagnostics

Overlap statistics are DIAGNOSTICS (they use the full overlap, holdout included, but
nothing is fitted on them). Log returns; vol annualized (x sqrt 252 / 52 / 12); TE = annualized std of the
difference. W = Friday-to-Friday weeks.

{_md_table(ck)}

Tier-D maturity calibration (par bond on END-of-month Moody's yields vs LQD, monthly
{ETF_START['ig_credit'][:7]}..{CALIBRATION_END[:7]}; correlation is ~0.8 for every maturity):

{_md_table(mt, "{:.2f}")}

**Averaging damping (tier D).** Same {n_star}y par bond, month-average vs end-of-month
yields, 1986-2016: vol ratio {damp['vol_ratio']:.2f}, corr {damp['corr']:.2f}. So the tier-D
segment's variance is understated by roughly {(1 - damp['vol_ratio']**2) * 100:.0f}% and its
timing is blurred (it also correlates only {g('moodys recon (month-avg yields) vs LQD')['corr']:.2f}
with LQD month by month). Treat 1919-1925 as level information, not as a vol measurement.

## Stale pricing

For each source and era: share of exactly-zero daily returns, autocorrelation of daily log
returns at lags 1-5, annualized vol from daily, weekly and monthly returns, and the
ratios weekly/daily and monthly/daily. With i.i.d. returns both ratios are ~1; smoothed
(stale) prices give positive autocorrelation and ratios > 1; ETF price noise around NAV
gives negative autocorrelation and ratios < 1.

{_md_table(st_fmt)}

What this shows:

* **High-yield fund NAVs are heavily smoothed.** VWEHX lag-1 autocorrelation is
  0.12-0.30 in normal eras and {st.query("source=='VWEHX' and era=='2007-2009'")['ac1'].iloc[0]:.2f} in 2007-09, with
  monthly-from-daily vol ratios of 1.3-1.4 (2.2 in 2007-09). On the HYG overlap,
  VWEHX daily vol is {vw[0]['vol_ratio']:.2f}x HYG's, weekly {vw[1]['vol_ratio']:.2f}x, monthly {vw[2]['vol_ratio']:.2f}x; daily
  correlation is only {vw[0]['corr']:.2f}. Other HY funds are similar or worse (FHIIX lag-1 AC ~0.5 in
  the 1990s-2000s). VWEHX was chosen because it is among the least smoothed and tracks
  HYG best at the monthly horizon. ~40% of VWEHX daily returns are exactly zero: NAVs are
  quoted to the cent on a ~$5-10 price, so one tick is 10-20 bp.
* **Consequence for `cov_rv`.** In the hy_credit fund era (1985-2007-04) daily-return
  realized vol understates true vol by roughly a factor 1.3-1.4 (variance by ~1.7-2x)
  and daily correlations with liquid assets are attenuated. Modeling agents should
  either use multi-day/weekly aggregation or a variance-ratio correction for these
  years; the data are left unadjusted.
* **IG funds are much less affected**: the IG blends have lag-1 AC 0.08-0.15 in the
  1980s-90s and -0.08..0.10 afterwards (LQD itself has 0.12 in 2020-26); the
  monthly/daily vol ratio is ~1.3 for the 1980s FBNDX blend (partly 1980-82 rate-vol
  clustering, not only staleness) and 0.8-1.1 later.
* **Lead-lag** (daily correlation of the fund today with the ETF k days earlier; a
  stale fund loads on lagged ETF returns):

{_md_table(llt)}

  VWEHX picks up a large share of HYG's move one and two days later, the signature
  of matrix pricing; the IG blend's lagged loadings are much smaller. The modest
  same-day IG correlation (~0.67) with almost no lagged loading points to
  non-synchronous measurement (evaluated bond prices are struck around the 3pm bond
  close, the ETF closes at 4pm) and ETF premium noise rather than to smoothing.
* **ETFs**: in 2007-09 LQD has NEGATIVE lag-1/2 autocorrelation (ac1
  {st.query("source=='LQD' and era=='2007-2009'")['ac1'].iloc[0]:.2f}, ac2 {st.query("source=='LQD' and era=='2007-2009'")['ac2'].iloc[0]:.2f}, weekly/daily vol {st.query("source=='LQD' and era=='2007-2009'")['w_over_d'].iloc[0]:.2f}); HYG is mildly negative at lags 2-3. Big one-day reversals
  (LQD 2008-09-29 -9.1% then +9.8%; HYG 2008-10-10 -6.7% then +12.3%) when the ETF price
  overshot NAV. In 2020-26, weekly vol exceeds daily vol for both (ratio ~1.16), driven by
  March 2020, when the ETFs traded at multi-day discounts to NAV.
  The ETF-era daily vol of hy_credit therefore includes some liquidity/premium noise.
* Era jump at the splice: hy_credit daily vol jumps when HYG replaces VWEHX
  in 2007-04 partly for this measurement reason, not only because of the crisis.

## Annualized vol by decade (final series)

{_md_table(dv)}

## Episodes (cumulative simple return over the window, %)

{_md_table((ep.set_index('episode') * 100).reset_index(), "{:.1f}")}

These look right: IG gains in Oct 1987 (flight to quality) while HY loses; HY -5.9% in
1990 and +29% in 1991; HY down and IG flat in Aug-Oct 1998; HY -7% in May-Jul 2002;
Lehman Sep-Nov 2008 IG -9.5%, HY -26.5%; Mar 2020 IG -6.3%, HY -10.0%; 2022 IG -17.9%
(duration), HY -11.0%. The Ibbotson long-term corporate index (high grade, long) GAINED
in 2008 (+8.8%) — a reminder that the pre-1980 tier-C series measures high-grade long-
bond risk and will understate spread-driven drawdowns relative to an LQD-like
(A/BBB) portfolio.

## Predictors (`data/processed/predictors/credit_spreads.parquet`)

Long format `obs_date, available_date, series, value`. Values in PERCENT (percentage
points), e.g. 0.94 = 94 bp. Monthly series use obs_date = month-end.

{_md_table(pr, "{:.2f}")}

* `moodys_*_m`: Moody's seasoned Aaa/Baa monthly averages (H.15 to 2016-09, Welch-Goyal
  2016-10..2025-12; the two agree exactly where they overlap). available_date = next
  business day after month-end (daily yields are published with a one-day lag, so the
  average is known then). They end 2025-12: no reachable source for 2026 Moody's data.
* `moodys_baa_minus_ust10y_m`: Baa minus the H.15 10-year CMT monthly average (from 1953-04).
* `*_d`: daily Baa-Aaa and Baa-10y from H.15, 1986-01-02..2016-10-07, available next
  business day.
* `gz_credit_spread_m`, `gz_excess_bond_premium_m`: Gilchrist-Zakrajsek (2012) from the
  Fed, monthly 1973-01..2026-10. available_date = end of the FOLLOWING month (the file
  is published with a lag and revised). The EBP is the residual of a regression
  estimated on the full sample, so it carries mild look-ahead in its loadings; the GZ
  spread is a plain average of bond spreads. The GZ spread is the only series here that
  runs to 2026.
* Not available: ICE BofA HY/IG OAS (see Sources).

## Caveats

* The fund-blend weights ({w85:.2f}/{w80:.2f}), the Ibbotson/T-bill weight ({wc:.2f}) and the
  tier-D maturity ({n_star}y) are calibrated on LQD data up to {CALIBRATION_END} only and
  applied backwards; durations of all these portfolios drifted over time (LQD's own
  duration rose from ~7 to ~8.5+).
* Before 2002 ig_credit is a blend of two open-end funds rebalanced daily (investable, but
  priced at NAV, hence tier B); its daily covariance with equities is affected by NAV
  staleness; IG fund staleness is mild (see table).
* hy_credit 1985-2007 (VWEHX, tier B): at the 21-day target horizon its vol is
  {diag['vwehx_hyg_horizon']['r21']:.2f}x HYG's on the overlap ({diag['vwehx_hyg_horizon']['r21_0709']:.2f}x in
  2007-09; {diag['vwehx_hyg_horizon']['r63']:.2f}x at 63 days). Its vol LEVEL should not be pooled with HYG's.
* VWEHX is a relatively high-quality HY fund (BB-heavy); its 1990 and 2008 drawdowns are
  milder than lower-quality funds (FHIIX 1990 -12.8%). HYG is also tilted to larger,
  liquid issuers.
* Pre-1985 hy_credit monthly returns span ex-dividend date to ex-dividend date (a few
  days off month-end); the measured effect of this on 1985-2026 is negligible
  (corr 0.996, vol ratio 1.00), but correlations with month-end-sampled assets in
  1980-84 are slightly attenuated.
* Tier D (1919-1925 ig_credit) is month-average-yield based: vol understated (~15%)
  and timing blurred.
* Moody's yields end 2025-12; GZ ends 2026-10; returns run to {ig_d.index[-1].date()}.
"""
    path = REPORTS / "data" / "credit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Build ig_credit / hy_credit returns and credit-spread predictors")
    ap.add_argument("--refresh", action="store_true", help="re-download raw files")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args(argv)
    diag = build(refresh=a.refresh)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(diag["segments"][["asset", "start", "end", "freq", "source", "tier"]].to_string(index=False))
    if not a.no_report:
        write_report(diag)


if __name__ == "__main__":
    main()
