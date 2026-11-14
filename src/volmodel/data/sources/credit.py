"""Credit total returns: ``ig_credit`` (LQD-like) and ``hy_credit`` (HYG-like).

placeholder - construction filled in below
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

from ...paths import ASSET_DIR, PREDICTOR_DIR, RAW, REPORTS

CACHE = RAW / "credit"
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
    div = pd.Series(0.0, index=df.index)
    for v in j.get("events", {}).get("dividends", {}).values():
        d = (pd.Timestamp(v["date"], unit="s", tz="UTC").tz_convert(tz)
             .tz_localize(None).normalize())
        if d in div.index:
            div[d] += v["amount"]
    df["div"] = div
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
OVERLAP_END_M = None  # set in build: last complete month


def yahoo_returns(sym: str, refresh: bool = False) -> pd.Series:
    """Daily simple total returns from Yahoo adjclose (distributions reinvested)."""
    df = yahoo_frame(sym, refresh)
    r = df["adjclose"].pct_change().dropna()
    if sym in DAILY_START:
        r = r.loc[DAILY_START[sym]:]
    r.name = sym
    return r


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


def to_log(r):
    return np.log1p(r)


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
                  axis=1, keys=["a", "b"]).dropna()
    la, lb = np.log1p(x["a"]), np.log1p(x["b"])
    return {"freq": freq, "n": len(x), "start": x.index[0].date(), "end": x.index[-1].date(),
            "corr": la.corr(lb), "vol_a": la.std() * np.sqrt(ann), "vol_b": lb.std() * np.sqrt(ann),
            "vol_ratio": la.std() / lb.std(), "te": (la - lb).std() * np.sqrt(ann)}


def fmt_stats(s: dict) -> str:
    return (f"{s['freq']} {s['start']}..{s['end']} n={s['n']}: corr={s['corr']:.3f} "
            f"vol {s['vol_a']:.3f}/{s['vol_b']:.3f} ratio={s['vol_ratio']:.2f} TE={s['te']:.3f}")


def vol_match_weight(a: pd.Series, b: pd.Series, target: pd.Series) -> float:
    """w in [0,1] such that the monthly vol of w*a + (1-w)*b equals that of target."""
    x = pd.concat([a, b, target], axis=1, keys=["a", "b", "t"]).dropna()
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


def calibrate_recon_maturity(lqd_m: pd.Series, refresh: bool = False) -> tuple[int, pd.DataFrame]:
    """Pick the par-bond maturity whose returns from END-of-month Moody's yields
    (daily H.15, 1986-2016) match LQD's monthly vol on 2002-08..2016-09. The tier-D
    series then uses that maturity with month-AVERAGE yields (the only ones before
    1986), so its residual vol damping is purely the averaging effect."""
    h = h15_series(["RIMLPAAAR_N.B", "RIMLPBAAR_N.B"], refresh)
    yd = ((h["RIMLPAAAR_N.B"] + h["RIMLPBAAR_N.B"]) / 200).dropna()
    yeom = yd.resample("ME").last()
    rows = []
    for n in TIER_D_MATURITIES:
        s = overlap_stats(par_bond_returns(yeom, n), lqd_m, "M", "2002-08", "2016-09")
        rows.append({"maturity": n, "vol_ratio_vs_LQD": s["vol_ratio"], "corr": s["corr"]})
    t = pd.DataFrame(rows).set_index("maturity")
    return int((t["vol_ratio_vs_LQD"] - 1).abs().idxmin()), t


def build(refresh: bool = False, write: bool = True) -> dict:
    """Build both assets and the predictor file; return diagnostics for the report."""
    rets = {s: yahoo_returns(s, refresh) for s in IG_SYMBOLS + HY_SYMBOLS}
    g = goyal_monthly(refresh)
    h = h15_series(["RIMLPAAAR_N.M", "RIMLPBAAR_N.M"], refresh)
    diag: dict = {"rets": rets}
    M = {s: resample_ret(r, "ME") for s, r in rets.items()}
    lqd, hyg = rets["LQD"], rets["HYG"]
    last_m = last_complete_month(lqd.index)
    ov = (ETF_START["ig_credit"][:7], last_m)  # LQD overlap window (months)

    # ---------------- ig_credit, daily: two fund blends, then LQD
    w80 = vol_match_weight(M["VWESX"].loc[ov[0]:ov[1]], M["FBNDX"].loc[ov[0]:ov[1]], M["LQD"].loc[ov[0]:ov[1]])
    w85 = vol_match_weight(M["VWESX"].loc[ov[0]:ov[1]], M["VFSTX"].loc[ov[0]:ov[1]], M["LQD"].loc[ov[0]:ov[1]])
    ig80 = (w80 * rets["VWESX"] + (1 - w80) * rets["FBNDX"]).dropna()
    ig85 = (w85 * rets["VWESX"] + (1 - w85) * rets["VFSTX"]).dropna()
    ig_d = _splice([(ig80, SRC["ig80"], "A"), (ig85, SRC["ig85"], "A"),
                    (lqd.loc[ETF_START["ig_credit"]:], SRC["LQD"], "A")])
    ig_m = to_monthly(ig_d, MONTHLY_FROM_DAILY["ig_credit"], last_m)

    # ---------------- ig_credit, monthly: Ibbotson LT corp + T-bill blend (C), Moody's recon (D)
    gg = g.loc[: "2025-12"]
    wc = vol_match_weight(gg["corpr"].loc[ov[0]:], gg["Rfree"].loc[ov[0]:], M["LQD"].loc[ov[0]:"2025-12"])
    corp = (wc * g["corpr"] + (1 - wc) * g["Rfree"]).dropna()
    n_star, mat_table = calibrate_recon_maturity(M["LQD"], refresh)
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
    hy_d = _splice([(rets["VWEHX"], SRC["VWEHX"], "A"),
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
    r, M = diag["rets"], None
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
    # effect of ex-dividend-date sampling (pre-1985 HY monthly), measured on 1985-2026
    chk["VWEHX ex-div-date-sampled vs month-end (1985-2026)"] = [exdiv_sampling_check()]
    return chk


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
    x = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
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
    notes = {
        SRC["ig80"]: (f"daily constant-mix {w80:.2f} VWESX + {1-w80:.2f} FBNDX (weight = vol match to LQD, monthly); "
                      f"vs LQD {short('blend_VWESX_FBNDX vs LQD')}; vs VWESX/VFSTX blend {short('blend_VWESX_FBNDX vs blend_VWESX_VFSTX (1985-2002)')}; "
                      "open-end NAVs: stale pricing, daily vol understated"),
        SRC["ig85"]: (f"daily constant-mix {w85:.2f} VWESX + {1-w85:.2f} VFSTX (vol match to LQD, monthly); "
                      f"vs LQD {short('blend_VWESX_VFSTX vs LQD')}; daily {short('blend_VWESX_VFSTX vs LQD', 0)}; "
                      "open-end NAVs: stale pricing, daily vol understated"),
        SRC["LQD"]: "iShares iBoxx IG corporate ETF, adjclose (verified = close + distributions)",
        SRC["corp"]: (f"{wc:.2f} x Ibbotson/SBBI long-term corporate TR + {1-wc:.2f} x T-bill (Welch-Goyal corpr, Rfree), "
                      f"duration-shortened to LQD; vs LQD {short('ibbotson_ltcorp_rf_blend vs LQD')}; "
                      f"vs fund blends {short('ibbotson_ltcorp_rf_blend vs fund blends (1980-2002)')}"),
        SRC["recon"]: (f"{diag['n_star']}y constant-maturity par bond priced off month-AVERAGE (Aaa+Baa)/2 Moody's yields (H.15); "
                       f"vs Ibbotson blend {short('moodys recon (month-avg yields) vs ibbotson blend (1926-1979)')}; "
                       f"vs LQD {short('moodys recon (month-avg yields) vs LQD')}; averaging damps vol and blurs timing"),
        SRC["VWEHX"]: (f"Vanguard High-Yield Corporate NAV; vs HYG {short('VWEHX vs HYG')}; daily {short('VWEHX vs HYG', 0)}; "
                       "matrix-priced NAV: daily vol ~half of HYG, positive autocorrelation"),
        SRC["VWEHX_m"]: ("Yahoo has one NAV per month before 1985 (ex-div date, ~25th-30th): returns span ex-date to ex-date; "
                         f"sampling effect measured 1985-2026: {short('VWEHX ex-div-date-sampled vs month-end (1985-2026)')}"),
        SRC["HYG"]: "iShares iBoxx HY corporate ETF, adjclose (verified = close + distributions)",
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
