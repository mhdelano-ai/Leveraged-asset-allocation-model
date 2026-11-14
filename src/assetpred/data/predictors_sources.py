"""Source readers for the raw predictor panel.

Every function returns a monthly ``pd.Series`` indexed by calendar month end.
Unless the docstring says otherwise, the index is the month the value *refers
to* (or, for market data, the month whose close it was observed at); the
publication lag is applied later by ``predictors.build`` from the spec's
``lag_months``. Functions whose name ends in ``_asof`` already return what was
knowable at each month end and must not be shifted again.

All downloads go through ``fetch``/``fred`` so the manifest records provenance.
"""
from __future__ import annotations

import io
import warnings
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd

from .fetch import fetch, fred

ME = pd.offsets.MonthEnd(0)


def last_complete_month(today: pd.Timestamp | None = None) -> pd.Timestamp:
    """Last month end whose close has already happened."""
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    return today - pd.offsets.MonthEnd(1)  # on a month-end day the close is still ahead


def to_month_end(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index) + ME
    return s[~s.index.duplicated(keep="last")].sort_index()


def month_end_last(s: pd.Series) -> pd.Series:
    """Last available observation in each calendar month (daily -> month end)."""
    s = s.dropna()
    return s.groupby(pd.DatetimeIndex(s.index) + ME).last()


def month_mean(s: pd.Series) -> pd.Series:
    s = s.dropna()
    return s.groupby(pd.DatetimeIndex(s.index) + ME).mean()


# --------------------------------------------------------------------- FRED
@lru_cache(maxsize=None)
def fred_raw(series_id: str) -> pd.Series:
    df = pd.read_csv(fred(series_id))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                  index=pd.to_datetime(df.iloc[:, 0]), name=series_id)
    return s.dropna()


def fred_monthly(series_id: str) -> pd.Series:
    """Monthly FRED series (dated first of month) re-dated to month end."""
    return to_month_end(fred_raw(series_id))


def fred_daily_me(series_id: str) -> pd.Series:
    return month_end_last(fred_raw(series_id))


def weekly_asof(s: pd.Series, release_days: int, window: int = 1) -> pd.Series:
    """Value knowable at each month end for a weekly series.

    ``s`` is indexed by the week-ending date; an observation is treated as
    published ``release_days`` after it. ``window`` > 1 averages the last
    ``window`` published weeks (e.g. 4-week average of jobless claims).
    """
    s = s.dropna().sort_index()
    if window > 1:
        s = s.rolling(window, min_periods=window).mean().dropna()
    avail = s.copy()
    avail.index = s.index + pd.Timedelta(days=release_days)
    months = pd.date_range(avail.index[0] + ME, last_complete_month(), freq="ME")
    out = avail.reindex(avail.index.union(months)).ffill().reindex(months)
    return out


def yoy(s: pd.Series, log: bool = True) -> pd.Series:
    """12-month change of a monthly month-end series (log by default)."""
    prev = s.copy()
    prev.index = prev.index + pd.offsets.MonthEnd(12)
    prev = prev.reindex(s.index)
    return np.log(s / prev) if log else s / prev - 1


# ------------------------------------------------------------------ ALFRED
def _alfred_chunk(series_id: str, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """One ALFRED request: the full vintage in force at each of up to 12 dates."""
    ids = ",".join([series_id] * len(dates))
    vd = ",".join(d.strftime("%Y-%m-%d") for d in dates)
    tag = f"{dates[0]:%Y%m}-{dates[-1]:%Y%m}"
    path = fetch(f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={ids}&vintage_date={vd}",
                 f"alfred/{series_id}/{tag}.csv")
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df.pop(df.columns[0]))
    return df.apply(pd.to_numeric, errors="coerce")


def alfred_asof(series_id: str, start: str, end: str | None = None,
                transforms: dict | None = None) -> dict[str, pd.Series]:
    """Real-time values from ALFRED vintages.

    ``start`` must be on or after the series' first vintage (ALFRED returns 404
    for an as-of date before it).

    For each month end t in [start, end] the vintage in force at t (all
    releases up to and including day t) is downloaded. From it we keep the
    latest observation (``level``) and any ``transforms`` computed *within that
    vintage*, so growth rates never mix vintages or base-year rebasings.
    ``transforms`` maps name -> (kind, months), kind in {"logdiff"}.
    Returns name -> Series indexed by t (already as-of; do not lag again),
    plus ``refdate``: the month end of the latest observation in each vintage.
    """
    transforms = transforms or {}
    end = pd.Timestamp(end) + ME if end else last_complete_month()
    months = list(pd.date_range(pd.Timestamp(start) + ME, end, freq="ME"))
    out = {"level": {}, **{k: {} for k in transforms}}
    refdate = {}
    for i in range(0, len(months), 12):
        chunk = months[i:i + 12]
        df = _alfred_chunk(series_id, chunk)
        cols = list(df.columns)
        if len(cols) != len(chunk):
            raise RuntimeError(f"ALFRED {series_id}: expected {len(chunk)} vintages, got {len(cols)}")
        for t, c in zip(chunk, cols):
            v = df[c].dropna()
            if v.empty:
                continue
            last = v.index[-1]
            out["level"][t] = v.iloc[-1]
            refdate[t] = last + ME
            for name, (kind, k) in transforms.items():
                ref = last - pd.DateOffset(months=k)
                if kind == "logdiff" and ref in v.index and v[ref] > 0:
                    out[name][t] = np.log(v.iloc[-1] / v[ref])
    res = {k: pd.Series(d, dtype=float).sort_index() for k, d in out.items()}
    res["refdate"] = pd.Series(refdate).sort_index()
    return res


def release_timed(values: pd.Series, refdate: pd.Series, min_lag: int) -> pd.Series:
    """Place revised monthly values at the month end they were first published.

    ``values`` is indexed by reference month end. ``refdate`` (from
    ``alfred_asof``) gives, for each as-of month end t, the latest reference
    month published by t; month m is first known at the first t with
    refdate[t] >= m. Months already published when the vintages begin, or not
    yet in them, fall back to a fixed ``min_lag``; a value is never placed
    earlier than m + min_lag. Returns a Series indexed by availability month.
    """
    first_known: dict[pd.Timestamp, pd.Timestamp] = {}
    covered = refdate.iloc[0]
    for t, ref in refdate.items():
        while covered < ref:
            covered = covered + pd.offsets.MonthEnd(1)
            first_known[covered] = t
    out = {}
    for m, v in values.dropna().items():
        avail = m + pd.offsets.MonthEnd(min_lag)
        if m in first_known:
            avail = max(avail, first_known[m])
        elif m > refdate.iloc[-1]:
            avail = max(avail, refdate.index[-1] + pd.offsets.MonthEnd(1))
        out[avail] = v  # if two months land together, the later reference month wins
    return pd.Series(out, dtype=float).sort_index()


# ----------------------------------------------------------------- Shiller
@lru_cache(maxsize=None)
def shiller() -> pd.DataFrame:
    """Shiller ie_data: monthly P (average of daily closes), D, E (index points)."""
    url = ("https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/"
           "1449043e-a1ed-4e2e-8e9c-7b9a2a064493/ie_data.xls")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = pd.read_excel(fetch(url, "shiller/ie_data.xls"), sheet_name="Data", header=None, skiprows=8)
    raw = raw[pd.to_numeric(raw[0], errors="coerce").notna()]
    ym = raw[0].astype(float).map(lambda x: f"{x:.2f}")
    idx = pd.to_datetime(ym.str.replace(".", "-", regex=False) + "-01") + ME
    df = pd.DataFrame({"P": raw[1].values, "D": raw[2].values, "E": raw[3].values}, index=idx)
    return df.apply(pd.to_numeric, errors="coerce")


# ------------------------------------------------------------- Goyal-Welch
@lru_cache(maxsize=None)
def goyal_welch() -> pd.DataFrame:
    """Goyal-Welch(-Zafirov) monthly predictor file (updated to 2025)."""
    url = "https://docs.google.com/spreadsheets/d/1qwpl2R_DNujpU5YUkk8lacP1tTeMb9iJ/export?format=xlsx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(fetch(url, "goyal/PredictorData2025.xlsx"), sheet_name="Monthly")
    df.index = pd.to_datetime(df.pop("yyyymm").astype(int).astype(str), format="%Y%m") + ME
    return df.apply(pd.to_numeric, errors="coerce")


def sp500_month_end() -> pd.Series:
    """S&P 500 price index at month-end close: Goyal-Welch ``Index`` (month-end
    close from 1926) until FRED's SP500 daily closes begin (Nov 2016), FRED after.
    The two agree to the cent over the overlap except one bad GW value (2017-11)."""
    gw = goyal_welch()["Index"].dropna()
    gw = gw[gw.index >= "1926-01-31"]
    fr = fred_daily_me("SP500")
    fr = fr[fr.index > fr.index[0]]  # first FRED month is partial
    return pd.concat([gw[gw.index < fr.index[0]], fr]).sort_index()


# -------------------------------------------------------------- Fed curves
@lru_cache(maxsize=None)
def fed_curve(kind: str) -> pd.DataFrame:
    """Daily Fed staff yield curves: 'nominal' (GSW, feds200628) or 'tips' (GSW, feds200805)."""
    file = {"nominal": "feds200628.csv", "tips": "feds200805.csv"}[kind]
    path = fetch(f"https://www.federalreserve.gov/data/yield-curve-tables/{file}", f"fed/{file}")
    with open(path) as f:
        for n, line in enumerate(f):
            if line.startswith("Date,"):
                break
    df = pd.read_csv(path, skiprows=n, na_values=["NA"], low_memory=False)
    df.index = pd.to_datetime(df.pop("Date"))
    return df.apply(pd.to_numeric, errors="coerce")


@lru_cache(maxsize=None)
def acm() -> pd.DataFrame:
    """NY Fed ACM term structure model, month-end sheet."""
    url = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
    df = pd.read_excel(fetch(url, "nyfed/ACMTermPremium.xls"), sheet_name="ACM Monthly")
    df.index = pd.to_datetime(df.pop("DATE"), format="%d-%b-%Y") + ME
    return df.apply(pd.to_numeric, errors="coerce")


@lru_cache(maxsize=None)
def ebp() -> pd.DataFrame:
    """Gilchrist-Zakrajsek spread and excess bond premium (Fed, monthly)."""
    path = fetch("https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv", "fed/ebp_csv.csv")
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df.pop("date")) + ME
    return df.apply(pd.to_numeric, errors="coerce")


# ------------------------------------------------------- uncertainty / risk
def epu_news() -> pd.Series:
    url = "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx"
    df = pd.read_excel(fetch(url, "epu/US_Policy_Uncertainty_Data.xlsx"), sheet_name="Main News Index")
    df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]
    idx = pd.to_datetime(dict(year=df["Year"].astype(int), month=df["Month"].astype(int), day=1)) + ME
    return pd.Series(df["News_Based_Policy_Uncert_Index"].astype(float).values, index=idx).sort_index()


@lru_cache(maxsize=None)
def gpr() -> pd.DataFrame:
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    df = pd.read_excel(fetch(url, "gpr/data_gpr_export.xls"))
    df.index = pd.to_datetime(df.pop("month")) + ME
    return df.apply(pd.to_numeric, errors="coerce")


@lru_cache(maxsize=None)
def baker_wurgler() -> pd.DataFrame:
    url = "https://pages.stern.nyu.edu/~jwurgler/data/Investor_Sentiment_Data_20190327_POST.xlsx"
    df = pd.read_excel(fetch(url, "wurgler/Investor_Sentiment_Data_20190327_POST.xlsx"), sheet_name="DATA")
    df = df[pd.to_numeric(df["yearmo"], errors="coerce").notna()]
    df.index = pd.to_datetime(df.pop("yearmo").astype(int).astype(str), format="%Y%m") + ME
    return df.apply(pd.to_numeric, errors="coerce")


def finra_margin() -> pd.DataFrame:
    url = "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx"
    df = pd.read_excel(fetch(url, "finra/margin-statistics.xlsx"))
    df.index = pd.to_datetime(df.pop("Year-Month").astype(str), format="%Y-%m") + ME
    return df.apply(pd.to_numeric, errors="coerce").sort_index()


# ------------------------------------------------------------ CBOE
def cboe_vix() -> pd.Series:
    path = fetch("https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv", "cboe/VIX_History.csv")
    df = pd.read_csv(path)
    s = pd.Series(df["CLOSE"].astype(float).values, index=pd.to_datetime(df["DATE"], format="%m/%d/%Y"))
    return month_end_last(s)


def _cboe_pc(name: str) -> pd.Series:
    path = fetch(f"https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/{name}.csv", f"cboe/{name}.csv")
    lines = open(path, encoding="latin-1").read().splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.upper().startswith(("DATE,", "TRADE_DATE,")))
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), skipinitialspace=True)
    s = pd.Series(pd.to_numeric(df.iloc[:, -1], errors="coerce").values,
                  index=pd.to_datetime(df.iloc[:, 0], format="mixed", errors="coerce"))
    return s[s.index.notna()].dropna()


def cboe_equity_pc_daily() -> pd.Series:
    old = _cboe_pc("equitypcarchive")
    new = _cboe_pc("equitypc")
    return pd.concat([old[old.index < new.index[0]], new]).sort_index()


# ------------------------------------------------------------ commodities
def gold_monthly() -> pd.Series:
    path = fetch("https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv", "gold/monthly.csv")
    df = pd.read_csv(path)
    return pd.Series(df["Price"].astype(float).values, index=pd.to_datetime(df["Date"]) + ME).sort_index()


def eia_wti_future(n: int) -> pd.Series:
    path = fetch(f"https://www.eia.gov/dnav/pet/hist_xls/RCLC{n}d.xls", f"eia/RCLC{n}d.xls")
    df = pd.read_excel(path, sheet_name="Data 1", header=2)
    return pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(df.iloc[:, 0])).dropna()


# ----------------------------------------------------- Ken French intl
def _french_zip_table(zipname: str, member: str, title: str) -> pd.DataFrame:
    path = fetch(f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{zipname}.zip", f"french/{zipname}.zip")
    with zipfile.ZipFile(path) as z:
        text = z.read(next(n for n in z.namelist() if n.endswith(member))).decode("latin-1")
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if title in l)
    rows = []
    for l in lines[start + 1:]:
        parts = l.split()
        if parts and parts[0].isdigit():
            rows.append(parts)
        elif rows:
            break
    df = pd.DataFrame(rows).set_index(0).astype(float).replace(-99.99, np.nan)
    return df


def french_intl_market(with_div: bool) -> pd.Series:
    """Monthly USD value-weighted return (%) of French's international index
    (EAFE + Canada countries, ``Ind_all``), all firms with B/M data."""
    zipname = "F-F_International_Indices" if with_div else "F-F_International_Indices_Wout_Div"
    df = _french_zip_table(zipname, "Ind_all.Dat", "Value-Weight Dollar Returns      All 4 Data Items Not Reqd")
    df = df[df.index.str.len() == 6]
    s = df[1] / 100
    s.index = pd.to_datetime(s.index, format="%Y%m") + ME
    return s


def french_intl_bm() -> pd.Series:
    """Annual market B/M of the international index; row y = ratios at the end of
    December y-1 (book equity for fiscal years ending in y-1)."""
    df = _french_zip_table("F-F_International_Indices", "Ind_all.Dat", "Value-Weight Ratios      All 4 Data Items Not Required")
    return pd.Series(df[1].values, index=df.index.astype(int))
