"""Cash (T-bill) returns and point-in-time macro predictors.

Run ``python -m assetforecast.data.sources.cash_macro [--refresh]``.

Outputs
-------
data/processed/assets/cash_daily.parquet    index ``date`` (every US trading day
                                            1926-07-01 .. latest), ``ret``, ``source``, ``tier``
data/processed/assets/cash_monthly.parquet  index month-end ``date`` 1871-01 .. latest
data/processed/predictors/macro.parquet     long: obs_date, available_date, series, value
reports/data/cash_macro.md                  provenance, diagnostics, predictor table

Cash construction (see the report for the numbers)
--------------------------------------------------
* 1926-07 .. last French month: Ken French ``RF`` (1-month T-bill: Ibbotson to
  2024-05, ICE BofA 1-month T-bill index after). Monthly ``ret`` = RF. The
  French *daily* RF column is rounded to 0.01 % per day, which is too coarse
  (it does not compound back to the monthly value), so the daily series spreads
  each month's RF geometrically over that month's trading days (dates from the
  French daily file):  (1+RF_m)^(1/n_m) - 1.  Daily compounds exactly to monthly.
* After the last French month: 4-week T-bill secondary-market rate (H.15,
  discount basis, via DBnomics). The yield observed at the previous trading
  day accrues over the calendar days to the current trading day (money-market
  convention, actual/360). Trading dates come from SPY on Yahoo. Yields missing
  on a trading day (bond-market holidays, or the last few days before H.15 is
  updated) are carried forward at most 7 calendar days.
* Before 1926-07 (monthly only, tier D): 1920-01 .. 1926-06 yield on 3-6 month
  Treasury notes and certificates (NBER macrohistory m13029a); 1871-01 ..
  1919-12 New York prime commercial-paper rate (NBER m13002) minus its mean
  spread over the Treasury yield in 1920-01 .. 1926-06. Monthly ret = yield/1200.
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

from ...paths import ASSET_DIR, PREDICTOR_DIR, RAW, REPORTS

RAW_DIR = RAW / "cash_macro"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
DBNOMICS = "https://api.db.nomics.world/v22/series/"
NBER = "https://data.nber.org/databases/macrohistory/rectdata/"
BLS_V1 = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/"
SHILLER = ["http://www.econ.yale.edu/~shiller/data/ie_data.xls",
           "https://www.econ.yale.edu/~shiller/data/ie_data.xls"]

MAX_YIELD_FFILL_DAYS = 7

# --------------------------------------------------------------------------- http


def _get(url: str, *, method: str = "GET", json_body=None, timeout: int = 90, tries: int = 3,
         ua: str = UA) -> bytes:
    last = None
    for i in range(tries):
        try:
            r = requests.request(method, url, json=json_body, timeout=timeout, headers={"User-Agent": ua})
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:  # pragma: no cover - network
            last = e
            time.sleep(min(60, 5 * 2 ** i))
    raise RuntimeError(f"download failed: {url}: {last}")


def _cached(name: str, fetch, refresh: bool) -> bytes:
    path = RAW_DIR / name
    if path.exists() and not refresh:
        return path.read_bytes()
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


# ------------------------------------------------------------------------ sources


def french_rf(refresh: bool = False) -> tuple[pd.Series, pd.Series]:
    """(monthly RF indexed by month-end, daily RF indexed by trading date), decimals."""
    out = []
    for fname in ["F-F_Research_Data_Factors_CSV.zip", "F-F_Research_Data_Factors_daily_CSV.zip"]:
        raw = _cached(fname, lambda f=fname: _get(FRENCH + f), refresh)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        rows = {}
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5 or not parts[0].isdigit():
                if rows and "Annual" in line:  # monthly file: annual block follows
                    break
                continue
            rows[parts[0]] = float(parts[4]) / 100.0
        s = pd.Series(rows)
        if len(next(iter(rows))) == 6:
            s.index = pd.to_datetime(s.index, format="%Y%m") + pd.offsets.MonthEnd(0)
        else:
            s.index = pd.to_datetime(s.index, format="%Y%m%d")
        out.append(s.sort_index().rename("rf"))
    return out[0], out[1]


def dbnomics(code: str, refresh: bool = False) -> pd.Series:
    """A DBnomics series 'PROVIDER/DATASET/SERIES' -> float Series (NaN for 'NA')."""
    name = "dbnomics_" + code.replace("/", "__") + ".json"
    raw = _cached(name, lambda: _get(DBNOMICS + code + "?observations=1&format=json"), refresh)
    doc = json.loads(raw)["series"]["docs"][0]
    s = pd.Series(pd.to_numeric(pd.Series(doc["value"]), errors="coerce").values,
                  index=pd.Index(doc["period"]), name=code)
    p = s.index[0]
    if len(p) == 7:  # YYYY-MM
        s.index = pd.to_datetime(s.index, format="%Y-%m") + pd.offsets.MonthEnd(0)
    else:
        s.index = pd.to_datetime(s.index)
        if code.endswith(".M") or code.endswith("_N.M"):
            s.index = s.index + pd.offsets.MonthEnd(0)
    return s.sort_index()


def nber(code: str, refresh: bool = False) -> pd.Series:
    """NBER macrohistory rectangular file (year, month, value); '.' = missing."""
    chapter = code[1:3]
    raw = _cached(f"nber_{code}.dat", lambda: _get(f"{NBER}{chapter}/{code}.dat"), refresh).decode()
    vals = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        y, m, v = int(parts[0]), int(parts[1]), parts[2]
        vals[pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)] = np.nan if v == "." else float(v)
    return pd.Series(vals, name=code).sort_index()


def bls_recent(series: list[str], refresh: bool = False) -> dict[str, pd.Series]:
    """Last ten years of BLS series from the public v1 API (no key; 25 requests/day/IP)."""
    end = pd.Timestamp.today().year
    body = {"seriesid": series, "startyear": str(end - 9), "endyear": str(end)}

    def fetch():
        raw = _get(BLS_V1, method="POST", json_body=body)
        if json.loads(raw).get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS API: {raw[:300]!r}")
        return raw

    raw = _cached("bls_v1_recent.json", fetch, refresh)
    out = {}
    for s in json.loads(raw)["Results"]["series"]:
        vals = {}
        for x in s["data"]:
            if not x["period"].startswith("M") or x["period"] == "M13":
                continue
            d = pd.Timestamp(int(x["year"]), int(x["period"][1:]), 1) + pd.offsets.MonthEnd(0)
            vals[d] = pd.to_numeric(x["value"], errors="coerce")
        out[s["seriesID"]] = pd.Series(vals, dtype=float).sort_index()
    return out


def shiller_cpi(refresh: bool = False) -> pd.Series:
    """Shiller's monthly CPI (pre-1913 = Warren-Pearson wholesale prices, rescaled)."""
    def fetch():
        for u in SHILLER:
            try:
                return _get(u, tries=1)
            except RuntimeError:
                continue
        raise RuntimeError("Shiller ie_data.xls unreachable")

    raw = _cached("shiller_ie_data.xls", fetch, refresh)
    x = pd.read_excel(io.BytesIO(raw), sheet_name="Data", header=None, skiprows=8, engine="xlrd")
    date = pd.to_numeric(x[0], errors="coerce")
    cpi = pd.to_numeric(x[4], errors="coerce")
    ok = date.notna() & cpi.notna()
    yr = np.floor(date[ok]).astype(int)
    mo = np.round((date[ok] - yr) * 100).astype(int)
    idx = pd.to_datetime(dict(year=yr, month=mo, day=1)) + pd.offsets.MonthEnd(0)
    return pd.Series(cpi[ok].values, index=pd.DatetimeIndex(idx), name="shiller_cpi").sort_index()


def yahoo_dates(symbol: str, start: pd.Timestamp, refresh: bool = False) -> pd.DatetimeIndex:
    """Trading dates of a liquid ETF from ``start`` to now (US/Eastern dates)."""
    p1 = int(pd.Timestamp(start).tz_localize("UTC").timestamp())
    p2 = int(time.time()) + 86400
    url = f"{YAHOO}{symbol}?period1={p1}&period2={p2}&interval=1d&events=div,splits"
    def fetch():
        try:
            # Yahoo rate-limits full Chrome UA strings from this network; a bare one works
            return _get(url, tries=4, ua="Mozilla/5.0")
        except RuntimeError:  # rate-limited: try the second host
            return _get(url.replace("query1.", "query2."), tries=4, ua="Mozilla/5.0")

    raw = _cached(f"yahoo_{symbol}_dates.json", fetch, refresh)
    res = json.loads(raw)["chart"]["result"][0]
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(res["meta"]["exchangeTimezoneName"])
    close = res["indicators"]["quote"][0]["close"]
    d = pd.DatetimeIndex(ts.tz_localize(None).normalize())
    keep = [c is not None for c in close]
    return pd.DatetimeIndex(sorted(set(d[keep]))).rename("date")


# --------------------------------------------------------------------------- cash


def _bill_mm_rate(discount_pct: pd.Series, days: int = 28) -> pd.Series:
    """Discount-basis bill rate (%) -> money-market (actual/360) simple rate, decimal."""
    d = discount_pct / 100.0
    return d / (1.0 - d * days / 360.0)


def _yield_daily_returns(dates: pd.DatetimeIndex, mm_rate: pd.Series, prev_date: pd.Timestamp) -> pd.DataFrame:
    """Daily returns on ``dates``: rate known at the previous trading day, accrued act/360."""
    rate = mm_rate.dropna().sort_index()
    dates = pd.DatetimeIndex(dates)
    prevs = pd.DatetimeIndex([prev_date]).append(dates[:-1])
    pos = rate.index.searchsorted(prevs, side="right") - 1
    if (pos < 0).any():
        raise ValueError("no yield before " + str(prevs[pos < 0][0]))
    stale = (prevs - rate.index[pos]).days.values
    days = (dates - prevs).days.values
    ret = rate.values[pos] * days / 360.0
    ret = np.where(stale > MAX_YIELD_FFILL_DAYS, np.nan, ret)
    # carried: the previous trading day is after the last published yield
    carried = prevs > rate.index[-1]
    return pd.DataFrame({"ret": ret, "stale_days": stale, "carried": carried}, index=dates.rename("date"))


def build_cash(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    diag: dict = {}
    rf_m, rf_d = french_rf(refresh)
    last_french = rf_m.index[-1]
    diag["french_monthly"] = (rf_m.index[0], last_french)
    diag["french_daily"] = (rf_d.index[0], rf_d.index[-1])

    # --- French era, daily = monthly spread geometrically over trading days
    cal = rf_d.loc[: last_french].index
    month = cal + pd.offsets.MonthEnd(0)
    n = pd.Series(1, index=cal).groupby(month).transform("size").values
    daily_fr = (1 + rf_m.reindex(month).values) ** (1.0 / n) - 1
    daily_fr = pd.Series(daily_fr, index=cal)
    diag["french_daily_rounding_max_abs_bp"] = float((daily_fr - rf_d.reindex(cal)).abs().max() * 1e4)
    diag["french_daily_raw_compound_gap_bp_yr"] = float(
        ((1 + rf_d).groupby(rf_d.index + pd.offsets.MonthEnd(0)).prod() - 1 - rf_m).dropna().mean() * 12e4)

    # --- yield-based tail
    w4 = dbnomics("FED/H15/RIFSGFSW04_N.B", refresh)
    m3 = dbnomics("FED/H15/RIFSGFSM03_N.B", refresh)
    mm4 = _bill_mm_rate(w4, 28)
    mm3 = _bill_mm_rate(m3, 91)

    # splice diagnostics: monthly compounded yield-based returns on the French calendar
    def monthly_from_yield(mm, start):
        dd = cal[cal >= start]
        r = _yield_daily_returns(dd[1:], mm, dd[0])["ret"]
        g = (1 + r).groupby(r.index + pd.offsets.MonthEnd(0))
        m = g.prod() - 1
        full = g.size() == pd.Series(1, index=dd).groupby(dd + pd.offsets.MonthEnd(0)).size().reindex(m.index)
        m = m[full & g.apply(lambda x: x.notna().all())]
        return m.loc[: last_french].iloc[1:]

    splice = {}
    for label, mm, start in [("4-week bill", mm4, w4.dropna().index[0]),
                             ("3-month bill", mm3, pd.Timestamp("1954-01-04"))]:
        ym = monthly_from_yield(mm, start)
        fr = rf_m.reindex(ym.index)
        for per, sl in [("full", slice(None)), ("2015+", slice("2015", None)), ("last 24m", slice(ym.index[-24], None))]:
            a, b = ym.loc[sl], fr.loc[sl]
            diff = (a - b) * 12e4  # bp/yr
            splice[(label, per)] = dict(
                start=a.index[0].strftime("%Y-%m"), end=a.index[-1].strftime("%Y-%m"), n=len(a),
                mean_diff_bp_yr=float(diff.mean()), rmse_bp_yr=float(np.sqrt((diff ** 2).mean())),
                corr=float(np.corrcoef(a, b)[0, 1]),
                ann_yield=float(a.mean() * 12 * 100), ann_french=float(b.mean() * 12 * 100))
    diag["splice"] = splice

    spy = yahoo_dates("SPY", last_french - pd.Timedelta(days=40), refresh)
    tail_dates = spy[spy > last_french]
    # the French calendar and SPY must agree on the overlap
    ov = spy[(spy > last_french - pd.Timedelta(days=35)) & (spy <= last_french)]
    diag["calendar_overlap_mismatch"] = sorted(set(ov) ^ set(cal[cal >= ov[0]])) if len(ov) else None
    tail = _yield_daily_returns(tail_dates, mm4, cal[-1])
    diag["tail"] = (tail.index[0], tail.index[-1], int(tail["carried"].sum()), w4.dropna().index[-1])

    src_tail = np.where(tail["carried"], "h15_tbill4w_ffill", "h15_tbill4w")
    daily = pd.concat([
        pd.DataFrame({"ret": daily_fr, "source": "french_rf", "tier": "C"}),
        pd.DataFrame({"ret": tail["ret"], "source": src_tail, "tier": "D"}),
    ])
    daily.index.name = "date"

    # --- monthly
    m_d = (1 + daily["ret"]).groupby(daily.index + pd.offsets.MonthEnd(0)).prod() - 1
    m_tail = m_d.loc[m_d.index > last_french]
    tbill = nber("m13029a", refresh)
    cp = nber("m13002", refresh)
    ov_idx = pd.date_range("1920-01-31", "1926-06-30", freq="ME")
    spread = float((cp.reindex(ov_idx) - tbill.reindex(ov_idx)).mean())
    diag["cp_spread_pct"] = spread
    # check of the proxy against French where both exist
    chk = pd.date_range("1926-07-31", "1929-12-31", freq="ME")
    diag["proxy_check_1926_29"] = dict(
        french=float(rf_m.reindex(chk).mean() * 1200), treas_3_6m=float(tbill.reindex(chk).mean()),
        cp_adj=float((cp.reindex(chk) - spread).mean()), cp_raw=float(cp.reindex(chk).mean()))
    pre = pd.concat([
        pd.DataFrame({"ret": (cp - spread).loc["1871-01":"1919-12"] / 1200,
                      "source": "nber_m13002_cp_minus_spread", "tier": "D"}),
        pd.DataFrame({"ret": tbill.loc["1920-01":"1926-06"] / 1200, "source": "nber_m13029a_treas_3_6m",
                      "tier": "D"}),
    ])
    monthly = pd.concat([
        pre,
        pd.DataFrame({"ret": rf_m, "source": "french_rf", "tier": "C"}),
        pd.DataFrame({"ret": m_tail, "source": "h15_tbill4w", "tier": "D"}),
    ])
    monthly.index.name = "date"
    gap = ((m_d - monthly["ret"].reindex(m_d.index)).abs()).max()
    diag["compound_max_abs_gap"] = float(gap)
    return daily, monthly, diag


# ------------------------------------------------------------------------- macro

def _me_plus(obs: pd.DatetimeIndex, months: int) -> pd.DatetimeIndex:
    """Last day of the month ``months`` after obs (obs are month-ends)."""
    return pd.DatetimeIndex(obs) + pd.offsets.MonthEnd(months)


def _long(series: str, values: pd.Series, avail: pd.Series | None = None) -> pd.DataFrame:
    """Long rows; ``avail`` is indexed by obs_date (None: available at obs_date, i.e. market data)."""
    values = values.dropna()
    av = values.index.to_series() if avail is None else pd.Series(avail).reindex(values.index)
    if av.isna().any():
        raise ValueError(f"{series}: missing available_date")
    return pd.DataFrame({"obs_date": values.index, "available_date": pd.DatetimeIndex(av.values),
                         "series": series, "value": values.values.astype(float)})


# Releases known to have slipped past the end of the following month.
UNRATE_OVERRIDES = {"2025-09-30": "2025-11-20"}   # 2025 shutdown: Sept report on Nov 20

LAGS = {  # series -> description used in the report
    "cpi": "end of month M+1 (from 1947; BLS releases ~10th-15th of M+1); end of M+3 for 1871-1946",
    "unrate": "end of M+1 (from 1960; Employment Situation ~first Friday of M+1); end of M+2 before 1960; "
              "Sept-2025 report -> 2025-11-20",
    "ip": "end of M+1 (from 1960; G.17 ~15th-17th of M+1); end of M+2 before 1960",
    "market": "end of month M + 1 business day (monthly average of daily market rates; H.15 publishes a "
              "day's rates the next business day)",
    "h10": "end of month M + 5 business days (H.10 dollar index is released weekly, on Mondays, for the "
           "previous week)",
}
# Publication lags of monthly averages of daily market data (business days after month-end).
MARKET_LAG_BDAYS = 1   # H.15 rates (and the historical NBER compilations treated alike)
H10_LAG_BDAYS = 5      # H.10 broad dollar index: weekly release


def build_macro(cash_monthly: pd.DataFrame, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    diag: dict = {}
    frames = []

    # ---------------- CPI (NSA, never revised)
    cpi_db = dbnomics("BLS/cu/CUUR0000SA0", refresh)
    bls = bls_recent(["CUUR0000SA0", "LNS14000000"], refresh)
    cpi_recent = bls["CUUR0000SA0"]
    both = cpi_db.index.intersection(cpi_recent.dropna().index)
    diag["cpi_dbnomics_vs_api_maxdiff"] = float((cpi_db[both] - cpi_recent[both]).abs().max()) if len(both) else None
    cpi = cpi_db.combine_first(cpi_recent)
    cpi.loc[cpi_recent.index] = cpi_recent
    sh = shiller_cpi(refresh)
    pre = sh.loc[:"1912-12"] * (cpi.loc["1913-01-31"] / sh.loc["1913-01-31"])
    cpi = pd.concat([pre, cpi.loc["1913-01":]]).sort_index()
    cpi_av = pd.Series(np.where(cpi.index >= "1947-01-01", _me_plus(cpi.index, 1), _me_plus(cpi.index, 3)),
                       index=cpi.index)
    infl_yoy = 100 * np.log(cpi / cpi.shift(12, freq="ME").reindex(cpi.index))
    dlog = np.log(cpi).diff()
    dlog[cpi.index.to_series().diff().dt.days > 31] = np.nan
    infl_vol = 100 * np.sqrt(12) * dlog.rolling(36, min_periods=30).std()
    frames += [_long("macro_cpi_level", cpi, cpi_av),
               _long("macro_infl_yoy", infl_yoy, cpi_av),
               _long("macro_infl_vol_36m", infl_vol, cpi_av)]

    # ---------------- short rate (%, annualized), long yield, term spread
    tb3 = dbnomics("FED/H15/RIFSGFSM03_N.M", refresh)
    t29a, t29b, cp = nber("m13029a", refresh), nber("m13029b", refresh), nber("m13002", refresh)
    spread = float((cp - t29a).loc["1920-01":"1926-06"].mean())
    short = pd.concat([(cp - spread).loc["1871-01":"1919-12"], t29a.loc["1920-01":"1930-12"],
                       t29b.loc["1931-01":"1933-12"], tb3.loc["1934-01":]]).dropna().sort_index()
    rr = nber("m13019", refresh)            # high-grade railroad bond yields
    ust_a, ust_b = nber("m13033a", refresh), nber("m13033b", refresh)
    gs10 = dbnomics("FED/H15/RIFLGFCY10_N.M", refresh)
    long = pd.concat([rr.loc["1871-01":"1918-12"], ust_a.loc["1919-01":"1941-09"],
                      ust_b.loc["1941-10":"1953-03"], gs10.loc["1953-04":]]).dropna().sort_index()
    for a, b, lbl in [(rr, ust_a, "railroad vs US long 1919-1936"), (ust_a, ust_b, "m13033a vs b 1941-44"),
                      (ust_b, gs10, "m13033b vs GS10 1953-67")]:
        j = pd.concat([a, b], axis=1).dropna()
        diag[f"long splice {lbl}"] = (len(j), float((j.iloc[:, 0] - j.iloc[:, 1]).mean()),
                                      float(j.corr().iloc[0, 1]))
    term = (long - short).dropna()

    def mkt_av(s: pd.Series, bdays: int = MARKET_LAG_BDAYS) -> pd.Series:
        return pd.Series(s.index + pd.offsets.BDay(bdays), index=s.index)
    frames += [_long("macro_short_rate", short, mkt_av(short)),
               _long("macro_long_yield", long, mkt_av(long)),
               _long("macro_term_spread", term, mkt_av(term))]
    real = (short - infl_yoy).dropna()
    frames.append(_long("macro_real_short_rate", real, cpi_av.reindex(real.index)))

    # ---------------- policy rate: NY Fed discount rate, then effective fed funds
    disc = nber("m13009", refresh)
    ff = dbnomics("FED/H15/RIFSPFF_N.M", refresh)
    pol_parts = [disc.loc[:"1954-06"].dropna(), ff.loc["1954-07":].dropna()]
    policy = pd.concat(pol_parts)
    frames.append(_long("macro_policy_rate", policy, mkt_av(policy)))
    for k in (3, 12):
        ch = pd.concat([p - p.shift(k, freq="ME").reindex(p.index) for p in pol_parts]).dropna()
        frames.append(_long(f"macro_policy_rate_chg_{k}m", ch, mkt_av(ch)))

    # ---------------- unemployment and a real-time-safe Sahm indicator
    un_db = dbnomics("BLS/ln/LNS14000000", refresh)
    un_recent = bls["LNS14000000"]
    both = un_db.index.intersection(un_recent.dropna().index)
    diag["unrate_dbnomics_vs_api_maxdiff"] = float((un_db[both] - un_recent[both]).abs().max()) if len(both) else None
    un = un_db.combine_first(un_recent)
    un.loc[un_recent.index] = un_recent
    un = un.dropna()
    un_av = pd.Series(np.where(un.index >= "1960-01-01", _me_plus(un.index, 1), _me_plus(un.index, 2)),
                      index=un.index)
    for o, a in UNRATE_OVERRIDES.items():
        if pd.Timestamp(o) in un_av.index:
            un_av[pd.Timestamp(o)] = pd.Timestamp(a)
    # Sahm: 3-month average of the published rate (months with no release, e.g. Oct-2025,
    # are skipped: the average is over the last 3 *published* months) minus the minimum
    # of that average over the prior 12 published months.
    avg3 = un.rolling(3).mean()
    gap = (avg3 - avg3.shift(1).rolling(12, min_periods=12).min()).dropna()
    # available once the latest month entering it is published (avail is monotone after max)
    gap_av = un_av.cummax().reindex(gap.index)
    chg12 = (un - un.shift(12, freq="ME").reindex(un.index)).dropna()
    frames += [_long("macro_unrate", un, un_av),
               _long("macro_unrate_chg_12m", chg12, un_av.reindex(chg12.index)),
               _long("macro_sahm_gap", gap, gap_av),
               _long("macro_sahm_recession", (gap >= 0.5).astype(float), gap_av)]
    diag["unrate_missing_months"] = [d.strftime("%Y-%m") for d in
                                     pd.date_range(un.index[0], un.index[-1], freq="ME").difference(un.index)]

    # ---------------- industrial production (SA, revised)
    ip = dbnomics("FED/G17_IP_MAJOR_INDUSTRY_GROUPS/IP.B50001.S", refresh).dropna()
    ip_av = pd.Series(np.where(ip.index >= "1960-01-01", _me_plus(ip.index, 1), _me_plus(ip.index, 2)),
                      index=ip.index)
    ip_yoy = (100 * np.log(ip / ip.shift(12, freq="ME").reindex(ip.index))).dropna()
    frames.append(_long("macro_ip_yoy", ip_yoy, ip_av.reindex(ip_yoy.index)))

    # ---------------- trade-weighted dollar, 12m log change (old goods-only broad to 2006, then new broad)
    old = dbnomics("FED/H10/V0.JRXWTFB_N.M", refresh).dropna()
    new = dbnomics("FED/H10/JRXWTFB_N.M", refresh).dropna()
    yo = 100 * np.log(old / old.shift(12, freq="ME").reindex(old.index))
    yn = 100 * np.log(new / new.shift(12, freq="ME").reindex(new.index))
    j = pd.concat([yo, yn], axis=1).dropna()
    diag["usd_splice"] = (len(j), float(j.corr().iloc[0, 1]), float((j.iloc[:, 0] - j.iloc[:, 1]).mean()))
    usd = pd.concat([yo.loc[:"2006-12"], yn.loc["2007-01":]]).dropna()
    frames.append(_long("macro_usd_broad_yoy", usd, mkt_av(usd, H10_LAG_BDAYS)))

    out = pd.concat(frames, ignore_index=True)
    out["obs_date"] = pd.to_datetime(out["obs_date"])
    out["available_date"] = pd.to_datetime(out["available_date"])
    out = out.sort_values(["series", "obs_date"]).reset_index(drop=True)
    return out, diag


# ------------------------------------------------------------------------ report

SERIES_META = {
    "macro_cpi_level": ("CPI-U NSA, 1982-84=100 (pre-1913 Shiller/Warren-Pearson wholesale index, rescaled)",
                        "BLS CUUR0000SA0 (DBnomics + BLS API v1); Shiller ie_data", "cpi"),
    "macro_infl_yoy": ("12-month log change of CPI, %", "derived from CPI", "cpi"),
    "macro_infl_vol_36m": ("trailing 36-month std of monthly log CPI change, annualized %", "derived from CPI", "cpi"),
    "macro_short_rate": ("short rate, % p.a.: CP minus spread (to 1919), 3-6m Treasury (1920-33), "
                         "3m T-bill secondary-market monthly avg (1934+)",
                         "NBER m13002, m13029a/b; H.15 RIFSGFSM03_N.M", "market"),
    "macro_long_yield": ("long yield, %: high-grade railroad bonds (to 1918), US long-term bonds (1919-1953-03), "
                         "10y CMT monthly avg (1953-04+)", "NBER m13019, m13033a/b; H.15 RIFLGFCY10_N.M", "market"),
    "macro_term_spread": ("long yield minus short rate, pp", "derived", "market"),
    "macro_real_short_rate": ("short rate minus trailing YoY inflation, pp", "derived", "cpi"),
    "macro_policy_rate": ("NY Fed discount rate (1914-11..1954-06), effective fed funds monthly avg (1954-07+), %",
                          "NBER m13009; H.15 RIFSPFF_N.M", "market"),
    "macro_policy_rate_chg_3m": ("3-month change in policy rate, pp (computed within each source)", "derived", "market"),
    "macro_policy_rate_chg_12m": ("12-month change in policy rate, pp (within source)", "derived", "market"),
    "macro_unrate": ("civilian unemployment rate, SA, %", "BLS LNS14000000 (DBnomics + BLS API v1)", "unrate"),
    "macro_unrate_chg_12m": ("12-month change in unemployment rate, pp", "derived", "unrate"),
    "macro_sahm_gap": ("3m avg unemployment minus its min over prior 12 months, pp", "derived", "unrate"),
    "macro_sahm_recession": ("1 if sahm_gap >= 0.5 else 0 (real-time-safe recession flag)", "derived", "unrate"),
    "macro_ip_yoy": ("12-month log change in industrial production (SA), %",
                     "Fed G.17 IP.B50001.S via DBnomics", "ip"),
    "macro_usd_broad_yoy": ("12-month log change of nominal broad dollar index, %: goods-only broad (to 2006), "
                            "broad (2007+)", "Fed H.10 V0.JRXWTFB_N.M, JRXWTFB_N.M via DBnomics", "h10"),
}


def _fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def write_report(daily, monthly, cdiag, macro, mdiag, path: Path) -> None:
    L = []
    a = L.append
    a("# Cash and macro predictors (`cash_macro` track)\n")
    a(f"Generated by `python -m assetforecast.data.sources.cash_macro` on {pd.Timestamp.today():%Y-%m-%d}.\n")
    a("## Cash (T-bill) returns\n")
    a(f"- Daily: {daily.index[0]:%Y-%m-%d} .. {daily.index[-1]:%Y-%m-%d}, {len(daily):,} trading days, "
      f"{int(daily['ret'].isna().sum())} NaN.")
    a(f"- Monthly: {monthly.index[0]:%Y-%m} .. {monthly.index[-1]:%Y-%m}, {len(monthly):,} months, "
      f"{int(monthly['ret'].isna().sum())} NaN. The last monthly row is month-to-date through "
      f"{daily.index[-1]:%Y-%m-%d} (compounded daily).")
    a("\n| segment | freq | start | end | source | tier |\n|---|---|---|---|---|---|")
    for name, df in [("daily", daily), ("monthly", monthly)]:
        g = df.groupby((df["source"] != df["source"].shift()).cumsum())
        for _, s in g:
            a(f"| {name} | {'D' if name == 'daily' else 'M'} | {s.index[0]:%Y-%m-%d} | {s.index[-1]:%Y-%m-%d} | "
              f"{s['source'].iloc[0]} | {s['tier'].iloc[0]} |")
    a("\nSources: `french_rf` = Ken French `RF` (1-month T-bill; Ibbotson to 2024-05, ICE BofA 1-month "
      "T-bill index from 2024-06). Daily values spread each month's RF geometrically over the month's "
      "trading days (French daily file dates), because the French *daily* RF column is rounded to 0.01 %/day "
      f"(max abs difference between ours and French daily: {cdiag['french_daily_rounding_max_abs_bp']:.2f} bp/day; "
      f"compounding French's rounded daily RF misses monthly RF by {cdiag['french_daily_raw_compound_gap_bp_yr']:.0f} bp/yr on average). "
      "`h15_tbill4w` = 4-week bill secondary-market discount rate (Fed H.15 via DBnomics), converted to an "
      "act/360 money-market rate; the rate known at the previous trading day accrues over the calendar days "
      "to the current day; trading dates from SPY (Yahoo). `h15_tbill4w_ffill` = same, using the last "
      f"published yield (H.15 data end {cdiag['tail'][3]:%Y-%m-%d}; carried at most {MAX_YIELD_FFILL_DAYS} days). "
      "`nber_m13029a_treas_3_6m` = yield on 3-6 month Treasury notes and certificates; "
      "`nber_m13002_cp_minus_spread` = NYC prime commercial paper rate minus "
      f"{cdiag['cp_spread_pct']:.2f} pp (its mean excess over m13029a in 1920-01..1926-06). Pre-1926 monthly ret = yield/1200.\n")
    p = cdiag["proxy_check_1926_29"]
    a(f"Proxy check 1926-07..1929-12 (annualized %): French RF {p['french']:.2f}; Treasury 3-6m {p['treas_3_6m']:.2f}; "
      f"CP minus spread {p['cp_adj']:.2f}; raw CP {p['cp_raw']:.2f}.\n")
    a("### Splice diagnostics: yield-based monthly cash vs French RF (overlap)\n")
    a("| yield | window | months | mean diff (bp/yr) | RMSE (bp/yr) | corr | yield-based % p.a. | French % p.a. |")
    a("|---|---|---|---|---|---|---|---|")
    for (lbl, per), d in cdiag["splice"].items():
        a(f"| {lbl} | {per} {d['start']}..{d['end']} | {d['n']} | {d['mean_diff_bp_yr']:+.1f} | {d['rmse_bp_yr']:.1f} | "
          f"{d['corr']:.4f} | {d['ann_yield']:.2f} | {d['ann_french']:.2f} |")
    a(f"\nCompounded daily vs monthly: max abs gap {cdiag['compound_max_abs_gap']:.2e}. "
      f"SPY vs French calendar mismatches on the last month of overlap: {cdiag['calendar_overlap_mismatch']}.\n")

    a("## Macro predictors (`data/processed/predictors/macro.parquet`)\n")
    a("Long format `obs_date, available_date, series, value`. `obs_date` is the month-end of the "
      "reference month. One vintage per observation (latest available), so `available_date` is the "
      "*first-release date* assumption applied to latest-vintage values.\n")
    a("| series | description | start | end | n | freq | source | assumed availability |")
    a("|---|---|---|---|---|---|---|---|")
    for s, g in macro.groupby("series"):
        desc, src, lag = SERIES_META[s]
        a(f"| `{s}` | {desc} | {g['obs_date'].min():%Y-%m} | {g['obs_date'].max():%Y-%m} | {len(g)} | M | {src} | {LAGS[lag]} |")
    a("\nDiagnostics:\n")
    for k, v in mdiag.items():
        a(f"- {k}: {v}")
    a("""
### Point-in-time caveats

- **CPI NSA** is never revised: latest vintage = real-time value. Before ~1940 the BLS index was
  not published monthly in real time and pre-1913 values (Shiller, from Warren-Pearson wholesale
  prices) are a later reconstruction, much more volatile than a consumer index: treat
  `macro_infl_vol_36m` before 1913 (and in the 1910s-40s) as a different regime.
- **Unemployment (SA)** is revised each January for five years of seasonal factors; revisions are
  typically <= 0.1-0.2 pp. The October-2025 rate was never published (2025 shutdown) and is left
  missing; the Sahm average runs over the last three *published* months.
- **Industrial production** is revised materially (monthly revisions and annual benchmarks); only
  the latest vintage is available here (ALFRED is not reachable). Use with caution.
- **Recession indicator**: NBER dates are deliberately not used. `macro_sahm_recession` is built
  from published unemployment and is available when its latest input is.
- Market-rate series (short rate, long yield, policy rate, dollar) are monthly averages of daily market
  observations. The last day's H.15 rate is published the next business day, so they are dated available
  one business day after month-end; the broad dollar (H.10, released weekly on Mondays) five business days
  after month-end. Long-yield splices across sources change
  the level (railroad bonds yield more than governments): see diagnostics above.
- The long-yield splice in 1941-10 (m13033a -> m13033b, different bond baskets) shifts the level
  by about 0.5 pp (overlap corr is low because both are nearly flat under the 1940s peg).
- Unemployment before the BLS API window (last ten years) comes from the DBnomics copy (vintage of
  early 2025), the last ten years from the BLS API: up to 0.1 pp vintage differences between the two.
- The October-2025 CPI was never published, so `macro_infl_yoy` (and real rate) for 2026-10 and
  `macro_unrate_chg_12m` for 2026-10 are missing rather than imputed.
- The BLS v1 API allows 25 requests/day per IP; the loader makes one request and caches it.
- Monthly French RF is rounded to 0.01 %/month (0.12 % p.a.), which is why the correlation with the
  yield-based series drops in the flat-rate last 24 months while the mean difference stays ~5 bp/yr.
- Pre-1926 cash is a model (tier D); its level could be off by ~0.5-1 pp p.a., which is
  immaterial for volatility but matters for long-run means.
""")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))


# -------------------------------------------------------------------------- main


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-download instead of using data/raw/cash_macro")
    args = ap.parse_args(argv)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    daily, monthly, cdiag = build_cash(args.refresh)
    macro, mdiag = build_macro(monthly, args.refresh)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTOR_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(ASSET_DIR / "cash_daily.parquet")
    monthly.to_parquet(ASSET_DIR / "cash_monthly.parquet")
    macro.to_parquet(PREDICTOR_DIR / "macro.parquet", index=False)
    write_report(daily, monthly, cdiag, macro, mdiag, REPORTS / "data" / "cash_macro.md")
    print(f"cash daily {daily.index[0]:%Y-%m-%d}..{daily.index[-1]:%Y-%m-%d} ({len(daily)}), "
          f"monthly {monthly.index[0]:%Y-%m}..{monthly.index[-1]:%Y-%m}; macro rows {len(macro)}, "
          f"series {macro['series'].nunique()}")


if __name__ == "__main__":
    main()
