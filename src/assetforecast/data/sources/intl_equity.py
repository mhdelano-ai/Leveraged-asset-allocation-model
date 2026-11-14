"""International equity total returns: ``intl_dev`` (developed ex-US, EFA-like)
and ``em`` (emerging markets, EEM-like). USD, unhedged, simple TOTAL returns.

Run ``python -m assetforecast.data.sources.intl_equity`` to (re)build
``data/processed/assets/{intl_dev,em}_{daily,monthly}.parquet`` and
``reports/data/intl_equity.md``. Raw downloads are cached under
``data/raw/intl_equity/``; pass ``--refresh`` to re-download.

Construction (diagnostics and justification in ``reports/data/intl_equity.md``)
------------------------------------------------------------------------------
intl_dev
  * 1975-01 .. 1985-12  monthly, Ken French "International Indices" (Ind_all):
                        countries at EAFE weights, value-weighted within country,
                        USD, with dividends (MSCI raw data to 2006)        -> tier C
  * 1986-01 .. 2001-08  daily, AQR market return "Global Ex USA" (cap-weighted
                        developed ex-US incl. Canada) = excess + AQR T-bill  -> tier B
                        Convention: index, FOREIGN closes.
  * 2001-09 .. today    daily, iShares MSCI EAFE ETF (EFA), close + validated
                        distributions                                       -> tier A
                        Convention: US close.
em
  * 1989-07 .. 1994-05  monthly, Ken French Emerging Markets Mkt-RF + RF  -> tier C
  * 1994-06 .. 2003-04  daily, Vanguard Emerging Markets Stock Index Fund
                        (VEIEX) adjclose; open-end NAV, mostly foreign-close
                        prices in this era (fair-value pricing grew from 2001) -> tier B
  * 2003-05 .. today    daily, iShares MSCI Emerging Markets ETF (EEM), close +
                        validated distributions                             -> tier A
                        Convention: US close.

Tiers (project-wide): A = daily price of an exchange-traded fund (market
close); B = daily open-end mutual-fund NAV, index, or reconstruction; C =
monthly index total return; D = monthly model-based reconstruction.

ETF total returns are close + Yahoo distributions, each validated against the
ex-date price move relative to peer funds (``distributions.validate_distributions``).
Confirmed bad NAV prints are dropped (``BAD_PRICES``; the next return spans them).

Splices are at calendar-month boundaries, so every month is single-source.
Nothing is interpolated; index days with a ~zero total return (|r| < 0.001%:
all markets closed, e.g. Jan 1) are dropped. Each daily file keeps its source's own
calendar (the AQR index has non-US trading days such as US Thanksgiving).
The monthly files run to the last COMPLETE calendar month.
"""
from __future__ import annotations

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

CACHE = RAW / "intl_equity"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
AQR_BASE = "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
MSCI_URL = "https://app2.msci.com/products/service/index/indexmaster/getLevelDataForGraph"
YAHOO_URL = "https://{host}.finance.yahoo.com/v8/finance/chart/{sym}"


def _get(url: str, params: dict | None = None, tries: int = 6, sleep: float = 2.0,
         timeout: float = 180) -> bytes:
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": UA, "Accept": "*/*"})
            if r.status_code == 200 and r.content:
                return r.content
            last = f"HTTP {r.status_code}"
            if r.status_code == 404:
                break
        except requests.RequestException as e:  # pragma: no cover - network
            last = repr(e)
        time.sleep(sleep * (i + 1))
    raise RuntimeError(f"GET failed for {url}: {last}")


def _cached(name: str, fetch, refresh: bool = False) -> bytes:
    path = CACHE / name
    if path.exists() and not refresh:
        return path.read_bytes()
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def yahoo(sym: str, refresh: bool = False) -> pd.DataFrame:
    """Daily close / adjclose from the Yahoo chart API (cached JSON)."""
    def fetch():
        p = {"period1": 0, "period2": int(time.time()) + 86400, "interval": "1d",
             "events": "div,splits", "includeAdjustedClose": "true"}
        last = None
        for host in ["query1", "query2", "query1", "query2"]:
            try:
                b = _get(YAHOO_URL.format(host=host, sym=sym), p, tries=3)
                json.loads(b)["chart"]["result"][0]["timestamp"]
                time.sleep(1.0)
                return b
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(3)
        raise RuntimeError(f"yahoo {sym}: {last}")
    raw = json.loads(_cached(f"yahoo/{sym.replace('^', '_')}.json", fetch, refresh))
    res = raw["chart"]["result"][0]
    tz = res["meta"].get("exchangeTimezoneName", "America/New_York")
    idx = (pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz)
           .tz_localize(None).normalize())
    q = res["indicators"]
    df = pd.DataFrame({"close": q["quote"][0]["close"],
                       "adjclose": q.get("adjclose", [{}])[0].get("adjclose")}, index=idx)
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].dropna(how="all").astype(float)
    ev = res.get("events", {}).get("dividends", {})
    to_date = lambda t: pd.to_datetime(t, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()  # noqa: E731
    div = pd.Series({to_date(int(v["date"])): float(v["amount"]) for v in ev.values()}, dtype=float)
    div = div.groupby(level=0).sum() if len(div) else div
    div, _ = dist.dedupe_repeats(div)
    df["div"] = dist.align_dividends(div, df.index[df["close"].notna()]).reindex(df.index).fillna(0.0)
    return df


# ------------------------------------------------------------------ parsers
def french_zip(fname: str, refresh: bool = False) -> dict[str, str]:
    b = _cached(f"french/{fname}", lambda: _get(FRENCH_BASE + fname), refresh)
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        return {n: z.read(n).decode("latin-1") for n in z.namelist()}


def french_factors(fname: str, refresh: bool = False) -> pd.DataFrame:
    """First table of a Ken French factor CSV, in DECIMALS, indexed by date."""
    (text,) = french_zip(fname, refresh).values()
    rows, header, started = [], None, False
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not started:
            if len(parts) > 1 and parts[0] == "" and "Mkt-RF" in parts:
                header, started = parts[1:], True
            continue
        if not parts[0].isdigit():
            break
        rows.append([parts[0]] + [float(x) for x in parts[1:]])
    df = pd.DataFrame(rows, columns=["d"] + header)
    fmt = "%Y%m%d" if len(df["d"].iloc[0]) == 8 else "%Y%m"
    idx = pd.to_datetime(df["d"], format=fmt)
    if fmt == "%Y%m":
        idx = idx + pd.offsets.MonthEnd(0)
    df = df.drop(columns="d").set_index(idx).rename_axis("date")
    return df.where(df > -99.0) / 100.0


def french_intl_index(member: str = "Ind_all.Dat", refresh: bool = False) -> pd.Series:
    """KF International Indices: EAFE-country-weighted, value-weighted within
    country, USD, WITH dividends, monthly. Returns the 'Mkt' column (decimals)
    of the first block ('Value-Weight Dollar Returns, All 4 Data Items Not Reqd')."""
    text = french_zip("F-F_International_Indices.zip", refresh)[member]
    out, block = {}, 0
    for line in text.splitlines():
        if "Returns" in line:
            block += 1
            continue
        if block != 1:
            continue
        parts = line.split()
        if parts and parts[0].isdigit() and len(parts[0]) == 6:
            v = float(parts[1])
            if v > -99.0:
                out[pd.Timestamp(parts[0][:4] + "-" + parts[0][4:] + "-01") + pd.offsets.MonthEnd(0)] = v / 100
    s = pd.Series(out, name="ret").sort_index()
    s.index.name = "date"
    return s


def aqr_sheet(freq: str, sheet: str, refresh: bool = False) -> pd.DataFrame:
    """AQR 'Betting Against Beta' workbook sheet (MKT = excess returns, RF = T-bill)."""
    name = f"Betting-Against-Beta-Equity-Factors-{'Daily' if freq == 'D' else 'Monthly'}.xlsx"
    b = _cached(f"aqr/{name}", lambda: _get(AQR_BASE + name, timeout=600), refresh)
    df = pd.read_excel(io.BytesIO(b), sheet_name=sheet, header=18, index_col=0)
    df = df[df.index.notna()]
    df.index = pd.to_datetime(df.index.astype(str), format="%m/%d/%Y")
    df.index.name = "date"
    return df.apply(pd.to_numeric, errors="coerce")


def msci(code: str, variant: str = "GRTR", freq: str = "DAILY", refresh: bool = False) -> pd.Series:
    """MSCI index levels in USD from the public graph endpoint (serves 1997+ only;
    GRTR/NETR start 2000-12-29)."""
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y%m%d")
    params = {"currency_symbol": "USD", "index_variant": variant, "start_date": "19970101",
              "end_date": end, "data_frequency": freq, "index_codes": code}
    b = _cached(f"msci/{code}_{variant}_{freq}.json", lambda: _get(MSCI_URL, params), refresh)
    lv = json.loads(b)["indexes"]["INDEX_LEVELS"]
    s = pd.Series({pd.Timestamp(str(x["calc_date"])): x["level_eod"] for x in lv}, name=code)
    s.index.name = "date"
    return s.sort_index()


# ------------------------------------------------------------- base series
SOURCE = {
    "kf_intl_index": "french_intl_indices_eafe_weighted",
    "aqr_gxus": "aqr_mkt_global_ex_usa_plus_rf",
    "EFA": "yahoo_EFA",
    "kf_em": "french_emerging_mkt_plus_rf",
    "VEIEX": "yahoo_VEIEX",
    "EEM": "yahoo_EEM",
}
# (key, tier, first date, last date) per asset; freq M segments come first.
PLAN = {
    "intl_dev": {"monthly": ("kf_intl_index", "C", "1975-01-31", "1985-12-31"),
                 "daily": [("aqr_gxus", "B", "1986-01-01", "2001-08-31"),
                           ("EFA", "A", "2001-09-01", None)]},
    "em": {"monthly": ("kf_em", "C", "1989-07-31", "1994-05-31"),
           "daily": [("VEIEX", "B", "1994-06-01", "2003-04-30"),
                     ("EEM", "A", "2003-05-01", None)]},
}
# Confirmed bad prints: the PRICE is dropped (the next return spans it; never interpolated).
# VEIEX 2001-07-16: NAV +5.6% then -5.9% the next day, while EM peers TEDMX (-0.6%, -0.8%),
# FOSFX and SCINX were flat (independent audit, section C-2).
BAD_PRICES = {"VEIEX": ["2001-07-16"]}
# ETFs whose return is close + VALIDATED distributions; peers are total returns (adjclose)
# of related funds plus the S&P 500 price index.
ETF_PEERS = {"EFA": ["VEA", "VTMGX", "^GSPC"], "EEM": ["VWO", "IEMG", "^GSPC"]}
CONVENTION = {
    "kf_intl_index": "monthly index (foreign closes; month-end timing)",
    "aqr_gxus": "daily index, foreign local closes converted to USD",
    "EFA": "US-listed ETF, US close (16:00 ET)",
    "kf_em": "monthly index (foreign closes)",
    "VEIEX": "open-end fund NAV: foreign closes, partially fair-valued to US close from ~2001",
    "EEM": "US-listed ETF, US close (16:00 ET)",
}


def yahoo_returns(sym: str, refresh: bool = False) -> pd.Series:
    """Daily total returns: close + validated distributions for the ETFs in
    ``ETF_PEERS``; adjclose for funds and diagnostics. ``BAD_PRICES`` are dropped."""
    if sym in ETF_PEERS:
        return etf_returns(sym, refresh)[0]
    a = yahoo(sym, refresh)["adjclose"].dropna()
    a = a[a > 0]
    a = a.drop(pd.DatetimeIndex(BAD_PRICES.get(sym, [])), errors="ignore")
    return a.pct_change().iloc[1:].rename(sym)


def etf_returns(sym: str, refresh: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    df = yahoo(sym, refresh).dropna(subset=["close"])
    df = df[df["close"] > 0]
    peers = {}
    for p in ETF_PEERS[sym]:
        y = yahoo(p, refresh)
        col = "close" if p.startswith("^") else "adjclose"
        peers[p] = y[col].dropna().pct_change()
    div, table = dist.validate_distributions(df["close"], df["div"], pd.DataFrame(peers))
    r = dist.total_return(df["close"], div).iloc[1:].dropna()
    return r.rename(sym), table


ZERO_DAY = 1e-5


def aqr_total(col: str, refresh: bool = False) -> pd.Series:
    """AQR daily market TOTAL return = excess return + AQR daily T-bill."""
    mkt = aqr_sheet("D", "MKT", refresh)[col]
    rf = aqr_sheet("D", "RF", refresh).iloc[:, 0]
    rf = rf.reindex(mkt.index.union(rf.index)).ffill().reindex(mkt.index)
    r = (mkt + rf).dropna()
    # AQR lists every weekday; on all-closed days (Jan 1, Dec 25, ...) the total
    # return is ~0 up to T-bill rounding. Drop those (|r| < 0.001%).
    return r[r.abs() >= ZERO_DAY].rename(col)


def kf_em_monthly(refresh: bool = False) -> pd.Series:
    f = french_factors("Emerging_5_Factors_CSV.zip", refresh)
    return (f["Mkt-RF"] + f["RF"]).dropna().rename("kf_em")


def kf_dev_monthly(refresh: bool = False) -> pd.Series:
    f = french_factors("Developed_ex_US_3_Factors_CSV.zip", refresh)
    return (f["Mkt-RF"] + f["RF"]).dropna().rename("kf_dev")


def kf_dev_daily(refresh: bool = False) -> pd.Series:
    f = french_factors("Developed_ex_US_3_Factors_Daily_CSV.zip", refresh)
    return (f["Mkt-RF"] + f["RF"]).dropna().rename("kf_dev_d")


def msci_returns(code: str, freq: str = "DAILY", refresh: bool = False) -> pd.Series:
    lv = msci(code, "GRTR", freq, refresh)
    r = lv.pct_change().iloc[1:]
    if freq == "DAILY":
        r = r[r.abs() > 1e-12]          # weekday holidays carry the previous level
    else:
        r.index = r.index + pd.offsets.MonthEnd(0)
    return r.rename(f"msci_{code}")


def base_series(key: str, refresh: bool = False) -> pd.Series:
    if key == "kf_intl_index":
        return french_intl_index(refresh=refresh)
    if key == "aqr_gxus":
        return aqr_total("Global Ex USA", refresh)
    if key == "kf_em":
        return kf_em_monthly(refresh)
    return yahoo_returns(key, refresh)


# ------------------------------------------------------------------ build
def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby(pd.Grouper(freq="ME"))
    m = pd.DataFrame({
        "ret": g["ret"].apply(lambda x: float(np.prod(1 + x.values) - 1) if len(x) else np.nan),
        "source": g["source"].agg(lambda x: x.mode().iloc[0] if len(x) else None),
        "tier": g["tier"].agg(lambda x: x.mode().iloc[0] if len(x) else None),
    }).dropna(subset=["ret"])
    last = daily.index[-1]
    if last < last + pd.offsets.BMonthEnd(0):     # last month incomplete
        m = m.loc[: last - pd.offsets.MonthEnd(1)]
    return m


def build_asset(asset: str, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    plan = PLAN[asset]
    parts = []
    for key, tier, start, end in plan["daily"]:
        r = base_series(key, refresh).loc[start:end]
        parts.append(pd.DataFrame({"ret": r.astype(float), "source": SOURCE[key], "tier": tier}))
    daily = pd.concat(parts)
    daily.index = pd.DatetimeIndex(daily.index).as_unit("ns")
    daily.index.name = "date"
    assert daily.index.is_monotonic_increasing and daily.index.is_unique
    key, tier, start, end = plan["monthly"]
    pre = base_series(key, refresh).loc[start:end]
    pre = pd.DataFrame({"ret": pre.astype(float), "source": SOURCE[key], "tier": tier})
    monthly = pd.concat([pre, to_monthly(daily)])
    monthly.index = pd.DatetimeIndex(monthly.index).as_unit("ns")
    monthly.index.name = "date"
    for df in (daily, monthly):
        df["source"] = df["source"].astype(str)
        df["tier"] = df["tier"].astype(str)
    assert monthly.index.is_unique and monthly.index.is_monotonic_increasing
    return daily, monthly


def build(refresh: bool = False, write: bool = True) -> dict:
    out = {}
    for asset in PLAN:
        daily, monthly = build_asset(asset, refresh)
        out[asset] = (daily, monthly)
        if write:
            ASSET_DIR.mkdir(parents=True, exist_ok=True)
            daily.to_parquet(ASSET_DIR / f"{asset}_daily.parquet")
            monthly.to_parquet(ASSET_DIR / f"{asset}_monthly.parquet")
    return out


# ------------------------------------------------------------ diagnostics
def _align(*xs, lo=None, hi=None) -> pd.DataFrame:
    df = pd.concat([np.log1p(x) for x in xs], axis=1, join="inner").dropna()
    return df.loc[lo:hi]


def _vol(x: pd.Series, k: int = 1, per_year: int = 252) -> float:
    """Annualized vol from overlapping k-period log sums (uncentered)."""
    s = x.rolling(k).sum().dropna() if k > 1 else x
    return float(np.sqrt((s ** 2).mean() * per_year / k))


def daily_pair(ref: pd.Series, new: pd.Series, lo=None, hi=None) -> dict:
    df = _align(ref, new, lo=lo, hi=hi)
    x, y = df.iloc[:, 0], df.iloc[:, 1]
    x3, y3 = x.rolling(3).sum(), y.rolling(3).sum()
    return {"start": df.index[0].date(), "end": df.index[-1].date(), "n": len(df),
            "corr_1d": x.corr(y), "corr_3d": x3.corr(y3),
            "vol_ref_1d": _vol(x), "vol_new_1d": _vol(y),
            "vol_ratio_1d": _vol(y) / _vol(x), "vol_ratio_3d": _vol(y, 3) / _vol(x, 3),
            "te_3d": float((y3 - x3).dropna().std() * np.sqrt(252 / 3))}


def monthly_pair(ref: pd.Series, new: pd.Series, lo=None, hi=None) -> dict:
    df = pd.concat([ref, new], axis=1, join="inner").dropna().loc[lo:hi]
    x, y = np.log1p(df.iloc[:, 0]), np.log1p(df.iloc[:, 1])
    return {"start": df.index[0].date(), "end": df.index[-1].date(), "n": len(df),
            "corr": x.corr(y), "vol_ref": x.std() * np.sqrt(12), "vol_new": y.std() * np.sqrt(12),
            "vol_ratio": y.std() / x.std(), "te": (y - x).std() * np.sqrt(12),
            "mean_diff_ann": (y - x).mean() * 12}


def us_link(r: pd.Series, us: pd.Series, lo=None, hi=None) -> dict:
    """Synchronous vs lagged correlation with the US market and the 1d/3d/monthly vol."""
    df = _align(r, us, lo=lo, hi=hi)
    x, u = df.iloc[:, 0], df.iloc[:, 1]
    m = np.log1p((1 + r.loc[lo:hi]).groupby(pd.Grouper(freq="ME")).prod() - 1).iloc[1:-1]
    return {"start": df.index[0].date(), "end": df.index[-1].date(),
            "corr_us_1d": x.corr(u), "corr_us_3d": x.rolling(3).sum().corr(u.rolling(3).sum()),
            "corr_us_lag1": x.corr(u.shift(1)), "autocorr_1d": x.autocorr(1),
            "vol_1d": _vol(x), "vol_3d": _vol(x, 3), "vol_monthly": _vol(m, 1, 12),
            "var_ratio_monthly_to_1d": (_vol(m, 1, 12) / _vol(x)) ** 2}


def _monthly_from_daily(r: pd.Series) -> pd.Series:
    m = (1 + r).groupby(pd.Grouper(freq="ME")).prod() - 1
    return m.iloc[1:-1]


DIAG_FUNDS_PRE1986 = ["SCINX", "PRITX", "VWIGX", "VTRIX", "FOSFX", "TEMFX"]


def diagnostics(out: dict, refresh: bool = False) -> dict:
    us = yahoo("^GSPC", refresh)["close"].pct_change().iloc[1:].rename("us")
    eafe_d, em_d = msci_returns("990300", refresh=refresh), msci_returns("891800", refresh=refresh)
    eafe_m = msci_returns("990300", "END_OF_MONTH", refresh)
    em_m = msci_returns("891800", "END_OF_MONTH", refresh)
    s = {k: base_series(k, refresh) for k in SOURCE}
    s["aqr_gxus"] = s["aqr_gxus"].loc["1986":]      # 1984-85 is Canada only
    D_tables = {sym: etf_returns(sym, refresh)[1] for sym in ETF_PEERS}
    kfd_d, kfd_m = kf_dev_daily(refresh), kf_dev_monthly(refresh)
    y = {sym: yahoo_returns(sym, refresh) for sym in
         ["VTMGX", "VEA", "VGTSX", "VWO", "TEDMX"] + DIAG_FUNDS_PRE1986}
    D: dict = {"div_tables": D_tables}

    # --- splice overlaps (full overlap of the two sources)
    D["splice"] = pd.DataFrame({
        "intl_dev: KF EAFE-wt (C) -> AQR GxUS (B), monthly":
            monthly_pair(s["kf_intl_index"], _monthly_from_daily(s["aqr_gxus"])),
        "intl_dev: KF EAFE-wt -> AQR, monthly, 1986-1995 only":
            monthly_pair(s["kf_intl_index"], _monthly_from_daily(s["aqr_gxus"]), "1986", "1995"),
        "intl_dev: AQR GxUS (B) -> EFA (A), monthly":
            monthly_pair(_monthly_from_daily(s["aqr_gxus"]), _monthly_from_daily(s["EFA"])),
        "em: KF EM (C) -> VEIEX (B), monthly":
            monthly_pair(s["kf_em"], _monthly_from_daily(s["VEIEX"])),
        "em: KF EM -> VEIEX, monthly, 1994-06..2000":
            monthly_pair(s["kf_em"], _monthly_from_daily(s["VEIEX"]), "1994-06", "2000-12"),
        "em: VEIEX (B) -> EEM (A), monthly":
            monthly_pair(_monthly_from_daily(s["VEIEX"]), _monthly_from_daily(s["EEM"])),
    }).T
    D["splice_daily"] = pd.DataFrame({
        "intl_dev: AQR GxUS -> EFA (2001-09..)": daily_pair(s["aqr_gxus"], s["EFA"], "2001-09"),
        "intl_dev: AQR GxUS -> EFA, 2001-09..2003-12": daily_pair(s["aqr_gxus"], s["EFA"], "2001-09", "2003-12"),
        "em: VEIEX -> EEM (2003-05..)": daily_pair(s["VEIEX"], s["EEM"], "2003-05"),
        "em: VEIEX -> EEM, 2003-05..2005-12": daily_pair(s["VEIEX"], s["EEM"], "2003-05", "2005-12"),
    }).T

    # --- every candidate vs the MSCI index (2001+): monthly and daily
    D["vs_msci_monthly"] = pd.DataFrame({
        "intl_dev KF EAFE-wt": monthly_pair(eafe_m, s["kf_intl_index"]),
        "intl_dev AQR GxUS": monthly_pair(eafe_m, _monthly_from_daily(s["aqr_gxus"])),
        "intl_dev KF Dev ex US": monthly_pair(eafe_m, kfd_m),
        "intl_dev EFA": monthly_pair(eafe_m, _monthly_from_daily(s["EFA"])),
        "intl_dev VTMGX": monthly_pair(eafe_m, _monthly_from_daily(y["VTMGX"])),
        "intl_dev VEA": monthly_pair(eafe_m, _monthly_from_daily(y["VEA"])),
        "em KF EM": monthly_pair(em_m, s["kf_em"]),
        "em VEIEX": monthly_pair(em_m, _monthly_from_daily(s["VEIEX"])),
        "em EEM": monthly_pair(em_m, _monthly_from_daily(s["EEM"])),
        "em VWO": monthly_pair(em_m, _monthly_from_daily(y["VWO"])),
        "em TEDMX (active)": monthly_pair(em_m, _monthly_from_daily(y["TEDMX"])),
    }).T
    D["vs_msci_daily"] = pd.DataFrame({
        "intl_dev AQR GxUS": daily_pair(eafe_d, s["aqr_gxus"], "2001-01"),
        "intl_dev KF Dev ex US": daily_pair(eafe_d, kfd_d, "2001-01"),
        "intl_dev EFA": daily_pair(eafe_d, s["EFA"], "2001-09"),
        "intl_dev VTMGX": daily_pair(eafe_d, y["VTMGX"], "2001-01"),
        "em VEIEX": daily_pair(em_d, s["VEIEX"], "2001-01"),
        "em EEM": daily_pair(em_d, s["EEM"], "2003-05"),
        "em VWO": daily_pair(em_d, y["VWO"], "2005-04"),
    }).T

    # --- rejected long-history active funds, pre-1986, vs KF EAFE-weighted index
    rows = {}
    for sym in DIAG_FUNDS_PRE1986:
        if y[sym].index[0] < pd.Timestamp("1985-06-01"):
            rows[f"{sym} vs KF EAFE-wt, ..1985"] = monthly_pair(
                s["kf_intl_index"], _monthly_from_daily(y[sym]), None, "1985-12")
        rows[f"{sym} vs KF EAFE-wt, full"] = monthly_pair(s["kf_intl_index"], _monthly_from_daily(y[sym]))
    rows["TEDMX vs KF EM, full"] = monthly_pair(s["kf_em"], _monthly_from_daily(y["TEDMX"]))
    rows["KF Dev ex US vs AQR GxUS, monthly, 1990-07..2001-08"] = monthly_pair(
        _monthly_from_daily(s["aqr_gxus"]), kfd_m, "1990-07", "2001-08")
    D["rejected"] = pd.DataFrame(rows).T
    D["kf_vs_aqr_daily"] = pd.DataFrame({
        "KF Dev ex US vs AQR GxUS, 1990-07..2001-08": daily_pair(s["aqr_gxus"], kfd_d, "1990-07", "2001-08"),
        "VTMGX vs AQR GxUS, 1999-08..2001-08": daily_pair(s["aqr_gxus"], y["VTMGX"], "1999-08-18", "2001-08"),
    }).T

    # --- asynchronous-close diagnostics: each segment and reference index vs US
    a = {}
    a["intl_dev AQR GxUS [segment] 1986-01..2001-08"] = us_link(s["aqr_gxus"], us, "1986-01", "2001-08")
    a["intl_dev AQR GxUS (not used) 2001-09..2026-08"] = us_link(s["aqr_gxus"], us, "2001-09", "2026-08")
    a["intl_dev MSCI EAFE GR index 2001-09..2026-08"] = us_link(eafe_d, us, "2001-09", "2026-08")
    a["intl_dev EFA [segment] 2001-09..2026-08"] = us_link(s["EFA"], us, "2001-09", "2026-08")
    a["intl_dev VTMGX fund NAV 2001-09..2026-08"] = us_link(y["VTMGX"], us, "2001-09", "2026-08")
    a["em VEIEX [segment] 1994-06..2003-04"] = us_link(s["VEIEX"], us, "1994-06", "2003-04")
    a["em VEIEX (not used) 2003-05..2026-10"] = us_link(s["VEIEX"], us, "2003-05", "2026-10")
    a["em MSCI EM GR index 2003-05..2026-10"] = us_link(em_d, us, "2003-05", "2026-10")
    a["em EEM [segment] 2003-05..2026-10"] = us_link(s["EEM"], us, "2003-05", "2026-10")
    D["async"] = pd.DataFrame(a).T

    # --- year-by-year synchronous / lagged US correlation (fair-value drift)
    yy = {}
    for name, r in [("AQR_GxUS", s["aqr_gxus"]), ("MSCI_EAFE", eafe_d), ("EFA", s["EFA"]),
                    ("VEIEX", s["VEIEX"]), ("MSCI_EM", em_d), ("EEM", s["EEM"])]:
        df = _align(r, us)
        df.columns = ["x", "u"]
        df["ul"] = df["u"].shift(1)
        g = df.groupby(df.index.year)
        yy[(name, "sync")] = g.apply(lambda d: d.x.corr(d.u))
        yy[(name, "lagUS")] = g.apply(lambda d: d.x.corr(d.ul))
    D["by_year"] = pd.DataFrame(yy).rename_axis("year")

    # --- vol by decade, episodes, outliers on the FINAL series
    dec, epi, outl = {}, {}, {}
    for asset, (daily, monthly) in out.items():
        lm = np.log1p(monthly["ret"])
        ld = np.log1p(daily["ret"])
        decade = lambda idx: (idx.year // 10) * 10  # noqa: E731
        dec[(asset, "monthly")] = lm.groupby(decade(lm.index)).apply(lambda x: _vol(x, 1, 12))
        dec[(asset, "daily_1d")] = ld.groupby(decade(ld.index)).apply(lambda x: _vol(x))
        dec[(asset, "daily_3d")] = ld.groupby(decade(ld.index)).apply(lambda x: _vol(x, 3))
        for lab, lo, hi in [("1987-10", "1987-10", "1987-10"), ("1990-01..09", "1990-01", "1990-09"),
                            ("1997-07..1998-08", "1997-07", "1998-08"), ("1998-08", "1998-08", "1998-08"),
                            ("2008-09..2009-02", "2008-09", "2009-02"), ("2008-10", "2008-10", "2008-10"),
                            ("2020-03", "2020-03", "2020-03"), ("2022", "2022-01", "2022-12")]:
            x = monthly["ret"].loc[lo:hi]
            epi[(asset, lab)] = float(np.prod(1 + x) - 1) if len(x) else np.nan
        big = daily.loc[daily["ret"].abs() > 0.08, ["ret", "source"]]
        outl[asset] = big
    D["decade"] = pd.DataFrame(dec).rename_axis("decade")
    D["episodes"] = pd.Series(epi).unstack(0)
    D["outliers"] = outl
    D["us_ref"] = "^GSPC price (Yahoo), US close"
    return D


# --------------------------------------------------------------- segments
NOTES = {
    "kf_intl_index": ("Ken French F-F_International_Indices.zip, Ind_all.Dat, first block "
                      "(value-weight dollar returns, all firms), Mkt column. Countries at EAFE "
                      "weights; MSCI raw data 1975-2006, Bloomberg after. Excludes Canada."),
    "aqr_gxus": ("AQR Betting-Against-Beta daily workbook, MKT sheet 'Global Ex USA' (excess) + "
                 "RF sheet. Cap-weighted developed ex-US incl. Canada (EFA excludes Canada). "
                 "Index convention: foreign closes. Weekdays with |r|<0.001% (all closed) dropped."),
    "EFA": ("iShares MSCI EAFE ETF, Yahoo close + distributions, each validated by the ex-date price test "
            "against VEA/VTMGX/S&P 500 (all passed). US-close prices."),
    "kf_em": ("Ken French Emerging_5_Factors_CSV.zip, Mkt-RF + RF (RF = US 1m T-bill), "
              "Bloomberg-based value-weighted EM market."),
    "VEIEX": ("Vanguard Emerging Markets Stock Index Inv (tracked MSCI Select EM Free in this era), "
              "Yahoo adjclose. Open-end NAV (tier B): foreign closes, with growing fair-value pricing after 2001. "
              "Bad NAV print 2001-07-16 (+5.6% then -5.9%, EM peers flat) dropped: the 07-17 return spans it."),
    "EEM": ("iShares MSCI Emerging Markets ETF, Yahoo close + distributions, each validated by the ex-date price "
            "test against VWO/IEMG/S&P 500 (all passed). US-close prices."),
}


def _fmt(d: dict, keys) -> str:
    return ", ".join(f"{k}={d[k]:.3f}" for k in keys)


def segments(out: dict, D: dict | None = None) -> pd.DataFrame:
    """Provenance rows in the segments.csv schema (asset, start, end, freq, source, tier, notes)."""
    key_of = {v: k for k, v in SOURCE.items()}
    over = {}
    if D is not None:
        sp, sd = D["splice"], D["splice_daily"]
        mk = ["corr", "vol_ratio", "te"]
        dk = ["corr_1d", "corr_3d", "vol_ratio_1d", "vol_ratio_3d"]
        over = {
            "aqr_gxus": "Overlap vs KF EAFE-wt monthly 1986-2025: " + _fmt(sp.iloc[0], mk) + ".",
            "EFA": ("Overlap vs AQR GxUS: monthly " + _fmt(sp.iloc[2], mk) + "; daily "
                    + _fmt(sd.iloc[0], dk) + " (async closes)."),
            "VEIEX": ("Overlap vs KF EM monthly 1994-06..2000: " + _fmt(sp.iloc[4], mk)
                      + "; full overlap " + _fmt(sp.iloc[3], mk) + "."),
            "EEM": ("Overlap vs VEIEX: monthly " + _fmt(sp.iloc[5], mk) + "; daily "
                    + _fmt(sd.iloc[2], dk) + "."),
        }
    rows = []
    for asset, (daily, monthly) in out.items():
        k, tier, *_ = PLAN[asset]["monthly"]
        m = monthly[monthly["source"] == SOURCE[k]]
        rows.append((asset, m.index.min().date(), m.index.max().date(), "M", SOURCE[k], tier,
                     NOTES[k]))
        for (src, tier), g in daily.groupby(["source", "tier"], sort=False):
            k = key_of[src]
            rows.append((asset, g.index.min().date(), g.index.max().date(), "D", src, tier,
                         (NOTES[k] + " " + over.get(k, "")).strip()))
    return pd.DataFrame(rows, columns=["asset", "start", "end", "freq", "source", "tier", "notes"])


def index_chain(asset: str, refresh: bool = False) -> pd.Series:
    """ALTERNATIVE daily series in a single FOREIGN-CLOSE (index) convention, not
    written to disk: intl_dev = AQR GxUS (1986..2000) + MSCI EAFE GR (2001..);
    em = MSCI EM GR (2001..). For sensitivity checks of the async-close effect."""
    if asset == "intl_dev":
        a = aqr_total("Global Ex USA", refresh).loc["1986":"2000-12-31"]
        return pd.concat([a, msci_returns("990300", refresh=refresh).loc["2001-01-01":]])
    return msci_returns("891800", refresh=refresh).loc["2001-01-01":]


# ----------------------------------------------------------------- report
def _md(df: pd.DataFrame, digits: int = 3, index: bool = True) -> str:
    df = df.copy()
    if index:
        df = df.reset_index()
    def f(v):
        if isinstance(v, (int, np.integer)):
            return str(v)
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    cols = [str(c) if not isinstance(c, tuple) else " ".join(map(str, c)) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f(v) for v in r) + " |")
    return "\n".join(lines)


def write_report(out: dict, D: dict, seg: pd.DataFrame, path: Path | None = None) -> Path:
    from ...paths import REPORTS
    path = path or REPORTS / "data" / "intl_equity.md"
    A, sp, sd = D["async"], D["splice"], D["splice_daily"]
    vm, vd = D["vs_msci_monthly"], D["vs_msci_daily"]
    res = []
    for asset, (d, m) in out.items():
        res.append(f"* **{asset}** — daily {d.index[0].date()} .. {d.index[-1].date()} "
                   f"({len(d)} obs); monthly {m.index[0].date()} .. {m.index[-1].date()} "
                   f"({len(m)} months; daily era compounded; current incomplete month excluded).")
    s = [
        "# intl_equity — international developed (`intl_dev`) and emerging (`em`) equity",
        "",
        f"Built by `python -m assetforecast.data.sources.intl_equity` on {pd.Timestamp.today().date()}. "
        "USD, unhedged, SIMPLE TOTAL returns in decimals (not excess; cash is another track).",
        "",
        "## Result", "", *res, "",
        "## Sources", "",
        "Used:",
        "* **Ken French data library** — `F-F_International_Indices.zip` (`Ind_all.Dat`: EAFE-country-weighted, "
        "value-weighted within country, USD, with dividends, monthly 1975-01..2025-12; MSCI raw data "
        "1975-2006, Bloomberg after) and `Emerging_5_Factors_CSV.zip` (EM Mkt-RF + RF, monthly 1989-07..).",
        "* **AQR** `Betting-Against-Beta-Equity-Factors-Daily.xlsx` — sheet MKT column `Global Ex USA` "
        "(excess return) + sheet RF. Daily from 1984, but until 1985-10 it is Canada only and until "
        "1985-12 Canada + Australia; Europe and Japan enter 1986-01, so it is used from 1986-01. Runs to 2026-08.",
        "* **Yahoo chart API**: EFA and EEM as close + distributions, every distribution validated against "
        "the ex-date price move relative to peer funds (`distributions.validate_distributions`: "
        + "; ".join(f"{s} {len(t)} tested, {int(t['action'].str.startswith('replaced').sum())} changed"
                    for s, t in D.get("div_tables", {}).items()) + "); VEIEX adjclose (open-end fund NAV, tier B), "
        f"bad print(s) dropped: {BAD_PRICES['VEIEX']}.",
        "",
        "Diagnostics only (not in the files):",
        "* **MSCI** public graph endpoint `app2.msci.com/.../getLevelDataForGraph` — works, but refuses "
        "start dates before 1997-01-01 and serves gross/net TOTAL return levels only from 2000-12-29 "
        "(price index from 1997-01). EAFE 990300 and EM 891800 GRTR daily and month-end, to 2026-11-13. "
        "Used as the reference index for every overlap test. MSCI EAFE from 1970 / EM from 1988 is NOT "
        "reachable here.",
        "* Ken French `Developed_ex_US_3_Factors(_Daily)` (from 1990-07) — tracks MSCI EAFE slightly better "
        "monthly than AQR but worse daily (its daily FX timing makes it more synchronous with the US); "
        "AQR also starts 4.5 years earlier. Not used.",
        "* Yahoo: VTMGX, VEA, VGTSX, VWO, TEDMX and long-lived active international funds SCINX (1980), "
        "PRITX (1980), VWIGX (1981), VTRIX (1983), FOSFX (1984), TEMFX (1986). See *Rejected candidates*.",
        "* `^GSPC` (Yahoo, price) as the US-close reference for correlation diagnostics.",
        "",
        "## Segments", "",
        _md(seg[["asset", "start", "end", "freq", "tier", "source", "notes"]], index=False),
        "",
        "Splices are on calendar-month boundaries, so every month in the monthly files is single-source "
        "and monthly == compounded daily exactly. Price conventions by segment:",
        "",
        _md(pd.DataFrame({"segment source": list(CONVENTION), "price-time convention": list(CONVENTION.values())}),
            index=False),
        "",
        "## Splice / overlap diagnostics", "",
        "Monthly log returns on the full overlap of the two sources (`vol_ratio` = new / ref, `te` = "
        "annualized tracking error, `mean_diff_ann` = new − ref):", "",
        _md(sp, 3),
        "",
        "Daily log returns on the overlap. `corr_3d`/`vol_ratio_3d` use overlapping 3-day sums "
        "(the ARCHITECTURE correlation convention):", "",
        _md(sd, 3),
        "",
        "Reading: the monthly overlaps show no level or vol jump at the splices (vol ratios 0.98-1.04 "
        "outside the noisy 1994-2000 EM window). The daily overlaps are dominated by the "
        "asynchronous-close effect (next section), not by index differences.",
        "",
        "### Every candidate vs the MSCI index (2001+)", "",
        "Monthly (reference = MSCI EAFE GR or MSCI EM GR, month-ends aligned):", "",
        _md(vm, 3), "",
        "Daily (same reference; MSCI weekday holidays with zero change removed):", "",
        _md(vd, 3), "",
        "### Rejected candidates", "",
        "Old actively managed international funds would extend DAILY history to 1980, but they are poor "
        "index proxies: monthly correlation with the EAFE-weighted index is 0.54-0.93 before 1986 and "
        "tracking error 6-10 %/yr (an index fund on the same data shows 1.5-3 %). They were not used; "
        "1975-1985 is monthly tier C instead. TEDMX (Templeton Developing Markets, active, 1991) likewise. "
        "VTMGX (EAFE index fund from 1999-08) would add two years of fund-NAV data (tier B) before EFA, at the cost of "
        "a third price convention (partially fair-valued NAV); not used.", "",
        _md(D["rejected"], 3), "",
        _md(D["kf_vs_aqr_daily"], 3), "",
        "## Asynchronous closes", "",
        "Correlation with the US market (^GSPC, US close) on the same day (`corr_us_1d`), with overlapping "
        "3-day sums (`corr_us_3d`) and with the PREVIOUS US day (`corr_us_lag1`); own first-order "
        "autocorrelation; annualized vol from 1-day, overlapping 3-day and monthly returns; and the "
        "monthly-to-daily variance ratio (vol_monthly / vol_1d)^2:", "",
        _md(A, 3), "",
        "What this shows:",
        f"* **Index / foreign-close data** (AQR, MSCI, VEIEX before ~2001) react to the US day one day "
        f"late: lag-1 US correlation {A.iloc[0]['corr_us_lag1']:.2f} (AQR 1986-2001), "
        f"{A.iloc[2]['corr_us_lag1']:.2f} (MSCI EAFE 2001+), {A.iloc[7]['corr_us_lag1']:.2f} (MSCI EM). "
        f"Same-day correlation with the US is badly understated (EAFE {A.iloc[2]['corr_us_1d']:.2f} vs "
        f"{A.iloc[2]['corr_us_3d']:.2f} with 3-day sums). Daily returns are positively autocorrelated "
        "(staggered closes and stale prices), so the 1-day vol UNDERSTATES monthly vol: variance ratio "
        f"{A.iloc[0]['var_ratio_monthly_to_1d']:.2f} for the AQR segment 1986-2001, "
        f"{A.iloc[5]['var_ratio_monthly_to_1d']:.2f} for the VEIEX segment 1994-2003.",
        f"* **US-listed ETFs** (EFA, EEM) price at the US close: same-day US correlation is high "
        f"({A.iloc[3]['corr_us_1d']:.2f}, {A.iloc[8]['corr_us_1d']:.2f}) and 3-day correlation barely "
        f"changes it. But they OVERREACT to the US day and partially revert the next day (lag-1 US "
        f"correlation {A.iloc[3]['corr_us_lag1']:.2f} / {A.iloc[8]['corr_us_lag1']:.2f}, own autocorrelation "
        f"{A.iloc[3]['autocorr_1d']:.2f} / {A.iloc[8]['autocorr_1d']:.2f}), so 1-day vol OVERSTATES monthly "
        f"vol: EFA {A.iloc[3]['vol_1d']:.3f} vs {A.iloc[3]['vol_monthly']:.3f} monthly (variance ratio "
        f"{A.iloc[3]['var_ratio_monthly_to_1d']:.2f}); EEM {A.iloc[8]['vol_1d']:.3f} vs "
        f"{A.iloc[8]['vol_monthly']:.3f} (ratio {A.iloc[8]['var_ratio_monthly_to_1d']:.2f}). On the same "
        f"dates the MSCI index has 1-day vol {A.iloc[2]['vol_1d']:.3f} (EAFE) and {A.iloc[7]['vol_1d']:.3f} (EM): "
        f"EFA's daily vol is {vd.loc['intl_dev EFA','vol_ratio_1d']:.2f}x the index's and EEM's "
        f"{vd.loc['em EEM','vol_ratio_1d']:.2f}x, while MONTHLY vols agree within 2 %.",
        "* **Open-end fund NAVs** (VTMGX, VEIEX) sit in between and drift over time: Vanguard's fair-value "
        "pricing moved them from foreign-close behaviour (lag-1 US corr ≈ 0.3-0.45 in the 1990s) to "
        "near-US-close behaviour by 2007 (see year table).",
        "* **Consequence for the model.** The daily-RV level of `intl_dev` and `em` has a structural break "
        "at the ETF splice (2001-09 and 2003-05) that is NOT a change in risk: the monthly-to-daily variance "
        "ratio drops from ~1.5-1.8 (index/NAV era) to ~0.6-0.65 (ETF era). Models trained on daily RV "
        "across the splice, and the RV→annual bridge, should treat `source`/`tier` (or the splice date) as a "
        "regime, or use the 3-day / monthly measures. 3-day overlapping sums remove most of the "
        "correlation bias for index data (EAFE–US 0.50 → 0.73) but not all of it, and do not fully remove "
        "the ETF vol excess (EFA 3-day vol still 1.07x the index).",
        "* `index_chain()` in the module returns a single-convention foreign-close alternative "
        "(AQR → MSCI EAFE GR; MSCI EM GR from 2001) for sensitivity checks; it is not written to disk.",
        "",
        "Same-day (`sync`) and lagged (`lagUS`) correlation with ^GSPC by calendar year:", "",
        _md(D["by_year"].round(2), 2), "",
        "## Annualized volatility by decade", "",
        "Uncentered, log returns: `monthly` from the monthly file (√12), `daily_1d` from daily returns (√252), "
        "`daily_3d` from overlapping 3-day sums. The 2000s daily row mixes conventions (index to 2001-08 / "
        "VEIEX to 2003-04, ETF after).", "",
        _md(D["decade"], 3), "",
        "## Known episodes (compounded from the monthly files)", "",
        _md(D["episodes"], 3), "",
        "1987-10 (intl_dev −12.5 %, daily low −13.5 % on 1987-10-20, the day after the US crash: foreign-close "
        "convention), 1990 Japan crash (−31 % Jan-Sep 1990), the 1997-98 Asia/Russia crisis for EM (−53 % "
        "1997-07..1998-08, −26 % in 1998-08), 2008 and 2020-03 all look right.", "",
        "Daily returns beyond ±8 %:", "",
    ]
    for asset, big in D["outliers"].items():
        b = big.copy()
        b.index = b.index.date
        s += [f"*{asset}*", "", _md(b, 4), ""]
    s += [
        "## Caveats", "",
        "* `intl_dev` 1986-2001 (AQR) includes Canada (~5-8 % weight; EFA excludes it, VEA includes it). "
        "Canada trades in US hours, so this segment is slightly more synchronous with the US than EAFE.",
        "* The 1975-1985 segment (KF, tier C) uses EAFE country weights with Ken French's own value-weighted "
        "country portfolios, not MSCI's constituents; its monthly tracking error vs MSCI EAFE is 1.5 %/yr "
        "on 2001-2025. MSCI EAFE 1970-1974 was not reachable.",
        "* EM monthly 1989-07..1994-05 (KF) is the only pre-VEIEX EM source found; KF EM vs MSCI EM price "
        "index 1997-2000: corr 0.99. MSCI EM 1988 monthly was not reachable. No daily EM source before 1994.",
        "* VEIEX 1994-2000 tracked MSCI EM less tightly than later (monthly corr 0.95 and TE 8 %/yr vs KF EM, "
        "0.97 vs the MSCI EM price index 1997-2000), partly because it tracked 'Select EM Free' (country "
        "subset) and partly NAV timing. VEIEX carried purchase/redemption fees in the 1990s that are not in "
        "the NAV return.",
        "* After 9/11 the US-close series have no prices 2001-09-11..14; the 2001-09-17 return spans five "
        "foreign trading days (VEIEX −10.4 %).",
        "* Yahoo `adjclose` is a vendor series and can be revised; raw JSON is cached in `data/raw/intl_equity/`.",
        "* Daily files keep each source's own calendar: the AQR segment includes non-US trading days "
        "(e.g. US Thanksgiving) when foreign markets traded; the assembler should compound or drop such "
        "days consistently with cash.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(s) + "\n")
    return path


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Build intl_dev / em total-return files")
    ap.add_argument("--refresh", action="store_true", help="re-download all raw files")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args(argv)
    out = build(refresh=a.refresh)
    D = diagnostics(out, refresh=False) if not a.no_report else None
    seg = segments(out, D)
    if D is not None:
        print("report:", write_report(out, D, seg))
    pd.set_option("display.width", 200)
    print(seg.drop(columns="notes").to_string(index=False))


if __name__ == "__main__":
    main()
