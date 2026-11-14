"""Raw-source loaders for the asset-return panel.

Every function returns *simple total returns* (decimals) indexed by calendar
month-end (monthly loaders) or by observation date (daily loaders). Everything
is computed from end-of-period levels; no source used here is a monthly average.

Downloads go through `fetch.fetch` / `fetch.fred` so the manifest records
provenance. Yahoo Finance histories (used only for mutual-fund NAV series that
have no free alternative, ETF validation and recent-month extension) are cached
as CSV under data/raw/yahoo/ and recorded in the same manifest by `_record`.
"""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import time
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import config
from .fetch import MANIFEST, _load_manifest, fetch, fred

FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
MSCI = ("https://app2.msci.com/products/service/index/indexmaster/getLevelDataForGraph?"
        "currency_symbol=USD&index_variant={variant}&start_date={start}&end_date={end}"
        "&data_frequency={freq}&index_codes={code}")
MSCI_CODES = {"world_ex_usa": "991000", "eafe": "990300", "em": "891800"}
NAREIT = "https://www.reit.com/sites/default/files/returns/MonthlyHistoricalReturns.xls"
AQR = "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
GSW_NOMINAL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
GSW_TIPS = "https://www.federalreserve.gov/data/yield-curve-tables/feds200805.csv"
LBMA_GOLD_PM = "https://prices.lbma.org.uk/json/gold_pm.json"


def month_end(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx).to_period("M").to_timestamp(how="end").normalize()


def _monthly_from_levels(levels: pd.Series) -> pd.Series:
    """Month-end-to-month-end simple return from a level series (last obs in each month)."""
    lv = levels.dropna().groupby(levels.dropna().index.to_period("M")).last()
    r = lv.pct_change().iloc[1:]
    r.index = r.index.to_timestamp(how="end").normalize()
    return r


def compound_monthly(daily: pd.Series) -> pd.Series:
    """Compound daily simple returns into calendar-month returns (empty months -> NaN)."""
    d = daily.dropna()
    g = np.log1p(d).groupby(d.index.to_period("M"))
    out = np.expm1(g.sum())
    out.index = out.index.to_timestamp(how="end").normalize()
    return out


# ---------------------------------------------------------------- Ken French
def _french_zip(name: str, refresh: bool = False) -> str:
    p = fetch(FRENCH + name, f"french/{name}", refresh=refresh)
    with zipfile.ZipFile(p) as z:
        member = [m for m in z.namelist() if not m.endswith("/")][0]
        return z.read(member).decode("latin-1")


def _french_table(text: str, daily: bool) -> pd.DataFrame:
    """First data block of a French CSV (monthly or daily), percent -> decimal."""
    lines = text.replace("\r", "").split("\n")
    start = next(i for i, l in enumerate(lines) if l.replace(" ", "").startswith(",Mkt-RF"))
    cols = [c.strip() for c in lines[start].split(",")[1:]]
    rows = []
    width = 8 if daily else 6
    for l in lines[start + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != width:
            break
        rows.append([parts[0]] + [float(x) for x in parts[1:]])
    df = pd.DataFrame(rows, columns=["date"] + cols)
    fmt = "%Y%m%d" if daily else "%Y%m"
    df.index = pd.to_datetime(df.pop("date"), format=fmt)
    if not daily:
        df.index = month_end(df.index)
    df = df.replace(-99.99, np.nan) / 100.0
    return df


def french_us(daily: bool = False) -> pd.DataFrame:
    """CRSP value-weighted US market total return and the 1-month T-bill (RF)."""
    name = "F-F_Research_Data_Factors_daily_CSV.zip" if daily else "F-F_Research_Data_Factors_CSV.zip"
    t = _french_table(_french_zip(name), daily)
    return pd.DataFrame({"mkt": t["Mkt-RF"] + t["RF"], "rf": t["RF"]})


def french_region(region: str, daily: bool = False) -> pd.Series:
    """Fama/French regional market return in USD (Mkt-RF + RF), total return."""
    names = {
        ("dev_ex_us", False): "Developed_ex_US_3_Factors_CSV.zip",
        ("dev_ex_us", True): "Developed_ex_US_3_Factors_Daily_CSV.zip",
        ("emerging", False): "Emerging_5_Factors_CSV.zip",
    }
    t = _french_table(_french_zip(names[(region, daily)]), daily)
    return (t["Mkt-RF"] + t["RF"]).rename(region)


def french_eafe_1975() -> pd.Series:
    """Fama/French international index (EAFE-weighted countries), USD, with dividends.

    `Ind_all.Dat` in F-F_International_Indices.zip, first block: value-weight dollar
    returns, "Mkt" column. Monthly from 1975-01.
    """
    p = fetch(FRENCH + "F-F_International_Indices.zip", "french/F-F_International_Indices.zip")
    with zipfile.ZipFile(p) as z:
        text = z.read("Ind_all.Dat").decode("latin-1")
    lines = text.replace("\r", "").split("\n")
    h = next(i for i, l in enumerate(lines) if l.strip())
    assert "Value-Weight Dollar Returns" in lines[h] and "Not Reqd" in lines[h], lines[h]
    assert lines[h + 2].split()[0] == "Mkt", lines[h + 2]
    rows = []
    for l in lines[h + 3:]:
        parts = l.split()
        if not parts or not (parts[0].isdigit() and len(parts[0]) == 6):
            break
        rows.append((parts[0], float(parts[1])))
    s = pd.Series({pd.Timestamp(d[:4] + "-" + d[4:] + "-01"): v for d, v in rows}, name="ff_eafe")
    s.index = month_end(s.index)
    return s.replace(-99.99, np.nan) / 100.0


# ---------------------------------------------------------------- MSCI
def msci(index: str, daily: bool = False, end: str | None = None) -> pd.Series:
    """MSCI gross total return index (USD) levels -> returns. Available from 2000-12-29."""
    code = MSCI_CODES[index]
    end = end or pd.Timestamp.today().strftime("%Y%m%d")
    freq = "DAILY" if daily else "END_OF_MONTH"
    url = MSCI.format(variant="GRTR", start="19970101", end=end, freq=freq, code=code)
    p = fetch(url, f"msci/{code}_GRTR_{freq}.json")
    lv = json.loads(p.read_text())["indexes"]["INDEX_LEVELS"]
    s = pd.Series({pd.Timestamp(str(x["calc_date"])): x["level_eod"] for x in lv}).sort_index()
    if daily:
        s = s[s.index.dayofweek < 5]
        return s.pct_change().iloc[1:].rename(f"msci_{index}")
    return _monthly_from_levels(s).rename(f"msci_{index}")


# ---------------------------------------------------------------- NAREIT
def nareit(column: str = "All Equity REITs") -> pd.Series:
    """FTSE Nareit monthly total return (month-end index values), from 1972-01."""
    p = fetch(NAREIT, "nareit/MonthlyHistoricalReturns.xls")
    raw = pd.read_excel(p, sheet_name="Index Data", header=None)
    hdr = raw.iloc[5].ffill()
    # columns: [name] Total Return, Index, Price Return, Index, Income Return, Dividend Yield
    first = [i for i, v in enumerate(raw.iloc[5]) if v == column][0]
    assert str(raw.iloc[6, first]).strip() == "Total" and str(raw.iloc[7, first + 1]).strip() == "Index"
    idx = pd.to_datetime(raw.iloc[9:, 0], errors="coerce")
    level = pd.to_numeric(raw.iloc[9:, first + 1], errors="coerce")
    s = pd.Series(level.values, index=idx).dropna()
    s = s[s.index.notna()]
    del hdr
    return _monthly_from_levels(s).rename("nareit")


# ---------------------------------------------------------------- AQR
def aqr_commodities() -> pd.Series:
    """AQR 'Commodities for the Long Run': excess return of the equal-weight
    commodity futures portfolio (over T-bills), month-end, from 1877."""
    p = fetch(AQR + "Commodities-for-the-Long-Run-Index-Level-Data-Monthly.xlsx",
              "aqr/Commodities-for-the-Long-Run-Index-Level-Data-Monthly.xlsx")
    raw = pd.read_excel(p, sheet_name=0, header=None)
    hrow = next(i for i in range(30) if str(raw.iloc[i, 1]).startswith("Excess return of equal-weight"))
    d = raw.iloc[hrow + 1:, [0, 1]].dropna()
    s = pd.Series(pd.to_numeric(d[1]).values, index=month_end(pd.to_datetime(d[0])), name="aqr_ew_xs")
    return s


def aqr_credit_xs() -> pd.Series:
    """AQR / Asvanunt-Richardson (2015): IG corporate return in excess of
    duration-matched Treasuries (CORP_XS), monthly 1926-2014."""
    p = fetch(AQR + "Credit-Risk-Premium-Preliminary-Paper-Data.xlsx",
              "aqr/Credit-Risk-Premium-Preliminary-Paper-Data.xlsx")
    raw = pd.read_excel(p, sheet_name=0, header=None)
    hrow = next(i for i in range(30) if str(raw.iloc[i, 0]).strip() == "Date")
    assert str(raw.iloc[hrow, 1]).strip() == "CORP_XS"
    d = raw.iloc[hrow + 1:, [0, 1]].dropna()
    return pd.Series(pd.to_numeric(d[1]).values, index=month_end(pd.to_datetime(d[0])), name="corp_xs")


# ---------------------------------------------------------------- Fed curves
def _gsw(url: str, name: str) -> pd.DataFrame:
    p = fetch(url, name)
    text = p.read_text()
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Date,"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), index_col=0, parse_dates=True, na_values=["NA", ""])
    return df[df["BETA0"].notna()].sort_index()


def gsw_nominal() -> pd.DataFrame:
    return _gsw(GSW_NOMINAL, "fed/feds200628.csv")


def gsw_tips() -> pd.DataFrame:
    return _gsw(GSW_TIPS, "fed/feds200805.csv")


def fred_series(series_id: str) -> pd.Series:
    p = fred(series_id)
    df = pd.read_csv(p, index_col=0, parse_dates=True, na_values=["."])
    return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().rename(series_id)


# ---------------------------------------------------------------- Cash extension
def tbill_4wk_monthly() -> pd.Series:
    """Holding-period return over month m of a 4-week bill bought at the last
    observation of month m-1 (FRED DTB4WK, secondary-market discount rate),
    rolled at the same yield for the days in the month."""
    d = fred_series("DTB4WK")
    last = d.groupby(d.index.to_period("M")).last()
    dates = d.index.to_series().groupby(d.index.to_period("M")).last()
    out = {}
    for i in range(1, len(last)):
        prev_rate = last.iloc[i - 1]
        days = (dates.iloc[i] - dates.iloc[i - 1]).days
        price = 1 - prev_rate / 100 * 28 / 360
        out[last.index[i].to_timestamp(how="end").normalize()] = (1 / price) ** (days / 28) - 1
    # the next month after the last observed month can also be priced (rate is known)
    p_last = last.index[-1]
    nxt = (p_last + 1).to_timestamp(how="end").normalize()
    price = 1 - last.iloc[-1] / 100 * 28 / 360
    out[nxt] = (1 / price) ** (((p_last + 1).days_in_month) / 28) - 1
    return pd.Series(out, name="dtb4wk")


# ---------------------------------------------------------------- Yahoo
def _record(name: str, url: str, content: bytes) -> None:
    with open(MANIFEST.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = _load_manifest()
        manifest[name] = {
            "url": url,
            "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def yahoo(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Daily OHLC, dividends and capital-gain distributions from Yahoo (cached)."""
    name = f"yahoo/{ticker.replace('^', '_')}.csv"
    path = config.RAW / name
    if not path.exists() or refresh:
        import yfinance as yf

        df = None
        for attempt in range(5):
            try:
                df = yf.Ticker(ticker).history(period="max", auto_adjust=False, actions=True)
                if len(df):
                    break
            except Exception:  # noqa: BLE001 - rate limits surface as varied errors
                pass
            time.sleep(5 * (attempt + 1))
        if df is None or not len(df):
            raise RuntimeError(f"yfinance returned no data for {ticker}")
        df.index = df.index.tz_localize(None).normalize()
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = df.to_csv().encode()
        path.write_bytes(buf)
        _record(name, f"yfinance:{ticker} history(period='max', auto_adjust=False, actions=True)", buf)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[~df.index.duplicated(keep="last")].sort_index()


def yahoo_daily_tr(ticker: str) -> pd.Series:
    """Daily total return from unadjusted close plus cash distributions
    (dividends and capital gains reinvested at the ex-date close)."""
    d = yahoo(ticker)
    d = d[d["Close"].notna() & (d["Close"] > 0)]
    dist = d.get("Dividends", 0).fillna(0)
    if "Capital Gains" in d:
        dist = dist + d["Capital Gains"].fillna(0)
    r = (d["Close"] + dist) / d["Close"].shift() - 1
    return r.iloc[1:].rename(ticker)


def yahoo_monthly_tr(ticker: str) -> pd.Series:
    """Calendar-month total return; the partial first month is dropped."""
    r = yahoo_daily_tr(ticker)
    m = compound_monthly(r)
    return m.iloc[1:].rename(ticker)


# ---------------------------------------------------------------- LBMA gold
def lbma_gold_pm() -> pd.Series | None:
    """LBMA Gold Price PM (USD) daily, or None if the LBMA endpoint refuses us.

    The JSON is a list of {"d": "YYYY-MM-DD", "v": [USD, GBP, EUR]}. If you can
    obtain it another way, drop it at data/raw/lbma/gold_pm.json and rebuild.
    """
    try:
        p = fetch(LBMA_GOLD_PM, "lbma/gold_pm.json", retries=1)
    except RuntimeError:
        return None
    rows = json.loads(p.read_text())
    s = pd.Series({pd.Timestamp(r["d"]): r["v"][0] for r in rows if r.get("v") and r["v"][0]})
    return s.sort_index().rename("lbma_pm")
