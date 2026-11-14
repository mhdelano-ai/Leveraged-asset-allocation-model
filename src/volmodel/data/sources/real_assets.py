"""Gold and broad-commodity total-return histories (track ``real_assets``).

Run ``python -m volmodel.data.sources.real_assets`` to download (cached under
``data/raw/real_assets/``), splice and write

    data/processed/assets/gold_{daily,monthly}.parquet
    data/processed/assets/commodities_{daily,monthly}.parquet
    reports/data/real_assets.md

Every output row carries ``ret`` (simple total return), ``source`` and ``tier``.
Commodity rows built from an excess-return (futures-only) source also carry
``ret_excess_native`` (the native excess return) and ``cash_ret`` (the cash
return that was added: Ken French / Ibbotson 1-month T-bill, ``RF``), so the
total return can be re-derived with the project's cash series later:
``ret = (1 + ret_excess_native) * (1 + cash_ret) - 1``.

See reports/data/real_assets.md for sources, segments and diagnostics.
"""
from __future__ import annotations

import io
import json
import random
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ...paths import ASSET_DIR, RAW, REPORTS

RAW_DIR = RAW / "real_assets"
REPORT_PATH = REPORTS / "data" / "real_assets.md"

UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "*/*",
}

# --------------------------------------------------------------------------- #
# HTTP with cache
# --------------------------------------------------------------------------- #


def _fetch(url: str, path: Path, *, refresh: bool = False, tries: int = 8,
           ok=(200,), pause: float = 1.0) -> Path:
    """Download ``url`` to ``path`` unless cached. Backs off on 429/5xx."""
    if path.exists() and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=120)
        except requests.RequestException as e:  # network hiccup
            last = repr(e)
            time.sleep(5 * (k + 1))
            continue
        if r.status_code in ok:
            path.write_bytes(r.content)
            time.sleep(pause)
            return path
        last = f"HTTP {r.status_code}"
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(15 * (k + 1), 120) + random.random() * 5)
            continue
        break
    raise RuntimeError(f"download failed ({last}): {url}")


# ---------------------------------------------------------------- Yahoo
YAHOO_URL = ("https://{host}.finance.yahoo.com/v8/finance/chart/{sym}"
             "?period1=-2208988800&period2={p2}&interval=1d&events=div%2Csplits")


def yahoo_path(sym: str) -> Path:
    return RAW_DIR / "yahoo" / (sym.replace("^", "_").replace("=", "_") + ".json")


def fetch_yahoo(sym: str, refresh: bool = False) -> Path:
    p = yahoo_path(sym)
    if p.exists() and not refresh:
        return p
    p2 = int(time.time()) + 86400
    for k, host in enumerate(["query1", "query2"] * 4):
        try:
            return _fetch(YAHOO_URL.format(host=host, sym=sym, p2=p2), p,
                          refresh=True, tries=3, pause=2.0)
        except RuntimeError:
            time.sleep(20 * (k + 1))
    raise RuntimeError(f"Yahoo download failed for {sym}")


def read_yahoo(sym: str) -> pd.DataFrame:
    """Daily close / adjclose (dividend- and split-adjusted) from cached chart JSON."""
    j = json.loads(yahoo_path(sym).read_text())["chart"]["result"][0]
    ts = j.get("timestamp") or []
    q = j["indicators"]["quote"][0]
    adj = j["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    tz = j["meta"].get("exchangeTimezoneName", "America/New_York")
    idx = (pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz)
           .tz_localize(None).normalize())
    df = pd.DataFrame({"close": q["close"], "adjclose": adj},
                      index=pd.DatetimeIndex(idx, name="date"), dtype=float)
    df = df[~df.index.duplicated(keep="last")].dropna(how="all")
    return df


def yahoo_meta(sym: str) -> dict:
    return json.loads(yahoo_path(sym).read_text())["chart"]["result"][0]["meta"]


# ---------------------------------------------------------------- Bundesbank
BBK_URL = "https://api.statistiken.bundesbank.de/rest/data/BBEX3/{key}?format=csv&lang=en"


def bbk_path(key: str) -> Path:
    return RAW_DIR / "bundesbank" / f"BBEX3.{key}.csv"


def read_bbk(key: str) -> pd.Series:
    rows = []
    for line in bbk_path(key).read_text(encoding="utf-8-sig").splitlines():
        parts = line.split(",")
        if len(parts) >= 2 and len(parts[0]) == 10 and parts[0][4] == "-":
            try:
                rows.append((pd.Timestamp(parts[0]), float(parts[1])))
            except ValueError:
                pass  # "." = no value
    s = pd.Series(dict(rows), dtype=float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.index.name = "date"
    return s


# ---------------------------------------------------------------- DBnomics / IMF IFS
DBN_URL = "https://api.db.nomics.world/v22/series/{prov}/{ds}/{code}?observations=1"


def dbn_path(prov: str, ds: str, code: str) -> Path:
    return RAW_DIR / "dbnomics" / f"{prov}_{ds}_{code}.json"


def read_dbn(prov: str, ds: str, code: str) -> pd.Series:
    d = json.loads(dbn_path(prov, ds, code).read_text())["series"]["docs"][0]
    s = pd.Series(d["value"], index=pd.PeriodIndex(d["period"], freq="M").to_timestamp("M"))
    return pd.to_numeric(s, errors="coerce").dropna()


# ---------------------------------------------------------------- Ken French RF
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{f}"
FRENCH_FILES = {"daily": "F-F_Research_Data_Factors_daily_CSV.zip",
                "monthly": "F-F_Research_Data_Factors_CSV.zip"}


def read_french_rf(freq: str) -> pd.Series:
    """1-month T-bill return (Ibbotson; ICE BofA from 2024-06), decimal."""
    z = zipfile.ZipFile(RAW_DIR / "french" / FRENCH_FILES[freq])
    txt = z.read(z.namelist()[0]).decode("latin-1")
    out = {}
    for line in txt.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        if freq == "daily" and len(parts[0]) == 8:
            out[pd.Timestamp(parts[0])] = float(parts[4]) / 100
        elif freq == "monthly" and len(parts[0]) == 6:
            out[pd.Period(parts[0][:4] + "-" + parts[0][4:], "M").to_timestamp("M")] = float(parts[4]) / 100
        elif freq == "monthly" and len(parts[0]) == 4:
            break  # annual block follows the monthly block
    s = pd.Series(out, dtype=float).sort_index()
    s.index.name = "date"
    return s


# ---------------------------------------------------------------- AQR
AQR_URL = ("https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
           "Commodities-for-the-Long-Run-Index-Level-Data-Monthly.xlsx")
AQR_FILE = "Commodities-for-the-Long-Run-Index-Level-Data-Monthly.xlsx"


def read_aqr() -> pd.DataFrame:
    """AQR 'Commodities for the Long Run' monthly returns (excess of collateral)."""
    raw = pd.read_excel(RAW_DIR / "aqr" / AQR_FILE, sheet_name=0, header=None)
    hdr = raw.index[raw.iloc[:, 1].astype(str).str.startswith("Excess return of equal")][0]
    df = raw.iloc[hdr + 1:, :6].copy()
    df.columns = ["date", "ew_excess", "ew_excess_spot", "ew_carry_adj", "ew_spot", "ew_carry"]
    df = df.dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    df = df.set_index("date").astype(float)
    return df


# ---------------------------------------------------------------- World Bank pink sheet
WB_URL = ("https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/"
          "related/CMO-Historical-Data-Monthly.xlsx")
WB_FILE = "CMO-Historical-Data-Monthly.xlsx"


def read_worldbank(col: str = "Gold") -> pd.Series:
    """Monthly AVERAGE price from the World Bank pink sheet."""
    raw = pd.read_excel(RAW_DIR / "worldbank" / WB_FILE, sheet_name="Monthly Prices", header=None)
    hdr = raw.index[raw.iloc[:, 0].astype(str).str.match(r"^\d{4}M\d{2}$")][0] - 2
    names = raw.iloc[hdr].astype(str).str.strip()
    j = int(np.where(names == col)[0][0])
    body = raw.iloc[hdr + 2:, [0, j]].dropna(subset=[0])
    body = body[body[0].astype(str).str.match(r"^\d{4}M\d{2}$")]
    idx = pd.PeriodIndex([s.replace("M", "-") for s in body[0]], freq="M").to_timestamp("M")
    return pd.Series(pd.to_numeric(body[j].values, errors="coerce"), index=idx).dropna()


# --------------------------------------------------------------------------- #
# Download everything
# --------------------------------------------------------------------------- #
GOLD_YAHOO = ["GLD", "IAU", "GC=F"]
COMM_YAHOO = ["DBC", "GSG", "DJP", "GCC", "PCRIX", "QRAAX", "^SPGSCI", "^BCOM"]
BBK_KEYS = ["D.XAU.DEM.EA.AC.C01", "D.USD.DEM.AA.AC.000"]
IFS_CODES = ["M.US.RAFAGOLDM_USD", "M.US.RAFAGOLDV_OZT"]


def download_all(refresh: bool = False, refresh_yahoo: bool = False) -> None:
    for k in BBK_KEYS:
        _fetch(BBK_URL.format(key=k), bbk_path(k), refresh=refresh)
    for c in IFS_CODES:
        _fetch(DBN_URL.format(prov="IMF", ds="IFS", code=c), dbn_path("IMF", "IFS", c), refresh=refresh)
    for f in FRENCH_FILES.values():
        _fetch(FRENCH_URL.format(f=f), RAW_DIR / "french" / f, refresh=refresh)
    _fetch(AQR_URL, RAW_DIR / "aqr" / AQR_FILE, refresh=refresh)
    _fetch(WB_URL, RAW_DIR / "worldbank" / WB_FILE, refresh=refresh)
    for s in GOLD_YAHOO + COMM_YAHOO:
        fetch_yahoo(s, refresh=refresh or refresh_yahoo)


# --------------------------------------------------------------------------- #
# Splice helpers and diagnostics
# --------------------------------------------------------------------------- #


@dataclass
class Segment:
    asset: str
    start: pd.Timestamp
    end: pd.Timestamp
    freq: str
    source: str
    tier: str
    notes: str = ""


def to_monthly(r: pd.Series) -> pd.Series:
    """Compound simple daily returns into calendar month-end returns."""
    r = r.dropna()
    return (1 + r).groupby(pd.Grouper(freq="ME")).prod(min_count=1).sub(1).dropna()


def overlap_stats(a: pd.Series, b: pd.Series, freq: str) -> dict:
    """Overlap diagnostics on LOG returns: n, ann. vols, vol ratio a/b, corr, TE."""
    k = 252 if freq == "D" else 12
    x = pd.concat([np.log1p(a), np.log1p(b)], axis=1, keys=["a", "b"], sort=True).dropna()
    if len(x) < 3:
        return {"n": len(x)}
    va, vb = x.a.std() * np.sqrt(k), x.b.std() * np.sqrt(k)
    return {
        "n": len(x), "start": x.index[0].date(), "end": x.index[-1].date(),
        "vol_a": va, "vol_b": vb, "vol_ratio": va / vb,
        "corr": x.a.corr(x.b), "te": (x.a - x.b).std() * np.sqrt(k),
        "mean_diff_ann": (x.a - x.b).mean() * k,
    }


def fmt_stats(s: dict) -> str:
    if s.get("n", 0) < 3:
        return f"n={s.get('n', 0)}"
    return (f"n={s['n']} {s['start']}..{s['end']}: vol {s['vol_a']:.3f} vs {s['vol_b']:.3f} "
            f"(ratio {s['vol_ratio']:.3f}), corr {s['corr']:.3f}, TE {s['te']:.3f}, "
            f"mean diff {s['mean_diff_ann']:+.3f}/yr")


def _frame(r: pd.Series, source: str, tier: str, **extra) -> pd.DataFrame:
    df = pd.DataFrame({"ret": r.astype(float)})
    df["source"] = source
    df["tier"] = tier
    for k, v in extra.items():
        df[k] = v
    return df


def first_full_month_start(idx: pd.DatetimeIndex) -> pd.Timestamp:
    """First day of the first calendar month fully covered by a price index.

    A price series whose first observation is in month m yields returns for
    a full calendar month only from month m+1 (its base is m's last price).
    """
    return (idx[0] + pd.offsets.MonthBegin(1)).normalize()


def price_returns(p: pd.Series, start=None, end=None) -> pd.Series:
    """Simple returns from a price series; first return is the one ending after ``start``."""
    p = p.dropna()
    r = p.pct_change().iloc[1:]
    if start is not None:
        r = r.loc[pd.Timestamp(start):]
    if end is not None:
        r = r.loc[:pd.Timestamp(end)]
    return r


def ann_vol_by_decade(r: pd.Series, freq: str) -> pd.Series:
    k = 252 if freq == "D" else 12
    x = np.log1p(r.dropna())
    return x.groupby((x.index.year // 10) * 10).std() * np.sqrt(k)
