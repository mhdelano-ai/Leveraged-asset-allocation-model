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


# ---------------------------------------------------------------- FT markets (fallback / check)
# Unadjusted daily OHLC from markets.ft.com, one year per request. Used only for
# vehicles that never distribute (GLD grantor trust, GSG grantor trust, DJP
# ETN), so close-to-close = total return; and as an independent check on Yahoo.
FT_XID = {
    "GLD": "139276",       # SPDR Gold Shares (NYSE Arca)
    "DJP": "4935806",      # iPath Bloomberg Commodity Index TR ETN
    "GSG": "5195936",      # iShares S&P GSCI Commodity-Indexed Trust
    "DBC": "4092531",      # Invesco DB Commodity Index Tracking (distributes: check only)
    "BCOM": "570179",      # Bloomberg Commodity Index (excess return), daily from 1991
    "BCOMTR": "591643",    # Bloomberg Commodity Index Total Return (FT history from ~2008)
    "GC1": "1045492",      # COMEX gold front-month continuous (GC.1:CMX)
}
FT_XID["CMOD"] = "377694284"  # Invesco Bloomberg Commodity UCITS ETF (LSE, USD): check only
FT_FIRST_YEAR = {"CMOD": 2017, "GLD": 2004, "DJP": 2006, "GSG": 2006, "DBC": 2006, "BCOM": 1991,
                 "BCOMTR": 2005, "GC1": 1975}
FT_URL = ("https://markets.ft.com/data/equities/ajax/get-historical-prices?"
          "startDate={a}&endDate={b}&symbol={xid}")


def fetch_ft(sym: str, first_year: int | None = None, refresh_last: bool = True) -> None:
    this_year = pd.Timestamp.today().year
    for y in range(first_year or FT_FIRST_YEAR[sym], this_year + 1):
        p = RAW_DIR / "ft" / sym / f"{y}.json"
        a, b = f"{y}%2F01%2F01", f"{y}%2F12%2F31"
        _fetch(FT_URL.format(a=a, b=b, xid=FT_XID[sym]), p,
               refresh=(refresh_last and y == this_year), pause=0.7)


def read_ft(sym: str) -> pd.Series:
    import re
    rows = {}
    for p in sorted((RAW_DIR / "ft" / sym).glob("*.json")):
        html = json.loads(p.read_text())["html"]
        for tr in re.findall(r"<tr>(.*?)</tr>", html, flags=re.S):
            date = re.search(r'hide-small-below">([^<]+)</span>', tr)
            nums = re.findall(r"<td>([\d,\.]+)</td>", tr)
            if date and len(nums) >= 4:
                rows[pd.Timestamp(date.group(1))] = float(nums[3].replace(",", ""))
    s = pd.Series(rows, dtype=float).sort_index()
    s.index.name = "date"
    return s


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
# Yahoo is OPTIONAL here: during the build (2026-11-14) the Yahoo chart API
# returned HTTP 429 for > 1 hour to every client in this environment, so all
# daily market data come from markets.ft.com. If Yahoo files are cached they
# are used only as a cross-check (see ``yahoo_crosscheck``).
YAHOO_CHECK = ["GLD", "DBC"]
BBK_KEYS = ["D.XAU.DEM.EA.AC.C01", "D.USD.DEM.AA.AC.000"]
IFS_CODES = ["M.US.RAFAGOLDM_USD", "M.US.RAFAGOLDV_OZT"]


def download_all(refresh: bool = False, try_yahoo: bool = False) -> None:
    """Fetch every raw input into data/raw/real_assets (cached).

    ``refresh`` re-downloads the slowly-revised files; FT files for the
    current year are always refreshed so the data run to the latest close.
    """
    for k in BBK_KEYS:
        _fetch(BBK_URL.format(key=k), bbk_path(k), refresh=refresh)
    for c in IFS_CODES:
        _fetch(DBN_URL.format(prov="IMF", ds="IFS", code=c), dbn_path("IMF", "IFS", c), refresh=refresh)
    for f in FRENCH_FILES.values():
        _fetch(FRENCH_URL.format(f=f), RAW_DIR / "french" / f, refresh=refresh)
    _fetch(AQR_URL, RAW_DIR / "aqr" / AQR_FILE, refresh=refresh)
    _fetch(WB_URL, RAW_DIR / "worldbank" / WB_FILE, refresh=refresh)
    for s in FT_XID:
        fetch_ft(s)
    if try_yahoo:
        for s in YAHOO_CHECK:
            try:
                fetch_yahoo(s)
            except RuntimeError as e:
                print(f"yahoo unavailable ({e}); continuing without the cross-check")


def clean_spikes(p: pd.Series, k: float = 0.05) -> tuple[pd.Series, list]:
    """Drop isolated bad prints: a jump > k immediately reversed (to within half).

    markets.ft.com occasionally publishes garbage index values on exchange
    holidays (e.g. BCOM 2011-09-05 = 140.40 between 162.51 and 161.22;
    2023-12-25 = 1496.35). Removing the print merges the two returns around
    it into one genuine return; nothing is interpolated.
    """
    p = p.dropna()
    lp = np.log(p)
    a, b = lp.diff(), -lp.diff(-1)
    bad = ((a.abs() > k) & (b.abs() > k) & (np.sign(a) != np.sign(b))
           & ((a + b).abs() < 0.5 * np.minimum(a.abs(), b.abs())))
    return p[~bad], list(p.index[bad].date)


def ft_prices(sym: str) -> pd.Series:
    return clean_spikes(read_ft(sym))[0]


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


# --------------------------------------------------------------------------- #
# GOLD
# --------------------------------------------------------------------------- #
OZ_PER_KG = 32.1507466
# Free-market start. Before 15 Aug 1971 the private London/Zurich/Frankfurt
# price was tethered to the $35 official price (the two-tier market of
# 1968-71 kept it within ~$35-43, ann. vol < 10%). Convertibility was
# suspended on 15 Aug 1971; from then on no official price constrained the
# market. Returns start with the first full month after that, so every
# monthly return is a complete month of free-market data.
GOLD_START = pd.Timestamp("1971-09-01")


def gold_frankfurt_usd() -> pd.Series:
    """Frankfurt kilo-bar fixing (DEM/kg) / Frankfurt USD fixing (DEM per USD) -> USD/oz.

    Both are Frankfurt exchange fixings set around 13:00 CET on the same day,
    so the conversion is synchronous.
    """
    g, fx = read_bbk(BBK_KEYS[0]), read_bbk(BBK_KEYS[1])
    both = pd.concat([g, fx], axis=1, keys=["g", "fx"], sort=True).dropna()
    return (both.g / both.fx / OZ_PER_KG).rename("gold_frankfurt_usd")


def gold_ifs_monthend() -> pd.Series:
    """IMF IFS: US official gold 'at market price' (USD mn) / volume (mn fine oz).

    IFS values official gold at the end-of-period London price, so the ratio
    is a month-end market price per ounce.
    """
    v = read_dbn("IMF", "IFS", IFS_CODES[0]) / read_dbn("IMF", "IFS", IFS_CODES[1])
    return v.rename("gold_ifs_monthend")


def build_gold() -> tuple[pd.DataFrame, pd.DataFrame, list[Segment], dict]:
    """Gold: Frankfurt fixing (1971-09..1998-12) -> COMEX front month (1999..2004-11) -> GLD."""
    segs: list[Segment] = []
    diag: dict = {}
    fra = gold_frankfurt_usd()
    ifs = gold_ifs_monthend()
    gc, gc_bad = clean_spikes(read_ft("GC1"))
    gld, gld_bad = clean_spikes(read_ft("GLD"))
    diag["_removed_prints"] = {"GC1": gc_bad, "GLD": gld_bad}

    gld_start = first_full_month_start(gld.index)       # GLD listed 2004-11-18 -> 2004-12-01
    fra_end = fra.index[-1]                               # 1998-12-30
    d1 = price_returns(fra, GOLD_START, None)
    d2 = price_returns(gc, fra_end + pd.Timedelta(days=1), gld_start - pd.Timedelta(days=1))
    d3 = price_returns(gld, gld_start)
    daily = pd.concat([
        _frame(d1, "bundesbank:BBEX3 Frankfurt gold fixing DEM/kg / Frankfurt USD fixing", "B"),
        _frame(d2, "ft:GC.1:CMX COMEX gold front-month close", "B"),
        _frame(d3, "ft:GLD close (no distributions; = total return)", "A"),
    ]).sort_index()
    daily.index.name = "date"
    segs += [
        Segment("gold", d1.index[0], d1.index[-1], "D",
                "Bundesbank BBEX3 Frankfurt kilo-bar fixing (DEM/kg) / Frankfurt USD fixing (DEM per USD)", "B",
                "starts after the 15 Aug 1971 closing of the gold window; both fixings ~13:00 CET, synchronous; "
                "series ends 1998-12-30 (the next segment supplies the 31 Dec 1998 return)"),
        Segment("gold", d2.index[0], d2.index[-1], "D", "markets.ft.com GC.1:CMX COMEX gold front-month (continuous) close", "B",
                "futures price change as spot proxy; settles 13:30 ET (GLD closes 16:00 ET)"),
        Segment("gold", d3.index[0], d3.index[-1], "D", "markets.ft.com GLD close (SPDR Gold Shares; never distributes)", "A",
                "physical gold less 0.40%/yr expense ratio"),
    ]
    monthly = _monthly_from_daily(daily)

    # --- diagnostics ------------------------------------------------------
    ifs_r = ifs.pct_change()
    rgc, rgld, rfra = gc.pct_change().dropna(), gld.pct_change().dropna(), fra.pct_change().dropna()
    diag["Frankfurt vs IFS month-end (M)"] = overlap_stats(to_monthly(d1), ifs_r, "M")
    diag["GC1 vs Frankfurt (D)"] = overlap_stats(rgc, rfra.loc[:fra_end], "D")
    diag["GC1 vs Frankfurt (M)"] = overlap_stats(to_monthly(rgc), to_monthly(rfra), "M")
    diag["GC1 vs IFS month-end (M, 1999-2004)"] = overlap_stats(to_monthly(rgc).loc["1999":"2004"], ifs_r, "M")
    diag["GC1 vs GLD (D)"] = overlap_stats(rgc, rgld, "D")
    diag["GC1 vs GLD (M)"] = overlap_stats(to_monthly(rgc), to_monthly(rgld), "M")
    diag["GLD vs IFS month-end (M)"] = overlap_stats(to_monthly(d3), ifs_r, "M")
    wb = read_worldbank("Gold").pct_change()
    diag["World Bank month-AVERAGE vs IFS month-END (M)"] = overlap_stats(wb.loc[GOLD_START:], ifs_r.loc[GOLD_START:], "M")
    diag["_ac1"] = {"World Bank month-average": np.log1p(wb.loc[GOLD_START:].dropna()).autocorr(),
                    "IFS month-end": np.log1p(ifs_r.loc[GOLD_START:].dropna()).autocorr()}
    return daily, monthly, segs, diag


def _monthly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby(pd.Grouper(freq="ME"))
    m = pd.DataFrame({"ret": to_monthly(daily["ret"])})
    m["source"] = g["source"].last().reindex(m.index)
    m["tier"] = g["tier"].last().reindex(m.index)
    for c in daily.columns.difference(["ret", "source", "tier"]):
        if c == "ret_excess_native":
            m[c] = (1 + daily[c]).groupby(pd.Grouper(freq="ME")).prod(min_count=1).sub(1).reindex(m.index)
        elif c == "cash_ret":
            m[c] = (1 + daily[c]).groupby(pd.Grouper(freq="ME")).prod(min_count=1).sub(1).reindex(m.index)
    m.index.name = "date"
    return m


# --------------------------------------------------------------------------- #
# COMMODITIES
# --------------------------------------------------------------------------- #
# Target: Bloomberg Commodity Index Total Return (BCOM TR): ~20 futures,
# energy capped near 30%, the most diversified of the long-lived broad
# indices and the closest to AQR's equal-weight history (see report).
COMM_AQR_START = pd.Timestamp("1926-07-31")   # first month with a T-bill return (Ken French RF)
BCOMTR_NATIVE_FROM = pd.Timestamp("2009-07-01")  # FT BCOMTR history starts 2009-06-22


def build_commodities() -> tuple[pd.DataFrame, pd.DataFrame, list[Segment], dict]:
    segs: list[Segment] = []
    diag: dict = {}
    bcom, b_bad = clean_spikes(read_ft("BCOM"))
    btr, t_bad = clean_spikes(read_ft("BCOMTR"))
    diag["_removed_prints"] = {"BCOM": b_bad, "BCOMTR": t_bad}
    rfd, rfm = read_french_rf("daily"), read_french_rf("monthly")

    # --- daily segment 1: BCOM excess-return index + 1-month T-bill ------------
    er = price_returns(bcom, "1991-02-01", BCOMTR_NATIVE_FROM - pd.Timedelta(days=1))
    cash = rfd.reindex(er.index)
    # Index days on which the US T-bill file has no entry (US holidays on
    # which BCOM still printed): no cash accrues on that index day; the
    # T-bill accrual of a holiday is carried by the next business day in the
    # French file, so nothing is lost or double-counted.
    cash = cash.fillna(0.0)
    tr1 = (1 + er) * (1 + cash) - 1
    seg1 = _frame(tr1, "ft:BCOM (excess return) + KenFrench daily RF", "B",
                  ret_excess_native=er, cash_ret=cash)
    # --- daily segment 2: native BCOM TR ---------------------------------------
    tr2 = price_returns(btr, BCOMTR_NATIVE_FROM)
    seg2 = _frame(tr2, "ft:BCOMTR (Bloomberg Commodity Index Total Return)", "B")
    daily = pd.concat([seg1, seg2]).sort_index()
    daily.index.name = "date"
    segs += [
        Segment("commodities", tr1.index[0], tr1.index[-1], "D",
                "markets.ft.com BCOM (Bloomberg Commodity Index, excess return) compounded with Ken French daily RF (1-month T-bill)", "B",
                "ret_excess_native = BCOM ER; cash_ret = French RF. Index launched 1998 (DJ-AIG); 1991-98 is the provider's backfill"),
        Segment("commodities", tr2.index[0], tr2.index[-1], "D",
                "markets.ft.com BCOMTR (Bloomberg Commodity Index Total Return; collateral = 3-month T-bill)", "B",
                "native total return; same index as the previous segment"),
    ]
    mdaily = _monthly_from_daily(daily)

    # --- monthly segment 0: AQR equal-weight excess return + T-bill ------------
    aqr = read_aqr()
    ex = aqr["ew_excess"].loc[COMM_AQR_START: mdaily.index[0] - pd.offsets.MonthEnd(1)]
    c = rfm.reindex(ex.index)
    assert c.notna().all()
    m0 = _frame((1 + ex) * (1 + c) - 1, "aqr:Commodities for the Long Run EW excess return + KenFrench monthly RF", "C",
                ret_excess_native=ex, cash_ret=c)
    monthly = pd.concat([m0, mdaily]).sort_index()
    monthly.index.name = "date"
    segs.insert(0, Segment("commodities", ex.index[0], ex.index[-1], "M",
                           "AQR 'Commodities for the Long Run' (Levine, Ooi, Richardson, Sasseville 2018) equal-weight "
                           "futures excess return, compounded with Ken French monthly RF", "C",
                           "EQUAL-WEIGHT across available contracts (few, mostly agricultural, before the 1960s-70s); "
                           "not the BCOM weighting used after 1991"))

    # --- diagnostics ------------------------------------------------------------
    rb_er = bcom.pct_change().dropna()
    rb_tr = (1 + rb_er) * (1 + rfd.reindex(rb_er.index).fillna(0)) - 1
    aqr_tr = ((1 + aqr["ew_excess"]) * (1 + rfm.reindex(aqr.index)) - 1).dropna()
    diag["BCOM ER + RF vs BCOMTR (D)"] = overlap_stats(rb_tr, btr.pct_change(), "D")
    diag["BCOM ER + RF vs BCOMTR (M)"] = overlap_stats(to_monthly(rb_tr), to_monthly(btr.pct_change().dropna()), "M")
    for a, b in [("1991", "1998"), ("1999", "2006"), ("2007", "2015"), ("2016", "2025")]:
        diag[f"AQR-EW vs BCOM TR (M, {a}-{b})"] = overlap_stats(aqr_tr.loc[a:b], to_monthly(daily["ret"]), "M")
    diag["AQR-EW vs BCOM TR (M, 1991-2025 all)"] = overlap_stats(aqr_tr, to_monthly(daily["ret"]), "M")
    etf = {s: ft_prices(s).pct_change().dropna() for s in ["DBC", "GSG", "DJP", "CMOD"] if (RAW_DIR / "ft" / s).exists()}
    for s, r in etf.items():
        diag[f"{s} vs BCOM TR (D)"] = overlap_stats(r, daily["ret"], "D")
        diag[f"{s} vs BCOM TR (M)"] = overlap_stats(to_monthly(r), to_monthly(daily["ret"]), "M")
    if "DBC" in etf and "GSG" in etf:
        diag["GSG vs DBC (M)"] = overlap_stats(to_monthly(etf["GSG"]), to_monthly(etf["DBC"]), "M")
    # DJP's sensitivity to the index (see report): regression slope on monthly log returns
    if "DJP" in etf:
        x = pd.concat([np.log1p(to_monthly(etf["DJP"])), np.log1p(to_monthly(daily["ret"]))],
                      axis=1, keys=["e", "i"]).dropna().iloc[1:]
        diag["_djp_beta_monthly"] = float(np.polyfit(x.i, x.e, 1)[0])
    diag["_ac1_monthly_log"] = {"AQR-EW 1926-1990": np.log1p(m0["ret"]).autocorr(),
                                "BCOM TR 1991-": np.log1p(mdaily["ret"]).autocorr()}
    return daily, monthly, segs, diag


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def save(asset: str, daily: pd.DataFrame, monthly: pd.DataFrame) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for freq, df in [("daily", daily), ("monthly", monthly)]:
        df = df.sort_index()
        df.index = pd.DatetimeIndex(df.index, name="date")
        df["source"] = df["source"].astype(str)
        df["tier"] = df["tier"].astype(str)
        df.to_parquet(ASSET_DIR / f"{asset}_{freq}.parquet")


def segments_frame(segs: list[Segment]) -> pd.DataFrame:
    return pd.DataFrame([{
        "asset": s.asset, "start": pd.Timestamp(s.start).date(), "end": pd.Timestamp(s.end).date(),
        "freq": s.freq, "source": s.source, "tier": s.tier, "notes": s.notes} for s in segs])


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def decade_table(series: dict[str, tuple[pd.Series, str]]) -> pd.DataFrame:
    cols = {}
    for name, (r, f) in series.items():
        cols[name] = ann_vol_by_decade(r, f)
    t = pd.DataFrame(cols)
    t.index = [f"{int(i)}s" for i in t.index]
    t.index.name = "decade"
    return t.reset_index()


def diag_table(diag: dict) -> pd.DataFrame:
    rows = []
    for k, s in diag.items():
        if isinstance(s, dict) and s.get("n", 0) >= 3:
            rows.append({"comparison (a vs b)": k, "n": s["n"], "start": s["start"], "end": s["end"],
                         "vol a": s["vol_a"], "vol b": s["vol_b"], "ratio a/b": s["vol_ratio"],
                         "corr": s["corr"], "TE": s["te"], "mean diff/yr": s["mean_diff_ann"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _yearly_vol(r: pd.Series, k: int) -> pd.Series:
    x = np.log1p(r.dropna())
    return x.groupby(x.index.year).std() * np.sqrt(k)


def write_report(gold, comm) -> str:
    gd, gm, gsegs, gdiag = gold
    cd, cm, csegs, cdiag = comm
    seg = segments_frame(gsegs + csegs)
    L = []
    w = L.append
    w("# Real assets: gold and broad commodities\n")
    w(f"Generated by `python -m volmodel.data.sources.real_assets` on {pd.Timestamp.today().date()}. "
      "Returns are SIMPLE TOTAL returns in USD. Overlap statistics use log returns; vols annualized "
      "(252 days / 12 months); TE = annualized std of the return difference.\n")
    w("## History achieved\n")
    w(_md_table(pd.DataFrame([
        {"asset": "gold", "daily from": gd.index[0].date(), "monthly from": gm.index[0].date(),
         "last date": gd.index[-1].date(), "n daily": len(gd), "n monthly": len(gm)},
        {"asset": "commodities", "daily from": cd.index[0].date(), "monthly from": cm.index[0].date(),
         "last date": cd.index[-1].date(), "n daily": len(cd), "n monthly": len(cm)},
    ])))
    w("\nThe last month in each monthly file is the current partial month (compounded daily to the last close).\n")
    w("## Segments\n")
    w(_md_table(seg))
    w("\nMonthly rows in daily eras are compounded daily returns (identical by construction).\n")
    w(GOLD_TEXT)
    w("### Gold overlap diagnostics (a vs b)\n")
    w(_md_table(diag_table(gdiag)))
    ac = gdiag["_ac1"]
    w(f"\nLag-1 autocorrelation of monthly log returns 1971-09..: month-AVERAGE (World Bank) "
      f"{ac['World Bank month-average']:.2f} vs month-END (IFS) {ac['IFS month-end']:.2f}. "
      "Averaging damps vol by ~14% and induces the textbook ~0.25 autocorrelation, which is why no month-average "
      "series is used in the gold history (the World Bank file is kept only for this check).\n")
    w(f"Bad prints removed from FT series: {gdiag['_removed_prints']} (GC1 removals are all before 1999, i.e. outside the segment used).\n")
    w(COMM_TEXT)
    w("### Commodity overlap diagnostics (a vs b)\n")
    w(_md_table(diag_table(cdiag)))
    w(f"\nDJP monthly regression slope on BCOM TR: {cdiag.get('_djp_beta_monthly', float('nan')):.3f}. "
      "Bad prints removed: " + str(cdiag["_removed_prints"]) + ".\n")
    ac = cdiag["_ac1_monthly_log"]
    w("Lag-1 autocorrelation of monthly log returns: " + ", ".join(f"{k} {v:.2f}" for k, v in ac.items()) + ".\n")
    w("## Annualized volatility by decade\n")
    t = decade_table({"gold daily": (gd["ret"], "D"), "gold monthly": (gm["ret"], "M"),
                      "commodities daily": (cd["ret"], "D"), "commodities monthly": (cm["ret"], "M")})
    w(_md_table(t))
    w("\n## Annualized volatility by year (daily data)\n")
    yv = pd.DataFrame({"gold": _yearly_vol(gd["ret"], 252), "commodities": _yearly_vol(cd["ret"], 252)})
    yv.index.name = "year"
    w(_md_table(yv.reset_index().astype({"year": int})))
    w("\n## Known episodes (cumulative total return between month-ends)\n")
    rows = []
    for asset, m, eps in [("gold", gm, GOLD_EPISODES), ("commodities", cm, COMM_EPISODES)]:
        px = (1 + m["ret"]).cumprod()
        for a, b, what in eps:
            rows.append({"asset": asset, "from": a, "to": b, "episode": what,
                         "return": px.loc[b].iloc[-1] / px.loc[a].iloc[-1] - 1})
    w(_md_table(pd.DataFrame(rows)))
    w(CAVEATS)
    txt = "\n".join(L) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(txt)
    return txt


GOLD_EPISODES = [("1979-06", "1980-01", "1979-80 spike"), ("1980-01", "1982-06", "1980-82 collapse"),
                 ("2008-02", "2008-10", "2008 GFC drawdown"), ("2011-08", "2013-12", "2011-13 bust"),
                 ("2020-02", "2020-07", "2020 rally")]
COMM_EPISODES = [("1972-12", "1974-02", "1973-74 boom"), ("1985-11", "1986-07", "1986 oil collapse"),
                 ("1990-07", "1990-10", "1990 Gulf war spike"), ("2008-06", "2008-12", "2008 crash"),
                 ("2014-06", "2016-01", "2014-16 bust"), ("2020-01", "2020-04", "2020 Covid/oil"),
                 ("2021-12", "2022-06", "2022 spike")]

GOLD_TEXT = """
## Gold

**Target**: unlevered physical gold in USD (GLD-like).

**Where the history starts, and why: 1971-09 (daily and monthly).** Until 1968 the London price was
held at the official $35 by the Gold Pool. From March 1968 to August 1971 a two-tier market let the private
price float, but US convertibility of dollars into gold at $35 still anchored it (it stayed in a $35-43 range;
Frankfurt-fixing daily vol over Jun-1968..Aug-1971 was 8.4%/yr). On 15 Aug 1971 the gold window closed and no
official price constrained the market any longer. Returns therefore start with the first complete month after
that: September 1971. Data from the pegged/anchored era are NOT included. Two later institutional changes are
recorded but not used to cut the sample: the end of the two-tier arrangement (Nov 1973) and legal US private
ownership plus COMEX futures (31 Dec 1974). Users who want only the fully liberalised market can start at 1975-01;
the 1972-74 years (annual vol 18-36%) look like later high-vol regimes, not like a pegged market.

**Sources.**
- 1971-09..1998-12: Deutsche Bundesbank BBEX3 `D.XAU.DEM.EA.AC.C01` (Frankfurt exchange fixing for 1 kg bars,
  DEM) divided by `D.USD.DEM.AA.AC.000` (Frankfurt official USD fixing). Both are set by the Frankfurt exchange
  at about 13:00 CET, so the USD conversion is synchronous. Tier B (daily reconstructed). Against the IMF IFS
  month-end London price it has monthly vol ratio 1.02 and correlation 0.99.
- 1999-01..2004-11: COMEX front-month gold futures continuous close (markets.ft.com `GC.1:CMX`). Tier B.
  Against GLD over 2004-2026 it has the same vol (ratio 1.00 daily and monthly) and a mean difference of +0.4%/yr
  (the GLD fee); daily correlation is only 0.89 because COMEX settles at 13:30 ET and GLD closes at 16:00 ET.
- 2004-12..: GLD close (markets.ft.com). GLD never distributes, so close-to-close is the total return. Tier A.
- Cross-checks only: IMF IFS (via DBnomics) US official gold at market value / fine ounces = end-of-month
  London price, monthly 1950-2025; World Bank pink-sheet monthly AVERAGE price.

LBMA's own price feed (prices.lbma.org.uk) is behind a Cloudflare challenge (HTTP 403) from this environment, and
the World Gold Council download needs a login; FRED times out. The Bundesbank Frankfurt series was the only
keyless daily source found for 1971-1998.
"""

COMM_TEXT = """
## Commodities

**Target: Bloomberg Commodity Index Total Return (BCOM TR).** About 20 futures, energy capped near 30%,
collateral = 3-month T-bills. Reasons: (1) it is the most diversified long-lived broad index, (2) its vol is
closest to AQR's equal-weight long history (vol ratio AQR/BCOM 0.91 over 1991-2025, corr 0.95), so the
1990/91 splice creates the smallest artificial regime change, and (3) a daily index history exists from 1991.
DBC (the universe's listed proxy) is energy-heavy (~55-60%) and its vol is **1.17x** BCOM TR on the 2006-2026
overlap; GSG (S&P GSCI) is **1.44x**. If DBC itself is what will be held, scale expectations accordingly — a
DBC-only history would start in 2006 and would need a 1.17x composition jump at any splice.

**Why the recent era is tier B, not A.** No US-listed fund tracks BCOM TR cleanly at the daily level:
- DJP (iPath BCOM TR ETN, 2006-) moves **~1.09-1.15x** the index at every horizon (monthly regression slope above;
  cumulative ratio to the index fell ~13% by 2020, then rose ~6%). Closes match Nasdaq's exactly, so this is not an FT
  data error; it is unexplained here, so DJP is reported but not used.
- CMOD (Invesco BCOM UCITS, LSE, 2017-) tracks it (vol ratio 0.99) but closes in London, which would make
  daily correlations with US assets asynchronous.
So the index itself is used throughout 1991-2026 (tier B).

**Sources.**
- 1926-07..1991-01 (monthly): AQR "Commodities for the Long Run" equal-weight portfolio EXCESS return
  (futures return without collateral) compounded with Ken French monthly `RF` (1-month T-bill, Ibbotson; ICE
  BofA 1M T-bill from 2024-06). Tier C. Starts 1926-07 because that is where the T-bill series starts (AQR itself
  starts 1877). Stored with `ret_excess_native` and `cash_ret` so that `ret = (1+ret_excess_native)(1+cash_ret)-1`
  can be re-derived with the cash track's series.
- 1991-02..2009-06 (daily): BCOM (excess-return headline index, markets.ft.com) compounded daily with Ken French
  daily `RF`; same two extra columns. On BCOM print days missing from the French file (US holidays when the index
  printed) cash_ret = 0 — the French file books the holiday's interest on the next business day, so the total
  accrual is unchanged. Tier B.
  The excess+cash construction reproduces native BCOM TR almost exactly (2009-2026: daily corr 1.000, vol ratio
  1.000, TE 0.1%/yr, mean diff -0.1%/yr; the residual is 1M vs 3M T-bill).
- 2009-07..: BCOMTR (native total return, markets.ft.com). Tier B.

**Composition change (the one that matters for vol).** The 1991 splice moves from an EQUAL-WEIGHT basket to
BCOM weights. Overlap vol ratios AQR-EW / BCOM TR: 1991-98 0.88, 1999-2006 **0.75** (energy rally), 2007-15
0.97, 2016-25 0.95. Before ~1960 the AQR basket holds few, mostly agricultural contracts, and some periods
(WWII price controls, 1950s-60s support programmes) show very low vol (1960s 8%). Treat the pre-1970 part as
a different asset with the same name: use it for long-run tail scenarios if at all, not to calibrate today's vol.
The pre-1991 part is probably 5-15% LESS volatile than BCOM would have been.

Yahoo index tickers (^SPGSCI, ^BCOM, ...) could not be examined: the Yahoo chart API returned HTTP 429 to
every request from this environment for over an hour. The S&P GSCI TR from 1970 was not obtainable keylessly
(S&P DJI site), and FT's GSCI spot (GNX) is a SPOT index (no roll yield) and is not used.
"""

CAVEATS = """
## Caveats and open items

- **Data vendor.** All daily market data after 1998 come from markets.ft.com's public historical-price endpoint
  (unadjusted OHLC, one request per calendar year, cached in `data/raw/real_assets/ft/`). FT closes for GLD and
  DJP match Nasdaq's exactly over 2016-2026. FT occasionally prints garbage on exchange holidays; isolated
  reversed jumps > 5% are removed (listed above), nothing is interpolated.
- **Index close times.** BCOM/BCOMTR daily values use futures settlement prices (13:30-14:30 ET for most
  contracts), and the 1971-98 gold series is a 13:00 CET fixing. Daily correlations with US closes are therefore
  understated at lag 0 (e.g. GC1 vs Frankfurt daily corr 0.28, monthly 0.96); the 3-day-return correlation
  rule in ARCHITECTURE.md is needed for these assets too, not just for international equity.
- **Gold fees.** Pre-2004 gold has no expense ratio; GLD carries 0.40%/yr. Irrelevant for vol.
- **GC.1 roll.** The continuous front-month series can jump at rolls; on 2004-2026 it has the same vol as GLD
  (ratio 1.005 daily) and only +0.4%/yr mean difference, so roll jumps are small in FT's series.
- **Frankfurt reversals.** The 1973-1985 Frankfurt series has a handful of large one-day reversals (e.g. 1980-01-28
  -8.3% then +7.4%); they are present in the DEM gold price itself (not the FX rate) during the 1980 frenzy and
  are kept as genuine.
- **BCOM 1991-1998** is the index provider's backfill (index launched July 1998).
- **Cash for excess-return segments** is Ken French RF, the same file the cash track caches; if the project's
  cash series differs, recompute `ret` from `ret_excess_native` and the project's cash.
- **Not verified:** the cause of DJP's ~1.1x sensitivity to BCOM TR; LBMA daily fixes (blocked); Yahoo-based
  cross-checks (blocked).
"""


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-download slowly-revised raw files too")
    ap.add_argument("--offline", action="store_true", help="build from the raw cache only")
    ap.add_argument("--try-yahoo", action="store_true", help="also try Yahoo for cross-check files")
    a = ap.parse_args(argv)
    if not a.offline:
        download_all(refresh=a.refresh, try_yahoo=a.try_yahoo)
    gold = build_gold()
    comm = build_commodities()
    save("gold", gold[0], gold[1])
    save("commodities", comm[0], comm[1])
    write_report(gold, comm)
    print(segments_frame(gold[2] + comm[2]).to_string())
    print(f"wrote {ASSET_DIR}/gold_*.parquet, commodities_*.parquet and {REPORT_PATH}")


if __name__ == "__main__":
    main()
