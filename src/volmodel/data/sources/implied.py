"""Option-implied volatility indices (and a few related gauges) as predictors.

Run ``python -m volmodel.data.sources.implied`` to (re)build
``data/processed/predictors/implied.parquet`` and ``reports/data/implied.md``.
Raw downloads are cached under ``data/raw/implied/``; pass ``--refresh`` to
re-download them (``--refresh-yahoo`` / ``--refresh-cboe`` for one source).

Output: long frame ``obs_date, available_date, series, value``. These are
end-of-day market closes, so ``available_date == obs_date``.

Units
-----
* Every ``iv_*`` volatility index is stored as ANNUALIZED VOL IN DECIMALS
  (VIX 20 -> 0.20).
* ``iv_move`` (ICE BofA MOVE) is a *yield* (normal / basis-point) vol: MOVE
  100 = 100bp/yr of yield vol -> stored as 0.0100. Multiply by modified
  duration to get an approximate price vol.
* ``iv_skew_index`` (Cboe SKEW) is NOT a vol: it is 100 - 10*(risk-neutral
  skewness), typically 100-160. Stored as published (level).
* ``iv_vvix`` is the vol of VIX, in decimals (VVIX 90 -> 0.90).
* Derived ``*_slope`` series are ratios minus one (dimensionless).

Sources
-------
* Cboe CDN ``cdn.cboe.com/api/global/us_indices/daily_prices/<SYM>_History.csv``
  (primary for every index it serves).
* Cboe CDN archives ``resources/us/indices/vxoarchive.xls`` (VXO 1986-2003)
  and ``vxocurrent.csv`` (VXO 2004-2021-01).
* Yahoo chart API, used (a) for series Cboe does not serve here (MOVE, the
  tail of VXO), and (b) to extend a Cboe series backwards before its first
  CDN date, but only if Yahoo and Cboe agree on their overlap (checked and
  reported). Yahoo is shared with other agents: everything is cached, a
  browser User-Agent is used, requests are spaced and retried with backoff.
  If Yahoo is unavailable the build still succeeds with Cboe data and the
  report says which Yahoo pieces are missing.

Cleaning (never interpolates; dropped values are simply absent):
* non-positive values dropped;
* isolated bad prints dropped: a value whose log-distance from BOTH
  neighbours exceeds ``SPIKE_LOG`` in the same direction while the two
  neighbours agree within ``SPIKE_NEIGHBOUR_LOG`` (e.g. VXD 2021-07-13 = 2.71
  between ~14 and ~15), unless VIX itself moved the same way by more than
  ``REF_CONFIRM_LOG`` that day (genuine market-wide jump);
* stale repeats dropped: within a run of >= ``STALE_RUN`` identical
  consecutive closes only the first is kept (thin ETF vol indices such as
  VXIEF carry values forward for weeks).
"""
from __future__ import annotations

import argparse
import io
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ...paths import PREDICTOR_DIR, RAW, REPORTS

TRACK = "implied"
CACHE = RAW / TRACK
OUT_FILE = PREDICTOR_DIR / "implied.parquet"
REPORT_FILE = REPORTS / "data" / "implied.md"
COLUMNS = ["obs_date", "available_date", "series", "value"]

BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
CBOE_CSV = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
CBOE_RES = "https://cdn.cboe.com/resources/us/indices/{name}"
YAHOO_URL = "https://{host}.finance.yahoo.com/v8/finance/chart/{sym}"

# "today" for live/dead classification; a series is live if its last close is
# within LIVE_DAYS calendar days of the newest close in the whole file.
LIVE_DAYS = 10
SPIKE_LOG = 0.4
SPIKE_NEIGHBOUR_LOG = 0.15
REF_CONFIRM_LOG = 0.15
STALE_RUN = 4


@dataclass(frozen=True)
class Spec:
    series: str
    name: str
    cboe: str | None          # Cboe CDN symbol (primary)
    yahoo: str | None         # Yahoo symbol (fallback / backward extension)
    scale: float              # multiply the published level by this
    unit: str
    notes: str = ""
    start: str | None = None  # drop an unreliable start-up period before this date


SPECS: list[Spec] = [
    # --- S&P 500 term structure -------------------------------------------
    Spec("iv_vix9d", "Cboe S&P 500 9-day vol (VIX9D)", "VIX9D", "^VIX9D", 0.01, "decimal vol"),
    Spec("iv_vix", "Cboe S&P 500 30-day vol (VIX)", "VIX", "^VIX", 0.01, "decimal vol",
         "Sep-2003 redefinition (SPX full strip, model-free) back-calculated to 1990; "
         "the pre-2003 'old VIX' is VXO."),
    Spec("iv_vix3m", "Cboe S&P 500 3-month vol (VIX3M, formerly VXV)", "VIX3M", "^VIX3M", 0.01,
         "decimal vol", "Renamed VXV -> VIX3M in 2017; same index."),
    Spec("iv_vix6m", "Cboe S&P 500 6-month vol (VIX6M, formerly VXMT)", "VIX6M", "^VIX6M", 0.01,
         "decimal vol", "Renamed VXMT -> VIX6M in 2017; close-only (O=H=L=C) until ~2013-12, "
         "consistent with retrospective calculation."),
    Spec("iv_vix1y", "Cboe S&P 500 1-year vol (VIX1Y)", "VIX1Y", "^VIX1Y", 0.01, "decimal vol",
         "Close-only (O=H=L=C) until ~2018-04: early history is retrospectively calculated."),
    Spec("iv_vxo", "Cboe S&P 100 30-day ATM vol (VXO, the pre-2003 'old VIX')", None, "^VXO", 0.01,
         "decimal vol",
         "Old methodology: OEX at-the-money, TRADING-day convention (inflates level vs "
         "calendar-day VIX). Discontinued by Cboe in 2021."),
    # --- other US equity ----------------------------------------------------
    Spec("iv_vxn", "Cboe Nasdaq-100 vol (VXN)", "VXN", "^VXN", 0.01, "decimal vol"),
    Spec("iv_rvx", "Cboe Russell 2000 vol (RVX)", "RVX", "^RVX", 0.01, "decimal vol"),
    Spec("iv_vxd", "Cboe DJIA vol (VXD)", "VXD", "^VXD", 0.01, "decimal vol"),
    # --- international equity ----------------------------------------------
    Spec("iv_vxefa", "Cboe EFA ETF vol (VXEFA)", "VXEFA", "^VXEFA", 0.01, "decimal vol",
         "Close-only until ~2013-07 (retrospective)."),
    Spec("iv_vxeem", "Cboe EEM ETF vol (VXEEM)", "VXEEM", "^VXEEM", 0.01, "decimal vol"),
    # --- rates and credit ---------------------------------------------------
    Spec("iv_vxtlt", "Cboe 20+yr Treasury ETF (TLT) vol (VXTLT)", "VXTLT", "^VXTLT", 0.01,
         "decimal vol"),
    Spec("iv_vxief", "Cboe 7-10yr Treasury ETF (IEF) vol (VXIEF)", "VXIEF", None, 0.01,
         "decimal vol", "Thin: long stale runs (e.g. one value carried 2020-07..2020-12, dropped); no data 2022-11-08..2025-07-13."),
    Spec("iv_vxlqd", "Cboe IG corporate ETF (LQD) vol (VXLQD)", "VXLQD", None, 0.01,
         "decimal vol", "Starts 2020-11; no data 2022-11-08..2025-07-13."),
    Spec("iv_vxhyg", "Cboe HY corporate ETF (HYG) vol (VXHYG)", "VXHYG", None, 0.01,
         "decimal vol", "Noisy; one garbage print (-19,738,470 on 2017-10-30) removed; "
         "no data 2021-09-24..2025-05-14."),
    Spec("iv_move", "ICE BofA MOVE (1-month Treasury option normal vol)", None, "^MOVE", 1e-4,
         "decimal YIELD vol (bp/1e4)", "Yield (basis-point) vol, not price vol."),
    # --- real assets --------------------------------------------------------
    Spec("iv_gvz", "Cboe gold ETF (GLD) vol (GVZ)", "GVZ", "^GVZ", 0.01, "decimal vol"),
    Spec("iv_ovx", "Cboe crude oil ETF (USO) vol (OVX)", "OVX", "^OVX", 0.01, "decimal vol",
         "USO options; peaked at 3.25 on 2020-04-21 (negative WTI front month)."),
    # --- higher moments -----------------------------------------------------
    Spec("iv_vvix", "Cboe VIX of VIX (VVIX)", "VVIX", "^VVIX", 0.01, "decimal vol of VIX",
         "2006-03..05 dropped: 3-8 prints a month, levels 16-92 vs ~90 after (start-up noise).",
         start="2006-06-01"),
    Spec("iv_skew_index", "Cboe S&P 500 SKEW index", "SKEW", "^SKEW", 1.0,
         "index level (NOT a vol)", "100 - 10 x risk-neutral skewness of 30-day SPX returns."),
]
BY_SERIES = {s.series: s for s in SPECS}

DERIVED_NOTES = {
    "iv_vix_term_slope": ("VIX3M / VIX - 1 (same-day closes)", "ratio - 1"),
    "iv_vix1y_term_slope": ("VIX1Y / VIX - 1 (same-day closes)", "ratio - 1"),
    "iv_vix_vxo_spliced": ("VIX from 1990-01-02; before that VXO x k, k = exp(mean log(VIX/VXO)) "
                           "over 1990-1994", "decimal vol"),
}

# Natural predictor per asset id: (asset, candidate series in order of
# preference -- the first one that was built is reported, comment).
ASSET_MAP = [
    ("us_large", "iv_vix1y", "1-yr point matches the horizon; iv_vix / iv_vix_vxo_spliced for long history"),
    ("us_small", "iv_rvx", "30-day only; no small-cap term structure"),
    ("intl_dev", "iv_vxefa", "EFA ETF options (dev ex-US/Canada)"),
    ("em", "iv_vxeem", "EEM ETF options"),
    ("ust_int", ("iv_move", "iv_vxief"), "yield vol x duration (~6.5) ~ price vol; iv_vxief is thin/gappy"),
    ("ust_long", "iv_vxtlt", "TLT ETF options; iv_move x duration (~16) as alternative"),
    ("tips", None, "no TIPS implied vol exists; proxy with iv_move / iv_vxief"),
    ("ig_credit", "iv_vxlqd", "short and gappy; rates gauges (iv_move) are the practical proxy"),
    ("hy_credit", "iv_vxhyg", "noisy, long gap 2021-2025; equity vol (iv_vix, iv_rvx) is the practical proxy"),
    ("reits", None, "no REIT implied vol index; iv_rvx / iv_vix as proxies"),
    ("gold", "iv_gvz", "GLD ETF options"),
    ("commodities", "iv_ovx", "crude only (USO options); energy-heavy indices like DBC/BCOM partly"),
]


# --------------------------------------------------------------------- fetch
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": BROWSER_UA, "Accept": "*/*"})
    return s


def _cached(path: Path, fetch, refresh: bool) -> bytes | None:
    """Return cached bytes, downloading via ``fetch()`` if absent or refresh."""
    if path.exists() and not refresh:
        return path.read_bytes()
    try:
        data = fetch()
    except Exception as e:  # keep the stale cache if a refresh fails
        print(f"  ! download failed for {path.name}: {e}")
        return path.read_bytes() if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _get_bytes(url: str, params: dict | None = None, tries: int = 5) -> bytes:
    s = _session()
    last = None
    for i in range(tries):
        try:
            r = s.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.content
            last = f"HTTP {r.status_code}"
            if r.status_code in (403, 404):
                break
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(2.0 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


def cboe_history(sym: str, refresh: bool = False) -> pd.Series | None:
    """Close series from the Cboe CDN ``<SYM>_History.csv`` (index level)."""
    raw = _cached(CACHE / "cboe" / f"{sym}_History.csv",
                  lambda: _get_bytes(CBOE_CSV.format(sym=sym)), refresh)
    if raw is None:
        return None
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip().upper() for c in df.columns]
    col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
    idx = pd.to_datetime(df["DATE"].astype(str).str.strip(), format="%m/%d/%Y")
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(), index=idx, name=sym)
    return s[~s.index.duplicated(keep="last")].sort_index().dropna()


def _first_number(x) -> float:
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    try:
        return float(str(x).split()[0])
    except (ValueError, IndexError):
        return np.nan


def cboe_vxo(refresh: bool = False) -> pd.Series | None:
    """VXO 1986-01-02 .. 2021-01-07 from the two Cboe archive files."""
    a = _cached(CACHE / "cboe" / "vxoarchive.xls",
                lambda: _get_bytes(CBOE_RES.format(name="vxoarchive.xls")), refresh)
    b = _cached(CACHE / "cboe" / "vxocurrent.csv",
                lambda: _get_bytes(CBOE_RES.format(name="vxocurrent.csv")), refresh)
    parts = []
    if a is not None:
        x = pd.read_excel(io.BytesIO(a), header=None)
        x = x.iloc[:, :5]
        x.columns = ["date", "open", "high", "low", "close"]
        x["date"] = pd.to_datetime(x["date"], errors="coerce", format="mixed")
        x = x.dropna(subset=["date"])
        # one cell reads "32.50  1.01" (2000-10-18): take the first number
        parts.append(pd.Series(x["close"].map(_first_number).to_numpy(), index=x["date"]))
    if b is not None:
        x = pd.read_csv(io.BytesIO(b), skiprows=2)
        x.columns = [c.strip().lower() for c in x.columns]
        d = pd.to_datetime(x["date"].astype(str).str.strip(), format="mixed")
        parts.append(pd.Series(pd.to_numeric(x["close"], errors="coerce").to_numpy(), index=d))
    if not parts:
        return None
    s = pd.concat(parts).sort_index()
    s.index = pd.DatetimeIndex(s.index).normalize()
    return s[~s.index.duplicated(keep="last")].dropna().rename("VXO")


_YAHOO_DOWN = False  # circuit breaker: once Yahoo keeps refusing, stop asking this run


def yahoo_series(sym: str, refresh: bool = False) -> pd.Series | None:
    """Daily close of a Yahoo index ticker (NOT adjusted; indices have no dividends)."""
    path = CACHE / "yahoo" / f"{sym.replace('^', '_')}.json"

    def fetch() -> bytes:
        global _YAHOO_DOWN
        if _YAHOO_DOWN:
            raise RuntimeError("skipped: Yahoo rate-limited earlier in this run (or cache-only)")
        s = _session()
        params = {"period1": 0, "period2": int(time.time()) + 86400, "interval": "1d",
                  "includePrePost": "false"}
        last = None
        for i in range(6):
            host = "query1" if i % 2 == 0 else "query2"
            try:
                r = s.get(YAHOO_URL.format(host=host, sym=sym), params=params, timeout=60)
                if r.status_code == 200:
                    time.sleep(1.5 + random.random())  # be polite: shared endpoint
                    return r.content
                last = f"HTTP {r.status_code}"
                if r.status_code == 404:
                    return r.content  # cache the "no data" answer too
            except requests.RequestException as e:
                last = repr(e)
            time.sleep(min(60.0, 10.0 * (i + 1)) + random.uniform(0, 5))
        _YAHOO_DOWN = True
        raise RuntimeError(last)

    raw = _cached(path, fetch, refresh)
    if raw is None:
        return None
    try:
        j = json.loads(raw)["chart"]
        res = (j.get("result") or [None])[0]
    except (ValueError, KeyError):
        return None
    if not res or not res.get("timestamp"):
        return None
    tz = res.get("meta", {}).get("exchangeTimezoneName", "America/New_York")
    idx = (pd.to_datetime(res["timestamp"], unit="s", utc=True)
           .tz_convert(tz).tz_localize(None).normalize())
    close = res["indicators"]["quote"][0].get("close") or []
    s = pd.Series(pd.to_numeric(pd.Series(close), errors="coerce").to_numpy(), index=idx, name=sym)
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    return s[s.index.dayofweek < 5]


# ------------------------------------------------------------------- cleaning
def clean(s: pd.Series, ref: pd.Series | None = None,
          start: str | None = None) -> tuple[pd.Series, dict]:
    """Drop non-positive values, isolated bad prints and stale repeats.

    A candidate spike is kept if ``ref`` (VIX) moved the same way by more than
    ``REF_CONFIRM_LOG`` on that day (e.g. VIX9D on 2011-08-08 is genuine).
    ``start`` drops a documented unreliable start-up period.
    """
    s = s.dropna()
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    n0 = len(s)
    nonpos = int((s <= 0).sum())
    s = s[s > 0]
    lx = np.log(s)
    d_prev = lx - lx.shift(1)
    d_next = lx - lx.shift(-1)
    neigh = (lx.shift(-1) - lx.shift(1)).abs()
    up = (d_prev > SPIKE_LOG) & (d_next > SPIKE_LOG)
    down = (d_prev < -SPIKE_LOG) & (d_next < -SPIKE_LOG)
    spike = (up | down) & (neigh < SPIKE_NEIGHBOUR_LOG)
    if ref is not None:
        r = np.log(ref[ref > 0])
        dr = (r - r.shift(1)).reindex(s.index)
        confirmed = (up & (dr > REF_CONFIRM_LOG)) | (down & (dr < -REF_CONFIRM_LOG))
        spike &= ~confirmed.fillna(False)
    spikes = list(s.index[spike].date)
    s = s[~spike]
    same = s.diff() == 0
    run_id = (~same).cumsum()
    run_len = s.groupby(run_id).transform("size")
    stale = same & (run_len >= STALE_RUN)
    s = s[~stale]
    return s, {"n_raw": n0, "nonpositive": nonpos, "spikes": spikes, "stale": int(stale.sum())}


def overlap_stats(a: pd.Series, b: pd.Series) -> dict:
    """Agreement of two level series on common dates (levels in index points)."""
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(j) < 20:
        return {"n": len(j)}
    d = j.iloc[:, 0] - j.iloc[:, 1]
    return {"n": len(j), "first": j.index[0].date(), "last": j.index[-1].date(),
            "corr": float(j.corr().iloc[0, 1]), "mean_diff": float(d.mean()),
            "median_abs_diff": float(d.abs().median()),
            "share_abs_gt_0.1": float((d.abs() > 0.1).mean())}


# -------------------------------------------------------------------- build
def build(refresh_cboe: bool = False, refresh_yahoo: bool = False) -> tuple[pd.DataFrame, dict]:
    levels: dict[str, pd.Series] = {}
    diag: dict[str, dict] = {}
    vix_ref = cboe_history("VIX", refresh_cboe)  # reference for confirming big moves
    for spec in SPECS:
        print(f"{spec.series}: ", end="", flush=True)
        info: dict = {"sources": []}
        primary = None
        if spec.series == "iv_vxo":
            primary = cboe_vxo(refresh_cboe)
            if primary is not None:
                info["sources"].append("cboe:vxoarchive.xls+vxocurrent.csv")
        elif spec.cboe:
            primary = cboe_history(spec.cboe, refresh_cboe)
            if primary is not None:
                info["sources"].append(f"cboe:{spec.cboe}_History.csv")
        y = yahoo_series(spec.yahoo, refresh_yahoo) if spec.yahoo else None
        info["yahoo_available"] = y is not None and len(y) > 0
        if y is not None and len(y):
            info["yahoo_range"] = (y.index[0].date(), y.index[-1].date(), len(y))
        if primary is not None and y is not None and len(y):
            st = overlap_stats(primary, y)
            info["overlap"] = st
            agree = st.get("n", 0) >= 250 and st["median_abs_diff"] < 0.02 * np.median(primary) \
                and st["corr"] > 0.99
            info["yahoo_agrees"] = bool(agree)
            pieces = [primary]
            if agree and y.index[0] < primary.index[0] - pd.Timedelta(days=7):
                pre = y[y.index < primary.index[0]]
                pieces.insert(0, pre)
                info["sources"].insert(0, f"yahoo:{spec.yahoo} {pre.index[0].date()}..{pre.index[-1].date()}")
            if agree and y.index[-1] > primary.index[-1] + pd.Timedelta(days=7):
                post = y[y.index > primary.index[-1]]
                pieces.append(post)
                info["sources"].append(f"yahoo:{spec.yahoo} {post.index[0].date()}..{post.index[-1].date()}")
            s = pd.concat(pieces).sort_index()
        elif primary is not None:
            s = primary
        elif y is not None and len(y):
            s = y
            info["sources"].append(f"yahoo:{spec.yahoo}")
        else:
            print("NO DATA")
            diag[spec.series] = info
            continue
        s, cinfo = clean(s, ref=None if spec.series == "iv_vix" else vix_ref,
                         start=spec.start)
        info.update(cinfo)
        levels[spec.series] = s
        diag[spec.series] = info
        print(f"{s.index[0].date()}..{s.index[-1].date()} n={len(s)} via {info['sources']}")

    # values in output units
    vals = {k: v * BY_SERIES[k].scale for k, v in levels.items()}

    # derived series
    if "iv_vix" in vals and "iv_vix3m" in vals:
        j = pd.concat([vals["iv_vix3m"], vals["iv_vix"]], axis=1, join="inner").dropna()
        vals["iv_vix_term_slope"] = j.iloc[:, 0] / j.iloc[:, 1] - 1
    if "iv_vix" in vals and "iv_vix1y" in vals:
        j = pd.concat([vals["iv_vix1y"], vals["iv_vix"]], axis=1, join="inner").dropna()
        vals["iv_vix1y_term_slope"] = j.iloc[:, 0] / j.iloc[:, 1] - 1
    if "iv_vix" in vals and "iv_vxo" in vals:
        vix, vxo = vals["iv_vix"], vals["iv_vxo"]
        j = pd.concat([vix, vxo], axis=1, join="inner").dropna().loc["1990":"1994"]
        k = float(np.exp(np.mean(np.log(j.iloc[:, 0] / j.iloc[:, 1]))))
        pre = vxo[vxo.index < vix.index[0]] * k
        vals["iv_vix_vxo_spliced"] = pd.concat([pre, vix]).sort_index()
        r = np.log(j.iloc[:, 0] / j.iloc[:, 1])
        diag["iv_vix_vxo_spliced"] = {
            "k": k, "overlap_n": len(j), "corr": float(j.corr().iloc[0, 1]),
            "log_ratio_sd": float(r.std()),
            "pre_range": (pre.index[0].date(), pre.index[-1].date(), len(pre)),
            "k_full_1990_2003": float(np.exp(np.mean(np.log(
                pd.concat([vix, vxo], axis=1, join="inner").dropna().loc[:"2003"]
                .pipe(lambda d: d.iloc[:, 0] / d.iloc[:, 1]))))),
        }

    frames = []
    for name, s in vals.items():
        s = s.dropna()
        frames.append(pd.DataFrame({"obs_date": s.index, "available_date": s.index,
                                    "series": name, "value": s.to_numpy(dtype=float)}))
    df = pd.concat(frames, ignore_index=True)
    df["obs_date"] = pd.to_datetime(df["obs_date"]).astype("datetime64[ns]")
    df["available_date"] = df["obs_date"]
    df["series"] = df["series"].astype(str)
    df = df.sort_values(["series", "obs_date"]).reset_index(drop=True)[COLUMNS]
    return df, diag


# ------------------------------------------------------------------- report
def _fmt(x, nd=3):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    latest = df["obs_date"].max()
    g = df.groupby("series")
    t = pd.DataFrame({
        "start": g["obs_date"].min().dt.date, "end": g["obs_date"].max().dt.date,
        "n": g.size(), "median": g["value"].median(), "min": g["value"].min(),
        "max": g["value"].max(), "last": g["value"].last(),
    })
    t["live"] = (latest - g["obs_date"].max()).dt.days <= LIVE_DAYS
    return t


def write_report(df: pd.DataFrame, diag: dict, path: Path = REPORT_FILE) -> None:
    t = summary_table(df)
    latest = df["obs_date"].max().date()
    L = []
    L.append("# Implied-volatility predictors (`implied` track)\n")
    L.append("Built by `python -m volmodel.data.sources.implied` -> "
             "`data/processed/predictors/implied.parquet` (long: `obs_date, available_date, "
             "series, value`; end-of-day closes, so `available_date = obs_date`). "
             f"Newest close in the file: **{latest}**. A series is *live* if its last close is "
             f"within {LIVE_DAYS} calendar days of that.\n")
    L.append("Units: all vol indices are **annualized vol in decimals** (VIX 20 -> 0.20). "
             "Exceptions: `iv_move` is a **yield (bp) vol in decimals** (MOVE 100 -> 0.0100; "
             "x modified duration ~ price vol); `iv_skew_index` is the **SKEW level** (not a vol); "
             "`*_term_slope` are ratios minus one; `iv_vvix` is the vol of VIX.\n")
    L.append("## Series\n")
    L.append("| series | description | unit | start | end | n | live | median | min | max | source | notes |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, r in t.iterrows():
        if name in BY_SERIES:
            sp = BY_SERIES[name]
            desc, unit, notes = sp.name, sp.unit, sp.notes
            d = diag.get(name, {})
            src = "; ".join(d.get("sources", []))
            extra = []
            if d.get("spikes"):
                extra.append("bad prints dropped: " + ", ".join(str(x) for x in d["spikes"]))
            if d.get("stale"):
                extra.append(f"{d['stale']} stale repeats dropped")
            if d.get("nonpositive"):
                extra.append(f"{d['nonpositive']} non-positive dropped")
            notes = " ".join([notes] + [e + "." for e in extra]).strip()
        else:
            desc, unit = DERIVED_NOTES[name]
            src, notes = "derived", ""
            if name == "iv_vix_vxo_spliced" and name in diag:
                notes = (f"k = {diag[name]['k']:.4f} (1990-94 overlap, n={diag[name]['overlap_n']}, "
                         f"corr {diag[name]['corr']:.3f}); spliced VXO part "
                         f"{diag[name]['pre_range'][0]}..{diag[name]['pre_range'][1]}.")
        L.append(f"| `{name}` | {desc} | {unit} | {r.start} | {r.end} | {r.n} | "
                 f"{'live' if r.live else '**dead**'} | {r['median']:.4g} | {r['min']:.4g} | "
                 f"{r['max']:.4g} | {src} | {notes} |")
    missing = [s.series for s in SPECS if s.series not in t.index]
    if missing:
        L.append(f"\nNot built (no data obtainable here): {', '.join('`'+m+'`' for m in missing)}.\n")

    no_y = [sp.yahoo for sp in SPECS if sp.yahoo and not diag.get(sp.series, {}).get("yahoo_available")]
    if no_y:
        L.append("\n> **Yahoo gap.** No Yahoo response was available (not in cache, and the chart API "
                 "answered HTTP 429 for the whole build session) for: " + ", ".join(no_y) + ". "
                 "Consequences: `iv_move` is absent; VIX3M/VXN/RVX/VXD/GVZ/OVX start at their Cboe CDN "
                 "dates (Sep 2009) instead of their earlier Yahoo history; VXO ends 2021-01-07 (Cboe "
                 "archive) rather than at its 2021 discontinuation. Re-running the module when Yahoo "
                 "responds fills these automatically (Yahoo data are spliced in only after the overlap "
                 "check below passes).\n")
    L.append("\n## Cboe vs Yahoo on the overlap\n")
    L.append("Levels in published index points. Yahoo is used to extend a Cboe series only "
             "when it agrees (n >= 250, corr > 0.99, median |diff| < 2% of the median level).\n")
    L.append("| series | Yahoo symbol | Yahoo range | overlap n | corr | mean diff | median abs diff | share abs diff > 0.1 | used to extend |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for sp in SPECS:
        d = diag.get(sp.series, {})
        if not sp.yahoo:
            continue
        yr = d.get("yahoo_range")
        yr_s = f"{yr[0]}..{yr[1]} ({yr[2]})" if yr else "unavailable"
        o = d.get("overlap", {})
        used = any(s.startswith("yahoo") for s in d.get("sources", []))
        L.append(f"| `{sp.series}` | {sp.yahoo} | {yr_s} | {o.get('n', '')} | {_fmt(o.get('corr'), 4)} | "
                 f"{_fmt(o.get('mean_diff'))} | {_fmt(o.get('median_abs_diff'))} | "
                 f"{_fmt(o.get('share_abs_gt_0.1'))} | {'yes' if used else 'no'} |")

    L.append("\n## Mapping to the 12 assets\n")
    L.append("| asset | natural implied predictor | from | live | comment |")
    L.append("|---|---|---|---|---|")
    for asset, cands, comment in ASSET_MAP:
        cands = (cands,) if isinstance(cands, str) else (cands or ())
        ser = next((c for c in cands if c in t.index), None)
        if ser:
            L.append(f"| {asset} | `{ser}` | {t.loc[ser, 'start']} | "
                     f"{'live' if t.loc[ser, 'live'] else 'dead'} | {comment} |")
        else:
            L.append(f"| {asset} | none | | | {comment} |")
    L.append(METHOD_NOTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


METHOD_NOTES = """
## Methodology breaks and caveats

* **VIX redefinition (2003-09-22).** Cboe switched VIX from the S&P 100
  at-the-money, trading-day formula to the S&P 500 model-free (full strip,
  calendar-day) formula and back-calculated the new VIX to 1990. The old index
  was renamed **VXO**. `iv_vix` is the new definition throughout; `iv_vxo` is
  the old one (1986-2021). VXO's trading-day convention inflates its level
  relative to a calendar-day measure, while VIX's full strip adds the skew
  premium; on 1990-2003 the two largely offset (VXO/VIX ~ 1.04 on average,
  corr ~ 0.98). VXO prints 1.50 on 1987-10-19: kept, it is a real close.
* **`iv_vix_vxo_spliced`** is VIX from 1990, extended to 1986 with VXO x k.
  It exists only for the 1986-1989 pre-history (it includes the 1987 crash);
  it is a level rescale, not a re-computation, and should be used with that
  in mind.
* **Retrospective history.** VIX1Y, VIX6M, VXEFA and VIX9D have long stretches
  where open = high = low = close, i.e. only a daily close was computed
  (from historical option quotes) before live dissemination. They are not
  look-ahead (computed from data available on each day) but the index did not
  exist in real time; VIX1Y in particular is close-only until ~2018-04.
* **Cboe CDN history starts.** The CDN files for VIX3M, VXN, RVX, VXD, GVZ,
  OVX start in Sep 2009 even though the indices were computed earlier
  (VXV/VIX3M from 2007-12, VXN 2001, RVX 2004, VXD 1997, GVZ 2008-06,
  OVX 2007-05). Earlier history is taken from Yahoo only where Yahoo matches
  Cboe on the overlap (see table above). VXN was originally computed with
  the old VXO-style formula and later restated by Cboe with the VIX formula;
  whether Yahoo's pre-2009 VXN is the restated series could not be verified.
* **ETF vol indices (VXIEF, VXLQD, VXHYG).** Thinly traded underlying options:
  noisy, stale runs, and a multi-year hole (Cboe stopped and later resumed
  calculating them: VXHYG has no data 2021-09..2025-05, VXIEF/VXLQD 2022-11..2025-07). Treat as live but
  unreliable; do not use them to define an estimation sample.
* **MOVE** is a normal (yield) vol of 1-month options on 2/5/10/30y
  Treasuries; it is not a price vol and its scale differs from VXTLT.
* **OVX** spiked to 3.25 on 2020-04-21 (negative WTI): genuine, kept.
* **Holidays.** Cboe files contain trading days only (no stale holiday rows);
  missing business days per year match the NYSE holiday count. Yahoo rows with
  null closes are dropped. Nothing is filled or interpolated.
* **Not obtainable here.** On the Cboe CDN these return 403: VXO_History
  (VXO comes from the archive files instead), EVZ (euro FX vol, dead),
  VXV/VXMT/VXST legacy names, TYVIX/VXTYN (10-yr note vol, discontinued
  2020), SRVIX (swaption vol), VXXLE, and per-index "archive" files for
  VXN/RVX/VXD. Not attempted on Yahoo because the chart API was rate-limited.
* **Residual noise.** The bad-print filter is deliberately conservative; thin
  series (VXHYG, VXIEF, VXLQD, VXEFA, VXD) still contain day-to-day noise and
  some two-day bad patches. For monthly forecasting prefer month averages or
  medians of daily values over a single month-end print.
"""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true", help="re-download everything")
    ap.add_argument("--refresh-cboe", action="store_true")
    ap.add_argument("--refresh-yahoo", action="store_true")
    ap.add_argument("--yahoo-cache-only", action="store_true",
                    help="use only Yahoo responses already cached; never call Yahoo")
    a = ap.parse_args(argv)
    if a.yahoo_cache_only:
        global _YAHOO_DOWN
        _YAHOO_DOWN = True
    df, diag = build(refresh_cboe=a.refresh or a.refresh_cboe,
                     refresh_yahoo=a.refresh or a.refresh_yahoo)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)
    write_report(df, diag)
    print(f"\nwrote {OUT_FILE} ({len(df)} rows, {df['series'].nunique()} series) and {REPORT_FILE}")
    print(summary_table(df).to_string())


if __name__ == "__main__":
    main()
