"""Broad US TIPS (``tips``, TIP-ETF-like): longest credible total-return history.

Run ``python -m volmodel.data.sources.tips`` to (re)build

* ``data/processed/assets/tips_{daily,monthly}.parquet``
* ``data/processed/predictors/tips_real_yields.parquet``
* ``reports/data/tips.md`` (sources, segments, overlap diagnostics, vol tables)

Raw downloads are cached under ``data/raw/tips/``; pass ``--refresh`` to
re-download them.

Construction
------------
TIPS were first auctioned in January 1997, so nothing before then is credible
(and no pre-1997 history is synthesized). Daily segments:

* 1997-01-30 .. 2003-12-31  PIMCO Real Return Instl (PRRIX), daily NAV with
  its month-end distributions re-spread as daily accruals (see below) -> tier B
* 2004-01-02 .. today       iShares TIPS Bond ETF (TIP), Yahoo close + dividend
  events (ex-date reinvestment, == adjclose except where Yahoo duplicates a
  dividend, e.g. TIP 2008-04-03), every distribution validated against the
  ex-date price move (``distributions.validate_distributions``) -> tier A

Tiers (project-wide): A = daily price of an exchange-traded fund (market
close); B = daily open-end mutual-fund NAV, index, or reconstruction; C =
monthly index total return; D = monthly model-based reconstruction.

Confirmed bad NAV prints are dropped before the returns are computed
(``BAD_PRICES``: PRRIX 2000-12-12, +0.84% then -0.99% with ACITX and VIPSX
unchanged); the next return spans the gap.

The monthly file is the compounded daily series (first full month 1997-02; the
current incomplete month is left out).

Why PRRIX and why the accrual fix: PIMCO (and American Century) bond funds
*declare dividends daily and pay them monthly*, so the NAV does not drop when a
month-end distribution is paid. Yahoo's adjclose nevertheless books the whole
month's income as a one-day return on the payment date, which puts a spurious
+0.3..1.4% spike on every month-end -- harmless for monthly returns, but it
inflates daily variance by a large fraction in the low-vol late 1990s. We
rebuild the total return from the raw NAV: distributions paid on the last
trading day of a month are treated as income accrued evenly per calendar day
over that month (weekend/holiday days go to the next trading day), so the
monthly total return is preserved exactly up to intra-month reinvestment;
other distributions (the December capital-gain/inflation specials) do drop
the NAV on their ex-date and are kept as ex-date cash flows. The diagnostics
in the report verify both classifications from the NAV behaviour.

Rejected / diagnostic-only sources (numbers in the report):

* VIPSX (Vanguard IPS, 2000-06-29+): Yahoo is missing most of its 2000-2002
  quarterly dividends (NAV drops of ~1% with no dividend), so its adjclose
  understates return; used only as a post-2004 diagnostic vs TIP.
* ACITX (American Century Inflation-Adjusted Bond, 1997-02+): daily-accrual
  distributions *and* several one-day bad NAV prints in 1997-98 that reverse
  the next day; diagnostic only.
* GSW (Fed, feds200805) zero-coupon real curve + CPI-U accrual: a daily
  reconstruction for 1999+ is computed only to validate the fund segment. It
  does not extend the history (the funds start in 1997) and its daily tracking
  is weak because the early curve fits are noisy.

Every return is a SIMPLE TOTAL return (distributions reinvested) in decimals,
not excess. Nothing is interpolated.

Predictors (``tips_real_yields``): daily GSW zero-coupon real yields and
breakevens (inflation compensation) at 5y and 10y, in PERCENT, continuously
compounded, as published. ``available_date`` = ``obs_date`` + 10 business days
(the Fed posts the file weekly-ish with a lag of about one week; on 2026-11-14
the file ended 2026-11-06). The history is the current vintage, which the Fed
can refit/revise without notice.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ...paths import ASSET_DIR, PREDICTOR_DIR, RAW, REPORTS
from . import distributions as dist

ASSET = "tips"
CACHE = RAW / "tips"
REPORT = REPORTS / "data" / "tips.md"
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
PLAIN_UA = "python-requests/2 (volmodel data loader)"

YAHOO_URL = "https://{host}.finance.yahoo.com/v8/finance/chart/{sym}"
GSW_TIPS_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200805.csv"
CPI_URL = "https://api.db.nomics.world/v22/series/BLS/cu/CUUR0000SA0"

# Daily segments: (symbol, first return date used, last return date used or None)
SEGMENTS = [("PRRIX", "1997-01-30", "2003-12-31"), ("TIP", "2004-01-02", None)]
DIAG_SYMBOLS = ["ACITX", "VIPSX", "VAIPX", "SCHP", "BPRIX"]
TIER = {"PRRIX": "B", "TIP": "A"}
SOURCE = {
    "PRRIX": "yahoo:PRRIX NAV + distributions, month-end distributions spread as daily accrual "
             "(PIMCO Real Return Instl, active)",
    "TIP": "yahoo:TIP close + dividends (iShares TIPS Bond ETF)",
}
NOTES = {
    "PRRIX": "Earliest TIPS fund (inception 1997-01-29). Daily-declared dividends re-spread over "
             "their accrual month; December specials kept on ex-date. Active manager: some "
             "non-TIPS exposure. Tier B: open-end fund NAV (also an accrual reconstruction and an active "
             "mandate). Bad NAV print 2000-12-12 (+0.84%/-0.99%, ACITX and VIPSX flat) dropped: the 12-13 "
             "return spans it.",
    "TIP": "Market price (not NAV): includes premium/discount noise, notably 2008 Q4. Every Yahoo distribution "
           "validated by the ex-date price test against GSW real-yield changes and VIPSX/SCHP (none rejected; "
           "Yahoo's duplicate of 2008-04-01 on 2008-04-03 removed).",
}
PREDICTOR_SERIES = {  # GSW mnemonic -> series name (percent, cont. compounded zero-coupon)
    "TIPSY05": "real_yield_5y",
    "TIPSY10": "real_yield_10y",
    "BKEVEN05": "breakeven_5y",
    "BKEVEN10": "breakeven_10y",
}
GSW_PUBLICATION_LAG_BDAYS = 10
# Confirmed bad NAV prints (price dropped; the next return spans it). PRRIX 2000-12-12:
# +0.84% then -0.99% the next day while ACITX and VIPSX were unchanged (audit C-2).
BAD_PRICES = {"PRRIX": ["2000-12-12"]}
# TIP's distributions follow monthly CPI accruals and swing several-fold from month to
# month, so their size relative to neighbours is not evidence: only an overwhelming
# price-test residual can reject one (``ratio_reject`` disabled).
TIP_VALIDATION = dict(ratio_reject=np.inf)
TIP_PEERS = ["VIPSX", "SCHP"]          # total returns (adjclose) of other TIPS funds


# --------------------------------------------------------------------- fetch
def _get(url: str, params: dict | None = None, ua: str = BROWSER_UA,
         tries: int = 6, sleep: float = 2.0, timeout: int = 90) -> requests.Response:
    last: Exception | None = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": ua, "Accept": "*/*"},
                             timeout=timeout)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {url}")
        except requests.RequestException as e:  # pragma: no cover - network
            last = e
        time.sleep(sleep * (2 ** i))
    raise RuntimeError(f"download failed: {url}: {last}")


def _cached(name: str, refresh: bool) -> Path | None:
    p = CACHE / name
    return p if p.exists() and p.stat().st_size > 0 and not refresh else None


def fetch_yahoo(sym: str, refresh: bool = False) -> Path:
    """Full daily history of ``sym`` from the Yahoo chart API, cached as JSON."""
    name = f"yahoo_{sym}.json"
    if (p := _cached(name, refresh)):
        return p
    CACHE.mkdir(parents=True, exist_ok=True)
    params = {"period1": 0, "period2": int(time.time()) + 86400, "interval": "1d",
              "events": "div,splits", "includeAdjustedClose": "true"}
    last = None
    for host in ("query1", "query2"):
        try:
            r = _get(YAHOO_URL.format(host=host, sym=sym), params=params)
            j = r.json()
            if j["chart"]["result"] and j["chart"]["result"][0].get("timestamp"):
                (CACHE / name).write_text(r.text)
                time.sleep(1.5)  # be polite: other agents share this endpoint
                return CACHE / name
            last = RuntimeError(f"empty Yahoo result for {sym}")
        except Exception as e:  # pragma: no cover - network
            last = e
    raise RuntimeError(f"Yahoo download failed for {sym}: {last}")


def fetch_gsw(refresh: bool = False) -> Path:
    name = "feds200805.csv"
    if (p := _cached(name, refresh)):
        return p
    CACHE.mkdir(parents=True, exist_ok=True)
    r = _get(GSW_TIPS_URL, ua=BROWSER_UA, timeout=180)
    (CACHE / name).write_bytes(r.content)
    return CACHE / name


def fetch_cpi(refresh: bool = False) -> Path:
    """CPI-U NSA (BLS CUUR0000SA0) via DBnomics; used only for the GSW reconstruction."""
    name = "cpi_u_nsa.json"
    if (p := _cached(name, refresh)):
        return p
    CACHE.mkdir(parents=True, exist_ok=True)
    r = _get(CPI_URL, params={"observations": 1, "format": "json"}, ua=PLAIN_UA)
    (CACHE / name).write_text(r.text)
    return CACHE / name


# --------------------------------------------------------------------- parse
def parse_yahoo(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Return (frame with close/adj/div by exchange-local date, dividend series)."""
    res = json.loads(Path(path).read_text())["chart"]["result"][0]
    tz = res["meta"].get("exchangeTimezoneName", "America/New_York")

    def to_date(sec):
        return (pd.to_datetime(sec, unit="s", utc=True).tz_convert(tz)
                .tz_localize(None).normalize())

    idx = to_date(res["timestamp"])
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"close": q["close"], "adj": res["indicators"]["adjclose"][0]["adjclose"]},
                      index=idx, dtype=float)
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"])
    divs = res.get("events", {}).get("dividends", {})
    div = pd.Series({to_date(int(k)): float(v["amount"]) for k, v in divs.items()},
                    dtype=float).sort_index()
    div = div.groupby(level=0).sum()
    # Yahoo occasionally repeats a distribution a few days later (TIP: 0.632 on
    # 2008-04-01 and again 2008-04-03, with a single NAV drop). Drop exact repeats.
    if len(div) > 1:
        gap = div.index.to_series().diff().dt.days
        dup = (gap <= 5) & np.isclose(div.values, div.shift(1).values)
        div = div[~dup.values]
    df["div"] = div.reindex(df.index).fillna(0.0)
    unmatched = div.index.difference(df.index)
    if len(unmatched):
        # Never silently drop a distribution: book it on the next trading day.
        for d in unmatched:
            nxt = df.index[df.index > d]
            if len(nxt):
                df.loc[nxt[0], "div"] += div[d]
    return df, div


def parse_gsw(path: Path) -> pd.DataFrame:
    lines = Path(path).read_text().splitlines()
    hdr = next(i for i, ln in enumerate(lines) if ln.startswith("Date,"))
    g = pd.read_csv(path, skiprows=hdr, index_col=0, na_values=["NA", ""])
    g.index = pd.to_datetime(g.index)
    g.index.name = "date"
    return g.sort_index().astype(float)


def parse_cpi(path: Path) -> pd.Series:
    d = json.loads(Path(path).read_text())["series"]["docs"][0]
    s = pd.Series(d["value"], index=pd.PeriodIndex(d["period"], freq="M"))
    return pd.to_numeric(s, errors="coerce").dropna()


# ----------------------------------------------------------------- returns
def adjclose_return(df: pd.DataFrame) -> pd.Series:
    """Yahoo-style total return: distributions reinvested on their ex/pay date.

    Computed from close + (de-duplicated) dividend events rather than Yahoo's
    adjclose, which carries duplicated dividends through (TIP 2008-04-03).
    """
    return (df["close"] + df["div"]) / df["close"].shift(1) - 1


def accrual_total_return(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Total return for a fund whose regular distributions accrue daily.

    Distributions paid on the last trading day of a calendar month are spread
    over that month's trading days in proportion to calendar days elapsed since
    the previous trading day (days after the last trading day are booked on
    it). Other distributions are treated as ex-date cash flows. Returns the
    daily return series and a table classifying each distribution.
    """
    idx = df.index
    month = idx.to_period("M")
    last_td = pd.Series(idx, index=idx).groupby(month).transform("max")
    is_last = pd.Series(idx == last_td.values, index=idx)
    paid = df["div"]
    accr_amt = paid.where(is_last, 0.0)
    exdate_amt = paid.where(~is_last, 0.0)

    prev = pd.Series(idx, index=idx).shift(1)
    month_start = idx.to_period("M").to_timestamp()
    start = np.maximum(prev.fillna(month_start[0] - pd.Timedelta(days=1)).values,
                       (month_start - pd.Timedelta(days=1)).values)
    cal_days = pd.Series((idx.values - start) / np.timedelta64(1, "D"), index=idx)
    month_end = idx.to_period("M").to_timestamp(how="end").normalize()
    cal_days = cal_days + np.where(is_last, (month_end - idx).days, 0)
    w = cal_days / cal_days.groupby(month).transform("sum")
    month_div = accr_amt.groupby(month).transform("sum")
    accrued = month_div * w

    ret = (df["close"] + accrued + exdate_amt) / df["close"].shift(1) - 1
    pr = df["close"].pct_change()
    cls = pd.DataFrame({
        "amount": paid[paid > 0],
        "yield": (paid / df["close"].shift(1))[paid > 0],
        "nav_ret_on_day": pr[paid > 0],
        "treated_as": np.where(is_last[paid > 0], "daily_accrual", "ex_date"),
    })
    return ret, cls


def gsw_zero_yield(g: pd.DataFrame, n) -> pd.Series:
    """Svensson zero-coupon real yield (decimal, cont. comp.) at maturity ``n`` years.

    ``n`` may be a scalar or a Series aligned with ``g``.
    """
    b0, b1, b2, b3 = (g[c] for c in ("BETA0", "BETA1", "BETA2", "BETA3"))
    t1, t2 = g["TAU1"], g["TAU2"]
    x = n / t1
    y = b0 + b1 * (1 - np.exp(-x)) / x + b2 * ((1 - np.exp(-x)) / x - np.exp(-x))
    z = n / t2
    sv = (b3 * ((1 - np.exp(-z)) / z - np.exp(-z))).fillna(0.0)
    return (y + sv) / 100


def reference_cpi(dates: pd.DatetimeIndex, cpi: pd.Series) -> pd.Series:
    """TIPS reference CPI: CPI(M-3) + (d-1)/days(M) * (CPI(M-2) - CPI(M-3))."""
    m = dates.to_period("M")
    a = cpi.reindex(m - 3).values
    b = cpi.reindex(m - 2).values
    frac = (dates.day - 1) / dates.days_in_month
    return pd.Series(a + frac * (b - a), index=dates)


def gsw_reconstruction(g: pd.DataFrame, cpi: pd.Series, maturity: float) -> pd.Series:
    """Daily total return of a constant-maturity zero-coupon TIPS (real return x index ratio)."""
    g = g.dropna(subset=["BETA0", "BETA1", "BETA2", "TAU1"])
    dates = g.index
    dt = pd.Series(dates, index=dates).diff().dt.days / 365.25
    y_prev = gsw_zero_yield(g, maturity).shift(1)
    y_now = gsw_zero_yield(g, maturity - dt)
    real_lr = y_prev * maturity - y_now * (maturity - dt)
    rc = reference_cpi(dates, cpi)
    infl_lr = np.log(rc / rc.shift(1))
    return np.expm1(real_lr + infl_lr).rename(f"gsw_{maturity:g}y")


# ------------------------------------------------------------- diagnostics
def overlap(a: pd.Series, b: pd.Series, start=None, end=None) -> dict:
    """Overlap statistics of candidate ``a`` vs reference ``b`` (daily simple returns)."""
    x = pd.concat([a, b], axis=1, sort=True).loc[start:end].dropna()
    x.columns = ["a", "b"]
    out: dict = {"n_days": len(x)}
    if len(x) < 40:
        return out
    out.update(start=x.index[0].date(), end=x.index[-1].date(),
               corr_d=x.a.corr(x.b), vol_ratio_d=x.a.std() / x.b.std(),
               te_d=(x.a - x.b).std() * np.sqrt(252),
               vol_a_d=x.a.std() * np.sqrt(252), vol_b_d=x.b.std() * np.sqrt(252))
    l3 = np.log1p(x).rolling(3).sum().dropna()
    out["corr_3d"] = l3.a.corr(l3.b)
    m = (1 + x).resample("ME").prod() - 1
    m = m[x.resample("ME").size() >= 15]
    out["n_months"] = len(m)
    if len(m) >= 6:
        out.update(corr_m=m.a.corr(m.b), vol_ratio_m=m.a.std() / m.b.std(),
                   te_m=(m.a - m.b).std() * np.sqrt(12),
                   vol_a_m=m.a.std() * np.sqrt(12), vol_b_m=m.b.std() * np.sqrt(12),
                   ret_diff_ann=(m.a - m.b).mean() * 12)
    return out


def empirical_duration(r: pd.Series, y: pd.Series) -> pd.Series:
    """Per calendar year: -beta of weekly returns on weekly change in yield (decimal)."""
    x = pd.concat([np.log1p(r).rename("r"), y.rename("y")], axis=1, sort=True).dropna()
    wk = pd.DataFrame({"r": x.r.resample("W-WED").sum(),
                       "dy": x.y.resample("W-WED").last().diff()}).dropna()
    def beta(g):
        return -np.cov(g.r, g.dy)[0, 1] / g.dy.var() if len(g) > 20 else np.nan
    return wk.groupby(wk.index.year).apply(beta)


# ------------------------------------------------------------------- build
def load_all(refresh: bool = False) -> dict:
    data = {}
    for sym in [s for s, *_ in SEGMENTS] + DIAG_SYMBOLS:
        data[sym] = parse_yahoo(fetch_yahoo(sym, refresh))
    data["gsw"] = parse_gsw(fetch_gsw(refresh))
    try:
        data["cpi"] = parse_cpi(fetch_cpi(refresh))
    except Exception as e:  # pragma: no cover - diagnostic only
        print(f"warning: CPI unavailable ({e}); skipping GSW reconstruction")
        data["cpi"] = None
    return data


def validated_return(df: pd.DataFrame, data: dict) -> tuple[pd.Series, pd.DataFrame]:
    """ETF total return with every distribution validated against the ex-date move
    relative to GSW real-yield changes and other TIPS funds' total returns."""
    g = data["gsw"]
    peers = pd.DataFrame({"ry5": g["TIPSY05"].diff() / 100, "ry10": g["TIPSY10"].diff() / 100,
                          **{p: data[p][0]["adj"].pct_change() for p in TIP_PEERS}})
    div, table = dist.validate_distributions(df["close"], df["div"], peers, **TIP_VALIDATION)
    return dist.total_return(df["close"], div), table


def build_daily(data: dict) -> tuple[pd.DataFrame, dict]:
    pieces, extra = [], {}
    for sym, start, end in SEGMENTS:
        df, _ = data[sym]
        df, dropped = dist.drop_bad_prices(df, BAD_PRICES.get(sym, []))
        extra[f"{sym}_dropped"] = dropped
        if sym == "PRRIX":
            r, cls = accrual_total_return(df)
            extra["prrix_classification"] = cls
        else:
            r, extra[f"{sym}_div_table"] = validated_return(df, data)
        r = r.loc[start:end].dropna()
        pieces.append(pd.DataFrame({"ret": r, "source": SOURCE[sym], "tier": TIER[sym]}))
    daily = pd.concat(pieces)
    daily.index.name = "date"
    assert daily.index.is_monotonic_increasing and not daily.index.duplicated().any()
    return daily, extra


def build_monthly(daily: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize() if today is None else today
    g = daily["ret"].groupby(daily.index.to_period("M"))
    ret = np.exp(np.log1p(daily["ret"]).groupby(daily.index.to_period("M")).sum()) - 1
    first_day = g.apply(lambda s: s.index.min())
    src = daily.groupby(daily.index.to_period("M"))["source"].agg(lambda s: " + ".join(dict.fromkeys(s)))
    tier = daily.groupby(daily.index.to_period("M"))["tier"].agg(lambda s: max(s))  # worst tier
    m = pd.DataFrame({"ret": ret, "source": src, "tier": tier})
    # drop the first month if it starts after its first trading days (partial) and current month
    first = m.index[0]
    if first_day[first].day > 7:
        m = m.drop(first)
    cur = today.to_period("M")
    last_obs = daily.index.max()
    if m.index[-1] == cur or last_obs < (m.index[-1].to_timestamp(how="end") - pd.offsets.BDay(1)).normalize():
        m = m.drop(m.index[-1])
    m.index = m.index.to_timestamp(how="end").normalize()
    m.index.name = "date"
    return m


def build_predictors(g: pd.DataFrame) -> pd.DataFrame:
    cols = list(PREDICTOR_SERIES)
    w = g[cols].rename(columns=PREDICTOR_SERIES)
    long = (w.reset_index().melt(id_vars="date", var_name="series", value_name="value")
            .dropna(subset=["value"]).rename(columns={"date": "obs_date"}))
    long["available_date"] = long["obs_date"] + pd.offsets.BDay(GSW_PUBLICATION_LAG_BDAYS)
    long = long[["obs_date", "available_date", "series", "value"]]
    return long.sort_values(["series", "obs_date"]).reset_index(drop=True)


def segments_table(daily: pd.DataFrame) -> pd.DataFrame:
    """Rows in the shape of ``data/processed/segments.csv`` (for the assembler)."""
    rows = []
    for sym, *_ in SEGMENTS:
        s = daily[daily["source"] == SOURCE[sym]]
        rows.append({"asset": ASSET, "start": s.index.min().date(), "end": s.index.max().date(),
                     "freq": "D", "source": SOURCE[sym], "tier": TIER[sym], "notes": NOTES[sym]})
    return pd.DataFrame(rows)


def diagnostics(data: dict, daily: pd.DataFrame) -> dict:
    r = {s: adjclose_return(data[s][0]) for s in [s for s, *_ in SEGMENTS] + DIAG_SYMBOLS}
    r["PRRIX_fix"] = accrual_total_return(data["PRRIX"][0])[0]
    r["ACITX_fix"] = accrual_total_return(data["ACITX"][0])[0]
    out = {"returns": r}
    pairs = [
        ("PRRIX_fix vs TIP (splice overlap, 2004-2006)", "PRRIX_fix", "TIP", "2004-01", "2006-12"),
        ("PRRIX_fix vs TIP (2004-2026)", "PRRIX_fix", "TIP", "2004-01", None),
        ("PRRIX_fix vs ACITX_fix (1997-02..2003)", "PRRIX_fix", "ACITX_fix", "1997-02", "2003-12"),
        ("PRRIX_fix vs VIPSX (2000-07..2003; VIPSX divs missing)", "PRRIX_fix", "VIPSX", "2000-07", "2003-12"),
        ("PRRIX ex-date TR (=adjclose) vs PRRIX_fix (1997-2003)", "PRRIX", "PRRIX_fix", "1997-02", "2003-12"),
        ("ACITX_fix vs TIP (2004-2026)", "ACITX_fix", "TIP", "2004-01", None),
        ("VIPSX vs TIP (2004-2026)", "VIPSX", "TIP", "2004-01", None),
        ("VIPSX vs TIP (2004-2007)", "VIPSX", "TIP", "2004-01", "2007-12"),
        ("VIPSX vs TIP (2008-2012)", "VIPSX", "TIP", "2008-01", "2012-12"),
        ("VIPSX vs TIP (2013-2026)", "VIPSX", "TIP", "2013-01", None),
        ("VAIPX vs TIP (2005-07..2026)", "VAIPX", "TIP", "2005-07", None),
        ("SCHP vs TIP (2010-09..2026)", "SCHP", "TIP", "2010-09", None),
    ]
    if data.get("cpi") is not None:
        for n in (7, 10):
            r[f"GSW_{n}y"] = gsw_reconstruction(data["gsw"], data["cpi"], n)
        pairs += [
            ("PRRIX_fix vs GSW_10y recon (1999-02..2003)", "PRRIX_fix", "GSW_10y", "1999-02", "2003-12"),
            ("PRRIX_fix vs GSW_7y recon (1999-02..2003)", "PRRIX_fix", "GSW_7y", "1999-02", "2003-12"),
            ("ACITX_fix vs GSW_10y recon (1999-02..2003)", "ACITX_fix", "GSW_10y", "1999-02", "2003-12"),
            ("VIPSX vs GSW_10y recon (2000-07..2003)", "VIPSX", "GSW_10y", "2000-07", "2003-12"),
            ("TIP vs GSW_10y recon (2004-2024)", "TIP", "GSW_10y", "2004-01", "2024-12"),
        ]
    out["pairs"] = {name: overlap(r[a], r[b], s, e) for name, a, b, s, e in pairs}
    y7 = gsw_zero_yield(data["gsw"].dropna(subset=["BETA0", "TAU1"]), 7.0)
    out["duration"] = pd.DataFrame({k: empirical_duration(r[k], y7)
                                    for k in ["PRRIX_fix", "ACITX_fix", "VIPSX", "TIP", "SCHP"]})
    out["duration"]["spliced"] = empirical_duration(daily["ret"], y7)
    return out


# ------------------------------------------------------------------ report
def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.3f}"
    return str(v)


def _vol_table(daily: pd.Series, others: dict, periods) -> pd.DataFrame:
    rows = {}
    for label, a, b in periods:
        row = {}
        for name, s in {"tips (spliced)": daily, **others}.items():
            x = s.loc[a:b].dropna()
            row[name] = x.std() * np.sqrt(252) * 100 if len(x) > 15 else np.nan
        rows[label] = row
    return pd.DataFrame(rows).T


def write_report(daily, monthly, preds, extra, diag, data) -> str:
    seg = segments_table(daily)
    L = ["# tips — broad US TIPS total return (TIP-like)", "",
         f"Built by `python -m volmodel.data.sources.tips` on {pd.Timestamp.today().date()}.", "",
         "## Result", "",
         f"* daily: {daily.index[0].date()} .. {daily.index[-1].date()} ({len(daily)} obs)",
         f"* monthly: {monthly.index[0].date()} .. {monthly.index[-1].date()} ({len(monthly)} months;"
         " compounded daily; current incomplete month excluded)",
         "* returns are SIMPLE TOTAL returns in decimals (not excess).",
         "* TIPS were first issued in Jan 1997; nothing earlier is reported or synthesized.", "",
         "## Sources", "",
         "* Yahoo chart API (close + dividend events; Yahoo adjclose used only as a cross-check): TIP, PRRIX, "
         "ACITX, VIPSX, VAIPX, SCHP, BPRIX. Total return = (close_t + div_t)/close_{t-1} − 1. Yahoo "
         "repeats TIP's 2008-04-01 distribution (0.632) on 2008-04-03 with no second price drop; "
         "exact repeats within 5 days are dropped (it is the only one among the symbols used).",
         f"* Fed GSW TIPS curve `{GSW_TIPS_URL}` (Gürkaynak-Sack-Wright 2010): real zero curve "
         "parameters, zero real yields and breakevens, daily from 1999-01-04.",
         "* CPI-U NSA (BLS CUUR0000SA0) via DBnomics — only for the GSW reconstruction check.", "",
         "## Segments", "", "| start | end | freq | tier | source | notes |", "|---|---|---|---|---|---|"]
    for _, s in seg.iterrows():
        L.append(f"| {s.start} | {s.end} | {s.freq} | {s.tier} | {s.source} | {s.notes} |")
    L += ["", "Monthly file = compounded daily; first month 1997-02 (Jan 1997 has only 2 trading days "
          "of fund history and is dropped).", "",
          "## Distribution treatment (why PRRIX is rebuilt from NAV)", "",
          "PIMCO and American Century declare dividends daily and pay monthly: the NAV does not drop "
          "on month-end payment dates, yet Yahoo's adjclose books the month's income as a one-day "
          "return on that date. Classification of PRRIX distributions 1997-2003 (NAV return on the "
          "payment day should be ≈ −yield for ex-date distributions, ≈ 0 for accruals):", ""]
    cls = extra["prrix_classification"].loc[:"2004-01"]
    t = cls.groupby("treated_as").agg(n=("amount", "size"), mean_yield=("yield", "mean"),
                                      mean_nav_ret=("nav_ret_on_day", "mean"))
    L += ["| treated as | n | mean distribution yield | mean NAV return on pay day |", "|---|---|---|---|"]
    for k, v in t.iterrows():
        L.append(f"| {k} | {int(v.n)} | {v.mean_yield*100:.2f}% | {v.mean_nav_ret*100:.2f}% |")
    rr = diag["returns"]
    L += ["", "Effect on the 1997-2003 PRRIX daily vol (annualized): adjclose "
          f"{rr['PRRIX'].loc['1997-02':'2003'].std()*np.sqrt(252)*100:.2f}% vs rebuilt "
          f"{rr['PRRIX_fix'].loc['1997-02':'2003'].std()*np.sqrt(252)*100:.2f}%; "
          "monthly returns are essentially unchanged (see the adjclose-vs-fix row below).", "",
          "## Overlap diagnostics", "",
          "`a vs b`: corr / vol ratio (a/b) / tracking error (annualized) on daily, 3-day-overlapping "
          "log (corr only) and monthly returns; `ret diff` = annualized mean monthly return a − b.", "",
          "| pair | days | corr d | corr 3d | vol ratio d | TE d | months | corr m | vol ratio m | TE m | vol a m | vol b m | ret diff |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, d in diag["pairs"].items():
        L.append("| " + " | ".join([name] + [_fmt(d.get(k, "")) for k in
                 ["n_days", "corr_d", "corr_3d", "vol_ratio_d", "te_d", "n_months", "corr_m",
                  "vol_ratio_m", "te_m", "vol_a_m", "vol_b_m", "ret_diff_ann"]]) + " |")
    L += ["", "Reading: daily correlations between funds are depressed by NAV rounding to the cent "
          "(a $10 NAV has a 0.1% tick, comparable to a quiet day's move in 1997-2000) and by the ETF's "
          "price noise; 3-day and monthly correlations are the relevant check of co-movement and "
          "vol level.", "",
          "## Duration profile", "",
          "Empirical real duration = −β of weekly log returns on weekly changes in the GSW 7y zero "
          "real yield, per calendar year (noisy; a level check, not a precise duration).", "",
          _md(diag["duration"], 1), "",
          "## Annualized volatility by period (daily returns, %)", ""]
    others = {k: rr[k] for k in ["PRRIX_fix", "ACITX_fix", "VIPSX", "VAIPX", "SCHP", "TIP"]}
    if "GSW_10y" in rr:
        others["GSW_10y recon"] = rr["GSW_10y"]
    years = [(str(y), str(y), str(y)) for y in range(1997, daily.index[-1].year + 1)]
    episodes = [("2008-09..2008-12 (Lehman / TIPS liquidity)", "2008-09-01", "2008-12-31"),
                ("2013-05..2013-09 (taper tantrum)", "2013-05-01", "2013-09-30"),
                ("2020-03 (COVID dash for cash)", "2020-03-01", "2020-03-31"),
                ("2022 (inflation / hiking)", "2022-01-01", "2022-12-31")]
    vt = _vol_table(daily["ret"], others, years + episodes)
    L += [_md(vt, 2), ""]
    yr = monthly.index.year
    mvol = monthly["ret"].groupby(yr // 5 * 5).std() * np.sqrt(12) * 100
    span = pd.Series(yr, index=yr).groupby(yr // 5 * 5).agg(["min", "max"])
    L += ["Monthly-return vol of the spliced series by 5-year block (%): " +
          ", ".join(f"{span.loc[k, 'min']}-{span.loc[k, 'max']}: {v:.2f}" for k, v in mvol.items()), "",
          "## Predictors: `predictors/tips_real_yields.parquet`", "",
          "Long format `obs_date, available_date, series, value`. Series: " +
          ", ".join(f"`{v}` (GSW `{k}`)" for k, v in PREDICTOR_SERIES.items()) +
          ". Values in PERCENT, zero-coupon, continuously compounded, as published by the Fed. "
          f"`available_date = obs_date + {GSW_PUBLICATION_LAG_BDAYS} business days`: the Fed posts "
          "this staff research file roughly weekly with about a one-week lag (on 2026-11-14 the "
          "file's last observation was 2026-11-06), so two weeks is a conservative bound. "
          f"Rows: {len(preds)}, {preds.obs_date.min().date()} .. {preds.obs_date.max().date()}. "
          "The file is the current vintage (the Fed may refit history), and 1999-2003 fits rest on "
          "few TIPS issues and are noisy (e.g. 1999-01-06 2y breakeven jumps to 2.05%).", "",
          "## Caveats", "",
          "* 1997-2003 is an ACTIVE fund (PIMCO Real Return): mostly TIPS, but with some nominal, "
          "foreign linker and derivative exposure; its 2004+ monthly vol ratio vs TIP (table above) "
          "shows it ran somewhat more risk than the index ETF. ACITX (closer to a plain TIPS fund) "
          "has bad one-day NAV prints in 1997-98 and was rejected for the daily series.",
          "* The accrual re-spread assumes income accrues evenly per calendar day within the month. "
          "Real accruals vary with the CPI index-ratio path; the error is a few bp per day at most "
          "and does not affect monthly returns.",
          "* NAV quantization: 1997-2000 fund NAVs move in 0.1% ticks; ~35-40% of days show zero "
          "NAV change. Daily variance in that era is noisy (rounding adds roughly tick²/6 per day).",
          "* TIP is an ETF market price: in 2008 Q4 it traded at sizeable discounts/premia to NAV, so "
          "its daily vol then exceeds the Vanguard fund's NAV vol (see episode table). This is "
          "investable reality, not an error.",
          "* Calendar-year TR check vs Vanguard IPS (VIPSX, NAV): TIP − VIPSX is within ±0.4pp in "
          "most years, but +2.3pp in 2008 and −1.9pp in 2009 (TIP market price ended 2008 rich vs "
          "NAV; the 2-year totals agree within ~0.5pp), and +1.4pp in 2016 where SCHP agrees with "
          "TIP (VIPSX is missing Yahoo dividends that year). TIP 2013 −8.49%, 2022 −12.24%.",
          "* VIPSX on Yahoo misses most 2000-2002 dividends; its early overlap numbers understate "
          "its return (ret diff) but its vol/corr are still informative.",
          "* GSW reconstruction (constant-maturity zero, real return × CPI index ratio) validates "
          "the fund era at the monthly level only; daily correlations are low because early curve "
          "fits are noisy. It is not spliced in (funds cover 1997+ at a higher tier).",
          "* Zero days: the spliced daily series includes days with exactly 0 return "
          "(NAV unchanged); these are real observations, not fills.", ""]
    text = "\n".join(L)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)
    return text


def _md(df: pd.DataFrame, digits: int) -> str:
    """Minimal markdown table (avoids a tabulate dependency)."""
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = ["| " + " | ".join([str(i)] + ["" if pd.isna(v) else f"{v:.{digits}f}" for v in r]) + " |"
            for i, r in zip(df.index, df.values)]
    return "\n".join([head, sep] + rows)


def build(refresh: bool = False, report: bool = True) -> dict:
    data = load_all(refresh)
    daily, extra = build_daily(data)
    monthly = build_monthly(daily)
    preds = build_predictors(data["gsw"])
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTOR_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(ASSET_DIR / f"{ASSET}_daily.parquet")
    monthly.to_parquet(ASSET_DIR / f"{ASSET}_monthly.parquet")
    preds.to_parquet(PREDICTOR_DIR / "tips_real_yields.parquet", index=False)
    out = {"daily": daily, "monthly": monthly, "predictors": preds, "segments": segments_table(daily)}
    if report:
        diag = diagnostics(data, daily)
        write_report(daily, monthly, preds, extra, diag, data)
        out["diagnostics"] = diag
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-download raw files")
    args = ap.parse_args(argv)
    out = build(refresh=args.refresh)
    d, m = out["daily"], out["monthly"]
    print(f"tips daily  {d.index[0].date()} .. {d.index[-1].date()}  n={len(d)}")
    print(f"tips monthly {m.index[0].date()} .. {m.index[-1].date()}  n={len(m)}")
    print(out["segments"].to_string(index=False))
    for k, v in out["diagnostics"]["pairs"].items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    print(f"report: {REPORT}")


if __name__ == "__main__":
    main()
