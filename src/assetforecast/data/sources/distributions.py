"""Shared tools for building total returns from prices plus distributions.

Used by every track that holds a fund or ETF (credit, treasuries, reits,
intl_equity, tips, us_equity). Three problems are handled here, each as a
tested function rather than a list of hard-coded dates:

1. **Spurious or inflated vendor distributions** (``validate_distributions``).
   Yahoo's dividend events (and so its ``adjclose``) sometimes contain
   distributions that were never paid, or wrong amounts (IEF 2003: five; LQD
   2003-12-16). A distribution that was really paid lowers the price on its
   ex-date by its amount, so the TOTAL return that day, (close + div) /
   previous close - 1, is in line with what related markets did. A distribution
   the price never paid out leaves a positive residual of about its own size.
   Each listed distribution is therefore tested against peers (rate changes,
   related ETFs): a regression of the fund's total return on the peers is fitted
   on the non-ex-dates within +-``window`` trading days, and the ex-date residual
   is compared with the listed yield. When a second dividend record is
   available (Nasdaq for Nasdaq-listed ETFs) it is used to arbitrate.

2. **Daily-accrual funds** (``accrual_total_return``). Many bond funds declare
   income daily and pay it monthly: the NAV does not drop when the month-end
   distribution is paid, yet ``adjclose`` books a whole month of income on one
   day. Distributions on the last trading day of a month are spread over that
   month (by calendar days); the monthly total return is unchanged up to
   intra-month reinvestment. Distributions on other days, and the part of a
   month-end distribution well above the fund's regular income (a capital
   gain paid with the income), are ex-date cash flows.

3. **Bad prints** (``drop_bad_prices``): a confirmed bad price is removed so the
   next return spans it. Nothing is ever interpolated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ helpers
def total_return(close: pd.Series, div: pd.Series) -> pd.Series:
    """(close_t + div_t) / close_{t-1} - 1; dividends on their ex-date."""
    div = div.reindex(close.index).fillna(0.0)
    return (close + div) / close.shift(1) - 1


def drop_bad_prices(df: pd.DataFrame, dates) -> tuple[pd.DataFrame, list]:
    """Remove the rows (prices) of confirmed bad prints; the next return then spans
    the gap. A distribution on a dropped date is carried to the next row."""
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates))) if len(dates) else pd.DatetimeIndex([])
    hit = dates.intersection(df.index)
    out = df.copy()
    if "div" in out.columns:
        for d in hit:
            nxt = out.index[out.index > d]
            if len(nxt) and out.at[d, "div"] != 0:
                out.at[nxt[0], "div"] += out.at[d, "div"]
    return out.drop(hit), [str(d.date()) for d in hit]


def dedupe_repeats(div: pd.Series, days: int = 5) -> tuple[pd.Series, list]:
    """Drop exact repeats of a distribution within ``days`` calendar days (Yahoo
    occasionally lists one twice, e.g. TIP 2008-04-01 and 2008-04-03)."""
    div = div[div > 0].sort_index()
    if len(div) < 2:
        return div, []
    gap = div.index.to_series().diff().dt.days
    dup = (gap <= days) & np.isclose(div.values, div.shift(1).values)
    return div[~dup.values], [str(d.date()) for d in div.index[dup.values]]


def align_dividends(div: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Dividend amounts on the trading days of ``index``; an event dated on a
    non-trading day is booked on the next trading day (never dropped)."""
    out = pd.Series(0.0, index=index)
    for d, v in div.items():
        if d in out.index:
            out[d] += v
        else:
            nxt = index[index > d]
            if len(nxt):
                out[nxt[0]] += v
    return out


# ------------------------------------------------------ distribution checks
def price_test(close: pd.Series, div: pd.Series, peers: pd.DataFrame, window: int = 250,
               min_fit: int = 60) -> pd.DataFrame:
    """Ex-date test of every listed distribution against peers.

    For each date with ``div > 0``: OLS of the fund's total return on ``peers``
    (plus an intercept), fitted on days within +-``window`` trading days that are
    not ex-dates; ``resid`` = ex-date total return minus the fitted value,
    ``sd`` = robust (MAD) residual sd, ``z`` = resid / sd, and ``implied`` = the
    distribution amount the price behaviour supports (listed amount minus
    resid x previous close). A genuine distribution has resid ~ 0 (implied ~
    listed); one that was never paid has resid ~ +yield (implied ~ 0).
    """
    close = close.dropna()
    div = div.reindex(close.index).fillna(0.0)
    tr = total_return(close, div)
    X = peers.reindex(close.index)
    is_ex = div > 0
    rows = []
    pos = {d: i for i, d in enumerate(close.index)}
    for d in close.index[is_ex.values]:
        i = pos[d]
        if i == 0:
            continue
        prev = float(close.iloc[i - 1])
        amt = float(div.iloc[i])
        row = dict(date=d, amount=amt, prev_close=prev, yld=amt / prev, resid=np.nan, sd=np.nan,
                   z=np.nan, implied=np.nan, n_fit=0)
        w = slice(max(1, i - window), min(len(close), i + window + 1))
        y, xx = tr.iloc[w], X.iloc[w]
        ok = (y.notna() & xx.notna().all(axis=1) & ~is_ex.iloc[w]).values
        xd = X.iloc[i]
        if ok.sum() >= min_fit and xd.notna().all() and np.isfinite(tr.iloc[i]):
            A = np.c_[np.ones(ok.sum()), xx.values[ok]]
            beta, *_ = np.linalg.lstsq(A, y.values[ok], rcond=None)
            e = y.values[ok] - A @ beta
            sd = 1.4826 * np.median(np.abs(e - np.median(e)))
            res = float(tr.iloc[i] - np.r_[1.0, xd.values] @ beta)
            row.update(resid=res, sd=sd, z=res / sd if sd > 0 else np.nan,
                       implied=amt - res * prev, n_fit=int(ok.sum()))
        rows.append(row)
    cols = ["date", "amount", "prev_close", "yld", "resid", "sd", "z", "implied", "n_fit"]
    return pd.DataFrame(rows, columns=cols).set_index("date")


def validate_distributions(close: pd.Series, div: pd.Series, peers: pd.DataFrame,
                           second: pd.Series | None = None, z_reject: float = 3.0,
                           z_strong: float = 8.0, ratio_reject: float = 1.5,
                           share_reject: float = 0.3, band_sd: float = 2.0, window: int = 250,
                           ratio_events: int = 12, agree_abs: float = 0.001, agree_rel: float = 0.02,
                           match_days: int = 3) -> tuple[pd.Series, pd.DataFrame]:
    """Validate listed distributions; return (corrected dividends on ``close``'s
    index, table with one row per listed distribution and the action taken).

    Rules, in order:

    * second source available for the date (an entry within ``match_days``
      calendar days): if the amounts agree (within ``agree_abs`` or
      ``agree_rel``) the distribution is kept; if not, whichever amount is
      closer to the price-implied amount is used (``replaced:second_source``
      when the second source wins);
    * otherwise price behaviour and magnitude decide together. The ex-date
      residual must look like an unpaid distribution: more than
      ``share_reject`` of the listed yield is not reflected in the price, and
      the residual does not exceed the listed yield by more than ``band_sd``
      residual sds (a residual far larger than the distribution is an
      unexplained market move, not evidence about the distribution). Then the
      distribution is REJECTED if the residual is clear (z > ``z_reject``) and
      the amount is anomalous for this fund (more than ``ratio_reject`` x the
      median of its ``ratio_events`` neighbouring distributions on each side),
      or if the residual alone is overwhelming (z > ``z_strong``). Requiring
      both price and magnitude evidence keeps the false-rejection rate low for
      funds where the test has little power (a monthly bond-ETF payment is only
      2-3 residual sds). A rejected amount is replaced by the price-implied
      amount, floored at 0 (``replaced:price_implied``).

    Peers must be TOTAL returns (or price returns of daily-accrual funds, whose
    NAV never drops): a peer whose price drops on the same ex-dates would make
    every genuine distribution look spurious.
    """
    close = close.dropna()
    div = div.reindex(close.index).fillna(0.0)
    t = price_test(close, div, peers, window=window)
    a = t["amount"]
    t["ratio"] = [a.iloc[i] / a.drop(a.index[i]).iloc[max(0, i - ratio_events):i + ratio_events].median()
                  if len(a) > 1 else 1.0 for i in range(len(a))]
    t["second"] = np.nan
    t["action"] = "kept"
    t["used"] = t["amount"]
    sec = second.dropna().sort_index() if second is not None and len(second) else None
    for d, r in t.iterrows():
        s_amt = np.nan
        if sec is not None:
            near = sec[(sec.index >= d - pd.Timedelta(days=match_days)) &
                       (sec.index <= d + pd.Timedelta(days=match_days))]
            if len(near):
                s_amt = float(near.iloc[np.argmin(np.abs((near.index - d).days))])
        t.at[d, "second"] = s_amt
        if np.isfinite(s_amt):
            if abs(r.amount - s_amt) <= max(agree_abs, agree_rel * r.amount):
                t.at[d, "action"] = "kept:second_source_agrees"
            elif np.isfinite(r.implied) and abs(s_amt - r.implied) < abs(r.amount - r.implied):
                t.at[d, "action"] = "replaced:second_source"
                t.at[d, "used"] = s_amt
            else:
                t.at[d, "action"] = "kept:price_supports_listed_over_second_source"
            continue
        unpaid_like = (np.isfinite(r.z) and r.resid > share_reject * r.yld
                       and r.resid - r.yld < band_sd * r.sd)
        if unpaid_like and ((r.z > z_reject and r.ratio > ratio_reject) or r.z > z_strong):
            t.at[d, "action"] = "replaced:price_implied"
            t.at[d, "used"] = max(0.0, float(r.implied))
    out = div.copy()      # untestable events (e.g. on the first row) are kept as listed
    out.loc[t.index] = t["used"].values
    return out, t


# ------------------------------------------------------- daily accrual funds
def accrual_total_return(df: pd.DataFrame, regular_multiple: float = 2.0, median_window: int = 13,
                         accrual_start=None) -> tuple[pd.Series, pd.DataFrame]:
    """Total return of a fund whose regular income accrues daily (NAV does not drop
    when the month-end distribution is paid).

    ``df`` has ``close`` and ``div`` (amount on the vendor's ex/pay date). A
    distribution paid on the last trading day of a calendar month is income
    accrued over that month: it is spread over the month's trading days in
    proportion to calendar days elapsed since the previous trading day (days after
    the last trading day are booked on it). If such a distribution is more than
    ``regular_multiple`` times the fund's regular month-end income (the rolling
    median of month-end amounts over ``median_window`` events), the excess is a
    capital gain paid with the income: the NAV drops by it, so it is an ex-date
    cash flow. Distributions on any other day are ex-date cash flows.

    Distributions before ``accrual_start`` (a fund that dropped its NAV on the
    ex-date in its early years) are all ex-date cash flows.

    Monthly total returns equal those of ex-date reinvestment up to intra-month
    compounding. Returns the daily return series and a table classifying every
    distribution.
    """
    idx = df.index
    month = idx.to_period("M")
    last_td = pd.Series(idx, index=idx).groupby(month).transform("max")
    is_last = pd.Series(idx == last_td.values, index=idx)
    if accrual_start is not None:
        is_last &= idx >= pd.Timestamp(accrual_start)
    paid = df["div"].reindex(idx).fillna(0.0)

    me_amt = paid[(paid > 0) & is_last]
    regular = me_amt.rolling(median_window, center=True, min_periods=3).median().reindex(idx)
    regular = regular.ffill().bfill() if len(me_amt) else regular
    cap = (regular_multiple * regular).where(is_last & (paid > 0))
    big = is_last & (paid > 0) & (paid > cap)
    accr_amt = paid.where(is_last, 0.0)
    accr_amt = accr_amt.where(~big, regular.where(big, 0.0))
    exdate_amt = paid - accr_amt

    prev = pd.Series(idx, index=idx).shift(1)
    month_start = month.to_timestamp()
    start = np.maximum(prev.fillna(month_start[0] - pd.Timedelta(days=1)).values,
                       (month_start - pd.Timedelta(days=1)).values)
    cal_days = pd.Series((idx.values - start) / np.timedelta64(1, "D"), index=idx)
    month_end = month.to_timestamp(how="end").normalize()
    cal_days = cal_days + np.where(is_last, (month_end - idx).days, 0)
    w = cal_days / cal_days.groupby(month).transform("sum")
    month_div = accr_amt.groupby(month).transform("sum")
    accrued = month_div * w

    ret = (df["close"] + accrued + exdate_amt) / df["close"].shift(1) - 1
    pr = df["close"].pct_change()
    m = paid > 0
    cls = pd.DataFrame({
        "amount": paid[m],
        "yield": (paid / df["close"].shift(1))[m],
        "nav_ret_on_day": pr[m],
        "accrued_part": accr_amt[m],
        "exdate_part": exdate_amt[m],
        "treated_as": np.where(~is_last[m], "ex_date", np.where(big[m], "accrual+ex_date_excess",
                                                                 "daily_accrual")),
    })
    return ret, cls


# --------------------------------------------------------- second sources
def parse_nasdaq_dividends(path: Path) -> pd.Series:
    """Cash dividends from a cached Nasdaq ``/api/quote/<sym>/dividends`` response,
    indexed by ex-date (empty if the listing has none, e.g. NYSE Arca ETFs)."""
    path = Path(path)
    if not path.exists():
        return pd.Series(dtype=float)
    d = json.loads(path.read_text())
    rows = (((d.get("data") or {}).get("dividends") or {}).get("rows")) or []
    out = {}
    for r in rows:
        if str(r.get("type", "")).lower() != "cash":
            continue
        try:
            dt = pd.Timestamp(pd.to_datetime(r["exOrEffDate"], format="%m/%d/%Y"))
            amt = float(str(r["amount"]).replace("$", "").replace(",", ""))
        except (KeyError, ValueError):
            continue
        out[dt] = out.get(dt, 0.0) + amt
    s = pd.Series(out, dtype=float).sort_index()
    s.index.name = "date"
    return s


NASDAQ_DIV_URL = "https://api.nasdaq.com/api/quote/{sym}/dividends?assetclass=etf"


def fetch_nasdaq_dividends(sym: str, path: Path, refresh: bool = False) -> pd.Series:
    """Nasdaq dividend history (cached at ``path``). Network failures are not
    fatal: the function then returns whatever is cached (possibly nothing)."""
    path = Path(path)
    if refresh or not path.exists():
        import requests
        try:
            r = requests.get(NASDAQ_DIV_URL.format(sym=sym),
                             headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=40)
            if r.status_code == 200 and r.text.startswith("{"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(r.text)
        except Exception:  # noqa: BLE001 - optional second source
            pass
    return parse_nasdaq_dividends(path)


def changed_distributions(table: pd.DataFrame) -> pd.DataFrame:
    """Rows of a ``validate_distributions`` table whose amount was changed."""
    t = table[table["action"].str.startswith("replaced")]
    return t[["amount", "used", "second", "implied", "z", "yld", "action"]]
