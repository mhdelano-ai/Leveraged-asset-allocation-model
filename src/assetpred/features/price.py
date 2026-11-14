"""Price-based features built from the monthly and daily return panels.

All price features are computed on the monthly excess log return
``xr = log1p(r) - log1p(cash)`` (or on its cumulative index), so that a column
means the same thing for bonds, equities and commodities. Volatility-scaled
versions put assets with very different risk on one scale for pooled models.

Volatility is measured two ways and the two are never spliced:
* ``vol_m_*`` from monthly excess returns, available for every asset once it
  has enough history;
* ``vol_d_*`` from daily total returns, only where a daily series exists
  (NaN otherwise). Daily sources differ from the monthly ones for REITs (VNQ vs
  the Nareit index) and commodities (BCOM ETN vs the AQR equal-weight
  portfolio); see ``asset_meta_daily.csv``. Daily vol of the mutual-fund series
  (ig_credit, hy_credit) understates true vol because NAVs are smoothed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import schema
from .common import Feat, Inputs, to_month_end

PRICE = "price"
TRADING_DAYS = 252


def _monthly_daily_sums(daily: pd.DataFrame, months: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per month: sum of squared daily log returns and number of daily obs."""
    lr = np.log1p(daily)
    key = to_month_end(lr.index)
    sq = (lr ** 2).groupby(key).sum(min_count=1)
    n = lr.notna().groupby(key).sum().astype(float)
    sq = sq.reindex(months).fillna(0.0)
    n = n.reindex(months).fillna(0.0)
    return sq, n


def _daily_vol(sq, n, months: int, min_obs: int) -> pd.DataFrame:
    ss = sq.rolling(months, min_periods=months).sum()
    nn = n.rolling(months, min_periods=months).sum()
    vol = np.sqrt(TRADING_DAYS * ss / nn)
    return vol.where(nn >= min_obs)


def _daily_corr(daily: pd.DataFrame, a: str, b: str, months_index: pd.DatetimeIndex,
                window: int, min_obs: int) -> pd.Series:
    x, y = np.log1p(daily[a]), np.log1p(daily[b])
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    key = to_month_end(x.index)
    parts = pd.DataFrame({"n": 1.0, "x": x, "y": y, "xx": x * x, "yy": y * y, "xy": x * y})
    s = parts.groupby(key).sum().reindex(months_index).fillna(0.0)
    s = s.rolling(window, min_periods=window).sum()
    n = s["n"]
    cov = n * s["xy"] - s["x"] * s["y"]
    var = (n * s["xx"] - s["x"] ** 2) * (n * s["yy"] - s["y"] ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.sqrt(var)
    return corr.where(n >= min_obs)


def _roll_sum(x: pd.DataFrame, k: int) -> pd.DataFrame:
    return x.rolling(k, min_periods=k).sum()


def _xs_rank(df: pd.DataFrame, min_assets: int = 4) -> pd.DataFrame:
    """Cross-sectional percentile rank on each date, scaled to [0, 1]."""
    n = df.notna().sum(axis=1)
    r = df.rank(axis=1, method="average")
    out = r.sub(1).div(n - 1, axis=0)
    return out.where(n >= min_assets, np.nan)


def price_features(inp: Inputs) -> list[Feat]:
    months = inp.returns.index
    xr = schema.excess_log_returns(inp.returns, inp.cash)
    idx = xr.cumsum().where(xr.notna())  # log excess-return index, per asset
    P = np.exp(idx)

    vol_m_12 = xr.rolling(12, min_periods=12).std() * np.sqrt(12)
    vol_m_36 = xr.rolling(36, min_periods=24).std() * np.sqrt(12)
    sig_m = vol_m_36 / np.sqrt(12)  # monthly sd used to scale momentum

    mom = {k: _roll_sum(xr, k) for k in (1, 3, 6, 12)}
    mom_12_1 = _roll_sum(xr, 11).shift(1)
    lt_rev = _roll_sum(xr, 48).shift(12)  # months t-59 .. t-12

    def sharpe(m, k):
        return m / (sig_m * np.sqrt(k))

    mom_1_s, mom_3_s, mom_12_1_s = sharpe(mom[1], 1), sharpe(mom[3], 3), sharpe(mom_12_1, 11)

    trend_ma10 = idx - np.log(P.rolling(10, min_periods=10).mean())
    trend_ma3_12 = np.log(P.rolling(3, min_periods=3).mean()) - np.log(P.rolling(12, min_periods=12).mean())
    # 12-month drawdown: against the index levels at month ends t-12..t (13 levels)
    dd_12 = idx - idx.rolling(13, min_periods=13).max()
    # running drawdown; 0 is the index level before the asset's first return
    dd_max = (idx - np.fmax(idx.expanding().max(), 0.0)).where(idx.expanding().count() >= 12)

    sq, n = _monthly_daily_sums(inp.daily.reindex(columns=xr.columns), months)
    vol_d_1 = _daily_vol(sq, n, 1, 10)
    vol_d_3 = _daily_vol(sq, n, 3, 40)
    vol_d_12 = _daily_vol(sq, n, 12, 150)
    vol_d_ratio = np.log(vol_d_1 / vol_d_12)
    for v in (vol_d_1, vol_d_3, vol_d_ratio):
        v.where(xr.notna(), inplace=True)

    eq = xr["us_eq"]
    cov = xr.rolling(36, min_periods=24).cov(eq)
    beta_eq = cov.div(eq.rolling(36, min_periods=24).var(), axis=0)

    f = []
    A = lambda name, v, d: f.append(Feat(name, PRICE, v, d))  # noqa: E731
    A("mom_1m", mom[1], "Excess log return of month t (short-term reversal / momentum). "
      "For hy_credit (smoothed fund NAV, monthly autocorrelation ~0.16) part of any signal is stale pricing.")
    A("mom_3m", mom[3], "Sum of excess log returns over months t-2..t.")
    A("mom_6m", mom[6], "Sum of excess log returns over months t-5..t.")
    A("mom_12m", mom[12], "Sum of excess log returns over months t-11..t.")
    A("mom_12_1", mom_12_1, "12-minus-1 momentum: sum of excess log returns over months t-11..t-1 (skips month t).")
    A("lt_rev_60_12", lt_rev, "Long-term reversal: sum of excess log returns over months t-59..t-12 (48 months).")
    A("mom_1m_vs", mom_1_s, "mom_1m divided by the asset's trailing monthly volatility (36m, min 24).")
    A("mom_3m_vs", mom_3_s, "mom_3m / (trailing monthly vol x sqrt(3)): a t-statistic-like 3-month trend.")
    A("mom_12_1_vs", mom_12_1_s, "mom_12_1 / (trailing monthly vol x sqrt(11)): volatility-scaled time-series momentum, comparable across assets.")
    A("trend_ma10", trend_ma10, "Log of the excess-return index over its 10-month simple moving average (months t-9..t).")
    A("trend_ma3_12", trend_ma3_12, "Log ratio of the 3-month to the 12-month moving average of the excess-return index.")
    A("dd_12m", dd_12, "Drawdown of the excess-return index from its maximum over the last 12 months (log, <= 0).")
    A("dd_max", dd_max, "Drawdown of the excess-return index from its running maximum since the asset's history began (log, <= 0); needs 12 months.")
    A("vol_m_12m", vol_m_12, "Annualised sd of monthly excess log returns over months t-11..t.")
    A("vol_m_36m", vol_m_36, "Annualised sd of monthly excess log returns over months t-35..t (min 24). Understated for smoothed NAV series (hy_credit, ig_credit).")
    A("vol_d_1m", vol_d_1, "Annualised realised vol from daily log returns in month t (sqrt(252 x mean r^2), min 10 days). "
      "NaN where no daily series exists: dev_eq before 1990-07, em_eq before 2001, reits/gold before 2004, commodities before 2006-10. "
      "Daily sources differ from monthly for reits (VNQ) and commodities (BCOM ETN).")
    A("vol_d_3m", vol_d_3, "Annualised realised vol from daily log returns over months t-2..t (min 40 days); same coverage caveats as vol_d_1m.")
    A("vol_d_ratio", vol_d_ratio, "Log of 1-month over 12-month daily realised vol (vol regime shift; self-normalised so comparable across assets and daily sources).")
    A("beta_eq_36m", beta_eq, "Rolling 36-month (min 24) beta of the asset's monthly excess return on US equity excess return.")
    A("xs_rank_mom_12_1_vs", _xs_rank(mom_12_1_s), "Cross-sectional percentile rank (0..1) of mom_12_1_vs among assets with a value on date t (needs >= 4 assets).")
    A("xs_rank_mom_3m_vs", _xs_rank(mom_3_s), "Cross-sectional percentile rank (0..1) of mom_3m_vs among assets with a value on date t (needs >= 4 assets).")
    A("xs_rank_lt_rev", _xs_rank(lt_rev / vol_m_36), "Cross-sectional percentile rank of vol-scaled long-term reversal (lt_rev_60_12 / vol_m_36m).")

    # ---- global price states
    sbd = _daily_corr(inp.daily, "us_eq", "ust_int", months, 3, 40)
    sbm = xr["us_eq"].rolling(36, min_periods=24).corr(xr["ust_int"])
    pos = (mom_12_1 > 0).astype(float).where(mom_12_1.notna())
    breadth = pos.mean(axis=1).where(pos.notna().sum(axis=1) >= 5)
    G = lambda name, v, d: f.append(Feat(name, PRICE, v, d))  # noqa: E731
    G("g_eq_mom_12_1", mom_12_1["us_eq"], "US equity 12-minus-1 momentum (global equity trend state).")
    G("g_bond_mom_12_1", mom_12_1["ust_int"], "Intermediate Treasury 12-minus-1 momentum (global rates trend state).")
    G("g_cmdty_mom_12_1", mom_12_1["commodities"], "Commodity 12-minus-1 momentum (inflation / growth state).")
    G("g_eq_vol_d_1m", vol_d_1["us_eq"], "US equity realised vol from daily returns in month t (annualised).")
    G("g_eq_dd_max", dd_max["us_eq"], "US equity drawdown from its running maximum (excess-return index).")
    G("g_sb_corr_d_3m", sbd, "Correlation of daily US equity and 5y Treasury returns over months t-2..t (min 40 joint days).")
    G("g_sb_corr_m_36m", sbm, "Correlation of monthly US equity and 5y Treasury excess returns over 36 months (min 24).")
    G("g_trend_breadth", breadth, "Share of assets with positive 12-minus-1 momentum on date t (NaN when fewer than 5 assets have a value).")
    return f


def realised_variance_us_eq(inp: Inputs) -> pd.Series:
    """Sum of squared daily US equity log returns in each month, x 1e4 (monthly %^2)."""
    sq, n = _monthly_daily_sums(inp.daily[["us_eq"]], inp.returns.index)
    return (sq["us_eq"] * 1e4).where(n["us_eq"] >= 10)
