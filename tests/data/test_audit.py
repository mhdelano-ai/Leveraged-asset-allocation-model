"""Regression tests from the independent data audit (reports/data/audit.md).

Two kinds of test:

* properties the audit VERIFIED and that must stay true (``Panel.asof`` exposes
  nothing after its cutoff, derived predictors use trailing windows only,
  assembly preserves returns, cash never contaminates excess returns, ...);
* defects the audit CONFIRMED, written as ``xfail(strict=True)``: each states the
  correct property, fails today, and will turn into an unexpected pass (an error
  under strict mode) once the data are fixed - at which point remove the marker.

All tests read data/processed (``realdata``); the construction-dependence test
also needs the raw download cache in data/raw and skips without it.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from volmodel.data.panel import load_panel
from volmodel.paths import PROCESSED, RAW

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(not (PROCESSED / "panel_daily.parquet").exists(), reason="panel not assembled"),
]


@pytest.fixture(scope="module")
def P():
    return load_panel()


@pytest.fixture(scope="module")
def W(P):
    return P.predictors_wide()


# ===================================================================== asof
ADVERSARIAL_DATES = [
    "1926-07-01", "1929-10-28", "1951-12-31", "1962-01-02", "1973-01-04",   # first day, crash, Saturday era, splices
    "1987-10-19", "1993-01-29", "1993-02-01", "1999-12-31", "2001-09-12",   # splice eve/day, closure day
    "2002-07-30", "2002-07-31", "2003-12-16", "2008-09-15", "2008-10-15",
    "2008-11-29", "2008-11-30", "2009-02-28", "2014-12-31", "2015-01-01",   # weekend month-end, dev cutoff
    "2020-03-16", "2025-10-31", "2025-11-20", "2026-10-31", "2026-11-13",
]


def _check_view(P, v, t):
    t = pd.Timestamp(t)
    assert v.asof_date == t
    for frame in (v.daily, v.monthly, v.cash_daily, v.cash_monthly, v.tiers):
        assert len(frame) == 0 or frame.index.max() <= t
    # nothing on or before t is dropped (asof is a cut, not a filter)
    assert len(v.daily) == int((P.daily.index <= t).sum())
    assert len(v.monthly) == int((P.monthly.index <= t).sum())
    p = v.predictors
    assert (p["available_date"] <= t).all()
    assert (p["obs_date"] <= t).all()
    assert len(p) == int((P.predictors["available_date"] <= t).sum())
    # the wide view: every non-NaN cell must be backed by a long row published by t
    w = v.predictors_wide()
    if len(w):
        assert w.index.max() <= t
        long = p.drop_duplicates(["series", "obs_date"], keep="first")
        backed = set(zip(long["series"], long["obs_date"]))
        cells = w.stack().dropna()
        assert all((s, d) in backed for d, s in cells.index)
    # latest values equal the last published value of each series
    lat = v.predictor_latest()
    for s in ["iv_vix", "macro_unrate", "credit_gz_spread_m", "tips_real_yield_10y", "ust_cmt_yield_10y"]:
        q = p[p["series"] == s]
        if len(q):
            q = q.sort_values("obs_date")
            assert lat[s] == pytest.approx(q["value"].iloc[-1])
        else:
            assert s not in lat.index


@pytest.mark.parametrize("t", ADVERSARIAL_DATES)
def test_asof_exposes_nothing_after_cutoff(P, t):
    _check_view(P, P.asof(t), t)


def test_asof_random_dates(P):
    rng = np.random.default_rng(20261114)
    days = pd.date_range("1926-07-01", "2026-11-13", freq="D")
    for t in rng.choice(days, 15, replace=False):
        _check_view(P, P.asof(t), t)


def test_asof_monthly_has_no_partial_month(P):
    v = P.asof("2008-10-15")
    assert v.monthly.index.max() == pd.Timestamp("2008-09-30")
    v = P.asof("2008-11-28")   # last trading day; month-end row is dated 11-30 > t
    assert v.monthly.index.max() == pd.Timestamp("2008-10-31")


def test_nested_asof_to_earlier_date_equals_direct(P):
    """Models (har, statespace, ml, bridge) call view.asof(earlier); it must equal panel.asof(earlier)."""
    v = P.asof("2010-06-30")
    a, b = v.asof("2007-12-31"), P.asof("2007-12-31")
    pd.testing.assert_frame_equal(a.daily, b.daily)
    pd.testing.assert_frame_equal(a.monthly, b.monthly)
    pd.testing.assert_frame_equal(a.predictors_wide(), b.predictors_wide())


def test_writing_into_view_does_not_touch_root(P):
    v = P.asof("2008-10-15")
    before = P.daily.loc["2008-10-15", "us_large"]
    v.daily.iloc[-1, 0] = 999.0
    assert P.daily.loc["2008-10-15", "us_large"] == before


def test_asof_of_a_view_never_widens(P):
    v = P.asof("2008-10-15").asof("2026-06-30")
    w = v.predictors_wide()
    assert w.index.max() <= pd.Timestamp("2008-10-15")


# ====================================================== predictor timing
def test_available_date_never_before_obs(P):
    p = P.predictors
    assert (p["available_date"] >= p["obs_date"]).all()


@pytest.mark.parametrize("series,min_days", [
    ("macro_cpi_level", 28), ("macro_infl_yoy", 28), ("macro_infl_vol_36m", 28), ("macro_real_short_rate", 28),
    ("macro_unrate", 28), ("macro_unrate_chg_12m", 28), ("macro_sahm_gap", 28), ("macro_sahm_recession", 28),
    ("macro_ip_yoy", 28), ("credit_gz_spread_m", 28), ("tips_real_yield_10y", 14), ("tips_breakeven_10y", 14),
    ("credit_moodys_baa_yield_m", 1), ("credit_moodys_baa_aaa_spread_d", 1),
])
def test_publication_lags_at_least(P, series, min_days):
    q = P.predictors[P.predictors["series"] == series]
    assert len(q)
    assert ((q["available_date"] - q["obs_date"]).dt.days >= min_days).all()


def test_monthly_predictors_stamped_at_month_end(P):
    p = P.predictors
    monthly = [s for s in p["series"].unique() if s.startswith("macro_") or s.endswith("_m")]
    q = p[p["series"].isin(monthly)]
    assert q["obs_date"].dt.is_month_end.all()


def _cmp(a: pd.Series, b: pd.Series, tol: float):
    j = pd.concat([a, b], axis=1, sort=True).dropna()
    assert len(j) > 0.95 * a.notna().sum()
    assert (j.iloc[:, 0] - j.iloc[:, 1]).abs().max() < tol


def _etf_duration(r: pd.Series, y: pd.Series, window=252, h=5) -> pd.Series:
    p = np.log1p(r).fillna(0).cumsum()
    df = pd.concat([p.rename("p"), y.rename("y")], axis=1, sort=True).dropna()
    rr, dy = df["p"].diff(h), -df["y"].diff(h)
    cov = (rr * dy).rolling(window).mean() - rr.rolling(window).mean() * dy.rolling(window).mean()
    var = (dy * dy).rolling(window).mean() - dy.rolling(window).mean() ** 2
    return (cov / var).dropna()


@pytest.mark.parametrize("asset,ycol,series", [("ust_int", "ust_ladder_yield_int", "ust_etf_duration_est_int"),
                                                ("ust_long", "ust_ladder_yield_long", "ust_etf_duration_est_long")])
def test_etf_duration_is_trailing_regression(P, W, asset, ycol, series):
    """Reproduce the rolling-regression ETF duration from panel returns; and check truncation invariance."""
    r = P.daily[asset].loc["2002-07-31":]
    _cmp(W[series], _etf_duration(r, W[ycol]), 1e-9)
    for T in ["2008-10-15", "2014-12-31", "2020-03-20"]:
        trunc = _etf_duration(r.loc[:T], W[ycol].loc[:T])
        assert trunc.iloc[-1] == pytest.approx(W[series].loc[:T].dropna().iloc[-1], abs=1e-9)


def test_macro_derived_series_are_trailing(W):
    cpi = W["macro_cpi_level"].dropna()
    dl = np.log(cpi).diff()
    dl[cpi.index.to_series().diff().dt.days > 31] = np.nan
    vol = np.sqrt(12) * dl.rolling(36, min_periods=30).std()
    _cmp(W["macro_infl_vol_36m"].loc[:"2025-09"], vol.loc[:"2025-09"], 1e-9)
    _cmp(W["macro_infl_yoy"], np.log(cpi / cpi.shift(12, freq="ME").reindex(cpi.index)), 1e-12)
    _cmp(W["macro_term_spread"], W["macro_long_yield"] - W["macro_short_rate"], 1e-12)
    _cmp(W["macro_real_short_rate"], W["macro_short_rate"] - W["macro_infl_yoy"], 1e-12)
    un = W["macro_unrate"].dropna()
    a3 = un.rolling(3).mean()
    _cmp(W["macro_sahm_gap"], a3 - a3.shift(1).rolling(12, min_periods=12).min(), 1e-12)
    _cmp(W["iv_vix_term_slope"], W["iv_vix3m"] / W["iv_vix"] - 1, 1e-12)
    _cmp(W["iv_vix1y_term_slope"], W["iv_vix1y"] / W["iv_vix"] - 1, 1e-12)


def test_sahm_available_when_latest_input_published(P):
    p = P.predictors
    gap = p[p["series"] == "macro_sahm_gap"].set_index("obs_date")["available_date"]
    un = p[p["series"] == "macro_unrate"].set_index("obs_date")["available_date"]
    j = pd.concat([gap, un.cummax()], axis=1, keys=["g", "u"], sort=True).dropna()
    assert (j["g"] >= j["u"]).all()


# ================================================================ assembly
def test_monthly_equals_compounded_daily(P):
    m = (1 + P.daily).groupby(pd.Grouper(freq="ME")).prod(min_count=1) - 1
    for a in P.daily.columns:
        first = P.daily[a].first_valid_index() + pd.offsets.MonthEnd(0)
        idx = m.index[(m.index > first) & (m.index <= P.monthly.index[-1])]
        assert (m[a].reindex(idx) - P.monthly[a].reindex(idx)).abs().max() < 1e-12


def test_no_partial_final_month(P):
    last = P.monthly.index[-1]
    assert P.daily.index[-1] > last
    assert P.daily.loc[:last].index[-1] == P.daily.index[P.daily.index <= last][-1]
    nxt = last + pd.offsets.MonthEnd(1)
    assert P.daily.index[-1] < nxt


def test_folding_preserves_compounded_returns(P):
    from volmodel.paths import ASSET_DIR
    for a in P.daily.columns:
        t = pd.read_parquet(ASSET_DIR / f"{a}_daily.parquet")["ret"].dropna()
        p = P.daily[a].dropna()
        assert abs(np.log1p(t).sum() - np.log1p(p).sum()) < 1e-10
        yt = np.log1p(t).groupby(t.index.year).sum()
        yp = np.log1p(p).groupby(p.index.year).sum().reindex(yt.index).fillna(0)
        assert (yt - yp).abs().max() < 1e-10


def test_tiers_cover_exactly_the_monthly_data(P):
    assert not (P.tiers.isna() & P.monthly.notna()).any().any()
    assert not (P.tiers.notna() & P.monthly.isna()).any().any()
    assert set(pd.unique(P.tiers.stack().dropna())) <= {"A", "B", "C", "D"}


# ==================================================================== cash
def test_cash_never_contaminates_excess(P):
    assert P.cash_daily.notna().all() and P.cash_monthly.notna().all()
    assert not (P.excess_daily.isna() & P.daily.notna()).any().any()
    assert not (P.excess_monthly.isna() & P.monthly.notna()).any().any()
    assert P.cash_daily.abs().max() < 0.001          # < 25%/yr on any day
    assert P.cash_monthly.min() > -0.001


@pytest.mark.parametrize("year,lo,hi", [(1935, 0.0, 0.5), (1945, 0.2, 0.5), (1960, 2.0, 3.5), (1974, 7.0, 9.0),
                                        (1981, 13.5, 16.0), (2009, 0.0, 0.3), (2015, 0.0, 0.2), (2024, 4.8, 5.6)])
def test_cash_levels_match_known_bill_rates(P, year, lo, hi):
    c = P.cash_monthly.loc[str(year)]
    assert lo <= ((1 + c).prod() - 1) * 100 <= hi


def test_cash_splice_to_4w_bill_is_continuous(P):
    c = P.cash_daily.loc["2026-08-01":"2026-10-31"]
    by_m = c.groupby(c.index.month).mean() * 252
    assert abs(by_m.loc[10] - by_m.loc[9]) < 0.005


# ================================================== genuine events kept
@pytest.mark.parametrize("asset,date,lo,hi", [
    ("gold", "1980-01-23", -0.18, -0.15),        # Frankfurt fixing after COMEX -17.4% on 01-22: genuine
    ("gold", "1985-03-20", 0.07, 0.09),          # COMEX +11.8% on 03-19: genuine, lagged by the fixing time
    ("us_large", "1987-10-19", -0.20, -0.17),
    ("reits", "2004-04-12", -0.06, -0.04),       # every REIT peer fund fell ~5%
])
def test_genuine_extreme_days_are_kept(P, asset, date, lo, hi):
    assert lo <= P.daily.loc[date, asset] <= hi


def test_known_dropped_days_stay_gaps(P):
    assert np.isnan(P.daily.loc["1973-01-04", "ust_long"])   # H.15 20y re-basing
    assert np.isnan(P.daily.loc["1993-07-29", "reits"])      # FRESX bad print


# ============================================ confirmed defects (fixed; were strict xfails)
# AUDIT A-1 (fixed): Yahoo lists non-existent/inflated IEF and LQD distributions (IEF 2003-07-01, 08-01, 09-02,
# 10-01, 12-16; LQD 2003-12-16) that adjclose credits; ETF returns are now close + validated distributions.
def test_ief_lqd_2003_distributions(P):
    assert P.daily.loc["2003-12-16", "ust_int"] < 0.005
    assert P.daily.loc["2003-12-16", "ig_credit"] < 0.006
    assert abs((1 + P.daily["ust_int"].loc["2003"]).prod() - 1 - 0.022) < 0.01


# AUDIT C-2 (fixed): VEIEX NAV bad print 2001-07-16/17 (+5.6% then -5.9%; peer EM funds -0.6%, -0.8%).
# The bad price is dropped, so 2001-07-16 is a gap (NaN) and the next return spans it.
def test_em_veiex_2001_bad_print(P):
    assert abs(P.daily.loc["2001-07-16", "em"]) < 0.03 or np.isnan(P.daily.loc["2001-07-16", "em"])
    assert abs(P.daily.loc["2001-07-17", "em"]) < 0.03


# AUDIT C-2 (fixed): FBNDX (41% of ig_credit 1980-84) bad prints that reverse next day, e.g. 1983-11-28/29
# +3.6%/-3.4% with VWESX and Treasuries flat. Dropped prices leave a NaN day; the next return spans it.
def test_ig_credit_fbndx_bad_prints(P):
    for d in ["1983-11-28", "1984-02-02", "1984-11-14"]:
        x = P.daily.loc[d, "ig_credit"]
        assert np.isnan(x) or abs(x) < 0.007
        assert abs(P.daily.loc[P.daily.index[P.daily.index > d][0], "ig_credit"]) < 0.007


# AUDIT C-1 (fixed): daily-accrual bond funds (VWEHX, VFSTX, FBNDX, VWESX 1985+) do not drop NAV on the ex-date;
# adjclose booked a month of income on one day. The income is now accrued over the month as for PRRIX.
def test_no_month_end_income_lumps_in_hy_credit(P):
    r = P.daily["hy_credit"].loc["1986":"2006"].dropna()
    month_last = r.groupby(r.index.to_period("M")).tail(1).index
    lump = r.loc[month_last].mean()
    other = r.drop(month_last).mean()
    assert lump < 3 * abs(other) + 0.001


@pytest.mark.skipif(not (RAW / "credit" / "yahoo_VWESX.json").exists(), reason="raw credit cache missing")
def test_ig_credit_history_independent_of_holdout(P):
    """AUDIT G-1 (fixed): ig_credit before 2002 is a blend whose weights were vol-matched to LQD on
    2002-08..2026-10, i.e. on post-2014 holdout data. The weight from 2002-08..2014-12 alone is 0.75, not 0.70.

    The weight is recovered by regressing the panel on the two funds' raw adjclose returns. Since the C-1 fix the
    panel accrues each fund's month-end income over the month while raw adjclose books it on the month's last
    trading day, so those days measure the accrual treatment, not the blend weight, and are left out."""
    def adj(sym):
        res = json.loads((RAW / "credit" / f"yahoo_{sym}.json").read_text())["chart"]["result"][0]
        ts = (pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York")
              .tz_localize(None).normalize())
        s = pd.Series(res["indicators"]["adjclose"][0]["adjclose"], index=ts)
        return s[~s.index.duplicated(keep="last")].dropna().pct_change().dropna()
    vw, vf = adj("VWESX"), adj("VFSTX")
    ig = P.daily["ig_credit"].loc["1990-01-02":"2002-07-30"]
    j = pd.concat([ig, vw, vf], axis=1, keys=["ig", "a", "b"], sort=True).dropna()
    j = j.drop(j.groupby(j.index.to_period("M")).tail(1).index)     # month-end income days (see docstring)
    w = np.linalg.lstsq((j["a"] - j["b"]).to_numpy()[:, None], (j["ig"] - j["b"]).to_numpy(), rcond=None)[0][0]
    assert abs(w - 0.752) < 0.01      # the dev-only (<= 2014) vol-match weight
