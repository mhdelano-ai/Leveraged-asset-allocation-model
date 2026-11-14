"""Tests for the shared distribution / bad-print tools (volmodel.data.sources.distributions)
and their use in the tracks (credit blends, accrual funds, calibration cutoff)."""
import numpy as np
import pandas as pd
import pytest

from volmodel.data.sources import credit as C
from volmodel.data.sources import distributions as D
from volmodel.paths import ASSET_DIR, RAW


def _market(n=1500, seed=0):
    """A fund that tracks a peer with noise, paying 0.25 monthly on the first trading day."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    peer = pd.Series(rng.normal(0, 0.004, n), idx)
    tr = 0.9 * peer + pd.Series(rng.normal(0, 0.0008, n), idx)
    first = pd.Series(idx, index=idx).groupby(idx.to_period("M")).transform("min") == idx
    div = pd.Series(np.where(first, 0.25, 0.0), idx)
    close = pd.Series(100.0, idx)
    for i in range(1, n):        # price drops by the distribution on its ex-date
        close.iloc[i] = (close.iloc[i - 1] * (1 + tr.iloc[i])) - div.iloc[i]
    return close, div, pd.DataFrame({"peer": peer})


def test_genuine_distributions_are_kept():
    close, div, peers = _market()
    used, t = D.validate_distributions(close, div, peers)
    assert (t["action"] == "kept").all()
    assert np.allclose(used[div > 0], 0.25)
    assert (t["implied"] - 0.25).abs().median() < 0.05


def test_spurious_and_inflated_distributions_are_replaced():
    close, div, peers = _market()
    exd = div.index[div > 0]
    bad_extra, bad_inflated = exd[20], exd[40]
    listed = div.copy()
    listed[bad_inflated] = 0.75                      # 3x the real amount, price dropped only 0.25
    d0 = close.index[close.index.get_loc(exd[30]) + 10]
    listed[d0] = 1.0                                 # never paid: price did not drop at all
    used, t = D.validate_distributions(close, listed, peers)
    assert t.loc[bad_inflated, "action"] == "replaced:price_implied"
    assert abs(used[bad_inflated] - 0.25) < 0.1
    assert t.loc[d0, "action"] == "replaced:price_implied"
    assert used[d0] < 0.1
    assert (t.drop([bad_inflated, d0])["action"] == "kept").all()


def test_second_source_arbitrates_disagreement():
    close, div, peers = _market()
    exd = div.index[div > 0]
    listed = div.copy()
    listed[exd[10]] = 0.70
    second = div[div > 0].copy()                     # second record has the true amounts
    used, t = D.validate_distributions(close, listed, peers, second=second)
    assert t.loc[exd[10], "action"] == "replaced:second_source"
    assert used[exd[10]] == pytest.approx(0.25)
    assert (t.drop(exd[10])["action"] == "kept:second_source_agrees").all()


def test_market_noise_is_not_mistaken_for_an_unpaid_distribution():
    close, div, peers = _market()
    exd = div.index[div > 0]
    i = close.index.get_loc(exd[25])
    close = close.copy()
    close.iloc[i:] *= 1.03                           # a 3% unexplained jump on an ex-date (>> yield)
    used, t = D.validate_distributions(close, div, peers)
    assert t.loc[exd[25], "action"] == "kept"


def test_drop_bad_prices_spans_the_gap_and_keeps_distributions():
    idx = pd.bdate_range("2001-07-12", periods=5)
    df = pd.DataFrame({"close": [10.0, 10.1, 10.7, 10.05, 10.1], "div": [0, 0, 0.05, 0, 0]}, idx)
    out, dropped = D.drop_bad_prices(df, ["2001-07-16"])
    assert dropped == ["2001-07-16"]
    assert "2001-07-16" not in out.index.strftime("%Y-%m-%d")
    r = D.total_return(out["close"], out["div"])
    assert r.loc["2001-07-17"] == pytest.approx((10.05 + 0.05) / 10.1 - 1)


def test_dedupe_repeats():
    div = pd.Series([0.632, 0.632, 0.5], index=pd.to_datetime(["2008-04-01", "2008-04-03", "2008-05-01"]))
    out, dup = D.dedupe_repeats(div)
    assert dup == ["2008-04-03"] and len(out) == 2


def test_accrual_spreads_month_end_income_and_preserves_monthly_returns():
    idx = pd.bdate_range("1990-01-02", "1990-06-29")
    rng = np.random.default_rng(3)
    close = pd.Series(10 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx)))), idx)
    last = pd.Series(idx, index=idx).groupby(idx.to_period("M")).transform("max") == idx
    div = pd.Series(np.where(last, 0.08, 0.0), idx)
    div["1990-03-15"] = 0.30                         # an ex-date distribution (NAV drops)
    close.loc["1990-03-15":] -= 0.30
    df = pd.DataFrame({"close": close, "div": div})
    r, cls = D.accrual_total_return(df)
    lump = D.total_return(close, div)
    m_new = (1 + r).groupby(idx.to_period("M")).prod() - 1
    m_old = (1 + lump).groupby(idx.to_period("M")).prod() - 1
    assert np.allclose(m_new.iloc[1:], m_old.iloc[1:], atol=2e-4)        # unchanged up to intra-month compounding
    assert r[last].iloc[1:].abs().max() < 0.01                            # no month-end income lump
    assert lump[last].iloc[1:].min() > 0.006
    assert cls.loc["1990-03-15", "treated_as"] == "ex_date"
    assert (cls.loc[last[last].index, "treated_as"] == "daily_accrual").all()


def test_accrual_splits_capital_gain_paid_with_month_end_income():
    idx = pd.bdate_range("1993-01-04", "1994-06-30")
    close = pd.Series(10.0, idx)
    last = pd.Series(idx, index=idx).groupby(idx.to_period("M")).transform("max") == idx
    div = pd.Series(np.where(last, 0.06, 0.0), idx)
    div["1993-12-31"] = 0.30                         # income 0.06 + capital gain 0.24 (NAV drops 0.24)
    close.loc["1993-12-31":] -= 0.24
    r, cls = D.accrual_total_return(pd.DataFrame({"close": close, "div": div}))
    assert cls.loc["1993-12-31", "treated_as"] == "accrual+ex_date_excess"
    assert cls.loc["1993-12-31", "exdate_part"] == pytest.approx(0.24)
    assert abs(r.loc["1993-12-31"]) < 0.001


def test_accrual_start_keeps_early_distributions_on_ex_date():
    idx = pd.bdate_range("1984-01-02", "1985-12-31")
    close = pd.Series(10.0, idx)
    last = pd.Series(idx, index=idx).groupby(idx.to_period("M")).transform("max") == idx
    div = pd.Series(np.where(last, 0.08, 0.0), idx)
    r, cls = D.accrual_total_return(pd.DataFrame({"close": close, "div": div}), accrual_start="1985-01-01")
    assert (cls.loc[:"1984-12-31", "treated_as"] == "ex_date").all()
    assert (cls.loc["1985-01-01":, "treated_as"] == "daily_accrual").all()


def test_blend_spans_a_day_missing_in_one_fund():
    idx = pd.bdate_range("1983-11-23", periods=5)
    a = pd.Series([0.01, 0.02, -0.01, 0.0, 0.01], idx)
    b = pd.Series([0.0, 0.03, 0.01, 0.02], idx.delete(2))      # b has no price on day 3
    out = C.blend_returns(0.5, a, b)
    assert idx[2] not in out.index
    assert out[idx[3]] == pytest.approx(0.5 * ((1 - 0.01) * 1.0 - 1) + 0.5 * 0.01)
    assert np.allclose(out[[idx[0], idx[1], idx[4]]], [0.005, 0.025, 0.015])


def test_calibration_cutoff_precedes_holdout():
    assert C.CALIBRATION_END == "2014-12-31"
    assert pd.Timestamp(C.CALIBRATION_END) < pd.Timestamp("2015-01-01")


@pytest.mark.realdata
def test_credit_segments_record_dev_only_calibration_and_hy_horizon_ratio():
    seg = pd.read_csv(ASSET_DIR.parent / "segments.csv")
    ig = seg[seg.asset == "ig_credit"]
    for src in ["blend_VWESX_FBNDX", "blend_VWESX_VFSTX", "ibbotson_ltcorp_rf_blend"]:
        assert "2002-07..2014-12 only" in ig[ig.source == src]["notes"].iloc[0]
    hy = seg[(seg.asset == "hy_credit") & (seg.source == "yahoo_VWEHX")].iloc[0]
    assert hy["tier"] == "B" and "21-day-horizon vol ratio VWEHX/HYG 0.8" in hy["notes"]
    assert set(seg[seg.source.isin(["blend_VWESX_FBNDX", "blend_VWESX_VFSTX", "yahoo_VEIEX",
                                     "yahoo:VGSIX adjclose (Vanguard REIT Index Inv)"])]["tier"]) == {"B"}


@pytest.mark.realdata
def test_fund_returns_have_no_month_end_lumps():
    if not (RAW / "credit" / "yahoo_VWEHX.json").exists():
        pytest.skip("raw credit cache missing")
    r, cls, dropped = C.fund_returns("VWEHX")
    x = r.loc["1986":"2006"]
    last = x.groupby(x.index.to_period("M")).tail(1).index
    assert x.loc[last].mean() < 3 * abs(x.drop(last).mean()) + 0.001
    assert dropped == ["1991-08-19"]
    raw = C.yahoo_returns("VWEHX")
    m_new = np.expm1(np.log1p(r.loc["1986":"2006"]).resample("ME").sum())
    m_old = np.expm1(np.log1p(raw.loc["1986":"2006"]).resample("ME").sum())
    assert (m_new - m_old).abs().max() < 1e-3     # monthly unchanged up to intra-month compounding
