"""Tests for the panel assembler (``assetforecast.data.assemble``).

Offline unit tests use synthetic frames; ``realdata`` tests read the assembled files in
data/processed (run ``python -m assetforecast.data.assemble`` first).
"""
import numpy as np
import pandas as pd
import pytest

from assetforecast.data import assemble as am
from assetforecast.paths import ASSET_DIR, PROCESSED
from assetforecast.universe import ASSET_IDS


# ---------------------------------------------------------------- unit tests
def _frame(dates, rets, src="s", tier="A"):
    return pd.DataFrame({"ret": rets, "source": src, "tier": tier}, index=pd.DatetimeIndex(dates))


def test_fold_compounds_off_calendar_into_next_day():
    cal = pd.DatetimeIndex(["2001-01-02", "2001-01-03", "2001-01-05", "2001-01-08"])
    d = _frame(["2001-01-02", "2001-01-04", "2001-01-05", "2001-01-06", "2001-01-08"],
               [0.01, 0.02, -0.01, 0.03, 0.0])
    d.loc["2001-01-04", "tier"] = "C"
    f = am.fold_to_calendar(d, cal)
    assert f.n_off_calendar == 2
    assert list(f.frame.index) == [pd.Timestamp(x) for x in ["2001-01-02", "2001-01-05", "2001-01-08"]]
    assert f.frame.loc["2001-01-05", "ret"] == pytest.approx(1.02 * 0.99 - 1)
    assert f.frame.loc["2001-01-08", "ret"] == pytest.approx(0.03)
    assert f.frame.loc["2001-01-05", "tier"] == "C"            # worst tier of the folded days
    # total compounded return is preserved; 01-03 has no return (stays a gap, not a zero)
    assert np.prod(1 + f.frame["ret"]) == pytest.approx(np.prod(1 + d["ret"]))
    assert pd.Timestamp("2001-01-03") not in f.frame.index


def test_fold_across_month_boundary_is_counted():
    cal = pd.DatetimeIndex(["2001-01-30", "2001-02-01"])
    f = am.fold_to_calendar(_frame(["2001-01-30", "2001-01-31"], [0.01, 0.02]), cal)
    assert f.n_boundary_moves == 1
    assert f.frame.index[-1] == pd.Timestamp("2001-02-01")


def test_month_is_complete():
    assert am.month_is_complete(pd.Timestamp("2026-10-30"))       # Friday, last weekday of Oct 2026
    assert not am.month_is_complete(pd.Timestamp("2026-11-13"))
    assert am.month_is_complete(pd.Timestamp("2024-03-28"))       # Good Friday 2024-03-29
    assert am.month_is_complete(pd.Timestamp("2021-05-28"))       # Memorial Day 2021-05-31


def test_worst_tier_and_runs():
    assert am.worst_tier(["A", "C", "B"]) == "C"
    assert np.isnan(am.worst_tier([np.nan]))
    s = pd.Series(["a", "a", "b", "b", "a"], index=pd.date_range("2000-01-03", periods=5))
    r = am._runs(s)
    assert [(x[0], x[3]) for x in r] == [("a", 2), ("b", 2), ("a", 1)]


def test_assemble_synthetic_monthly_and_partial_months():
    cal = pd.bdate_range("2000-01-03", "2000-04-14")
    cash_d = _frame(cal, 0.0001, "c", "C")
    me = pd.date_range("1999-10-31", "2000-04-30", freq="ME")
    cash_m = _frame(me, 0.002, "c", "C")
    d = _frame(cal[cal >= "2000-01-10"], 0.001, "etf", "A")       # partial first month
    m = _frame(pd.date_range("1999-11-30", "2000-03-31", freq="ME"), 0.05, "idx", "C")
    A = am.assemble(cash_d, cash_m, {"em": am.AssetFiles("em", d, m)})
    assert A.monthly.index[0] == pd.Timestamp("1999-11-30")
    assert A.monthly.index[-1] == pd.Timestamp("2000-03-31")        # April incomplete
    assert A.monthly.loc["2000-01-31", "em"] == pytest.approx(0.05)  # partial daily month -> track file
    nfeb = ((cal >= "2000-02-01") & (cal <= "2000-02-29")).sum()
    assert A.monthly.loc["2000-02-29", "em"] == pytest.approx(1.001 ** nfeb - 1)
    assert A.tiers.loc["2000-01-31", "em"] == "C" and A.tiers.loc["2000-02-29", "em"] == "A"
    assert A.daily.index[0] == pd.Timestamp("2000-01-10")


# ---------------------------------------------------------------- real data
need = pytest.mark.skipif(not (PROCESSED / "panel_daily.parquet").exists(), reason="panel not assembled")


@pytest.fixture(scope="module")
def panel():
    from assetforecast.data.panel import load_panel
    return load_panel()


@pytest.fixture(scope="module")
def seg():
    return pd.read_csv(PROCESSED / "segments.csv", parse_dates=["start", "end"])


@pytest.mark.realdata
@need
def test_load_panel_and_asof(panel):
    assert list(panel.daily.columns) == [a for a in ASSET_IDS if a in panel.daily.columns]
    assert panel.cash_daily.index.equals(panel.daily.index)
    t = pd.Timestamp("2008-10-15")
    v = panel.asof(t)
    assert v.daily.index.max() <= t and v.monthly.index.max() <= t
    assert (v.predictors["available_date"] <= t).all()
    w = v.predictors_wide()
    assert w.index.max() <= t
    assert len(v.eligible()) >= 5


@pytest.mark.realdata
@need
def test_daily_index_is_cash_calendar(panel):
    d = panel.daily
    assert d.index.is_monotonic_increasing and d.index.is_unique
    cal = pd.DatetimeIndex(pd.read_parquet(ASSET_DIR / "cash_daily.parquet").index).as_unit("ns")
    span = cal[(cal >= d.index[0]) & (cal <= d.index[-1])]
    assert d.index.equals(span)
    assert d.index[-1] == cal[-1]
    assert panel.cash_daily.notna().all() and panel.cash_monthly.notna().all()


@pytest.mark.realdata
@need
def test_no_impossible_returns_and_no_cash_nans(panel):
    assert (panel.daily.stack().dropna() > -1).all() and (panel.monthly.stack().dropna() > -1).all()
    assert not (panel.daily.notna() & panel.excess_daily.isna()).any().any()
    assert not (panel.monthly.notna() & panel.excess_monthly.isna()).any().any()


@pytest.mark.realdata
@need
def test_no_partial_final_month(panel):
    last = panel.monthly.index[-1]
    days_in_last = panel.daily.index[panel.daily.index.to_period("M") == last.to_period("M")]
    assert am.month_is_complete(days_in_last[-1])
    assert panel.daily.index[-1].to_period("M") >= last.to_period("M")
    if panel.daily.index[-1].to_period("M") > last.to_period("M"):
        assert not am.month_is_complete(panel.daily.index[-1])


@pytest.mark.realdata
@need
def test_monthly_equals_compounded_daily_in_daily_eras(panel):
    d = panel.daily
    per = d.index.to_period("M")
    cal = pd.DatetimeIndex(pd.read_parquet(ASSET_DIR / "cash_daily.parquet").index).as_unit("ns")
    cs = pd.Series(cal, index=cal)
    first_day = cs.groupby(cal.to_period("M")).min()
    last_day = cs.groupby(cal.to_period("M")).max()
    comp = np.expm1(np.log1p(d).groupby(per).sum(min_count=1))
    for a in d.columns:
        fv, lv = d[a].first_valid_index(), d[a].last_valid_index()
        ok = (first_day >= fv) & (last_day <= lv)
        months = ok.index[ok.values]
        months = months[months <= panel.monthly.index[-1].to_period("M")]
        m = panel.monthly[a]
        m.index = m.index.to_period("M")
        np.testing.assert_allclose(comp.loc[months, a].values, m.loc[months].values, atol=1e-12, err_msg=a)


@pytest.mark.realdata
@need
def test_cash_monthly_covers_all_assets(panel):
    first = min(panel.monthly[a].first_valid_index() for a in panel.monthly.columns)
    assert panel.monthly.index[0] == first
    assert panel.cash_monthly.loc[first:].notna().all()


@pytest.mark.realdata
@need
def test_tiers_match_data(panel):
    t = panel.tiers
    assert t.index.equals(panel.monthly.index)
    assert (t.notna() == panel.monthly.notna()).all().all()
    assert set(pd.unique(t.stack().dropna())) <= set("ABCD")


@pytest.mark.realdata
@need
def test_segments_agree_with_files(panel, seg):
    assert list(seg.columns) == am.SEGMENT_COLUMNS
    for a in panel.daily.columns:
        f = pd.read_parquet(ASSET_DIR / f"{a}_daily.parquet")
        srcs = set(f["source"].astype(str))
        assert srcs == set(seg[(seg.asset == a) & (seg.freq == "D")]["source"]), a
        # no data before the documented start
        assert panel.daily[a].first_valid_index() >= seg[(seg.asset == a) & (seg.freq == "D")]["start"].min()
        assert panel.monthly[a].first_valid_index() >= seg[seg.asset == a]["start"].min() - pd.offsets.MonthEnd(1)
    assert seg["tier"].isin(list("ABCD")).all()


@pytest.mark.realdata
@need
def test_predictors_clean_and_in_standard_units():
    p = pd.read_parquet(PROCESSED / "predictors.parquet")
    cat = pd.read_csv(PROCESSED / "predictors_catalog.csv")
    assert list(p.columns) == am.PREDICTOR_COLUMNS
    assert list(cat.columns) == am.CATALOG_COLUMNS
    assert (p["available_date"] >= p["obs_date"]).all()
    assert not p.duplicated(["series", "obs_date"]).any()
    assert cat["series"].is_unique and set(cat["series"]) == set(p["series"])
    assert not p["series"].str.contains("excess_bond_premium").any()
    assert cat["notes"].str.contains("excess bond premium", case=False).any()
    q = p.groupby("series")["value"].apply(lambda s: s.abs().quantile(0.99))
    dec = cat.loc[cat.unit.str.startswith("decimal") & ~cat.unit.str.startswith("decimal vol"), "series"]
    assert (q[dec] < 0.5).all(), q[dec][q[dec] >= 0.5]          # rates/spreads/yoy in decimals
    vols = cat.loc[cat.unit.str.startswith("decimal vol"), "series"]
    assert (q[vols] < 4).all() and (p[p.series.isin(vols)]["value"] > 0).all()
    dur = cat.loc[cat.unit == "years", "series"]
    assert ((q[dur] > 3) & (q[dur] < 30)).all()


@pytest.mark.realdata
@pytest.mark.slow
@need
def test_build_is_idempotent():
    out = am.build(write=False, evidence=False, verbose=False)
    d = pd.read_parquet(PROCESSED / "panel_daily.parquet")
    m = pd.read_parquet(PROCESSED / "panel_monthly.parquet")
    pd.testing.assert_frame_equal(out["A"].daily.astype(float), d, check_freq=False)
    pd.testing.assert_frame_equal(out["A"].monthly.astype(float), m, check_freq=False)
