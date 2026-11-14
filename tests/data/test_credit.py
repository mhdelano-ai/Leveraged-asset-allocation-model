"""Tests for the credit track (ig_credit, hy_credit, credit_spreads predictors)."""
import numpy as np
import pandas as pd
import pytest

from assetforecast.data.sources import credit
from assetforecast.paths import ASSET_DIR, PREDICTOR_DIR

ASSETS = ["ig_credit", "hy_credit"]


# ------------------------------------------------------------------ offline unit tests
def test_par_bond_constant_yield_earns_coupon():
    y = pd.Series(0.06, index=pd.date_range("2000-01-31", periods=24, freq="ME"))
    r = credit.par_bond_returns(y, 10).dropna()
    # unchanged yield: price ~ par (one month of roll-down on a par bond is ~0) + coupon/12
    assert np.allclose(r, 0.005, atol=2e-4)


def test_par_bond_duration_sign_and_size():
    idx = pd.date_range("2000-01-31", periods=2, freq="ME")
    r = credit.par_bond_returns(pd.Series([0.05, 0.06], index=idx), 10).iloc[-1]
    # +100bp on a 10y par bond: modified duration ~7.7 -> price change about -7%
    assert -0.08 < r < -0.06


def test_vol_match_weight_recovers_scale():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-31", periods=400, freq="ME")
    a = pd.Series(rng.normal(0, 0.03, 400), idx)
    b = pd.Series(np.zeros(400), idx)
    w = credit.vol_match_weight(a, b, 0.6 * a)
    assert abs(w - 0.6) < 0.01


def test_stale_stats_detects_smoothing():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2000-01-03", periods=3000)
    e = pd.Series(rng.normal(0, 0.004, 3000), idx)
    smooth = 0.5 * e + 0.3 * e.shift(1) + 0.2 * e.shift(2)
    s0 = credit.stale_stats(e)
    s1 = credit.stale_stats(smooth.dropna())
    assert abs(s0["ac1"]) < 0.06 and s1["ac1"] > 0.3
    assert s1["m_over_d"] > 1.15 > s0["m_over_d"] - 0.15


# ------------------------------------------------------------------ processed files
def _load(asset, freq):
    p = ASSET_DIR / f"{asset}_{freq}.parquet"
    if not p.exists():
        pytest.skip(f"{p} not built")
    return pd.read_parquet(p)


@pytest.mark.realdata
@pytest.mark.parametrize("asset", ASSETS)
def test_files_follow_contract(asset):
    for freq in ("daily", "monthly"):
        df = _load(asset, freq)
        assert list(df.columns) == ["ret", "source", "tier"]
        assert df.index.name == "date"
        assert df.index.is_monotonic_increasing and df.index.is_unique
        assert df["ret"].notna().all()
        assert set(df["tier"].unique()) <= set("ABCD")
        assert (df["ret"] > -0.5).all() and (df["ret"] < 0.5).all()
    m = _load(asset, "monthly")
    assert (m.index == m.index + pd.offsets.MonthEnd(0)).all()


@pytest.mark.realdata
@pytest.mark.parametrize("asset", ASSETS)
def test_monthly_equals_compounded_daily(asset):
    d, m = _load(asset, "daily"), _load(asset, "monthly")
    c = np.expm1(np.log1p(d["ret"]).resample("ME").sum())
    common = c.index[1:-1].intersection(m.index)
    assert len(common) > 200
    assert np.allclose(c.loc[common], m.loc[common, "ret"], atol=1e-10)


@pytest.mark.realdata
def test_history_spans():
    ig_d, ig_m = _load("ig_credit", "daily"), _load("ig_credit", "monthly")
    hy_d, hy_m = _load("hy_credit", "daily"), _load("hy_credit", "monthly")
    assert ig_m.index[0] <= pd.Timestamp("1926-01-31")
    assert ig_d.index[0] <= pd.Timestamp("1980-01-31")
    assert hy_m.index[0] <= pd.Timestamp("1980-12-31")
    assert hy_d.index[0] <= pd.Timestamp("1985-01-31")
    for d in (ig_d, hy_d):
        assert d.index[-1] >= pd.Timestamp("2026-10-01")
        # business-day data, no multi-week holes in the daily file
        assert d.index.to_series().diff().dt.days.max() <= 7
    # the ETF eras are the ETFs
    assert (ig_d.loc["2003":, "source"] == "yahoo_LQD").all()
    assert (hy_d.loc["2008":, "source"] == "yahoo_HYG").all()
    # no forward-filled pre-1985 HY fund data in the daily file
    assert (hy_d.loc[:"1985-03", "ret"] == 0).mean() < 0.6


@pytest.mark.realdata
def test_known_episodes():
    ig = _load("ig_credit", "monthly")["ret"]
    hy = _load("hy_credit", "monthly")["ret"]

    def cum(s, a, b):
        return np.expm1(np.log1p(s.loc[a:b]).sum())
    assert cum(hy, "2008-01", "2008-12") < -0.15
    assert cum(hy, "2009-01", "2009-12") > 0.20
    assert cum(ig, "2022-01", "2022-12") < -0.12
    assert cum(hy, "2020-03", "2020-03") < -0.05
    assert cum(hy, "1990-01", "1990-12") < 0
    assert cum(hy, "1991-01", "1991-12") > 0.2
    assert cum(ig, "1994-01", "1994-12") < 0


@pytest.mark.realdata
def test_vol_levels_plausible():
    for asset, lo, hi in (("ig_credit", 0.04, 0.12), ("hy_credit", 0.05, 0.15)):
        m = _load(asset, "monthly")["ret"].loc["1985":]
        v = np.log1p(m).std() * np.sqrt(12)
        assert lo < v < hi, (asset, v)


@pytest.mark.realdata
def test_predictor_file():
    p = PREDICTOR_DIR / "credit_spreads.parquet"
    if not p.exists():
        pytest.skip("credit_spreads not built")
    df = pd.read_parquet(p)
    assert list(df.columns) == ["obs_date", "available_date", "series", "value"]
    assert df["value"].notna().all()
    assert (df["available_date"] > df["obs_date"]).all()
    assert not df.duplicated(["series", "obs_date"]).any()
    s = set(df["series"])
    assert {"moodys_baa_minus_aaa_m", "moodys_baa_minus_ust10y_m", "gz_credit_spread_m"} <= s
    baa_aaa = df[df.series == "moodys_baa_minus_aaa_m"].set_index("obs_date")["value"]
    assert 0.2 < baa_aaa.median() < 2.0  # percent
    assert baa_aaa.loc["2008-12"].iloc[0] > 2.5  # crisis peak (3.38 in Dec 2008)


# ------------------------------------------------------------------ network
@pytest.mark.network
def test_yahoo_download(tmp_path, monkeypatch):
    monkeypatch.setattr(credit, "CACHE", tmp_path)
    df = credit.yahoo_frame("LQD", refresh=True)
    assert df.index[0] <= pd.Timestamp("2002-08-01")
    assert (df["div"] > 0).sum() > 200
    assert (tmp_path / "yahoo_LQD.json").exists()


@pytest.mark.network
def test_fed_ebp_download(tmp_path, monkeypatch):
    monkeypatch.setattr(credit, "CACHE", tmp_path)
    e = credit.fed_ebp(refresh=True)
    assert e.index[0] == pd.Timestamp("1973-01-31")
    assert {"gz_spread", "ebp"} <= set(e.columns)
