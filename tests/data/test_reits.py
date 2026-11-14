"""Tests for the ``reits`` loader. Offline tests read data/processed (``realdata``);
download tests are marked ``network``."""
import numpy as np
import pandas as pd
import pytest

from volmodel.data.sources import reits as R
from volmodel.paths import ASSET_DIR

DAILY = ASSET_DIR / "reits_daily.parquet"
MONTHLY = ASSET_DIR / "reits_monthly.parquet"


@pytest.fixture(scope="module")
def daily():
    if not DAILY.exists():
        pytest.skip("processed reits files not built")
    return pd.read_parquet(DAILY)


@pytest.fixture(scope="module")
def monthly():
    if not MONTHLY.exists():
        pytest.skip("processed reits files not built")
    return pd.read_parquet(MONTHLY)


@pytest.mark.realdata
def test_format(daily, monthly):
    for df in (daily, monthly):
        assert list(df.columns) == ["ret", "source", "tier"]
        assert df.index.name == "date"
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_unique and df.index.is_monotonic_increasing
        assert df["ret"].notna().all()
        assert df["ret"].gt(-1).all()
        assert set(df["tier"]) <= set("ABCD")
        assert df["source"].notna().all()
    assert (monthly.index == monthly.index + pd.offsets.MonthEnd(0)).all()
    # no weekend dates in the daily file
    assert (daily.index.dayofweek < 5).all()


@pytest.mark.realdata
def test_coverage(daily, monthly):
    assert monthly.index[0] == pd.Timestamp("1972-01-31")
    assert daily.index[0] <= pd.Timestamp("1986-11-30")
    assert daily.index[-1] >= pd.Timestamp("2026-10-30")
    # monthly runs to the last COMPLETE month of the daily data
    assert monthly.index[-1] <= daily.index[-1] + pd.offsets.MonthEnd(0)
    assert monthly.index[-1] >= daily.index[-1] - pd.offsets.MonthEnd(1) - pd.Timedelta(days=1)
    # contiguous months, no gaps
    full = pd.date_range(monthly.index[0], monthly.index[-1], freq="ME")
    assert monthly.index.equals(full)
    # daily: no gap longer than the 9/11 market closure
    assert daily.index.to_series().diff().max() <= pd.Timedelta(days=7)  # 2001-09-11 closure


@pytest.mark.realdata
def test_monthly_equals_compounded_daily(daily, monthly):
    m = (1 + daily["ret"]).groupby(pd.Grouper(freq="ME")).prod() - 1
    common = m.index[1:].intersection(monthly.index)   # first daily month is partial
    assert len(common) > 400
    assert np.allclose(m.loc[common], monthly.loc[common, "ret"], atol=1e-12)
    assert (monthly.loc[common, "tier"] != "C").all()
    assert (monthly.loc[: common[0] - pd.offsets.MonthEnd(1), "tier"] == "C").all()


@pytest.mark.realdata
def test_tiers_and_segments(daily, monthly):
    seg = R.segments(daily, monthly)
    assert list(seg["tier"]) == ["C", "B", "A", "A"]
    assert set(daily.loc[:"1996-05-31", "tier"]) == {"B"}
    assert set(daily.loc["1996-06-03":, "tier"]) == {"A"}
    # segments abut: each starts after the previous ends
    d = seg[seg["freq"] == "D"]
    assert (d["start"].iloc[1:].values > d["end"].iloc[:-1].values).all()


@pytest.mark.realdata
def test_known_episodes(daily, monthly):
    m = monthly["ret"]

    def cum(a, b):
        return (1 + m.loc[a:b]).prod() - 1

    assert cum("1973-01", "1974-12") < -0.25
    assert m.loc["1987-10"].iloc[0] < -0.10
    assert cum("1998-01", "1998-12") < -0.10
    assert cum("2007-02", "2009-02") < -0.60
    assert m.loc["2020-03"].iloc[0] < -0.15
    assert cum("2022-01", "2022-12") < -0.20
    d = daily["ret"]
    assert d.loc["1987-10-19"] < -0.05
    assert d.loc["2008-12-01"] < -0.15
    assert d.abs().max() < 0.25


@pytest.mark.realdata
def test_vol_regimes(daily, monthly):
    v = R.vol_by_decade(monthly, daily)
    # GFC decade far more volatile than the 1980s/90s; full-history vol plausible
    assert v.loc[2000, "vol_monthly"] > 1.7 * v.loc[1990, "vol_monthly"]
    assert 0.12 < np.log1p(monthly["ret"]).std() * np.sqrt(12) < 0.25
    assert 0.20 < np.log1p(daily.loc["2004-10":, "ret"]).std() * np.sqrt(252) < 0.35


def test_overlap_stats_identity():
    x = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500),
                  index=pd.bdate_range("2000-01-03", periods=500))
    s = R.overlap_stats(x, x, 252)
    assert s["corr"] == pytest.approx(1) and s["te"] == pytest.approx(0)
    assert s["vol_ratio"] == pytest.approx(1)


@pytest.mark.network
def test_download_nareit(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "CACHE", tmp_path)
    n = R.nareit_monthly(refresh=True)
    s = n[R.NAREIT_SERIES].dropna()
    assert s.index[0] == pd.Timestamp("1972-01-31")
    assert s.index[-1] >= pd.Timestamp("2026-09-30")
    assert {"Equity REITs", "Mortgage REITs"} <= set(n.columns)
    assert s.loc["2008"].add(1).prod() - 1 < -0.30


@pytest.mark.network
def test_download_yahoo(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "CACHE", tmp_path)
    p = R.yahoo_adjclose("VNQ", refresh=True)
    assert p.index[0] <= pd.Timestamp("2004-10-01")
    assert len(p) > 5000
