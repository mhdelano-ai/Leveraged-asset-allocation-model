"""Tests for the real_assets track (gold, commodities)."""
import numpy as np
import pandas as pd
import pytest

from assetforecast.data.sources import real_assets as ra
from assetforecast.paths import ASSET_DIR

ASSETS = ["gold", "commodities"]


def _load(asset, freq):
    p = ASSET_DIR / f"{asset}_{freq}.parquet"
    if not p.exists():
        pytest.skip(f"{p} not built")
    return pd.read_parquet(p)


# ------------------------------------------------------------------ offline
def test_to_monthly_compounds():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    r = pd.Series(0.001, index=idx)
    m = ra.to_monthly(r)
    assert list(m.index) == list(pd.date_range("2020-01-31", periods=3, freq="ME"))
    assert np.isclose(m.iloc[0], 1.001 ** len(idx[idx.month == 1]) - 1)


def test_overlap_stats_identity():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500),
                  index=pd.bdate_range("2000-01-03", periods=500))
    s = ra.overlap_stats(r, r, "D")
    assert np.isclose(s["corr"], 1) and np.isclose(s["vol_ratio"], 1) and s["te"] < 1e-12


@pytest.mark.realdata
@pytest.mark.parametrize("asset", ASSETS)
@pytest.mark.parametrize("freq", ["daily", "monthly"])
def test_format(asset, freq):
    df = _load(asset, freq)
    assert {"ret", "source", "tier"} <= set(df.columns)
    assert df.index.name == "date" and df.index.is_monotonic_increasing and df.index.is_unique
    assert df["ret"].notna().all()
    assert set(df["tier"]) <= set("ABCD")
    assert (df["ret"] > -0.6).all() and (df["ret"] < 1.0).all()
    if freq == "monthly":
        assert (df.index == df.index + pd.offsets.MonthEnd(0)).all()
    # runs to (near) the present
    assert df.index[-1] >= pd.Timestamp("2026-10-01")


@pytest.mark.realdata
@pytest.mark.parametrize("asset", ASSETS)
def test_monthly_equals_compounded_daily(asset):
    d, m = _load(asset, "daily"), _load(asset, "monthly")
    cd = ra.to_monthly(d["ret"])
    common = cd.index.intersection(m.index)[:-1]  # last month may be partial in both
    assert len(common) > 100
    assert np.allclose(cd.loc[common], m.loc[common, "ret"], atol=1e-10)
    # every month that has daily data is taken from the daily data
    assert set(cd.index[:-1]) <= set(m.index)


@pytest.mark.realdata
@pytest.mark.parametrize("asset", ASSETS)
def test_no_monthly_gaps(asset):
    m = _load(asset, "monthly")
    full = pd.date_range(m.index[0], m.index[-1], freq="ME")
    assert m.index.equals(full)


@pytest.mark.realdata
def test_gold_history_and_episodes():
    d, m = _load("gold", "daily"), _load("gold", "monthly")
    assert m.index[0] == pd.Timestamp("1971-09-30")       # free-market start
    assert d.index[0] < pd.Timestamp("1971-09-03")
    px = (1 + m["ret"]).cumprod()
    # 1979-80 spike: Jan-1980 month-end level > 1.5x Jun-1979
    assert px.loc["1980-01-31"] / px.loc["1979-06-30"] > 1.5
    # 2011 peak -> 2013 trough: drawdown of more than 25%
    assert px.loc["2013-12-31"] / px.loc["2011-08-31"] < 0.75
    # 1980 daily vol far above 1990s daily vol
    v = np.log1p(d["ret"]).groupby(d.index.year).std() * np.sqrt(252)
    assert v.loc[1980] > 0.35 and v.loc[1995] < 0.12
    assert (m.loc["2004-12":, "tier"] == "A").all()


@pytest.mark.realdata
def test_commodities_episodes():
    m = _load("commodities", "monthly")
    px = (1 + m["ret"]).cumprod()
    assert px.loc["2008-12-31"] / px.loc["2008-06-30"] < 0.65   # 2008 crash
    assert px.loc["2016-01-31"] / px.loc["2014-06-30"] < 0.65   # 2014-16 bust
    assert px.loc["2022-06-30"] / px.loc["2021-12-31"] > 1.10   # 2022 spike
    d = _load("commodities", "daily")
    assert (d["tier"] == "B").all()                              # BCOM index throughout
    assert d.index[0] < pd.Timestamp("1991-03-01")
    assert m.index[0] == pd.Timestamp("1926-07-31")
    # 1990 Gulf-war spike and the 2020-04 low are visible
    assert px.loc["1990-10-31"] / px.loc["1990-07-31"] > 1.05
    assert px.loc["2020-04-30"] / px.loc["2020-01-31"] < 0.85


@pytest.mark.realdata
def test_commodities_excess_reconciles():
    m = _load("commodities", "monthly")
    if "ret_excess_native" not in m.columns:
        pytest.skip("no excess-return segment")
    x = m.dropna(subset=["ret_excess_native"])
    assert np.allclose((1 + x["ret_excess_native"]) * (1 + x["cash_ret"]) - 1, x["ret"])


# ------------------------------------------------------------------ network
@pytest.mark.network
def test_bundesbank_responds(tmp_path):
    p = ra._fetch(ra.BBK_URL.format(key=ra.BBK_KEYS[1]) + "&startPeriod=1990-01-01&endPeriod=1990-01-31",
                  tmp_path / "x.csv", refresh=True)
    assert "1990-01-02" in p.read_text(encoding="utf-8-sig")


@pytest.mark.network
def test_ft_responds(tmp_path, monkeypatch):
    monkeypatch.setattr(ra, "RAW_DIR", tmp_path)
    ra.fetch_ft("BCOM", first_year=2025, refresh_last=False)
    s = ra.read_ft("BCOM")
    assert s.loc["2025"].size > 240 and (s > 0).all()
