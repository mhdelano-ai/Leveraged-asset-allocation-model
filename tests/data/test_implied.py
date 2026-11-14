"""Tests for the implied-volatility predictor track."""
import numpy as np
import pandas as pd
import pytest

from assetforecast.data.panel import PREDICTOR_COLUMNS
from assetforecast.data.sources import implied as m


# ----------------------------------------------------------------- offline unit
def test_clean_drops_isolated_bad_print_and_stale_runs():
    idx = pd.bdate_range("2020-01-01", periods=12)
    s = pd.Series([15, 15.5, 2.7, 15.2, 16, 16, 16, 16, 16, 17, 18, 18], index=idx, dtype=float)
    c, info = m.clean(s)
    assert idx[2] not in c.index and info["spikes"] == [idx[2].date()]
    # run of five 16s: only the first survives; a pair of 18s is kept
    assert (c == 16).sum() == 1 and info["stale"] == 4
    assert (c == 18).sum() == 2


def test_clean_keeps_spike_confirmed_by_vix():
    idx = pd.bdate_range("2011-08-03", periods=5)
    s = pd.Series([30, 37.6, 68.0, 43.4, 40], index=idx, dtype=float)
    vix = pd.Series([23, 32, 48, 35, 39], index=idx, dtype=float)
    assert idx[2] in m.clean(s, ref=vix)[0].index


def test_clean_drops_nonpositive():
    idx = pd.bdate_range("2020-01-01", periods=4)
    c, info = m.clean(pd.Series([7.7, -19738470.0, 7.8, 7.9], index=idx))
    assert (c > 0).all() and info["nonpositive"] == 1


# ------------------------------------------------------------ processed file
@pytest.fixture(scope="module")
def df():
    if not m.OUT_FILE.exists():
        pytest.skip("implied.parquet not built")
    return pd.read_parquet(m.OUT_FILE)


@pytest.mark.realdata
def test_schema(df):
    assert list(df.columns) == PREDICTOR_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(df["obs_date"])
    assert pd.api.types.is_datetime64_any_dtype(df["available_date"])
    assert (df["available_date"] == df["obs_date"]).all()
    assert df["series"].str.startswith("iv_").all()
    assert np.isfinite(df["value"]).all()


@pytest.mark.realdata
def test_no_duplicates(df):
    assert not df.duplicated(["series", "obs_date"]).any()


@pytest.mark.realdata
def test_core_series_present_and_long(df):
    g = df.groupby("series")["obs_date"]
    for name, start in [("iv_vix", "1990-01-02"), ("iv_vxo", "1986-01-02"),
                        ("iv_vix1y", "2007-01-31"), ("iv_skew_index", "1990-01-31"),
                        ("iv_gvz", "2009-12-31"), ("iv_ovx", "2009-12-31"),
                        ("iv_vxeem", "2011-12-31"), ("iv_vxefa", "2008-12-31"),
                        ("iv_vxtlt", "2004-12-31")]:
        assert name in g.groups, name
        assert g.min()[name] <= pd.Timestamp(start), name


@pytest.mark.realdata
def test_value_ranges(df):
    w = df.groupby("series")["value"]
    lo, hi = w.min(), w.max()
    for s in lo.index:
        if s == "iv_skew_index":
            assert 95 < lo[s] and hi[s] < 200
        elif s.endswith("_slope"):
            assert -0.8 < lo[s] and hi[s] < 1.5
        elif s == "iv_move":
            assert 0.002 < lo[s] and hi[s] < 0.03       # 20bp .. 300bp yield vol
        elif s == "iv_vvix":
            assert 0.3 < lo[s] and hi[s] < 3.0
        elif s == "iv_ovx":
            assert 0.1 < lo[s] and hi[s] < 4.0          # 2020-04-21 = 3.25 is real
        else:
            assert 0.02 < lo[s] and hi[s] < 1.6, s       # decimals, VXO 1987 = 1.50
    med = w.median()
    assert 0.12 < med["iv_vix"] < 0.25


@pytest.mark.realdata
def test_live_flags(df):
    t = m.summary_table(df)
    assert t.loc["iv_vix", "live"] and t.loc["iv_vix1y", "live"]
    assert not t.loc["iv_vxo", "live"]


@pytest.mark.realdata
def test_vix_vxo_relationship(df):
    w = df.pivot(index="obs_date", columns="series", values="value")
    j = w[["iv_vix", "iv_vxo"]].dropna().loc[:"2003"]
    assert j.corr().iloc[0, 1] > 0.95
    sp = w["iv_vix_vxo_spliced"].dropna()
    assert sp.index[0] == pd.Timestamp("1986-01-02")
    assert np.allclose(sp.loc["1990":], w["iv_vix"].dropna())


# --------------------------------------------------------------------- network
@pytest.mark.network
def test_cboe_download(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "CACHE", tmp_path)
    s = m.cboe_history("VIX1Y")
    assert s.index[0] <= pd.Timestamp("2007-01-31") and len(s) > 4000
    assert 8 < s.median() < 40


@pytest.mark.network
def test_cboe_vxo_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "CACHE", tmp_path)
    s = m.cboe_vxo()
    assert s.index[0] == pd.Timestamp("1986-01-02")
    assert s.loc["1987-10-19"] > 100


@pytest.mark.network
def test_yahoo_download(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "CACHE", tmp_path)
    monkeypatch.setattr(m, "_YAHOO_DOWN", False)
    s = m.yahoo_series("^VIX")
    if s is None:
        pytest.skip("Yahoo rate-limited")
    assert len(s) > 5000 and 8 < s.median() < 40
