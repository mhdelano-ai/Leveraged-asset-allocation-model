"""Tests of assetforecast.covhistory: synthetic and seeded, plus realdata checks."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from assetforecast import covhistory as ch
from assetforecast.bridge import _anchor, get_bridge, vol_uncertainty
from assetforecast.data.panel import Panel
from assetforecast.data.synthetic import make_synthetic_panel
from assetforecast.paths import COV_HISTORY, FORECASTS_OUT


def _stored(panel: Panel, origins) -> dict:
    """A cheap PD 21-day 'forecast' per origin: 12 x mean squared monthly returns, last 60 months."""
    out = {}
    for o in origins:
        v = panel.asof(o)
        a = v.eligible(36)
        x = v.excess_monthly[a].iloc[-60:].fillna(0.0).to_numpy()
        C = 12 * x.T @ x / len(x) + 1e-4 * np.eye(len(a))
        out[pd.Timestamp(o)] = pd.DataFrame(C, a, a)
    return out


@pytest.fixture(scope="module")
def synth():
    return make_synthetic_panel(seed=7)


@pytest.fixture(scope="module")
def origins(synth):
    return list(pd.date_range("1979-06-30", "2015-12-31", freq="ME")[::5])


@pytest.fixture(scope="module")
def built(synth, origins):
    stored = _stored(synth, origins)
    hist, orig = ch.history_from_forecasts(stored, synth, "vr_market")
    return stored, hist, orig


def test_layout_and_per_origin_facts(synth, origins, built):
    stored, hist, orig = built
    assert list(hist.columns) == ch.HIST_COLUMNS
    assert list(orig.columns) == ch.ORIGIN_COLUMNS
    assert list(orig["origin"]) == origins
    for o, row in orig.set_index("origin").iterrows():
        assert row["assets"].split("|") == list(stored[o].index) == synth.asof(o).eligible(36)
        assert row["n_assets"] == len(stored[o])
        assert row["bridge_anchor"] == _anchor(synth.asof(o))
        assert row["bridge_anchor"].month == 12 and row["bridge_anchor"] <= o
        assert row["tail_s2_origin"] == vol_uncertainty(synth.asof(o))["s2"]
    assert len(hist) == int((orig["n_assets"] ** 2).sum())


def test_cov_21d_is_the_stored_forecast_and_annual_is_the_cold_bridge(synth, built):
    stored, hist, _ = built
    for o, A in ch._matrices(hist, "cov_annual").items():
        C = ch._matrices(hist, "cov_21d")[o]
        assert C.equals(stored[o])
        cold = get_bridge("vr_market").cov_annual(stored[o], synth.asof(o))   # fresh instance, one origin
        np.testing.assert_array_equal(A.to_numpy(), cold.to_numpy())
        assert list(A.index) == list(stored[o].index)


def test_variance_ratios_change_only_at_the_anchor(built):
    _, hist, orig = built
    d = hist[hist["asset_i"] == hist["asset_j"]].copy()
    d["vr"] = d["cov_annual"] / d["cov_21d"]
    d = d.merge(orig[["origin", "bridge_anchor", "assets"]], on="origin")
    n_groups = 0
    for _, g in d.groupby(["bridge_anchor", "assets", "asset_i"]):
        assert np.ptp(g["vr"].to_numpy()) < 1e-12
        n_groups += g["origin"].nunique() > 1
    assert n_groups > 0                       # some anchors cover several origins


def test_no_look_ahead(synth, origins):
    t = pd.Timestamp("2001-07-31")
    rng = np.random.default_rng(0)
    m2, d2 = synth.monthly.copy(), synth.daily.copy()
    later = m2.index > t
    m2.loc[later] = m2.loc[later] * 3 + rng.normal(0, 0.05, m2.loc[later].shape)
    d2.loc[d2.index > t] = d2.loc[d2.index > t] * -2
    altered = Panel(daily=d2, monthly=m2, cash_daily=synth.cash_daily, cash_monthly=synth.cash_monthly,
                    tiers=synth.tiers, predictors=synth.predictors)
    early = [o for o in origins if o <= t]
    stored = _stored(synth, early)
    h1, o1 = ch.history_from_forecasts(stored, synth)
    h2, o2 = ch.history_from_forecasts(stored, altered)
    pd.testing.assert_frame_equal(h1, h2)
    pd.testing.assert_frame_equal(o1, o2)


def test_pd_and_symmetric(built):
    _, hist, _ = built
    c = ch.check_matrices(hist)
    assert c["all_pd"]
    assert c["cov_annual"]["max_rel_asymmetry"] == 0.0


def test_eligibility_mismatch_raises(synth):
    o = pd.Timestamp("1995-12-31")
    st = _stored(synth, [o])
    a = list(st[o].index)[:-1]
    with pytest.raises(ch.CovHistoryError, match="eligible"):
        ch.history_from_forecasts({o: st[o].loc[a, a]}, synth)


def test_refuses_dropped_tiers(monkeypatch, tmp_path):
    monkeypatch.setenv("ASSETFORECAST_DROP_TIERS", "D")
    with pytest.raises(ch.CovHistoryError, match="ASSETFORECAST_DROP_TIERS"):
        ch.build(results_dir=tmp_path)
    with pytest.raises(ch.CovHistoryError, match="ASSETFORECAST_DROP_TIERS"):
        ch.truncation_check(pd.DataFrame(columns=ch.HIST_COLUMNS), [])


def test_write_load_round_trip(built, tmp_path):
    stored, hist, orig = built
    hist = hist.copy()
    hist.attrs["meta"] = {"model": "synthetic"}
    p = ch.write(hist, orig, tmp_path / "cov_history.parquet")
    assert (tmp_path / ch.ORIGINS_FILE).exists()
    meta = json.loads((tmp_path / ch.META_FILE).read_text())
    assert meta["model"] == "synthetic" and meta["n_origins"] == len(orig) and meta["checks"]["all_pd"]
    a21, aa = ch.load("21d", p), ch.load("annual", p)
    assert sorted(a21) == sorted(stored)
    ref = ch._matrices(hist, "cov_annual")
    for o in stored:
        pd.testing.assert_frame_equal(a21[o], stored[o], check_exact=True)
        pd.testing.assert_frame_equal(aa[o], ref[o], check_exact=True)
    o2 = pd.read_csv(tmp_path / ch.ORIGINS_FILE, parse_dates=["origin", "bridge_anchor"])
    assert list(o2.columns) == ch.ORIGIN_COLUMNS
    np.testing.assert_allclose(o2["tail_s2_origin"], orig["tail_s2_origin"], rtol=1e-9)
    with pytest.raises(ValueError):
        ch.load("monthly", p)


def test_to_cov_columns(built):
    from assetforecast.returns.schema import COV_COLUMNS
    _, hist, orig = built
    df = ch.to_cov_columns(hist)
    assert list(df.columns) == COV_COLUMNS
    assert (df["model"] == "af_primary").all() and (df["horizon"] == 12).all()
    assert df["horizon"].dtype == np.int64 and pd.api.types.is_datetime64_any_dtype(df["date"])
    np.testing.assert_array_equal(df["cov"].to_numpy(), hist["cov_annual"].to_numpy())
    dates = pd.DatetimeIndex(df["date"].unique())
    assert (dates == dates + pd.offsets.MonthEnd(0)).all()
    with pytest.raises(ValueError):
        ch.to_cov_columns(hist, horizon=1)


# ------------------------------------------------------------------ real data
@pytest.mark.realdata
def test_published_origin_reproduced():
    """The committed forecast's cov_21d, bridged here, gives its cov_annual (8-decimal files)."""
    from assetforecast.data.panel import load_panel
    d = FORECASTS_OUT / "latest"
    if not (d / "cov_21d.csv").exists():
        pytest.skip("no published forecast")
    o = pd.Timestamp(json.loads((d / "forecast.json").read_text())["origin"])
    C = pd.read_csv(d / "cov_21d.csv", index_col=0)
    hist, orig = ch.history_from_forecasts({o: C}, load_panel(drop_tiers=""))
    A = ch._matrices(hist, "cov_annual")[o]
    pub = pd.read_csv(d / "cov_annual.csv", index_col=0)
    assert np.abs(A.to_numpy() - pub.to_numpy()).max() < 5e-8


@pytest.mark.realdata
def test_written_history_matches_latest_and_is_pd():
    if not COV_HISTORY.exists():
        pytest.skip("results/covariance/cov_history.parquet not built")
    meta = json.loads((COV_HISTORY.parent / ch.META_FILE).read_text())
    assert meta["panel"]["tiers_dropped"] == [] and meta["checks"]["all_pd"]
    A = ch.load("annual")
    last = max(A)
    pub = pd.read_csv(FORECASTS_OUT / "latest" / "cov_annual.csv", index_col=0)
    if str(last.date()) == json.loads((FORECASTS_OUT / "latest" / "forecast.json").read_text())["origin"]:
        assert np.abs(A[last].reindex_like(pub).to_numpy() - pub.to_numpy()).max() <= 5e-9 + 1e-15
