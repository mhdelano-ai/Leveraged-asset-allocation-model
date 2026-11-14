"""assetforecast.forecast: the consistency check against the covariance history
(synthetic) and the layout of the published forecast (realdata)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from assetforecast import covhistory as ch
from assetforecast import forecast as F
from assetforecast.data.synthetic import make_synthetic_panel
from assetforecast.paths import FORECASTS_OUT
from assetforecast.returns import schema


@pytest.fixture(scope="module")
def history(tmp_path_factory):
    panel = make_synthetic_panel(seed=3)
    out = {}
    for o in pd.date_range("1990-01-31", "1992-12-31", freq="ME")[::6]:
        v = panel.asof(o)
        a = v.eligible(36)
        x = v.excess_monthly[a].iloc[-60:].fillna(0.0).to_numpy()
        out[o] = pd.DataFrame(12 * x.T @ x / len(x) + 1e-4 * np.eye(len(a)), a, a)
    hist, orig = ch.history_from_forecasts(out, panel, "vr_market")
    p = tmp_path_factory.mktemp("cov") / "cov_history.parquet"
    ch.write(hist, orig, p)
    return p, ch.load("21d", p), ch.load("annual", p)


def test_history_check_passes_on_the_same_matrices(history):
    p, C21, CA = history
    o = max(C21)
    res = F.check_against_history(o, C21[o], CA[o], p)
    assert res["cov_21d_max_abs_diff"] == 0.0 and res["cov_annual_max_abs_diff"] == 0.0


def test_history_check_refuses_differences(history):
    p, C21, CA = history
    o = max(C21)
    bad = CA[o].copy()
    bad.iloc[0, 0] *= 1 + 1e-9
    with pytest.raises(F.ForecastConsistencyError, match="differs"):
        F.check_against_history(o, C21[o], bad, p)
    rev = list(CA[o].index)[::-1]
    with pytest.raises(F.ForecastConsistencyError, match="assets differ"):
        F.check_against_history(o, C21[o].loc[rev, rev], CA[o], p)
    with pytest.raises(F.ForecastConsistencyError, match="no origin"):
        F.check_against_history(o + pd.offsets.MonthEnd(1), C21[o], CA[o], p)
    with pytest.raises(F.ForecastConsistencyError, match="no covariance history"):
        F.check_against_history(o, C21[o], CA[o], p.parent / "missing.parquet")


# ------------------------------------------------------------------ the published files
@pytest.fixture(scope="module")
def latest():
    d = FORECASTS_OUT / "latest"
    if not (d / "expected_returns.csv").exists():
        pytest.skip("no published forecast with the return side")
    return d


@pytest.mark.realdata
def test_published_layout(latest):
    meta = json.loads((latest / "forecast.json").read_text())
    vols = pd.read_csv(latest / "vols.csv", index_col=0)
    ports = pd.read_csv(latest / "portfolios.csv", index_col=0)
    for df in (vols, ports):
        assert not any("ex_mean" in c for c in df.columns)   # scale-mixture quantiles left the files
    er = pd.read_csv(latest / "expected_returns.csv")
    assert list(er.columns) == ["asset", "name", "group", "mean_12m", "se_12m", "ci90_lo", "ci90_hi",
                                "n_months", "first_month", "mean_12m_arith"]
    da = pd.read_csv(latest / "distribution_assets.csv")
    assert list(da.columns) == ["asset", "mean", "scale", "sd", *schema.QUANTILE_COLUMNS]
    dp = pd.read_csv(latest / "distribution_portfolios.csv")
    assert list(dp.columns) == ["portfolio", "model", "shape", "mean", "scale", "sd",
                                *schema.QUANTILE_COLUMNS, "weights"]
    assert set(dp["shape"]) == {"portfolios", "assets"}
    assert set(dp.loc[dp["shape"] == "portfolios", "model"]) == {"dist_hm_af"}
    assert set(dp.loc[dp["shape"] == "assets", "model"]) == {"dist_hm_af.assetshape"}
    assert all(isinstance(json.loads(w), dict) for w in dp["weights"])
    sh = pd.read_csv(latest / "shape.csv")
    assert list(sh["tau"]) == [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    for k in ("expected_returns", "distribution", "checks"):
        assert k in meta
    assert "method" in meta["expected_returns"] and "method" in meta["distribution"]
    assert "tail_model" not in meta
    assert meta["checks"]["covariance_history"]["cov_annual_max_abs_diff"] == 0.0


@pytest.mark.realdata
def test_published_numbers_hang_together(latest):
    er = pd.read_csv(latest / "expected_returns.csv", index_col=0)
    cov = pd.read_csv(latest / "cov_annual.csv", index_col=0)
    vols = pd.read_csv(latest / "vols.csv", index_col=0)
    da = pd.read_csv(latest / "distribution_assets.csv", index_col=0)
    dp = pd.read_csv(latest / "distribution_portfolios.csv")
    ports = pd.read_csv(latest / "portfolios.csv", index_col=0)
    arith = np.exp(er["mean_12m"] + 0.5 * np.diag(cov.loc[er.index, er.index])) - 1
    np.testing.assert_allclose(er["mean_12m_arith"], arith, atol=2e-5)
    np.testing.assert_allclose(da["mean"], er.loc[da.index, "mean_12m"], atol=1e-9)
    np.testing.assert_allclose(da["scale"], vols.loc[da.index, "vol_12m_return"], atol=1e-5)
    q = da[schema.QUANTILE_COLUMNS].to_numpy()
    assert (np.diff(q, axis=1) > 0).all()
    own = dp[dp["shape"] == "portfolios"].set_index("portfolio")
    np.testing.assert_allclose(own["scale"], ports.loc[own.index, "vol_12m_return"], atol=1e-5)
