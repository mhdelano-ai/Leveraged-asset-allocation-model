import numpy as np
import pandas as pd
import pytest

from assetforecast.data.synthetic import make_synthetic_panel
from assetforecast.eval.harness import (ForecastError, forecast_path, load_forecasts, load_forecasts_long,
                                   origins_for, run_model, run_models, validate_forecast)
from assetforecast.eval.targets import compute_targets, observed_through
from assetforecast.models import base
from assetforecast.models.base import CorrForecaster, CovForecaster, VolForecaster


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(seed=5, start="1990-01-01", end="2001-12-31", monthly_only_years=3)


class ConstVol(VolForecaster):
    name = "t_const"

    def forecast_vol(self, view, assets):
        return pd.Series(0.15, index=assets)


class SpyVol(VolForecaster):
    """Records every view it sees and asserts the view cannot contain the future."""
    name = "t_spy"

    def __init__(self):
        self.seen = []

    def forecast_vol(self, view, assets):
        t = view.asof_date
        assert view.daily.index.max() <= t
        assert view.monthly.index.max() <= t
        assert view.cash_daily.index.max() <= t
        assert (view.predictors["available_date"] <= t).all()
        assert all(view.monthly[a].notna().sum() >= 36 for a in assets)
        self.seen.append((t, list(assets)))
        x = view.excess_monthly[assets].iloc[-36:]
        return np.sqrt((x ** 2).mean() * 12)


class SampleCov(CovForecaster):
    name = "t_samplecov"

    def forecast_cov(self, view, assets):
        x = view.excess_monthly[assets].iloc[-48:].fillna(0.0)
        return x.T @ x / len(x) * 12 + np.eye(len(assets)) * 1e-6


class Bad(CorrForecaster):
    def __init__(self, how, at=None):
        self.how, self.at, self.name = how, at, f"t_bad_{how}"

    def forecast_corr(self, view, assets):
        n = len(assets)
        c = pd.DataFrame(np.eye(n), index=assets, columns=assets)
        if self.at is not None and view.asof_date < pd.Timestamp(self.at):
            return c
        if self.how == "nonpd":
            c.iloc[0, 1] = c.iloc[1, 0] = 0.9
            c.iloc[0, 2] = c.iloc[2, 0] = 0.9
            c.iloc[1, 2] = c.iloc[2, 1] = -0.9
        elif self.how == "order":
            c = c.iloc[::-1, ::-1]
        elif self.how == "missing":
            c = c.iloc[1:, 1:]
        elif self.how == "nan":
            c.iloc[0, 1] = np.nan
        elif self.how == "asym":
            c.iloc[0, 1] = 0.3
        elif self.how == "diag":
            c.iloc[0, 0] = 1.1
        elif self.how == "raise":
            raise ZeroDivisionError("boom")
        return c


def test_origins(panel):
    o = origins_for(panel)
    assert o == sorted(o) and o[-1] == observed_through(panel) == pd.Timestamp("2001-12-31")
    o12 = origins_for(panel, start="1995-01-01", step=12)
    assert all(t.month == 12 for t in o12) and o12[0] == pd.Timestamp("1995-12-31")
    o3 = origins_for(panel, start="1995-02-01", end="1996-12-31", step=3, include_latest=False)
    assert [t.month for t in o3] == [3, 6, 9, 12, 3, 6, 9, 12]


def test_walk_forward_no_lookahead_and_roundtrip(panel, tmp_path):
    m = SpyVol()
    r = run_model(m, "vol", panel, results_dir=tmp_path)
    ts = [t for t, _ in m.seen]
    assert ts == sorted(ts) and len(ts) == r.n_new > 0
    assert r.n_skipped_few_assets > 0                     # early origins have < 2 eligible assets
    assert len(m.seen[0][1]) >= 2 and len(m.seen[-1][1]) > len(m.seen[0][1])
    df = pd.read_parquet(forecast_path("vol", "t_spy", tmp_path))
    assert list(df.columns) == ["origin", "asset", "vol"]
    d = load_forecasts("vol", "t_spy", tmp_path)
    t_last, a_last = m.seen[-1]
    assert list(d[t_last].index) == a_last
    # the live origin is stored but has no target
    T = compute_targets(panel)
    assert t_last == pd.Timestamp("2001-12-31") and t_last not in set(T.origins)


@pytest.mark.parametrize("how,msg", [
    ("nonpd", "not positive definite"), ("order", "wrong order"), ("missing", "wrong assets"),
    ("nan", "non-finite"), ("asym", "not symmetric"), ("diag", "diagonal not 1"), ("raise", "ZeroDivisionError"),
])
def test_invalid_forecasts_rejected(panel, tmp_path, how, msg):
    with pytest.raises(ForecastError) as e:
        run_model(Bad(how), "corr", panel, results_dir=tmp_path, start="1999-01-01")
    s = str(e.value)
    assert msg in s and f"t_bad_{how}" in s and "1999-01-31" in s


def test_validate_vol_rules():
    a = ["x", "y"]
    with pytest.raises(ForecastError, match="> 0"):
        validate_forecast("vol", pd.Series([0.1, 0.0], index=a), a, "m", "2000-01-31")
    with pytest.raises(ForecastError, match="pd.Series"):
        validate_forecast("vol", np.array([0.1, 0.2]), a, "m", "2000-01-31")
    with pytest.raises(ForecastError, match="variance must be > 0"):
        validate_forecast("cov", pd.DataFrame([[-1.0, 0], [0, 1]], index=a, columns=a), a, "m", "2000-01-31")
    ok = validate_forecast("corr", pd.DataFrame([[1.0, 0.2], [0.2 + 1e-14, 1]], index=a, columns=a), a, "m",
                           "2000-01-31")
    assert ok.iloc[0, 1] == ok.iloc[1, 0]


def test_resume_force_and_checkpoint_on_failure(panel, tmp_path):
    r1 = run_model(ConstVol(), "vol", panel, results_dir=tmp_path, end="1998-12-31")
    r2 = run_model(ConstVol(), "vol", panel, results_dir=tmp_path)
    assert r2.n_skipped_stored == r1.n_new and r2.n_new > 0
    n_all = r1.n_new + r2.n_new
    r3 = run_model(ConstVol(), "vol", panel, results_dir=tmp_path)
    assert r3.n_new == 0 and r3.n_skipped_stored == n_all
    r4 = run_model(ConstVol(), "vol", panel, results_dir=tmp_path, start="2000-01-01", force=True)
    assert r4.n_new == 24 and r4.n_skipped_stored == 0
    df = load_forecasts_long("vol", "t_const", tmp_path)
    assert df["origin"].nunique() == n_all and not df.duplicated(["origin", "asset"]).any()

    # a model that fails mid-run keeps the origins completed before the failure
    bad = Bad("nonpd", at="1999-06-30")
    bad.name = "t_flaky"
    with pytest.raises(ForecastError):
        run_model(bad, "corr", panel, results_dir=tmp_path)
    kept = load_forecasts_long("corr", "t_flaky", tmp_path)["origin"]
    assert kept.max() == pd.Timestamp("1999-05-31")
    bad.at = "2100-01-01"                                  # "fixed" model resumes
    r = run_model(bad, "corr", panel, results_dir=tmp_path)
    assert r.n_skipped_stored == kept.nunique() and r.n_new == 31


def test_cov_derivations(panel, tmp_path):
    run_model(SampleCov(), "cov", panel, results_dir=tmp_path, start="1999-01-01")
    cov = load_forecasts("cov", "t_samplecov", tmp_path)
    vol = load_forecasts("vol", "cov:t_samplecov", tmp_path)
    corr = load_forecasts("corr", "cov:t_samplecov", tmp_path)
    o = max(cov)
    C = cov[o]
    assert np.allclose(vol[o].to_numpy(), np.sqrt(np.diag(C)))
    d = np.sqrt(np.diag(C))
    assert np.allclose(corr[o].to_numpy(), C.to_numpy() / np.outer(d, d))
    assert list(corr[o].index) == list(C.index)


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_run_models_parallel(panel, tmp_path):
    """Process-parallel path with test models registered only for this test (fork inherits them)."""
    reg = base._REGISTRY["vol"]
    base._load_all()
    reg["t_par_a"] = ConstVol
    reg["t_par_b"] = ConstVol
    try:
        res = run_models("vol", ["t_par_a", "t_par_b"], source="synthetic", jobs=2, mp_start="fork",
                         results_dir=tmp_path, step=24)
    finally:
        reg.pop("t_par_a"), reg.pop("t_par_b")
    assert [r.model for r in res] == ["t_par_a", "t_par_b"]
    assert all(r.error is None and r.n_new > 0 for r in res)
    a = load_forecasts_long("vol", "t_par_a", tmp_path)
    assert a["origin"].is_monotonic_increasing
