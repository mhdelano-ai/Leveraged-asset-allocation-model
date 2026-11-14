"""The return side's walk-forward engine (``assetforecast.returns.evaluation``):
on synthetic data (signal recovery, noise, leakage, contract), the replication
runner, and the reproduction of assetpred's archived forecasts.

Ported from assetpred's ``tests/test_harness.py``. assetpred's reference model
``ridge`` (``models/linear.py``) was not ported; ``ridge_z`` stands in for it."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from assetforecast.returns import config, schema
from assetforecast.returns.evaluation import leakage, scoring, walkforward
from assetforecast.returns.evaluation.cv import month_ordinal, purged_splits
from assetforecast.returns.evaluation.synthetic import make_synthetic
from assetforecast.returns.evaluation.walkforward import WalkForwardSettings, walk_forward
from assetforecast.returns.models import base, registry
from assetforecast.returns.models.base import (ForecastContractError, Forecaster, LeakageError,
                                   gaussian_frame, sort_quantiles, validate_predictions)

SETTINGS = WalkForwardSettings(oos_start="1980-01-31", refit_every=12, min_history=60)
QC = schema.QUANTILE_COLUMNS


def _score(fc, xr, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return scoring.score_forecasts(fc, xr, subperiods=(), mt_reps=199, **kw)


def _get(scores, model, h, metric, bench="hist_mean", asset="pooled"):
    s = scores[(scores.model == model) & (scores.horizon == h) & (scores.metric == metric)
               & (scores.benchmark == bench) & (scores.asset == asset) & (scores.period == "full")]
    assert len(s) == 1, (model, h, metric, bench, asset)
    return float(s.value.iloc[0])


def _run(models, data, horizons, settings=SETTINGS):
    return pd.concat([walk_forward(m, data.features, data.xr, h, settings)
                      for m in models for h in horizons], ignore_index=True)


@pytest.fixture(scope="module")
def signal_data():
    return make_synthetic(n_assets=4, beta=0.35, start="1960-01-31", end="2008-12-31", seed=11)


@pytest.fixture(scope="module")
def noise_data():
    return make_synthetic(n_assets=4, beta=0.0, start="1960-01-31", end="2008-12-31", seed=12)


@pytest.fixture(scope="module")
def signal_scores(signal_data):
    fc = _run(["hist_mean", "vol_ewma", "ridge_z"], signal_data, (1, 3))
    return fc, _score(fc, signal_data.xr)[0]


@pytest.fixture(scope="module")
def noise_scores(noise_data):
    fc = _run(["hist_mean", "hist_quantile", "zero", "vol_ewma", "ridge_z"], noise_data, (1, 12))
    return fc, _score(fc, noise_data.xr)[0]


# ---------------------------------------------------------------- CV splitter
@pytest.mark.parametrize("mode", ["expanding", "kfold"])
@pytest.mark.parametrize("h", [1, 3, 12])
def test_purged_splits_never_overlap(mode, h):
    dates = pd.date_range("1970-01-31", periods=240, freq="ME")
    idx = pd.MultiIndex.from_product([dates, ["a", "b"]], names=["date", "asset"])
    m = month_ordinal(idx.get_level_values("date"))
    folds = purged_splits(idx, h, n_splits=4, mode=mode)
    assert len(folds) >= 3
    for tr, va in folds:
        assert len(np.intersect1d(tr, va)) == 0
        vlo, vhi = m[va].min(), m[va].max()
        # training targets (s, s+h] must not intersect validation targets (vlo, vhi+h]
        s = m[tr]
        assert not np.any((s < vhi + h) & (s + h > vlo))
        if mode == "expanding":
            assert s.max() + h <= vlo


# ---------------------------------------------------------------- the purge
class _Recorder(Forecaster):
    """Records what the engine hands it."""
    refit_every = 1
    log: list = []

    def fit(self, train):
        _Recorder.log.append((train.origin, train.horizon, train.X.index.get_level_values("date").max(),
                              train.xr.index.max(), train.y.isna().any(),
                              train.realized_targets().dropna(how="all").index.max()))
        self._sd = 0.05
        return self

    def predict(self, X):
        return gaussian_frame(np.zeros(len(X)), np.full(len(X), self._sd), X.index)


@pytest.mark.parametrize("h", [1, 3, 12])
def test_engine_only_hands_realized_targets(signal_data, h):
    _Recorder.log = []
    st = WalkForwardSettings(oos_start="1985-01-31", end="1987-12-31")
    walk_forward(_Recorder, signal_data.features, signal_data.xr, h, st)
    assert len(_Recorder.log) == 36
    for origin, hh, xmax, xrmax, ynan, realized_max in _Recorder.log:
        assert xmax + pd.offsets.MonthEnd(hh) <= origin
        assert xmax + pd.offsets.MonthEnd(hh) == origin  # and uses everything allowed
        assert xrmax <= origin
        assert not ynan
        assert realized_max + pd.offsets.MonthEnd(hh) <= origin


class _MemorizeLastTarget(Forecaster):
    """Predicts each asset's most recent target it can find. Without a purge the
    most recent h=12 target would share 11 of 12 months with the one being
    forecast. It also tries to rebuild targets from the return history itself."""
    refit_every = 1

    def fit(self, train):
        y = train.y.unstack("asset")
        own = train.realized_targets()            # its own attempt, from xr
        last = own.ffill().iloc[-1].fillna(y.ffill().iloc[-1].reindex(own.columns))
        self._last = last.to_dict()
        self._sd = train.realized_targets().std().to_dict()
        return self

    def predict(self, X):
        a = X.index.get_level_values("asset")
        return gaussian_frame([self._last[k] for k in a], [self._sd[k] for k in a], X.index)


def test_memorizing_model_is_neutralized_by_purge(noise_data):
    fc = pd.concat([walk_forward("hist_mean", noise_data.features, noise_data.xr, 12, SETTINGS),
                    walk_forward("vol_ewma", noise_data.features, noise_data.xr, 12, SETTINGS),
                    walk_forward(_MemorizeLastTarget, noise_data.features, noise_data.xr, 12,
                                 SETTINGS, name="memorize")])
    sc = _score(fc, noise_data.xr, multiple_testing=False)[0]
    assert _get(sc, "memorize", 12, "r2_oos") < 0


def test_disabling_the_purge_is_caught(noise_data, monkeypatch):
    # If the engine's cutoff were wrong, TrainingSet refuses to be built.
    monkeypatch.setattr(walkforward, "_training_cutoff", lambda o, h: o - 1)
    with pytest.raises(LeakageError):
        walk_forward(_MemorizeLastTarget, noise_data.features, noise_data.xr, 12, SETTINGS)


def test_training_set_rejects_future_returns(noise_data):
    t = pd.Timestamp("1990-01-31")
    with pytest.raises(LeakageError):
        base.TrainingSet(horizon=1, origin=t, xr=noise_data.xr.loc[:"1990-02-28"], assets=(),
                         seed=0, _features=noise_data.features.iloc[:0],
                         _targets=pd.Series(dtype=float))


def test_audit_catches_feature_equal_to_future_target(noise_data):
    for h in (1, 12):
        y = schema.make_targets(noise_data.xr, h).stack().rename_axis(["date", "asset"])
        feats = noise_data.features.copy()
        feats["leak"] = y.reindex(feats.index)
        with pytest.raises(leakage.SuspectFeatureError, match="leak"):
            leakage.audit_feature_leakage(feats, noise_data.xr, h)
    # a one-month-ahead return leaked into a feature is caught at every horizon
    feats = noise_data.features.copy()
    feats["leak1"] = noise_data.xr.shift(-1).stack().rename_axis(["date", "asset"]).reindex(feats.index)
    with pytest.raises(leakage.SuspectFeatureError, match="leak1"):
        leakage.audit_feature_leakage(feats, noise_data.xr, 12)


def test_audit_passes_clean_features(signal_data):
    for h in (1, 3, 12):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = leakage.audit_feature_leakage(signal_data.features, signal_data.xr, h)
        assert (out.level != "error").all()


def test_lag_features_shifts_within_asset(signal_data):
    lagged = leakage.lag_features(signal_data.features, 1)
    f = signal_data.features["signal"].unstack("asset")
    g = lagged["signal"].unstack("asset")
    pd.testing.assert_frame_equal(g.iloc[1:], f.shift(1).iloc[1:], check_freq=False)


# ---------------------------------------------------------------- signal / noise
def test_planted_signal_is_recovered(signal_scores):
    _, sc = signal_scores
    for h in (1, 3):
        assert _get(sc, "ridge_z", h, "r2_oos") > 0.02
        assert _get(sc, "ridge_z", h, "cw_p") < 0.01
        assert _get(sc, "ridge_z", h, "crps_skill") > 0
        assert _get(sc, "ridge_z", h, "crps_skill", bench="vol_ewma") > 0
    # per-asset R^2 positive for most assets
    assets = sc[(sc.model == "ridge_z") & (sc.metric == "r2_oos") & (sc.horizon == 1)
                & (sc.asset != "pooled") & (sc.period == "full")]
    assert (assets.value > 0).mean() >= 0.75


def test_pure_noise_gives_no_edge(noise_scores):
    _, sc = noise_scores
    for h in (1, 12):
        assert _get(sc, "ridge_z", h, "r2_oos") <= 0.005
        assert _get(sc, "zero", h, "r2_oos") <= 0.005
        # baselines sharing the historical mean get exactly zero
        assert _get(sc, "vol_ewma", h, "r2_oos") == pytest.approx(0.0, abs=1e-12)
        assert _get(sc, "hist_quantile", h, "r2_oos") == pytest.approx(0.0, abs=1e-12)


def test_baseline_quantiles_calibrated_under_noise(noise_scores):
    _, sc = noise_scores
    for m in ("hist_mean", "hist_quantile"):
        for h in (1, 12):
            # sampling error of coverage is driven by non-overlapping observations
            n_indep = _get(sc, m, h, "n_indep", "none") * 4  # 4 roughly independent assets
            for lvl in (50, 90):
                p = lvl / 100
                tol = 3 * np.sqrt(p * (1 - p) / n_indep)
                assert abs(_get(sc, m, h, f"cover_{lvl}", "none") - p) < tol
            assert _get(sc, m, h, "calib_p", "none") > 0.01
        assert _get(sc, m, 1, "kupiec_p_q05", "none") > 0.01


def test_vol_baseline_gets_distribution_not_return_credit():
    d = make_synthetic(n_assets=3, beta=0.0, stoch_vol=True, start="1960-01-31",
                       end="2008-12-31", seed=5)
    fc = _run(["hist_mean", "vol_ewma"], d, (1,))
    sc = _score(fc, d.xr)[0]
    assert _get(sc, "vol_ewma", 1, "r2_oos") == pytest.approx(0.0, abs=1e-12)
    assert _get(sc, "vol_ewma", 1, "crps_skill") > 0.005


# ---------------------------------------------------------------- contract
def test_quantiles_monotone_and_contract(signal_scores, noise_scores):
    for fc, _ in (signal_scores, noise_scores):
        assert list(fc.columns) == schema.FORECAST_COLUMNS
        q = fc[QC].to_numpy()
        assert (np.diff(q, axis=1) >= -1e-12).all()
        assert (fc["sd"] > 0).all()
        assert not fc.duplicated(["model", "horizon", "date", "asset"]).any()


def test_validate_predictions_rejects_crossing_and_sort_fixes():
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2000-01-31"), "us_large")], names=["date", "asset"])
    p = gaussian_frame([0.0], [0.05], idx)
    bad = p.copy()
    bad[["q05", "q10"]] = bad[["q10", "q05"]].to_numpy()
    with pytest.raises(ForecastContractError):
        validate_predictions(bad, idx)
    validate_predictions(sort_quantiles(bad), idx)
    nan = p.copy()
    nan["sd"] = np.nan
    with pytest.raises(ForecastContractError):
        validate_predictions(nan, idx)


def test_eligibility_and_entry(signal_data):
    st = WalkForwardSettings(oos_start="1970-01-31", end="1980-12-31")
    fc = walk_forward("hist_mean", signal_data.features, signal_data.xr, 1, st)
    late = signal_data.xr.columns[-1]
    first = signal_data.xr[late].first_valid_index()
    entry = fc[fc.asset == late].date.min()
    n_hist = signal_data.xr[late].loc[:entry].notna().sum()
    assert n_hist == st.min_history
    assert fc[fc.asset != late].date.min() == pd.Timestamp("1970-01-31")
    assert entry > first


def test_refit_schedule_and_fresh_objects(signal_data):
    made = []

    class Counting(_Recorder):
        refit_every = None

        def __init__(self):
            made.append(self)

    st = WalkForwardSettings(oos_start="1985-01-31", end="1990-12-31", refit_every=12)
    walk_forward(Counting, signal_data.features, signal_data.xr, 1, st)
    # one probe object for the schedule + one fresh object per annual refit
    assert len(made) == 1 + 6


def test_point_forecaster_residual_distribution(signal_data):
    fc = walk_forward("ridge_z", signal_data.features, signal_data.xr, 3,
                      WalkForwardSettings(oos_start="1990-01-31", end="1995-12-31"))
    sd_by_asset = fc.groupby("asset")["sd"].mean()
    true_sd = signal_data.sigma.iloc[0] * np.sqrt(3)
    ratio = sd_by_asset / true_sd.reindex(sd_by_asset.index)
    assert ((ratio > 0.6) & (ratio < 1.5)).all()


# ---------------------------------------------------------------- cache, runner
def test_cache_and_parallel(signal_data, tmp_path):
    st = WalkForwardSettings(oos_start="1995-01-31", end="1999-12-31")
    status = walkforward.run_many(["hist_mean", "vol_ewma", "zero"], [1, 3], signal_data.features,
                                  signal_data.xr, st, results_dir=tmp_path, n_jobs=2)
    assert (status.status == "ran").all() and len(status) == 6
    status = walkforward.run_many(["hist_mean", "vol_ewma", "zero"], [1, 3], signal_data.features,
                                  signal_data.xr, st, results_dir=tmp_path, n_jobs=2)
    assert (status.status == "cached").all()
    xr2 = signal_data.xr.copy()
    xr2.iloc[-1, 0] += 0.01
    assert walkforward.run_cached("hist_mean", signal_data.features, xr2, 1, st,
                                  results_dir=tmp_path) == "ran"
    fc = walkforward.load_forecasts(["hist_mean", "zero"], [1, 3], tmp_path)
    serial = walk_forward("zero", signal_data.features, signal_data.xr, 3, st)
    pd.testing.assert_frame_equal(
        fc[(fc.model == "zero") & (fc.horizon == 3)].reset_index(drop=True), serial)


# ---------------------------------------------------------------- the replication runner
def _fake_inputs(monkeypatch, xr):
    """Stand in for ``returns.inputs`` (the panel) with an in-memory return panel."""
    import types

    import assetforecast.returns as returns_pkg
    fake = types.ModuleType("assetforecast.returns.inputs")
    fake.load_return_panel = lambda max_date=None: xr
    fake.excess_returns = lambda panel: panel
    monkeypatch.setitem(__import__("sys").modules, "assetforecast.returns.inputs", fake)
    monkeypatch.setattr(returns_pkg, "inputs", fake, raising=False)


def test_runner_missing_data_message(tmp_path, monkeypatch):
    from assetforecast import paths
    from assetforecast.returns import run
    monkeypatch.setattr(paths, "FEATURES", tmp_path / "features.parquet")
    monkeypatch.setattr(paths, "RESULTS_RETURNS", tmp_path / "results")
    with pytest.raises(SystemExit, match="Feature panel is missing"):
        run.main(["--models", "hist_mean"])


def test_runner_refuses_inputs_the_models_were_not_frozen_for(noise_data, monkeypatch):
    from assetforecast.returns import run
    from assetforecast.returns.models import ml_mean
    # synthetic feature names: "all" does not resolve to the pinned 82 columns
    monkeypatch.setattr(ml_mean, "resolve_columns", lambda fs, cols: list(cols))
    with pytest.raises(SystemExit, match="FROZEN_FEATURE_COLUMNS"):
        run.check_frozen_inputs(noise_data.features, noise_data.xr)
    # an asset with no pooling group
    xr = noise_data.xr.rename(columns={noise_data.xr.columns[0]: "us_eq"})
    with pytest.raises(SystemExit, match="us_eq"):
        run.check_frozen_inputs(noise_data.features.iloc[:0], xr)


def test_runner_max_date_needs_a_fresh_paths(monkeypatch):
    from assetforecast import paths
    from assetforecast.returns import run
    monkeypatch.setenv("ASSETFORECAST_MAX_DATE", "")  # restored (unset) after the test
    monkeypatch.setattr(paths, "_DEV", False)
    run._set_max_date(None)
    with pytest.raises(SystemExit, match="fresh process"):
        run._set_max_date("2010-12-31")


@pytest.mark.slow
def test_runner_end_to_end(tmp_path, monkeypatch):
    import json

    from assetforecast import paths
    from assetforecast.returns import run
    d = make_synthetic(n_assets=3, beta=0.3, start="1975-01-31", end="2004-12-31", seed=3)
    d.features.to_parquet(tmp_path / "features.parquet")
    monkeypatch.setenv("ASSETFORECAST_MAX_DATE", "")
    monkeypatch.setattr(paths, "_DEV", False)
    monkeypatch.setattr(paths, "FEATURES", tmp_path / "features.parquet")
    monkeypatch.setattr(paths, "RESULTS_RETURNS", tmp_path / "results")
    monkeypatch.setattr(run, "check_frozen_inputs", lambda f, x: None)  # synthetic feature names
    _fake_inputs(monkeypatch, d.xr)
    assert run.main(["--models", "ridge_z", "--reps", "99", "--jobs", "2"]) == 0
    out = tmp_path / "results"
    sc = pd.read_parquet(out / "scores" / "replication.parquet")
    assert list(sc.columns) == scoring.SCORE_COLUMNS
    assert {"ridge_z", "hist_mean", "vol_ewma", "*", "*mean"} <= set(sc.model)
    assert set(sc.horizon) == {12}
    assert {"full", "dev", "1990s"} <= set(sc.period)
    assert _get(sc, "ridge_z", 12, "r2_oos") > 0
    meta = json.loads((out / "scores" / "replication.json").read_text())
    assert meta["label"] == config.REPLICATION_LABEL and meta["frozen_at"] == config.FROZEN_AT
    for m in ("ridge_z", "hist_mean", "vol_ewma"):
        assert (out / "forecasts" / m / "h12.parquet").exists()
        side = json.loads((out / "forecasts" / m / "h12.json").read_text())
        assert side["label"] == config.REPLICATION_LABEL
    assert (out / "scores" / "replication_cumdsse.parquet").exists()
    # a second run reuses the cached forecasts (the label does not break the cache)
    assert walkforward.run_cached("hist_mean", d.features, d.xr, 12,
                                  results_dir=out) == "cached"
    # the leak audit stops a run whose features contain the future
    y = schema.make_targets(d.xr, 12).stack().rename_axis(["date", "asset"])
    f = d.features.copy()
    f["leak"] = y.reindex(f.index)
    f.to_parquet(tmp_path / "features.parquet")
    with pytest.raises(SystemExit, match="future returns"):
        run.main(["--models", "hist_mean"])


def test_score_forecasts_with_explicit_targets(noise_scores, noise_data):
    fc, _ = noise_scores
    fc = fc[fc.horizon == 12]
    kw = dict(subperiods=(), multiple_testing=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a, ca = scoring.score_forecasts(fc, noise_data.xr, **kw)
        y = schema.make_targets_long(noise_data.xr, 12)
        b, cb = scoring.score_forecasts(fc, None, targets=y, **kw)
        # targets replace the sums of xr: shifting them changes the scores
        c, _ = scoring.score_forecasts(fc, None, targets=y + 0.01, **kw)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ca, cb)
    key = (c.metric == "mse") & (c.asset == "pooled") & (c.model == "hist_mean")
    assert not np.isclose(c[key].value.iloc[0], a[key].value.iloc[0])
    with pytest.raises(ValueError, match="one horizon"):
        scoring.score_forecasts(noise_scores[0], None, targets=y, **kw)


# ---------------------------------------------------------------- reproduction of assetpred
# assetpred's frozen run, archived unmodified: inputs (11 assets, OLD ids) and the
# h=12 forecasts every model produced. The ported engine and models must give the
# same numbers on them. The only arrangement is the old universe (the archived
# return panel's columns) and the old pooling-group table.
OLD_ASSET_GROUP = {
    "us_eq": "equity", "dev_eq": "equity", "em_eq": "equity", "reits": "equity",
    "ust_int": "rates", "ust_long": "rates", "tips": "rates",
    "ig_credit": "credit", "hy_credit": "credit",
    "gold": "real", "commodities": "real",
}
ARCHIVE_FINGERPRINT = "bf5526544f8567df"  # data fingerprint in the archived forecasts' sidecars


@pytest.fixture(scope="module")
def archived_inputs():
    from assetforecast import paths
    d = paths.ARCHIVE_AP / "data"
    if not (d / "features.parquet").exists():
        pytest.skip("no archived assetpred inputs")
    features = pd.read_parquet(d / "features.parquet")
    returns = pd.read_parquet(d / "asset_returns.parquet")
    cash = pd.read_parquet(d / "cash.parquet")["cash"]
    xr = schema.excess_log_returns(returns, cash)
    meta = pd.read_csv(d / "feature_meta.csv").set_index("name")
    return features, xr, meta


@pytest.mark.realdata
def test_archived_inputs_are_the_frozen_runs(archived_inputs):
    from assetforecast.returns.models import ml_mean
    features, xr, meta = archived_inputs
    assert walkforward.data_fingerprint(features.sort_index(), xr) == ARCHIVE_FINGERPRINT
    assert set(xr.columns) == set(OLD_ASSET_GROUP)
    # the feature sets the frozen models resolve on the archived panel are assetpred's:
    # all but the discontinued Baker-Wurgler pair; families and hindsight from its metadata
    cols = list(features.columns)
    frozen = list(config.FROZEN_FEATURE_COLUMNS)
    assert ml_mean.resolve_columns("all", cols) == frozen
    assert ml_mean.resolve_columns("price", cols) == [c for c in frozen if meta.family[c] == "price"]
    assert ml_mean.resolve_columns("all-nohind", cols) == [c for c in frozen if not meta.hindsight[c]]


@pytest.mark.slow
@pytest.mark.realdata
@pytest.mark.parametrize("name", config.REPLICATION_MODELS)
def test_reproduces_archived_assetpred_forecasts(archived_inputs, monkeypatch, name):
    from assetforecast import paths
    from assetforecast.returns.models import ml_mean
    features, xr, _ = archived_inputs
    monkeypatch.setattr(ml_mean, "ASSET_GROUP", OLD_ASSET_GROUP)
    new = walk_forward(name, features, xr, 12).set_index(["date", "asset"]).sort_index()
    old = pd.read_parquet(paths.ARCHIVE_AP / "forecasts" / name / "h12.parquet")
    old = old.set_index(["date", "asset"]).sort_index()
    assert new.index.equals(old.index)
    cols = ["mean", "sd", *schema.QUANTILE_COLUMNS]
    diff = (new[cols] - old[cols]).abs().to_numpy()
    assert diff.max() <= 1e-10, (name, diff.max())



def test_registry_discovers_and_validates():
    names = registry.available()
    # exactly the replication list: baselines, the six frozen mean models, two robustness rows
    assert set(names) == set(config.REPLICATION_MODELS)
    m = registry.create("hist_mean")
    assert m.name == "hist_mean"
    with pytest.raises(ValueError):
        registry.register("bad name", lambda: None)
    with pytest.raises(KeyError):
        registry.get_factory("nope")


# ---------------------------------------------------------------- multiple testing by period / family
@pytest.fixture(scope="module")
def mt_scores(noise_scores, noise_data):
    fc, _ = noise_scores
    kw = dict(subperiods=(("late", "1995-01-01", "2008-12-31"),), mt_reps=99)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        plain = scoring.score_forecasts(fc, noise_data.xr, **kw)[0]
        fam = scoring.score_forecasts(
            fc, noise_data.xr, mt_periods=("full", "late"),
            families={"mean": ("ridge_z", "zero", "not_in_run"), "dist": ("hist_quantile",)}, **kw)[0]
    return plain, fam


def test_multiple_testing_per_period(mt_scores):
    plain, fam = mt_scores
    for metric in ("spa_p_mse", "spa_p_crps", "spa_q_bh_mse", "mcs_p_mse", "in_mcs_crps",
                   "cw_q_bh", "cw_q_by_all", "dm_crps_q_bh"):
        assert set(plain[plain.metric == metric].period) == {"full"}, metric
        assert set(fam[fam.metric == metric].period) == {"full", "late"}, metric
    # the late-period q-values adjust the late-period p-values, not the full ones
    late = fam[(fam.period == "late") & (fam.benchmark == "hist_mean") & (fam.asset != "pooled")
               & (fam.model == "ridge_z")]
    p = late[late.metric == "cw_p"].set_index(["horizon", "asset"]).value.sort_index()
    q = late[late.metric == "cw_q_bh"].set_index(["horizon", "asset"]).value.sort_index()
    np.testing.assert_allclose(q.to_numpy(), scoring.inf.fdr(p.to_numpy(), "bh"))


def test_spa_families(mt_scores):
    plain, fam = mt_scores
    star = fam[fam.model.str.startswith("*")]
    got = set(zip(star.model, star.benchmark, star.metric))
    assert ("*mean", "hist_mean", "spa_p_mse") in got
    assert ("*dist", "hist_mean", "spa_p_crps") in got
    assert ("*dist", "vol_ewma", "spa_p_crps") in got
    assert ("*mean", "vol_ewma", "spa_p_mse") not in got
    assert not any(m == "*mean" and k == "spa_p_crps" for m, _, k in got)
    assert ("*dist", "hist_mean", "spa_q_bh_crps") in got
    # the all-model rows are unchanged by adding families
    key = ["model", "horizon", "asset", "period", "benchmark", "metric"]
    a = plain[plain.model == "*"].set_index(key).value
    b = fam[(fam.model == "*") & (fam.period == "full")].set_index(key).value
    pd.testing.assert_series_equal(a.sort_index(), b.reindex(a.index).sort_index())
    # FDR of SPA rows is within one label and period
    g = fam[(fam.model == "*mean") & (fam.period == "late") & (fam.asset != "pooled")]
    p = g[g.metric == "spa_p_mse"].set_index(["horizon", "asset"]).value.sort_index()
    q = g[g.metric == "spa_q_bh_mse"].set_index(["horizon", "asset"]).value.sort_index()
    np.testing.assert_allclose(q.to_numpy(), scoring.inf.fdr(p.to_numpy(), "bh"))
    with pytest.raises(ValueError, match="families"):
        scoring.score_forecasts(pd.DataFrame(columns=schema.FORECAST_COLUMNS), None,
                                families={"other": ("ridge_z",)})


def test_summary_by_period(mt_scores):
    _, fam = mt_scores
    full, late = scoring.summary(fam), scoring.summary(fam, "late")
    assert not any(m.startswith("*") for m, _ in full.index)
    assert set(full.index) == set(late.index)
    assert not np.allclose(full["r2_oos"].dropna(), late["r2_oos"].dropna())
