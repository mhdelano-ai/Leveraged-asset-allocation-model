"""The return-side predictor track (``rp_`` series): the ported specs and the
archived panel they produced, the long-format track and its round trip, the
point-in-time accessors in ``assetforecast.returns.inputs``, and the check that
adding the track leaves every covariance-side forecast unchanged.

Ported from assetpred's ``tests/test_predictors.py``; the assertions on the
panel's values are unchanged and run on the archived wide panel (the
``--from-committed`` source of the track). Network tests keep their mark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from assetforecast.data import assemble as am
from assetforecast.data.panel import Panel
from assetforecast.data.sources import return_predictors as P
from assetforecast.paths import PROCESSED
from assetforecast.returns import config as rconfig
from assetforecast.returns import inputs
from assetforecast.universe import ASSET_IDS

OOS_START = "1990-01-31"   # assetpred's first scored origin (used only by test_history_lengths)


@pytest.fixture(scope="module")
def committed():
    return P.from_committed()


@pytest.fixture(scope="module")
def panel(committed) -> pd.DataFrame:
    return committed[0]


@pytest.fixture(scope="module")
def meta(committed) -> pd.DataFrame:
    return committed[1]


@pytest.fixture(scope="module")
def track(committed, tmp_path_factory):
    """The track written by ``write_track`` to a temporary directory (never data/processed)."""
    d = tmp_path_factory.mktemp("track")
    P.write_track(*committed, out_dir=d)
    return d


# ------------------------------------------------------------------ contract
def test_contract(panel):
    assert panel.index.name == "date"
    assert panel.columns.is_unique
    assert all(np.issubdtype(t, np.floating) for t in panel.dtypes)
    assert panel.index[-1] <= pd.Timestamp.today().normalize() - pd.offsets.MonthEnd(1)
    assert 60 <= panel.shape[1] <= 150
    assert not np.isinf(panel.to_numpy()).any()
    P.check_month_end(panel.index, "archived panel")
    assert panel.notna().any(axis=1).all()          # no empty month inside the span


def test_meta_matches_panel(panel, meta):
    assert list(meta.columns) == P.META_COLUMNS
    assert list(meta["name"]) == list(panel.columns)
    for _, r in meta.iterrows():
        s = panel[r["name"]].dropna()
        assert s.index[0] == pd.Timestamp(r["first_date"]), r["name"]
        assert s.index[-1] == pd.Timestamp(r["last_date"]), r["name"]


def test_meta_values(meta):
    assert set(meta["family"]) <= set(P.FAMILIES)
    assert set(P.FAMILIES) <= set(rconfig.FEATURE_FAMILIES)
    assert "price" not in set(meta["family"])  # price features are built from asset returns elsewhere
    assert set(meta["realtime"]) <= set(P.REALTIME)
    assert set(meta["scope"]) <= {"global", "asset"}
    glob = meta["scope"] == "global"
    assert (meta.loc[glob, "asset"] == "").all()
    assert meta.loc[~glob, "asset"].isin(ASSET_IDS).all()
    assert (meta["lag_months"].astype(int) >= 0).all()
    assert (meta["description"].str.len() > 20).all()
    assert (meta["source"].str.len() > 0).all()


def test_hindsight_flag_matches_description(meta):
    flagged = meta["hindsight"].astype(bool)
    assert flagged.any()
    assert (meta["description"].str.contains(r"\[HINDSIGHT\]") == flagged).all()


def test_meta_matches_specs(meta):
    specs = {s.name: s for s in P.SPECS}
    assert set(specs) == set(meta["name"])
    for _, r in meta.iterrows():
        sp = specs[r["name"]]
        assert (sp.family, sp.scope, sp.asset or "", sp.lag, sp.realtime) == (
            r["family"], r["scope"], r["asset"], int(r["lag_months"]), r["realtime"])


def test_asset_ids_renamed_and_nothing_else(meta):
    """The only metadata change from assetpred is the asset id rename."""
    old = P.read_meta(P.ARCHIVE_META)
    changed = old["asset"] != meta["asset"]
    assert dict(old.loc[changed, "asset"].value_counts()) == {"us_eq": 8, "dev_eq": 2}
    assert (meta.loc[changed, "asset"].map({"us_large": "us_eq", "intl_dev": "dev_eq"})
            == old.loc[changed, "asset"]).all()
    pd.testing.assert_frame_equal(old.drop(columns="asset"), meta.drop(columns="asset"))
    per_asset = meta[meta.scope == "asset"].groupby("asset").size().to_dict()
    assert per_asset == {"us_large": 8, "ust_long": 5, "tips": 5, "ust_int": 3, "ig_credit": 3,
                         "intl_dev": 2, "gold": 1, "commodities": 1}


# ---------------------------------------------------------------- lag rules
# Revised series allowed without a whole-month shift: weekly releases lagged in
# days, and term-premium models whose only inputs are month-end yields.
ZERO_LAG_REVISED = {"jobless_claims_4wk", "nfci", "anfci", "stl_fsi", "acm_tp_02y", "acm_tp_05y",
                    "acm_tp_10y", "kw_tp_10y"}


def test_lag_rules(meta):
    for _, r in meta.iterrows():
        lag = int(r["lag_months"])
        if r["realtime"] == "vintage":
            assert lag == 0, r["name"]  # vintages are as-of already
        if r["realtime"] == "lagged_revised" and r["name"] not in ZERO_LAG_REVISED:
            assert lag >= 1, r["name"]
        if r["family"] == "macro" and r["realtime"] == "market":
            assert lag == 0


def test_release_lag_cpi(panel):
    # CPI-U NSA y/y: March 2020 = log(258.115/254.202) is first known in April.
    assert panel.loc["2020-04-30", "cpi_yoy"] == pytest.approx(np.log(258.115 / 254.202), abs=2e-4)
    # April 2020 (published 12 May) appears only at the end of May.
    assert panel.loc["2020-05-31", "cpi_yoy"] == pytest.approx(np.log(256.389 / 255.548), abs=2e-4)


def test_release_lag_labour_market(panel):
    # April 2020 employment report came out 8 May 2020.
    assert panel.loc["2020-04-30", "unemployment_rt"] == pytest.approx(4.4)
    assert panel.loc["2020-05-31", "unemployment_rt"] == pytest.approx(14.7)
    assert panel.loc["2020-04-30", "payrolls_yoy_rt"] > 0
    assert panel.loc["2020-05-31", "payrolls_yoy_rt"] < -0.10
    # Claims for the week ending 28 Mar 2020 (6.9m) were published 2 Apr.
    assert panel.loc["2020-03-31", "jobless_claims_4wk"] < 1.5e6
    assert panel.loc["2020-04-30", "jobless_claims_4wk"] > 3e6


def test_market_series_not_lagged(panel):
    assert panel.loc["2020-03-31", "vix"] == pytest.approx(53.54)
    assert panel.loc["2008-10-31", "vix"] == pytest.approx(59.89)
    assert panel.loc["2020-03-31", "tbill_3m"] < 0.3


def test_vintage_series_start_with_their_vintages(meta):
    first = pd.to_datetime(meta.set_index("name")["first_date"])
    # ALFRED vintages: payrolls from May 1955, unemployment Mar 1960, M2 Feb 1980.
    assert pd.Timestamp("1955-05-31") <= first["payrolls_yoy_rt"] <= pd.Timestamp("1956-01-31")
    assert pd.Timestamp("1960-03-31") <= first["unemployment_rt"] <= pd.Timestamp("1960-12-31")
    assert pd.Timestamp("1980-02-29") <= first["m2_yoy_rt"] <= pd.Timestamp("1980-12-31")


def test_history_lengths(meta):
    first = pd.to_datetime(meta.set_index("name")["first_date"])
    for name in ["tbill_3m", "baa_aaa_spread", "us_cape", "cpi_yoy", "ip_yoy_rt", "unemployment_rt",
                 "ust_zero_05y", "gz_spread", "umich_sentiment"]:
        assert first[name] <= pd.Timestamp("1975-01-31"), name
    for fam, g in meta.groupby("family"):
        assert (pd.to_datetime(g["first_date"]) <= pd.Timestamp(OOS_START)).mean() > 0.6, fam


# ------------------------------------------------------- long-format track
def test_track_round_trip(panel, meta, track):
    long = pd.read_parquet(track / "returns.parquet")
    assert list(long.columns) == P.LONG_COLUMNS
    assert len(long) == int(panel.notna().to_numpy().sum())
    assert (long["obs_date"] == long["available_date"]).all()
    assert long["series"].str.startswith(P.PREFIX).all()
    assert not long.duplicated(["series", "obs_date"]).any()
    w = P.to_wide(long, names=list(meta["name"]))
    assert w.index.equals(panel.index) and list(w.columns) == list(panel.columns)
    a, b = panel.to_numpy(), w.to_numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.array_equal(a, b, equal_nan=True)                  # bit-for-bit, NaN pattern included
    pd.testing.assert_frame_equal(P.read_meta(track / "returns_meta.csv"), meta)


def test_no_name_collisions_with_other_tracks():
    cat = pd.read_csv(PROCESSED / "predictors_catalog.csv")
    other = set(cat.loc[cat["group"] != P.TRACK, "series"])
    assert not any(s.startswith(P.PREFIX) for s in other)
    assert not {P.PREFIX + s.name for s in P.SPECS} & other


# ------------------------------------------------------- panel and accessors
def _with_track(base: Panel, long: pd.DataFrame) -> Panel:
    """``base`` with its rp_ series replaced by ``long``, rows ordered as the assembler orders them."""
    p = base.predictors
    p = p[~p["series"].astype(str).str.startswith(P.PREFIX)]
    if len(long):
        p = pd.concat([p, long[am.PREDICTOR_COLUMNS]], ignore_index=True)
    p = p.sort_values(["available_date", "series", "obs_date"], kind="stable").reset_index(drop=True)
    return Panel(daily=base.daily, monthly=base.monthly, cash_daily=base.cash_daily,
                 cash_monthly=base.cash_monthly, tiers=base.tiers, predictors=p)


@pytest.fixture(scope="module")
def real_panel():
    from assetforecast.data.panel import load_panel
    if not (PROCESSED / "panel_daily.parquet").exists():
        pytest.skip("panel not assembled")
    return load_panel()


@pytest.fixture(scope="module")
def with_without(real_panel, track):
    long = pd.read_parquet(track / "returns.parquet")
    return _with_track(real_panel, long), _with_track(real_panel, long.iloc[:0])


@pytest.mark.realdata
@pytest.mark.parametrize("t", ["1926-01-31", "1960-02-29", "2008-10-15", "2020-03-31", "2026-10-31", "2026-11-13"])
def test_asof_exposes_exactly_rows_up_to_t(with_without, track, panel, t):
    full, _ = with_without
    t = pd.Timestamp(t)
    v = full.asof(t)
    rp = v.predictors[v.predictors["series"].str.startswith(P.PREFIX)]
    allrp = full.predictors[full.predictors["series"].str.startswith(P.PREFIX)]
    expect = allrp[allrp["available_date"] <= t]
    assert len(rp) == len(expect) == int(panel.loc[:t].notna().to_numpy().sum())
    pd.testing.assert_frame_equal(rp.sort_values(["series", "obs_date"]).reset_index(drop=True),
                                  expect.sort_values(["series", "obs_date"]).reset_index(drop=True))
    w, m = inputs.return_predictors(v, meta_path=track / "returns_meta.csv")
    ref = panel.loc[:t]
    assert w.index.equals(ref.index) and w.index.max() <= t
    assert np.array_equal(w.to_numpy(), ref.to_numpy(), equal_nan=True)
    # metadata of a view describes the view only
    last = pd.to_datetime(m.loc[m["last_date"] != "", "last_date"])
    assert (last <= t).all()


@pytest.mark.realdata
def test_return_predictors_full_panel_equals_archive(with_without, track, panel, meta):
    w, m = inputs.return_predictors(with_without[0], meta_path=track / "returns_meta.csv")
    assert w.index.equals(panel.index) and list(w.columns) == list(panel.columns)
    assert np.array_equal(w.to_numpy(), panel.to_numpy(), equal_nan=True)
    pd.testing.assert_frame_equal(m, meta)
    with pytest.raises(ValueError, match="no rp_ series"):
        inputs.return_predictors(with_without[1], meta_path=track / "returns_meta.csv")


@pytest.mark.realdata
def test_store_track_matches_archive_once_migrated(panel, meta):
    """After the in-place migration, the store's rp_ series are the archived panel exactly."""
    from assetforecast.data.panel import load_panel
    p = load_panel()
    if not p.predictors["series"].str.startswith(P.PREFIX).any():
        pytest.skip("data/processed/predictors.parquet has no rp_ series yet (not migrated)")
    w, m = inputs.return_predictors(p)
    assert np.array_equal(w.to_numpy(), panel.to_numpy(), equal_nan=True) and w.index.equals(panel.index)
    pd.testing.assert_frame_equal(m, meta)


@pytest.mark.realdata
def test_load_return_panel_and_excess_returns():
    rp = inputs.load_return_panel()
    assert rp.monthly.index[0] == pd.Timestamp(rconfig.RETURN_START)
    assert rp.cash_monthly.index.equals(rp.monthly.index) and rp.tiers.index.equals(rp.monthly.index)
    assert not rp.tiers.isin(list(rconfig.RETURN_DROP_TIERS)).any().any()
    assert rp.daily.index[0] >= pd.Timestamp(rconfig.RETURN_START) - pd.offsets.MonthBegin(1)
    assert rp.predictors["obs_date"].min() < pd.Timestamp(rconfig.RETURN_START)   # predictors not cut
    xr = inputs.excess_returns(rp)
    assert list(xr.columns) == list(ASSET_IDS)
    assert xr.index[-1] == inputs.observed_through(rp) == rp.monthly.index[-1]
    assert xr.index[0] == pd.Timestamp(rconfig.RETURN_START)
    v = rp.asof("2008-10-15")
    assert inputs.observed_through(v) == pd.Timestamp("2008-09-30")
    assert inputs.excess_returns(v).index[-1] == pd.Timestamp("2008-09-30")
    cut = inputs.load_return_panel(max_date="2013-12-31")
    assert inputs.observed_through(cut) == pd.Timestamp("2013-12-31")
    assert cut.monthly.index[0] == pd.Timestamp(rconfig.RETURN_START)


# ------------------------------------------- covariance side is unaffected
# Models that read predictors, one per access path (``ml.model._fingerprint`` and
# ``ml.features._macro_features``; ``macro.features._series_tables``;
# ``structural.core.monthly_predictors`` via ``predictors_wide``;
# ``condcorr.data.predictors_at``), plus models that read none.
INVARIANCE_MODELS = [("vol", "ref_blend"), ("vol", "macro_full"), ("vol", "macro_implied"),
                     ("vol", "ml_glm_macro"), ("vol", "structural_cev"),
                     ("corr", "condcorr_sb_macro"), ("corr", "condcorr_msw_infl"), ("corr", "bench_corr_blend"),
                     ("cov", "structural_factor_global"), ("cov", "pair_cov_ref")]
INVARIANCE_ORIGINS = ["1975-12-31", "2008-10-31", "2026-10-31"]


@pytest.mark.realdata
@pytest.mark.slow
def test_rp_series_do_not_change_covariance_forecasts(with_without):
    import warnings

    from assetforecast.models.base import get_model
    from assetforecast.models.ml.model import _fingerprint
    full, bare = with_without
    for t in INVARIANCE_ORIGINS:
        # the ML cache key changes (it hashes every predictor value); it is only a cache key
        assert _fingerprint(full.asof(t)) != _fingerprint(bare.asof(t))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kind, name in INVARIANCE_MODELS:
            m_full, m_bare = get_model(kind, name), get_model(kind, name)
            f = {"vol": "forecast_vol", "corr": "forecast_corr", "cov": "forecast_cov"}[kind]
            for t in INVARIANCE_ORIGINS:
                vf, vb = full.asof(t), bare.asof(t)
                assets = vb.eligible(min_months=36)
                a, b = getattr(m_full, f)(vf, assets), getattr(m_bare, f)(vb, assets)
                assert a.index.equals(b.index), (name, t)
                assert np.array_equal(np.asarray(a, float), np.asarray(b, float)), (name, t)


# ------------------------------------------------------------------ network
@pytest.mark.network
def test_rebuild_matches_committed(panel):
    specs = [s for s in P.SPECS if s.name in {"cpi_yoy", "vix", "ust_zero_10y", "unemployment_rt"}]
    fresh, _ = P.build(specs, end=panel.index[-1])
    for c in fresh.columns:
        a, b = fresh[c].dropna(), panel[c].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 100
        assert np.allclose(a[common], b[common], atol=1e-6, equal_nan=True), c


@pytest.mark.network
def test_alfred_asof_semantics():
    from assetforecast.data.sources import return_predictors_sources as S
    # Industrial production for Sep 2026 was released on 16 Oct 2026.
    r = S.alfred_asof("INDPRO", "2026-09", "2026-10")["level"]
    assert r.index.tolist() == [pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-31")]
    assert r.iloc[0] != r.iloc[1]
