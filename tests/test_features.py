"""Checks on the feature panel: the data contract, point-in-time construction
(truncation test), alignment against realised returns, and the per-asset slots."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from assetpred import config, schema
from assetpred.features import build as B
from assetpred.features.common import load_inputs


@pytest.fixture(scope="module")
def inputs():
    return load_inputs()


@pytest.fixture(scope="module")
def built(inputs):
    return B.build_features(inputs)


@pytest.fixture(scope="module")
def features(built):
    return built[0]


@pytest.fixture(scope="module")
def meta(built):
    return built[1]


@pytest.fixture(scope="module")
def xr(inputs):
    return schema.excess_log_returns(inputs.returns, inputs.cash)


# ------------------------------------------------------------------ contract
def test_committed_files_match_a_fresh_build(features, meta):
    disk = schema.load_features()
    pd.testing.assert_frame_equal(disk, features)
    disk_meta = pd.read_csv(schema.FEATURE_META, keep_default_na=False)
    assert list(disk_meta.columns) == B.META_COLUMNS
    assert list(disk_meta["name"]) == list(meta["name"])
    assert list(disk_meta["first_date"]) == list(meta["first_date"])


def test_index_contract(features, inputs):
    assert list(features.index.names) == ["date", "asset"]
    d = pd.DatetimeIndex(features.index.get_level_values("date"))
    assert d.tz is None
    assert (d == d + pd.offsets.MonthEnd(0)).all()
    assert features.index.is_monotonic_increasing and not features.index.has_duplicates
    assert set(features.index.get_level_values("asset")) == set(config.ASSETS)
    # exactly one row per (month, asset) with a return, through the last month
    expected = inputs.returns.stack().dropna().index
    assert len(features) == len(expected)
    assert set(map(tuple, features.index)) == {(pd.Timestamp(a), b) for a, b in expected}
    assert d.max() == inputs.returns.index.max()


def test_columns_are_float_and_finite(features):
    assert all(pd.api.types.is_float_dtype(t) for t in features.dtypes)
    assert not np.isinf(features.to_numpy()).any()
    assert 60 <= features.shape[1] <= 150
    assert features.notna().any().all(), "a feature is entirely missing"


def test_meta_matches_columns(features, meta):
    assert list(meta.columns) == B.META_COLUMNS
    assert list(meta["name"]) == list(features.columns)
    assert set(meta["family"]) == set(config.FEATURE_FAMILIES)
    assert set(meta["scope"]) <= {"asset", "global"}
    assert meta["hindsight"].isin([True, False]).all()
    assert (meta["description"].str.len() > 20).all()
    for _, r in meta.iterrows():
        first = features[r["name"]].dropna().index.get_level_values("date").min()
        assert pd.Timestamp(r["first_date"]) == first, r["name"]


def test_scope_is_truthful(features, meta):
    nun = features.groupby(level="date").nunique(dropna=False)
    for _, r in meta.iterrows():
        varies = (nun[r["name"]] > 1).any()
        if r["scope"] == "global":
            assert not varies, f"{r['name']} is labelled global but varies across assets"
        else:
            assert varies, f"{r['name']} is labelled asset but never varies across assets"


def test_hindsight_flags(meta, inputs):
    flagged = set(meta.loc[meta["hindsight"], "name"])
    assert {"g_acm_tp_10y", "cfnai_ma3", "nfci", "nfci_chg_3m", "bw_sentiment",
            "excess_bond_premium"} <= flagged
    # nothing derived only from returns or non-hindsight series is flagged
    assert not any(n.startswith(("mom", "vol", "dd", "trend")) for n in flagged)
    assert "bw_sentiment_pit" not in flagged and "carry" not in flagged and "value" not in flagged


# ------------------------------------------------------------------ point in time
TRUNCATION_DATES = ["1975-06-30", "1989-12-31", "1999-01-31", "2004-12-31",
                    "2008-09-30", "2019-01-31", "2024-04-30", "2025-12-31", "2026-09-30"]


@pytest.mark.parametrize("T", TRUNCATION_DATES)
def test_truncation_invariance(inputs, features, meta, T):
    """Rebuilding from inputs cut at month T leaves every row dated <= T unchanged."""
    cut, cut_meta = B.build_features(inputs.truncate(T))
    full = features.loc[features.index.get_level_values("date") <= pd.Timestamp(T)]
    assert list(cut.columns) == list(full.columns)
    assert cut.index.get_level_values("date").max() == pd.Timestamp(T)
    pd.testing.assert_frame_equal(cut, full, check_exact=True)
    assert list(cut_meta["hindsight"]) == list(meta["hindsight"])


def test_truncation_catches_a_leak(inputs, monkeypatch):
    """Sanity check of the test itself: a planted feature that uses month t+1
    (or a full-sample z-score) makes the truncated build differ."""
    from assetpred.features.common import Feat

    orig = B.all_features

    def leaky(inp):
        x = schema.excess_log_returns(inp.returns, inp.cash)
        tb = inp.predictors["tbill_3m"]
        return orig(inp) + [Feat("leak_fwd", "price", x.shift(-1), "planted look-ahead"),
                            Feat("leak_fullz", "macro", (tb - tb.mean()) / tb.std(), "planted full-sample z")]

    monkeypatch.setattr(B, "all_features", leaky)
    T = pd.Timestamp("2010-06-30")
    full, _ = B.build_features(inputs)
    cut, _ = B.build_features(inputs.truncate(T))
    full = full.loc[full.index.get_level_values("date") <= T]
    for c in ("leak_fwd", "leak_fullz"):
        assert not full[c].equals(cut[c]), c
    pd.testing.assert_frame_equal(cut.drop(columns=["leak_fwd", "leak_fullz"]),
                                  full.drop(columns=["leak_fwd", "leak_fullz"]))


def test_mom_1m_is_the_return_of_month_t(features, xr):
    s = xr.stack().rename_axis(["date", "asset"]).reindex(features.index)
    np.testing.assert_allclose(features["mom_1m"].to_numpy(), s.to_numpy(), rtol=0, atol=1e-12)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < 60:
        return np.nan
    return a[ok].rank().corr(b[ok].rank())


def test_no_implausible_correlation_with_next_month(features, xr):
    """A feature that is really the next month's return (a shifted index) would
    correlate far beyond the ~0.1-0.2 a genuine monthly predictor reaches."""
    nxt = xr.shift(-1).stack().rename_axis(["date", "asset"]).reindex(features.index)
    assets = features.index.get_level_values("asset")
    bad = []
    for c in features:
        pooled = _spearman(features[c], nxt)
        per = [abs(_spearman(features[c][assets == a], nxt[assets == a])) for a in config.ASSETS]
        worst = np.nanmax(per) if np.isfinite(per).any() else 0.0
        if abs(np.nan_to_num(pooled)) > 0.25 or worst > 0.4:
            bad.append((c, pooled, worst))
    assert not bad, bad


def test_price_features_line_up_with_returns(features, xr):
    """Momentum at t must match returns ending at t, not t+1."""
    m3 = xr.rolling(3, min_periods=3).sum().stack().rename_axis(["date", "asset"]).reindex(features.index)
    np.testing.assert_allclose(features["mom_3m"].dropna(), m3.loc[features["mom_3m"].dropna().index], atol=1e-12)
    m121 = xr.rolling(11, min_periods=11).sum().shift(1).stack().rename_axis(["date", "asset"])
    got = features["mom_12_1"].dropna()
    np.testing.assert_allclose(got, m121.loc[got.index], atol=1e-12)


# ------------------------------------------------------------------ slots
def test_slots_empty_where_no_credible_measure(features):
    for slot, empty in {"carry": ["em_eq", "hy_credit", "reits"],
                        "value": ["em_eq", "hy_credit", "reits", "commodities"]}.items():
        for a in empty:
            assert features[slot].xs(a, level="asset").isna().all(), (slot, a)
        filled = set(config.ASSETS) - set(empty)
        for a in filled:
            assert features[slot].xs(a, level="asset").notna().any(), (slot, a)


def test_slots_respect_series_ends(features):
    after = features.index.get_level_values("date") > pd.Timestamp("2025-12-31")
    dev = features.index.get_level_values("asset") == "dev_eq"
    assert features.loc[after & dev, ["carry", "value"]].isna().all().all()
    cm = features.index.get_level_values("asset") == "commodities"
    assert features.loc[(features.index.get_level_values("date") > pd.Timestamp("2024-04-30")) & cm, "carry"].isna().all()


def test_global_features_repeat_across_assets(features):
    d = pd.Timestamp("2010-12-31")
    x = features.xs(d, level="date")
    assert (x["g_tbill"].nunique() == 1) and (x["cpi_yoy"].nunique() == 1)
