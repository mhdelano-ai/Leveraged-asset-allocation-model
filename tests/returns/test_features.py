"""Checks on the feature panel built on the 12-asset return panel: the data
contract, point-in-time construction (truncation tests), alignment against
realised returns, the per-asset slots, the pinned feature set, and equality
with assetpred's archived panel where nothing forces a difference.

Ported from assetpred's ``tests/test_features.py``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from assetforecast import paths
from assetforecast.returns import config, schema
from assetforecast.returns.features import build as B
from assetforecast.returns.features import predictors as FP
from assetforecast.returns.features import select
from assetforecast.returns.features.common import Feat, Inputs

ARCHIVE = paths.ARCHIVE_AP / "data"
OLD_TO_NEW = {"us_eq": "us_large", "dev_eq": "intl_dev", "em_eq": "em"}

# First month with a non-missing slot value, per asset (None = never). Pinned so
# that a slot keyed by an id outside the universe (silently dropped) or a slot
# re-sourced by accident shows up here.
SLOT_FIRST = {
    "carry": {"us_large": "1954-01-31", "us_small": None, "intl_dev": "1975-12-31", "em": None,
              "ust_int": "1962-02-28", "ust_long": "1971-08-31", "tips": "1999-01-31",
              "ig_credit": "1954-01-31", "hy_credit": None, "reits": None, "gold": "1971-09-30",
              "commodities": "1985-12-31"},
    "value": {"us_large": "1930-12-31", "us_small": None, "intl_dev": "1981-05-31", "em": None,
              "ust_int": "1966-05-31", "ust_long": "1976-07-31", "tips": "2003-12-31",
              "ig_credit": "1958-03-31", "hy_credit": None, "reits": None, "gold": "1979-12-31",
              "commodities": None},
}


# ------------------------------------------------------------------ fixtures (real panel)
@pytest.fixture(scope="module")
def panel():
    from assetforecast.returns.inputs import load_return_panel
    try:
        return load_return_panel()
    except (FileNotFoundError, ValueError) as e:
        pytest.skip(f"no return panel: {e}")


@pytest.fixture(scope="module")
def inputs(panel):
    from assetforecast.returns.features.common import inputs_from_panel
    return inputs_from_panel(panel)


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


@pytest.fixture(scope="module")
def archived():
    if not (ARCHIVE / "features.parquet").exists():
        pytest.skip("no archived assetpred inputs")
    old = pd.read_parquet(ARCHIVE / "features.parquet").rename(index=OLD_TO_NEW, level="asset")
    return old, pd.read_csv(ARCHIVE / "feature_meta.csv", keep_default_na=False)


# ------------------------------------------------------------------ contract
@pytest.mark.realdata
def test_committed_files_match_a_fresh_build(features, meta):
    disk = pd.read_parquet(paths.FEATURES)
    pd.testing.assert_frame_equal(disk, features)
    disk_meta = pd.read_csv(paths.FEATURE_META, keep_default_na=False)
    assert list(disk_meta.columns) == B.META_COLUMNS
    assert list(disk_meta["name"]) == list(meta["name"])
    assert list(disk_meta["first_date"]) == list(meta["first_date"])


@pytest.mark.realdata
def test_index_contract(features, inputs, panel):
    from assetforecast.returns.inputs import observed_through
    assert list(features.index.names) == ["date", "asset"]
    d = pd.DatetimeIndex(features.index.get_level_values("date"))
    assert d.tz is None
    assert (d == d + pd.offsets.MonthEnd(0)).all()
    assert features.index.is_monotonic_increasing and not features.index.has_duplicates
    assert set(features.index.get_level_values("asset")) == set(config.ASSETS)
    # exactly one row per (month, asset) with a return, through the last complete month
    expected = inputs.returns.stack().dropna().index
    assert len(features) == len(expected)
    assert set(map(tuple, features.index)) == {(pd.Timestamp(a), b) for a, b in expected}
    assert d.max() == observed_through(panel) == inputs.returns.index.max()
    assert d.min() == pd.Timestamp(config.RETURN_START)


@pytest.mark.realdata
def test_columns_are_the_old_84_in_order(features, archived):
    old, _ = archived
    assert list(features.columns) == list(old.columns)
    assert features.shape[1] == 84
    assert all(pd.api.types.is_float_dtype(t) for t in features.dtypes)
    assert not np.isinf(features.to_numpy()).any()
    assert features.notna().any().all(), "a feature is entirely missing"


@pytest.mark.realdata
def test_meta_matches_columns_and_the_archive(features, meta, archived):
    assert list(meta.columns) == B.META_COLUMNS
    assert list(meta["name"]) == list(features.columns)
    assert set(meta["family"]) == set(config.FEATURE_FAMILIES)
    assert (meta["description"].str.len() > 20).all()
    for _, r in meta.iterrows():
        first = features[r["name"]].dropna().index.get_level_values("date").min()
        assert pd.Timestamp(r["first_date"]) == first, r["name"]
    old_meta = archived[1]
    for c in ("name", "family", "scope"):
        assert list(meta[c]) == list(old_meta[c]), c
    assert list(meta["hindsight"]) == [str(v) == "True" for v in old_meta["hindsight"]]


@pytest.mark.realdata
def test_scope_is_truthful(features, meta):
    nun = features.groupby(level="date").nunique(dropna=False)
    for _, r in meta.iterrows():
        varies = (nun[r["name"]] > 1).any()
        if r["scope"] == "global":
            assert not varies, f"{r['name']} is labelled global but varies across assets"
        else:
            assert varies, f"{r['name']} is labelled asset but never varies across assets"


@pytest.mark.realdata
def test_hindsight_flags(meta):
    flagged = set(meta.loc[meta["hindsight"], "name"])
    assert {"g_acm_tp_10y", "cfnai_ma3", "nfci", "nfci_chg_3m", "bw_sentiment",
            "excess_bond_premium"} <= flagged
    assert not any(n.startswith(("mom", "vol", "dd", "trend")) for n in flagged)
    assert "bw_sentiment_pit" not in flagged and "carry" not in flagged and "value" not in flagged


# ------------------------------------------------------------------ the return panel underneath
@pytest.mark.realdata
def test_no_row_in_a_month_average_month(features):
    """Every row sits in a month whose return is not tier D (month-average eras)."""
    from assetforecast.data.panel import load_panel
    tiers = load_panel().tiers.stack()
    tiers.index = tiers.index.set_names(["date", "asset"])
    t = tiers.reindex(features.index)
    assert t.notna().all()
    assert not (t == "D").any(), t[t == "D"].head()


@pytest.mark.realdata
def test_partial_month_is_not_used(panel, inputs):
    """The daily panel runs into the current month; the inputs stop at the last complete one."""
    t = inputs.returns.index.max()
    assert inputs.daily.index.max() <= t
    assert inputs.predictors.index.max() <= t
    assert panel.daily.index.max() >= t  # (the panel itself may hold the partial month)


# ------------------------------------------------------------------ equality with the archive
@pytest.mark.realdata
def test_predictor_only_features_equal_the_archive(features, meta, archived):
    """Global valuation/macro/sentiment features depend only on the predictors
    (except vrp, which reads us_large's daily returns): unchanged to 1e-12."""
    old, _ = archived
    names = meta.loc[(meta.scope == "global") & (meta.family != "price") & (meta.name != "vrp"), "name"]
    assert len(names) == 51
    on, oo = features.groupby(level="date").first(), old.groupby(level="date").first()
    d = on.index.intersection(oo.index)
    assert len(d) > 1000
    for c in names:
        a, b = on.loc[d, c], oo.loc[d, c]
        assert (a.isna() == b.isna()).all(), c
        assert np.nanmax(np.abs(a - b).to_numpy(), initial=0.0) <= 1e-12, c


@pytest.mark.realdata
def test_slots_equal_the_archive_where_the_formula_is_unchanged(features, archived):
    old, _ = archived
    idx = features.index.intersection(old.index)
    assets = idx.get_level_values("asset")
    for slot in ("carry", "value"):
        a, b = features.loc[idx, slot], old.loc[idx, slot]
        for asset in set(config.ASSETS) - {"ust_int", "us_small"}:
            m = assets == asset
            np.testing.assert_allclose(a[m].to_numpy(), b[m].to_numpy(), rtol=0, atol=1e-12,
                                       err_msg=f"{slot} {asset}")
        m = assets == "ust_int"  # forced: 7-year zero instead of 5-year
        assert not np.allclose(a[m].to_numpy(), b[m].to_numpy(), equal_nan=True)


# ------------------------------------------------------------------ point in time
TRUNCATION_DATES = ["1975-06-30", "1990-01-31", "1999-01-31", "2004-12-31", "2008-09-30",
                    "2014-12-31", "2019-01-31", "2024-04-30", "2025-12-31"]


@pytest.mark.realdata
@pytest.mark.parametrize("T", TRUNCATION_DATES)
def test_truncation_invariance(inputs, features, meta, T):
    """Rebuilding from inputs cut at month T leaves every row dated <= T unchanged."""
    cut, cut_meta = B.build_features(inputs.truncate(T))
    full = features.loc[features.index.get_level_values("date") <= pd.Timestamp(T)]
    assert list(cut.columns) == list(full.columns)
    assert cut.index.get_level_values("date").max() == pd.Timestamp(T)
    pd.testing.assert_frame_equal(cut, full, check_exact=True)
    assert list(cut_meta["hindsight"]) == list(meta["hindsight"])


@pytest.mark.realdata
@pytest.mark.parametrize("T", ["1990-01-31", "2008-09-15", "2014-12-31", "latest"])
def test_truncation_invariance_of_the_panel(features, panel, T):
    """The whole chain, from the panel as it stood at T (``load_return_panel(T)``:
    returns, daily data and predictors cut by availability) to the features,
    reproduces the full build on rows dated <= T. A mid-month T checks that
    the partial month's daily returns are not used."""
    from assetforecast.returns.features.common import inputs_from_panel
    from assetforecast.returns.inputs import load_return_panel, observed_through
    T = observed_through(panel) if T == "latest" else pd.Timestamp(T)
    cut, _ = B.build_features(inputs_from_panel(load_return_panel(T)))
    last = T + pd.offsets.MonthEnd(0) if T == T + pd.offsets.MonthEnd(0) else T - pd.offsets.MonthEnd(1)
    full = features.loc[features.index.get_level_values("date") <= last]
    assert cut.index.get_level_values("date").max() == last
    pd.testing.assert_frame_equal(cut, full, check_exact=True)


@pytest.mark.realdata
def test_truncation_catches_a_leak(inputs, monkeypatch):
    """Sanity check of the test itself: a planted feature that uses month t+1
    (or a full-sample z-score) makes the truncated build differ."""
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


@pytest.mark.realdata
def test_mom_1m_is_the_return_of_month_t(features, xr):
    s = xr.stack().rename_axis(["date", "asset"]).reindex(features.index)
    np.testing.assert_allclose(features["mom_1m"].to_numpy(), s.to_numpy(), rtol=0, atol=1e-12)


@pytest.mark.realdata
def test_price_features_line_up_with_returns(features, xr):
    """Momentum at t must match returns ending at t, not t+1."""
    m3 = xr.rolling(3, min_periods=3).sum().stack().rename_axis(["date", "asset"]).reindex(features.index)
    np.testing.assert_allclose(features["mom_3m"].dropna(), m3.loc[features["mom_3m"].dropna().index], atol=1e-12)
    m121 = xr.rolling(11, min_periods=11).sum().shift(1).stack().rename_axis(["date", "asset"])
    got = features["mom_12_1"].dropna()
    np.testing.assert_allclose(got, m121.loc[got.index], atol=1e-12)


@pytest.mark.realdata
def test_global_price_states_read_us_large(features, xr):
    g = features.groupby(level="date").first()
    m121 = xr["us_large"].rolling(11, min_periods=11).sum().shift(1)
    np.testing.assert_allclose(g["g_eq_mom_12_1"].dropna(), m121.loc[g["g_eq_mom_12_1"].dropna().index], atol=1e-12)
    m121 = xr["ust_int"].rolling(11, min_periods=11).sum().shift(1)
    np.testing.assert_allclose(g["g_bond_mom_12_1"].dropna(), m121.loc[g["g_bond_mom_12_1"].dropna().index], atol=1e-12)


@pytest.mark.slow
@pytest.mark.realdata
def test_leakage_audit_flags_nothing_at_12_months(features, xr):
    from assetforecast.returns.evaluation.leakage import audit_feature_leakage
    flagged = audit_feature_leakage(features, xr, 12, raise_on_error=False)
    assert flagged.empty, flagged.to_string()


# ------------------------------------------------------------------ slots
@pytest.mark.realdata
def test_slot_coverage_is_pinned(features):
    for slot, table in SLOT_FIRST.items():
        assert set(table) == set(config.ASSETS)
        for a, first in table.items():
            s = features[slot].xs(a, level="asset").dropna()
            got = None if s.empty else s.index.min()
            assert got == (None if first is None else pd.Timestamp(first)), (slot, a, got)


@pytest.mark.realdata
def test_slots_respect_series_ends(features):
    dates = features.index.get_level_values("date")
    assets = features.index.get_level_values("asset")
    assert features.loc[(dates > pd.Timestamp("2025-12-31")) & (assets == "intl_dev"), ["carry", "value"]].isna().all().all()
    assert features.loc[(dates > pd.Timestamp("2024-04-30")) & (assets == "commodities"), "carry"].isna().all()


@pytest.mark.realdata
def test_global_features_repeat_across_assets(features):
    x = features.xs(pd.Timestamp("2010-12-31"), level="date")
    assert (x["g_tbill"].nunique() == 1) and (x["cpi_yoy"].nunique() == 1)


def test_slot_definitions_cover_only_universe_ids():
    assert set(FP.CARRY_DEF) <= set(config.ASSETS)
    assert set(FP.VALUE_DEF) <= set(config.ASSETS)
    assert "us_small" not in FP.CARRY_DEF and "us_small" not in FP.VALUE_DEF
    assert "ust_zero_07y" in FP.CARRY_DEF["ust_int"][1] and "ust_zero_07y" in FP.VALUE_DEF["ust_int"][1]


def _toy_inputs() -> Inputs:
    """A small synthetic input set (every predictor the formulas read, 12 assets)."""
    rng = np.random.default_rng(0)
    months = pd.date_range("1980-01-31", periods=180, freq="ME")
    days = pd.bdate_range("1980-01-01", months[-1])
    assets = list(config.ASSETS)
    returns = pd.DataFrame(rng.normal(0.005, 0.04, (len(months), len(assets))), months, assets)
    daily = pd.DataFrame(rng.normal(0.0002, 0.01, (len(days), len(assets))), days, assets)
    names = pd.read_csv(ARCHIVE / "predictor_meta.csv", keep_default_na=False)["name"] \
        if (ARCHIVE / "predictor_meta.csv").exists() else None
    if names is None:
        pytest.skip("no archived predictor metadata")
    preds = pd.DataFrame(np.abs(rng.normal(2, 0.5, (len(months), len(names)))), months, list(names))
    meta = pd.DataFrame({"name": list(names), "hindsight": "False"})
    return Inputs(returns, pd.Series(0.003, months), daily, preds, meta)


def test_a_slot_on_an_unknown_id_fails_loudly(monkeypatch):
    inp = _toy_inputs()
    B.build_features(inp)  # clean
    monkeypatch.setitem(FP.CARRY_DEF, "us_eq", ("old id", ("us_div_yield",)))
    with pytest.raises(ValueError, match="outside the universe"):
        B.build_features(inp)


def test_a_slot_column_on_an_unknown_id_fails_loudly(monkeypatch):
    inp = _toy_inputs()
    orig = FP._slots

    def stray(p, assets):
        out = orig(p, assets)
        out[0].values["dev_eq"] = 1.0  # what a missed rename would do
        FP._check_slot_ids("carry", out[0].values, FP.CARRY_DEF, assets)
        return out

    monkeypatch.setattr(FP, "_slots", stray)
    with pytest.raises(ValueError, match="outside the universe"):
        B.build_features(inp)
    # and build_features itself refuses any asset frame with a foreign column
    monkeypatch.setattr(FP, "_slots", orig)
    real = B.all_features
    monkeypatch.setattr(B, "all_features", lambda i: real(i) + [
        Feat("stray", "price", pd.DataFrame({"em_eq": 1.0}, index=i.returns.index), "a column on an old id")])
    with pytest.raises(ValueError, match="outside the universe"):
        B.build_features(inp)


# ------------------------------------------------------------------ the pinned feature set
@pytest.fixture
def selector_on(tmp_path):
    """Point the selector at a (features, meta) pair; restore afterwards."""
    def point(features, meta):
        f, m = tmp_path / "features.parquet", tmp_path / "feature_meta.csv"
        features.to_parquet(f)
        meta.to_csv(m, index=False)
        select.use_source(m, f)
    yield point
    select.use_source()


@pytest.mark.realdata
def test_columns_are_pinned_on_the_full_panel(features, meta, selector_on):
    selector_on(features, meta)
    frozen = list(config.FROZEN_FEATURE_COLUMNS)
    assert select.columns() == frozen
    fam = meta.set_index("name")["family"]
    hind = meta.set_index("name")["hindsight"]
    assert select.columns(families=("price",)) == [c for c in frozen if fam[c] == "price"]
    assert select.columns(hindsight=False) == [c for c in frozen if not hind[c]]
    assert select.columns(discontinued=True) == list(features.columns)
    # the old rule, on the full panel, agrees
    old_rule = select.stale_by_old_rule(features)
    assert [c for c in features.columns if not old_rule[c]] == frozen


@pytest.mark.realdata
@pytest.mark.parametrize("T", ["2014-12-31", "2019-01-31"])
def test_columns_do_not_depend_on_where_the_panel_ends(features, meta, selector_on, T):
    """The old rule would change the set under a development cut-off (at 2014-12
    it keeps the Baker-Wurgler pair); the pinned set does not."""
    cut = features.loc[features.index.get_level_values("date") <= pd.Timestamp(T)]
    selector_on(cut, meta)
    assert select.columns() == list(config.FROZEN_FEATURE_COLUMNS)
    assert set(select.columns()) != {c for c, s in select.stale_by_old_rule(cut).items() if not s}


@pytest.mark.realdata
def test_columns_on_the_archived_metadata():
    if not (ARCHIVE / "feature_meta.csv").exists():
        pytest.skip("no archived assetpred inputs")
    select.use_source(ARCHIVE / "feature_meta.csv", ARCHIVE / "features.parquet")
    try:
        meta = pd.read_csv(ARCHIVE / "feature_meta.csv").set_index("name")
        frozen = list(config.FROZEN_FEATURE_COLUMNS)
        assert select.columns() == frozen
        assert select.columns(families=("price",)) == [c for c in frozen if meta.family[c] == "price"]
        assert select.columns(hindsight=False) == [c for c in frozen if not meta.hindsight[c]]
    finally:
        select.use_source()


def test_missing_metadata_fails_loudly(tmp_path):
    select.use_source(tmp_path / "nope.csv", tmp_path / "nope.parquet")
    try:
        with pytest.raises(select.FeatureMetaMissing):
            select.columns()
        assert not issubclass(select.FeatureMetaMissing, OSError)
    finally:
        select.use_source()
