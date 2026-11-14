"""Offline checks of the committed predictor panel and its metadata, plus
network tests that rebuild a few series from source."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from assetpred import config, schema
from assetpred.data import predictors as P


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return schema.load_predictors()


@pytest.fixture(scope="module")
def meta() -> pd.DataFrame:
    return pd.read_csv(schema.PREDICTOR_META, keep_default_na=False)


# ------------------------------------------------------------------ contract
def test_contract(panel):
    assert panel.index.name == "date"
    assert panel.columns.is_unique
    assert all(np.issubdtype(t, np.floating) for t in panel.dtypes)
    assert panel.index[-1] <= pd.Timestamp.today().normalize() - pd.offsets.MonthEnd(1)
    assert 60 <= panel.shape[1] <= 150
    assert not np.isinf(panel.to_numpy()).any()


def test_meta_matches_panel(panel, meta):
    assert list(meta.columns) == P.META_COLUMNS
    assert list(meta["name"]) == list(panel.columns)
    for _, r in meta.iterrows():
        s = panel[r["name"]].dropna()
        assert s.index[0] == pd.Timestamp(r["first_date"]), r["name"]
        assert s.index[-1] == pd.Timestamp(r["last_date"]), r["name"]


def test_meta_values(meta):
    assert set(meta["family"]) <= set(config.FEATURE_FAMILIES)
    assert "price" not in set(meta["family"])  # price features are built from asset returns elsewhere
    assert set(meta["realtime"]) <= set(P.REALTIME)
    assert set(meta["scope"]) <= {"global", "asset"}
    glob = meta["scope"] == "global"
    assert (meta.loc[glob, "asset"] == "").all()
    assert meta.loc[~glob, "asset"].isin(list(config.ASSETS)).all()
    assert (meta["lag_months"].astype(int) >= 0).all()
    assert (meta["description"].str.len() > 20).all()
    assert (meta["source"].str.len() > 0).all()


def test_hindsight_flag_matches_description(meta):
    flagged = meta["hindsight"].astype(str) == "True"
    assert flagged.any()
    assert (meta["description"].str.contains(r"\[HINDSIGHT\]") == flagged).all()


def test_meta_matches_specs(meta):
    specs = {s.name: s for s in P.SPECS}
    assert set(specs) == set(meta["name"])
    for _, r in meta.iterrows():
        sp = specs[r["name"]]
        assert (sp.family, sp.scope, sp.asset or "", sp.lag, sp.realtime) == (
            r["family"], r["scope"], r["asset"], int(r["lag_months"]), r["realtime"])


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
        assert (pd.to_datetime(g["first_date"]) <= pd.Timestamp(config.OOS_START)).mean() > 0.6, fam


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
    from assetpred.data import predictors_sources as S
    # Industrial production for Sep 2026 was released on 16 Oct 2026.
    r = S.alfred_asof("INDPRO", "2026-09", "2026-10")["level"]
    assert r.index.tolist() == [pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-31")]
    assert r.iloc[0] != r.iloc[1]
