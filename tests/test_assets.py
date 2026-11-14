"""Checks on the committed asset panel (offline) and on the source code paths (network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from assetpred import config, schema
from assetpred.data import assets_bonds as B

PROC = config.PROCESSED


@pytest.fixture(scope="module")
def panel():
    return schema.load_asset_returns()


@pytest.fixture(scope="module")
def cash():
    return schema.load_cash()


@pytest.fixture(scope="module")
def meta():
    return pd.read_csv(schema.ASSET_META, parse_dates=["segment_start", "segment_end"])


@pytest.fixture(scope="module")
def validation():
    return pd.read_csv(PROC / "asset_validation.csv")


# ------------------------------------------------------------------ contract
def test_columns_are_the_universe(panel):
    assert list(panel.columns) == list(config.ASSETS)
    assert panel.index.name == "date"
    assert (panel.dtypes == float).all()


def test_nan_only_before_history_starts(panel):
    for a in panel:
        s = panel[a]
        first = s.first_valid_index()
        assert first is not None, a
        assert s.loc[first:].notna().all(), f"{a} has gaps after it starts"


def test_all_assets_reach_the_last_month(panel):
    assert panel.index.max() >= pd.Timestamp("2026-10-31")
    assert panel.iloc[-1].notna().all()


def test_cash_covers_panel(panel, cash):
    assert cash.name == "cash"
    assert cash.index.min() <= panel.index.min() and cash.index.max() >= panel.index.max()
    c = cash.loc[panel.index.min():panel.index.max()]
    assert len(c) == len(panel) and c.notna().all()
    assert c.min() > -0.002 and c.max() < 0.02


def test_excess_returns_and_targets(panel, cash):
    xr = schema.excess_log_returns(panel, cash)
    assert xr.shape == panel.shape
    y = schema.make_targets(xr, 12)
    assert y["us_eq"].notna().sum() == panel["us_eq"].notna().sum() - 12


# ------------------------------------------------------------------ sanity bounds
VOL_RANGE = {  # annualised vol of monthly total returns over each asset's full history
    "us_eq": (0.12, 0.25), "dev_eq": (0.12, 0.25), "em_eq": (0.15, 0.30),
    "ust_int": (0.02, 0.08), "ust_long": (0.06, 0.16), "tips": (0.03, 0.09),
    "ig_credit": (0.04, 0.13), "hy_credit": (0.05, 0.13), "reits": (0.12, 0.25),
    "gold": (0.12, 0.25), "commodities": (0.10, 0.25),
}


def test_monthly_returns_within_bounds(panel):
    assert panel.min().min() > -0.45 and panel.max().max() < 0.6
    for a, (lo, hi) in VOL_RANGE.items():
        vol = panel[a].std() * np.sqrt(12)
        assert lo < vol < hi, (a, vol)


def test_mean_excess_returns_plausible(panel, cash):
    xr = schema.excess_log_returns(panel, cash)
    m = xr.mean() * 12
    assert ((m > -0.02) & (m < 0.10)).all(), m


def test_no_smoothing_signature(panel):
    """Returns from averaged prices show first-order autocorrelation ~0.25+.

    hy_credit (fund NAV, evaluated pricing) is allowed a little more."""
    for a in panel:
        ac1 = panel[a].dropna().autocorr()
        assert ac1 < (0.25 if a == "hy_credit" else 0.2), (a, ac1)


def test_known_months(panel):
    """Spot checks against well-known monthly moves."""
    assert panel.loc["1987-10-31", "us_eq"] < -0.20
    assert panel.loc["2008-10-31", "us_eq"] < -0.15
    assert panel.loc["2008-10-31", "ust_long"] > -0.05  # Treasuries rallied in late 2008
    assert panel.loc["2008-12-31", "ust_long"] > 0.08
    assert panel.loc["2008-10-31", "hy_credit"] < -0.10
    assert panel.loc["2020-03-31", "reits"] < -0.15
    assert panel.loc["2022-09-30", "ust_long"] < -0.05
    assert panel.loc["1980-01-31":"1981-12-31", "ust_int"].std() > 0.02  # Volcker-era volatility


# ------------------------------------------------------------------ meta & splices
def test_meta_segments_tile_each_series(panel, cash, meta):
    assert set(meta["asset"]) == set(config.ASSETS) | {"cash"}
    for asset, g in meta.groupby("asset", sort=False):
        g = g.sort_values("segment_start")
        s = cash if asset == "cash" else panel[asset]
        assert g["segment_start"].iloc[0] == s.first_valid_index(), asset
        assert g["segment_end"].iloc[-1] == panel.index.max(), asset
        nxt = g["segment_end"].iloc[:-1] + pd.offsets.MonthEnd(1)
        assert (nxt.to_numpy() == g["segment_start"].iloc[1:].to_numpy()).all(), asset
        assert g["source"].str.len().gt(10).all() and g["description"].str.len().gt(10).all()


def test_splice_overlaps_agree(validation):
    sp = validation[validation["check"] == "splice_overlap"]
    assert len(sp) >= 6
    assert (sp["n"] >= 24).all()
    assert (sp["corr"] > 0.95).all(), sp[["asset", "a", "b", "corr"]]
    assert ((sp["vol_ratio"] - 1).abs() < 0.2).all(), sp[["asset", "a", "b", "vol_ratio"]]


def test_no_jump_at_splice_boundaries(panel, meta):
    """The first month of a new segment should not be an outlier for that asset."""
    for asset, g in meta[meta.asset != "cash"].groupby("asset"):
        s = panel[asset].dropna()
        z = (s - s.mean()) / s.std()
        for d in g["segment_start"].iloc[1:]:
            assert abs(z.loc[d]) < 4, (asset, d, z.loc[d])


def test_etf_validation(validation):
    etf = validation[validation["check"] == "etf"]
    assert set(etf["asset"]) == set(config.ASSETS)
    assert (etf["n"] >= 100).all()
    assert (etf["corr"] > 0.85).all(), etf[["asset", "b", "corr"]]
    # synthetic Treasuries track the ETFs closely
    t = etf.set_index(["asset", "b"])
    assert t.loc[("ust_int", "IEI"), "corr"] > 0.98
    assert t.loc[("ust_long", "TLT"), "corr"] > 0.98


# ------------------------------------------------------------------ daily
def test_daily_file(panel):
    d = pd.read_parquet(PROC / "asset_returns_daily.parquet")
    assert d.index.is_monotonic_increasing and not d.index.has_duplicates
    assert set(d.columns) <= set(config.ASSETS)
    assert d.index.max() <= panel.index.max()
    assert d.abs().max().max() < 0.3  # largest is 1987-10-19 (-17%)
    pre1953_sat = d.loc[:"1952-12-31"].index.dayofweek == 5
    assert pre1953_sat.any()  # NYSE Saturday sessions are kept


def test_daily_compounds_to_monthly(validation):
    dv = validation[validation["check"] == "daily_vs_monthly"].set_index("asset")
    same_source = ["us_eq", "dev_eq", "em_eq", "ust_int", "ust_long", "tips", "ig_credit", "hy_credit"]
    assert (dv.loc[same_source, "corr"] > 0.999).all(), dv["corr"]
    assert (dv.loc[same_source, "tracking_error"] < 0.005).all()


# ------------------------------------------------------------------ bond pricing (no data needed)
def _flat(rate_pct):
    return pd.Series({"BETA0": rate_pct, "BETA1": 0.0, "BETA2": 0.0, "BETA3": 0.0, "TAU1": 1.0, "TAU2": np.nan})


def test_par_bond_prices_at_par():
    p = _flat(5.0)
    c = B.par_coupon(5.0, p)
    assert abs(B.hold_return(5.0, p, p, 1e-9)) < 1e-9
    assert 0.05 < c < 0.052  # semiannual-coupon equivalent of 5% continuous


def test_flat_curve_earns_the_yield():
    p = _flat(5.0)
    r = B.hold_return(10.0, p, p, 1 / 12)
    assert abs(r - (np.exp(0.05 / 12) - 1)) < 1e-6


def test_rate_rise_loses_about_duration():
    r = B.hold_return(20.0, _flat(5.0), _flat(6.0), 1e-9)
    assert -0.13 < r < -0.11  # modified duration of a 20y 5% par bond is ~12.5


# ------------------------------------------------------------------ network
@pytest.mark.network
def test_rebuild_matches_committed(panel, meta):
    """Sources re-parsed from the cached/downloaded raw files reproduce the first segment."""
    from assetpred.data import assets

    segs, _ = assets.monthly_segments()
    for asset, seglist in segs.items():
        s0 = seglist[0].series
        first_end = meta[meta.asset == asset]["segment_end"].min()
        common = panel[asset].dropna().loc[:first_end].index.intersection(s0.index)[:120]
        if len(common):
            assert np.allclose(panel.loc[common, asset], s0.loc[common], atol=1e-10), asset
