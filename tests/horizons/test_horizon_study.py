"""The horizon study: targets and forecasts against brute force, and no look-ahead."""
import numpy as np
import pandas as pd
import pytest

from assetforecast import horizons
from assetforecast.data.synthetic import make_synthetic_panel
from assetforecast.horizons import risk, signals, value


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(seed=3, start="1990-01-01", end="2012-12-31")


def _xr(seed=0, n=420, assets=("a", "b", "c")):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1960-01-31", periods=n, freq="ME")
    xr = pd.DataFrame(rng.normal(0.004, 0.04, (n, len(assets))), index=idx, columns=list(assets))
    xr.iloc[:100, 2] = np.nan
    feats = pd.DataFrame({s: rng.normal(size=n * len(assets)) for s in signals.SIGNALS},
                         index=pd.MultiIndex.from_product([idx, assets], names=["date", "asset"]))
    return xr, feats


def test_period_mask_partitions_by_window_close():
    o = pd.DatetimeIndex(["1989-12-31", "1990-01-31", "2013-12-31", "2014-01-31", "2015-01-31"])
    assert horizons.period_mask(o, 12, "early").tolist() == [True, False, False, False, False]
    assert horizons.period_mask(o, 12, "dev").tolist() == [False, True, True, False, False]
    assert horizons.period_mask(o, 12, "after").tolist() == [False, False, False, False, True]
    assert horizons.period_mask(o, 1, "dev").tolist() == [False, True, True, True, False]


def test_hist_mean_is_h_times_expanding_mean():
    xr, _ = _xr()
    hm = signals.hist_mean(xr, 36)
    t = xr.index[200]
    assert hm.loc[t, "a"] == pytest.approx(36 * xr["a"].loc[:t].mean())
    assert np.isnan(hm.loc[xr.index[130], "c"])        # 31 months of history < 60


@pytest.mark.parametrize("h", [1, 12, 60])
def test_signal_forecasts_do_not_see_the_future(h):
    xr, feats = _xr()
    cut = xr.index[300]
    base = signals.run(feats, xr, horizons=(h,), start="1960-01-31")
    xr2, feats2 = xr.copy(), feats.copy()
    xr2.loc[xr2.index > cut] += 0.5
    feats2.loc[feats2.index.get_level_values("date") > cut] += 3.0
    alt = signals.run(feats2, xr2, horizons=(h,), start="1960-01-31")
    key = ["model", "horizon", "date", "asset"]
    a = base[base["date"] <= cut].set_index(key)["mean"]
    b = alt[alt["date"] <= cut].set_index(key)["mean"]
    assert len(a) > 1000 and a.index.equals(b.index)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=1e-12)


def test_signal_slope_matches_ols_on_purged_pairs():
    xr, feats = _xr()
    h, a, s = 12, "a", "carry"
    fc = signals.run(feats, xr, horizons=(h,), start="1960-01-31")
    t = xr.index[250]
    got = fc[(fc["model"] == f"sig_{s}") & (fc["date"] == t) & (fc["asset"] == a)]["mean"].item()
    x = feats[s].xs(a, level="asset")
    y = signals.targets(xr, h)[a]
    use = x.index <= t - pd.offsets.MonthEnd(h)
    b = np.polyfit(x[use], y[use], 1)[0]
    want = h * xr[a].loc[:t].mean() + b * (x[t] - x[use].mean())
    assert got == pytest.approx(want, abs=1e-10)


def _brute(x, t, h, k, a, b=None):
    w = x.loc[(x.index > t) & (x.index <= t + pd.offsets.MonthEnd(h))].fillna(0.0)
    s = w.rolling(k).sum().dropna()
    if b is None:
        return (s[a] ** 2).mean() * 252 / k
    return (s[a] * s[b]).sum() / np.sqrt((s[a] ** 2).sum() * (s[b] ** 2).sum())


def test_realized_and_trailing_match_brute_force(panel):
    vol, corr = risk.run(panel, horizons=(3, 12), ks=(21, 1), start="1990-01-31")
    x = panel.excess_daily
    done = vol.dropna(subset=["target", "trail_12"])
    for _, r in done.sample(12, random_state=1).iterrows():
        assert r["target"] == pytest.approx(_brute(x, r["date"], r["horizon"], r["k"], r["asset"]), rel=1e-9)
        back = r["date"] - pd.offsets.MonthEnd(12)
        assert r["trail_12"] == pytest.approx(_brute(x, back, 12, r["k"], r["asset"]), rel=1e-9)
    for _, r in corr.dropna(subset=["target"]).sample(8, random_state=2).iterrows():
        a, b = r["pair"].split("|")
        assert r["target"] == pytest.approx(_brute(x, r["date"], r["horizon"], r["k"], a, b), abs=1e-9)
    assert corr["target"].abs().max() <= 1 + 1e-9


def test_risk_forecasts_do_not_see_the_future(panel):
    from dataclasses import replace
    cut = pd.Timestamp("2011-06-30")
    d2 = panel.daily.copy()
    d2.loc[d2.index > cut] *= 3.0
    p2 = replace(panel, daily=d2)
    cols = ["naive", "trail_3", "trail_12", "trail_60", "half", "har"]
    for h in (1, 12):
        a, _ = risk.run(panel, horizons=(h,), ks=(21,), start="1990-01-31")
        b, _ = risk.run(p2, horizons=(h,), ks=(21,), start="1990-01-31")
        a, b = a[a["date"] <= cut], b[b["date"] <= cut]
        assert len(a) > 200 and a["trail_60"].notna().any()
        np.testing.assert_allclose(a[cols].to_numpy(), b[cols].to_numpy(), rtol=1e-10, equal_nan=True)


def test_monthly_fallback_at_one_month(panel):
    vol, _ = risk.run(panel, horizons=(1,), ks=(21,), start="1990-01-31")
    r = vol.dropna(subset=["target"]).iloc[500]
    x = panel.excess_daily[r["asset"]]
    m = x.loc[(x.index > r["date"]) & (x.index <= r["date"] + pd.offsets.MonthEnd(1))].sum()
    assert r["target"] == pytest.approx(12 * m * m, rel=1e-9)


def test_positions_average_the_live_tranches():
    w = pd.DataFrame({"a": [1.0, 0.0, 0.5, np.nan, 1.5]})
    pos = value._positions(w, 3)["a"]
    assert pos.iloc[2] == pytest.approx(0.5) and np.isnan(pos.iloc[3])
    assert pos.iloc[4] == pytest.approx(1.0)     # mean of 0.5 and 1.5; the missing tranche is skipped


@pytest.mark.realdata
def test_copies_reproduce_frozen_models_at_twelve_months():
    from assetforecast.horizons import mlcopies, run
    ml = mlcopies.load((12,))
    if ml.empty:
        pytest.skip("ML copies not cached (python -m assetforecast.horizons.mlcopies)")
    chk = run.check_copies(ml)
    assert chk and all(v["max_abs_diff"] == 0 and v["unmatched"] == 0 for v in chk.values())
