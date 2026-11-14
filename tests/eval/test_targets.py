import numpy as np
import pandas as pd
import pytest

from volmodel.data.panel import PREDICTOR_COLUMNS, Panel
from volmodel.data.synthetic import make_synthetic_panel
from volmodel.eval.targets import (compute_targets, get_targets, load_targets, observed_through,
                                   panel_fingerprint)


def tiny_panel(daily: pd.DataFrame, monthly_extra: dict | None = None, month_start="2000-01-31") -> Panel:
    """Panel from daily simple returns (cash = 0); monthly = compounded daily.

    ``monthly_extra`` maps asset -> Series of monthly SIMPLE returns used where
    the asset has no daily data.
    """
    months = pd.date_range(month_start, daily.index.max() + pd.offsets.MonthEnd(0), freq="ME")
    monthly = (1 + daily).groupby(daily.index + pd.offsets.MonthEnd(0)).prod(min_count=1) - 1
    monthly = monthly.reindex(months)
    for a, s in (monthly_extra or {}).items():
        monthly[a] = s.reindex(months)
    tiers = pd.DataFrame(np.where(monthly.notna(), "B", None), index=months, columns=monthly.columns)
    return Panel(daily=daily, monthly=monthly,
                 cash_daily=pd.Series(0.0, index=daily.index), cash_monthly=pd.Series(0.0, index=months),
                 tiers=tiers, predictors=pd.DataFrame(columns=PREDICTOR_COLUMNS))


def test_hand_computed_target():
    days = pd.bdate_range("2001-01-01", "2001-12-31")
    r = 0.01 * np.where(np.arange(len(days)) % 2 == 0, 1.0, -1.0)      # log returns +-1%
    daily = pd.DataFrame({"a": np.expm1(r), "b": np.expm1(r), "c": np.expm1(-r), "d": np.nan}, index=days)
    # d: monthly only, log returns alternating +-5% in 2001
    m01 = pd.date_range("2001-01-31", "2001-12-31", freq="ME")
    dm = np.expm1(0.05 * np.where(np.arange(12) % 2 == 0, 1.0, -1.0) + 0.001)
    p = tiny_panel(daily, {"d": pd.Series(dm, index=m01)})
    T = compute_targets(p)
    o = pd.Timestamp("2000-12-31")
    assert list(T.origins) == [o]                    # the only fully observed window
    v = T.vol_at(o)
    assert v.loc["a", "vol"] == pytest.approx(0.01 * np.sqrt(252))
    assert v.loc["c", "vol"] == pytest.approx(0.01 * np.sqrt(252))
    assert not v.loc["a", "proxy"]
    lm = np.log1p(dm)
    assert v.loc["d", "vol"] == pytest.approx(np.sqrt(12 * np.mean(lm ** 2)))
    assert v.loc["d", "proxy"]
    C = T.corr_matrix(o, ["a", "b", "c", "d"])
    assert C.loc["a", "b"] == pytest.approx(1.0)
    assert C.loc["a", "c"] == pytest.approx(-1.0)
    assert np.isnan(C.loc["a", "d"])                 # no joint daily data
    # covariance on the daily-based set = D C D
    S = T.cov_matrix(o)
    assert list(S.index) == ["a", "b", "c"]
    assert S.loc["a", "c"] == pytest.approx(-(0.01 ** 2) * 252)
    f = T.fwd_ret.set_index("asset")["ret"]
    assert f["a"] == pytest.approx(r.sum())
    assert f["d"] == pytest.approx(lm.sum())


def _naive(panel: Panel, origin, frac=0.9, per_month=10, pair_frac=0.5):
    """Slow per-origin reference implementation of the target definitions."""
    end = origin + pd.offsets.MonthEnd(12)
    x = panel.excess_daily
    w = x[(x.index > origin) & (x.index <= end)]
    n_days = int(w.notna().any(axis=1).sum())
    xm = panel.excess_monthly
    wm = xm[(xm.index > origin) & (xm.index <= end)]
    out, daily = {}, []
    for a in x.columns:
        s = w[a].dropna()
        months = s.groupby(s.index + pd.offsets.MonthEnd(0)).size().reindex(wm.index, fill_value=0)
        if n_days and len(s) >= frac * n_days and months.min() >= per_month:
            out[a] = (np.sqrt((s ** 2).sum() * 252 / len(s)), False)
            daily.append(a)
        elif wm[a].notna().sum() == 12:
            out[a] = (np.sqrt(12 * (wm[a] ** 2).mean()), True)
    y = w.rolling(3, min_periods=3).sum().iloc[2:]
    corr = {}
    for i, a in enumerate(daily):
        for b in daily[i + 1:]:
            j = y[[a, b]].dropna()
            if len(j) >= pair_frac * (n_days - 2):
                corr[(a, b)] = (j[a] * j[b]).sum() / np.sqrt((j[a] ** 2).sum() * (j[b] ** 2).sum())
    return out, corr


def test_matches_naive_reference_with_gaps():
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2001-01-01", "2004-12-31")
    n = len(days)
    daily = pd.DataFrame(rng.standard_normal((n, 5)) * 0.01, index=days, columns=list("abcde"))
    daily.loc[rng.random(n) < 0.03, "b"] = np.nan                   # scattered holes: still daily
    daily.loc[rng.random(n) < 0.15, "c"] = np.nan                   # too sparse: proxy
    daily.loc["2002-06-01":"2002-06-30", "d"] = np.nan              # a missing month: proxy for those windows
    daily.loc[:"2002-03-15", "e"] = np.nan                          # starts mid-year
    p = tiny_panel(daily)
    T = compute_targets(p)
    assert len(T.origins) > 20
    for o in T.origins[::5]:
        ref, ref_corr = _naive(p, o)
        v = T.vol_at(o)
        assert set(v.index) == set(ref), o
        for a, (vol, px) in ref.items():
            assert v.loc[a, "vol"] == pytest.approx(vol, rel=1e-10), (o, a)
            assert bool(v.loc[a, "proxy"]) == px, (o, a)
        got = {(r.asset_i, r.asset_j): r.corr for r in T.corr[T.corr["origin"] == o].itertuples()}
        assert set(got) == set(ref_corr), o
        for k, c in ref_corr.items():
            assert got[k] == pytest.approx(c, rel=1e-9, abs=1e-12)
    assert T.vol[T.vol["asset"] == "c"]["proxy"].all()      # 15% holes: monthly proxy
    assert not T.vol[T.vol["asset"] == "b"]["proxy"].any()


def test_window_must_be_fully_observed_and_partial_month():
    days = pd.bdate_range("2001-01-01", "2003-06-12")                 # June 2003 is partial
    daily = pd.DataFrame({"a": 0.001, "b": -0.001}, index=days)
    p = tiny_panel(daily)
    assert p.monthly.index.max() == pd.Timestamp("2003-06-30")
    assert observed_through(p) == pd.Timestamp("2003-05-31")
    T = compute_targets(p)
    assert T.origins.max() == pd.Timestamp("2002-05-31")


def test_synthetic_targets_sane_and_cached(tmp_path):
    p = make_synthetic_panel(seed=2, start="1990-01-01", end="2001-12-31", monthly_only_years=3)
    T = get_targets(p, tmp_path)
    assert (tmp_path / "targets" / "vol.parquet").exists()
    T2 = load_targets(tmp_path)
    pd.testing.assert_frame_equal(T.vol, T2.vol)
    pd.testing.assert_frame_equal(T.corr, T2.corr)
    assert T2.meta["fingerprint"] == panel_fingerprint(p)
    # daily targets start once daily data covers a whole window; proxies before
    first_daily = T.vol[~T.vol["proxy"]]["origin"].min()
    assert first_daily == pd.Timestamp("1992-12-31")
    assert T.vol[T.vol["origin"] < first_daily]["proxy"].all()
    # realized vols are in the right ballpark of the generator's long-run vols
    m = T.vol[~T.vol["proxy"]].groupby("asset")["vol"].mean()
    assert 0.03 < m["ust_int"] < 0.12 and 0.10 < m["em"] < 0.40
    # realized corr matrices on the daily set are PD
    for o in T.origins[-30::7]:
        assert np.linalg.eigvalsh(T.cov_matrix(o).to_numpy()).min() > 0
    with pytest.raises(ValueError):
        compute_targets(p.asof("2000-01-31"))
    # cache invalidates when the panel changes
    p2 = make_synthetic_panel(seed=3, start="1990-01-01", end="2001-12-31", monthly_only_years=3)
    assert panel_fingerprint(p2) != panel_fingerprint(p)
    T3 = get_targets(p2, tmp_path)
    assert not np.allclose(T3.vol["vol"].to_numpy()[:50], T.vol["vol"].to_numpy()[:50])
