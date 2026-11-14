import json

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


K = 21


def test_hand_computed_target():
    days = pd.bdate_range("2001-01-01", "2001-12-31")
    M = len(days) - (K - 1)                                             # overlapping 21-day sums
    assert M == 241
    A = len(days) / K                                                   # annualization n_rows / k (261 rows)
    r = 0.01 * np.where(np.arange(len(days)) % 2 == 0, 1.0, -1.0)      # log returns +-1%
    mu = 0.001                                                          # constant drift
    daily = pd.DataFrame({"a": np.expm1(r), "b": np.expm1(r), "c": np.expm1(-r), "d": np.nan,
                          "e": np.expm1(mu)}, index=days)
    # d: monthly only, log returns alternating +-5% in 2001
    m01 = pd.date_range("2001-01-31", "2001-12-31", freq="ME")
    dm = np.expm1(0.05 * np.where(np.arange(12) % 2 == 0, 1.0, -1.0) + 0.001)
    p = tiny_panel(daily, {"d": pd.Series(dm, index=m01)})
    T = compute_targets(p)
    assert T.meta["horizon_days"] == K and T.meta["version"] == 3
    o = pd.Timestamp("2000-12-31")
    assert list(T.origins) == [o]                    # the only fully observed window
    v = T.vol_at(o)
    # every 21-day sum of alternating +-1% is +-1%: var = (n_rows/21) * 1e-4. (The 1-day measure
    # gives n_rows * 1e-4: strong negative autocorrelation no longer inflates the target.)
    assert v.loc["a", "vol"] == pytest.approx(0.01 * np.sqrt(A))
    assert v.loc["c", "vol"] == pytest.approx(0.01 * np.sqrt(A))
    assert not v.loc["a", "proxy"]
    # uncentered: a constant return is all "variance"
    assert v.loc["e", "vol"] == pytest.approx(np.sqrt(A) * K * mu)
    lm = np.log1p(dm)
    assert v.loc["d", "vol"] == pytest.approx(np.sqrt(12 * np.mean(lm ** 2)))
    assert v.loc["d", "proxy"]
    C = T.corr_matrix(o, ["a", "b", "c", "d", "e"])
    assert C.loc["a", "b"] == pytest.approx(1.0)
    assert C.loc["a", "c"] == pytest.approx(-1.0)
    assert np.isnan(C.loc["a", "d"])                 # no correlations from monthly proxies
    # sums of a alternate +1%, -1%, ... starting with + : 121 positive, 120 negative
    S = T.cov_matrix(o)
    assert list(S.index) == ["a", "b", "c", "e"]
    assert S.loc["a", "c"] == pytest.approx(-A * 0.01 ** 2)
    assert S.loc["a", "e"] == pytest.approx(A * (K * mu) * 0.01 / M)
    assert T.corr[T.corr["origin"] == o]["n_obs"].eq(M).all()
    f = T.fwd_ret.set_index("asset")["ret"]
    assert f["a"] == pytest.approx(r.sum())
    assert f["d"] == pytest.approx(lm.sum())


def _naive(panel: Panel, origin, frac=0.9, per_month=10, k=K):
    """Slow per-origin reference implementation of the target definitions."""
    end = origin + pd.offsets.MonthEnd(12)
    x = panel.excess_daily
    w = x[(x.index > origin) & (x.index <= end)]
    n_days = int(w.notna().any(axis=1).sum())
    xm = panel.excess_monthly
    wm = xm[(xm.index > origin) & (xm.index <= end)]
    y = w.fillna(0.0).rolling(k).sum().iloc[k - 1:]                    # k-day sums inside the window
    out, daily = {}, []
    for a in x.columns:
        s = w[a].dropna()
        months = s.groupby(s.index + pd.offsets.MonthEnd(0)).size().reindex(wm.index, fill_value=0)
        if n_days and len(s) >= frac * n_days and months.min() >= per_month:
            out[a] = (np.sqrt((y[a] ** 2).mean() * len(w) / k), False)       # n_rows / k
            daily.append(a)
        elif wm[a].notna().sum() == 12:
            out[a] = (np.sqrt(12 * (wm[a] ** 2).mean()), True)
    corr = {}
    for i, a in enumerate(daily):
        for b in daily[i + 1:]:
            corr[(a, b)] = (y[a] * y[b]).sum() / np.sqrt((y[a] ** 2).sum() * (y[b] ** 2).sum())
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
        for key, c in ref_corr.items():
            assert got[key] == pytest.approx(c, rel=1e-9, abs=1e-12)
        # the realized matrix is PSD (a Gram matrix) and equals D C D
        S = T.cov_matrix(o)
        assert np.linalg.eigvalsh(S.to_numpy()).min() > -1e-15
    assert T.vol[T.vol["asset"] == "c"]["proxy"].all()      # 15% holes: monthly proxy
    assert not T.vol[T.vol["asset"] == "b"]["proxy"].any()  # 3% holes lose no window
    with pytest.raises(ValueError):
        compute_targets(p, horizon_days=0)


def test_nan_means_no_trade_and_contributes_zero():
    """A NaN day is a non-trading day whose return is carried by the next row:
    it contributes 0 to the k-day sums, does not invalidate them, and moving
    a return across a gap barely changes the target."""
    rng = np.random.default_rng(11)
    days = pd.bdate_range("2001-01-01", "2006-12-31")
    n = len(days)
    x = rng.standard_normal(n) * 0.01                                  # daily LOG returns
    gap = (rng.random(n) < 0.06) & (np.arange(n) < n - 1)              # ~6% non-trading days
    gap &= ~np.roll(gap, 1)                                            # isolated gaps
    folded = x.copy()
    folded[1:] += np.where(gap[:-1], x[:-1], 0.0)                      # next return spans the gap
    folded[gap] = np.nan
    zeros = np.where(gap, 0.0, folded)                                 # same series, gaps as 0 returns
    daily = pd.DataFrame({"full": np.expm1(x), "gappy": np.expm1(folded), "zeros": np.expm1(zeros),
                          "z": np.expm1(rng.standard_normal(n) * 0.01 + 0.5 * x)}, index=days)
    T = compute_targets(tiny_panel(daily))
    V = T.vol_wide()
    assert not T.vol_wide("proxy").any().any()                         # every window daily-based
    assert len(V) == len(T.origins) > 50
    # NaN contributes exactly 0
    assert np.allclose(V["gappy"], V["zeros"], rtol=1e-12)
    cz = T.corr.set_index(["origin", "asset_i", "asset_j"])["corr"]
    o = T.origins[10]
    assert cz[(o, "gappy", "z")] == pytest.approx(cz[(o, "zeros", "z")], rel=1e-12)
    # gap-spanning returns: the 21-day target hardly notices the gaps
    rel = (V["gappy"] / V["full"] - 1).abs()
    assert rel.max() < 0.02 and rel.mean() < 0.005
    assert (T.corr[(T.corr.asset_i == "full") & (T.corr.asset_j == "gappy")]["corr"] > 0.99).all()


def test_smoothed_and_overshooting_prices_get_the_same_target():
    """The point of the 21-day definition: two pricing conventions of the same
    risk (smoothed NAV: positively autocorrelated daily returns; overreacting
    ETF: negatively autocorrelated) with the same monthly variance get the same
    target on average, although their 1-day vols differ by ~1.6x."""
    rng = np.random.default_rng(5)
    days = pd.bdate_range("1990-01-01", "2019-12-31")
    n = len(days)
    e = rng.standard_normal(n) * 0.16 / np.sqrt(252)                  # true risk: 16% vol, iid
    theta, psi = 0.3, 0.15
    smooth = np.empty(n)                                               # r_t = (1-theta) e_t + theta r_{t-1}
    smooth[0] = e[0]
    for t in range(1, n):
        smooth[t] = (1 - theta) * e[t] + theta * smooth[t - 1]
    over = (1 + psi) * e - psi * np.r_[0.0, e[:-1]]                    # overshoot, then partial reversal
    daily = pd.DataFrame({"raw": np.expm1(e), "smooth": np.expm1(smooth), "over": np.expm1(over)}, index=days)
    T = compute_targets(tiny_panel(daily, month_start="1990-01-31"))
    var = T.vol_wide("var")
    assert not T.vol_wide("proxy").any().any() and len(var) > 300
    ratio = var.mean() / var["raw"].mean()
    # theory (21-day sums): smoothed 1 - 2 theta / ((1 - theta^2) 21) = 0.969 of the true variance,
    # overshooting 1 + (2 psi + 2 psi^2) / 21 = 1.016; 1-day: 0.54 and 1.345
    assert 0.95 < ratio["smooth"] < 0.99 and 1.0 < ratio["over"] < 1.035
    assert abs(ratio["over"] / ratio["smooth"] - 1) < 0.06
    # ... although their 1-day vols differ by ~1.6x (the old target)
    d1 = (np.log1p(daily) ** 2).mean()
    assert d1["over"] / d1["smooth"] > 2.2 and d1["smooth"] / d1["raw"] < 0.6
    # the targets agree origin by origin too, not just on average
    assert (np.log(var["smooth"] / var["over"]).abs() < 0.15).mean() > 0.95
    # and they measure what the monthly-only era measures: the 12 * mean(r_m^2) proxy
    P = compute_targets(tiny_panel(daily, month_start="1990-01-31"), min_daily_frac=2.0)
    assert P.vol["proxy"].all()
    pv = P.vol_wide("var").mean()
    assert np.allclose(var.mean() / pv, 1.0, atol=0.06)


def _even_calendar(days_per_year: int, years: range) -> pd.DatetimeIndex:
    """``days_per_year`` trading days spread evenly over each calendar year."""
    out = []
    for y in years:
        cal = pd.date_range(f"{y}-01-01", f"{y}-12-31")
        out.append(cal[np.linspace(0, len(cal) - 1, days_per_year).round().astype(int)])
    return out[0].append(out[1:])


def test_annualization_follows_the_calendar():
    """Same daily variance, 300 vs 252 trading days a year (e.g. Saturday trading
    before 1952): the annual target scales with the number of trading days."""
    years = range(1800, 2100)
    v = 0.01 ** 2
    res = {}
    for n in (252, 300):
        days = _even_calendar(n, years)
        x = np.random.default_rng(n).standard_normal(len(days)) * np.sqrt(v)
        r = 0.01 * np.where(np.arange(len(days)) % 2 == 0, 1.0, -1.0)    # deterministic +-1% pattern
        daily = pd.DataFrame({"iid": np.expm1(x), "alt": np.expm1(r)}, index=days)
        T = compute_targets(tiny_panel(daily, month_start="1800-01-31"))
        assert not T.vol["proxy"].any()
        rows = T.vol_wide("n_days")["iid"]                                   # = n_rows here (one asset trades daily)
        assert rows.between(n - 1, n + 1).all()
        var = T.vol_wide("var")
        # truth for a simulated process: (rows in the window) x daily variance
        assert np.allclose(var["alt"], rows * v / K)                         # every 21-day sum is +-1%
        assert (var["iid"] / (rows * v)).mean() == pytest.approx(1.0, abs=0.06)
        # the monthly proxy measures the same quantity on either calendar
        P = compute_targets(tiny_panel(daily, month_start="1800-01-31"), min_daily_frac=2.0)
        assert P.vol_wide("var")["iid"].mean() / (n * v) == pytest.approx(1.0, abs=0.06)
        res[n] = var.mean()
    assert res[300]["alt"] / res[252]["alt"] == pytest.approx(300 / 252, rel=2e-3)
    assert res[300]["iid"] / res[252]["iid"] == pytest.approx(300 / 252, rel=0.08)


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
    # a cache built under another definition is never silently reused
    meta_path = tmp_path / "targets" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, "version": 2}))
    with pytest.raises(ValueError, match="another definition"):
        load_targets(tmp_path)
    T1 = get_targets(p, tmp_path)                              # rebuilt, same fingerprint
    assert T1.meta["version"] == 3 and load_targets(tmp_path).meta["version"] == 3
    T10 = get_targets(p, tmp_path, horizon_days=10)            # other horizon: own directory
    assert (tmp_path / "targets_h10" / "meta.json").exists() and T10.meta["horizon_days"] == 10
    assert panel_fingerprint(p, 10) != panel_fingerprint(p)
    assert load_targets(tmp_path).meta["horizon_days"] == 21
    assert not np.allclose(T10.vol["vol"].to_numpy()[-50:], T.vol["vol"].to_numpy()[-50:])
    # cache invalidates when the panel changes
    p2 = make_synthetic_panel(seed=3, start="1990-01-01", end="2001-12-31", monthly_only_years=3)
    assert panel_fingerprint(p2) != panel_fingerprint(p)
    T3 = get_targets(p2, tmp_path)
    assert not np.allclose(T3.vol["vol"].to_numpy()[:50], T.vol["vol"].to_numpy()[:50])
