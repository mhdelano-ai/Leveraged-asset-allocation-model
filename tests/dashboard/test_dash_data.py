"""Every value in dashboard/data/* is recomputed here from the result files, independently of
``assetforecast.dashboard.data`` (only unit conversion and row selection are repeated)."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from assetforecast import paths
from assetforecast.dashboard import data as dd
from assetforecast.returns import config as rc

D = paths.DASHBOARD_DATA
LATEST = paths.FORECASTS_OUT / "latest"
F, R, A = paths.RESULTS_FINAL, paths.RESULTS_RETURNS, paths.ARCHIVE_AP

pytestmark = pytest.mark.skipif(not (D / "verdicts.csv").exists(), reason="dashboard data not built")


def csv(name):
    return pd.read_csv(D / f"{name}.csv", keep_default_na=True)


def close(a, b, tol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    assert np.allclose(a, b, atol=tol, equal_nan=True), np.nanmax(np.abs(a - b))


def pval(x):
    return np.nan if pd.isna(x) else ("<0.001" if x < 0.0005 else round(float(x), 3))


def same_p(col, ref):
    assert [str(a) for a in col.fillna("nan")] == [str(b) for b in pd.Series(ref, dtype=object).map(
        lambda v: "nan" if (not isinstance(v, str) and pd.isna(v)) else (v if isinstance(v, str) else str(float(v))))]


def test_fresh_build_equals_committed():
    """The committed files are exactly what the builders write now."""
    frames = dd.build_all()
    for d in dd.DATASETS:
        committed = pd.read_csv(D / d.file)
        fresh = pd.read_csv(pd.io.common.StringIO(frames[d.id].to_csv(index=False)))
        pd.testing.assert_frame_equal(committed, fresh, check_dtype=False)


def test_forecast_assets_and_portfolios():
    v = pd.read_csv(LATEST / "vols.csv")
    fa = csv("forecast_assets")
    assert list(fa.asset) == list(v.asset)
    for col, src in [("vol12", "vol_12m_return"), ("vol21", "vol_21d_horizon"), ("last12", "realized_last_12m"),
                     ("longrun", "realized_long_run"), ("p10", "family_p10"), ("p90", "family_p90")]:
        close(fa[col], (100 * v[src]).round(1))
    acc = pd.read_csv(F / "accuracy.csv").query("sample == 'last30y'").set_index("asset")
    close(fa.miss_lo, acc.two_thirds_lo_pct.loc[fa.asset].round(0))
    close(fa.miss_hi, acc.two_thirds_hi_pct.loc[fa.asset].round(0))
    assert "q05" not in fa.columns and "q01" not in fa.columns   # the removed scale-mixture quantiles
    p = pd.read_csv(LATEST / "portfolios.csv")
    fp = csv("forecast_portfolios")
    close(fp.vol12, (100 * p.vol_12m_return).round(1))
    close(fp.vol21, (100 * p.vol_21d_horizon).round(1))


def test_corr():
    c = pd.read_csv(LATEST / "corr.csv", index_col=0)
    fc = csv("forecast_corr")
    assert len(fc) == c.size
    for _, r in fc.iterrows():
        assert r["corr"] == pytest.approx(round(c.loc[r.asset_i, r.asset_j], 2))
        assert r.pair == f"{r.asset_i}~{r.asset_j}"


def test_histories_candidates_ranks():
    d = pd.read_csv(F / "dashboard_vol.csv").set_index(["origin", "asset"])
    h = csv("history_vol").set_index(["origin", "asset"])
    close(h.forecast, (100 * d.primary.loc[h.index]).round(1))
    close(h.realized, (100 * d.realized.loc[h.index]).round(1))
    k = pd.read_csv(F / "dashboard_corr.csv").set_index(["origin", "pair"])
    hc = csv("history_corr").set_index(["origin", "pair"])
    close(hc.forecast, k.primary.loc[hc.index].round(2))
    hh = pd.read_csv(F / "h2h_vol.csv").query("subset == 'daily' and era == 'all'")
    cand = csv("candidates")
    for _, r in cand.iterrows():
        for per in ("dev", "holdout"):
            assert r[per] == pytest.approx(round(hh[(hh.role == r.role) & (hh.period == per)].qlike.iloc[0], 3))
    lb = pd.read_csv(F / "vol_leaderboard.csv").query("subset == 'daily' and era == 'all'")
    mr = csv("model_ranks")
    for _, r in mr.iterrows():
        assert r.rank_holdout == lb[(lb.model == r.model) & (lb.period == "holdout")]["rank"].iloc[0]


def test_expectations_generated_from_final():
    e = pd.read_csv(F / "expectations.csv")
    x = csv("expectations")
    assert list(x["id"]) == list(e["id"])
    num = re.compile(r"[-+]?\d+\.\d+|\d+")
    from_terms = {n for _, rep in dd.EXPECTATION_TERMS for n in num.findall(rep)}   # "5-year", "12 independent"
    for (_, a), (_, b) in zip(e.iterrows(), x.iterrows()):
        # every number shown appears, as written, in the scoring output
        src_nums = num.findall(a.evidence)
        for n in num.findall(b.evidence):
            assert n in src_nums or n in from_terms, (n, b.evidence)
        assert b.outcome == ("not_met" if a.status.startswith("not met") else
                             "partly" if "point estimate" in a.status else "met")


def test_outlook_equals_published_forecast():
    """The page's 'now' figures are what forecasts/latest publishes."""
    er = pd.read_csv(LATEST / "expected_returns.csv").set_index("asset")
    da = pd.read_csv(LATEST / "distribution_assets.csv").set_index("asset")
    o = csv("outlook_assets").set_index("asset")
    assert list(o.index) == list(er.index)
    for col, src in [("er", "mean_12m"), ("er_lo", "ci90_lo"), ("er_hi", "ci90_hi"), ("er_se", "se_12m"),
                     ("er_arith", "mean_12m_arith")]:
        close(o[col], (100 * er[src]).round(1))
    close(o.vol12, (100 * da.scale.loc[o.index]).round(1))
    close(o.sd, (100 * da.sd.loc[o.index]).round(1))
    for q in dd.QS:
        close(o[q], (100 * da[q].loc[o.index]).round(1))
    close((100 * da["mean"]).round(1).loc[o.index], o.er)   # distribution centred on the expected return
    dp = pd.read_csv(LATEST / "distribution_portfolios.csv")
    op = csv("outlook_portfolios")
    assert list(zip(op.portfolio, op["shape"])) == list(zip(dp.portfolio, dp["shape"]))
    for q in (*dd.QS, "mean", "scale", "sd"):
        close(op[{"mean": "er", "scale": "vol12"}.get(q, q)], (100 * dp[q]).round(1))
    assert set(op[op.published == "yes"]["shape"]) == {"portfolios"}
    fp = csv("forecast_portfolios").set_index("portfolio")
    close(op.set_index("portfolio").vol12, fp.vol12.loc[op.portfolio])
    sh = pd.read_csv(LATEST / "shape.csv")
    s = csv("shape")
    close(s.z_assets, sh.z_assets.round(3))
    close(s.z_portfolios, sh.z_portfolios.round(3))


def test_snapshot():
    f = json.loads((LATEST / "forecast.json").read_text())
    s = csv("snapshot").iloc[0]
    assert s.origin == f["origin"] and s.data_through == f["data_through"]
    assert s.replication_label == rc.REPLICATION_LABEL and s.frozen_at == rc.FROZEN_AT
    sp = f["distribution"]["shape_portfolios"]
    assert s.shape_p_episodes_q01 == sp["episodes_q01"] and s.shape_p_n == sp["n_residuals"]
    assert s.oos_from == pd.Timestamp(rc.OOS_ORIGIN_START).strftime("%B %Y")


def _score(df, model, h, asset, period, bench, metric, target=None):
    m = ((df.model == model) & (df.horizon.astype(int) == h) & (df.asset == asset) & (df.period == period)
         & (df.benchmark == bench) & (df.metric == metric))
    if target:
        m &= df.target == target
    v = df.loc[m, "value"]
    assert len(v) <= 1
    return float(v.iloc[0]) if len(v) else np.nan


def test_mean_r2():
    arc = pd.read_parquet(A / "scores/frozen.parquet")
    rep = pd.read_parquet(R / "scores/replication.parquet")
    m = csv("mean_r2")
    for _, r in m.iterrows():
        sc = arc if r.source == "original" else rep
        assert r.r2 == pytest.approx(round(100 * _score(sc, r.model, r.horizon, "pooled", r.period, "hist_mean", "r2_oos"), 2))
        assert r.r2_detectable == pytest.approx(round(100 * _score(sc, "hist_mean", r.horizon, "pooled", r.period, "none",
                                                                   "r2_detectable"), 1))
        assert str(r.cw_p) == str(pval(_score(sc, r.model, r.horizon, "pooled", r.period, "hist_mean", "cw_p")))


def test_mean_cum():
    m = csv("mean_cum")
    for source, path in (("original", A / "scores/frozen_cumdsse.parquet"), ("replication", R / "scores/replication_cumdsse.parquet")):
        c = pd.read_parquet(path).query("asset == 'pooled'")
        c = c[c.model.isin(rc.FROZEN_MEAN)]
        c = c.assign(date=pd.to_datetime(c.date).dt.strftime("%Y-%m")).set_index(["horizon", "model", "date"]).cum_dsse
        g = m[m.source == source].set_index(["horizon", "model", "date"]).cum
        close(g, c.loc[g.index].round(5))


def test_dist_calib_and_original():
    sc = pd.read_parquet(R / "distribution/scores.parquet")
    for _, r in csv("dist_calib").iterrows():
        for k in ("cover_90", "hit_q05", "hit_q01"):
            assert r[k] == pytest.approx(round(100 * _score(sc, r.model, 12, r.asset, r.period, "none", k, r.target), 1))
        assert str(r.calib_p) == str(pval(_score(sc, r.model, 12, r.asset, r.period, "none", "calib_p", r.target)))
    arc = pd.read_parquet(A / "scores/frozen.parquet")
    for _, r in csv("dist_original").iterrows():
        assert r.crps_skill_vol == pytest.approx(round(100 * _score(arc, r.model, r.horizon, "pooled", "holdout", "vol_ewma", "crps_skill"), 2))


def test_strategy():
    a = pd.read_csv(A / "strategy/summary_by_period.csv")
    b = pd.read_csv(R / "strategy/summary_by_period.csv")
    s = csv("strategy")
    assert len(s[(s.source == "original")]) == 3 * len(rc.FROZEN_MEAN) * 3
    for _, r in s.iterrows():
        src = a if r.source == "original" else b
        col = "cov_model" if r.family == "risk" else "mean_model"
        g = src[(src.family == r.family) & (src.universe == "all") & (src.horizon == r.horizon) & (src.period == r.period)
                & (src.rule == r.rule) & (src[col] == r.model)]
        assert len(g) == 1
        assert r.d_cer == pytest.approx(round(100 * g.d_cer.iloc[0], 2))
        assert r.ci_lo == pytest.approx(round(100 * g.d_cer_ci_lo.iloc[0], 2))


def test_verdict_figures():
    v = csv("verdicts").set_index("question")
    arc = pd.read_parquet(A / "scores/frozen.parquet")
    rep = pd.read_parquet(R / "scores/replication.parquet")
    for q, sc, h, per in [("mean_orig_h1", arc, 1, "holdout"), ("mean_orig_h12", arc, 12, "holdout"),
                          ("mean_rep", rep, 12, "holdout"), ("mean_rep_ap", rep, 12, "holdout_ap")]:
        r2 = {m: _score(sc, m, h, "pooled", per, "hist_mean", "r2_oos") for m in rc.FROZEN_MEAN}
        best = max(r2, key=r2.get)
        assert v.loc[q, "best_model"] == best
        assert v.loc[q, "r2"] == pytest.approx(round(100 * r2[best], 2))
        assert str(v.loc[q, "spa_p"]) == str(pval(_score(sc, "*mean", h, "pooled", per, "hist_mean", "spa_p_mse")))
    ds = pd.read_parquet(R / "distribution/scores.parquet")
    assert v.loc["calib_assets", "cover90"] == pytest.approx(
        round(100 * _score(ds, "dist_hm_af", 12, "pooled", "holdout", "none", "cover_90", "assets"), 1))
    st = pd.read_csv(R / "strategy/summary_by_period.csv")
    g = st[(st.family == "mean") & (st.universe == "all") & (st.period == "holdout") & st.mean_model.isin(rc.FROZEN_MEAN)
           & st.rule.isin(["mv", "mv_vt", "timing"])]
    assert int(v.loc["strat_rep", "n"]) == len(g) and int(v.loc["strat_rep", "n_neg"]) == int((g.d_cer < 0).sum())
    summ = json.loads((F / "summary.json").read_text())["headline"]["holdout"]["vol_vs_primary"]["reference"]
    assert v.loc["cov_vol", "skill"] == pytest.approx(round(summ["primary_skill_pct"], 2))
    for q in v.index:
        assert (v.loc[q, "label"] == rc.REPLICATION_LABEL) == (v.loc[q, "evidence"] == "replication")


def test_frontier_equals_published_forecast():
    """The frontier datasets are the published frontier in percent; group shares and counts recomputed."""
    from assetforecast.universe import BY_ID
    f, p = pd.read_csv(LATEST / "frontier.csv"), pd.read_csv(LATEST / "frontier_points.csv")
    d, q, st = csv("frontier"), csv("frontier_points").set_index("point"), csv("frontier_settings").iloc[0]
    assert list(d.id) == list(f.curve + "~" + f.i.astype(str))
    for c in ("vol", "growth", "growth_lo", "growth_hi"):
        close(d[c], (100 * f[c]).round(2))
        close(q[c], (100 * p.set_index("point")[c]).round(2).loc[q.index])
    close(d.invested, (100 * f.invested).round(1))
    wcols = [c for c in f.columns if c.startswith("w_")]
    for g in ("equity", "rates", "credit", "real"):
        close(d[g], (100 * f[[c for c in wcols if BY_ID[c[2:]].group == g]].sum(axis=1)).round(1))
    for c in wcols:
        close(d[c], (100 * f[c]).round(1))
    pp = p.set_index("point").loc[q.index]
    for c in ("gap", "gap_lo", "gap_hi"):
        close(q[c], (100 * pp[c]).round(2))
    assert set(q.kind) == {"asset", "policy", "special", "match"} and q.label.notna().all()
    assert (q.loc[q.kind == "asset", "group"] == [BY_ID[a].group for a in q.index[q.kind == "asset"]]).all()
    assert q.loc["p6040", "weights"] == "60% US large, 40% UST 7y"
    assert d.loc[d.id == "levered~0", "weights"].iloc[0] == "100% cash"
    assert "borrowed" in d.loc[d.invested.idxmax(), "weights"]
    m = json.loads((LATEST / "forecast.json").read_text())["frontier"]
    assert st.borrow_spread == 100 * m["borrow_spread"] and st.max_invested == 100 * m["max_leverage"]
    match = p[p.kind == "match"]
    assert st.n_match == len(match) == 3
    assert st.n_gap_incl_zero == ((match.gap_lo <= 0) & (match.gap_hi >= 0)).sum()
    lev = f[f.curve == "levered"]
    assert st.n_assets_held == (lev[wcols].max() >= 0.005).sum() and st.n_assets == len(wcols)
    top = lev.loc[lev.vol.idxmax()]
    assert st.top_vol == round(100 * top.vol, 1) and st.top_growth == round(100 * top.growth, 1)


def test_manifest_covers_every_file():
    man = json.loads(dd.MANIFEST.read_text())
    files = {m["file"] for m in man}
    assert files == {f"data/{p.name}" for p in D.glob("*.csv")}
    assert len({m["id"] for m in man}) == len(man)
    for m in man:
        assert m["title"] and m["description"] and m["file"] == f"data/{m['id']}.csv"
    for old in ("forecast_assets", "forecast_corr", "forecast_portfolios", "history_vol", "history_corr",
                "candidates", "model_ranks", "expectations"):
        assert old in {m["id"] for m in man}


def test_portfolio_se_narrowing_and_zero_intervals():
    """The figures quoted beside the portfolio and expected-return tables, recomputed."""
    f = json.loads((LATEST / "forecast.json").read_text())
    dp = pd.read_csv(LATEST / "distribution_portfolios.csv")
    res = pd.read_parquet(R / "distribution/residuals.parquet")
    res = res[(res.kind == "portfolios") & (pd.to_datetime(res.origin) == pd.Timestamp(f["origin"]))].set_index("name")
    op = csv("outlook_portfolios")
    close(op.er_se, (100 * res.se.loc[op.portfolio]).round(1))
    from scipy.stats import norm
    narrower = 100 * (1 - (dp.q75 - dp.q25) / (2 * norm.ppf(0.75) * res.pscale.loc[dp.portfolio].to_numpy()))
    close(op.iqr_narrower, narrower.round(0))
    snap = csv("snapshot").iloc[0]
    assert snap.shape_p_sd == pytest.approx(round(f["distribution"]["shape_portfolios"]["sd_z"], 3))
    assert snap.p6040_sd == pytest.approx(100 * dp.set_index(["portfolio", "shape"]).sd[("p6040", "portfolios")], abs=0.006)
    sh = pd.read_parquet(R / "distribution/shape_history.parquet")
    sh = sh[(pd.to_datetime(sh.origin) == pd.Timestamp(f["origin"])) & (sh.kind == "portfolios")]
    floored = [f"{round(100 * t):g}%" for t, z, zg, zr in zip(sh.tau, sh.z, sh.z_gauss, sh.z_raw)
               if np.isclose(z, zg) and not np.isclose(z, zr)]
    assert snap.shape_p_floored == " and ".join(floored)
    er = pd.read_csv(LATEST / "expected_returns.csv")
    assert snap.er_n_zero == int(((er.ci90_lo <= 0) & (er.ci90_hi >= 0)).sum())
    assert snap.er_n == len(er)


def test_correlation_verdict_from_final():
    hc = pd.read_csv(F / "h2h_corr.csv").query("subset == 'daily' and period == 'holdout' and role == 'primary'").iloc[0]
    v = csv("verdicts").set_index("question")
    assert v.loc["cov_corr", "skill"] == pytest.approx(round(hc.skill_pct, 2))
    assert v.loc["cov_corr", "status"] == ("pass" if hc.skill_pct > 0 and hc.dm_p < 0.05 else "weak" if hc.skill_pct > 0 else "none")
