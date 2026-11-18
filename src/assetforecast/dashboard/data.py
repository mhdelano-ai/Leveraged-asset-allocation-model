"""Build every file in ``dashboard/data/`` from the results, and the dataset manifest.

    python -m assetforecast.dashboard            # writes dashboard/data/*.csv and dashboard/datasets.json

Each dataset is one CSV attached to the published dashboard and bound to the
page by ``data-source`` / ``data-field`` / ``data-where`` marks. The host
compares a number the page shows with its cell, so every value is written in
the units and rounding the page prints (percent with one or two decimals,
p-values with three, ``<0.001`` below that) and the page prints it unchanged.

Sources (nothing here re-estimates a statistic; the arithmetic is unit
conversion, rounding and selecting rows):

* ``forecasts/latest/``  the published forecast (``python -m assetforecast.forecast``), with the
                         efficient frontier it implies (``assetforecast.frontier``)
* ``results/final/``     the covariance model's frozen evaluation (``scripts/score_final.py``)
* ``results/returns/``   the merge-time replication, distribution scores and strategy test
* ``results/archive/assetpred/``  the original return-forecasting project's holdout, unmodified

Conclusions are not written here: ``verdicts`` decides which of the page's
prewritten sentences applies (``verdicts.py``) and supplies its figures.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from .. import paths
from ..returns import config as rc
from ..universe import BY_ID as ASSET
from . import glossary as gl
from . import verdicts as vd

MANIFEST = paths.ROOT / "dashboard" / "datasets.json"
RULES = ("mv", "mv_vt", "timing")                       # the pre-declared rules of the mean test
RISK_RULES = ("mv", "mv_vt", "timing", "vt_rp")         # the risk-model test adds inverse-vol
QS = ("q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99")
ARCHIVE_HORIZONS = (1, 3, 12)
ARCHIVE_PERIOD = "holdout"                              # assetpred's own holdout label (origins from 2015-01)


@dataclass
class Sources:
    """Where the inputs live (tests may point these elsewhere)."""
    latest: Path = paths.FORECASTS_OUT / "latest"
    final: Path = paths.RESULTS_FINAL
    returns: Path = paths.RESULTS_RETURNS
    archive: Path = paths.ARCHIVE_AP
    processed: Path = paths.PROCESSED
    _cache: dict = field(default_factory=dict, repr=False)

    def read(self, rel: str, root: str = "latest") -> pd.DataFrame:
        key = (root, rel)
        if key not in self._cache:
            p = getattr(self, root) / rel
            self._cache[key] = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        return self._cache[key]

    def json(self, rel: str, root: str = "latest") -> dict:
        return json.loads((getattr(self, root) / rel).read_text())


# --------------------------------------------------------------------------- display units

def pc(x, dp: int = 1):
    """Fraction -> percent, rounded for display (works on scalars and Series)."""
    return (100 * x).round(dp) if isinstance(x, (pd.Series, pd.DataFrame)) else (
        round(100 * float(x), dp) if vd.ok(x) else np.nan)


def pc_exact(x, dp: int) -> float:
    """Fraction -> percent rounded half-up on the decimal text the published file holds (0.10295 -> 10.30)."""
    from decimal import ROUND_HALF_UP, Decimal
    return float((Decimal(str(x)) * 100).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))


def pv(x):
    """A p-value as the page prints it: three decimals, '<0.001' below that."""
    if isinstance(x, pd.Series):
        return x.map(pv)
    if not vd.ok(x):
        return np.nan
    return "<0.001" if x < 0.0005 else round(float(x), 3)


def ym(x) -> str:
    return pd.Timestamp(x).strftime("%Y-%m") if vd.ok(x) and str(x) not in ("", "NaT") else ""


def short_list(weights: dict) -> str:
    return ", ".join(f"{round(100 * w):g}% {gl.SHORT_ASSET.get(a, a)}" for a, w in weights.items())


# --------------------------------------------------------------------------- covariance side (existing datasets)

def forecast_assets(s: Sources) -> pd.DataFrame:
    v = s.read("vols.csv")
    acc = s.read("accuracy.csv", "final")
    acc = acc[acc["sample"] == "last30y"].set_index("asset")
    return pd.DataFrame({
        "asset": v.asset, "name": v.name, "group": v.group, "vol12": pc(v.vol_12m_return),
        "vol21": pc(v.vol_21d_horizon), "last12": pc(v.realized_last_12m), "longrun": pc(v.realized_long_run),
        "p10": pc(v.family_p10), "p90": pc(v.family_p90), "since": v.history_starts,
        "miss_lo": v.asset.map(acc.two_thirds_lo_pct).round(0).astype(int),
        "miss_hi": v.asset.map(acc.two_thirds_hi_pct).round(0).astype(int),
    })


def forecast_corr(s: Sources) -> pd.DataFrame:
    c = s.read("corr.csv").set_index("Unnamed: 0").rename_axis(None).stack().reset_index()
    c.columns = ["asset_i", "asset_j", "corr"]
    c["corr"] = c["corr"].round(2)
    c["pair"] = c.asset_i + "~" + c.asset_j
    return c


def forecast_portfolios(s: Sources) -> pd.DataFrame:
    p = s.read("portfolios.csv")
    return pd.DataFrame({"portfolio": p.portfolio, "label": p.portfolio.map(gl.PORTFOLIOS),
                         "vol12": pc(p.vol_12m_return), "vol21": pc(p.vol_21d_horizon)})


def _quarterly(d: pd.DataFrame) -> pd.DataFrame:
    """Quarter-end origins, plus every one of the last 12 (whose windows are still open)."""
    m = d.origin.str[5:7].astype(int)
    live_from = sorted(d.origin.unique())[-12]
    return d[(m % 3 == 0) | (d.origin >= live_from)]


def history_vol(s: Sources) -> pd.DataFrame:
    d = s.read("dashboard_vol.csv", "final")
    h = _quarterly(d[d.origin >= "1926-06"])
    return pd.DataFrame({"origin": h.origin, "asset": h.asset, "period": h.period, "realized": pc(h.realized),
                         "proxy": h.proxy.astype(str).str.lower(), "forecast": pc(h.primary),
                         "reference": pc(h.reference), "p10": pc(h.fam_p10), "p90": pc(h.fam_p90)})


def history_corr(s: Sources) -> pd.DataFrame:
    k = s.read("dashboard_corr.csv", "final")
    k = _quarterly(k[k.pair.isin(["us_large~ust_int", "us_large~ust_long"])])
    return pd.DataFrame({"origin": k.origin, "pair": k.pair, "realized": k.realized.round(2),
                         "forecast": k.primary.round(2), "reference": k.reference.round(2)})


CANDIDATE_ROLES = {"primary": "Selected: family average", "runner_up": "Simple runner-up",
                   "single": "Best single model (development)", "reference": "Reference: long-run / 5-year blend",
                   "ref_last12m": "Benchmark: last 12 months"}


def candidates(s: Sources) -> pd.DataFrame:
    hh = s.read("h2h_vol.csv", "final")
    hh = hh[(hh.subset == "daily") & (hh.era == "all") & hh.period.isin(["dev", "holdout"])
            & hh.role.isin(CANDIDATE_ROLES)]
    cand = hh.pivot(index="role", columns="period", values="qlike").round(3)
    rf = hh.pivot(index="role", columns="period", values="rf").round(2)
    cand["label"], cand["rf_dev"], cand["rf_holdout"] = cand.index.map(CANDIDATE_ROLES), rf["dev"], rf["holdout"]
    out = cand.loc[list(CANDIDATE_ROLES)].reset_index()
    out.columns.name = None
    return out[["role", "dev", "holdout", "label", "rf_dev", "rf_holdout"]]


NAMED_MODELS = {"combo_vol_hier_trim": "Selected", "statespace_2c_daily": "Best single in development",
                "ref_blend": "Reference", "ref_last12m": "Last 12 months", "combo_vol_cal_m_hl12": "Simple runner-up",
                "bench_vol_m_hl120": "10-year average"}


def model_ranks(s: Sources) -> pd.DataFrame:
    lb = s.read("vol_leaderboard.csv", "final")
    lb = lb[(lb.subset == "daily") & (lb.era == "all")]
    a, b = lb[lb.period == "dev"].set_index("model"), lb[lb.period == "holdout"].set_index("model")
    rk = pd.DataFrame({"family": a.family, "rank_dev": a["rank"].astype(int), "rank_holdout": b["rank"].astype(int),
                       "qlike_dev": a.qlike.round(3), "qlike_holdout": b.qlike.round(3)}).reset_index()
    rk["label"] = rk.model.map(NAMED_MODELS).fillna("")
    return rk


# Plain-language rewording of results/final/expectations.csv, applied in order. Only names change; every
# number is carried over as written (tests check this).
EXPECTATION_TERMS = (
    (r"\bprimary vol QLIKE\b", "the selected model's volatility loss (QLIKE)"),
    (r"\bholdout QLIKE primary\b", "holdout loss: selected"),
    (r"\bprimary qlike_rel\b", "the selected model's covariance loss"),
    (r"\bholdout qlike_rel\b", "holdout covariance loss"),
    (r"\bpair_cov_ref\b", "the reference matrix"),
    (r"\bref_last12m\b", "last-12-months reference"),
    (r"\bref_blend\b", "long-run/5-year reference"),
    (r"\bmean RV/PV\b", "mean realized/predicted variance"),
    (r"\bVar\(z\)", "the variance of standardized 12-month returns"),
    (r"\bp6040\b", "60/40"), (r"\bp_balanced\b", "balanced"), (r"\bp_equity\b", "all-equity"),
    (r"\berc\b", "risk parity"), (r"\bgmv_long\b", "long-only minimum variance"), (r"\bgmv\b", "minimum variance"),
    (r"\bvr_market\b", "with the bridge"), (r"\bidentity\b", "without the bridge"),
    (r"chi2 \(non-overlap, ~12 yrs\)", "chance range for 12 independent years"),
    (r"\bdev:", "development:"), (r"\bdev\b", "development"), (r"\bdiff\b", "difference"),
)


def reword(text: str) -> str:
    for pat, rep in EXPECTATION_TERMS:
        text = re.sub(pat, rep, text)
    return text


def expectation_outcome(status: str) -> str:
    s = status.strip().lower()
    if s.startswith("met"):
        return "partly" if ("not significant" in s or "point estimate" in s) else "met"
    if s.startswith("not met"):
        return "not_met"
    raise ValueError(f"unknown expectation status {status!r}")


def expectations(s: Sources) -> pd.DataFrame:
    e = s.read("expectations.csv", "final")
    cap = lambda t: t[:1].upper() + t[1:]  # noqa: E731
    return pd.DataFrame({"id": e["id"], "expectation": e.expectation.map(reword).map(cap),
                         "status": e.status.map(reword).map(cap), "outcome": e.status.map(expectation_outcome),
                         "evidence": e.evidence.map(reword).map(cap)})


# --------------------------------------------------------------------------- what the package says now

def outlook_assets(s: Sources) -> pd.DataFrame:
    er = s.read("expected_returns.csv")
    d = s.read("distribution_assets.csv").set_index("asset").loc[er.asset].reset_index()
    out = pd.DataFrame({"asset": er.asset, "name": er.name, "er": pc(er.mean_12m), "er_lo": pc(er.ci90_lo),
                        "er_hi": pc(er.ci90_hi), "er_se": pc(er.se_12m), "er_arith": pc(er.mean_12m_arith),
                        "vol12": pc(d.scale), "sd": pc(d.sd)})
    for q in QS:
        out[q] = pc(d[q])
    out["includes_zero"] = np.where((er.ci90_lo <= 0) & (er.ci90_hi >= 0), "yes", "no")
    out["n_months"] = er.n_months
    out["history_from"] = er.first_month
    return out


def outlook_portfolios(s: Sources) -> pd.DataFrame:
    d = s.read("distribution_portfolios.csv")
    out = pd.DataFrame({"portfolio": d.portfolio, "shape": d["shape"], "model": d.model,
                        "label": d.portfolio.map(gl.PORTFOLIOS), "shape_label": d["shape"].map(gl.SHAPES),
                        "published": np.where(d["shape"] == "portfolios", "yes", "no"),
                        "er": pc(d["mean"]), "vol12": pc(d.scale), "sd": pc(d.sd)})
    for q in QS:
        out[q] = pc(d[q])
    # The portfolio expected return's standard error and the predictive scale are not in the published
    # file; they are the values distribution.portfolio_frame computed for the same origin, kept in
    # results/returns/distribution/residuals.parquet. Refuse if that row is not the published forecast.
    f = s.json("forecast.json")
    r = s.read("distribution/residuals.parquet", "returns")
    r = r[(r.kind == "portfolios") & (pd.to_datetime(r.origin) == pd.Timestamp(f["origin"]))].set_index("name")
    if not np.allclose(r["mean"].loc[d.portfolio].to_numpy(), d["mean"].to_numpy(), atol=1e-5):
        raise ValueError("residuals.parquet portfolio means differ from the published forecast")
    se, ps = r.se.loc[d.portfolio].to_numpy(), r.pscale.loc[d.portfolio].to_numpy()
    out["er_se"] = pc(pd.Series(se), 1).to_numpy()
    out["pscale"] = pc(pd.Series(ps), 1).to_numpy()
    # how much narrower the central half of the published range is than a normal with the predictive scale
    iqr_normal = 2 * norm.ppf(0.75) * ps
    out["iqr_narrower"] = (100 * (1 - (d.q75 - d.q25).to_numpy() / iqr_normal)).round(0).astype(int)
    out["weights"] = d.weights.map(lambda w: short_list(json.loads(w)))
    return out


def _floored(s: Sources) -> dict[str, pd.Series]:
    """Per kind, whether each published shape quantile sits on the normal floor (its empirical value was
    thinner and was replaced), from the shape history at the forecast origin."""
    f = s.json("forecast.json")
    h = s.read("distribution/shape_history.parquet", "returns")
    h = h[pd.to_datetime(h.origin) == pd.Timestamp(f["origin"])]
    out = {}
    for kind in ("assets", "portfolios"):
        g = h[h.kind == kind].set_index("tau")
        out[kind] = pd.Series(np.where(np.isclose(g.z, g.z_gauss) & ~np.isclose(g.z, g.z_raw), "yes", "no"), index=g.index)
    return out


def shape(s: Sources) -> pd.DataFrame:
    sh = s.read("shape.csv")
    fl = _floored(s)
    return pd.DataFrame({"tau": (100 * sh.tau).round(0).astype(int), "z_assets": sh.z_assets.round(3),
                         "z_portfolios": sh.z_portfolios.round(3), "z_normal": norm.ppf(sh.tau).round(3),
                         "floor_assets": fl["assets"].reindex(sh.tau).to_numpy(),
                         "floor_portfolios": fl["portfolios"].reindex(sh.tau).to_numpy()})


def snapshot(s: Sources) -> pd.DataFrame:
    f = s.json("forecast.json")
    w0, w1 = (pd.Timestamp(x.strip()) for x in f["window"].split(" to "))
    oos = pd.Timestamp(rc.OOS_ORIGIN_START)
    dist = f["distribution"]
    row = {"key": "now", "origin": f["origin"], "data_through": f["data_through"],
           "window": f"{w0:%B %Y} to {w1:%B %Y}", "window_start": f"{w0:%Y-%m-%d}", "window_end": f"{w1:%Y-%m-%d}",
           "frozen_at": rc.FROZEN_AT, "oos_from": f"{oos:%B %Y}",
           "first_oos_outcome": f"{oos + pd.offsets.MonthEnd(12):%B %Y}",
           "return_panel_from": pd.Timestamp(rc.RETURN_START).strftime("%B %Y"),
           "replication_label": rc.REPLICATION_LABEL, "er_level": round(100 * f["expected_returns"]["level"])}
    for k, blk in (("p", "shape_portfolios"), ("a", "shape_assets")):
        b = dist[blk]
        row.update({f"shape_{k}_episodes_q01": b["episodes_q01"], f"shape_{k}_episodes_q05": b["episodes_q05"],
                    f"shape_{k}_n": b["n_residuals"], f"shape_{k}_series": b["n_series"],
                    f"shape_{k}_from": ym(b["first_origin"]), f"shape_{k}_to": ym(b["last_origin"]),
                    f"shape_{k}_sd": round(b["sd_z"], 3)})
    fl = _floored(s)["portfolios"]
    row["shape_p_floored"] = " and ".join(f"{round(100 * t):g}%" for t in fl.index if fl[t] == "yes")
    # 60/40 published row: predictive sd against the 12-month volatility, and the narrowing of the central half
    op = outlook_portfolios(s).set_index(["portfolio", "shape"])
    p = op.loc[("p6040", "portfolios")]
    raw = s.read("distribution_portfolios.csv").set_index(["portfolio", "shape"]).loc[("p6040", "portfolios")]
    row.update({"p6040_sd": pc_exact(raw.sd, 2), "p6040_vol12": pc_exact(raw.scale, 2), "p6040_iqr_narrower": p.iqr_narrower,
                "p6040_q05_assetshape": op.loc[("p6040", "assets")].q05})
    # how many of the asset expected-return intervals include zero, and which lower bounds are essentially zero
    er = s.read("expected_returns.csv")
    inc = (er.ci90_lo <= 0) & (er.ci90_hi >= 0)
    near = er[(~inc) & (er.ci90_lo.abs() < 0.0005)]
    row.update({"er_n": len(er), "er_n_zero": int(inc.sum()),
                "er_near_zero": " and ".join(near.name) if len(near) else "none"})
    return pd.DataFrame([row])


# --------------------------------------------------------------------------- the return-side evidence

def _archive_scores(s: Sources) -> pd.DataFrame:
    return s.read("scores/frozen.parquet", "archive")


def _replication_scores(s: Sources) -> pd.DataFrame:
    return s.read("scores/replication.parquet", "returns")


def _dist_scores(s: Sources) -> pd.DataFrame:
    return s.read("distribution/scores.parquet", "returns")


def archive_dist_family(scores: pd.DataFrame) -> tuple[str, ...]:
    """assetpred's frozen distribution models: every scored ``dist_*`` model that is not a composite."""
    return tuple(sorted(m for m in scores.model.unique() if m.startswith("dist_") and "__" not in m))


def _span(cum: pd.DataFrame, horizon: int, first: str, last: str | None = None) -> tuple[str, str]:
    """First and last scored origin of a period (pooled cumulative series, any model)."""
    c = cum[(cum.asset == "pooled") & (cum.horizon.astype(int) == horizon)]
    d = pd.to_datetime(c.date)
    d = d[(d >= pd.Timestamp(first)) & ((d <= pd.Timestamp(last)) if last else True)]
    return ym(d.min()), ym(d.max())


def _replication_span(s: Sources, period: str) -> tuple[str, str]:
    lo, hi = rc.PERIODS[period]
    return _span(s.read("scores/replication_cumdsse.parquet", "returns"), 12, lo, hi)


def _archive_span(s: Sources, h: int) -> tuple[str, str]:
    return _span(s.read("scores/frozen_cumdsse.parquet", "archive"), h, rc.PERIODS["holdout_ap"][0])


def mean_r2(s: Sources) -> pd.DataFrame:
    rows = []
    cells = [("original", h, ARCHIVE_PERIOD, _archive_scores(s)) for h in ARCHIVE_HORIZONS]
    cells += [("replication", 12, p, _replication_scores(s)) for p in ("holdout", "holdout_ap")]
    for source, h, period, sc in cells:
        det = vd.get(sc, "hist_mean", h, "pooled", period, "none", "r2_detectable")
        for m in (*rc.FROZEN_MEAN, *rc.ROBUSTNESS):
            g = lambda metric: vd.get(sc, m, h, "pooled", period, "hist_mean", metric)  # noqa: E731
            rows.append({"id": f"{source}~{h}~{period}~{m}", "source": source, "horizon": h, "period": period,
                         "model": m, "label": gl.label(m), "role": "frozen" if m in rc.FROZEN_MEAN else "robustness",
                         "r2": pc(g("r2_oos"), 2), "cw_p": pv(g("cw_p")), "dm_p": pv(g("dm_mse_p1")),
                         "r2_detectable": pc(det, 1)})
    return pd.DataFrame(rows)


def mean_cum(s: Sources) -> pd.DataFrame:
    parts = []
    for source, cum, hs in (("original", s.read("scores/frozen_cumdsse.parquet", "archive"), ARCHIVE_HORIZONS),
                            ("replication", s.read("scores/replication_cumdsse.parquet", "returns"), (12,))):
        c = cum[(cum.asset == "pooled") & cum.model.isin(rc.FROZEN_MEAN) & cum.horizon.astype(int).isin(hs)]
        parts.append(pd.DataFrame({"series": source + "~" + c.horizon.astype(int).astype(str), "source": source,
                                   "horizon": c.horizon.astype(int), "model": c.model,
                                   "date": pd.to_datetime(c.date).dt.strftime("%Y-%m"), "cum": c.cum_dsse.round(5)}))
    return pd.concat(parts, ignore_index=True)


DIST_ROWS = (  # (model, target, assets-or-'pooled'-or-'each', benchmarks for skill)
    ("dist_hm_af", "assets"), ("hist_mean", "assets"), ("vol_ewma", "assets"),
    ("dist_hm_af", "portfolios"), ("dist_hm_af.assetshape", "portfolios"), ("hist_mean", "portfolios"),
)


def dist_calib(s: Sources) -> pd.DataFrame:
    sc = _dist_scores(s)
    rows = []
    for model, target in DIST_ROWS:
        names = ["pooled"]
        if model not in ("hist_mean", "vol_ewma"):
            have = sc[(sc.model == model) & (sc.target == target)].asset.unique()
            names += [a for a in (rc.ASSETS if target == "assets" else gl.PORTFOLIOS) if a in have]
        for a in names:
            for period in ("holdout", "full"):
                g = lambda metric, bench="none": vd.get(sc, model, 12, a, period, bench, metric, target)  # noqa: E731
                if not vd.ok(g("cover_90")):
                    continue
                name = "All assets" if (a == "pooled" and target == "assets") else (
                    "All three portfolios" if a == "pooled" else rc.ASSETS.get(a, gl.PORTFOLIOS.get(a, a)))
                r = {"id": f"{model}~{target}~{a}~{period}", "model": model, "label": gl.label(model),
                     "target": target, "asset": a, "name": name, "period": period}
                for k in ("cover_50", "cover_80", "cover_90", "cover_98", "hit_q01", "hit_q05", "hit_q95", "hit_q99"):
                    r[k] = pc(g(k), 1)
                r.update({"calib_p": pv(g("calib_p")), "kupiec_p_q05": pv(g("kupiec_p_q05")),
                          "kupiec_p_q01": pv(g("kupiec_p_q01")), "n_indep": round(g("n_indep"), 1),
                          "crps_skill_hm": pc(g("crps_skill", "hist_mean"), 2),
                          "crps_p_hm": pv(g("dm_crps_p", "hist_mean")),
                          "crps_skill_vol": pc(g("crps_skill", "vol_ewma"), 2),
                          "crps_p_vol": pv(g("dm_crps_p", "vol_ewma"))})
                rows.append(r)
    return pd.DataFrame(rows)


def dist_original(s: Sources) -> pd.DataFrame:
    sc = _archive_scores(s)
    rows = []
    for h in ARCHIVE_HORIZONS:
        for m in archive_dist_family(sc):
            g = lambda metric, bench="none": vd.get(sc, m, h, "pooled", ARCHIVE_PERIOD, bench, metric)  # noqa: E731
            rows.append({"id": f"{h}~{m}", "horizon": h, "model": m, "label": gl.label(m),
                         "crps_skill_vol": pc(g("crps_skill", "vol_ewma"), 2), "crps_p_vol": pv(g("dm_crps_p", "vol_ewma")),
                         "cover_90": pc(g("cover_90"), 1), "calib_p": pv(g("calib_p"))})
    return pd.DataFrame(rows)


def strategy(s: Sources) -> pd.DataFrame:
    a = s.read("strategy/summary_by_period.csv", "archive")
    b = s.read("strategy/summary_by_period.csv", "returns")
    sel = []
    a = a[(a.family == "mean") & (a.universe == "all") & (a.period == ARCHIVE_PERIOD)
          & a.mean_model.isin(rc.FROZEN_MEAN) & a.rule.isin(RULES)]
    sel.append(a.assign(source="original"))
    m = b[(b.family == "mean") & (b.universe == "all") & b.period.isin(["holdout", "holdout_ap"])
          & b.mean_model.isin((*rc.FROZEN_MEAN, *rc.ROBUSTNESS)) & b.rule.isin(RULES)]
    r = b[(b.family == "risk") & (b.universe == "all") & b.period.isin(["holdout", "holdout_ap"])
          & b.rule.isin(RISK_RULES)]
    sel += [m.assign(source="replication"), r.assign(source="replication")]
    d = pd.concat(sel, ignore_index=True)
    d["horizon"] = d.horizon.astype(int)
    who = np.where(d.family == "risk", d.cov_model, d.mean_model)
    out = pd.DataFrame({
        "id": d.source + "~" + d.family + "~" + d.horizon.astype(str) + "~" + d.period + "~" + d.rule + "~" + who,
        "source": d.source, "family": d.family, "horizon": d.horizon, "period": d.period, "rule": d.rule,
        "rule_label": d.rule.map(gl.rule_label), "model": who,
        "label": np.where(d.family == "risk", "Package covariance vs static sample", d.mean_model.map(gl.label)),
        "role": np.where(d.mean_model.isin(rc.ROBUSTNESS), "robustness", "frozen"),
        "d_cer": pc(d.d_cer, 2), "ci_lo": pc(d.d_cer_ci_lo, 2), "ci_hi": pc(d.d_cer_ci_hi, 2),
        "d_cer_p": pv(d.d_cer_p), "d_cer_q": pv(d.d_cer_q_bh), "d_sharpe": d.d_sharpe.round(2)})
    return out


# --------------------------------------------------------------------------- verdicts

VERDICT_UNITS = {  # field -> converter to display units
    **dict.fromkeys(("r2", "skill", "best_d_cer", "worst_d_cer"), lambda x: pc(x, 2)),
    **dict.fromkeys(("r2_detectable", "cover90", "hit_q05", "hit_q01", "hit_q95", "hit_q99"), lambda x: pc(x, 1)),
    **dict.fromkeys(("spa_p", "calib_p", "kupiec_p_q05", "kupiec_p_q01", "p", "min_q"), pv),
    "n_indep": lambda x: round(x, 1) if vd.ok(x) else np.nan,
}


def _verdict_row(question: str, evidence: str, horizon, period: str, span: tuple[str, str], v: dict) -> dict:
    row = {"question": question, "evidence": evidence, "horizon": horizon, "period": period,
           "first_origin": span[0], "last_origin": span[1]}
    for k, x in v.items():
        row[k] = VERDICT_UNITS[k](x) if k in VERDICT_UNITS else x
    for k in ("best_model", "worst_model"):
        if v.get(k) and v[k] in gl.MODELS:
            row[k.replace("model", "label")] = gl.label(v[k])
    for k in ("best_rule", "worst_rule"):
        if v.get(k):
            row[k + "_label"] = gl.rule_label(v[k])
    row["label"] = rc.REPLICATION_LABEL if evidence == "replication" else ""
    return row


def verdict_rows(s: Sources) -> list[dict]:
    rows = []
    # 1. the covariance model's own evaluation (results/final; genuine out-of-sample)
    hl = s.json("summary.json", "final")["headline"]["holdout"]
    hold = tuple(ym(x) for x in s.json("summary.json", "final")["holdout"])
    vr = hl["vol_vs_primary"]["reference"]
    rows.append(_verdict_row("cov_vol", "covariance", 12, "holdout", hold, {
        "status": vd.skill_verdict(vr["primary_skill_pct"], vr["dm_p"]), "skill": vr["primary_skill_pct"] / 100,
        "skill_lo": round(vr["ci90"][0], 1), "skill_hi": round(vr["ci90"][1], 1), "p": vr["dm_p"],
        "q_model": round(hl["vol_qlike"]["primary"], 3), "q_ref": round(hl["vol_qlike"]["reference"], 3)}))
    cr = hl["cov_vs_primary"]["reference"]
    rows.append(_verdict_row("cov_matrix", "covariance", 12, "holdout", hold, {
        "status": vd.skill_verdict(-cr["diff"], cr["dm_p"]), "diff": round(cr["diff"], 3),
        "diff_lo": round(cr["ci90"][0], 3), "diff_hi": round(cr["ci90"][1], 3), "p": cr["dm_p"],
        "q_model": round(hl["cov_qlike_rel"]["primary"], 3), "q_ref": round(hl["cov_qlike_rel"]["reference"], 3)}))
    hc = s.read("h2h_corr.csv", "final")
    hc = hc[(hc.subset == "daily") & (hc.period == "holdout") & (hc.role == "primary")].iloc[0]
    rows.append(_verdict_row("cov_corr", "covariance", 12, "holdout", hold, {
        "status": vd.skill_verdict(hc.skill_pct, hc.dm_p), "skill": hc.skill_pct / 100,
        "skill_lo": round(hc.skill_lo, 1), "skill_hi": round(hc.skill_hi, 1), "p": hc.dm_p}))
    rows.append(_verdict_row("cov_expect", "covariance", 12, "holdout", hold,
                             vd.expectations_verdict(s.read("expectations.csv", "final").status)))

    # 2. the original return-forecasting project's holdout (archived; genuine out-of-sample)
    a = _archive_scores(s)
    astrat = s.read("strategy/summary_by_period.csv", "archive")
    per_h = {}
    for h in ARCHIVE_HORIZONS:
        span = _archive_span(s, h)
        per_h[h] = {"mean": vd.mean_verdict(a, rc.FROZEN_MEAN, h, ARCHIVE_PERIOD),
                    "dist": vd.dist_skill_verdict(a, archive_dist_family(a), h, ARCHIVE_PERIOD),
                    "strat": vd.strategy_verdict(astrat, rc.FROZEN_MEAN, RULES, h, ARCHIVE_PERIOD)}
        for k, v in per_h[h].items():
            rows.append(_verdict_row(f"{k}_orig_h{h}", "original", h, ARCHIVE_PERIOD, span, v))
    span = (_archive_span(s, ARCHIVE_HORIZONS[0])[0], max(_archive_span(s, h)[1] for h in ARCHIVE_HORIZONS))
    for k in ("mean", "dist"):
        rows.append(_verdict_row(f"{k}_orig", "original", "all", ARCHIVE_PERIOD, span,
                                 vd.combine_edges([per_h[h][k] for h in ARCHIVE_HORIZONS])))
    rows.append(_verdict_row("strat_orig", "original", "all", ARCHIVE_PERIOD, span,
                             vd.combine_strategy([per_h[h]["strat"] for h in ARCHIVE_HORIZONS])))

    # 3. the merge-time replication on the 12-asset panel (seen data; descriptive)
    r, d = _replication_scores(s), _dist_scores(s)
    rstrat = s.read("strategy/summary_by_period.csv", "returns")
    for period, suf in (("holdout", ""), ("holdout_ap", "_ap")):
        span = _replication_span(s, period)
        rows.append(_verdict_row(f"mean_rep{suf}", "replication", 12, period, span,
                                 vd.mean_verdict(r, rc.FROZEN_MEAN, 12, period)))
        rows.append(_verdict_row(f"strat_rep{suf}", "replication", 12, period, span,
                                 vd.strategy_verdict(rstrat, rc.FROZEN_MEAN, RULES, 12, period)))
    span = _replication_span(s, "holdout")
    for period in ("holdout", "full"):
        sp = span if period == "holdout" else _span(s.read("scores/replication_cumdsse.parquet", "returns"), 12,
                                                     "1900-01-31")
        suf = "" if period == "holdout" else "_full"
        for q, model, target in (("calib_assets", "dist_hm_af", "assets"), ("calib_ports", "dist_hm_af", "portfolios"),
                                 ("calib_ports_as", "dist_hm_af.assetshape", "portfolios")):
            rows.append(_verdict_row(q + suf, "replication", 12, period, sp,
                                     vd.calib_verdict(d, model, "pooled", period, target=target)))
    for bench, q in (("vol_ewma", "dist_rep_vol"), ("hist_mean", "dist_rep_hm")):
        sk = vd.get(d, "dist_hm_af", 12, "pooled", "holdout", bench, "crps_skill", "assets")
        p = vd.get(d, "dist_hm_af", 12, "pooled", "holdout", bench, "dm_crps_p", "assets")
        rows.append(_verdict_row(q, "replication", 12, "holdout", span,
                                 {"status": vd.skill_verdict(sk, p), "skill": sk, "p": p}))
    rv = vd.strategy_verdict(rstrat, ("hist_mean", "none"), RISK_RULES, 12, "holdout", family="risk")
    rk = rstrat[(rstrat.family == "risk") & (rstrat.universe == "all") & (rstrat.horizon.astype(int) == 12)
                & (rstrat.period == "holdout") & rstrat.rule.isin(RISK_RULES)]
    behind = rk[rk.d_cer < 0]
    rv["behind_list"] = "; ".join(f"{gl.rule_label(r.rule)} ({pc(r.d_cer, 2):+.2f}% a year, p = {pv(r.d_cer_p)})"
                                  for r in behind.itertuples()) or "none"
    rows.append(_verdict_row("risk_rep", "replication", 12, "holdout", span, rv))
    return rows


def verdicts(s: Sources) -> pd.DataFrame:
    rows = verdict_rows(s)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    out = pd.DataFrame(rows, columns=cols)
    for c in out.columns:   # counts print as integers
        if (c == "n" or c.startswith("n_")) and c != "n_indep":
            out[c] = out[c].astype("Int64")
    return out


# --------------------------------------------------------------------------- efficient frontier

FRONTIER_GROUPS = ("equity", "rates", "credit", "real")
FRONTIER_POINTS = {  # point id -> label (assets take their own name)
    "min_variance": "Minimum variance", "max_sharpe": "Best return per unit of risk", "risk_parity": "Risk parity",
}
FRONTIER_KINDS = ("policy", "special", "match", "asset")


def _weights(f: pd.DataFrame) -> pd.DataFrame:
    w = f[[c for c in f.columns if c.startswith("w_")]]
    return w.rename(columns=lambda c: c[2:])


def weight_list(w: pd.Series) -> str:
    """'37% TIPS, 36% High yield, ...' for weights of at least half a percent, largest first, then cash."""
    held = w[w >= 0.005].sort_values(ascending=False, kind="stable")
    parts = [f"{round(100 * v):g}% {gl.SHORT_ASSET.get(a, a)}" for a, v in held.items()]
    rest = round(100 * (1 - float(w.sum())))
    if rest > 0:
        parts.append(f"{rest:g}% cash")
    return ", ".join(parts) if parts else "100% cash"


def _frontier_frame(f: pd.DataFrame) -> pd.DataFrame:
    """The columns the frontier line and the reference points share, in display units."""
    w = _weights(f)
    out = pd.DataFrame({"vol": pc(f.vol, 2), "growth": pc(f.growth, 2), "growth_lo": pc(f.growth_lo, 2),
                        "growth_hi": pc(f.growth_hi, 2), "invested": pc(f.invested, 1)})
    for g in FRONTIER_GROUPS:
        out[g] = pc(w[[a for a in w.columns if ASSET[a].group == g]].sum(axis=1), 1)
    for a in w.columns:
        out[f"w_{a}"] = pc(w[a], 1)
    out["weights"] = [weight_list(w.loc[i]) for i in w.index]
    return out


def frontier(s: Sources) -> pd.DataFrame:
    f = s.read("frontier.csv")
    out = _frontier_frame(f)
    out.insert(0, "i", f.i)
    out.insert(0, "curve", f.curve)
    out.insert(0, "id", f.curve + "~" + f.i.astype(str))
    return out


def frontier_points(s: Sources) -> pd.DataFrame:
    f = s.read("frontier_points.csv")
    f = f.assign(_k=f.kind.map({k: i for i, k in enumerate(FRONTIER_KINDS)})).sort_values("_k", kind="stable")
    f = f.drop(columns="_k").reset_index(drop=True)
    out = _frontier_frame(f)
    label = np.where(f.kind == "asset", f.name, np.where(
        f.kind == "policy", f.name.map(gl.PORTFOLIOS), np.where(
            f.kind == "match", "Frontier at the volatility of " + f.name.map(gl.PORTFOLIO_SHORT),
            f.point.map(FRONTIER_POINTS))))
    if pd.isna(label).any():
        raise ValueError("a frontier point has no label")
    out.insert(0, "label", label)
    out.insert(0, "group", [ASSET[a].group if k == "asset" else "" for a, k in zip(f.point, f.kind)])
    out.insert(0, "kind", f.kind)
    out.insert(0, "point", f.point)
    for c in ("gap", "gap_lo", "gap_hi"):
        out[c] = pc(f[c], 2)
    return out


def frontier_settings(s: Sources) -> pd.DataFrame:
    m = s.json("forecast.json")["frontier"]
    f, p = s.read("frontier.csv"), s.read("frontier_points.csv")
    wc = f[f.curve == "with_cash"]
    match = p[p.kind == "match"]
    top = wc.loc[wc.vol.idxmax()]
    return pd.DataFrame([{
        "key": "now", "level": round(100 * m["interval_level"]),
        "top_vol": pc(top.vol, 1), "top_growth": pc(top.growth, 1),
        "top_growth_lo": pc(top.growth_lo, 1), "top_growth_hi": pc(top.growth_hi, 1),
        "n_match": len(match), "n_gap_incl_zero": int(((match.gap_lo <= 0) & (match.gap_hi >= 0)).sum()),
        "n_assets_held": int((_weights(wc).max() >= 0.005).sum()), "n_assets": len(_weights(wc).columns),
    }])


# --------------------------------------------------------------------------- registry

@dataclass(frozen=True)
class Dataset:
    id: str
    title: str
    description: str
    sources: tuple[str, ...]
    build: Callable[[Sources], pd.DataFrame]

    @property
    def file(self) -> str:
        return f"{self.id}.csv"


DATASETS: tuple[Dataset, ...] = (
    Dataset("snapshot", "Forecast date and labels",
            "One row: the forecast origin and window, the out-of-sample start for the return side, the replication label and the size of the shape samples.",
            ("forecasts/latest/forecast.json", "src/assetforecast/returns/config.py"), snapshot),
    Dataset("outlook_assets", "Expected return and range, by asset",
            "Per asset: expected 12-month log excess return with its 90% interval, and the quantiles of the predictive distribution, in percent.",
            ("forecasts/latest/expected_returns.csv", "forecasts/latest/distribution_assets.csv"), outlook_assets),
    Dataset("outlook_portfolios", "Expected return and range, policy portfolios",
            "Per policy portfolio and shape: expected 12-month log excess return, volatility and predictive quantiles; the asset-shape rows are a sensitivity check.",
            ("forecasts/latest/distribution_portfolios.csv",), outlook_portfolios),
    Dataset("frontier", "Efficient frontier",
            "Portfolios along the long-only frontier, fully invested and with cash (nothing borrowed): volatility, expected compound growth with its range, share invested and weights, in percent.",
            ("forecasts/latest/frontier.csv",), frontier),
    Dataset("frontier_points", "Portfolios and assets beside the frontier",
            "The policy portfolios, minimum variance, best return per unit of risk, risk parity, the frontier portfolio at each policy portfolio's volatility, and the twelve assets, on the frontier's axes.",
            ("forecasts/latest/frontier_points.csv",), frontier_points),
    Dataset("frontier_settings", "Frontier settings",
            "One row: the interval level, the top of the frontier and the counts quoted beside it.",
            ("forecasts/latest/forecast.json", "forecasts/latest/frontier.csv", "forecasts/latest/frontier_points.csv"), frontier_settings),
    Dataset("shape", "Shape of the predictive distribution",
            "Quantiles of past standardized 12-month outcomes, pooled over assets and over portfolios, beside the normal.",
            ("forecasts/latest/shape.csv",), shape),
    Dataset("verdicts", "Verdicts and their figures",
            "One row per question the page answers: which prewritten sentence applies (status) and the figures it quotes.",
            ("results/final/summary.json", "results/final/expectations.csv", "results/archive/assetpred/scores/frozen.parquet",
             "results/archive/assetpred/strategy/summary_by_period.csv", "results/returns/scores/replication.parquet",
             "results/returns/distribution/scores.parquet", "results/returns/strategy/summary_by_period.csv"), verdicts),
    Dataset("forecast_assets", "Volatility forecast by asset",
            "Per asset: 12-month and 21-day volatility forecasts, the range across model families, recent and long-run realized volatility, in percent.",
            ("forecasts/latest/vols.csv", "results/final/accuracy.csv"), forecast_assets),
    Dataset("forecast_corr", "Forecast correlations",
            "Every pair of assets with its forecast correlation over the next 12 months.",
            ("forecasts/latest/corr.csv",), forecast_corr),
    Dataset("forecast_portfolios", "Volatility forecast, policy portfolios",
            "Per policy portfolio: 12-month and 21-day volatility forecasts, in percent.",
            ("forecasts/latest/portfolios.csv",), forecast_portfolios),
    Dataset("history_vol", "Forecast and realized volatility through time",
            "Quarterly forecast origins from 1926: the selected model's 12-month volatility forecast, the family range and the volatility then realized, per asset.",
            ("results/final/dashboard_vol.csv",), history_vol),
    Dataset("history_corr", "Stock-bond correlation through time",
            "Quarterly forecast origins: forecast and realized correlation of US large caps with Treasuries.",
            ("results/final/dashboard_corr.csv",), history_corr),
    Dataset("candidates", "Volatility loss in and out of sample",
            "The selected model, the runner-up, the best single model and two references: volatility loss and realized/forecast variance before 2014 and on 2014-2025.",
            ("results/final/h2h_vol.csv",), candidates),
    Dataset("model_ranks", "Model ranks before and after 2014",
            "Each of 70 volatility models' rank on 1926-2013 and on the 2014-2025 holdout.",
            ("results/final/vol_leaderboard.csv",), model_ranks),
    Dataset("expectations", "Five written expectations",
            "The five expectations committed before the covariance holdout was scored, with their outcome, reworded from the scoring output.",
            ("results/final/expectations.csv",), expectations),
    Dataset("mean_r2", "Expected-return models against the historical mean",
            "Pooled out-of-sample R-squared of each frozen mean model against the historical mean: the original holdout at 1, 3 and 12 months and the 12-month replication.",
            ("results/archive/assetpred/scores/frozen.parquet", "results/returns/scores/replication.parquet"), mean_r2),
    Dataset("mean_cum", "Cumulative squared-error difference",
            "Running sum of the historical mean's squared error minus each frozen model's, pooled over assets, by forecast month: rising means the model was ahead.",
            ("results/archive/assetpred/scores/frozen_cumdsse.parquet", "results/returns/scores/replication_cumdsse.parquet"), mean_cum),
    Dataset("dist_calib", "Calibration of the predictive distributions",
            "Coverage of central intervals, tail hit rates and calibration tests of the published distribution and two baselines, assets and portfolios (replication).",
            ("results/returns/distribution/scores.parquet",), dist_calib),
    Dataset("dist_original", "Original distribution models",
            "The original project's five distribution models on its own holdout: CRPS gain over the volatility baseline and 90% coverage, at 1, 3 and 12 months.",
            ("results/archive/assetpred/scores/frozen.parquet",), dist_original),
    Dataset("strategy", "Portfolio test",
            "Net certainty-equivalent gain of each pre-declared rule on a model's expected returns over the same rule on historical means, and of the package covariance over a static one.",
            ("results/archive/assetpred/strategy/summary_by_period.csv", "results/returns/strategy/summary_by_period.csv"), strategy),
)
BY_ID = {d.id: d for d in DATASETS}


def build_all(s: Sources | None = None) -> dict[str, pd.DataFrame]:
    s = s or Sources()
    return {d.id: d.build(s) for d in DATASETS}


def manifest() -> list[dict]:
    return [{"id": d.id, "file": f"data/{d.file}", "title": d.title, "description": d.description,
             "sources": list(d.sources)} for d in DATASETS]


def write(frames: dict[str, pd.DataFrame], out: Path = paths.DASHBOARD_DATA, manifest_path: Path = MANIFEST) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for d in DATASETS:
        p = out / d.file
        frames[d.id].to_csv(p, index=False)
        written.append(p)
    for p in set(out.glob("*.csv")) - set(written):
        p.unlink()
    manifest_path.write_text(json.dumps(manifest(), indent=1) + "\n")
    return written
