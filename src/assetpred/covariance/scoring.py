"""Out-of-sample evaluation of covariance forecasts.

:func:`score_covariances` returns a tidy table with ``evaluation.scoring``'s
columns ``model, horizon, asset, period, benchmark, metric, value``; here
``asset`` names the *target* of the metric:

* ``joint``     the whole vector of eligible assets;
* ``port_ew``, ``port_6040``, ``port_iv``   fixed test portfolios (equal
  weight; 60% us_eq / 40% ust_int; inverse volatility from the trailing 36-month
  sample sd at the origin, the same weights for every model);
* ``mvp``       the global minimum-variance portfolio each model implies
  (unconstrained, fully invested: w = S^-1 1 / 1'S^-1 1);
* ``sb``        the us_eq / ust_int pair (stock-bond correlation).

Centring. All models are scored around the same mean, mu = h * expanding mean
of monthly excess returns through t (the ``hist_mean`` benchmark's mean), so
only the covariance differs; e = y - mu.

Per-date losses (lower is better), averaged over dates:
  nll       Gaussian negative log score per asset: [n log 2 pi + log|S| + e'S^-1 e] / (2n).
            With the realized outer product ee' as the proxy, QLIKE
            (log|S| + tr(S^-1 ee')) is the same loss up to constants, so it
            is not reported separately.
  fro       weighted Frobenius loss sum_ij (e~_i e~_j - S~_ij)^2 / n^2 with
            e~ = e / s, S~ = S / ss', s_i = sqrt(h * monthly sample var at t)
            (weights known at t, common to all models; robust to proxy noise
            in the sense of Patton 2011 / Laurent-Rombouts-Violante 2013).
  qlike     portfolio QLIKE  log v + r^2 / v  with v = w'Sw, r = w'e.
  mvp_sq    (w_mvp'e)^2, the squared realized minimum-variance return (Engle &
            Colacito 2006): lower = the model found a less risky portfolio.
  nll       (``sb``) bivariate Gaussian negative log score of the pair.
Descriptives: mahal = e'S^-1 e / n (1 if calibrated), cover90 (share of dates
with e'S^-1 e below the chi2_n 90% quantile), bias_ratio = mean(r^2)/mean(v),
z2 = mean(r^2/v), realized_vol / pred_vol (annualized, mvp), gross_lev (mvp),
cond_median / cond_max (condition number of the implied correlation matrix),
logdet, corr_pred (sb), corr_real (sb, model '*').

Inference vs a benchmark (``benchmark`` = ``'sample'``, and in the frozen
run also ``'lw'``, the well-conditioned static benchmark, so that "dynamics
beat shrinkage" has its own test): d_t = L_bench - L_model,
``diff`` = mean d (positive = model better), ``dm_stat``/``dm_p`` (one-sided,
model better; Newey-West with ``inference.hac_lags`` >= 1.5h lags, Student t),
``dm_p2`` (two-sided), and for h > 1 the non-overlapping strands: for each of
the h offsets the dates t0, t0+h, ... give independent-target series;
``strand_p_median`` and ``strand_share_05`` (share of strands with one-sided
p < 0.05). ``n_indep`` = n_dates / h is the number of non-overlapping
observations. These rows exist for every period.

Search-adjusted results, for each period in ``mt_periods`` (``("full",)`` in a
dev run; ``config.MT_PERIODS`` = full, pre_holdout, holdout in the full run),
when the period has at least ``MIN_MT_DATES`` dates:
  * ``model='*'``, ``metric='spa_p_nll'``, ``asset`` in {joint, sb},
    ``benchmark`` = the benchmark: Hansen SPA, does any model of the
    benchmark's family beat it (family for ``sample``: every other model;
    for ``lw``: the dynamic frozen models, see ``run.SPA_FAMILY``);
  * ``metric='mcs_p_nll'`` / ``'in_mcs_nll'`` (asset joint, benchmark none);
  * ``metric='dm_q_bh_nll'`` (asset joint and sb): Benjamini-Hochberg q-values
    of ``dm_p_nll`` across the family x horizons, per (period, benchmark).
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy import stats

from .. import config, schema
from ..evaluation import inference as inf
from ..evaluation.scoring import DEFAULT_SUBPERIODS, SCORE_COLUMNS
from . import linalg as la
from .engine import to_matrices

log = logging.getLogger(__name__)

PORTFOLIOS = ("port_ew", "port_6040", "port_iv")
STOCK, BOND = "us_eq", "ust_int"
TEST_LOSSES = {  # (target, metric) pairs that get inference vs the benchmark
    ("joint", "nll"), ("joint", "fro"), ("port_ew", "qlike"), ("port_6040", "qlike"),
    ("port_iv", "qlike"), ("mvp", "mvp_sq"), ("sb", "nll"),
}


def _weights(kind: str, assets, X_t: pd.DataFrame) -> np.ndarray:
    n = len(assets)
    if kind == "port_ew":
        return np.full(n, 1.0 / n)
    if kind == "port_6040":
        w = np.zeros(n)
        w[assets.index(STOCK)] = 0.6
        w[assets.index(BOND)] = 0.4
        return w
    if kind == "port_iv":
        sd = X_t[list(assets)].iloc[-36:].std(ddof=1).to_numpy(float)
        iv = 1.0 / sd
        return iv / iv.sum()
    raise ValueError(kind)


def _per_date(S: np.ndarray, e: np.ndarray, y: np.ndarray, s_ref: np.ndarray,
              W: dict, assets) -> dict:
    n = len(e)
    out = {}
    L = np.linalg.cholesky(S)
    logdet = 2 * np.log(np.diag(L)).sum()
    u = np.linalg.solve(L, e)
    m2 = float(u @ u)
    out["joint", "nll"] = 0.5 * (n * math.log(2 * math.pi) + logdet + m2) / n
    out["joint", "logdet"] = logdet / n
    out["joint", "mahal"] = m2 / n
    out["joint", "cover90"] = float(m2 <= stats.chi2.ppf(0.9, n))
    et = e / s_ref
    St = S / np.outer(s_ref, s_ref)
    out["joint", "fro"] = float(((np.outer(et, et) - St) ** 2).sum() / n ** 2)
    R, _ = la.cov_to_corr(S)
    ev = np.linalg.eigvalsh(R)
    out["joint", "cond"] = float(ev[-1] / max(ev[0], 1e-300))
    for p, w in W.items():
        v = float(w @ S @ w)
        r = float(w @ e)
        out[p, "qlike"] = math.log(v) + r * r / v
        out[p, "v"] = v
        out[p, "r2"] = r * r
    one = np.ones(n)
    Si1 = np.linalg.solve(S, one)
    wm = Si1 / Si1.sum()
    rm = float(wm @ e)
    out["mvp", "mvp_sq"] = rm * rm
    out["mvp", "v"] = float(wm @ S @ wm)
    out["mvp", "gross_lev"] = float(np.abs(wm).sum())
    if STOCK in assets and BOND in assets:
        ix = [assets.index(STOCK), assets.index(BOND)]
        S2 = S[np.ix_(ix, ix)]
        e2 = e[ix]
        det = S2[0, 0] * S2[1, 1] - S2[0, 1] ** 2
        q = (S2[1, 1] * e2[0] ** 2 - 2 * S2[0, 1] * e2[0] * e2[1] + S2[0, 0] * e2[1] ** 2) / det
        out["sb", "nll"] = 0.5 * (2 * math.log(2 * math.pi) + math.log(det) + q) / 2
        out["sb", "corr_pred"] = S2[0, 1] / math.sqrt(S2[0, 0] * S2[1, 1])
    return out


def per_date_table(cov: pd.DataFrame, xr: pd.DataFrame) -> pd.DataFrame:
    """Per-date losses for every (model, horizon) on the dates and assets common
    to all models at that horizon. Long: horizon, date, model, target, metric, value."""
    rows = []
    for h, fh in cov.groupby("horizon", sort=True):
        h = int(h)
        Y = schema.make_targets(xr, h)
        mats = {m: to_matrices(g) for m, g in fh.groupby("model")}
        dates = sorted(set.intersection(*[set(d) for d in mats.values()]))
        for t in dates:
            assets = None
            for m in mats:
                a = mats[m][t][0]
                assets = list(a) if assets is None else [x for x in assets if x in a]
            assets = tuple(assets)
            y = Y.loc[t, list(assets)].to_numpy(float)
            if not np.isfinite(y).all():
                continue
            X_t = xr.loc[:t]
            mu = h * X_t[list(assets)].mean().to_numpy(float)
            s_ref = np.sqrt(h * X_t[list(assets)].var(ddof=1).to_numpy(float))
            e = y - mu
            W = {p: _weights(p, assets, X_t) for p in PORTFOLIOS
                 if p != "port_6040" or (STOCK in assets and BOND in assets)}
            for m, d in mats.items():
                a_m, S = d[t]
                ix = [a_m.index(a) for a in assets]
                S = S[np.ix_(ix, ix)]
                for (tg, met), v in _per_date(S, e, y, s_ref, W, assets).items():
                    rows.append((h, t, m, tg, met, v))
            rows.append((h, t, "*", "joint", "n_assets", float(len(assets))))
            if STOCK in assets and BOND in assets:
                rows.append((h, t, "*", "sb", "e_stock", e[assets.index(STOCK)]))
                rows.append((h, t, "*", "sb", "e_bond", e[assets.index(BOND)]))
    return pd.DataFrame(rows, columns=["horizon", "date", "model", "target", "metric", "value"])


def _strands(d: np.ndarray, h: int) -> tuple[float, float]:
    if h == 1:
        return np.nan, np.nan
    ps = []
    for o in range(h):
        ds = d[o::h]
        if len(ds) >= 10:
            ps.append(inf.mean_test(ds, 1)[1])
    ps = np.array([p for p in ps if np.isfinite(p)])
    if not len(ps):
        return np.nan, np.nan
    return float(np.median(ps)), float((ps < 0.05).mean())


MIN_MT_DATES = 24   # fewest dates for SPA / MCS in a period


def _family(models, bm, benchmarks, spa_family):
    """Models tested jointly against ``bm`` (SPA, BH). ``spa_family[bm]`` if
    given, else every other model (the original behaviour for ``sample``)."""
    if spa_family and bm in spa_family:
        return [m for m in spa_family[bm] if m in models and m != bm]
    return [m for m in models if m != bm]


def score_covariances(cov: pd.DataFrame, xr: pd.DataFrame, benchmark: str = "sample",
                      subperiods=DEFAULT_SUBPERIODS, reps: int = 999,
                      seed: int = config.SEED, extra_benchmarks=(),
                      mt_periods=("full",), spa_family: dict | None = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score covariance forecasts; returns (scores, per-date table).

    ``benchmark`` is the main benchmark. Each name in ``extra_benchmarks``
    (e.g. ``lw``, well-conditioned but static) gets the same pairwise tests
    (rows with ``benchmark`` = that name) for every other model.
    ``mt_periods``: periods (labels from "full" and ``subperiods``) for which
    the search-adjusted results are computed: SPA (``spa_p_nll`` on targets
    ``joint`` and ``sb``, ``model='*'``), the MCS on the joint nll
    (``mcs_p_nll``/``in_mcs_nll``, ``benchmark='none'``) and BH q-values of
    ``dm_p_nll`` (``dm_q_bh_nll``, across the benchmark's family x horizons),
    each needing at least ``MIN_MT_DATES`` dates. ``spa_family`` maps a
    benchmark to the models in its SPA / BH family (default: all others).
    """
    pdt = per_date_table(cov, xr)
    rows = []
    periods = [("full", None, None)] + list(subperiods or ())
    benchmarks = [benchmark] + [b for b in extra_benchmarks if b != benchmark]
    for h, ph in pdt.groupby("horizon", sort=True):
        h = int(h)
        wide = ph.pivot_table(index="date", columns=["model", "target", "metric"], values="value")
        models = sorted(m for m in ph["model"].unique() if m != "*")
        for bm in benchmarks:
            if bm not in models:
                raise ValueError(f"h={h}: benchmark {bm!r} missing")
        for plabel, start, end in periods:
            sel = wide
            if start is not None:
                sel = wide.loc[(wide.index >= pd.Timestamp(start)) & (wide.index <= pd.Timestamp(end))]
            nd = len(sel)
            if nd < 12:
                continue
            base = dict(n_dates=nd, n_indep=nd / h,
                        n_assets_mean=sel[("*", "joint", "n_assets")].mean())
            for k, v in base.items():
                rows.append(("*", h, "joint", plabel, "none", k, v))
            if ("*", "sb", "e_stock") in sel:
                es, eb = sel[("*", "sb", "e_stock")], sel[("*", "sb", "e_bond")]
                rows.append(("*", h, "sb", plabel, "none", "corr_real", float(np.corrcoef(es, eb)[0, 1])))
            for m in models:
                g = sel[m]
                for (tg, met) in g.columns:
                    col = g[(tg, met)]
                    if met in ("v", "r2"):
                        continue
                    if met == "cond":
                        rows.append((m, h, tg, plabel, "none", "cond_median", col.median()))
                        rows.append((m, h, tg, plabel, "none", "cond_max", col.max()))
                        continue
                    rows.append((m, h, tg, plabel, "none", met, col.mean()))
                for p in PORTFOLIOS:
                    if (p, "v") in g:
                        v, r2 = g[(p, "v")], g[(p, "r2")]
                        rows.append((m, h, p, plabel, "none", "bias_ratio", r2.mean() / v.mean()))
                        rows.append((m, h, p, plabel, "none", "z2", (r2 / v).mean()))
                ann = math.sqrt(12.0 / h)
                rows.append((m, h, "mvp", plabel, "none", "realized_vol", math.sqrt(g[("mvp", "mvp_sq")].mean()) * ann))
                rows.append((m, h, "mvp", plabel, "none", "pred_vol", np.sqrt(g[("mvp", "v")]).mean() * ann))
                for bm in benchmarks:
                    if m == bm:
                        continue
                    gb = sel[bm]
                    for tg, met in TEST_LOSSES:
                        if (tg, met) not in g or (tg, met) not in gb:
                            continue
                        lm, lb = g[(tg, met)].to_numpy(float), gb[(tg, met)].to_numpy(float)
                        d = lb - lm
                        stat, p1, p2 = inf.diebold_mariano(lm, lb, h)
                        sp, sh = _strands(d, h)
                        for k, v in dict(diff=d.mean(), dm_stat=stat, dm_p=p1, dm_p2=p2,
                                         strand_p_median=sp, strand_share_05=sh).items():
                            rows.append((m, h, tg, plabel, bm, f"{k}_{met}", v))
                    rm = math.sqrt(g[("mvp", "mvp_sq")].mean())
                    rb = math.sqrt(gb[("mvp", "mvp_sq")].mean())
                    rows.append((m, h, "mvp", plabel, bm, "vol_ratio", rm / rb))
        # data-snooping-robust tests, per multiple-testing period
        for plabel, start, end in periods:
            if plabel not in mt_periods:
                continue
            w = wide
            if start is not None:
                w = wide.loc[(wide.index >= pd.Timestamp(start)) & (wide.index <= pd.Timestamp(end))]
            for tg in ("joint", "sb"):
                if (models[0], tg, "nll") not in w:
                    continue
                L = w.xs((tg, "nll"), axis=1, level=["target", "metric"])[models].dropna()
                if len(L) < MIN_MT_DATES:
                    continue
                for bm in benchmarks:
                    fam = _family(models, bm, benchmarks, spa_family)
                    if fam:
                        rows.append(("*", h, tg, plabel, bm, "spa_p_nll",
                                     inf.spa_test(L[bm], L[fam], h, reps=reps, seed=seed + h)))
                if tg == "joint" and len(models) >= 2:
                    mcs = inf.model_confidence_set(L, h, size=0.10, reps=reps, seed=seed + h)
                    for m in mcs.index:
                        rows.append((m, h, "joint", plabel, "none", "mcs_p_nll", mcs.at[m, "mcs_p"]))
                        rows.append((m, h, "joint", plabel, "none", "in_mcs_nll", mcs.at[m, "in_mcs"]))
    scores = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    scores["value"] = scores["value"].astype(float)
    # BH q-values of the one-sided DM p-values, per (period, benchmark, target),
    # across the benchmark's family x horizons
    extra = []
    nd = scores[(scores.model == "*") & (scores.metric == "n_dates") & (scores.asset == "joint")]
    for plabel in mt_periods:
        ok_h = set(nd[(nd.period == plabel) & (nd.value >= MIN_MT_DATES)].horizon)
        for bm in benchmarks:
            fam = set(_family(sorted(set(scores.model) - {"*"}), bm, benchmarks, spa_family))
            for tg in ("joint", "sb"):
                sel = ((scores.metric == "dm_p_nll") & (scores.asset == tg) & (scores.period == plabel)
                       & (scores.benchmark == bm) & scores.model.isin(fam) & scores.horizon.isin(ok_h))
                if sel.any():
                    q = scores[sel].copy()
                    q["metric"] = "dm_q_bh_nll"
                    q["value"] = inf.fdr(q["value"].to_numpy(), "bh")
                    extra.append(q)
    if extra:
        scores = pd.concat([scores, *extra], ignore_index=True)
    return scores, pdt


def summary(scores: pd.DataFrame, benchmark: str = "sample", period: str = "full") -> pd.DataFrame:
    """One row per (model, horizon) for one period, tests against ``benchmark``."""
    s = scores[scores.period == period]
    pick = [
        ("joint", "none", "nll", "nll"),
        ("joint", benchmark, "diff_nll", "d_nll"),
        ("joint", benchmark, "dm_p_nll", "p_nll"),
        ("joint", benchmark, "strand_p_median_nll", "strand_p"),
        ("joint", benchmark, "diff_fro", "d_fro"),
        ("joint", benchmark, "dm_p_fro", "p_fro"),
        ("joint", "none", "mahal", "mahal"),
        ("port_ew", "none", "bias_ratio", "ew_bias"),
        ("port_6040", benchmark, "dm_p_qlike", "p_6040"),
        ("mvp", "none", "realized_vol", "mvp_vol"),
        ("mvp", benchmark, "dm_p_mvp_sq", "p_mvp"),
        ("sb", benchmark, "diff_nll", "d_sb"),
        ("joint", "none", "cond_median", "cond"),
        ("joint", "none", "in_mcs_nll", "in_mcs"),
        ("joint", benchmark, "dm_q_bh_nll", "q_nll"),
    ]
    parts = []
    for tg, bm, met, col in pick:
        x = s[(s.asset == tg) & (s.benchmark == bm) & (s.metric == met)]
        parts.append(x.set_index(["model", "horizon"])["value"].rename(col))
    out = pd.concat(parts, axis=1)
    return out[~out.index.get_level_values("model").isin(["*"])]
