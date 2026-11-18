"""Verdict rules: which of the page's prewritten sentences applies, decided from the results.

The dashboard page holds every sentence that states a conclusion; this module
decides, from the scores, which one is true and supplies the figures it
quotes. Each function returns a plain dict (fractions, not percent) that
``data.verdicts`` turns into one row of ``verdicts.csv``. The rules are
deliberately simple, are stated on the page, and were carried over from
``assetpred.dashboard.narrative`` (``mean_verdict``, ``strategy_verdict``),
restricted to what the merged package publishes:

* An edge **holds** (``pass``) when the point estimate favours the model *and*
  the test that allows for the search over models is below ``ALPHA``: the SPA
  test over the pre-registered family for means and distributions,
  Benjamini-Hochberg q-values for the strategy comparisons, the
  Diebold-Mariano p-value for a single pre-declared comparison.
* It is **not significant** (``weak``) when the point estimate favours the
  model but that test does not.
* There is **no edge** (``none``) when the point estimate does not favour the
  model. For the strategy test, ``worse`` replaces ``weak``/``none`` when more
  than half the comparisons lost to the historical-mean version.

Calibration is described, not passed or failed: ``ok`` when the 90% interval
held within ``CALIB_TOL`` of 90% of outcomes, else ``wide`` or ``narrow``; the
PIT test's verdict is reported separately (``calib_test``).

Verdicts are taken over the frozen test families only; robustness rows and
the all-model SPA rows (model ``'*'``, which depend on which unrelated models
share a run) are never used.
"""
from __future__ import annotations

import math

import pandas as pd

ALPHA = 0.05      # one threshold for p-values and q-values everywhere on the page
CALIB_TOL = 0.02  # |coverage of the 90% interval - 0.90| within which coverage is called accurate


def ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


_KEYS = ["model", "horizon", "asset", "period", "benchmark", "metric"]
_INDEX: dict[int, tuple[pd.DataFrame, dict]] = {}


def _index(scores: pd.DataFrame) -> dict:
    hit = _INDEX.get(id(scores))
    if hit is not None and hit[0] is scores:
        return hit[1]
    has_t = "target" in scores.columns
    cols = _KEYS + (["target"] if has_t else [])
    d = scores.assign(horizon=scores.horizon.astype(int))
    idx: dict = {}
    for key, v in zip(zip(*(d[c] for c in cols)), d["value"]):
        idx.setdefault(key[:6], []).append((key[6] if has_t else None, v))
    _INDEX[id(scores)] = (scores, idx)
    return idx


def get(scores: pd.DataFrame, model, horizon, asset, period, benchmark, metric, target=None) -> float:
    """One value of a tidy scores table (model, horizon, asset, period, benchmark, metric[, target], value),
    or NaN when absent. ``target`` selects among tables that carry a ``target`` column."""
    hits = _index(scores).get((model, int(horizon), asset, period, benchmark, metric), [])
    if target is not None:
        hits = [h for h in hits if h[0] == target]
    if len(hits) > 1:
        raise ValueError(f"{len(hits)} rows for {(model, horizon, asset, period, benchmark, metric, target)}")
    return float(hits[0][1]) if hits else float("nan")


def _strategy_status(n: int, n_pos: int, n_q_pos: int, n_neg: int) -> str:
    if n_q_pos:
        return "pass"
    if 2 * n_neg > n:
        return "worse"
    return "weak" if n_pos else "none"


def _edge(point_favours: bool, test_p: float) -> str:
    if point_favours and ok(test_p) and test_p < ALPHA:
        return "pass"
    return "weak" if point_favours else "none"


# --------------------------------------------------------------------------- expected return

def mean_verdict(scores: pd.DataFrame, family, horizon: int, period: str, spa_label: str = "*mean") -> dict:
    """Does any frozen mean model beat the historical mean (pooled out-of-sample R^2, family SPA)?"""
    r2 = {m: get(scores, m, horizon, "pooled", period, "hist_mean", "r2_oos") for m in family}
    r2 = {m: v for m, v in r2.items() if ok(v)}
    out = {"n_models": len(r2), "n_positive": sum(v > 0 for v in r2.values()),
           "spa_p": get(scores, spa_label, horizon, "pooled", period, "hist_mean", "spa_p_mse"),
           "r2_detectable": get(scores, "hist_mean", horizon, "pooled", period, "none", "r2_detectable"),
           "n_indep": get(scores, "hist_mean", horizon, "pooled", period, "none", "n_indep")}
    if not r2:
        return {**out, "status": "na", "best_model": "", "r2": float("nan")}
    best = max(r2, key=r2.get)
    return {**out, "status": _edge(r2[best] > 0, out["spa_p"]), "best_model": best, "r2": r2[best]}


# --------------------------------------------------------------------------- distributions

def dist_skill_verdict(scores: pd.DataFrame, family, horizon: int, period: str, bench: str = "vol_ewma",
                       spa_label: str = "*dist") -> dict:
    """Does any frozen distribution model beat ``bench`` on CRPS (family SPA)?"""
    sk = {m: get(scores, m, horizon, "pooled", period, bench, "crps_skill") for m in family}
    sk = {m: v for m, v in sk.items() if ok(v)}
    out = {"n_models": len(sk), "n_positive": sum(v > 0 for v in sk.values()),
           "spa_p": get(scores, spa_label, horizon, "pooled", period, bench, "spa_p_crps")}
    if not sk:
        return {**out, "status": "na", "best_model": "", "skill": float("nan")}
    best = max(sk, key=sk.get)
    return {**out, "status": _edge(sk[best] > 0, out["spa_p"]), "best_model": best, "skill": sk[best]}


def calib_verdict(scores: pd.DataFrame, model: str, asset: str, period: str, horizon: int = 12,
                  target: str | None = None) -> dict:
    """How often the 90% interval held, and whether the PIT test rejects calibration."""
    g = lambda metric: get(scores, model, horizon, asset, period, "none", metric, target)  # noqa: E731
    c90, cp = g("cover_90"), g("calib_p")
    if not ok(c90):
        status = "na"
    elif abs(c90 - 0.90) <= CALIB_TOL:
        status = "ok"
    else:
        status = "wide" if c90 > 0.90 else "narrow"
    return {"status": status, "cover90": c90, "calib_p": cp,
            "calib_test": "na" if not ok(cp) else ("rejects" if cp < ALPHA else "accepts"),
            "hit_q05": g("hit_q05"), "hit_q01": g("hit_q01"), "hit_q95": g("hit_q95"), "hit_q99": g("hit_q99"),
            "kupiec_p_q05": g("kupiec_p_q05"), "kupiec_p_q01": g("kupiec_p_q01"), "n_indep": g("n_indep")}


# --------------------------------------------------------------------------- portfolio value

def strategy_verdict(summary: pd.DataFrame, models, rules, horizon: int, period: str,
                     universe: str = "all", family: str = "mean") -> dict:
    """Net certainty-equivalent gain of each rule run on a model's means over the same rule on historical means.

    ``summary`` has the columns of ``strategy/summary_by_period.csv``."""
    s = summary
    g = s[(s.family == family) & (s.universe == universe) & (s.horizon.astype(int) == int(horizon))
          & (s.period == period) & s.mean_model.isin(list(models)) & s.rule.isin(list(rules))]
    g = g[g.d_cer.notna()]
    n = len(g)
    if not n:
        return {"status": "na", "n": 0}
    pos, neg = g[g.d_cer > 0], g[g.d_cer < 0]
    sig_q = pos[pos.d_cer_q_bh.notna() & (pos.d_cer_q_bh < ALPHA)]
    b, w = g.loc[g.d_cer.idxmax()], g.loc[g.d_cer.idxmin()]
    return {"status": _strategy_status(n, len(pos), len(sig_q), len(neg)), "n": n, "n_pos": len(pos), "n_sig_pos": int((pos.d_cer_p < ALPHA).sum()),
            "n_q_pos": len(sig_q), "n_neg": len(neg), "n_sig_neg": int((neg.d_cer_p < ALPHA).sum()),
            "best_d_cer": float(b.d_cer), "best_model": b.mean_model, "best_rule": b.rule,
            "worst_d_cer": float(w.d_cer), "worst_model": w.mean_model, "worst_rule": w.rule,
            "min_q": float(g.d_cer_q_bh.min()) if g.d_cer_q_bh.notna().any() else float("nan")}


# --------------------------------------------------------------------------- covariance side

def skill_verdict(skill: float, p: float) -> str:
    """A single pre-declared comparison where positive skill favours the selected model."""
    return "na" if not ok(skill) else _edge(skill > 0, p)


def expectations_verdict(statuses) -> dict:
    """Count the written expectations met (in full, or in point estimate only)."""
    st = [str(s).strip().lower() for s in statuses]
    met = [s for s in st if s.startswith("met")]
    partial = [s for s in met if "not significant" in s or "point estimate" in s]
    n, k = len(st), len(met)
    status = "na" if not n else ("all" if k == n else ("none" if k == 0 else "some"))
    return {"status": status, "n": n, "n_met": k, "n_met_point_only": len(partial), "n_not_met": n - k}


# --------------------------------------------------------------------------- several horizons at once

def combine_edges(vs) -> dict:
    """One verdict over several cells (horizons) of ``mean_verdict`` / ``dist_skill_verdict``."""
    st = [v["status"] for v in vs if v.get("status") != "na"]
    status = "na" if not st else ("pass" if "pass" in st else ("weak" if "weak" in st else "none"))
    return {"status": status, "n_cells": len(st), "n_pass": st.count("pass"), "n_weak": st.count("weak")}


def combine_strategy(vs) -> dict:
    """One verdict over several cells of ``strategy_verdict``: counts summed, same rule."""
    vs = [v for v in vs if v.get("status") != "na"]
    if not vs:
        return {"status": "na", "n": 0}
    tot = {k: sum(int(v[k]) for v in vs) for k in ("n", "n_pos", "n_sig_pos", "n_q_pos", "n_neg", "n_sig_neg")}
    status = _strategy_status(tot["n"], tot["n_pos"], tot["n_q_pos"], tot["n_neg"])
    return {"status": status, "n_cells": len(vs), **tot}
