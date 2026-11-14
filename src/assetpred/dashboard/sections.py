"""Assemble the data behind each section of the page as plain dicts/lists.

Each builder takes already-loaded results and returns JSON-ready structures
(numbers through :func:`extract.clean`). Missing inputs yield ``None`` for that
section so the page can say so instead of failing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, schema
from . import glossary as gl
from .extract import (ASSETS, BENCH, COV_BENCH, HORIZONS, POOLED, VOL_BENCH, Scores, clean,
                      load_cov_last, load_forecast_last, series_points)

TAIL_Q = ("q01", "q05", "q95", "q99")
TAIL_NOMINAL = {"q01": 0.01, "q05": 0.05, "q95": 0.05, "q99": 0.01}
COVER_LEVELS = (50, 80, 90, 98)
PINBALL_Q = tuple(schema.QUANTILE_COLUMNS)
DECADES = ("1990s", "2000s", "2010s", "2020s")


# --------------------------------------------------------------------------- periods

def periods(sc: Scores, cum: pd.DataFrame | None, dev: bool) -> dict:
    have = sc.periods
    holdout = "holdout" in have
    headline = "holdout" if holdout else "full"
    context = "pre_holdout" if "pre_holdout" in have else None
    order = [p for p in ("holdout", "pre_holdout") if p in have] + [p for p in DECADES if p in have] + ["full"]
    span = {}
    if cum is not None and not cum.empty:
        c = cum[cum.asset == POOLED]
        for h, g in c.groupby("horizon"):
            span[int(h)] = [pd.Timestamp(g.date.min()).strftime("%Y-%m"), pd.Timestamp(g.date.max()).strftime("%Y-%m")]
    hs = pd.Timestamp(config.HOLDOUT_START).strftime("%Y-%m")
    de = pd.Timestamp(config.DEV_END).strftime("%Y-%m")
    t12 = (pd.Timestamp(config.DEV_END) + pd.offsets.MonthEnd(12)).strftime("%Y-%m")
    labels = {
        "holdout": f"Holdout (origins from {hs})",
        "pre_holdout": f"Development period (origins to {de}; 3- and 12-month targets end as late as {t12})",
        "full": ("Development sample (all scored origins)" if dev or not holdout
                 else "Whole sample (development + holdout)"),
        **{d: f"{d} origins" for d in DECADES},
    }
    if holdout:
        labels["2010s"] = "2010s origins (straddle development and holdout)"
        labels["2020s"] = "2020s origins (holdout)"
    else:
        labels["2010s"] = f"2010s origins (to {de} only)"
    return {"have": have, "headline": headline, "context": context, "order": order, "labels": labels,
            "holdout_present": holdout, "span": span}


def _adj_period(sc: Scores, period: str, metric: str, label: str | None = None) -> str:
    """Multiple-testing rows (SPA, MCS, q-values) exist only for the periods in
    the run's ``mt_periods``. Use the headline period when it has them, else 'full'."""
    d = sc.df
    g = d[(d.metric == metric) & (d.period == period)]
    if label is not None:
        g = g[g.model == label]
    return period if not g.empty else "full"


ROBUST_PARENT = {m: m.split(".", 1)[0] for m in getattr(config, "ROBUSTNESS", ())}


def _hindsight_names() -> list[str]:
    """Features flagged as estimated with hindsight (the ones the .nohind rows drop)."""
    try:
        fm = pd.read_csv(schema.FEATURE_META)
        return sorted(fm.loc[fm.hindsight.astype(str).str.lower() == "true", "name"].tolist())
    except Exception:
        return []


# --------------------------------------------------------------------------- mean

def mean_section(sc: Scores, cum: pd.DataFrame | None, P: dict) -> dict | None:
    family = [m for m in config.FROZEN_MEAN if sc.has(m)]
    if not family:
        return None
    extra = [m for m in (config.COMPOSITE,) if sc.has(m)]
    robust = [m for m in ROBUST_PARENT if sc.has(m)]
    refs = [m for m in ("zero",) if sc.has(m)]
    models = family + extra
    hp, cp = P["headline"], P["context"]
    q_period = _adj_period(sc, hp, "cw_q_bh")
    spa_period = _adj_period(sc, hp, "spa_p_mse", "*mean")
    mcs_period = _adj_period(sc, hp, "in_mcs_mse")
    out = {"models": models, "family": family, "extra": extra, "robust": robust, "robust_parent": ROBUST_PARENT,
           "refs": refs, "q_period": q_period, "spa_period": spa_period, "mcs_period": mcs_period,
           "hindsight_features": _hindsight_names(),
           "pooled": {}, "grid": {}, "sub": {}, "cum": {}, "info": {}, "weights": {}}
    for h in HORIZONS:
        spa = sc.star("spa_p_mse", h, POOLED, BENCH, spa_period, family, label="*mean")
        spa_all = sc.star("spa_p_mse", h, POOLED, BENCH, spa_period, family, label="*")
        out["info"][h] = {
            "n_indep": clean(sc.get(BENCH, h, POOLED, hp, "none", "n_indep")),
            "n_dates": clean(sc.get(BENCH, h, POOLED, hp, "none", "n_dates")),
            "r2_detectable": clean(sc.get(BENCH, h, POOLED, hp, "none", "r2_detectable")),
            "spa_p": clean(spa[0]), "spa_run": spa[1], "spa_n": len(family) if spa[1] else None,
            "spa_all_p": clean(spa_all[0]), "spa_all_n": sc.run_size(spa_all[1], BENCH),
        }
        role = {**{m: "family" for m in family}, **{m: "composite" for m in extra},
                **{m: "robust" for m in robust}, **{m: "ref" for m in refs}}
        rows = []
        for m in family + extra + robust + refs:
            rows.append({
                "model": m, "role": role[m], "ref": m in refs, "parent": ROBUST_PARENT.get(m),
                "r2": clean(sc.get(m, h, POOLED, hp, BENCH, "r2_oos")),
                "r2_ctx": clean(sc.get(m, h, POOLED, cp, BENCH, "r2_oos")) if cp else None,
                "cw_p": clean(sc.get(m, h, POOLED, hp, BENCH, "cw_p")),
                "dm_p": clean(sc.get(m, h, POOLED, hp, BENCH, "dm_mse_p1")),
                "in_mcs": clean(sc.get(m, h, POOLED, mcs_period, "none", "in_mcs_mse")),
                "mcs_p": clean(sc.get(m, h, POOLED, mcs_period, "none", "mcs_p_mse")),
            })
        out["pooled"][h] = rows
        # share of the benchmark's squared error each asset carries in the pooled R² (display reduction)
        sse = {a: sc.get(BENCH, h, a, hp, "none", "mse") * sc.get(BENCH, h, a, hp, "none", "n_obs") for a in ASSETS}
        tot = sum(v for v in sse.values() if np.isfinite(v))
        out["weights"][h] = {a: clean(v / tot) if tot > 0 and np.isfinite(v) else None for a, v in sse.items()}
        grid = {"assets": ASSETS, "detectable": {}, "cells": {}}
        for a in ASSETS:
            grid["detectable"][a] = clean(sc.get(BENCH, h, a, hp, "none", "r2_detectable"))
        for m in models + robust:
            grid["cells"][m] = {a: {"r2": clean(sc.get(m, h, a, hp, BENCH, "r2_oos")),
                                    "cw_p": clean(sc.get(m, h, a, hp, BENCH, "cw_p")),
                                    "q": clean(sc.get(m, h, a, q_period, BENCH, "cw_q_bh")),
                                    "q_all": clean(sc.get(m, h, a, q_period, BENCH, "cw_q_by_all"))}
                                for a in ASSETS}
        out["grid"][h] = grid
        out["sub"][h] = {m: {p: clean(sc.get(m, h, POOLED, p, BENCH, "r2_oos")) for p in P["order"]}
                         for m in models + robust + refs}
        out.setdefault("sub_detectable", {})[h] = {p: clean(sc.get(BENCH, h, POOLED, p, "none", "r2_detectable"))
                                                   for p in P["order"]}
        if cum is not None:
            ch = cum[(cum.horizon == h) & cum.model.isin(models + robust)]
            out["cum"][h] = {m: {a: series_points(g.date, g.cum_dsse, 4) for a, g in gm.groupby("asset")}
                             for m, gm in ch.groupby("model")}
    return out


# --------------------------------------------------------------------------- distribution

def dist_section(sc: Scores, P: dict) -> dict | None:
    family = [m for m in config.FROZEN_DIST if sc.has(m)]
    if not family:
        return None
    extra = [m for m in (config.COMPOSITE,) if sc.has(m)]
    models = family + extra
    refs = [m for m in ("vol_ewma_fhs", "hist_quantile", VOL_BENCH, BENCH) if sc.has(m)]
    hp, cp = P["headline"], P["context"]
    q_period = _adj_period(sc, hp, "dm_crps_q_bh")
    spa_period = _adj_period(sc, hp, "spa_p_crps", "*dist")
    mcs_period = _adj_period(sc, hp, "in_mcs_crps")
    out = {"models": models, "family": family, "extra": extra, "refs": refs, "q_period": q_period,
           "spa_period": spa_period, "mcs_period": mcs_period,
           "pooled": {}, "grid": {}, "info": {}, "pinball": {}, "tails": {}}
    for h in HORIZONS:
        sv = sc.star("spa_p_crps", h, POOLED, VOL_BENCH, spa_period, family, label="*dist")
        sh = sc.star("spa_p_crps", h, POOLED, BENCH, spa_period, family, label="*dist")
        sa = sc.star("spa_p_crps", h, POOLED, VOL_BENCH, spa_period, family, label="*")
        out["info"][h] = {
            "spa_vol": clean(sv[0]), "spa_hist": clean(sh[0]), "spa_run": sv[1],
            "spa_n": len(family) if sv[1] else None,
            "spa_all_vol": clean(sa[0]), "spa_all_n": sc.run_size(sa[1], VOL_BENCH),
            "n_indep": clean(sc.get(BENCH, h, POOLED, hp, "none", "n_indep")),
        }
        rows = []
        for m in models + refs:
            r = {"model": m, "ref": m in refs, "role": "family" if m in family else "composite" if m in extra else "ref",
                 "skill_vol": clean(sc.get(m, h, POOLED, hp, VOL_BENCH, "crps_skill")),
                 "skill_vol_ctx": clean(sc.get(m, h, POOLED, cp, VOL_BENCH, "crps_skill")) if cp else None,
                 "skill_hist": clean(sc.get(m, h, POOLED, hp, BENCH, "crps_skill")),
                 "p_vol": clean(sc.get(m, h, POOLED, hp, VOL_BENCH, "dm_crps_p")),
                 "p_hist": clean(sc.get(m, h, POOLED, hp, BENCH, "dm_crps_p")),
                 "in_mcs": clean(sc.get(m, h, POOLED, mcs_period, "none", "in_mcs_crps")),
                 "calib_p": clean(sc.get(m, h, POOLED, hp, "none", "calib_p")),
                 "mean_sd": clean(sc.get(m, h, POOLED, hp, "none", "mean_sd"))}
            for lv in COVER_LEVELS:
                r[f"cover_{lv}"] = clean(sc.get(m, h, POOLED, hp, "none", f"cover_{lv}"))
            rows.append(r)
            out["tails"].setdefault(h, []).append(
                {"model": m, "ref": m in refs,
                 **{f"hit_{q}": clean(sc.get(m, h, POOLED, hp, "none", f"hit_{q}")) for q in TAIL_Q},
                 **{f"kupiec_{q}": clean(sc.get(m, h, POOLED, hp, "none", f"kupiec_p_{q}")) for q in TAIL_Q}})
            out["pinball"].setdefault(h, {})[m] = {
                "vol": [clean(sc.get(m, h, POOLED, hp, VOL_BENCH, f"pinball_skill_{q}")) for q in PINBALL_Q],
                "hist": [clean(sc.get(m, h, POOLED, hp, BENCH, f"pinball_skill_{q}")) for q in PINBALL_Q]}
        out["pooled"][h] = rows
        out["grid"][h] = {m: {a: {"skill": clean(sc.get(m, h, a, hp, VOL_BENCH, "crps_skill")),
                                  "p": clean(sc.get(m, h, a, hp, VOL_BENCH, "dm_crps_p")),
                                  "q": clean(sc.get(m, h, a, q_period, VOL_BENCH, "dm_crps_q_bh"))}
                              for a in ASSETS} for m in models}
    out["quantiles"] = [int(q[1:]) for q in PINBALL_Q]
    return out


# --------------------------------------------------------------------------- covariance

def cov_section(cs: pd.DataFrame | None, per_date: pd.DataFrame | None, P: dict, meta: dict | None = None) -> dict | None:
    if cs is None or cs.empty:
        return None
    have = set(cs.period)
    hp = "holdout" if "holdout" in have else "full"
    cp = "pre_holdout" if "pre_holdout" in have else None
    star_rows = cs.model.astype(str).str.startswith("*")
    present = [m for m in pd.unique(cs.model) if not str(m).startswith("*")]
    models = [m for m in config.FROZEN_COV if m in present] + sorted(m for m in present if "__sd_" in m)
    benches = [b for b in (COV_BENCH, "lw") if b in set(cs.benchmark)]
    static = "lw" if "lw" in benches else COV_BENCH   # the benchmark the verdict rests on
    ix = cs[~star_rows].set_index(["model", "horizon", "asset", "period", "benchmark", "metric"])["value"]
    ix = ix[~ix.index.duplicated()]

    def g(m, h, a, p, b, k):
        try:
            return clean(ix.loc[(m, h, a, p, b, k)])
        except KeyError:
            return None

    def star(h, a, p, k, b=COV_BENCH):
        s = cs[star_rows & (cs.horizon == h) & (cs.asset == a) & (cs.period == p) & (cs.metric == k)]
        if b is not None:
            s = s[s.benchmark == b]
        return clean(s.value.iloc[0]) if len(s) else None

    def mt(metric):
        return hp if not cs[(cs.metric == metric) & (cs.period == hp)].empty else "full"

    q_period, spa_period, mcs_period = mt("dm_q_bh_nll"), mt("spa_p_nll"), mt("in_mcs_nll")
    periods_order = [p for p in ("holdout", "pre_holdout", *DECADES, "full") if p in have]
    fam = (meta or {}).get("spa_family", {}).get("lw") or [m for m in config.FROZEN_COV if m != "lw"]
    out = {"models": models, "headline": hp, "context": cp, "q_period": q_period, "spa_period": spa_period,
           "mcs_period": mcs_period, "benches": benches, "static": static, "lw_family": fam,
           "periods": periods_order, "joint": {}, "sb": {}, "ports": {}, "info": {}, "sb_series": {}}
    for h in HORIZONS:
        out["info"][h] = {"spa": {b: {"joint": star(h, "joint", spa_period, "spa_p_nll", b),
                                      "sb": star(h, "sb", spa_period, "spa_p_nll", b)} for b in benches},
                          "n_indep": star(h, "joint", hp, "n_indep", "none"),
                          "n_assets": star(h, "joint", hp, "n_assets_mean", "none")}
        rows = []
        for m in [*models, *[b for b in benches if b not in models]]:
            r = {"model": m, "ref": m == COV_BENCH, "in_family": m in fam,
                 "nll": g(m, h, "joint", hp, "none", "nll"),
                 "in_mcs": g(m, h, "joint", mcs_period, "none", "in_mcs_nll"),
                 "mahal": g(m, h, "joint", hp, "none", "mahal"),
                 "cover90": g(m, h, "joint", hp, "none", "cover90"),
                 "cond": g(m, h, "joint", hp, "none", "cond_median"), "vs": {}}
            for b in benches:
                if b == m:
                    continue
                r["vs"][b] = {"diff": g(m, h, "joint", hp, b, "diff_nll"),
                              "diff_ctx": g(m, h, "joint", cp, b, "diff_nll") if cp else None,
                              "p": g(m, h, "joint", hp, b, "dm_p_nll"),
                              "q": g(m, h, "joint", q_period, b, "dm_q_bh_nll"),
                              "strand": g(m, h, "joint", hp, b, "strand_p_median_nll")}
            rows.append(r)
        out["joint"][h] = rows
        out["sb"][h] = {
            "real": {p: star(h, "sb", p, "corr_real", None) for p in periods_order},
            "rows": [{"model": m, "ref": m in benches,
                      "pred": {p: g(m, h, "sb", p, "none", "corr_pred") for p in periods_order},
                      "vs": {b: {"diff": {p: g(m, h, "sb", p, b, "diff_nll") for p in periods_order},
                                 "p": {p: g(m, h, "sb", p, b, "dm_p_nll") for p in periods_order},
                                 "q": g(m, h, "sb", q_period, b, "dm_q_bh_nll")}
                             for b in benches if b != m}}
                     for m in [*models, *[b for b in benches if b not in models]]]}
        prow = []
        for m in models:
            if m == static:
                continue
            r = {"model": m, "bench": static}
            for a, k in (("port_ew", "qlike"), ("port_6040", "qlike"), ("port_iv", "qlike"), ("mvp", "mvp_sq")):
                r[a] = {"diff": g(m, h, a, hp, static, f"diff_{k}"), "p": g(m, h, a, hp, static, f"dm_p_{k}")}
            r["mvp_real"] = g(m, h, "mvp", hp, "none", "realized_vol")
            r["mvp_pred"] = g(m, h, "mvp", hp, "none", "pred_vol")
            prow.append(r)
        out["ports"][h] = prow
    if per_date is not None and not per_date.empty:
        pdd = per_date[(per_date.target == "sb")]
        for h in HORIZONS:
            ph = pdd[pdd.horizon == h]
            ser = {}
            for m in [*config.FROZEN_COV, COV_BENCH]:
                s = ph[(ph.model == m) & (ph.metric == "corr_pred")].sort_values("date")
                if len(s):
                    ser[m] = series_points(s.date, s.value, 3)
            e = ph[ph.metric.isin(["e_stock", "e_bond"])].pivot_table(index="date", columns="metric", values="value")
            real = None
            if {"e_stock", "e_bond"} <= set(e.columns) and h == 1:
                # display-only reference line: trailing 24-month correlation of the realized
                # 1-month errors (the scored corr_real is per subperiod and is in the table)
                rc = e["e_stock"].rolling(24, min_periods=24).corr(e["e_bond"])
                real = series_points(rc.index, rc.values, 3)
            out["sb_series"][h] = {"models": ser, "realized_24m": real}
    return out


# --------------------------------------------------------------------------- now

def _last_return_date() -> pd.Timestamp | None:
    try:
        return schema.load_asset_returns().index.max()
    except Exception:  # processed data absent (e.g. in a test fixture)
        return None


def now_section(root, sc: Scores | None) -> dict:
    mean_m, dist_m = gl.composite_parts()
    last_ret = _last_return_date()
    out = {"composite": config.COMPOSITE, "mean_model": mean_m, "dist_model": dist_m, "h": {},
           "primary_cov": config.PRIMARY_COV, "cov": {}}
    for h in HORIZONS:
        comp = load_forecast_last(root, config.COMPOSITE, h)
        source, note = config.COMPOSITE, None
        if comp is None:
            comp = load_forecast_last(root, dist_m, h)
            source = dist_m
            note = (f"The composite forecast ({config.COMPOSITE}) is not in this results folder. Shown instead: "
                    f"the composite's distribution model ({dist_m}), which is centred on the historical mean, "
                    f"with the {mean_m} mean forecast marked separately.")
        if comp is None:
            out["h"][h] = None
            continue
        origin = pd.Timestamp(comp["date"].iloc[0])
        hm = _forecast_at(root, BENCH, h, origin)
        mm = _forecast_at(root, mean_m, h, origin) if source != config.COMPOSITE else None
        realized_by = origin + pd.offsets.MonthEnd(h)
        live = None if last_ret is None else bool(realized_by > last_ret)
        rows = []
        for a in ASSETS:
            r = comp[comp.asset == a]
            if r.empty:
                continue
            r = r.iloc[0]
            row = {"asset": a, "mean": clean(r["mean"]), "sd": clean(r["sd"]),
                   **{q: clean(r[q]) for q in schema.QUANTILE_COLUMNS}}
            row["hist_mean"] = clean(hm.loc[a, "mean"]) if hm is not None and a in hm.index else None
            row["model_mean"] = clean(mm.loc[a, "mean"]) if mm is not None and a in mm.index else None
            rows.append(row)
        out["h"][h] = {"origin": origin.strftime("%Y-%m"), "target_end": realized_by.strftime("%Y-%m"),
                       "live": live, "source": source, "note": note, "rows": rows}
        cv = load_cov_last(root, config.PRIMARY_COV, h)
        if cv is not None:
            names = [a for a in ASSETS if a in set(cv.asset_i) | set(cv.asset_j)]
            M = pd.DataFrame(np.nan, index=names, columns=names)
            for _, r in cv.iterrows():
                M.loc[r.asset_i, r.asset_j] = r["cov"]
                M.loc[r.asset_j, r.asset_i] = r["cov"]
            sd = np.sqrt(np.diag(M.to_numpy(float)))
            C = M.to_numpy(float) / np.outer(sd, sd)
            out["cov"][h] = {"origin": pd.Timestamp(cv.date.iloc[0]).strftime("%Y-%m"), "assets": names,
                             "corr": [[clean(round(x, 3)) for x in row] for row in C],
                             "vol_ann": [clean(v * np.sqrt(12.0 / h)) for v in sd]}
    return out


def _forecast_at(root, model, h, origin) -> pd.DataFrame | None:
    from ..evaluation.walkforward import forecast_path
    p = forecast_path(model, h, root)
    if not p.exists():
        return None
    f = pd.read_parquet(p, columns=["date", "asset", "mean", "sd"])
    f = f[f.date == origin]
    return f.set_index("asset") if len(f) else None


# --------------------------------------------------------------------------- strategy

STRAT_KEEP = ["d_sharpe", "d_sharpe_p", "d_sharpe_q_bh", "d_cer", "d_cer_p", "d_cer_ci_lo", "d_cer_ci_hi",
              "d_cer_q_bh", "d_cer_boot_p", "breakeven_cost_bp", "d_turnover_ann", "n_eff"]
LEVEL_KEEP = ["sharpe", "cer", "ann_mean", "ann_vol", "max_drawdown", "turnover_ann", "avg_leverage", "n_months"]


def strategy_section(st: dict | None) -> dict | None:
    if st is None:
        return None
    m = st["metrics"]
    meta = st.get("meta", {})
    bh_family = list(meta.get("bh_family") or config.FROZEN_MEAN)
    have = set(m.period)
    hp = "holdout" if "holdout" in have else "full"
    net = m[(m.period == hp) & (m.basis == "net")]
    comp = net[net.benchmark != "none"]
    keys = ["family", "horizon", "universe", "rule", "mean_model", "cov_model", "benchmark"]
    t = comp[comp.metric.isin(STRAT_KEEP)].pivot_table(index=keys, columns="metric", values="value",
                                                       observed=True).reset_index()
    rows = []
    for _, r in t.iterrows():
        d = {k: (int(r[k]) if k == "horizon" else r[k]) for k in keys}
        d.update({k: clean(r.get(k)) for k in STRAT_KEEP})
        mm = d["mean_model"]
        d["role"] = ("family" if mm in bh_family else "composite" if mm == config.COMPOSITE
                     else "robust" if mm in ROBUST_PARENT else "bench" if mm in ("hist_mean", "none") else "other")
        rows.append(d)
    lv = net[(net.benchmark == "none") & net.metric.isin(LEVEL_KEEP)].pivot_table(
        index=["strategy", "family", "horizon", "universe", "rule", "mean_model", "cov_model"],
        columns="metric", values="value", observed=True).reset_index()
    levels = []
    for _, r in lv.iterrows():
        d = {k: (int(r[k]) if k == "horizon" else r[k]) for k in
             ["strategy", "family", "horizon", "universe", "rule", "mean_model", "cov_model"]}
        d.update({k: clean(r.get(k)) for k in LEVEL_KEEP})
        levels.append(d)
    series = {}
    R = st.get("returns")
    if R is not None:
        sel = R[(R.universe == "all") & R.family.isin(["mean", "static"])]
        for s, g in sel.groupby("strategy"):
            g = g.sort_values("date")
            series[s] = series_points(g.date, g.net, 4)
    return {"period": hp, "bh_family": bh_family, "period_rule": meta.get("period_rule"), "has_holdout_period": "holdout" in have, "rows": rows, "levels": levels,
            "series": series,
            "meta": {k: meta.get(k) for k in ("cov_model", "rule_settings", "cost_bp_one_way",
                                              "financing_spread_annual", "n_strategies", "n_comparisons",
                                              "boot_reps")},
            "diag": meta.get("diagnostics", {})}


# --------------------------------------------------------------------------- method and data

def method_section(root, metas: dict, cov_meta: dict, dev: bool) -> dict:
    out = {"trials": {}, "runs": {}, "data": {}}
    for k in ("mean", "dist", "cov"):
        t = _csv(root / f"trials_{k}.csv")
        if t is None and not dev:
            t = _csv(root / "dev" / f"trials_{k}.csv")
        if t is None:
            out["trials"][k] = None
            continue
        if k == "mean":
            rows = [{"name": r["name"], "desc": r.get("description"), "status": r.get("status")}
                    for _, r in t.iterrows()]
        elif k == "dist":
            rows = [{"name": r["name"], "desc": r.get("description"), "status": r.get("run")}
                    for _, r in t.iterrows()]
        else:
            rows = [{"name": r["model"], "desc": r.get("kind"), "status": r.get("run_name")}
                    for _, r in t.iterrows()]
        out["trials"][k] = {"n": int(len(t)), "rows": [{kk: (None if pd.isna(v) else v) for kk, v in r.items()}
                                                        for r in rows]}
    out["runs"] = {r: {"models": m.get("models"), "data": m.get("data"), "dev": m.get("dev"),
                       "reps": m.get("reps")} for r, m in metas.items()}
    if cov_meta:
        out["runs"]["(covariance) " + str(cov_meta.get("run", ""))] = {
            "models": cov_meta.get("models"), "data": cov_meta.get("data"), "dev": cov_meta.get("dev"),
            "reps": cov_meta.get("reps")}
    am = _csv(schema.ASSET_META)
    if am is not None:
        assets = []
        for a in ASSETS:
            g = am[am.asset == a]
            if g.empty:
                continue
            assets.append({"asset": a, "start": str(g.segment_start.min())[:7], "end": str(g.segment_end.max())[:7],
                           "segments": [{"start": str(r.segment_start)[:7], "end": str(r.segment_end)[:7],
                                         "source": r.source, "desc": r.description} for _, r in g.iterrows()]})
        out["data"]["assets"] = assets
        cash = am[am.asset == "cash"]
        out["data"]["cash"] = [{"start": str(r.segment_start)[:7], "end": str(r.segment_end)[:7],
                                "source": r.source} for _, r in cash.iterrows()]
    av = _csv(schema.ASSET_META.parent / "asset_validation.csv")
    if av is not None:
        etf = av[av.check == "etf"]
        out["data"]["validation"] = [{"asset": r.asset, "etf": r.b, "start": str(r.start)[:7], "n": int(r.n),
                                      "corr": clean(r["corr"]), "vol_ratio": clean(r["vol_ratio"])}
                                     for _, r in etf.iterrows()]
    pm = _csv(schema.PREDICTOR_META)
    if pm is not None:
        out["data"]["predictors"] = {"n": int(len(pm)),
                                     "by_family": {k: int(v) for k, v in pm.family.value_counts().items()},
                                     "hindsight": int(pm.hindsight.astype(str).str.lower().eq("true").sum())}
    fm = _csv(schema.FEATURE_META)
    if fm is not None:
        out["data"]["features"] = {"n": int(len(fm)),
                                   "by_family": {k: int(v) for k, v in fm.family.value_counts().items()}}
    fps = {m.get("data") for m in metas.values() if m.get("data")}
    out["data_fingerprints_differ"] = len(fps) > 1
    return out


def _csv(p):
    try:
        return pd.read_csv(p) if p.exists() else None
    except Exception:
        return None
