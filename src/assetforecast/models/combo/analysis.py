"""Evaluation of the combo family: every table in reports/models/combo.md.

    export ASSETFORECAST_MAX_DATE=2014-12-31 OMP_NUM_THREADS=1
    python -m assetforecast.models.combo.analysis --forecast --out combo_tables.md

``--forecast`` first re-runs every registered combo through the harness from
scratch (``force=True``; vol, then corr, then cov) into the results directory
in use, so the whole family is reproduced by this one command once its
components are stored. Unregistered variants (everything tried and dropped)
are computed in memory with the same classes, walked forward over the same
origins, and never stored. Nothing here feeds back into a model: the
registered models were fixed before these tables existed.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np
import pandas as pd

from ...data.panel import load_panel
from ...eval import inference as inf
from ...eval.harness import load_forecasts, load_forecasts_long, origins_for, run_model
from ...eval.losses import corr_losses, cov_losses, vol_losses
from ...eval.portfolio import RealizedCache, score_portfolios, summarize
from ...eval.targets import get_targets
from ...paths import RESULTS
from ...universe import ASSET_IDS
from ..base import get_model, list_models
from . import CORR_MODELS, COV_PAIRS, FASTSLOW, VOL_MODELS
from .corr import CorrCombo
from .store import family_of, load
from .vol import VolCombo

ERAS = {"1926-45": ("1926-01-01", "1945-12-31"), "1946-79": ("1946-01-01", "1979-12-31"),
        "1980-99": ("1980-01-01", "1999-12-31"), "2000-13": ("2000-01-01", "2013-12-31")}
LEADERS = ["statespace_2c_daily", "ml_glm", "har_pooled_logcvx_d", "har_pooled_qlike_d", "garch_d_rm2006",
           "bench_vol_mr_est", "bench_vol_m_hl12", "regime_bocpd", "macro_base_h"]
REFS = ["ref_last12m", "ref_blend"]
CORR_SINGLES = ["ref_corr_last120m", "bench_corr_d_hl36", "bench_corr_shrink_block", "mcorr_nls", "mcorr_dcc",
                "mcorr_shrink_block", "condcorr_msw", "condcorr_msw_infl"]
warnings.filterwarnings("ignore", category=inf.InferenceWarning)


# ------------------------------------------------------------------ helpers
def md(df: pd.DataFrame, fmt="{:.3f}") -> str:
    def f(x):
        if isinstance(x, (float, np.floating)):
            return "" if not np.isfinite(x) else fmt.format(x)
        return str(x)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [f(x) for x in r.to_numpy()]) + " |")
    return "\n".join(lines)


def walk(models: dict, kind: str, panel) -> dict[str, pd.DataFrame]:
    """Walk every model forward over the harness origins (one view per origin); long frames."""
    rows = {n: [] for n in models}
    for t in origins_for(panel):
        view = panel.asof(t)
        a = view.eligible()
        if len(a) < 2:
            continue
        for n, m in models.items():
            if kind == "vol":
                f = m.forecast_vol(view, a)
                rows[n].append(pd.DataFrame({"origin": t, "asset": a, "vol": f.to_numpy()}))
            else:
                c = m.forecast_corr(view, a).to_numpy()
                rows[n].append(pd.DataFrame({"origin": t, "asset_i": np.repeat(a, len(a)),
                                             "asset_j": np.tile(a, len(a)), "corr": c.ravel()}))
    out = {}
    for n, r in rows.items():
        df = pd.concat(r, ignore_index=True)
        df["origin"] = df["origin"].astype("datetime64[ns]")
        out[n] = df
    return out


def cov_dict(vol_long: pd.DataFrame, corr: dict) -> dict:
    """{origin: cov DataFrame} = diag(v) C diag(v) from a vol long frame and a corr dict."""
    out = {}
    for o, g in vol_long.groupby("origin", sort=True):
        o = pd.Timestamp(o)
        C = corr[o]
        v = g.set_index("asset")["vol"].reindex(C.index).to_numpy()
        out[o] = pd.DataFrame(C.to_numpy() * np.outer(v, v), index=C.index, columns=C.columns)
    return out


def cov_long(d: dict) -> pd.DataFrame:
    parts = []
    for o, H in d.items():
        a = list(H.index)
        parts.append(pd.DataFrame({"origin": o, "asset_i": np.repeat(a, len(a)), "asset_j": np.tile(a, len(a)),
                                   "cov": H.to_numpy().ravel()}))
    df = pd.concat(parts, ignore_index=True)
    df["origin"] = df["origin"].astype("datetime64[ns]")
    return df


def era_of(idx: pd.DatetimeIndex) -> np.ndarray:
    lab = np.full(len(idx), "", dtype=object)
    for e, (a, b) in ERAS.items():
        lab[(idx >= a) & (idx <= b)] = e
    return lab


def skill_row(lm: pd.Series, lb: pd.Series) -> dict:
    d = pd.concat([lm, lb], axis=1).dropna()
    s = inf.skill_score(d.iloc[:, 0], d.iloc[:, 1], warn=False)
    t = inf.dm_test(d.iloc[:, 0], d.iloc[:, 1], warn=False)
    return {"skill%": s.skill, "ci_lo": s.ci_low, "ci_hi": s.ci_high, "dm_p": t.pvalue}


# -------------------------------------------------------------- vol section
def vol_variants(panel) -> dict:
    singles = [m for m in list_models("vol") if family_of(m) != "combo"]
    v = {
        "x_median": VolCombo(group="family", agg="median"),
        "x_invq": VolCombo(group="family", agg="invq"),
        "x_best_family": VolCombo(group="family", agg="best"),
        "x_best_model": VolCombo(group="none", agg="best"),
        "x_stack_models": VolCombo(group="none", agg="stack"),
        "x_ew_cal": VolCombo(group="none", calibrate=True),
        "x_hier_calgroup": VolCombo(group="family", calibrate="group"),
        "x_cal_m_hl12_group": VolCombo(["bench_vol_m_hl12"], group="none", calibrate="group"),
    }
    for m in singles:
        v[f"cal:{m}"] = VolCombo([m], group="none", calibrate=True)
    for n, m in v.items():
        m.name = n
    return v


def logspace_hier(origins_long: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic: hierarchical mean in LOG variance (geometric), from stored forecasts."""
    singles = [m for m in list_models("vol") if family_of(m) != "combo"]
    fams: dict[str, list] = {}
    for m in singles:
        fams.setdefault(family_of(m), []).append(m)
    base = load("vol", singles[0])
    acc = []
    for f, mem in sorted(fams.items()):
        acc.append(np.mean([np.log(load("vol", m).on_grid(base.origins)) for m in mem], axis=0))
    G = np.exp(np.mean(acc, axis=0))                       # O x N variance
    o = origins_long[["origin", "asset"]].copy()
    oi = base.origins.get_indexer(o["origin"])
    ai = o["asset"].map({a: i for i, a in enumerate(ASSET_IDS)}).to_numpy(int)
    o["vol"] = np.sqrt(G[oi, ai])
    return o


def vol_tables(T, fc: dict, out: list, panel):
    A, P = [], []
    for n, df in fc.items():
        r = vol_losses(df, T, n)
        A.append(r["asset"])
        P.append(r["pooled"])
    A = pd.concat(A, ignore_index=True)
    P = pd.concat(P, ignore_index=True)
    A["ratio"] = (A["vol_r"] / A["vol_f"]) ** 2
    reg = list(VOL_MODELS)
    var = [n for n in fc if n.startswith("x_")]
    show = reg + var + LEADERS + REFS
    res = {}
    for subset, ref in (("daily", "ref_last12m"), ("all", "ref_blend")):
        p = P[P["subset"] == subset]
        L = p.pivot(index="origin", columns="model", values="qlike")
        L = L[[c for c in fc]].dropna()
        a = A[A["origin"].isin(L.index)]
        if subset == "daily":
            a = a[~a["proxy"]]
        rf = a.groupby(["model", "origin"])["ratio"].mean().groupby("model").mean()
        best = "statespace_2c_daily"
        rows = {}
        for m in show:
            d = {"qlike": L[m].mean(), "R/F": rf[m]}
            for k, v in skill_row(L[m], L[ref]).items():
                d[f"{k} vs {ref}"] = v
            if m != best:
                s = skill_row(L[m], L[best])
                d[f"skill% vs {best}"], d["ci_lo2"], d["ci_hi2"], d["dm_p2"] = s["skill%"], s["ci_lo"], s["ci_hi"], s["dm_p"]
            s = skill_row(L[m], L["x_best_model"])
            d["skill% vs past-best"], d["dm_p3"] = s["skill%"], s["dm_p"]
            rows[m] = d
        tab = pd.DataFrame(rows).T
        tab.index.name = "model"
        out.append(f"### Vol, subset `{subset}`: {len(L)} common origins {L.index.min():%Y-%m}..{L.index.max():%Y-%m}"
                   f" (skill = % lower mean QLIKE, 90% Fieller-EWC CI; dm_p = DM-EWC two-sided; "
                   f"R/F = mean realized/forecast variance; past-best = `x_best_model`)\n\n"
                   + md(tab.sort_values("qlike")))
        # eras
        lab = era_of(L.index)
        e = pd.DataFrame({er: L[lab == er][show].mean() for er in ERAS})
        e.loc["n_origins"] = [int((lab == er).sum()) for er in ERAS]
        e["26-45 / ref_last12m"] = e["1926-45"] / e.loc["ref_last12m", "1926-45"]
        e.loc["n_origins", "26-45 / ref_last12m"] = np.nan
        e.index.name = "model"
        out.append(f"### Vol QLIKE by era, subset `{subset}`\n\n" + md(e))
        # by asset (common cells)
        a2 = a[a["model"].isin(show)]
        cnt = a2.groupby(["origin", "asset"])["model"].nunique()
        keep = cnt[cnt == len(show)].index
        a2 = a2.set_index(["origin", "asset"]).loc[lambda x: x.index.isin(keep)].reset_index()
        ba = a2.pivot_table(index="model", columns="asset", values="qlike", aggfunc="mean")
        ba = ba.reindex(index=show, columns=[c for c in ASSET_IDS if c in ba.columns])
        ba.index.name = "model"
        out.append(f"### Vol QLIKE by asset, subset `{subset}` (common cells)\n\n" + md(ba))
        # era-level ratio R/F
        rfe = a.assign(era=era_of(pd.DatetimeIndex(a["origin"]))).groupby(["model", "era", "origin"])["ratio"].mean() \
            .groupby(["model", "era"]).mean().unstack()
        rfe = rfe.reindex(index=show, columns=list(ERAS))
        rfe.index.name = "model"
        out.append(f"### Mean realized/forecast variance by era, subset `{subset}`\n\n" + md(rfe))
        res[subset] = (L, a)
    return res, P, A


def calibration_tables(res: dict, fc: dict, out: list):
    singles = sorted(n[4:] for n in fc if n.startswith("cal:"))
    pairs = [(m, f"cal:{m}") for m in singles] + [("combo_vol_hier", "combo_vol_hier_cal"),
                                                  ("combo_vol_stack", "combo_vol_stack_cal"),
                                                  ("combo_vol_ew", "x_ew_cal"),
                                                  ("combo_vol_hier", "x_hier_calgroup"),
                                                  ("bench_vol_m_hl12", "x_cal_m_hl12_group")]
    for subset in ("daily", "all"):
        L, a = res[subset]
        rf = a.groupby(["model", "origin"])["ratio"].mean().groupby("model").mean()
        rows = {}
        lab = era_of(L.index)
        for raw, cal in pairs:
            s = skill_row(L[cal], L[raw])
            d = {"raw": L[raw].mean(), "calibrated": L[cal].mean(), "skill%": s["skill%"], "ci_lo": s["ci_lo"],
                 "ci_hi": s["ci_hi"], "dm_p": s["dm_p"], "R/F raw": rf[raw], "R/F cal": rf[cal]}
            for er in ERAS:
                m = lab == er
                d[f"d {er}"] = L[cal][m].mean() - L[raw][m].mean()
            rows[f"{cal}" if not cal.startswith("cal:") else raw] = d
        tab = pd.DataFrame(rows).T.sort_values("calibrated")
        tab.index.name = "model (calibrated vs raw)"
        n_imp = int((tab["skill%"] > 0).sum())
        n_sig = int(((tab["skill%"] > 0) & (tab["dm_p"] < 0.10)).sum())
        n_sigw = int(((tab["skill%"] < 0) & (tab["dm_p"] < 0.10)).sum())
        out.append(f"### Level calibration, subset `{subset}`: calibrated vs raw (skill of the calibrated "
                   f"version; d <era> = change in era mean QLIKE, negative = calibration helped). "
                   f"Improves {n_imp} of {len(tab)}; significantly better (p<0.10) {n_sig}, "
                   f"significantly worse {n_sigw}\n\n" + md(tab))


def learned_vs_equal(res: dict, out: list):
    comps = [("combo_vol_stack", "combo_vol_hier"), ("x_invq", "combo_vol_hier"),
             ("x_best_family", "combo_vol_hier"), ("x_stack_models", "combo_vol_ew"),
             ("x_best_model", "combo_vol_ew"), ("combo_vol_stack_cal", "combo_vol_hier_cal"),
             ("combo_vol_hier", "combo_vol_ew"), ("combo_vol_hier_trim", "combo_vol_hier"),
             ("x_median", "combo_vol_hier"), ("x_logspace_hier", "combo_vol_hier")]
    for subset in ("daily", "all"):
        L, _ = res[subset]
        lab = era_of(L.index)
        rows = {}
        for a, b in comps:
            d = skill_row(L[a], L[b])
            for er in ERAS:
                m = lab == er
                d[f"skill% {er}"] = 100 * (1 - L[a][m].mean() / L[b][m].mean())
            rows[f"{a} vs {b}"] = d
        tab = pd.DataFrame(rows).T
        tab.index.name = "comparison"
        out.append(f"### Learned vs equal weights and aggregation choices, subset `{subset}`\n\n" + md(tab))


def mcs_table(res: dict, out: list):
    for subset in ("daily", "all"):
        L, _ = res[subset]
        cols = list(VOL_MODELS) + LEADERS + REFS + ["x_best_model"]
        r = inf.model_confidence_set(L[cols], alpha=0.10, seed=0, warn=False)
        s = r.summary()
        s.index.name = "model"
        out.append(f"### Model confidence set (alpha 0.10, R statistic, block bootstrap), subset `{subset}`: "
                   f"{len(r.included)} of {len(cols)} models retained\n\n" + md(s[["mean_loss", "mcs_pvalue",
                                                                                   "in_mcs"]]))


def weight_tables(panel, out: list):
    years = [1929, 1935, 1945, 1955, 1965, 1975, 1985, 1995, 2005, 2014]
    st = get_model("vol", "combo_vol_stack")
    rows = {}
    for y in years:
        E = pd.Timestamp(f"{y}-12-31")
        e = st.estimate(panel.asof(E), E)
        rows[y] = dict(zip(e.groups, e.weights)) | {"lam": e.lam, "years": e.n_years}
    w = pd.DataFrame(rows).T
    w.index.name = "December"
    out.append("### `combo_vol_stack` family weights at selected estimation dates (after shrinkage; "
               "equal weight = 1/9 = 0.111)\n\n" + md(w))
    # full path stability: mean absolute yearly change in weights
    allE = [pd.Timestamp(f"{y}-12-31") for y in range(1929, 2015)]
    W = np.array([st.estimate(panel.asof(E), E).weights for E in allE])
    raw = []
    for E, wv in zip(allE, W):
        e = st.estimate(panel.asof(E), E)
        raw.append((wv - (1 - e.lam) / len(wv)) / max(e.lam, 1e-12))
    raw = np.array(raw)
    ch = np.abs(np.diff(W, axis=0)).sum(axis=1) / 2
    chr_ = np.abs(np.diff(raw, axis=0)).sum(axis=1) / 2
    out.append(f"Yearly weight turnover (half the sum of absolute changes), 1930-2014: shrunk weights mean "
               f"{ch.mean():.3f} (max {ch.max():.3f}); unshrunk optimal weights w* mean {chr_.mean():.3f} "
               f"(max {chr_.max():.3f}). Unshrunk w* at 2014-12: "
               + ", ".join(f"{g} {x:.2f}" for g, x in zip(st.estimate(panel.asof(allE[-1]), allE[-1]).groups,
                                                         raw[-1])))
    # calibration constants
    rows = {}
    cals = {n: get_model("vol", n) for n in ["combo_vol_hier_cal", "combo_vol_stack_cal",
                                             "combo_vol_cal_m_hl12", "combo_vol_cal_rm2006"]}
    for y in years:
        E = pd.Timestamp(f"{y}-12-31")
        rows[y] = {}
        for n, m in cals.items():
            e = m.estimate(panel.asof(E), E)
            rows[y][f"{n} c"] = e.scale
            rows[y][f"{n} c_hat"] = e.scale_raw
    c = pd.DataFrame(rows).T
    c.index.name = "December"
    out.append("### Calibration constants `c` (applied) and `c_hat` (raw mean realized/forecast on past "
               "cells)\n\n" + md(c))
    g = VolCombo(["bench_vol_m_hl12"], group="none", calibrate="group")
    rows = {y: dict(g.estimate(panel.asof(pd.Timestamp(f"{y}-12-31")), pd.Timestamp(f"{y}-12-31")).scale_by_group)
            for y in years}
    gt = pd.DataFrame(rows).T
    gt.index.name = "December"
    out.append("### Per-asset-group calibration constants for `bench_vol_m_hl12` (variant "
               "`x_cal_m_hl12_group`)\n\n" + md(gt))


# ------------------------------------------------------------- corr section
def corr_variants() -> dict:
    v = {
        "x_corr_ew_shrink": CorrCombo(group="none", block_shrink=0.2),
        "x_corr_fastslow_infl": CorrCombo(["condcorr_msw_infl", "mcorr_shrink_block"], group="none",
                                          weights={"condcorr_msw_infl": 0.5, "mcorr_shrink_block": 0.5}),
        "x_corr_dcc_nls": CorrCombo(["mcorr_dcc", "mcorr_nls"], group="none",
                                    weights={"mcorr_dcc": 0.5, "mcorr_nls": 0.5}),
        "x_corr_fastslow_shrink": CorrCombo(list(FASTSLOW), group="none", weights=FASTSLOW, block_shrink=0.2),
    }
    for n, m in v.items():
        m.name = n
    return v


def corr_tables(T, cfc: dict, vols: dict, out: list):
    names = list(cfc)
    Pp, Pr = [], []
    for n in names:
        r = corr_losses(cfc[n], T, n)
        Pp.append(r["pooled"])
        Pr.append(r["pair"])
    P = pd.concat(Pp, ignore_index=True)
    PR = pd.concat(Pr, ignore_index=True)
    ref = "ref_corr_last120m"
    corr_dicts = {n: _long_to_dict_corr(cfc[n]) for n in names}
    for subset in ("daily", "stock_bond", "full12"):
        L = P[P["subset"] == subset].pivot(index="origin", columns="model", values="mse")[names].dropna()
        B = P[P["subset"] == subset].pivot(index="origin", columns="model", values="bias")[names].reindex(L.index)
        rows = {}
        lab = era_of(L.index)
        for n in names:
            d = {"mse": L[n].mean(), "bias": B[n].mean()}
            if n != ref:
                d.update(skill_row(L[n], L[ref]))
            for er in ERAS:
                if (lab == er).sum():
                    d[er] = L[n][lab == er].mean()
            rows[n] = d
        tab = pd.DataFrame(rows).T.sort_values("mse")
        tab.index.name = "model"
        out.append(f"### Correlation MSE, subset `{subset}`: {len(L)} common origins "
                   f"{L.index.min():%Y-%m}..{L.index.max():%Y-%m} (skill vs `{ref}`)\n\n" + md(tab, "{:.4f}"))
    # stock-bond pair by era
    pr = PR[(PR["asset_i"] == "us_large") & (PR["asset_j"] == "ust_int")]
    w = pr.pivot(index="origin", columns="model", values="se")[names].dropna()
    lab = era_of(w.index)
    t = pd.DataFrame({er: w[lab == er].mean() for er in ERAS if (lab == er).any()})
    t["all"] = w.mean()
    t.index.name = "model"
    out.append(f"### us_large~ust_int squared error by era ({len(w)} origins)\n\n" + md(t, "{:.4f}"))
    # covariance loss with fixed vols
    cache: dict = {}
    for vname, vlong in vols.items():
        Q = []
        for n in names:
            Q.append(cov_losses(cov_long(cov_dict(vlong, corr_dicts[n])), T, n, _cache=cache)["pooled"])
        Q = pd.concat(Q, ignore_index=True)
        rows = {}
        for subset in ("daily", "stock_bond", "core9", "full12"):
            L = Q[Q["subset"] == subset].pivot(index="origin", columns="model", values="qlike_rel")[names].dropna()
            for n in names:
                rows.setdefault(n, {})[f"{subset} ({len(L)})"] = L[n].mean()
                if subset == "daily":
                    t = inf.dm_test(L[n], L[ref], warn=False) if n != ref else None
                    rows[n]["daily diff vs ref"] = L[n].mean() - L[ref].mean()
                    rows[n]["daily dm_p"] = t.pvalue if t else np.nan
                    lab = era_of(L.index)
                    for er in ERAS:
                        rows[n][f"daily {er}"] = L[n][lab == er].mean()
        tab = pd.DataFrame(rows).T
        tab = tab.sort_values(tab.columns[0])
        tab.index.name = "corr model"
        out.append(f"### Covariance loss `qlike_rel` with every correlation paired with `{vname}` vols\n\n"
                   + md(tab))


def _long_to_dict_corr(df):
    from ...eval.harness import long_to_dict
    return long_to_dict("corr", df)


# -------------------------------------------------------------- cov section
def cov_tables(panel, T, covs: dict, out: list):
    names = list(covs)
    Q = []
    cache: dict = {}
    for n in names:
        Q.append(cov_losses(cov_long(covs[n]), T, n, _cache=cache)["pooled"])
    Q = pd.concat(Q, ignore_index=True)
    ref = "ref_blend+ref_corr_last120m"
    for subset in ("daily", "sbg4", "core9", "full12"):
        L = Q[Q["subset"] == subset].pivot(index="origin", columns="model", values="qlike_rel")[names].dropna()
        lab = era_of(L.index)
        rows = {}
        for n in names:
            d = {"qlike_rel": L[n].mean()}
            if n != ref:
                t = inf.dm_test(L[n], L[ref], warn=False)
                d["diff vs ref"], d["dm_p"] = L[n].mean() - L[ref].mean(), t.pvalue
            for er in ERAS:
                if (lab == er).any():
                    d[er] = L[n][lab == er].mean()
            rows[n] = d
        tab = pd.DataFrame(rows).T.sort_values("qlike_rel")
        tab.index.name = "cov model"
        out.append(f"### Covariance `qlike_rel`, subset `{subset}`: {len(L)} common origins "
                   f"{L.index.min():%Y-%m}..{L.index.max():%Y-%m}\n\n" + md(tab))
    # portfolios
    t0 = time.perf_counter()
    R = score_portfolios(panel, covs, cache=RealizedCache(panel))
    secs = time.perf_counter() - t0
    S = summarize(R, eras={k: v for k, v in ERAS.items()})
    def pick(port, metric, era="all"):
        x = S[(S["portfolio"] == port) & (S["metric"] == metric) & (S["era"] == era)]
        return x.set_index("model")["mean"].reindex(names)
    cols = {}
    for port in ("p6040", "p_balanced", "p_equity", "erc", "invvol", "ew", "random"):
        cols[f"{port} qlike"] = pick(port, "qlike")
        cols[f"{port} RV/PV"] = pick(port, "var_ratio")
    tab = pd.DataFrame(cols)
    tab.index.name = "cov model"
    n_or = int(S[(S["portfolio"] == "p6040") & (S["metric"] == "qlike") & (S["era"] == "all")]["n_origins"].max())
    out.append(f"### Portfolio risk prediction (common origins, {n_or} for p6040): QLIKE and mean realized/"
               f"predicted variance\n\n" + md(tab))
    cols = {}
    for port in ("gmv", "gmv_long", "erc_vt"):
        for met in ("real_var", "var_ratio", "qlike", "turnover", "gross"):
            if port == "erc_vt" and met not in ("var_ratio", "turnover", "gross"):
                continue
            cols[f"{port} {met}"] = pick(port, met)
    cols["erc_vt |vol-10%|"] = pick("erc_vt", "vt_abs_err")
    tab = pd.DataFrame(cols)
    tab["gmv real_var"] *= 1e4
    tab["gmv_long real_var"] *= 1e4
    tab = tab.rename(columns={"gmv real_var": "gmv real_var x1e4", "gmv_long real_var": "gmv_long real_var x1e4"})
    tab.index.name = "cov model"
    out.append("### Optimized portfolios: realized variance (lower is better), realized/predicted variance, "
               "turnover (one-way, monthly)\n\n" + md(tab, "{:.3f}"))
    t = {}
    for n in names:
        for port in ("gmv", "gmv_long", "p_balanced", "random"):
            for era in ERAS:
                t.setdefault(n, {})[f"{port} {era}"] = pick(port, "var_ratio", era)[n]
    tab = pd.DataFrame(t).T
    tab.index.name = "cov model"
    out.append("### Realized/predicted variance by era\n\n" + md(tab, "{:.2f}"))
    # significance of GMV realized variance vs the reference composite
    rows = {}
    for port in ("gmv", "gmv_long"):
        w = R[(R["portfolio"] == port) & (R["metric"] == "real_var")].assign(model=lambda x: x["model"].astype(str)) \
            .pivot_table(index="origin", columns="model", values="value", aggfunc="first")[names].dropna()
        for n in names:
            if n == ref:
                continue
            s = skill_row(w[n], w[ref])
            rows.setdefault(n, {})[f"{port} real_var skill%"] = s["skill%"]
            rows[n][f"{port} ci"] = f"[{s['ci_lo']:.1f}, {s['ci_hi']:.1f}]"
            rows[n][f"{port} dm_p"] = s["dm_p"]
    tab = pd.DataFrame(rows).T
    tab.index.name = "cov model"
    out.append(f"### Minimum-variance realized variance, skill vs `{ref}` (portfolio scoring {secs:.0f}s)\n\n"
               + md(tab, "{:.3f}"))


# ------------------------------------------------------------------- main
def run_registered(panel):
    t = {}
    for kind, names in (("vol", VOL_MODELS), ("corr", CORR_MODELS), ("cov", COV_PAIRS)):
        for n in names:
            r = run_model(n, kind, panel, force=True)
            t[f"{kind}/{n}"] = (r.seconds, r.model_seconds)
            print(r, file=sys.stderr, flush=True)
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m assetforecast.models.combo.analysis")
    ap.add_argument("--forecast", action="store_true", help="re-run every registered combo first (force)")
    ap.add_argument("--out", help="write the markdown tables here (default stdout)")
    a = ap.parse_args(argv)
    t_all = time.perf_counter()
    panel = load_panel()
    out = [f"# combo tables (generated {pd.Timestamp.now():%Y-%m-%d %H:%M}, results {RESULTS})"]
    if a.forecast:
        times = run_registered(panel)
        out.append("### Run times (full monthly run, seconds: total / model)\n\n" + "\n".join(
            f"- `{k}`: {s:.1f} / {m:.1f}" for k, (s, m) in times.items()))
    T = get_targets(panel, RESULTS)

    # ---- vol
    t0 = time.perf_counter()
    singles = [m for m in list_models("vol") if family_of(m) != "combo"]
    fc = {n: load_forecasts_long("vol", n) for n in list(VOL_MODELS) + singles}
    variants = vol_variants(panel)
    fc.update(walk(variants, "vol", panel))
    fc["x_logspace_hier"] = logspace_hier(fc["combo_vol_hier"])
    print(f"vol variants walked in {time.perf_counter() - t0:.0f}s", file=sys.stderr, flush=True)
    out.append("## Volatility")
    res, P, A = vol_tables(T, fc, out, panel)
    calibration_tables(res, fc, out)
    learned_vs_equal(res, out)
    mcs_table(res, out)
    weight_tables(panel, out)

    # ---- corr
    out.append("## Correlation")
    cfc = {n: load_forecasts_long("corr", n) for n in list(CORR_MODELS) + CORR_SINGLES}
    cfc.update(walk(corr_variants(), "corr", panel))
    vols = {"ref_blend": fc["ref_blend"], "combo_vol_hier": fc["combo_vol_hier"]}
    corr_tables(T, cfc, vols, out)

    # ---- cov
    out.append("## Covariance and portfolios")
    covs = {n: load_forecasts("cov", n) for n in COV_PAIRS}
    for n in ("structural_factor_global", "structural_factor_lognormal"):
        covs[n] = load_forecasts("cov", n)
    cd = {n: load_forecasts("corr", n) for n in ("ref_corr_last120m", "mcorr_nls", "combo_corr_hier_shrink",
                                                   "combo_corr_hier")}
    covs["ref_blend+ref_corr_last120m"] = cov_dict(fc["ref_blend"], cd["ref_corr_last120m"])
    covs["statespace_2c_daily+mcorr_nls"] = cov_dict(fc["statespace_2c_daily"], cd["mcorr_nls"])
    covs["combo_vol_stack+combo_corr_hier_shrink"] = cov_dict(fc["combo_vol_stack"], cd["combo_corr_hier_shrink"])
    covs["combo_vol_cal_rm2006+combo_corr_hier"] = cov_dict(fc["combo_vol_cal_rm2006"], cd["combo_corr_hier"])
    covs["x_cov_simple_fastslow"] = cov_dict(fc["combo_vol_cal_m_hl12"], load_forecasts("corr", "combo_corr_fastslow"))
    cov_tables(panel, T, covs, out)

    out.append(f"_total analysis time {time.perf_counter() - t_all:.0f}s_")
    text = "\n\n".join(out) + "\n"
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
