#!/usr/bin/env python3
"""Final evaluation of every stored model: development period and holdout, kept apart.

    export OMP_NUM_THREADS=1 ASSETFORECAST_RESULTS=$PWD/results/full      # no ASSETFORECAST_MAX_DATE
    python scripts/score_final.py [--out results/final] [--dev-results results/dev]

Reads the stored forecasts and cached targets in ``$ASSETFORECAST_RESULTS`` (the
full-history re-run) and writes small tidy CSV/JSON files to ``--out``. Nothing
is chosen here: the roles below were fixed in ``reports/SELECTION.md`` before
any post-2013 origin was scored.

Periods (by forecast ORIGIN):
    dev      origins <= 2013-12-31 (windows end by 2014-12)
    holdout  2014-01-31 .. 2025-10-31 (last origin with a fully observed window)
    full     both
Eras (development only): pre-1926, 1926-1945, 1946-1979, 1980-1999, 2000-2013.

Comparison rule: within every table, models are compared on IDENTICAL cells
(origin x asset for vol, origin x pair for corr, origin for cov), the cells
where every model in the table has a forecast and a target. Pooled losses are
the per-origin mean over those cells, then the mean over origins (each origin
counts once). ``n_origins`` / ``n_cells`` in every output say how many.

Inference: ``assetforecast.eval.inference`` defaults (DM with the EWC long-run
variance and t(B) reference; Fieller-EWC 90% skill intervals; Goetze-Kuensch
MCS). On the holdout (142 overlapping origins, about 12 independent years) B
is clipped at its floor of 4, so intervals are wide by construction.

Outputs (``--out``):
    coverage.csv              per stored model: origins, gaps, asset-set mismatches
    vol_leaderboard.csv       every vol model: QLIKE, R/F by period/era/subset, skill vs best reference
    vol_leaderboard_asset.csv every vol model per asset and period, skill vs best reference
    corr_leaderboard.csv      every corr model: pairwise MSE, bias, skill vs best reference
    cov_leaderboard.csv       every cov model: qlike_rel, frob_scaled, difference vs pair_cov_ref
    h2h_vol.csv / h2h_vol_pairwise.csv / h2h_vol_asset.csv   the four candidates (+2 vol references)
    h2h_cov_pairwise.csv      the four candidates, pairwise covariance-loss tests
    h2h_corr.csv              the four candidates' implied correlations, MSE
    mcs.csv                   Model Confidence Sets (dev, holdout) for the four candidates
    portfolio_summary.csv     portfolio metrics (summarize) with HAC 90% intervals, by period/era
    portfolio_dm.csv          primary vs the other candidates on portfolio QLIKE and GMV variance
    bridge_calibration.csv    Var(z) per asset / portfolio, identity vs vr_market, by period
    bridge_tails.csv          tail shares (1, 2 sigma; mixture 5%, 1%) with block-bootstrap intervals
    expectations.csv          the five pre-registered expectations and their outcome
    episodes_vol.csv          primary vs realized vol per asset at the holdout's December origins and Feb 2020
    episodes_worst.csv        largest holdout misses of the primary
    episodes_corr.csv         stock-bond correlation forecasts vs realized, 2019-12 .. 2023-12
    dashboard_vol.csv         per asset per origin: realized, candidates, family-mean 10-90% range
    dashboard_corr.csv        per key pair per origin: realized and the candidates' forecasts
    dashboard_portfolio.csv   60/40 and balanced: realized and predicted vol per origin
    accuracy.csv              log(realized / forecast vol) distribution of the primary, noise split
    data_correction.csv       dev-period scores before vs after the data correction
    verification.csv          independent recomputation of targets and losses from panel returns
    summary.json              headline numbers and run metadata
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEV_END = pd.Timestamp("2013-12-31")
HOLDOUT_START = pd.Timestamp("2014-01-31")
HOLDOUT_END = pd.Timestamp("2025-10-31")
PERIODS = ("dev", "holdout", "full")
ERAS = {
    "pre-1926": (None, "1925-12-31"),
    "1926-1945": ("1926-01-01", "1945-12-31"),
    "1946-1979": ("1946-01-01", "1979-12-31"),
    "1980-1999": ("1980-01-01", "1999-12-31"),
    "2000-2013": ("2000-01-01", "2013-12-31"),
}
LAST30_START = pd.Timestamp("1995-11-30")      # last 30 years of scored origins (to 2025-10)

CANDIDATES = {"primary": "combo_cov_main", "runner_up": "combo_cov_simple",
              "single": "pair_cov_single", "reference": "pair_cov_ref"}
ROLE_OF = {v: k for k, v in CANDIDATES.items()}
VOL_REFS = ["ref_blend", "ref_last12m", "ref_last60m", "ref_longrun"]
CORR_REFS = ["ref_corr_last120m", "ref_corr_longrun"]      # ref_corr_zero is a diagnostic, never "best"
COV_REF = "pair_cov_ref"
H2H_VOL_EXTRA = ["ref_last12m", "ref_blend"]               # the references named in expectation 1
KEY_PAIRS = [("us_large", "ust_int"), ("us_large", "ust_long"), ("us_large", "intl_dev"),
             ("us_large", "hy_credit"), ("us_large", "gold"), ("ust_long", "tips")]
POLICY_FIXED = ["p6040", "p_balanced", "erc", "random"]     # expectation 3
LEVEL = 0.90


# =========================================================== pure helpers
def period_mask(origins, period: str) -> np.ndarray:
    """Boolean mask of origins in ``period`` ('dev', 'holdout', 'full')."""
    o = pd.DatetimeIndex(origins)
    dev = np.asarray(o <= DEV_END)
    hold = np.asarray((o >= HOLDOUT_START) & (o <= HOLDOUT_END))
    if period == "dev":
        return dev
    if period == "holdout":
        return hold
    if period == "full":
        return dev | hold
    raise ValueError(period)


def era_mask(origins, era: str) -> np.ndarray:
    lo, hi = ERAS[era]
    o = pd.DatetimeIndex(origins)
    m = np.ones(len(o), bool)
    if lo is not None:
        m &= np.asarray(o >= pd.Timestamp(lo))
    if hi is not None:
        m &= np.asarray(o <= pd.Timestamp(hi))
    return m & period_mask(o, "dev")


def family_of(model: str) -> str:
    """Model family = name prefix ('pair_cov_*' -> 'pairs'; 'cov:x' -> family of x)."""
    m = model[4:] if model.startswith("cov:") else model
    p = m.split("_")[0]
    return "pairs" if p == "pair" else p


def to_cube(df: pd.DataFrame, value: str, item_col, models=None, items=None, origins=None):
    """Long frame (model, origin, item[, ...]) -> (origins, items, models, O x I x M array).

    ``item_col`` is a column name or a function of the frame returning item labels.
    Duplicate (model, origin, item) keys are an error.
    """
    it = item_col(df) if callable(item_col) else df[item_col]
    it = pd.Series(np.asarray(it, dtype=object), index=df.index)
    models = list(dict.fromkeys(df["model"])) if models is None else list(models)
    items = list(dict.fromkeys(it)) if items is None else list(items)
    origins = pd.DatetimeIndex(sorted(df["origin"].unique())) if origins is None else pd.DatetimeIndex(origins)
    C = np.full((len(origins), len(items), len(models)), np.nan)
    oi = origins.get_indexer(pd.DatetimeIndex(df["origin"]))
    ii = pd.Index(items).get_indexer(it)
    mi = pd.Index(models).get_indexer(df["model"])
    ok = (oi >= 0) & (ii >= 0) & (mi >= 0)
    key = oi[ok] * (len(items) * len(models)) + ii[ok] * len(models) + mi[ok]
    if len(np.unique(key)) != len(key):
        raise ValueError("duplicate (model, origin, item) rows")
    C[oi[ok], ii[ok], mi[ok]] = df[value].to_numpy(float)[ok]
    return origins, items, models, C


def common_cells(C: np.ndarray) -> np.ndarray:
    """O x I mask of cells where every model has a finite value."""
    return np.isfinite(C).all(axis=2)


def pool_cells(C: np.ndarray, cells: np.ndarray):
    """Per-origin mean over ``cells`` (O x I) of an O x I x M cube -> (O x M, counts)."""
    w = cells[:, :, None]
    cnt = cells.sum(axis=1)
    num = np.where(w, np.nan_to_num(C), 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / cnt[:, None]
    out[cnt == 0] = np.nan
    return out, cnt


def subset_cells(cells: np.ndarray, items: list, members: list) -> np.ndarray:
    """Restrict ``cells`` to ``members`` and keep only origins where ALL members are present."""
    idx = [items.index(m) for m in members]
    out = np.zeros_like(cells)
    if not idx:
        return out
    full = cells[:, idx].all(axis=1)
    out[np.ix_(full, idx)] = True
    return out


def pair_label(a: str, b: str) -> str:
    return f"{a}~{b}"


def hac_mean_ci(x: pd.Series, level: float = LEVEL) -> tuple[float, float, float, float]:
    """Mean of a monthly series with a HAC (EWC, inference defaults) t-interval.

    Returns (mean, lo, hi, n_eff); NaN interval below 6 observations.
    """
    from scipy import stats
    from assetforecast.eval.inference import long_run_variance
    x = x.dropna()
    m = float(x.mean()) if len(x) else np.nan
    if len(x) < 6:
        return m, np.nan, np.nan, np.nan
    r = long_run_variance(x)
    se = math.sqrt(r.lrv / r.n_obs)
    q = stats.t.ppf(0.5 + level / 2, r.df) if np.isfinite(r.df) else stats.norm.ppf(0.5 + level / 2)
    return m, m - q * se, m + q * se, float(r.n_eff)


def compare(la: pd.Series, lb: pd.Series, skill: bool = True, level: float = LEVEL) -> dict:
    """DM (EWC) test of la - lb, its 90% interval, and the Fieller skill of a vs b.

    Skill is the % reduction of a's mean loss relative to b (positive = a better).
    """
    from scipy import stats
    from assetforecast.eval.inference import dm_test, skill_score
    out = {"mean_diff": np.nan, "diff_lo": np.nan, "diff_hi": np.nan, "dm_p": np.nan,
           "skill_pct": np.nan, "skill_lo": np.nan, "skill_hi": np.nan, "n_eff": np.nan, "df": np.nan}
    d = pd.concat({"a": la, "b": lb}, axis=1).dropna()
    if len(d) < 6:
        return out
    r = dm_test(d["a"], d["b"], warn=False)
    q = stats.t.ppf(0.5 + level / 2, r.df) if np.isfinite(r.df) else stats.norm.ppf(0.5 + level / 2)
    out.update(mean_diff=r.mean_diff, diff_lo=r.mean_diff - q * r.se, diff_hi=r.mean_diff + q * r.se,
               dm_p=r.pvalue, n_eff=r.n_eff, df=r.df)
    if skill and d["b"].mean() > 0 and (d["a"] >= 0).all():
        try:
            s = skill_score(d["a"], d["b"], level=level, warn=False)
            out.update(skill_pct=s.skill, skill_lo=s.ci_low, skill_hi=s.ci_high)
        except ValueError:
            pass
    return out


def block_bootstrap_mean(X, block: int = 24, n_boot: int = 2000, level: float = LEVEL, seed: int = 0):
    """Mean of all finite entries of a T x K array with a circular moving-block interval.

    Rows (origins) are resampled in blocks of ``block`` consecutive rows, keeping
    the cross-section of each row together (overlap and cross-correlation are
    preserved inside blocks). Returns (mean, lo, hi).
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    T = X.shape[0]
    fin = np.isfinite(X)
    if not fin.any():
        return np.nan, np.nan, np.nan
    S = np.where(fin, X, 0.0).sum(axis=1)
    N = fin.sum(axis=1).astype(float)
    m = float(S.sum() / N.sum())
    if T < 2:
        return m, np.nan, np.nan
    b = max(1, min(int(block), T))
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(T / b))
    starts = rng.integers(0, T, size=(n_boot, nb))
    idx = ((starts[:, :, None] + np.arange(b)[None, None, :]) % T).reshape(n_boot, -1)[:, :T]
    with np.errstate(invalid="ignore", divide="ignore"):
        stats_ = S[idx].sum(axis=1) / N[idx].sum(axis=1)
    lo, hi = np.nanquantile(stats_, [0.5 - level / 2, 0.5 + level / 2])
    return m, float(lo), float(hi)


def log_ratio_summary(lr) -> dict:
    """Distribution of log(realized / forecast vol): quantiles and the central two-thirds in %."""
    lr = np.asarray(lr, float)
    lr = lr[np.isfinite(lr)]
    if lr.size == 0:
        return {"n": 0}
    q = np.quantile(lr, [0.05, 1 / 6, 0.5, 5 / 6, 0.95])
    return {"n": int(lr.size), "mean": float(lr.mean()), "sd": float(lr.std(ddof=1)) if lr.size > 1 else np.nan,
            "q05": q[0], "q1_6": q[1], "median": q[2], "q5_6": q[3], "q95": q[4],
            "two_thirds_lo_pct": 100 * (math.exp(q[1]) - 1), "two_thirds_hi_pct": 100 * (math.exp(q[3]) - 1),
            "ninety_lo_pct": 100 * (math.exp(q[0]) - 1), "ninety_hi_pct": 100 * (math.exp(q[4]) - 1)}


def kday_var(X, k: int = 21) -> np.ndarray:
    """Independent re-implementation of the target: (n_rows / k) * mean of squared
    overlapping k-row sums (missing = 0), uncentered; for a T x N array, returns N
    variances and the cross-product matrix. Written without assetforecast.eval code."""
    X = np.nan_to_num(np.asarray(X, float))
    T = X.shape[0]
    c = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)])
    s = c[k:] - c[:-k]                      # T - k + 1 sums
    G = s.T @ s / s.shape[0] * (T / k)
    return np.diag(G).copy(), G


def qlike_var(vr, vf):
    r = np.asarray(vr, float) / np.asarray(vf, float)
    return r - np.log(r) - 1.0


def qlike_rel_independent(H, S) -> float:
    """[log det H + tr(H^-1 S) - sum log S_ii - N] / N, with plain numpy."""
    H, S = np.asarray(H, float), np.asarray(S, float)
    n = len(H)
    sign, ld = np.linalg.slogdet(H)
    if sign <= 0:
        return np.nan
    tr = float(np.trace(np.linalg.solve(H, S)))
    return (ld + tr - float(np.log(np.diag(S)).sum()) - n) / n


def noise_sd_logvol(n_days: int = 252, k: int = 21, n_sim: int = 20000, seed: int = 0) -> float:
    """Sd of log(realized vol) of the k-day target for iid normal days (pure measurement noise)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_sim, n_days))
    c = np.concatenate([np.zeros((n_sim, 1)), np.cumsum(x, axis=1)], axis=1)
    s = c[:, k:] - c[:, :-k]
    v = (s ** 2).mean(axis=1) * n_days / k
    return float(np.std(0.5 * np.log(v), ddof=1))


def noise_sd_logvol_proxy(n_months: int = 12) -> float:
    """Sd of log(realized vol) of the monthly proxy 12*mean(r_m^2) under normality: 0.5 sqrt(psi'(n/2))."""
    from scipy.special import polygamma
    return float(0.5 * math.sqrt(polygamma(1, n_months / 2)))


def verdict_between(x: float, lo: float, hi: float) -> bool:
    return bool(np.isfinite(x) and lo <= x <= hi)


# ====================================================== real-data stages
def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Ctx:
    """Everything loaded once."""

    def __init__(self, results: Path, out: Path, dev_results: Path | None):
        from assetforecast.data.panel import load_panel
        from assetforecast.eval.targets import load_targets, panel_fingerprint
        if os.environ.get("ASSETFORECAST_MAX_DATE"):
            raise SystemExit("unset ASSETFORECAST_MAX_DATE: the final evaluation scores the full history")
        self.results, self.out, self.dev_results = results, out, dev_results
        self.panel = load_panel()
        self.targets = load_targets(results)
        fp = panel_fingerprint(self.panel)
        self.fingerprint_ok = fp == self.targets.meta.get("fingerprint")
        if not self.fingerprint_ok:
            raise SystemExit(f"cached targets in {results}/targets do not match the panel "
                             f"({self.targets.meta.get('fingerprint')} vs {fp}); rebuild them first")
        from assetforecast.eval.harness import stored_models
        self.models = {k: stored_models(k, results) for k in ("vol", "corr", "cov")}
        self.tables: dict[str, pd.DataFrame] = {}
        self.summary: dict = {"results_dir": str(results), "targets_observed_through":
                              self.targets.meta.get("observed_through"), "targets_fingerprint": fp,
                              "dev_end": str(DEV_END.date()), "holdout": [str(HOLDOUT_START.date()),
                                                                          str(HOLDOUT_END.date())]}

    def write(self, name: str, df: pd.DataFrame, digits: int = 6):
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].round(digits)
        p = self.out / name
        df.to_csv(p, index=False)
        self.tables[name] = df
        _log(f"wrote {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p} ({len(df)} rows)")


# ------------------------------------------------------------- coverage
def stage_coverage(ctx: Ctx):
    from assetforecast.eval.harness import load_forecasts_long, origins_for
    panel = ctx.panel
    origins = pd.DatetimeIndex(origins_for(panel))
    cnt = panel.monthly.notna().cumsum()
    cnt = cnt.reindex(origins, method="ffill")
    elig = {o: [a for a in cnt.columns if cnt.at[o, a] >= 36] for o in origins}
    expected = pd.DatetimeIndex([o for o in origins if len(elig[o]) >= 2])
    scored = set(ctx.targets.origins)
    rows = []
    for kind, models in ctx.models.items():
        for m in models:
            df = load_forecasts_long(kind, m, ctx.results)
            a_col = "asset" if kind == "vol" else "asset_i"
            got = df.groupby("origin")[a_col].agg(lambda s: tuple(sorted(set(s))))
            go = pd.DatetimeIndex(got.index)
            missing = expected.difference(go)
            extra = go.difference(expected)
            mism = sum(1 for o in expected.intersection(go) if got[o] != tuple(sorted(elig[o])))
            nonfinite = int((~np.isfinite(df[{"vol": "vol", "corr": "corr", "cov": "cov"}[kind]])).sum())
            hold = go[(go >= HOLDOUT_START) & (go <= HOLDOUT_END)]
            meta_p = ctx.results / "forecasts" / kind / f"{m}.meta.json"
            meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
            rows.append({"kind": kind, "model": m, "family": family_of(m), "n_origins": len(go),
                         "first": go.min().date() if len(go) else None, "last": go.max().date() if len(go) else None,
                         "n_expected": len(expected), "n_missing": len(missing), "n_extra": len(extra),
                         "first_missing": missing.min().date() if len(missing) else None,
                         "n_asset_set_mismatch": mism, "n_nonfinite": nonfinite,
                         "n_holdout_origins": len(hold),
                         "n_holdout_scorable": len([o for o in hold if o in scored]),
                         "meta_finished": meta.get("finished") or meta.get("last_run") or meta.get("time")})
    cov = pd.DataFrame(rows)
    ctx.write("coverage.csv", cov)
    exp_hold = int(((expected >= HOLDOUT_START) & (expected <= HOLDOUT_END)).sum())
    ctx.summary["coverage"] = {
        "n_expected_origins": len(expected), "first_expected": str(expected.min().date()),
        "last_expected": str(expected.max().date()), "expected_holdout_origins": exp_hold,
        "models": {k: len(v) for k, v in ctx.models.items()},
        "models_incomplete": cov.loc[(cov.n_missing > 0) | (cov.n_asset_set_mismatch > 0) | (cov.n_nonfinite > 0),
                                     "model"].tolist()}
    return cov


# ------------------------------------------------------- vol leaderboard
def vol_cube(asset_frame: pd.DataFrame, models: list[str]):
    from assetforecast.universe import ASSET_IDS
    o, items, models, Q = to_cube(asset_frame, "qlike", "asset", models=models, items=list(ASSET_IDS))
    R = to_cube(asset_frame.assign(rf=(asset_frame["vol_r"] / asset_frame["vol_f"]) ** 2), "rf", "asset",
                models=models, items=items, origins=o)[3]
    LE = to_cube(asset_frame, "log_err", "asset", models=models, items=items, origins=o)[3]
    px = asset_frame.drop_duplicates(["origin", "asset"]).pivot(index="origin", columns="asset", values="proxy")
    PX = px.reindex(index=o, columns=items).fillna(False).to_numpy(bool)
    return o, items, models, Q, R, LE, PX


def vol_subset_cells(cells, PX, items, subset):
    from assetforecast.eval.losses import DEFAULT_SUBSETS
    if subset == "all":
        return cells
    if subset == "daily":
        return cells & ~PX
    return subset_cells(cells, items, DEFAULT_SUBSETS[subset])


def leaderboard(L: pd.DataFrame, extra: dict[str, pd.DataFrame], cnt: pd.Series, refs: list[str],
                kind: str, subset: str, skill: bool = True, eras: bool = True, fixed_ref: str | None = None,
                infer: bool = True) -> list[dict]:
    """Rows per period (and dev era) x model from an origin x model loss frame on common cells."""
    rows = []
    for period in PERIODS:
        m = period_mask(L.index, period)
        Lp = L[m].dropna(how="any")
        if Lp.empty:
            continue
        present = [r for r in refs if r in Lp.columns]
        ref = fixed_ref if fixed_ref is not None else (min(present, key=lambda r: Lp[r].mean()) if present else None)
        for mod in Lp.columns:
            row = {"kind": kind, "model": mod, "family": family_of(mod), "period": period, "era": "all",
                   "subset": subset, "n_origins": len(Lp), "n_cells": int(cnt.reindex(Lp.index).sum()),
                   "first": Lp.index.min().date(), "last": Lp.index.max().date(), "loss": Lp[mod].mean()}
            for k, X in extra.items():
                row[k] = X.loc[Lp.index, mod].mean()
            row["ref"] = ref
            if infer and ref is not None and mod != ref:
                row.update(compare(Lp[mod], Lp[ref], skill=skill))
            rows.append(row)
        if eras and period == "dev":
            for era in ERAS:
                em = era_mask(Lp.index, era)
                Le = Lp[em]
                if Le.empty:
                    continue
                for mod in Le.columns:
                    row = {"kind": kind, "model": mod, "family": family_of(mod), "period": "dev", "era": era,
                           "subset": subset, "n_origins": len(Le), "n_cells": int(cnt.reindex(Le.index).sum()),
                           "first": Le.index.min().date(), "last": Le.index.max().date(), "loss": Le[mod].mean()}
                    for k, X in extra.items():
                        row[k] = X.loc[Le.index, mod].mean()
                    row["ref"] = ref
                    rows.append(row)
    return rows


def stage_vol(ctx: Ctx):
    from assetforecast.eval.losses import DEFAULT_SUBSETS, compute_losses
    models = ctx.models["vol"]
    _log(f"vol losses for {len(models)} models")
    lv = compute_losses("vol", models, ctx.targets, results_dir=ctx.results)
    af = lv["asset"]
    o, items, models, Q, R, LE, PX = vol_cube(af, models)
    cells = common_cells(Q)
    rows = []
    for subset in ["daily", "all", *DEFAULT_SUBSETS]:
        c = vol_subset_cells(cells, PX, items, subset)
        Lq, cnt = pool_cells(Q, c)
        Lr, _ = pool_cells(R, c)
        L = pd.DataFrame(Lq, index=o, columns=models)
        X = {"rf": pd.DataFrame(Lr, index=o, columns=models)}
        rows += leaderboard(L, X, pd.Series(cnt, index=o), VOL_REFS, "vol", subset)
    lb = pd.DataFrame(rows).rename(columns={"loss": "qlike"})
    lb["rank"] = lb.groupby(["period", "era", "subset"])["qlike"].rank(method="min")
    ctx.write("vol_leaderboard.csv", lb)
    # per asset
    rows = []
    for j, a in enumerate(items):
        ok = cells[:, j]
        if not ok.any():
            continue
        L = pd.DataFrame(Q[:, j, :], index=o, columns=models).where(pd.Series(ok, index=o), axis=0)
        Rr = pd.DataFrame(R[:, j, :], index=o, columns=models)
        npx = pd.Series(PX[:, j] & ok, index=o)
        for period in PERIODS:
            m = period_mask(o, period) & ok
            Lp = L[m]
            if len(Lp) < 2:
                continue
            ref = min(VOL_REFS, key=lambda r: Lp[r].mean())
            for mod in models:
                row = {"asset": a, "model": mod, "family": family_of(mod), "period": period, "n_origins": len(Lp),
                       "n_proxy": int(npx[m].sum()), "qlike": Lp[mod].mean(), "rf": Rr.loc[Lp.index, mod].mean(),
                       "ref": ref}
                if mod != ref:
                    c = compare(Lp[mod], Lp[ref])
                    row.update({k: c[k] for k in ("skill_pct", "skill_lo", "skill_hi", "dm_p")})
                rows.append(row)
    la = pd.DataFrame(rows)
    la["rank"] = la.groupby(["asset", "period"])["qlike"].rank(method="min")
    ctx.write("vol_leaderboard_asset.csv", la)
    ctx.vol_af = af
    return lb


# ------------------------------------------------------ corr leaderboard
def corr_cube(pair_frame: pd.DataFrame, models: list[str]):
    from assetforecast.universe import ASSET_IDS
    pairs = [pair_label(a, b) for i, a in enumerate(ASSET_IDS) for b in ASSET_IDS[i + 1:]]
    lab = lambda d: d["asset_i"] + "~" + d["asset_j"]
    o, items, models, SE = to_cube(pair_frame, "se", lab, models=models, items=pairs)
    ER = to_cube(pair_frame, "err", lab, models=models, items=items, origins=o)[3]
    return o, items, models, SE, ER


def corr_subset_cells(cells, items, subset):
    from assetforecast.eval.losses import DEFAULT_SUBSETS
    if subset == "daily":
        return cells
    mem = set(DEFAULT_SUBSETS[subset])
    return subset_cells(cells, items, [p for p in items if set(p.split("~")) <= mem])


def stage_corr(ctx: Ctx):
    from assetforecast.eval.losses import DEFAULT_SUBSETS, compute_losses
    models = ctx.models["corr"]
    _log(f"corr losses for {len(models)} models")
    lc = compute_losses("corr", models, ctx.targets, results_dir=ctx.results)
    o, items, models, SE, ER = corr_cube(lc["pair"], models)
    cells = common_cells(SE)
    rows = []
    for subset in ["daily", *DEFAULT_SUBSETS]:
        c = corr_subset_cells(cells, items, subset)
        Ls, cnt = pool_cells(SE, c)
        Le, _ = pool_cells(ER, c)
        L = pd.DataFrame(Ls, index=o, columns=models)
        rows += leaderboard(L, {"bias": pd.DataFrame(Le, index=o, columns=models)}, pd.Series(cnt, index=o),
                            CORR_REFS, "corr", subset)
    lb = pd.DataFrame(rows).rename(columns={"loss": "mse"})
    lb["rank"] = lb.groupby(["period", "era", "subset"])["mse"].rank(method="min")
    ctx.write("corr_leaderboard.csv", lb)
    return lb


# ------------------------------------------------------- cov leaderboard
def stage_cov(ctx: Ctx):
    from assetforecast.eval.losses import compute_losses
    models = ctx.models["cov"]
    _log(f"cov losses for {len(models)} models")
    lc = compute_losses("cov", models, ctx.targets, results_dir=ctx.results)
    pooled = lc["pooled"]
    rows = []
    for subset in pooled["subset"].unique():
        p = pooled[pooled["subset"] == subset]
        dims = p.pivot_table(index="origin", columns="model", values="dim")
        Lq = p.pivot_table(index="origin", columns="model", values="qlike_rel").reindex(columns=models)
        Lf = p.pivot_table(index="origin", columns="model", values="frob_scaled").reindex(columns=models)
        ok = Lq.notna().all(axis=1) & Lf.notna().all(axis=1) & (dims.nunique(axis=1) == 1)
        Lq, Lf = Lq[ok], Lf[ok]
        cnt = dims[ok].iloc[:, 0]
        rq = leaderboard(Lq, {"frob_scaled": Lf}, cnt, [COV_REF], "cov", subset, skill=False, fixed_ref=COV_REF)
        rf = leaderboard(Lf, {}, cnt, [COV_REF], "cov", subset, skill=True, eras=False, fixed_ref=COV_REF)
        fmap = {(r["model"], r["period"]): r for r in rf}
        for r in rq:
            if r["era"] == "all" and (r["model"], r["period"]) in fmap:
                f = fmap[(r["model"], r["period"])]
                r.update({"frob_diff": f.get("mean_diff"), "frob_dm_p": f.get("dm_p"),
                          "frob_skill_pct": f.get("skill_pct"), "frob_skill_lo": f.get("skill_lo"),
                          "frob_skill_hi": f.get("skill_hi")})
        rows += rq
    lb = pd.DataFrame(rows).rename(columns={"loss": "qlike_rel", "n_cells": "n_asset_origins"})
    for c in ("skill_pct", "skill_lo", "skill_hi"):
        lb = lb.drop(columns=c, errors="ignore")
    lb["rank"] = lb.groupby(["period", "era", "subset"])["qlike_rel"].rank(method="min")
    ctx.write("cov_leaderboard.csv", lb)
    ctx.cov_pooled = pooled
    return lb


# ---------------------------------------------------------- head to head
def stage_h2h(ctx: Ctx):
    from assetforecast.eval.inference import model_confidence_set
    from assetforecast.eval.losses import DEFAULT_SUBSETS, compute_losses
    cands = list(CANDIDATES.values())
    vmods = [f"cov:{m}" for m in cands] + H2H_VOL_EXTRA
    lv = compute_losses("vol", vmods, ctx.targets, results_dir=ctx.results)
    af = lv["asset"]
    ctx.h2h_af = af
    o, items, models, Q, R, LE, PX = vol_cube(af, vmods)
    cells = common_cells(Q)
    rows, prow = [], []
    pairs = [(a, b) for i, a in enumerate(vmods) for b in vmods[i + 1:]]
    Ls = {}
    for subset in ["daily", "all", *DEFAULT_SUBSETS]:
        c = vol_subset_cells(cells, PX, items, subset)
        Lq, cnt = pool_cells(Q, c)
        Lr, _ = pool_cells(R, c)
        L = pd.DataFrame(Lq, index=o, columns=vmods).dropna()
        Rr = pd.DataFrame(Lr, index=o, columns=vmods).loc[L.index]
        cn = pd.Series(cnt, index=o).loc[L.index]
        Ls[subset] = L
        slices = [(p, "all", period_mask(L.index, p)) for p in PERIODS] + \
                 [("dev", e, era_mask(L.index, e)) for e in ERAS]
        for period, era, m in slices:
            if m.sum() < 2:
                continue
            Lp = L[m]
            for mod in vmods:
                rows.append({"subset": subset, "period": period, "era": era, "model": mod,
                             "role": ROLE_OF.get(mod[4:], mod), "n_origins": len(Lp), "n_cells": int(cn[m].sum()),
                             "qlike": Lp[mod].mean(), "rf": Rr.loc[Lp.index, mod].mean()})
            for a, b in pairs:
                r = compare(Lp[a], Lp[b])
                prow.append({"subset": subset, "period": period, "era": era, "model_a": a, "model_b": b,
                             "qlike_a": Lp[a].mean(), "qlike_b": Lp[b].mean(), "n_origins": len(Lp), **r})
    ctx.write("h2h_vol.csv", pd.DataFrame(rows))
    ctx.write("h2h_vol_pairwise.csv", pd.DataFrame(prow))
    # per asset
    rows = []
    for j, a in enumerate(items):
        ok = cells[:, j]
        L = pd.DataFrame(Q[:, j, :], index=o, columns=vmods)[ok]
        Rr = pd.DataFrame(R[:, j, :], index=o, columns=vmods)[ok]
        LEa = pd.DataFrame(LE[:, j, :], index=o, columns=vmods)[ok]
        for period in PERIODS:
            m = period_mask(L.index, period)
            Lp = L[m]
            if len(Lp) < 2:
                continue
            for mod in vmods:
                row = {"asset": a, "period": period, "model": mod, "role": ROLE_OF.get(mod[4:], mod),
                       "n_origins": len(Lp), "n_proxy": int(PX[:, j][ok][m].sum()), "qlike": Lp[mod].mean(),
                       "rf": Rr.loc[Lp.index, mod].mean(), "mean_log_err": LEa.loc[Lp.index, mod].mean()}
                if mod != vmods[0]:
                    c = compare(Lp[vmods[0]], Lp[mod])
                    row.update({"primary_skill_vs_this": c["skill_pct"], "skill_lo": c["skill_lo"],
                                "skill_hi": c["skill_hi"], "dm_p": c["dm_p"]})
                rows.append(row)
    ctx.write("h2h_vol_asset.csv", pd.DataFrame(rows))

    # covariance pairwise among the four candidates
    pooled = ctx.cov_pooled
    prow = []
    Lcov = {}
    for subset in pooled["subset"].unique():
        p = pooled[(pooled["subset"] == subset) & pooled["model"].isin(cands)]
        Lq = p.pivot_table(index="origin", columns="model", values="qlike_rel").reindex(columns=cands).dropna()
        Lf = p.pivot_table(index="origin", columns="model", values="frob_scaled").reindex(columns=cands).loc[Lq.index]
        Lcov[subset] = Lq
        slices = [(pp, "all", period_mask(Lq.index, pp)) for pp in PERIODS] + \
                 [("dev", e, era_mask(Lq.index, e)) for e in ERAS]
        for period, era, m in slices:
            if m.sum() < 6:
                continue
            for i, a in enumerate(cands):
                for b in cands[i + 1:]:
                    r = compare(Lq[m][a], Lq[m][b], skill=False)
                    rf = compare(Lf[m][a], Lf[m][b], skill=True)
                    prow.append({"subset": subset, "period": period, "era": era, "model_a": a, "model_b": b,
                                 "n_origins": int(m.sum()), "qlike_rel_a": Lq[m][a].mean(),
                                 "qlike_rel_b": Lq[m][b].mean(), **r,
                                 "frob_a": Lf[m][a].mean(), "frob_b": Lf[m][b].mean(),
                                 "frob_skill_pct": rf["skill_pct"], "frob_skill_lo": rf["skill_lo"],
                                 "frob_skill_hi": rf["skill_hi"], "frob_dm_p": rf["dm_p"]})
    ctx.write("h2h_cov_pairwise.csv", pd.DataFrame(prow))

    # implied correlations of the candidates
    lc = compute_losses("corr", [f"cov:{m}" for m in cands], ctx.targets, results_dir=ctx.results)
    ctx.h2h_corr_pair = lc["pair"]
    co, citems, cmods, SE, ER = corr_cube(lc["pair"], [f"cov:{m}" for m in cands])
    ccells = common_cells(SE)
    rows = []
    for subset in ["daily", *DEFAULT_SUBSETS]:
        c = corr_subset_cells(ccells, citems, subset)
        Lsq, cnt = pool_cells(SE, c)
        L = pd.DataFrame(Lsq, index=co, columns=cmods).dropna()
        Eb = pd.DataFrame(pool_cells(ER, c)[0], index=co, columns=cmods).loc[L.index]
        for period in PERIODS:
            m = period_mask(L.index, period)
            if m.sum() < 6:
                continue
            for mod in cmods:
                row = {"subset": subset, "period": period, "model": mod, "role": ROLE_OF[mod[4:]],
                       "n_origins": int(m.sum()), "n_pairs": int(pd.Series(cnt, index=co).loc[L.index][m].sum()),
                       "mse": L[m][mod].mean(), "bias": Eb[m][mod].mean()}
                if mod != f"cov:{COV_REF}":
                    row.update({k: v for k, v in compare(L[m][mod], L[m][f"cov:{COV_REF}"]).items()
                                if k in ("skill_pct", "skill_lo", "skill_hi", "dm_p")})
                rows.append(row)
    ctx.write("h2h_corr.csv", pd.DataFrame(rows))

    # Model Confidence Sets
    rows = []
    for loss, L in (("vol_qlike_daily", Ls["daily"][[f"cov:{m}" for m in cands]]),
                    ("cov_qlike_rel_daily", Lcov["daily"])):
        for period in ("dev", "holdout"):
            Lp = L[period_mask(L.index, period)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = model_confidence_set(Lp, alpha=0.10, seed=0, warn=False)
            s = r.summary()
            for mod, x in s.iterrows():
                rows.append({"loss": loss, "period": period, "model": mod,
                             "role": ROLE_OF.get(mod.replace("cov:", ""), mod), "mean_loss": x["mean_loss"],
                             "mcs_pvalue": x["mcs_pvalue"], "in_mcs_90": bool(x["in_mcs"]),
                             "elimination_rank": x["elimination_rank"], "n_obs": r.n_obs,
                             "block_length": r.block_length})
    ctx.write("mcs.csv", pd.DataFrame(rows))
    ctx.h2h_L = Ls
    ctx.h2h_Lcov = Lcov


# ------------------------------------------------------------ portfolios
PORT_ERAS = {"dev": (None, str(DEV_END.date())), "holdout": (str(HOLDOUT_START.date()), str(HOLDOUT_END.date())),
             "full": (None, str(HOLDOUT_END.date())),
             **{k: v for k, v in ERAS.items() if k != "pre-1926"}}


def stage_portfolio(ctx: Ctx):
    from assetforecast.eval.portfolio import RealizedCache, load_cov_forecasts, score_portfolios, summarize
    cands = list(CANDIDATES.values())
    fc = load_cov_forecasts(cands, root=ctx.results / "forecasts" / "cov")
    fc = {m: {o: H for o, H in d.items() if o <= HOLDOUT_END} for m, d in fc.items()}
    _log("scoring portfolios (4 candidates)")
    cache = RealizedCache(ctx.panel)
    ctx.port_cache = cache
    res = score_portfolios(ctx.panel, fc, cache=cache)
    ctx.port_res = res
    summ = summarize(res, eras=PORT_ERAS)
    keep_metrics = ["var_ratio", "qlike", "real_var", "pred_vol", "real_vol", "turnover", "vt_abs_err", "gross",
                    "log_ratio"]
    summ = summ[summ["metric"].isin(keep_metrics)]
    # HAC intervals on the same common origins as summarize
    rows = []
    for (port, metric), g in res[res["metric"].isin(keep_metrics)].groupby(["portfolio", "metric"]):
        W = g.pivot_table(index="origin", columns="model", values="value").reindex(columns=cands).dropna()
        for era, (lo, hi) in PORT_ERAS.items():
            m = np.ones(len(W), bool)
            if lo:
                m &= W.index >= pd.Timestamp(lo)
            if hi:
                m &= W.index <= pd.Timestamp(hi)
            if m.sum() < 2:
                continue
            for mod in cands:
                mean, clo, chi, neff = hac_mean_ci(W[m][mod])
                rows.append({"model": mod, "role": ROLE_OF[mod], "portfolio": port, "metric": metric, "era": era,
                             "mean": mean, "ci_lo": clo, "ci_hi": chi, "n_eff": neff, "n_origins": int(m.sum())})
    ci = pd.DataFrame(rows)
    chk = summ.merge(ci, on=["model", "portfolio", "metric", "era"], suffixes=("_summ", ""))
    maxdiff = float((chk["mean_summ"] - chk["mean"]).abs().max())
    ctx.summary["portfolio_summarize_vs_own_mean_maxdiff"] = maxdiff
    ci = ci.rename(columns={"era": "period"})
    ctx.write("portfolio_summary.csv", ci)
    # DM: primary vs others on QLIKE (fixed portfolios) and realized variance (GMV)
    rows = []
    for (port, metric), g in res[res["metric"].isin(["qlike", "real_var", "var_ratio"])].groupby(["portfolio", "metric"]):
        if metric == "real_var" and port not in ("gmv", "gmv_long"):
            continue
        W = g.pivot_table(index="origin", columns="model", values="value").reindex(columns=cands).dropna()
        for period in PERIODS:
            Wp = W[period_mask(W.index, period)]
            if len(Wp) < 6:
                continue
            for b in cands[1:]:
                if metric == "var_ratio":
                    # distance of the ratio from 1 is not a loss; compare |log ratio| instead
                    la, lb_ = np.log(Wp[cands[0]]).abs(), np.log(Wp[b]).abs()
                    r = compare(la, lb_, skill=False)
                    mname = "abs_log_var_ratio"
                else:
                    r = compare(Wp[cands[0]], Wp[b])
                    mname = metric
                rows.append({"portfolio": port, "metric": mname, "period": period, "model_a": cands[0], "model_b": b,
                             "n_origins": len(Wp), "mean_a": (la.mean() if metric == "var_ratio" else Wp[cands[0]].mean()),
                             "mean_b": (lb_.mean() if metric == "var_ratio" else Wp[b].mean()), **r})
    ctx.write("portfolio_dm.csv", pd.DataFrame(rows))
    return ci


# ------------------------------------------------------------------ bridge
def stage_bridge(ctx: Ctx):
    from scipy.stats import norm
    from assetforecast.bridge import ScaleMixture, _anchor, get_bridge, vol_uncertainty
    from assetforecast.eval.portfolio import POLICIES, annual_z_scores, calibration_summary, load_cov_forecasts
    from assetforecast.universe import ASSET_IDS
    prim = CANDIDATES["primary"]
    fc = load_cov_forecasts([prim], root=ctx.results / "forecasts" / "cov")[prim]
    fc = {o: H for o, H in fc.items() if o <= HOLDOUT_END}
    panel = ctx.panel
    _log(f"bridging {len(fc)} origins (identity, vr_market)")
    A = {"identity": {}, "vr_market": {}}
    bridges = {k: get_bridge(k) for k in A}
    s2, memo = {}, {}
    for o, H in fc.items():
        v = panel.asof(o)
        for k, b in bridges.items():
            A[k][o] = b.cov_annual(H, v)
        an = _anchor(v)
        if an not in memo:
            memo[an] = vol_uncertainty(v.asof(an))["s2"]
        s2[o] = memo[an]
    s2 = pd.Series(s2).sort_index()
    pol = {**POLICIES, **{f"a_{a}": {"weights": {a: 1.0}, "required": (a,)} for a in ASSET_IDS}}
    _log("annual z-scores")
    z = annual_z_scores(panel, A, cache=getattr(ctx, "port_cache", None), n_random=10, policies=pol)
    z["origin"] = pd.to_datetime(z["origin"])
    keep_ports = ["p6040", "p_balanced", "p_equity", "ew", "invvol", "erc", "gmv_long", "random_pooled",
                  *[f"a_{a}" for a in ASSET_IDS]]
    rows = []
    for period in PERIODS:
        zp = z[period_mask(z["origin"], period)]
        if zp.empty:
            continue
        cs = calibration_summary(zp)
        cs = cs[cs["portfolio"].isin(keep_ports)].copy()
        cs.insert(0, "period", period)
        rows.append(cs)
    cal = pd.concat(rows, ignore_index=True).rename(columns={"model": "bridge"})
    ctx.write("bridge_calibration.csv", cal)
    # tails
    zz = z[z["metric"] == "z"].copy()
    zz["portfolio"] = zz["portfolio"].astype(str)
    qm = {}
    rows = []
    groups = {"assets_pooled": [f"a_{a}" for a in ASSET_IDS], "policy_pooled": ["p6040", "p_balanced", "p_equity"],
              "p6040": ["p6040"], "p_balanced": ["p_balanced"], "p_equity": ["p_equity"]}
    for br in A:
        g = zz[zz["model"] == br]
        for period in ("dev", "holdout", "full"):
            gp = g[period_mask(g["origin"], period)].copy()
            if gp.empty:
                continue
            gp["dev_"] = gp["value"] - gp.groupby("portfolio")["value"].transform("mean")
            med = float(s2.reindex(pd.DatetimeIndex(gp["origin"].unique())).median())
            mix = ScaleMixture(med)
            exp = {"gt1": mix.share_beyond(1.0), "gt2": mix.share_beyond(2.0), "lt_q05_mix": 0.05, "lt_q01_mix": 0.01}
            exp_n = {"gt1": 2 * norm.sf(1), "gt2": 2 * norm.sf(2), "lt_q05_mix": np.nan, "lt_q01_mix": np.nan}
            for name, ports in groups.items():
                h = gp[gp["portfolio"].isin(ports)]
                if h.empty:
                    continue
                so = s2.reindex(pd.DatetimeIndex(h["origin"])).to_numpy()
                for v in np.unique(so):
                    if v not in qm:
                        mx = ScaleMixture(float(v))
                        qm[v] = (mx.quantile(0.05), mx.quantile(0.01))
                q05 = np.array([qm[v][0] for v in so])
                q01 = np.array([qm[v][1] for v in so])
                d = h["dev_"].to_numpy()
                ind = {"gt1": np.abs(d) > 1, "gt2": np.abs(d) > 2, "lt_q05_mix": d < q05, "lt_q01_mix": d < q01}
                for stat, x in ind.items():
                    W = pd.DataFrame({"origin": h["origin"].to_numpy(), "p": h["portfolio"].to_numpy(),
                                      "x": x.astype(float)}).pivot_table(index="origin", columns="p", values="x")
                    W = W.reindex(pd.date_range(W.index.min(), W.index.max(), freq="ME"))
                    mean, lo, hi = block_bootstrap_mean(W.to_numpy(), block=24, seed=1)
                    rows.append({"bridge": br, "period": period, "group": name, "stat": stat, "n": int(np.isfinite(W.to_numpy()).sum()),
                                 "n_origins": int(W.notna().any(axis=1).sum()), "share": mean, "ci_lo": lo, "ci_hi": hi,
                                 "expected_mixture": exp[stat], "expected_normal": exp_n[stat], "s2_median": med})
    ctx.write("bridge_tails.csv", pd.DataFrame(rows))
    ctx.bridge_cal = cal
    ctx.summary["bridge_s2"] = {"min": float(s2.min()), "median": float(s2.median()), "max": float(s2.max()),
                                "holdout_median": float(s2[period_mask(s2.index, "holdout")].median())}


# ------------------------------------------------------------- episodes
def stage_episodes(ctx: Ctx):
    af = ctx.h2h_af
    prim = f"cov:{CANDIDATES['primary']}"
    w = af.pivot_table(index=["origin", "asset"], columns="model", values="vol_f")
    real = af.drop_duplicates(["origin", "asset"]).set_index(["origin", "asset"])[["vol_r", "proxy"]]
    d = real.join(w).reset_index()
    d = d[period_mask(d["origin"], "holdout")]
    names = {f"cov:{v}": k for k, v in CANDIDATES.items()}
    d = d.rename(columns={**names, "vol_r": "realized"})
    sel = sorted({*[o for o in d["origin"].unique() if pd.Timestamp(o).month == 12],
                  pd.Timestamp("2013-12-31"), pd.Timestamp("2020-02-29")})
    e = d[d["origin"].isin(sel)].copy()
    lab = lambda o: (f"{o.year + 1} calendar year" if o.month == 12 else
                     f"{(o + pd.offsets.MonthBegin(1)).strftime('%Y-%m')}..{(o + pd.offsets.MonthEnd(12)).strftime('%Y-%m')}")
    e["window"] = [lab(pd.Timestamp(o)) for o in e["origin"]]
    for k in ("primary", "runner_up", "single", "reference", "ref_last12m", "ref_blend"):
        if k in e:
            e[f"{k}_ratio"] = e["realized"] / e[k]
    cols = ["origin", "window", "asset", "realized", "proxy", "primary", "runner_up", "single", "reference",
            "ref_last12m", "primary_ratio", "reference_ratio", "runner_up_ratio", "single_ratio"]
    ctx.write("episodes_vol.csv", e[cols].sort_values(["asset", "origin"]))
    d["log_ratio"] = np.log(d["realized"] / d["primary"])
    worst = d.loc[d.groupby("asset")["log_ratio"].apply(lambda s: s.abs().idxmax())]
    worst = worst.assign(kind="worst_per_asset")
    top = d.reindex(d["log_ratio"].abs().sort_values(ascending=False).index).head(15).assign(kind="top15_overall")
    under = d.groupby("asset")["log_ratio"].agg(lambda s: float((s > np.log(1.5)).mean())).rename("share_realized_gt_1p5x")
    cols = ["kind", "origin", "asset", "realized", "primary", "reference", "runner_up", "single", "log_ratio"]
    out = pd.concat([worst[cols], top[cols]], ignore_index=True)
    out = out.merge(under.reset_index(), on="asset", how="left")
    ctx.write("episodes_worst.csv", out)
    # stock-bond correlation
    cp = ctx.h2h_corr_pair
    cp = cp[((cp["asset_i"] == "us_large") & cp["asset_j"].isin(["ust_int", "ust_long"]))]
    cw = cp.pivot_table(index=["origin", "asset_i", "asset_j"], columns="model", values="corr_f")
    cr = cp.drop_duplicates(["origin", "asset_i", "asset_j"]).set_index(["origin", "asset_i", "asset_j"])["corr_r"]
    c = cw.join(cr).reset_index().rename(columns={**names, "corr_r": "realized"})
    c = c[(c["origin"] >= pd.Timestamp("2018-12-31")) & (c["origin"] <= pd.Timestamp("2023-12-31"))]
    c["pair"] = c["asset_i"] + "~" + c["asset_j"]
    ctx.write("episodes_corr.csv", c[["origin", "pair", "realized", "primary", "runner_up", "single", "reference"]]
              .sort_values(["pair", "origin"]))


# ------------------------------------------------------------- dashboard
def stage_dashboard(ctx: Ctx):
    from assetforecast.eval.harness import load_forecasts_long
    T = ctx.targets
    vols = {}
    for role, m in CANDIDATES.items():
        v = load_forecasts_long("vol", f"cov:{m}", ctx.results)
        vols[role] = v.set_index(["origin", "asset"])["vol"]
    df = pd.DataFrame(vols)
    # family means of the non-combo vol models (mean variance within family)
    base = [m for m in ctx.models["vol"] if not m.startswith("combo_")]
    fam = {}
    for m in base:
        v = load_forecasts_long("vol", m, ctx.results).set_index(["origin", "asset"])["vol"] ** 2
        fam.setdefault(family_of(m), []).append(v)
    F = pd.DataFrame({f: pd.concat(vs, axis=1).mean(axis=1) for f, vs in fam.items()}) ** 0.5
    df["fam_p10"] = F.quantile(0.10, axis=1)
    df["fam_p90"] = F.quantile(0.90, axis=1)
    df["n_families"] = F.notna().sum(axis=1)
    tv = T.vol.set_index(["origin", "asset"])[["vol", "proxy"]].rename(columns={"vol": "realized"})
    df = df.join(tv, how="left").reset_index()
    df["period"] = np.where(period_mask(df["origin"], "dev"), "dev",
                            np.where(period_mask(df["origin"], "holdout"), "holdout", "live"))
    df.loc[df["origin"] > HOLDOUT_END, "period"] = "live"
    df["origin"] = pd.DatetimeIndex(df["origin"]).strftime("%Y-%m")
    cols = ["origin", "asset", "period", "realized", "proxy", "primary", "runner_up", "single", "reference",
            "fam_p10", "fam_p90", "n_families"]
    ctx.write("dashboard_vol.csv", df[cols], digits=4)
    ctx.summary["dashboard_vol_families"] = sorted(fam)
    # correlations of key pairs
    rows = []
    corr = {}
    for role, m in CANDIDATES.items():
        c = load_forecasts_long("corr", f"cov:{m}", ctx.results)
        corr[role] = c.set_index(["origin", "asset_i", "asset_j"])["corr"]
    C = pd.DataFrame(corr)
    R = T.corr.set_index(["origin", "asset_i", "asset_j"])["corr"].rename("realized")
    C = C.join(R, how="left").reset_index()
    sel = []
    for a, b in KEY_PAIRS:
        # forecasts hold both orientations; targets only (i, j) in panel order, as KEY_PAIRS are
        s = C[(C.asset_i == a) & (C.asset_j == b)].copy()
        if s.duplicated(["origin"]).any():
            raise ValueError(f"duplicate origins for pair {a}~{b}")
        s["pair"] = f"{a}~{b}"
        sel.append(s)
    C = pd.concat(sel, ignore_index=True)
    C["origin"] = pd.DatetimeIndex(C["origin"]).strftime("%Y-%m")
    ctx.write("dashboard_corr.csv", C[["origin", "pair", "realized", "primary", "runner_up", "single", "reference"]],
              digits=4)
    # portfolios
    res = ctx.port_res
    r = res[res["portfolio"].isin(["p6040", "p_balanced"]) & res["metric"].isin(["pred_vol", "real_vol"])]
    W = r.pivot_table(index=["origin", "portfolio"], columns=["metric", "model"], values="value")
    out = pd.DataFrame({"realized": W[("real_vol", CANDIDATES["primary"])]})
    for role, m in CANDIDATES.items():
        out[role] = W[("pred_vol", m)]
    out = out.reset_index()
    out["origin"] = pd.DatetimeIndex(out["origin"]).strftime("%Y-%m")
    ctx.write("dashboard_portfolio.csv", out, digits=4)


# -------------------------------------------------------------- accuracy
def stage_accuracy(ctx: Ctx):
    af = ctx.h2h_af
    prim = f"cov:{CANDIDATES['primary']}"
    d = af[af["model"] == prim].copy()
    d["lr"] = np.log(d["vol_r"] / d["vol_f"])
    nd = noise_sd_logvol(252)
    nd295 = noise_sd_logvol(295)
    npx = noise_sd_logvol_proxy(12)
    samples = {"full": period_mask(d["origin"], "full"),
               "last30y": period_mask(d["origin"], "full") & np.asarray(d["origin"] >= LAST30_START),
               "dev": period_mask(d["origin"], "dev"), "holdout": period_mask(d["origin"], "holdout")}
    rows = []
    for name, m in samples.items():
        dm = d[m]
        for a, g in [("pooled", dm), *dm.groupby("asset")]:
            s = log_ratio_summary(g["lr"])
            share_px = float(g["proxy"].mean())
            nvar = (1 - share_px) * nd ** 2 + share_px * npx ** 2
            s.update({"sample": name, "asset": a, "share_proxy": share_px, "noise_sd": math.sqrt(nvar),
                      "forecast_err_sd": math.sqrt(max(s.get("sd", np.nan) ** 2 - nvar, 0.0))
                      if s["n"] > 1 else np.nan})
            fe = s["forecast_err_sd"]
            # two-thirds band of the TRUE vol around the forecast if errors were normal with that sd
            s["two_thirds_true_lo_pct"] = 100 * (math.exp(s["mean"] - 0.9674 * fe) - 1) if np.isfinite(fe) else np.nan
            s["two_thirds_true_hi_pct"] = 100 * (math.exp(s["mean"] + 0.9674 * fe) - 1) if np.isfinite(fe) else np.nan
            rows.append(s)
    out = pd.DataFrame(rows)
    out = out[["sample", "asset", "n", "share_proxy", "mean", "sd", "q05", "q1_6", "median", "q5_6", "q95",
               "two_thirds_lo_pct", "two_thirds_hi_pct", "ninety_lo_pct", "ninety_hi_pct", "noise_sd",
               "forecast_err_sd", "two_thirds_true_lo_pct", "two_thirds_true_hi_pct"]]
    ctx.write("accuracy.csv", out)
    ctx.summary["noise_sd_logvol"] = {"daily_252": nd, "daily_295": nd295, "monthly_proxy": npx}


# --------------------------------------------------------- data correction
def stage_correction(ctx: Ctx):
    from assetforecast.eval.harness import load_forecasts_long, stored_models
    from assetforecast.eval.losses import compute_losses
    from assetforecast.eval.targets import load_targets
    dev = ctx.dev_results
    if dev is None or not (dev / "targets" / "meta.json").exists():
        _log("no pre-fix dev results; skipping the data-correction comparison")
        return
    Tpre = load_targets(dev)
    rows = []
    for kind in ("vol", "corr", "cov"):
        both = [m for m in stored_models(kind, dev) if m in set(ctx.models[kind])]
        if not both:
            continue
        _log(f"data correction: {kind}, {len(both)} models")
        pre = compute_losses(kind, both, Tpre, results_dir=dev)
        post = compute_losses(kind, both, ctx.targets, results_dir=ctx.results)
        subsets = ["daily", "all"] if kind == "vol" else ["daily"]
        for subset in subsets:
            res = {}
            for tag, L in (("pre", pre), ("post", post)):
                if kind == "vol":
                    af = L["asset"]
                    af = af[af["origin"] <= DEV_END]
                    o, items, mods, Q, R, LE, PX = vol_cube(af, both)
                    c = vol_subset_cells(common_cells(Q), PX, items, subset)
                    M, cnt = pool_cells(Q, c)
                    Rm, _ = pool_cells(R, c)
                    res[tag] = (pd.DataFrame(M, index=o, columns=mods).dropna(),
                                pd.DataFrame(Rm, index=o, columns=mods))
                elif kind == "corr":
                    pf = L["pair"]
                    pf = pf[pf["origin"] <= DEV_END]
                    o, items, mods, SE, ER = corr_cube(pf, both)
                    M, cnt = pool_cells(SE, common_cells(SE))
                    res[tag] = (pd.DataFrame(M, index=o, columns=mods).dropna(), None)
                else:
                    p = L["pooled"]
                    p = p[(p["subset"] == subset) & (p["origin"] <= DEV_END)]
                    W = p.pivot_table(index="origin", columns="model", values="qlike_rel").reindex(columns=both).dropna()
                    res[tag] = (W, None)
            Lpre, Lpost = res["pre"][0], res["post"][0]
            refs = {"vol": VOL_REFS, "corr": CORR_REFS, "cov": [COV_REF]}[kind]
            rk_pre, rk_post = Lpre.mean().rank(), Lpost.mean().rank()
            ref_pre = min([r for r in refs if r in Lpre], key=lambda r: Lpre[r].mean(), default=None)
            ref_post = min([r for r in refs if r in Lpost], key=lambda r: Lpost[r].mean(), default=None)
            for m in both:
                row = {"kind": kind, "model": m, "family": family_of(m), "subset": subset,
                       "n_origins_pre": len(Lpre), "n_origins_post": len(Lpost),
                       "loss_pre": Lpre[m].mean(), "loss_post": Lpost[m].mean(),
                       "rank_pre": rk_pre[m], "rank_post": rk_post[m], "ref_pre": ref_pre, "ref_post": ref_post}
                row["change_pct"] = 100 * (row["loss_post"] / row["loss_pre"] - 1) if kind != "cov" else np.nan
                row["change_abs"] = row["loss_post"] - row["loss_pre"]
                if kind == "vol":
                    row["rf_pre"] = res["pre"][1].loc[Lpre.index, m].mean()
                    row["rf_post"] = res["post"][1].loc[Lpost.index, m].mean()
                if kind != "cov" and ref_pre and m != ref_pre:
                    row["skill_vs_ref_pre"] = 100 * (1 - Lpre[m].mean() / Lpre[ref_pre].mean())
                if kind != "cov" and ref_post and m != ref_post:
                    row["skill_vs_ref_post"] = 100 * (1 - Lpost[m].mean() / Lpost[ref_post].mean())
                if kind == "cov":
                    row["diff_vs_ref_pre"] = Lpre[m].mean() - Lpre[COV_REF].mean() if COV_REF in Lpre else np.nan
                    row["diff_vs_ref_post"] = Lpost[m].mean() - Lpost[COV_REF].mean() if COV_REF in Lpost else np.nan
                rows.append(row)
        if kind == "vol":
            # how much the forecasts themselves moved (same model, same dev origins, corrected data)
            ch = []
            for m in both:
                a = load_forecasts_long("vol", m, dev).set_index(["origin", "asset"])["vol"]
                b = load_forecasts_long("vol", m, ctx.results).set_index(["origin", "asset"])["vol"]
                j = pd.concat({"a": a, "b": b}, axis=1).dropna()
                j = j[j.index.get_level_values(0) <= DEV_END]
                x = np.log(j["b"] / j["a"]).abs()
                ch.append({"model": m, "fc_mean_abs_log_change": float(x.mean()), "fc_max_abs_log_change": float(x.max()),
                           "fc_share_changed_1pct": float((x > 0.01).mean())})
            chd = pd.DataFrame(ch)
            for r in rows:
                if r["kind"] == "vol":
                    hit = chd[chd["model"] == r["model"]]
                    if len(hit):
                        r.update(hit.iloc[0].drop("model").to_dict())
    ctx.write("data_correction.csv", pd.DataFrame(rows))


# ------------------------------------------------------------ verification
VERIFY_ORIGINS = ["1931-06-30", "1938-12-31", "1949-06-30", "1962-12-31", "1974-06-30", "1987-09-30",
                  "1999-12-31", "2008-08-31", "2013-12-31", "2016-06-30", "2020-02-29", "2021-12-31"]
VERIFY_ASSETS = ["us_large", "ust_long", "gold", "ig_credit"]


def _independent_window(panel, origin, assets):
    """Own computation of the 21-day realized matrix and coverage for one window."""
    o = pd.Timestamp(origin)
    end = o + pd.offsets.MonthEnd(12)
    R = panel.daily.loc[(panel.daily.index > o) & (panel.daily.index <= end)]
    cash = panel.cash_daily.loc[R.index]
    X = np.log1p(R[assets]).sub(np.log1p(cash), axis=0)
    trade = R.notna().any(axis=1)                       # rows with any asset (all 12) trading
    n_days = int(trade.sum())
    n_rows = len(R)
    ok = {}
    for a in assets:
        cnt = X[a].notna()
        per_m = cnt.groupby(X.index.to_period("M")).sum()
        ok[a] = bool(n_days > 0 and cnt.sum() >= 0.9 * n_days and len(per_m) == 12 and per_m.min() >= 10)
    var, G = kday_var(X.to_numpy(), 21)
    # monthly proxy
    m = panel.monthly.loc[(panel.monthly.index > o) & (panel.monthly.index <= end), assets]
    cm = panel.cash_monthly.loc[m.index]
    xm = np.log1p(m).sub(np.log1p(cm), axis=0)
    proxy = (xm ** 2).mean() * 12
    proxy[xm.notna().sum() < 12] = np.nan
    return var, G, ok, proxy, n_rows


def stage_verify(ctx: Ctx):
    panel, T = ctx.panel, ctx.targets
    prim = CANDIDATES["primary"]
    H = pd.read_parquet(ctx.results / "forecasts" / "cov" / f"{prim}.parquet")
    H["origin"] = pd.to_datetime(H["origin"])
    hv = pd.read_parquet(ctx.results / "forecasts" / "vol" / "combo_vol_hier_trim.parquet")
    hv["origin"] = pd.to_datetime(hv["origin"])
    af = ctx.h2h_af
    af = af[af["model"] == f"cov:{prim}"].set_index(["origin", "asset"])
    tv = T.vol.set_index(["origin", "asset"])
    rows = []
    for os_ in VERIFY_ORIGINS:
        o = pd.Timestamp(os_)
        var, G, ok, proxy, n_rows = _independent_window(panel, o, VERIFY_ASSETS)
        for j, a in enumerate(VERIFY_ASSETS):
            if (o, a) not in tv.index:
                continue
            t = tv.loc[(o, a)]
            mine = math.sqrt(var[j]) if ok[a] else (math.sqrt(proxy[a]) if np.isfinite(proxy[a]) else np.nan)
            rows.append({"check": "target_vol", "origin": o.date(), "asset": a, "n_rows": n_rows,
                         "proxy_mine": not ok[a], "proxy_stored": bool(t["proxy"]), "independent": mine,
                         "stored": float(t["vol"])})
            g = H[(H["origin"] == o) & (H["asset_i"] == a) & (H["asset_j"] == a)]
            if len(g) and np.isfinite(mine):
                vf = float(g["cov"].iloc[0])
                q = float(qlike_var(mine ** 2, vf))
                stored = float(af.loc[(o, a), "qlike"]) if (o, a) in af.index else np.nan
                rows.append({"check": "qlike_primary", "origin": o.date(), "asset": a, "n_rows": n_rows,
                             "independent": q, "stored": stored})
                hh = hv[(hv["origin"] == o) & (hv["asset"] == a)]
                if len(hh):
                    rows.append({"check": "cov_diag_vs_combo_vol_hier_trim", "origin": o.date(), "asset": a,
                                 "independent": math.sqrt(vf), "stored": float(hh["vol"].iloc[0])})
    # correlation us_large ~ ust_int
    for os_ in ["1975-06-30", "2008-08-31", "2021-12-31"]:
        o = pd.Timestamp(os_)
        var, G, ok, proxy, n_rows = _independent_window(panel, o, ["us_large", "ust_int"])
        mine = G[0, 1] / math.sqrt(G[0, 0] * G[1, 1])
        t = T.corr[(T.corr["origin"] == o) & (T.corr["asset_i"] == "us_large") & (T.corr["asset_j"] == "ust_int")]
        rows.append({"check": "target_corr us_large~ust_int", "origin": o.date(), "asset": "pair", "n_rows": n_rows,
                     "independent": mine, "stored": float(t["corr"].iloc[0]) if len(t) else np.nan})
    # covariance loss of the primary on the complete daily set
    from assetforecast.universe import ASSET_IDS
    for os_ in ["1990-06-30", "2016-06-30"]:
        o = pd.Timestamp(os_)
        assets = list(ASSET_IDS)
        var, G, ok, proxy, n_rows = _independent_window(panel, o, assets)
        S_assets = [a for a in assets if ok[a] and var[assets.index(a)] > 0]
        g = H[H["origin"] == o]
        Hm = g.pivot(index="asset_i", columns="asset_j", values="cov")
        S_assets = [a for a in S_assets if a in Hm.index]
        ix = [assets.index(a) for a in S_assets]
        S = G[np.ix_(ix, ix)]
        q = qlike_rel_independent(Hm.loc[S_assets, S_assets].to_numpy(), S)
        p = ctx.cov_pooled
        st = p[(p["model"] == prim) & (p["origin"] == o) & (p["subset"] == "daily")]
        rows.append({"check": f"cov_qlike_rel_primary (dim {len(S_assets)})", "origin": o.date(), "asset": "daily set",
                     "n_rows": n_rows, "independent": q, "stored": float(st["qlike_rel"].iloc[0]) if len(st) else np.nan,
                     "dim_stored": int(st["dim"].iloc[0]) if len(st) else None})
    v = pd.DataFrame(rows)
    v["abs_diff"] = (v["independent"] - v["stored"]).abs()
    v["rel_diff"] = v["abs_diff"] / v["stored"].abs()
    ctx.write("verification.csv", v, digits=10)
    # sanity: known holdout episodes
    tvv = T.vol
    s = {}
    for a, o in (("us_large", "2019-12-31"), ("us_large", "2020-02-29"), ("us_large", "2016-12-31"),
                 ("ust_long", "2021-12-31"), ("ust_long", "2016-12-31"), ("ust_int", "2021-12-31")):
        r = tvv[(tvv.asset == a) & (tvv.origin == pd.Timestamp(o))]
        s[f"{a} {o}"] = float(r["vol"].iloc[0]) if len(r) else None
    ctx.summary["sanity_realized_vol"] = s
    ctx.summary["verification_max_rel_diff"] = float(v["rel_diff"].max())


# ------------------------------------------------------------ expectations
def stage_expectations(ctx: Ctx):
    t = ctx.tables
    rows = []
    prim = f"cov:{CANDIDATES['primary']}"
    pw = t["h2h_vol_pairwise.csv"]
    def pr(period, b):
        r = pw[(pw.subset == "daily") & (pw.period == period) & (pw.era == "all") & (pw.model_a == prim) & (pw.model_b == b)]
        return r.iloc[0]
    # 1
    h = {b: pr("holdout", b) for b in H2H_VOL_EXTRA}
    dv = {b: pr("dev", b) for b in H2H_VOL_EXTRA}
    best_h = min(H2H_VOL_EXTRA, key=lambda b: h[b]["qlike_b"])
    best_d = min(H2H_VOL_EXTRA, key=lambda b: dv[b]["qlike_b"])
    lower = all(h[b]["qlike_a"] < h[b]["qlike_b"] for b in H2H_VOL_EXTRA)
    sig = h[best_h]["dm_p"] < 0.10
    reversal = (h[best_h]["skill_pct"] < -dv[best_d]["skill_pct"]) if not lower else False
    status = "met" if lower and sig else ("met (point estimate; not significant)" if lower else
                                          ("not met: reversal larger than the development gap" if reversal else "not met"))
    txt = "; ".join(f"holdout QLIKE primary {h[b]['qlike_a']:.3f} vs {b} {h[b]['qlike_b']:.3f}, skill "
                    f"{h[b]['skill_pct']:+.1f}% [{h[b]['skill_lo']:+.1f}, {h[b]['skill_hi']:+.1f}], p={h[b]['dm_p']:.2f}"
                    for b in H2H_VOL_EXTRA)
    txt += "; dev: " + "; ".join(f"vs {b} {dv[b]['skill_pct']:+.1f}% [{dv[b]['skill_lo']:+.1f}, {dv[b]['skill_hi']:+.1f}]"
                                for b in H2H_VOL_EXTRA)
    rows.append({"id": 1, "expectation": "primary vol QLIKE below the best reference (ref_last12m, ref_blend) on the holdout",
                 "status": status, "evidence": txt})
    # 2
    cp = t["h2h_cov_pairwise.csv"]
    r = cp[(cp.subset == "daily") & (cp.period == "holdout") & (cp.era == "all") &
           (cp.model_a == CANDIDATES["primary"]) & (cp.model_b == COV_REF)].iloc[0]
    rd = cp[(cp.subset == "daily") & (cp.period == "dev") & (cp.era == "all") &
            (cp.model_a == CANDIDATES["primary"]) & (cp.model_b == COV_REF)].iloc[0]
    lower = r["qlike_rel_a"] < r["qlike_rel_b"]
    status = ("met" if r["dm_p"] < 0.10 else "met (point estimate; not significant)") if lower else "not met"
    rows.append({"id": 2, "expectation": "primary qlike_rel below pair_cov_ref on the holdout", "status": status,
                 "evidence": f"holdout qlike_rel {r['qlike_rel_a']:.3f} vs {r['qlike_rel_b']:.3f}, diff "
                             f"{r['mean_diff']:+.3f} [{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}], p={r['dm_p']:.3f}, "
                             f"n={int(r['n_origins'])}; dev diff {rd['mean_diff']:+.3f} p={rd['dm_p']:.3f}"})
    # 3
    ps = t["portfolio_summary.csv"]
    sel = ps[(ps.model == CANDIDATES["primary"]) & (ps.metric == "var_ratio") & (ps.portfolio.isin(POLICY_FIXED))]
    hv = sel[sel.period == "holdout"].set_index("portfolio")
    dvv = sel[sel.period == "dev"].set_index("portfolio")
    inside = all(verdict_between(hv.at[p, "mean"], 0.75, 1.35) for p in POLICY_FIXED)
    rows.append({"id": 3, "expectation": "fixed-weight portfolios (60/40, balanced, risk parity, random): mean RV/PV "
                                        "in [0.75, 1.35] on the holdout",
                 "status": "met" if inside else "not met",
                 "evidence": "; ".join(f"{p} {hv.at[p, 'mean']:.2f} [{hv.at[p, 'ci_lo']:.2f}, {hv.at[p, 'ci_hi']:.2f}] "
                                       f"(dev {dvv.at[p, 'mean']:.2f})" for p in POLICY_FIXED)})
    # 4
    sel = ps[(ps.model == CANDIDATES["primary"]) & (ps.metric == "var_ratio") & (ps.portfolio.isin(["gmv", "gmv_long"]))]
    hv = sel[sel.period == "holdout"].set_index("portfolio")
    dvv = sel[sel.period == "dev"].set_index("portfolio")
    above = all(hv.at[p, "mean"] > 1.2 for p in ("gmv", "gmv_long"))
    st = "met" if above else ("partly met" if any(hv.at[p, "mean"] > 1.2 for p in ("gmv", "gmv_long")) else "not met")
    rows.append({"id": 4, "expectation": "minimum-variance portfolios realize more variance than predicted, factor > 1.2",
                 "status": st,
                 "evidence": "; ".join(f"{p} {hv.at[p, 'mean']:.2f} [{hv.at[p, 'ci_lo']:.2f}, {hv.at[p, 'ci_hi']:.2f}] "
                                       f"(dev {dvv.at[p, 'mean']:.2f})" for p in ("gmv", "gmv_long"))})
    # 5
    bc = t["bridge_calibration.csv"]
    ev, closer = [], []
    for p in ("p6040", "p_balanced", "p_equity"):
        g = bc[(bc.period == "holdout") & (bc.portfolio == p)].set_index("bridge")
        if not {"identity", "vr_market"} <= set(g.index):
            continue
        a, b = g.loc["identity"], g.loc["vr_market"]
        closer.append(abs(b["var_z"] - 1) < abs(a["var_z"] - 1))
        ev.append(f"{p}: identity {a['var_z']:.2f} [{a['ci_lo']:.2f}, {a['ci_hi']:.2f}], vr_market {b['var_z']:.2f} "
                  f"[{b['ci_lo']:.2f}, {b['ci_hi']:.2f}], chi2 (non-overlap, ~{b['n_nonoverlap']:.0f} yrs) "
                  f"[{b['chi2_lo']:.2f}, {b['chi2_hi']:.2f}]")
    if all(closer):
        st = "met in point estimate (vr_market nearer 1 for all); inconclusive statistically"
    elif not any(closer):
        st = "not met in point estimate (vr_market further from 1 for all); inconclusive statistically"
    else:
        st = "mixed (vr_market nearer 1 for some policy portfolios only); inconclusive statistically"
    rows.append({"id": 5, "expectation": "with the bridge, Var(z) of policy portfolios nearer 1 than without",
                 "status": st, "evidence": "; ".join(ev)})
    ctx.write("expectations.csv", pd.DataFrame(rows))
    ctx.summary["expectations"] = {int(r["id"]): r["status"] for r in rows}
    # headline: primary vs the other candidates and the vol references, daily subset
    head = {}
    hv = t["h2h_vol.csv"]
    for period in ("dev", "holdout"):
        h = {"vol_qlike": {}, "vol_vs_primary": {}, "cov_qlike_rel": {}, "cov_vs_primary": {}}
        g = hv[(hv.subset == "daily") & (hv.period == period) & (hv.era == "all")]
        h["n_origins"] = int(g["n_origins"].iloc[0])
        for _, r in g.iterrows():
            h["vol_qlike"][r["role"]] = round(float(r["qlike"]), 4)
        for b in [f"cov:{m}" for m in list(CANDIDATES.values())[1:]] + H2H_VOL_EXTRA:
            r = pw[(pw.subset == "daily") & (pw.period == period) & (pw.era == "all") & (pw.model_a == prim) &
                   (pw.model_b == b)].iloc[0]
            h["vol_vs_primary"][ROLE_OF.get(b[4:], b)] = {
                "primary_skill_pct": round(float(r["skill_pct"]), 2), "ci90": [round(float(r["skill_lo"]), 2),
                                                                           round(float(r["skill_hi"]), 2)],
                "dm_p": round(float(r["dm_p"]), 4)}
        for b in list(CANDIDATES.values())[1:]:
            r = cp[(cp.subset == "daily") & (cp.period == period) & (cp.era == "all") &
                   (cp.model_a == CANDIDATES["primary"]) & (cp.model_b == b)].iloc[0]
            h["cov_qlike_rel"][ROLE_OF[b]] = round(float(r["qlike_rel_b"]), 4)
            h["cov_qlike_rel"]["primary"] = round(float(r["qlike_rel_a"]), 4)
            h["cov_vs_primary"][ROLE_OF[b]] = {"diff": round(float(r["mean_diff"]), 4),
                                               "ci90": [round(float(r["diff_lo"]), 4), round(float(r["diff_hi"]), 4)],
                                               "dm_p": round(float(r["dm_p"]), 4)}
        head[period] = h
    ctx.summary["headline"] = head


# ================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.environ.get("ASSETFORECAST_RESULTS", str(ROOT / "results" / "full")))
    ap.add_argument("--out", default=str(ROOT / "results" / "final"))
    ap.add_argument("--dev-results", default=str(ROOT / "results" / "dev"),
                    help="pre-correction development results (forecasts + targets); '' to skip")
    ap.add_argument("--stages", nargs="*", default=None, help="subset of stages (debugging)")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = Ctx(Path(a.results), out, Path(a.dev_results) if a.dev_results else None)
    stages = [("coverage", stage_coverage), ("vol", stage_vol), ("corr", stage_corr), ("cov", stage_cov),
              ("h2h", stage_h2h), ("portfolio", stage_portfolio), ("bridge", stage_bridge),
              ("episodes", stage_episodes), ("dashboard", stage_dashboard), ("accuracy", stage_accuracy),
              ("correction", stage_correction), ("verify", stage_verify), ("expectations", stage_expectations)]
    for name, fn in stages:
        if a.stages and name not in a.stages:
            continue
        t1 = time.time()
        _log(f"stage {name}")
        fn(ctx)
        ctx.summary.setdefault("stage_seconds", {})[name] = round(time.time() - t1, 1)
    ctx.summary["run_seconds"] = round(time.time() - t0, 1)
    ctx.summary["generated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if not a.stages:
        (out / "summary.json").write_text(json.dumps(ctx.summary, indent=2, default=str))
    _log("done")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
