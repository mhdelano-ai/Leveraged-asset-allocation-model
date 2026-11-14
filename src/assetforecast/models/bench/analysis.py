"""Tables for reports/models/bench.md.

    ASSETFORECAST_MAX_DATE=2014-12-31 OMP_NUM_THREADS=1 \
        python -m assetforecast.models.bench.analysis [--diag-dir DIR] [--step 1] [--out FILE]

1. Runs the DIAGNOSTIC variants (the a-priori half-life grid 6/12/36/60/120
   months for monthly and 21-day-sum inputs, the 1-day x ratio input, the
   dropped variants) into ``--diag-dir`` (default: a scratch folder, never
   ``results/``), skipping origins already stored.
2. Scores registered bench models (read from the results directory, so run
   ``python -m assetforecast.eval.run forecast`` for them first) against the
   reference models and prints markdown tables: overall, eras, assets, skill
   and DM tests against the best reference, the half-life curves by asset
   group, and daily-versus-monthly tests.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ...data.panel import load_panel
from ...eval.harness import load_forecasts_long, run_model
from ...eval.inference import dm_test, panel_dm_test, skill_score
from ...eval.losses import corr_losses, cov_losses, vol_losses
from ...eval.targets import get_targets
from ...paths import RESULTS
from ...universe import BY_ID
from . import CORR_MODELS, VOL_MODELS
from .corr import EWCorr, ShrinkCorr
from .vol import DailyRatioVol, EWVol, MREstVol

GRID = [6, 12, 36, 60, 120]
ERAS = ["1925-12-31", "1945-12-31", "1979-12-31", "1999-12-31"]
REF_VOL = ["ref_longrun", "ref_last12m", "ref_last60m", "ref_blend"]
REF_CORR = ["ref_corr_longrun", "ref_corr_last120m", "ref_corr_zero"]
COMPOSITE_VOL = "bench_vol_d_hl12"   # common vol paired with every corr model for cov qlike_rel
GROUPS = ["equity", "rates", "credit", "real"]


def diag_vol():
    d = {}
    for h in GRID:
        d[f"diag_vol_m_hl{h}"] = EWVol(h, daily=False)
        d[f"diag_vol_d_hl{h}"] = EWVol(h, daily=True)
    for h in (6, 12, 36):
        d[f"diag_vol_d1r_hl{h}"] = DailyRatioVol(h)
    d["diag_vol_mr_est_free"] = MREstVol(constrained=False)
    return d


def diag_corr():
    d = {}
    for h in GRID:
        d[f"diag_corr_m_hl{h}"] = EWCorr(h, daily=False)
        d[f"diag_corr_d_hl{h}"] = EWCorr(h, daily=True)
    d["diag_corr_shrink_const"] = ShrinkCorr(36, "const")
    return d


def run_diag(panel, diag_dir: Path, step: int):
    for kind, models in (("vol", diag_vol()), ("corr", diag_corr())):
        for name, m in models.items():
            m.name = name
            t0 = time.perf_counter()
            r = run_model(m, kind, panel, results_dir=diag_dir, step=step)
            print(f"<!-- diag {kind}/{name}: {r.n_new} new in {time.perf_counter() - t0:.0f}s -->", flush=True)


# ------------------------------------------------------------------ helpers
def _load(kind, name, diag_dir):
    for d in (RESULTS, diag_dir):
        try:
            return load_forecasts_long(kind, name, d)
        except Exception:
            continue
    raise FileNotFoundError(f"{kind}/{name}")


def _md(df: pd.DataFrame, fmt="{:.4f}") -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else fmt.format(v)
        return str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for i, row in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [f(v) for v in row.values]) + " |")
    return "\n".join(lines)


def _era(o: pd.Series) -> pd.Series:
    b = pd.to_datetime(ERAS)
    lab = np.array(["pre-1926", "1926-1945", "1946-1979", "1980-1999", "2000-2013"], dtype=object)
    return pd.Series(lab[np.searchsorted(b.values, o.values, side="left")], index=o.index)


def _vol_frames(names, T, diag_dir):
    A, P = [], []
    for n in names:
        out = vol_losses(_load("vol", n, diag_dir), T, n)
        A.append(out["asset"])
        P.append(out["pooled"])
    return pd.concat(A, ignore_index=True), pd.concat(P, ignore_index=True)


def _wide(P, subset, metric):
    return P[P["subset"] == subset].pivot_table(index="origin", columns="model", values=metric).dropna()


def _common_cells(A, names, daily_only):
    a = A[A["model"].isin(names)]
    if daily_only:
        a = a[~a["proxy"]]
    w = a.pivot_table(index=["origin", "asset"], columns="model", values="qlike").dropna()
    return w[names]


def _test_vs(w: pd.DataFrame, model: str, bench: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = skill_score(w[model], w[bench], warn=False)
        d = dm_test(w[model], w[bench], warn=False)
    return s.skill, s.ci_low, s.ci_high, d.pvalue


def _panel_test(cells: pd.DataFrame, a: str, b: str):
    long = cells[[a, b]].stack().rename("loss").reset_index().rename(columns={"level_2": "model"})
    long.columns = ["origin", "asset", "model", "loss"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = panel_dm_test(long, a, b, warn=False)
    return r.mean_diff, r.pvalue


# --------------------------------------------------------------------- vol
def vol_tables(T, diag_dir) -> str:
    out = []
    reg = list(VOL_MODELS)
    A, P = _vol_frames(reg + REF_VOL, T, diag_dir)
    for subset in ("daily", "all"):
        w = _wide(P, subset, "qlike")[reg + REF_VOL]
        best = w[REF_VOL].mean().idxmin()
        rows = {}
        for m in reg + REF_VOL:
            sk, lo, hi, p = _test_vs(w, m, best) if m != best else (0.0, np.nan, np.nan, np.nan)
            rows[m] = dict(qlike=w[m].mean(), skill_pct=sk, ci90_lo=lo, ci90_hi=hi, dm_p=p)
        t = pd.DataFrame(rows).T.sort_values("qlike")
        t.index.name = "model"
        out.append(f"#### Vol, subset `{subset}`: {len(w)} common origins "
                   f"{w.index.min():%Y-%m}..{w.index.max():%Y-%m}; skill and DM vs best ref `{best}`\n\n"
                   + _md(t, "{:.3f}"))
        e = w.groupby(_era(w.index.to_series()).values).mean().T
        e.loc["n_origins"] = w.groupby(_era(w.index.to_series()).values).size()
        e.index.name = "model"
        out.append(f"#### Vol QLIKE by era, subset `{subset}`\n\n" + _md(e, "{:.3f}"))
        cells = _common_cells(A[A["origin"].isin(w.index)], reg + REF_VOL, subset == "daily")
        ba = cells.groupby(level="asset").mean().T
        ba = ba[[a for a in BY_ID if a in ba.columns]]
        ba.loc["n_cells"] = cells.groupby(level="asset").size()
        ba.index.name = "model"
        out.append(f"#### Vol QLIKE by asset, subset `{subset}`\n\n" + _md(ba, "{:.3f}"))
    return "\n\n".join(out)


def halflife_tables(T, diag_dir) -> str:
    out = []
    names = [f"diag_vol_{s}_hl{h}" for s in ("m", "d") for h in GRID]
    A, P = _vol_frames(names, T, diag_dir)
    for subset in ("daily", "all"):
        w = _wide(P, subset, "qlike")[names]
        cells = _common_cells(A[A["origin"].isin(w.index)], names, subset == "daily")
        grp = cells.index.get_level_values("asset").map(lambda a: BY_ID[a].group)
        rows = {}
        for s in ("m", "d"):
            for label, data in [("overall (pooled)", w)] + [(g, cells[grp == g]) for g in GROUPS]:
                if not len(data):
                    continue
                cols = [f"diag_vol_{s}_hl{h}" for h in GRID]
                mean = data[cols].mean()
                rows[(s, label)] = {f"hl{h}": mean[c] for h, c in zip(GRID, cols)}
                rows[(s, label)]["best"] = f"hl{GRID[int(np.argmin(mean.values))]}"
                rows[(s, label)]["n"] = len(data)
        t = pd.DataFrame(rows).T
        t.index = [f"{'monthly' if s == 'm' else '21d-sum'} / {g}" for s, g in t.index]
        t.index.name = "input / group"
        out.append(f"#### QLIKE by half-life (months), subset `{subset}`, common origins "
                   f"{w.index.min():%Y-%m}..{w.index.max():%Y-%m}\n\n" + _md(t, "{:.3f}"))
        # flatness: each hl vs the best hl of the same input, DM p-value (pooled series)
        fl = {}
        for s in ("m", "d"):
            cols = [f"diag_vol_{s}_hl{h}" for h in GRID]
            b = w[cols].mean().idxmin()
            for h, c in zip(GRID, cols):
                sk, lo, hi, p = _test_vs(w, c, b) if c != b else (0.0, np.nan, np.nan, np.nan)
                fl[(s, h)] = dict(pct_worse=-sk, ci90_lo=-hi, ci90_hi=-lo, dm_p=p)
        f = pd.DataFrame(fl).T
        f.index = [f"{'monthly' if s == 'm' else '21d-sum'} hl{h}" for s, h in f.index]
        f.index.name = "variant"
        out.append(f"#### Flatness, subset `{subset}`: % higher mean QLIKE than the best half-life "
                   f"of the same input (90% CI), DM p-value\n\n" + _md(f, "{:.3f}"))
    return "\n\n".join(out)


def daily_tables(T, diag_dir) -> str:
    out = []
    names = ([f"diag_vol_{s}_hl{h}" for s in ("m", "d") for h in GRID]
             + [f"diag_vol_d1r_hl{h}" for h in (6, 12, 36)])
    A, P = _vol_frames(names, T, diag_dir)
    w = _wide(P, "daily", "qlike")[names]
    cells = _common_cells(A[A["origin"].isin(w.index)], names, True)
    grp = cells.index.get_level_values("asset").map(lambda a: BY_ID[a].group)
    # restrict to cells where daily inputs could matter: origin after the asset's daily start
    rows = {}
    pairs = [(f"diag_vol_d_hl{h}", f"diag_vol_m_hl{h}") for h in GRID] + \
            [(f"diag_vol_d1r_hl{h}", f"diag_vol_d_hl{h}") for h in (6, 12, 36)] + \
            [(f"diag_vol_d1r_hl{h}", f"diag_vol_m_hl{h}") for h in (6, 12, 36)]
    for a, b in pairs:
        sk, lo, hi, p = _test_vs(w, a, b)
        md, pp = _panel_test(cells, a, b)
        r = dict(skill_pct=sk, ci90_lo=lo, ci90_hi=hi, dm_p=p, panel_diff=md, panel_p=pp)
        for g in GROUPS:
            c = cells[grp == g]
            r[f"skill_{g}"] = 100 * (1 - c[a].mean() / c[b].mean())
        rows[f"{a.replace('diag_vol_', '')} vs {b.replace('diag_vol_', '')}"] = r
    t = pd.DataFrame(rows).T
    t.index.name = "comparison"
    out.append(f"#### Daily inputs vs monthly, subset `daily` ({len(w)} origins "
               f"{w.index.min():%Y-%m}..{w.index.max():%Y-%m}; {len(cells)} asset-origin cells). "
               "skill = % lower mean QLIKE of the first model (90% Fieller CI, pooled series); "
               "dm_p = DM on the pooled series; panel_* = Driscoll-Kraay panel DM on asset cells; "
               "skill_<group> = point skill on that group's cells\n\n" + _md(t, "{:.3f}"))
    return "\n\n".join(out)


def dropped_tables(T, diag_dir) -> str:
    names = ["bench_vol_mr_est", "diag_vol_mr_est_free", "bench_vol_mr_fixed"]
    A, P = _vol_frames(names, T, diag_dir)
    rows = {}
    for subset in ("daily", "all"):
        w = _wide(P, subset, "qlike")[names]
        for n in names:
            sk, lo, hi, p = _test_vs(w, n, "bench_vol_mr_est") if n != "bench_vol_mr_est" else (0, np.nan, np.nan, np.nan)
            rows[(subset, n)] = dict(qlike=w[n].mean(), skill_vs_mr_est=sk, ci90_lo=lo, ci90_hi=hi, dm_p=p)
    t = pd.DataFrame(rows).T
    t.index = [f"{s} / {n}" for s, n in t.index]
    t.index.name = "subset / model"
    return "#### Dropped vol variant (unconstrained estimated blend)\n\n" + _md(t, "{:.3f}")


# -------------------------------------------------------------------- corr
def _corr_frames(names, T, diag_dir):
    A, P = [], []
    for n in names:
        o = corr_losses(_load("corr", n, diag_dir), T, n)
        A.append(o["pair"])
        P.append(o["pooled"])
    return pd.concat(A, ignore_index=True), pd.concat(P, ignore_index=True)


def _cov_frame(names, T, diag_dir):
    vol = _load("vol", COMPOSITE_VOL, diag_dir)[["origin", "asset", "vol"]]
    P = []
    for n in names:
        c = _load("corr", n, diag_dir)
        c = c.merge(vol.rename(columns={"asset": "asset_i", "vol": "vi"}), on=["origin", "asset_i"]) \
             .merge(vol.rename(columns={"asset": "asset_j", "vol": "vj"}), on=["origin", "asset_j"])
        c["cov"] = c["corr"] * c["vi"] * c["vj"]
        P.append(cov_losses(c[["origin", "asset_i", "asset_j", "cov"]], T, n)["pooled"])
    return pd.concat(P, ignore_index=True)


def corr_tables(T, diag_dir) -> str:
    out = []
    reg = list(CORR_MODELS)
    names = reg + REF_CORR
    A, P = _corr_frames(names, T, diag_dir)
    w = _wide(P, "daily", "mse")[names]
    refs = [r for r in REF_CORR if r != "ref_corr_zero"]
    best = w[refs].mean().idxmin()
    C = _cov_frame(names, T, diag_dir)
    wc = _wide(C, "daily", "qlike_rel")[names]
    bestc = wc[refs].mean().idxmin()
    rows = {}
    for m in names:
        sk, lo, hi, p = _test_vs(w, m, best) if m != best else (0.0, np.nan, np.nan, np.nan)
        dc = wc[m] - wc[bestc]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pc = dm_test(wc[m], wc[bestc], warn=False).pvalue if m != bestc else np.nan
        rows[m] = dict(mse=w[m].mean(), skill_pct=sk, ci90_lo=lo, ci90_hi=hi, dm_p=p,
                       cov_qlike_rel=wc[m].mean(), cov_diff=dc.mean(), cov_dm_p=pc)
    t = pd.DataFrame(rows).T.sort_values("mse")
    t.index.name = "model"
    out.append(f"#### Correlation, subset `daily`: {len(w)} common origins {w.index.min():%Y-%m}.."
               f"{w.index.max():%Y-%m}; skill/DM on pooled MSE vs best ref `{best}`; cov_qlike_rel = "
               f"multivariate QLIKE of `{COMPOSITE_VOL}` vols x this correlation ({len(wc)} origins), "
               f"cov_diff / cov_dm_p vs `{bestc}`\n\n" + _md(t, "{:.4f}"))
    e = w.groupby(_era(w.index.to_series()).values).mean().T
    e.loc["n_origins"] = w.groupby(_era(w.index.to_series()).values).size()
    e.index.name = "model"
    out.append("#### Correlation MSE by era\n\n" + _md(e, "{:.4f}"))
    # by pair group
    a = A[A["origin"].isin(w.index)]
    a = a.assign(gp=[ "-".join(sorted((BY_ID[i].group, BY_ID[j].group))) for i, j in zip(a["asset_i"], a["asset_j"])])
    cnt = a.groupby(["origin", "asset_i", "asset_j"])["model"].nunique()
    full = cnt[cnt == len(names)].index
    a = a.set_index(["origin", "asset_i", "asset_j"]).loc[lambda x: x.index.isin(full)].reset_index()
    g = a.pivot_table(index="model", columns="gp", values="se", aggfunc="mean").reindex(names)
    g.loc["n_cells"] = a[a["model"] == names[0]].groupby("gp").size()
    g.index.name = "model"
    out.append("#### Correlation MSE by group pair (common pair-origin cells)\n\n" + _md(g, "{:.4f}"))
    # half-life curve and daily vs monthly
    dn = [f"diag_corr_{s}_hl{h}" for s in ("m", "d") for h in GRID] + ["diag_corr_shrink_const"]
    A2, P2 = _corr_frames(dn, T, diag_dir)
    w2 = _wide(P2, "daily", "mse")[dn]
    C2 = _cov_frame(dn, T, diag_dir)
    wc2 = _wide(C2, "daily", "qlike_rel")[dn]
    rows = {}
    for s in ("m", "d"):
        rows[f"{'monthly' if s == 'm' else '21d-sum'} mse"] = {f"hl{h}": w2[f"diag_corr_{s}_hl{h}"].mean() for h in GRID}
        rows[f"{'monthly' if s == 'm' else '21d-sum'} cov qlike_rel"] = {f"hl{h}": wc2[f"diag_corr_{s}_hl{h}"].mean() for h in GRID}
    t = pd.DataFrame(rows).T
    t.index.name = "input / loss"
    out.append(f"#### Correlation by half-life ({len(w2)} origins)\n\n" + _md(t, "{:.4f}"))
    rows = {}
    for h in GRID:
        a_, b_ = f"diag_corr_d_hl{h}", f"diag_corr_m_hl{h}"
        sk, lo, hi, p = _test_vs(w2, a_, b_)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pc = dm_test(wc2[a_], wc2[b_], warn=False).pvalue
        rows[f"d vs m hl{h}"] = dict(skill_mse=sk, ci90_lo=lo, ci90_hi=hi, dm_p=p,
                                     cov_diff=(wc2[a_] - wc2[b_]).mean(), cov_dm_p=pc)
    sk, lo, hi, p = _test_vs(w2, "diag_corr_shrink_const", "diag_corr_d_hl36")
    rows["shrink_const vs d_hl36"] = dict(skill_mse=sk, ci90_lo=lo, ci90_hi=hi, dm_p=p,
                                          cov_diff=(wc2["diag_corr_shrink_const"] - wc2["diag_corr_d_hl36"]).mean())
    t = pd.DataFrame(rows).T
    t.index.name = "comparison"
    out.append("#### Correlation: 21-day-sum vs monthly inputs (skill = % lower MSE; cov_diff < 0 favours the first)\n\n"
               + _md(t, "{:.4f}"))
    return "\n\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag-dir", default=os.environ.get("BENCH_DIAG_DIR",
                                                        str(Path(tempfile.gettempdir()) / "bench_diag")))
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--skip-run", action="store_true")
    a = ap.parse_args(argv)
    diag = Path(a.diag_dir)
    panel = load_panel()
    if not a.skip_run:
        run_diag(panel, diag, a.step)
    T = get_targets(panel, RESULTS)
    parts = [vol_tables(T, diag), halflife_tables(T, diag), daily_tables(T, diag),
             dropped_tables(T, diag), corr_tables(T, diag)]
    text = "\n\n".join(parts) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
