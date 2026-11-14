"""Per-origin losses from stored forecasts and cached targets, plus a scoreboard.

All loss frames are TIDY: one row per (model, origin[, asset | pair | subset])
with loss columns, so the inference code can pivot to origin x model with
:func:`loss_matrix`. Every row carries ``origin``, ``dim`` (number of assets
behind the row) and the proxy information (``proxy`` for per-asset rows,
``n_proxy`` / ``n_proxy_excluded`` for pooled rows).

Vol (per asset; v = variance, s = vol; f = forecast, r = realized)
    qlike         v_r/v_f - log(v_r/v_f) - 1      (0 iff perfect; scale-free)
    se_vol        (s_f - s_r)^2                   (scale-DEPENDENT)
    se_vol_scaled ((s_f - s_r) / c_a)^2, c_a = the asset's mean realized vol
                  over all origins (a fixed constant per asset)
    se_var_scaled ((v_f - v_r) / c_a^2)^2  (MSE on variance, fixed scale)
    se_logvol     (log s_f - log s_r)^2           (scale-free)
    log_err       log(s_f / s_r)                  (signed bias; > 0 = over-forecast)

Pooling across assets (vol): equal-weight MEAN over the assets in a subset.
QLIKE and log errors are scale-free, so an equal-weight mean treats a 6% vol
bond fund and a 25% vol EM fund alike. Plain ``se_vol`` is dominated by the
high-vol assets, so it is reported but the comparable pooled MSE is
``se_vol_scaled``: dividing by a fixed per-asset constant keeps it a
consistent loss under noisy unbiased targets (Patton 2011) — unlike dividing
by the realized value, which would not. QLIKE is the headline metric.

Robustness to target noise (matters for monthly-proxy rows): QLIKE and MSE on
VARIANCE (``se_var_scaled``) rank forecasts correctly under any unbiased
variance proxy (Patton 2011). Squared errors of vol / log vol and ``log_err``
are NOT robust: with a noisy proxy they favour forecasts biased towards the
proxy's median. They are fine on daily-based targets (small noise) but read
them on the "daily" subset; ``log_err`` on proxy rows is biased downwards.

Subsets (rows of the ``pooled`` frames, column ``subset``):
    "daily"   all assets with a DAILY-based target at that origin (vol/corr/cov)
    "all"     vol only: also includes monthly-proxy targets
    named     fixed lists (DEFAULT_SUBSETS or the caller's): a row exists only
              at origins where EVERY member has a forecast and a target (for
              corr/cov: a daily-based one), so eras can be compared like for
              like. ``n_proxy`` counts proxy targets inside a vol subset.
Because the eligible set grows through time, compare eras on a named subset.

Corr: per pair ``err = corr_f - corr_r``, ``se = err^2`` on pairs with a
realized value (both assets daily-based); pooled ``mse`` (mean over pairs) and
``bias`` (mean err).

Cov (on the daily-based set, H forecast, S realized, N = dim):
    stein        [tr(H^-1 S) - log det(H^-1 S) - N] / N   (0 iff H == S; per-asset
                 normalization makes it comparable across dimensions; ranks like
                 the multivariate QLIKE log det H + tr(H^-1 S))
    frob_scaled  mean_ij ((H_ij - S_ij) / (c_i c_j))^2 = ||C^-1 (H-S) C^-1||_F^2 / N^2
    target_pd    whether S was positive definite (stein is NaN otherwise)

Public API::

    vol_losses(fc_long, targets, model, subsets=None) -> {"asset": df, "pooled": df}
    corr_losses(fc_long, targets, model, subsets=None) -> {"pair": df, "pooled": df}
    cov_losses(fc_long, targets, model, subsets=None) -> {"pooled": df}
    compute_losses(kind, models, targets, results_dir=None, subsets=None) -> dict of frames
    loss_matrix(frame, metric, subset="daily", asset=None, pair=None) -> origin x model
    scoreboard(losses, kind, subset="daily", eras=None, metrics=None) -> dict of tables
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..universe import ASSET_IDS
from .harness import load_forecasts_long
from .targets import Targets

#: Fixed named subsets for like-for-like comparisons across eras. These are
#: placeholders chosen before the real panel exists; revisit once its
#: coverage is known (``coverage_summary(targets)`` shows when each asset's
#: daily targets start).
DEFAULT_SUBSETS: dict[str, list[str]] = {
    "stock_bond": ["us_large", "ust_int"],
    "core6": ["us_large", "us_small", "intl_dev", "ust_int", "ust_long", "gold"],
    "full12": list(ASSET_IDS),
}

VOL_METRICS = ["qlike", "se_logvol", "log_err", "se_var_scaled", "se_vol_scaled", "se_vol"]
CORR_METRICS = ["mse", "bias"]
COV_METRICS = ["stein", "frob_scaled"]
PRIMARY = {"vol": "qlike", "corr": "mse", "cov": "stein"}


def _subsets(subsets):
    return DEFAULT_SUBSETS if subsets is None else subsets


# ---------------------------------------------------------------------- vol
def vol_losses(fc: pd.DataFrame, targets: Targets, model: str,
               subsets: dict[str, list[str]] | None = None) -> dict[str, pd.DataFrame]:
    """Per-asset and pooled vol losses. ``fc`` long: origin, asset, vol."""
    scale = targets.vol_scale()
    t = targets.vol[["origin", "asset", "vol", "proxy"]].rename(columns={"vol": "vol_r"})
    d = fc[["origin", "asset", "vol"]].rename(columns={"vol": "vol_f"}).merge(t, on=["origin", "asset"])
    vf, vr = d["vol_f"].to_numpy(float), d["vol_r"].to_numpy(float)
    ratio = (vr / vf) ** 2
    d["qlike"] = ratio - np.log(ratio) - 1.0
    d["se_vol"] = (vf - vr) ** 2
    c = d["asset"].map(scale).to_numpy(float)
    d["se_vol_scaled"] = ((vf - vr) / c) ** 2
    d["se_var_scaled"] = ((vf ** 2 - vr ** 2) / c ** 2) ** 2
    d["log_err"] = np.log(vf / vr)
    d["se_logvol"] = d["log_err"] ** 2
    d.insert(0, "model", model)
    d = d.sort_values(["origin", "asset"], kind="stable").reset_index(drop=True)

    pooled = []
    def pool(rows: pd.DataFrame, name: str, need: int | None = None):
        if rows.empty:
            return
        g = rows.groupby("origin", sort=True)
        p = g[VOL_METRICS].mean()
        p["dim"] = g.size()
        p["n_proxy"] = g["proxy"].sum().astype(int)
        if need is not None:
            p = p[p["dim"] == need]
        p["subset"] = name
        pooled.append(p.reset_index())

    pool(d, "all")
    pool(d[~d["proxy"]], "daily")
    for name, members in _subsets(subsets).items():
        pool(d[d["asset"].isin(members)], name, need=len(set(members)))
    pooled = pd.concat(pooled, ignore_index=True) if pooled else _empty_pooled(VOL_METRICS + ["n_proxy"])
    pooled.insert(0, "model", model)
    cols = ["model", "origin", "subset", "dim", "n_proxy"] + VOL_METRICS
    return {"asset": d, "pooled": pooled[cols]}


def _empty_pooled(cols):
    return pd.DataFrame({"origin": pd.Series([], dtype="datetime64[ns]"), "subset": pd.Series([], dtype="str"),
                         "dim": pd.Series([], dtype=int), **{c: [] for c in cols}})


# --------------------------------------------------------------------- corr
def corr_losses(fc: pd.DataFrame, targets: Targets, model: str,
                subsets: dict[str, list[str]] | None = None) -> dict[str, pd.DataFrame]:
    """Per-pair and pooled corr losses. ``fc`` long: origin, asset_i, asset_j, corr."""
    t = targets.corr[["origin", "asset_i", "asset_j", "corr"]].rename(columns={"corr": "corr_r"})
    d = fc[["origin", "asset_i", "asset_j", "corr"]].rename(columns={"corr": "corr_f"}) \
        .merge(t, on=["origin", "asset_i", "asset_j"])
    d["err"] = d["corr_f"] - d["corr_r"]
    d["se"] = d["err"] ** 2
    d.insert(0, "model", model)
    d = d.sort_values(["origin", "asset_i", "asset_j"], kind="stable").reset_index(drop=True)

    pooled = []
    def pool(rows, name, need_assets=None):
        if rows.empty:
            return
        g = rows.groupby("origin", sort=True)
        p = pd.DataFrame({"mse": g["se"].mean(), "bias": g["err"].mean(), "n_pairs": g.size()})
        both = pd.concat([rows[["origin", "asset_i"]].set_axis(["origin", "a"], axis=1),
                          rows[["origin", "asset_j"]].set_axis(["origin", "a"], axis=1)])
        p["dim"] = both.groupby("origin")["a"].nunique()
        if need_assets is not None:
            k = len(need_assets)
            p = p[p["n_pairs"] == k * (k - 1) // 2]
        p["subset"] = name
        pooled.append(p.reset_index())

    pool(d, "daily")
    for name, members in _subsets(subsets).items():
        m = set(members)
        pool(d[d["asset_i"].isin(m) & d["asset_j"].isin(m)], name, need_assets=m)
    pooled = pd.concat(pooled, ignore_index=True) if pooled else _empty_pooled(CORR_METRICS + ["n_pairs"])
    pooled.insert(0, "model", model)
    pooled["n_proxy"] = 0
    return {"pair": d, "pooled": pooled[["model", "origin", "subset", "dim", "n_pairs", "n_proxy"] + CORR_METRICS]}


# ---------------------------------------------------------------------- cov
def _stein_frob(H: np.ndarray, S: np.ndarray, c: np.ndarray):
    n = len(H)
    frob = float(np.mean(((H - S) / np.outer(c, c)) ** 2))
    sign_s, logdet_s = np.linalg.slogdet(S)
    if sign_s <= 0 or not np.isfinite(logdet_s):
        return np.nan, frob, False
    _, logdet_h = np.linalg.slogdet(H)
    tr = float(np.trace(np.linalg.solve(H, S)))
    return (tr - (logdet_s - logdet_h) - n) / n, frob, True


def cov_losses(fc: pd.DataFrame, targets: Targets, model: str,
               subsets: dict[str, list[str]] | None = None,
               _cache: dict | None = None) -> dict[str, pd.DataFrame]:
    """Pooled cov losses per origin and subset. ``fc`` long: origin, asset_i, asset_j, cov."""
    scale = targets.vol_scale()
    cache = {} if _cache is None else _cache
    have = set(targets.origins)
    named = list(_subsets(subsets).items())
    rows = []
    fc = fc[fc["origin"].isin(have)]
    for o, g in fc.groupby("origin", sort=True):
        o = pd.Timestamp(o).as_unit("ns")
        names = list(dict.fromkeys(g["asset_i"]))
        n = len(names)
        if len(g) == n * n and list(g["asset_j"].iloc[:n]) == names:
            F = g["cov"].to_numpy(float).reshape(n, n)
        else:
            F = g.pivot(index="asset_i", columns="asset_j", values="cov").reindex(
                index=names, columns=names).to_numpy(float)
        fpos = {a: k for k, a in enumerate(names)}
        key = ("set", o)
        if key not in cache:
            vt = targets.vol_at(o)
            cache[key] = (targets.complete_daily_set(o), set(vt.index[vt["proxy"].to_numpy(bool)]))
        comp, proxies = cache[key]
        n_px = len(set(names) & proxies)
        cands = [("daily", [a for a in comp if a in fpos])]
        for name, members in named:
            mem = [a for a in comp if a in set(members)]
            if len(mem) == len(set(members)) and all(a in fpos for a in mem):
                cands.append((name, mem))
        for name, S_assets in cands:
            if len(S_assets) < 2:
                continue
            sk = ("S", o, tuple(S_assets))
            if sk not in cache:
                cache[sk] = (targets.cov_array(o, S_assets), scale.reindex(S_assets).to_numpy(float))
            S, c = cache[sk]
            ix = [fpos[a] for a in S_assets]
            H = F[np.ix_(ix, ix)]
            st, fr, ok = _stein_frob(H, S, c)
            rows.append((model, o, name, len(S_assets), n_px, st, fr, ok))
    cols = ["model", "origin", "subset", "dim", "n_proxy_excluded"] + COV_METRICS + ["target_pd"]
    pooled = pd.DataFrame(rows, columns=cols)
    pooled["origin"] = pooled["origin"].astype("datetime64[ns]")
    return {"pooled": pooled.sort_values(["origin", "subset"], kind="stable").reset_index(drop=True)}


# ------------------------------------------------------------- many models
def compute_losses(kind: str, models: list[str], targets: Targets, results_dir: Path | None = None,
                   subsets: dict[str, list[str]] | None = None) -> dict[str, pd.DataFrame]:
    """Losses for several stored models, concatenated.

    ``models`` are names under ``forecasts/<kind>/``; for kind vol/corr a name
    ``"cov:<name>"`` scores the vols/corrs implied by a stored cov model.
    Returns the same keys as the single-model functions.
    """
    fn = {"vol": vol_losses, "corr": corr_losses, "cov": cov_losses}[kind]
    acc: dict[str, list] = {}
    cache: dict = {}
    for m in models:
        fc = load_forecasts_long(kind, m, results_dir)
        out = fn(fc, targets, m, subsets, _cache=cache) if kind == "cov" else fn(fc, targets, m, subsets)
        for k, v in out.items():
            acc.setdefault(k, []).append(v)
    return {k: pd.concat(v, ignore_index=True) for k, v in acc.items()}


# ---------------------------------------------------------------- outputs
def loss_matrix(frame: pd.DataFrame, metric: str, subset: str | None = "daily",
                asset: str | None = None, pair: tuple[str, str] | None = None) -> pd.DataFrame:
    """Pivot a tidy loss frame to origin x model (for DM / MCS / HAC tests).

    Pass a ``pooled`` frame with ``subset``, a vol ``asset`` frame with
    ``asset``, or a corr ``pair`` frame with ``pair``. Rows are NOT restricted
    to common origins; use ``.dropna()`` for that.
    """
    f = frame
    if asset is not None:
        f = f[f["asset"] == asset]
    elif pair is not None:
        a, b = pair
        f = f[((f["asset_i"] == a) & (f["asset_j"] == b)) | ((f["asset_i"] == b) & (f["asset_j"] == a))]
    elif subset is not None and "subset" in f.columns:
        f = f[f["subset"] == subset]
    w = f.pivot_table(index="origin", columns="model", values=metric, aggfunc="mean")
    models = list(dict.fromkeys(frame["model"]))
    return w.reindex(columns=[m for m in models if m in w.columns]).sort_index()


def _era_labels(origins: pd.Series, eras) -> pd.Series:
    b = sorted(pd.Timestamp(x) for x in eras)
    edges = [pd.Timestamp.min] + b + [pd.Timestamp.max]
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo_s = "start" if lo == pd.Timestamp.min else (lo + pd.offsets.MonthEnd(1)).strftime("%Y-%m")
        hi_s = "end" if hi == pd.Timestamp.max else hi.strftime("%Y-%m")
        labels.append(f"{lo_s}..{hi_s}")
    idx = np.searchsorted(np.array(b, dtype="datetime64[ns]"), origins.to_numpy(dtype="datetime64[ns]"),
                          side="left")
    return pd.Series(pd.Categorical(np.asarray(labels, dtype=object)[idx], categories=labels, ordered=True),
                     index=origins.index)


def scoreboard(losses: dict[str, pd.DataFrame], kind: str, subset: str = "daily",
               eras=None, metrics: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Mean losses by model, on the origins COMMON to all models compared.

    Returns ``overall`` (model x metrics, plus n origins, sorted by the
    primary metric), ``by_era`` (model x era, primary metric; eras split at
    the caller's boundary dates, each boundary closing an era) and, for vol,
    ``by_asset`` (model x asset, QLIKE, on (origin, asset) pairs common to all
    models within the common origins); for corr, ``by_pair`` (mse per pair).
    """
    metrics = metrics or {"vol": VOL_METRICS, "corr": CORR_METRICS, "cov": COV_METRICS}[kind]
    primary = metrics[0]
    pooled = losses["pooled"]
    p = pooled[pooled["subset"] == subset]
    models = list(dict.fromkeys(p["model"]))
    w = loss_matrix(p, primary, subset=None).dropna()
    common = w.index
    p = p[p["origin"].isin(common)]
    out = {}
    overall = p.groupby("model")[metrics].mean().reindex(models)
    overall.insert(0, "n_origins", len(common))
    if len(common):
        overall.insert(1, "first", common.min().strftime("%Y-%m"))
        overall.insert(2, "last", common.max().strftime("%Y-%m"))
    out["overall"] = overall.sort_values(primary)

    if eras:
        lab = _era_labels(p["origin"], eras)
        pe = p.assign(era=lab.astype(str).to_numpy())
        order = [c for c in lab.cat.categories if c in set(pe["era"])]
        e = pe.pivot_table(index="model", columns="era", values=primary, aggfunc="mean")
        e = e.reindex(index=out["overall"].index, columns=order)
        e.loc["n_origins"] = pe.groupby("era")["origin"].nunique().reindex(order).astype(float)
        out["by_era"] = e

    if kind == "vol" and "asset" in losses:
        a = losses["asset"]
        a = a[a["origin"].isin(common)]
        if subset == "daily":
            a = a[~a["proxy"]]
        cnt = a.groupby(["origin", "asset"])["model"].nunique()
        keep = cnt[cnt == len(models)].index
        a = a.set_index(["origin", "asset"]).loc[lambda x: x.index.isin(keep)].reset_index()
        t = a.pivot_table(index="model", columns="asset", values=primary, aggfunc="mean")
        cols = [x for x in ASSET_IDS if x in t.columns] + [x for x in t.columns if x not in ASSET_IDS]
        out["by_asset"] = t.reindex(index=out["overall"].index, columns=cols)
    if kind == "corr" and "pair" in losses:
        a = losses["pair"]
        a = a[a["origin"].isin(common)]
        t = a.assign(pair=a["asset_i"] + "~" + a["asset_j"]).pivot_table(
            index="model", columns="pair", values="se", aggfunc="mean")
        rank = {x: k for k, x in enumerate(ASSET_IDS)}
        cols = sorted(t.columns, key=lambda c: tuple(rank.get(x, 99) for x in c.split("~")))
        out["by_pair"] = t.reindex(index=out["overall"].index, columns=cols)
    return out


def coverage_summary(targets: Targets) -> pd.DataFrame:
    """Per asset: first origin with any target, first with a daily-based one, counts."""
    v = targets.vol
    g = v.groupby("asset")
    out = pd.DataFrame({
        "first_target": g["origin"].min(),
        "first_daily": v[~v["proxy"]].groupby("asset")["origin"].min(),
        "n_daily": v[~v["proxy"]].groupby("asset").size(),
        "n_proxy": v[v["proxy"]].groupby("asset").size(),
    })
    out[["n_daily", "n_proxy"]] = out[["n_daily", "n_proxy"]].fillna(0).astype(int)
    return out.reindex([a for a in targets.assets if a in out.index])
