"""Assemble one tidy scores table for a set of forecasts.

:func:`score_forecasts` returns a long table with columns

    model, horizon, asset, period, benchmark, metric, value

* ``asset`` is an asset id or ``"pooled"`` (all assets; tests aggregate per
  date first). ``period`` is ``"full"`` or a subperiod label.
* ``benchmark`` is ``"none"`` for a model's own accuracy, else the model it is
  compared with: point accuracy against ``hist_mean`` only; distribution skill
  against ``hist_mean`` and, separately, ``vol_ewma``.
* ``model == "*"`` rows are about the whole set of models in a cell (SPA).

Every comparison at a horizon uses the same rows: the (origin, asset) pairs
where *every* model in the run has a forecast and the target is realized.

Metrics (``benchmark == "none"``):
  n_obs, n_dates, n_indep (= n_dates / h, the non-overlapping observations),
  r2_detectable (rough OOS R^2 needed for 5% significance given n_indep),
  mse, crps (grid CRPS), mean_sd, pinball_qXX, cover_50/80/90/98,
  calib_wald / calib_p (HAC PIT-bin test), hit_qXX / kupiec_p_qXX /
  christoffersen_p_qXX for the tail quantiles, mcs_p_mse / in_mcs_mse and
  mcs_p_crps / in_mcs_crps (model confidence set at 10%, full period).
Metrics vs ``hist_mean``:
  r2_oos, cw_stat, cw_p (one-sided), dm_mse_stat, dm_mse_p (two-sided),
  cw_q_bh (BH across this model's asset x horizon cells),
  cw_q_by_all (BY across all models and cells).
Metrics vs ``hist_mean`` and vs ``vol_ewma``:
  crps_skill (1 - CRPS/CRPS_bench), dm_crps_stat, dm_crps_p (one-sided,
  model better), dm_crps_q_bh (BH across this model's cells), pinball_skill_qXX.
Rows with ``model == "*"`` (full period):
  spa_p_mse, spa_p_crps (Hansen SPA: does *any* model beat the benchmark in
  this cell?), spa_q_bh_mse, spa_q_bh_crps (BH across cells).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, schema
from ..models.baselines import BENCHMARK, VOL_BENCHMARK
from . import inference as inf
from . import metrics as mt

log = logging.getLogger(__name__)

DEFAULT_SUBPERIODS = (
    ("1990s", "1990-01-01", "1999-12-31"),
    ("2000s", "2000-01-01", "2009-12-31"),
    ("2010s", "2010-01-01", "2019-12-31"),
    ("2020s", "2020-01-01", "2029-12-31"),
)
SCORE_COLUMNS = ["model", "horizon", "asset", "period", "benchmark", "metric", "value"]
POOLED = "pooled"


def realized_targets(xr: pd.DataFrame, horizon: int) -> pd.Series:
    y = schema.make_targets(xr, horizon).stack().dropna()
    y.index = y.index.set_names(["date", "asset"])
    return y.rename("y")


class _Block:
    """Arrays for one model on the common rows of a horizon."""

    def __init__(self, F: pd.DataFrame, y: np.ndarray):
        self.mean = F["mean"].to_numpy(float)
        self.sd = F["sd"].to_numpy(float)
        self.Q = F[mt.QCOLS].to_numpy(float)
        self.se = mt.squared_error(y, self.mean)
        self.pin = mt.pinball(y, self.Q)
        self.crps = 2.0 * self.pin @ mt.CRPS_WEIGHTS


def _absolute(b: _Block, y, sel, dates, h, pooled, out):
    ys, Qs = y[sel], b.Q[sel]
    d = dates[sel]
    n = int(sel.sum())
    nd = len(np.unique(d))
    out["n_obs"] = n
    out["n_dates"] = nd
    out["n_indep"] = nd / h
    out["r2_detectable"] = inf.detectable_r2(nd / h)
    out["mse"] = b.se[sel].mean()
    out["crps"] = b.crps[sel].mean()
    out["mean_sd"] = b.sd[sel].mean()
    pin = b.pin[sel].mean(axis=0)
    for k, c in enumerate(mt.QCOLS):
        out[f"pinball_{c}"] = pin[k]
    for lvl in mt.INTERVALS:
        out[f"cover_{lvl}"] = mt.coverage(ys, Qs, lvl)
    bins = mt.pit_bins(ys, Qs)
    w, p = inf.calibration_test(bins, mt.bin_probabilities(), h, dates=d if pooled else None)
    out["calib_wald"], out["calib_p"] = w, p
    for c in mt.TAIL_QUANTILES:
        hits, prob = mt.tail_hits(ys, Qs, c)
        out[f"hit_{c}"] = hits.mean()
        if pooled:
            dev = pd.Series(hits - prob).groupby(d).mean().to_numpy()
            _, _, p2, _ = inf.mean_test(dev, h)
            out[f"kupiec_p_{c}"] = p2
        else:
            out[f"kupiec_p_{c}"] = inf.kupiec(hits, prob, h)[2]
            out[f"christoffersen_p_{c}"] = inf.christoffersen(hits, h)


def _relative(b: _Block, bb: _Block, y, sel, dates, h, pooled, point, out):
    d = dates[sel] if pooled else None
    if point:
        sse_b = bb.se[sel].sum()
        out["r2_oos"] = 1 - b.se[sel].sum() / sse_b if sse_b > 0 else np.nan
        out["cw_stat"], out["cw_p"] = inf.clark_west(y[sel], bb.mean[sel], b.mean[sel], h, dates=d)
        s, _, p2 = inf.diebold_mariano(b.se[sel], bb.se[sel], h, dates=d)
        out["dm_mse_stat"], out["dm_mse_p"] = s, p2
    cb = bb.crps[sel].sum()
    out["crps_skill"] = 1 - b.crps[sel].sum() / cb if cb > 0 else np.nan
    s, p1, _ = inf.diebold_mariano(b.crps[sel], bb.crps[sel], h, dates=d)
    out["dm_crps_stat"], out["dm_crps_p"] = s, p1
    pb = bb.pin[sel].sum(axis=0)
    pm = b.pin[sel].sum(axis=0)
    for k, c in enumerate(mt.QCOLS):
        out[f"pinball_skill_{c}"] = 1 - pm[k] / pb[k] if pb[k] > 0 else np.nan


def _per_date(loss: np.ndarray, sel, dates, pooled) -> pd.Series:
    s = pd.Series(loss[sel], index=dates[sel])
    return s.groupby(level=0).sum() if pooled else s


def score_forecasts(forecasts: pd.DataFrame, xr: pd.DataFrame,
                    benchmarks=(BENCHMARK, VOL_BENCHMARK), subperiods=DEFAULT_SUBPERIODS,
                    mt_reps: int = 999, mcs_size: float = 0.10, seed: int = config.SEED,
                    multiple_testing: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score forecasts (``schema.FORECAST_COLUMNS``, any models and horizons).

    Returns ``(scores, cumulative)``: the tidy scores table described in the
    module docstring, and the Goyal-Welch series (model, horizon, asset, date,
    cum_dsse = running sum of SE(hist_mean) - SE(model)) for every model.
    """
    rows: list[tuple] = []
    cum_frames = []
    for h, fh in forecasts.groupby("horizon", sort=True):
        h = int(h)
        models = sorted(fh["model"].unique())
        for bm in benchmarks:
            if bm not in models:
                raise ValueError(f"h={h}: benchmark {bm!r} forecasts are missing; run it too")
        y_all = realized_targets(xr, h)
        frames = {m: g.set_index(["date", "asset"]).sort_index() for m, g in fh.groupby("model")}
        common = y_all.index
        for m, F in frames.items():
            if F.index.has_duplicates:
                raise ValueError(f"{m} h={h}: duplicate (date, asset) forecasts")
            common = common.intersection(F.index)
        common = common.sort_values()
        for m, F in frames.items():
            dropped = len(F.index.difference(common).intersection(y_all.index))
            if dropped:
                log.warning("%s h=%d: %d scorable rows dropped (not common to all models)", m, h, dropped)
        if len(common) == 0:
            log.warning("h=%d: no common scorable rows", h)
            continue
        y = y_all.reindex(common).to_numpy(float)
        dates = np.asarray(common.get_level_values("date"))
        assets = np.asarray(common.get_level_values("asset")).astype(str)
        blocks = {m: _Block(F.loc[common], y) for m, F in frames.items()}

        periods = [("full", None, None)] + [p for p in (subperiods or ())]
        cells = [(a, assets == a, False) for a in sorted(pd.unique(assets))]
        cells.append((POOLED, np.ones(len(y), bool), True))
        for plabel, start, end in periods:
            pm = np.ones(len(y), bool)
            if start is not None:
                pm = (dates >= np.datetime64(pd.Timestamp(start))) & (dates <= np.datetime64(pd.Timestamp(end)))
            if pm.sum() == 0:
                continue
            for a, amask, pooled in cells:
                sel = amask & pm
                if len(np.unique(dates[sel])) < 12:
                    continue
                for m, b in blocks.items():
                    out: dict = {}
                    _absolute(b, y, sel, dates, h, pooled, out)
                    rows += [(m, h, a, plabel, "none", k, v) for k, v in out.items()]
                    for bm in benchmarks:
                        if bm == m:
                            continue
                        out = {}
                        _relative(b, blocks[bm], y, sel, dates, h, pooled, bm == BENCHMARK, out)
                        rows += [(m, h, a, plabel, bm, k, v) for k, v in out.items()]

        # cumulative SSE difference vs hist_mean (full period)
        bb = blocks[BENCHMARK]
        for m, b in blocks.items():
            if m == BENCHMARK:
                continue
            for a, amask, pooled in cells:
                s = _per_date(bb.se - b.se, amask, dates, True).cumsum()
                cum_frames.append(pd.DataFrame({"model": m, "horizon": h, "asset": a,
                                                "date": s.index, "cum_dsse": s.to_numpy()}))

        if multiple_testing:
            rng_seed = seed + h
            for a, amask, pooled in cells:
                if len(np.unique(dates[amask])) < 24:
                    continue
                for loss in ("se", "crps"):
                    L = pd.DataFrame({m: _per_date(getattr(b, loss), amask, dates, pooled)
                                      for m, b in blocks.items()})
                    tag = "mse" if loss == "se" else "crps"
                    for bm in benchmarks:
                        others = [m for m in L.columns if m != bm]
                        if loss == "se" and bm != BENCHMARK:
                            continue
                        if others:
                            p = inf.spa_test(L[bm], L[others], h, reps=mt_reps, seed=rng_seed)
                            rows.append(("*", h, a, "full", bm, f"spa_p_{tag}", p))
                    if L.shape[1] >= 2:
                        mcs = inf.model_confidence_set(L, h, size=mcs_size, reps=mt_reps, seed=rng_seed)
                        for m in mcs.index:
                            rows.append((m, h, a, "full", "none", f"mcs_p_{tag}", mcs.at[m, "mcs_p"]))
                            rows.append((m, h, a, "full", "none", f"in_mcs_{tag}", mcs.at[m, "in_mcs"]))

    scores = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    scores["value"] = scores["value"].astype(float)
    scores = _add_fdr(scores)
    cum = pd.concat(cum_frames, ignore_index=True) if cum_frames else \
        pd.DataFrame(columns=["model", "horizon", "asset", "date", "cum_dsse"])
    return scores, cum


def _add_fdr(scores: pd.DataFrame) -> pd.DataFrame:
    """Append FDR q-values across the asset x horizon grid (full period, assets only)."""
    base = scores[(scores.period == "full") & (scores.asset != POOLED)]
    extra = []

    def add(df, metric, method, by):
        if df.empty:
            return
        groups = df.groupby(by) if by else [(None, df)]
        for _, g in groups:
            q = inf.fdr(g["value"].to_numpy(), method)
            e = g.copy()
            e["metric"] = metric
            e["value"] = q
            extra.append(e)

    cw = base[base.metric == "cw_p"]
    add(cw, "cw_q_bh", "bh", ["model", "benchmark"])
    add(cw, "cw_q_by_all", "by", ["benchmark"])
    add(base[base.metric == "dm_crps_p"], "dm_crps_q_bh", "bh", ["model", "benchmark"])
    for tag in ("mse", "crps"):
        add(base[base.metric == f"spa_p_{tag}"], f"spa_q_bh_{tag}", "bh", ["benchmark"])
    if extra:
        scores = pd.concat([scores, *extra], ignore_index=True)
    return scores


def summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Compact pooled, full-period view: one row per (model, horizon)."""
    s = scores[(scores.asset == POOLED) & (scores.period == "full")]
    pick = {
        ("r2_oos", BENCHMARK): "r2_oos",
        ("cw_p", BENCHMARK): "cw_p",
        ("crps_skill", BENCHMARK): "crps_skill_vs_hist",
        ("crps_skill", VOL_BENCHMARK): "crps_skill_vs_vol",
        ("cover_90", "none"): "cover_90",
        ("calib_p", "none"): "calib_p",
        ("n_indep", "none"): "n_indep",
    }
    s = s[[(m, b) in pick for m, b in zip(s.metric, s.benchmark)]]
    s = s.assign(col=[pick[(m, b)] for m, b in zip(s.metric, s.benchmark)])
    return s.pivot_table(index=["model", "horizon"], columns="col", values="value").reindex(
        columns=list(pick.values()))


def write_scores(scores: pd.DataFrame, cumulative: pd.DataFrame, run_name: str,
                 results_dir: Path | None = None, meta: dict | None = None) -> Path:
    """Write ``results/scores/<run_name>.{parquet,csv,json}`` and
    ``<run_name>_cumdsse.parquet``; return the parquet path."""
    root = (Path(results_dir) if results_dir is not None else config.RESULTS) / "scores"
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{run_name}.parquet"
    scores.to_parquet(p, index=False)
    scores.to_csv(root / f"{run_name}.csv", index=False, float_format="%.6g")
    cumulative.to_parquet(root / f"{run_name}_cumdsse.parquet", index=False)
    (root / f"{run_name}.json").write_text(json.dumps(meta or {}, indent=1, default=str))
    return p
