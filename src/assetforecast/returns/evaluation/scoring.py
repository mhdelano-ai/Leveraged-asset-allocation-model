"""Assemble one tidy scores table for a set of forecasts.

:func:`score_forecasts` returns a long table with columns

    model, horizon, asset, period, benchmark, metric, value

* ``asset`` is an asset id or ``"pooled"`` (all assets; tests aggregate per
  date first). ``period`` is ``"full"`` or a subperiod label.
* ``benchmark`` is ``"none"`` for a model's own accuracy, else the model it is
  compared with: point accuracy against ``hist_mean`` only; distribution skill
  against ``hist_mean`` and, separately, ``vol_ewma``.
* ``model == "*"`` rows are about the whole set of models in a cell (SPA);
  ``model == "*mean"`` / ``"*dist"`` rows are the same test restricted to one
  pre-registered family (``families``, from ``config.SPA_FAMILIES``).

Every comparison at a horizon uses the same rows: the (origin, asset) pairs
where *every* model in the run has a forecast and the target is realized.

Metrics (``benchmark == "none"``):
  n_obs, n_dates, n_indep (= n_dates / h, the non-overlapping observations),
  r2_detectable (rough OOS R^2 needed for 5% significance given n_indep),
  mse, crps (grid CRPS), mean_sd, pinball_qXX, cover_50/80/90/98,
  calib_f / calib_p (PIT test at q10/q50/q90, ``inference.calibration_test``),
  hit_qXX / kupiec_p_qXX / christoffersen_p_qXX (assets only) for the tail
  quantiles, mcs_p_mse / in_mcs_mse and
  mcs_p_crps / in_mcs_crps (model confidence set at 10%, each period in
  ``mt_periods``).
  For ``asset == "pooled"`` kupiec_p_qXX is ``inference.kupiec_pooled``: the
  null is that *every* asset's tail hit rate is nominal (per-asset, per-strand
  mid-p binomial tests, Cauchy-combined). It holds its size on the real panel
  shapes (0.7-5.8% at 5% in Monte Carlo); an earlier pooled-hit-rate t-test
  did not (up to 87%) and was replaced.
Metrics vs ``hist_mean``:
  r2_oos, cw_stat, cw_p (one-sided), dm_mse_stat, dm_mse_p (two-sided),
  dm_mse_p1 (one-sided, model better), cw_q_bh (BH across this model's
  asset x horizon cells), cw_q_by_all (BY across all models and cells).
  Read cw_p only together with r2_oos: Clark-West tests the population null
  that the extra predictors have no coefficient, and a model with very noisy
  forecasts can get a small cw_p while its OOS R^2 is far below zero (the
  harness tests show a memorizing model with R^2 = -0.56 and cw_p = 0.02).
  A model has beaten the benchmark in a cell only if r2_oos > 0 *and* the
  multiple-testing-adjusted evidence (cw_q_bh, spa_p_*) holds up.
Metrics vs ``hist_mean`` and vs ``vol_ewma``:
  crps_skill (1 - CRPS/CRPS_bench), dm_crps_stat, dm_crps_p (one-sided,
  model better), dm_crps_q_bh (BH across this model's cells), pinball_skill_qXX.
Rows with ``model == "*"`` (each period in ``mt_periods``):
  spa_p_mse (vs hist_mean), spa_p_crps (vs hist_mean and vs vol_ewma): Hansen
  SPA, does *any* model in the run beat the benchmark in this cell? The
  p-value depends on which models happen to be in the run.
Rows with ``model == "*mean"`` / ``"*dist"`` (each period in ``mt_periods``,
only when the family has members in the run): the SPA test over that family
only. ``*mean``: spa_p_mse vs hist_mean. ``*dist``: spa_p_crps vs hist_mean
and vs vol_ewma. These are the pre-registered tests to read.
spa_q_bh_mse / spa_q_bh_crps: BH across the asset x horizon cells, within one
model label ("*", "*mean", "*dist"), benchmark and period.

Multiple-testing rows (SPA, MCS, every *_q_* column) exist only for the
periods in ``mt_periods`` (default ``("full",)``; a full run of
``evaluation.run`` uses ``config.MT_PERIODS``). Cells with fewer than 24 dates
get no SPA/MCS rows.

Known size distortions: an independent Monte Carlo on the real panel shapes
found Clark-West (cw_p) and SPA (spa_p_*) slightly oversized at h = 12:
7.5-10.5% rejections of a true null at a nominal 5%. Treat h = 12 p-values
near 0.05 as weaker evidence than they look.
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


def _absolute(b: _Block, y, sel, dates, h, pooled, out, assets=None):
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
    f, p = inf.calibration_test(ys, Qs, h, dates=d if pooled else None)
    out["calib_f"], out["calib_p"] = f, p
    for c in mt.TAIL_QUANTILES:
        hits, prob = mt.tail_hits(ys, Qs, c)
        out[f"hit_{c}"] = hits.mean()
        if pooled:
            out[f"kupiec_p_{c}"] = inf.kupiec_pooled(hits, prob, h, d, assets[sel])
        else:
            out[f"kupiec_p_{c}"] = inf.kupiec(hits, prob, h)[1]
            out[f"christoffersen_p_{c}"] = inf.christoffersen(hits, h)


def _relative(b: _Block, bb: _Block, y, sel, dates, h, pooled, point, out):
    d = dates[sel] if pooled else None
    if point:
        sse_b = bb.se[sel].sum()
        out["r2_oos"] = 1 - b.se[sel].sum() / sse_b if sse_b > 0 else np.nan
        out["cw_stat"], out["cw_p"] = inf.clark_west(y[sel], bb.mean[sel], b.mean[sel], h, dates=d)
        s, p1, p2 = inf.diebold_mariano(b.se[sel], bb.se[sel], h, dates=d)
        out["dm_mse_stat"], out["dm_mse_p"], out["dm_mse_p1"] = s, p2, p1
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


# SPA tests run within each named family: family name -> ((loss tag, benchmarks), ...)
FAMILY_TESTS = {
    "mean": (("mse", (BENCHMARK,)),),
    "dist": (("crps", (BENCHMARK, VOL_BENCHMARK)),),
}


def score_forecasts(forecasts: pd.DataFrame, xr: pd.DataFrame,
                    benchmarks=(BENCHMARK, VOL_BENCHMARK), subperiods=DEFAULT_SUBPERIODS,
                    mt_reps: int = 999, mcs_size: float = 0.10, seed: int = config.SEED,
                    multiple_testing: bool = True, mt_periods=("full",),
                    families: dict | None = None, targets: pd.Series | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score forecasts (``schema.FORECAST_COLUMNS``, any models and horizons).

    ``targets``: realized targets as a Series indexed (date, asset), used in
    place of the h-month sums of ``xr`` (for targets that are not sums of a
    monthly panel, e.g. policy portfolios). It is used for every horizon in
    ``forecasts``, so pass forecasts of one horizon with it; ``xr`` is then
    ignored and may be None. Rows with NaN targets are dropped.

    ``mt_periods``: periods (``"full"`` or subperiod labels) for which the
    multiple-testing rows (SPA, MCS, FDR q-values) are computed. ``families``:
    family name -> model names (``config.SPA_FAMILIES``); each family present in
    the run gets its own SPA rows with model ``"*<name>"`` (see
    ``FAMILY_TESTS`` for the loss and benchmarks per family).

    Returns ``(scores, cumulative)``: the tidy scores table described in the
    module docstring, and the Goyal-Welch series (model, horizon, asset, date,
    cum_dsse = running sum of SE(hist_mean) - SE(model)) for every model.
    """
    families = dict(families or {})
    unknown = set(families) - set(FAMILY_TESTS)
    if unknown:
        raise ValueError(f"no SPA test defined for families {sorted(unknown)}; "
                         f"known: {sorted(FAMILY_TESTS)}")
    mt_periods = tuple(mt_periods)
    if targets is not None:
        if not isinstance(targets, pd.Series) or targets.index.nlevels != 2:
            raise ValueError("targets must be a Series indexed (date, asset)")
        if targets.index.has_duplicates:
            raise ValueError("targets: duplicate (date, asset) rows")
        if forecasts["horizon"].nunique() > 1:
            raise ValueError("targets given with forecasts of several horizons; score one horizon at a time")
    rows: list[tuple] = []
    cum_frames = []
    for h, fh in forecasts.groupby("horizon", sort=True):
        h = int(h)
        models = sorted(fh["model"].unique())
        for bm in benchmarks:
            if bm not in models:
                raise ValueError(f"h={h}: benchmark {bm!r} forecasts are missing; run it too")
        if targets is None:
            y_all = realized_targets(xr, h)
        else:
            y_all = targets.dropna().rename("y")
            y_all.index = y_all.index.set_names(["date", "asset"])
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
                    _absolute(b, y, sel, dates, h, pooled, out, assets)
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
            pmasks = {}
            for plabel, start, end in periods:
                if start is None:
                    pmasks[plabel] = np.ones(len(y), bool)
                else:
                    pmasks[plabel] = (dates >= np.datetime64(pd.Timestamp(start))) & \
                        (dates <= np.datetime64(pd.Timestamp(end)))
            for plabel in mt_periods:
                if plabel not in pmasks:
                    continue
                for a, amask, pooled in cells:
                    sel = amask & pmasks[plabel]
                    if len(np.unique(dates[sel])) < 24:
                        continue
                    rows += _multiple_testing_cell(blocks, benchmarks, families, sel, dates, pooled,
                                                   h, a, plabel, mt_reps, mcs_size, rng_seed)

    scores = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    scores["value"] = scores["value"].astype(float)
    scores = _add_fdr(scores, mt_periods)
    cum = pd.concat(cum_frames, ignore_index=True) if cum_frames else \
        pd.DataFrame(columns=["model", "horizon", "asset", "date", "cum_dsse"])
    return scores, cum


def _multiple_testing_cell(blocks, benchmarks, families, sel, dates, pooled, h, a, plabel,
                           reps, mcs_size, rng_seed) -> list[tuple]:
    """SPA (all models, and per family) and MCS rows for one (horizon, asset, period) cell."""
    rows = []
    L = {tag: pd.DataFrame({m: _per_date(getattr(b, loss), sel, dates, pooled)
                            for m, b in blocks.items()})
         for loss, tag in (("se", "mse"), ("crps", "crps"))}
    for tag, frame in L.items():
        for bm in benchmarks:
            if tag == "mse" and bm != BENCHMARK:
                continue
            others = [m for m in frame.columns if m != bm]
            if others:
                p = inf.spa_test(frame[bm], frame[others], h, reps=reps, seed=rng_seed)
                rows.append(("*", h, a, plabel, bm, f"spa_p_{tag}", p))
        if frame.shape[1] >= 2:
            mcs = inf.model_confidence_set(frame, h, size=mcs_size, reps=reps, seed=rng_seed)
            for m in mcs.index:
                rows.append((m, h, a, plabel, "none", f"mcs_p_{tag}", mcs.at[m, "mcs_p"]))
                rows.append((m, h, a, plabel, "none", f"in_mcs_{tag}", mcs.at[m, "in_mcs"]))
    for fam, members in families.items():
        for tag, fam_benchmarks in FAMILY_TESTS[fam]:
            for bm in fam_benchmarks:
                if bm not in blocks:
                    continue
                present = [m for m in members if m in blocks and m != bm]
                if not present:
                    continue
                frame = L[tag]
                p = inf.spa_test(frame[bm], frame[present], h, reps=reps, seed=rng_seed)
                rows.append((f"*{fam}", h, a, plabel, bm, f"spa_p_{tag}", p))
    return rows


def _add_fdr(scores: pd.DataFrame, mt_periods=("full",)) -> pd.DataFrame:
    """Append FDR q-values across the asset x horizon grid (assets only), separately
    for each period in ``mt_periods``."""
    base = scores[scores.period.isin(list(mt_periods)) & (scores.asset != POOLED)]
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
    add(cw, "cw_q_bh", "bh", ["model", "benchmark", "period"])
    add(cw, "cw_q_by_all", "by", ["benchmark", "period"])
    add(base[base.metric == "dm_crps_p"], "dm_crps_q_bh", "bh", ["model", "benchmark", "period"])
    for tag in ("mse", "crps"):
        # within one SPA label ("*", "*mean", "*dist"), benchmark and period
        add(base[base.metric == f"spa_p_{tag}"], f"spa_q_bh_{tag}", "bh",
            ["model", "benchmark", "period"])
    if extra:
        scores = pd.concat([scores, *extra], ignore_index=True)
    return scores


def summary(scores: pd.DataFrame, period: str = "full") -> pd.DataFrame:
    """Compact pooled view of one period: one row per (model, horizon)."""
    s = scores[(scores.asset == POOLED) & (scores.period == period)
               & ~scores.model.str.startswith("*")]
    pick = {
        ("r2_oos", BENCHMARK): "r2_oos",
        ("cw_p", BENCHMARK): "cw_p",
        ("dm_mse_p1", BENCHMARK): "dm_mse_p1",
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
    if results_dir is None:
        from ... import paths  # imported late, see walkforward._default_results
        results_dir = paths.RESULTS_RETURNS
    root = Path(results_dir) / "scores"
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{run_name}.parquet"
    scores.to_parquet(p, index=False)
    scores.to_csv(root / f"{run_name}.csv", index=False, float_format="%.6g")
    cumulative.to_parquet(root / f"{run_name}_cumdsse.parquet", index=False)
    (root / f"{run_name}.json").write_text(json.dumps(meta or {}, indent=1, default=str))
    return p
