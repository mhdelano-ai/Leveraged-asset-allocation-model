"""Read the results files and reduce them to the small tables the page shows.

Nothing here re-estimates a statistic: every number is a value read from the
scores, covariance-scores, strategy or forecast files, or a display reduction
of them (picking a row, taking the last forecast origin, a cumulative sum of
monthly returns for a wealth curve, a trailing correlation of realized errors
for a reference line). Where the page needs a number the files do not carry,
the page says it is unavailable instead of computing it here.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, schema
from ..evaluation.walkforward import forecast_path

log = logging.getLogger(__name__)

KEY = ["model", "horizon", "asset", "period", "benchmark", "metric"]
POOLED = "pooled"
BENCH = "hist_mean"
VOL_BENCH = "vol_ewma"
COV_BENCH = "sample"
HORIZONS = tuple(config.HORIZONS)
ASSETS = list(config.ASSETS)


class MissingInput(SystemExit):
    """A required results file is absent; the message says which and what exists."""


# --------------------------------------------------------------------------- loading

def _listing(d: Path, suffix: str = ".parquet") -> list[str]:
    if not d.exists():
        return []
    return sorted(p.name[: -len(suffix)] for p in d.glob(f"*{suffix}")
                  if not p.name.endswith(("_cumdsse.parquet", "_per_date.parquet")))


def load_score_runs(root: Path, runs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Merge one or more marginal score runs.

    Rows for a real model are taken from the first run (in the order given)
    that has them; rows for model ``'*'`` (SPA, about the whole set of models
    in a run) are kept per run. Returns (scores with a ``run`` column,
    cumulative dSSE, {run: meta}).
    """
    sdir = root / "scores"
    frames, cums, metas = [], [], {}
    for r in runs:
        p = sdir / f"{r}.parquet"
        if not p.exists():
            raise MissingInput(
                f"scores run {r!r} not found at {p}.\nAvailable runs in {sdir}: "
                + (", ".join(_listing(sdir)) or "(none)"))
        df = pd.read_parquet(p)
        df["run"] = r
        frames.append(df)
        cp = sdir / f"{r}_cumdsse.parquet"
        if cp.exists():
            c = pd.read_parquet(cp)
            c["run"] = r
            cums.append(c)
        mp = sdir / f"{r}.json"
        metas[r] = json.loads(mp.read_text()) if mp.exists() and mp.read_text().strip() else {}
    s = pd.concat(frames, ignore_index=True)
    s["horizon"] = s["horizon"].astype(int)
    star = s[is_set_label(s.model)]
    real = s[~is_set_label(s.model)].drop_duplicates(KEY, keep="first")
    s = pd.concat([real, star], ignore_index=True)
    cum = pd.concat(cums, ignore_index=True).drop_duplicates(
        ["model", "horizon", "asset", "date"], keep="first") if cums else None
    return s, cum, metas


def load_cov_run(root: Path, run: str | None):
    if run is None:
        return None, None, {}
    sdir = root / "scores"
    p = sdir / f"{run}.parquet"
    if not p.exists():
        raise MissingInput(f"covariance scores run {run!r} not found at {p}.\nAvailable runs in {sdir}: "
                           + (", ".join(r for r in _listing(sdir) if r.startswith("cov")) or "(none)")
                           + "\nPass --no-covariance to build the page without the covariance section.")
    s = pd.read_parquet(p)
    s["horizon"] = s["horizon"].astype(int)
    pd_path = sdir / f"{run}_per_date.parquet"
    per_date = pd.read_parquet(pd_path) if pd_path.exists() else None
    mp = sdir / f"{run}.json"
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    return s, per_date, meta


def load_strategy(sdir: Path | None):
    if sdir is None or not (sdir / "metrics.parquet").exists():
        return None
    out = {"metrics": pd.read_parquet(sdir / "metrics.parquet")}
    rp = sdir / "returns.parquet"
    out["returns"] = pd.read_parquet(rp, columns=["strategy", "family", "rule", "mean_model", "cov_model",
                                                  "universe", "horizon", "date", "net", "gross"]) \
        if rp.exists() else None
    mp = sdir / "meta.json"
    out["meta"] = json.loads(mp.read_text()) if mp.exists() else {}
    return out


def load_forecast_last(root: Path, model: str, horizon: int) -> pd.DataFrame | None:
    """Forecast rows of ``model`` at its last origin for this horizon (None if absent)."""
    p = forecast_path(model, horizon, root)
    if not p.exists():
        return None
    f = pd.read_parquet(p)
    if f.empty:
        return None
    last = f["date"].max()
    return f[f["date"] == last][schema.FORECAST_COLUMNS].reset_index(drop=True)


def load_cov_last(root: Path, model: str, horizon: int) -> pd.DataFrame | None:
    p = root / "covariances" / model / f"h{horizon:02d}.parquet"
    if not p.exists():
        return None
    c = pd.read_parquet(p)
    if c.empty:
        return None
    return c[c["date"] == c["date"].max()].reset_index(drop=True)


def load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


# --------------------------------------------------------------------------- lookup

def is_set_label(models: pd.Series) -> pd.Series:
    """Rows about a set of models ('*', '*mean', '*dist', ...), not one model."""
    return models.astype(str).str.startswith("*")


@dataclass
class Scores:
    """Indexed access to a tidy scores table."""

    df: pd.DataFrame
    _ix: pd.Series = field(init=False, repr=False)

    def __post_init__(self):
        real = self.df[~is_set_label(self.df.model)]
        self._ix = real.set_index(KEY)["value"].sort_index()

    @property
    def periods(self) -> list[str]:
        return list(pd.unique(self.df["period"]))

    @property
    def models(self) -> list[str]:
        return sorted(m for m in pd.unique(self.df["model"]) if not str(m).startswith("*"))

    def has(self, model: str) -> bool:
        return model in set(self.df["model"])

    def run_size(self, run: str | None, exclude: str | None = None) -> int | None:
        """Number of models in a scores run other than ``exclude`` (the set an SPA test searched)."""
        if run is None or "run" not in self.df:
            return None
        ms = set(self.df.loc[(self.df.run == run) & ~is_set_label(self.df.model), "model"])
        ms.discard(exclude)
        return len(ms)

    def get(self, model, h, asset, period, bench, metric) -> float:
        try:
            v = self._ix.loc[(model, int(h), asset, period, bench, metric)]
        except KeyError:
            return float("nan")
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        return float(v)

    def star(self, metric, h, asset=POOLED, bench=BENCH, period="full", prefer_models=(),
             label: str = "*") -> tuple[float, str | None]:
        """Value of a set-level row (model ``label``: '*' for every model in the
        run, '*mean' / '*dist' for a pre-registered SPA family); among several
        runs, the one whose models overlap ``prefer_models`` most."""
        d = self.df
        g = d[(d.model == label) & (d.metric == metric) & (d.horizon == int(h)) & (d.asset == asset)
              & (d.benchmark == bench) & (d.period == period)]
        if g.empty:
            return float("nan"), None
        if "run" in g and g["run"].nunique() > 1 and prefer_models:
            best, score = None, -1
            for r in g["run"].unique():
                ms = set(self.df.loc[(self.df.run == r) & ~is_set_label(self.df.model), "model"])
                k = len(ms & set(prefer_models))
                if k > score:
                    best, score = r, k
            g = g[g.run == best]
        row = g.iloc[0]
        return float(row["value"]), (row["run"] if "run" in g else None)


def clean(x):
    """JSON-safe float (None for NaN/inf), rounded to keep the page small."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    if not np.isfinite(xf):
        return None
    if xf == 0:
        return 0.0
    return float(f"{xf:.5g}")


def series_points(dates, values, digits: int = 5) -> list:
    out = []
    for d, v in zip(dates, values):
        if v is None or not np.isfinite(v):
            continue
        out.append([pd.Timestamp(d).strftime("%Y-%m"), float(f"{float(v):.{digits}g}")])
    return out
