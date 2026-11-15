"""Skill of the return forecasters against the expanding mean, by horizon.

For each (model, horizon, asset or ``pooled``, period):

``r2``       out-of-sample R squared, ``1 - SSE(model) / SSE(hist_mean)``, on the
             origins where both have a forecast and the target is realized
``cw_t``, ``cw_p``  Clark-West statistic and one-sided p-value (model better),
             Bartlett long-run variance with ``1.5 h`` lags
             (``returns.evaluation.inference``); pooled rows sum the loss
             terms over assets per origin first
``n``        origins; ``n_indep = n / h``: about how many non-overlapping windows
``thin``     ``n_indep < MIN_INDEP``: too few independent windows for the test
             to mean anything (the long-run variance with ``1.5 h`` lags is
             then estimated from less than one of its own bandwidths)

p-values are not adjusted for the number of cells in the table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..returns.evaluation import inference
from . import MIN_INDEP, PERIODS, period_mask
from .signals import BENCH, targets


def _cell(y, fb, fm, dates, h):
    sse_b, sse_m = float(((y - fb) ** 2).sum()), float(((y - fm) ** 2).sum())
    nd = len(np.unique(dates))
    t, p = inference.clark_west(y, fb, fm, h, dates=dates)
    return {"r2": 1 - sse_m / sse_b if sse_b > 0 else np.nan, "cw_t": t, "cw_p": p,
            "n": nd, "n_indep": nd / h, "thin": nd / h < MIN_INDEP,
            "rmse_bench_ann": float(np.sqrt(sse_b / len(y)) * np.sqrt(12 / h))}


def score(forecasts: pd.DataFrame, xr: pd.DataFrame, periods=PERIODS) -> pd.DataFrame:
    rows = []
    for h, fh in forecasts.groupby("horizon"):
        y = targets(xr, h).stack().dropna()
        y.index = y.index.set_names(["date", "asset"])
        wide = fh.pivot_table(index=["date", "asset"], columns="model", values="mean")
        wide = wide.join(y.rename("y"), how="inner").dropna(subset=["y", BENCH])
        dates = wide.index.get_level_values("date")
        assets = wide.index.get_level_values("asset").to_numpy()
        for m in wide.columns.drop(["y", BENCH]):
            ok = wide[m].notna().to_numpy()
            for per in periods:
                sel = ok & period_mask(dates, h, per)
                for a in [*pd.unique(assets), "pooled"]:
                    s = sel if a == "pooled" else sel & (assets == a)
                    if s.sum() < 12:
                        continue
                    w = wide[s]
                    rows.append({"model": m, "horizon": h, "asset": a, "period": per,
                                 **_cell(w["y"].to_numpy(), w[BENCH].to_numpy(), w[m].to_numpy(),
                                         dates[s].to_numpy(), h)})
    return pd.DataFrame(rows)
