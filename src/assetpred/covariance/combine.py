"""Pair any correlation model with per-asset predictive sds from a marginal model.

    joint = combine(cov_forecasts, marginal_forecasts)

For each (horizon, origin) the correlation matrix R implied by the covariance
model's matrix is combined with the marginal model's ``sd`` column:

    Sigma = diag(sd) R diag(sd)

so the joint forecast's variances equal the marginal model's sd^2 exactly
(consistent with the marginals by construction), and it is PSD whenever R is
(R is the correlation of a PSD matrix; the result is still eigen-floored and
checked). Assets present in only one of the two inputs are dropped for that
origin; origins present in only one are dropped.

Gaussian copula reading: with the marginal model's own quantiles, R is the
copula correlation; Sigma is the covariance only if the marginals are
(near) Gaussian. The marginal ``sd`` includes the marginal model's own
mean-uncertainty inflation; the correlation model's R includes the engine's,
which barely changes correlations.

From the command line: ``python -m assetpred.covariance.run --dev --combine
vc_ewma_ewma:vol_ewma`` reads the marginal forecasts from
``results/dev/forecasts/vol_ewma/hXX.parquet`` and scores the pair as model
``vc_ewma_ewma__sd_vol_ewma``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema
from . import linalg as la
from .engine import check_matrix, from_matrices, to_matrices


def combined_name(corr_model: str, sd_model: str) -> str:
    return f"{corr_model}__sd_{sd_model}"


def combine_matrix(S: np.ndarray, sd: np.ndarray) -> np.ndarray:
    R, _ = la.cov_to_corr(S)
    out = la.make_psd(la.corr_to_cov(R, np.asarray(sd, dtype=float)))
    return out


def combine(cov: pd.DataFrame, marginals: pd.DataFrame, name: str | None = None) -> pd.DataFrame:
    """``cov``: COV_COLUMNS for one model (any horizons); ``marginals``:
    FORECAST_COLUMNS (or at least model, horizon, date, asset, sd) for one model."""
    if cov["model"].nunique() != 1 or marginals["model"].nunique() != 1:
        raise ValueError("combine one covariance model with one marginal model at a time")
    name = name or combined_name(cov["model"].iloc[0], marginals["model"].iloc[0])
    out = []
    for h, ch in cov.groupby("horizon"):
        mh = marginals[marginals["horizon"] == h]
        sd_tab = mh.pivot(index="date", columns="asset", values="sd")
        mats = {}
        for t, (assets, S) in to_matrices(ch).items():
            if t not in sd_tab.index:
                continue
            sd_row = sd_tab.loc[t]
            keep = [a for a in assets if a in sd_row.index and np.isfinite(sd_row[a]) and sd_row[a] > 0]
            if not keep:
                continue
            ix = [assets.index(a) for a in keep]
            Sc = combine_matrix(S[np.ix_(ix, ix)], sd_row[keep].to_numpy(float))
            check_matrix(Sc, f"{name} h={h} {t.date()}")
            mats[t] = (tuple(keep), Sc)
        if mats:
            out.append(from_matrices(mats, name, int(h)))
    if not out:
        return pd.DataFrame(columns=schema.COV_COLUMNS)
    return pd.concat(out, ignore_index=True)


def load_marginals(sd_model: str, horizons, results_dir: Path | None = None) -> pd.DataFrame:
    from ..evaluation.walkforward import load_forecasts
    return load_forecasts([sd_model], horizons, results_dir)
