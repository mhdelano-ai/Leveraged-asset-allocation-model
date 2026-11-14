"""Leakage guards that sit outside any model.

The walk-forward purge (``walkforward``) stops a model from *training* on
targets that are not yet realized. It cannot stop a *feature* dated t from
containing information published after t: such a feature is as available at
prediction time as at training time. Two defenses live here:

* :func:`audit_feature_leakage` flags features whose rank correlation with the
  forward target (or with next month's excess return) is so high that it can
  only be look-ahead. It catches gross errors (a target copied into the
  features, a shift in the wrong direction), not subtle one-month lag slips.
* :func:`lag_features` shifts every feature one or more extra months. Running a
  model on lagged features (``run --feature-lag 1``) is a cheap sensitivity
  test: genuine, slow-moving predictors lose little; a feature whose apparent
  skill depends on precise timing at the edge of availability collapses.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .. import schema


class SuspectFeatureError(RuntimeError):
    pass


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 24:
        return np.nan
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def audit_feature_leakage(features: pd.DataFrame, xr: pd.DataFrame, horizon: int,
                          error_corr: float = 0.9, warn_corr: float = 0.5,
                          next_month_error_corr: float = 0.6,
                          raise_on_error: bool = True) -> pd.DataFrame:
    """Rank-correlate each feature with the forward h-month target and with
    next month's excess return, per asset and pooled.

    Thresholds are set far above anything a genuine monthly predictor attains
    (a 0.6 rank correlation with next month's return would be an R^2 near 0.36).
    Returns a table of flagged features (empty if clean); raises
    :class:`SuspectFeatureError` for errors when ``raise_on_error``.
    """
    y = schema.make_targets(xr, horizon).stack().rename("y").rename_axis(["date", "asset"])
    nxt = xr.shift(-1).stack().rename("next").rename_axis(["date", "asset"])
    df = features.join(y, how="left").join(nxt, how="left")
    assets = df.index.get_level_values("asset")
    rows = []
    for col in features.columns:
        f = df[col].to_numpy(dtype=float)
        if not np.isfinite(f).any():
            continue
        cy = _spearman(f, df["y"].to_numpy(dtype=float))
        cn = _spearman(f, df["next"].to_numpy(dtype=float))
        per = [abs(_spearman(df[col].to_numpy(float)[assets == a], df["y"].to_numpy(float)[assets == a]))
               for a in pd.unique(assets)]
        per = [p for p in per if np.isfinite(p)]
        cmax = max(per) if per else np.nan
        level = None
        if (np.nan_to_num(abs(cy)) > error_corr or np.nan_to_num(cmax) > error_corr
                or np.nan_to_num(abs(cn)) > next_month_error_corr):
            level = "error"
        elif np.nan_to_num(abs(cy)) > warn_corr:
            level = "warn"
        if level:
            rows.append({"feature": col, "horizon": horizon, "level": level,
                         "corr_target_pooled": cy, "corr_target_max_asset": cmax,
                         "corr_next_month": cn})
    out = pd.DataFrame(rows, columns=["feature", "horizon", "level", "corr_target_pooled",
                                      "corr_target_max_asset", "corr_next_month"])
    if len(out):
        warn = out[out.level == "warn"]
        if len(warn):
            warnings.warn(f"features with suspiciously strong correlation to the h={horizon} "
                          f"target (check their lagging): {list(warn.feature)}", stacklevel=2)
        err = out[out.level == "error"]
        if len(err) and raise_on_error:
            raise SuspectFeatureError(
                f"features look like they contain future returns at h={horizon}: "
                f"{list(err.feature)}. A feature dated t must use only data published by "
                f"the close of month t (see schema docstring).\n{err.to_string(index=False)}")
    return out


def lag_features(features: pd.DataFrame, months: int) -> pd.DataFrame:
    """Shift every feature ``months`` months later within each asset."""
    if months == 0:
        return features
    return features.groupby(level="asset", sort=False).shift(months)
