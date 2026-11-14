"""Bench volatility models (see the package docstring for formulas)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import VolForecaster
from .core import Moments, ew_mean, ew_series, segment_start

MR_RECENT_HL = 3      # months: "current level" component of the mean-reversion blends
MR_HALF_LIFE = 6      # months: a-priori half-life of log-variance deviations
RATIO_HL = 36         # months: half-life of the trailing 21-day/1-day variance ratio
MIN_RATIO_MONTHS = 12 # months in the current tier segment before the ratio is used
MIN_TRAIN = 120       # (asset, origin) rows before the estimated blend is trusted
MIN_HIST = 36         # months of history an (asset, origin) needs to enter training


def mr_weight(half_life: float = MR_HALF_LIFE, horizon: int = 12) -> float:
    """Weight on the current level in a 12-month-average variance forecast when
    the deviation from the long-run level decays as AR(1) with the given
    half-life in months: mean_{h=1..12} phi^h, phi = 0.5^(1/H). H=6 -> 0.51."""
    phi = 0.5 ** (1.0 / half_life)
    return float(phi * (1 - phi ** horizon) / (horizon * (1 - phi)))


def _finish(var: np.ndarray, assets: list[str], fallback: np.ndarray) -> pd.Series:
    var = np.where(np.isfinite(var) & (var > 0), var, fallback)
    if not (np.isfinite(var) & (var > 0)).all():          # last resort, never expected
        med = np.nanmedian(np.where(var > 0, var, np.nan))
        var = np.where(np.isfinite(var) & (var > 0), var, med if np.isfinite(med) else 0.04)
    return pd.Series(np.sqrt(var), index=assets)


class EWVol(VolForecaster):
    """12 x exponentially weighted mean of a per-month variance measurement.
    ``daily=False``: monthly squared returns only. ``daily=True``: 21-day sums
    of daily returns where they exist, monthly squared returns otherwise."""

    def __init__(self, hl: float | None, daily: bool):
        self.hl, self.daily = hl, daily

    def forecast_vol(self, view, assets):
        mo = Moments(view, assets, daily=self.daily)
        x = mo.x if self.daily else mo.r2
        fb = 12.0 * ew_mean(mo.r2, None)
        return _finish(12.0 * ew_mean(x, self.hl), assets, fb)


class DailyRatioVol(VolForecaster):
    """1-day realized variance mapped to the 21-day scale by a trailing ratio:

        var = 12 * rho * EW_hl(x1),   rho = EW_36(xd) / EW_36(x1)

    both computed only on months of the asset's CURRENT tier segment (so no
    pre-splice scale is carried across a detectable splice). Falls back to
    the 21-day-sum model with the same half-life when the segment holds fewer
    than 12 months with both measurements."""

    def __init__(self, hl: float, ratio_hl: float = RATIO_HL):
        self.hl, self.ratio_hl = hl, ratio_hl

    def forecast_vol(self, view, assets):
        mo = Moments(view, assets, daily=True, one_day=True)
        seg = segment_start(view, assets)
        months = mo.months.values
        segm = np.stack([months >= (np.datetime64(s) if pd.notna(s) else months[-1] + 1)
                         for s in seg.reindex(assets)], 1)
        both = np.isfinite(mo.xd) & np.isfinite(mo.x1) & segm
        rho = ew_mean(mo.xd, self.ratio_hl, both) / ew_mean(mo.x1, self.ratio_hl, both)
        v1 = ew_mean(mo.x1, self.hl, np.isfinite(mo.x1) & segm)
        var = 12.0 * rho * v1
        fb = 12.0 * ew_mean(mo.x, self.hl)
        var = np.where(both.sum(0) >= MIN_RATIO_MONTHS, var, fb)
        return _finish(var, assets, fb)


class MRFixedVol(VolForecaster):
    """var = w * recent + (1 - w) * long-run, w = mr_weight(6) = 0.51 fixed;
    recent = 12 EW_3(x), long-run = 12 mean(x) over all history."""

    def __init__(self, w: float | None = None):
        self.w = mr_weight() if w is None else w

    def forecast_vol(self, view, assets):
        mo = Moments(view, assets, daily=True)
        rec = 12.0 * ew_mean(mo.x, MR_RECENT_HL)
        lng = 12.0 * ew_mean(mo.x, None)
        return _finish(self.w * rec + (1 - self.w) * lng, assets, lng)


class MREstVol(VolForecaster):
    """log var = a + b log recent + (1 - b) log long-run, times a smearing
    factor mean(exp(residual)) (Duan 1983) so the forecast targets E[var]
    rather than exp(E[log var]). Pooled OLS over
    assets and past origins whose 12-month window is fully observed by the
    last December-end <= origin (re-estimated once a year). Training target:
    the sum of ``x`` over the 12 months after each past origin - the target
    estimator on daily data, monthly squared returns before. Falls back to
    the fixed blend with fewer than 120 training rows. The weights sum to one
    (scale equivariance: doubling every variance doubles the forecast) and b
    is clipped to [0, 1]; the intercept absorbs any average bias.

    ``constrained=False`` is the dropped first version (free coefficients on
    both logs, lognormal s^2/2 correction), kept for the report."""

    def __init__(self, constrained: bool = True):
        self.constrained = constrained

    def forecast_vol(self, view, assets):
        allc = list(view.monthly.columns)
        mo = Moments(view, allc, daily=True)
        x = mo.x
        rec = 12.0 * ew_series(x, MR_RECENT_HL)
        lng = 12.0 * ew_series(x, None)
        hist = np.cumsum(np.isfinite(x), 0)
        ok = np.isfinite(x)
        cs = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(np.where(ok, x, 0.0), 0)])
        cn = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(ok, 0)])
        M = mo.M
        rv = np.full_like(x, np.nan)
        if M > 12:
            s = np.arange(M - 12)
            full = (cn[s + 13] - cn[s + 1]) == 12
            rv[s] = np.where(full, cs[s + 13] - cs[s + 1], np.nan)
        is_dec = mo.months.month == 12
        dec = np.nonzero(is_dec)[0]
        cut = dec[-1] if len(dec) else -1
        idx = np.arange(M)[:, None]
        train = (idx + 12 <= cut) & (hist >= MIN_HIST) & (rec > 0) & (lng > 0) & (rv > 0)
        j = [allc.index(a) for a in assets]
        fixed = mr_weight() * rec[-1, j] + (1 - mr_weight()) * lng[-1, j]
        if train.sum() < MIN_TRAIN:
            return _finish(fixed, assets, lng[-1, j])
        lr, ll = np.log(rec), np.log(lng)
        if not self.constrained:
            X = np.column_stack([np.ones(train.sum()), lr[train], ll[train]])
            y = np.log(rv[train])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            s2 = float(np.mean((y - X @ beta) ** 2))
            self.last_fit = dict(beta=beta, s2=s2, n=int(train.sum()), cut=mo.months[cut])
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                lv = beta[0] + beta[1] * lr[-1, j] + beta[2] * ll[-1, j] + 0.5 * s2
            return _finish(np.exp(lv), assets, fixed)
        y = np.log(rv[train]) - ll[train]
        z = lr[train] - ll[train]
        X = np.column_stack([np.ones(len(z)), z])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        b = float(np.clip(beta[1], 0.0, 1.0))
        a = float(np.mean(y - b * z))
        smear = float(np.mean(np.exp(y - a - b * z)))      # Duan (1983) retransformation
        self.last_fit = dict(a=a, b=b, smear=smear, n=int(train.sum()), cut=mo.months[cut])
        with np.errstate(divide="ignore", invalid="ignore"):
            lv = a + b * lr[-1, j] + (1 - b) * ll[-1, j]
        return _finish(smear * np.exp(lv), assets, fixed)
        return _finish(np.exp(lv), assets, fixed)
