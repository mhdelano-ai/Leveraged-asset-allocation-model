"""Bench correlation models (see the package docstring for formulas)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...universe import BY_ID
from ..base import CorrForecaster
from .core import Moments, ew_corr, nearest_corr

BLEND_W = 0.5   # weight on the recent correlation (same AR(1), H = 6 months argument as the vol blend)


def _frame(c, assets):
    return pd.DataFrame(nearest_corr(c), index=assets, columns=assets)


class EWCorr(CorrForecaster):
    """Pairwise uncentered EW correlation of per-month co-moments (21-day sums
    of daily data where both assets have them, monthly returns otherwise, or
    monthly only), repaired to positive definite."""

    def __init__(self, hl: float | None, daily: bool):
        self.hl, self.daily = hl, daily

    def forecast_corr(self, view, assets):
        mo = Moments(view, assets, pairs=True, daily=self.daily)
        return _frame(ew_corr(mo, self.hl)[0], assets)


def block_target(c: np.ndarray, groups: list[str]) -> np.ndarray:
    """Average correlation within each group and between each pair of groups."""
    g = np.asarray(groups)
    n = len(g)
    F = np.eye(n)
    off = ~np.eye(n, dtype=bool)
    for a in np.unique(g):
        for b in np.unique(g):
            m = np.outer(g == a, g == b) & off
            if m.any():
                F[m] = c[m].mean()
    return F


def shrink_intensity(c: np.ndarray, F: np.ndarray, neff: np.ndarray) -> float:
    """Schaefer-Strimmer / Ledoit-Wolf style intensity for a correlation matrix:
    sum Var(r_ij) / sum (r_ij - f_ij)^2 over i < j, with the large-sample
    normal variance (1 - r^2)^2 / n_eff, clipped to [0, 1]."""
    iu = np.triu_indices(len(c), 1)
    r, f, n = c[iu], F[iu], neff[iu]
    var = np.where(n > 1, (1 - r ** 2) ** 2 / np.maximum(n, 1), 0.0)
    den = float(((r - f) ** 2).sum())
    if den <= 0:
        return 1.0
    return float(np.clip(var.sum() / den, 0.0, 1.0))


class ShrinkCorr(CorrForecaster):
    """EW 21-day-sum correlation shrunk toward the block (asset-group) target
    with an intensity estimated from the view."""

    def __init__(self, hl: float = 36, target: str = "block"):
        self.hl, self.target = hl, target

    def forecast_corr(self, view, assets):
        mo = Moments(view, assets, pairs=True, daily=True)
        c, neff = ew_corr(mo, self.hl)
        if self.target == "block":
            F = block_target(c, [BY_ID[a].group if a in BY_ID else a for a in assets])
        else:
            F = block_target(c, ["all"] * len(assets))
        d = shrink_intensity(c, F, neff)
        self.last_delta = d
        return _frame(d * F + (1 - d) * c, assets)


class BlendCorr(CorrForecaster):
    """0.5 x EW(hl=12) + 0.5 x all-history correlation, 21-day sums where
    available, monthly otherwise."""

    def __init__(self, hl_recent: float = 12, w: float = BLEND_W):
        self.hl, self.w = hl_recent, w

    def forecast_corr(self, view, assets):
        mo = Moments(view, assets, pairs=True, daily=True)
        rec = ew_corr(mo, self.hl)[0]
        lng = ew_corr(mo, None)[0]
        return _frame(self.w * rec + (1 - self.w) * lng, assets)
