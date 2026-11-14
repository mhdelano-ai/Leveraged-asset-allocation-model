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


def _block_ids(groups: list[str]) -> np.ndarray:
    """Integer id of the (group, group) block each off-diagonal entry is in."""
    g = np.asarray(groups)
    u = {x: k for k, x in enumerate(dict.fromkeys(groups))}
    gi = np.array([u[x] for x in g])
    lo, hi = np.minimum.outer(gi, gi), np.maximum.outer(gi, gi)
    return lo * len(u) + hi


def block_target(c: np.ndarray, groups: list[str]) -> np.ndarray:
    """Average correlation within each group and between each pair of groups."""
    n = len(groups)
    b = _block_ids(groups)
    off = ~np.eye(n, dtype=bool)
    F = np.eye(n)
    for k in np.unique(b[off]):
        m = (b == k) & off
        F[m] = c[m].mean()
    return F


def shrink_intensity(c: np.ndarray, F: np.ndarray, neff: np.ndarray,
                     groups: list[str] | None = None) -> float:
    """Schaefer-Strimmer / Ledoit-Wolf style intensity for a correlation matrix:
    sum Var(r_ij) / sum (r_ij - f_ij)^2 over i < j, with the large-sample
    normal variance (1 - r^2)^2 / n_eff, clipped to [0, 1]. Entries alone in
    their block (``groups`` given) have f_ij = r_ij - shrinkage cannot move
    them - and are left out of both sums."""
    n = len(c)
    iu = np.triu_indices(n, 1)
    keep = np.ones(len(iu[0]), bool)
    if groups is not None:
        b = _block_ids(groups)[iu]
        _, inv, cnt = np.unique(b, return_inverse=True, return_counts=True)
        keep = cnt[inv] >= 2
    r, f, ne = c[iu][keep], F[iu][keep], neff[iu][keep]
    if not len(r):
        return 0.0
    var = np.where(ne > 1, (1 - r ** 2) ** 2 / np.maximum(ne, 1), 0.0)
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
        groups = ([BY_ID[a].group if a in BY_ID else a for a in assets] if self.target == "block"
                  else ["all"] * len(assets))
        F = block_target(c, groups)
        d = shrink_intensity(c, F, neff, groups)
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
