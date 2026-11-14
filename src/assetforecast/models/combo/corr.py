"""Correlation combinations of stored forecasts (see the package docstring).

``C = sum_g w_g C_g`` where ``C_g`` is the mean of the member matrices in
group ``g`` (a family, or a single member). Every stored member matrix is a
valid correlation matrix (unit diagonal, positive definite: the harness
validates it), and a convex combination of such matrices is again one, so no
repair is ever needed.

Optional block shrinkage: ``C <- (1 - d) C + d B(C)``, where ``B(C)``
replaces every entry by the mean of its asset-group block (within-group
off-diagonal mean; between-group mean; unit diagonal; groups from
``universe``). ``B(C)`` is the average of ``P C P'`` over all permutations
``P`` that only reorder assets within their groups, i.e. an average of
positive definite correlation matrices, so it is itself one and the shrunk
matrix stays positive definite for any ``d`` in [0, 1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import CorrForecaster
from ...universe import BY_ID
from .store import MissingComponentError, family_of, load, results_dir_in_use


def block_average(C: np.ndarray, groups: list[str]) -> np.ndarray:
    """Block-mean target of a correlation matrix (see module docstring)."""
    groups = np.asarray(groups)
    n = len(groups)
    B = np.eye(n)
    labs = list(dict.fromkeys(groups))
    idx = {g: np.flatnonzero(groups == g) for g in labs}
    for gi in labs:
        for gj in labs:
            I, J = idx[gi], idx[gj]
            sub = C[np.ix_(I, J)]
            if gi == gj:
                if len(I) < 2:
                    continue
                m = (sub.sum() - np.trace(sub)) / (len(I) * (len(I) - 1))
                blk = np.full((len(I), len(I)), m)
                np.fill_diagonal(blk, 1.0)
            else:
                blk = np.full((len(I), len(J)), sub.mean())
            B[np.ix_(I, J)] = blk
    return B


class CorrCombo(CorrForecaster):
    """Convex combination of stored correlation forecasts."""

    def __init__(self, members: list[str] | None = None, *, group: str = "family",
                 weights: dict | None = None, block_shrink: float = 0.0,
                 exclude: tuple = ("ref_corr_zero",), exclude_families: tuple = ("combo",),
                 results_dir=None):
        if group not in ("family", "none"):
            raise ValueError("group must be 'family' or 'none'")
        if not 0.0 <= block_shrink <= 1.0:
            raise ValueError("block_shrink must be in [0, 1]")
        self.members = list(members) if members is not None else None
        self.group, self.weights, self.block_shrink = group, weights, float(block_shrink)
        self.exclude, self.exclude_families = tuple(exclude), tuple(exclude_families)
        self.results_dir = results_dir

    def member_list(self) -> list[str]:
        if self.members is not None:
            return list(self.members)
        from ..base import list_models
        out = [m for m in list_models("corr")
               if family_of(m) not in self.exclude_families and m not in self.exclude]
        out += [f"cov:{m}" for m in list_models("cov")
                if family_of(m) not in self.exclude_families and m not in self.exclude]
        return out

    def groups(self) -> list[tuple[str, list[str]]]:
        members = self.member_list()
        if self.group == "none":
            return [(m, [m]) for m in members]
        fams: dict[str, list[str]] = {}
        for m in members:
            fams.setdefault(family_of(m), []).append(m)
        return sorted(fams.items())

    def group_weights(self, groups) -> np.ndarray:
        if self.weights is None:
            return np.full(len(groups), 1.0 / len(groups))
        w = np.array([float(self.weights[g]) for g, _ in groups])
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError("combination weights must be non-negative with a positive sum")
        return w / w.sum()

    def forecast_corr(self, view, assets):
        root = results_dir_in_use(self.results_dir)
        groups = self.groups()
        if not groups:
            raise MissingComponentError(f"combo {getattr(self, 'name', '?')!r} has no members")
        t = pd.Timestamp(view.asof_date if view.asof_date is not None else view.monthly.index[-1])
        w = self.group_weights(groups)
        C = np.zeros((len(assets), len(assets)))
        for wg, (_, mem) in zip(w, groups):
            Cg = np.mean([load("corr", m, root).at(t, assets) for m in mem], axis=0)
            C += wg * Cg
        if self.block_shrink > 0:
            B = block_average(C, [BY_ID[a].group for a in assets])
            C = (1 - self.block_shrink) * C + self.block_shrink * B
        C = (C + C.T) / 2.0
        np.fill_diagonal(C, 1.0)
        return pd.DataFrame(C, index=list(assets), columns=list(assets))
