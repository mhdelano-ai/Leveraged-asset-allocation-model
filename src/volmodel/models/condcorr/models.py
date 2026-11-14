"""The condcorr CorrForecasters: conditional block correlations embedded in
the unconditional base matrix (``ref_corr_last120m``)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import CorrForecaster
from ..ref import _CorrRecent
from .data import fisher_z
from .embed import embed_anchor_targets
from .markov import predict_ms
from .stockbond import CORE_PAIRS, MACRO, RECENT, SHORT, STRESS, STRESS_PAIRS, predict_stockbond


class CondCorrBase(CorrForecaster):
    """Base matrix plus conditional Fisher-z shifts of chosen block pairs."""

    def __init__(self):
        self.base = _CorrRecent()           # ref_corr_last120m (read-only reuse)
        self.last_info: dict = {}

    def base_matrix(self, view, assets):
        return self.base.forecast_corr(view, assets).to_numpy(float)

    def targets(self, view, assets, C0) -> dict:  # pragma: no cover - abstract
        """{(a, b): target correlation} for the anchor pairs (empty = base matrix)."""
        return {}

    def forecast_corr(self, view, assets):
        assets = list(assets)
        C0 = self.base_matrix(view, assets)
        T = self.targets(view, assets, C0)
        C, lam = embed_anchor_targets(C0, assets, T)
        self.last_info["lambda"] = lam
        self.last_info["targets"] = T
        return pd.DataFrame(C, index=assets, columns=assets)


def block_targets(assets, C0, z_pred: dict) -> dict:
    """Anchor targets for a block: every anchor pair moves by the MEAN Fisher-z
    shift of the predicted pairs relative to the base matrix (the pairs are
    near-duplicates; a common shift avoids amplifying their differences
    through the nearly collinear anchors)."""
    pos = {a: i for i, a in enumerate(assets)}
    z0 = {p: fisher_z(C0[pos[p[0]], pos[p[1]]]) for p in z_pred}
    dz = float(np.mean([z_pred[p] - z0[p] for p in z_pred]))
    return {p: float(np.tanh(z0[p] + dz)) for p in z_pred}


class SBRegression(CondCorrBase):
    def __init__(self, wanted: list[str], pairs=CORE_PAIRS):
        super().__init__()
        self.wanted = list(wanted)
        self.pairs = list(pairs)

    def targets(self, view, assets, C0):
        pairs = [p for p in self.pairs if p[0] in assets and p[1] in assets]
        pred = predict_stockbond(view, pairs, self.wanted)
        self.last_info = {"pred": pred}
        if pred is None:
            return {}
        if len(self.pairs) == len(CORE_PAIRS) and set(self.pairs) == set(CORE_PAIRS):
            return block_targets(assets, C0, pred.z)
        return {p: float(np.tanh(z)) for p, z in pred.z.items()}   # distinct pairs: own targets


class SBMarkov(CondCorrBase):
    """Stock-bond correlation from a two-state Markov-switching model on
    (us_large, ust_int) - or ust_long where ust_int is not yet eligible."""

    def __init__(self, tvtp: bool = False):
        super().__init__()
        self.tvtp = tvtp

    def targets(self, view, assets, C0):
        pairs = [p for p in CORE_PAIRS if p[0] in assets and p[1] in assets]
        if not pairs:
            self.last_info = {"pred": None}
            return {}
        a, b = pairs[0]                             # ust_int first if eligible
        f = predict_ms(view, a, b, tvtp=self.tvtp)
        self.last_info = {"pred": f}
        if f is None:
            return {}
        pos = {x: i for i, x in enumerate(assets)}
        dz = float(fisher_z(f.rho) - fisher_z(C0[pos[a], pos[b]]))
        return block_targets(assets, C0, {p: fisher_z(C0[pos[p[0]], pos[p[1]]]) + dz for p in pairs})


SB_BASE = [RECENT]
SB_STATE = [RECENT, SHORT] + MACRO
SB_MACRO = list(MACRO)
