"""Unlevered portfolio composition.

Inverse-volatility weighting with **cluster risk budgets**. The cluster layer is
not decoration: naive inverse-vol across this universe hands roughly half the
book to five sleeves that are all equity beta -- US, international, EM, REITs and
high yield. Their correlations converge in a crisis, so the "diversified"
portfolio is a levered equity fund wearing a disguise, and it is levered on the
strength of a diversification benefit that is not there.

Budgets are expressed as shares of *risk*, not capital.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Sleeve -> cluster. Anything unmapped falls into "other".
DEFAULT_CLUSTERS = {
    "us_equity": "growth",
    "intl_equity": "growth",
    "em_equity": "growth",
    "reit": "growth",
    "hy_credit": "growth",
    "long_ust": "rates",
    "interm_ust": "rates",
    "ig_credit": "credit",
    "gold": "real",
    "commodities": "real",
}

DEFAULT_BUDGETS = {"growth": 0.40, "rates": 0.30, "real": 0.20, "credit": 0.10}

# Gold's 1971-80 run was a one-off monetary regime change and will not repeat; an
# optimiser fed that decade adores gold. Cap it a priori rather than letting the
# search decide. (Its -70% 1980-2001 collapse is the antidote, and is in-sample.)
DEFAULT_CAPS = {"gold": 0.15, "commodities": 0.20, "em_equity": 0.15, "hy_credit": 0.15}


def inverse_vol_weights(vols: pd.DataFrame, eligible: pd.DataFrame | None = None) -> pd.DataFrame:
    """Weights proportional to 1/sigma, renormalised to sum to 1 per row."""
    inv = 1.0 / vols.replace(0.0, np.nan)
    if eligible is not None:
        inv = inv.where(eligible.reindex_like(inv).fillna(False), 0.0)
    inv = inv.fillna(0.0)
    totals = inv.sum(axis=1).replace(0.0, np.nan)
    return inv.div(totals, axis=0).fillna(0.0)


def apply_caps(weights: pd.DataFrame, caps: dict[str, float] | None = None) -> pd.DataFrame:
    """Cap individual sleeves and redistribute the excess proportionally."""
    caps = DEFAULT_CAPS if caps is None else caps
    out = weights.copy()
    for _ in range(8):
        over = pd.Series(False, index=out.columns)
        for col, cap in caps.items():
            if col in out.columns:
                over[col] = bool((out[col] > cap + 1e-9).any())
        if not over.any():
            break
        for col, cap in caps.items():
            if col in out.columns:
                out[col] = out[col].clip(upper=cap)
        deficit = 1.0 - out.sum(axis=1)
        free = [c for c in out.columns if c not in caps or caps.get(c, 1.0) >= 1.0]
        free = free or list(out.columns)
        free_mass = out[free].sum(axis=1).replace(0.0, np.nan)
        for col in free:
            out[col] = out[col] + deficit * (out[col] / free_mass).fillna(0.0)
    totals = out.sum(axis=1).replace(0.0, np.nan)
    return out.div(totals, axis=0).fillna(0.0)


def apply_cluster_budgets(
    weights: pd.DataFrame,
    vols: pd.DataFrame,
    *,
    clusters: dict[str, str] | None = None,
    budgets: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rescale so each cluster contributes its budgeted share of standalone risk.

    Uses standalone risk (w * sigma) rather than a full covariance contribution:
    it needs no matrix inverse, is stable when a sleeve has little history, and
    the difference is immaterial once the crisis-covariance floor is doing the
    real work in the sizer.
    """
    clusters = DEFAULT_CLUSTERS if clusters is None else clusters
    budgets = DEFAULT_BUDGETS if budgets is None else budgets

    risk = weights * vols.reindex_like(weights).ffill()
    out = weights.copy()

    present: dict[str, list[str]] = {}
    for col in weights.columns:
        present.setdefault(clusters.get(col, "other"), []).append(col)

    active = {c: budgets.get(c, 0.0) for c in present}
    total_budget = sum(active.values())
    if total_budget <= 0:
        return weights
    active = {c: v / total_budget for c, v in active.items()}

    for cluster, cols in present.items():
        target = active.get(cluster, 0.0)
        cluster_risk = risk[cols].sum(axis=1)
        total_risk = risk.sum(axis=1).replace(0.0, np.nan)
        current = (cluster_risk / total_risk).replace([np.inf, -np.inf], np.nan)
        scale = (target / current).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        scale = scale.clip(upper=20.0)
        for col in cols:
            out[col] = out[col] * scale

    totals = out.sum(axis=1).replace(0.0, np.nan)
    return out.div(totals, axis=0).fillna(0.0)


def build(
    returns: pd.DataFrame,
    vols: pd.DataFrame,
    *,
    eligible: pd.DataFrame | None = None,
    clusters: dict[str, str] | None = None,
    budgets: dict[str, float] | None = None,
    caps: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Full base-weight pipeline: inverse-vol, cluster budgets, then caps."""
    w = inverse_vol_weights(vols, eligible)
    w = apply_cluster_budgets(w, vols, clusters=clusters, budgets=budgets)
    return apply_caps(w, caps)
