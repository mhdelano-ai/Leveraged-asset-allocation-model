"""Registered predictive-distribution models (location: historical mean).

Every model here is a :class:`~assetpred.models.dist_core.Composite`; pair
any of them with a registered mean model through
:func:`~assetpred.models.dist_core.register_combo` (see ``dist_combos.py``).
The configurations tried and their dev scores are logged in
``results/dev/trials_dist.csv``.
"""
from __future__ import annotations

from .dist_core import Composite
from .dist_scale import AggScale, CondShapeGBM, EWMAScale, GBMScale, LinearScale
from .registry import register


# Factories are module-level functions (picklable for worker processes).
@register("dist_ewma_fhs")
def _ewma_fhs():
    """vol_ewma_fhs re-expressed as a Composite (identical forecasts); the
    location-pluggable non-ML reference."""
    return Composite([EWMAScale(shape="std")])


@register("dist_ewma_grp")
def _ewma_grp():
    """EWMA scale; shape per asset class shrunk to pooled, not rescaled."""
    return Composite([EWMAScale(shape="group")])


@register("dist_lin")
def _lin():
    """Linear QLIKE log-variance model (EWMA, slow EWMA, 12m vol, daily vol, VIX,
    down-move); pooled shape."""
    return Composite([LinearScale(shape="pooled")])


@register("dist_gbm_vol")
def _gbm_vol():
    """Linear start + boosted corrections on volatility/stress features; pooled OOF shape."""
    return Composite([GBMScale(features="vol", shape="pooled")])


@register("dist_gbm_all")
def _gbm_all():
    """As dist_gbm_vol with every live non-hindsight feature."""
    return Composite([GBMScale(features="all", shape="pooled")])


@register("dist_lin_agg")
def _lin_agg():
    """Linear QLIKE monthly variance, aggregated affinely to h months; pooled shape."""
    return Composite([AggScale(monthly="linear", shape="pooled")])


@register("dist_gbm_agg")
def _gbm_agg():
    """As dist_lin_agg with the boosted (vol-feature) monthly variance model; annual refits."""
    return Composite([AggScale(monthly="gbm_vol", shape="pooled", refit_every=12)])


@register("dist_lin_agg_grp")
def _lin_agg_grp():
    """dist_lin_agg with the shape per asset class (shrunk to pooled)."""
    return Composite([AggScale(monthly="linear", shape="group")])


@register("dist_lin_agg_cshape")
def _lin_agg_cshape():
    """dist_lin_agg scale + conditional shape (quantile GBM on out-of-fold z; annual refits)."""
    return Composite([CondShapeGBM(scale="agg_linear", refit_every=12)])


@register("dist_ens")
def _ens():
    """Equal-weight quantile average of dist_lin_agg and dist_ewma_fhs."""
    return Composite([AggScale(monthly="linear", shape="pooled"), EWMAScale(shape="std")])


@register("dist_lin_agg_rw")
def _lin_agg_rw():
    """dist_lin_agg with the pooled shape weighted by recency (half-life 240 months).
    Post-hoc trial (motivated by the dev-period q99 hit deficit)."""
    return Composite([AggScale(monthly="linear", shape="pooled", half_life=240)])


@register("dist_lin_agg_ret")
def _lin_agg_ret():
    """Ablation of dist_lin_agg: monthly-return inputs only (no daily RV, no VIX)."""
    return Composite([AggScale(monthly="linear", shape="pooled", inputs="returns")])
