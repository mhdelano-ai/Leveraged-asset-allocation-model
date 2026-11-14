"""Machine-learning vol forecasts on a pooled panel (family ``ml``).

One pooled learner across all assets and month-end origins predicts the
ratio of next-12-month realized variance to the asset's own long-run
variance from trailing realized variances (monthly- and daily-based), a
global volatility state, asset group and optionally macro/implied state.
See ``features``, ``learners``, ``model`` and ``reports/models/ml.md``.

    ml_glm             pooled gamma GLM (log link, L2 by blocked CV)   [linear control]
    ml_gbm_mono        LightGBM, monotone in trailing variances
    ml_gbm             LightGBM, unconstrained
    ml_rf              random forest (Poisson splits)
    ml_glm_macro       ml_glm + macro/implied features
    ml_gbm_mono_macro  ml_gbm_mono + macro/implied features
    ml_avg             mean of ml_glm and ml_gbm_mono variance forecasts
"""
from __future__ import annotations

from ..base import register
from .features import CORE_FEATURES, MACRO_FEATURES
from .learners import GBM, GLM, RF, Average
from .model import MLVol

MACRO = CORE_FEATURES + MACRO_FEATURES

SPECS = {
    "ml_glm": (lambda f: GLM(f), CORE_FEATURES),
    "ml_gbm_mono": (lambda f: GBM(f, monotone=True), CORE_FEATURES),
    "ml_gbm": (lambda f: GBM(f, monotone=False), CORE_FEATURES),
    "ml_rf": (lambda f: RF(f), CORE_FEATURES),
    "ml_glm_macro": (lambda f: GLM(f), MACRO),
    "ml_gbm_mono_macro": (lambda f: GBM(f, monotone=True), MACRO),
    "ml_avg": (lambda f: Average([GLM(f), GBM(f, monotone=True)]), CORE_FEATURES),
}


def _factory(make, feats):
    return lambda: MLVol(make, feats)


for _name, (_make, _feats) in SPECS.items():
    register("vol", _name)(_factory(_make, _feats))
