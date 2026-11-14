"""HAR family: direct multi-horizon regressions for the 12-month variance.

The realized 12-month target variance is regressed on trailing variances
measured over several horizons (1, 3, 12, 60 months and the full history),
in a form normalized by each asset's long-run variance, and forecast directly
(no iteration). Definitions, formulas and the reasons for every fixed number
are in ``features.py`` (regressors) and ``model.py`` (estimation), and in
reports/models/har.md.

Eight pre-specified variants, chosen so that each pair isolates one choice:

    name                 regressors      form     pooling
    har_pooled_log_m     monthly         log      pooled
    har_pooled_log_d     daily-mapped    log      pooled      (vs _m: daily information)
    har_pooled_logcvx_d  daily-mapped    log      pooled, b >= 0, sum b <= 1   (vs _log_d: constraints)
    har_pooled_logfe_d   daily-mapped    log      pooled slopes, shrunk asset intercepts
    har_asset_log_d      daily-mapped    log      per asset   (vs pooled)
    har_pooled_qlike_m   monthly         QLIKE    pooled, convex
    har_pooled_qlike_d   daily-mapped    QLIKE    pooled, convex   (vs logcvx_d: log vs level/QLIKE)
    har_asset_qlike_d    daily-mapped    QLIKE    per asset, convex
"""
from __future__ import annotations

from ..base import register
from .model import HarVol

SPECS = {
    "har_pooled_log_m": ("M", "log", "pooled"),
    "har_pooled_log_d": ("D", "log", "pooled"),
    "har_pooled_logcvx_d": ("D", "logcvx", "pooled"),
    "har_pooled_logfe_d": ("D", "log", "fe"),
    "har_asset_log_d": ("D", "log", "asset"),
    "har_pooled_qlike_m": ("M", "qlike", "pooled"),
    "har_pooled_qlike_d": ("D", "qlike", "pooled"),
    "har_asset_qlike_d": ("D", "qlike", "asset"),
}


def _factory(spec):
    return lambda: HarVol(*spec)


for _name, _spec in SPECS.items():
    register("vol", _name)(_factory(_spec))

__all__ = ["HarVol", "SPECS"]
