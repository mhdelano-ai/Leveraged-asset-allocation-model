"""Full forecasters = a registered mean model + a registered dist_* model.

:func:`~assetpred.models.dist_core.register_combo` builds
:class:`~assetpred.models.dist_core.MeanPlusDist`: the ``mean`` column comes
from the mean model (fitted on its own refit cadence, on the original
features), the quantiles are that mean plus the dist model's offsets (fitted
on residuals about the historical mean, see ``dist_core``). Add a line to
``COMBOS`` (or call ``register_combo`` from any module under
``assetpred/models``) and run it like any model::

    python -m assetpred.evaluation.run --dev --models ridge__dist_lin_agg ...

Registry name: ``<mean>__<dist>`` unless a name is given.
"""
from __future__ import annotations

from .dist_core import register_combo

# (mean model, dist model[, name]); the mean model must be registered.
COMBOS = [
    ("ridge", "dist_lin_agg"),        # demonstration with the repo's reference point model
    ("hist_mean", "dist_lin_agg"),    # sanity: same forecasts as dist_lin_agg itself
    ("ens_z", "dist_lin_agg"),        # config.COMPOSITE: the frozen full-distribution forecast
]

for _c in COMBOS:
    register_combo(*_c)
