"""Structural family: forecasts built from the economic structure of each asset.

A bond's return volatility is its duration times the volatility of its yield.
Duration is observable today and moves mechanically with the yield level;
yield volatility is the slowly varying piece forecast from history. Credit
adds a spread component whose volatility is proportional to the spread level
(duration-times-spread); TIPS take the nominal yield-vol structure with a
shrunk real/nominal ratio. The covariance model is a small factor model with
observable or slowly estimated exposures. See reports/models/structural.md.

Registered models
    vol  structural_normal      beta = 0 (basis-point yield vol is stable)
    vol  structural_lognormal   beta = 1 (relative yield vol is stable)
    vol  structural_sqrt        beta = 1/2 (a-priori midpoint, CEV square root)
    vol  structural_cev         beta estimated walk-forward (once a year)
    cov  structural_factor      factor model (see factor.py)
"""
from __future__ import annotations

from ..base import register
from .vol import StructuralVol

register("vol", "structural_normal")(lambda: StructuralVol(beta=0.0))
register("vol", "structural_lognormal")(lambda: StructuralVol(beta=1.0))
register("vol", "structural_sqrt")(lambda: StructuralVol(beta=0.5))
register("vol", "structural_cev")(lambda: StructuralVol(beta="est"))

from .factor import StructuralFactorCov  # noqa: E402

register("cov", "structural_factor_lognormal")(lambda: StructuralFactorCov(beta=1.0))
register("cov", "structural_factor_normal")(lambda: StructuralFactorCov(beta=0.0))
