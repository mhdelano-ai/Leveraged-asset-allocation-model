"""statespace: latent log-volatility models with explicit measurement noise.

Every realized-variance measurement is a noisy observation of a persistent
latent log variance ``h``. The models here filter ``h`` with a state-space
model whose measurement-noise variance is (approximately) known from the
estimator, and forecast the 12-month target as the conditional mean of the
summed monthly variances, ``s * sum_k E[exp(h_{t+k})]`` (lognormal
retransformation with the predictive variance).

    statespace_ar1_asset    AR(1) latent log variance, monthly returns only,
                            parameters estimated asset by asset (no pooling)
    statespace_ar1          the same with persistence, vol-of-vol and noise
                            variance pooled across assets and a hierarchical
                            (empirical Bayes) prior on each asset's level
    statespace_ar1_grid     statespace_ar1's parameters, but an EXACT
                            non-Gaussian filter on a grid (r | h ~ N(0, e^h))
                            instead of the Kalman filter on log r^2
    statespace_2c           pooled, two AR(1) components (slow + fast)
    statespace_ar1_daily    statespace_ar1 + a second measurement, the log of
                            the month's daily realized variance, whose level
                            relative to h is a learnt offset state (random
                            walk, re-opened at splices)
    statespace_2c_daily     two components + the daily measurement
    statespace_factor       common log-vol factor (loadings by asset group)
                            + idiosyncratic AR(1), monthly returns only
    statespace_factor_daily the factor model + the daily measurement

See ``models.py`` for the equations and ``reports/models/statespace.md``.
"""
from __future__ import annotations

from ..base import register
from .models import FactorSpec, FactorVol, GridVol, Spec, StateSpaceVol

register("vol", "statespace_ar1_asset")(lambda: StateSpaceVol(Spec(ncomp=1, daily=False, pooled=False)))
register("vol", "statespace_ar1")(lambda: StateSpaceVol(Spec(ncomp=1, daily=False, pooled=True)))
register("vol", "statespace_ar1_grid")(GridVol)
register("vol", "statespace_2c")(lambda: StateSpaceVol(Spec(ncomp=2, daily=False, pooled=True)))
register("vol", "statespace_ar1_daily")(lambda: StateSpaceVol(Spec(ncomp=1, daily=True, pooled=True)))
register("vol", "statespace_2c_daily")(lambda: StateSpaceVol(Spec(ncomp=2, daily=True, pooled=True)))
register("vol", "statespace_factor")(lambda: FactorVol(FactorSpec(daily=False, refit=1)))
register("vol", "statespace_factor_daily")(lambda: FactorVol(FactorSpec(daily=True, refit=1)))
