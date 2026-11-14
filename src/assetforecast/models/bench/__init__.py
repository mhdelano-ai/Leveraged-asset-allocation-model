"""bench: the strong simple benchmarks.

Exponentially weighted averages of per-month variance / co-moment
measurements, mean-reversion blends and shrinkage - the forecasts that
sophisticated models most often fail to beat. Building blocks and the exact
per-month estimators are in ``core`` (read its docstring); the report is
``reports/models/bench.md``.

Notation: ``x_m`` = month m's variance measured from 21-day sums of daily
log excess returns where the asset has daily data (the target's own
estimator), else ``r_m^2``; ``EW_h(.)`` = mean over months with weights
``0.5 ** (age / h)``, h in months, missing months dropped; ``EW_inf`` = equal
weights over all history.

Volatility (annualized variance; vol = sqrt)

    bench_vol_m_hl12     12 EW_12(r^2)       monthly returns only, half-life 1 year
    bench_vol_m_hl36     12 EW_36(r^2)       3 years
    bench_vol_m_hl120    12 EW_120(r^2)      10 years
    bench_vol_d_hl12     12 EW_12(x)         21-day sums of daily data, else monthly
    bench_vol_d_hl36     12 EW_36(x)
    bench_vol_d1r_hl12   12 rho EW_12(x1),   x1 = sum of squared 1-day returns in the
                         month, rho = EW_36(xd)/EW_36(x1) on the current tier segment
    bench_vol_mr_fixed   w 12 EW_3(x) + (1-w) 12 EW_inf(x),  w = 0.51 (AR(1) half-life 6m)
    bench_vol_mr_est     exp(a + b log 12EW_3(x) + c log 12EW_inf(x) + s^2/2), (a,b,c)
                         pooled OLS on past fully observed 12-month windows, yearly

Correlation (uncentered, pairwise, repaired to PD by eigenvalue clipping)

    bench_corr_m_hl36        EW_36 of monthly co-moments
    bench_corr_m_hl120       EW_120 of monthly co-moments
    bench_corr_d_hl36        EW_36 of 21-day-sum co-moments where both assets have
                             daily data, monthly co-moments otherwise
    bench_corr_d_hl120       same, half-life 10 years
    bench_corr_shrink_block  bench_corr_d_hl36 shrunk toward the asset-group block
                             target, intensity estimated (Schaefer-Strimmer / LW)
    bench_corr_blend         0.5 EW_12 + 0.5 EW_inf, 21-day sums where available
"""
from __future__ import annotations

from ..base import register
from .corr import BlendCorr, EWCorr, ShrinkCorr, block_target, shrink_intensity
from .core import Moments, ew_corr, ew_mean, ew_series, nearest_corr
from .vol import DailyRatioVol, EWVol, MREstVol, MRFixedVol, mr_weight

VOL_MODELS = {
    "bench_vol_m_hl12": lambda: EWVol(12, daily=False),
    "bench_vol_m_hl36": lambda: EWVol(36, daily=False),
    "bench_vol_m_hl120": lambda: EWVol(120, daily=False),
    "bench_vol_d_hl12": lambda: EWVol(12, daily=True),
    "bench_vol_d_hl36": lambda: EWVol(36, daily=True),
    "bench_vol_d1r_hl12": lambda: DailyRatioVol(12),
    "bench_vol_mr_fixed": MRFixedVol,
    "bench_vol_mr_est": MREstVol,
}
CORR_MODELS = {
    "bench_corr_m_hl36": lambda: EWCorr(36, daily=False),
    "bench_corr_m_hl120": lambda: EWCorr(120, daily=False),
    "bench_corr_d_hl36": lambda: EWCorr(36, daily=True),
    "bench_corr_d_hl120": lambda: EWCorr(120, daily=True),
    "bench_corr_shrink_block": lambda: ShrinkCorr(36, "block"),
    "bench_corr_blend": BlendCorr,
}
for _n, _f in VOL_MODELS.items():
    register("vol", _n)(_f)
for _n, _f in CORR_MODELS.items():
    register("corr", _n)(_f)
