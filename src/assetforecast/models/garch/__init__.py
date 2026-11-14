"""GARCH family: conditional-variance models for the 12-month average variance.

At a 12-month horizon a GARCH(1,1) has mean-reverted almost entirely to its
long-run variance, so what matters is how that level is estimated and
whether the model has a slowly moving component. Members (all Student-t,
zero mean, variance targeting; see ``reports/models/garch.md``):

    garch_d_garch11   daily GARCH(1,1), VR-adjusted returns, 20-year window
    garch_d_gjr       daily GJR-GARCH(1,1) (asymmetric baseline)
    garch_d_cgarch    daily Engle-Lee component GARCH (slow long-run component)
    garch_d_midas     daily GARCH-MIDAS: unit-mean daily GARCH x long-run
                      component driven by 36 months of realized variance
    garch_d_rm2006    daily RiskMetrics-2006 long-memory ARCH (fixed constants)
    garch_w_cgarch    component GARCH on non-overlapping 5-day returns
    garch_m_garch11   monthly GARCH(1,1), parameters pooled across assets
    garch_m_cgarch    monthly component GARCH, parameters pooled across assets

Daily/weekly inputs are rescaled to the 21-day convention by a causal,
segment-local variance ratio (``prep``). Parameters are re-estimated on data
through each December 31 and the variance state is filtered with fixed
parameters to the origin. The forecast is the SUM of the multi-step
conditional variances over the next 12 months' steps (trading days counted
over the trailing year, so ~295 before 1952), in closed form. Assets
without enough daily data fall back to the pooled monthly GARCH(1,1); every
eligible asset gets a finite forecast.
"""
from __future__ import annotations

from ..base import register
from .core import CGarch, Garch11, GJR, Midas, RM2006
from .models import HighFreq, MonthlyPooled

register("vol", "garch_d_garch11")(lambda: HighFreq(Garch11, k=1))
register("vol", "garch_d_gjr")(lambda: HighFreq(GJR, k=1))
register("vol", "garch_d_cgarch")(lambda: HighFreq(CGarch, k=1))
register("vol", "garch_d_midas")(lambda: HighFreq(Midas, k=1))
register("vol", "garch_d_rm2006")(lambda: HighFreq(RM2006, k=1))
register("vol", "garch_w_cgarch")(lambda: HighFreq(CGarch, k=5))
register("vol", "garch_m_garch11")(lambda: MonthlyPooled(Garch11))
register("vol", "garch_m_cgarch")(lambda: MonthlyPooled(CGarch))
