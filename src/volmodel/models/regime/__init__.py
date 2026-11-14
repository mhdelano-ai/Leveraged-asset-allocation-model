"""Regime-switching and structural-break volatility models (family ``regime``).

Markov switching (filtered regime probabilities x the 12-step-ahead regime
distribution):

    regime_ms2          per asset, 2 states, Student-t(5), monthly returns
    regime_ms3          per asset, 3 states, Student-t(5), monthly returns
    regime_ms2_weekly   per asset, 2 states, 5-day sums of daily returns mapped
                        to the 21-day scale per splice segment
    regime_common3      one 3-state regime shared by all assets (multivariate
                        t with a fixed correlation matrix), asset-specific
                        variance per regime

Structural breaks in the level:

    regime_bocpd        Bayesian online change-point detection, hazard 1/240
    regime_bocpd_eb     the same with the hazard chosen each year by the marginal
                        likelihood of all assets (empirical Bayes)
    regime_bocpd_weekly the same as regime_bocpd on 5-day sums (hazard scaled to
                        calendar time)
    regime_break_blend  ref_blend with the long-run level taken since the last
                        kappa-2 (HAC CUSUM-of-squares, 1%) break

See ``models.py`` for definitions and ``reports/models/regime.md`` for results.
"""
from ..base import register
from .models import BOCPDEmpiricalBayes, BOCPDMonthly, BOCPDWeekly, BreakBlend, CommonMonthly, MSMonthly, MSWeekly

register("vol", "regime_ms2")(lambda: MSMonthly(2))
register("vol", "regime_ms3")(lambda: MSMonthly(3))
register("vol", "regime_ms2_weekly")(lambda: MSWeekly(2))
register("vol", "regime_common3")(lambda: CommonMonthly(3))
register("vol", "regime_bocpd")(lambda: BOCPDMonthly(1 / 240))
register("vol", "regime_bocpd_eb")(BOCPDEmpiricalBayes)
register("vol", "regime_bocpd_weekly")(lambda: BOCPDWeekly(1 / 240))
register("vol", "regime_break_blend")(BreakBlend)
