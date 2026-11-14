"""Macro family: volatility forecasts conditioned on macroeconomic state and
option-implied volatility, measured as an INCREMENT over an own-history base.

    macro_base       own history only: pooled QLIKE regression of next-year
                     variance on log slow / 12-month / 3-month variance
    macro_state      base + pre-specified macro state variables per asset group
    macro_state_ip   macro_state + industrial-production growth (latest
                     vintage, heavily revised: a sensitivity check only)
    macro_implied    base + option-implied variance (VIX; MOVE x duration;
                     GVZ / OVX) where it exists
    macro_full       base + macro state + implied

See ``model.py`` for the estimator and fallback chain, ``features.py`` for
every input, and reports/models/macro.md for results.
"""
from ..base import register
from .model import MacroVol

register("vol", "macro_base")(lambda: MacroVol(macro=False, implied=False))
register("vol", "macro_state")(lambda: MacroVol(macro=True, implied=False))
register("vol", "macro_state_ip")(lambda: MacroVol(macro=True, implied=False, ip=True))
register("vol", "macro_implied")(lambda: MacroVol(macro=False, implied=True))
register("vol", "macro_full")(lambda: MacroVol(macro=True, implied=True))
