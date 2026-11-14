"""condcorr: correlations conditioned on economic state.

Question: do observable conditions tell us which correlation regime the next
12 months will be in? The block that matters is equity x rates (the
stock-bond correlation). Every model here starts from the unconditional
``ref_corr_last120m`` matrix and moves only the equity x rates block (and,
for ``condcorr_stress``, equity x credit/REIT), in Fisher-z, by an amount
predicted from the state; see ``models``, ``stockbond``, ``markov``,
``embed`` and reports/models/condcorr.md.

    condcorr_sb_base    baseline correction from the recent 12m correlation only
    condcorr_sb_state   + 63-day correlation + inflation level, inflation vol, long yield
    condcorr_sb_macro   inflation level, inflation vol, long yield only (no recent corr)
    condcorr_msw        two-state Markov switching on (us_large, ust_int), filtered
    condcorr_msw_infl   same with inflation-dependent transition probabilities
    condcorr_stress     equity x HY credit / REIT correlations from recent corr, equity
                        vol ratio and the Baa-Aaa spread (stock-bond left at the base)
"""
from __future__ import annotations

from ..base import register
from .models import SB_BASE, SB_MACRO, SB_STATE, SBMarkov, SBRegression
from .stockbond import STRESS, STRESS_PAIRS

register("corr", "condcorr_sb_base")(lambda: SBRegression(SB_BASE))
register("corr", "condcorr_sb_state")(lambda: SBRegression(SB_STATE))
register("corr", "condcorr_sb_macro")(lambda: SBRegression(SB_MACRO))
register("corr", "condcorr_msw")(lambda: SBMarkov(tvtp=False))
register("corr", "condcorr_msw_infl")(lambda: SBMarkov(tvtp=True))
register("corr", "condcorr_stress")(lambda: SBRegression(STRESS, STRESS_PAIRS))
