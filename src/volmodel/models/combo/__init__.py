"""combo: forecast combination and level calibration of STORED forecasts.

Members are other registered models' stored forecasts (read with
``combo.store`` from the results directory in use; nothing is re-run and
nothing missing is ever substituted: a missing component forecast raises
``MissingComponentError`` naming the component and the origin). The combos
therefore have to be run AFTER their components.

Volatility (variance space: QLIKE is minimized by the conditional mean of the
variance, and an arithmetic mean of variances is unbiased when its members
are; a mean of vols or log variances is biased low by Jensen whenever the
members disagree, and QLIKE punishes under-forecasts most)

    combo_vol_ew          equal weight over ALL registered vol models of other families
    combo_vol_hier        mean within family, then equal weight across the families
    combo_vol_hier_trim   family means, trimmed mean across families (drop highest and lowest)
    combo_vol_stack       family means, simplex weights minimizing past QLIKE (pooled over assets),
                          shrunk toward equal: w = lam w* + (1-lam)/g, lam = Y/(Y+20), Y training years
    combo_vol_hier_cal    combo_vol_hier x walk-forward level constant c
    combo_vol_stack_cal   combo_vol_stack x walk-forward level constant c
    combo_vol_cal_m_hl12  bench_vol_m_hl12 (12-month EWMA of squared monthly returns) x c
    combo_vol_cal_rm2006  garch_d_rm2006 (RiskMetrics 2006, published constants) x c

    c = 1 + lam (c_hat - 1), c_hat = mean over past cells of realized / forecast variance
    (the QLIKE-optimal multiplicative constant), pooled over assets, every origin weighted
    once. Weights and c are re-estimated each December from cells (s, a) with
    s + 12 months <= E, realized values computed from the view (``combo.train``).

Correlation (convex combinations of valid correlation matrices are valid)

    combo_corr_ew          equal weight over all registered corr models of other families and
                           the correlations of the registered cov models, except ref_corr_zero
    combo_corr_hier        family means, equal weight across families
    combo_corr_hier_shrink combo_corr_hier shrunk 20% toward its own asset-group block means
    combo_corr_fastslow    0.5 condcorr_msw (fast, Markov stock-bond) + 0.5 mcorr_shrink_block
                           (slow, well conditioned)

Covariance (``Composite`` of a vol combo and a corr combo; both computed from
the stored components, so they do not depend on the combos' own files)

    combo_cov_hier         combo_vol_hier x combo_corr_hier                 (pre-specified)
    combo_cov_cal          combo_vol_hier_cal x combo_corr_hier_shrink      (pre-specified)
    combo_cov_main         combo_vol_hier_trim x combo_corr_ew              (the recommendation; chosen
                           after the vol and corr evaluation, see the report)
    combo_cov_simple       combo_vol_cal_m_hl12 x bench_corr_d_hl36         (the sceptic's runner-up;
                           first registered with combo_corr_fastslow, changed after the evaluation)

Report: ``reports/models/combo.md``; evaluation: ``python -m volmodel.models.combo.analysis``.
"""
from __future__ import annotations

from ..base import Composite, register
from .corr import CorrCombo, block_average
from .store import MissingComponentError, family_of, load
from .train import estimation_date, training_data
from .vol import Estimate, VolCombo, aggregate, stack_weights

VOL_MODELS = {
    "combo_vol_ew": lambda: VolCombo(group="none", agg="mean"),
    "combo_vol_hier": lambda: VolCombo(group="family", agg="mean"),
    "combo_vol_hier_trim": lambda: VolCombo(group="family", agg="trim"),
    "combo_vol_stack": lambda: VolCombo(group="family", agg="stack"),
    "combo_vol_hier_cal": lambda: VolCombo(group="family", agg="mean", calibrate=True),
    "combo_vol_stack_cal": lambda: VolCombo(group="family", agg="stack", calibrate=True),
    "combo_vol_cal_m_hl12": lambda: VolCombo(["bench_vol_m_hl12"], group="none", calibrate=True),
    "combo_vol_cal_rm2006": lambda: VolCombo(["garch_d_rm2006"], group="none", calibrate=True),
}
FASTSLOW = {"condcorr_msw": 0.5, "mcorr_shrink_block": 0.5}
BLOCK_SHRINK = 0.2
CORR_MODELS = {
    "combo_corr_ew": lambda: CorrCombo(group="none"),
    "combo_corr_hier": lambda: CorrCombo(group="family"),
    "combo_corr_hier_shrink": lambda: CorrCombo(group="family", block_shrink=BLOCK_SHRINK),
    "combo_corr_fastslow": lambda: CorrCombo(list(FASTSLOW), group="none", weights=FASTSLOW),
}
COV_PAIRS = {
    "combo_cov_hier": ("combo_vol_hier", "combo_corr_hier"),
    "combo_cov_cal": ("combo_vol_hier_cal", "combo_corr_hier_shrink"),
    "combo_cov_main": ("combo_vol_hier_trim", "combo_corr_ew"),
    "combo_cov_simple": ("combo_vol_cal_m_hl12", "bench_corr_d_hl36"),
}
# a single stored corr model read through the same store (so a missing one raises the same way)
_CORR_FOR_COV = CORR_MODELS | {"bench_corr_d_hl36": lambda: CorrCombo(["bench_corr_d_hl36"], group="none")}


def _composite(vol_name, corr_name):
    def make():
        v, c = VOL_MODELS[vol_name](), _CORR_FOR_COV[corr_name]()
        v.name, c.name = vol_name, corr_name
        return Composite(v, c)
    return make


for _n, _f in VOL_MODELS.items():
    register("vol", _n)(_f)
for _n, _f in CORR_MODELS.items():
    register("corr", _n)(_f)
for _n, (_v, _c) in COV_PAIRS.items():
    register("cov", _n)(_composite(_v, _c))
