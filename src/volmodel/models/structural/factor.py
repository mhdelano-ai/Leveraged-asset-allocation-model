"""Structural factor covariance model.

    Sigma_t = B_t F_t B_t' + diag(resid_t)

Factors (monthly, from the view; longest histories available):
    EQ  US large-cap log excess return (1871-)
    RT  rate shock: Treasury yield change implied by ust_int (ust_long before
        1953), divided by y^beta (the model's yield-vol functional form)
    CS  credit-spread shock (DTS): the part of the investment-grade credit
        return not explained by RT, divided by duration x spread level (1919-)
    CM  commodity log excess return (1926-)

Exposures. Each asset regresses on a fixed, economically chosen subset of
factors; the bond regressors are scaled by OBSERVABLE exposures known at the
start of each month (duration x y^beta for rates, duration x spread for the
credit spread), so the estimated coefficient c is a stable number and today's
loading is c x today's exposure. Coefficients come from weighted least squares
on the asset's whole history (weights: the asset's own structural scale, so
the residual is homoskedastic in normalized units).

    us_large            EQ (it is the equity factor)
    us_small, intl_dev,
    em, reits           EQ, RT
    ust_int, ust_long   RT x D y^beta
    tips                RT x D_real y_int^beta
    ig_credit           RT x D_c y_int^beta, CS x D_c s
    hy_credit           RT x D_c y_int^beta, CS x D_c s, EQ
    gold                RT, CM
    commodities         CM

Factor covariance: variances by the ref_blend rule (geometric mean of the
full-history and last-60-month mean squares); correlations as
ref_corr_last120m (last 120 months shrunk halfway to the full history),
repaired to positive definite. Residual variances: same blend of the
normalized residuals, rescaled by today's structural scale, floored at 2% of
the asset's systematic variance (no asset is exactly spanned by the factors
for the next year). The matrix is positive definite by construction.

Assets whose regressors are missing at the origin (no predictors, as in the
synthetic panel) are modelled on whatever factors exist; if a factor history
is missing entirely, the asset falls back to EQ only, and the vols of the
non-modelled part default to ref_blend through the residual.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import CovForecaster
from ..ref import nearest_corr, pairwise_corr
from . import core
from .vol import CREDIT, StructuralState, TIPS_PRE_REAL_YIELD

RESID_FLOOR = 0.02
CORR_RECENT, CORR_WEIGHT = 120, 0.5     # as ref_corr_last120m
MIN_FIT = 24                            # months needed to estimate a loading

SPEC = {
    "us_large": ["EQ"],
    "us_small": ["EQ", "RT"], "intl_dev": ["EQ", "RT"], "em": ["EQ", "RT"], "reits": ["EQ", "RT"],
    "ust_int": ["RT"], "ust_long": ["RT"], "tips": ["RT"],
    "ig_credit": ["RT", "CS"], "hy_credit": ["RT", "CS", "EQ"],
    "gold": ["RT", "CM"], "commodities": ["CM"],
}
FACTORS = ["EQ", "RT", "CS", "CM"]
# v2 ("global"): a non-US equity factor (developed ex-US, from 1975) for the
# two non-US equity assets, which share a regional component the US factor
# cannot carry; everything else as SPEC.
SPEC_GLOBAL = dict(SPEC, intl_dev=["XU"], em=["EQ", "XU", "RT"])
FACTORS_GLOBAL = FACTORS + ["XU"]


def factor_cov(F: pd.DataFrame) -> np.ndarray:
    v = np.array([core.blend_var(F[k]) for k in F.columns])
    long = pairwise_corr(F)
    recent = pairwise_corr(F.iloc[-CORR_RECENT:])
    n = F.iloc[-CORR_RECENT:].notna().to_numpy(float)
    recent = np.where((n.T @ n) < 24, long, recent)
    c = nearest_corr(CORR_WEIGHT * recent + (1 - CORR_WEIGHT) * long)
    s = np.sqrt(v)
    return c * np.outer(s, s)


class StructuralFactorCov(CovForecaster):
    def __init__(self, beta=1.0, etf_adjust=True, global_eq=False, dimson=False):
        self.beta, self.etf_adjust = beta, etf_adjust
        self.spec = SPEC_GLOBAL if global_eq else SPEC
        self.factors = FACTORS_GLOBAL if global_eq else FACTORS
        self.dimson = dimson

    # ---------------------------------------------------------------- pieces
    def _setup(self, view):
        st = StructuralState(view, beta=self.beta, how="blend", etf_adjust=self.etf_adjust)
        X, idx = st.X, st.X.index
        nan = pd.Series(np.nan, index=idx)
        y_int = st.y["int"]
        sc_int = core.cev_scale(y_int, st.beta)
        g = st.g()
        F = pd.DataFrame({"EQ": X["us_large"], "RT": g / sc_int.shift(1),
                          "CS": nan.copy(), "CM": X["commodities"], "XU": X["intl_dev"]}, index=idx)
        # exposures S[a][k] known at month-end (applied with a one-month lag in fits)
        S = {a: {} for a in SPEC}
        R = {a: pd.Series(1.0, index=idx) for a in SPEC}   # residual scales
        for a, (k, _) in {"ust_int": ("int", None), "ust_long": ("long", None)}.items():
            sk = st.D[k] * core.cev_scale(st.y[k], st.beta)
            S[a]["RT"] = sk
            R[a] = sk
        yr = st.pm["tips_real_yield_10y"].ffill().fillna(TIPS_PRE_REAL_YIELD)
        Dr = pd.Series(core.par_duration(yr.to_numpy(), core.MAT_TIPS), index=idx)
        S["tips"]["RT"] = Dr * sc_int
        R["tips"] = Dr
        s = core.spread_level(st.pm)
        ycr = y_int + s
        for a, mats in CREDIT.items():
            Dc = pd.Series(core.par_duration(ycr.to_numpy(), mats), index=idx).where(ycr.notna())
            S[a]["RT"] = Dc * sc_int
            S[a]["CS"] = Dc * s
            R[a] = Dc * s
        # credit-spread factor: IG return net of its rate part, per unit of D x s
        try:
            p = st.credit_parts("ig_credit")
            Dc = S["ig_credit"]["CS"] / s
            uc = X["ig_credit"] / Dc.shift(1)
            F["CS"] = (uc - p["b"] * g) / s.shift(1)
        except (KeyError, ZeroDivisionError, ValueError):
            pass
        return st, X, F, S, R

    def forecast_cov(self, view, assets):
        with np.errstate(all="ignore"):
            st, X, F, S, R = self._setup(view)
        FACT = self.factors
        k = len(FACT)
        Fv = F[FACT].copy()
        avail = [f for f in FACT if Fv[f].notna().sum() >= MIN_FIT]
        B = np.zeros((len(assets), k))
        resid = np.zeros(len(assets))
        for i, a in enumerate(assets):
            regs = [f for f in self.spec.get(a, ["EQ"]) if f in avail] or (["EQ"] if "EQ" in avail else [])
            y = X[a]
            r_scale = R[a]
            r_scale = r_scale if r_scale.notna().iloc[-1] and np.isfinite(r_scale.iloc[-1]) else pd.Series(1.0, index=X.index)
            cols, today = {}, {}
            for f in regs:
                sc = S[a].get(f)
                if sc is None or not np.isfinite(sc.iloc[-1]):
                    sc = pd.Series(1.0, index=X.index)
                cols[f] = sc.shift(1) * Fv[f]
                today[f] = float(sc.iloc[-1])
                if self.dimson:     # one lag; the loading is the sum (Dimson 1979)
                    cols[f + "_l1"] = sc.shift(2) * Fv[f].shift(1)
            w = r_scale.shift(1)
            D = pd.DataFrame(cols)
            ok = y.notna() & w.notna() & (D.notna().all(axis=1) if regs else True)
            ok &= w > 0
            yn = (y / w)[ok]
            if regs and ok.sum() >= MIN_FIT:
                Dn = D[ok].div(w[ok], axis=0)
                c, *_ = np.linalg.lstsq(Dn.to_numpy(), yn.to_numpy(), rcond=None)
                e = pd.Series(yn.to_numpy() - Dn.to_numpy() @ c, index=yn.index)
                coef = dict(zip(D.columns, c))
                for f in regs:
                    B[i, FACT.index(f)] = (coef[f] + coef.get(f + "_l1", 0.0)) * today[f]
            else:
                e = (y / w).dropna()
            resid[i] = core.blend_var(e) * float(r_scale.iloc[-1]) ** 2
        Fc = np.zeros((k, k))
        if avail:
            ia = [FACT.index(f) for f in avail]
            Fc[np.ix_(ia, ia)] = factor_cov(Fv[avail])
        sysv = np.einsum("ij,jk,ik->i", B, Fc, B)
        resid = np.where(np.isfinite(resid), resid, 0.0)
        resid = np.maximum(resid, RESID_FLOOR * sysv)
        # an asset with neither a factor fit nor residual history: ref variance
        x = X[assets]
        fallback = 12.0 * (x ** 2).mean().to_numpy()
        bad = (sysv + resid) <= 0
        resid[bad] = fallback[bad]
        cov = B @ Fc @ B.T + np.diag(resid)
        cov = (cov + cov.T) / 2.0
        return pd.DataFrame(cov, index=assets, columns=assets)
