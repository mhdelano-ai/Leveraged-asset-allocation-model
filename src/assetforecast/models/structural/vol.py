"""Structural volatility models: slow observable exposures x stable factor vols.

For every asset in which the family makes a structural claim:

Treasuries (ust_int, ust_long)
    sigma^2 = D_t^2 * y_t^(2 beta) * V,   V = blend_var(r_m / (D_{m-1} y_{m-1}^beta))
    D = modified duration of the asset's par ladder at today's yield, times the
    ETF/ladder effective-duration ratio in the ETF era; r = monthly log excess
    return. beta = 0: basis-point vol is the stable quantity ("normal");
    beta = 1: relative yield vol is ("lognormal"); in between: CEV.

TIPS
    sigma^2 = D_real,t^2 * kappa^2 * V_nom,t
    V_nom,t is the nominal 10y yield-variance forecast of the same functional
    form (from ust_int), kappa^2 = (n kappa_hat^2 + 24) / (n + 24) shrinks the
    ratio of TIPS-implied real-yield variance to nominal yield variance over the
    n common months toward 1 (prior worth 24 months).

Credit (ig_credit, hy_credit), duration-times-spread
    u = r / D_c (D_c: par bond at Treasury yield + Baa spread), g = Treasury
    yield change implied by ust_int (ust_long before 1953), b = OLS loading of u
    on g over all history, z = (u - b g) / s_{m-1} (DTS shock), and
    sigma^2 = D_c,t^2 * [ b^2 V_g,t + s_t^2 blend_var(z) ].

Everything else (equities, REITs, gold, commodities), and any structural asset
whose inputs are missing at the origin, gets ``ref_blend``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import VolForecaster
from ..ref import window_var
from . import core

TIPS_PRIOR_MONTHS = 24     # prior weight on kappa = 1 (real-yield vol = nominal-yield vol)
TIPS_PRE_REAL_YIELD = 0.03 # real yield assumed for TIPS durations before the GSW real curve (1997-98)
TREASURY = {"ust_int": ("int", core.MAT_INT), "ust_long": ("long", core.MAT_LONG)}
CREDIT = {"ig_credit": core.MAT_IG, "hy_credit": core.MAT_HY}
STRUCTURAL_ASSETS = ("ust_int", "ust_long", "tips", "ig_credit", "hy_credit")


def ref_blend_var(view, assets) -> pd.Series:
    x = view.excess_monthly[assets]
    return np.exp(0.5 * (np.log(window_var(x, None)) + np.log(window_var(x, 60))))


class StructuralState:
    """All monthly structural series for one view (computed once per call)."""

    def __init__(self, view, beta="est", how="blend", etf_adjust=True):
        self.view = view
        self.how = how
        self.X = view.excess_monthly
        pm = core.monthly_predictors(view)
        self.pm = pm
        self.y = core.yield_levels(pm)
        self.D = core.treasury_durations(pm, etf_adjust=etf_adjust)
        X = self.X
        self.u = {k: X[a] / self.D[k].shift(1) for a, (k, _) in TREASURY.items() if a in X}
        if beta == "est":
            beta = core.estimate_beta(self.u, {k: self.y[k] for k in self.u}, view.asof_date
                                      if view.asof_date is not None else X.index[-1])
        self.beta = float(beta)

    # -- rates ---------------------------------------------------------------
    def scale(self, k: str) -> pd.Series:
        return core.cev_scale(self.y[k], self.beta)

    def rate_var(self, k: str) -> tuple[float, float]:
        """(forecast yield variance today, in decimal^2/yr; today's duration)."""
        n = self.u[k] / self.scale(k).shift(1)
        V = core.blend_var(n, how=self.how)
        sc = self.scale(k).iloc[-1]
        return V * sc ** 2, self.D[k].iloc[-1]

    def treasury(self, a: str) -> float:
        k = TREASURY[a][0]
        V, D = self.rate_var(k)
        return D ** 2 * V

    def g(self) -> pd.Series:
        """Treasury yield change implied by returns: ust_int, else ust_long."""
        gi = self.u.get("int", pd.Series(np.nan, index=self.X.index))
        gl = self.u.get("long", pd.Series(np.nan, index=self.X.index))
        return gi.combine_first(gl)

    # -- TIPS ----------------------------------------------------------------
    def tips(self) -> float:
        yr = self.pm["tips_real_yield_10y"].ffill().fillna(TIPS_PRE_REAL_YIELD)
        Dr = pd.Series(core.par_duration(yr.to_numpy(), core.MAT_TIPS), index=yr.index)
        ut = self.X["tips"] / Dr.shift(1)
        un = self.u["int"]
        ok = ut.notna() & un.notna()
        n = int(ok.sum())
        if n == 0:
            k2 = 1.0
        else:
            k2_hat = float((ut[ok] ** 2).mean() / (un[ok] ** 2).mean())
            k2 = (n * k2_hat + TIPS_PRIOR_MONTHS * 1.0) / (n + TIPS_PRIOR_MONTHS)
        V, _ = self.rate_var("int")
        return Dr.iloc[-1] ** 2 * k2 * V

    # -- credit --------------------------------------------------------------
    def credit_parts(self, a: str):
        s = core.spread_level(self.pm)
        ycr = self.y["int"] + s
        Dc = pd.Series(core.par_duration(ycr.to_numpy(), CREDIT[a]), index=ycr.index).where(ycr.notna())
        uc = self.X[a] / Dc.shift(1)
        g = self.g()
        s1 = s.shift(1)
        ok = uc.notna() & g.notna() & s1.notna()
        b = float((uc[ok] * g[ok]).sum() / (g[ok] ** 2).sum())
        z = ((uc - b * g) / s1).where(ok)
        # Treasury variance at the g point, same functional form
        ng = g / core.cev_scale(self.y["int"], self.beta).shift(1)
        Vg = core.blend_var(ng, how=self.how) * core.cev_scale(self.y["int"], self.beta).iloc[-1] ** 2
        Vz = core.blend_var(z, how=self.how)
        return dict(D=Dc.iloc[-1], b=b, Vg=Vg, s=s.iloc[-1], Vz=Vz)

    def credit(self, a: str) -> float:
        p = self.credit_parts(a)
        return p["D"] ** 2 * (p["b"] ** 2 * p["Vg"] + p["s"] ** 2 * p["Vz"])

    def variance(self, a: str) -> float:
        if a in TREASURY:
            return self.treasury(a)
        if a == "tips":
            return self.tips()
        if a in CREDIT:
            return self.credit(a)
        return np.nan


def _safe(fn, *args) -> float:
    try:
        with np.errstate(all="ignore"):
            v = float(fn(*args))
    except (KeyError, ValueError, ZeroDivisionError, IndexError):
        return np.nan
    return v if np.isfinite(v) and v > 0 else np.nan


class StructuralVol(VolForecaster):
    """Structural vol for Treasuries, TIPS and credit; ref_blend elsewhere.

    ``beta``: CEV elasticity of yield vol to the yield level (0 normal, 1
    lognormal, "est" estimated walk-forward). ``how``: window form of every
    variance estimate ("blend" as ref_blend, "full", "recent").
    ``assets_on``: the structural assets this instance treats (others: ref_blend).
    """

    def __init__(self, beta="est", how="blend", etf_adjust=True, assets_on=STRUCTURAL_ASSETS):
        self.beta, self.how, self.etf_adjust = beta, how, etf_adjust
        self.assets_on = tuple(assets_on)

    def forecast_vol(self, view, assets):
        var = ref_blend_var(view, assets).reindex(assets)
        todo = [a for a in assets if a in self.assets_on]
        if todo:
            st = StructuralState(view, beta=self.beta, how=self.how, etf_adjust=self.etf_adjust)
            for a in todo:
                v = _safe(st.variance, a)
                if np.isfinite(v):
                    var[a] = v
        return np.sqrt(var).reindex(assets)
