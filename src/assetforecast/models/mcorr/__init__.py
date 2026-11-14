"""mcorr: unconditional-structure estimators of the correlation matrix.

The problem these models attack is ESTIMATION ERROR. A 12-asset correlation
matrix estimated from monthly returns rests on a few hundred independent
months at best and on 36 at worst (an asset is forecast once it has 36
months of history), and the histories are very unequal (us_large 1871,
tips 1997). Each model below is a different classical remedy; all share
one data convention so that comparisons between them isolate the remedy:

* input: monthly (calendar-month, ~21-trading-day) log excess returns from
  ``view.excess_monthly`` - the target's own horizon, free of the 1-day
  asynchronous-close and stale-price biases, and available in every era;
* weighting: exponential, half-life 10 years (EW_HALFLIFE = 120 months);
* second moments uncentered (as the target);
* effective sample sizes in independent months (Kish), never days.

Registered models (kind ``corr``):

    mcorr_pairwise      each pair on its own common sample, then PD repair
                        (the unequal-history CONTROL)
    mcorr_stambaugh     ML under monotone missingness: long histories on their
                        full sample, each shorter asset attached by regression
                        on all longer ones (PD by construction)
    mcorr_block         block equicorrelation (groups equity/rates/credit/real)
                        of mcorr_stambaugh: one number per block pair
    mcorr_shrink_block  mcorr_stambaugh shrunk toward mcorr_block, pair-specific
                        empirical-Bayes (Ledoit-Wolf type) intensity
    mcorr_nls           Ledoit-Wolf (2020) analytical nonlinear shrinkage of the
                        eigenvalues of mcorr_stambaugh
    mcorr_fa3           3-factor model (iterated principal-axis factoring) of mcorr_stambaugh plus
                        uniquenesses
    mcorr_factor2       observed factors: equity (us_large) and rates (ust_long),
                        loadings of every other asset on its own sample
    mcorr_dcc           corrected DCC around a slowly moving mcorr_shrink_block
                        target, composite likelihood over all pairs; forecast =
                        average expected correlation over the 12 months

See reports/models/mcorr.md for formulas, choices and results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import CorrForecaster, register
from .core import (EW_HALFLIFE, block_average, ew_weights, groups_of, kish, nonlinear_shrink,
                   observed_factor_cov, pair_neff, pairwise_corr, pca_factor_corr, rmt_clip,
                   shrink_to_target, stambaugh_corr, cov_to_corr, ensure_pd, nearest_corr)
from . import dcc as _dcc

FACTORS = ("us_large", "ust_long")     # equity and rates factors: the two longest monthly histories
MIN_FACTOR_MONTHS = 36
PCA_FACTORS = 3


def _monthly(view, cols) -> np.ndarray:
    X = view.excess_monthly[list(cols)].to_numpy(float)
    keep = np.isfinite(X).any(axis=1)
    if not keep.any():
        return X[:0]
    return X[np.argmax(keep):]


def harmonic_pair_n(n: np.ndarray) -> float:
    """Harmonic mean of the off-diagonal pair effective sizes: the n whose
    1/n equals the average sampling variance factor across pairs."""
    iu = np.triu_indices(len(n), 1)
    v = np.maximum(n[iu], 3.0)
    return float(len(v) / np.sum(1.0 / v)) if len(v) else 3.0


class McorrStatic(CorrForecaster):
    """Static estimators; ``method`` picks the remedy (see module docstring)."""

    METHODS = ("pairwise", "stambaugh", "block", "shrink_block", "nls", "fa3", "factor2",
               "rmt", "shrink_block_full", "lw_common")

    def __init__(self, method: str, halflife: float | None = EW_HALFLIFE):
        if method not in self.METHODS:
            raise ValueError(method)
        self.method, self.halflife = method, halflife
        self.last_info: dict = {}

    def estimate(self, X: np.ndarray, assets: list[str], view=None) -> np.ndarray:
        w = ew_weights(len(X), self.halflife)
        m = self.method
        if m == "pairwise":
            return pairwise_corr(X, w)
        if m == "factor2":
            return self._factor2(X, w, assets, view)
        R = stambaugh_corr(X, w)
        if m == "stambaugh":
            return R
        if m == "block":
            return block_average(R, groups_of(assets))
        if m in ("shrink_block", "shrink_block_full"):
            out, info = shrink_to_target(R, block_average(R, groups_of(assets)), pair_neff(X, w))
            self.last_info = info
            return out
        if m == "lw_common":                      # dropped variant: one intensity for all pairs
            Tg = block_average(R, groups_of(assets))
            n = pair_neff(X, w)
            iu = np.triu_indices(len(R), 1)
            v = (1 - R[iu] ** 2) ** 2 / np.maximum(n[iu], 3.0)
            d2 = np.sum((R[iu] - Tg[iu]) ** 2)
            delta = float(np.clip(v.sum() / d2, 0, 1)) if d2 > 0 else 1.0
            self.last_info = {"delta": delta}
            return ensure_pd(delta * Tg + (1 - delta) * R)
        n = harmonic_pair_n(pair_neff(X, w))
        self.last_info = {"n": n}
        if m == "nls":
            return nonlinear_shrink(R, n)
        if m == "rmt":
            return rmt_clip(R, n)
        if m == "fa3":
            return pca_factor_corr(R, PCA_FACTORS)
        raise ValueError(m)

    def _factor2(self, X, w, assets, view):
        if view is None:
            raise ValueError("factor2 needs the view")
        n_obs = view.excess_monthly.notna().sum()
        facs = [f for f in FACTORS if f in view.excess_monthly.columns and n_obs[f] >= MIN_FACTOR_MONTHS]
        if not facs:                               # no factor history: fall back to the ML matrix
            return stambaugh_corr(X, w)
        cols = list(assets) + [f for f in facs if f not in assets]
        Z = _monthly(view, cols)
        wz = ew_weights(len(Z), self.halflife)
        Xa = Z[:, : len(assets)]
        Fx = Z[:, [cols.index(f) for f in facs]]
        is_f = np.array([a in facs for a in assets])
        fcol = np.array([facs.index(a) if a in facs else -1 for a in assets])
        return ensure_pd(cov_to_corr(observed_factor_cov(Xa, Fx, wz, is_f, fcol)))

    def forecast_corr(self, view, assets):
        assets = list(assets)
        X = _monthly(view, assets)
        C = self.estimate(X, assets, view)
        return pd.DataFrame(C, index=assets, columns=assets)


class McorrDCC(CorrForecaster):
    """See ``dcc.py``."""

    def __init__(self, horizon: int = _dcc.HORIZON):
        self.horizon = horizon
        self._params: dict = {}
        self._targets: dict = {}
        self.last_info: dict = {}

    def params(self, X: np.ndarray, years: np.ndarray, months: np.ndarray,
               assets: list[str]) -> tuple[float, float]:
        """(a, b) estimated on data through the latest December at or before
        the last month of the view (re-estimated once a year)."""
        dec = np.flatnonzero(months == 12)
        if dec.size == 0:
            return 0.0, 0.0
        r = int(dec[-1])
        key = (tuple(assets), r, int(years[r]))
        if key not in self._params:
            E = _dcc.standardize(X[: r + 1])
            prep = _dcc._Prepared(E, years[: r + 1], assets, self._targets)
            self._params[key] = _dcc.fit_dcc(prep)
        return self._params[key]

    def forecast_corr(self, view, assets):
        assets = list(assets)
        idx = view.excess_monthly.index
        X = view.excess_monthly[assets].to_numpy(float)
        keep = np.isfinite(X).any(axis=1)
        first = int(np.argmax(keep))
        X = X[first:]
        dates = idx[first:]
        years = np.asarray(dates.year)
        a, b = self.params(X, years, np.asarray(dates.month), assets)
        E = _dcc.standardize(X)
        S_T = _dcc.target_corr(E, assets)
        N = len(assets)
        kappa = float(np.mean((a + b) ** np.arange(self.horizon))) if a + b > 0 else 0.0
        R1 = S_T.copy()          # next month's conditional correlation (target where a pair is not active)
        if a > 0:
            prep = _dcc._Prepared(E, years, assets, self._targets)
            if prep.ok:
                _, qd, qo, _ = prep.run(a, b, loglik=False)
                i, j = prep.iu
                pact = prep.pact[-1]
                r_next = qo / np.sqrt(qd[i] * qd[j])
                R1[i[pact], j[pact]] = r_next[pact]
                R1[j[pact], i[pact]] = r_next[pact]
                R1 = nearest_corr(R1)   # pairs started at different dates: repair if needed
        # E[R_{t+h}] ~ S + (a+b)^(h-1) (R_{t+1} - S); averaged over h = 1..12 this is a
        # convex combination of two PD matrices, hence PD.
        C = ensure_pd((1.0 - kappa) * S_T + kappa * R1)
        iu = np.triu_indices(N, 1)
        self.last_info = {"a": a, "b": b, "kappa": kappa,
                          "mean_abs_dev": float(np.mean(np.abs(C[iu] - S_T[iu]))) if N > 1 else 0.0}
        return pd.DataFrame(C, index=assets, columns=assets)


for _m in ("pairwise", "stambaugh", "block", "shrink_block", "nls", "fa3", "factor2"):
    register("corr", f"mcorr_{_m}")((lambda m=_m: (lambda: McorrStatic(m)))())
register("corr", "mcorr_dcc")(McorrDCC)
