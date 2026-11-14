"""Covariance models. Registered names are listed in :data:`MODELS`; the frozen
holdout list is :data:`FROZEN`.

Baselines
---------
``sample``      THE benchmark. Expanding pairwise-complete sample covariance of
                realized overlapping h-month sums (rows s <= t-h): variances
                per asset over its whole history, correlations over each
                pair's common history, projected to the nearest correlation
                matrix when the pairwise matrix is not PSD. Its diagonal (after
                the engine's mean-uncertainty factor) equals ``hist_mean``'s
                sd^2 exactly.
``sample_iid``  h x the monthly sample covariance (i.i.d. scaling).
``ewma_cov``    RiskMetrics: h x EWMA covariance of monthly returns,
                lambda = 0.97 (the RiskMetrics monthly value), no mean reversion.
``lw``          ``sample`` with its correlation shrunk toward the constant-
                correlation target, Ledoit-Wolf (2004) intensity estimated on
                the balanced window of monthly returns of the eligible assets.

Volatility x correlation (``VolCorr``)
--------------------------------------
Sigma_h = D R D. D_ii^2 = vr_i * (h * LR_i + c_v * (s2_i - LR_i)): the current
monthly variance s2_i mean-reverts to the long-run LR_i at rate phi_v per month
(c_v = sum_k phi_v^k), and vr_i = var(h-sum) / (h var(month)) carries serial
correlation into the h-month scale. With ``vol='ewma'`` this is exactly the
``vol_ewma`` marginal baseline. ``vol='daily'`` takes s2 from daily lead-lag
realized variance (21- and 63-day), level-matched to the monthly series.

R is a long-run correlation level plus the *anomaly* of a dynamic estimate
relative to its own long-run level, weighted by c_r / h (c_r = sum_k phi_r^k):

    R_h = R^LR + (c_r / h) * (R_dyn - R_dyn^LR)       (space='corr', default)
    z(R_h) = z(R^LR) + (c_r / h) * (z(R_dyn) - z(R_dyn^LR))   (space='z')

then projected to the nearest correlation matrix. The level comes from monthly
returns (base='1') or from overlapping h-month sums (base='h'); the dynamics
from recent monthly or daily data, so level differences between scales
(non-synchronous closes, stale NAVs) cancel. On dev data base='h' and
space='z' both lost clearly (h-month correlations from overlapping sums are
noisy; Fisher z blows up anomalies of pairs near |rho| = 1).
corr = 'sample' (no anomaly), 'ewma' (monthly EWMA, lambda_r), 'roll'
(rolling window), 'dcc' (DCC(1,1), composite likelihood, refit yearly; its own
mean reversion replaces c_r/h), 'daily' (126-day daily lead-lag correlation
vs its full-history daily level; monthly EWMA anomaly for pairs without
matching daily data). ``shrink`` shrinks the final correlation toward
constant correlation ('lw' = Ledoit-Wolf intensity, or a number).

``ewma_mr`` is the full-matrix analogue: Sigma_h = (h - c) S_h / h + c * EWMA_1
with S_h the benchmark (a convex combination of PSD matrices).

Suffixes of the registered names: ``_m`` correlation level from monthly
returns (``base='1'``) instead of overlapping h-month sums (``base='h'``,
round-1 names, which also use Fisher-z anomalies: ``space='z'``); ``_lw``
Ledoit-Wolf shrinkage of the final correlation toward constant correlation,
``_lwb`` LW applied to the static level only, before the dynamic anomaly is
added (so a sign switch is not pulled back to the cross-pair average),
``_lwi`` LW toward identity,
``_eig`` eigenvalue floor of 0.05 on the correlation matrix; ``_rw``
phi_r = 1 (the correlation anomaly does not mean-revert); ``_novr`` i.i.d.
variance scaling (no variance ratio). ``vc_daily_macro_m_lw``
(:class:`MacroCorr`) forecasts daily correlation anomalies with a per-pair
ridge on macro features.

All smoothing constants are fixed a priori (RiskMetrics-style monthly values);
the few variants tried are logged in ``results/dev/trials_cov.csv``.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from . import estimators as est
from . import linalg as la
from .base import CovForecaster, CovTrainingSet, register

BENCHMARK = "sample"


class SampleCov(CovForecaster):
    """Expanding sample covariance of realized h-month sums (the benchmark)."""

    def predict(self, train: CovTrainingSet) -> np.ndarray:
        var, R = est.h_sample(train)
        return la.corr_to_cov(R, np.sqrt(var))


class SampleIID(CovForecaster):
    """h x monthly sample covariance."""

    def predict(self, train):
        _, var, R = est.m_sample(train)
        return train.horizon * la.corr_to_cov(R, np.sqrt(var))


class EWMACov(CovForecaster):
    """RiskMetrics EWMA covariance of monthly returns times h."""

    def __init__(self, lam: float = 0.97):
        self.lam = lam

    def predict(self, train):
        S = est.ewma_cov(train, self.lam)
        R, sd = la.cov_to_corr(S)
        return train.horizon * la.corr_to_cov(la.nearest_corr(R), sd)


class EWMAMeanReverting(CovForecaster):
    """(h - c) S_h / h + c * EWMA monthly covariance, c = sum_k phi^k."""

    def __init__(self, lam: float = 0.97, phi: float = 0.9):
        self.lam, self.phi = lam, phi

    def predict(self, train):
        h = train.horizon
        var, R = est.h_sample(train)
        S_h = la.corr_to_cov(R, np.sqrt(var))
        E = est.ewma_cov(train, self.lam)
        Re, sd = la.cov_to_corr(E)
        E = la.corr_to_cov(la.nearest_corr(Re), sd)
        c = est.term_weight(self.phi, h)
        return (h - c) / h * S_h + c * E


class LedoitWolf(CovForecaster):
    """Benchmark variances with LW constant-correlation shrinkage of the correlation."""

    def predict(self, train):
        var, R = est.h_sample(train)
        bal = train.X.dropna()
        delta = la.lw_constant_corr_intensity(bal.to_numpy(float)) if len(bal) >= 24 else 0.5
        return la.corr_to_cov(la.shrink_corr(R, delta), np.sqrt(var))


class VolCorr(CovForecaster):
    """Volatility x correlation model; see the module docstring."""

    def __init__(self, vol: str = "ewma", corr: str = "ewma", lam_v: float = 0.94,
                 phi_v: float = 0.9, lam_r: float = 0.97, phi_r: float = 0.95,
                 window: int = 60, daily_window: int = 126, shrink=None,
                 base: str = "h", space: str = "corr", shrink_target: str = "constant",
                 vr: bool = True):
        self.vol, self.corr = vol, corr
        self.shrink_target, self.vr = shrink_target, vr
        self.base, self.space = base, space
        self.lam_v, self.phi_v = lam_v, phi_v
        self.lam_r, self.phi_r = lam_r, phi_r
        self.window, self.daily_window = window, daily_window
        self.shrink = shrink
        self.refit_every = 12 if corr == "dcc" else 1
        self.uses_daily = "daily" in (vol, corr)
        self._ab = None

    # ---- parameters (DCC only) ----
    def fit(self, train):
        if self.corr == "dcc":
            self._ab = est.fit_dcc(est.dcc_residuals(train))
        return self

    # ---- volatility ----
    def h_var(self, train) -> np.ndarray:
        h = train.horizon
        _, lr, _ = est.m_sample(train)
        vr = est.variance_ratio(train) if self.vr else np.ones(len(lr))
        if self.vol == "sample":
            return est.h_sample(train)[0]
        s2 = est.ewma_var(train, self.lam_v)
        if self.vol == "daily":
            d = est.daily_var_now(train)
            s2 = np.where(np.isfinite(d), d, s2)
        elif self.vol != "ewma":
            raise ValueError(self.vol)
        c = est.term_weight(self.phi_v, h)
        hv = vr * (h * lr + c * (s2 - lr))
        return np.maximum(hv, 0.05 * h * lr)

    # ---- correlation ----
    def h_corr(self, train) -> np.ndarray:
        h = train.horizon
        _, R_1 = est.m_sample(train)[1:]
        R_h = est.h_sample(train)[1] if self.base == "h" else R_1
        if self.shrink == "lwbase":
            # condition the static level only; the dynamic anomaly is added unshrunk
            R_h = la.shrink_corr(R_h, self._lw_delta(train), self.shrink_target)
        w = est.term_weight(self.phi_r, h) / h
        sp = self.space
        if self.corr == "sample":
            R = R_h
        elif self.corr == "ewma":
            R = la.blend_corr(R_h, est.ewma_corr(train, self.lam_r), R_1, w, sp)
        elif self.corr == "roll":
            R = la.blend_corr(R_h, est.rolling_corr(train, self.window), R_1, w, sp)
        elif self.corr == "dcc":
            if self._ab is None:
                self.fit(train)
            Rf, Qbar = est.dcc_forecast_corr(est.dcc_residuals(train), *self._ab, h)
            R = la.blend_corr(R_h, Rf, Qbar, 1.0, sp)
        elif self.corr == "daily":
            Rr, Rl = est.daily_corr(train, self.daily_window)
            an = la.corr_anomaly(Rr, Rl, sp)
            an_m = la.corr_anomaly(est.ewma_corr(train, self.lam_r), R_1, sp)
            R = la.apply_anomaly(R_h, np.where(np.isfinite(an), an, an_m), w, sp)
        else:
            raise ValueError(self.corr)
        if isinstance(self.shrink, str) and self.shrink.startswith("eig"):
            return la.eig_floor_corr(R, float(self.shrink[3:]))
        if self.shrink is not None and self.shrink != "lwbase":
            delta = self._lw_delta(train) if self.shrink == "lw" else float(self.shrink)
            R = la.shrink_corr(R, delta, self.shrink_target)
        return R

    @staticmethod
    def _lw_delta(train) -> float:
        bal = train.X.dropna()
        return la.lw_constant_corr_intensity(bal.to_numpy(float)) if len(bal) >= 24 else 0.5

    def predict(self, train):
        return la.corr_to_cov(self.h_corr(train), np.sqrt(self.h_var(train)))


class MacroCorr(VolCorr):
    """VolCorr with a feature-based forecast of the daily correlation anomaly.

    For each pair with matching daily data, a ridge regression (refit every 12
    months, rows s <= t - h only) predicts the Fisher-z anomaly of the realized
    daily correlation over the next h months relative to its long-run level,
    from the current 6-month anomaly and the global macro features
    ``MACRO_FEATURES`` (inflation and the real T-bill rate: the usual
    explanation of the sign of the stock-bond correlation). Features are
    standardized on the training rows; alpha = ``alpha`` x rows (fixed). The
    predicted future daily correlation's deviation from its long-run level is
    added (weight 1) to the monthly long-run correlation; pairs without daily
    data use the monthly EWMA anomaly as in ``VolCorr``.
    """
    MACRO_FEATURES = ("cpi_yoy", "g_real_tbill")
    uses_features = True

    def __init__(self, alpha: float = 0.1, recent_months: int = 6, **kw):
        kw.setdefault("vol", "daily")
        kw.setdefault("base", "1")
        kw.setdefault("shrink", "lw")
        super().__init__(corr="macro", **kw)
        self.alpha, self.recent_months = alpha, recent_months
        self.uses_daily = True
        self.refit_every = 12
        self._coef = {}

    def _macro(self, train):
        F = train.features
        cols = list(self.MACRO_FEATURES)
        if F is None or not len(F):
            return pd.DataFrame(columns=cols)
        g = F[cols].groupby(level="date").first()
        return g

    def _rows(self, train):
        panel = est.daily_corr_panel(train, self.recent_months)
        if panel is None:
            return None
        months, rec, lr, fut, idx = panel
        macro = self._macro(train).reindex(months)
        return months, la.fisher_z(rec) - la.fisher_z(lr), la.fisher_z(fut) - la.fisher_z(lr), lr, macro, idx

    def fit(self, train):
        from sklearn.linear_model import Ridge
        self._coef = {}
        r = self._rows(train)
        if r is None:
            return self
        months, an, tgt, lr, macro, idx = r
        k = len(idx)
        Mx = macro.to_numpy(float)
        for i in range(k):
            for j in range(i + 1, k):
                X = np.column_stack([an[:, i, j], Mx])
                y = tgt[:, i, j]
                ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
                if ok.sum() < 60:
                    continue
                mu, sd = X[ok].mean(0), X[ok].std(0) + 1e-12
                m = Ridge(alpha=self.alpha * ok.sum()).fit((X[ok] - mu) / sd, y[ok])
                self._coef[(i, j)] = (mu, sd, m.coef_.copy(), float(m.intercept_))
        return self

    def h_corr(self, train):
        if self.corr != "macro":
            return super().h_corr(train)
        h = train.horizon
        _, R_1 = est.m_sample(train)[1:]
        n = len(train.assets)
        w_m = est.term_weight(self.phi_r, h) / h
        an = la.corr_anomaly(est.ewma_corr(train, self.lam_r), R_1, "corr")
        W = np.full((n, n), w_m)
        r = self._rows(train)
        if r is not None:
            months, an_d, _, lr, macro, idx = r
            x_m = macro.to_numpy(float)[-1] if len(macro) else np.full(len(self.MACRO_FEATURES), np.nan)
            for (i, j), (mu, sd, b, c) in self._coef.items():
                x = np.r_[an_d[-1, i, j], x_m]
                if not np.isfinite(x).all() or not np.isfinite(lr[-1, i, j]):
                    continue
                zf = la.fisher_z(lr[-1, i, j]) + c + ((x - mu) / sd) @ b
                d = float(np.tanh(zf) - lr[-1, i, j])
                a, bb = idx[i], idx[j]
                an[a, bb] = an[bb, a] = d
                W[a, bb] = W[bb, a] = 1.0
        R = la.apply_anomaly(R_1, an, W, "corr")
        bal = train.X.dropna()
        delta = la.lw_constant_corr_intensity(bal.to_numpy(float)) if len(bal) >= 24 else 0.5
        return la.shrink_corr(R, delta, self.shrink_target)


# --------------------------------------------------------------------------
# Registry. Names are fixed; add variants here, never edit a scored one.
# --------------------------------------------------------------------------
_VC = partial(VolCorr, space="z")   # round-1 names: Fisher-z anomalies, h-month base
MODELS = {
    "sample": SampleCov,
    "sample_iid": SampleIID,
    "ewma_cov": EWMACov,
    "ewma_mr": EWMAMeanReverting,
    "lw": LedoitWolf,
    # round 1: vol x corr, correlation level from h-month sums, Fisher-z anomalies
    "vc_ewma_sample": partial(_VC, vol="ewma", corr="sample"),
    "vc_ewma_ewma": partial(_VC, vol="ewma", corr="ewma"),
    "vc_ewma_roll": partial(_VC, vol="ewma", corr="roll"),
    "vc_ewma_dcc": partial(_VC, vol="ewma", corr="dcc"),
    "vc_ewma_daily": partial(_VC, vol="ewma", corr="daily"),
    "vc_daily_sample": partial(_VC, vol="daily", corr="sample"),
    "vc_daily_ewma": partial(_VC, vol="daily", corr="ewma"),
    "vc_daily_daily": partial(_VC, vol="daily", corr="daily"),
    "vc_ewma_ewma_lw": partial(_VC, vol="ewma", corr="ewma", shrink="lw"),
    "vc_sample_ewma": partial(_VC, vol="sample", corr="ewma"),
    # round 2: correlation level from monthly returns ("_m"), anomalies in
    # correlation units, optional Ledoit-Wolf shrinkage ("_lw")
    "vc_ewma_sample_m": partial(VolCorr, vol="ewma", corr="sample", base="1"),
    "vc_ewma_sample_m_lw": partial(VolCorr, vol="ewma", corr="sample", base="1", shrink="lw"),
    "vc_ewma_ewma_m": partial(VolCorr, vol="ewma", corr="ewma", base="1"),
    "vc_ewma_ewma_m_lw": partial(VolCorr, vol="ewma", corr="ewma", base="1", shrink="lw"),
    "vc_ewma_dcc_m_lw": partial(VolCorr, vol="ewma", corr="dcc", base="1", shrink="lw"),
    "vc_ewma_daily_m_lw": partial(VolCorr, vol="ewma", corr="daily", base="1", shrink="lw"),
    "vc_daily_ewma_m_lw": partial(VolCorr, vol="daily", corr="ewma", base="1", shrink="lw"),
    "vc_daily_daily_m_lw": partial(VolCorr, vol="daily", corr="daily", base="1", shrink="lw"),
    "vc_daily_sample_m_lw": partial(VolCorr, vol="daily", corr="sample", base="1", shrink="lw"),
    # round 3: correlation regimes (no mean reversion of the anomaly, "_rw"),
    # shrinkage toward identity ("_lwi"), i.i.d. variance scaling ("_novr")
    "vc_daily_ewma_m_lw_rw": partial(VolCorr, vol="daily", corr="ewma", base="1", shrink="lw", phi_r=1.0),
    "vc_daily_daily_m_lw_rw": partial(VolCorr, vol="daily", corr="daily", base="1", shrink="lw", phi_r=1.0),
    "vc_daily_ewma_m_lwi_rw": partial(VolCorr, vol="daily", corr="ewma", base="1", shrink="lw",
                                      shrink_target="identity", phi_r=1.0),
    "vc_daily_ewma_m_lw_novr": partial(VolCorr, vol="daily", corr="ewma", base="1", shrink="lw", vr=False),
    # round 4: eigenvalue floor on the correlation matrix instead of LW ("_eig")
    "vc_daily_ewma_m_eig_rw": partial(VolCorr, vol="daily", corr="ewma", base="1", shrink="eig0.05", phi_r=1.0),
    "vc_daily_ewma_m_eig": partial(VolCorr, vol="daily", corr="ewma", base="1", shrink="eig0.05"),
    "vc_ewma_ewma_m_eig": partial(VolCorr, vol="ewma", corr="ewma", base="1", shrink="eig0.05"),
    # round 6: LW applied to the static level only ("_lwb"), so regime shifts
    # in the dynamic anomaly are not pulled toward the cross-pair average
    "vc_daily_ewma_m_lwb": partial(VolCorr, vol="daily", corr="ewma", base="1", shrink="lwbase"),
    "vc_daily_daily_m_lwb": partial(VolCorr, vol="daily", corr="daily", base="1", shrink="lwbase"),
    "vc_ewma_ewma_m_lwb": partial(VolCorr, vol="ewma", corr="ewma", base="1", shrink="lwbase"),
    # round 5: feature-based (ridge) forecast of daily correlation anomalies
    "vc_daily_macro_m_lw": MacroCorr,
}

for _n, _f in MODELS.items():
    register(_n, _f)

# Frozen list for the holdout run, chosen on dev data only (origins 1990-01 to
# DEV_END - h). The benchmark ``sample`` is always run as well.
FROZEN: tuple[str, ...] = (
    "vc_daily_ewma_m_lwb",  # primary: daily-RV vols; monthly corr level, LW-shrunk; monthly-EWMA anomaly unshrunk
    "vc_ewma_ewma_m_lwb",   # monthly-only analogue; its diagonal is exactly vol_ewma's sd^2
    "vc_daily_daily_m_lw",  # daily lead-lag correlation anomaly, then LW (best stock-bond score on dev)
    "ewma_cov",             # RiskMetrics baseline (dynamic, no conditioning)
    "lw",                   # static shrinkage baseline (conditioning, no dynamics)
)
