"""HAR-type direct 12-month variance regressions: estimation and forecasting.

Normalized form. With y the realized 12-month target variance and LR the
asset's long-run (expanding) variance at the origin, every regressor enters
as a ratio R_h = RV_h / LR, so coefficients are comparable across assets
and pooling is meaningful:

  log form     log(y / LR) = c + sum_h b_h log R_h + e
               forecast  = LR * exp(c + sum_h b_h log R_h) * phi
               phi = mean(exp(e_hat)) over the training rows (Duan's
               smearing estimator): E[y | x] = exp(x b) E[exp(e)], so the
               forecast targets the conditional MEAN of the variance, which
               is what QLIKE rewards, not exp(E[log y]) (the median under
               lognormal errors). No normality is assumed.
               The implied weight on the long run is 1 - sum_h b_h: the
               forecast is a geometric combination LR^(1 - sum b) prod RV_h^b_h.
  qlike form   y_hat = LR * kappa * (w_0 + sum_h w_h R_h)
               = kappa * (w_0 LR + sum_h w_h RV_h),  w = softmax (w >= 0,
               sum w = 1), kappa > 0; fitted by minimizing the mean QLIKE
               y / y_hat - log(y / y_hat) - 1 over training rows. Linear in
               the variances, so no retransformation: QLIKE is minimized by
               the conditional mean. kappa absorbs any common scale error.

Constraints: ``log`` is free OLS; ``logcvx`` imposes b_h >= 0 and
sum_h b_h <= 1 (a convex combination of horizons plus a pull to the long-run
mean, fixed a priori, not tuned); ``qlike`` is convex by construction.

Pooling: ``pooled`` (one set of coefficients for all assets, common
intercept); ``fe`` (pooled slopes, asset intercepts shrunk toward the common
one: delta_a = mean residual_a * n_a / (n_a + 120), n_a the asset's training
origins - the prior is worth ten years of monthly origins, i.e. about ten
independent annual targets); ``asset`` (each asset alone when it has at least
120 training origins of its own, else the pooled fit).

Estimation (walk-forward, inside the model): at origin t the coefficients are
those estimated at e = the last December 31 <= t, from the view truncated at
e: every (month-end s, asset) with at least 36 months of monthly history at
s and a realized target fully observed by e (s + 12 months <= e), computed
by ``assetforecast.eval.targets.compute_targets`` on that view - the same
estimator as the scored target, derived from the view, never from cached
files. Rows overlap (monthly origins, 12-month targets); that does not bias
the point estimates, only their standard errors, which the model never uses.
A fit needs at least 120 distinct training origins; until then the forecast
falls back to the next model in the chain (daily -> monthly -> prior), the
prior being the geometric mean of LR and the 60-month variance (ref_blend).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ...eval.targets import compute_targets
from ..base import VolForecaster
from .features import DAILY_REGS, MONTHLY_REGS, har_features

MIN_ASSET_MONTHS = 36     # the harness's eligibility rule, applied to training rows too
MIN_ORIGINS = 120         # distinct training origins (ten years) before a fit is trusted
FE_PRIOR = 120.0          # shrinkage of asset intercepts, in training origins
LOG_FLOOR = np.log(0.05)  # floor on log(RV_h / LR): bounds the leverage of a near-zero
                          # short-window measurement (a vol below 22% of the long run)


# ------------------------------------------------------------------ fitting
@dataclass
class Fit:
    form: str                     # log | logcvx | qlike
    regs: tuple
    coef: np.ndarray              # log: b_h ; qlike: w_h (horizons, excluding LR)
    const: float                  # log: c ; qlike: log kappa
    phi: float = 1.0              # smearing factor (log forms)
    w_lr: float = 0.0             # weight on the long run (diagnostic)
    n_rows: int = 0
    n_origins: int = 0
    asset_shift: dict | None = None   # fe: asset -> delta_a

    def predict(self, R: np.ndarray, assets=None) -> np.ndarray:
        """R: n x H ratios RV_h / LR. Returns y / LR."""
        if self.form == "qlike":
            return np.exp(self.const) * (self.w_lr + R @ self.coef)
        z = np.maximum(np.log(R), LOG_FLOOR)
        lin = self.const + z @ self.coef
        if self.asset_shift is not None and assets is not None:
            lin = lin + np.array([self.asset_shift.get(a, 0.0) for a in assets])
        return np.exp(lin) * self.phi


def _fit_log(R, y, convex: bool) -> tuple[np.ndarray, float, np.ndarray]:
    Z = np.maximum(np.log(R), LOG_FLOOR)
    ly = np.log(y)
    X = np.column_stack([np.ones(len(ly)), Z])
    if not convex:
        beta = np.linalg.lstsq(X, ly, rcond=None)[0]
    else:
        # min ||ly - X beta||^2  s.t. b >= 0, sum b <= 1 (intercept free); tiny QP
        A = X.T @ X / len(ly)
        g = X.T @ ly / len(ly)
        H = Z.shape[1]
        cons = [{"type": "ineq", "fun": lambda bb: 1.0 - bb[1:].sum(), "jac": lambda bb: np.r_[0.0, -np.ones(H)]}]
        x0 = np.r_[ly.mean(), np.full(H, 0.5 / H)]
        res = minimize(lambda bb: bb @ A @ bb - 2 * g @ bb, x0, jac=lambda bb: 2 * (A @ bb - g),
                       bounds=[(None, None)] + [(0.0, None)] * H, constraints=cons, method="SLSQP",
                       options={"maxiter": 500, "ftol": 1e-12})
        beta = res.x
        beta[1:] = np.maximum(beta[1:], 0.0)
    resid = ly - X @ beta
    return beta[1:], float(beta[0]), resid


def _fit_qlike(R, y) -> tuple[np.ndarray, float, float]:
    H = R.shape[1]
    Rf = np.column_stack([np.ones(len(y)), R])           # LR column first

    def unpack(th):
        a = np.r_[0.0, th[:H]]
        w = np.exp(a - a.max())
        return w / w.sum(), th[H]

    def obj(th):
        w, lk = unpack(th)
        f = np.exp(lk) * (Rf @ w)
        u = y / f
        val = np.mean(u - np.log(u) - 1.0)
        # gradient
        dl_df = (1.0 - u) / f                              # d/df of u - log u
        g_lk = np.mean(dl_df * f)
        dfdw = np.exp(lk) * Rf                             # n x (H+1)
        gw = (dl_df[:, None] * dfdw).mean(0)
        J = np.diag(w) - np.outer(w, w)                    # dw/da
        ga = J @ gw
        return val, np.r_[ga[1:], g_lk]

    th0 = np.r_[np.zeros(H), 0.0]
    res = minimize(obj, th0, jac=True, method="L-BFGS-B", options={"maxiter": 1000, "gtol": 1e-10})
    w, lk = unpack(res.x)
    return w[1:], float(lk), float(w[0])


def fit_one(form: str, regs: tuple, R: np.ndarray, y: np.ndarray, origins: np.ndarray,
            assets: np.ndarray | None = None, fe: bool = False) -> Fit | None:
    n_orig = len(np.unique(origins))
    if n_orig < MIN_ORIGINS:
        return None
    if form == "qlike":
        w, lk, w0 = _fit_qlike(R, y)
        return Fit(form, regs, w, lk, 1.0, w0, len(y), n_orig)
    b, c, resid = _fit_log(R, y, convex=(form == "logcvx"))
    shift = None
    if fe:
        df = pd.DataFrame({"a": assets, "o": origins, "r": resid})
        g = df.groupby("a")
        n_a = g["o"].nunique()
        shift = (g["r"].mean() * n_a / (n_a + FE_PRIOR)).to_dict()
        resid = resid - df["a"].map(shift).to_numpy()
    phi = float(np.mean(np.exp(resid)))
    return Fit(form, regs, b, c, phi, 1.0 - float(b.sum()), len(y), n_orig, shift)


# ------------------------------------------------------------- the model
class HarVol(VolForecaster):
    """See the module docstring. ``regs`` is "M" (monthly regressors) or "D"
    (daily-mapped short windows, falling back to "M" where unavailable)."""

    def __init__(self, regs: str = "D", form: str = "log", pool: str = "pooled"):
        assert regs in ("M", "D") and form in ("log", "logcvx", "qlike") and pool in ("pooled", "fe", "asset")
        self.regs, self.form, self.pool = regs, form, pool
        self._cache: dict = {}
        self._cache_owner = None

    # ---------------------------------------------------------- training set
    @staticmethod
    def training_rows(view_e) -> dict:
        F = har_features(view_e)
        tv = compute_targets(dataclasses.replace(view_e, asof_date=None)).vol
        grid, assets = F["grid"], F["assets"]
        Y = np.full((len(grid), len(assets)), np.nan)
        if len(tv):
            w = tv.pivot(index="origin", columns="asset", values="var")
            w = w.reindex(index=pd.DatetimeIndex(grid).astype("datetime64[ns]"), columns=assets)
            Y = w.to_numpy(float)
        elig = F["n_obs"] >= MIN_ASSET_MONTHS
        base = elig & np.isfinite(Y) & (Y > 0) & np.isfinite(F["LR"]) & (F["LR"] > 0)
        return {"F": F, "Y": Y, "base": base}

    @staticmethod
    def _design(F, regs, mask):
        gi, ai = np.nonzero(mask)
        R = np.column_stack([F[r][gi, ai] / F["LR"][gi, ai] for r in regs])
        return gi, ai, R

    def _fit_set(self, tr: dict, regs: tuple) -> dict:
        """{'pooled': Fit|None, 'asset': {asset: Fit}} for one regressor set."""
        F, Y = tr["F"], tr["Y"]
        mask = tr["base"] & (F["daily_ok"] if regs == DAILY_REGS else True)
        gi, ai, R = self._design(F, regs, mask)
        ok = np.isfinite(R).all(1) & (R > 0).all(1)
        gi, ai, R = gi[ok], ai[ok], R[ok]
        y = Y[gi, ai] / F["LR"][gi, ai]
        an = np.asarray(F["assets"], dtype=object)[ai]
        out = {"pooled": fit_one(self.form, regs, R, y, gi, an, fe=(self.pool == "fe")), "asset": {}}
        if self.pool == "asset":
            for a in np.unique(an):
                s = an == a
                f = fit_one(self.form, regs, R[s], y[s], gi[s])
                if f is not None:
                    out["asset"][a] = f
        return out

    def fit(self, view_e) -> dict:
        tr = self.training_rows(view_e)
        fits = {"M": self._fit_set(tr, MONTHLY_REGS)}
        if self.regs == "D":
            fits["D"] = self._fit_set(tr, DAILY_REGS)
        return fits

    # -------------------------------------------------------------- forecast
    @staticmethod
    def estimation_date(t) -> pd.Timestamp:
        t = pd.Timestamp(t)
        return pd.Timestamp(year=t.year if t.month == 12 and t.day == 31 else t.year - 1, month=12, day=31)

    def _fits_for(self, view):
        if self._cache_owner is not view._shared:
            self._cache, self._cache_owner = {}, view._shared
        e = self.estimation_date(view.asof_date)
        if e not in self._cache:
            self._cache[e] = self.fit(view.asof(e))
        return self._cache[e]

    def forecast_var(self, view, assets: list[str]) -> tuple[pd.Series, pd.Series]:
        """Variance forecasts and the route used for each asset (diagnostic)."""
        fits = self._fits_for(view)
        F = har_features(view)
        cols = F["assets"]
        i = len(F["grid"]) - 1
        out, route = {}, {}
        for a in assets:
            j = cols.index(a)
            LR = F["LR"][i, j]
            v, how = np.nan, "prior"
            chain = []
            if self.regs == "D" and F["daily_ok"][i, j]:
                chain.append(("D", DAILY_REGS))
            chain.append(("M", MONTHLY_REGS))
            for key, regs in chain:
                fs = fits[key]
                f = fs["asset"].get(a) if self.pool == "asset" else None
                tag = f"{key}:asset" if f is not None else f"{key}:pooled"
                f = f or fs["pooled"]
                if f is None:
                    continue
                R = np.array([[F[r][i, j] / LR for r in regs]])
                if np.isfinite(R).all() and (R > 0).all():
                    v = float(f.predict(R, [a])[0] * LR)
                    how = tag
                    break
            if not (np.isfinite(v) and v > 0):
                v, how = float(np.sqrt(LR * F["M60"][i, j])), "prior"
            out[a], route[a] = v, how
        return pd.Series(out).reindex(assets), pd.Series(route).reindex(assets)

    def forecast_vol(self, view, assets):
        v, _ = self.forecast_var(view, assets)
        return np.sqrt(v).rename(None)
