"""The ``macro`` volatility model: a pooled QLIKE (Gamma-GLM) regression of
next-year variance on the asset's own history, plus economic state variables
added ON TOP of it, so that their contribution is the increment over the base.

Asset ``a`` in group ``g`` at origin ``s`` (variances annualized, 21-day-return
scale; inputs in ``features``)::

    stage 1 (base)   log E[y] = alpha_g + b1 log v_slow + b2 log v12 + b3 log v3
    stage 2 (state)  log E[y] = base_g(s, a) + delta_0 + sum_k gamma_k z_{k,a,s}

Both stages minimize ``nu * sum_i w_i [y_i / mu_i + log mu_i] + penalty``. The
bracket is QLIKE up to constants (the Gamma log-likelihood with a log link),
so the fit targets E[y] (not a geometric mean) and is consistent under the
noisy-but-unbiased training targets. ``w_i = 1 / (12 m_s)``, ``m_s`` = the
group's assets at month ``s``: each group-month counts 1/12, i.e. one
independent observation per group-year (12-month overlap; the assets of a
group share most of their volatility). Stage 1 is unpenalized and uses every
usable row of the group. Stage 2 keeps the stage-1 forecast as a fixed offset
and estimates a level shift ``delta_0`` and the state slopes on the rows where
all of its regressors exist, with a Gaussian prior N(0, TAU^2) on each (in
log-variance units per 1 sd of the regressor); ``nu = 1 / phi``, ``phi`` the
weighted mean of ``(y / mu_base - 1)^2`` on those rows, so the prior is
weighed against the likelihood of about ``n_years`` observations. Keeping the
base fixed stops a short sample (say 2003-2013 for MOVE) from re-estimating
the own-history slopes, which would otherwise swing wildly.

State variables ``z`` are standardized on the training rows (implied-vol
regressors are centred per asset, because their level differs by instrument
and carries a variance risk premium, and scaled by the pooled sd); at the
forecast they are clipped to +-4 sd.

Information sets and the fallback chain. At each origin and for each asset,
the candidate state variables of its group that are FRESH at the origin form
its information set (a stale or unpublished predictor is simply absent). The
stage-2 model for that set is trained only on rows where all of them existed.
If that leaves fewer than ``MIN_TRAIN_MONTHS`` distinct training months, the
variable with the shortest training history is dropped and the step repeated
(nested models by information set), down to the empty set = the base model.
If the group's base itself has too few training months (the first years of
the panel), the fixed geometric mean of ``v_slow`` and ``v12`` is used.
Re-estimation is yearly: the fit date is the last 31 December <= origin and
only targets whose window ends by then are used. A fit is a pure function of
the data up to its fit date, so it is cached in the panel's shared cache and
reused across calls and variants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import VolForecaster
from .features import ReturnFeatures, state_features

TAU = 0.25                 # prior sd of a state coefficient per 1-sd move (log-variance units)
MIN_TRAIN_MONTHS = 60      # 5 years of training origins before a regressor may enter
MIN_ELIGIBLE_MONTHS = 36   # same as the harness eligibility rule
Z_CLIP = 4.0

GROUPS = {
    "equity": ["us_large", "us_small", "intl_dev", "em", "reits"],
    "credit": ["ig_credit", "hy_credit"],
    "rates": ["ust_int", "ust_long", "tips"],
    "real": ["gold", "commodities"],
}
GROUP_OF = {a: g for g, al in GROUPS.items() for a in al}

# pre-specified on economic grounds (see the report); never searched
MACRO_SETS = {
    "equity": ["cs", "ts", "ivol", "sahm"],
    "credit": ["cs", "ts", "sahm", "ly"],
    "rates": ["ly", "ivol", "dpol", "ts"],
    "real": ["ivol", "infl", "rsr", "usd"],
}
IMPLIED_SETS = {
    "equity": ["vix"],
    "credit": ["vix"],
    "rates": ["move_d"],
    "real": ["iv_own"],
}
IMPLIED = {"vix", "move_d", "iv_own"}
CACHE_KEY = "macro_family_fits"


def fit_date(t: pd.Timestamp) -> pd.Timestamp:
    dec = pd.Timestamp(year=t.year, month=12, day=31)
    return dec if t >= dec else pd.Timestamp(year=t.year - 1, month=12, day=31)


# ------------------------------------------------------------------ GLM
def gamma_glm(X, y, w, pen, nu=1.0, beta0=None, max_iter=100, tol=1e-9):
    """Minimize nu * sum w (y exp(-Xb) + Xb) + 0.5 * sum pen b^2 by damped Newton."""
    n, p = X.shape
    b = np.zeros(p) if beta0 is None else beta0.copy()
    if beta0 is None:
        b[0] = np.log(np.sum(w * y) / np.sum(w))

    def obj(b):
        eta = X @ b
        return nu * np.sum(w * (y * np.exp(-eta) + eta)) + 0.5 * np.sum(pen * b * b)

    f = obj(b)
    for _ in range(max_iter):
        eta = X @ b
        r = y * np.exp(-eta)
        g = nu * X.T @ (w * (1.0 - r)) + pen * b
        H = nu * (X * (w * r)[:, None]).T @ X + np.diag(pen) + 1e-10 * np.eye(p)
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-6:
            bn = b - t * step
            fn = obj(bn)
            if np.isfinite(fn) and fn <= f + 1e-12:
                break
            t *= 0.5
        b, f_old, f = bn, f, fn
        if abs(f_old - f) < tol * (1 + abs(f)) and np.max(np.abs(t * step)) < 1e-7:
            break
    return b


def gamma_glm_offset(X, y, w, offset, pen, nu):
    """Same objective with a fixed offset: log mu = offset + X b."""
    return gamma_glm(X, y * np.exp(-offset), w, pen, nu=nu, beta0=np.zeros(X.shape[1]))


class _Data:
    """Everything a fit and a forecast need, from one view."""

    def __init__(self, view, need_state: bool):
        self.rf = ReturnFeatures(view)
        self.assets = self.rf.assets
        self.col = {a: j for j, a in enumerate(self.assets)}
        self.grid = self.rf.grid
        self.base = self.rf.base_matrix()
        self.state = state_features(view, self.grid, self.assets) if need_state else {}
        self.eligible = self.rf.n_months >= MIN_ELIGIBLE_MONTHS


class MacroVol(VolForecaster):
    def __init__(self, macro: bool, implied: bool, ip: bool = False, homog: bool = False):
        self.macro, self.implied, self.ip, self.homog = macro, implied, ip, homog
        self.last_fits: dict = {}

    # ---------------------------------------------------------- candidates
    def candidates(self, group: str) -> list[str]:
        c = []
        if self.macro:
            c += MACRO_SETS[group] + (["ip"] if self.ip else [])
        if self.implied:
            c += IMPLIED_SETS[group]
        return c

    # ---------------------------------------------------------------- fit
    @staticmethod
    def _train_rows(d: _Data, F: pd.Timestamp, assets: list[str], feats: tuple):
        """Boolean (G, len(assets)) mask of usable training cells."""
        rows = np.asarray(d.rf.y_end <= F)
        cols = [d.col[a] for a in assets]
        m = rows[:, None] & np.isfinite(d.rf.y[:, cols]) & d.eligible[:, cols]
        m &= np.isfinite(d.base[:, cols, :]).all(axis=-1)
        for f in feats:
            m &= np.isfinite(d.state[f][:, cols])
        return m

    def _fit(self, d: _Data, F, group: str, feats: tuple, cache: dict):
        key = (F, group, feats, self.homog)
        if key in cache:
            return cache[key]
        assets = GROUPS[group]
        cols = [d.col[a] for a in assets]
        mask = self._train_rows(d, F, assets, feats)
        n_months = int(mask.any(axis=1).sum())
        base = self._fit(d, F, group, (), cache) if feats else None
        if n_months < MIN_TRAIN_MONTHS or (feats and base is None):
            cache[key] = None
            return None
        gi, ai = np.nonzero(mask)
        y = d.rf.y[:, cols][gi, ai]
        X0 = np.column_stack([np.ones(len(y)), d.base[:, cols, :][gi, ai, :]])
        w = 1.0 / (12.0 * mask.sum(axis=1)[gi])
        if not feats:
            # stage 1, the own-history base: unpenalized, on every usable row
            if self.homog:
                # slopes sum to one: log E[y] = a + xs + c1 (x12 - xs) + c2 (x3 - xs)
                xs = X0[:, 1]
                Xh = np.column_stack([X0[:, 0], X0[:, 2] - xs, X0[:, 3] - xs])
                c = gamma_glm_offset(Xh, y, w, xs, np.zeros(3), 1.0)
                b0 = np.array([c[0], 1.0 - c[1] - c[2], c[1], c[2]])
            else:
                b0 = gamma_glm(X0, y, w, np.zeros(X0.shape[1]))
            mu0 = np.exp(X0 @ b0)
            phi = float(np.sum(w * (y / mu0 - 1.0) ** 2) / np.sum(w))
            fit = dict(beta=b0, feats=feats, centers={}, scales={}, phi=phi, nu=1.0 / max(phi, 1e-6),
                       n_months=n_months, n_rows=len(y), group=group, F=F, delta=np.zeros(1),
                       delta_se=np.zeros(1))
            cache[key] = fit
            return fit
        # stage 2: shift + state coefficients on top of the base forecast (an offset),
        # on the rows where every regressor exists; all of it shrunk toward 0
        offset = X0 @ base["beta"]
        mu0 = np.exp(offset)
        phi = float(np.sum(w * (y / mu0 - 1.0) ** 2) / np.sum(w))
        nu = 1.0 / max(phi, 1e-6)
        centers, scales, Z = {}, {}, []
        for f in feats:
            raw = d.state[f][:, cols][gi, ai]
            if f in IMPLIED:
                c = {assets[k]: float(raw[ai == k].mean()) for k in np.unique(ai)}
                c["_pooled"] = float(raw.mean())
                cen = raw - np.array([c[assets[k]] for k in ai])
            else:
                c = {"_pooled": float(raw.mean())}
                cen = raw - c["_pooled"]
            sd = float(cen.std()) or 1.0
            centers[f], scales[f] = c, sd
            Z.append(cen / sd)
        Xs = np.column_stack([np.ones(len(y))] + Z)
        pen = np.full(Xs.shape[1], 1.0 / TAU ** 2)
        delta = gamma_glm_offset(Xs, y, w, offset, pen, nu)
        r = y * np.exp(-(offset + Xs @ delta))
        H = nu * (Xs * (w * r)[:, None]).T @ Xs + np.diag(pen)
        delta_se = np.sqrt(np.diag(np.linalg.inv(H)))          # posterior sd (Laplace)
        beta = base["beta"].copy()
        beta[0] += delta[0]
        beta = np.concatenate([beta, delta[1:]])
        fit = dict(beta=beta, feats=feats, centers=centers, scales=scales, phi=phi, nu=nu,
                   n_months=n_months, n_rows=len(y), group=group, F=F, delta=delta,
                   delta_se=delta_se)
        cache[key] = fit
        return fit

    def _feature_months(self, d, F, group, f):
        return int(self._train_rows(d, F, GROUPS[group], (f,)).any(axis=1).sum())

    def select_fit(self, d: _Data, F, asset: str, cache: dict):
        """Walk the fallback chain for one asset at the last grid date."""
        group = GROUP_OF[asset]
        j = d.col[asset]
        feats = [f for f in self.candidates(group) if np.isfinite(d.state[f][-1, j])]
        while True:
            fit = self._fit(d, F, group, tuple(feats), cache)
            if fit is not None:
                return fit
            if not feats:
                break
            months = [self._feature_months(d, F, group, f) for f in feats]
            feats.pop(int(np.argmin(months)))
        return None

    # ------------------------------------------------------------ forecast
    def forecast_vol(self, view, assets):
        d = _Data(view, need_state=bool(self.macro or self.implied))
        t = d.grid[-1]
        F = fit_date(t)
        cache = view._shared.setdefault(CACHE_KEY, {})
        out = {}
        self.last_fits = {}
        for a in assets:
            j = d.col[a]
            xb = d.base[-1, j, :]
            if not np.isfinite(xb).all():
                x = view.excess_monthly[a].dropna()
                out[a] = float(np.sqrt(max(12.0 * np.mean(x.to_numpy() ** 2), 1e-6)))
                continue
            fit = self.select_fit(d, F, a, cache)
            self.last_fits[a] = fit
            if fit is None:                                  # earliest years: fixed blend
                eta = 0.5 * (xb[0] + xb[1])
            else:
                z = []
                for f in fit["feats"]:
                    c = fit["centers"][f]
                    v = (d.state[f][-1, j] - c.get(a, c["_pooled"])) / fit["scales"][f]
                    z.append(np.clip(v, -Z_CLIP, Z_CLIP))
                x = np.concatenate([[1.0], xb, z])
                eta = float(x @ fit["beta"])
            eta = float(np.clip(eta, np.log(1e-4), np.log(4.0)))
            out[a] = float(np.sqrt(np.exp(eta)))
        return pd.Series(out, index=assets, dtype=float)
