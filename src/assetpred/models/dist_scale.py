"""Scale/shape components for :mod:`assetpred.models.dist_core`.

All components model the residual ``e = y - h * mean(xr)`` about the
historical mean as ``e = sigma(x) * z``: a conditional scale ``sigma(x)`` and a
standardized shape for ``z``.

* :class:`EWMAScale` -- the ``vol_ewma`` scale (RiskMetrics EWMA, lambda 0.94,
  mean-reverting projection, variance ratio, sqrt(1 + h/T)); with
  ``shape="std"`` it reproduces ``vol_ewma_fhs`` exactly.
* :class:`LinearScale` -- QLIKE (Gaussian quasi-likelihood) regression of the
  log h-month residual variance on a handful of volatility state variables,
  pooled across assets in asset-relative units (a HAR-type model with
  implied volatility). Non-ML reference for the ML scale.
* :class:`GBMScale` -- gradient boosting (LightGBM, gamma deviance = QLIKE)
  started from the linear model's log variance (``init_score``) and adding
  trees chosen by purged expanding CV (0 trees allowed, i.e. it can fall back
  to the linear model). Shape from purged out-of-fold standardized residuals.
* :class:`AggScale` -- a one-month variance model (linear or boosted)
  mapped to h months affinely, ``v_a * (alpha + beta * r(x))``, with alpha,
  beta fitted by QLIKE on out-of-fold monthly forecasts. The direct
  log-linear h-month fit (LinearScale/GBMScale at h > 1) understates width in
  high-volatility states because the h-month variance is a *sum* of
  mean-reverting monthly variances; this structure fixed that on the dev
  period and is the basis of the nominated models.
* :class:`CondShapeGBM` -- conditional shape: LightGBM quantile regression of
  the out-of-fold standardized residual ``z`` on the state, started from the
  unconditional quantiles; tests whether tails/asymmetry move beyond scale.

Units. A pooled model needs comparable inputs across assets: the target is
``u = e^2 / v_a`` with ``v_a`` the asset's mean squared residual in the
training window; asset-level volatility features enter as log variance minus
the asset's training-window mean of that log variance (so daily and monthly
sources with different levels are each self-normalised); global features are
centred on the training window.

Shape. ``"std"``: pooled empirical z quantiles rescaled to unit sd (the
``vol_ewma_fhs`` convention); ``"pooled"``: pooled, centred, not rescaled
(keeps any systematic scale miss); ``"group"``: per asset class (equity,
credit, rates, real assets) shrunk to the pooled shape (``dist_core.shape_quantiles``).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import TAUS, InsufficientDataError, TrainingSet, empirical_z_quantiles
from .dist_core import (ASSET_GROUP, DistComponent, grid_mean, lookup_shape,
                        residuals, shape_quantiles, centered_quantiles)
from ..evaluation.cv import purged_splits

# asset-level volatility features (annualised sd unless noted) -> log variance, asset-relative
ASSET_VOL = ("vol_d_1m", "vol_d_3m", "vol_m_12m", "vol_m_36m")
DERIVED_LV = ("_d_lewma94", "_d_lewma985")            # already log variance
# other state variables for the "vol" set
STATE = ("vol_d_ratio", "dd_12m", "dd_max", "mom_1m_vs", "mom_3m_vs", "mom_12_1_vs")
GLOBAL_VOL = ("log_vix", "vix_rel_12m", "vrp", "g_eq_vol_d_1m", "g_default_spread_z",
              "g_gz_spread", "g_sb_corr_d_3m", "g_eq_dd_max")


def _allowed_all(columns) -> list:
    """Every feature that is neither hindsight nor discontinued (synthetic panels: all)."""
    try:
        from ..features import select
        ok = set(select.columns(hindsight=False))
        meta = set(select.feature_meta().index)
    except Exception:  # no processed data (tests on synthetic panels)
        ok, meta = set(), set()
    return [c for c in columns if (c in ok or c not in meta) and not str(c).startswith("_d_")]


class FeatureMap:
    """Training-window normalisation into asset-relative units (see module doc)."""

    def __init__(self, extra: tuple = ()):
        self.extra = tuple(extra)

    def fit(self, X: pd.DataFrame):
        assets = X.index.get_level_values("asset")
        self._amean = {}
        for c in ASSET_VOL + DERIVED_LV:
            if c in X.columns:
                lv = self._lv(X[c], c)
                self._amean[c] = (lv.groupby(assets).mean().to_dict(), float(np.nanmean(lv)))
        self._gmean = {c: float(np.nanmean(X[c])) for c in GLOBAL_VOL if c in X.columns}
        if "g_eq_vol_d_1m" in self._gmean:
            self._gmean["g_eq_vol_d_1m"] = float(np.nanmean(np.log(X["g_eq_vol_d_1m"].clip(lower=1e-6))))
        return self

    @staticmethod
    def _lv(s: pd.Series, c: str) -> pd.Series:
        if c in DERIVED_LV:
            return s.astype(float)
        return 2 * np.log(s.astype(float).clip(lower=1e-6))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assets = X.index.get_level_values("asset")
        out = {}
        for c, (am, pm) in self._amean.items():
            mu = np.array([am.get(a, pm) for a in assets], float)
            mu = np.where(np.isfinite(mu), mu, pm)
            out["r_" + c] = self._lv(X[c], c).to_numpy() - mu
        for c, m in self._gmean.items():
            v = X[c].astype(float).to_numpy()
            if c == "g_eq_vol_d_1m":
                v = np.log(np.clip(v, 1e-6, None))
            out["g_" + c] = v - m
        for c in STATE + self.extra:
            if c in X.columns and c not in out:
                out[c] = X[c].astype(float).to_numpy()
        return pd.DataFrame(out, index=X.index)


def _linear_design(F: pd.DataFrame, inputs: str = "all") -> np.ndarray:
    """Small, always-defined design for the linear QLIKE model. ``inputs="returns"``
    keeps only what monthly returns give (EWMAs, 12m vol, down-move): no daily
    realized vol, no VIX (an ablation)."""
    def col(name, fallback=None):
        v = F[name].to_numpy(float) if name in F.columns else np.full(len(F), np.nan)
        if fallback is not None:
            v = np.where(np.isfinite(v), v, fallback)
        return np.where(np.isfinite(v), v, 0.0)
    ew = col("r__d_lewma94")
    slow = col("r__d_lewma985", ew)
    m12 = col("r_vol_m_12m", slow)
    d1 = col("r_vol_d_1m", ew)
    vix = F["g_log_vix"].to_numpy(float) if "g_log_vix" in F.columns else np.full(len(F), np.nan)
    has_vix = np.isfinite(vix).astype(float)
    vix = np.where(np.isfinite(vix), vix, 0.0)
    neg = np.minimum(col("mom_1m_vs"), 0.0)
    if inputs == "returns":
        return np.column_stack([np.ones(len(F)), ew, slow, m12, neg])
    return np.column_stack([np.ones(len(F)), ew, slow, m12, d1, vix, has_vix, neg])


def qlike_fit(D: np.ndarray, u: np.ndarray, ridge: float = 1e-3, iters: int = 50) -> np.ndarray:
    """Minimise mean(u * exp(-D b) + D b) (QLIKE / Gamma deviance, log link) by Newton."""
    n, p = D.shape
    b = np.zeros(p)
    b[0] = np.log(max(u.mean(), 1e-12))
    pen = ridge * np.eye(p)
    pen[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(D @ b, -20, 20)
        w = u * np.exp(-eta)
        g = D.T @ (1 - w) / n + pen @ b
        H = (D * w[:, None]).T @ D / n + pen
        step = np.linalg.solve(H + 1e-9 * np.eye(p), g)
        b -= step
        if np.abs(step).max() < 1e-8:
            break
    return b


def _qlike(u, eta):
    return float(np.mean(u * np.exp(-eta) + eta))


class _ResidualScaleBase(DistComponent):
    """Shared: residuals, target units, standardized residuals, shape."""

    shape: str = "pooled"

    def _prepare(self, train: TrainingSet):
        e = residuals(train)
        if len(e) < 60:
            raise InsufficientDataError("too few residuals for a scale model")
        X = train.X.loc[e.index]
        assets = e.index.get_level_values("asset")
        v = (e ** 2).groupby(assets).mean()
        v = v.where(e.groupby(assets).size() >= 12)
        self._va = v.dropna().to_dict()
        self._va_pool = float((e ** 2).mean())
        # widen for the estimation error of the historical mean, as the baselines do
        nT = train.xr.notna().sum()
        self._infl = np.sqrt(1.0 + train.horizon / nT.clip(lower=1)).to_dict()
        va = np.array([self._va.get(a, self._va_pool) for a in assets])
        u = np.clip((e ** 2).to_numpy() / va, 1e-8, None)
        return e, X, u, va

    def _va_of(self, X):
        return np.array([self._va.get(a, self._va_pool) for a in X.index.get_level_values("asset")])

    def _infl_of(self, X):
        return np.array([self._infl.get(a, 1.0) for a in X.index.get_level_values("asset")])

    def _set_shape(self, z: pd.Series, horizon: int):
        z = z[np.isfinite(z.to_numpy())]
        if self.shape == "std":
            self._shapes = {None: empirical_z_quantiles(z.to_numpy())}
            self._zsd = 1.0
        else:
            self._shapes = shape_quantiles(z, horizon, self.shape,
                                           half_life=getattr(self, "half_life", None))
            self._zsd = float(z.std())

    def predict_offsets(self, X):
        sig = self.sigma(X)
        zq = lookup_shape(self._shapes, X.index.get_level_values("asset"))
        return sig * self._zsd, sig[:, None] * zq

    def describe(self):
        return _default_describe(self)


def _default_describe(obj):
    items = {k: v for k, v in sorted(vars(obj).items())
             if not k.startswith("_") and not k.endswith("_")
             and isinstance(v, (int, float, str, bool, type(None), tuple))}
    return f"{type(obj).__qualname__}({items})"


# --------------------------------------------------------------------------
class EWMAScale(DistComponent):
    """``vol_ewma`` scale (see ``baselines.VolEWMA``) with a choice of shape."""

    def __init__(self, lam: float = 0.94, phi: float = 0.9, shape: str = "std", refit_every: int = 1):
        self.lam, self.phi, self.shape, self.refit_every = lam, phi, shape, refit_every

    def describe(self):
        return _default_describe(self)

    def fit(self, train: TrainingSet):
        xr = train.xr.dropna(axis=1, how="all")
        if xr.empty:
            raise InsufficientDataError("no return history")
        Y = train.realized_targets()[xr.columns]
        h = train.horizon
        mu = xr.mean()
        ewma = ((xr - mu) ** 2).ewm(alpha=1 - self.lam, adjust=False, ignore_na=True).mean()
        lr = xr.var(ddof=1)
        vr = (Y.var(ddof=1) / (h * lr)).clip(0.25, 4.0).fillna(1.0)
        decay = (self.phi ** np.arange(1, h + 1)).sum()
        infl = np.sqrt(1.0 + h / xr.notna().sum().clip(lower=1))
        hv = (vr * (h * lr + decay * (ewma.iloc[-1] - lr))).clip(lower=0.05 * h * lr)
        self._sd = (np.sqrt(hv) * infl).to_dict()
        hv_path = (vr * (h * lr + decay * (ewma - lr))).clip(lower=0.05 * h * lr, axis=1)
        z = ((Y - h * mu) / np.sqrt(hv_path)).stack().dropna()
        z.index = z.index.set_names(["date", "asset"])
        if self.shape == "std":
            self._shapes = {None: empirical_z_quantiles(z.to_numpy())}
            self._zsd = 1.0
        else:
            self._shapes = shape_quantiles(z, h, self.shape)
            self._zsd = float(z.std())
        return self

    def sigma(self, X):
        try:
            return np.array([self._sd[a] for a in X.index.get_level_values("asset")], float)
        except KeyError as e:
            raise InsufficientDataError(f"no EWMA scale for {e}") from None

    def predict_offsets(self, X):
        sig = self.sigma(X)
        zq = lookup_shape(self._shapes, X.index.get_level_values("asset"))
        return sig * self._zsd, sig[:, None] * zq


# --------------------------------------------------------------------------
class LinearScale(_ResidualScaleBase):
    """QLIKE log-variance regression on 7 state variables (see ``_linear_design``)."""

    def __init__(self, shape: str = "pooled", refit_every: int = 1, ridge: float = 1e-3,
                 oof_splits: int = 5, inputs: str = "all"):
        self.shape, self.refit_every, self.ridge, self.inputs = shape, refit_every, ridge, inputs
        self.oof_splits = oof_splits    # 0: shape from in-sample standardized residuals

    def describe(self):
        return _default_describe(self)

    def fit_core(self, X: pd.DataFrame, u: np.ndarray):
        self._fm = FeatureMap().fit(X)
        self._b = qlike_fit(_linear_design(self._fm.transform(X), self.inputs), u, self.ridge)
        return self

    def log_rel_var(self, X) -> np.ndarray:
        return _linear_design(self._fm.transform(X), self.inputs) @ self._b

    def fit(self, train: TrainingSet):
        e, X, u, va = self._prepare(train)
        sig = np.full(len(e), np.nan)
        if self.oof_splits:
            for tr, vl in purged_splits(X, train.horizon, self.oof_splits, mode="kfold", min_train_dates=60):
                m = LinearScale(ridge=self.ridge, inputs=self.inputs).fit_core(X.iloc[tr], u[tr])
                sig[vl] = np.sqrt(va[vl] * np.exp(np.clip(m.log_rel_var(X.iloc[vl]), -10, 10)))
        self.fit_core(X, u)
        if np.isnan(sig).all():
            sig = np.sqrt(va * np.exp(np.clip(self.log_rel_var(X), -10, 10)))
        self.sig_oof_ = pd.Series(sig, index=e.index)
        self.z_oof_ = pd.Series(e.to_numpy() / sig, index=e.index)
        self._set_shape(self.z_oof_, train.horizon)
        return self

    def sigma(self, X):
        return self._infl_of(X) * np.sqrt(self._va_of(X) * np.exp(np.clip(self.log_rel_var(X), -10, 10)))


# --------------------------------------------------------------------------
_LGB_BASE = dict(learning_rate=0.03, num_leaves=7, max_depth=3, min_data_in_leaf=200,
                 bagging_fraction=0.7, bagging_freq=1, feature_fraction=0.7,
                 lambda_l2=10.0, num_threads=1, verbose=-1)


class GBMScale(_ResidualScaleBase):
    """Linear QLIKE start + LightGBM gamma-deviance corrections (see module doc).

    ``features``: ``"vol"`` (volatility/stress state) or ``"all"`` (every
    non-hindsight, live feature plus the volatility transforms).
    ``trees``: candidate numbers of boosting rounds, chosen by purged
    expanding CV on QLIKE (``0`` = linear model only).
    """

    def __init__(self, features: str = "vol", shape: str = "pooled", refit_every: int = 12,
                 trees=(0, 50, 100, 200, 400), cv_splits: int = 3, oof_splits: int = 5):
        self.features, self.shape, self.refit_every = features, shape, refit_every
        self.trees, self.cv_splits, self.oof_splits = tuple(trees), cv_splits, oof_splits

    def describe(self):
        return _default_describe(self)

    def _design(self, fm: FeatureMap, X: pd.DataFrame) -> pd.DataFrame:
        F = fm.transform(X)
        if self.features == "all":
            extra = [c for c in self._all_cols if c not in F.columns]
            F = pd.concat([F, X[extra].astype(float)], axis=1)
        return F

    def _fit_one(self, X, u, n_trees, seed):
        import lightgbm as lgb
        lin = LinearScale(ridge=1e-3).fit_core(X, u)
        F = self._design(lin._fm, X)
        init = lin.log_rel_var(X)
        booster = None
        if n_trees > 0:
            ds = lgb.Dataset(F.to_numpy(float), label=u, init_score=init, free_raw_data=False)
            booster = lgb.train(dict(_LGB_BASE, objective="gamma", seed=seed), ds, num_boost_round=n_trees)
        return lin, booster

    def _predict_one(self, lin, booster, X, n_trees=None):
        init = lin.log_rel_var(X)
        if booster is None:
            return init
        F = self._design(lin._fm, X).to_numpy(float)
        return init + booster.predict(F, raw_score=True, num_iteration=n_trees)

    def fit(self, train: TrainingSet):
        e, X, u, va = self._prepare(train)
        self._all_cols = _allowed_all(train.X.columns) if self.features == "all" else []
        h = train.horizon
        # 1) number of trees by purged expanding CV on QLIKE
        cand = sorted(set(self.trees))
        n_best = 0
        folds = purged_splits(X, h, self.cv_splits, mode="expanding", min_train_dates=60)
        if len(cand) > 1 and folds:
            loss = np.zeros(len(cand))
            for tr, va_ in folds:
                lin, bst = self._fit_one(X.iloc[tr], u[tr], max(cand), train.seed)
                # u in the validation block uses the full-window v_a; that only shifts
                # every candidate's loss by the same constant per asset
                for j, n in enumerate(cand):
                    eta = self._predict_one(lin, bst if n > 0 else None, X.iloc[va_], n if n > 0 else None)
                    loss[j] += _qlike(u[va_], eta) * len(va_)
            n_best = cand[int(np.argmin(loss))]
        self.n_trees_ = n_best
        # 2) out-of-fold standardized residuals for the shape (purged k-fold)
        sig_oof = np.full(len(e), np.nan)
        for tr, va_ in purged_splits(X, h, self.oof_splits, mode="kfold", min_train_dates=60):
            lin, bst = self._fit_one(X.iloc[tr], u[tr], n_best, train.seed)
            sig_oof[va_] = np.sqrt(va[va_] * np.exp(np.clip(self._predict_one(lin, bst, X.iloc[va_]), -10, 10)))
        # 3) final fit on the whole window
        self._lin, self._bst = self._fit_one(X, u, n_best, train.seed)
        if np.isnan(sig_oof).all():
            sig_oof = self.sigma(X) / self._infl_of(X)
        z = pd.Series(e.to_numpy() / sig_oof, index=e.index)
        self.sig_oof_ = pd.Series(sig_oof, index=e.index)
        self.z_oof_ = z
        self._set_shape(z, h)
        return self

    def sigma(self, X):
        eta = np.clip(self._predict_one(self._lin, self._bst, X), -10, 10)
        return self._infl_of(X) * np.sqrt(self._va_of(X) * np.exp(eta))


# --------------------------------------------------------------------------
class CondShapeGBM(_ResidualScaleBase):
    """Conditional shape on top of a scale model.

    The scale model (``scale``: ``"linear"``, ``"agg_linear"`` (:class:`AggScale`)
    or a :class:`GBMScale` feature set ``"gbm_vol"``/``"gbm_all"``) gives
    sigma(x) and out-of-fold z.
    LightGBM quantile regressions of z on the state (start: the unconditional
    z quantile) at taus 5..95%; the number of rounds (shared by all taus,
    0 allowed) is chosen by purged expanding CV on pinball loss. The 1% / 99%
    quantiles extend the conditional 5% / 95% by the unconditional tail gap
    scaled with the conditional spread. Each row's z quantiles are recentred
    so their grid mean equals the unconditional shape's: the shape may tilt
    (median vs mean) but cannot move the location.
    """

    INNER = (1, 2, 3, 4, 5, 6, 7)   # indices of 5%..95% in TAUS

    def __init__(self, scale: str = "linear", refit_every: int = 12, trees=(0, 50, 100, 200),
                 cv_splits: int = 3):
        self.scale, self.refit_every, self.trees, self.cv_splits = scale, refit_every, tuple(trees), cv_splits

    def describe(self):
        return _default_describe(self)

    def _state(self, X):
        F = self._fm.transform(X)
        sm = self._scale_model
        F["lsig"] = np.log(sm.sigma(X) / sm._infl_of(X) / np.sqrt(sm._va_of(X)))
        return F.to_numpy(float)

    def fit(self, train: TrainingSet):
        import lightgbm as lgb
        if self.scale == "linear":
            sm = LinearScale().fit(train)
        elif self.scale.startswith("agg_"):
            sm = AggScale(monthly=self.scale.split("_", 1)[1]).fit(train)
        else:
            sm = GBMScale(features=self.scale.split("_", 1)[1]).fit(train)
        e, X, _, _ = self._prepare(train)
        z = sm.z_oof_.reindex(e.index)
        self._scale_model = sm
        self._fm = FeatureMap().fit(X)
        zz = z.to_numpy()
        ok = np.isfinite(zz)
        X, zz = X[ok], zz[ok]
        self._u = centered_quantiles(zz)               # unconditional shape (centred)
        zz = zz - np.nanmean(zz)
        S = self._state(X)
        h = train.horizon
        cand = sorted(set(self.trees))

        def fit_taus(rows, n):
            out = {}
            for k in self.INNER:
                if n == 0:
                    out[k] = None
                    continue
                ds = lgb.Dataset(S[rows], label=zz[rows], init_score=np.full(len(rows), self._u[k]))
                out[k] = lgb.train(dict(_LGB_BASE, objective="quantile", alpha=float(TAUS[k]),
                                        seed=train.seed), ds, num_boost_round=n)
            return out

        def pred_taus(models, rows, n):
            Q = np.tile(self._u, (len(rows), 1))
            for k, m in models.items():
                if m is not None:
                    Q[:, k] = self._u[k] + m.predict(S[rows], raw_score=True, num_iteration=n)
            return Q

        n_best = 0
        folds = purged_splits(X, h, self.cv_splits, mode="expanding", min_train_dates=60)
        if len(cand) > 1 and folds:
            loss = np.zeros(len(cand))
            for tr, va in folds:
                ms = fit_taus(tr, max(cand))
                for j, n in enumerate(cand):
                    Q = pred_taus(ms if n > 0 else {}, va, n)
                    r = zz[va][:, None] - Q[:, list(self.INNER)]
                    t = TAUS[list(self.INNER)]
                    loss[j] += np.maximum(t * r, (t - 1) * r).sum()
            n_best = cand[int(np.argmin(loss))]
        self.n_trees_ = n_best
        self._models = fit_taus(np.arange(len(zz)), n_best) if n_best > 0 else {}
        self._n = n_best
        self._zsd = float(np.std(zz))
        return self

    def sigma(self, X):
        return self._scale_model.sigma(X)

    def z_quantiles(self, X):
        S = self._state(X)
        Q = np.tile(self._u, (len(X), 1))
        for k, m in self._models.items():
            if m is not None:
                Q[:, k] = self._u[k] + m.predict(S, raw_score=True, num_iteration=self._n)
        Q[:, 1:8] = np.sort(Q[:, 1:8], axis=1)
        u = self._u
        lo = (Q[:, 1] - Q[:, 4]) / (u[1] - u[4])
        hi = (Q[:, 7] - Q[:, 4]) / (u[7] - u[4])
        Q[:, 0] = Q[:, 1] + (u[0] - u[1]) * np.clip(lo, 0.25, 4)
        Q[:, 8] = Q[:, 7] + (u[8] - u[7]) * np.clip(hi, 0.25, 4)
        Q -= (grid_mean(Q) - grid_mean(u[None, :]))[:, None]
        return Q

    def predict_offsets(self, X):
        sig = self.sigma(X)
        return sig * self._zsd, sig[:, None] * self.z_quantiles(X)


# --------------------------------------------------------------------------
def monthly_training_set(train: TrainingSet) -> TrainingSet:
    """Horizon-1 view of an h-month training set: same feature rows (dated
    <= origin - h), one-month targets xr[s+1] from ``train.xr``."""
    if train.horizon == 1:
        return train
    y1 = train.xr.shift(-1).stack().dropna()
    y1.index = y1.index.set_names(["date", "asset"])
    tg = y1.reindex(train._features.index)
    return TrainingSet(horizon=1, origin=train.origin, xr=train.xr, assets=train.assets,
                       seed=train.seed, _features=train._features, _targets=tg)


def _fit_affine_qlike(u: np.ndarray, r: np.ndarray) -> tuple[float, float]:
    """alpha, beta >= 0 minimising mean(u / (alpha + beta r) + log(alpha + beta r))."""
    from scipy.optimize import minimize
    ok = np.isfinite(u) & np.isfinite(r)
    u, r = u[ok], r[ok]

    def f(p):
        a, b = np.exp(p)
        s = a + b * r
        return float(np.mean(u / s + np.log(s)))
    best = None
    for a0, b0 in ((0.5, 0.5), (0.1, 0.9), (0.9, 0.1)):
        res = minimize(f, np.log([a0, b0]), method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9, maxiter=2000))
        if best is None or res.fun < best.fun:
            best = res
    a, b = np.exp(best.x)
    return float(a), float(b)


class AggScale(_ResidualScaleBase):
    """Monthly conditional variance model, aggregated to h months arithmetically.

    A one-month scale model (``monthly``: ``"linear"`` = :class:`LinearScale`,
    ``"gbm_vol"`` = :class:`GBMScale` on volatility features) gives the
    relative monthly variance r(x) = sigma_1^2(x) / v1_a. The h-month residual
    variance is ``v_a * (alpha + beta * r(x))`` with alpha, beta >= 0 fitted by
    QLIKE on the h-month residuals using the monthly model's *out-of-fold*
    r: the h-month variance is a sum of mean-reverting monthly variances, so
    it is affine (not log-linear) in today's variance, and alpha/beta play the
    role of the EWMA projection's mean-reversion and variance-ratio terms.
    For h = 1 it is the monthly model recalibrated. Shape: as configured, from
    the out-of-fold h-month standardized residuals.
    """

    def __init__(self, monthly: str = "linear", shape: str = "pooled", refit_every: int = 1,
                 half_life: float | None = None, inputs: str = "all"):
        self.monthly, self.shape, self.refit_every = monthly, shape, refit_every
        self.inputs = inputs           # linear monthly model inputs (see _linear_design)
        self.half_life = half_life     # months; recency weighting of the pooled shape

    def describe(self):
        return _default_describe(self)

    def fit(self, train: TrainingSet):
        t1 = monthly_training_set(train)
        if self.monthly == "linear":
            m = LinearScale(shape="pooled", refit_every=1, inputs=self.inputs)
        elif self.monthly.startswith("gbm_"):
            m = GBMScale(features=self.monthly.split("_", 1)[1], shape="pooled")
        else:
            raise ValueError(self.monthly)
        m.fit(t1)
        self._m = m
        r_oof = (m.sig_oof_ ** 2 / np.array([m._va.get(a, m._va_pool)
                                              for a in m.sig_oof_.index.get_level_values("asset")]))
        e, X, u, va = self._prepare(train)
        r = r_oof.reindex(e.index).to_numpy()
        self.alpha_, self.beta_ = _fit_affine_qlike(u, r)
        sig = np.sqrt(va * (self.alpha_ + self.beta_ * r))
        self.sig_oof_ = pd.Series(sig, index=e.index)
        self.z_oof_ = pd.Series(e.to_numpy() / sig, index=e.index)
        self._set_shape(self.z_oof_, train.horizon)
        return self

    def _r(self, X):
        m = self._m
        s1 = m.sigma(X) / m._infl_of(X)
        return s1 ** 2 / m._va_of(X)

    def sigma(self, X):
        return self._infl_of(X) * np.sqrt(self._va_of(X) * (self.alpha_ + self.beta_ * self._r(X)))
