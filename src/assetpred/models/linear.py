"""Pooled ridge regression: a reference ML model and a template for point models.

Predicts each asset's deviation from its own training-window mean target with
one ridge regression pooled across assets. Features are median-imputed and
standardized on the training window only; the penalty is chosen by purged
expanding-window CV inside ``fit_point``. The predictive distribution comes
from :class:`~assetpred.models.base.PointForecaster` (out-of-fold residuals).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.cv import purged_splits
from .base import InsufficientDataError, PointForecaster
from .registry import register


class PooledRidge(PointForecaster):
    def __init__(self, alphas=(0.001, 0.01, 0.1, 1.0, 10.0), cv_splits: int = 4,
                 features: tuple | None = None):
        self.alphas = tuple(alphas)
        self.cv_splits = cv_splits
        self.features = features

    # -- ridge on standardized, imputed features; penalty alpha * n_rows ----
    # Everything is numpy inside the CV loops: this model is refit many times.
    @staticmethod
    def _prep_fit(Xa: np.ndarray, ya: np.ndarray, codes: np.ndarray, n_assets: int):
        with np.errstate(all="ignore"):
            med = np.nanmedian(Xa, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Z = np.where(np.isfinite(Xa), Xa, med)
        mu, sd = Z.mean(axis=0), Z.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Z = (Z - mu) / sd
        cnt = np.bincount(codes, minlength=n_assets)
        amean = np.bincount(codes, weights=ya, minlength=n_assets) / np.maximum(cnt, 1)
        amean = np.where(cnt > 0, amean, ya.mean())
        return Z, ya - amean[codes], (med, mu, sd), amean

    @staticmethod
    def _solve(Z, r, alpha):
        p = Z.shape[1]
        return np.linalg.solve(Z.T @ Z + alpha * len(r) * np.eye(p), Z.T @ r)

    @staticmethod
    def _apply(Xa, codes, prep, amean, beta):
        med, mu, sd = prep
        Z = (np.where(np.isfinite(Xa), Xa, med) - mu) / sd
        return amean[codes] + Z @ beta

    def _cols(self, X):
        return X[list(self.features)] if self.features else X

    def _codes(self, X):
        assets = X.index.get_level_values("asset")
        return np.array([self._asset_pos.setdefault(a, len(self._asset_pos)) for a in assets])

    def fit_point(self, X, y, horizon):
        if len(y) < 24:
            raise InsufficientDataError("too few rows for ridge")
        self._asset_pos = {}
        Xa = self._cols(X).to_numpy(dtype=float)
        ya = y.to_numpy(dtype=float)
        codes = self._codes(X)
        k = len(self._asset_pos)
        alpha = self.alphas[0]
        folds = purged_splits(X, horizon, self.cv_splits, mode="expanding") if len(self.alphas) > 1 else []
        if folds:
            losses = np.zeros(len(self.alphas))
            for tr, va in folds:
                Z, r, prep, amean = self._prep_fit(Xa[tr], ya[tr], codes[tr], k)
                for j, a in enumerate(self.alphas):
                    b = self._solve(Z, r, a)
                    pred = self._apply(Xa[va], codes[va], prep, amean, b)
                    losses[j] += float(((ya[va] - pred) ** 2).sum())
            alpha = self.alphas[int(np.argmin(losses))]
        Z, r, self._prep, self._amean = self._prep_fit(Xa, ya, codes, k)
        self._beta = self._solve(Z, r, alpha)
        self._fallback = float(ya.mean())
        self.alpha_ = alpha

    def predict_point(self, X):
        assets = X.index.get_level_values("asset")
        known = np.array([a in self._asset_pos for a in assets])
        codes = np.array([self._asset_pos.get(a, 0) for a in assets])
        pred = self._apply(self._cols(X).to_numpy(dtype=float), codes, self._prep, self._amean, self._beta)
        # an asset with no training rows gets the pooled mean plus the signal part
        pred = np.where(known, pred, pred - self._amean[codes] + self._fallback)
        return pred


register("ridge", PooledRidge)
