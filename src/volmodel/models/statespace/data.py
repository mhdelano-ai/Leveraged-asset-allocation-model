"""Observations for the state-space models, built from a Panel view only.

For every asset ``i`` and calendar month ``t`` of the view:

``r2``  squared monthly LOG EXCESS return. Its expectation given the latent
        state is the month's variance, which is what the 21-day target
        averages (12 x mean of monthly variances = the annual target).
``y1``  ``log(r2 + DELTA * mbar) - KAPPA``: the monthly measurement of the
        latent log variance ``h``. ``mbar`` is the mean of ``r2`` over the
        previous 36 months (causal). The small offset (half a percent of
        recent variance; Fuller 1996, Breidt & Carriquiry 1996) removes the
        ``log 0`` inlier tail of a log chi-square(1) without moving its bulk:
        for normal returns ``y1 = h + e`` with ``E e = 0`` and
        ``Var e = SIGMA1_NORMAL = 3.24`` instead of 4.93. The exact constant
        is re-calibrated in each model (see ``scale`` in ``models.py``), so
        fat tails do not bias the forecast level.
``y2``  ``log RV1``: log of the sum of squared DAILY log excess returns in the
        month, where the asset has daily data on at least 80% of the month's
        calendar rows and at least 15 days. Its level relative to ``h``
        (the 1-day/21-day variance ratio) is NOT assumed: the models carry a
        learnt offset state ``c`` for it.
``reset`` True in a month where the asset's data tier changes inside its
        daily era (a source splice visible in the view): the offset ``c``
        then gets fresh prior variance.
``dact`` True from the asset's first ``y2`` month on (the offset random
        walk only runs once daily data exist).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

DELTA = 0.005                 # inlier offset, share of trailing mean r2
KAPPA = -1.0979738144849467   # E log(z^2 + DELTA), z ~ N(0, 1)
SIGMA1_NORMAL = 3.23807524702426   # Var log(z^2 + DELTA)
MIN_DAYS = 15
MIN_DAY_FRAC = 0.8


@dataclass
class Obs:
    months: pd.DatetimeIndex
    assets: list[str]
    r2: np.ndarray      # (T, A)
    y1: np.ndarray      # (T, A)
    y2: np.ndarray      # (T, A)
    reset: np.ndarray   # (T, A) bool
    dact: np.ndarray    # (T, A) bool
    nobs: np.ndarray    # (T, A) number of y1 observations up to and including t

    def upto(self, t) -> int:
        """Number of months <= t."""
        return int(self.months.searchsorted(pd.Timestamp(t), side="right"))


def build_obs(view) -> Obs:
    xm = view.excess_monthly
    assets = list(xm.columns)
    r = xm.to_numpy(float)
    r2 = r ** 2
    r2s = pd.DataFrame(r2, index=xm.index)
    mbar = r2s.rolling(36, min_periods=1).mean().shift(1).to_numpy()
    floor = np.where(np.isfinite(mbar), DELTA * mbar, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        y1 = np.log(np.maximum(r2 + floor, 1e-14)) - KAPPA
    y1[~np.isfinite(r2)] = np.nan

    # daily realized variance per calendar month
    y2 = np.full_like(y1, np.nan)
    xd = view.excess_daily
    if len(xd):
        me = xd.index + pd.offsets.MonthEnd(0)
        xd2 = xd.reindex(columns=assets)
        sq = (xd2 ** 2).groupby(me).sum(min_count=1)
        cnt = xd2.notna().groupby(me).sum()
        rows = pd.Series(1, index=xd.index).groupby(me).sum()
        ok = (cnt.ge(MIN_DAYS)) & (cnt.ge(rows.to_numpy()[:, None] * MIN_DAY_FRAC)) & (sq > 0)
        rv = sq.where(ok)
        rv = rv.reindex(xm.index)
        with np.errstate(divide="ignore", invalid="ignore"):
            y2 = np.log(rv.to_numpy(float))
        y2[~np.isfinite(y2)] = np.nan
        # a partial first daily month fails the 80%-of-rows rule above
    has2 = np.isfinite(y2)
    dact = np.cumsum(has2, axis=0) > 0

    tiers = view.tiers.reindex(index=xm.index, columns=assets)
    tv = tiers.to_numpy(object)
    prev = np.vstack([tv[:1], tv[:-1]])
    tv_s = np.where(pd.isna(tv), "", tv).astype(str)
    pv_s = np.where(pd.isna(prev), "", prev).astype(str)
    changed = (tv_s != pv_s) & (tv_s != "") & (pv_s != "")
    # a splice is relevant only once the offset is live (after the first y2)
    prev_dact = np.vstack([np.zeros((1, len(assets)), bool), dact[:-1]])
    reset = changed & prev_dact

    nobs = np.cumsum(np.isfinite(y1), axis=0)
    return Obs(months=xm.index, assets=assets, r2=r2, y1=y1, y2=y2, reset=reset,
               dact=dact, nobs=nobs)


def fingerprint(view) -> str:
    """Hash of everything in the view the models read (returns and tiers)."""
    h = hashlib.sha1()
    for df in (view.monthly, view.daily):
        h.update(np.ascontiguousarray(df.to_numpy(float)).tobytes())
        h.update(str(df.index[-1] if len(df) else "").encode())
        h.update(",".join(map(str, df.columns)).encode())
    h.update(np.ascontiguousarray(view.cash_monthly.to_numpy(float)).tobytes())
    h.update(np.ascontiguousarray(view.cash_daily.to_numpy(float)).tobytes())
    h.update(pd.util.hash_pandas_object(view.tiers.fillna(""), index=True).values.tobytes())
    return h.hexdigest()
