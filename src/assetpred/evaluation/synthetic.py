"""Synthetic panels that obey the data contract, for testing models and the harness.

Monthly excess log return of asset a over month t+1::

    xr[t+1, a] = mu_a + sigma[t, a] * (beta * s[t, a] + sqrt(1 - beta^2) * eps[t+1, a])

where ``s`` is a unit-variance AR(1) signal observed at t (feature ``signal``),
so the population one-month R^2 of the signal is beta^2. Optional stochastic
volatility makes ``sigma`` persistent and time-varying (predictable scale,
unpredictable mean). Other features: per-asset AR(1) noise (``noise_k``), a
global AR(1) noise repeated across assets (``global_noise``), and trailing
12-month return (``mom12``, legitimately known at t).

Use it to check a new model: on ``beta=0`` it must not beat ``hist_mean`` out
of sample; on ``beta>0`` it should.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config, schema


@dataclass
class SyntheticData:
    returns: pd.DataFrame     # wide simple total returns (contract: asset_returns)
    cash: pd.Series           # simple T-bill return (contract: cash)
    xr: pd.DataFrame          # wide monthly excess log returns
    features: pd.DataFrame    # long (date, asset)
    signal: pd.DataFrame      # wide true signal s[t, a]
    sigma: pd.DataFrame       # wide true conditional monthly sd for month t+1


def _ar1(rng, n, k, rho):
    e = rng.standard_normal((n, k))
    x = np.empty((n, k))
    x[0] = e[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + np.sqrt(1 - rho ** 2) * e[t]
    return x


def make_synthetic(n_assets: int = 4, start: str = "1960-01-31", end: str = "2012-12-31",
                   beta: float = 0.0, signal_rho: float = 0.9, n_noise: int = 3,
                   stoch_vol: bool = False, late_start: bool = True, seed: int = 0
                   ) -> SyntheticData:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="ME")
    n = len(dates)
    assets = list(config.ASSETS)[:n_assets]
    mu = rng.uniform(0.001, 0.006, n_assets)
    base_sd = rng.uniform(0.02, 0.06, n_assets)
    s = _ar1(rng, n, n_assets, signal_rho)
    if stoch_vol:
        lv = _ar1(rng, n, n_assets, 0.95) * 0.45
        sigma = base_sd * np.exp(lv - 0.45 ** 2 / 2)
    else:
        sigma = np.tile(base_sd, (n, 1))
    eps = rng.standard_normal((n, n_assets))
    xr = np.full((n, n_assets), np.nan)
    xr[1:] = mu + sigma[:-1] * (beta * s[:-1] + np.sqrt(1 - beta ** 2) * eps[1:])
    xr[0] = mu + base_sd * eps[0]
    xr = pd.DataFrame(xr, index=dates, columns=assets)
    if late_start and n_assets > 1:
        xr.iloc[:120, -1] = np.nan
    cash = pd.Series(0.003 + 0.001 * _ar1(rng, n, 1, 0.98)[:, 0], index=dates, name="cash")
    returns = np.expm1(xr.add(np.log1p(cash), axis=0))
    xr = schema.excess_log_returns(returns, cash)

    cols = {"signal": s}
    for k in range(n_noise):
        cols[f"noise_{k}"] = _ar1(rng, n, n_assets, 0.9)
    cols["global_noise"] = np.repeat(_ar1(rng, n, 1, 0.95), n_assets, axis=1)
    cols["mom12"] = xr.rolling(12, min_periods=12).sum().to_numpy()
    long = {}
    for name, arr in cols.items():
        long[name] = pd.DataFrame(arr, index=dates, columns=assets).stack(future_stack=True)
    feats = pd.DataFrame(long).astype(float)
    feats.index = feats.index.set_names(["date", "asset"])
    # drop rows for assets before their returns begin
    first = xr.apply(lambda c: c.first_valid_index())
    keep = [d >= first[a] for d, a in feats.index]
    feats = feats[np.array(keep)].sort_index()
    return SyntheticData(returns=returns, cash=cash, xr=xr, features=feats,
                         signal=pd.DataFrame(s, index=dates, columns=assets),
                         sigma=pd.DataFrame(sigma, index=dates, columns=assets))
