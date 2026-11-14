"""A synthetic Panel with known properties, for tests and for development
before (or without) the real data. Shapes, dtypes and conventions match
``load_panel()`` exactly.

The generator is a two-factor model with persistent stochastic volatility,
staggered asset start dates, and an early monthly-only era, so code that
handles unequal histories and mixed frequency gets exercised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..universe import ASSET_IDS
from .panel import PREDICTOR_COLUMNS, Panel

# long-run annual vols and loadings on (growth, rates) factors
_SPEC = {
    "us_large": (0.16, 1.0, 0.0), "us_small": (0.21, 1.1, 0.0),
    "intl_dev": (0.17, 0.9, 0.0), "em": (0.23, 0.9, 0.0),
    "ust_int": (0.06, -0.1, 1.0), "ust_long": (0.12, -0.15, 1.0),
    "tips": (0.06, 0.0, 0.8), "ig_credit": (0.07, 0.3, 0.8),
    "hy_credit": (0.09, 0.7, 0.2), "reits": (0.20, 0.8, 0.1),
    "gold": (0.16, 0.0, 0.2), "commodities": (0.18, 0.3, -0.1),
}


def make_synthetic_panel(seed: int = 0, start: str = "1975-01-01", end: str = "2015-12-31",
                         monthly_only_years: int = 10, stagger: bool = True) -> Panel:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    n, k = len(days), len(ASSET_IDS)

    def log_vol_path(persistence=0.995, vol_of_vol=0.035):
        x = np.zeros(n)
        e = rng.standard_normal(n) * vol_of_vol
        for i in range(1, n):
            x[i] = persistence * x[i - 1] + e[i]
        return np.exp(x - x.var() / 2)

    market_state = log_vol_path()
    factors = rng.standard_normal((n, 2)) * market_state[:, None]
    rets = np.empty((n, k))
    for j, a in enumerate(ASSET_IDS):
        vol, b_g, b_r = _SPEC[a]
        load = np.array([b_g, b_r])
        idio_share = 0.5
        sys = factors @ load / max(np.linalg.norm(load), 1e-9) * np.sqrt(1 - idio_share)
        idio = rng.standard_normal(n) * log_vol_path() * np.sqrt(idio_share)
        rets[:, j] = (sys + idio) * vol / np.sqrt(252) + 0.03 / 252
    cash_d = pd.Series(0.03 / 252, index=days, name="cash")
    daily_full = pd.DataFrame(np.expm1(rets) + cash_d.values[:, None], index=days, columns=ASSET_IDS)

    monthly = (1 + daily_full).groupby(pd.Grouper(freq="ME")).prod() - 1
    cash_m = (1 + cash_d).groupby(pd.Grouper(freq="ME")).prod() - 1
    cash_m.name = "cash"

    # staggered starts: asset j begins j years after the panel start
    first_daily = days[0] + pd.DateOffset(years=monthly_only_years)
    daily = daily_full.loc[first_daily:].copy()
    tiers = pd.DataFrame("C", index=monthly.index, columns=ASSET_IDS, dtype=object)
    tiers.loc[first_daily:] = "B"
    if stagger:
        for j, a in enumerate(ASSET_IDS):
            s = days[0] + pd.DateOffset(years=j)
            monthly.loc[monthly.index < s, a] = np.nan
            daily.loc[daily.index < s, a] = np.nan
            tiers.loc[tiers.index < s, a] = np.nan

    m_idx = monthly.index
    state_m = pd.Series(market_state, index=days).groupby(pd.Grouper(freq="ME")).mean()
    pred = pd.concat([
        pd.DataFrame({"obs_date": m_idx, "available_date": m_idx, "series": "synthetic_implied_vol",
                      "value": 0.16 * state_m.values * np.exp(rng.standard_normal(len(m_idx)) * 0.05)}),
        pd.DataFrame({"obs_date": m_idx, "available_date": m_idx + pd.offsets.MonthEnd(1),
                      "series": "synthetic_macro", "value": rng.standard_normal(len(m_idx)).cumsum()}),
    ], ignore_index=True)[PREDICTOR_COLUMNS]

    return Panel(daily=daily, monthly=monthly, cash_daily=cash_d.reindex(daily.index),
                 cash_monthly=cash_m, tiers=tiers, predictors=pred)
