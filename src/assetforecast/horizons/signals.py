"""The benchmark and the single-signal return forecasters at every horizon.

Target: ``y[t, a] = xr[t+1] + ... + xr[t+h]``, the h-month log excess return.

``hist_mean``  ``h`` times the mean monthly log excess return through ``t``
               (at least 60 months): the bar. At h = 12 it is the package's
               published expected return.
``sig_<x>``    ``hist_mean + b * (x[t] - xbar)``: a predictive regression of the
               target on one signal, per asset. ``b`` and ``xbar`` come from
               the pairs ``(x[s], y[s])`` with ``s + h <= t`` (the purge: the
               target is realized at ``t``), expanding, re-estimated every
               month, at least ``MIN_PAIRS`` pairs. With ``b = 0`` it is the
               benchmark, so the Clark-West test for nested models applies.
``sig_combo``  the mean of the ``sig_<x>`` forecasts an asset has at ``t``
               (Rapach, Strauss and Zhou 2010).

The signals are columns of the committed feature panel (``returns/features``,
unchanged), fixed here before anything was run: an asset's own carry and
value, three price signals, and four market-wide valuation and rate series.
No slope is shrunk or sign-restricted. Overlapping targets bias long-horizon
slopes toward predictability in sample (Stambaugh 1999; Boudoukh, Richardson
and Whitelaw 2008); scoring out of sample is the guard against that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import EARLY_START, HORIZONS

SIGNALS = ("carry", "value", "mom_12_1", "trend_ma10", "lt_rev_60_12",
           "g_term_spread", "g_default_spread", "g_us_cape_ey_z", "g_tbill")
MIN_PAIRS = 120       # realized (signal, target) pairs before a slope is used
MIN_HISTORY = 60      # months of returns before an asset is forecast (as hist_mean)
BENCH = "hist_mean"
COMBO = "sig_combo"


def targets(xr: pd.DataFrame, h: int) -> pd.DataFrame:
    """Forward h-month sums dated at the origin (NaN until fully realized)."""
    return xr.rolling(h, min_periods=h).sum().shift(-h)


def hist_mean(xr: pd.DataFrame, h: int) -> pd.DataFrame:
    n = xr.notna().cumsum()
    m = xr.fillna(0.0).cumsum() / n.where(n > 0)
    return (h * m).where(n >= MIN_HISTORY)


def _signal_forecast(x: pd.DataFrame, y: pd.DataFrame, base: pd.DataFrame, h: int) -> pd.DataFrame:
    """``base + b (x - xbar)`` with b, xbar from pairs dated at least h months back."""
    ok = x.notna() & y.notna()
    x0, y0 = x.where(ok, 0.0), y.where(ok, 0.0)
    lag = lambda d: d.cumsum().shift(h)          # sums over pairs s <= t - h
    n, sx, sy = lag(ok.astype(float)), lag(x0), lag(y0)
    sxx, sxy = lag(x0 * x0), lag(x0 * y0)
    vx = sxx - sx * sx / n
    b = (sxy - sx * sy / n) / vx.where(vx > 1e-12 * sxx.abs().clip(lower=1e-300))
    f = base + b * (x - sx / n)
    return f.where(n >= MIN_PAIRS)


def run(features: pd.DataFrame, xr: pd.DataFrame, horizons=HORIZONS, start: str = EARLY_START
        ) -> pd.DataFrame:
    """Long frame ``model, horizon, date, asset, mean`` for the benchmark, every
    signal and the combination, origins from ``start``."""
    wide = {s: features[s].unstack("asset").reindex(index=xr.index, columns=xr.columns)
            for s in SIGNALS}
    out = []
    for h in horizons:
        y, base = targets(xr, h), hist_mean(xr, h)
        fc = {BENCH: base}
        for s in SIGNALS:
            fc[f"sig_{s}"] = _signal_forecast(wide[s], y, base, h)
        stack = np.stack([fc[f"sig_{s}"].to_numpy() for s in SIGNALS])
        with np.errstate(invalid="ignore"):
            cnt = np.isfinite(stack).sum(axis=0)
            combo = np.where(cnt > 0, np.nansum(stack, axis=0) / np.maximum(cnt, 1), np.nan)
        fc[COMBO] = pd.DataFrame(combo, index=xr.index, columns=xr.columns)
        for name, f in fc.items():
            f = f.loc[pd.Timestamp(start):]
            s = f.stack().dropna()
            s.index = s.index.set_names(["date", "asset"])
            d = s.rename("mean").reset_index()
            d.insert(0, "horizon", h)
            d.insert(0, "model", name)
            out.append(d)
    return pd.concat(out, ignore_index=True)
