"""Stock-bond correlation as a function of observable state (regression).

Target (training) for origin ``s`` and core pair ``p`` (us_large with
ust_int / ust_long):

    y_ps = z(realized 12m corr over (s, s+12m]) - z(ref_p(s))

where ``z`` = Fisher z and ``ref_p(s)`` is the pair value of the reference
baseline ``ref_corr_last120m`` at ``s`` (half last-10y, half all-history
monthly correlation). The model therefore predicts the ERROR OF THE BASELINE
and its prior centre (all coefficients 0) is the baseline itself, so every
coefficient measures incremental value over it.

    y_ps = b0 + b_rec * (z(rc12_ps) - z(ref_ps))          recent 12m corr (target estimator)
              + b_sht * (z(rc63_ps) - z(ref_ps))          last 63 days, 1-day returns
              + g1 * infl_yoy_s + g2 * infl_vol36_s + g3 * long_yield_s
              + e

All slopes >= 0 (economics: correlation is persistent; high and volatile
inflation and high yields mean supply/monetary shocks dominate and stocks
and bonds fall together), ridge-shrunk toward 0 (see ``regression``),
pooled over the core pairs with common coefficients (the target is already
relative to each pair's own baseline, so no pair intercepts are needed). Re-estimated every December on the
windows fully observed by then; applied monthly with the current predictors.

Information-set ladder at origin ``t`` (explicit, no silent imputation):
  1. a predictor not observable at ``t`` is dropped;
  2. a predictor set with fewer than ``MIN_TRAIN`` training months is reduced
     by dropping the predictor with the fewest observations, repeatedly;
  3. with fewer than ``MIN_TRAIN`` training months even for the intercept
     (origins before ~1962), the model returns None and the caller uses the
     unconditional base matrix (= ``ref_corr_last120m``) unchanged.
Training starts with windows opening in 1952 (after the March 1951
Treasury-Fed Accord ended the bond-yield peg: before it bond returns were
administered, and the stock-bond correlation had a different mechanism).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import (fisher_z, month_grid, predictors_at, realized_corr_series, ref_pair_corr,
                   short_corr_series)
from .regression import RidgeFit, fit_ridge

H = 12
TRAIN_START = pd.Timestamp("1952-01-31")
MIN_TRAIN = 120               # training months (10 years) before a fit is trusted
TAU = 0.2                     # prior sd of a 1-sd predictor effect on Fisher z
CORE_PAIRS = [("us_large", "ust_int"), ("us_large", "ust_long")]
MACRO = ["macro_infl_yoy", "macro_infl_vol_36m", "macro_long_yield"]
RECENT, SHORT, EQVOL = "recent12", "short63", "eqvol_ratio"
STRESS_PAIRS = [("us_large", "hy_credit"), ("us_large", "reits")]
STRESS = [RECENT, EQVOL, "credit_moodys_baa_aaa_spread_m"]


def eqvol_ratio(view, grid, asset: str = "us_large", days: int = 63, long_days: int = 2520) -> pd.Series:
    """log(trailing 63-day vol / trailing 10-year vol) of ``asset``'s 1-day log
    excess returns: the real-time equity stress state (both legs use 1-day
    returns from the same sources, so pricing conventions largely cancel)."""
    d = view.excess_daily[asset]
    out = pd.Series(np.nan, index=grid)
    if d.notna().sum() < days:
        return out
    x = d.to_numpy(float)
    ok = np.isfinite(x)
    x2 = np.where(ok, x * x, 0.0)
    P2, Pn = np.r_[0, np.cumsum(x2)], np.r_[0, np.cumsum(ok)]
    e = d.index.searchsorted(grid, side="right")
    s_short, s_long = np.maximum(e - days, 0), np.maximum(e - long_days, 0)
    n_s, n_l = Pn[e] - Pn[s_short], Pn[e] - Pn[s_long]
    with np.errstate(invalid="ignore", divide="ignore"):
        v_s = (P2[e] - P2[s_short]) / n_s
        v_l = (P2[e] - P2[s_long]) / n_l
        r = 0.5 * np.log(v_s / v_l)
    ok_r = (n_s >= 50) & (n_l >= 750) & np.isfinite(r)
    return pd.Series(np.where(ok_r, r, np.nan), index=grid)


def fit_date(origin: pd.Timestamp) -> pd.Timestamp:
    """Latest December month-end <= origin (parameters are refit once a year)."""
    o = pd.Timestamp(origin)
    y = o.year if o.month == 12 else o.year - 1
    return pd.Timestamp(year=y, month=12, day=31)


@dataclass
class SBPrediction:
    z: dict                    # pair -> predicted Fisher z
    fit: RidgeFit | None
    used: list[str]
    ref_z: dict                # pair -> baseline z at the origin
    x: dict                    # predictor values at the origin


def build_design(view, pairs, wanted):
    """Per-pair frames (index = month-end) with target ``y`` (deviation of the
    realized corr over the NEXT 12 months from the baseline; NaN if not yet
    observed in the view) and the predictor columns in ``wanted``."""
    grid = month_grid(view)
    rc, _ = realized_corr_series(view, pairs, grid)
    ref = ref_pair_corr(view, pairs, grid)
    sh = short_corr_series(view, pairs, grid) if SHORT in wanted else None
    macro = [w for w in wanted if w.startswith(("macro_", "credit_"))]
    ev = eqvol_ratio(view, grid) if EQVOL in wanted else None
    mp = predictors_at(view, macro, grid) if macro else None
    out = {}
    for a, b in pairs:
        col = f"{a}~{b}"
        zr = fisher_z(ref[col].to_numpy())
        f = pd.DataFrame(index=grid)
        fut = rc[col].shift(-H)                    # realized over (s, s+12m]
        f["y"] = fisher_z(fut.to_numpy()) - zr
        f["ref_z"] = zr
        if RECENT in wanted:
            f[RECENT] = fisher_z(rc[col].to_numpy()) - zr
        if sh is not None:
            f[SHORT] = fisher_z(sh[col].to_numpy()) - zr
        for m in macro:
            f[m] = mp[m].to_numpy()
        if ev is not None:
            f[EQVOL] = ev.to_numpy()
        out[(a, b)] = f
    return out


def predict_stockbond(view, pairs, wanted: list[str], tau: float = TAU) -> SBPrediction | None:
    """Fit (on data up to the last December) and predict at the view's last month-end."""
    if not pairs:
        return None
    frames = build_design(view, pairs, wanted)
    origin = next(iter(frames.values())).index[-1]
    F = fit_date(origin)
    last_s = F - pd.offsets.MonthEnd(H)        # training origins s with s + 12m <= F
    # 1. predictors observable now (for every pair)
    used = [w for w in wanted if all(np.isfinite(fr[w].iloc[-1]) for fr in frames.values())]

    def train_rows(cols):
        parts = []
        for fr in frames.values():
            tr = fr.loc[(fr.index >= TRAIN_START) & (fr.index <= last_s), ["y"] + cols]
            parts.append(tr.dropna())
        return parts

    # 2. reduce until enough training months
    while True:
        parts = train_rows(used)
        n_months = len(pd.Index(np.unique(np.concatenate([p.index.values for p in parts]))))
        if n_months >= MIN_TRAIN or not used:
            break
        counts = {w: sum(fr.loc[(fr.index >= TRAIN_START) & (fr.index <= last_s), w].notna().sum()
                         for fr in frames.values()) for w in used}
        used.remove(min(counts, key=counts.get))
    if n_months < MIN_TRAIN:
        return None                                # 3. fallback to the base matrix
    k = len(parts)
    X = np.vstack([p[used].to_numpy(float) for p in parts]) if used else np.zeros((sum(map(len, parts)), 0))
    y = np.concatenate([p["y"].to_numpy(float) for p in parts])
    w = np.full(len(y), 1.0 / (H * k))
    fit = fit_ridge(X, y, w, used, tau=tau, nonneg=[True] * len(used))
    z, refz, xs = {}, {}, {}
    for pair, fr in frames.items():
        x = fr[used].iloc[-1].to_numpy(float)
        refz[pair] = float(fr["ref_z"].iloc[-1])
        z[pair] = refz[pair] + fit.predict(x)
        xs[pair] = dict(zip(used, x))
    return SBPrediction(z=z, fit=fit, used=used, ref_z=refz, x=xs)
