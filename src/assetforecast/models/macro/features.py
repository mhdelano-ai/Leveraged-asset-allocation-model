"""Features for the ``macro`` family, all computed from a Panel view.

Everything is laid out on a GRID of month-ends (the view's monthly index, plus
the view's ``asof_date`` when it is not a month-end) and the 12 assets:

Own-history (target-scale variances of 21-day log excess returns, annualized)
    v_slow   EWMA of squared MONTHLY log excess returns, half-life 60 months,
             x 12 (all history; a slow average of past variance).
    v12      last 12 months: the target's own estimator (overlapping 21-day
             sums, uncentered, x n_rows/21) on daily data when the asset has
             full daily coverage of the window, else 12 x mean of the 12
             squared monthly returns (the target's monthly proxy).
    v3       last 63 trading days, same 21-day estimator (43 overlapping sums,
             annualized with the row count of the trailing 12 months); else
             12 x mean of the last 3 squared monthly returns.
    Because both estimators measure the 21-day variance, no 1-day vol is ever
    used and no 1-day/21-day scale ratio has to be carried across splices.

Training target (only used where ``s + 12 months <= fit date``)
    y        next-12-month variance by the same rule as ``v12`` applied to the
             window (s, s + 12 months] (daily estimator with coverage, else the
             monthly proxy, which needs all 12 months).

State variables (``state_features``)
    Each predictor's value AS KNOWN AT each grid date ``g``: only rows of the
    view's long predictor frame with ``available_date <= g`` are used.
    Monthly series: the latest published observation, stale (NaN) if its
    obs_date is more than ``MONTHLY_MAX_AGE_DAYS`` before ``g``. Daily series:
    the mean of the last ``DAILY_AVG_OBS`` published closes (the implied-vol
    track recommends averages over single prints), stale if the last close is
    more than ``DAILY_MAX_AGE_DAYS`` old.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...universe import ASSET_IDS

K = 21                      # target horizon in trading days
HALF_LIFE_SLOW = 60         # months: a business cycle; also ref_last60m's window
RECENT_DAYS = 63            # one quarter of trading days
MIN_DAILY_FRAC = 0.90       # coverage rule of the target module
MIN_WINDOW_ROWS_12M = 150   # a 12-month daily window must really be daily
MONTHLY_MAX_AGE_DAYS = 190  # ~6 months: covers the M+3 pre-1947 lag plus one missing print
DAILY_MAX_AGE_DAYS = 31
DAILY_AVG_OBS = 21
VAR_FLOOR = 1e-5            # annualized variance floor (vol 0.3%) before logs


# ------------------------------------------------------------------ grid
def make_grid(view) -> pd.DatetimeIndex:
    g = pd.DatetimeIndex(view.monthly.index)
    t = view.asof_date
    if t is None:
        t = max(g[-1], view.daily.index[-1]) if len(view.daily) else g[-1]
    t = pd.Timestamp(t)
    g = g[g <= t]
    if len(g) == 0 or g[-1] != t:
        g = g.append(pd.DatetimeIndex([t]))
    return g


def _shift_months(grid: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([d + pd.offsets.MonthEnd(n) if d.is_month_end else
                             d + pd.DateOffset(months=n) for d in grid])


# --------------------------------------------------------- own history
class ReturnFeatures:
    """Base features and training targets on the grid (arrays G x A)."""

    def __init__(self, view, assets=ASSET_IDS):
        self.assets = list(assets)
        self.grid = make_grid(view)
        G, A = len(self.grid), len(self.assets)
        rm = view.excess_monthly.reindex(columns=self.assets)
        mdates = rm.index
        R = rm.to_numpy(float)
        present = np.isfinite(R)
        R2 = np.where(present, R * R, 0.0)
        # month position for every grid date: number of month-ends <= g
        mpos = mdates.searchsorted(self.grid, side="right")          # 0..len
        self.n_months = np.vstack([np.zeros(A), np.cumsum(present, axis=0)])[mpos]  # history length

        # ---- slow EWMA of monthly squared returns
        lam = 0.5 ** (1.0 / HALF_LIFE_SLOW)
        num = np.zeros(A)
        den = np.zeros(A)
        ew = np.full((len(mdates) + 1, A), np.nan)
        for i in range(len(mdates)):
            num = lam * num + R2[i]
            den = lam * den + present[i]
            with np.errstate(invalid="ignore", divide="ignore"):
                ew[i + 1] = np.where(den > 0, 12.0 * num / den, np.nan)
        self.v_slow = ew[mpos]

        # ---- monthly window helpers (prefix sums)
        cs2 = np.vstack([np.zeros(A), np.cumsum(R2, axis=0)])
        cnt = np.vstack([np.zeros(A), np.cumsum(present, axis=0)])

        def mwin(lo, hi, need):
            """12 x mean of squared monthly returns over month positions [lo, hi)."""
            lo = np.clip(lo, 0, len(mdates))
            hi = np.clip(hi, 0, len(mdates))
            n = cnt[hi] - cnt[lo]
            s = cs2[hi] - cs2[lo]
            with np.errstate(invalid="ignore", divide="ignore"):
                v = 12.0 * s / n
            return np.where(n >= need, v, np.nan)

        m12_back = mwin(mpos - 12, mpos, 1)
        m3_back = mwin(mpos - 3, mpos, 1)
        m12_fwd = mwin(mpos, mpos + 12, 12)
        fwd_complete = (mpos + 12) <= len(mdates)

        # ---- daily 21-day-sum machinery
        dd = view.excess_daily.reindex(columns=self.assets)
        ddates = dd.index
        X = dd.to_numpy(float)
        ok = np.isfinite(X)
        X0 = np.where(ok, X, 0.0)
        D = len(ddates)
        if D > K:
            C = np.vstack([np.zeros(A), np.cumsum(X0, axis=0)])        # C[n] = sum rows < n
            S = np.zeros((D, A))
            S[K - 1:] = C[K:] - C[:D - K + 1]                          # sum of rows j-20..j
            Q = S * S
            Q[:K - 1] = 0.0
            CQ = np.vstack([np.zeros(A), np.cumsum(Q, axis=0)])
            N = np.vstack([np.zeros(A), np.cumsum(ok, axis=0)])
            first_row = np.where(ok.any(axis=0), ok.argmax(axis=0), D + 1)
        dpos = ddates.searchsorted(self.grid, side="right")             # rows <= g
        grid_fwd = _shift_months(self.grid, 12)
        grid_back = _shift_months(self.grid, -12)
        dpos_fwd = ddates.searchsorted(grid_fwd, side="right")
        dpos_back = ddates.searchsorted(grid_back, side="right")

        def dwin(a, b, rows_year, min_rows):
            """Target estimator on rows [a, b): (rows_year / 21) * mean(S^2); NaN without coverage."""
            out = np.full((G, A), np.nan)
            if D <= K:
                return out
            a = np.asarray(a)
            b = np.asarray(b)
            M = b - a - (K - 1)
            good = (M >= 1) & ((b - a) >= min_rows)
            ai = np.clip(a + K - 1, 0, D)
            bi = np.clip(b, 0, D)
            meanq = (CQ[bi] - CQ[ai]) / np.maximum(M, 1)[:, None]
            cov = (N[bi] - N[np.clip(a, 0, D)]) >= MIN_DAILY_FRAC * (b - a)[:, None]
            started = first_row[None, :] <= a[:, None]
            v = (rows_year / K)[:, None] * meanq
            return np.where(good[:, None] & cov & started, v, np.nan)

        rows_back = (dpos - dpos_back).astype(float)
        d12_back = dwin(dpos_back, dpos, rows_back, MIN_WINDOW_ROWS_12M)
        d3_back = dwin(dpos - RECENT_DAYS, dpos, np.maximum(rows_back, 1.0), RECENT_DAYS)
        rows_fwd = (dpos_fwd - dpos).astype(float)
        d12_fwd = dwin(dpos, dpos_fwd, rows_fwd, MIN_WINDOW_ROWS_12M)
        # a forward window is observed only if the view's daily data reach its end
        last_daily = ddates[-1] if D else pd.Timestamp.min
        fwd_obs_daily = np.asarray(grid_fwd <= last_daily)
        d12_fwd = np.where(fwd_obs_daily[:, None], d12_fwd, np.nan)
        # the 63-day window needs the asset to have traded across the whole 12 months
        d3_back = np.where(np.isfinite(d12_back), d3_back, np.nan)

        self.v12 = np.where(np.isfinite(d12_back), d12_back, m12_back)
        self.v3 = np.where(np.isfinite(d3_back), d3_back, m3_back)
        self.y = np.where(np.isfinite(d12_fwd), d12_fwd, np.where(fwd_complete[:, None], m12_fwd, np.nan))
        self.y_end = grid_fwd                                          # end of each target window
        # fallbacks for thin recent windows: v3 <- v12 <- v_slow
        self.v12 = np.where(np.isfinite(self.v12), self.v12, self.v_slow)
        self.v3 = np.where(np.isfinite(self.v3), self.v3, self.v12)

    def base_matrix(self) -> np.ndarray:
        """(G, A, 3): log v_slow, log v12, log v3 (floored)."""
        f = [np.log(np.maximum(v, VAR_FLOOR)) for v in (self.v_slow, self.v12, self.v3)]
        out = np.stack(f, axis=-1)
        return np.where(np.isfinite(np.stack([self.v_slow, self.v12, self.v3], -1)), out, np.nan)


# --------------------------------------------------------- predictors
def _series_tables(view) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per series: (available_date, obs_date, value) sorted by available date,
    earliest vintage per obs_date (the Panel's convention)."""
    p = view.predictors
    if len(p) == 0:
        return {}
    p = p.drop_duplicates(["series", "obs_date"], keep="first")
    p = p[np.isfinite(p["value"].to_numpy(float))]
    out = {}
    for s, g in p.groupby("series", sort=False):
        g = g.sort_values("available_date", kind="stable")
        out[s] = (g["available_date"].to_numpy("datetime64[ns]"),
                  g["obs_date"].to_numpy("datetime64[ns]"),
                  g["value"].to_numpy(float))
    return out


def asof_monthly(tab, grid: pd.DatetimeIndex, max_age_days=MONTHLY_MAX_AGE_DAYS) -> np.ndarray:
    """Latest published observation at each grid date (NaN if none or stale)."""
    out = np.full(len(grid), np.nan)
    if tab is None:
        return out
    av, ob, val = tab
    g = grid.to_numpy("datetime64[ns]")
    k = np.searchsorted(av, g, side="right")
    # index of the max obs_date among the first n rows
    obi = ob.astype("int64")
    best = np.zeros(len(ob), dtype=int)
    cur = 0
    for i in range(len(ob)):
        if obi[i] >= obi[cur]:
            cur = i
        best[i] = cur
    has = k > 0
    idx = best[np.maximum(k - 1, 0)]
    age = (g - ob[idx]).astype("timedelta64[D]").astype(float)
    fresh = has & (age <= max_age_days)
    out[fresh] = val[idx[fresh]]
    return out


def asof_daily(tab, grid: pd.DatetimeIndex, n_obs=DAILY_AVG_OBS,
               max_age_days=DAILY_MAX_AGE_DAYS) -> np.ndarray:
    """Mean of the last ``n_obs`` published closes at each grid date."""
    out = np.full(len(grid), np.nan)
    if tab is None:
        return out
    av, ob, val = tab
    g = grid.to_numpy("datetime64[ns]")
    k = np.searchsorted(av, g, side="right")
    cv = np.concatenate([[0.0], np.cumsum(val)])
    lo = np.maximum(k - n_obs, 0)
    n = k - lo
    has = n >= min(5, n_obs)
    last_ob = ob[np.maximum(k - 1, 0)]
    age = (g - last_ob).astype("timedelta64[D]").astype(float)
    fresh = has & (age <= max_age_days)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (cv[k] - cv[lo]) / n
    out[fresh] = m[fresh]
    return out


def _log_pos(x):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)


def credit_spread(tabs, grid) -> np.ndarray:
    """Log credit spread: Moody's Baa-Aaa (1919-); where it is stale at the
    LAST grid date only (Moody's ends 2025-12), the Gilchrist-Zakrajsek
    spread mapped to the Baa-Aaa scale by a log-log regression on the months
    both are published in the view (needs >= 60). History is never imputed."""
    baa = asof_monthly(tabs.get("credit_moodys_baa_aaa_spread_m"), grid)
    out = _log_pos(baa)
    if np.isfinite(out[-1]):
        return out
    gz_tab = tabs.get("credit_gz_spread_m")
    b_tab = tabs.get("credit_moodys_baa_aaa_spread_m")
    if gz_tab is None or b_tab is None:
        return out
    gz_now = asof_monthly(gz_tab, grid[-1:])[0]
    if not (np.isfinite(gz_now) and gz_now > 0):
        return out
    gz = pd.Series(gz_tab[2], index=gz_tab[1])
    bb = pd.Series(b_tab[2], index=b_tab[1])
    both = pd.concat([gz, bb], axis=1, join="inner").dropna()
    both = both[(both > 0).all(axis=1)]
    if len(both) < 60:
        return out
    lx, ly = np.log(both.iloc[:, 0].to_numpy()), np.log(both.iloc[:, 1].to_numpy())
    b = np.cov(lx, ly)[0, 1] / np.var(lx, ddof=1)
    a = ly.mean() - b * lx.mean()
    out = out.copy()
    out[-1] = a + b * np.log(gz_now)
    return out


DURATION_SERIES = {"ust_int": "ust_ladder_duration_int", "tips": "ust_ladder_duration_int",
                   "ust_long": "ust_ladder_duration_long"}
OWN_IMPLIED = {"gold": "iv_gvz", "commodities": "iv_ovx"}


def state_features(view, grid: pd.DatetimeIndex, assets=ASSET_IDS) -> dict[str, np.ndarray]:
    """name -> (G, A) array of state variables as known at each grid date."""
    tabs = _series_tables(view)
    G, A = len(grid), len(assets)

    def common(x):
        return np.repeat(np.asarray(x, float)[:, None], A, axis=1)

    m = {s: asof_monthly(tabs.get(s), grid) for s in [
        "macro_term_spread", "macro_infl_vol_36m", "macro_infl_yoy", "macro_long_yield",
        "macro_sahm_gap", "macro_policy_rate_chg_12m", "macro_real_short_rate",
        "macro_usd_broad_yoy", "macro_ip_yoy"]}
    out = {
        "cs": common(credit_spread(tabs, grid)),
        "ts": common(m["macro_term_spread"]),
        "ivol": common(_log_pos(m["macro_infl_vol_36m"])),
        "infl": common(m["macro_infl_yoy"]),
        "ly": common(m["macro_long_yield"]),
        "sahm": common(m["macro_sahm_gap"]),
        "dpol": common(np.abs(m["macro_policy_rate_chg_12m"])),
        "rsr": common(m["macro_real_short_rate"]),
        "usd": common(np.abs(m["macro_usd_broad_yoy"])),
        "ip": common(m["macro_ip_yoy"]),
        # implied: log implied VARIANCE (2 log vol) of a ~1-month average
        "vix": common(2.0 * _log_pos(asof_daily(tabs.get("iv_vix_vxo_spliced"), grid))),
    }
    move = asof_daily(tabs.get("iv_move"), grid)
    md = np.full((G, A), np.nan)
    own = np.full((G, A), np.nan)
    for j, a in enumerate(assets):
        if a in DURATION_SERIES:
            dur = asof_daily(tabs.get(DURATION_SERIES[a]), grid)
            md[:, j] = 2.0 * _log_pos(move * dur)          # yield vol x duration = price vol
        if a in OWN_IMPLIED:
            own[:, j] = 2.0 * _log_pos(asof_daily(tabs.get(OWN_IMPLIED[a]), grid))
    out["move_d"] = md
    out["iv_own"] = own
    return out
