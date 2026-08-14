"""Return and risk-adjusted performance statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .drawdown import depth, max_drawdown, ulcer_index, underwater_fraction

TRADING_DAYS = 252
DAYS_PER_YEAR = 365.25


def _years(returns: pd.Series) -> float:
    if isinstance(returns.index, pd.DatetimeIndex) and len(returns) > 1:
        return (returns.index[-1] - returns.index[0]).days / DAYS_PER_YEAR
    return len(returns) / TRADING_DAYS


def total_return(returns: pd.Series) -> float:
    return float(np.prod(1.0 + np.asarray(returns, dtype=float)) - 1.0)


def cagr(returns: pd.Series) -> float:
    years = _years(returns)
    if years <= 0:
        return 0.0
    growth = float(np.prod(1.0 + np.asarray(returns, dtype=float)))
    if growth <= 0:
        return -1.0
    return growth ** (1.0 / years) - 1.0


def ann_vol(returns: pd.Series) -> float:
    return float(np.std(np.asarray(returns, dtype=float), ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf_daily: pd.Series | float = 0.0) -> float:
    excess = returns - rf_daily if not np.isscalar(rf_daily) else returns - rf_daily
    sd = float(np.std(excess, ddof=1))
    if sd == 0:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf_daily: pd.Series | float = 0.0) -> float:
    excess = returns - rf_daily
    downside = np.minimum(np.asarray(excess, dtype=float), 0.0)
    dd = float(np.sqrt(np.mean(downside**2)))
    if dd == 0:
        return 0.0
    return float(np.mean(excess) / dd * np.sqrt(TRADING_DAYS))


def calmar(returns: pd.Series) -> float:
    d = depth(returns)
    return cagr(returns) / d if d > 1e-9 else np.inf


def worst_rolling(returns: pd.Series, window: int = TRADING_DAYS) -> float:
    """Worst compounded return over any rolling window of ``window`` days."""
    if len(returns) < window:
        return total_return(returns)
    logs = np.log1p(np.asarray(returns, dtype=float))
    csum = np.concatenate([[0.0], np.cumsum(logs)])
    rolls = csum[window:] - csum[:-window]
    return float(np.expm1(rolls.min()))


def var_cvar(returns: pd.Series, q: float = 0.05) -> tuple[float, float]:
    """Historical daily VaR and CVaR at quantile ``q`` (both negative)."""
    r = np.asarray(returns, dtype=float)
    var = float(np.quantile(r, q))
    tail = r[r <= var]
    return var, float(tail.mean()) if tail.size else var


def capture_ratios(returns: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    """Up-capture and down-capture versus a benchmark, on shared dates."""
    joined = pd.concat([returns.rename("s"), benchmark.rename("b")], axis=1, join="inner").dropna()
    up, down = joined[joined["b"] > 0], joined[joined["b"] < 0]
    up_cap = float(up["s"].mean() / up["b"].mean()) if len(up) else np.nan
    down_cap = float(down["s"].mean() / down["b"].mean()) if len(down) else np.nan
    return up_cap, down_cap


def beta_alpha(returns: pd.Series, benchmark: pd.Series, rf_daily: pd.Series | float = 0.0):
    """CAPM beta and annualised alpha versus a benchmark."""
    joined = pd.concat([returns.rename("s"), benchmark.rename("b")], axis=1, join="inner").dropna()
    if isinstance(rf_daily, pd.Series):
        rf = rf_daily.reindex(joined.index).ffill().fillna(0.0)
    else:
        rf = pd.Series(rf_daily, index=joined.index)
    xs, xb = joined["s"] - rf, joined["b"] - rf
    var_b = float(np.var(xb, ddof=1))
    if var_b == 0:
        return np.nan, np.nan
    beta = float(np.cov(xs, xb, ddof=1)[0, 1] / var_b)
    alpha = float((xs.mean() - beta * xb.mean()) * TRADING_DAYS)
    return beta, alpha


def summary(
    returns: pd.Series,
    *,
    benchmark: pd.Series | None = None,
    rf_daily: pd.Series | float = 0.0,
) -> dict:
    """Full statistics block for one return stream."""
    rf = rf_daily.reindex(returns.index).ffill().fillna(0.0) if isinstance(rf_daily, pd.Series) else rf_daily
    var5, cvar5 = var_cvar(returns, 0.05)
    out = {
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "years": round(_years(returns), 1),
        "cagr": cagr(returns),
        "vol": ann_vol(returns),
        "sharpe": sharpe(returns, rf),
        "sortino": sortino(returns, rf),
        "max_dd": max_drawdown(returns),
        "calmar": calmar(returns),
        "ulcer": ulcer_index(returns),
        "underwater_frac": underwater_fraction(returns),
        "worst_12m": worst_rolling(returns, TRADING_DAYS),
        "worst_day": float(np.min(returns)),
        "var_95": var5,
        "cvar_95": cvar5,
        "skew": float(pd.Series(returns).skew()),
        "kurtosis": float(pd.Series(returns).kurtosis()),
    }
    if benchmark is not None:
        up, down = capture_ratios(returns, benchmark)
        beta, alpha = beta_alpha(returns, benchmark, rf)
        out |= {"up_capture": up, "down_capture": down, "beta": beta, "alpha": alpha}
    return out
