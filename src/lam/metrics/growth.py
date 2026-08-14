"""Growth-rate mathematics for a concentrated levered book.

When the objective is "maximise return" rather than "beat a benchmark under a
constraint", the governing equation is not the Sharpe ratio. It is the geometric
growth rate, and for a book holding leverage ``L`` in one asset it is

    g(L) = L*mu_excess + r_f - (1/2)*L^2*sigma^2 - (L-1)*spread - costs

The quadratic term is what makes this problem interesting. Growth rises linearly
in leverage and falls quadratically, so there is an interior optimum

    L* = mu_excess / sigma^2

-- the Kelly criterion. Above ``L*`` more leverage *lowers* compound return while
still raising volatility, which is the trap every 3x ETF holder walks into. The
functions here measure ``L*`` three ways, because they disagree and the
disagreement is the point:

``kelly_analytic``
    From sample moments. Assumes returns are roughly lognormal, which they are
    not, so it is biased high.
``optimal_static_leverage``
    Argmax of the *realised* compound return over a leverage grid, financing and
    trading costs included. This is the honest number.
``regime_kelly``
    ``L*`` estimated separately inside volatility and trend states. The spread
    between states is the entire economic case for a dynamic leverage rule: if
    ``L*`` were constant there would be nothing for trend following or volatility
    targeting to do.

Every estimate here is a full-sample, in-sample quantity computed with hindsight.
They are diagnostics that say what was achievable, never signals -- nothing in
``lam.alloc`` may read them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeaxis import day_deltas
from .core import ann_vol, cagr, sharpe
from .drawdown import max_drawdown

TRADING_DAYS = 252
FINANCING_BASIS = 360.0


def constant_leverage_returns(
    asset: pd.Series,
    rf_annual: pd.Series,
    leverage: float,
    *,
    borrow_spread: float = 0.0050,
    lend_spread: float = -0.0010,
    cost_per_unit: float = 0.0002,
) -> pd.Series:
    """Daily-reset constant-leverage returns, financed at ACT/360.

    Daily reset is what a leveraged ETF does mechanically and what a margin book
    with a tight leverage band approximates. It is also the conservative choice:
    resetting daily locks in the volatility drag, whereas letting leverage drift
    lets a winning position run.

    The reset is not free. After a move of ``r`` the book's exposure has drifted
    to ``L(1+r)`` against a target of ``L(1+r_p)``, so it must trade
    ``L|L-1| |r - carry|`` of equity every day -- zero at ``L = 1``, and rising
    with the square of leverage. At 3x on the Nasdaq-100 that is roughly 8x
    annual turnover paid purely to stay level.
    """
    net, _ = _levered_path(
        asset, rf_annual, leverage,
        borrow_spread=borrow_spread, lend_spread=lend_spread,
        cost_per_unit=cost_per_unit,
    )
    return pd.Series(net, index=asset.index, name=f"L{leverage:g}")


def _levered_path(
    asset: pd.Series,
    rf_annual: pd.Series,
    leverage: float,
    *,
    borrow_spread: float,
    lend_spread: float,
    cost_per_unit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Net daily returns and daily turnover for a daily-reset levered book."""
    rf = rf_annual.reindex(asset.index).ffill().fillna(0.0).to_numpy(dtype=float)
    r = asset.to_numpy(dtype=float)
    days = day_deltas(asset.index, prepend=1.0)

    spread = borrow_spread if leverage > 1.0 else lend_spread
    carry = (rf + spread) * days / FINANCING_BASIS

    gross = leverage * r + (1.0 - leverage) * carry
    turnover = abs(leverage * (leverage - 1.0)) * np.abs(r - carry)
    net = gross - turnover * cost_per_unit

    # Once equity is gone it stays gone. Without this the series "recovers" on
    # negative capital, which is how naive high-leverage backtests report a
    # finite CAGR for a book that was wiped out in 1987.
    wiped = np.flatnonzero(net <= -1.0)
    if wiped.size:
        first = int(wiped[0])
        net[first] = -1.0
        net[first + 1 :] = 0.0
        turnover[first + 1 :] = 0.0
    return net, turnover


def leverage_curve(
    asset: pd.Series,
    rf_annual: pd.Series,
    *,
    levels: np.ndarray | None = None,
    borrow_spread: float = 0.0050,
    lend_spread: float = -0.0010,
    cost_per_unit: float = 0.0002,
) -> pd.DataFrame:
    """Realised statistics of constant leverage across a grid of ``L``.

    This single table answers the question the strategy exists to improve on:
    *how far does brute-force leverage get you, and where does it stop working?*
    """
    if levels is None:
        levels = np.round(np.arange(0.5, 5.01, 0.25), 4)

    rf_daily = rf_annual.reindex(asset.index).ffill().fillna(0.0) / 252.0
    rows = []
    for lev in levels:
        r = constant_leverage_returns(
            asset, rf_annual, float(lev),
            borrow_spread=borrow_spread, lend_spread=lend_spread,
            cost_per_unit=cost_per_unit,
        )
        ruined = bool((r <= -1.0 + 1e-12).any())
        rows.append(
            {
                "leverage": float(lev),
                "cagr": cagr(r),
                "vol": ann_vol(r),
                "sharpe": sharpe(r, rf_daily),
                "max_dd": max_drawdown(r),
                "terminal_wealth": float(np.prod(1.0 + r.to_numpy())),
                "worst_day": float(r.min()),
                "ruined": ruined,
            }
        )
    return pd.DataFrame(rows).set_index("leverage")


def optimal_static_leverage(
    asset: pd.Series,
    rf_annual: pd.Series,
    *,
    search: np.ndarray | None = None,
    **kwargs,
) -> tuple[float, float]:
    """Leverage maximising realised compound return, and that return.

    Searched on a fine grid rather than solved: the objective is the realised
    path, which has no closed form once fat tails, financing and turnover costs
    are in it.
    """
    if search is None:
        search = np.round(np.arange(0.1, 5.01, 0.05), 4)
    curve = leverage_curve(asset, rf_annual, levels=search, **kwargs)
    best = curve["cagr"].idxmax()
    return float(best), float(curve.loc[best, "cagr"])


def vol_matched_static(
    strategy: pd.Series,
    asset: pd.Series,
    rf_annual: pd.Series,
    *,
    borrow_spread: float = 0.0050,
    lend_spread: float = -0.0010,
    cost_per_unit: float = 0.0002,
    tolerance: float = 0.0005,
) -> dict:
    """The decisive comparison: constant leverage carrying the *same* volatility.

    A dynamic strategy that runs at higher average leverage than the index will
    naturally show a higher compound return, and reporting that as evidence the
    timing works is the most common error in this literature. The honest question
    is whether the strategy beats a dumb constant-leverage book *sized to the same
    realised risk*. Whatever is left over is what the trend gate, the volatility
    target and the throttle actually earned.

    Leverage is solved by bisection on realised volatility, which is monotone in
    ``L``, so the match is exact rather than approximate.
    """
    joined = pd.concat(
        [strategy.rename("s"), asset.rename("a")], axis=1, join="inner"
    ).dropna()
    target_vol = ann_vol(joined["s"])

    lo, hi = 0.0, 12.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        candidate = constant_leverage_returns(
            joined["a"], rf_annual, mid,
            borrow_spread=borrow_spread, lend_spread=lend_spread,
            cost_per_unit=cost_per_unit,
        )
        if ann_vol(candidate) < target_vol:
            lo = mid
        else:
            hi = mid
        if abs(ann_vol(candidate) - target_vol) < tolerance:
            break

    matched = constant_leverage_returns(
        joined["a"], rf_annual, mid,
        borrow_spread=borrow_spread, lend_spread=lend_spread,
        cost_per_unit=cost_per_unit,
    )
    return {
        "matched_leverage": float(mid),
        "target_vol": float(target_vol),
        "matched_vol": ann_vol(matched),
        "strategy_cagr": cagr(joined["s"]),
        "matched_cagr": cagr(matched),
        "edge_bps": (cagr(joined["s"]) - cagr(matched)) * 1e4,
        "strategy_dd": max_drawdown(joined["s"]),
        "matched_dd": max_drawdown(matched),
        "dd_saved_pp": (max_drawdown(joined["s"]) - max_drawdown(matched)) * 100,
        "returns": matched,
    }


def kelly_analytic(asset: pd.Series, rf_annual: pd.Series, *, spread: float = 0.0050) -> float:
    """``L* = mu_excess / sigma^2`` from sample moments.

    Biased high for real equity returns. The derivation assumes the loss from
    leverage is entirely the variance term, but a fat left tail costs more than
    its variance contribution implies, and the daily reset makes that cost
    path-dependent. Compare against :func:`optimal_static_leverage`.
    """
    rf = rf_annual.reindex(asset.index).ffill().fillna(0.0)
    days = day_deltas(asset.index, prepend=1.0)
    carry = (rf.to_numpy() + spread) * days / FINANCING_BASIS
    excess = asset.to_numpy(dtype=float) - carry

    mu = float(np.mean(excess)) * TRADING_DAYS
    var = float(np.var(excess, ddof=1)) * TRADING_DAYS
    return mu / var if var > 0 else np.nan


def kelly_bootstrap(
    asset: pd.Series,
    rf_annual: pd.Series,
    *,
    n: int = 2000,
    block: int = 21,
    spread: float = 0.0050,
    seed: int = 0,
) -> dict:
    """Sampling distribution of the Kelly optimum under a moving-block bootstrap.

    The width of this interval is the single strongest argument against betting
    full Kelly. ``L*`` is a ratio of two noisily estimated moments, and the
    numerator -- the equity risk premium -- is the hardest number in finance to
    estimate. Blocks preserve volatility clustering, which an i.i.d. bootstrap
    destroys and which is exactly what makes the tail expensive.
    """
    rf = rf_annual.reindex(asset.index).ffill().fillna(0.0)
    days = day_deltas(asset.index, prepend=1.0)
    carry = (rf.to_numpy() + spread) * days / FINANCING_BASIS
    excess = asset.to_numpy(dtype=float) - carry

    n_obs = len(excess)
    n_blocks = int(np.ceil(n_obs / block))
    rng = np.random.default_rng(seed)

    draws = np.empty(n)
    starts = rng.integers(0, max(1, n_obs - block), size=(n, n_blocks))
    offsets = np.arange(block)
    for i in range(n):
        idx = (starts[i][:, None] + offsets).ravel()[:n_obs]
        sample = excess[np.minimum(idx, n_obs - 1)]
        var = float(np.var(sample, ddof=1))
        draws[i] = float(np.mean(sample)) / var if var > 0 else np.nan

    draws = draws[np.isfinite(draws)]
    return {
        "point": kelly_analytic(asset, rf_annual, spread=spread),
        "mean": float(draws.mean()),
        "p05": float(np.quantile(draws, 0.05)),
        "p50": float(np.quantile(draws, 0.50)),
        "p95": float(np.quantile(draws, 0.95)),
        "frac_below_1": float((draws < 1.0).mean()),
        "n": int(len(draws)),
    }


def growth_decomposition(
    asset: pd.Series,
    rf_annual: pd.Series,
    leverage: float,
    *,
    borrow_spread: float = 0.0050,
    cost_per_unit: float = 0.0002,
) -> dict:
    """Break annualised log growth at ``leverage`` into its four sources.

    Reported in basis points per year so the terms can be read against each
    other::

        g = L*mu_arith + (1-L)*(rf + spread) - (1/2)*L^2*sigma^2 - trading

    The carry term carries its own sign: positive below 1x, where the book lends
    out the cash it is not using, and negative above it. Netting it as a separate
    "cash" income line *and* a "financing" cost line double-counts the cash --
    at 2x that error is worth the entire risk-free rate.

    Annualisation uses the **observed** number of observations per year rather
    than a hardcoded 252. A synthetic business-day index runs 261 days a year,
    and forcing 252 onto it understates every term by 3.6% -- enough to look like
    a modelling error in the residual when it is only a calendar convention.

    The residual is what the lognormal approximation genuinely misses: skew,
    kurtosis, and the path dependence of the daily reset.
    """
    rf = rf_annual.reindex(asset.index).ffill().fillna(0.0)
    days = day_deltas(asset.index, prepend=1.0)
    r = asset.to_numpy(dtype=float)
    years = (asset.index[-1] - asset.index[0]).days / 365.25
    per_year = len(asset) / years

    mu_arith = float(np.mean(r)) * per_year
    var = float(np.var(r, ddof=1)) * per_year
    spread = borrow_spread if leverage > 1.0 else 0.0
    rate = float(np.sum((rf.to_numpy() + spread) * days / FINANCING_BASIS)) / years

    net, turnover_path = _levered_path(
        asset, rf_annual, leverage,
        borrow_spread=borrow_spread, lend_spread=-0.0010,
        cost_per_unit=cost_per_unit,
    )

    drift = leverage * mu_arith
    carry = (1.0 - leverage) * rate
    variance_drag = 0.5 * leverage**2 * var
    trading = float(np.sum(turnover_path)) / years * cost_per_unit
    predicted = drift + carry - variance_drag - trading
    actual = float(np.log1p(np.maximum(net, -0.999999)).sum()) / years

    return {
        "leverage": leverage,
        "drift_bps": drift * 1e4,
        "carry_bps": carry * 1e4,
        "variance_drag_bps": -variance_drag * 1e4,
        "trading_cost_bps": -trading * 1e4,
        "predicted_log_growth_bps": predicted * 1e4,
        "actual_log_growth_bps": actual * 1e4,
        "residual_bps": (actual - predicted) * 1e4,
        "realised_cagr": cagr(pd.Series(net, index=asset.index)),
    }


def regime_kelly(
    asset: pd.Series,
    rf_annual: pd.Series,
    *,
    vol_halflife: float = 40.0,
    trend_window: int = 200,
    spread: float = 0.0050,
) -> pd.DataFrame:
    """Kelly optimum conditional on the previous day's volatility and trend state.

    This is the empirical justification for the whole dynamic-leverage apparatus.
    Static leverage implicitly assumes one ``L*`` for all time. If ``L*`` varies
    materially with observable, *lagged* state, then a rule that levers up in the
    favourable state and down in the unfavourable one earns more compound return
    than any constant can -- and the size of the gap bounds how much such a rule
    can possibly be worth.

    State is measured strictly on data available the previous close, so the
    buckets are formed from information a trader had. Bucket boundaries, however,
    use the full-sample volatility distribution: this is a diagnostic, not a
    signal, and it is not tradable as written.
    """
    rf = rf_annual.reindex(asset.index).ffill().fillna(0.0)
    days = day_deltas(asset.index, prepend=1.0)
    carry = pd.Series((rf.to_numpy() + spread) * days / FINANCING_BASIS, index=asset.index)
    excess = asset - carry

    vol = asset.ewm(halflife=vol_halflife, min_periods=60).std() * np.sqrt(TRADING_DAYS)
    price = (1.0 + asset.fillna(0.0)).cumprod()
    above = price > price.rolling(trend_window, min_periods=trend_window // 2).mean()

    lagged_vol = vol.shift(1)
    up = above.shift(1).fillna(False).astype(bool).reindex(asset.index).fillna(False)
    down = ~up
    quintile = pd.qcut(lagged_vol.dropna(), 5, labels=[1, 2, 3, 4, 5]).reindex(asset.index)

    rows = []
    for label, mask in [
        ("vol Q1 (calmest)", quintile == 1),
        ("vol Q2", quintile == 2),
        ("vol Q3", quintile == 3),
        ("vol Q4", quintile == 4),
        ("vol Q5 (wildest)", quintile == 5),
        ("above 200d MA", up),
        ("below 200d MA", down),
        ("calm + uptrend", quintile.isin([1, 2]) & up),
        ("wild + downtrend", quintile.isin([4, 5]) & down),
    ]:
        m = mask.reindex(asset.index).fillna(False)
        sub = excess[m]
        if len(sub) < 100:
            continue
        mu = float(sub.mean()) * TRADING_DAYS
        var = float(sub.var(ddof=1)) * TRADING_DAYS
        rows.append(
            {
                "state": label,
                "days": int(len(sub)),
                "share": len(sub) / len(asset),
                "excess_cagr": mu,
                "vol": float(np.sqrt(var)),
                "kelly": mu / var if var > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("state")
