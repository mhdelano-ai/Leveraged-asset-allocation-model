"""Daily backtest loop.

No-lookahead is enforced *structurally*, not by convention. The allocator never
sees the return matrix: it is handed a causal ``base_weights`` composition and a
causal ``target_leverage`` series, both computed beforehand from rolling
right-aligned estimators, and the loop applies row ``t`` of those to the returns
of day ``t + execution_lag``. Path-dependent state -- realised drawdown, the
leverage ratchet, margin calls -- is the only thing computed inside the loop, and
it depends solely on equity already realised.

``execution_lag_days`` defaults to 1: a signal formed at the close of day t-1 is
traded into day t's return. This is what makes 1987 honest -- a signal computed
on Friday 1987-10-16 eats the full -20.5% of Black Monday at whatever leverage
was on. Any design that dodges that by trading intraday on the 19th is fiction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..timeaxis import day_deltas
from .accounting import apply_rebalance, mark_to_market, needs_rebalance


@dataclass
class EngineConfig:
    """Cost and rebalancing assumptions."""

    # One-way transaction cost per unit turnover, per asset (decimals).
    cost_per_unit: np.ndarray | None = None
    default_cost: float = 0.0005
    # Backtesting 2020s spreads through the 1970s is fantasy; scale by era.
    cost_era_multipliers: tuple[tuple[str, float], ...] = (
        ("1990-01-01", 4.0),
        ("2000-01-01", 2.0),
        ("2100-01-01", 1.0),
    )
    borrow_spread: float = 0.0050
    lend_spread: float = -0.0010
    execution_lag_days: int = 1
    rebalance_freq: str = "M"  # M, W, Q or None
    abs_band: float = 0.010
    rel_band: float = 0.25
    lev_band: float = 0.25
    risk_off_threshold: float = 0.10
    max_leverage: float = 3.0
    leverage_ratchet_up: float = 0.10  # max daily increase in leverage


@dataclass
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    leverage: pd.Series
    weights: pd.DataFrame
    # The standing order the book is being held against. Weights drift between
    # rebalances, so `weights` is what is held and this is what it is aiming at;
    # reporting the former as "the allocation" would hand back a number that is
    # not the number to trade to.
    target_weights: pd.DataFrame
    turnover: pd.Series
    financing_cost: pd.Series
    transaction_cost: pd.Series
    throttle: pd.Series
    events: list[dict] = field(default_factory=list)

    @property
    def ruined(self) -> bool:
        return bool((self.equity <= 1e-12).any())

    def summary_costs(self) -> dict:
        years = (self.returns.index[-1] - self.returns.index[0]).days / 365.25
        return {
            "financing_drag_bps": -self.financing_cost.sum() / years * 1e4,
            "transaction_drag_bps": self.transaction_cost.sum() / years * 1e4,
            "avg_leverage": float(self.leverage.mean()),
            "frac_levered": float((self.leverage > 1.0).mean()),
            "ann_turnover": float(self.turnover.sum() / years),
        }


def _era_costs(index: pd.DatetimeIndex, config: EngineConfig, n_assets: int) -> np.ndarray:
    base = (
        np.full(n_assets, config.default_cost)
        if config.cost_per_unit is None
        else np.asarray(config.cost_per_unit, dtype=float)
    )
    mult = np.ones(len(index))
    for cutoff, factor in config.cost_era_multipliers:
        mask = index < pd.Timestamp(cutoff)
        mult = np.where(mask & (mult == 1.0), factor, mult)
    # Later cutoffs must not overwrite earlier (stricter) ones.
    mult = np.ones(len(index))
    bounds = [(pd.Timestamp(c), f) for c, f in config.cost_era_multipliers]
    prev = pd.Timestamp("1800-01-01")
    for cutoff, factor in bounds:
        mult[(index >= prev) & (index < cutoff)] = factor
        prev = cutoff
    return np.outer(mult, base)


def _calendar_triggers(index: pd.DatetimeIndex, freq: str | None) -> np.ndarray:
    if not freq:
        return np.zeros(len(index), dtype=bool)
    series = pd.Series(np.arange(len(index)), index=index)
    keep = series.groupby(index.to_period(freq)).max().to_numpy()
    flags = np.zeros(len(index), dtype=bool)
    flags[keep] = True
    return flags


def run(
    asset_returns: pd.DataFrame,
    base_weights: pd.DataFrame,
    target_leverage: pd.Series,
    *,
    rf_annual: pd.Series,
    config: EngineConfig | None = None,
    throttle=None,
    vehicle=None,
    vehicle_vix: pd.Series | None = None,
    initial_equity: float = 1.0,
) -> BacktestResult:
    """Run the daily loop.

    Parameters
    ----------
    asset_returns:
        Daily total returns per sleeve. Defines the date index.
    base_weights:
        Causal unlevered composition, rows summing to ~1.
    target_leverage:
        Causal gross leverage target before the drawdown throttle.
    rf_annual:
        Financing base rate as an annual decimal.
    throttle:
        Object with ``__call__(drawdown_depth) -> float`` in [0, 1]. Sees only
        realised equity.
    vehicle:
        Optional financing vehicle supplying borrow rates, leverage caps,
        achievable weights and margin calls.
    """
    config = config or EngineConfig()

    frame = asset_returns.dropna(how="all").copy()
    if vehicle is not None:
        from ..financing.base import VehicleContext

        frame = vehicle.prepare(
            VehicleContext(
                returns=frame,
                rf=rf_annual.reindex(frame.index).ffill(),
                vix=vehicle_vix,
                sleeve_names=tuple(frame.columns),
            )
        )
    weights_frame = base_weights.reindex(frame.index).ffill()
    lev = target_leverage.reindex(frame.index).ffill()

    common = frame.index
    rets = frame.to_numpy(dtype=float)
    rets = np.nan_to_num(rets)
    base = np.nan_to_num(weights_frame.to_numpy(dtype=float))
    lev_arr = np.nan_to_num(lev.to_numpy(dtype=float))
    rf = np.nan_to_num(rf_annual.reindex(common).ffill().to_numpy(dtype=float))

    n_days, n_assets = rets.shape
    costs = _era_costs(common, config, n_assets)
    cal = _calendar_triggers(common, config.rebalance_freq)
    day_counts = day_deltas(common, prepend=1.0)

    lag = max(1, int(config.execution_lag_days))
    max_lev = min(config.max_leverage, vehicle.max_leverage if vehicle else np.inf)

    equity = np.full(n_days, np.nan)
    port_ret = np.zeros(n_days)
    lev_path = np.zeros(n_days)
    phi_path = np.ones(n_days)
    turn_path = np.zeros(n_days)
    fin_path = np.zeros(n_days)
    tc_path = np.zeros(n_days)
    weight_path = np.zeros((n_days, n_assets))
    target_path = np.zeros((n_days, n_assets))
    events: list[dict] = []

    E = initial_equity
    w = np.zeros(n_assets)
    peak = E
    prev_desired = 0.0
    prev_phi = 1.0
    current_target: np.ndarray | None = None

    for t in range(n_days):
        if t >= lag:
            borrow = rf[t] + (vehicle.borrow_spread(t) if vehicle else config.borrow_spread)
            lend = rf[t] + config.lend_spread
            E_new, w_drift, _, fin = mark_to_market(
                E, w, rets[t],
                borrow_rate=borrow, lend_rate=lend, day_count=day_counts[t],
            )
            port_ret[t] = E_new / E - 1.0 if E > 0 else -1.0
            fin_path[t] = fin
            E = E_new
            if E <= 1e-12:
                events.append({"date": common[t], "type": "ruin"})
                equity[t:] = 0.0
                break
        else:
            w_drift = w

        peak = max(peak, E)
        dd_depth = max(0.0, 1.0 - E / peak) if peak > 0 else 0.0
        phi = float(throttle(dd_depth)) if throttle is not None else 1.0

        # Signal row t is applied to day t+lag's returns.
        desired = min(lev_arr[t] * phi, max_lev)
        # Leverage falls freely but rises slowly: stops the book re-levering into
        # bear-market rallies, which is where a large share of realised drawdown
        # in naive vol-targeted strategies actually comes from.
        #
        # The ratchet is anchored to the previous *desired* leverage, which
        # advances every day, not to the leverage currently held. Anchoring it to
        # the held book deadlocks against the rebalance band: the cap keeps the
        # candidate within `ratchet` of what is held, the band only trades once
        # the gap exceeds `lev_band`, and with ratchet < lev_band the gap never
        # gets there. Leverage could then only rise on calendar triggers, one
        # ratchet step per month -- 1.2x a year against a stated limit of 25x.
        desired = min(desired, prev_desired + config.leverage_ratchet_up)
        prev_desired = desired

        comp = base[t]
        comp_sum = comp.sum()
        candidate = desired * comp / comp_sum if comp_sum > 1e-12 else np.zeros(n_assets)
        if vehicle is not None:
            candidate = vehicle.achievable_weights(candidate, t)

        # The *standing order* only refreshes on a rebalance event. Signals move
        # a little every day, so comparing the book against a target recomputed
        # daily makes it chase a moving goalpost -- that alone drove 15-30x
        # annual turnover and ~90-200bp of costs. A real mandate re-solves the
        # target on its rebalance schedule and holds it in between.
        if current_target is None:
            current_target = candidate

        fired = needs_rebalance(
            w_drift, current_target,
            calendar_trigger=bool(cal[t]),
            abs_band=config.abs_band,
            rel_band=config.rel_band,
            lev_band=config.lev_band,
            risk_off_move=abs(phi - prev_phi),
            risk_off_threshold=config.risk_off_threshold,
        )
        # A materially different candidate is itself a reason to trade, even
        # mid-period: this is how a risk-off signal reaches the book promptly.
        lev_gap = abs(float(candidate.sum()) - float(current_target.sum()))
        if lev_gap > config.lev_band:
            fired = True

        if fired and E > 0:
            current_target = candidate
            E, w, tc, turn = apply_rebalance(E, w_drift, current_target, costs[t])
            tc_path[t], turn_path[t] = tc, turn
            port_ret[t] -= tc
        else:
            w = w_drift

        if vehicle is not None and E > 0:
            action = vehicle.check_margin(E, w, t)
            if action is not None:
                scale, slippage, note = action
                w = w * scale
                E *= 1.0 - slippage
                port_ret[t] -= slippage
                events.append(
                    {"date": common[t], "type": "margin_call", "scale": scale, "note": note}
                )

        equity[t] = E
        # Report *economic* exposure. For LETFs the capital weights sum to <= 1
        # while the notional exposure is up to 3x, so summing weights would
        # understate the risk being run.
        lev_path[t] = (
            vehicle.notional_leverage(w) if vehicle is not None else float(w.sum())
        )
        phi_path[t] = phi
        weight_path[t] = w
        target_path[t] = current_target if current_target is not None else w
        prev_phi = phi

    equity = pd.Series(equity, index=common, name="equity").ffill()
    return BacktestResult(
        returns=pd.Series(port_ret, index=common, name="strategy"),
        equity=equity,
        leverage=pd.Series(lev_path, index=common, name="leverage"),
        weights=pd.DataFrame(weight_path, index=common, columns=frame.columns),
        target_weights=pd.DataFrame(target_path, index=common, columns=frame.columns),
        turnover=pd.Series(turn_path, index=common, name="turnover"),
        financing_cost=pd.Series(fin_path, index=common, name="financing"),
        transaction_cost=pd.Series(tc_path, index=common, name="tcost"),
        throttle=pd.Series(phi_path, index=common, name="throttle"),
        events=events,
    )
