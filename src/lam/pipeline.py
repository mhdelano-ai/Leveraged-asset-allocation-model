"""One place that turns (panel, parameters, vehicle) into a backtest result.

Both the CLI scripts and the optimiser call this, so a parameter set always means
the same thing everywhere. The optimiser evaluates thousands of candidates, so
the expensive causal signals are cached per panel: they depend only on the panel
and a couple of parameters, not on the leverage or throttle settings that the
search actually varies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .alloc.dd_throttle import DrawdownThrottle
from .alloc.stack import StackParams, build as build_stack
from .data import yahoo
from .data.panels import Panel
from .engine.backtest import BacktestResult, EngineConfig, run
from .metrics.constraint import ConstraintResult, design_budget, evaluate
from .metrics.core import summary


@dataclass
class RunOutput:
    result: BacktestResult
    constraint: ConstraintResult
    stats: dict
    params: StackParams
    budget: float

    def headline(self) -> str:
        s = self.stats
        return (
            f"CAGR {s['cagr']:6.2%} | vol {s['vol']:5.1%} | Sharpe {s['sharpe']:4.2f} | "
            f"maxDD {s['max_dd']:7.2%} | Calmar {s['calmar']:4.2f} | "
            f"{'PASS' if self.constraint.passed else 'FAIL'}"
        )


_signal_cache: dict[tuple, object] = {}


def _vix_series() -> pd.Series | None:
    try:
        return yahoo.prices("^VIX", field="close")
    except Exception:  # noqa: BLE001 - VIX is optional conditioning
        return None


def stack_for(
    panel: Panel,
    params: StackParams,
    *,
    use_trend: bool = True,
    use_crisis_cov: bool = True,
):
    """Causal signals for a panel, memoised on the parameters that affect them.

    The cache key deliberately excludes ``sigma_target`` and ``max_leverage``:
    those only rescale the finished volatility forecast, so a cached stack is
    reused and rescaled. Building the covariance path dominates runtime, and the
    optimiser varies those two more than anything else.
    """
    key = (
        panel.name,
        round(params.vol_halflife, 4),
        round(params.trend_fast_weight, 4),
        round(params.growth_budget, 4),
        use_trend,
        use_crisis_cov,
    )
    if key not in _signal_cache:
        _signal_cache[key] = build_stack(
            panel.returns,
            panel.rf,
            params=params,
            eligible=panel.eligible,
            stress_reference=panel.benchmark,
            use_trend=use_trend,
            use_crisis_cov=use_crisis_cov,
        )
    return _signal_cache[key].rescale(params.sigma_target, params.max_leverage)


def clear_signal_cache() -> None:
    _signal_cache.clear()


def engine_config(panel: Panel, params: StackParams, **overrides) -> EngineConfig:
    cfg = EngineConfig(
        cost_per_unit=panel.costs,
        max_leverage=params.max_leverage,
        leverage_ratchet_up=params.leverage_ratchet_up,
        lev_band=params.lev_band,
    )
    return replace(cfg, **overrides) if overrides else cfg


def execute(
    panel: Panel,
    params: StackParams | None = None,
    *,
    vehicle=None,
    use_trend: bool = True,
    use_crisis_cov: bool = True,
    use_throttle: bool = True,
    constraint_step: int = 5,
    **engine_overrides,
) -> RunOutput:
    """Run one full backtest and evaluate the drawdown constraint."""
    params = params or StackParams()
    signals = stack_for(panel, params, use_trend=use_trend, use_crisis_cov=use_crisis_cov)

    budget = design_budget(panel.benchmark, margin=params.dd_budget_margin)
    throttle = DrawdownThrottle(budget=budget) if use_throttle else None

    result = run(
        panel.returns,
        signals.base_weights,
        signals.target_leverage,
        rf_annual=panel.rf,
        config=engine_config(panel, params, **engine_overrides),
        throttle=throttle,
        vehicle=vehicle,
        vehicle_vix=_vix_series(),
    )

    rf_daily = panel.rf / 252.0
    stats = summary(result.returns, benchmark=panel.benchmark, rf_daily=rf_daily)
    stats |= result.summary_costs()
    stats["margin_calls"] = sum(1 for e in result.events if e["type"] == "margin_call")
    stats["ruined"] = result.ruined

    constraint = evaluate(result.returns, panel.benchmark, step=constraint_step)
    return RunOutput(
        result=result, constraint=constraint, stats=stats, params=params, budget=budget
    )


def benchmark_stats(panel: Panel) -> dict:
    return summary(panel.benchmark, benchmark=panel.benchmark, rf_daily=panel.rf / 252.0)


def reference_portfolios(panel: Panel) -> dict[str, pd.Series]:
    """Unlevered null benchmarks the strategy must beat to be worth anything."""
    out: dict[str, pd.Series] = {"spx": panel.benchmark}

    cols = panel.returns.columns
    if "us_equity" in cols and "interm_ust" in cols:
        eq = panel.returns["us_equity"].fillna(0.0)
        bond = panel.returns["interm_ust"].fillna(0.0)
        out["60_40"] = (0.6 * eq + 0.4 * bond).rename("60_40")

    # Unlevered inverse-vol risk parity over whatever sleeves are live.
    vols = panel.returns.ewm(halflife=60, min_periods=60).std()
    inv = (1.0 / vols).where(panel.eligible, 0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = inv.div(inv.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0).shift(1)
    out["risk_parity"] = (weights * panel.returns.fillna(0.0)).sum(axis=1).rename("risk_parity")
    return out
