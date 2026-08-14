"""Constrained parameter search: maximise CAGR subject to the drawdown constraint.

Method is a Sobol scan followed by local refinement around the best *feasible
plateau* -- not a fine grid (eight dimensions makes that hopeless) and not a
gradient method (the objective is non-smooth and the constraint is
path-dependent).

Two choices matter more than the search algorithm.

**Optimise the CVaR of window excesses, gate on the true maximum.** There are
thousands of overlapping 3-year windows, so ``max_w excess(w)`` is an extreme
order statistic determined by a single path; optimising it directly fits that one
window. The soft objective therefore penalises the worst 5% on average, while
feasibility is still decided by the hard maximum.

**Select a plateau, never a peak.** A parameter set that is feasible only in a
narrow spike is an artefact. :func:`plateau_radius` measures how far every
parameter can move before feasibility is lost, and selection requires a minimum
radius.

Draws are evaluated in an order that groups identical *structural* parameters
together, because those force an expensive signal rebuild while the rest are
nearly free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from scipy.stats import qmc

from ..alloc.stack import StackParams
from ..data.panels import Panel
from ..pipeline import execute

# Search box. Structural parameters (rebuild signals) are listed first.
SEARCH_SPACE = {
    "vol_halflife": (20.0, 90.0),
    "trend_fast_weight": (0.0, 0.60),
    "growth_budget": (0.20, 0.60),
    "sigma_target": (0.04, 0.30),
    "max_leverage": (1.0, 3.0),
    "dd_budget_margin": (0.35, 1.00),
    "leverage_ratchet_up": (0.02, 0.40),
    "lev_band": (0.10, 0.50),
}
STRUCTURAL = ("vol_halflife", "trend_fast_weight", "growth_budget")

# Discretisation of the structural axes, so the signal cache actually hits.
STRUCTURAL_GRID = {
    "vol_halflife": [20.0, 30.0, 40.0, 60.0, 90.0],
    "trend_fast_weight": [0.0, 0.15, 0.30, 0.45, 0.60],
    "growth_budget": [0.20, 0.30, 0.40, 0.50, 0.60],
}


@dataclass
class Trial:
    params: StackParams
    cagr: float
    max_dd: float
    bench_dd: float
    sharpe: float
    calmar: float
    worst_excess: float
    cvar5_excess: float
    feasible: bool
    avg_leverage: float
    objective: float

    def row(self) -> dict:
        return asdict(self.params) | {
            "cagr": self.cagr,
            "max_dd": self.max_dd,
            "bench_dd": self.bench_dd,
            "sharpe": self.sharpe,
            "calmar": self.calmar,
            "worst_excess": self.worst_excess,
            "cvar5_excess": self.cvar5_excess,
            "feasible": self.feasible,
            "avg_leverage": self.avg_leverage,
            "objective": self.objective,
        }


def _snap(value: float, options: list[float]) -> float:
    return min(options, key=lambda o: abs(o - value))


def sample_params(n: int, *, seed: int = 0) -> list[StackParams]:
    """Sobol draws, snapped on the structural axes and ordered for cache reuse."""
    keys = list(SEARCH_SPACE)
    sampler = qmc.Sobol(d=len(keys), scramble=True, seed=seed)
    raw = sampler.random(n)
    lows = np.array([SEARCH_SPACE[k][0] for k in keys])
    highs = np.array([SEARCH_SPACE[k][1] for k in keys])
    scaled = qmc.scale(raw, lows, highs)

    out = []
    for row in scaled:
        kwargs = dict(zip(keys, row))
        for k in STRUCTURAL:
            kwargs[k] = _snap(kwargs[k], STRUCTURAL_GRID[k])
        out.append(StackParams(**kwargs))
    out.sort(key=lambda p: tuple(getattr(p, k) for k in STRUCTURAL))
    return out


def objective(cagr: float, worst_excess: float, cvar5: float, *, l1=10.0, l2=5.0) -> float:
    """Soft objective. Infeasible points still get gradient toward feasibility."""
    return cagr - l1 * max(0.0, worst_excess) - l2 * max(0.0, cvar5)


def evaluate(
    panel: Panel,
    params: StackParams,
    *,
    tolerance: float = 0.0,
    constraint_step: int = 21,
    vehicle=None,
) -> Trial:
    out = execute(panel, params, vehicle=vehicle, constraint_step=constraint_step)
    c = out.constraint
    feasible = (
        c.strategy_depth <= c.benchmark_depth + tolerance
        and c.worst_excess <= tolerance
        and not out.stats["ruined"]
    )
    return Trial(
        params=params,
        cagr=out.stats["cagr"],
        max_dd=out.stats["max_dd"],
        bench_dd=-c.benchmark_depth,
        sharpe=out.stats["sharpe"],
        calmar=out.stats["calmar"],
        worst_excess=c.worst_excess,
        cvar5_excess=c.cvar5_excess,
        feasible=feasible,
        avg_leverage=out.stats["avg_leverage"],
        objective=objective(out.stats["cagr"], c.worst_excess - tolerance, c.cvar5_excess - tolerance),
    )


def scan(
    panel: Panel,
    n: int = 512,
    *,
    seed: int = 0,
    tolerance: float = 0.0,
    constraint_step: int = 21,
    vehicle=None,
    progress: bool = True,
) -> pd.DataFrame:
    """Sobol scan over the search box. Returns one row per trial."""
    draws = sample_params(n, seed=seed)
    rows = []
    for i, params in enumerate(draws):
        try:
            trial = evaluate(
                panel, params, tolerance=tolerance,
                constraint_step=constraint_step, vehicle=vehicle,
            )
            rows.append(trial.row())
        except Exception as exc:  # noqa: BLE001 - a bad corner should not kill the scan
            rows.append({**asdict(params), "cagr": np.nan, "feasible": False, "error": str(exc)})
        if progress and (i + 1) % 50 == 0:
            done = pd.DataFrame(rows)
            best = done.loc[done["feasible"], "cagr"].max() if done["feasible"].any() else np.nan
            print(f"  {i + 1}/{len(draws)} trials, best feasible CAGR {best:.2%}", flush=True)
    return pd.DataFrame(rows)


def plateau_radius(
    panel: Panel,
    params: StackParams,
    *,
    tolerance: float = 0.0,
    radii: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25),
    constraint_step: int = 21,
) -> float:
    """Largest relative perturbation under which every parameter stays feasible.

    Each parameter is moved to +/- ``r`` of its range with the others held. A
    point that survives a 15% move in every direction is a plateau; one that
    fails at 5% is a spike and should be discarded however good its CAGR.
    """
    keys = list(SEARCH_SPACE)
    best = 0.0
    for r in radii:
        ok = True
        for k in keys:
            lo, hi = SEARCH_SPACE[k]
            span = hi - lo
            for sign in (-1.0, 1.0):
                value = float(np.clip(getattr(params, k) + sign * r * span, lo, hi))
                if k in STRUCTURAL:
                    value = _snap(value, STRUCTURAL_GRID[k])
                trial = evaluate(
                    panel, replace(params, **{k: value}),
                    tolerance=tolerance, constraint_step=constraint_step,
                )
                if not trial.feasible:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            best = r
        else:
            break
    return best


def select(
    results: pd.DataFrame, *, min_plateau_rows: int = 5
) -> tuple[pd.Series | None, pd.DataFrame]:
    """Best feasible trial and the feasible frontier, ranked by CAGR."""
    feasible = results[results.get("feasible", False)].copy()
    if feasible.empty:
        return None, feasible
    feasible = feasible.sort_values("cagr", ascending=False)
    return feasible.iloc[0], feasible
