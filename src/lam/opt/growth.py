"""Search for the return-maximising configuration of a concentrated levered book.

The objective here is compound return, not return subject to a drawdown
constraint, which changes what the search has to defend against.

**Unconstrained CAGR maximisation has a degenerate answer and a fragile one.**
The degenerate answer is that with no drawdown limit the optimiser will happily
accept a -95% path if it ends higher, and no investor would hold it. The fragile
answer is that the in-sample argmax of compound return sits on the edge of the
leverage cliff -- the growth curve in leverage is flat near its peak and falls off
a precipice just beyond, so a point chosen at the peak on 40 years of data is one
estimation error away from the wrong side.

Three devices deal with that:

``dd_ceiling``
    A dial, not a fixed rule. Sweeping it traces the whole return/drawdown
    frontier so the choice of tolerance stays the investor's, and the search
    reports what each level of pain buys.
``half_sample_cagr``
    Every trial records the *worse* of its first-half and second-half compound
    return. A configuration that made all its money in one regime is visible
    immediately.
``leverage_headroom``
    Distance from the trial's leverage to the leverage at which realised
    compound return would have peaked. Points sitting past the peak are earning
    less return for more risk and are rejected outright.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from scipy.stats import qmc

from ..alloc.growth import GrowthParams
from ..data.panels import Panel
from ..metrics.core import cagr
from ..metrics.drawdown import max_drawdown
from ..pipeline import execute_growth

# Search box. Structural parameters (those that force a signal rebuild) first.
SEARCH_SPACE = {
    "vol_halflife": (20.0, 90.0),
    "trend_fast_weight": (0.0, 0.60),
    "trend_floor": (0.0, 0.50),
    "stress_cap": (0.20, 1.00),
    "sigma_target": (0.10, 0.60),
    "max_leverage": (1.0, 4.0),
    "dd_budget": (0.30, 0.80),
    "leverage_ratchet_up": (0.05, 0.50),
}
STRUCTURAL = ("vol_halflife", "trend_fast_weight", "trend_floor", "stress_cap")

# Discretisation of the structural axes so the signal cache actually hits.
STRUCTURAL_GRID = {
    "vol_halflife": [20.0, 30.0, 40.0, 60.0, 90.0],
    "trend_fast_weight": [0.0, 0.15, 0.30, 0.45, 0.60],
    "trend_floor": [0.0, 0.10, 0.25, 0.50],
    "stress_cap": [0.20, 0.35, 0.50, 0.75, 1.00],
}


@dataclass
class GrowthTrial:
    params: GrowthParams
    cagr: float
    asset_cagr: float
    vol: float
    sharpe: float
    max_dd: float
    asset_dd: float
    calmar: float
    worst_12m: float
    avg_leverage: float
    max_leverage_used: float
    half_sample_cagr: float
    ruined: bool

    def row(self) -> dict:
        return asdict(self.params) | {
            "cagr": self.cagr,
            "asset_cagr": self.asset_cagr,
            "excess_cagr": self.cagr - self.asset_cagr,
            "vol": self.vol,
            "sharpe": self.sharpe,
            "max_dd": self.max_dd,
            "asset_dd": self.asset_dd,
            "calmar": self.calmar,
            "worst_12m": self.worst_12m,
            "avg_leverage": self.avg_leverage,
            "max_leverage_used": self.max_leverage_used,
            "half_sample_cagr": self.half_sample_cagr,
            "ruined": self.ruined,
        }


def _snap(value: float, options: list[float]) -> float:
    return min(options, key=lambda o: abs(o - value))


def sample_params(n: int, *, seed: int = 0) -> list[GrowthParams]:
    """Sobol draws, snapped on the structural axes and ordered for cache reuse."""
    keys = list(SEARCH_SPACE)
    sampler = qmc.Sobol(d=len(keys), scramble=True, seed=seed)
    scaled = qmc.scale(
        sampler.random(n),
        np.array([SEARCH_SPACE[k][0] for k in keys]),
        np.array([SEARCH_SPACE[k][1] for k in keys]),
    )

    out = []
    for row in scaled:
        kwargs = dict(zip(keys, row))
        for k in STRUCTURAL:
            kwargs[k] = _snap(kwargs[k], STRUCTURAL_GRID[k])
        out.append(GrowthParams(**kwargs))
    out.sort(key=lambda p: tuple(getattr(p, k) for k in STRUCTURAL))
    return out


def half_sample_cagr(returns: pd.Series) -> float:
    """The worse of the two half-sample compound returns.

    A single number that separates a configuration which worked throughout from
    one that earned everything in a single regime -- the 1990s for a Nasdaq book,
    or the post-2009 bull for anything levered.
    """
    if len(returns) < 500:
        return cagr(returns)
    mid = len(returns) // 2
    return min(cagr(returns.iloc[:mid]), cagr(returns.iloc[mid:]))


def evaluate(panel: Panel, params: GrowthParams, *, vehicle=None) -> GrowthTrial:
    out = execute_growth(panel, params, vehicle=vehicle)
    s = out.stats
    return GrowthTrial(
        params=params,
        cagr=s["cagr"],
        asset_cagr=out.benchmark_stats["cagr"],
        vol=s["vol"],
        sharpe=s["sharpe"],
        max_dd=s["max_dd"],
        asset_dd=out.benchmark_stats["max_dd"],
        calmar=s["calmar"],
        worst_12m=s["worst_12m"],
        avg_leverage=s["avg_leverage"],
        max_leverage_used=s["max_leverage_used"],
        half_sample_cagr=half_sample_cagr(out.result.returns),
        ruined=bool(s["ruined"]),
    )


def scan(
    panel: Panel,
    n: int = 512,
    *,
    seed: int = 0,
    vehicle=None,
    progress: bool = True,
) -> pd.DataFrame:
    """Sobol scan over the search box. One row per trial, no filtering applied.

    Filtering is deliberately left to :func:`select` so a single scan can be
    re-read at every drawdown tolerance without recomputation.
    """
    draws = sample_params(n, seed=seed)
    rows = []
    for i, params in enumerate(draws):
        try:
            rows.append(evaluate(panel, params, vehicle=vehicle).row())
        except Exception as exc:  # noqa: BLE001 - a bad corner should not kill the scan
            rows.append({**asdict(params), "cagr": np.nan, "ruined": True, "error": str(exc)})
        if progress and (i + 1) % 50 == 0:
            done = pd.DataFrame(rows)
            alive = done[~done["ruined"].fillna(True)]
            best = alive["cagr"].max() if len(alive) else np.nan
            print(f"  {i + 1}/{len(draws)} trials, best surviving CAGR {best:.2%}", flush=True)
    return pd.DataFrame(rows)


def select(
    results: pd.DataFrame,
    *,
    dd_ceiling: float | None = None,
    min_half_sample: float | None = None,
) -> tuple[pd.Series | None, pd.DataFrame]:
    """Best trial under a drawdown tolerance, and the surviving frontier.

    ``dd_ceiling`` is a positive depth (``0.50`` means "no worse than -50%").
    """
    alive = results[~results["ruined"].fillna(True)].copy()
    alive = alive[alive["cagr"].notna()]
    if dd_ceiling is not None:
        alive = alive[alive["max_dd"] >= -abs(dd_ceiling)]
    if min_half_sample is not None:
        alive = alive[alive["half_sample_cagr"] >= min_half_sample]
    if alive.empty:
        return None, alive
    alive = alive.sort_values("cagr", ascending=False)
    return alive.iloc[0], alive


def frontier(
    results: pd.DataFrame,
    ceilings: tuple[float, ...] = (0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85),
    *,
    min_half_sample: float | None = None,
) -> pd.DataFrame:
    """Best achievable compound return at each drawdown tolerance.

    The central deliverable of a return-maximising study. "Maximise return" has
    no answer until someone says how much drawdown they will sit through, and
    this table makes the exchange rate explicit.
    """
    rows = []
    for ceiling in ceilings:
        best, pool = select(results, dd_ceiling=ceiling, min_half_sample=min_half_sample)
        if best is None:
            rows.append({"dd_ceiling": ceiling, "n_feasible": 0})
            continue
        rows.append(
            {
                "dd_ceiling": ceiling,
                "n_feasible": int(len(pool)),
                "cagr": best["cagr"],
                "asset_cagr": best["asset_cagr"],
                "max_dd": best["max_dd"],
                "sharpe": best["sharpe"],
                "avg_leverage": best["avg_leverage"],
                "sigma_target": best["sigma_target"],
                "max_leverage": best["max_leverage"],
                "half_sample_cagr": best["half_sample_cagr"],
            }
        )
    return pd.DataFrame(rows).set_index("dd_ceiling")


def leverage_headroom(panel: Panel, params: GrowthParams) -> dict:
    """How far the configuration sits from the top of its own growth curve.

    Compound return as a function of the risk target is a concave curve with a
    long flat top and a cliff beyond it. Being 20% *below* the peak costs almost
    nothing; being 20% past it costs return *and* adds drawdown. The sign of
    ``headroom`` is therefore the number to read, not its size.
    """
    scale = np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0])
    rows = []
    for s in scale:
        trial = evaluate(panel, replace(params, sigma_target=params.sigma_target * float(s)))
        rows.append({"scale": float(s), "sigma": params.sigma_target * float(s),
                     "cagr": trial.cagr, "max_dd": trial.max_dd})
    curve = pd.DataFrame(rows).set_index("scale")
    peak = curve["cagr"].idxmax()
    return {
        "peak_scale": float(peak),
        "peak_sigma": float(curve.loc[peak, "sigma"]),
        "headroom": float(peak) - 1.0,
        "cagr_at_peak": float(curve.loc[peak, "cagr"]),
        "cagr_here": float(curve.loc[1.0, "cagr"]),
        "past_peak": bool(peak < 1.0),
        "curve": curve,
    }


def ablation(panel: Panel, params: GrowthParams) -> pd.DataFrame:
    """What each layer contributes, by removing it.

    Ablation is the only honest way to justify a layer. A component that cannot
    be shown to change the result is decoration, and decoration in a levered
    strategy is a place for overfitting to hide.
    """
    variants = {
        "full stack": {},
        "no trend gate": {"use_trend": False},
        "no vol regime": {"use_regime": False},
        "no drawdown throttle": {"use_throttle": False},
        "no trend, no regime": {"use_trend": False, "use_regime": False},
    }
    rows = []
    for label, kwargs in variants.items():
        out = execute_growth(panel, params, **kwargs)
        rows.append(
            {
                "variant": label,
                "cagr": out.stats["cagr"],
                "vol": out.stats["vol"],
                "sharpe": out.stats["sharpe"],
                "max_dd": out.stats["max_dd"],
                "calmar": out.stats["calmar"],
                "avg_leverage": out.stats["avg_leverage"],
                "half_sample_cagr": half_sample_cagr(out.result.returns),
            }
        )
    rows.append(
        {
            "variant": "buy and hold asset",
            "cagr": cagr(panel.benchmark),
            "vol": float(panel.benchmark.std() * np.sqrt(252)),
            "sharpe": np.nan,
            "max_dd": max_drawdown(panel.benchmark),
            "calmar": np.nan,
            "avg_leverage": 1.0,
            "half_sample_cagr": half_sample_cagr(panel.benchmark),
        }
    )
    return pd.DataFrame(rows).set_index("variant")
