"""The drawdown constraint: "drawdowns must not exceed the S&P 500's".

Two conditions, both required:

1. **Full sample** -- the strategy's worst peak-to-trough is no deeper than the
   benchmark's over the same dates.
2. **Rolling 3-year** -- in *every* rolling 3y window, the strategy's worst
   drawdown within that window is no deeper than the benchmark's within the same
   window.

Condition 2 is what gives the exercise teeth. On its own, condition 1 is nearly
free: the S&P's full-sample depth is ~-55%, so a strategy could take one
catastrophic -54% loss at a completely different time and still "pass". The
rolling test requires the strategy to be no worse *contemporaneously*, across
every three-year stretch of history.

Everything here uses **positive depth** (0.55 == a 55% loss), so "exceeds" means
"greater than" and comparisons read the way the requirement is written.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .drawdown import WINDOW_3Y, depth, rolling_max_drawdown


@dataclass
class ConstraintResult:
    passed: bool
    full_sample_pass: bool
    rolling_pass: bool
    strategy_depth: float
    benchmark_depth: float
    worst_excess: float
    cvar5_excess: float
    n_windows: int
    n_violating_windows: int
    worst_window_start: pd.Timestamp | None = None
    worst_window_end: pd.Timestamp | None = None
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] strategy maxDD {self.strategy_depth:.2%} vs "
            f"benchmark {self.benchmark_depth:.2%} | worst rolling-3y excess "
            f"{self.worst_excess:+.2%} | {self.n_violating_windows}/{self.n_windows} "
            f"windows violating"
        )


def evaluate(
    strategy: pd.Series,
    benchmark: pd.Series,
    *,
    window: int = WINDOW_3Y,
    step: int = 1,
    tolerance: float = 0.0,
) -> ConstraintResult:
    """Evaluate both drawdown conditions on a shared date index.

    ``tolerance`` allows a small slack (in depth units, e.g. 0.005 == 0.5pp)
    before a window counts as violating; the default is a strict reading.
    """
    joined = pd.concat(
        [strategy.rename("s"), benchmark.rename("b")], axis=1, join="inner"
    ).dropna()
    if joined.empty:
        raise ValueError("strategy and benchmark share no dates")

    s, b = joined["s"], joined["b"]
    s_depth, b_depth = depth(s), depth(b)
    full_pass = s_depth <= b_depth + tolerance

    s_roll = -rolling_max_drawdown(s, window, step=step)
    b_roll = -rolling_max_drawdown(b, window, step=step)
    excess = s_roll - b_roll

    worst_idx = int(np.argmax(excess))
    worst_excess = float(excess[worst_idx])
    n_violating = int((excess > tolerance).sum())
    rolling_pass = worst_excess <= tolerance

    # CVaR of the worst 5% of window excesses. The optimiser targets this rather
    # than the raw max: with thousands of overlapping windows the max is an
    # extreme order statistic set by a single path, so optimising it directly
    # fits one window. The max is still checked as a hard gate.
    k = max(1, int(np.ceil(0.05 * excess.size)))
    cvar5 = float(np.sort(excess)[-k:].mean())

    starts = np.arange(0, len(joined) - window + 1, step)
    if starts.size:
        ws = joined.index[starts[min(worst_idx, starts.size - 1)]]
        we = joined.index[min(starts[min(worst_idx, starts.size - 1)] + window - 1, len(joined) - 1)]
    else:
        ws = we = None

    return ConstraintResult(
        passed=bool(full_pass and rolling_pass),
        full_sample_pass=bool(full_pass),
        rolling_pass=bool(rolling_pass),
        strategy_depth=s_depth,
        benchmark_depth=b_depth,
        worst_excess=worst_excess,
        cvar5_excess=cvar5,
        n_windows=int(excess.size),
        n_violating_windows=n_violating,
        worst_window_start=ws,
        worst_window_end=we,
        detail={
            "median_excess": float(np.median(excess)),
            "p95_excess": float(np.quantile(excess, 0.95)),
            "frac_violating": float(n_violating / excess.size) if excess.size else 0.0,
        },
    )


def excess_drawdown_path(
    strategy: pd.Series,
    benchmark: pd.Series,
    *,
    window: int = WINDOW_3Y,
    step: int = 5,
) -> pd.Series:
    """Rolling-3y excess drawdown, indexed by window *end* date.

    The headline chart of the whole project: the constraint holds exactly when
    this line never rises above zero.
    """
    joined = pd.concat(
        [strategy.rename("s"), benchmark.rename("b")], axis=1, join="inner"
    ).dropna()
    s_roll = -rolling_max_drawdown(joined["s"], window, step=step)
    b_roll = -rolling_max_drawdown(joined["b"], window, step=step)
    starts = np.arange(0, len(joined) - window + 1, step)
    ends = joined.index[np.minimum(starts + window - 1, len(joined) - 1)]
    return pd.Series(s_roll - b_roll, index=ends, name="excess_dd")


def design_budget(benchmark: pd.Series, margin: float = 0.80) -> float:
    """Drawdown budget to *design* to, as a positive depth.

    Deliberately tighter than the benchmark's realised depth. The historical
    record contains only ~7 independent major drawdown events, so the observed
    benchmark depth is itself a small-sample statistic. Designing straight to it
    leaves no room for the next crisis being worse than any in the sample --
    which is the single most likely way an in-sample "PASS" becomes an
    out-of-sample failure.
    """
    return margin * depth(benchmark)
