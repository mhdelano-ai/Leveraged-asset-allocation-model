"""Drawdown throttle -- the layer that actually defends the hard constraint.

Trend and volatility targeting are *forecasts*, and they can both be wrong at the
same time. The throttle is a **realised-loss feedback loop**: it is the only
layer that references the constraint itself, and the only one that mechanically
guarantees exposure reaches zero before the drawdown budget is spent.

Shape::

    phi(D) = clip(2 - 2*D/D*, 0, 1)

Full risk while the drawdown is shallower than half the budget, then linear to
zero at the budget. The dead zone matters: a throttle that engages from the first
dollar of loss is textbook CPPI cash-lock -- it delevers into every ordinary dip
and misses the recovery, destroying return in exchange for protection against
drawdowns that were never going to threaten the budget.

Hysteresis handles the other half of that failure. Once exposure has been cut to
zero, a small bounce should not immediately re-lever into what is usually a
bear-market rally; risk stays capped until a genuine recovery is underway.

The budget is deliberately **static**, not tied to the benchmark's live
drawdown. Tracking the benchmark looks like a faithful reading of "drawdowns not
exceeding the S&P 500", but it creates a perverse controller that grants the most
risk exactly when the S&P is deepest -- and the S&P may recover along a path the
strategy does not. The *relative* comparison belongs in the constraint evaluator;
the controller should answer only to its own losses.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DrawdownThrottle:
    """Stateful exposure multiplier driven by realised drawdown.

    Parameters
    ----------
    budget:
        Drawdown depth (positive, e.g. ``0.44``) at which exposure reaches zero.
    dead_zone:
        Fraction of the budget that incurs no throttling at all.
    recovery_level:
        Fraction of the budget the drawdown must recover to before the
        hysteresis cap lifts.
    locked_cap:
        Maximum exposure while the hysteresis lock is active.
    floor:
        Minimum multiplier once fully throttled. Kept at 0 by default; the
        hysteresis, not a floor, is what prevents permanent cash-lock.
    """

    budget: float = 0.44
    dead_zone: float = 0.5
    recovery_level: float = 0.4
    locked_cap: float = 0.5
    floor: float = 0.0

    _locked: bool = False

    def reset(self) -> None:
        self._locked = False

    def __call__(self, drawdown_depth: float) -> float:
        d = max(0.0, float(drawdown_depth))
        budget = max(self.budget, 1e-6)
        start = self.dead_zone * budget

        if d <= start:
            phi = 1.0
        elif d >= budget:
            phi = self.floor
        else:
            phi = (budget - d) / (budget - start)
            phi = max(self.floor, min(1.0, phi))

        if phi <= self.floor + 1e-12:
            self._locked = True
        if self._locked:
            if d <= self.recovery_level * budget:
                self._locked = False
            else:
                phi = min(phi, self.locked_cap)
        return phi
