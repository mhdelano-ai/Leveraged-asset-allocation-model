"""Common interface for the two leverage vehicles.

The same allocation logic is run through both so the head-to-head measures the
vehicles, not two different strategies. A vehicle may:

* transform the sleeve return stream (a leveraged ETF *is* a different
  instrument, with its own fee and financing drag baked in);
* refuse to express the requested weights (no 3x broad-commodity fund exists);
* charge a borrowing spread that varies with market stress;
* force a deleverage (margin accounts only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass
class VehicleContext:
    """Everything a vehicle needs to price and police a position."""

    returns: pd.DataFrame
    rf: pd.Series
    vix: pd.Series | None = None
    sleeve_names: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)


class FinancingVehicle(Protocol):
    name: str
    max_leverage: float

    def prepare(self, ctx: VehicleContext) -> pd.DataFrame:
        """Return the instrument-level return matrix the engine should trade."""

    def borrow_spread(self, t: int) -> float:
        """Spread over the base rate charged on borrowed cash on day ``t``."""

    def achievable_weights(self, target_notional: np.ndarray, t: int) -> np.ndarray:
        """Convert desired notional exposure into weights the vehicle can hold."""

    def notional_leverage(self, held_weights: np.ndarray) -> float:
        """Economic exposure implied by the held weights."""

    def check_margin(
        self, equity: float, weights: np.ndarray, t: int
    ) -> tuple[float, float, str] | None:
        """Return ``(scale, slippage, note)`` if a forced deleverage fires."""
