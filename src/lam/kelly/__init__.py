"""Kelly-optimal allocation across the four-asset stock/bond universe.

Exposed as :func:`build_panel` (data) and the solvers in :mod:`lam.kelly.solve`.
"""

from .panel import ASSETS, KellyPanel, PANELS, PRETTY, PROXY_NOTES, build_panel
from .solve import (
    KellySolution,
    bootstrap_weights,
    constrained_shrunk_kelly,
    empirical_kelly,
    fraction_curve,
    gaussian_kelly,
    growth_rate,
    score,
    shrunk_kelly,
    summarise,
    walk_forward,
)

__all__ = [
    "ASSETS",
    "KellyPanel",
    "KellySolution",
    "PANELS",
    "PRETTY",
    "PROXY_NOTES",
    "bootstrap_weights",
    "constrained_shrunk_kelly",
    "build_panel",
    "empirical_kelly",
    "fraction_curve",
    "gaussian_kelly",
    "growth_rate",
    "score",
    "shrunk_kelly",
    "summarise",
    "walk_forward",
]
