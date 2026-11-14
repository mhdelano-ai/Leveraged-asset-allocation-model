"""Evaluation core: realized targets, walk-forward harness, losses, scoreboard.

See the docstrings of ``targets``, ``harness`` and ``losses`` for the exact
definitions; ``python -m assetforecast.eval.run --help`` for the CLI.
"""
from .harness import (ForecastError, load_forecasts, load_forecasts_long, origins_for,  # noqa: F401
                      run_model, run_models, stored_models, validate_forecast)
from .losses import (compute_losses, corr_losses, cov_losses, loss_matrix,  # noqa: F401
                     scoreboard, vol_losses)
from .targets import Targets, compute_targets, get_targets, load_targets, observed_through  # noqa: F401
