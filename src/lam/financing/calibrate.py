"""Fit and validate the synthetic-LETF model against real funds.

The model is::

    r_letf = k*r_u - (k-1)*(rf + s)*dt/360 - TER*dt/365

with everything known except the swap spread ``s``. Rearranging gives a direct
estimator: the daily residual between ``k*r_u`` and the fund's actual return,
net of the known financing and fee terms, must be exactly ``(k-1)*s*dt/360``.

Fitting each fund independently is the point. If the functional form is right,
UPRO (3x equity) and SSO (2x equity) -- different multipliers, different funds,
separate regressions -- should imply the *same* spread. They do, to within a
basis point, which is far stronger evidence than any single fit's R-squared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import lbma, synth_equity, yahoo
from ..data.rates import bill_rate_bey
from ..metrics.core import cagr
from ..timeaxis import day_deltas
from .letf import simulate_letf

# fund -> (multiplier, expense ratio, underlying key)
#
# The expense ratios are the stated prospectus figures, but note what the fit can
# and cannot see: the estimator solves for ``s`` given ``TER``, so only the *sum*
# ``(k-1)*s + TER`` is identified by the data. An error in the quoted expense
# ratio moves the fitted spread by an offsetting amount and leaves the simulated
# fund unchanged -- which is why :func:`validate` compares whole return paths
# rather than reporting the split as if it were measured.
FUNDS = {
    "UPRO": (3.0, 0.0091, "spx"),
    "SSO": (2.0, 0.0089, "spx"),
    "TMF": (3.0, 0.0106, "tlt"),
    "UGL": (2.0, 0.0095, "gold"),
    "TQQQ": (3.0, 0.0084, "ndx"),
    "QLD": (2.0, 0.0095, "ndx"),
}


def _underlyings() -> dict[str, pd.Series]:
    from ..data import nasdaq

    return {
        "spx": synth_equity.spx_total_return(),
        "ndx": nasdaq.ndx_total_return(),
        "tlt": yahoo.total_return_prices("TLT").pct_change().dropna(),
        # GLD, not the LBMA fix. UGL marks at the US close while the London PM
        # fix is set mid-afternoon UK time, so the two are measuring different
        # moments of the same day -- that timing mismatch alone drops the daily
        # correlation to ~0.66 and makes the fit meaningless. LBMA remains the
        # right source for the long history; it is just not the right
        # comparison for a US-listed fund.
        "gold": yahoo.total_return_prices("GLD").pct_change().dropna(),
    }


def fit_spread(fund: str) -> dict:
    """Estimate the embedded swap spread for one leveraged fund."""
    k, ter, key = FUNDS[fund]
    actual = yahoo.total_return_prices(fund).pct_change().dropna()
    underlying = _underlyings()[key]
    rf = bill_rate_bey()

    joined = pd.concat(
        [actual.rename("letf"), underlying.rename("u"), rf.rename("rf")], axis=1, join="inner"
    ).dropna()
    if len(joined) < 250:
        raise ValueError(f"{fund}: only {len(joined)} overlapping days")

    dt = day_deltas(joined.index, prepend=1.0)
    residual = k * joined["u"].to_numpy() - joined["letf"].to_numpy()
    known = (k - 1.0) * joined["rf"].to_numpy() * dt / 360.0 + ter * dt / 365.0
    spread_num = float(np.sum(residual - known))
    spread_den = float(np.sum((k - 1.0) * dt / 360.0))
    spread = spread_num / spread_den if spread_den else np.nan

    years = (joined.index[-1] - joined.index[0]).days / 365.25
    gross_gap = float(np.sum(residual)) / years
    avg_rf = float(joined["rf"].mean())

    # Decompose the gap. Most of it is the plain cost of borrowing (k-1) units
    # of cash, which *any* leverage vehicle pays -- quoting it as "LETF drag"
    # overstates the wrapper's inefficiency. The part attributable to the
    # wrapper is the swap spread plus the expense ratio.
    financing_cost = (k - 1.0) * avg_rf
    wrapper_cost = (k - 1.0) * spread + ter

    return {
        "fund": fund,
        "multiplier": k,
        "expense_ratio": ter,
        "implied_spread": spread,
        "avg_rf": avg_rf,
        "gross_gap_ann": gross_gap,
        "financing_cost_ann": financing_cost,
        "wrapper_cost_ann": wrapper_cost,
        "wrapper_vs_ter": wrapper_cost / ter if ter else np.nan,
        "n_days": len(joined),
        "start": str(joined.index[0].date()),
        "end": str(joined.index[-1].date()),
    }


def validate(fund: str, spread: float | None = None) -> dict:
    """Compare a simulated fund against the real one."""
    k, ter, key = FUNDS[fund]
    actual = yahoo.total_return_prices(fund).pct_change().dropna()
    underlying = _underlyings()[key]
    rf = bill_rate_bey()
    if spread is None:
        spread = fit_spread(fund)["implied_spread"]

    joined = pd.concat(
        [actual.rename("letf"), underlying.rename("u")], axis=1, join="inner"
    ).dropna()
    dt = day_deltas(joined.index, prepend=1.0)
    sim = simulate_letf(
        joined["u"], rf, multiplier=k, expense_ratio=ter, swap_spread=spread, day_counts=dt
    )

    a, b = sim, joined["letf"]
    return {
        "fund": fund,
        "spread_used": spread,
        "corr": float(a.corr(b)),
        "cagr_sim": cagr(a),
        "cagr_actual": cagr(b),
        "cagr_diff_bps": (cagr(a) - cagr(b)) * 1e4,
        "te_ann": float((a - b).std() * np.sqrt(252)),
        "n_days": len(joined),
        "start": str(joined.index[0].date()),
    }


def calibrate_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    fits = pd.DataFrame([fit_spread(f) for f in FUNDS]).set_index("fund")

    def group_mean(key: str) -> float:
        members = [f for f in FUNDS if FUNDS[f][2] == key]
        return float(fits.loc[members, "implied_spread"].mean())

    equity = group_mean("spx")
    grouped = {
        "spx": equity,
        "tlt": group_mean("tlt"),
        "gold": equity,
        # Fitted on its own funds. TQQQ and QLD price the same underlying at
        # different multipliers, so agreement between them is an independent
        # check of the functional form on a second index.
        "ndx": group_mean("ndx"),
    }
    checks = pd.DataFrame(
        [validate(f, grouped[FUNDS[f][2]]) for f in FUNDS]
    ).set_index("fund")
    return fits, checks
