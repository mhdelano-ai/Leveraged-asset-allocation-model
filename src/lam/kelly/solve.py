"""Kelly solvers.

Two things separate this from the textbook formula, and both change the answer
materially:

1. **The Gaussian formula is used as a reference, not as the answer.** The
   closed form ``f* = Sigma^-1 mu`` maximises expected log wealth only for
   continuously rebalanced lognormal returns. The number reported as *the* Kelly
   portfolio is instead the direct maximiser of realised monthly log growth --
   ``max_w (1/T) sum log(1 + rf_t + w'x_t)`` -- which uses the actual return
   distribution, fat tails and all, and cannot be talked into a weight that would
   have bankrupted the account in some month of the sample.

2. **Leverage is priced.** Full Kelly on a bond-heavy book asks for multiples of
   capital, and gross exposure above 1x has to be borrowed. Every objective here
   charges ``financing_spread`` over the bill rate on the borrowed part, so the
   optimiser sees the real trade-off instead of free leverage.

The estimation-error machinery (:func:`bootstrap_weights`, :func:`walk_forward`)
is not decoration. For this universe the sampling error on the weights is larger
than the weights, and a Kelly answer quoted without it is a point estimate of a
quantity that cannot be estimated from the available history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .panel import ASSETS, KellyPanel

MONTHS = 12

# Portfolio margin / box-spread financing over the bill rate, the same order of
# magnitude the rest of the project uses for the cheap borrowing vehicle.
DEFAULT_SPREAD = 0.006

# Box bound on any single weight. Not an economic constraint -- it exists so the
# optimiser terminates. A solution that sits *on* this bound is the interesting
# result, not a nuisance: it says the sample's log-growth surface is still rising
# at 20x notional, i.e. that unconstrained Kelly is not identified from this data.
BOX = 20.0

MODES = ("unconstrained", "long_only", "long_only_unlevered")

# Fixed comparison portfolios, written as sleeve maps so adding a sleeve to the
# universe cannot silently reweight a benchmark.
BENCHMARKS = {
    "60_40": {"us_equity": 0.36, "intl_equity": 0.24, "us_bonds": 0.28, "intl_bonds": 0.12},
    "all_us_equity": {"us_equity": 1.0},
}


def _benchmark(name: str) -> np.ndarray:
    weights = BENCHMARKS[name]
    return np.array([weights.get(asset, 0.0) for asset in ASSETS], dtype=float)


@dataclass(frozen=True)
class KellySolution:
    """A weight vector and what it did in-sample."""

    panel: str
    mode: str
    fraction: float
    weights: pd.Series
    growth: float  # annualised expected log growth, net of financing
    arith: float  # annualised arithmetic return
    vol: float  # annualised volatility
    gross: float  # sum of |weights|
    worst_month: float

    def as_row(self) -> dict:
        row = {"panel": self.panel, "mode": self.mode, "fraction": self.fraction}
        row.update({a: float(self.weights[a]) for a in ASSETS})
        row["cash"] = 1.0 - float(self.weights.sum())
        row["gross"] = self.gross
        row["growth"] = self.growth
        row["arith"] = self.arith
        row["vol"] = self.vol
        row["worst_month"] = self.worst_month
        return row


def _arrays(panel: KellyPanel) -> tuple[np.ndarray, np.ndarray]:
    return panel.excess.to_numpy(), panel.cash.to_numpy()


def portfolio_returns(
    weights: np.ndarray,
    excess: np.ndarray,
    cash: np.ndarray,
    *,
    financing_spread: float = DEFAULT_SPREAD,
) -> np.ndarray:
    """Monthly simple returns of a book held at fixed weights.

    Unspent capital earns the bill rate; gross exposure above 1x pays the bill
    rate plus ``financing_spread`` on the excess.
    """
    borrowed = max(np.abs(weights).sum() - 1.0, 0.0)
    return cash + excess @ weights - borrowed * financing_spread / MONTHS


def growth_rate(
    weights: np.ndarray,
    excess: np.ndarray,
    cash: np.ndarray,
    *,
    financing_spread: float = DEFAULT_SPREAD,
) -> float:
    """Annualised realised log growth. ``-inf`` if the book is ever wiped out."""
    gross_return = 1.0 + portfolio_returns(
        weights, excess, cash, financing_spread=financing_spread
    )
    if np.min(gross_return) <= 1e-9:
        return -np.inf
    return float(MONTHS * np.mean(np.log(gross_return)))


def gaussian_kelly(panel: KellyPanel) -> pd.Series:
    """The textbook ``Sigma^-1 mu`` on annualised excess returns.

    Reported for comparison only. It ignores financing, ignores higher moments
    and inverts a covariance matrix estimated from the same sample it is scored
    on, which for four correlated assets is where most of the instability enters.
    """
    excess = panel.excess
    mu = excess.mean().to_numpy() * MONTHS
    sigma = excess.cov().to_numpy() * MONTHS
    weights = np.linalg.solve(sigma, mu)
    return pd.Series(weights, index=ASSETS, name="gaussian_kelly")


def _bounds_and_constraints(mode: str, n: int, max_gross: float | None):
    if mode == "unconstrained":
        bounds = [(-BOX, BOX)] * n
    elif mode in {"long_only", "long_only_unlevered"}:
        bounds = [(0.0, BOX)] * n
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown mode {mode!r}")

    cons = []
    cap = 1.0 if mode == "long_only_unlevered" else max_gross
    if cap is not None:
        cons.append({"type": "ineq", "fun": lambda w, cap=cap: cap - np.abs(w).sum()})
    return bounds, cons


def empirical_kelly(
    panel: KellyPanel,
    *,
    mode: str = "unconstrained",
    fraction: float = 1.0,
    max_gross: float | None = None,
    financing_spread: float = DEFAULT_SPREAD,
) -> KellySolution:
    """Maximise realised monthly log growth over the sample.

    ``fraction`` scales the solved full-Kelly weights afterwards -- half Kelly is
    half the bet, not the optimum of a different objective.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    excess, cash = _arrays(panel)
    n = excess.shape[1]

    def negative_growth(w: np.ndarray) -> float:
        g = growth_rate(w, excess, cash, financing_spread=financing_spread)
        return 1e6 if not np.isfinite(g) else -g

    bounds, cons = _bounds_and_constraints(mode, n, max_gross)

    # Several starts: the objective is concave in w on the region where no month
    # wipes out, but the wipe-out barrier and the |w| financing kink can strand a
    # single start on the boundary.
    starts = [np.zeros(n), np.full(n, 0.25), np.full(n, 0.6)]
    if mode == "unconstrained":
        try:
            starts.append(np.clip(gaussian_kelly(panel).to_numpy(), -3.0, 3.0))
        except np.linalg.LinAlgError:
            # A singular sample covariance (a perfectly redundant sleeve, or fewer
            # months than assets) kills the closed form but not the search.
            pass

    best, best_val = None, np.inf
    for start in starts:
        start = np.clip(start, [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(
            negative_growth,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if res.fun < best_val:
            best, best_val = res.x, res.fun
    assert best is not None

    weights = pd.Series(best * fraction, index=ASSETS, name="weights")
    realised = portfolio_returns(
        weights.to_numpy(), excess, cash, financing_spread=financing_spread
    )
    return KellySolution(
        panel=panel.name,
        mode=mode,
        fraction=fraction,
        weights=weights,
        growth=growth_rate(
            weights.to_numpy(), excess, cash, financing_spread=financing_spread
        ),
        arith=float(realised.mean() * MONTHS),
        vol=float(realised.std(ddof=1) * np.sqrt(MONTHS)),
        gross=float(weights.abs().sum()),
        worst_month=float(realised.min()),
    )


def bootstrap_weights(
    panel: KellyPanel,
    *,
    mode: str = "unconstrained",
    draws: int = 400,
    block: int = 12,
    seed: int = 0,
    financing_spread: float = DEFAULT_SPREAD,
) -> pd.DataFrame:
    """Circular block bootstrap of the Kelly weights.

    Blocks of ``block`` months preserve the volatility clustering and the
    stock/bond correlation runs that a plain iid resample would destroy -- and
    those runs are precisely what the bond weights are sensitive to.
    """
    rng = np.random.default_rng(seed)
    excess, cash = _arrays(panel)
    T, n = excess.shape
    n_blocks = int(np.ceil(T / block))

    rows = []
    for _ in range(draws):
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([(s + np.arange(block)) % T for s in starts])[:T]
        sample = KellyPanel(
            name=panel.name,
            returns=pd.DataFrame(
                excess[idx] + cash[idx, None], columns=ASSETS, index=panel.returns.index
            ),
            cash=pd.Series(cash[idx], index=panel.returns.index, name="cash"),
            tickers=panel.tickers,
        )
        sol = empirical_kelly(
            sample, mode=mode, financing_spread=financing_spread
        )
        rows.append(sol.weights.to_dict() | {"gross": sol.gross})
    return pd.DataFrame(rows)


def walk_forward(
    panel: KellyPanel,
    *,
    mode: str = "long_only_unlevered",
    fractions: tuple[float, ...] = (1.0, 0.5, 0.25),
    min_months: int = 120,
    refit_every: int = 12,
    max_gross: float | None = None,
    financing_spread: float = DEFAULT_SPREAD,
) -> pd.DataFrame:
    """Out-of-sample test: fit Kelly on the past, hold it forward.

    Answers the only question that matters about a historically fitted Kelly
    portfolio -- whether the weights it produces from data available at the time
    would have out-grown simple fixed allocations afterwards.
    """
    excess, cash = _arrays(panel)
    T = excess.shape[0]
    if T <= min_months:
        raise ValueError(f"panel {panel.name} has {T} months; need > {min_months}")

    index = panel.returns.index
    names = [f"kelly_{f:g}x" for f in fractions] + ["60_40", "all_us_equity", "cash"]
    out = {name: [] for name in names}
    dates = []

    weights: dict[float, np.ndarray] = {}
    for t in range(min_months, T):
        if (t - min_months) % refit_every == 0 or not weights:
            train = KellyPanel(
                name=panel.name,
                returns=panel.returns.iloc[:t],
                cash=panel.cash.iloc[:t],
                tickers=panel.tickers,
            )
            base = empirical_kelly(
                train,
                mode=mode,
                max_gross=max_gross,
                financing_spread=financing_spread,
            ).weights.to_numpy()
            weights = {f: base * f for f in fractions}

        step_excess = excess[t : t + 1]
        step_cash = cash[t : t + 1]
        for f in fractions:
            out[f"kelly_{f:g}x"].append(
                portfolio_returns(
                    weights[f], step_excess, step_cash, financing_spread=financing_spread
                )[0]
            )
        sixty_forty = _benchmark("60_40")
        out["60_40"].append(
            portfolio_returns(sixty_forty, step_excess, step_cash, financing_spread=financing_spread)[0]
        )
        all_equity = _benchmark("all_us_equity")
        out["all_us_equity"].append(
            portfolio_returns(all_equity, step_excess, step_cash, financing_spread=financing_spread)[0]
        )
        out["cash"].append(step_cash[0])
        dates.append(index[t])

    return pd.DataFrame(out, index=pd.DatetimeIndex(dates))


def _constant_correlation_target(sigma: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf's constant-correlation shrinkage target."""
    vols = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(vols, vols)
    n = corr.shape[0]
    off = (corr.sum() - n) / (n * (n - 1))
    target_corr = np.full((n, n), off)
    np.fill_diagonal(target_corr, 1.0)
    return target_corr * np.outer(vols, vols)


def shrunk_kelly(
    panel: KellyPanel,
    *,
    mu_shrink: float = 0.5,
    cov_shrink: float = 0.2,
) -> pd.Series:
    """Kelly on shrunk inputs -- the only version worth acting on.

    Two priors, both of which say the sample means are largely noise:

    * **Means** are pulled ``mu_shrink`` of the way toward an equal-Sharpe prior
      (``mu_i = sbar * sigma_i``), the natural no-information prior for asset
      classes: it assumes each is compensated in proportion to its risk, and
      declines to bet on one having out-earned another over 30 years. Without it
      the optimiser reads 1993-2026's international underperformance as a
      permanent negative risk premium and shorts the asset class.
    * **Covariance** is pulled ``cov_shrink`` toward constant correlation, which
      stops the optimiser from financing an enormous long/short position out of
      the 0.84 US/international equity correlation and the 0.86 US/international
      bond correlation -- pair correlations estimated on ~150 months whose
      inverse the weights depend on directly.
    """
    excess = panel.excess
    mu = excess.mean().to_numpy() * MONTHS
    sigma = excess.cov().to_numpy() * MONTHS

    vols = np.sqrt(np.diag(sigma))
    grand_sharpe = float(np.mean(mu / vols))
    mu = (1.0 - mu_shrink) * mu + mu_shrink * grand_sharpe * vols
    sigma = (1.0 - cov_shrink) * sigma + cov_shrink * _constant_correlation_target(sigma)

    return pd.Series(np.linalg.solve(sigma, mu), index=ASSETS, name="shrunk_kelly")


def score(
    weights: np.ndarray | pd.Series,
    panel: KellyPanel,
    *,
    financing_spread: float = DEFAULT_SPREAD,
) -> dict:
    """In-sample growth/risk of an arbitrary weight vector on a panel."""
    w = np.asarray(weights, dtype=float)
    excess, cash = _arrays(panel)
    realised = portfolio_returns(w, excess, cash, financing_spread=financing_spread)
    equity = np.cumprod(1.0 + realised)
    drawdown = float((equity / np.maximum.accumulate(equity) - 1.0).min())
    return {
        "growth": growth_rate(w, excess, cash, financing_spread=financing_spread),
        "arith": float(realised.mean() * MONTHS),
        "vol": float(realised.std(ddof=1) * np.sqrt(MONTHS)),
        "sharpe": (
            float((realised - cash).mean() * MONTHS / (realised.std(ddof=1) * np.sqrt(MONTHS)))
            if realised.std(ddof=1) > 0
            else 0.0
        ),
        "max_drawdown": drawdown,
        "worst_month": float(realised.min()),
        "gross": float(np.abs(w).sum()),
    }


def summarise(returns: pd.Series) -> dict:
    """Growth and risk of a realised monthly return stream, ruin included.

    A levered book can lose more than 100% in a month, and once it does every
    conventional statistic downstream is meaningless -- ``(1+r).prod()`` happily
    reports a CAGR for a path that passed through negative wealth. So ruin is
    detected first and reported as a date, not smuggled into a return number.
    """
    wealth = (1.0 + returns).cumprod()
    ruined = wealth <= 0.0
    if ruined.any():
        when = returns.index[ruined.argmax()]
        return {
            "ruined": True,
            "ruin_date": when,
            "months_to_ruin": int(ruined.argmax()) + 1,
            "cagr": float("nan"),
            "log_growth": float("-inf"),
            "vol": float(returns.std(ddof=1) * np.sqrt(MONTHS)),
            "max_drawdown": -1.0,
            "worst_month": float(returns.min()),
        }
    years = len(returns) / MONTHS
    return {
        "ruined": False,
        "ruin_date": None,
        "months_to_ruin": None,
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0),
        "log_growth": float(MONTHS * np.log1p(returns).mean()),
        "vol": float(returns.std(ddof=1) * np.sqrt(MONTHS)),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "worst_month": float(returns.min()),
    }


def constrained_shrunk_kelly(
    panel: KellyPanel,
    *,
    mode: str = "long_only_unlevered",
    mu_shrink: float = 0.5,
    cov_shrink: float = 0.2,
    max_gross: float | None = None,
) -> pd.Series:
    """Maximise the Gaussian log-growth ``w'mu - 0.5 w'Sigma w`` on shrunk inputs,
    subject to the same real-world constraints as :func:`empirical_kelly`.

    This is the version a person can actually hold: it keeps Kelly's objective,
    replaces the sample means with something that is not mostly noise, and honours
    a no-shorting, no-leverage account.
    """
    excess = panel.excess
    mu = excess.mean().to_numpy() * MONTHS
    sigma = excess.cov().to_numpy() * MONTHS
    vols = np.sqrt(np.diag(sigma))
    grand_sharpe = float(np.mean(mu / vols))
    mu = (1.0 - mu_shrink) * mu + mu_shrink * grand_sharpe * vols
    sigma = (1.0 - cov_shrink) * sigma + cov_shrink * _constant_correlation_target(sigma)

    n = len(mu)
    bounds, cons = _bounds_and_constraints(mode, n, max_gross)

    def negative(w: np.ndarray) -> float:
        return -(w @ mu - 0.5 * w @ sigma @ w)

    res = minimize(
        negative,
        np.full(n, 0.25),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return pd.Series(res.x, index=ASSETS, name="constrained_shrunk_kelly")


def fraction_curve(
    panel: KellyPanel,
    *,
    mode: str = "unconstrained",
    max_gross: float | None = None,
    fractions: np.ndarray | None = None,
    min_months: int = 120,
    refit_every: int = 12,
    financing_spread: float = DEFAULT_SPREAD,
) -> pd.DataFrame:
    """Log growth against the Kelly fraction, in sample and out.

    The in-sample curve is the textbook parabola peaking at 1.0x -- that is what
    "optimal" means. The out-of-sample curve is the answer to the only question
    worth asking about it, and the gap between the two peaks is the cost of having
    estimated the inputs from the same data.
    """
    if fractions is None:
        fractions = np.round(np.arange(0.0, 1.51, 0.05), 3)

    excess, cash = _arrays(panel)
    full = empirical_kelly(
        panel, mode=mode, max_gross=max_gross, financing_spread=financing_spread
    ).weights.to_numpy()

    # Each fraction gets its own walk-forward pass: scaling the *weights* is not
    # the same as scaling a realised return stream once financing is charged on
    # the borrowed part only.
    rows = []
    for f in fractions:
        in_sample = growth_rate(full * f, excess, cash, financing_spread=financing_spread)
        oos = walk_forward(
            panel,
            mode=mode,
            fractions=(float(f),),
            min_months=min_months,
            refit_every=refit_every,
            max_gross=max_gross,
            financing_spread=financing_spread,
        )[f"kelly_{f:g}x"]
        stats = summarise(oos)
        rows.append(
            {
                "fraction": float(f),
                "in_sample_growth": in_sample,
                "out_of_sample_growth": stats["log_growth"],
                "out_of_sample_vol": stats["vol"],
                "out_of_sample_max_drawdown": stats["max_drawdown"],
                "ruined": stats["ruined"],
            }
        )
    return pd.DataFrame(rows).set_index("fraction")
