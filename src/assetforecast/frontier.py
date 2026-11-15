"""The efficient frontier the published forecast implies: long-only portfolios, cash and borrowing.

    python -m assetforecast.frontier [--dir DIR]

(without ``--dir``: ``forecasts/latest`` and the dated folder it is a copy of)

Reads a published forecast folder (``cov_annual.csv``, ``expected_returns.csv`` and the
policy weights and origin in ``forecast.json``), the T-bill return of the origin month
(``data/processed/cash_monthly.parquet``) and Robinhood's margin rates
(``data/processed/margin_rates.json``), and writes, into the same folder,

    frontier.csv          portfolios along two frontiers, with their weights
    frontier_points.csv   the reference points: the assets, the policy portfolios, minimum
                          variance, best return per unit of risk, risk parity, and the
                          frontier portfolio at each policy portfolio's volatility with its
                          growth advantage over that portfolio (``gap``)

and a ``frontier`` block (method, settings and the borrowing terms) in ``forecast.json``.
``assetforecast.forecast.write`` calls :func:`publish` on every forecast it writes, so the
frontier always belongs to the covariance and expected returns beside it.

This is arithmetic on the published forecast, not a model and not a tested strategy: it
shows what a mean-variance optimizer makes of the historical means and the covariance
forecast. Nothing here has out-of-sample evidence of its own.

Definitions (everything is a 12-month figure in excess of T-bills; x are weights as a
share of capital, the rest is cash when they sum to less than one and borrowing when more)
-----------------------------------------------------------------------------------------
drift       ``a_i = mean_12m_i + 0.5 cov_annual_ii``: the expected simple-return rate of an
            asset whose expected log return is ``mean_12m_i``.
cost        ``borrow_spread * max(sum(x) - 1, 0)``: borrowing costs T-bills plus the spread
            (see Borrowing).
volatility  ``sqrt(x' cov_annual x)``.
growth      ``x'a - cost - 0.5 x' cov_annual x``: the expected log return of the portfolio
            rebalanced to x (the expected compound growth rate). For a single asset it is
            ``mean_12m`` exactly. For a mix it exceeds ``x'mean_12m``, the approximation the
            published portfolio rows use, by the diversification term
            ``0.5 (sum x_i cov_ii - x' cov x)``.
band        ``growth +/- sum_i x_i (ci90_hi_i - ci90_lo_i) / 2``: the published 90% intervals
            of the assets' means, added as if their errors were perfectly correlated. This is
            the convention of the published portfolio standard error and it is on the wide
            side. It is the range of the growth of that fixed portfolio, and covers sampling
            error of the means only, not error in the covariance forecast.
gap         for a frontier portfolio x matched to a policy portfolio w at the same volatility:
            ``growth(x) - growth(w)``, with the band ``+/- sum_i |x_i - w_i| half_i``, the
            widest difference the assets' 90% intervals allow.

Frontiers
---------
``unlevered``  least variance for each level of drift, long-only and fully invested
               (``x >= 0, sum(x) = 1``), from the minimum-variance portfolio to the asset
               with the highest drift.
``levered``    the same with cash and borrowing allowed (``x >= 0, sum(x) <= MAX_LEVERAGE``)
               and drift counted net of the borrowing cost, from all cash up to the
               volatility of the most volatile asset. Below full investment it is the best
               return-per-risk portfolio mixed with cash; the spread makes it bend at 100%.

At a given volatility the variance term of growth is fixed, so the portfolio with the most
net drift is also the one with the most growth: these are frontiers in growth as well.

Borrowing
---------
Borrowed money is a Robinhood margin loan. ``data/processed/margin_rates.json`` holds
Robinhood's published schedule (one rate on the whole balance, set by the size of the
balance; the date it is stated as of; the day count of its interest, a 360-day year),
downloaded by ``python -m assetforecast.data.sources.margin_rates --refresh``, and
``MARGIN_TIER`` is the balance tier the frontier uses. Returns here are in excess of T-bills, so what enters is the spread

    borrow_spread = margin rate x 365/day_count - 12 x the T-bill return of the origin month

(:func:`borrow_terms`), held for the 12 months: Robinhood's rate floats with the federal
funds target, and T-bills with it. Not counted: the first $1,000 that a Gold subscription
lends free, the compounding of interest charged every 30 days, and a change of tier or of
schedule. ``MAX_LEVERAGE`` is Regulation T's 50% initial margin; Robinhood's maintenance
requirements vary by security and can bind sooner. For an origin before the schedule's date
(development runs) the spread is measured in the schedule's month: today's terms, not the
terms of that day.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import paths
from .universe import BY_ID

__all__ = ["MARGIN_FILE", "MARGIN_TIER", "margin_schedule", "MAX_LEVERAGE", "N_POINTS", "METHOD", "Inputs", "borrow_terms",
           "load_inputs", "drift",
           "stats", "min_variance", "max_sharpe", "risk_parity", "frontier_point", "at_volatility",
           "curve", "build", "publish"]

MARGIN_TIER = 0          # the tier the frontier borrows at: a balance up to $50,000, the dearest
MAX_LEVERAGE = 2.0       # total exposure as a multiple of capital (Regulation T: 50% initial margin)
CASH_FILE = paths.PROCESSED / "cash_monthly.parquet"
MARGIN_FILE = paths.PROCESSED / "margin_rates.json"       # written by assetforecast.data.sources.margin_rates
N_POINTS = 41            # portfolios per frontier
FRONTIER_FILE, POINTS_FILE = "frontier.csv", "frontier_points.csv"
_TOL = 1e-9

METHOD = (
    "Arithmetic on the published forecast, not a model and not a tested strategy. Weights x are shares of "
    "capital, long-only; the rest is cash, or borrowing at T-bills plus borrow_spread when sum(x) > 1, up to "
    "max_leverage. Borrowing is a Robinhood margin loan (block 'borrow'): borrow_spread = margin_rate x "
    "365/day_count - tbill_rate, with margin_rate the published rate of the balance tier used, charged on the "
    "whole balance, and tbill_rate 12 x the T-bill return of tbill_month, the origin month (the schedule's "
    "month for an earlier origin); the spread is held for the 12 months. Not counted: the first $1,000 a Gold "
    "subscription lends free, compounding of the interest charged every 30 days, changes of tier or schedule, "
    "and maintenance requirements tighter than max_leverage. drift_i = mean_12m_i + 0.5 cov_annual_ii. volatility = sqrt(x'cov_annual x). growth = "
    "x'drift - borrow_spread x max(sum(x) - 1, 0) - 0.5 x'cov_annual x, the expected log excess return of the "
    "portfolio rebalanced to x; for one asset it equals mean_12m, for a mix it exceeds x'mean_12m (the "
    "approximation of the published portfolio rows) by 0.5 (sum x_i cov_ii - x'cov x). growth_lo/growth_hi = "
    "growth -/+ sum_i x_i (ci90_hi_i - ci90_lo_i)/2: the 90% intervals of the means added as if perfectly "
    "correlated; sampling error of the means only. Curve 'unlevered': least variance per level of drift with "
    "sum(x) = 1. Curve 'levered': the same with sum(x) <= max_leverage and drift net of the borrowing cost, "
    "from all cash to the volatility of the most volatile asset. Points: min_variance and max_sharpe "
    "(most drift per unit of volatility) are long-only and fully invested; risk_parity gives every asset the "
    "same share of portfolio variance; 'match' rows are the levered frontier at a policy portfolio's volatility, "
    "with gap = its growth minus the policy portfolio's and gap_lo/gap_hi = gap -/+ sum_i |x_i - w_i| "
    "(ci90_hi_i - ci90_lo_i)/2, the widest the 90% intervals of the means allow.")


class Inputs:
    """The published numbers the frontier is made of, aligned on one asset order."""

    def __init__(self, cov: pd.DataFrame, expected: pd.DataFrame, policies: dict[str, dict[str, float]],
                 spread: float, borrow: dict | None = None):
        assets = [a for a in cov.index if a in expected.index]
        if assets != list(cov.index) or list(cov.columns) != assets:
            raise ValueError("cov_annual and expected_returns do not cover the same assets in the same order")
        self.assets = assets
        self.cov = cov.to_numpy(float)
        self.mean = expected.loc[assets, "mean_12m"].to_numpy(float)
        self.half = ((expected["ci90_hi"] - expected["ci90_lo"]) / 2).loc[assets].to_numpy(float)
        self.spread, self.borrow = float(spread), borrow      # over T-bills, a year; and where it comes from
        self.policies = {k: np.array([float(w.get(a, 0.0)) for a in assets]) for k, w in policies.items()}
        if not (np.all(np.linalg.eigvalsh(self.cov) > 0) and np.isfinite(self.mean).all()
                and np.isfinite(self.half).all()):
            raise ValueError("the covariance must be positive definite and every asset needs a mean and interval")
        if not (np.isfinite(self.spread) and self.spread >= 0):
            raise ValueError("the borrowing spread must be a non-negative number")


def _tier_label(lo: float, hi: float | None) -> str:
    return f"${lo:,.0f} or more" if hi is None else f"up to ${hi:,.0f}" if lo == 0 else f"${lo:,.0f} up to ${hi:,.0f}"


def margin_schedule() -> dict:
    """Robinhood's published margin rates as last downloaded (``assetforecast.data.sources.margin_rates``)."""
    return json.loads(MARGIN_FILE.read_text())


def borrow_terms(origin, tier: int = MARGIN_TIER, cash: pd.Series | None = None,
                 schedule: dict | None = None) -> dict:
    """The cost of a Robinhood margin loan at ``origin`` as a spread over T-bills (module docstring).

    ``cash`` is the simple monthly T-bill return by month-end (default: the processed panel's);
    ``schedule`` the margin rates (default: :func:`margin_schedule`).
    """
    MARGIN = margin_schedule() if schedule is None else schedule
    t = MARGIN["tiers"][tier]
    lo, hi, rate = t["from"], t["below"], t["rate"]
    if cash is None:
        cash = pd.read_parquet(CASH_FILE)["cash"]
    month = max(pd.Timestamp(origin), pd.Timestamp(MARGIN["as_of"])) + pd.offsets.MonthEnd(0)
    if month not in cash.index or not np.isfinite(cash.loc[month]):
        raise ValueError(f"no T-bill return for {month.date()}: the borrowing spread cannot be measured")
    tbill = round(12 * float(cash.loc[month]), 6)
    cost = round(rate * 365 / MARGIN["day_count"], 6)
    return {"lender": MARGIN["lender"], "source": MARGIN["source"], "as_of": MARGIN["as_of"],
            "retrieved": MARGIN["retrieved"], "tier": _tier_label(lo, hi), "margin_rate": rate,
            "day_count": MARGIN["day_count"], "annual_cost": cost, "tbill_rate": tbill,
            "tbill_month": str(month.date()), "spread": round(cost - tbill, 6),
            "schedule": [{"balance": _tier_label(t["from"], t["below"]), "rate": t["rate"]} for t in MARGIN["tiers"]]}


def load_inputs(d: Path) -> Inputs:
    d = Path(d)
    cov = pd.read_csv(d / "cov_annual.csv", index_col=0)
    exp = pd.read_csv(d / "expected_returns.csv", index_col=0)
    meta = json.loads((d / "forecast.json").read_text())
    borrow = borrow_terms(meta["origin"])
    return Inputs(cov, exp, meta["portfolio_weights"], borrow["spread"], borrow)


def drift(inp: Inputs) -> np.ndarray:
    return inp.mean + 0.5 * np.diag(inp.cov)


def stats(inp: Inputs, x: np.ndarray, spread: float | None = None) -> dict:
    """Volatility, growth and its band, drift net of cost, and total exposure of weights ``x``."""
    x = np.asarray(x, float)
    spread = inp.spread if spread is None else spread
    var = float(x @ inp.cov @ x)
    net = float(x @ drift(inp)) - spread * max(float(x.sum()) - 1.0, 0.0)
    g, band = net - 0.5 * var, float(x @ inp.half)
    return {"vol": np.sqrt(var), "growth": g, "growth_lo": g - band, "growth_hi": g + band,
            "drift": net, "invested": float(x.sum())}


# ------------------------------------------------------------------ small convex programs
def _solve(fun, jac, x0, bounds, constraints) -> np.ndarray:
    """SLSQP from an interior start (it stalls when started on the boundary); a failed solve is an error."""
    r = minimize(fun, x0, jac=jac, bounds=bounds, constraints=constraints, method="SLSQP",
                 options={"ftol": 1e-13, "maxiter": 2000})
    if not r.success:
        raise RuntimeError(f"frontier optimisation failed: {r.message}")
    return r.x


def _clean(x: np.ndarray) -> np.ndarray:
    x = np.where(x < 1e-7, 0.0, x)
    return x


def _min_var(inp: Inputs, cons: list, x0: np.ndarray, extra: int = 0) -> np.ndarray:
    """Least ``x'cov x`` over ``x >= 0`` (and ``extra`` trailing non-negative slack variables)."""
    n = len(inp.assets)
    S = inp.cov / np.diag(inp.cov).mean()            # scaled so the objective is of order one

    def f(z):
        return float(z[:n] @ S @ z[:n])

    def g(z):
        return np.concatenate([2 * S @ z[:n], np.zeros(extra)])

    return _solve(f, g, x0, [(0.0, None)] * (n + extra), cons)


def min_variance(inp: Inputs) -> np.ndarray:
    n = len(inp.assets)
    x = _min_var(inp, [{"type": "eq", "fun": lambda z: z.sum() - 1, "jac": lambda z: np.ones(n)}],
                 np.full(n, 1 / n))
    return _clean(x) / _clean(x).sum()


def max_sharpe(inp: Inputs) -> np.ndarray:
    """Most drift per unit of volatility, long-only and fully invested."""
    a, n = drift(inp), len(inp.assets)
    if a.max() <= 0:
        raise ValueError("no asset has a positive drift")
    k = int(a.argmax())
    y0 = np.full(n, 1 / max(a.sum(), a[k]))
    y = _clean(_min_var(inp, [{"type": "eq", "fun": lambda z: z @ a - 1, "jac": lambda z: a}], y0))
    return y / y.sum()


def risk_parity(inp: Inputs) -> np.ndarray:
    """Every asset contributes the same share of portfolio variance (long-only, fully invested)."""
    n = len(inp.assets)
    sd = np.sqrt(np.diag(inp.cov))
    R = inp.cov / np.outer(sd, sd)

    def f(u):
        return 0.5 * float(u @ R @ u) - np.log(u).sum() / n

    def g(u):
        return R @ u - 1 / (n * u)

    u = _solve(f, g, np.ones(n), [(1e-8, None)] * n, [])
    x = u / sd
    return x / x.sum()


def frontier_point(inp: Inputs, target: float, levered: bool, spread: float | None = None,
                   max_leverage: float = MAX_LEVERAGE) -> np.ndarray:
    """Least-variance weights with (net) drift at least ``target``.

    ``levered=False``: ``sum(x) = 1``. ``levered=True``: ``sum(x) <= max_leverage`` and the drift
    is net of ``spread`` on the borrowed share, which enters as a slack variable ``b >= sum(x) - 1``.
    """
    a, n = drift(inp), len(inp.assets)
    spread = inp.spread if spread is None else spread
    one, k = np.ones(n), float(np.abs(a).max())      # k scales the drift constraint to order one
    if not levered:
        cons = [{"type": "eq", "fun": lambda z: z.sum() - 1, "jac": lambda z: one},
                {"type": "ineq", "fun": lambda z: (z @ a - target) / k, "jac": lambda z: a / k}]
        return _clean(_min_var(inp, cons, np.full(n, 1 / n)))
    e = np.concatenate([one, [0.0]])
    cons = [{"type": "ineq", "fun": lambda z: z[n] - z[:n].sum() + 1, "jac": lambda z: np.concatenate([-one, [1.0]])},
            {"type": "ineq", "fun": lambda z: max_leverage - z[:n].sum(), "jac": lambda z: -e},
            {"type": "ineq", "fun": lambda z: (z[:n] @ a - spread * z[n] - target) / k,
             "jac": lambda z: np.concatenate([a, [-spread]]) / k}]
    start = np.concatenate([np.full(n, 1 / n), [0.5]])
    return _clean(_min_var(inp, cons, start, extra=1)[:n])


def _max_target(inp: Inputs, levered: bool, spread: float | None, max_leverage: float) -> float:
    a = drift(inp)
    spread = inp.spread if spread is None else spread
    top = float(a.max())
    return max(top, max_leverage * top - spread * (max_leverage - 1)) if levered else top


def at_volatility(inp: Inputs, vol: float, levered: bool = True, spread: float | None = None,
                  max_leverage: float = MAX_LEVERAGE) -> np.ndarray:
    """The frontier portfolio whose volatility is ``vol`` (bisection on the drift target)."""
    lo = 0.0 if levered else float(drift(inp) @ min_variance(inp))
    hi = _max_target(inp, levered, spread, max_leverage)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        x = frontier_point(inp, mid, levered, spread, max_leverage)
        if np.sqrt(x @ inp.cov @ x) < vol:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    return frontier_point(inp, 0.5 * (lo + hi), levered, spread, max_leverage)


def curve(inp: Inputs, levered: bool, n_points: int = N_POINTS, spread: float | None = None,
          max_leverage: float = MAX_LEVERAGE) -> list[np.ndarray]:
    """Weights along a frontier, evenly spaced in volatility."""
    top_vol = float(np.sqrt(np.diag(inp.cov)).max())
    if levered:
        vols = np.linspace(0.0, top_vol, n_points)
        return [np.zeros(len(inp.assets))] + [at_volatility(inp, v, True, spread, max_leverage) for v in vols[1:]]
    x_min = min_variance(inp)
    k = int(drift(inp).argmax())
    x_top = np.zeros(len(inp.assets))
    x_top[k] = 1.0
    v0, v1 = np.sqrt(x_min @ inp.cov @ x_min), np.sqrt(inp.cov[k, k])
    vols = np.linspace(v0, v1, n_points)
    return [x_min] + [at_volatility(inp, v, False) for v in vols[1:-1]] + [x_top]


# ------------------------------------------------------------------ tables
def _row(inp: Inputs, x: np.ndarray, spread: float | None, **head) -> dict:
    return {**head, **stats(inp, x, spread), **{f"w_{a}": float(w) for a, w in zip(inp.assets, x)}}


def build(inp: Inputs, spread: float | None = None, max_leverage: float = MAX_LEVERAGE,
          n_points: int = N_POINTS) -> dict:
    """``{"frontier": DataFrame, "points": DataFrame, "meta": dict}`` for one forecast."""
    spread = inp.spread if spread is None else spread
    rows = []
    for name, lev in (("unlevered", False), ("levered", True)):
        for i, x in enumerate(curve(inp, lev, n_points, spread, max_leverage)):
            rows.append(_row(inp, x, spread, curve=name, i=i))
    frontier = pd.DataFrame(rows)

    n = len(inp.assets)
    pts = [_row(inp, np.eye(n)[i], spread, point=a, kind="asset", name=BY_ID[a].name if a in BY_ID else a)
           for i, a in enumerate(inp.assets)]
    for k, w in inp.policies.items():
        pts.append(_row(inp, w, spread, point=k, kind="policy", name=k))
    specials = {"min_variance": min_variance(inp), "max_sharpe": max_sharpe(inp), "risk_parity": risk_parity(inp)}
    for k, w in specials.items():
        pts.append(_row(inp, w, spread, point=k, kind="special", name=k))
    for k, w in inp.policies.items():
        x = at_volatility(inp, float(np.sqrt(w @ inp.cov @ w)), True, spread, max_leverage)
        row = _row(inp, x, spread, point=f"match_{k}", kind="match", name=k)
        gap, band = row["growth"] - stats(inp, w, spread)["growth"], float(np.abs(x - w) @ inp.half)
        pts.append({**row, "gap": gap, "gap_lo": gap - band, "gap_hi": gap + band})
    points = pd.DataFrame(pts)
    head = ["point", "kind", "name", *stats(inp, np.zeros(n), spread), "gap", "gap_lo", "gap_hi"]
    points = points[head + [c for c in points.columns if c.startswith("w_")]]
    meta = {"method": METHOD, "borrow_spread": spread, "max_leverage": max_leverage, "n_points": n_points,
            "interval_level": 0.90, "files": [FRONTIER_FILE, POINTS_FILE]}
    if inp.borrow is not None and spread == inp.spread:
        meta["borrow"] = inp.borrow
    return {"frontier": frontier, "points": points, "meta": meta}


def publish(d: Path) -> dict:
    """Write the frontier of the forecast published in ``d`` into ``d`` and record it in ``forecast.json``."""
    d = Path(d)
    res = build(load_inputs(d))
    res["frontier"].round(5).to_csv(d / FRONTIER_FILE, index=False)
    res["points"].round(5).to_csv(d / POINTS_FILE, index=False)
    meta = json.loads((d / "forecast.json").read_text())
    meta.pop("frontier", None)
    meta["frontier"] = res["meta"]
    (d / "forecast.json").write_text(json.dumps(meta, indent=1))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m assetforecast.frontier", description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, default=None,
                    help="a published forecast folder (default: forecasts/latest and the dated folder it copies)")
    a = ap.parse_args(argv)
    if a.dir is None:
        a.dir = paths.FORECASTS_OUT / "latest"
        dated = paths.FORECASTS_OUT / json.loads((a.dir / "forecast.json").read_text())["origin"]
        if dated.is_dir():
            publish(dated)
    res = publish(a.dir)
    pd.set_option("display.width", 200)
    cols = ["point", "kind", "vol", "growth", "growth_lo", "growth_hi", "invested"]
    print(f"frontier of {a.dir} -> {FRONTIER_FILE}, {POINTS_FILE}")
    b = res["meta"]["borrow"]
    print(f"borrowing: {b['lender']}, balance {b['tier']}, {b['margin_rate']:.2%} as of {b['as_of']} "
          f"= T-bills ({b['tbill_rate']:.2%}, {b['tbill_month']}) + {b['spread']:.2%}")
    print(res["points"][cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
