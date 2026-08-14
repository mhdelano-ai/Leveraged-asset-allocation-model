"""Out-of-sample validation. This is where the project earns its conclusions.

Satisfying a drawdown constraint *in sample* is close to meaningless: the
degenerate solution is zero leverage, and any search with enough parameters will
find a set that happens to dodge the specific crashes present in the sample. The
statistics that matter are all here.

The most important number in the whole study is not a return -- it is the
effective sample size. 1971-2026 contains ~13,900 trading days but only about
**seven independent major drawdown events** (1973-74, 1980-82, 1987, 2000-02,
2007-09, 2020, 2022). For a constraint defined on drawdowns, n is 7, not 13,900.
Every technique here exists to keep that number in view.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..alloc.stack import StackParams
from ..data.panels import Panel
from ..metrics.constraint import evaluate
from ..metrics.core import ann_vol, cagr, sharpe
from ..metrics.drawdown import max_drawdown
from ..pipeline import execute

# Named stress episodes. Dates are generous at the edges so an episode is not
# missed by a few days of timing difference.
STRESS_PERIODS = [
    ("1973-74 oil shock", "1973-01-01", "1974-12-31"),
    ("1970s stagflation", "1972-01-01", "1982-12-31"),
    ("1980-82 Volcker", "1980-01-01", "1982-12-31"),
    ("1987 crash", "1987-08-01", "1987-12-31"),
    ("Black Monday only", "1987-10-16", "1987-10-23"),
    ("1990 recession", "1990-06-01", "1990-12-31"),
    ("1994 bond massacre", "1994-01-01", "1994-12-31"),
    ("1998 LTCM", "1998-07-01", "1998-10-31"),
    ("2000-02 dotcom", "2000-03-01", "2002-12-31"),
    ("2007-09 GFC", "2007-10-01", "2009-03-31"),
    ("2011 EU crisis", "2011-05-01", "2011-12-31"),
    ("2013 taper tantrum", "2013-05-01", "2013-09-30"),
    ("2015-16 China", "2015-06-01", "2016-02-29"),
    ("Q4 2018", "2018-10-01", "2018-12-31"),
    ("2020 COVID", "2020-02-01", "2020-04-30"),
    ("2022 inflation", "2022-01-01", "2022-12-31"),
    ("2023-26 recent", "2023-01-01", "2026-12-31"),
]


def stress_table(strategy: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Strategy versus benchmark drawdown in each named episode."""
    rows = []
    for label, start, end in STRESS_PERIODS:
        s = strategy.loc[start:end]
        b = benchmark.loc[start:end]
        if len(s) < 5 or len(b) < 5:
            continue
        s_dd, b_dd = max_drawdown(s), max_drawdown(b)
        rows.append(
            {
                "episode": label,
                "start": start[:7],
                "days": len(s),
                "strat_return": float(np.prod(1 + s) - 1),
                "bench_return": float(np.prod(1 + b) - 1),
                "strat_dd": s_dd,
                "bench_dd": b_dd,
                "excess_dd": (-s_dd) - (-b_dd),
                "pass": s_dd >= b_dd,
            }
        )
    return pd.DataFrame(rows).set_index("episode")


def regime_table(
    strategy: pd.Series, benchmark: pd.Series, rf: pd.Series
) -> pd.DataFrame:
    """Performance split by rate regime and by stock/bond correlation regime.

    The correlation split is the one that matters most. Risk parity's whole edge
    depends on negative stock/bond correlation, which is a post-2000 phenomenon
    -- the 1970s through 1990s had it firmly *positive*. A strategy that passes
    only in the negative-correlation regime is fitted to a sign that has already
    flipped once and flipped back in 2022.
    """
    rows = []
    rf_change = rf.rolling(252).mean().diff(252)
    for label, mask in [
        ("rising rates", rf_change > 0.005),
        ("falling rates", rf_change < -0.005),
        ("high rates (>5%)", rf > 0.05),
        ("low rates (<3%)", rf < 0.03),
        ("pre-1990", pd.Series(strategy.index < pd.Timestamp("1990-01-01"), index=strategy.index)),
        ("1990-2010", pd.Series(
            (strategy.index >= pd.Timestamp("1990-01-01"))
            & (strategy.index < pd.Timestamp("2010-01-01")), index=strategy.index)),
        ("2010+", pd.Series(strategy.index >= pd.Timestamp("2010-01-01"), index=strategy.index)),
    ]:
        m = mask.reindex(strategy.index).fillna(False)
        s, b = strategy[m], benchmark.reindex(strategy.index)[m]
        if len(s) < 250:
            continue
        rows.append(
            {
                "regime": label,
                "days": len(s),
                "strat_cagr": cagr(s),
                "bench_cagr": cagr(b),
                "strat_vol": ann_vol(s),
                "strat_dd": max_drawdown(s),
                "bench_dd": max_drawdown(b),
                "pass": max_drawdown(s) >= max_drawdown(b),
            }
        )
    return pd.DataFrame(rows).set_index("regime")


def walk_forward(
    panel: Panel,
    candidates: list[StackParams],
    *,
    initial_train_years: int = 20,
    step_years: int = 2,
    tolerance: float = 0.0,
    constraint_step: int = 21,
) -> tuple[pd.Series, pd.DataFrame]:
    """Anchored walk-forward. Returns the stitched OOS series and a fold table.

    On each fold the best *feasible* candidate on the training window is chosen,
    then applied unchanged to the next ``step_years``. The stitched result is the
    headline number -- never the in-sample optimum.
    """
    index = panel.returns.index
    start = index[0]
    folds = []
    oos_parts = []

    train_end = start + pd.DateOffset(years=initial_train_years)
    while train_end < index[-1]:
        test_end = min(train_end + pd.DateOffset(years=step_years), index[-1])

        best, best_cagr = None, -np.inf
        for params in candidates:
            out = execute(panel, params, constraint_step=constraint_step)
            r = out.result.returns.loc[:train_end]
            b = panel.benchmark.loc[:train_end]
            if len(r) < 500:
                continue
            c = evaluate(r, b, step=constraint_step)
            feasible = c.worst_excess <= tolerance and c.strategy_depth <= c.benchmark_depth + tolerance
            score = cagr(r)
            if feasible and score > best_cagr:
                best, best_cagr = params, score

        if best is None:
            # No candidate was feasible in training; stand aside for this fold
            # rather than deploying a known-infeasible parameter set.
            chosen = replace(candidates[0], sigma_target=0.02, max_leverage=1.0)
        else:
            chosen = best

        out = execute(panel, chosen, constraint_step=constraint_step)
        segment = out.result.returns.loc[train_end:test_end]
        if len(segment):
            oos_parts.append(segment)
            folds.append(
                {
                    "train_end": train_end.date(),
                    "test_end": test_end.date(),
                    "days": len(segment),
                    "oos_cagr": cagr(segment),
                    "oos_dd": max_drawdown(segment),
                    "bench_dd": max_drawdown(panel.benchmark.loc[train_end:test_end]),
                    "is_cagr": best_cagr if best is not None else np.nan,
                    "sigma_target": chosen.sigma_target,
                    "max_leverage": chosen.max_leverage,
                    "dd_margin": chosen.dd_budget_margin,
                    "found_feasible": best is not None,
                }
            )
        train_end = test_end

    stitched = pd.concat(oos_parts).sort_index() if oos_parts else pd.Series(dtype=float)
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    return stitched.rename("oos"), pd.DataFrame(folds)


def walk_forward_growth(
    panel: Panel,
    candidates: list,
    *,
    initial_train_years: int = 15,
    step_years: int = 2,
    dd_ceiling: float = 0.50,
) -> tuple[pd.Series, pd.DataFrame]:
    """Anchored walk-forward for the return-maximising objective.

    On each fold the candidate with the highest training-window compound return
    *that stayed inside the drawdown ceiling* is selected, then applied unchanged
    to the next ``step_years``. The stitched out-of-sample series is the headline
    result; the in-sample optimum never is.

    Each candidate is executed **once** over the full panel and then sliced.
    That is legitimate precisely because every signal is causal: the returns on
    ``[start, train_end]`` depend on no data after ``train_end``, so slicing a
    full run gives the same numbers as running only the training window -- at a
    fraction of the cost.

    The honest limitation is the same one the constrained model carries: the
    candidate pool is produced by a scan over the full sample, so the pool embeds
    full-sample information even though the per-fold choice does not. This
    measures parameter **stability**, not true out-of-sample discovery.
    """
    from ..pipeline import execute_growth

    runs = []
    for params in candidates:
        out = execute_growth(panel, params)
        runs.append((params, out.result.returns))

    index = panel.returns.index
    train_end = index[0] + pd.DateOffset(years=initial_train_years)
    folds, oos_parts = [], []

    while train_end < index[-1]:
        test_end = min(train_end + pd.DateOffset(years=step_years), index[-1])

        best, best_cagr, best_returns = None, -np.inf, None
        for params, returns in runs:
            train = returns.loc[:train_end]
            if len(train) < 500:
                continue
            if max_drawdown(train) < -abs(dd_ceiling):
                continue
            score = cagr(train)
            if score > best_cagr:
                best, best_cagr, best_returns = params, score, returns

        if best is None:
            # Nothing cleared the ceiling in training. Standing aside is the only
            # defensible action -- deploying a candidate known to breach it would
            # make the out-of-sample number meaningless.
            train_end = test_end
            continue

        segment = best_returns.loc[train_end:test_end]
        if len(segment):
            oos_parts.append(segment)
            folds.append(
                {
                    "train_end": train_end.date(),
                    "test_end": test_end.date(),
                    "days": len(segment),
                    "oos_cagr": cagr(segment),
                    "oos_dd": max_drawdown(segment),
                    "asset_cagr": cagr(panel.benchmark.loc[train_end:test_end]),
                    "asset_dd": max_drawdown(panel.benchmark.loc[train_end:test_end]),
                    "is_cagr": best_cagr,
                    "sigma_target": best.sigma_target,
                    "max_leverage": best.max_leverage,
                    "trend_floor": best.trend_floor,
                    "stress_cap": best.stress_cap,
                    "dd_budget": best.dd_budget,
                }
            )
        train_end = test_end

    stitched = pd.concat(oos_parts).sort_index() if oos_parts else pd.Series(dtype=float)
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    return stitched.rename("oos"), pd.DataFrame(folds)


def edge_attribution(
    strategy: pd.Series,
    reference: pd.Series,
    *,
    episodes=STRESS_PERIODS,
) -> pd.DataFrame:
    """Where a strategy's advantage over a reference book was actually earned.

    An outperformance figure spread over forty years and one earned entirely in a
    single two-year window are the same number and completely different claims.
    This decomposes the total log-return edge by episode, so a result that rests
    on one crisis says so out loud.

    The reference should be the volatility-matched constant-leverage book, not
    the unlevered index -- otherwise the "edge" is mostly just leverage.
    """
    joined = pd.concat(
        [strategy.rename("s"), reference.rename("r")], axis=1, join="inner"
    ).dropna()
    edge = np.log1p(joined["s"]) - np.log1p(joined["r"])
    total = float(edge.sum())
    # Shares of a negative total invert every sign and read as nonsense -- a
    # window that lost money shows up as a positive "share of the edge". When
    # there is no edge to apportion, report the contributions and nothing else.
    share_is_meaningful = total > 1e-9

    def _share(value: float) -> float:
        return value / total if share_is_meaningful else np.nan

    rows = []
    covered = pd.Series(False, index=edge.index)
    for label, start, end in episodes:
        window = edge.loc[start:end]
        if len(window) < 5:
            continue
        # "1970s stagflation" nests other episodes; only disjoint windows are
        # marked as covered so the residual line stays meaningful.
        rows.append(
            {
                "episode": label,
                "days": len(window),
                "edge_log": float(window.sum()),
                "share_of_total": _share(float(window.sum())),
            }
        )
        covered.loc[window.index] = True

    rows.append(
        {
            "episode": "all other days",
            "days": int((~covered).sum()),
            "edge_log": float(edge[~covered].sum()),
            "share_of_total": float(edge[~covered].sum()) / total if total else np.nan,
        }
    )
    rows.append(
        {"episode": "TOTAL", "days": len(edge), "edge_log": total, "share_of_total": 1.0}
    )
    return pd.DataFrame(rows).set_index("episode")


def edge_excluding(
    strategy: pd.Series,
    reference: pd.Series,
    windows: list[tuple[str, str]],
) -> dict:
    """Annualised edge with named date ranges removed from both series.

    The blunt version of :func:`edge_attribution`: drop the crisis and see what
    is left. If the answer is nothing, the strategy is a bet that the next
    drawdown looks like the last one.
    """
    joined = pd.concat(
        [strategy.rename("s"), reference.rename("r")], axis=1, join="inner"
    ).dropna()
    mask = pd.Series(True, index=joined.index)
    for start, end in windows:
        mask.loc[start:end] = False
    kept = joined[mask]
    years = len(kept) / 252.0
    if years <= 0:
        return {}
    s_g = float(np.log1p(kept["s"]).sum()) / years
    r_g = float(np.log1p(kept["r"]).sum()) / years
    return {
        "days_kept": int(len(kept)),
        "days_dropped": int((~mask).sum()),
        "strategy_log_growth": s_g,
        "reference_log_growth": r_g,
        "edge_bps": (s_g - r_g) * 1e4,
    }


def deep_history_scenario(
    params,
    *,
    start: str = "1928-01-03",
    end: str = "1945-12-31",
    assumed_rf: float = 0.02,
) -> dict:
    """What the rule would have done through 1929-32, as a labelled counterfactual.

    The S&P panel begins in 1960 because that is where a daily T-bill series
    begins, and a levered book without a measured financing rate is not a
    backtest. But 1929-32 is the deepest drawdown in the record -- -86% -- and a
    return-maximising leverage rule that has never been shown against it is
    untested where it matters most.

    So this is run separately and labelled for what it is: the price and dividend
    data are real, the **financing rate is assumed constant** at ``assumed_rf``,
    and short rates in that era in fact ranged from roughly 5% in 1929 to near
    zero through the later 1930s. Read the drawdown, not the return.
    """
    from ..alloc.growth import GrowthParams
    from ..data import panels as panels_mod
    from ..data import synth_equity
    from ..pipeline import execute_growth

    params = params or GrowthParams()
    series = synth_equity.reconstructed_tr_returns().loc[start:end]
    rf = pd.Series(assumed_rf, index=series.index)

    panel = panels_mod.Panel(
        name="1929",
        returns=series.rename("sp500").to_frame(),
        benchmark=series.rename("benchmark"),
        rf=rf,
        eligible=pd.DataFrame(True, index=series.index, columns=["sp500"]),
        description=f"Counterfactual: real prices, assumed flat {assumed_rf:.0%} financing.",
    )
    out = execute_growth(panel, params)
    return {
        "start": str(series.index[0].date()),
        "end": str(series.index[-1].date()),
        "assumed_rf": assumed_rf,
        "strategy_cagr": out.stats["cagr"],
        "strategy_dd": out.stats["max_dd"],
        "index_cagr": cagr(series),
        "index_dd": max_drawdown(series),
        "avg_leverage": out.stats["avg_leverage"],
        "max_leverage_used": out.stats["max_leverage_used"],
        "ruined": out.stats["ruined"],
        "returns": out.result.returns,
    }


def deflated_sharpe(observed_sharpe: float, n_trials: int, n_obs: int) -> dict:
    """Haircut a Sharpe ratio for the number of configurations tried.

    With thousands of trials the best in-sample Sharpe is inflated even when
    every candidate is worthless. This reports the expected maximum under the
    null so the achieved figure can be read against it.
    """
    from scipy.stats import norm

    if n_trials < 2 or n_obs < 2:
        return {"observed": observed_sharpe, "expected_max_null": 0.0,
                "excess_over_null": observed_sharpe, "n_trials": n_trials}
    euler = 0.5772156649
    # Expected maximum of n_trials standard normals.
    e_max_z = (1 - euler) * norm.ppf(1 - 1.0 / n_trials) + euler * norm.ppf(
        1 - 1.0 / (n_trials * np.e)
    )
    # A per-period Sharpe estimated from n_obs observations has standard error
    # ~1/sqrt(n_obs); annualising multiplies by sqrt(252). Applying sqrt(252)
    # twice (once inside a sqrt(252/n_obs) factor and again outside) inflates
    # this by ~16x and produces absurd thresholds like 8.
    e_max_ann = e_max_z / np.sqrt(n_obs) * np.sqrt(252.0)
    return {
        "observed": observed_sharpe,
        "n_trials": n_trials,
        "n_obs": n_obs,
        "expected_max_null": float(e_max_ann),
        "excess_over_null": float(observed_sharpe - e_max_ann),
    }


def permutation_test(
    panel: Panel,
    params: StackParams,
    *,
    n: int = 200,
    seed: int = 0,
) -> dict:
    """Does trend carry information, or is it just a leverage-timing artefact?

    Shuffles the trend signal's dates while leaving the vol-target and throttle
    machinery intact, building a null distribution of CAGR.
    """
    from ..pipeline import stack_for
    from ..engine.backtest import run
    from ..pipeline import engine_config
    from ..alloc.dd_throttle import DrawdownThrottle
    from ..metrics.constraint import design_budget

    signals = stack_for(panel, params)
    actual = execute(panel, params)
    rng = np.random.default_rng(seed)
    budget = design_budget(panel.benchmark, margin=params.dd_budget_margin)

    null_cagrs = []
    for _ in range(n):
        perm = rng.permutation(len(signals.base_weights))
        shuffled = pd.DataFrame(
            signals.base_weights.to_numpy()[perm],
            index=signals.base_weights.index,
            columns=signals.base_weights.columns,
        )
        res = run(
            panel.returns, shuffled, signals.target_leverage,
            rf_annual=panel.rf, config=engine_config(panel, params),
            throttle=DrawdownThrottle(budget=budget),
        )
        null_cagrs.append(cagr(res.returns))

    null = np.array(null_cagrs)
    observed = actual.stats["cagr"]
    return {
        "observed_cagr": observed,
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "percentile": float((null < observed).mean()),
        "n": n,
    }


def cost_sensitivity(panel: Panel, params: StackParams, multiples=(1.0, 2.0, 3.0)) -> pd.DataFrame:
    """Re-run at inflated transaction and financing costs."""
    rows = []
    for m in multiples:
        out = execute(
            panel, params,
            cost_per_unit=panel.costs * m,
            borrow_spread=0.0050 * m,
        )
        rows.append(
            {
                "cost_multiple": m,
                "cagr": out.stats["cagr"],
                "max_dd": out.stats["max_dd"],
                "sharpe": out.stats["sharpe"],
                "passes": out.constraint.passed,
            }
        )
    return pd.DataFrame(rows).set_index("cost_multiple")
