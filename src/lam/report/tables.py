"""Formatted tables for the console and the markdown report."""

from __future__ import annotations

import pandas as pd

# Columns where True genuinely means "passed"; everything else boolean reads
# as yes/no so a False cannot masquerade as a failure.
PASS_COLUMNS = {"pass", "passes", "feasible", "found_feasible", "levered"}

PCT = {"cagr", "vol", "max_dd", "strat_cagr", "bench_cagr", "strat_vol", "strat_dd",
       "bench_dd", "excess_dd", "strat_return", "bench_return", "worst_12m",
       "underwater_frac", "alpha", "worst_excess", "cvar5_excess", "sigma_target",
       "oos_cagr", "oos_dd", "is_cagr", "var_95", "cvar_95", "worst_day",
       # Return-maximising model.
       "asset_cagr", "asset_dd", "excess_cagr", "half_sample_cagr", "dd_ceiling",
       "excess_return", "share", "asset return", "strategy return",
       "share of days", "realised_cagr", "matched_cagr", "strategy_cagr",
       "target_vol", "matched_vol", "strategy_dd", "matched_dd",
       "index_cagr", "index_dd", "share_of_total", "implied_spread",
       "wrapper_cost_ann", "cagr_sim", "cagr_actual", "te_ann", "expense_ratio"}


def fmt(df: pd.DataFrame, decimals: int = 2) -> str:
    out = df.copy()
    for col in out.columns:
        if col in PCT and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: f"{v:.{decimals}%}" if pd.notna(v) else "-")
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: f"{v:,.{decimals}f}" if pd.notna(v) else "-")
        elif pd.api.types.is_bool_dtype(out[col]):
            # Only genuine pass/fail columns get PASS/FAIL. Rendering
            # "ruined=False" as "FAIL" inverts its meaning.
            if col in PASS_COLUMNS:
                out[col] = out[col].map(lambda v: "PASS" if v else "FAIL")
            else:
                out[col] = out[col].map(lambda v: "yes" if v else "no")
    return out.to_string()


def to_markdown(df: pd.DataFrame, decimals: int = 2) -> str:
    out = df.copy()
    for col in out.columns:
        if col in PCT and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: f"{v:.{decimals}%}" if pd.notna(v) else "-")
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: f"{v:,.{decimals}f}" if pd.notna(v) else "-")
        elif pd.api.types.is_bool_dtype(out[col]):
            # Only genuine pass/fail columns get PASS/FAIL. Rendering
            # "ruined=False" as "FAIL" inverts its meaning.
            if col in PASS_COLUMNS:
                out[col] = out[col].map(lambda v: "PASS" if v else "FAIL")
            else:
                out[col] = out[col].map(lambda v: "yes" if v else "no")
    return out.to_markdown()


def summary_row(name: str, stats: dict, constraint=None) -> dict:
    row = {
        "strategy": name,
        "cagr": stats.get("cagr"),
        "vol": stats.get("vol"),
        "sharpe": stats.get("sharpe"),
        "max_dd": stats.get("max_dd"),
        "calmar": stats.get("calmar"),
        "ulcer": stats.get("ulcer"),
        "avg_lev": stats.get("avg_leverage"),
    }
    if constraint is not None:
        row |= {"worst_excess": constraint.worst_excess, "passes": constraint.passed}
    return row


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("strategy")


def frontier_table(
    results: pd.DataFrame,
    tolerances=(0.0, 0.01, 0.02, 0.03, 0.05),
    *,
    benchmark_dd: float | None = None,
) -> pd.DataFrame:
    """Best achievable CAGR as the rolling-window tolerance is relaxed.

    The full-sample condition stays strict throughout; only the rolling-3y test
    is loosened. This isolates how much of the return ceiling comes from the
    rolling requirement rather than from crash protection -- which turns out to
    be almost all of it, because the S&P's *calmest* three-year windows, not its
    crashes, are what cap leverage.

    The final row drops the rolling test entirely, giving the looser reading of
    "drawdowns not exceeding the S&P 500" (worst peak-to-trough only).
    """
    rows = []
    clean = results.dropna(subset=["cagr", "worst_excess"])

    def full_sample_ok(df: pd.DataFrame) -> pd.DataFrame:
        if "bench_dd" in df.columns:
            return df[df["max_dd"] >= df["bench_dd"]]
        if benchmark_dd is not None:
            return df[df["max_dd"] >= benchmark_dd]
        return df

    cases = [(f"{tol * 100:.0f}", clean[clean["worst_excess"] <= tol]) for tol in tolerances]
    cases.append(("full-sample only", clean))

    for label, subset in cases:
        ok = full_sample_ok(subset)
        best = ok.loc[ok["cagr"].idxmax()] if len(ok) else None
        rows.append(
            {
                "rolling_tolerance_pp": label,
                "n_feasible": int(len(ok)),
                "best_cagr": float(best["cagr"]) if best is not None else float("nan"),
                "best_max_dd": float(best["max_dd"]) if best is not None else float("nan"),
                "best_sharpe": float(best["sharpe"]) if best is not None else float("nan"),
                "best_avg_lev": float(best["avg_leverage"]) if best is not None else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("rolling_tolerance_pp")
