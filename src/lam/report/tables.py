"""Formatted tables for the console and the markdown report."""

from __future__ import annotations

import pandas as pd

# Columns where True genuinely means "passed"; everything else boolean reads
# as yes/no so a False cannot masquerade as a failure.
PASS_COLUMNS = {"pass", "passes", "feasible", "found_feasible", "levered"}

PCT = {"cagr", "vol", "max_dd", "strat_cagr", "bench_cagr", "strat_vol", "strat_dd",
       "bench_dd", "excess_dd", "strat_return", "bench_return", "worst_12m",
       "underwater_frac", "alpha", "worst_excess", "cvar5_excess", "sigma_target",
       "oos_cagr", "oos_dd", "is_cagr", "var_95", "cvar_95", "worst_day"}


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


def frontier_table(results: pd.DataFrame, tolerances=(0.0, 0.01, 0.02, 0.03, 0.05)) -> pd.DataFrame:
    """Best achievable CAGR as the rolling-window tolerance is relaxed.

    The full-sample condition stays strict throughout; only the rolling-3y test
    is loosened. This isolates how much of the return ceiling is imposed by the
    rolling requirement rather than by crash protection.
    """
    rows = []
    clean = results.dropna(subset=["cagr", "worst_excess"])
    for tol in tolerances:
        ok = clean[clean["worst_excess"] <= tol]
        if "bench_dd" in clean.columns:
            # Full-sample condition: strategy depth <= benchmark depth, i.e. the
            # (negative) max drawdown is no lower than the benchmark's.
            ok = ok[ok["max_dd"] >= ok["bench_dd"]]
        rows.append(
            {
                "tolerance_pp": tol * 100,
                "n_feasible": int(len(ok)),
                "best_cagr": float(ok["cagr"].max()) if len(ok) else float("nan"),
                "best_max_dd": float(ok.loc[ok["cagr"].idxmax(), "max_dd"]) if len(ok) else float("nan"),
                "best_sharpe": float(ok.loc[ok["cagr"].idxmax(), "sharpe"]) if len(ok) else float("nan"),
                "best_avg_lev": float(ok.loc[ok["cagr"].idxmax(), "avg_leverage"]) if len(ok) else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("tolerance_pp")
