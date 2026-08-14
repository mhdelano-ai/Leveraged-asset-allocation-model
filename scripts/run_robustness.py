"""Robustness checks for the drawdown-constrained model.

    python scripts/run_robustness.py --scan output/scan_panelL.parquet

These numbers were originally produced ad hoc, which meant the README could drift
away from the code. They live in a script now so they can be regenerated with
everything else.

Four questions, and the answers are the uncomfortable part of Model A:

1. **Is the solution a plateau or a spike?** How far every parameter can move
   before feasibility is lost, and which axis breaks first.
2. **What does safety margin cost?** Best CAGR when the constraint must be
   cleared by a genuine margin rather than by a basis point.
3. **Does it hold on a different universe?** Panel L parameters run on X and M.
4. **Does it survive higher costs?**
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lam.alloc.stack import StackParams
from lam.data import panels
from lam.opt import search, validate
from lam.report import tables
from lam import pipeline

PARAM_FIELDS = list(StackParams.__dataclass_fields__)


def _load(path: str) -> StackParams:
    data = json.loads(Path(path).read_text())
    return StackParams(**{k: float(v) for k, v in data.items() if k in PARAM_FIELDS})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="output/scan_panelL.parquet")
    ap.add_argument("--robust", default="docs/robust_params.json")
    ap.add_argument("--fullsample", default="docs/fullsample_params.json")
    args = ap.parse_args()

    panel = panels.build("L")
    robust, full = _load(args.robust), _load(args.fullsample)
    print(panel, "\n")

    # ------------------------------------------------------------- plateau
    print("=" * 78)
    print("1. PLATEAU OR SPIKE")
    print("=" * 78)
    scan_path = Path(args.scan)
    rows = []
    named = {"robust (committed)": robust, "full-sample (committed)": full}
    if scan_path.exists():
        df = pd.read_parquet(scan_path)
        feasible = df[df["feasible"]].sort_values("cagr", ascending=False)
        if len(feasible):
            best = feasible.iloc[0]
            named["best in-sample"] = StackParams(
                **{k: float(best[k]) for k in PARAM_FIELDS if k in best}
            )
    for label, params in named.items():
        out = pipeline.execute(panel, params, constraint_step=21)
        radius, binding = search.plateau_radius_detail(panel, params, constraint_step=21)
        rows.append(
            {
                "parameters": label,
                "cagr": out.stats["cagr"],
                "max_dd": out.stats["max_dd"],
                "worst_excess": out.constraint.worst_excess,
                "slack_pp": -out.constraint.worst_excess * 100,
                "plateau_radius": radius,
                "breaks_first": binding or "-",
            }
        )
    print(tables.fmt(pd.DataFrame(rows).set_index("parameters")))
    print(
        "\n  A point that is feasible only in a narrow spike is an artefact. The\n"
        "  parameter that breaks first is the more useful diagnostic: if feasibility\n"
        "  dies the moment the risk target rises, the configuration is not a robust\n"
        "  design, it is sitting exactly on the constraint boundary."
    )

    # -------------------------------------------------------------- margin
    print("\n" + "=" * 78)
    print("2. WHAT SAFETY MARGIN COSTS")
    print("=" * 78)
    if scan_path.exists():
        df = pd.read_parquet(scan_path)
        rows = []
        for margin in (0.0, 0.01, 0.02, 0.03, 0.05):
            pool = df[
                (df["worst_excess"] <= -margin)
                & (df["max_dd"] >= df["bench_dd"] + margin)
                & df["cagr"].notna()
            ]
            if pool.empty:
                rows.append({"required_margin_pp": margin * 100, "n_feasible": 0})
                continue
            best = pool.loc[pool["cagr"].idxmax()]
            rows.append(
                {
                    "required_margin_pp": margin * 100,
                    "n_feasible": len(pool),
                    "cagr": best["cagr"],
                    "max_dd": best["max_dd"],
                    "avg_leverage": best["avg_leverage"],
                    "sigma_target": best["sigma_target"],
                }
            )
        print(tables.fmt(pd.DataFrame(rows).set_index("required_margin_pp")))
    else:
        print(f"  no scan at {scan_path}")

    # ---------------------------------------------------------- cross-panel
    print("\n" + "=" * 78)
    print("3. CROSS-PANEL TRANSFER (Panel L parameters, unchanged)")
    print("=" * 78)
    rows = []
    for name in ("L", "X", "M"):
        p = panels.build(name)
        for label, params in (("robust", robust), ("full-sample", full)):
            out = pipeline.execute(p, params, constraint_step=21)
            c = out.constraint
            rows.append(
                {
                    "panel": f"{name} / {label}",
                    "cagr": out.stats["cagr"],
                    "max_dd": out.stats["max_dd"],
                    "bench_dd": -c.benchmark_depth,
                    "worst_excess": c.worst_excess,
                    "passes": c.passed,
                }
            )
    print(tables.fmt(pd.DataFrame(rows).set_index("panel")))
    print(
        "\n  Panel M holds a single crisis and cannot validate a drawdown constraint;\n"
        "  it is shown for completeness, not as evidence."
    )

    # ---------------------------------------------------------------- costs
    print("\n" + "=" * 78)
    print("4. COST SENSITIVITY")
    print("=" * 78)
    for label, params in (("robust", robust), ("full-sample", full)):
        print(f"\n  {label}")
        print(tables.fmt(validate.cost_sensitivity(panel, params)))

    # --------------------------------------------------------------- stress
    print("\n" + "=" * 78)
    print("5. STRESS EPISODES")
    print("=" * 78)
    for label, params in (("robust", robust), ("full-sample", full)):
        out = pipeline.execute(panel, params, constraint_step=21)
        table = validate.stress_table(out.result.returns, panel.benchmark)
        print(f"\n  {label}: passes {int(table['pass'].sum())} of {len(table)} episodes")
        print(tables.fmt(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
