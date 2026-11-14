"""Mechanical look-ahead test for every registered model.

For each model and each test origin t:
  A. forecast from full_panel.asof(t), with ONE model instance that is first
     called on the LATEST origin and then walks backwards, so that any cache
     built from later data has every chance to contaminate earlier forecasts;
  B. forecast from a panel that physically ends at t (load_panel(max_date=t)),
     with a fresh model instance.
If A == B the forecast at t cannot depend on anything after t.

Usage: python scripts/leak_test.py [--kinds vol corr cov] [--models ...] [--jobs 4]
Respects VOLMODEL_MAX_DATE (the "full" panel is then the development panel).
"""
import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ORIGINS = ["1951-03-31", "1975-06-30", "1987-10-31", "1999-12-31", "2008-09-30", "2013-12-31"]


def check(kind, name, origins):
    warnings.simplefilter("ignore")
    from volmodel.data.panel import load_panel
    from volmodel.models import get_model
    fn = {"vol": "forecast_vol", "corr": "forecast_corr", "cov": "forecast_cov"}[kind]
    out, t0 = [], time.time()
    try:
        full = load_panel()
        origins = [o for o in origins if pd.Timestamp(o) <= full.monthly.index.max()]
        m = get_model(kind, name)
        a = {}
        for o in sorted(origins, reverse=True):
            v = full.asof(o)
            a[o] = getattr(m, fn)(v, v.eligible())
        for o in origins:
            p = load_panel(max_date=o)
            v = p.asof(o)
            b = getattr(get_model(kind, name), fn)(v, v.eligible())
            x, y = np.asarray(a[o], float), np.asarray(b, float)
            if x.shape != y.shape:
                out.append((kind, name, o, np.inf, "shape differs"))
                continue
            d = float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-12)))
            out.append((kind, name, o, d, ""))
    except Exception as e:  # report, do not hide
        out.append((kind, name, "-", np.nan, f"ERROR {e!r}"[:200]))
    return out, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["vol", "corr", "cov"])
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from volmodel.models import list_models
    todo = [(k, n) for k in a.kinds for n in list_models(k) if not a.models or n in a.models]
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for (k, n), fut in [((k, n), ex.submit(check, k, n, ORIGINS)) for k, n in todo]:
            res, sec = fut.result()
            rows += res
            worst = max((r[3] for r in res if r[3] == r[3]), default=float("nan"))
            flag = "ok  " if all((r[3] <= a.tol) for r in res) else "FAIL"
            print(f"{flag} {k:4s} {n:32s} max rel diff {worst:.2e}  {sec:6.1f}s  "
                  + "; ".join(f"{r[2]}:{r[4]}" for r in res if r[4]), flush=True)
    df = pd.DataFrame(rows, columns=["kind", "model", "origin", "max_rel_diff", "note"])
    if a.out:
        df.to_csv(a.out, index=False)
    bad = df[~(df["max_rel_diff"] <= a.tol)]
    print(f"\n{len(todo)} models, {len(df)} checks, {len(bad)} failing")
    sys.exit(1 if len(bad) else 0)
