"""Does the month-average (tier D) history help or hurt the leading models?

Step 1 (run once; writes to results/dev_noD):
  VOLMODEL_MAX_DATE=2014-12-31 VOLMODEL_DROP_TIERS=D VOLMODEL_RESULTS=$PWD/results/dev_noD \
    python -m volmodel.eval.run forecast --kind vol --model <models> --jobs 3
Step 2: VOLMODEL_MAX_DATE=2014-12-31 python scripts/tierD_sensitivity.py
Compares QLIKE on identical asset-dates (daily-based targets only), by era.
"""
import pathlib
import warnings

import pandas as pd

from volmodel.data.panel import load_panel
from volmodel.eval import harness as H
from volmodel.eval import losses as L
from volmodel.eval.targets import get_targets

warnings.simplefilter("ignore")
R = pathlib.Path(__file__).resolve().parents[1] / "results"
ERAS, LABELS = [1900, 1945, 1979, 1999, 2014], ["1926-45", "1946-79", "1980-99", "2000-13"]


def main():
    T = get_targets(load_panel())
    models = sorted(set(H.stored_models("vol", R / "dev_noD")) & set(H.stored_models("vol", R / "dev")))

    def cells(res):
        a = L.compute_losses("vol", models, T, R / res)["asset"]
        return a[~a.proxy].set_index(["origin", "asset", "model"])["qlike"].unstack("model")

    a, b = cells("dev").dropna(), cells("dev_noD").dropna()
    ix = a.index.intersection(b.index)

    def agg(pv):
        po = pv.loc[ix].groupby(level=0).mean()
        t = po.groupby(pd.cut(po.index.year, ERAS, labels=LABELS), observed=True).mean().T
        t["all"] = po.mean()
        return t

    ta, tb = agg(a), agg(b)
    out = pd.concat({"with D": ta, "without D": tb, "difference": tb - ta}, axis=1).round(3)
    print(f"{len(ix)} common asset-dates from {ix.get_level_values(0).min().date()}")
    print(out.sort_values(("without D", "all")).to_string())
    out.to_csv(R.parent / "reports" / "models" / "tierD_sensitivity.csv")


if __name__ == "__main__":
    main()
