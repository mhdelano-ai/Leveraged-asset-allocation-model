"""Build the small CSV files behind the published dashboard.

    python scripts/make_dashboard_data.py        # writes dashboard/data/*.csv

Inputs: forecasts/latest/ (python -m volmodel.forecast) and results/final/
(scripts/score_final.py). Volatilities are written in percent with one
decimal so that the page shows exactly what the files hold. The page itself
is dashboard/index.html; it is stored in the dashboard, not served from here.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F, FC, OUT = ROOT / "results" / "final", ROOT / "forecasts" / "latest", ROOT / "dashboard" / "data"


def pc(s):
    return (100 * s).round(1)


def quarterly(d):
    """Quarter-end origins, plus every one of the last 12 (whose windows are still open)."""
    m = d.origin.str[5:7].astype(int)
    live_from = sorted(d.origin.unique())[-12]
    return d[(m % 3 == 0) | (d.origin >= live_from)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v = pd.read_csv(FC / "vols.csv")
    acc = pd.read_csv(F / "accuracy.csv")
    acc = acc[acc["sample"] == "last30y"].set_index("asset")
    pd.DataFrame({
        "asset": v.asset, "name": v.name, "group": v.group, "vol12": pc(v.vol_12m_return),
        "vol21": pc(v.vol_21d_horizon), "last12": pc(v.realized_last_12m), "longrun": pc(v.realized_long_run),
        "p10": pc(v.family_p10), "p90": pc(v.family_p90), "q05": pc(v.q05_12m_ex_mean),
        "q01": pc(v.q01_12m_ex_mean), "since": v.history_starts,
        "miss_lo": v.asset.map(acc.two_thirds_lo_pct).round(0).astype(int),
        "miss_hi": v.asset.map(acc.two_thirds_hi_pct).round(0).astype(int),
    }).to_csv(OUT / "forecast_assets.csv", index=False)

    c = pd.read_csv(FC / "corr.csv", index_col=0).stack().reset_index()
    c.columns = ["asset_i", "asset_j", "corr"]
    c["corr"] = c["corr"].round(2)
    c["pair"] = c.asset_i + "~" + c.asset_j
    c.to_csv(OUT / "forecast_corr.csv", index=False)

    p = pd.read_csv(FC / "portfolios.csv")
    label = {"p6040": "60/40 (US large cap / intermediate Treasuries)", "p_balanced": "Balanced 12-asset mix",
             "p_equity": "Global equity mix"}
    pd.DataFrame({"portfolio": p.portfolio, "label": p.portfolio.map(label), "vol12": pc(p.vol_12m_return),
                  "vol21": pc(p.vol_21d_horizon), "q05": pc(p.q05_12m_ex_mean),
                  "q01": pc(p.q01_12m_ex_mean)}).to_csv(OUT / "forecast_portfolios.csv", index=False)

    d = pd.read_csv(F / "dashboard_vol.csv")
    h = quarterly(d[d.origin >= "1926-06"])
    pd.DataFrame({"origin": h.origin, "asset": h.asset, "period": h.period, "realized": pc(h.realized),
                  "proxy": h.proxy.astype(str).str.lower(), "forecast": pc(h.primary), "reference": pc(h.reference),
                  "p10": pc(h.fam_p10), "p90": pc(h.fam_p90)}).to_csv(OUT / "history_vol.csv", index=False)

    k = pd.read_csv(F / "dashboard_corr.csv")
    k = quarterly(k[k.pair.isin(["us_large~ust_int", "us_large~ust_long"])])
    pd.DataFrame({"origin": k.origin, "pair": k.pair, "realized": k.realized.round(2), "forecast": k.primary.round(2),
                  "reference": k.reference.round(2)}).to_csv(OUT / "history_corr.csv", index=False)

    q = pd.read_csv(F / "dashboard_portfolio.csv")
    q = quarterly(q[q.portfolio == "p6040"])
    pd.DataFrame({"origin": q.origin, "realized": pc(q.realized), "forecast": pc(q.primary),
                  "reference": pc(q.reference)}).to_csv(OUT / "history_6040.csv", index=False)

    hh = pd.read_csv(F / "h2h_vol.csv")
    hh = hh[(hh.subset == "daily") & (hh.era == "all") & hh.period.isin(["dev", "holdout"])]
    roles = {"primary": "Selected: family average", "runner_up": "Simple runner-up",
             "single": "Best single model (development)", "reference": "Reference: long-run / 5-year blend",
             "ref_last12m": "Benchmark: last 12 months"}
    hh = hh[hh.role.isin(roles)]
    cand = hh.pivot(index="role", columns="period", values="qlike").round(3)
    rf = hh.pivot(index="role", columns="period", values="rf").round(2)
    cand["label"], cand["rf_dev"], cand["rf_holdout"] = cand.index.map(roles), rf["dev"], rf["holdout"]
    cand.loc[list(roles)].reset_index().to_csv(OUT / "candidates.csv", index=False)

    lb = pd.read_csv(F / "vol_leaderboard.csv")
    lb = lb[(lb.subset == "daily") & (lb.era == "all")]
    a, b = lb[lb.period == "dev"].set_index("model"), lb[lb.period == "holdout"].set_index("model")
    rk = pd.DataFrame({"family": a.family, "rank_dev": a["rank"].astype(int), "rank_holdout": b["rank"].astype(int),
                       "qlike_dev": a.qlike.round(3), "qlike_holdout": b.qlike.round(3)}).reset_index()
    named = {"combo_vol_hier_trim": "Selected", "statespace_2c_daily": "Best single in development",
             "ref_blend": "Reference", "ref_last12m": "Last 12 months", "combo_vol_cal_m_hl12": "Simple runner-up",
             "bench_vol_m_hl120": "10-year average"}
    rk["label"] = rk.model.map(named).fillna("")
    rk.to_csv(OUT / "model_ranks.csv", index=False)
    print("wrote", sorted(f.name for f in OUT.glob("*.csv")))


if __name__ == "__main__":
    main()
