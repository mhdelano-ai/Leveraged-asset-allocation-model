"""Write the generated tables and figures-in-prose of reports/REPORT.md and README.md.

    python scripts/make_report_tables.py           # rewrite every <!-- gen:NAME --> block in place
    python scripts/make_report_tables.py --check   # exit 1 if any committed block differs from a fresh build

A block is everything between ``<!-- gen:NAME -->`` and ``<!-- /gen:NAME -->``;
the two markers may sit on one line (a number inside a sentence) or on lines of
their own (a table). Everything outside the markers is hand-written and never
touched. Every value comes from the same builders as the dashboard
(``assetforecast.dashboard.data``), so the report, the README and the page
cannot disagree; the covariance evaluation tables read ``results/final``.
"""
from __future__ import annotations

import argparse
import json
from decimal import ROUND_HALF_UP, Decimal
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assetforecast import paths  # noqa: E402
from assetforecast.dashboard import data as dd  # noqa: E402
from assetforecast.dashboard import glossary as gl  # noqa: E402
from assetforecast.returns import config as rc  # noqa: E402

TARGETS = (ROOT / "reports" / "REPORT.md", ROOT / "README.md")
MARK = re.compile(r"<!-- gen:([a-z0-9_]+) -->(.*?)<!-- /gen:\1 -->", re.S)
STATUS = {"pass": "beats it", "weak": "not significant", "none": "no edge", "worse": "worse", "na": "n/a"}


def s(x, dp=1, sign=False) -> str:
    """Format a display value (already in the dashboard's units); '−' for minus."""
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return "n/a"
    # half-up on the shortest decimal text of the value, so 19.15 prints as 19.2 here as it does in
    # results/final's own text (Python's format would give 19.1 from the binary value)
    d = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)
    t = f"{d:+.{dp}f}" if sign else f"{d:.{dp}f}"
    return t.replace("-", "−")


def p(x) -> str:
    return x if isinstance(x, str) else ("n/a" if pd.isna(x) else f"{x:.3f}")


def table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n" + "\n".join(out) + "\n"


def build_blocks(f: dict[str, pd.DataFrame] | None = None) -> dict[str, str]:
    f = f or dd.build_all()
    src = dd.Sources()
    b: dict[str, str] = {}
    snap = f["snapshot"].iloc[0]
    v = f["verdicts"].set_index("question")
    oa, op, fa = f["outlook_assets"].set_index("asset"), f["outlook_portfolios"], f["forecast_assets"].set_index("asset")
    pub = op[op["shape"] == "portfolios"].set_index("portfolio")
    corr = f["forecast_corr"].set_index("pair")["corr"]
    vols = src.read("vols.csv").set_index("asset")

    # ---- inline facts ---------------------------------------------------------
    b["window"] = snap.window
    b["data_through"] = snap.data_through
    b["sb_corr"] = s(corr["us_large~ust_int"], 2, True)
    b["vol_6040"] = s(f["forecast_portfolios"].set_index("portfolio").vol12["p6040"])
    b["er_6040"] = s(pub.er["p6040"], 1, True)
    b["q05_6040"] = s(pub.q05["p6040"])
    b["se_6040"] = s(pub.er_se["p6040"])
    b["er_zero_line"] = (f"{snap.er_n_zero} of the {snap.er_n} 90% intervals include zero, and the lower bound for "
                          f"{snap.er_near_zero} is essentially zero")
    b["port_narrow_line"] = (
        f"the portfolio shape keeps the spread of past portfolio outcomes, {snap.shape_p_sd:.3f} of the scale then "
        f"predicted (from {snap.shape_p_series} overlapping portfolio series), and its tails are floored at a normal "
        f"with that same spread, not unit spread. For 60/40 the range's standard deviation is {snap.p6040_sd:.2f}% "
        f"against a 12-month volatility forecast of {snap.p6040_vol12:.2f}%; its middle half (25–75%) is "
        f"{snap.p6040_iqr_narrower}% narrower than a normal at the predictive scale; and its {snap.shape_p_floored} "
        f"quantiles sit on the narrower floor")
    b["p6040_q05_as"] = s(snap.p6040_q05_assetshape)
    b["episodes_p05"] = str(snap.shape_p_episodes_q05)
    rr = v.loc["risk_rep"]
    b["risk_behind_line"] = rr.behind_list.replace("-", "−")
    cc = v.loc["cov_corr"]
    b["cov_corr_line"] = f"{s(cc.skill, 1, True)}% (p = {p(cc.p)})"
    cv = v.loc["cov_vol"]
    b["cov_vol_line"] = f"{s(cv.skill, 1, True)}% (p = {p(cv.p)})"
    cm = v.loc["cov_matrix"]
    b["cov_matrix_p"] = p(cm.p)
    b["oos_from"] = snap.oos_from
    b["first_oos_outcome"] = snap.first_oos_outcome
    b["episodes_p"] = f"{snap.shape_p_episodes_q01} and {snap.shape_p_episodes_q05}"
    b["episodes_a"] = f"{snap.shape_a_episodes_q01} and {snap.shape_a_episodes_q05}"
    b["return_panel_from"] = snap.return_panel_from
    b["replication_label"] = rc.REPLICATION_LABEL
    mr, mo = v.loc["mean_rep"], v.loc["mean_orig_h12"]
    b["mean_rep_line"] = (f"best pooled R² {s(mr.r2, 2, True)}% ({mr.best_label}), {mr.n_positive} of {mr.n_models} "
                          f"positive, about {s(mr.r2_detectable)}% needed to detect anything, family test p = {p(mr.spa_p)}")
    ca = v.loc["calib_assets"]
    b["calib_line"] = (f"{s(ca.cover90)}% of asset outcomes inside the 90% interval over the 2014–2025 origins "
                       f"(calibration test p = {p(ca.calib_p)}, about {s(ca.n_indep)} independent years)")
    sr = v.loc["strat_rep"]
    b["strat_rep_line"] = (f"{sr.n_pos} of {sr.n} rule-model pairs ahead, {sr.n_q_pos} significantly after "
                           f"adjustment; {sr.n_neg} behind, {sr.n_sig_neg} with p < 0.05")
    so = v.loc["strat_orig"]
    b["strat_orig_line"] = f"{so.n_neg} of {so.n} did worse, {so.n_sig_neg} of them with p < 0.05; {so.n_q_pos} gained significantly after adjustment"

    # ---- the answer and the covariance forecast ------------------------------
    names = {a: rc.ASSETS[a] for a in rc.ASSETS}
    b["answer_vols"] = "\n" + "\n".join(
        f"- {names[a]}: **{s(fa.vol12[a])}%**" for a in fa.index) + "\n"
    du = src.read("dashboard_vol.csv", "final")
    live = du[du.origin == du.origin.max()].set_index("asset")
    b["vol_table"] = table(
        ["asset", "**12-month return vol**", "21-day vol", "variance ratio", "range across model families (21-day)",
         "simple runner-up (21-day)", "realized, last 12 months", "long-run realized", "history from"],
        [[names[a], f"**{s(fa.vol12[a])}%**", f"{s(fa.vol21[a])}%", f"{vols.variance_ratio[a]:.2f}",
          f"{s(fa.p10[a])}–{s(fa.p90[a])}%", f"{s(100 * live.runner_up[a])}%", f"{s(fa.last12[a])}%",
          f"{s(fa.longrun[a])}%", str(fa.since[a])[:4]] for a in fa.index])
    ids = list(rc.ASSETS)
    ab = {a: gl.SHORT_ASSET[a] for a in ids}
    b["corr_table"] = table(["", *[ab[a] for a in ids[:-1]]],
                            [[ab[a]] + [s(corr[f"{a}~{c}"], 2) if j < i else "" for j, c in enumerate(ids[:-1])]
                             for i, a in enumerate(ids) if i > 0])

    # ---- expected returns and distributions -----------------------------------
    b["outlook_table"] = table(
        ["asset", "**expected return**", "90% interval", "arithmetic equivalent", "12-month vol", "1%", "5%",
         "median", "95%", "99%", "average since"],
        [[r["name"], f"**{s(r.er, 1, True)}%**", f"{s(r.er_lo)} to {s(r.er_hi)}%", f"{s(r.er_arith)}%",
          f"{s(r.vol12)}%", s(r.q01), s(r.q05), s(r.q50), s(r.q95), s(r.q99), r.history_from]
         for _, r in f["outlook_assets"].iterrows()])
    fp = f["forecast_portfolios"].set_index("portfolio")
    b["portfolio_table"] = table(
        ["portfolio", "shape", "expected return (s.e.)", "21-day vol", "12-month vol", "sd of the range", "1%", "5%",
         "median", "95%", "99%"],
        [[r.label if r["shape"] == "portfolios" else f"*{r.label}*", "published (portfolio outcomes)"
          if r["shape"] == "portfolios" else "*sensitivity (asset outcomes)*", f"{s(r.er, 1, True)}% ({s(r.er_se)})",
          f"{s(fp.vol21[r.portfolio])}%", f"{s(r.vol12)}%", f"{s(r.sd)}%", s(r.q01), s(r.q05), s(r.q50), s(r.q95), s(r.q99)]
         for _, r in pd.concat([op[op["shape"] == "portfolios"], op[op["shape"] == "assets"]]).iterrows()])
    b["portfolio_weights"] = "\n" + "\n".join(
        f"- {r.label}: {r.weights}" for _, r in pub.reset_index().iterrows()) + "\n"
    sh = f["shape"]
    b["shape_table"] = table(["quantile", *[f"{t}%" for t in sh.tau]],
                             [["single assets", *[s(x, 2) for x in sh.z_assets]],
                              ["policy portfolios", *[s(x, 2) for x in sh.z_portfolios]],
                              ["normal", *[s(x, 2) for x in sh.z_normal]]])

    # ---- covariance evaluation (results/final) --------------------------------
    hl = src.json("summary.json", "final")["headline"]
    roles = [("primary", "**Selected model** (trimmed family average)"), ("runner_up", "Simple runner-up (section 9)"),
             ("single", "Best single model in development (`statespace_2c_daily`)"),
             ("ref_blend", "Reference (long-run / 5-year blend)"), ("ref_last12m", "Reference (last 12 months)")]
    b["cov_scores"] = table(["model", "development, 1926–2013", "holdout, 2014–2025"],
                            [[lab, f"{hl['dev']['vol_qlike'][k]:.3f}", f"{hl['holdout']['vol_qlike'][k]:.3f}"]
                             for k, lab in roles])

    def sk(d):
        lo = "unbounded" if d["ci90"][0] < -100 else s(d["ci90"][0], 1, True)
        return f"{s(d['primary_skill_pct'], 1, True)}% [{lo}, {s(d['ci90'][1], 1, True)}], {d['dm_p']:.2f}"
    b["cov_skill"] = table(["against", "development [90% interval], p", "holdout [90% interval], p"],
                           [[lab, sk(hl["dev"]["vol_vs_primary"][k]), sk(hl["holdout"]["vol_vs_primary"][k])]
                            for k, lab in (roles[3], roles[4], roles[1], roles[2])])
    cvd, cvh = hl["dev"]["cov_vs_primary"]["reference"], hl["holdout"]["cov_vs_primary"]["reference"]
    b["cov_matrix_line"] = (f"development difference in covariance loss {s(cvd['diff'], 3, True)} (p = {cvd['dm_p']:.3f}); "
                            f"holdout {s(cvh['diff'], 3, True)} [{s(cvh['ci90'][0], 3, True)}, {s(cvh['ci90'][1], 3, True)}] "
                            f"(p = {cvh['dm_p']:.2f})")
    e = f["expectations"]
    b["expectations_table"] = table(["#", "expectation, written before the holdout", "outcome", "evidence"],
                                    [[r.id, r.expectation, f"**{r.status}**", r.evidence] for _, r in e.iterrows()])

    # ---- the return-side evidence -------------------------------------------
    tests = [("mean_orig_h1", "Original holdout, 1 month"), ("mean_orig_h3", "Original holdout, 3 months"),
             ("mean_orig_h12", "Original holdout, 12 months"),
             ("mean_rep", "Replication, 12 months, 2014–2025 origins *(seen data)*"),
             ("mean_rep_ap", "Replication, 12 months, 2015–2025 origins *(seen data)*")]
    mt = table(
        ["test", "origins", "best model", "best pooled R²", "models with R² > 0", "R² detectable", "family test p",
         "independent obs.", "verdict"],
        [[lab, f"{v.loc[q].first_origin} to {v.loc[q].last_origin}", v.loc[q].best_label,
          f"{s(v.loc[q].r2, 2, True)}%", f"{v.loc[q].n_positive} of {v.loc[q].n_models}",
          f"{s(v.loc[q].r2_detectable)}%", p(v.loc[q].spa_p), s(v.loc[q].n_indep), STATUS[v.loc[q].status]]
         for q, lab in tests])
    lines = mt.strip("\n").split("\n")
    b["mean_table_orig"] = "\n" + "\n".join(lines[:5]) + "\n"
    b["mean_table_rep"] = "\n" + "\n".join(lines[:2] + lines[5:]) + "\n"
    m = f["mean_r2"]
    cols = [("original", 1, "holdout", "orig. 1m"), ("original", 3, "holdout", "orig. 3m"),
            ("original", 12, "holdout", "orig. 12m"), ("replication", 12, "holdout", "repl. 2014–25"),
            ("replication", 12, "holdout_ap", "repl. 2015–25")]
    rows = []
    for model in (*rc.FROZEN_MEAN, *rc.ROBUSTNESS):
        lab = gl.label(model) + (" *(robustness)*" if model in rc.ROBUSTNESS else "")
        rows.append([lab] + [f"{s(m[(m.source == a) & (m.horizon == h) & (m.period == per) & (m.model == model)].r2.iloc[0], 2, True)}%"
                             for a, h, per, _ in cols])
    b["mean_models_orig"] = table(["model (pooled R² vs historical mean)", *[c[3] for c in cols[:3]]], [r[:4] for r in rows])
    b["mean_models_rep"] = table(["model (pooled R² vs historical mean)", *[c[3] for c in cols[3:]]], [[r[0], *r[4:]] for r in rows])
    b["dist_orig_table"] = table(
        ["horizon", "best model", "CRPS gain over EWMA volatility", "models ahead", "family test p", "verdict"],
        [[f"{h} month{'s' if h > 1 else ''}", v.loc[f"dist_orig_h{h}"].best_label, f"{s(v.loc[f'dist_orig_h{h}'].skill, 2, True)}%",
          f"{v.loc[f'dist_orig_h{h}'].n_positive} of {v.loc[f'dist_orig_h{h}'].n_models}", p(v.loc[f"dist_orig_h{h}"].spa_p),
          STATUS[v.loc[f"dist_orig_h{h}"].status]] for h in dd.ARCHIVE_HORIZONS])
    c = f["dist_calib"].set_index("id")
    crow = [("dist_hm_af~assets~pooled", "Published, all assets"), ("hist_mean~assets~pooled", "Historical mean, normal (assets)"),
            ("vol_ewma~assets~pooled", "EWMA volatility, normal (assets)"),
            ("dist_hm_af~portfolios~pooled", "Published, policy portfolios"),
            ("dist_hm_af.assetshape~portfolios~pooled", "*Portfolios with the asset shape*"),
            ("hist_mean~portfolios~pooled", "Historical mean, static normal (portfolios)")]
    b["calib_table"] = table(
        ["distribution", "origins", "50%", "80%", "90%", "98%", "below 5%", "below 1%", "above 95%", "calibration p",
         "independent obs."],
        [[lab, "2014–2025" if per == "holdout" else "all scored", *[s(c.loc[f"{k}~{per}", f"cover_{lv}"]) for lv in (50, 80, 90, 98)],
          s(c.loc[f"{k}~{per}", "hit_q05"]), s(c.loc[f"{k}~{per}", "hit_q01"]), s(c.loc[f"{k}~{per}", "hit_q95"]),
          p(c.loc[f"{k}~{per}", "calib_p"]), s(c.loc[f"{k}~{per}", "n_indep"])]
         for per in ("holdout", "full") for k, lab in crow if f"{k}~{per}" in c.index])
    dv, dh = v.loc["dist_rep_vol"], v.loc["dist_rep_hm"]
    b["dist_rep_line"] = (f"{s(dv.skill, 2, True)}% against the EWMA-volatility normal (p = {p(dv.p)}) and "
                          f"{s(dh.skill, 2, True)}% against the historical-mean normal (p = {p(dh.p)})")
    st = [("strat_orig_h1", "Original holdout, 1 month"), ("strat_orig_h3", "Original holdout, 3 months"),
          ("strat_orig_h12", "Original holdout, 12 months"), ("strat_rep", "Replication, 12 months, 2014–2025 *(seen data)*"),
          ("strat_rep_ap", "Replication, 12 months, 2015–2025 *(seen data)*")]
    stt = table(
        ["test", "pairs", "ahead", "ahead, p < 0.05", "ahead after adjustment", "behind", "behind, p < 0.05",
         "best (CER, % a year)", "worst", "verdict"],
        [[lab, r.n, r.n_pos, r.n_sig_pos, r.n_q_pos, r.n_neg, r.n_sig_neg,
          f"{s(r.best_d_cer, 2, True)} ({r.best_rule_label}, {r.best_label})",
          f"{s(r.worst_d_cer, 2, True)} ({r.worst_rule_label}, {r.worst_label})", STATUS[r.status]]
         for q, lab in st for r in [v.loc[q]]])
    lines = stt.strip("\n").split("\n")
    b["strategy_orig"] = "\n" + "\n".join(lines[:5]) + "\n"
    b["strategy_rep"] = "\n" + "\n".join(lines[:2] + lines[5:]) + "\n"
    sg = f["strategy"]
    rk = sg[(sg.family == "risk") & (sg.period == "holdout")]
    b["risk_table"] = table(["rule (historical-mean expected returns)", "CER gain, % a year", "95% interval", "p", "Sharpe difference"],
                            [[r.rule_label, s(r.d_cer, 2, True), f"{s(r.ci_lo, 2)} to {s(r.ci_hi, 2)}", p(r.d_cer_p),
                              s(r.d_sharpe, 2, True)] for _, r in rk.iterrows()])

    # ---- README ------------------------------------------------------------------
    b["readme_forecast"] = table(
        ["asset", "expected excess return (90% interval)", "12-month volatility", "5% quantile", "95% quantile"],
        [[r["name"], f"{s(r.er, 1, True)}% ({s(r.er_lo)} to {s(r.er_hi)})", f"{s(r.vol12)}%", f"{s(r.q05)}%",
          f"{s(r.q95)}%"] for _, r in f["outlook_assets"].iterrows()] +
        [[f"*{r.label}*", f"{s(r.er, 1, True)}% (s.e. {s(r.er_se)})", f"{s(r.vol12)}%", f"{s(r.q05)}%", f"{s(r.q95)}%"]
         for _, r in pub.reset_index().iterrows()])
    return b


def render(text: str, blocks: dict[str, str]) -> str:
    unknown = sorted({n for n, _ in MARK.findall(text)} - set(blocks))
    if unknown:
        raise KeyError(f"no generator for blocks {unknown}")
    return MARK.sub(lambda m: f"<!-- gen:{m.group(1)} -->{blocks[m.group(1)]}<!-- /gen:{m.group(1)} -->", text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if a committed block differs from a fresh build")
    a = ap.parse_args(argv)
    blocks = build_blocks()
    stale = []
    for path in TARGETS:
        old = path.read_text()
        new = render(old, blocks)
        if new != old:
            stale.append(path)
            if not a.check:
                path.write_text(new)
    if a.check:
        print("stale: " + ", ".join(str(p.relative_to(ROOT)) for p in stale) if stale else "all generated blocks current")
        return 1 if stale else 0
    print("rewrote " + ", ".join(str(p.relative_to(ROOT)) for p in stale) if stale else "nothing to change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
