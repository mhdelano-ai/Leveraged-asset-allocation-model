"""The verdict: plain sentences selected and filled in from the section data.

Every sentence that states a result is chosen by a rule on the numbers below,
so the holdout can overturn it. The rules are deliberately simple and stated on
the page:

* An edge **holds** when the point estimate favours the model *and* the test
  that allows for the search over models (SPA for a set of models, BH q-values
  for many cells) is below ``ALPHA``.
* It is **not significant** when the point estimate favours the model but that
  adjusted test does not.
* There is **no edge** when the point estimate does not favour the model.
"""
from __future__ import annotations

import math

from .. import config
from . import glossary as gl

ALPHA = 0.05          # one threshold for p-values and q-values everywhere on the page
CALIB_TOL = 0.02      # |coverage of the 90% interval - 0.90| within which coverage is called accurate

STATUS_LABEL = {"pass": "Edge holds", "weak": "Not significant", "none": "No edge", "na": "Not available",
                "calib_ok": "Calibrated", "calib_off": "Miscalibrated"}


def ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def pct(x, dp=2, sign=True) -> str:
    if not ok(x):
        return "n/a"
    s = f"{x * 100:+.{dp}f}%" if sign else f"{x * 100:.{dp}f}%"
    return s.replace("-", "−")


def num(x, dp=3, sign=False) -> str:
    if not ok(x):
        return "n/a"
    s = f"{x:+.{dp}f}" if sign else f"{x:.{dp}f}"
    return s.replace("-", "−")


def pval(x) -> str:
    if not ok(x):
        return "n/a"
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def eq(name: str, x) -> str:
    """'p = 0.031' or 'p < 0.001'."""
    v = pval(x)
    return f"{name} {v}" if v.startswith("<") else f"{name} = {v}"


def _spa_name(info: dict) -> str:
    n = info.get("spa_n")
    return f"SPA p ({n} models)" if n else "SPA p"


def hlist(hs) -> str:
    hs = [str(h) for h in hs]
    if not hs:
        return ""
    if len(hs) == 1:
        return f"{hs[0]} month" + ("" if hs[0] == "1" else "s")
    return ", ".join(hs[:-1]) + " and " + hs[-1] + " months"


def _best(rows, key):
    rows = [r for r in rows if not r.get("ref") and ok(r.get(key))]
    return max(rows, key=lambda r: r[key]) if rows else None


# --------------------------------------------------------------------------- components

def mean_verdict(mean: dict | None) -> dict:
    out = {"cells": {}, "text": [], "status_by_h": {}}
    if not mean:
        out["text"].append("No mean-forecast scores were found, so this part of the question is unanswered here.")
        return out
    hs = sorted(mean["pooled"])
    nmod = len(mean["models"])
    below_det, best_r2 = {}, {}
    for h in hs:
        b = _best(mean["pooled"][h], "r2")
        info = mean["info"][h]
        if b is None:
            out["cells"][h] = {"status": "na", "value": "n/a", "detail": ""}
            continue
        spa = info.get("spa_p")
        det = info.get("r2_detectable")
        if b["r2"] > 0 and ok(spa) and spa < ALPHA and ok(b.get("dm_p")) and b["dm_p"] < ALPHA:
            st = "pass"
        elif b["r2"] > 0:
            st = "weak"
        else:
            st = "none"
        out["status_by_h"][h] = st
        best_r2[h] = b
        below_det[h] = ok(det) and b["r2"] < det
        out["cells"][h] = {
            "status": st, "value": f"R² {pct(b['r2'])}",
            "detail": f"best of {nmod}: {gl.label(b['model'])}; detectable ≈ {pct(det, 1, False)}; "
                      f"{eq(_spa_name(info), spa)}"}
    passing = [h for h in hs if out["status_by_h"].get(h) == "pass"]
    if not passing:
        parts = ", ".join(f"{pct(best_r2[h]['r2'])} at {h}m" for h in hs if h in best_r2)
        dets = ", ".join(pct(mean["info"][h]["r2_detectable"], 1, False) for h in hs if h in best_r2)
        s = (f"No mean forecast reliably beats the historical mean at any horizon. The best pooled out-of-sample "
             f"R² values are {parts}")
        if all(below_det.get(h) for h in best_r2):
            s += f", each below the roughly {dets} the sample could detect at 1, 3 and 12 months."
        else:
            s += f" (detectable: about {dets})."
        spas = ", ".join(pval(mean["info"][h]["spa_p"]) for h in hs)
        nspa = mean["info"][hs[0]].get("spa_n") or nmod
        s += (f" Allowing for the search over the {nspa} models scored together, the test of whether any of them "
              f"beats the historical mean (SPA) gives p = {spas}.")
        out["text"].append(s)
    else:
        for h in passing:
            b = best_r2[h]
            out["text"].append(
                f"At {h} month{'s' if h > 1 else ''} the mean forecasts beat the historical mean: "
                f"{gl.label(b['model'])} has a pooled out-of-sample R² of {pct(b['r2'])} "
                f"(one-sided Diebold-Mariano {eq('p', b['dm_p'])}; SPA p across the {mean['info'][h].get('spa_n') or nmod} models = "
                f"{pval(mean['info'][h]['spa_p'])}).")
        rest = [h for h in hs if h not in passing]
        if rest:
            out["text"].append(f"At {hlist(rest)} no mean model reliably beats it.")
    # per-asset cells that survive the BH adjustment
    n_cells, n_pass, n_pos, examples = 0, 0, 0, []
    for h in hs:
        for m, cells in mean["grid"][h]["cells"].items():
            if m == config.COMPOSITE:
                continue  # same mean as its mean model; do not double count
            for a, c in cells.items():
                if not ok(c.get("r2")):
                    continue
                n_cells += 1
                if c["r2"] > 0:
                    n_pos += 1
                    if ok(c.get("q")) and c["q"] < ALPHA:
                        n_pass += 1
                        examples.append((c["r2"], h, m, a, c["q"]))
    out["n_cells"], out["n_cells_pass"], out["n_cells_pos"] = n_cells, n_pass, n_pos
    s = (f"Asset by asset, {n_pos} of {n_cells} model × asset × horizon cells have a positive R²; "
         f"{n_pass} of them {'survives' if n_pass == 1 else 'survive'} the Benjamini-Hochberg adjustment of the "
         f"Clark-West test at q < {ALPHA:g}")
    if examples:
        examples.sort(reverse=True)
        r2, h, m, a, q = examples[0]
        s += f" (largest: {gl.label(m)} on {config.ASSETS[a]} at {h}m, R² {pct(r2)}, {eq('q', q)})"
    out["text"].append(s + ".")
    return out


def dist_verdict(dist: dict | None) -> dict:
    out = {"cells": {}, "text": [], "status_by_h": {}, "calib_cells": {}}
    if not dist:
        out["text"].append("No distribution scores were found.")
        return out
    hs = sorted(dist["pooled"])
    best_by_h = {}
    for h in hs:
        rows = [r for r in dist["pooled"][h] if r["model"] in dist["models"]]
        b = _best(rows, "skill_vol")
        if b is None:
            out["cells"][h] = {"status": "na", "value": "n/a", "detail": ""}
            continue
        spa = dist["info"][h]["spa_vol"]
        st = "pass" if b["skill_vol"] > 0 and ok(spa) and spa < ALPHA else ("weak" if b["skill_vol"] > 0 else "none")
        out["status_by_h"][h] = st
        best_by_h[h] = b
        out["cells"][h] = {"status": st, "value": f"CRPS {pct(b['skill_vol'])}",
                           "detail": f"vs EWMA volatility; best: {gl.label(b['model'])}; "
                                     f"{eq(_spa_name(dist['info'][h]), spa)}"}
    passing = [h for h in hs if out["status_by_h"].get(h) == "pass"]
    weak = [h for h in hs if out["status_by_h"].get(h) == "weak"]
    if passing:
        sk = ", ".join(f"{pct(best_by_h[h]['skill_vol'], 1)} at {h}m" for h in passing)
        s = (f"The distribution forecasts beat the volatility-tracking baseline at {hlist(passing)}: the best "
             f"model improves CRPS by {sk} (SPA p = "
             + ", ".join(pval(dist["info"][h]["spa_vol"]) for h in passing) + ").")
        if weak:
            s += f" At {hlist(weak)} the point estimate is positive but not significant."
    elif weak:
        s = (f"The distribution forecasts score better than the volatility baseline at {hlist(weak)} (best: "
             + ", ".join(f"{pct(best_by_h[h]['skill_vol'], 1)} at {h}m" for h in weak)
             + "), but not significantly once the search over models is allowed for.")
    else:
        s = "No distribution model beats the volatility-tracking baseline."
    out["text"].append(s)
    split = [h for h in hs if out["status_by_h"].get(h) == "weak" and ok(best_by_h[h].get("p_vol"))
             and best_by_h[h]["p_vol"] < ALPHA]
    if split:
        n = dist["info"][split[0]].get("spa_n")
        out["text"].append(
            "Taken alone, the best model's one-sided Diebold-Mariano p-value against the volatility baseline is "
            + ", ".join(f"{pval(best_by_h[h]['p_vol'])} at {h}m" for h in split)
            + f"; the SPA test over all {n or 'the'} models in the scoring run is the stricter bar and gives "
            + ", ".join(f"{pval(dist['info'][h]['spa_vol'])}" for h in split) + ".")
    # ML vs linear
    ml_parts = []
    for h in hs:
        rows = {r["model"]: r for r in dist["pooled"][h]}
        ml = _best([rows[m] for m in gl.ML_DIST if m in rows], "skill_vol")
        lin = _best([rows[m] for m in gl.LINEAR_DIST if m in rows], "skill_vol")
        if ml and lin:
            ml_parts.append((h, ml, lin))
    if ml_parts:
        no_better = all(ml["skill_vol"] <= lin["skill_vol"] + 0.001 for _, ml, lin in ml_parts)
        detail = "; ".join(f"{h}m {pct(ml['skill_vol'], 1)} vs {pct(lin['skill_vol'], 1)}" for h, ml, lin in ml_parts)
        ml_name = gl.label(ml_parts[0][1]["model"]).lower()
        lin_name = gl.label(ml_parts[0][2]["model"]).lower()
        if no_better:
            out["text"].append(f"Machine learning adds nothing to the scale forecast: the {ml_name} scores no better "
                               f"than the {lin_name} ({detail}).")
        else:
            out["text"].append(f"The {ml_name} edges the {lin_name} on CRPS skill ({detail}); the results contain no "
                               f"direct test of that difference.")
    # which part of the distribution: pinball skill by quantile for the best model at h=1 (or first passing h)
    href = (passing or hs)[0]
    if href in best_by_h:
        m = best_by_h[href]["model"]
        sk = dist["pinball"][href][m]["vol"]
        qs = dist["quantiles"]
        pairs = [(q, s) for q, s in zip(qs, sk) if ok(s)]
        if pairs:
            centre = [s for q, s in pairs if 25 <= q <= 75]
            tails = [s for q, s in pairs if q <= 5 or q >= 95]
            hi = max(pairs, key=lambda t: t[1])
            lo = min(pairs, key=lambda t: t[1])
            where = ("in the tails" if tails and centre and sum(tails) / len(tails) > sum(centre) / len(centre)
                     else "in the centre")
            out["text"].append(
                f"By quantile ({gl.label(m)}, {href}m), the improvement over the volatility baseline is larger "
                f"{where}: from {pct(lo[1], 1)} at the {ordinal(lo[0])} percentile to {pct(hi[1], 1)} at the "
                f"{ordinal(hi[0])}.")
    # calibration of the forecast shown in "now"
    mean_m, dist_m = gl.composite_parts()
    cm = config.COMPOSITE if config.COMPOSITE in dist["models"] else dist_m
    tails_txt = []
    for h in hs:
        r = next((x for x in dist["pooled"][h] if x["model"] == cm), None)
        if r is None:
            continue
        c90, cp = r.get("cover_90"), r.get("calib_p")
        good = ok(c90) and abs(c90 - 0.90) <= CALIB_TOL and ok(cp) and cp >= ALPHA
        out["calib_cells"][h] = {"status": "calib_ok" if good else "calib_off",
                                 "value": f"90% band holds {pct(c90, 1, False)}",
                                 "detail": f"{gl.label(cm)}; {eq('PIT test p', cp)}"}
        t = next((x for x in dist["tails"][h] if x["model"] == cm), None)
        if t:
            tails_txt.append(f"{h}m: {pct(t['hit_q01'], 1, False)} below the 1st and {pct(t['hit_q99'], 1, False)} "
                             f"above the 99th percentile")
    if out["calib_cells"]:
        good = [h for h, c in out["calib_cells"].items() if c["status"] == "calib_ok"]
        off = [h for h, c in out["calib_cells"].items() if c["status"] != "calib_ok"]
        s = f"Calibration of the {gl.label(cm)}: "
        if good:
            s += f"accurate at {hlist(good)} (90% interval coverage within {CALIB_TOL * 100:.0f} points of 90% and PIT p ≥ {ALPHA:g})"
        if off:
            s += ("; " if good else "") + f"off at {hlist(off)}"
        s += ". Tail hit rates against nominal 1%: " + "; ".join(tails_txt) + "."
        out["text"].append(s)
    return out


def cov_verdict(cov: dict | None) -> dict:
    out = {"cells": {}, "text": [], "status_by_h": {}}
    if not cov:
        out["text"].append("No covariance scores were found, so the covariance evidence is not shown.")
        return out
    prim = config.PRIMARY_COV
    lw_parts, sb_parts, hs_pass = [], [], []
    for h in sorted(cov["joint"]):
        rows = {r["model"]: r for r in cov["joint"][h]}
        r = rows.get(prim)
        if r is None or not ok(r.get("diff")):
            out["cells"][h] = {"status": "na", "value": "n/a", "detail": ""}
            continue
        q = r.get("q") if ok(r.get("q")) else r.get("p")
        st = "pass" if r["diff"] > 0 and ok(q) and q < ALPHA else ("weak" if r["diff"] > 0 else "none")
        out["status_by_h"][h] = st
        if st == "pass":
            hs_pass.append(h)
        lw = rows.get("lw")
        lw_txt = f"; static shrinkage alone {num(lw['diff'], 2, True)}" if lw and ok(lw.get("diff")) else ""
        out["cells"][h] = {"status": st, "value": f"log score {num(r['diff'], 2, True)}",
                           "detail": f"{gl.label(prim)}, nats per asset; {eq('q', r.get('q'))}{lw_txt}"}
        if lw and ok(lw.get("diff")):
            lw_parts.append((h, r["diff"], lw["diff"]))
        sbrows = {x["model"]: x for x in cov["sb"][h]["rows"]}
        hp = cov["headline"]
        if prim in sbrows and "lw" in sbrows:
            sb_parts.append((h, sbrows[prim]["diff"].get(hp), sbrows["lw"]["diff"].get(hp),
                             sbrows[prim]["p"].get(hp)))
    if hs_pass:
        out["text"].append(f"The primary covariance model ({gl.label(prim)}) beats the static sample "
                           f"covariance on the joint log score at {hlist(hs_pass)} after adjustment.")
    else:
        out["text"].append("The primary covariance model does not reliably beat the static sample covariance.")
    if lw_parts:
        share = ", ".join(f"{h}m {num(l, 2)} of {num(p, 2)}" for h, p, l in lw_parts)
        extra = ", ".join(f"{num(p - l, 2, True)} at {h}m" for h, p, l in lw_parts)
        out["text"].append(
            f"Much of that gain is conditioning rather than dynamics: static shrinkage of the sample covariance "
            f"(Ledoit-Wolf) alone gains {share} nats per asset. What the dynamic model adds over static shrinkage "
            f"is {extra} (a difference of score differences; the results carry no direct test of it).")
    if sb_parts:
        real = cov["sb"][min(cov["sb"])]["real"]
        decs = [p for p in ("1990s", "2000s", "2010s", "2020s") if ok(real.get(p))]
        regime = ""
        if len(decs) >= 2:
            regime = (f" Realized stock-bond correlation (US equities vs intermediate Treasuries, 1-month) went from "
                      f"{num(real[decs[0]], 2, True)} in the {decs[0]} to {num(real[decs[-1]], 2, True)} in the "
                      f"{decs[-1]}.")
        txt = ", ".join(f"{h}m {num(p, 3, True)} vs {num(l, 3, True)}" for h, p, l, _ in sb_parts if ok(p) and ok(l))
        out["text"].append(
            f"On the stock-bond pair alone, the dynamic model's log-score gain over the sample covariance is "
            f"{txt} for static shrinkage; following the correlation regime is where dynamics pay.{regime}")
    return out


def strategy_verdict(st: dict | None) -> dict:
    out = {"cells": {}, "text": [], "status_by_h": {}}
    if not st:
        out["text"].append("No strategy-test results were found, so the economic-value test is not shown.")
        return out
    rows = [r for r in st["rows"] if r["family"] == "mean" and r["universe"] == "all"]
    hs = sorted({r["horizon"] for r in rows})
    for h in hs:
        g = [r for r in rows if r["horizon"] == h and ok(r.get("d_cer"))]
        if not g:
            continue
        b = max(g, key=lambda r: r["d_cer"])
        sig_q = [r for r in g if r["d_cer"] > 0 and ok(r.get("d_cer_q_bh")) and r["d_cer_q_bh"] < ALPHA]
        sig_p = [r for r in g if r["d_cer"] > 0 and ok(r.get("d_cer_p")) and r["d_cer_p"] < ALPHA]
        st_ = "pass" if sig_q else ("weak" if b["d_cer"] > 0 else "none")
        out["status_by_h"][h] = st_
        minq = min((r["d_cer_q_bh"] for r in g if ok(r.get("d_cer_q_bh"))), default=None)
        out["cells"][h] = {"status": st_, "value": f"CER {pct(b['d_cer'])}/yr",
                           "detail": f"best of {len(g)}: {gl.STRATEGY_RULES.get(b['rule'], (b['rule'],))[0]} with "
                                     f"{gl.label(b['mean_model'])}, net; smallest {eq('q', minq)}"}
        out.setdefault("_detail", {})[h] = (g, b, sig_p, sig_q, minq)
    d = out.get("_detail", {})
    if d:
        parts = []
        for h, (g, b, sig_p, sig_q, minq) in sorted(d.items()):
            npos = sum(1 for r in g if r["d_cer"] > 0)
            parts.append(f"{h}m: {npos} of {len(g)} positive, {len(sig_p)} with p < {ALPHA:g}, {len(sig_q)} after "
                         f"adjustment (smallest q {pval(minq)})")
        out["text"].append("Net of trading costs, each rule using a model's forecasts against the same rule using "
                           "historical means, in certainty-equivalent return: " + "; ".join(parts) + ".")
        anyq = any(v[3] for v in d.values())
        if not anyq:
            out["text"].append("No gain from the return forecasts survives the multiple-testing adjustment.")
    vs = [r for r in st["rows"] if r["family"] == "vs_static" and r["universe"] == "all" and r["horizon"] == 1
          and r["mean_model"] not in ("none",)]
    if vs:
        beat = [r for r in vs if ok(r.get("d_sharpe")) and r["d_sharpe"] > 0 and ok(r.get("d_sharpe_p"))
                and r["d_sharpe_p"] < ALPHA]
        pos = [r for r in vs if ok(r.get("d_sharpe")) and r["d_sharpe"] > 0]
        ew = next((lv for lv in st["levels"] if lv["rule"] == "ew" and lv["universe"] == "all"), None)
        s = (f"Against 1/N (Sharpe {num(ew['sharpe'], 2)} net) " if ew else "Against 1/N ")
        s += (f"{len(pos)} of {len(vs)} one-month optimized or timed portfolios have a higher Sharpe ratio, "
              f"{len(beat)} significantly at p < {ALPHA:g} (unadjusted).")
        out["text"].append(s)
    out.pop("_detail", None)
    return out


# --------------------------------------------------------------------------- assemble

def verdict(ctx: dict) -> dict:
    mv = mean_verdict(ctx.get("mean"))
    dv = dist_verdict(ctx.get("dist"))
    cv = cov_verdict(ctx.get("cov"))
    sv = strategy_verdict(ctx.get("strategy"))
    grid = [
        {"key": "mean", "label": "Expected return", "sub": "vs historical mean", "cells": mv["cells"]},
        {"key": "dist", "label": "Full distribution", "sub": "vs EWMA volatility baseline", "cells": dv["cells"]},
        {"key": "calib", "label": "Calibration", "sub": "forecast shown below", "cells": dv["calib_cells"]},
        {"key": "cov", "label": "Covariance", "sub": "vs static sample covariance", "cells": cv["cells"]},
        {"key": "strategy", "label": "Portfolio value", "sub": "net CER vs hist-mean rule", "cells": sv["cells"]},
    ]
    headline = _headline(mv, dv, cv, sv)
    return {"headline": headline, "grid": grid, "mean": mv["text"], "dist": dv["text"], "cov": cv["text"],
            "strategy": sv["text"], "alpha": ALPHA, "labels": STATUS_LABEL}


def _headline(mv, dv, cv, sv) -> str:
    mp = sorted(h for h, s in mv["status_by_h"].items() if s == "pass")
    mw = sorted(h for h, s in mv["status_by_h"].items() if s == "weak")
    dp = sorted(h for h, s in dv["status_by_h"].items() if s == "pass")
    cp = sorted(h for h, s in cv["status_by_h"].items() if s == "pass")
    sp = sorted(h for h, s in sv["status_by_h"].items() if s == "pass")
    out = []
    if mv["status_by_h"]:
        if mp:
            out.append(f"Mean forecasts beat the historical mean at {hlist(mp)}.")
        elif mw:
            out.append("No mean forecast beats the historical mean at any horizon once the search over models is "
                       "allowed for.")
        else:
            out.append("No mean forecast beats the historical mean at any horizon.")
    if dv["status_by_h"]:
        if dp:
            out.append(f"The distribution forecasts {'also' if mp else 'do'} beat a volatility-tracking baseline at "
                       f"{hlist(dp)}, by a small margin.")
        elif any(st == "weak" for st in dv["status_by_h"].values()):
            out.append("The distribution forecasts score better than a volatility-tracking baseline, but not "
                       "significantly once the search over models is allowed for.")
        else:
            out.append("No distribution forecast beats a volatility-tracking baseline.")
    if cv["status_by_h"]:
        out.append(f"Covariance forecasts beat the static sample covariance at {hlist(cp)}." if cp else
                   "Covariance forecasts do not reliably beat the static sample covariance.")
    if sv["status_by_h"]:
        out.append("In the portfolio test, return forecasts add value net of costs that survives multiple-testing "
                   f"adjustment at {hlist(sp)}." if sp else
                   "In the portfolio test, no gain from the return forecasts survives multiple-testing adjustment.")
    return " ".join(out) if out else "No results were found."


def ordinal(n: int) -> str:
    n = int(n)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"
