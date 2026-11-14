"""The verdict: plain sentences selected and filled in from the section data.

Every sentence that states a result is chosen by a rule on the numbers below,
so the holdout can overturn it. The rules are deliberately simple and stated on
the page:

* An edge **holds** when the point estimate favours the model *and* the test
  that allows for the search over models is below ``ALPHA``: the SPA test over
  the pre-registered family (``config.SPA_FAMILIES``) for the mean and
  distribution, Benjamini-Hochberg q-values for many cells or comparisons.
* It is **not significant** when the point estimate favours the model but that
  adjusted test does not.
* There is **no edge** when the point estimate does not favour the model.

Verdicts are taken over the frozen test families only. The composite forecast
and the robustness rows (``config.ROBUSTNESS``) are reported individually,
never as extra candidates. The all-model SPA rows (model ``'*'``) depend on
which unrelated models share a run and are never used for a verdict.
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


def plv(x) -> str:
    """A p-value inside a list introduced by 'p-values': '0.031' or '<0.001'."""
    v = pval(x)
    return "<0.001" if v.startswith("<") else v


def eq(name: str, x) -> str:
    """'p = 0.031' or 'p < 0.001'."""
    v = pval(x)
    return f"{name} {v}" if v.startswith("<") else f"{name} = {v}"


def hlist(hs) -> str:
    hs = [str(h) for h in hs]
    if not hs:
        return ""
    if len(hs) == 1:
        return f"{hs[0]} month" + ("" if hs[0] == "1" else "s")
    return ", ".join(hs[:-1]) + " and " + hs[-1] + " months"


def ordinal(n: int) -> str:
    n = int(n)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def by_h(vals: dict, fmt) -> str:
    """'+0.5% at 1m, −0.2% at 3m' from {h: value}."""
    return ", ".join(f"{fmt(v)} at {h}m" for h, v in vals.items())


def _best(rows, key, roles=("family",)):
    rows = [r for r in rows if r.get("role", "family") in roles and ok(r.get(key))]
    return max(rows, key=lambda r: r[key]) if rows else None


def _row(rows, model):
    return next((r for r in rows if r["model"] == model), None)


# --------------------------------------------------------------------------- mean

def mean_verdict(mean: dict | None) -> dict:
    out = {"cells": {}, "text": [], "status_by_h": {}}
    if not mean:
        out["text"].append("No mean-forecast scores were found, so this part of the question is unanswered here.")
        return out
    hs = sorted(mean["pooled"])
    nfam = len(mean["family"])
    below_det, best = {}, {}
    for h in hs:
        b = _best(mean["pooled"][h], "r2")
        info = mean["info"][h]
        if b is None:
            out["cells"][h] = {"status": "na", "value": "n/a", "detail": ""}
            continue
        spa, det = info.get("spa_p"), info.get("r2_detectable")
        if b["r2"] > 0 and ok(spa) and spa < ALPHA:
            st = "pass"
        elif b["r2"] > 0:
            st = "weak"
        else:
            st = "none"
        out["status_by_h"][h] = st
        best[h] = b
        below_det[h] = ok(det) and b["r2"] < det
        spa_txt = eq("SPA p (mean family)", spa) if ok(spa) else "family SPA not in the results"
        out["cells"][h] = {"status": st, "value": f"R² {pct(b['r2'])}",
                           "detail": f"best of {nfam}: {gl.label(b['model'])}; detectable ≈ "
                                     f"{pct(det, 1, False)}; {spa_txt}"}
    passing = [h for h in hs if out["status_by_h"].get(h) == "pass"]
    spas = {h: mean["info"][h]["spa_p"] for h in hs}
    if not passing:
        dets = ", ".join(pct(mean["info"][h]["r2_detectable"], 1, False) for h in hs if h in best)
        s = (f"No mean forecast reliably beats the historical mean at any horizon. The best pooled out-of-sample "
             f"R² among the {nfam} frozen mean models is {by_h({h: best[h]['r2'] for h in best}, pct)}")
        if all(below_det.get(h) for h in best):
            s += f", each below the roughly {dets} the sample could detect."
        else:
            s += f" (detectable: about {dets})."
        if any(ok(v) for v in spas.values()):
            s += (f" The test of whether any of the {nfam} beats the historical mean, allowing for the search over "
                  f"them (SPA), gives p-values of " + ", ".join(plv(spas[h]) for h in hs) + ".")
        out["text"].append(s)
    else:
        for h in passing:
            b = best[h]
            out["text"].append(
                f"At {hlist([h])} the mean forecasts beat the historical mean: {gl.label(b['model'])} has a pooled "
                f"out-of-sample R² of {pct(b['r2'])} (one-sided Diebold-Mariano {eq('p', b['dm_p'])}; SPA over the "
                f"{nfam} frozen mean models {eq('p', spas[h])}).")
        rest = [h for h in hs if h not in passing]
        if rest:
            out["text"].append(f"At {hlist(rest)} no mean model reliably beats it.")
    comp = {h: _row(mean["pooled"][h], config.COMPOSITE) for h in hs}
    if all(comp.values()):
        out["text"].append(
            f"The composite forecast, reported on its own, has pooled R² {by_h({h: comp[h]['r2'] for h in hs}, pct)}"
            f" (one-sided Diebold-Mariano p-values " + ", ".join(plv(comp[h]["dm_p"]) for h in hs) + ").")
    for rm in mean.get("robust", []):
        par = mean["robust_parent"].get(rm)
        pr = {h: (_row(mean["pooled"][h], par), _row(mean["pooled"][h], rm)) for h in hs}
        if all(a and b for a, b in pr.values()):
            out["text"].append(
                f"Robustness: {gl.label(par).lower()} without the features estimated over the full sample has R² "
                + ", ".join(f"{pct(b['r2'])} at {h}m (with them {pct(a['r2'])})" for h, (a, b) in pr.items()) + ".")
    # per-asset cells of the test family that survive the BH adjustment
    n_cells, n_pass, n_pos, examples = 0, 0, 0, []
    for h in hs:
        for m in mean["family"]:
            for a, c in mean["grid"][h]["cells"].get(m, {}).items():
                if not ok(c.get("r2")):
                    continue
                n_cells += 1
                if c["r2"] > 0:
                    n_pos += 1
                    if ok(c.get("q")) and c["q"] < ALPHA:
                        n_pass += 1
                        examples.append((c["r2"], h, m, a, c["q"]))
    out["n_cells"], out["n_cells_pass"], out["n_cells_pos"] = n_cells, n_pass, n_pos
    s = (f"Asset by asset, {n_pos} of {n_cells} model × asset × horizon cells of the frozen mean models have a "
         f"positive R²; {n_pass} {'survives' if n_pass == 1 else 'survive'} the Benjamini-Hochberg adjustment of the "
         f"Clark-West test at q < {ALPHA:g}")
    if examples:
        examples.sort(reverse=True)
        r2, h, m, a, q = examples[0]
        s += f" (largest: {gl.label(m)} on {config.ASSETS[a]} at {h}m, R² {pct(r2)}, {eq('q', q)})"
    out["text"].append(s + ".")
    return out


# --------------------------------------------------------------------------- distribution

def dist_verdict(dist: dict | None, show_kupiec_long: bool = True) -> dict:
    out = {"cells": {}, "text": [], "status_by_h": {}, "calib_cells": {}}
    if not dist:
        out["text"].append("No distribution scores were found.")
        return out
    hs = sorted(dist["pooled"])
    nfam = len(dist["family"])
    best = {}
    for h in hs:
        b = _best(dist["pooled"][h], "skill_vol")
        if b is None:
            out["cells"][h] = {"status": "na", "value": "n/a", "detail": ""}
            continue
        spa = dist["info"][h]["spa_vol"]
        st = "pass" if b["skill_vol"] > 0 and ok(spa) and spa < ALPHA else ("weak" if b["skill_vol"] > 0 else "none")
        out["status_by_h"][h] = st
        best[h] = b
        spa_txt = eq("SPA p (distribution family)", spa) if ok(spa) else "family SPA not in the results"
        out["cells"][h] = {"status": st, "value": f"CRPS {pct(b['skill_vol'])}",
                           "detail": f"vs EWMA volatility; best of {nfam}: {gl.label(b['model'])}; {spa_txt}"}
    passing = [h for h in hs if out["status_by_h"].get(h) == "pass"]
    weak = [h for h in hs if out["status_by_h"].get(h) == "weak"]
    if passing:
        s = (f"The distribution forecasts beat the volatility-tracking baseline at {hlist(passing)}: the best of the "
             f"{nfam} frozen distribution models improves CRPS by {by_h({h: best[h]['skill_vol'] for h in passing}, lambda v: pct(v, 1))}"
             f" (SPA over the family p-values " + ", ".join(plv(dist["info"][h]["spa_vol"]) for h in passing) + ").")
        if weak:
            s += f" At {hlist(weak)} the point estimate is positive but not significant."
    elif weak:
        s = (f"The distribution forecasts score better than the volatility baseline at {hlist(weak)} (best: "
             f"{by_h({h: best[h]['skill_vol'] for h in weak}, lambda v: pct(v, 1))}), but not significantly once "
             f"the search over the {nfam} frozen distribution models is allowed for (SPA p-values "
             + ", ".join(plv(dist["info"][h]["spa_vol"]) for h in weak) + ").")
    else:
        s = "No distribution model beats the volatility-tracking baseline."
    out["text"].append(s)
    comp = {h: _row(dist["pooled"][h], config.COMPOSITE) for h in hs}
    if all(comp.values()):
        out["text"].append(
            f"The composite forecast, reported on its own, improves CRPS over the volatility baseline by "
            f"{by_h({h: comp[h]['skill_vol'] for h in hs}, lambda v: pct(v, 1))} (one-sided Diebold-Mariano p-values "
            + ", ".join(plv(comp[h]["p_vol"]) for h in hs) + ").")
    # ML vs linear, within the family
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
    href = (passing or hs)[0]
    if href in best:
        m = best[href]["model"]
        pairs = [(q, s) for q, s in zip(dist["quantiles"], dist["pinball"][href][m]["vol"]) if ok(s)]
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
        r = _row(dist["pooled"][h], cm)
        if r is None:
            continue
        c90, cp = r.get("cover_90"), r.get("calib_p")
        good = ok(c90) and abs(c90 - 0.90) <= CALIB_TOL and ok(cp) and cp >= ALPHA
        out["calib_cells"][h] = {"status": "calib_ok" if good else "calib_off",
                                 "value": f"90% band holds {pct(c90, 1, False)}",
                                 "detail": f"{gl.label(cm)}; {eq('PIT test p', cp)}"}
        t = _row(dist["tails"][h], cm)
        if t:
            s = (f"{h}m: {pct(t['hit_q01'], 1, False)} below the 1st and {pct(t['hit_q99'], 1, False)} above the "
                 f"99th percentile")
            if h == 1 or show_kupiec_long:
                s += f" (Kupiec p = {pval(t['kupiec_q01'])} and {pval(t['kupiec_q99'])})"
            tails_txt.append(s)
    if out["calib_cells"]:
        good = [h for h, c in out["calib_cells"].items() if c["status"] == "calib_ok"]
        off = [h for h, c in out["calib_cells"].items() if c["status"] != "calib_ok"]
        s = f"Calibration of the {gl.label(cm).lower()}: "
        if good:
            s += (f"accurate at {hlist(good)} (90% interval coverage within {CALIB_TOL * 100:.0f} points of 90% and "
                  f"PIT p ≥ {ALPHA:g})")
        if off:
            s += ("; " if good else "") + f"off at {hlist(off)}"
        s += ". Tail hit rates against nominal 1%: " + "; ".join(tails_txt) + "."
        out["text"].append(s)
    return out


# --------------------------------------------------------------------------- covariance

def cov_verdict(cov: dict | None) -> dict:
    out = {"cells": {}, "text": [], "status_by_h": {}}
    if not cov:
        out["text"].append("No covariance scores were found, so the covariance evidence is not shown.")
        return out
    prim, static = config.PRIMARY_COV, cov["static"]
    has_lw = static == "lw"
    sname = "static shrinkage (Ledoit-Wolf)" if has_lw else "the static sample covariance"
    vs_static, vs_sample, sb = {}, {}, {}
    for h in sorted(cov["joint"]):
        rows = {r["model"]: r for r in cov["joint"][h]}
        r = rows.get(prim)
        v = (r or {}).get("vs", {}).get(static, {})
        if not ok(v.get("diff")):
            out["cells"][h] = {"status": "na", "value": "n/a", "detail": ""}
            continue
        q = v.get("q") if ok(v.get("q")) else v.get("p")
        st = "pass" if v["diff"] > 0 and ok(q) and q < ALPHA else ("weak" if v["diff"] > 0 else "none")
        out["status_by_h"][h] = st
        vs_static[h] = v
        vs_sample[h] = (r["vs"].get("sample") or {}) if has_lw else v
        extra = f"; vs raw sample {num(vs_sample[h].get('diff'), 2, True)}" if has_lw else ""
        out["cells"][h] = {"status": st, "value": f"log score {num(v['diff'], 3, True)}",
                           "detail": f"{gl.label(prim)} vs {sname}, nats per asset; "
                                     f"{eq('q' if ok(v.get('q')) else 'p', q)}{extra}"}
        sbr = next((x for x in cov["sb"][h]["rows"] if x["model"] == prim), None)
        if sbr and static in sbr["vs"]:
            sb[h] = sbr["vs"][static]
    hs = sorted(vs_static)
    passing = [h for h in hs if out["status_by_h"][h] == "pass"]
    spa = {h: cov["info"][h]["spa"].get(static, {}).get("joint") for h in hs}
    s = (f"Against {sname}, the primary covariance model ({gl.label(prim)}) has a joint log-score gain of "
         f"{by_h({h: vs_static[h]['diff'] for h in hs}, lambda x: num(x, 3, True))} nats per asset "
         f"(one-sided Diebold-Mariano p-values " + ", ".join(plv(vs_static[h].get("p")) for h in hs) +
         "; BH q-values " + ", ".join(plv(vs_static[h].get("q")) for h in hs) + ").")
    if any(ok(v) for v in spa.values()):
        s += (" The test of whether any dynamic model beats it (SPA) gives p-values of "
              + ", ".join(plv(spa[h]) for h in hs) + ".")
    s += (f" The edge holds at {hlist(passing)}." if passing else " No reliable edge after adjustment.")
    out["text"].append(s)
    if has_lw and all(ok(vs_sample[h].get("diff")) for h in hs):
        out["text"].append(
            f"Against the raw sample covariance the gain is much larger "
            f"({by_h({h: vs_sample[h]['diff'] for h in hs}, lambda x: num(x, 2, True))}), but most of that is "
            f"conditioning: the static Ledoit-Wolf matrix is itself far better than the sample covariance. The "
            f"comparison with Ledoit-Wolf is the one that measures dynamics.")
    if sb:
        hp = cov["headline"]
        real = cov["sb"][min(cov["sb"])]["real"]
        decs = [p for p in ("1990s", "2000s", "2010s", "2020s") if ok(real.get(p))]
        regime = ""
        if len(decs) >= 2:
            regime = (f" Realized stock-bond correlation (US equities vs intermediate Treasuries, 1-month) went "
                      f"from {num(real[decs[0]], 2, True)} in the {decs[0]} to {num(real[decs[-1]], 2, True)} in the "
                      f"{decs[-1]}.")
        out["text"].append(
            f"On the stock-bond pair alone, the gain over {sname} is "
            + ", ".join(f"{num(v['diff'].get(hp), 3, True)} at {h}m (p {pval(v['p'].get(hp))})" for h, v in sb.items())
            + f": following the stock-bond correlation regime is where the dynamic model earns its keep.{regime}")
    return out


# --------------------------------------------------------------------------- strategy

def strategy_verdict(st: dict | None) -> dict:
    out = {"cells": {}, "text": [], "status_by_h": {}}
    if not st:
        out["text"].append("No strategy-test results were found, so the economic-value test is not shown.")
        return out
    rows = [r for r in st["rows"] if r["family"] == "mean" and r["universe"] == "all"]
    fam = [r for r in rows if r.get("role") == "family"]
    hs = sorted({r["horizon"] for r in fam})
    detail = {}
    for h in hs:
        g = [r for r in fam if r["horizon"] == h and ok(r.get("d_cer"))]
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
        detail[h] = (g, sig_p, sig_q, minq)
    per = " in the holdout" if st.get("period") == "holdout" else ""
    if detail:
        parts = []
        for h, (g, sig_p, sig_q, minq) in sorted(detail.items()):
            npos = sum(1 for r in g if r["d_cer"] > 0)
            parts.append(f"{h}m: {npos} of {len(g)} positive, {len(sig_p)} with p < {ALPHA:g}, {len(sig_q)} after "
                         f"adjustment (smallest q {pval(minq)})")
        out["text"].append(f"Net of trading costs{per}, each rule using a frozen mean model's forecasts against the "
                           f"same rule using historical means, in certainty-equivalent return: " + "; ".join(parts) + ".")
        if not any(v[2] for v in detail.values()):
            out["text"].append("No gain from the return forecasts survives the multiple-testing adjustment.")
        # A null verdict hides the direction: say plainly when the forecasts cost money.
        losses = []
        for h, (g, _sp, _sq, _mq) in sorted(detail.items()):
            neg = [r for r in g if r["d_cer"] < 0]
            if 2 * len(neg) <= len(g):
                continue
            neg_p = [r for r in neg if ok(r.get("d_cer_p")) and r["d_cer_p"] < ALPHA]
            worst = min(neg, key=lambda r: r["d_cer"])
            losses.append(
                f"at {h}m, {len(neg)} of {len(g)} did worse than the historical-mean version, {len(neg_p)} of them "
                f"with unadjusted p < {ALPHA:g} (largest loss {pct(worst['d_cer'])} a year: "
                f"{gl.STRATEGY_RULES.get(worst['rule'], (worst['rule'],))[0]} with {gl.label(worst['mean_model'])})")
        if losses:
            out["text"].append("Most model-driven portfolios lost to their historical-mean counterparts: "
                               + "; ".join(losses) + ".")
    comp = [r for r in rows if r.get("role") == "composite" and ok(r.get("d_cer"))]
    if comp:
        cb = max(comp, key=lambda r: r["d_cer"])
        out["text"].append(
            f"The composite forecast, reported on its own (no q-value), gains at most {pct(cb['d_cer'])} a year "
            f"({gl.STRATEGY_RULES.get(cb['rule'], (cb['rule'],))[0]}, {cb['horizon']}m, {eq('p', cb.get('d_cer_p'))}).")
    vs = [r for r in st["rows"] if r["family"] == "vs_static" and r["universe"] == "all" and r["horizon"] == 1
          and r["mean_model"] != "none" and r.get("role") in ("family", "bench")
          and r.get("cov_model") == (st.get("meta") or {}).get("cov_model", r.get("cov_model"))]
    if vs:
        beat = [r for r in vs if ok(r.get("d_sharpe")) and r["d_sharpe"] > 0 and ok(r.get("d_sharpe_p"))
                and r["d_sharpe_p"] < ALPHA]
        pos = [r for r in vs if ok(r.get("d_sharpe")) and r["d_sharpe"] > 0]
        ew = next((lv for lv in st["levels"] if lv["rule"] == "ew" and lv["universe"] == "all"), None)
        s = (f"Against 1/N (Sharpe {num(ew['sharpe'], 2)} net) " if ew else "Against 1/N ")
        s += (f"{len(pos)} of {len(vs)} one-month optimized or timed portfolios have a higher Sharpe ratio, "
              f"{len(beat)} significantly at p < {ALPHA:g} (unadjusted).")
        out["text"].append(s)
    return out


# --------------------------------------------------------------------------- assemble

def verdict(ctx: dict, show_kupiec_long: bool = True) -> dict:
    mv = mean_verdict(ctx.get("mean"))
    dv = dist_verdict(ctx.get("dist"), show_kupiec_long)
    cv = cov_verdict(ctx.get("cov"))
    sv = strategy_verdict(ctx.get("strategy"))
    cov = ctx.get("cov")
    cov_sub = "vs static shrinkage (Ledoit-Wolf)" if cov and cov.get("static") == "lw" else "vs static sample covariance"
    grid = [
        {"key": "mean", "label": "Expected return", "sub": "vs historical mean", "cells": mv["cells"]},
        {"key": "dist", "label": "Full distribution", "sub": "vs EWMA volatility baseline", "cells": dv["cells"]},
        {"key": "calib", "label": "Calibration", "sub": "the composite forecast", "cells": dv["calib_cells"]},
        {"key": "cov", "label": "Covariance", "sub": cov_sub, "cells": cv["cells"]},
        {"key": "strategy", "label": "Portfolio value", "sub": "net CER vs hist-mean rule", "cells": sv["cells"]},
    ]
    return {"headline": _headline(mv, dv, cv, sv, cov), "grid": grid, "mean": mv["text"], "dist": dv["text"],
            "cov": cv["text"], "strategy": sv["text"], "alpha": ALPHA, "labels": STATUS_LABEL}


def _headline(mv, dv, cv, sv, cov) -> str:
    pick = lambda d, s: sorted(h for h, v in d["status_by_h"].items() if v == s)  # noqa: E731
    mp, mw, dp, cp, sp = pick(mv, "pass"), pick(mv, "weak"), pick(dv, "pass"), pick(cv, "pass"), pick(sv, "pass")
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
        elif any(v == "weak" for v in dv["status_by_h"].values()):
            out.append("The distribution forecasts score better than a volatility-tracking baseline, but not "
                       "significantly once the search over models is allowed for.")
        else:
            out.append("No distribution forecast beats a volatility-tracking baseline.")
    if cv["status_by_h"]:
        target = "static shrinkage of the covariance" if cov and cov.get("static") == "lw" else \
            "the static sample covariance"
        out.append(f"Dynamic covariance forecasts beat {target} at {hlist(cp)}." if cp else
                   f"Dynamic covariance forecasts do not reliably beat {target}.")
    if sv["status_by_h"]:
        out.append("In the portfolio test, return forecasts add value net of costs that survives multiple-testing "
                   f"adjustment at {hlist(sp)}." if sp else
                   "In the portfolio test, no gain from the return forecasts survives multiple-testing adjustment.")
    return " ".join(out) if out else "No results were found."
