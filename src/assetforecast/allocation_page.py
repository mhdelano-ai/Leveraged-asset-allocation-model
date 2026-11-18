"""Write ``reports/allocation.md`` and ``dashboard/allocation.html`` from ``results/allocation/``.

    python -m assetforecast.allocation_page [--fragment PATH]

Both read the tables ``assetforecast.allocation`` wrote and compute nothing
beyond formatting; the sentences quote those tables through :func:`facts`. The
page inlines the tables as JSON and draws them in the browser. ``--fragment``
also writes the page without its document wrapper (for hosts that supply
their own).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import allocation as A
from . import pagekit, paths
from .pagekit import CSS, EL_JS, TIP_JS, TOKENS, inline_json, md_table, num, pct, records

PAGE = paths.ROOT / "dashboard" / "allocation.html"
TABLES = ("sleeve_rules", "portfolio_rules", "risk_accuracy", "persistence", "means", "today", "menu")
PERIOD_NAMES = {"all": "whole replay", "live": "live funds only"}
SOURCE_NAMES = {"hist": "Whole past", "recent": "Last 5 years", "fcst": "Forecast, 12-month matrix",
                "fcst21": "Forecast, 21-day matrix"}
ITEM_NAMES = {**A.NAMES, "mix": "60/40 mix", "world_stocks|ust_int": "Stocks and Treasuries",
              "world_stocks|gold": "Stocks and gold", "ust_int|gold": "Treasuries and gold"}
SHORT = {"world_stocks": "Stocks", "ust_int": "Treasuries", "gold": "Gold", "mf": "Managed futures"}


def load(out: Path | None = None) -> dict:
    out = A.OUT if out is None else out
    d = {t: pd.read_csv(out / f"{t}.csv") for t in TABLES}
    d["paths"] = pd.read_csv(out / "paths.csv", parse_dates=["date"]).set_index("date")
    d["meta"] = json.loads((out / "meta.json").read_text())
    return d


def facts(d: dict) -> dict:
    """The numbers the summary sentences quote, each read from one table cell."""
    sl, pf, ps, mt, ra = d["sleeve_rules"], d["portfolio_rules"], d["persistence"], d["means"], d["risk_accuracy"]
    r = lambda t, p, rule: t[(t["period"] == p) & (t["rule"] == rule)].iloc[0]          # noqa: E731
    pooled = ps[ps["asset"] == "pooled"].iloc[0]
    acc = lambda s, item: ra[(ra["source"] == s) & (ra["kind"] == "vol") & (ra["item"] == item)].iloc[0]  # noqa: E731
    m = mt.set_index("asset")
    f = {"pooled": pooled, "means": m, "acc": {s: acc(s, "mix") for s in ("hist", "recent", "fcst", "fcst21")}}
    for p in A.PERIODS:
        f[p] = {"sleeve": {x: r(sl, p, x) for x in A.SLEEVE_RULES}, "portfolio": {x: r(pf, p, x) for x in A.PORTFOLIO_RULES}}
    return f


def _clear(row) -> str:
    """Whether a rule's Sharpe difference from the no-estimate rule is outside its 90% interval."""
    if not np.isfinite(row["diff_lo"]):
        return ""
    return "better" if row["diff_lo"] > 0 else "worse" if row["diff_hi"] < 0 else "no clear difference"


def answers(d: dict) -> list[tuple[str, str, str]]:
    """(question, verdict, evidence) for the three inputs. The verdicts are a reading of this run's tables."""
    f = facts(d)
    a, l, m, po = f["all"], f["live"], f["means"], f["pooled"]
    g = m.loc["gold"]
    mf = m.loc["mf"]
    best_live = max((l["sleeve"][x]["sharpe_diff"] for x in ("maxsharpe_hist", "maxsharpe_recent", "maxsharpe_mix")))
    ih_a, ih_l, eq_a = a["sleeve"]["invvol_hist"], l["sleeve"]["invvol_hist"], a["sleeve"]["equal"]
    tf, fx, th = a["portfolio"]["target_fcst"], a["portfolio"]["fixed"], a["portfolio"]["target_hist"]
    opts = ("opt_hist", "opt_recent", "opt_mix")
    oa = max((a["portfolio"][x] for x in opts), key=lambda r: r["sharpe_diff"])
    ol = l["portfolio"][oa["rule"]]                                    # the same rule, live funds only
    scaled_hist = {"worse": "was worse than not scaling", "better": "was better than not scaling",
                   "no clear difference": "made no clear difference"}[_clear(th)]
    return [
        ("Expected returns", "Do not forecast them, and do not optimize on them",
         f"A holding's return in one five-year block said nothing about the next (correlation {num(po['corr_mean'])} "
         f"over {po['pairs']:.0f} pairs of blocks). Gold's return per unit of risk was {num(g['sharpe_first_half'])} in the "
         f"first half of the window and {num(g['sharpe_second_half'])} in the second; managed futures went from "
         f"{num(mf['sharpe_first_half'])} to {num(mf['sharpe_second_half'])}. Since the live funds began, all three "
         f"optimizers trailed a third each (best of them {num(best_live)} in Sharpe ratio, none clearly different). "
         f"Over all four holdings, the optimizer that did best over the whole replay ({oa['name'].lower()}) had a Sharpe "
         f"ratio {num(oa['sharpe_diff'])} against the fixed 60/40 mix's ({_clear(oa)}), at {pct(oa['vol'], 1, False)} "
         f"volatility against {pct(fx['vol'], 1, False)}; since the live funds began, {num(ol['sharpe_diff'])} "
         f"({_clear(ol)}). Before 2007 the replay's managed futures are the paper factor."),
        ("Weights inside the sleeve", "Size by long-run risk, or simply a third each",
         f"Risk does carry over: a block's volatility against the next, correlation {num(po['corr_vol'])}. Weighting by "
         f"1/volatility on the whole past ran at {pct(ih_a['vol'], 1, False)} volatility against {pct(eq_a['vol'], 1, False)} "
         f"for a third each, with a worst fall of {pct(ih_a['worst_fall'], 0)} against {pct(eq_a['worst_fall'], 0)}. Its Sharpe "
         f"ratio was {num(ih_a['sharpe_diff'])} higher over the whole replay ({_clear(ih_a)}) and {num(ih_l['sharpe_diff'])} "
         f"since the live funds began ({_clear(ih_l)}). The forecast gave the same weights for "
         f"{a['sleeve']['invvol_fcst']['turnover'] / max(ih_a['turnover'], 1e-9):.0f} times the trading."),
        ("How much risk in total", "Your choice; use the forecast to know what you chose",
         f"No input picks the stock share: it is the risk you accept. Scaling the 60/40 mix to a 10% risk target with the "
         f"forecast gave about the same Sharpe ratio as leaving it fixed ({num(tf['sharpe'], 2, False)} against "
         f"{num(fx['sharpe'], 2, False)}, {_clear(tf)}) with a worst fall of {pct(tf['worst_fall'], 0)} against "
         f"{pct(fx['worst_fall'], 0)}; scaling with past volatility {scaled_hist} ({num(th['sharpe_diff'])}). The forecast's "
         f"12-month matrix stood {pct(f['acc']['fcst']['ratio'] - 1, 0)} above month-to-month volatility, so a target set "
         f"with it runs below the target."),
    ]


# ------------------------------------------------------------------ report
def _rules_md(t: pd.DataFrame, period: str, sleeve: bool) -> str:
    t = t[t["period"] == period]
    rows = []
    for _, r in t.iterrows():
        iv = "" if not np.isfinite(r["diff_lo"]) else f"{num(r['sharpe_diff'])} ({num(r['diff_lo'])} to {num(r['diff_hi'])})"
        held = " / ".join(f"{100 * r[f'w_{a}']:.0f}%" for a in (A.SLEEVE if sleeve else A.ASSETS))
        rows.append([r["name"], pct(r["growth"]), pct(r["vol"], 1, False), num(r["sharpe"], 2, False), pct(r["worst_fall"], 0),
                     iv, f"{100 * r['turnover']:.0f}%", held, *([] if sleeve else [f"{100 * r['exposure']:.0f}%"])])
    head = ["Rule", "Growth over T-bills", "Vol", "Sharpe", "Worst fall", "Sharpe against the first row (90% interval)",
            "Turnover a year", "Average held: " + ("Treasuries / gold / managed futures" if sleeve else
                                                    "stocks / Treasuries / gold / managed futures"),
            *([] if sleeve else ["Average exposure"])]
    b = t.iloc[0]
    return f"{b['start'][:7]} to {b['end'][:7]} ({b['n']} months):\n\n" + md_table(head, rows)


def write_report(d: dict | None = None) -> str:
    d = load() if d is None else d
    meta, s = d["meta"], d["meta"]["settings"]
    L = ["# How should the allocation be set?", "", f"*{meta['label']}.*", "",
         f"Written by `python -m assetforecast.allocation`; returns through {meta['returns_through']}. Four holdings: world "
         "stocks and the diversifier study's sleeve (intermediate Treasuries, gold, managed futures). Each weighting rule "
         f"was fixed before any was run, sees only what was known at each month-end, and is held over the next month, "
         f"from {meta['first_holding_month'][:7]}. Dashboard: `dashboard/allocation.html`.", "", "## The reading", ""]
    for q, v, e in answers(d):
        L += [f"**{q}: {v.lower()}.** {e}", ""]
    L += ["So the answer is a mix, split by input: nothing from return forecasts, long-run history for the weights inside "
          "the sleeve, and the covariance forecast for the size of the total risk. These verdicts are a reading of the "
          "tables below, on years that had all been seen.", ""]

    ps, mt = d["persistence"], d["means"]
    L += ["## Can each input be known in advance?", "",
          f"One block of {s['persist']} months against the next, non-overlapping, on each holding's whole history:", "",
          md_table(["Holding", "From", "Pairs of blocks", "Volatility", "Mean return", "Return per unit of risk"],
                 [[A.NAMES.get(r["asset"], "All four, pooled"), "" if pd.isna(r.get("start")) else str(r["start"])[:4],
                   f"{r['pairs']:.0f}", num(r["corr_vol"]), num(r["corr_mean"]), num(r["corr_sharpe"])]
                  for _, r in ps.iterrows()]), "",
          "The pooled row removes each holding's own average first. Managed futures' return looks persistent because its "
          "blocks before 2007 are the paper factor.", "",
          f"Historical means on the common window ({meta['common_start'][:4]} on), with the package's 90% interval:", "",
          md_table(["Holding", "Mean a year", "90% interval", "Vol", "Return per unit of risk", "Its interval",
                  f"To {mt['first_half_end'].iloc[0][:7]}", "After"],
                 [[r["name"], pct(r["mean"]), f"{pct(r['mean_lo'])} to {pct(r['mean_hi'])}", pct(r["vol"], 1, False),
                   num(r["sharpe"]), f"{num(r['sharpe_lo'])} to {num(r['sharpe_hi'])}", num(r["sharpe_first_half"]),
                   num(r["sharpe_second_half"])] for _, r in mt.iterrows()]), ""]

    ra = d["risk_accuracy"]
    v = ra[ra["kind"] == "vol"]
    c = ra[ra["kind"] == "corr"]
    L += ["## How well each source measured the next 12 months' risk", "",
          "Volatility: QLIKE loss (lower is better), the forecast's level over the realized level, and the correlation of "
          "forecast with realized.", "",
          md_table(["Source", *[f"{ITEM_NAMES[i]}: QLIKE" for i in ("world_stocks", "ust_int", "gold", "mix")],
                  "60/40 mix: level", "60/40 mix: correlation"],
                 [[SOURCE_NAMES[sname], *[num(g.set_index("item").loc[i, "qlike"], 3, False) for i in ("world_stocks", "ust_int", "gold", "mix")],
                   num(g.set_index("item").loc["mix", "ratio"], 2, False), num(g.set_index("item").loc["mix", "corr_with_realized"])]
                  for sname, g in v.groupby("source", sort=False)]), "",
          "Correlations: mean squared error against the realized 12-month correlation.", "",
          md_table(["Source", *[ITEM_NAMES[i] for i in c["item"].unique()]],
                 [[SOURCE_NAMES[sname], *[num(x, 3, False) for x in g["mse"]]] for sname, g in c.groupby("source", sort=False)]), "",
          "The forecast does not cover managed futures. The 21-day row was added after the first run showed the 12-month "
          "matrix standing above month-to-month volatility; no rule uses it.", ""]

    L += ["## The sleeve: ten ways to weight Treasuries, gold and managed futures", ""]
    for p in A.PERIODS:
        L += [_rules_md(d["sleeve_rules"], p, True), ""]
    L += ["## The whole portfolio", "",
          f"Scaled rules aim at {s['target_vol']:.0%} volatility and never hold more than {s['max_exposure']:.0%} of "
          "capital; when the target calls for more they stay fully invested, and when it calls for less the rest is in "
          "T-bills.", ""]
    for p in A.PERIODS:
        L += [_rules_md(d["portfolio_rules"], p, False), ""]

    mn = d["menu"]
    L += ["## The menu today", "",
          f"Stocks beside the sleeve at origin {meta['returns_through']}. \"Forecast vol\" is on the forecast's 12-month "
          "matrix; the replay columns are the whole replay and, in brackets, since the live funds began.", ""]
    for rule, g in mn.groupby("sleeve_rule", sort=False):
        L += [f"Sleeve: {A.RULE_NAMES[rule].lower()}.", "",
              md_table(["Stocks", "Treasuries", "Gold", "Managed futures", "Forecast vol", "Replay growth", "Replay vol",
                      "Replay worst fall"],
                     [[f"{r['w_world_stocks']:.0%}", f"{r['w_ust_int']:.0%}", f"{r['w_gold']:.0%}", f"{r['w_mf']:.0%}",
                       pct(r["vol_fcst"], 1, False),
                       f"{pct(r['growth_all'])} ({pct(r['growth_live'])})", f"{pct(r['vol_all'], 1, False)} ({pct(r['vol_live'], 1, False)})",
                       f"{pct(r['worst_fall_all'], 0)} ({pct(r['worst_fall_live'], 0)})"] for _, r in g.iterrows()]), ""]

    L += ["## Caveats", "",
          "- Every year in the replay had been seen before the rules were written, and the three sleeve assets were chosen "
          "on the same years. Nothing here is out-of-sample.",
          f"- Managed futures before {meta['live_from'][:7]} are a paper factor scaled to the live funds' volatility less a "
          "gap measured on 2007-2026. Rules that read managed futures' past return or risk before 2007 read numbers nobody "
          "had. The live-funds period is the one to trust, and it is short.",
          "- Differences in Sharpe ratio between rules are small beside their intervals. \"No clear difference\" is the "
          "common result, not a failure of the test.",
          "- No trading costs, fund fees or taxes. Month-end values miss falls inside a month.",
          "- The rules were run as written. Two changes came after seeing results: the 21-day row of the risk table, and "
          "a rerun of the scaled rules with exposure capped at 100% of capital instead of twice capital, when the package "
          "dropped borrowing.", ""]
    A.REPORT.write_text("\n".join(L))
    return str(A.REPORT)


# ------------------------------------------------------------------ page
JS = r"""
(() => {
  const D = JSON.parse(document.getElementById('al-data').textContent);
  const $ = (id) => document.getElementById(id);
  const NS = 'http://www.w3.org/2000/svg';
  const COL = { world_stocks: '--s1', ust_int: '--s2', gold: '--s3', mf: '--s4' };
  const pct = (v, d = 1, sign = true) => (v == null || !isFinite(v) ? '' :
    (sign ? (v > 0 ? '+' : v < 0 ? '−' : '') : (v < 0 ? '−' : '')) + Math.abs(100 * v).toFixed(d) + '%');
  const num = (v, d = 2, sign = true) => (v == null || !isFinite(v) ? '' :
    (v < 0 ? '−' : sign && v > 0 ? '+' : '') + Math.abs(v).toFixed(d));
""" + EL_JS + r"""  const svg = (host, w, h) => {
    host.innerHTML = '';
    const s = el('svg', { viewBox: `0 0 ${w} ${h}`, role: 'img' });
    host.appendChild(s);
    return s;
  };
""" + TIP_JS + r"""  const narrow = () => innerWidth < 560;

  // ---- does one block tell the next
  function persistence() {
    const rows = D.persistence, W = 460, rowH = 26, left = 150, right = 44, top = 22;
    for (const [id, key] of [['al-pvol', 'corr_vol'], ['al-pmean', 'corr_mean']]) {
      const H = top + rows.length * rowH + 8, s = svg($(id), W, H);
      const x = (v) => left + (v + 1) / 2 * (W - left - right);
      for (const t of [-1, -0.5, 0, 0.5, 1]) {
        s.appendChild(el('line', { x1: x(t), x2: x(t), y1: top - 4, y2: H - 6, class: t === 0 ? 'zero' : 'grid' }));
        s.appendChild(el('text', { x: x(t), y: 10, 'text-anchor': 'middle' }, num(t, 1, true) || '0'));
      }
      rows.forEach((r, i) => {
        const y = top + i * rowH, v = r[key], pooled = r.asset === 'pooled';
        s.appendChild(el('text', { x: 0, y: y + 15, class: 'lab', 'font-weight': pooled ? 600 : 400 }, pooled ? 'All four, pooled' : D.names[r.asset]));
        const x0 = Math.min(x(0), x(v)), w = Math.max(Math.abs(x(v) - x(0)), 1);
        const b = el('rect', { x: x0, y: y + 4, width: w, height: 14, rx: 3, fill: `var(${key === 'corr_vol' ? '--s1' : '--s2'})`,
          opacity: pooled ? 1 : 0.55 });
        bind(b, `<b>${pooled ? 'All four, pooled' : D.names[r.asset]}</b>${num(v)} over ${r.pairs} pairs of 5-year blocks` +
          (pooled ? '' : `, from ${String(r.start).slice(0, 4)}`));
        s.appendChild(b);
        s.appendChild(el('text', { x: v >= 0 ? x(v) + 5 : x(v) - 5, y: y + 15, 'text-anchor': v >= 0 ? 'start' : 'end', class: 'lab' }, num(v)));
      });
    }
  }

  // ---- return per unit of risk with its interval
  function means() {
    const rows = D.means, W = 460, rowH = 34, left = 150, right = 16, top = 22, H = top + rows.length * rowH + 6;
    const s = svg($('al-means'), W, H), lo = -0.5, hi = 1.5;
    const x = (v) => left + (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * (W - left - right);
    for (const t of [-0.5, 0, 0.5, 1, 1.5]) {
      s.appendChild(el('line', { x1: x(t), x2: x(t), y1: top - 4, y2: H - 4, class: t === 0 ? 'zero' : 'grid' }));
      s.appendChild(el('text', { x: x(t), y: 10, 'text-anchor': 'middle' }, t === 0 ? '0' : num(t, 1)));
    }
    rows.forEach((r, i) => {
      const y = top + i * rowH + 14;
      s.appendChild(el('text', { x: 0, y: y + 4, class: 'lab' }, r.name));
      s.appendChild(el('line', { x1: x(r.sharpe_lo), x2: x(r.sharpe_hi), y1: y, y2: y, stroke: 'var(--s1)', 'stroke-width': 2, 'stroke-linecap': 'round' }));
      const g = el('g');
      g.appendChild(el('rect', { x: left, y: y - 14, width: W - left - right, height: 28, fill: 'transparent' }));
      g.appendChild(el('path', { d: `M${x(r.sharpe_first_half)},${y - 7} l5,7 l-5,7 l-5,-7 z`, fill: 'var(--panel)', stroke: 'var(--fg)', 'stroke-width': 1.5 }));
      g.appendChild(el('rect', { x: x(r.sharpe_second_half) - 5, y: y - 5, width: 10, height: 10, fill: 'var(--panel)', stroke: 'var(--fg)', 'stroke-width': 1.5 }));
      g.appendChild(el('circle', { cx: x(r.sharpe), cy: y, r: 5, fill: 'var(--s1)', stroke: 'var(--panel)', 'stroke-width': 2 }));
      bind(g, `<b>${r.name}</b>Whole window ${num(r.sharpe)} (90% interval ${num(r.sharpe_lo)} to ${num(r.sharpe_hi)})<br>` +
        `To ${r.first_half_end.slice(0, 7)}: ${num(r.sharpe_first_half)}<br>After: ${num(r.sharpe_second_half)}<br>` +
        `Mean ${pct(r.mean)} a year (${pct(r.mean_lo)} to ${pct(r.mean_hi)})`);
      s.appendChild(g);
    });
  }

  // ---- the replay
  const state = () => ({ table: $('al-table').value, period: $('al-period').value, rule: $('al-rule').value });
  const tableRows = (st) => (st.table === 'sleeve' ? D.sleeve_rules : D.portfolio_rules).filter((r) => r.period === st.period);

  function fillRules() {
    const st = state(), rules = st.table === 'sleeve' ? D.meta.sleeve_rules : D.meta.portfolio_rules, sel = $('al-rule');
    const keep = rules.includes(sel.value) ? sel.value : rules[st.table === 'sleeve' ? 7 : 5];
    sel.innerHTML = rules.slice(1).map((r) => `<option value="${r}">${D.meta.rule_names[r]}</option>`).join('');
    sel.value = keep === rules[0] ? rules[1] : keep;
  }

  function diffs() {
    const st = state(), rows = tableRows(st), base = rows[0], rest = rows.slice(1);
    const W = 760, rowH = 26, left = narrow() ? 200 : 270, right = 20, top = 24, H = top + rest.length * rowH + 8;
    const s = svg($('al-diff'), W, H);
    const ext = Math.max(0.3, ...rest.map((r) => Math.max(Math.abs(r.diff_lo), Math.abs(r.diff_hi))));
    const m = Math.ceil(ext * 10) / 10, x = (v) => left + (v + m) / (2 * m) * (W - left - right);
    for (const t of [-m, -m / 2, 0, m / 2, m]) {
      s.appendChild(el('line', { x1: x(t), x2: x(t), y1: top - 4, y2: H - 6, class: t === 0 ? 'zero' : 'grid' }));
      s.appendChild(el('text', { x: x(t), y: 10, 'text-anchor': 'middle' }, t === 0 ? 'same as ' + (st.table === 'sleeve' ? 'a third each' : 'fixed 60/40') : num(t, 2)));
    }
    rest.forEach((r, i) => {
      const y = top + i * rowH + 12, clear = r.diff_lo > 0 || r.diff_hi < 0, sel = r.rule === st.rule;
      s.appendChild(el('text', { x: 0, y: y + 4, class: 'lab', 'font-weight': sel ? 600 : 400 }, r.name));
      const g = el('g');
      g.appendChild(el('rect', { x: left, y: y - 12, width: W - left - right, height: 24, fill: 'transparent' }));
      g.appendChild(el('line', { x1: x(r.diff_lo), x2: x(r.diff_hi), y1: y, y2: y, stroke: 'var(--s1)', 'stroke-width': 2, 'stroke-linecap': 'round', opacity: clear ? 1 : 0.45 }));
      g.appendChild(el('circle', { cx: x(r.sharpe_diff), cy: y, r: 5, fill: clear ? 'var(--s1)' : 'var(--panel)', stroke: 'var(--s1)', 'stroke-width': 2 }));
      bind(g, `<b>${r.name}</b>Sharpe ${num(r.sharpe, 2, false)} against ${num(base.sharpe, 2, false)}<br>Difference ${num(r.sharpe_diff)} ` +
        `(90% interval ${num(r.diff_lo)} to ${num(r.diff_hi)})<br>${clear ? 'Interval excludes zero' : 'Interval includes zero'}`);
      s.appendChild(g);
    });
  }

  function rulesTable() {
    const st = state(), rows = tableRows(st), sl = st.table === 'sleeve';
    const held = sl ? ['ust_int', 'gold', 'mf'] : D.meta.assets;
    $('al-rules').innerHTML = '<thead><tr><th>Rule</th><th>Growth over T-bills</th><th>Vol</th><th>Sharpe</th><th>Worst fall</th>' +
      '<th>Worst 12 months</th><th>Turnover a year</th>' + (sl ? '' : '<th>Average exposure</th>') +
      held.map((a) => `<th>${D.short[a]}, average (range)</th>`).join('') + '</tr></thead><tbody>' +
      rows.map((r, i) => `<tr class="${i === 0 ? 'base' : ''}"><td>${r.name}</td><td>${pct(r.growth)}</td><td>${pct(r.vol, 1, false)}</td>` +
        `<td>${num(r.sharpe, 2, false)}</td><td>${pct(r.worst_fall, 0)}</td><td>${pct(r.worst_12m, 0)}</td><td>${pct(r.turnover, 0, false)}</td>` +
        (sl ? '' : `<td>${pct(r.exposure, 0, false)}</td>`) +
        held.map((a) => `<td>${pct(r['w_' + a], 0, false)} (${pct(r['w_' + a + '_lo'], 0, false)} to ${pct(r['w_' + a + '_hi'], 0, false)})</td>`).join('') +
        '</tr>').join('') + '</tbody>';
    const b = rows[0];
    $('al-span').textContent = `${b.start.slice(0, 7)} to ${b.end.slice(0, 7)}, ${b.n} months.`;
  }

  function pathChart() {
    const st = state(), rules = st.table === 'sleeve' ? D.meta.sleeve_rules : D.meta.portfolio_rules, base = rules[0];
    const start = st.period === 'live' ? D.meta.live_from : D.paths.date[0];
    const i0 = D.paths.date.findIndex((d) => d >= start), dates = D.paths.date.slice(i0);
    const series = [base, st.rule].map((r) => {
      const v = D.paths[r], ref = i0 > 0 ? v[i0 - 1] : 0;
      return { rule: r, y: v.slice(i0).map((z) => z - ref) };
    });
    const W = 760, H = 300, left = 44, right = narrow() ? 20 : 150, top = 12, bottom = 26, s = svg($('al-path'), W, H);
    const all = series.flatMap((q) => q.y), lo = Math.min(0, ...all), hi = Math.max(...all);
    const x = (i) => left + i / (dates.length - 1) * (W - left - right), y = (v) => top + (hi - v) / (hi - lo) * (H - top - bottom);
    for (const m of [0.5, 1, 2, 4, 8, 16, 32]) {
      const v = Math.log(m); if (v < lo - 1e-9 || v > hi + 1e-9) continue;
      s.appendChild(el('line', { x1: left, x2: W - right, y1: y(v), y2: y(v), class: m === 1 ? 'zero' : 'grid' }));
      s.appendChild(el('text', { x: left - 6, y: y(v) + 4, 'text-anchor': 'end' }, m + '×'));
    }
    const step = dates.length > 300 ? 5 : 2, y0 = +dates[0].slice(0, 4);
    dates.forEach((d, i) => {
      if (d.slice(5, 7) === '01' && (+d.slice(0, 4) - y0) % step === 0 && +d.slice(0, 4) > y0)
        s.appendChild(el('text', { x: x(i), y: H - 8, 'text-anchor': 'middle' }, d.slice(0, 4)));
    });
    const cols = ['var(--base)', 'var(--s1)'];
    series.forEach((q, k) => {
      s.appendChild(el('path', { d: q.y.map((v, i) => (i ? 'L' : 'M') + x(i).toFixed(1) + ',' + y(v).toFixed(1)).join(''), fill: 'none',
        stroke: cols[k], 'stroke-width': 2, 'stroke-linejoin': 'round' }));
    });
    if (!narrow()) {
      const ends = series.map((q, k) => ({ k, y: y(q.y[q.y.length - 1]), name: D.meta.rule_names[q.rule] }));
      const gap = 30 - Math.abs(ends[0].y - ends[1].y);
      if (gap > 0) { const up = ends[0].y < ends[1].y ? 0 : 1; ends[up].y -= gap / 2 + 6; ends[1 - up].y += gap / 2 - 6; }
      ends.forEach((e) => {
        const words = e.name.split(' '), lines = ['']; words.forEach((w) => { if ((lines[lines.length - 1] + ' ' + w).length > 24) lines.push(w); else lines[lines.length - 1] = (lines[lines.length - 1] + ' ' + w).trim(); });
        lines.slice(0, 2).forEach((t, j) => s.appendChild(el('text', { x: W - right + 8, y: e.y + 4 + j * 13, class: 'lab' }, t)));
      });
    }
    $('al-pathleg').innerHTML = series.map((q, k) => `<span><span class="ln" style="background:${cols[k]}"></span>${D.meta.rule_names[q.rule]}</span>`).join('');
    const cross = el('line', { y1: top, y2: H - bottom, stroke: 'var(--border)', 'stroke-width': 1, visibility: 'hidden' });
    const dots = series.map((q, k) => el('circle', { r: 4.5, fill: cols[k], stroke: 'var(--panel)', 'stroke-width': 2, visibility: 'hidden' }));
    s.appendChild(cross); dots.forEach((c) => s.appendChild(c));
    const hit = el('rect', { x: left, y: top, width: W - left - right, height: H - top - bottom, fill: 'transparent' });
    const move = (e) => {
      const r = hit.getBoundingClientRect(), cx = (e.touches ? e.touches[0].clientX : e.clientX);
      const i = Math.max(0, Math.min(dates.length - 1, Math.round((cx - r.left) / r.width * (dates.length - 1))));
      cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('visibility', 'visible');
      dots.forEach((c, k) => { c.setAttribute('cx', x(i)); c.setAttribute('cy', y(series[k].y[i])); c.setAttribute('visibility', 'visible'); });
      showAt(r.left + (x(i) - left) / (W - left - right) * r.width, r.top + 10,
        `<b>${dates[i].slice(0, 7)}</b>` + series.map((q) => `${D.meta.rule_names[q.rule]}: ${Math.exp(q.y[i]).toFixed(2)}×`).join('<br>'));
    };
    const out = () => { hide(); cross.setAttribute('visibility', 'hidden'); dots.forEach((c) => c.setAttribute('visibility', 'hidden')); };
    hit.addEventListener('mousemove', move); hit.addEventListener('touchmove', move, { passive: true });
    hit.addEventListener('mouseleave', out); hit.addEventListener('touchend', out);
    s.appendChild(hit);
  }

  // ---- stacked weights
  function stack(host, rows, label, maxX, detail) {
    const W = 760, rowH = 30, left = narrow() ? 190 : 270, right = 110, top = 24, H = top + rows.length * rowH + 4;
    const s = svg(host, W, H), x = (v) => left + v / maxX * (W - left - right);
    for (let t = 0; t <= maxX + 1e-9; t += 0.5) {
      s.appendChild(el('line', { x1: x(t), x2: x(t), y1: top - 4, y2: H - 4, class: Math.abs(t - 1) < 1e-9 ? 'zero' : 'grid' }));
      s.appendChild(el('text', { x: x(t), y: 10, 'text-anchor': 'middle' }, pct(t, 0, false) + (Math.abs(t - 1) < 1e-9 && maxX > 1 ? ' = your capital' : '')));
    }
    rows.forEach((r, i) => {
      const y = top + i * rowH;
      s.appendChild(el('text', { x: 0, y: y + 16, class: 'lab' }, label(r)));
      let acc = 0;
      for (const a of D.meta.assets) {
        const w = r.w[a] || 0; if (w < 0.0005) continue;
        const px = x(acc + w) - x(acc);
        const b = el('rect', { x: x(acc) + 1, y: y + 4, width: Math.max(px - 2, 1), height: 18, rx: 3, fill: `var(${COL[a]})` });
        bind(b, `<b>${label(r)}</b>${D.names[a]}: ${pct(w, 0, false)} of capital` + (detail ? '<br>' + detail(r) : ''));
        s.appendChild(b);
        acc += w;
      }
      s.appendChild(el('text', { x: x(acc) + 6, y: y + 17, class: 'lab' }, r.tail || ''));
    });
  }
  const legend = (id, assets = D.meta.assets) => { $(id).innerHTML = assets.map((a) => `<span><span class="sw" style="background:var(${COL[a]})"></span>${D.names[a]}</span>`).join(''); };

  function menu() {
    const rule = $('al-sleeve').value, rows = D.menu.filter((r) => r.sleeve_rule === rule);
    const name = (r) => `${pct(r.stocks, 0, false)} stocks / ${pct(r.sleeve, 0, false)} sleeve`;
    stack($('al-menu'), rows.map((r) => ({ ...r, w: { world_stocks: r.w_world_stocks, ust_int: r.w_ust_int, gold: r.w_gold, mf: r.w_mf } })),
      name, 1, (r) => `Forecast volatility ${pct(r.vol_fcst, 1, false)}`);
    $('al-menut').innerHTML = '<thead><tr><th>Mix</th><th>Stocks</th><th>Treasuries</th><th>Gold</th><th>Managed futures</th>' +
      '<th>Forecast vol</th><th>Replay growth</th><th>Replay vol</th><th>Replay worst fall</th><th>Since 2007: growth</th><th>Worst fall</th></tr></thead><tbody>' +
      rows.map((r) => `<tr><td>${name(r)}</td><td>${pct(r.w_world_stocks, 0, false)}</td><td>${pct(r.w_ust_int, 0, false)}</td><td>${pct(r.w_gold, 0, false)}</td>` +
        `<td>${pct(r.w_mf, 0, false)}</td><td>${pct(r.vol_fcst, 1, false)}</td><td>${pct(r.growth_all)}</td>` +
        `<td>${pct(r.vol_all, 1, false)}</td><td>${pct(r.worst_fall_all, 0)}</td><td>${pct(r.growth_live)}</td><td>${pct(r.worst_fall_live, 0)}</td></tr>`).join('') + '</tbody>';
  }

  function todayRules() {
    const rows = D.today.filter((r) => r.table === 'sleeve').map((r) => ({ ...r, w: { ust_int: r.ust_int, gold: r.gold, mf: r.mf } }));
    stack($('al-today'), rows, (r) => r.name, 1, (r) => `Forecast volatility of this sleeve ${pct(r.vol_fcst, 1, false)}`);
  }

  function risk() {
    const v = D.risk_accuracy.filter((r) => r.kind === 'vol'), src = [...new Set(v.map((r) => r.source))];
    const items = ['world_stocks', 'ust_int', 'gold', 'mix'], at = (s, i) => v.find((r) => r.source === s && r.item === i);
    $('al-risk').innerHTML = '<thead><tr><th>Risk measured from</th>' + items.map((i) => `<th>${D.items[i]}: loss</th>`).join('') +
      '<th>60/40 mix: forecast level / realized</th><th>60/40 mix: correlation with realized</th></tr></thead><tbody>' +
      src.map((s) => `<tr><td>${D.sources[s]}</td>` + items.map((i) => `<td>${num(at(s, i).qlike, 3, false)}</td>`).join('') +
        `<td>${num(at(s, 'mix').ratio, 2, false)}</td><td>${num(at(s, 'mix').corr_with_realized)}</td></tr>`).join('') + '</tbody>';
  }

  const replay = () => { diffs(); rulesTable(); pathChart(); };
  $('al-table').addEventListener('change', () => { fillRules(); replay(); });
  $('al-period').addEventListener('change', replay);
  $('al-rule').addEventListener('change', replay);
  $('al-sleeve').addEventListener('change', menu);
  let t; addEventListener('resize', () => { clearTimeout(t); t = setTimeout(() => { replay(); menu(); todayRules(); }, 150); });
  legend('al-menuleg'); legend('al-todayleg', D.meta.assets.slice(1));
  fillRules(); persistence(); means(); risk(); replay(); menu(); todayRules();
})();
"""


def payload(d: dict) -> dict:
    p = d["paths"]
    return {"meta": d["meta"], "names": A.NAMES, "sources": SOURCE_NAMES, "items": ITEM_NAMES,
            "short": SHORT,
            **{t: records(d[t]) for t in TABLES},
            "paths": {"date": [x.strftime("%Y-%m-%d") for x in p.index], **{c: [round(float(v), 4) for v in p[c]] for c in p.columns}}}


def fragment(d: dict) -> str:
    meta, s = d["meta"], d["meta"]["settings"]
    ans = "".join(f'<div><div class="q">{q}</div><div class="verdict">{v}</div><p>{e}</p></div>' for q, v, e in answers(d))
    data = inline_json(payload(d))
    first = meta["first_holding_month"][:4]
    return f"""<title>Allocation Inputs Study</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>{TOKENS}{CSS}</style>
<main>
<header style="display:flex;flex-direction:column;gap:12px">
<h1>Forecast, history, or neither: what should set the allocation?</h1>
<p>Stocks beside a sleeve of intermediate Treasuries, gold and managed futures. Eighteen weighting rules, each fixed in
advance, replayed month by month from {first} with only what was known at the time. Returns through {meta['returns_through']}.</p>
<p class="label">{meta['label']}.</p>
</header>

<section aria-label="Answers"><div class="answers">{ans}</div>
<p class="note">The verdicts are a reading of the tables below. Every number on this page is written by
<code>python -m assetforecast.allocation</code>.</p></section>

<section><h2>Risk carries over from one five-year block to the next. Returns do not.</h2>
<p>Each holding's own history cut into non-overlapping blocks of {s['persist']} months. A bar is the correlation between a
block's value and the next block's; the pooled row asks about changes within a holding.</p>
<div class="two">
<div><h3>Volatility</h3><div id="al-pvol"></div></div>
<div><h3>Mean return</h3><div id="al-pmean"></div>
<p class="note">Managed futures' blocks before 2007 are a paper factor, which is why its return looks persistent.</p></div>
</div></section>

<section><h2>History does not pin the returns down</h2>
<div class="two">
<div><p>Return per unit of risk since {meta['common_start'][:4]}. The line is the package's 90% interval for the whole
window; the two outlined marks are the first and second half of it.</p>
<div class="legend"><span><span class="sw" style="background:var(--s1);border-radius:50%"></span>whole window, with 90% interval</span>
<span>◇ first half</span><span>□ second half</span></div></div>
<div><div id="al-means"></div></div>
</div></section>

<section><h2>The replay: each rule against using no estimate at all</h2>
<div class="controls">
<label>Decision<select id="al-table"><option value="sleeve">Weights inside the sleeve</option><option value="portfolio">The whole portfolio</option></select></label>
<label>Period<select id="al-period"><option value="all">Whole replay, from {first}</option><option value="live">Live managed-futures funds only, from {meta['live_from'][:7]}</option></select></label>
</div>
<h3>Sharpe ratio less that of the no-estimate rule, with its 90% interval</h3>
<div class="scroll"><div id="al-diff" style="min-width:560px"></div></div>
<p class="note">A filled dot means the interval excludes zero. <span id="al-span"></span> Whole-portfolio rules other than the
fixed mix aim at {s['target_vol']:.0%} volatility and never hold more than {s['max_exposure']:.0%} of capital; nothing
borrows.</p>
<div class="scroll"><table id="al-rules"></table></div>
<div class="controls"><label>Growth of 1 over T-bills: compare with<select id="al-rule"></select></label></div>
<div class="legend" id="al-pathleg"></div>
<div class="scroll"><div id="al-path" style="min-width:560px"></div></div>
</section>

<section><h2>What the covariance forecast adds</h2>
<p>Each source's reading of the next 12 months' volatility, scored at every month-end. Loss is QLIKE, lower is better.
The forecast tracks changes in risk, which the two historical windows do not, but its level stands above what month-to-month
returns then showed.</p>
<div class="scroll"><table id="al-risk"></table></div>
<p class="note">The forecast does not cover managed futures. The 21-day row was added after the first run and no rule uses it.</p>
</section>

<section><h2>The menu today</h2>
<p>The sleeve weights come from the rule chosen here; the stock share is the risk you choose.
Weights at origin {meta['returns_through']}.</p>
<div class="controls"><label>Sleeve weights<select id="al-sleeve"><option value="invvol_hist">1/volatility on the whole past</option><option value="equal">A third each</option></select></label></div>
<div class="legend" id="al-menuleg"></div>
<div class="scroll"><div id="al-menu" style="min-width:560px"></div></div>
<div class="scroll"><table id="al-menut"></table></div>
<p class="note">Forecast vol is on the forecast's 12-month matrix. Replay columns hold these stock and sleeve shares through the
whole replay.</p>
</section>

<section><h2>What each sleeve rule would hold today</h2>
<p>The rules that read returns disagree with each other and with themselves over time; the rules that read only risk land
close together.</p>
<div class="legend" id="al-todayleg"></div>
<div class="scroll"><div id="al-today" style="min-width:560px"></div></div>
</section>

<section><h2>What this does not show</h2>
<ul>
<li>Every year in the replay had been seen before the rules were written, and the three sleeve assets were chosen on the same years.</li>
<li>Managed futures before {meta['live_from'][:7]} are a paper factor scaled to the live funds' volatility, less a gap measured
on 2007 to 2026. The live-funds period is the one to trust, and it is short.</li>
<li>No trading costs, fund fees or taxes, and month-end values miss falls inside a month.</li>
<li>Most differences between rules are small beside their intervals.</li>
<li>The scaled rules were rerun once after the first results, with exposure capped at 100% of capital instead of twice
capital, when the package dropped borrowing.</li>
</ul></section>
</main>
<div id="tip" hidden></div>
<script type="application/json" id="al-data">{data}</script>
<script>{JS}</script>"""


def write_page(d: dict | None = None, fragment_path: Path | None = None) -> str:
    return pagekit.write_page(PAGE, fragment(load() if d is None else d), fragment_path)


def main(argv=None) -> int:
    return pagekit.main(__doc__, "python -m assetforecast.allocation_page", load, write_report, write_page, argv)


if __name__ == "__main__":
    sys.exit(main())
