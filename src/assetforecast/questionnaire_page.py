"""Write ``reports/questionnaire.md`` and ``dashboard/questionnaire.html`` from ``results/questionnaire/``.

    python -m assetforecast.questionnaire_page [--fragment PATH]

Both read the tables ``assetforecast.questionnaire`` wrote and compute nothing
beyond formatting. The page inlines the tables as JSON; the two questions pick one
row of ``answers.csv`` and the browser draws it. ``--fragment`` also writes the page
without its document wrapper (for hosts that supply their own).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from . import questionnaire as Q
from .allocation_page import CSS, TOKENS, _records, _table

PAGE = paths.ROOT / "dashboard" / "questionnaire.html"
TABLES = ("answers", "ladder", "funds", "frontier", "points", "inputs")
SHORT = {"world_stocks": "Stocks", "ust_int": "Treasuries", "gold": "Gold", "mf": "Managed futures"}
LOWER = {"ust_int": "Treasuries", "gold": "gold", "mf": "managed futures"}


def load(out: Path | None = None) -> dict:
    out = Q.OUT if out is None else out
    d = {t: pd.read_csv(out / f"{t}.csv", keep_default_na=False, na_values=[""]) for t in TABLES}
    d["funds"]["symbol"] = d["funds"]["symbol"].fillna("")
    d["meta"] = json.loads((out / "meta.json").read_text())
    return d


def _pct(v, d=1, sign=True):
    return "" if v is None or not np.isfinite(v) else f"{100 * v:{'+' if sign else ''}.{d}f}%"


def _held(r) -> str:
    return " / ".join(f"{100 * r[f'w_{a}']:.0f}%" for a in Q.ASSETS)


# ------------------------------------------------------------------ report
def write_report(d: dict | None = None) -> str:
    d = load() if d is None else d
    meta, s, ans, lad = d["meta"], d["meta"]["settings"], d["answers"], d["ladder"]
    sw = meta["sleeve_weights"]
    L = ["# Allocation questionnaire", "", f"*{meta['label']}.*", "",
         f"Written by `python -m assetforecast.questionnaire`; origin {meta['origin']}. Two answers, the account's tax "
         "treatment and the goal, give a stock share; the rest goes to the diversifier sleeve (intermediate Treasuries, "
         "gold, managed futures). Interactive page: `dashboard/questionnaire.html`.", "",
         "## The rule", "",
         f"1. Each goal declares the largest fall from a previous high it can sit through. The stock share is the largest, "
         f"in steps of {s['step']:.0%}, whose worst fall in the replay stayed within it, with the rest in the sleeve.",
         f"2. The sleeve is weighted by {s['sleeve_rule_name'].lower()}: today "
         + ", ".join(f"{100 * sw[a]:.0f}% {LOWER[a]}" for a in Q.SLEEVE) + ".",
         f"3. At 100% stocks nothing is left to backfill, so the sleeve goes on top, {s['overlay']:.0%} of capital: on a "
         f"margin loan in a taxable account (T-bills + {100 * s['spreads']['margin']:.2f}%), through stacked funds in a "
         "retirement account, where no loan is allowed.",
         "4. Money needed within 3 years is outside the model: T-bills.", "",
         _table(["Goal", "Fall it can sit through", "Stock share"],
                [[g["name"], "" if g["loss_limit"] is None else f"{g['loss_limit']:.0%}",
                  "T-bills" if g["loss_limit"] is None else f"{ans[ans['goal'] == k]['stocks'].iloc[0]:.0%}"]
                 for k, g in meta["goals"].items()]), "",
         "## Every answer", "",
         "Forecast vol and expected growth are on the four-holding frontier's inputs; growth is over T-bills, with the "
         "90% interval of the historical means. The last column is the frontier portfolio at the same volatility, with "
         "the account's financing, less the answer.", "",
         _table(["Account", "Goal", "Answer", "Stocks / Treasuries / gold / managed futures", "Forecast vol",
                 "Expected growth (90%)", "Replay growth", "Replay worst fall", "Frontier at the same risk, less this (range)"],
                [[meta["accounts"][r["account"]]["name"], meta["goals"][r["goal"]]["name"], r["headline"], _held(r),
                  _pct(r["vol"], 1, False), f"{_pct(r['growth'])} ({_pct(r['growth_lo'])} to {_pct(r['growth_hi'])})"
                  if r["cash"] < 1 else "", _pct(r["growth_all"]), _pct(r["worst_fall_all"], 0),
                  f"{_pct(r['gap'])} ({_pct(r['gap_lo'])} to {_pct(r['gap_hi'])})" if r["cash"] < 1 else ""]
                 for _, r in ans.iterrows()]), "",
         "## The ladder the stock share is read from", "",
         f"Stocks beside the sleeve, replayed from {meta['first_holding_month'][:7]} to {meta['returns_through'][:7]}, "
         "rebalanced monthly.", "",
         _table(["Stocks", "Sleeve", "Replay worst fall", "Replay growth over T-bills", "Replay vol", "Forecast vol"],
                [[f"{r['stocks']:.0%}", f"{r['sleeve']:.0%}", _pct(r["worst_fall_all"], 1), _pct(r["growth_all"]),
                  _pct(r["vol_all"], 1, False), _pct(r["vol_fcst"], 1, False)] for _, r in lad.iterrows()]), "",
         "## The frontier's inputs", "",
         _table(["Holding", "Expected return over T-bills", "90% interval", "History from", "Read from", "Forecast vol"],
                [[r["name"], _pct(r["mean_12m"]), f"{_pct(r['ci90_lo'])} to {_pct(r['ci90_hi'])}", str(r["first_month"]),
                  r["source"], _pct(r["vol"], 1, False)] for _, r in d["inputs"].iterrows()]), "",
         f"Managed futures are not in the covariance forecast: their volatility and correlations are the last "
         f"{s['recent']} months'. Their mean is the live funds' only.", "",
         "## Funds", "",
         f"One way to hold each answer, from `data/processed/diversifier_vehicles.json` (hand-recorded on "
         f"{meta['vehicles_retrieved']}; confirm fees on the issuer's page).", ""]
    for acc, goal in (("taxable", "income"), ("taxable", "growth"), ("roth", "growth")):
        r = ans[(ans["account"] == acc) & (ans["goal"] == goal)].iloc[0]
        f = d["funds"][(d["funds"]["account"] == acc) & (d["funds"]["goal"] == goal)]
        L += [f"{meta['accounts'][acc]['name']}, {meta['goals'][goal]['name'].lower()}: {r['headline']}.", "",
              _table(["Fund", "Name", "Share of capital", "Fee a year"],
                     [[x["symbol"], x["name"], f"{100 * x['weight']:.1f}%", _pct(x["fee"], 2, False)] for _, x in f.iterrows()]),
              "", f"Recorded fund fees: {_pct(r['fund_fees'], 2, False)} of capital a year"
              + (f"; loan: {_pct(r['loan_cost'], 2, False)}" if r["loan_cost"] > 0 else "")
              + (f". The same position through stacked funds: {_pct(r['stacked_fees'], 2, False)} in fees and no loan"
                 if np.isfinite(r.get("stacked_fees", np.nan)) else "")
              + (f". Stocks held: {100 * r['stocks_held']:.0f}% of capital" if r["financing"] == "stacked" else "") + ".", ""]
    L += ["## Caveats", "",
          "- The loss limits are judgements, and the stock shares follow from them. A different limit gives a different answer; the ladder shows every one.",
          "- The replay is the allocation study's: every year had been seen, the sleeve's assets were chosen on the same years, and managed futures before 2007-03 are a paper factor. A worst fall in the past is not a bound on the next one.",
          "- Month-end values miss falls inside a month. A margin loan can be called before a month ends.",
          "- Taxes, trading costs and fund fees are not charged. Traditional and Roth accounts get the same numbers.",
          "- The frontier optimizes on historical means. The allocation study found that doing so did not beat fixed weights; the gap is what the optimizer claims.",
          "- The stacked funds hold US or global stocks, not the study's world-stock composite, and none of them is in the study's returns."]
    Q.REPORT.write_text("\n".join(L) + "\n")
    return str(Q.REPORT)


# ------------------------------------------------------------------ page
EXTRA_CSS = """
/* layout: the two questions, then the answer they pick (headline, what to hold, the frontier), then how it was decided */
fieldset { border: 0; margin: 0; padding: 0; min-width: 0; display: flex; flex-direction: column; gap: 8px; }
legend { padding: 0; font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px; }
.opts { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 8px; }
.opt { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; align-items: start; padding: 10px 12px; cursor: pointer;
  border: 1px solid var(--border); border-radius: 8px; background: var(--panel); min-width: 0; }
.opt input { margin: 3px 0 0; accent-color: var(--fg); grid-row: span 2; }
.opt .t { font-weight: 600; font-size: .9375rem; }
.opt .d { grid-column: 2; font-size: .8125rem; color: var(--muted); }
.opt:has(input:checked) { border-color: var(--fg); box-shadow: inset 0 0 0 1px var(--fg); }
.opt:has(input:focus-visible) { outline: 2px solid var(--s1); outline-offset: 2px; }
.answer { border-top: 2px solid var(--fg); padding-top: 14px; gap: 16px; }
.answer .q { font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
.answer .verdict { font-family: var(--serif); font-size: 1.75rem; line-height: 1.15; text-wrap: balance; }
.figs { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px 24px; margin: 0; }
.figs div { min-width: 0; }
.figs dt { font-size: .75rem; color: var(--muted); }
.figs dd { margin: 0; font-size: 1.25rem; font-variant-numeric: tabular-nums; }
.figs dd small { display: block; font-size: .75rem; color: var(--muted); }
tr.pick td { font-weight: 600; background: color-mix(in srgb, var(--fg) 6%, transparent); }
td.l, th.l { text-align: left; }
td.wrap { white-space: normal; min-width: 180px; }
"""

JS = r"""
(() => {
  const D = JSON.parse(document.getElementById('qn-data').textContent);
  const $ = (id) => document.getElementById(id);
  const NS = 'http://www.w3.org/2000/svg';
  const A = D.meta.assets, COL = { world_stocks: '--s1', ust_int: '--s2', gold: '--s3', mf: '--s4' };
  const pct = (v, d = 1, sign = true) => (v == null || !isFinite(v) ? '' :
    (sign ? (v > 0 ? '+' : v < 0 ? '−' : '') : (v < 0 ? '−' : '')) + Math.abs(100 * v).toFixed(d) + '%');
  const el = (tag, attrs = {}, text) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  };
  const svg = (host, w, h, label) => {
    host.innerHTML = '';
    const s = el('svg', { viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': label });
    host.appendChild(s);
    return s;
  };
  const tip = $('tip');
  const showAt = (x, y, html) => {
    tip.innerHTML = html; tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let L = x - w / 2, T = y - h - 10;
    L = Math.max(8, Math.min(L, innerWidth - w - 8)); if (T < 8) T = y + 14;
    tip.style.left = L + 'px'; tip.style.top = T + 'px';
  };
  const hide = () => { tip.hidden = true; };
  const bind = (node, html) => {
    node.setAttribute('tabindex', '0');
    const on = () => { const r = node.getBoundingClientRect(); showAt(r.left + r.width / 2, r.top, html); };
    node.addEventListener('mouseenter', on); node.addEventListener('focus', on);
    node.addEventListener('mouseleave', hide); node.addEventListener('blur', hide);
  };
  const held = (r, pre = 'w_') => A.filter((a) => r[pre + a] > 0.0005).map((a) => `${D.short[a]} ${pct(r[pre + a], 0, false)}`).join('<br>');
  const picked = (name) => document.querySelector(`input[name="${name}"]:checked`).value;
  const answer = () => D.answers.find((r) => r.account === picked('qn-account') && r.goal === picked('qn-goal'));

  // ---- what to hold: one bar, shares of capital
  function bar(r) {
    const host = $('qn-bar');
    if (r.cash >= 1) { host.innerHTML = ''; $('qn-barleg').innerHTML = '<span><span class="sw" style="background:var(--base)"></span>T-bills 100%</span>'; return; }
    const W = 760, H = 62, total = A.reduce((s, a) => s + r['w_' + a], 0), x = (v) => v / Math.max(total, 1) * W;
    const s = svg(host, W, H, 'Shares of capital in each holding');
    let at = 0;
    A.forEach((a) => {
      const w = r['w_' + a]; if (w <= 0.0005) return;
      const b = el('rect', { x: x(at) + 1, y: 20, width: Math.max(x(w) - 2, 1), height: 26, rx: 4, fill: `var(${COL[a]})` });
      bind(b, `<b>${D.meta.names[a]}</b>${pct(w, 1, false)} of capital`);
      s.appendChild(b);
      if (x(w) > 44) s.appendChild(el('text', { x: x(at) + x(w) / 2, y: 60, 'text-anchor': 'middle', class: 'lab' }, pct(w, 0, false)));
      at += w;
    });
    if (total > 1.0005) {
      s.appendChild(el('line', { x1: x(1), x2: x(1), y1: 12, y2: 50, stroke: 'var(--fg)', 'stroke-width': 1.5 }));
      s.appendChild(el('text', { x: x(1) - 6, y: 12, 'text-anchor': 'end' }, 'capital, 100%'));
      s.appendChild(el('text', { x: x(1) + 6, y: 12 }, r.financing === 'margin' ? `borrowed, ${pct(r.borrowed, 0, false)}` : `carried inside funds, ${pct(r.borrowed, 0, false)}`));
    }
    $('qn-barleg').innerHTML = A.filter((a) => r['w_' + a] > 0.0005).map((a) =>
      `<span><span class="sw" style="background:var(${COL[a]})"></span>${D.meta.names[a]} ${pct(r['w_' + a], 1, false)}</span>`).join('');
  }

  // ---- where it sits: forecast risk against expected growth
  function chart(r) {
    const fin = D.meta.accounts[r.account].financing, lev = D.frontier.filter((p) => p.curve === 'levered_' + fin);
    const unl = D.frontier.filter((p) => p.curve === 'unlevered'), mix = D.points.filter((p) => p.kind === 'mix');
    const assets = D.points.filter((p) => p.kind === 'asset'), cash = r.cash >= 1;
    const W = 760, H = 430, m = { l: 46, r: 24, t: 18, b: 40 };
    const xmax = Math.ceil(Math.max(...lev.map((p) => p.vol), ...assets.map((p) => p.vol)) * 20 + 0.2) / 20;
    const ymax = Math.ceil(Math.max(...lev.map((p) => p.growth), cash ? 0 : r.growth_hi) * 50) / 50;
    const ymin = Math.floor(Math.min(0, cash ? 0 : r.growth_lo) * 50) / 50;
    const x = (v) => m.l + v / xmax * (W - m.l - m.r), y = (v) => H - m.b - (v - ymin) / (ymax - ymin) * (H - m.t - m.b);
    const s = svg($('qn-chart'), W, H, 'Forecast volatility against expected growth: the frontiers, the four holdings and this answer');
    for (let t = ymin; t <= ymax + 1e-9; t += 0.02) {
      s.appendChild(el('line', { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: Math.abs(t) < 1e-9 ? 'zero' : 'grid' }));
      s.appendChild(el('text', { x: m.l - 6, y: y(t) + 4, 'text-anchor': 'end' }, pct(t, 0, false)));
    }
    for (let t = 0; t <= xmax + 1e-9; t += 0.05) s.appendChild(el('text', { x: x(t), y: H - m.b + 16, 'text-anchor': 'middle' }, pct(t, 0, false)));
    s.appendChild(el('text', { x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle' }, 'Forecast volatility, next 12 months'));
    s.appendChild(el('text', { x: m.l, y: 10 }, 'Expected growth over T-bills, a year'));
    const line = (pts, attrs) => s.appendChild(el('path', { d: pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.vol).toFixed(1)},${y(p.growth).toFixed(1)}`).join(''), fill: 'none', ...attrs }));
    line(mix, { stroke: 'var(--muted)', 'stroke-width': 1.5, 'stroke-dasharray': '2 5', 'stroke-linecap': 'round' });
    line(unl, { stroke: 'var(--base)', 'stroke-width': 2 });
    line(lev, { stroke: 'var(--fg)', 'stroke-width': 2 });
    const dots = (pts, title) => pts.forEach((p) => {
      const c = el('circle', { cx: x(p.vol), cy: y(p.growth), r: 9, fill: 'transparent' });
      bind(c, `<b>${title(p)}</b>Volatility ${pct(p.vol, 1, false)}, growth ${pct(p.growth)}<br>${held(p)}`);
      s.appendChild(c);
    });
    dots(lev.slice(1), () => 'Frontier, ' + (fin === 'margin' ? 'with a margin loan' : 'with stacked funds'));
    dots(unl, () => 'Frontier, nothing borrowed');
    dots(mix, (p) => p.name);
    const side = { world_stocks: [9, 4, 'start'], ust_int: [-9, 4, 'end'], gold: [-9, 4, 'end'], mf: [9, 4, 'start'] };
    assets.forEach((p) => {
      const c = el('circle', { cx: x(p.vol), cy: y(p.growth), r: 5, fill: `var(${COL[p.point]})`, stroke: 'var(--bg)', 'stroke-width': 2 });
      bind(c, `<b>${p.name}</b>Volatility ${pct(p.vol, 1, false)}<br>Expected growth ${pct(p.growth)} (${pct(p.growth_lo)} to ${pct(p.growth_hi)})`);
      s.appendChild(c);
      const [dx, dy, anchor] = side[p.point];
      s.appendChild(el('text', { x: x(p.vol) + dx, y: y(p.growth) + dy, 'text-anchor': anchor }, D.short[p.point]));
    });
    if (cash) {
      s.appendChild(el('circle', { cx: x(0), cy: y(0), r: 7, fill: 'var(--fg)', stroke: 'var(--bg)', 'stroke-width': 2 }));
      s.appendChild(el('text', { x: x(0) + 12, y: y(0) - 8, class: 'lab', 'font-weight': 600 }, 'T-bills'));
      return;
    }
    const px = x(r.vol), py = y(r.growth), my = y(r.match_growth);
    s.appendChild(el('line', { x1: px, x2: px, y1: y(r.growth_lo), y2: y(r.growth_hi), stroke: 'var(--fg)', 'stroke-width': 1.5, opacity: 0.45 }));
    for (const v of [r.growth_lo, r.growth_hi]) s.appendChild(el('line', { x1: px - 5, x2: px + 5, y1: y(v), y2: y(v), stroke: 'var(--fg)', 'stroke-width': 1.5, opacity: 0.45 }));
    const mm = el('circle', { cx: px, cy: my, r: 6, fill: 'var(--bg)', stroke: 'var(--fg)', 'stroke-width': 2 });
    bind(mm, `<b>Frontier at the same risk</b>Growth ${pct(r.match_growth)}, ${pct(r.gap)} more than this answer (range ${pct(r.gap_lo)} to ${pct(r.gap_hi)})<br>${held(r, 'match_w_')}`);
    s.appendChild(mm);
    const me = el('circle', { cx: px, cy: py, r: 8, fill: 'var(--fg)', stroke: 'var(--bg)', 'stroke-width': 2 });
    bind(me, `<b>${r.headline}</b>Volatility ${pct(r.vol, 1, false)}<br>Expected growth ${pct(r.growth)} (${pct(r.growth_lo)} to ${pct(r.growth_hi)})<br>${held(r)}`);
    s.appendChild(me);
    const right = px < W - 200, dx = right ? 13 : -13, anchor = right ? 'start' : 'end';
    s.appendChild(el('text', { x: px + dx, y: py + 16, 'text-anchor': anchor, class: 'lab', 'font-weight': 600 }, 'Your mix'));
    s.appendChild(el('text', { x: px - 11, y: Math.min(my - 10, py - 14), 'text-anchor': 'end' }, 'Frontier at the same risk'));
  }

  function figures(r) {
    const cash = r.cash >= 1, f = (t, v, sub) => `<div><dt>${t}</dt><dd>${v}<small>${sub}</small></dd></div>`;
    $('qn-figs').innerHTML = cash ? f('Forecast volatility', '0%', 'T-bills are the yardstick everything here is measured against') :
      f('Forecast volatility', pct(r.vol, 1, false), 'next 12 months') +
      f('Expected growth over T-bills', pct(r.growth), `90% interval ${pct(r.growth_lo)} to ${pct(r.growth_hi)}`) +
      f('Worst fall in the replay', pct(r.worst_fall_all, 0), `limit for this goal ${pct(-r.loss_limit, 0)}`) +
      f('Growth in the replay', pct(r.growth_all), `${pct(r.growth_live)} since the live funds began`);
  }

  function compare(r) {
    const box = $('qn-compare');
    if (r.cash >= 1) { box.hidden = true; return; }
    box.hidden = false;
    const fin = D.meta.accounts[r.account].financing, zero = r.gap_lo <= 0 && r.gap_hi >= 0;
    $('qn-gap').textContent = `At the same forecast risk, the frontier portfolio (${fin === 'margin' ? 'margin loan allowed' : 'stacked funds allowed'}) expects ` +
      `${pct(r.gap)} a year ${r.gap >= 0 ? 'more' : 'less'} than this answer, with a range of ${pct(r.gap_lo)} to ${pct(r.gap_hi)}` +
      (zero ? ', which includes zero.' : '.') + ' It gets there by leaning on the holdings whose past returns were highest for their risk. ' +
      'The allocation study replayed that kind of rule and it did not beat fixed weights, so the gap is a claim of the optimizer.';
    const tot = (pre) => A.reduce((s, a) => s + r[pre + a], 0);
    $('qn-comparet').innerHTML = '<thead><tr><th></th>' + A.map((a) => `<th>${D.short[a]}</th>`).join('') +
      '<th>Total exposure</th><th>Expected growth</th></tr></thead><tbody>' +
      `<tr class="pick"><td>Your mix</td>${A.map((a) => `<td>${pct(r['w_' + a], 0, false)}</td>`).join('')}<td>${pct(tot('w_'), 0, false)}</td><td>${pct(r.growth)}</td></tr>` +
      `<tr><td>Frontier at the same risk</td>${A.map((a) => `<td>${pct(r['match_w_' + a], 0, false)}</td>`).join('')}<td>${pct(tot('match_w_'), 0, false)}</td><td>${pct(r.match_growth)}</td></tr></tbody>`;
  }

  function funds(r) {
    const rows = D.funds.filter((f) => f.account === r.account && f.goal === r.goal);
    $('qn-funds').innerHTML = '<thead><tr><th class="l">Fund</th><th class="l">Name</th><th>Share of capital</th><th>Fee a year</th><th class="l">Note</th></tr></thead><tbody>' +
      rows.map((f) => `<tr><td class="l">${f.symbol || ''}</td><td class="l">${f.name}</td><td>${pct(f.weight, 1, false)}</td><td>${f.fee == null ? '' : pct(f.fee, 2, false)}</td><td class="l wrap">${f.note || ''}</td></tr>`).join('') + '</tbody>';
    let cost = '';
    if (r.cash < 1) {
      cost = `Recorded fund fees come to ${pct(r.fund_fees, 2, false)} of capital a year` +
        (r.loan_cost > 0 ? `, and the loan costs ${pct(r.loan_cost, 2, false)} more than T-bills` : '') + '.';
      if (r.stacked_fees != null) cost += ` Holding the same position through stacked funds would cost ${pct(r.stacked_fees, 2, false)} in fees with no loan; their tax cost in this account is not measured.`;
      if (r.financing === 'stacked') cost += ` These funds leave ${pct(r.stocks_held, 0, false)} of capital in stocks, and their stocks are US or global funds, not the study's composite.`;
    }
    $('qn-cost').textContent = cost;
    $('qn-notes').innerHTML = D.meta.notes[r.account].map((n) => `<li>${n}</li>`).join('');
  }

  function ladder(r) {
    $('qn-ladder').innerHTML = '<thead><tr><th>Stocks</th><th>Sleeve</th><th>Worst fall in the replay</th><th>Growth over T-bills</th><th>Volatility in the replay</th><th>Forecast volatility</th></tr></thead><tbody>' +
      D.ladder.map((l) => `<tr${r.cash < 1 && Math.abs(l.stocks - r.stocks) < 1e-6 ? ' class="pick"' : ''}><td>${pct(l.stocks, 0, false)}</td><td>${pct(l.sleeve, 0, false)}</td>` +
        `<td>${pct(l.worst_fall_all, 1)}</td><td>${pct(l.growth_all)}</td><td>${pct(l.vol_all, 1, false)}</td><td>${pct(l.vol_fcst, 1, false)}</td></tr>`).join('') + '</tbody>';
  }

  function draw() {
    const r = answer();
    $('qn-headline').textContent = r.headline;
    $('qn-reason').textContent = r.reason;
    $('qn-context').textContent = `${D.meta.accounts[r.account].name} · ${D.meta.goals[r.goal].name}`;
    bar(r); figures(r); chart(r); compare(r); funds(r); ladder(r);
    try { localStorage.setItem('qn', JSON.stringify([r.account, r.goal])); } catch (e) { /* storage is a convenience */ }
  }
  try {
    const [a, g] = JSON.parse(localStorage.getItem('qn') || '[]');
    for (const [n, v] of [['qn-account', a], ['qn-goal', g]]) { const i = document.querySelector(`input[name="${n}"][value="${v}"]`); if (i) i.checked = true; }
  } catch (e) { /* defaults stand */ }
  document.querySelectorAll('input[type="radio"]').forEach((i) => i.addEventListener('change', draw));
  let t; addEventListener('resize', () => { clearTimeout(t); t = setTimeout(draw, 150); });
  draw();
})();
"""


def payload(d: dict) -> dict:
    return {"meta": d["meta"], "short": SHORT, **{t: _records(d[t]) for t in TABLES}}


def _options(name: str, items: dict, default: str, detail) -> str:
    return "".join(
        f'<label class="opt"><input type="radio" name="{name}" id="{name}-{k}" value="{k}"{" checked" if k == default else ""}>'
        f'<span class="t">{v["name"]}</span><span class="d">{detail(v)}</span></label>' for k, v in items.items())


def fragment(d: dict) -> str:
    meta, s = d["meta"], d["meta"]["settings"]
    data = json.dumps(payload(d), separators=(",", ":"), allow_nan=False, default=lambda o: None).replace("</", "<\\/")
    sw = meta["sleeve_weights"]
    limit = lambda g: (g["detail"] + (" Outside the model." if g["loss_limit"] is None                      # noqa: E731
                                      else f" Can sit through a fall of {g['loss_limit']:.0%}."))
    sleeve = ", ".join(f"{100 * sw[a]:.0f}% {LOWER[a]}" for a in Q.SLEEVE)
    return f"""<title>Allocation Questionnaire</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>{TOKENS}{CSS}{EXTRA_CSS}</style>
<main>
<header style="display:flex;flex-direction:column;gap:12px">
<h1>Two answers, one allocation</h1>
<p>The goal sets how much goes in stocks. The rest goes to the three diversifiers that held up beside stocks: intermediate
Treasuries, gold and managed futures. The account's tax treatment decides how anything above 100% can be held.
Forecast origin {meta['origin']}.</p>
<p class="label">{meta['label']}.</p>
</header>

<section aria-label="Questions">
<fieldset><legend>How is the account taxed?</legend><div class="opts">{_options("qn-account", meta["accounts"], "taxable", lambda v: v["detail"])}</div></fieldset>
<fieldset><legend>What is the money for?</legend><div class="opts">{_options("qn-goal", meta["goals"], "growth", limit)}</div></fieldset>
</section>

<section class="answer" aria-live="polite">
<div><div class="q" id="qn-context"></div><div class="verdict" id="qn-headline"></div></div>
<p id="qn-reason"></p>
<div id="qn-bar"></div>
<div class="legend" id="qn-barleg"></div>
<dl class="figs" id="qn-figs"></dl>
</section>

<section><h2>Where it sits against the efficient frontier</h2>
<p>The frontier is the most expected growth available at each level of forecast risk from these four holdings. The dotted
line is every stock share with the rest in the sleeve, which is the family this answer comes from.</p>
<div class="legend">
<span><span class="sw" style="background:var(--fg);border-radius:50%"></span>your mix, with the 90% interval of its growth</span>
<span><span class="ln" style="background:var(--fg)"></span>frontier, with this account's way of holding more than 100%</span>
<span><span class="ln" style="background:var(--base)"></span>frontier, nothing borrowed</span>
<span><span class="ln" style="background:repeating-linear-gradient(90deg,var(--muted) 0 3px,transparent 3px 7px)"></span>stocks plus sleeve, 0% to 100% stocks</span>
</div>
<div class="scroll"><div id="qn-chart" style="min-width:560px"></div></div>
<div id="qn-compare" style="display:flex;flex-direction:column;gap:14px">
<p id="qn-gap"></p>
<div class="scroll"><table id="qn-comparet"></table></div>
</div>
<p class="note">Risk is the package's 12-month covariance forecast; managed futures, which it does not cover, enter at
their last {s['recent']} months. Expected growth uses each holding's historical mean, and for managed futures only the live
funds since {meta['live_from'][:4]}. Borrowing is capped at {s['max_leverage']:.0f} times capital.</p>
</section>

<section><h2>One way to hold it</h2>
<div class="scroll"><table id="qn-funds"></table></div>
<p id="qn-cost"></p>
<ul id="qn-notes"></ul>
<p class="note">Funds and fees are a hand-recorded lookup from {meta['vehicles_retrieved']}: the cheapest recorded fund for each
exposure. Confirm on the issuer's page. No stock fund's fee is recorded.</p>
</section>

<section><h2>How the stock share is decided</h2>
<p>Each goal states the largest fall from a previous high it can sit through. The stock share is the largest whose worst
fall stayed within that when replayed month by month from {meta['first_holding_month'][:4]}, with the rest in the sleeve
({sleeve} today, by {s['sleeve_rule_name'].lower()}). At 100% stocks nothing is left to backfill, so the sleeve goes on
top, {s['overlay']:.0%} of capital: on a margin loan at T-bills + {100 * s['spreads']['margin']:.2f}% in a taxable account,
inside stacked funds in a retirement account.</p>
<div class="scroll"><table id="qn-ladder"></table></div>
</section>

<section><h2>What this does not show</h2>
<ul>
<li>The loss limits are judgements, and the stock shares follow from them. The table above shows what any other limit would give.</li>
<li>Every year in the replay had been seen before the rule was written, the three diversifiers were chosen on the same years,
and managed futures before {meta['live_from'][:7]} are a paper factor. A worst fall in the past does not bound the next one.</li>
<li>Month-end values miss falls inside a month, and a margin loan can be called before a month ends.</li>
<li>Taxes, trading costs and fund fees are not charged, so tax-deferred and Roth accounts get the same numbers.</li>
<li>This is arithmetic on the package's studies, not advice.</li>
</ul></section>
</main>
<div id="tip" hidden></div>
<script type="application/json" id="qn-data">{data}</script>
<script>{JS}</script>"""


def write_page(d: dict | None = None, fragment_path: Path | None = None) -> str:
    d = load() if d is None else d
    frag = fragment(d)
    PAGE.write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'
                    + frag + "\n</body>\n</html>\n")
    if fragment_path is not None:
        Path(fragment_path).write_text(frag)
    return str(PAGE)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m assetforecast.questionnaire_page", description=__doc__.split("\n")[0])
    ap.add_argument("--fragment", type=Path, default=None)
    a = ap.parse_args(argv)
    d = load()
    print(write_report(d), write_page(d, a.fragment))
    return 0


if __name__ == "__main__":
    sys.exit(main())
