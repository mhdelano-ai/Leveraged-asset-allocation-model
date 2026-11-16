"""Write ``reports/questionnaire.md`` and ``dashboard/questionnaire.html`` from ``results/questionnaire/``.

    python -m assetforecast.questionnaire_page [--fragment PATH]

Both read the tables ``assetforecast.questionnaire`` wrote and compute nothing
beyond formatting. The page inlines the tables as JSON; the two questions pick one
row of ``answers.csv`` and the browser draws it on the curve of ``curve.csv``.
``--fragment`` also writes the page without its document wrapper (for hosts that
supply their own).
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
SHORT = {"world_stocks": "Stocks", "ust_int": "Treasuries", "gold": "Gold", "mf": "Managed futures"}
LOWER = {"ust_int": "Treasuries", "gold": "gold", "mf": "managed futures"}


def load(out: Path | None = None) -> dict:
    out = Q.OUT if out is None else out
    d = {t: pd.read_csv(out / f"{t}.csv", keep_default_na=False, na_values=[""]) for t in Q.TABLES}
    d["funds"]["symbol"] = d["funds"]["symbol"].fillna("")
    d["meta"] = json.loads((out / "meta.json").read_text())
    return d


def _pct(v, d=1, sign=True):
    return "" if v is None or not np.isfinite(v) else f"{100 * v:{'+' if sign else ''}.{d}f}%"


def _held(r) -> str:
    return " / ".join(f"{100 * r[f'w_{a}']:.0f}%" for a in Q.ASSETS)


def _curve_md(cv: pd.DataFrame, every: float = 0.10) -> str:
    """The rows of one part of the curve at round exposures."""
    keep = cv[np.isclose((cv["stocks"] + cv["sleeve"]) / every, np.round((cv["stocks"] + cv["sleeve"]) / every))
              | (cv["segment"] == "unlevered")]
    return _table(["Stocks", "Sleeve", "Borrowed or carried", "Forecast vol", "Replay worst fall", "Replay growth over T-bills",
                   "Replay vol", "Expected growth (90%)"],
                  [[f"{r['stocks']:.0%}", f"{r['sleeve']:.0%}", f"{r['borrowed']:.0%}", _pct(r["vol"], 1, False),
                    _pct(r["worst_fall_all"], 1), _pct(r["growth_all"]), _pct(r["vol_all"], 1, False),
                    f"{_pct(r['growth'])} ({_pct(r['growth_lo'])} to {_pct(r['growth_hi'])})"] for _, r in keep.iterrows()])


# ------------------------------------------------------------------ report
def write_report(d: dict | None = None) -> str:
    d = load() if d is None else d
    meta, s, ans, cv = d["meta"], d["meta"]["settings"], d["answers"], d["curve"]
    sw, caps = meta["sleeve_weights"], s["overlay_caps"]
    L = ["# Allocation questionnaire", "", f"*{meta['label']}.*", "",
         f"Written by `python -m assetforecast.questionnaire`; origin {meta['origin']}. Two answers, the account's tax "
         "treatment and the goal, give a stock share; the rest goes to the diversifier sleeve (intermediate Treasuries, "
         "gold, managed futures). Interactive page: `dashboard/questionnaire.html`.", "",
         "Risk decides and returns do not. Every answer is a point on one curve of mixes, and the goal's tolerance for "
         "loss picks the point. Nothing is optimized on an expected return; expected growth is shown with its interval "
         "and is not an input.", "",
         "## The rule", "",
         f"1. Each goal declares the largest fall from a previous high it can sit through. The stock share is the largest, "
         f"in steps of {s['step']:.0%}, whose worst fall in the replay stayed within it, with the rest in the sleeve.",
         f"2. The sleeve is weighted by {s['sleeve_rule_name'].lower()}: today "
         + ", ".join(f"{100 * sw[a]:.0f}% {LOWER[a]}" for a in Q.SLEEVE) + ".",
         "3. At 100% stocks nothing is left to backfill, so the sleeve goes on top, the most whose worst fall also stays "
         f"within the limit: on a margin loan in a taxable account (T-bills + {100 * s['spreads']['margin']:.2f}%, up to "
         f"{caps['margin']:.0%} of capital), through stacked funds in a retirement account, where no loan is allowed "
         f"(up to {caps['stacked']:.0%}).",
         "4. Money needed within 3 years is outside the model: T-bills.", "",
         _table(["Goal", "Fall it can sit through", "Stock share"],
                [[g["name"], "" if g["loss_limit"] is None else f"{g['loss_limit']:.0%}",
                  "T-bills" if g["loss_limit"] is None else f"{ans[ans['goal'] == k]['stocks'].iloc[0]:.0%}"]
                 for k, g in meta["goals"].items()]), "",
         "## Every answer", "",
         "Forecast vol is on the package's 12-month covariance forecast. Expected growth is over T-bills, with the 90% "
         "interval of the historical means.", "",
         _table(["Account", "Goal", "Answer", "Stocks / Treasuries / gold / managed futures", "Forecast vol",
                 "Replay worst fall", "Replay growth", "Expected growth (90%)"],
                [[meta["accounts"][r["account"]]["name"], meta["goals"][r["goal"]]["name"], r["headline"], _held(r),
                  _pct(r["vol"], 1, False), _pct(r["worst_fall_all"], 0), _pct(r["growth_all"]),
                  f"{_pct(r['growth'])} ({_pct(r['growth_lo'])} to {_pct(r['growth_hi'])})" if r["cash"] < 1 else ""]
                 for _, r in ans.iterrows()]), "",
         "## The curve", "",
         f"Stocks beside the sleeve, replayed from {meta['first_holding_month'][:7]} to {meta['returns_through'][:7]}, "
         f"rebalanced monthly. Forecast risk is lowest at {meta['least_risk_stocks']:.0%} stocks.", "",
         "Nothing borrowed:", "", _curve_md(cv[cv["segment"] == "unlevered"]), "",
         f"100% stocks with the sleeve on top, on a margin loan at T-bills + {100 * s['spreads']['margin']:.2f}% "
         f"(up to {caps['margin']:.0%} of capital, {s['max_leverage']:.0f} times capital in all):", "",
         _curve_md(cv[cv["segment"] == "margin"]), "",
         f"100% stocks with the sleeve on top, inside stacked funds (up to {caps['stacked']:.0%} of capital, where those "
         "funds use all the capital; financed inside the funds, counted at the T-bill rate):", "",
         _curve_md(cv[cv["segment"] == "stacked"]), "",
         "## What risk and expected growth are measured on", "",
         _table(["Holding", "Forecast vol", "Expected return over T-bills", "90% interval", "History from", "Read from"],
                [[r["name"], _pct(r["vol"], 1, False), _pct(r["mean_12m"]), f"{_pct(r['ci90_lo'])} to {_pct(r['ci90_hi'])}",
                  str(r["first_month"]), r["source"]] for _, r in d["inputs"].iterrows()]), "",
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
          "- The loss limits are judgements, and the stock shares follow from them. A different limit gives a different answer; the curve shows every one.",
          f"- A goal that allows 100% stocks gets the end of the curve, the most exposure the account can carry: more sleeve on top lowered the replay's worst fall, so the loss limit never stops it. Forecast volatility rises with it. On margin the end is Regulation T's limit for opening a position ({s['max_leverage']:.0f} times capital); the broker's maintenance requirement, which is not modelled, decides how far it can fall before it is called.",
          "- The replay is the allocation study's: every year had been seen, the sleeve's assets were chosen on the same years, and managed futures before 2007-03 are a paper factor. A worst fall in the past is not a bound on the next one.",
          "- Month-end values miss falls inside a month. A margin loan can be called before a month ends.",
          "- Taxes, trading costs and fund fees are not charged. Traditional and Roth accounts get the same numbers.",
          "- Expected growth rests on historical means, which the allocation study found do not carry over. It is shown, with its interval, and decides nothing.",
          "- The stacked funds hold US or global stocks, not the study's world-stock composite, and none of them is in the study's returns."]
    Q.REPORT.write_text("\n".join(L) + "\n")
    return str(Q.REPORT)


# ------------------------------------------------------------------ page
EXTRA_CSS = """
/* layout: the two questions, then the answer they pick (headline, what to hold, its place on the curve), then how it was decided */
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
.ring { display: inline-block; width: 9px; height: 9px; border-radius: 50%; border: 2px solid var(--fg); vertical-align: -1px; margin-right: 6px; }
.band { display: inline-block; width: 16px; height: 11px; border-radius: 2px; background: color-mix(in srgb, var(--fg) 12%, transparent);
  vertical-align: -1px; margin-right: 6px; }
"""

JS = r"""
(() => {
  const D = JSON.parse(document.getElementById('qn-data').textContent);
  const $ = (id) => document.getElementById(id);
  const NS = 'http://www.w3.org/2000/svg';
  const A = D.meta.assets, S = D.meta.settings, COL = { world_stocks: '--s1', ust_int: '--s2', gold: '--s3', mf: '--s4' };
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
  const picked = (name) => document.querySelector(`input[name="${name}"]:checked`).value;
  const answer = () => D.answers.find((r) => r.account === picked('qn-account') && r.goal === picked('qn-goal'));
  const fin = (r) => D.meta.accounts[r.account].financing;
  const onTop = { margin: 'on a margin loan', stacked: 'inside stacked funds' };
  // this account's curve: every stock share with the rest in the sleeve, then the sleeve on top of 100% stocks
  const path = (r) => [...D.curve.filter((p) => p.segment === 'unlevered'), ...D.curve.filter((p) => p.segment === fin(r))];
  const same = (p, r) => Math.abs(p.stocks - r.stocks) < 1e-6 && Math.abs(p.sleeve - r.sleeve) < 1e-6;
  const mixName = (p) => (p.borrowed > 0 ? `100% stocks plus ${pct(p.sleeve, 0, false)} sleeve ${onTop[p.segment]}` :
    `${pct(p.stocks, 0, false)} stocks, ${pct(p.sleeve, 0, false)} sleeve`);
  const mixTip = (p) => `Forecast volatility ${pct(p.vol, 1, false)}<br>Worst fall in the replay ${pct(p.worst_fall_all, 0)}<br>` +
    `Expected growth ${pct(p.growth)} (${pct(p.growth_lo)} to ${pct(p.growth_hi)})`;

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

  // ---- where it sits: forecast risk across, expected growth (and how little is known of it) up
  function chart(r) {
    const pts = path(r), unl = pts.filter((p) => p.segment === 'unlevered'), top = pts.filter((p) => p.segment !== 'unlevered');
    const cash = r.cash >= 1, W = 760, H = 440, m = { l: 46, r: 24, t: 20, b: 40 };
    const xmax = Math.ceil(Math.max(...pts.map((p) => p.vol), ...D.inputs.map((p) => p.vol)) * 20) / 20;
    const ymax = Math.ceil(Math.max(...pts.map((p) => p.growth_hi)) * 50) / 50;
    const ymin = Math.floor(Math.min(0, ...pts.map((p) => p.growth_lo)) * 50) / 50;
    const x = (v) => m.l + v / xmax * (W - m.l - m.r), y = (v) => H - m.b - (v - ymin) / (ymax - ymin) * (H - m.t - m.b);
    const s = svg($('qn-chart'), W, H, 'Forecast volatility against expected growth for every mix the questionnaire can give, with this answer marked');
    for (let t = ymin; t <= ymax + 1e-9; t += 0.02) {
      s.appendChild(el('line', { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: Math.abs(t) < 1e-9 ? 'zero' : 'grid' }));
      s.appendChild(el('text', { x: m.l - 6, y: y(t) + 4, 'text-anchor': 'end' }, pct(t, 0, false)));
    }
    for (let t = 0; t <= xmax + 1e-9; t += 0.05) s.appendChild(el('text', { x: x(t), y: H - m.b + 16, 'text-anchor': 'middle' }, pct(t, 0, false)));
    s.appendChild(el('text', { x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle' }, 'Forecast volatility, next 12 months'));
    s.appendChild(el('text', { x: m.l, y: 11 }, 'Expected growth over T-bills, a year'));
    const xy = (p, k) => `${x(p.vol).toFixed(1)},${y(p[k]).toFixed(1)}`;
    const from = unl.findIndex((p) => Math.abs(p.stocks - D.meta.least_risk_stocks) < 1e-6), run = pts.slice(from);
    s.appendChild(el('path', { d: 'M' + run.map((p) => xy(p, 'growth_hi')).join('L') + 'L' + run.slice().reverse().map((p) => xy(p, 'growth_lo')).join('L') + 'Z',
      fill: 'var(--fg)', opacity: 0.07 }));
    const line = (ps, attrs) => s.appendChild(el('path', { d: 'M' + ps.map((p) => xy(p, 'growth')).join('L'), fill: 'none', stroke: 'var(--fg)', 'stroke-width': 2, 'stroke-linecap': 'round', ...attrs }));
    line(unl, {});
    if (top.length) line([unl[unl.length - 1], ...top], { 'stroke-dasharray': '1 6' });
    const side = { world_stocks: [9, 15, 'start'], ust_int: [-9, 4, 'end'], gold: [-9, 4, 'end'], mf: [9, 4, 'start'] };
    D.inputs.forEach((p) => {
      const c = el('circle', { cx: x(p.vol), cy: y(p.mean_12m), r: 5, fill: `var(${COL[p.asset]})`, stroke: 'var(--bg)', 'stroke-width': 2 });
      bind(c, `<b>${p.name}</b>Forecast volatility ${pct(p.vol, 1, false)}<br>Expected growth ${pct(p.mean_12m)} (${pct(p.ci90_lo)} to ${pct(p.ci90_hi)}), history from ${String(p.first_month).slice(0, 4)}`);
      const [dx, dy, anchor] = side[p.asset];
      s.appendChild(el('text', { x: x(p.vol) + dx, y: y(p.mean_12m) + dy, 'text-anchor': anchor }, D.short[p.asset]));
      s.appendChild(c);
    });
    unl.filter((p) => [0, 0.25, 0.5, 0.75].some((t) => Math.abs(p.stocks - t) < 1e-6)).forEach((p) => {
      s.appendChild(el('circle', { cx: x(p.vol), cy: y(p.growth), r: 2.5, fill: 'var(--fg)' }));
      s.appendChild(el('text', { x: x(p.vol) - 7, y: y(p.growth) + (p.stocks < 0.01 ? 15 : -7), 'text-anchor': 'end' }, pct(p.stocks, 0, false) + ' stocks'));
    });
    if (top.length) {
      const e = top[top.length - 1];
      s.appendChild(el('text', { x: x(e.vol) + 8, y: y(e.growth) - 4 }, `plus ${pct(e.sleeve, 0, false)} sleeve`));
    }
    pts.forEach((p) => {
      const c = el('circle', { cx: x(p.vol), cy: y(p.growth), r: 8, fill: 'transparent' });
      bind(c, `<b>${mixName(p)}</b>${mixTip(p)}`);
      s.appendChild(c);
    });
    D.answers.filter((o) => o.account === r.account && o.goal !== r.goal).forEach((o) => {
      const c = el('circle', { cx: x(o.vol), cy: y(o.growth), r: 5, fill: 'var(--bg)', stroke: 'var(--fg)', 'stroke-width': 2 });
      bind(c, `<b>${D.meta.goals[o.goal].name}</b>${o.headline}` + (o.cash >= 1 ? '' : '<br>' + mixTip(o)));
      s.appendChild(c);
    });
    const me = el('circle', { cx: x(r.vol), cy: y(r.growth), r: 8, fill: 'var(--fg)', stroke: 'var(--bg)', 'stroke-width': 2 });
    bind(me, `<b>${r.headline}</b>` + (cash ? 'The yardstick: growth here is measured over T-bills' : mixTip(r)));
    s.appendChild(me);
    s.appendChild(el('text', { x: x(r.vol) + 13, y: y(r.growth) + (cash ? -10 : 17), class: 'lab', 'font-weight': 600 }, cash ? 'Your mix: T-bills' : 'Your mix'));
  }

  function figures(r) {
    const cash = r.cash >= 1, f = (t, v, sub) => `<div><dt>${t}</dt><dd>${v}<small>${sub}</small></dd></div>`;
    $('qn-figs').innerHTML = cash ? f('Forecast volatility', '0%', 'T-bills are the yardstick everything here is measured against') :
      f('Forecast volatility', pct(r.vol, 1, false), 'next 12 months') +
      f('Worst fall in the replay', pct(r.worst_fall_all, 0), `limit for this goal ${pct(-r.loss_limit, 0)}`) +
      f('Growth in the replay', pct(r.growth_all), `${pct(r.growth_live)} since the live funds began`) +
      f('Expected growth over T-bills', pct(r.growth), `90% interval ${pct(r.growth_lo)} to ${pct(r.growth_hi)}`);
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
      else if (r.financing === 'margin') cost += ` Stacked funds cannot hold this position: they carry at most ${pct(S.overlay_caps.stacked, 0, false)} on top.`;
      if (r.financing === 'stacked') cost += ` These funds leave ${pct(r.stocks_held, 0, false)} of capital in stocks, and their stocks are US or global funds, not the study's composite.`;
    }
    $('qn-cost').textContent = cost;
    $('qn-notes').innerHTML = D.meta.notes[r.account].map((n) => `<li>${n}</li>`).join('');
  }

  function table(r) {
    const tenth = (p) => Math.abs((p.stocks + p.sleeve) * 10 - Math.round((p.stocks + p.sleeve) * 10)) < 1e-6;
    const rows = path(r).filter((p) => (p.segment === 'unlevered' ? Math.abs(p.stocks * 10 - Math.round(p.stocks * 10)) < 1e-6 : tenth(p)) || same(p, r));
    $('qn-curve').innerHTML = `<thead><tr><th>Stocks</th><th>Sleeve</th><th>${fin(r) === 'margin' ? 'Borrowed' : 'Carried inside funds'}</th><th>Forecast volatility</th>` +
      '<th>Worst fall in the replay</th><th>Growth in the replay</th><th>Expected growth (90% interval)</th></tr></thead><tbody>' +
      rows.map((p) => `<tr${r.cash < 1 && same(p, r) ? ' class="pick"' : ''}><td>${pct(p.stocks, 0, false)}</td><td>${pct(p.sleeve, 0, false)}</td><td>${p.borrowed > 0 ? pct(p.borrowed, 0, false) : ''}</td>` +
        `<td>${pct(p.vol, 1, false)}</td><td>${pct(p.worst_fall_all, 1)}</td><td>${pct(p.growth_all)}</td><td>${pct(p.growth)} (${pct(p.growth_lo)} to ${pct(p.growth_hi)})</td></tr>`).join('') + '</tbody>';
  }

  function draw() {
    const r = answer(), f = fin(r), cap = S.overlay_caps[f];
    $('qn-headline').textContent = r.headline;
    $('qn-reason').textContent = r.reason;
    $('qn-context').textContent = `${D.meta.accounts[r.account].name} · ${D.meta.goals[r.goal].name}`;
    $('qn-how').textContent = f === 'margin' ? `on a margin loan, up to ${pct(cap, 0, false)} of capital` :
      `inside stacked funds, up to ${pct(cap, 0, false)} of capital, where those funds use all of it`;
    $('qn-dash').textContent = `100% stocks with the sleeve on top, ${onTop[f]}`;
    bar(r); figures(r); chart(r); funds(r); table(r);
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
    return {"meta": d["meta"], "short": SHORT, **{t: _records(d[t]) for t in Q.TABLES}}


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
Risk decides every step; expected returns decide none. Forecast origin {meta['origin']}.</p>
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

<section><h2>Every mix the questionnaire can give, and where yours sits</h2>
<p>Left to right is forecast risk, which the package can measure. The line runs from all sleeve to all stocks. Past 100%
stocks it continues with the sleeve added on top, <span id="qn-how"></span>. The goal's loss limit picks the point.</p>
<p>The shaded band is the 90% interval of each mix's expected growth. It is wide, which is why no point is chosen on it.</p>
<div class="legend">
<span><span class="sw" style="background:var(--fg);border-radius:50%"></span>your mix</span>
<span><span class="ring"></span>this account's other goals</span>
<span><span class="ln" style="background:var(--fg)"></span>stocks with the rest in the sleeve</span>
<span><span class="ln" style="background:repeating-linear-gradient(90deg,var(--fg) 0 2px,transparent 2px 6px)"></span><span id="qn-dash"></span></span>
<span><span class="band"></span>90% interval of expected growth</span>
</div>
<div class="scroll"><div id="qn-chart" style="min-width:560px"></div></div>
<p class="note">Risk is the package's 12-month covariance forecast; managed futures, which it does not cover, enter at
their last {s['recent']} months. Expected growth uses each holding's historical mean, and for managed futures only the live
funds since {meta['live_from'][:4]}. Forecast risk is lowest at {meta['least_risk_stocks']:.0%} stocks, so the line bends
back below that and the band starts there. This is not a mean-variance frontier: nothing is optimized.</p>
<div class="scroll"><table id="qn-curve"></table></div>
<p class="note">The replay holds each mix from {meta['first_holding_month'][:4]}, rebalanced monthly, with the sleeve at
{sleeve} today (by {s['sleeve_rule_name'].lower()}). The row in bold is this answer.</p>
</section>

<section><h2>One way to hold it</h2>
<div class="scroll"><table id="qn-funds"></table></div>
<p id="qn-cost"></p>
<ul id="qn-notes"></ul>
<p class="note">Funds and fees are a hand-recorded lookup from {meta['vehicles_retrieved']}: the cheapest recorded fund for each
exposure. Confirm on the issuer's page. No stock fund's fee is recorded.</p>
</section>

<section><h2>How the point is picked</h2>
<ul>
<li>Each goal states the largest fall from a previous high it can sit through.</li>
<li>The stock share is the largest whose worst fall in the replay stayed within that, with the rest in the sleeve.</li>
<li>At 100% stocks nothing is left to backfill, so the sleeve goes on top, the most whose worst fall also stays within the
limit: on a margin loan at T-bills + {100 * s['spreads']['margin']:.2f}% in a taxable account, inside stacked funds in a
retirement account.</li>
<li>Money needed within 3 years is outside the model and gets T-bills.</li>
</ul>
</section>

<section><h2>What this does not show</h2>
<ul>
<li>The loss limits are judgements, and the stock shares follow from them. The table above shows what any other limit would give.</li>
<li>A goal that allows 100% stocks gets the end of the line, the most exposure the account can carry. More sleeve on top
lowered the replay's worst fall, so the loss limit never stops it; forecast risk rises with it. On margin the end is
Regulation T's limit for opening a position, {s['max_leverage']:.0f} times capital. The broker's maintenance requirement
is not modelled, and it decides how far the position can fall before it is called.</li>
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
