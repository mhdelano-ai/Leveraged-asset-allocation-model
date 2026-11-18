"""Write ``reports/questionnaire.md`` and ``dashboard/questionnaire.html`` from ``results/questionnaire/``.

    python -m assetforecast.questionnaire_page [--fragment PATH]

Both read the tables ``assetforecast.questionnaire`` wrote and compute nothing
beyond formatting. The page inlines the tables as JSON; the goal picks one row of
``answers.csv`` and the browser draws it on the path from ``curve.csv``; the account
picks the tax notes.
``--fragment`` also writes the page without its document wrapper (for hosts that
supply their own).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import pagekit, paths
from . import questionnaire as Q
from .allocation_page import SHORT
from .pagekit import CSS, EL_JS, TIP_JS, TOKENS, inline_json, md_table, records

PAGE = paths.ROOT / "dashboard" / "questionnaire.html"
LOWER = {"ust_int": "Treasuries", "gold": "gold", "mf": "managed futures"}


def load(out: Path | None = None) -> dict:
    out = Q.OUT if out is None else out
    d = {t: pd.read_csv(out / f"{t}.csv", keep_default_na=False, na_values=[""]) for t in Q.TABLES}
    for c in ("symbol", "note"):
        d["funds"][c] = d["funds"][c].fillna("")
    d["meta"] = json.loads((out / "meta.json").read_text())
    return d


def _pct(v, d=1, sign=True):
    if v is None or not np.isfinite(v):
        return ""
    v = 0.0 if round(100 * v, d) == 0 else v           # no signed zero
    return f"{100 * v:{'+' if sign and v != 0 else ''}.{d}f}%"


def _held(r) -> str:
    return " / ".join(f"{100 * r[f'w_{a}']:.0f}%" for a in Q.ASSETS)


def _sleeve(meta: dict) -> str:
    return ", ".join(f"{100 * meta['sleeve_weights'][a]:.0f}% {LOWER[a]}" for a in Q.SLEEVE)


def _path_md(cv: pd.DataFrame, every: float = 0.10) -> str:
    """The path at every tenth of the stock share."""
    keep = cv[np.isclose(cv["stocks"] / every, np.round(cv["stocks"] / every))]
    return md_table(["Stocks", "Diversifiers", "Forecast vol", "Replay worst fall", "Replay growth over T-bills",
                   "Expected growth (90%)"],
                  [[f"{r['stocks']:.0%}", f"{r['sleeve']:.0%}", _pct(r["vol"], 1, False), _pct(r["worst_fall_all"], 1),
                    _pct(r["growth_all"]), f"{_pct(r['growth'])} ({_pct(r['growth_lo'])} to {_pct(r['growth_hi'])})"]
                   for _, r in keep.iterrows()])


def facts(d: dict) -> dict:
    """Numbers the sentences quote, each read from one row of the path."""
    first = d["curve"].sort_values("i").iloc[0]                    # no stocks, only the diversifiers
    return {"sleeve_fall": float(first["worst_fall_all"])}


# ------------------------------------------------------------------ report
def write_report(d: dict | None = None) -> str:
    d = load() if d is None else d
    meta, s, ans, cv, f = d["meta"], d["meta"]["settings"], d["answers"], d["curve"], facts(d)
    L = ["# Allocation questionnaire", "", f"*{meta['label']}.*", "",
         f"Written by `python -m assetforecast.questionnaire`; origin {meta['origin']}. One choice: how much in world "
         "stocks. The rest of the capital is in the diversifiers (intermediate Treasuries, gold, managed futures). Nothing "
         "is borrowed: every answer holds at most 100% of capital. Interactive page: `dashboard/questionnaire.html`.", "",
         "Risk decides and returns do not. Nothing is chosen on an expected return; expected growth is shown with its "
         "interval and is not an input.", "",
         "## How the mix is set", "",
         "1. Stocks cause the large falls, so the goal's drawdown budget sets the stock share: the largest, in steps of "
         f"{s['step']:.0%}, whose worst fall in the replay stayed within the budget. The fall is measured on the whole "
         "portfolio, diversifiers included.",
         f"2. The diversifiers held up when stocks fell, so they take the rest of the capital, always in the same "
         f"proportions ({_sleeve(meta)}; {s['sleeve_rule_name'].lower()}).",
         "3. Money needed within 3 years has no budget: T-bills.", "",
         "## Every answer", "",
         "Forecast vol is on the package's 12-month covariance forecast. Expected growth is over T-bills, with the 90% "
         "interval of the historical means. The account's tax treatment does not change the mix; it changes the notes "
         "below.", "",
         md_table(["Goal", "Drawdown budget", "Answer", "Stocks / Treasuries / gold / managed futures", "Forecast vol",
                 "Replay worst fall", "Replay growth", "Expected growth (90%)", "Fund fees"],
                [[meta["goals"][r["goal"]]["name"], "" if pd.isna(r["loss_limit"]) else f"{r['loss_limit']:.0%}",
                  r["headline"], _held(r), _pct(r["vol"], 1, False), _pct(r["worst_fall_all"], 0), _pct(r["growth_all"]),
                  f"{_pct(r['growth'])} ({_pct(r['growth_lo'])} to {_pct(r['growth_hi'])})" if r["cash"] < 1 else "",
                  _pct(r["fund_fees"], 2, False) if r["cash"] < 1 else ""] for _, r in ans.iterrows()]), "",
         "## Tax notes", "",
         *[f"- **{meta['accounts'][k]['name']}.** " + " ".join(v) for k, v in meta["notes"].items()], "",
         "## Every stock share", "",
         f"Each stock share with the diversifiers in the rest of the capital, replayed from "
         f"{meta['first_holding_month'][:7]} to {meta['returns_through'][:7]}, rebalanced monthly.", "",
         _path_md(cv), "",
         "## What risk and expected growth are measured on", "",
         md_table(["Holding", "Forecast vol", "Expected return over T-bills", "90% interval", "History from", "Read from"],
                [[r["name"], _pct(r["vol"], 1, False), _pct(r["mean_12m"]), f"{_pct(r['ci90_lo'])} to {_pct(r['ci90_hi'])}",
                  str(r["first_month"]), r["source"]] for _, r in d["inputs"].iterrows()]), "",
         f"Managed futures are not in the covariance forecast: their volatility and correlations are the last "
         f"{s['recent']} months'. Their mean is the live funds' only.", "",
         "## Funds", "",
         f"One way to hold each answer, from `data/processed/diversifier_vehicles.json` (hand-recorded on "
         f"{meta['vehicles_retrieved']}; confirm fees on the issuer's page).", ""]
    for goal in ("medium", "income", "growth"):
        r = ans[ans["goal"] == goal].iloc[0]
        fd = d["funds"][d["funds"]["goal"] == goal]
        L += [f"{meta['goals'][goal]['name']}: {r['headline']}.", "",
              md_table(["Fund", "Name", "Share of capital", "Fee a year"],
                     [[x["symbol"], x["name"], f"{100 * x['weight']:.1f}%", _pct(x["fee"], 2, False)] for _, x in fd.iterrows()]),
              "", f"Recorded fund fees: {_pct(r['fund_fees'], 2, False)} of capital a year"
              + ("; no stock fund's fee is recorded." if r["stocks"] > 0 else "."), ""]
    L += ["## Caveats", "",
          "- The drawdown budgets are judgements, and the stock shares follow from them. The tables show what any other budget would give.",
          f"- The diversifiers fall too. With no stocks at all, the diversifiers alone fell {_pct(f['sleeve_fall'], 0)} in the replay.",
          "- The replay is the allocation study's: every year had been seen, the diversifiers were chosen on the same years, and managed futures before 2007-03 are a paper factor. A worst fall in the past is not a bound on the next one.",
          "- Month-end values miss falls inside a month.",
          "- Taxes, trading costs and fund fees are not charged.",
          "- Expected growth rests on historical means, which the allocation study found do not carry over. It is shown, with its interval, and decides nothing."]
    Q.REPORT.write_text("\n".join(L) + "\n")
    return str(Q.REPORT)


# ------------------------------------------------------------------ page
EXTRA_CSS = """
/* layout: the questions, then the answer (stocks, diversifiers), then every stock share, then how to hold it */
[hidden] { display: none !important; }
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
.dials { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px 32px; }
.dials > div { min-width: 0; display: flex; flex-direction: column; gap: 4px; }
.dials .big { font-family: var(--serif); font-size: 2.25rem; line-height: 1.1; font-variant-numeric: tabular-nums; }
.dials .big small { font-family: var(--sans); font-size: .9375rem; color: var(--muted); margin-left: 6px; }
.dials p { font-size: .875rem; color: var(--muted); }
.figs { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px 24px; margin: 0; }
.figs div { min-width: 0; }
.figs dt { font-size: .75rem; color: var(--muted); }
.figs dd { margin: 0; font-size: 1.25rem; font-variant-numeric: tabular-nums; }
.figs dd small { display: block; font-size: .75rem; color: var(--muted); }
tr.pick td { font-weight: 600; background: color-mix(in srgb, var(--fg) 6%, transparent); }
td.l, th.l { text-align: left; }
td.wrap { white-space: normal; min-width: 180px; }
ol { margin: 0; padding-left: 22px; max-width: 72ch; }
.ring { display: inline-block; width: 9px; height: 9px; border-radius: 50%; border: 2px solid var(--fg); vertical-align: -1px; margin-right: 6px; }
.band { display: inline-block; width: 16px; height: 11px; border-radius: 2px; background: color-mix(in srgb, var(--fg) 12%, transparent);
  vertical-align: -1px; margin-right: 6px; }
"""

JS = r"""
(() => {
  const D = JSON.parse(document.getElementById('qn-data').textContent);
  const $ = (id) => document.getElementById(id);
  const NS = 'http://www.w3.org/2000/svg';
  const A = D.meta.assets, COL = { world_stocks: '--s1', ust_int: '--s2', gold: '--s3', mf: '--s4' };
  const pct = (v, d = 1, sign = true) => {
    if (v == null || !isFinite(v)) return '';
    const t = Math.abs(100 * v).toFixed(d), z = Number(t) === 0;      // no signed zero
    return (z ? '' : v < 0 ? '−' : sign ? '+' : '') + t + '%';
  };
""" + EL_JS + r"""  const svg = (host, w, h, label) => {
    host.innerHTML = '';
    const s = el('svg', { viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': label });
    host.appendChild(s);
    return s;
  };
""" + TIP_JS + r"""  const picked = (name) => document.querySelector(`input[name="${name}"]:checked`).value;
  const answer = () => D.answers.find((r) => r.goal === picked('qn-goal'));
  const mixName = (p) => `${pct(p.stocks, 0, false)} stocks, ${pct(p.sleeve, 0, false)} diversifiers`;
  const mixTip = (p) => `Forecast volatility ${pct(p.vol, 1, false)}<br>Worst fall in the replay ${pct(p.worst_fall_all, 0)}<br>` +
    `Expected growth ${pct(p.growth)} (${pct(p.growth_lo)} to ${pct(p.growth_hi)})`;

  // ---- the answer: stocks, then the diversifiers in the rest
  function dials(r) {
    const cash = r.cash >= 1, g = D.meta.goals[r.goal];
    $('qn-dials').hidden = cash; $('qn-cash').hidden = !cash;
    if (cash) { $('qn-cash').textContent = r.reason; return; }
    $('qn-stocks').innerHTML = `${pct(r.stocks, 0, false)}<small>of capital in stocks</small>`;
    $('qn-stocks-why').textContent = `The most whose worst fall stayed within the ${pct(g.loss_limit, 0, false)} this goal can sit through. ` +
      `Replayed with the diversifiers beside it, this fell ${pct(Math.abs(r.worst_fall_all), 0, false)}.`;
    $('qn-div').innerHTML = `${pct(r.sleeve, 0, false)}<small>of capital in diversifiers</small>`;
    $('qn-div-why').textContent = r.sleeve > 0.0005 ? 'The rest of the capital, always in the same proportions. Nothing is borrowed.' :
      'Nothing is left beside the stocks, and nothing is borrowed.';
  }

  // ---- what to hold: one bar, shares of capital
  function bar(r) {
    const host = $('qn-bar');
    if (r.cash >= 1) { host.innerHTML = ''; $('qn-barleg').innerHTML = '<span><span class="sw" style="background:var(--base)"></span>T-bills 100%</span>'; return; }
    const parts = [...A.map((a) => [D.meta.names[a], r['w_' + a], `var(${COL[a]})`]), ['T-bills', r.cash, 'var(--base)']].filter((p) => p[1] > 0.0005);
    const W = 760, H = 62, x = (v) => v * W;
    const s = svg(host, W, H, 'Shares of capital in each holding');
    let at = 0;
    parts.forEach(([name, w, fill]) => {
      const b = el('rect', { x: x(at) + 1, y: 20, width: Math.max(x(w) - 2, 1), height: 26, rx: 4, fill });
      bind(b, `<b>${name}</b>${pct(w, 1, false)} of capital`);
      s.appendChild(b);
      if (x(w) > 44) s.appendChild(el('text', { x: x(at) + x(w) / 2, y: 60, 'text-anchor': 'middle', class: 'lab' }, pct(w, 0, false)));
      at += w;
    });
    $('qn-barleg').innerHTML = parts.map(([name, w, fill]) => `<span><span class="sw" style="background:${fill}"></span>${name} ${pct(w, 1, false)}</span>`).join('');
  }

  // ---- every stock share: forecast risk across, expected growth (and how little is known of it) up
  function chart(r) {
    const pts = D.curve, cash = r.cash >= 1;
    const W = 760, H = 440, m = { l: 46, r: 24, t: 20, b: 40 };
    const xmax = Math.ceil(Math.max(...pts.map((p) => p.vol), ...D.inputs.map((p) => p.vol)) * 20) / 20;
    const ymax = Math.ceil(Math.max(...pts.map((p) => p.growth_hi), ...D.inputs.map((p) => p.mean_12m)) * 50) / 50;
    const ymin = Math.floor(Math.min(0, ...pts.map((p) => p.growth_lo)) * 50) / 50;
    const x = (v) => m.l + v / xmax * (W - m.l - m.r), y = (v) => H - m.b - (v - ymin) / (ymax - ymin) * (H - m.t - m.b);
    const s = svg($('qn-chart'), W, H, 'Forecast volatility against expected growth for every stock share, with this answer marked');
    for (let t = ymin; t <= ymax + 1e-9; t += 0.02) {
      s.appendChild(el('line', { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: Math.abs(t) < 1e-9 ? 'zero' : 'grid' }));
      s.appendChild(el('text', { x: m.l - 6, y: y(t) + 4, 'text-anchor': 'end' }, pct(t, 0, false)));
    }
    for (let t = 0; t <= xmax + 1e-9; t += 0.05) s.appendChild(el('text', { x: x(t), y: H - m.b + 16, 'text-anchor': 'middle' }, pct(t, 0, false)));
    s.appendChild(el('text', { x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle' }, 'Forecast volatility, next 12 months'));
    s.appendChild(el('text', { x: m.l, y: 11 }, 'Expected growth over T-bills, a year'));
    const xy = (p, k) => `${x(p.vol).toFixed(1)},${y(p[k]).toFixed(1)}`;
    const run = pts.slice(pts.findIndex((p) => Math.abs(p.stocks - D.meta.least_risk_stocks) < 1e-6));   // the band from the least risky share on
    s.appendChild(el('path', { d: 'M' + run.map((p) => xy(p, 'growth_hi')).join('L') + 'L' + run.slice().reverse().map((p) => xy(p, 'growth_lo')).join('L') + 'Z',
      fill: 'var(--fg)', opacity: 0.07 }));
    s.appendChild(el('path', { d: 'M' + pts.map((p) => xy(p, 'growth')).join('L'), fill: 'none', stroke: 'var(--fg)', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }));
    const side = { world_stocks: [9, 4, 'start'], ust_int: [-9, 4, 'end'], gold: [-9, 4, 'end'], mf: [9, 4, 'start'] };
    D.inputs.forEach((p) => {
      const c = el('circle', { cx: x(p.vol), cy: y(p.mean_12m), r: 5, fill: `var(${COL[p.asset]})`, stroke: 'var(--bg)', 'stroke-width': 2 });
      bind(c, `<b>${p.name}</b>Forecast volatility ${pct(p.vol, 1, false)}<br>Expected growth ${pct(p.mean_12m)} (${pct(p.ci90_lo)} to ${pct(p.ci90_hi)}), history from ${String(p.first_month).slice(0, 4)}`);
      const [dx, dy, anchor] = side[p.asset];
      s.appendChild(el('text', { x: x(p.vol) + dx, y: y(p.mean_12m) + dy, 'text-anchor': anchor }, D.short[p.asset]));
      s.appendChild(c);
    });
    pts.filter((p) => [0, 0.5].some((t) => Math.abs(p.stocks - t) < 1e-6)).forEach((p) => {
      s.appendChild(el('circle', { cx: x(p.vol), cy: y(p.growth), r: 2.5, fill: 'var(--fg)' }));
      s.appendChild(el('text', { x: x(p.vol) + 7, y: y(p.growth) + 15 }, pct(p.stocks, 0, false) + ' stocks'));
    });
    pts.forEach((p) => {
      const c = el('circle', { cx: x(p.vol), cy: y(p.growth), r: 7, fill: 'transparent' });
      bind(c, `<b>${mixName(p)}</b>${mixTip(p)}`);
      s.appendChild(c);
    });
    D.answers.filter((o) => o.goal !== r.goal).forEach((o) => {
      const c = el('circle', { cx: x(o.vol), cy: y(o.growth), r: 5, fill: 'var(--bg)', stroke: 'var(--fg)', 'stroke-width': 2 });
      bind(c, `<b>${D.meta.goals[o.goal].name}</b>${o.headline}` + (o.cash >= 1 ? '' : '<br>' + mixTip(o)));
      s.appendChild(c);
    });
    const me = el('circle', { cx: x(r.vol), cy: y(r.growth), r: 8, fill: 'var(--fg)', stroke: 'var(--bg)', 'stroke-width': 2 });
    bind(me, `<b>${r.headline}</b>` + (cash ? 'The yardstick: growth here is measured over T-bills' : mixTip(r)));
    s.appendChild(me);
    s.appendChild(el('text', { x: x(r.vol) + (cash ? 13 : 0), y: y(r.growth) - (cash ? 10 : 15), 'text-anchor': cash ? 'start' : 'middle', class: 'lab', 'font-weight': 600 }, cash ? 'Your mix: T-bills' : 'Your mix'));
  }

  function figures(r) {
    const cash = r.cash >= 1, f = (t, v, sub) => `<div><dt>${t}</dt><dd>${v}<small>${sub}</small></dd></div>`;
    $('qn-figs').innerHTML = cash ? f('Forecast volatility', '0%', 'T-bills are the yardstick everything here is measured against') :
      f('Forecast volatility', pct(r.vol, 1, false), 'next 12 months') +
      f('Worst fall in the replay', pct(r.worst_fall_all, 0), `budget for this goal ${pct(-r.loss_limit, 0)}`) +
      f('Growth in the replay', pct(r.growth_all), `${pct(r.growth_live)} since the live funds began`) +
      f('Expected growth over T-bills', pct(r.growth), `90% interval ${pct(r.growth_lo)} to ${pct(r.growth_hi)}`);
  }

  function funds(r) {
    const rows = D.funds.filter((f) => f.goal === r.goal);
    $('qn-funds').innerHTML = '<thead><tr><th class="l">Fund</th><th class="l">Name</th><th>Share of capital</th><th>Fee a year</th><th class="l">Note</th></tr></thead><tbody>' +
      rows.map((f) => `<tr><td class="l">${f.symbol || ''}</td><td class="l">${f.name}</td><td>${pct(f.weight, 1, false)}</td><td>${f.fee == null ? '' : pct(f.fee, 2, false)}</td><td class="l wrap">${f.note || ''}</td></tr>`).join('') + '</tbody>';
    $('qn-cost').textContent = r.cash < 1 ? `Recorded fund fees come to ${pct(r.fund_fees, 2, false)} of capital a year.` : '';
    $('qn-notes').innerHTML = D.meta.notes[picked('qn-account')].map((n) => `<li>${n}</li>`).join('');
  }

  function table(r) {
    const rows = D.curve.filter((p) => Math.abs(p.stocks * 10 - Math.round(p.stocks * 10)) < 1e-6 || p.i === r.i);
    $('qn-curve').innerHTML = '<thead><tr><th>Stocks</th><th>Diversifiers</th><th>Forecast volatility</th>' +
      '<th>Worst fall in the replay</th><th>Growth in the replay</th><th>Expected growth (90% interval)</th></tr></thead><tbody>' +
      rows.map((p) => `<tr${r.cash < 1 && p.i === r.i ? ' class="pick"' : ''}><td>${pct(p.stocks, 0, false)}</td><td>${pct(p.sleeve, 0, false)}</td>` +
        `<td>${pct(p.vol, 1, false)}</td><td>${pct(p.worst_fall_all, 1)}</td><td>${pct(p.growth_all)}</td><td>${pct(p.growth)} (${pct(p.growth_lo)} to ${pct(p.growth_hi)})</td></tr>`).join('') + '</tbody>';
  }

  function draw() {
    const r = answer(), acc = D.meta.accounts[picked('qn-account')];
    $('qn-context').textContent = [D.meta.goals[r.goal].name, acc.name].join(' · ');
    dials(r); bar(r); figures(r); chart(r); funds(r); table(r);
    try { localStorage.setItem('qn', JSON.stringify([picked('qn-account'), r.goal])); } catch (e) { /* storage is a convenience */ }
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
    return {"meta": d["meta"], "short": SHORT, **{t: records(d[t]) for t in Q.TABLES}}


def _options(name: str, items: dict, default: str, detail) -> str:
    return "".join(
        f'<label class="opt"><input type="radio" name="{name}" id="{name}-{k}" value="{k}"{" checked" if k == default else ""}>'
        f'<span class="t">{v["name"]}</span><span class="d">{detail(k, v)}</span></label>' for k, v in items.items())


def fragment(d: dict) -> str:
    meta, s, f = d["meta"], d["meta"]["settings"], facts(d)
    data = inline_json(payload(d))
    budget = lambda k, g: (g["detail"] + (" No room for a fall." if g["loss_limit"] is None                 # noqa: E731
                                          else f" Can sit through a fall of {g['loss_limit']:.0%}."))
    return f"""<title>Allocation Questionnaire</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>{TOKENS}{CSS}{EXTRA_CSS}</style>
<main>
<header style="display:flex;flex-direction:column;gap:12px">
<h1>How much in stocks: the fall you can sit through decides</h1>
<p>Stocks cause the big falls, so the fall your goal can sit through sets how much goes in stocks. Intermediate Treasuries,
gold and managed futures held up when stocks fell, so they take the rest of the capital. Nothing is borrowed, and expected
returns decide nothing. Forecast origin {meta['origin']}.</p>
<p class="label">{meta['label']}.</p>
</header>

<section aria-label="Questions">
<fieldset><legend>What is the money for?</legend><div class="opts">{_options("qn-goal", meta["goals"], "growth", budget)}</div></fieldset>
<fieldset><legend>How is the account taxed? It changes the tax notes, not the mix</legend><div class="opts">{_options("qn-account", meta["accounts"], "taxable", lambda k, v: v["detail"])}</div></fieldset>
</section>

<section class="answer" aria-live="polite">
<div class="q" id="qn-context"></div>
<div class="dials" id="qn-dials">
<div><div class="big" id="qn-stocks"></div><p id="qn-stocks-why"></p></div>
<div><div class="big" id="qn-div"></div><p id="qn-div-why"></p></div>
</div>
<p id="qn-cash" hidden></p>
<div id="qn-bar"></div>
<div class="legend" id="qn-barleg"></div>
<dl class="figs" id="qn-figs"></dl>
</section>

<section><h2>Every stock share, and where yours sits</h2>
<p>The line is every stock share from 0% to 100%, with the diversifiers in the rest of the capital. Left to right is
forecast risk, which the package can measure. The drawdown budget picks the stock share.</p>
<p>The shaded band is the 90% interval of each point's expected growth. It is wide, which is why no point is chosen on it.</p>
<div class="legend">
<span><span class="sw" style="background:var(--fg);border-radius:50%"></span>your mix</span>
<span><span class="ring"></span>the other goals</span>
<span><span class="ln" style="background:var(--fg)"></span>every stock share, with the diversifiers beside it</span>
<span><span class="band"></span>90% interval of expected growth</span>
</div>
<div class="scroll"><div id="qn-chart" style="min-width:560px"></div></div>
<p class="note">Risk is the package's 12-month covariance forecast; managed futures, which it does not cover, enter at
their last {s['recent']} months. Expected growth uses each holding's historical mean, and for managed futures only the live
funds since {meta['live_from'][:4]}. Nothing is optimized on it.</p>
<div class="scroll"><table id="qn-curve"></table></div>
<p class="note">The replay holds each row's shares from {meta['first_holding_month'][:4]}, rebalanced monthly. The row in
bold is this answer.</p>
</section>

<section><h2>One way to hold it</h2>
<div class="scroll"><table id="qn-funds"></table></div>
<p id="qn-cost"></p>
<ul id="qn-notes"></ul>
<p class="note">Funds and fees are a hand-recorded lookup from {meta['vehicles_retrieved']}, using the cheapest recorded
funds that fit. Confirm on the issuer's page. No stock fund's fee is recorded.</p>
</section>

<section><h2>How the mix is set</h2>
<ol>
<li>Stocks: the largest share, in steps of {s['step']:.0%}, whose worst fall in the replay stayed within the goal's
drawdown budget. The fall is measured on the whole portfolio, diversifiers included.</li>
<li>Diversifiers: the rest of the capital, always {_sleeve(meta)}. Nothing is borrowed.</li>
<li>Money needed within 3 years has no budget and stays in T-bills.</li>
</ol>
</section>

<section><h2>What this does not show</h2>
<ul>
<li>The drawdown budgets are judgements, and the stock shares follow from them. The table above shows what any other budget would give.</li>
<li>The diversifiers fall too. With no stocks at all, the diversifiers alone fell {_pct(f['sleeve_fall'], 0)} in the replay.</li>
<li>Every year in the replay had been seen before the rule was written, the diversifiers were chosen on the same years,
and managed futures before {meta['live_from'][:7]} are a paper factor. A worst fall in the past does not bound the next one.</li>
<li>Month-end values miss falls inside a month. Taxes, trading costs and fund fees are not charged.</li>
<li>This is arithmetic on the package's studies, not advice.</li>
</ul></section>
</main>
<div id="tip" hidden></div>
<script type="application/json" id="qn-data">{data}</script>
<script>{JS}</script>"""


def write_page(d: dict | None = None, fragment_path: Path | None = None) -> str:
    return pagekit.write_page(PAGE, fragment(load() if d is None else d), fragment_path)


def main(argv=None) -> int:
    return pagekit.main(__doc__, "python -m assetforecast.questionnaire_page", load, write_report, write_page, argv)


if __name__ == "__main__":
    sys.exit(main())
