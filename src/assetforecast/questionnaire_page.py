"""Write ``reports/questionnaire.md`` and ``dashboard/questionnaire.html`` from ``results/questionnaire/``.

    python -m assetforecast.questionnaire_page [--fragment PATH]

Both read the tables ``assetforecast.questionnaire`` wrote and compute nothing
beyond formatting. The page inlines the tables as JSON; the questions pick one row
of ``answers.csv`` and the browser draws it on its route's path from ``curve.csv``.
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
LOWER = {"world_stocks": "stocks", "ust_int": "Treasuries", "gold": "gold", "mf": "managed futures"}


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


def _mix(meta: dict) -> str:
    return ", ".join(f"{100 * meta['mix'][a]:.0f}% {LOWER[a]}" for a in Q.ASSETS)


def _round(cv: pd.DataFrame, col: str, every: float = 0.10) -> pd.Series:
    return np.isclose(cv[col] / every, np.round(cv[col] / every))


def _path_md(cv: pd.DataFrame, route: str) -> str:
    """One route's path at round steps: of exposure while scaling, of the stock share while tilting."""
    p = cv[cv["route"] == route]
    keep = p[((p["segment"] == "scaled") & (_round(p, "exposure") | (p["i"] == p[p["segment"] == "scaled"]["i"].max())))
             | ((p["segment"] == "tilted") & _round(p, "stocks"))]
    return _table(["", "Stocks", "Diversifiers", "T-bills", "Above 100%", "Forecast vol", "Replay worst fall",
                   "Replay growth over T-bills", "Expected growth (90%)"],
                  [["the mix, scaled" if r["segment"] == "scaled" else "more stocks", f"{r['stocks']:.0%}",
                    f"{r['sleeve']:.0%}", f"{r['cash']:.0%}", f"{r['borrowed']:.0%}", _pct(r["vol"], 1, False),
                    _pct(r["worst_fall_all"], 1), _pct(r["growth_all"]),
                    f"{_pct(r['growth'])} ({_pct(r['growth_lo'])} to {_pct(r['growth_hi'])})"] for _, r in keep.iterrows()])


def facts(d: dict) -> dict:
    """Numbers the sentences quote, each read from the tables."""
    cv, meta = d["curve"], d["meta"]
    m = cv[(cv["route"] == "margin") & (cv["segment"] == "scaled")]
    one = m[np.isclose(m["exposure"], 1.0)].iloc[0]
    return {"margin_lo_at_1": float(one["growth_lo"]), "margin_lo_at_cap": float(m.iloc[-1]["growth_lo"]),
            "margin_cap": float(meta["caps"]["margin"]["exposure"]), "stacked_cap": float(meta["caps"]["stacked"]["exposure"]),
            "stacked_cap_fall": float(meta["caps"]["stacked"]["worst_fall_all"]),
            "margin_cap_fall": float(meta["caps"]["margin"]["worst_fall_all"])}


# ------------------------------------------------------------------ report
def write_report(d: dict | None = None) -> str:
    d = load() if d is None else d
    meta, s, ans, cv, f = d["meta"], d["meta"]["settings"], d["answers"], d["curve"], facts(d)
    lo, hi = meta["mix_stock_share_range"]
    L = ["# Allocation questionnaire", "", f"*{meta['label']}.*", "",
         f"Written by `python -m assetforecast.questionnaire`; origin {meta['origin']}. One mix of world stocks, "
         "intermediate Treasuries, gold and managed futures, held with T-bills beside it or above 100% of capital, as far "
         "as the goal's drawdown budget allows. Interactive page: `dashboard/questionnaire.html`.", "",
         "Risk decides and returns do not. The mix is set by each holding's volatility, and how much of it to hold by the "
         "fall the goal can sit through. Nothing is optimized on an expected return; expected growth is shown with its "
         "interval and is not an input.", "",
         "## The rule", "",
         f"1. The mix: the four holdings by {s['mix_rule']}. Today {_mix(meta)}. (Its stock share has been between "
         f"{lo:.0%} and {hi:.0%} at every month-end of the replay.)",
         "2. Scale it. Each goal declares the largest fall from a previous high it can sit through. The mix is held at "
         f"the largest multiple of capital, in steps of {s['step']:.0%}, whose worst fall in the replay stayed within "
         "that: with T-bills beside it below 100%, and above 100% by the account's route.",
         f"3. Routes. Stacked funds carry the extra exposure inside (no loan; the only route in a retirement account, the "
         f"default in a taxable one). A margin loan costs T-bills + {100 * s['spreads']['margin']:.2f}% (taxable only).",
         f"4. Where the account can carry no more. The mix reaches {f['stacked_cap']:.2f} times capital with stacked funds "
         f"(they stack a diversifier only on top of stocks, and the mix holds few) and {f['margin_cap']:.0f} times on "
         f"margin. Its worst fall there was {_pct(f['stacked_cap_fall'], 0)} and {_pct(f['margin_cap_fall'], 0)}. A goal "
         "with budget left takes it by holding stocks in place of diversifiers at that limit, up to 100% stocks.",
         "5. Money needed within 3 years has no budget: T-bills.", "",
         "## Every answer", "",
         "Forecast vol is on the package's 12-month covariance forecast. Expected growth is over T-bills, with the 90% "
         "interval of the historical means. Tax-deferred and Roth accounts get the same answers.", "",
         _table(["Account", "Above 100% by", "Goal", "Answer", "Stocks / Treasuries / gold / managed futures", "Forecast vol",
                 "Replay worst fall", "Replay growth", "Expected growth (90%)", "Fund fees", "Loan"],
                [[meta["accounts"][r["account"]]["name"], meta["routes"][r["route"]]["name"].lower(), meta["goals"][r["goal"]]["name"],
                  r["headline"], _held(r), _pct(r["vol"], 1, False), _pct(r["worst_fall_all"], 0), _pct(r["growth_all"]),
                  f"{_pct(r['growth'])} ({_pct(r['growth_lo'])} to {_pct(r['growth_hi'])})" if r["cash"] < 1 else "",
                  _pct(r["fund_fees"], 2, False) if r["cash"] < 1 else "", _pct(r["loan_cost"], 2, False) if r["loan_cost"] > 0 else ""]
                 for _, r in ans[ans["account"] != "roth"].iterrows()]), "",
         "## The path", "",
         f"Replayed from {meta['first_holding_month'][:7]} to {meta['returns_through'][:7]}, rebalanced monthly, each row "
         "at its own fixed shares.", "", "With stacked funds:", "", _path_md(cv, "stacked"), "",
         f"On a margin loan at T-bills + {100 * s['spreads']['margin']:.2f}%:", "", _path_md(cv, "margin"), "",
         "## What risk and expected growth are measured on", "",
         _table(["Holding", "Forecast vol", "Expected return over T-bills", "90% interval", "History from", "Read from"],
                [[r["name"], _pct(r["vol"], 1, False), _pct(r["mean_12m"]), f"{_pct(r['ci90_lo'])} to {_pct(r['ci90_hi'])}",
                  str(r["first_month"]), r["source"]] for _, r in d["inputs"].iterrows()]), "",
         f"Managed futures are not in the covariance forecast: their volatility and correlations are the last "
         f"{s['recent']} months'. Their mean is the live funds' only.", "",
         "## Funds", "",
         f"One way to hold each answer, from `data/processed/diversifier_vehicles.json` (hand-recorded on "
         f"{meta['vehicles_retrieved']}; confirm fees on the issuer's page).", ""]
    for route in meta["accounts"]["taxable"]["routes"]:
        for goal in ("medium", "income", "growth"):
            r = ans[(ans["account"] == "taxable") & (ans["route"] == route) & (ans["goal"] == goal)].iloc[0]
            fd = d["funds"][(d["funds"]["account"] == "taxable") & (d["funds"]["route"] == route) & (d["funds"]["goal"] == goal)]
            L += [f"{meta['goals'][goal]['name']}, {meta['routes'][route]['name'].lower()}: {r['headline']}.", "",
                  _table(["Fund", "Name", "Share of capital", "Fee a year"],
                         [[x["symbol"], x["name"], f"{100 * x['weight']:.1f}%", _pct(x["fee"], 2, False)] for _, x in fd.iterrows()]),
                  "", f"Recorded fund fees: {_pct(r['fund_fees'], 2, False)} of capital a year"
                  + (f"; loan: {_pct(r['loan_cost'], 2, False)}" if r["loan_cost"] > 0 else "")
                  + (f". Stocks held: {100 * r['stocks_held']:.0f}% of capital" if abs(r["stocks_held"] - r["stocks"]) > 5e-4 else "")
                  + ".", ""]
    L += ["## Caveats", "",
          "- The drawdown budgets are judgements, and the answers follow from them. The path shows what any other budget would give.",
          "- Every goal beyond 3 years ends where the account can carry no more, because the mix scaled that far still fell less than the budget. The budget then sets the stock share, and the diversifiers fill whatever the account can hold.",
          f"- On margin that limit is Regulation T's for opening a position, {s['max_leverage']:.0f} times capital. The broker's maintenance requirement, which is not modelled, decides how far the position can fall before it is called.",
          f"- On margin the lower end of expected growth falls as the mix is levered, from {_pct(f['margin_lo_at_1'])} at 100% of capital to {_pct(f['margin_lo_at_cap'])} at {f['margin_cap']:.0f} times: the diversifiers' return over the loan's cost is not established.",
          "- The replay is the allocation study's: every year had been seen, the diversifiers were chosen on the same years, and managed futures before 2007-03 are a paper factor. A worst fall in the past is not a bound on the next one.",
          "- Month-end values miss falls inside a month.",
          "- Taxes, trading costs and fund fees are not charged.",
          "- Expected growth rests on historical means, which the allocation study found do not carry over. It is shown, with its interval, and decides nothing.",
          "- The stacked funds hold US or global stocks, not the study's world-stock composite, and none of them is in the study's returns."]
    Q.REPORT.write_text("\n".join(L) + "\n")
    return str(Q.REPORT)


# ------------------------------------------------------------------ page
EXTRA_CSS = """
/* layout: the questions, then the answer they pick (headline, what to hold, its place on the path), then how it was decided */
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
.answer .verdict { font-family: var(--serif); font-size: 1.75rem; line-height: 1.15; text-wrap: balance; }
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
  const times = (k) => String(Math.round(k * 100) / 100) + ' times capital';
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
  // the route is asked only where the account has more than one
  const route = () => { const rs = D.meta.accounts[picked('qn-account')].routes; return rs.length > 1 && rs.includes(picked('qn-route')) ? picked('qn-route') : rs[0]; };
  const answer = () => D.answers.find((r) => r.account === picked('qn-account') && r.route === route() && r.goal === picked('qn-goal'));
  const path = (r) => D.curve.filter((p) => p.route === r.route);
  const same = (p, r) => p.i === r.i;
  const above = (rt) => (rt === 'margin' ? 'borrowed on margin' : 'carried inside stacked funds');
  const mixName = (p) => `${pct(p.stocks, 0, false)} stocks, ${pct(p.sleeve, 0, false)} diversifiers` +
    (p.cash > 0.0005 ? `, ${pct(p.cash, 0, false)} T-bills` : p.borrowed > 0.0005 ? `, ${pct(p.borrowed, 0, false)} ${above(p.route)}` : '');
  const mixTip = (p) => `Forecast volatility ${pct(p.vol, 1, false)}<br>Worst fall in the replay ${pct(p.worst_fall_all, 0)}<br>` +
    `Expected growth ${pct(p.growth)} (${pct(p.growth_lo)} to ${pct(p.growth_hi)})`;

  // ---- what to hold: one bar, shares of capital
  function bar(r) {
    const host = $('qn-bar');
    if (r.cash >= 1) { host.innerHTML = ''; $('qn-barleg').innerHTML = '<span><span class="sw" style="background:var(--base)"></span>T-bills 100%</span>'; return; }
    const parts = [...A.map((a) => [D.meta.names[a], r['w_' + a], `var(${COL[a]})`]), ['T-bills', r.cash, 'var(--base)']].filter((p) => p[1] > 0.0005);
    const W = 760, H = 62, total = parts.reduce((s, p) => s + p[1], 0), x = (v) => v / Math.max(total, 1) * W;
    const s = svg(host, W, H, 'Shares of capital in each holding');
    let at = 0;
    parts.forEach(([name, w, fill]) => {
      const b = el('rect', { x: x(at) + 1, y: 20, width: Math.max(x(w) - 2, 1), height: 26, rx: 4, fill });
      bind(b, `<b>${name}</b>${pct(w, 1, false)} of capital`);
      s.appendChild(b);
      if (x(w) > 44) s.appendChild(el('text', { x: x(at) + x(w) / 2, y: 60, 'text-anchor': 'middle', class: 'lab' }, pct(w, 0, false)));
      at += w;
    });
    if (total > 1.0005) {
      s.appendChild(el('line', { x1: x(1), x2: x(1), y1: 12, y2: 50, stroke: 'var(--fg)', 'stroke-width': 1.5 }));
      s.appendChild(el('text', { x: x(1) - 6, y: 12, 'text-anchor': 'end' }, 'capital, 100%'));
      s.appendChild(el('text', { x: x(1) + 6, y: 12 }, `${above(r.route)}, ${pct(r.borrowed, 0, false)}`));
    }
    $('qn-barleg').innerHTML = parts.map(([name, w, fill]) => `<span><span class="sw" style="background:${fill}"></span>${name} ${pct(w, 1, false)}</span>`).join('');
  }

  // ---- where it sits: forecast risk across, expected growth (and how little is known of it) up
  function chart(r) {
    const pts = path(r), scaled = pts.filter((p) => p.segment === 'scaled'), tilted = pts.filter((p) => p.segment === 'tilted');
    const zero = { vol: 0, growth: 0, growth_lo: 0, growth_hi: 0 }, cash = r.cash >= 1;
    const W = 760, H = 440, m = { l: 46, r: 24, t: 20, b: 40 };
    const xmax = Math.ceil(Math.max(...pts.map((p) => p.vol), ...D.inputs.map((p) => p.vol)) * 20) / 20;
    const ymax = Math.ceil(Math.max(...pts.map((p) => p.growth_hi)) * 50) / 50;
    const ymin = Math.floor(Math.min(0, ...pts.map((p) => p.growth_lo)) * 50) / 50;
    const x = (v) => m.l + v / xmax * (W - m.l - m.r), y = (v) => H - m.b - (v - ymin) / (ymax - ymin) * (H - m.t - m.b);
    const s = svg($('qn-chart'), W, H, 'Forecast volatility against expected growth along the path: the mix scaled, then stocks in place of diversifiers, with this answer marked');
    for (let t = ymin; t <= ymax + 1e-9; t += 0.02) {
      s.appendChild(el('line', { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: Math.abs(t) < 1e-9 ? 'zero' : 'grid' }));
      s.appendChild(el('text', { x: m.l - 6, y: y(t) + 4, 'text-anchor': 'end' }, pct(t, 0, false)));
    }
    for (let t = 0; t <= xmax + 1e-9; t += 0.05) s.appendChild(el('text', { x: x(t), y: H - m.b + 16, 'text-anchor': 'middle' }, pct(t, 0, false)));
    s.appendChild(el('text', { x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle' }, 'Forecast volatility, next 12 months'));
    s.appendChild(el('text', { x: m.l, y: 11 }, 'Expected growth over T-bills, a year'));
    const xy = (p, k) => `${x(p.vol).toFixed(1)},${y(p[k]).toFixed(1)}`, run = [zero, ...pts];
    s.appendChild(el('path', { d: 'M' + run.map((p) => xy(p, 'growth_hi')).join('L') + 'L' + run.slice().reverse().map((p) => xy(p, 'growth_lo')).join('L') + 'Z',
      fill: 'var(--fg)', opacity: 0.07 }));
    const line = (ps, attrs) => s.appendChild(el('path', { d: 'M' + ps.map((p) => xy(p, 'growth')).join('L'), fill: 'none', stroke: 'var(--fg)', 'stroke-width': 2, 'stroke-linecap': 'round', ...attrs }));
    line([zero, ...scaled], {});
    if (tilted.length) line([scaled[scaled.length - 1], ...tilted], { 'stroke-dasharray': '1 6' });
    const side = { world_stocks: [9, 4, 'start'], ust_int: [-9, 4, 'end'], gold: [-9, 4, 'end'], mf: [9, 4, 'start'] };
    D.inputs.forEach((p) => {
      const c = el('circle', { cx: x(p.vol), cy: y(p.mean_12m), r: 5, fill: `var(${COL[p.asset]})`, stroke: 'var(--bg)', 'stroke-width': 2 });
      bind(c, `<b>${p.name}</b>Forecast volatility ${pct(p.vol, 1, false)}<br>Expected growth ${pct(p.mean_12m)} (${pct(p.ci90_lo)} to ${pct(p.ci90_hi)}), history from ${String(p.first_month).slice(0, 4)}`);
      const [dx, dy, anchor] = side[p.asset];
      s.appendChild(el('text', { x: x(p.vol) + dx, y: y(p.mean_12m) + dy, 'text-anchor': anchor }, D.short[p.asset]));
      s.appendChild(c);
    });
    const mark = (p, text) => {
      s.appendChild(el('circle', { cx: x(p.vol), cy: y(p.growth), r: 2.5, fill: 'var(--fg)' }));
      s.appendChild(el('text', { x: x(p.vol) + 7, y: y(p.growth) + 15 }, text));
    };
    const one = scaled.find((p) => Math.abs(p.exposure - 1) < 1e-6), end = scaled[scaled.length - 1];
    if (one) mark(one, 'the mix, 100% of capital');
    if (end.exposure > 1.3) mark(end, 'the mix at ' + times(end.exposure));
    tilted.filter((p) => Math.abs(p.stocks - 0.5) < 1e-6 || p === tilted[tilted.length - 1]).forEach((p) => mark(p, pct(p.stocks, 0, false) + ' stocks'));
    pts.forEach((p) => {
      const c = el('circle', { cx: x(p.vol), cy: y(p.growth), r: 7, fill: 'transparent' });
      bind(c, `<b>${mixName(p)}</b>${mixTip(p)}`);
      s.appendChild(c);
    });
    D.answers.filter((o) => o.account === r.account && o.route === r.route && o.goal !== r.goal).forEach((o) => {
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
    const rows = D.funds.filter((f) => f.account === r.account && f.route === r.route && f.goal === r.goal);
    $('qn-funds').innerHTML = '<thead><tr><th class="l">Fund</th><th class="l">Name</th><th>Share of capital</th><th>Fee a year</th><th class="l">Note</th></tr></thead><tbody>' +
      rows.map((f) => `<tr><td class="l">${f.symbol || ''}</td><td class="l">${f.name}</td><td>${pct(f.weight, 1, false)}</td><td>${f.fee == null ? '' : pct(f.fee, 2, false)}</td><td class="l wrap">${f.note || ''}</td></tr>`).join('') + '</tbody>';
    let cost = '';
    if (r.cash < 1) {
      cost = `Recorded fund fees come to ${pct(r.fund_fees, 2, false)} of capital a year` +
        (r.loan_cost > 0 ? `, and the loan costs ${pct(r.loan_cost, 2, false)} more than T-bills` : '') + '.';
      if (r.financing === 'stacked') cost += (Math.abs(r.stocks_held - r.stocks) > 0.0005 ? ` These funds leave ${pct(r.stocks_held, 0, false)} of capital in stocks.` : '') +
        ' The stacked funds hold US or global stocks, not the study\'s composite.';
    }
    $('qn-cost').textContent = cost;
    $('qn-notes').innerHTML = D.meta.notes[r.account].map((n) => `<li>${n}</li>`).join('');
  }

  function table(r) {
    const round = (v) => Math.abs(v * 10 - Math.round(v * 10)) < 1e-6, pts = path(r);
    const last = pts.filter((p) => p.segment === 'scaled').length - 1;
    const rows = pts.filter((p) => (p.segment === 'scaled' ? round(p.exposure) || p.i === last : round(p.stocks)) || same(p, r));
    $('qn-curve').innerHTML = `<thead><tr><th class="l"></th><th>Stocks</th><th>Diversifiers</th><th>T-bills</th><th>${r.route === 'margin' ? 'Borrowed' : 'Inside funds'}</th><th>Forecast volatility</th>` +
      '<th>Worst fall in the replay</th><th>Growth in the replay</th><th>Expected growth (90% interval)</th></tr></thead><tbody>' +
      rows.map((p) => `<tr${r.cash < 1 && same(p, r) ? ' class="pick"' : ''}><td class="l">${p.segment === 'scaled' ? 'the mix at ' + pct(p.exposure, 0, false) : 'more stocks'}</td>` +
        `<td>${pct(p.stocks, 0, false)}</td><td>${pct(p.sleeve, 0, false)}</td><td>${p.cash > 0.0005 ? pct(p.cash, 0, false) : ''}</td><td>${p.borrowed > 0.0005 ? pct(p.borrowed, 0, false) : ''}</td>` +
        `<td>${pct(p.vol, 1, false)}</td><td>${pct(p.worst_fall_all, 1)}</td><td>${pct(p.growth_all)}</td><td>${pct(p.growth)} (${pct(p.growth_lo)} to ${pct(p.growth_hi)})</td></tr>`).join('') + '</tbody>';
  }

  function draw() {
    const acc = D.meta.accounts[picked('qn-account')];
    $('qn-route-set').hidden = acc.routes.length < 2;
    const r = answer(), cap = D.meta.caps[r.route];
    $('qn-headline').textContent = r.headline;
    $('qn-reason').textContent = r.reason;
    $('qn-context').textContent = [acc.name, acc.routes.length > 1 ? D.meta.routes[r.route].name : null, D.meta.goals[r.goal].name].filter(Boolean).join(' · ');
    $('qn-how').textContent = (r.route === 'margin' ? 'on a margin loan' : 'inside stacked funds') + `, as far as ${times(cap.exposure)}`;
    bar(r); figures(r); chart(r); funds(r); table(r);
    try { localStorage.setItem('qn', JSON.stringify([r.account, r.goal, picked('qn-route')])); } catch (e) { /* storage is a convenience */ }
  }
  try {
    const [a, g, rt] = JSON.parse(localStorage.getItem('qn') || '[]');
    for (const [n, v] of [['qn-account', a], ['qn-goal', g], ['qn-route', rt]]) { const i = document.querySelector(`input[name="${n}"][value="${v}"]`); if (i) i.checked = true; }
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
        f'<span class="t">{v["name"]}</span><span class="d">{detail(k, v)}</span></label>' for k, v in items.items())


def fragment(d: dict) -> str:
    meta, s, f = d["meta"], d["meta"]["settings"], facts(d)
    data = json.dumps(payload(d), separators=(",", ":"), allow_nan=False, default=lambda o: None).replace("</", "<\\/")
    budget = lambda k, g: (g["detail"] + (" No room for a fall." if g["loss_limit"] is None                 # noqa: E731
                                          else f" Can sit through a fall of {g['loss_limit']:.0%}."))
    how = lambda k, r: r["detail"] + (f" T-bills + {100 * s['spreads']['margin']:.2f}%." if k == "margin" else "")   # noqa: E731
    routes = {k: meta["routes"][k] for k in meta["accounts"]["taxable"]["routes"]}
    return f"""<title>Allocation Questionnaire</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>{TOKENS}{CSS}{EXTRA_CSS}</style>
<main>
<header style="display:flex;flex-direction:column;gap:12px">
<h1>One mix, sized to the fall you can sit through</h1>
<p>Stocks, intermediate Treasuries, gold and managed futures, each weighted by its risk: {_mix(meta)}. The goal's
drawdown budget decides how much of that mix to hold, with T-bills beside it or above 100% of capital. Expected returns
decide nothing. Forecast origin {meta['origin']}.</p>
<p class="label">{meta['label']}.</p>
</header>

<section aria-label="Questions">
<fieldset><legend>What is the money for?</legend><div class="opts">{_options("qn-goal", meta["goals"], "growth", budget)}</div></fieldset>
<fieldset><legend>How is the account taxed?</legend><div class="opts">{_options("qn-account", meta["accounts"], "taxable", lambda k, v: v["detail"])}</div></fieldset>
<fieldset id="qn-route-set"><legend>How should anything above 100% of capital be held?</legend><div class="opts">{_options("qn-route", routes, next(iter(routes)), how)}</div></fieldset>
</section>

<section class="answer" aria-live="polite">
<div><div class="q" id="qn-context"></div><div class="verdict" id="qn-headline"></div></div>
<p id="qn-reason"></p>
<div id="qn-bar"></div>
<div class="legend" id="qn-barleg"></div>
<dl class="figs" id="qn-figs"></dl>
</section>

<section><h2>The path, and where yours sits</h2>
<p>Left to right is forecast risk, which the package can measure. The solid line is the mix held at more and more of the
capital: with T-bills beside it at first, then above 100% <span id="qn-how"></span>. Where the account can carry no more,
the dotted line takes more risk the only way left, with stocks in place of diversifiers. The goal's drawdown budget picks
the point.</p>
<p>The shaded band is the 90% interval of each point's expected growth. It is wide, which is why no point is chosen on it.</p>
<div class="legend">
<span><span class="sw" style="background:var(--fg);border-radius:50%"></span>your mix</span>
<span><span class="ring"></span>the other goals</span>
<span><span class="ln" style="background:var(--fg)"></span>the mix, scaled</span>
<span><span class="ln" style="background:repeating-linear-gradient(90deg,var(--fg) 0 2px,transparent 2px 6px)"></span>stocks in place of diversifiers</span>
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

<section><h2>How the point is picked</h2>
<ol>
<li>One mix: the four holdings weighted by 1/volatility on the whole past.</li>
<li>Scale it to the goal's drawdown budget: the largest multiple of capital whose worst fall in the replay stayed within
the budget. Below 100% the rest is T-bills. Above 100% it is carried inside stacked funds, or on a margin loan in a taxable
account that chooses one.</li>
<li>If the account can carry no more before the budget is used, stocks take the place of diversifiers at that limit, up to
100% stocks. Stacked funds reach {f['stacked_cap']:.2f} times capital with this mix and a margin loan {f['margin_cap']:.0f} times.</li>
<li>Money needed within 3 years has no budget and stays in T-bills.</li>
</ol>
</section>

<section><h2>What this does not show</h2>
<ul>
<li>The drawdown budgets are judgements, and the answers follow from them. The table above shows what any other budget would give.</li>
<li>Every goal beyond 3 years ends where the account can carry no more: the mix scaled that far fell only
{_pct(f['stacked_cap_fall'], 0)} with stacked funds and {_pct(f['margin_cap_fall'], 0)} on margin. So in practice the budget
sets the stock share and the diversifiers fill whatever the account can hold.</li>
<li>On margin that limit is Regulation T's for opening a position, {s['max_leverage']:.0f} times capital. The broker's
maintenance requirement is not modelled, and it decides how far the position can fall before it is called.</li>
<li>On margin the lower end of expected growth falls as the mix is levered, from {_pct(f['margin_lo_at_1'])} at 100% of
capital to {_pct(f['margin_lo_at_cap'])} at {f['margin_cap']:.0f} times. Whether the diversifiers earn more than the loan
costs is not established.</li>
<li>Every year in the replay had been seen before the rule was written, the diversifiers were chosen on the same years,
and managed futures before {meta['live_from'][:7]} are a paper factor. A worst fall in the past does not bound the next one.</li>
<li>Month-end values miss falls inside a month. Taxes, trading costs and fund fees are not charged, so tax-deferred and
Roth accounts get the same numbers.</li>
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
