"""Write ``dashboard/horizons.html``: the horizon study as one self-contained page.

    python -m assetforecast.horizons.page [--fragment PATH]

The page inlines the score tables of ``horizons.TABLES`` as JSON and draws
them in the browser; no number is computed here beyond the few counts in the
summary sentences, which come from ``report``'s helpers on the same tables.
``--fragment`` also writes the page without its document wrapper (for hosts
that supply their own).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ..pagekit import document
from . import HORIZONS, LABEL, MIN_INDEP, PAGE
from . import report as R

TOKENS = """
:root {
  /* layout: one reading column; controls above a pair of asset-by-horizon grids, one per period */
  --bg: #f9f9f7; --panel: #fcfcfb; --fg: #0b0b0b; --muted: #52514e; --line: #e1e0d9; --border: #c3c2b7;
  --pos: #2a78d6; --neg: #d03b3b; --mid: #f0efec; --s1: #2a78d6; --s2: #eb6834;
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --serif: ui-serif, Georgia, Cambria, "Times New Roman", serif;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  color-scheme: dark; --bg: #0d0d0d; --panel: #1a1a19; --fg: #f0efec; --muted: #c3c2b7; --line: #2c2c2a;
  --border: #383835; --pos: #3987e5; --neg: #e66767; --mid: #383835; --s1: #3987e5; --s2: #d95926; } }
:root[data-theme="dark"] {
  color-scheme: dark; --bg: #0d0d0d; --panel: #1a1a19; --fg: #f0efec; --muted: #c3c2b7; --line: #2c2c2a;
  --border: #383835; --pos: #3987e5; --neg: #e66767; --mid: #383835; --s1: #3987e5; --s2: #d95926; }
"""

CSS = """
*, *::before, *::after { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font-family: var(--sans); font-size: 15px;
  line-height: 1.5; padding-block: 32px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main { max-width: 1080px; margin: 0 auto; display: flex; flex-direction: column; gap: 28px; }
h1 { font-family: var(--serif); font-size: 2rem; line-height: 1.15; margin: 0; font-weight: 600; text-wrap: balance; }
h2 { font-size: 1.25rem; margin: 0 0 8px; font-weight: 600; text-wrap: balance; }
h3 { font-size: .9375rem; margin: 0 0 6px; font-weight: 600; }
p { margin: 0; max-width: 68ch; }
.muted, .note { color: var(--muted); font-size: .8125rem; }
.label { border-left: 3px solid var(--border); padding: 2px 0 2px 12px; color: var(--muted); font-size: .8125rem; max-width: 80ch; }
.answers { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 20px 28px; }
.answers > div { min-width: 0; border-top: 2px solid var(--fg); padding-top: 10px; }
.answers .verdict { font-family: var(--serif); font-size: 1.25rem; line-height: 1.2; margin-bottom: 6px; }
.answers p { font-size: .875rem; color: var(--muted); }
.controls { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: end; }
.controls label { display: flex; flex-direction: column; gap: 3px; font-size: .75rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: .04em; }
select { font: inherit; font-size: .875rem; color: var(--fg); background: var(--panel); border: 1px solid var(--border);
  border-radius: 6px; padding: 5px 8px; max-width: 100%; }
select:focus-visible, td[tabindex]:focus-visible { outline: 2px solid var(--pos); outline-offset: 1px; }
.grids { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; }
.grids > div { min-width: 0; }
.scroll { overflow-x: auto; }
table { border-collapse: separate; border-spacing: 2px; font-variant-numeric: tabular-nums; font-size: .8125rem; width: 100%; }
th { font-weight: 500; color: var(--muted); text-align: right; padding: 3px 6px; white-space: nowrap; }
th.row { text-align: left; }
td { text-align: right; padding: 5px 6px; border-radius: 3px; white-space: nowrap; min-width: 44px; }
td.thin { color: var(--muted); background: transparent; text-align: center; }
tr.pooled td, tr.pooled th { font-weight: 600; }
tr.pooled + tr td, tr.pooled + tr th { padding-top: 9px; }
table.plain { border-spacing: 0; }
table.plain td, table.plain th { border-bottom: 1px solid var(--line); border-radius: 0; padding: 5px 8px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: center; font-size: .8125rem; color: var(--muted); }
.ramp { display: inline-block; width: 140px; height: 10px; border-radius: 2px;
  background: linear-gradient(90deg, var(--neg), var(--mid), var(--pos)); vertical-align: middle; margin: 0 6px; }
.sw { display: inline-block; width: 14px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
#tip { position: fixed; pointer-events: none; background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  padding: 7px 10px; font-size: .8125rem; box-shadow: 0 4px 16px rgba(0,0,0,.18); max-width: 260px; z-index: 5; }
#tip b { display: block; }
svg text { fill: var(--muted); font-size: 11px; font-family: var(--sans); }
svg .lab { fill: var(--fg); font-size: 12px; }
ul { margin: 0; padding-left: 20px; max-width: 72ch; }
li { margin-bottom: 4px; }
code { font-size: .875em; }
.bar { display: flex; justify-content: flex-end; }
.bar button { font: inherit; font-size: .8125rem; color: var(--muted); background: none; border: 1px solid var(--border);
  border-radius: 6px; padding: 3px 10px; cursor: pointer; }
"""

JS = r"""
(() => {
  const D = JSON.parse(document.getElementById('hz-data').textContent);
  const H = D.horizons, HL = D.hlabels, PER = D.periods, ASSETS = D.assets;
  const $ = (id) => document.getElementById(id);
  const Q = {
    returns: { rows: D.returns, val: 'r2', p: 'cw_p', vmax: 0.3, models: D.return_models, def: 'sig_combo',
      what: 'Out-of-sample R squared against the historical mean', bar: 'the historical mean' },
    vol: { rows: D.vol, val: 'skill', p: 'dm_p', vmax: 0.6, models: D.vol_models, def: 'half',
      what: 'Share of the naive forecast’s QLIKE loss removed', bar: 'the long-run volatility' },
    corr: { rows: D.corr, val: 'skill', p: 'dm_p', vmax: 0.5, models: D.corr_models, def: 'half',
      what: 'Share of the naive forecast’s squared error removed', bar: 'the long-run correlation' },
  };
  const pct = (v, d = 1) => (v == null ? '' : (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(100 * v).toFixed(d) + '%');
  const fill = (v, vmax) => {
    const s = Math.min(Math.abs(v) / vmax, 1) * 62;
    return `color-mix(in oklab, var(${v >= 0 ? '--pos' : '--neg'}) ${s.toFixed(0)}%, var(--mid))`;
  };
  const tip = $('tip');
  const showTip = (e, html) => {
    tip.innerHTML = html; tip.hidden = false;
    const r = e.currentTarget.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    let x = r.left + r.width / 2 - w / 2, y = r.top - h - 8;
    x = Math.max(8, Math.min(x, innerWidth - w - 8)); if (y < 8) y = r.bottom + 8;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  };
  const hideTip = () => { tip.hidden = true; };
  const bind = (el, html) => {
    el.tabIndex = 0;
    el.addEventListener('mouseenter', (e) => showTip(e, html)); el.addEventListener('focus', (e) => showTip(e, html));
    el.addEventListener('mouseleave', hideTip); el.addEventListener('blur', hideTip);
  };

  function pick() {
    const q = Q[$('hz-q').value], model = $('hz-model').value, k = +$('hz-k').value;
    const rows = q.rows.filter((r) => r.model === model && (r.k == null || r.k === k));
    const at = {};
    for (const r of rows) at[r.period + '|' + r.asset + '|' + r.horizon] = r;
    return { q, model, at };
  }

  function grid(el, period, { q, at }) {
    const cols = H.filter((h) => ['pooled', ...ASSETS].some((a) => at[period + '|' + a + '|' + h]));
    const t = document.createElement('table');
    t.innerHTML = '<thead><tr><th class="row">asset</th>' + cols.map((h) => `<th>${HL[h]}</th>`).join('') + '</tr></thead>';
    const tb = document.createElement('tbody');
    for (const a of ['pooled', ...ASSETS]) {
      const tr = document.createElement('tr'); if (a === 'pooled') tr.className = 'pooled';
      tr.innerHTML = `<th class="row">${a === 'pooled' ? 'all assets' : a}</th>`;
      for (const h of cols) {
        const r = at[period + '|' + a + '|' + h], td = document.createElement('td');
        if (!r) { td.className = 'thin'; td.textContent = ''; }
        else if (r.thin) {
          td.className = 'thin'; td.textContent = '·';
          bind(td, `<b>${a}, ${HL[h]}</b>Only ${r.n_indep.toFixed(1)} independent windows: too few to show.`);
        } else {
          const v = r[q.val]; td.textContent = pct(v, Math.abs(v) >= 0.995 ? 0 : 1); td.style.background = fill(v, q.vmax);
          const p = r[q.p];
          bind(td, `<b>${a === 'pooled' ? 'all assets' : a}, ${HL[h]}</b>${q.what}: ${pct(v)}` +
            (r.win != null ? `<br>Lower loss than naive on ${(100 * r.win).toFixed(0)}% of months` : '') +
            (p != null ? `<br>p-value (better than ${q.bar}): ${p.toFixed(2)}` : '') +
            `<br>${r.n_indep.toFixed(1)} independent windows`);
        }
        tr.append(td);
      }
      tb.append(tr);
    }
    t.append(tb); el.replaceChildren(t);
  }

  function line(el, { q, at }) {
    const W = Math.max(320, Math.min(el.clientWidth || 720, 760)), Ht = 240, m = { l: 46, r: 96, t: 14, b: 30 };
    const series = Object.keys(PER).map((p, i) => ({ p, color: `var(--s${i + 1})`,
      pts: H.map((h, j) => ({ j, h, r: at[p + '|pooled|' + h] })).filter((d) => d.r && !d.r.thin) }));
    const vals = series.flatMap((s) => s.pts.map((d) => d.r[q.val]));
    if (!vals.length) { el.replaceChildren(); return; }
    let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals); const pad = (hi - lo || 0.01) * 0.12; lo -= pad; hi += pad;
    const x = (j) => m.l + j * (W - m.l - m.r) / (H.length - 1), y = (v) => m.t + (hi - v) * (Ht - m.t - m.b) / (hi - lo);
    const step = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10].find((s) => (hi - lo) / s <= 6) || 20;
    let s = `<svg viewBox="0 0 ${W} ${Ht}" width="${W}" height="${Ht}" role="img" aria-label="Pooled skill by horizon, two periods">`;
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step)
      s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${pct(v, 0)}</text>`;
    s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(0)}" y2="${y(0)}" stroke="var(--muted)"/>`;
    H.forEach((h, j) => { s += `<text x="${x(j)}" y="${Ht - 8}" text-anchor="middle">${HL[h]}</text>`; });
    const ends = [];
    for (const sr of series) {
      if (!sr.pts.length) continue;
      s += `<path d="${sr.pts.map((d, i) => (i ? 'L' : 'M') + x(d.j) + ' ' + y(d.r[q.val])).join(' ')}" fill="none" stroke="${sr.color}" stroke-width="2" stroke-linejoin="round"/>`;
      for (const d of sr.pts) s += `<circle cx="${x(d.j)}" cy="${y(d.r[q.val])}" r="4" fill="${sr.color}" stroke="var(--bg)" stroke-width="2" data-p="${sr.p}" data-h="${d.h}"/>`;
      const e = sr.pts[sr.pts.length - 1]; ends.push({ y: y(e.r[q.val]), x: x(e.j), name: PER[sr.p] });
    }
    ends.sort((a, b) => a.y - b.y); if (ends.length === 2 && ends[1].y - ends[0].y < 14) { ends[0].y -= 7; ends[1].y += 7; }
    for (const e of ends) s += `<text class="lab" x="${e.x + 9}" y="${e.y + 4}">${e.name}</text>`;
    el.innerHTML = s + '</svg>';
    el.querySelectorAll('circle').forEach((c) => {
      const r = at[c.dataset.p + '|pooled|' + c.dataset.h];
      bind(c, `<b>${PER[c.dataset.p]}, ${HL[c.dataset.h]}</b>${q.what}: ${pct(r[q.val])}` + (r[q.p] != null ? `<br>p-value: ${r[q.p].toFixed(2)}` : ''));
    });
  }

  function plain(el, rows, key, val, fmt) {
    const names = [...new Set(rows.map(key))], per = Object.keys(PER);
    let s = '';
    for (const p of per) {
      const cols = H.filter((h) => rows.some((r) => r.period === p && r.horizon === h));
      s += `<h3>${PER[p]}</h3><div class="scroll"><table class="plain"><thead><tr><th class="row"></th>` + cols.map((h) => `<th>${HL[h]}</th>`).join('') + '</tr></thead><tbody>';
      for (const n of names) {
        s += `<tr><th class="row">${n}</th>` + cols.map((h) => {
          const r = rows.find((d) => d.period === p && d.horizon === h && key(d) === n);
          return `<td>${r && !r.thin ? fmt(r[val]) : '·'}</td>`;
        }).join('') + '</tr>';
      }
      s += '</tbody></table></div>';
    }
    el.innerHTML = s;
  }

  function render() {
    const sel = pick();
    $('hz-what').textContent = sel.q.what + '. Positive (blue): better than ' + sel.q.bar + '.';
    Object.keys(PER).forEach((p) => grid($('hz-grid-' + p), p, sel));
    line($('hz-line'), sel);
  }
  function models() {
    const q = Q[$('hz-q').value];
    $('hz-model').innerHTML = q.models.map((m) => `<option value="${m}"${m === q.def ? ' selected' : ''}>${m}</option>`).join('');
    $('hz-k-wrap').hidden = $('hz-q').value === 'returns';
  }
  $('hz-q').addEventListener('change', () => { models(); render(); });
  $('hz-model').addEventListener('change', render); $('hz-k').addEventListener('change', render);
  models(); render();
  const lab = { 'hist_mean/naive': 'baseline' };
  plain($('hz-sleeves'), D.sleeves, (r) => r.mu + ' + ' + r.var, 'd_cer', (v) => pct(v, 2));
  plain($('hz-minvar'), D.minvar, (r) => r.vol + ' vol, ' + r.corr + ' corr', 'ratio', (v) => v.toFixed(3));
  let w = innerWidth, t = null;
  addEventListener('resize', () => { if (innerWidth === w) return; w = innerWidth; clearTimeout(t); t = setTimeout(render, 150); });
  const btn = $('hz-theme'), root = document.documentElement;
  const dark = () => (root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches);
  const setLabel = () => { btn.textContent = dark() ? 'Light mode' : 'Dark mode'; }; setLabel();
  btn.addEventListener('click', () => { root.dataset.theme = dark() ? 'light' : 'dark'; setLabel(); });
})();
"""


def _records(df: pd.DataFrame, cols: list) -> list:
    d = df[df["period"].isin(list(R.PER))][cols].copy()
    d["thin"] = d["thin"].astype(bool)
    d = d.astype(object).where(d.notna(), None)
    return [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in d.to_dict("records")]


def answers(t: dict) -> list[tuple[str, str, str]]:
    """(quantity, verdict, one sentence with numbers from the tables)."""
    rs, vs, cs, sl = t["return_scores"], t["vol_scores"], t["corr_scores"], t["sleeves"]
    f, span = R._fmt, R.span
    rm =[m for m in ("sig_combo", "hz_ridge", "hz_gbm", "hz_ens") if m in set(rs["model"])]
    pr = {p: R.by_h(rs, "r2", "model", asset="pooled", period=p, model=rm) for p in R.PER}
    pv = {p: R.by_h(vs, "skill", "model", k=1, asset="pooled", period=p).loc["half"] for p in R.PER}
    pw = R.by_h(vs, "win", "model", k=1, asset="pooled", period="after").loc["half"]
    pc = {p: R.by_h(cs, "skill", "model", k=1, asset="pooled", period=p).loc["half"] for p in R.PER}
    sp = sl[(sl["asset"] == "pooled") & (sl["mu"] == "sig_combo") & (sl["var"] == "half")]
    sd = {p: R.by_h(sp, "d_cer", "period", period=p).iloc[0] for p in R.PER}
    sh = [1, 3, 6, 12]
    return [
        ("Returns", "Nothing reliable at any horizon",
         f"Up to 12 months the pooled R squared against the historical mean is {span(pr['dev'][sh], 'pct')} in 1990 to "
         f"2014 and {span(pr['after'][sh], 'pct')} after 2015. It reaches {f(pr['dev'][[36, 60]].max().max(), 'pct')} at 3 to 5 "
         "years before 2015, on about five independent windows that the later years cannot confirm."),
        ("Volatility", "Predictable out to about a year",
         f"The blended forecast removed {span(pv['dev'][sh], 'pct0')} of the naive loss at 1 to 12 months in 1990 to "
         f"2014. After 2015 the average is {span(pv['after'][sh], 'pct0')} because of the March 2020 crash, though it "
         f"was closer in {span(pw[sh], 'pct0')} of months."),
        ("Correlations", "Modestly predictable, and it held up",
         f"The blended forecast removed {span(pc['dev'], 'pct0')} of the naive squared error in 1990 to 2014 and "
         f"{span(pc['after'], 'pct0')} after 2015."),
        ("Combined", "No reliable gain for the whole model",
         f"Sizing positions with both forecasts changed the certainty-equivalent return by "
         f"{span(sd['dev'], 'pct')} a year in 1990 to 2014 and {span(sd['after'], 'pct')} after 2015, depending on "
         "the horizon."),
    ]


def body(t: dict) -> str:
    rs, vs, cs, sl, mv = (t[k] for k in ("return_scores", "vol_scores", "corr_scores", "sleeves", "minvar"))
    slp = sl[sl["asset"] == "pooled"]
    data = {
        "horizons": list(HORIZONS), "hlabels": R.HL, "periods": R.PER,
        "assets": [a for a in R.ASSET_ORDER if a in set(rs["asset"])],
        "return_models": [m for m in ("sig_combo", "hz_ens", "hz_ridge", "hz_gbm") if m in set(rs["model"])]
                         + sorted(m for m in set(rs["model"]) if m.startswith("sig_") and m != "sig_combo"),
        "vol_models": [m for m in ("half", "har", "trail_3", "trail_12", "trail_60", "published") if m in set(vs["model"])],
        "corr_models": [m for m in ("half", "trail_12", "trail_36", "trail_60", "published") if m in set(cs["model"])],
        "returns": _records(rs, ["model", "horizon", "asset", "period", "r2", "cw_p", "n_indep", "thin"]),
        "vol": _records(vs, ["k", "horizon", "model", "asset", "period", "skill", "win", "dm_p", "n_indep", "thin"]),
        "corr": _records(cs, ["k", "horizon", "model", "asset", "period", "skill", "win", "dm_p", "n_indep", "thin"]),
        "sleeves": _records(slp, ["horizon", "period", "mu", "var", "d_cer", "t", "thin"]),
        "minvar": _records(mv, ["horizon", "period", "vol", "corr", "ratio", "dm_t", "thin"]),
    }
    ans = "".join(f'<div><h3 class="muted">{q}</h3><div class="verdict">{v}</div><p>{s}</p></div>'
                  for q, v, s in answers(t))
    grids = "".join(f'<div><h3>{name}</h3><div class="scroll" id="hz-grid-{p}"></div></div>'
                    for p, name in R.PER.items())
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return f"""<main>
<div class="bar"><button id="hz-theme" type="button">Dark mode</button></div>
<header><h1>What is predictable, from 1 month to 10 years</h1></header>
<p class="label">{LABEL}. The published 12-month forecast and the frozen models are unchanged.</p>
<section class="answers">{ans}</section>
<section>
<h2>The map: skill by asset and horizon</h2>
<div class="controls">
<label for="hz-q">Quantity<select id="hz-q"><option value="returns">Returns</option><option value="vol">Volatility</option><option value="corr">Correlations</option></select></label>
<label for="hz-model">Forecaster<select id="hz-model"></select></label>
<label for="hz-k" id="hz-k-wrap">Risk measured from<select id="hz-k"><option value="1">daily returns</option><option value="21">21-day returns (package target)</option></select></label>
</div>
<p class="note" id="hz-what" style="margin:10px 0"></p>
<div class="legend"><span>worse<span class="ramp"></span>better</span><span>· fewer than {MIN_INDEP:g} independent windows</span><span>Hover or focus a cell for the test and sample size.</span></div>
<div class="grids" style="margin-top:14px">{grids}</div>
</section>
<section>
<h2>All assets together, by horizon</h2>
<p class="note">The same forecaster, pooled over assets. One line per period.</p>
<div id="hz-line" style="margin-top:8px"></div>
</section>
<section>
<h2>Is it worth money?</h2>
<p>Single-asset sleeves against cash, each forecast held for its own horizon, after trading costs. Gain in annualized certainty-equivalent return over the baseline (historical mean with long-run volatility), averaged over the 12 sleeves. Rows are return forecast + volatility forecast.</p>
<div id="hz-sleeves" style="margin-top:10px"></div>
<p style="margin-top:18px">Long-only minimum-variance portfolio built from the forecasts: realized volatility relative to the same portfolio built from naive forecasts. Below 1 means the forecasts lowered risk.</p>
<div id="hz-minvar" style="margin-top:10px"></div>
</section>
<section>
<h2>How to read this</h2>
<ul>
<li>Forecasts are made at every month end and scored once their window has closed. <b>1990 to 2014</b> are the years the package's models were developed on; <b>2015 on</b> are years already used once as a holdout. Neither column is a test.</li>
<li>Return forecasters: <code>sig_combo</code> averages nine single-signal regressions; <code>hz_ridge</code>, <code>hz_gbm</code> and <code>hz_ens</code> are the frozen learners re-aimed at each horizon under new names. The bar is the expanding historical mean.</li>
<li>Risk forecasters: <code>trail_N</code> is the last N months; <code>half</code> averages a trailing window with the long-run level; <code>har</code> is a log regression; <code>published</code> is the package's selected model at its own 12-month horizon. The bar is the long-run level.</li>
<li>Nothing is adjusted for the number of cells. With this many, isolated good cells are expected by chance.</li>
<li>Ten-year cells are mostly blank: 25 years hold at most two and a half non-overlapping windows.</li>
<li>Full method and tables: <code>reports/horizon_study.md</code>, <code>results/horizons/</code>.</li>
</ul>
</section>
</main>
<div id="tip" hidden></div>
<script type="application/json" id="hz-data">{payload}</script>
<script>{JS}</script>"""


def build(fragment: str | Path | None = None) -> str:
    b = body(R.load())
    head = (f'<title>Horizon Map</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            f"<style>{TOKENS}{CSS}</style>\n")
    PAGE.write_text(document(b, head, viewport="width=device-width, initial-scale=1"))
    if fragment:
        Path(fragment).write_text(head + b + "\n")
    return str(PAGE)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m assetforecast.horizons.page")
    ap.add_argument("--fragment", default=None)
    print(build(ap.parse_args(argv).fragment))
    return 0


if __name__ == "__main__":
    sys.exit(main())
