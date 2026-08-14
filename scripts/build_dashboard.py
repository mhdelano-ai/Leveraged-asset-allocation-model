"""Generator for docs/dashboard.html -- the live operating dashboard.

    python scripts/build_dashboard.py

Reuses the existing model rather than reimplementing it: lam.data.panels for
the Nasdaq-100 panel, lam.alloc.rules for the four-state rule (improved
variant, RuleParams(up_wild=1.0)), and lam.report.dashboard to shape the
result into the JSON payload embedded below. This script's only job is to
call that payload builder and splice the result into the page template --
see lam.report.dashboard's module docstring for why the shaping logic lives
there instead of here.

The page itself is a single self-contained HTML file: history is baked in at
generation time (it never changes, so there is no reason to compute it on
every page load), and only the last day or two of prices are ever fetched
live, client-side, against a Twelve Data API key the *reader* supplies and
that this script never sees. Re-run this script whenever the model, its
parameters, or the baked history changes; the live segment updates itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from lam.report.dashboard import load_and_build_payload

OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "dashboard.html"

# The placeholder this script substitutes the payload into. Chosen to be
# something that can never collide with real HTML/CSS/JS content or with
# JSON output, so a plain str.replace is safe -- no templating engine needed
# for a single substitution, and it avoids fighting the CSS/JS braces below
# that would make str.format unusable here.
PLACEHOLDER = "__LAM_DASHBOARD_PAYLOAD__"

# The page shell itself: hand-authored HTML/CSS/JS, developed and verified
# against Playwright (see tests/test_dashboard.py) before being embedded here.
# Only the payload placeholder is ever substituted; everything else is static.
TEMPLATE = '''
<title>Four-State Rotation — Live Position</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{
  color-scheme: light;
  --ground:#f1f4f3; --panel:#fbfcfc; --panel-2:#e9eeed;
  --ink:#101619; --ink-2:#4d5a5e; --muted:#7f8f92;
  --accent:#0d6c76; --accent-soft:#0d6c761f;
  --breach:#a83a30; --breach-soft:#a83a3021;
  --gain:#2c7a58;   --gain-soft:#2c7a5821;
  --warn:#a8751f;   --warn-soft:#a8751f21;
  --rule:#d8e0de; --hair:#e6ecea;
  --shadow:0 1px 2px rgba(16,22,25,.05), 0 8px 24px -16px rgba(16,22,25,.22);
  --serif: ui-serif, Georgia, "Iowan Old Style", "Times New Roman", serif;
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --ground:#0c1113; --panel:#141b1e; --panel-2:#1b2427;
    --ink:#e9efee; --ink-2:#a3b1b3; --muted:#78878a;
    --accent:#47b3bf; --accent-soft:#47b3bf24;
    --breach:#dd6a5f; --breach-soft:#dd6a5f26;
    --gain:#4faa83;   --gain-soft:#4faa8326;
    --warn:#d1a24a;   --warn-soft:#d1a24a26;
    --rule:#25302f; --hair:#1d2628;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --ground:#0c1113; --panel:#141b1e; --panel-2:#1b2427;
  --ink:#e9efee; --ink-2:#a3b1b3; --muted:#78878a;
  --accent:#47b3bf; --accent-soft:#47b3bf24;
  --breach:#dd6a5f; --breach-soft:#dd6a5f26;
  --gain:#4faa83;   --gain-soft:#4faa8326;
  --warn:#d1a24a;   --warn-soft:#d1a24a26;
  --rule:#25302f; --hair:#1d2628;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.55;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:900px; margin:0 auto; padding:0 16px 80px}
@media(max-width:640px){ .wrap{padding:0 12px 56px} }

header.top{padding:32px 0 4px}
.eyebrow{
  font-family:var(--mono); font-size:10.5px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted); margin:0 0 10px;
}
h1{
  font-family:var(--serif); font-weight:600; font-size:clamp(26px,5vw,38px);
  line-height:1.06; letter-spacing:-.02em; margin:0 0 6px;
}
.standfirst{ font-size:clamp(14px,1.8vw,16px); color:var(--ink-2); margin:0 0 4px; max-width:56ch }
.generated{ font-family:var(--mono); font-size:11.5px; color:var(--muted); margin:10px 0 0 }

main{ display:flex; flex-direction:column; gap:20px; margin-top:22px }

.card{
  background:var(--panel); border:1px solid var(--rule); border-radius:12px;
  padding:20px; box-shadow:var(--shadow);
}
h2{ font-family:var(--serif); font-size:17px; font-weight:650; margin:0 0 4px; letter-spacing:-.01em }
h3{ font-family:var(--sans); font-size:12.5px; font-weight:650; letter-spacing:.03em; color:var(--muted); margin:0 0 10px; text-transform:uppercase }

/* ---- status / setup banner ---- */
.status-card{ display:flex; flex-direction:column; gap:12px; border-left-width:4px; border-left-style:solid; }
.status-card.live{ border-left-color:var(--gain); }
.status-card.stale{ border-left-color:var(--warn); }
.status-line{ display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
.status-dot{ width:8px; height:8px; border-radius:50%; flex:none; position:relative; top:-1px; display:inline-block }
.status-card.live .status-dot{ background:var(--gain) }
.status-card.stale .status-dot{ background:var(--warn) }
.status-title{ font-weight:650; font-size:15px }
.status-reason{ font-size:13.5px; color:var(--ink-2); margin:0 }
.key-form{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:2px }
/* `hidden` and this class carry equal specificity, and this rule loads after
   the UA stylesheet's `[hidden]{display:none}`, so without this it would win
   and the form would stay visible even when JS sets the hidden attribute. */
.key-form[hidden]{ display:none }
.key-form input[type="password"], .key-form input[type="text"]{
  flex:1 1 220px; min-width:0; font-family:var(--mono); font-size:13px;
  padding:9px 11px; border-radius:8px; border:1px solid var(--rule);
  background:var(--panel-2); color:var(--ink);
}
button{
  font-family:var(--sans); font-size:13px; font-weight:650; cursor:pointer;
  padding:9px 14px; border-radius:8px; border:1px solid var(--accent);
  background:var(--accent); color:#fff; white-space:nowrap;
}
button.secondary{ background:transparent; color:var(--accent); }
button.ghost{ background:transparent; color:var(--muted); border-color:var(--rule); font-weight:500; }
.key-note{ font-size:12px; color:var(--muted); margin:0 }
.key-note a{ color:var(--accent) }

/* ---- position card ---- */
.position-top{ display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap }
.state-tag{
  font-family:var(--mono); font-size:11px; font-weight:650; letter-spacing:.04em;
  padding:5px 10px; border-radius:99px; display:inline-flex; align-items:center; gap:6px;
}
.state-tag.levered{ background:var(--accent-soft); color:var(--accent) }
.state-tag.unlevered{ background:var(--panel-2); color:var(--ink-2) }
.state-tag.wild-up{ background:var(--gain-soft); color:var(--gain) }
.state-tag.wild-down{ background:var(--breach-soft); color:var(--breach) }
.changed-flag{
  font-family:var(--sans); font-size:11px; font-weight:650; color:var(--warn);
  background:var(--warn-soft); padding:4px 9px; border-radius:99px;
}
.hold-line{ font-family:var(--serif); font-size:clamp(22px,4.5vw,30px); font-weight:600; margin:14px 0 2px; letter-spacing:-.015em }
.hold-sub{ font-size:13.5px; color:var(--muted); margin:0 0 16px }
.position-grid{ display:grid; grid-template-columns:1fr 1fr; gap:14px 20px; margin-top:6px }
@media(max-width:520px){ .position-grid{ grid-template-columns:1fr } }
.kv dt{ font-size:11.5px; color:var(--muted); margin:0 0 2px }
.kv dd{ margin:0; font-family:var(--mono); font-size:15px; font-variant-numeric:tabular-nums }
.account-row{ display:flex; align-items:center; gap:8px; margin-top:16px; padding-top:16px; border-top:1px solid var(--hair) }
.account-row label{ font-size:12.5px; color:var(--muted); white-space:nowrap }
.account-row input{
  font-family:var(--mono); font-size:14px; padding:7px 10px; border-radius:7px;
  border:1px solid var(--rule); background:var(--panel-2); color:var(--ink); width:130px;
}
.days-line{ font-size:13px; color:var(--muted); margin:16px 0 0; display:flex; align-items:center; gap:10px; flex-wrap:wrap }

/* ---- stats card ---- */
.statgrid{ display:grid; grid-template-columns:repeat(2,1fr); gap:1px; background:var(--rule); border:1px solid var(--rule); border-radius:10px; overflow:hidden }
@media(max-width:480px){ .statgrid{ grid-template-columns:1fr } }
.stat{ background:var(--panel); padding:14px 16px; display:flex; flex-direction:column; gap:3px }
.stat .k{ font-family:var(--sans); font-size:11px; color:var(--muted); letter-spacing:.02em }
.stat .v{ font-family:var(--mono); font-size:19px; font-variant-numeric:tabular-nums; letter-spacing:-.01em }
.stat .n{ font-family:var(--sans); font-size:11px; color:var(--muted) }
.stat .v.pos{ color:var(--gain) } .stat .v.neg{ color:var(--breach) }

/* ---- charts ---- */
figure{ margin:0 }
.chart-card{ padding:18px 16px 14px }
.chart-head{ display:flex; justify-content:space-between; align-items:baseline; gap:14px; flex-wrap:wrap; margin-bottom:10px }
.chart-title{ font-size:13.5px; font-weight:650 }
.legend{ display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--ink-2) }
.legend span{ display:inline-flex; align-items:center; gap:6px }
.swatch{ width:12px; height:3px; border-radius:2px; display:inline-block }
svg{ display:block; width:100%; height:auto; overflow:visible }
.grid{ stroke:var(--hair); stroke-width:1 }
.axis{ stroke:var(--rule); stroke-width:1 }
.tick{ font-family:var(--mono); font-size:9.5px; fill:var(--muted); font-variant-numeric:tabular-nums }
.axlab{ font-family:var(--sans); font-size:9.5px; fill:var(--muted); letter-spacing:.06em; text-transform:uppercase; font-weight:600 }
.s-strat{ stroke:var(--accent) } .s-qqq{ stroke:var(--muted) }
.f-strat{ fill:var(--accent) } .f-qqq{ fill:var(--muted) }
.line{ fill:none; stroke-width:1.8; stroke-linejoin:round; stroke-linecap:round }
figcaption{ font-size:12px; color:var(--muted); margin-top:10px }

/* ---- table ---- */
.scroll{ overflow-x:auto; -webkit-overflow-scrolling:touch }
table{ border-collapse:collapse; width:100%; font-size:13px; min-width:480px }
th,td{ padding:8px 10px; text-align:right; border-bottom:1px solid var(--hair) }
th:first-child,td:first-child{ text-align:left }
thead th{ font-family:var(--mono); font-size:10px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); font-weight:500; border-bottom:1px solid var(--rule) }
tbody td{ font-family:var(--mono); font-variant-numeric:tabular-nums }
tbody td:first-child{ font-family:var(--sans); font-weight:600; color:var(--ink) }

footer{ margin-top:36px; padding-top:20px; border-top:1px solid var(--rule); font-size:12.5px; color:var(--muted); display:flex; flex-direction:column; gap:8px }
footer a{ color:var(--accent) }
footer p{ margin:0; max-width:70ch }
.warning{ background:var(--breach-soft); border:1px solid var(--breach); border-left-width:3px; border-radius:8px; padding:14px 16px; font-size:13px; color:var(--ink-2) }
.warning b{ color:var(--breach) }

:where(button):focus-visible, :where(input):focus-visible{ outline:2px solid var(--accent); outline-offset:2px }
@media (prefers-reduced-motion: reduce){ *{ transition:none !important; animation:none !important } }
</style>

<div class="wrap">
  <header class="top">
    <p class="eyebrow">Nasdaq-100 &middot; four-state rotation, improved variant</p>
    <h1>Live Position</h1>
    <p class="standfirst">What the rule says to hold today, the market conditions behind it, and how far today is from a different answer.</p>
  </header>

  <main>
    <section class="card status-card" id="status-card">
      <div class="status-line">
        <span class="status-dot"></span>
        <span class="status-title" id="status-title">Checking&hellip;</span>
      </div>
      <p class="status-reason" id="status-reason"></p>
      <div class="key-form" id="key-form" hidden>
        <input type="password" id="key-input" placeholder="Twelve Data API key" autocomplete="off" spellcheck="false">
        <button id="key-save">Save key</button>
        <button class="ghost" id="key-clear" type="button">Clear</button>
      </div>
      <p class="key-note" id="key-note" hidden>
        Free key at <a href="https://twelvedata.com/pricing" target="_blank" rel="noopener">twelvedata.com</a> (800 calls/day).
        Stored only in this browser's localStorage &mdash; sent to Twelve Data and nowhere else, and never written into this file.
      </p>
    </section>

    <section class="card position-card">
      <div class="position-top">
        <span class="state-tag" id="state-tag"><span id="state-label">&mdash;</span></span>
        <span class="changed-flag" id="changed-flag" hidden>state changed</span>
      </div>
      <p class="hold-line" id="hold-line">&mdash;</p>
      <p class="hold-sub" id="hold-sub"></p>
      <div class="position-grid" id="share-grid"></div>
      <div class="account-row">
        <label for="account-value">Account value $</label>
        <input type="number" id="account-value" inputmode="decimal" min="0" step="100" placeholder="10000">
      </div>
      <p class="days-line">
        <span>In this state <strong id="days-in-state">&mdash;</strong> days</span>
        <button class="secondary" id="ack-button" type="button" style="display:none">Acknowledge this state</button>
        <span id="ack-note" style="display:none"></span>
      </p>
    </section>

    <section class="card stats-card">
      <h3>Market conditions</h3>
      <div class="statgrid">
        <div class="stat"><span class="k">QQQ vs 200-day SMA</span><span class="v" id="stat-sma">&mdash;</span></div>
        <div class="stat"><span class="k">20-day realised vol</span><span class="v" id="stat-vol">&mdash;</span><span class="n">threshold 28%</span></div>
        <div class="stat"><span class="k">Distance to trend flip</span><span class="v" id="stat-trend-flip">&mdash;</span></div>
        <div class="stat"><span class="k">Distance to vol flip</span><span class="v" id="stat-vol-flip">&mdash;</span></div>
      </div>
    </section>

    <figure class="card chart-card">
      <div class="chart-head">
        <span class="chart-title">Trailing 12 months</span>
        <span class="legend">
          <span><i class="swatch" style="background:var(--accent)"></i>Strategy</span>
          <span><i class="swatch" style="background:var(--muted)"></i>QQQ</span>
        </span>
      </div>
      <svg id="chart-trailing" viewBox="0 0 860 340" role="img" aria-label="Trailing twelve month growth of one dollar and drawdown, strategy versus QQQ"></svg>
      <figcaption id="caption-trailing"></figcaption>
    </figure>

    <figure class="card chart-card">
      <div class="chart-head">
        <span class="chart-title">Full history, 1985&ndash;present</span>
        <span class="legend">
          <span><i class="swatch" style="background:var(--accent)"></i>Strategy</span>
          <span><i class="swatch" style="background:var(--muted)"></i>QQQ</span>
        </span>
      </div>
      <svg id="chart-full" viewBox="0 0 860 340" role="img" aria-label="Full history growth of one dollar, log scale, and drawdown, strategy versus QQQ since 1985"></svg>
      <figcaption id="caption-full"></figcaption>
    </figure>

    <section class="card">
      <h3>Backtest headline, full sample</h3>
      <div class="scroll">
        <table>
          <thead><tr><th>Series</th><th>CAGR</th><th>Vol</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th><th>Switches/yr</th></tr></thead>
          <tbody id="headline-body"></tbody>
        </table>
      </div>
    </section>

    <div class="warning">
      <b>This reports what the rule says, not a trade ticket.</b>
      <span id="warning-text"></span>
    </div>
  </main>

  <footer id="footer">
  </footer>
</div>

<script>
"use strict";
const DATA = __LAM_DASHBOARD_PAYLOAD__;

/* =========================================================================
   Core state machine -- this MUST match lam.alloc.rules.four_state_signal
   exactly: same rolling windows, same one-day lag, same band hysteresis with
   its ffill+seed behaviour. tests/test_dashboard.py drives this exact
   function through Playwright and compares it against the Python model on a
   fixed price series; if you touch the maths here, that test is what catches
   a silent divergence between what the backtest says and what this page
   tells you to hold today. See four_state_signal's docstring in rules.py for
   why the trend leg runs on raw closes (price_basis) while nothing here
   needs a total-return series at all -- a live feed has no dividend stream
   to reinvest, so raw close *is* the only basis this page can ever compute.
   ========================================================================= */

function rollingMean(arr, window) {
  const n = arr.length, out = new Array(n).fill(NaN);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += arr[i];
    if (i >= window) sum -= arr[i - window];
    if (i >= window - 1) out[i] = sum / window;
  }
  return out;
}

function rollingStd(arr, window) {
  // Sample standard deviation (ddof=1), matching pandas' default -- see
  // four_state_signal's `.rolling(...).std()`.
  const n = arr.length, out = new Array(n).fill(NaN);
  let sum = 0, sumSq = 0;
  for (let i = 0; i < n; i++) {
    sum += arr[i]; sumSq += arr[i] * arr[i];
    if (i >= window) { sum -= arr[i - window]; sumSq -= arr[i - window] * arr[i - window]; }
    if (i >= window - 1) {
      const mean = sum / window;
      const variance = (sumSq - window * mean * mean) / (window - 1);
      out[i] = Math.sqrt(Math.max(variance, 0));
    }
  }
  return out;
}

const RULE_DEFAULTS = {
  sma_window: 200, vol_window: 20, vol_threshold: 0.28, levered_multiple: 3.0,
  up_calm: 3.0, up_wild: 0.0, down_calm: 1.0, down_wild: 0.0, sma_band: 0.01,
};

/**
 * dates: ISO date strings, ascending. closes: raw price levels, same length.
 * Output arrays are aligned to dates[1..n-1] (pct_change has nothing to
 * difference the first close against, exactly like pandas' behaviour on the
 * Python side once `.dropna()` runs) -- so out.dates[0] === dates[1].
 */
function computeStateSeries(dates, closes, params) {
  const p = Object.assign({}, RULE_DEFAULTS, params || {});
  const n = closes.length;
  const price = closes.slice(1);
  const outDates = dates.slice(1);
  const m = price.length;
  const ret = new Array(m);
  for (let i = 1; i < n; i++) ret[i - 1] = closes[i] / closes[i - 1] - 1;

  const sma = rollingMean(price, p.sma_window);
  const vol = rollingStd(ret, p.vol_window).map(s => s * Math.sqrt(252));

  // Hysteresis: decisive[j] = +1/0 only outside the band; NaN inside it, then
  // forward-filled so each of those days inherits the last decisive crossing.
  // With sma_band=0 nothing ever falls inside a zero-width band, which is
  // what makes that value reproduce a plain `price > sma`.
  const decisive = new Array(m).fill(NaN);
  for (let j = 0; j < m; j++) {
    if (Number.isNaN(sma[j])) continue;
    if (price[j] > sma[j] * (1 + p.sma_band)) decisive[j] = 1.0;
    else if (price[j] < sma[j] * (1 - p.sma_band)) decisive[j] = 0.0;
  }
  const first = sma.findIndex(v => !Number.isNaN(v));
  if (first >= 0 && Number.isNaN(decisive[first])) {
    decisive[first] = price[first] > sma[first] ? 1.0 : 0.0;
  }
  const aboveRaw = new Array(m).fill(NaN);
  let hold = NaN;
  for (let j = 0; j < m; j++) {
    if (!Number.isNaN(decisive[j])) hold = decisive[j];
    aboveRaw[j] = hold;
  }

  const state = new Array(m), exposure = new Array(m), pctAboveSma = new Array(m);
  for (let j = 0; j < m; j++) {
    pctAboveSma[j] = Number.isNaN(sma[j]) ? NaN : (price[j] / sma[j] - 1);
    if (j === 0) {
      // Mirrors four_state_signal's single "warmup" row: shift(1) has no
      // j-1 to look up at the very first position of index_tr's own domain.
      state[j] = "warmup"; exposure[j] = 0.0;
      continue;
    }
    // NaN vs a number is always false in IEEE 754, in JS exactly as in numpy,
    // so "still inside the pre-seed warmup" (aboveRaw[j-1]=NaN) reads as
    // "not above" here with no special-casing -- same as the Python side.
    const above = aboveRaw[j - 1] === 1.0;
    const calm = vol[j - 1] < p.vol_threshold;
    if (above && calm) { state[j] = "levered"; exposure[j] = p.up_calm; }
    else if (above && !calm) { state[j] = "cash (volatile uptrend)"; exposure[j] = p.up_wild; }
    else if (!above && calm) { state[j] = "unlevered"; exposure[j] = p.down_calm; }
    else { state[j] = "cash (volatile downtrend)"; exposure[j] = p.down_wild; }
  }

  return {
    dates: outDates, price, sma, vol, state, exposure, pctAboveSma,
    // Unlagged -- today's own banded reading off today's own close, i.e.
    // what tomorrow's (lagged) state will copy. See rules.py's `trend_held`.
    trendHeld: aboveRaw.map(v => v === 1.0),
  };
}

function instrumentFor(exposure, params) {
  if (exposure >= params.levered_multiple - 1e-9) return "levered";
  if (exposure >= 1e-9) return "unlevered";
  return "cash";
}

/* =========================================================================
   Small pure utilities shared by the chart and position code
   ========================================================================= */

function rebase(values) {
  const v0 = values[0];
  return values.map(v => v / v0);
}

function drawdownFromEquity(equity, priorPeak) {
  let peak = priorPeak === undefined ? -Infinity : priorPeak;
  return equity.map(v => { peak = Math.max(peak, v); return v / peak - 1.0; });
}

function dayCount(prevDateStr, dateStr) {
  const prev = Date.parse(prevDateStr + "T00:00:00Z");
  const cur = Date.parse(dateStr + "T00:00:00Z");
  return Math.round((cur - prev) / 86400000);
}

function fmtPct(v, digits) {
  digits = digits === undefined ? 1 : digits;
  return (v >= 0 ? "+" : "") + (v * 100).toFixed(digits) + "%";
}

/* =========================================================================
   localStorage -- API key, per-symbol daily cache, account value, ack state
   ========================================================================= */

const LS_KEY = "lam_dash_api_key";
const LS_ACCOUNT = "lam_dash_account_value";
const LS_ACK_STATE = "lam_dash_ack_state";
const CACHE_PREFIX = "lam_dash_cache_";

function safeGet(key) { try { return localStorage.getItem(key); } catch { return null; } }
function safeSet(key, val) { try { localStorage.setItem(key, val); } catch { /* storage full/blocked -- degrade to re-fetching, not fatal */ } }
function safeRemove(key) { try { localStorage.removeItem(key); } catch { /* no-op */ } }

function getApiKey() { const k = safeGet(LS_KEY); return k ? k.trim() : ""; }
function setApiKey(k) { safeSet(LS_KEY, k.trim()); }
function clearApiKey() { safeRemove(LS_KEY); }

function getAccountValue() { const v = parseFloat(safeGet(LS_ACCOUNT)); return Number.isFinite(v) && v > 0 ? v : 0; }
function setAccountValue(v) { safeSet(LS_ACCOUNT, String(v)); }

function getAckState() { return safeGet(LS_ACK_STATE); }
function setAckState(s) { safeSet(LS_ACK_STATE, s); }

function todayLocalStr() {
  const d = new Date();
  return d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0");
}

function loadCached(symbol) {
  try {
    const raw = safeGet(CACHE_PREFIX + symbol);
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}
function saveCached(symbol, series) {
  safeSet(CACHE_PREFIX + symbol, JSON.stringify({ fetchedOn: todayLocalStr(), dates: series.dates, closes: series.closes }));
}

/* =========================================================================
   Twelve Data fetch, with explicit, user-facing failure classification
   ========================================================================= */

async function fetchSeries(symbol, apiKey, opts) {
  opts = opts || {};
  const outputsize = opts.outputsize || 500;
  const timeoutMs = opts.timeoutMs || 12000;
  const url = "https://api.twelvedata.com/time_series?symbol=" + encodeURIComponent(symbol)
    + "&interval=1day&outputsize=" + outputsize + "&apikey=" + encodeURIComponent(apiKey);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let resp;
  try {
    resp = await fetch(url, { signal: controller.signal });
  } catch (err) {
    clearTimeout(timer);
    if (err && err.name === "AbortError") {
      return { ok: false, reason: "timeout", detail: "Twelve Data did not respond in time." };
    }
    return { ok: false, reason: "network", detail: "Could not reach Twelve Data (offline, or the request was blocked)." };
  }
  clearTimeout(timer);

  if (resp.status === 401) return { ok: false, reason: "bad_key", detail: "Twelve Data rejected the API key." };
  if (resp.status === 429) return { ok: false, reason: "rate_limited", detail: "Twelve Data's daily rate limit has been reached." };
  if (!resp.ok) return { ok: false, reason: "http_error", detail: "Twelve Data returned HTTP " + resp.status + "." };

  let body;
  try { body = await resp.json(); } catch { return { ok: false, reason: "malformed", detail: "Twelve Data's response was not valid JSON." }; }

  // Twelve Data returns HTTP 200 with a JSON error body for some failures
  // (an over-restricted demo key, an unknown symbol) -- status text, not the
  // HTTP code, is the reliable signal there.
  if (body.status === "error" || !Array.isArray(body.values) || body.values.length === 0) {
    return { ok: false, reason: "api_error", detail: (body && body.message) || "The response did not include the expected price data." };
  }

  // Most-recent-first on the wire; this page wants ascending order throughout.
  const rows = body.values.slice().reverse();
  return { ok: true, dates: rows.map(r => r.datetime), closes: rows.map(r => Number(r.close)) };
}

/* =========================================================================
   Live equity construction -- extends the equity curve past the baked
   history using the SAME return definitions backtest_four_state used, so the
   live tail is not a different model wearing the same chart:
     - levered days use TQQQ's own realised return (the real fund, not a
       simulated one -- more accurate than re-deriving the LETF formula for
       a handful of recent days, and we already fetched it)
     - unlevered days use QQQ's price return plus the same modelled dividend
       accrual nasdaq.total_return() uses, less the QQQ expense ratio
     - cash days accrue at the last known bill rate plus the cash spread,
       held flat -- there is no live short-rate feed here, and the live tail
       is short enough (days to a few weeks between dashboard regenerations)
       that a flat approximation is honest rather than sloppy
   Switch costs are *not* applied on the live tail (a few bp on a handful of
   days is not worth the complexity here; the baked backtest still charges
   them in full).
   ========================================================================= */

function buildLiveReturns(sig, qqqCloses) {
  const n = sig.dates.length;
  const stratRet = new Array(n).fill(0);
  const qqqRet = new Array(n).fill(0);
  const tqqqByDate = sig.tqqqByDate;
  for (let j = 1; j < n; j++) {
    const dt = dayCount(sig.dates[j - 1], sig.dates[j]);
    const priceRet = qqqCloses[j] / qqqCloses[j - 1] - 1;
    const divAccrual = Math.pow(1 + DATA.dividend_yield, dt / 365.25) - 1;
    qqqRet[j] = (1 + priceRet) * (1 + divAccrual) - 1;

    const inst = instrumentFor(sig.exposure[j], DATA.params);
    if (inst === "levered") {
      const p0 = tqqqByDate.get(sig.dates[j - 1]), p1 = tqqqByDate.get(sig.dates[j]);
      stratRet[j] = (p0 !== undefined && p1 !== undefined) ? (p1 / p0 - 1) : qqqRet[j] * DATA.params.levered_multiple;
    } else if (inst === "unlevered") {
      stratRet[j] = (1 + qqqRet[j]) * (1 - DATA.costs.qqq_expense_ratio * dt / 365.0) - 1;
    } else {
      stratRet[j] = (DATA.cash_rate_annual + DATA.costs.cash_spread) * dt / 360.0;
    }
  }
  return { stratRet, qqqRet };
}

/* =========================================================================
   Rendering
   ========================================================================= */

const STATE_META = {
  "levered": { cls: "levered", label: "Levered" },
  "cash (volatile uptrend)": { cls: "wild-up", label: "Volatile uptrend" },
  "unlevered": { cls: "unlevered", label: "Unlevered" },
  "cash (volatile downtrend)": { cls: "wild-down", label: "Volatile downtrend" },
  "warmup": { cls: "unlevered", label: "Warming up" },
};

function instrumentLabel(exposure, params) {
  const kind = instrumentFor(exposure, params);
  if (kind === "levered") return { ticker: "TQQQ", text: params.levered_multiple.toFixed(0) + "x leveraged (TQQQ)" };
  if (kind === "unlevered") return { ticker: "QQQ", text: "1x QQQ" };
  return { ticker: null, text: "Cash" };
}

function renderStatus(mode, asOf, detail) {
  const card = document.getElementById("status-card");
  const title = document.getElementById("status-title");
  const reasonEl = document.getElementById("status-reason");
  const keyForm = document.getElementById("key-form");
  const keyNote = document.getElementById("key-note");
  card.classList.remove("live", "stale");
  card.classList.add(mode === "live" ? "live" : "stale");
  if (mode === "live") {
    title.textContent = "Live — as of " + asOf;
    reasonEl.textContent = "QQQ and TQQQ fetched from Twelve Data and cached for the rest of today.";
    keyForm.hidden = true;
    keyNote.hidden = true;
  } else {
    title.textContent = "Showing backtested data as of " + asOf;
    reasonEl.textContent = detail || "";
    keyForm.hidden = false;
    keyNote.hidden = false;
  }
}

function renderPosition(ctx) {
  const meta = STATE_META[ctx.state] || STATE_META.unlevered;
  const tag = document.getElementById("state-tag");
  tag.className = "state-tag " + meta.cls;
  document.getElementById("state-label").textContent = meta.label;

  const inst = instrumentLabel(ctx.exposure, DATA.params);
  document.getElementById("hold-line").textContent = inst.text;
  document.getElementById("hold-sub").textContent = ctx.exposure > 0
    ? (ctx.exposure.toFixed(2) + "x notional exposure to the Nasdaq-100")
    : "No equity exposure";

  document.getElementById("days-in-state").textContent = ctx.daysInState;
  document.getElementById("changed-flag").hidden = !(ctx.daysInState <= 3);

  const grid = document.getElementById("share-grid");
  const accountValue = getAccountValue();
  if (ctx.live && inst.ticker && accountValue > 0) {
    const price = inst.ticker === "TQQQ" ? ctx.tqqqPrice : ctx.qqqPrice;
    const shares = accountValue / price;
    grid.innerHTML =
      '<div class="kv"><dt>Shares of ' + inst.ticker + '</dt><dd>' + shares.toFixed(shares < 10 ? 3 : 1) + '</dd></div>' +
      '<div class="kv"><dt>' + inst.ticker + ' price</dt><dd>$' + price.toFixed(2) + '</dd></div>';
  } else if (ctx.live && !inst.ticker) {
    grid.innerHTML = '<div class="kv" style="grid-column:1/-1"><dt>Position</dt><dd>100% cash</dd></div>';
  } else if (!ctx.live) {
    grid.innerHTML = '<div class="kv" style="grid-column:1/-1"><dt>Share counts</dt>'
      + '<dd style="font-family:inherit;font-size:13px;font-weight:400">Need a live price — add an API key above.</dd></div>';
  } else {
    grid.innerHTML = '<div class="kv" style="grid-column:1/-1"><dt>Share counts</dt>'
      + '<dd style="font-family:inherit;font-size:13px;font-weight:400">Enter an account value below.</dd></div>';
  }

  const ackBtn = document.getElementById("ack-button");
  const ackNote = document.getElementById("ack-note");
  if (getAckState() === ctx.state) {
    ackBtn.style.display = "none";
    ackNote.style.display = "inline";
    ackNote.textContent = "Acknowledged";
  } else {
    ackBtn.style.display = "inline-block";
    ackNote.style.display = "none";
    ackBtn.onclick = () => { setAckState(ctx.state); renderPosition(ctx); };
  }
}

function renderStats(ctx) {
  const smaEl = document.getElementById("stat-sma");
  smaEl.textContent = fmtPct(ctx.pctAboveSma);
  smaEl.className = "v " + (ctx.pctAboveSma >= 0 ? "pos" : "neg");

  const volEl = document.getElementById("stat-vol");
  volEl.textContent = (ctx.vol20 * 100).toFixed(1) + "%";
  volEl.className = "v " + (ctx.vol20 < DATA.params.vol_threshold ? "pos" : "neg");

  // Distance to a trend flip is measured from whichever band edge is
  // *currently held* (trendHeld), not from the SMA line itself -- inside the
  // band the relevant edge is the far one from today's reading, not the near
  // one, because hysteresis is already holding the state through the near one.
  // Worked entirely in percentage terms (price/sma - 1, already carried as
  // pctAboveSma) rather than off raw levels, so this reads the same way
  // whether the number came from a live fetch or the baked fallback -- the
  // baked payload never carries a raw QQQ price at all (see the module
  // docstring on why: it is built from ^NDX, not QQQ, and the two are not on
  // the same price scale).
  const band = DATA.params.sma_band;
  const edgeMult = ctx.trendHeld ? (1 - band) : (1 + band);
  const trendFlipEl = document.getElementById("stat-trend-flip");
  if (Number.isFinite(ctx.pctAboveSma)) {
    const dist = edgeMult / (1.0 + ctx.pctAboveSma) - 1.0;
    trendFlipEl.textContent = fmtPct(dist) + (ctx.trendHeld ? " to flip below" : " to flip above");
    trendFlipEl.className = "v";
  } else {
    trendFlipEl.textContent = "—";
  }

  const volFlipEl = document.getElementById("stat-vol-flip");
  const volGap = DATA.params.vol_threshold - ctx.vol20;
  volFlipEl.textContent = (volGap >= 0 ? "+" : "") + (volGap * 100).toFixed(1) + "pp";
  volFlipEl.className = "v " + (volGap >= 0 ? "pos" : "neg");
}

function svgEl(tag, attrs) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

function drawDualPanel(svgId, captionId, dates, stratEq, qqqEq, opts) {
  opts = opts || {};
  const svg = document.getElementById(svgId);
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (dates.length < 2) return;

  const W = 860, L = 44, R = 12, TOP = 14, H1 = 168, GAP = 28, H2 = 92;
  const axisY = TOP + H1 + GAP + H2;
  const ts = dates.map(d => Date.parse(d + "T00:00:00Z"));
  const t0 = ts[0], t1 = ts[ts.length - 1];
  const span = Math.max(t1 - t0, 1);
  const x = t => L + (t - t0) / span * (W - L - R);

  const stratDd = opts.stratDd || drawdownFromEquity(stratEq);
  const qqqDd = opts.qqqDd || drawdownFromEquity(qqqEq);

  let yEq;
  if (opts.log) {
    const lo = Math.log10(Math.min(...stratEq, ...qqqEq) * 0.92);
    const hi = Math.log10(Math.max(...stratEq, ...qqqEq) * 1.08);
    yEq = v => TOP + (hi - Math.log10(Math.max(v, 1e-9))) / (hi - lo) * H1;
  } else {
    const lo = Math.min(...stratEq, ...qqqEq) * 0.97;
    const hi = Math.max(...stratEq, ...qqqEq) * 1.03;
    yEq = v => TOP + (hi - v) / Math.max(hi - lo, 1e-9) * H1;
  }
  const ddLo = Math.min(-0.05, Math.min(...stratDd, ...qqqDd) * 1.15);
  const yDd = v => TOP + H1 + GAP + (0 - v) / (0 - ddLo) * H2;

  // gridlines, equity panel
  const eqTicks = opts.log
    ? [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192].filter(g => {
        const lo = Math.min(...stratEq, ...qqqEq) * 0.92, hi = Math.max(...stratEq, ...qqqEq) * 1.08;
        return g >= lo && g <= hi;
      })
    : (() => {
        const lo = Math.min(...stratEq, ...qqqEq), hi = Math.max(...stratEq, ...qqqEq);
        const step = (hi - lo) / 4;
        return [0, 1, 2, 3, 4].map(i => lo + i * step);
      })();
  eqTicks.forEach(g => {
    svg.appendChild(svgEl("line", { class: "grid", x1: L, x2: W - R, y1: yEq(g), y2: yEq(g) }));
    const t = svgEl("text", { class: "tick", x: L - 6, y: yEq(g) + 3, "text-anchor": "end" });
    t.textContent = opts.log ? ("$" + (g < 1 ? g.toFixed(2) : g.toFixed(0))) : g.toFixed(2) + "x";
    svg.appendChild(t);
  });

  // gridlines, drawdown panel
  const ddStep = ddLo <= -0.5 ? 0.2 : ddLo <= -0.2 ? 0.1 : 0.05;
  for (let g = 0; g > ddLo; g -= ddStep) {
    svg.appendChild(svgEl("line", { class: "grid", x1: L, x2: W - R, y1: yDd(g), y2: yDd(g) }));
    const t = svgEl("text", { class: "tick", x: L - 6, y: yDd(g) + 3, "text-anchor": "end" });
    t.textContent = (g * 100).toFixed(0) + "%";
    svg.appendChild(t);
  }

  svg.appendChild(svgEl("text", { class: "axlab", x: L, y: TOP - 4 })).textContent = opts.log ? "Growth of $1 (log)" : "Growth of $1";
  svg.appendChild(svgEl("text", { class: "axlab", x: L, y: TOP + H1 + GAP - 9 })).textContent = "Drawdown";
  svg.appendChild(svgEl("line", { class: "axis", x1: L, x2: W - R, y1: axisY, y2: axisY }));

  // x-axis ticks: date-proportional, a handful of evenly spaced labels. A
  // bare year is fine once ticks are a year or more apart (the full-history
  // chart); anything shorter needs month+year or every tick reads the same.
  const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const nTicks = 6;
  const yearOnly = span / nTicks > 340 * 86400000;
  for (let i = 0; i <= nTicks; i++) {
    const t = t0 + (span * i) / nTicks;
    const d = new Date(t);
    const label = yearOnly
      ? String(d.getUTCFullYear())
      : MONTHS[d.getUTCMonth()] + " ’" + String(d.getUTCFullYear()).slice(2);
    const el = svgEl("text", { class: "tick", x: x(t), y: axisY + 14, "text-anchor": "middle" });
    el.textContent = label;
    svg.appendChild(el);
  }

  const pathFor = (values, yFn) => {
    let d = "";
    for (let i = 0; i < values.length; i++) d += (i ? "L" : "M") + x(ts[i]).toFixed(1) + "," + yFn(values[i]).toFixed(1);
    return d;
  };
  svg.appendChild(svgEl("path", { class: "line s-qqq", d: pathFor(qqqEq, yEq) }));
  svg.appendChild(svgEl("path", { class: "line s-strat", d: pathFor(stratEq, yEq) }));
  svg.appendChild(svgEl("path", { class: "line s-qqq", d: pathFor(qqqDd, yDd) }));
  svg.appendChild(svgEl("path", { class: "line s-strat", d: pathFor(stratDd, yDd) }));

  const cap = document.getElementById(captionId);
  if (cap) {
    const stratTot = stratEq[stratEq.length - 1] / stratEq[0] - 1;
    const qqqTot = qqqEq[qqqEq.length - 1] / qqqEq[0] - 1;
    const stratWorstDd = Math.min(...stratDd), qqqWorstDd = Math.min(...qqqDd);
    cap.textContent = "Strategy " + fmtPct(stratTot, 0) + " (worst drawdown " + fmtPct(stratWorstDd, 0)
      + ") vs QQQ " + fmtPct(qqqTot, 0) + " (worst " + fmtPct(qqqWorstDd, 0) + ") over this window.";
  }
}

function renderHeadlineTable() {
  const body = document.getElementById("headline-body");
  const rows = [
    ["Strategy", DATA.headline.strategy],
    ["QQQ", DATA.headline.qqq],
  ];
  body.innerHTML = rows.map(([label, s]) => {
    const calmar = s.calmar === null || s.calmar === undefined ? "—" : s.calmar.toFixed(2);
    const switches = s.switches_per_year === undefined ? "—" : s.switches_per_year.toFixed(1);
    return "<tr><td>" + label + "</td><td>" + fmtPct(s.cagr) + "</td><td>" + (s.vol * 100).toFixed(1) + "%</td>"
      + "<td>" + s.sharpe.toFixed(2) + "</td><td>" + fmtPct(s.max_dd) + "</td><td>" + calmar + "</td><td>" + switches + "</td></tr>";
  }).join("");
}

function renderFooter() {
  const footer = document.getElementById("footer");
  footer.innerHTML =
    '<p>Panel N (Nasdaq-100 total return, 1985-10+). Strategy is the four-state rotation, improved variant '
    + '(<code>up_wild=1.0</code>): above the 200-day SMA but too volatile to lever, hold 1x QQQ rather than cash. '
    + 'Trend is evaluated on raw closes with a &plusmn;1% hysteresis band around the SMA, matching exactly what this '
    + 'page computes live -- see <code>lam.alloc.rules.four_state_signal</code>.</p>'
    + '<p>Live prices from <a href="https://twelvedata.com" target="_blank" rel="noopener">Twelve Data</a>. '
    + 'Backtest generated ' + DATA.generated_at + '. Full methodology at '
    + '<a href="growth_findings.html">growth_findings.html</a> and <a href="findings.html">findings.html</a>.</p>';
  document.getElementById("warning-text").textContent =
    "The strategy's full-sample worst drawdown is " + fmtPct(DATA.headline.strategy.max_dd, 1) + ". "
    + "Past performance of a backtested rule is not a projection of future results.";
}

/* =========================================================================
   Orchestration
   ========================================================================= */

function renderBaked() {
  const d = DATA.daily;
  const last = d.dates.length - 1;
  renderStatus("stale", DATA.generated_at, "No live fetch has completed yet.");
  renderPosition({
    state: DATA.current.state, exposure: DATA.current.exposure,
    daysInState: DATA.current.days_in_state, live: false,
  });
  renderStats({
    pctAboveSma: DATA.current.pct_above_sma, vol20: DATA.current.vol20,
    trendHeld: DATA.current.trend_held,
  });
  drawDualPanel("chart-trailing", "caption-trailing", d.dates, d.strategy_equity, d.qqq_equity);
  const m = DATA.monthly;
  drawDualPanel("chart-full", "caption-full", m.dates, m.strategy_equity, m.qqq_equity,
    { log: true, stratDd: m.strategy_drawdown, qqqDd: m.qqq_drawdown });
  renderHeadlineTable();
  renderFooter();
}

function renderLive(qqq, tqqq) {
  const sig = computeStateSeries(qqq.dates, qqq.closes, DATA.params);
  sig.tqqqByDate = new Map(tqqq.dates.map((d, i) => [d, tqqq.closes[i]]));
  const last = sig.state.length - 1;

  renderStatus("live", sig.dates[last]);
  renderPosition({
    state: sig.state[last], exposure: sig.exposure[last], daysInState: daysInState(sig.state),
    live: true, qqqPrice: sig.price[last], tqqqPrice: tqqq.closes[tqqq.closes.length - 1],
  });
  renderStats({
    pctAboveSma: sig.pctAboveSma[last], vol20: sig.vol[last], trendHeld: sig.trendHeld[last],
  });

  const { stratRet, qqqRet } = buildLiveReturns(sig, qqq.closes);

  // Trailing 12 months: computed fresh from the live series, which at up to
  // ~500 days is self-sufficient (well past the 200-day SMA warmup) -- no
  // need to splice the baked window on for this chart.
  const cutoff = Date.parse(sig.dates[last] + "T00:00:00Z") - 365 * 86400000;
  let start = sig.dates.findIndex(d => Date.parse(d + "T00:00:00Z") >= cutoff);
  if (start < 0) start = 0;
  const trailDates = sig.dates.slice(start);
  const trailStratEq = rebase(cumprod(stratRet.slice(start)));
  const trailQqqEq = rebase(cumprod(qqqRet.slice(start)));
  drawDualPanel("chart-trailing", "caption-trailing", trailDates, trailStratEq, trailQqqEq);

  // Full history: baked monthly, extended by whatever live days fall after
  // the bake's own last date -- see buildLiveReturns' docstring-comment for
  // why the extension uses the same return definitions as the backtest.
  const m = DATA.monthly;
  const ext = extendWithLive(m, sig.dates, stratRet, qqqRet, DATA.generated_at);
  drawDualPanel("chart-full", "caption-full", ext.dates, ext.stratEq, ext.qqqEq,
    { log: true, stratDd: ext.stratDd, qqqDd: ext.qqqDd });

  renderHeadlineTable();
  renderFooter();
}

function cumprod(rets) {
  const out = new Array(rets.length);
  let v = 1.0;
  for (let i = 0; i < rets.length; i++) { v *= (1 + rets[i]); out[i] = v; }
  return out;
}

function daysInState(state) {
  const current = state[state.length - 1];
  let n = 0;
  for (let i = state.length - 1; i >= 0; i--) { if (state[i] !== current) break; n++; }
  return n;
}

function extendWithLive(monthly, liveDates, stratRet, qqqRet, generatedAt) {
  const dates = monthly.dates.slice(), stratEq = monthly.strategy_equity.slice(), qqqEq = monthly.qqq_equity.slice();
  let sScale = stratEq[stratEq.length - 1], qScale = qqqEq[qqqEq.length - 1];
  for (let j = 0; j < liveDates.length; j++) {
    if (liveDates[j] <= generatedAt) continue;
    sScale *= (1 + stratRet[j]); qScale *= (1 + qqqRet[j]);
    dates.push(liveDates[j]); stratEq.push(sScale); qqqEq.push(qScale);
  }
  const stratPriorPeak = Math.max(...monthly.strategy_equity);
  const qqqPriorPeak = Math.max(...monthly.qqq_equity);
  const nBaked = monthly.dates.length;
  const stratDd = monthly.strategy_drawdown.concat(
    drawdownFromEquity(stratEq.slice(nBaked), stratPriorPeak));
  const qqqDd = monthly.qqq_drawdown.concat(
    drawdownFromEquity(qqqEq.slice(nBaked), qqqPriorPeak));
  return { dates, stratEq, qqqEq, stratDd, qqqDd };
}

async function loadLiveOrFail() {
  const key = getApiKey();
  if (!key) return { ok: false, reason: "no_key", detail: "Add a Twelve Data API key to fetch live prices." };

  const today = todayLocalStr();
  const cachedQqq = loadCached("QQQ"), cachedTqqq = loadCached("TQQQ");
  const haveQqq = cachedQqq && cachedQqq.fetchedOn === today;
  const haveTqqq = cachedTqqq && cachedTqqq.fetchedOn === today;
  if (haveQqq && haveTqqq) return { ok: true, qqq: cachedQqq, tqqq: cachedTqqq };

  const [qqqResult, tqqqResult] = await Promise.all([
    haveQqq ? Promise.resolve(Object.assign({ ok: true }, cachedQqq)) : fetchSeries("QQQ", key),
    haveTqqq ? Promise.resolve(Object.assign({ ok: true }, cachedTqqq)) : fetchSeries("TQQQ", key),
  ]);
  if (!qqqResult.ok) return qqqResult;
  if (!tqqqResult.ok) return tqqqResult;
  if (!haveQqq) saveCached("QQQ", qqqResult);
  if (!haveTqqq) saveCached("TQQQ", tqqqResult);
  return { ok: true, qqq: qqqResult, tqqq: tqqqResult };
}

const FAILURE_TEXT = {
  no_key: "No API key saved yet.",
  bad_key: "The saved API key was rejected by Twelve Data.",
  rate_limited: "Twelve Data's daily rate limit has been reached.",
  timeout: "Twelve Data did not respond in time.",
  network: "Could not reach Twelve Data -- check the connection.",
  http_error: "Twelve Data returned an unexpected error.",
  malformed: "Twelve Data's response could not be read.",
  api_error: "Twelve Data returned an error.",
};

function wireControls() {
  // Wired once, unconditionally, before the (possibly slow, possibly
  // failing) live fetch even starts -- the key-entry form has to work
  // exactly in the cases where the fetch below does not.
  document.getElementById("key-save").onclick = () => {
    const v = document.getElementById("key-input").value;
    if (v.trim()) { setApiKey(v); location.reload(); }
  };
  document.getElementById("key-clear").onclick = () => { clearApiKey(); location.reload(); };
  const accountInput = document.getElementById("account-value");
  accountInput.value = getAccountValue() || "";
  accountInput.onchange = (ev) => {
    setAccountValue(parseFloat(ev.target.value) || 0);
    main();
  };
}

async function main() {
  wireControls();
  renderBaked();

  const result = await loadLiveOrFail();
  if (!result.ok) {
    // FAILURE_TEXT is the user-facing summary; api_error/http_error also
    // carry Twelve Data's own message in `detail`, which is worth surfacing
    // since it usually says exactly what to fix (e.g. an unknown symbol).
    const summary = FAILURE_TEXT[result.reason] || "Live fetch failed.";
    const extra = (result.reason === "api_error" || result.reason === "http_error") && result.detail
      ? " " + result.detail : "";
    renderStatus("stale", DATA.generated_at, summary + extra);
    return;
  }
  renderLive(result.qqq, result.tqqq);
}

main();
</script>

'''


def main() -> int:
    payload = load_and_build_payload()
    # allow_nan=False: a NaN or Infinity here is always a bug (an un-trimmed
    # warmup row, or a calmar computed on a driftless synthetic series slipping
    # through) and belongs on the generator's stderr, not silently embedded as
    # a bare `NaN` token in what is about to become a JS object literal.
    data_json = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    if PLACEHOLDER not in TEMPLATE:
        raise RuntimeError("template is missing its payload placeholder")
    html = TEMPLATE.replace(PLACEHOLDER, data_json)

    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(html):,} bytes, payload {len(data_json):,} bytes)")
    print(f"  generated_at={payload['generated_at']}  current_state={payload['current']['state']!r}")
    print(f"  strategy CAGR={payload['headline']['strategy']['cagr']:.4%}  "
          f"max_dd={payload['headline']['strategy']['max_dd']:.4%}  "
          f"switches/yr={payload['headline']['strategy']['switches_per_year']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
