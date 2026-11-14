"""Make ``dashboard/index.html`` a self-contained page that opens offline in any browser.

The hand-written page (its prose, styles, ``data-*`` bindings and chart code) sits
between ``<!-- page -->`` and ``<!-- /page -->``. :func:`build` keeps that part
verbatim and rewrites everything around it: the document head, theme tokens,
the vendored d3 (``dashboard/vendor/``), every dataset in ``dashboard/datasets.json``
inlined as JSON, and a small stand-in for the ``dash`` runtime the page calls.
It also carries the "Refresh forecast" button (``REFRESH``; shown only on claude.ai).
No number is computed here; the values are the CSVs this package writes, copied
verbatim. ``python -m assetforecast.dashboard`` runs it after writing the data.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .. import paths

DASH = paths.ROOT / "dashboard"
PAGE = DASH / "index.html"
MANIFEST = DASH / "datasets.json"
# The "Refresh forecast" button: the scheduled task it starts (see refresh.json and REFRESH below).
REFRESH_CONFIG = DASH / "refresh.json"
D3 = DASH / "vendor" / "d3-7.9.0.min.js"
OPEN, CLOSE = "<!-- page -->", "<!-- /page -->"
NUM = re.compile(r"^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")

# Theme tokens the page reads (var(--...)). Colours follow the validated
# reference chart palette: the categorical series, surfaces, inks and status
# colours each have a light and a dark step.
TOKENS_COMMON = """
  --font-anthropic-sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-anthropic-serif: ui-serif, Georgia, Cambria, "Times New Roman", serif;
  --cds-font-size-caption: 0.8125rem; --cds-font-size-body: 0.9375rem;
  --cds-font-size-heading: 1.25rem; --cds-font-size-title: 2rem;
  --cds-font-weight-medium: 600;
  --cds-gap-xs: 4px; --cds-gap-sm: 8px; --cds-gap-md: 14px; --cds-gap-xl: 32px;
  --cds-pad-md: 12px; --cds-pad-lg: 16px; --cds-radius: 8px;
"""
TOKENS_LIGHT = """
  color-scheme: light;
  --color-bg: #f9f9f7; --color-panel: #fcfcfb; --color-fg: #0b0b0b; --color-fg-muted: #52514e;
  --color-border-line: #e1e0d9; --cds-border: #c3c2b7; --cds-text-secondary: #52514e;
  --cds-surface-popover: #fcfcfb; --cds-shadow-popover: 0 4px 16px rgba(0,0,0,.12);
  --cds-chart-muted: #898781; --cds-chart-grid: #e1e0d9; --cds-chart-axis: #c3c2b7; --cds-chart-reference: #52514e;
  --color-ok: #0a7d0a; --color-warn: #a86a00; --color-bad: #d03b3b;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
  --series-5: #e87ba4; --series-6: #008300; --series-7: #6250d6; --series-8: #e34948;
"""
TOKENS_DARK = """
  color-scheme: dark;
  --color-bg: #0d0d0d; --color-panel: #1a1a19; --color-fg: #f0efec; --color-fg-muted: #c3c2b7;
  --color-border-line: #2c2c2a; --cds-border: #383835; --cds-text-secondary: #c3c2b7;
  --cds-surface-popover: #1a1a19; --cds-shadow-popover: 0 4px 16px rgba(0,0,0,.5);
  --cds-chart-muted: #898781; --cds-chart-grid: #2c2c2a; --cds-chart-axis: #383835; --cds-chart-reference: #c3c2b7;
  --color-ok: #0ca30c; --color-warn: #fab219; --color-bad: #e66767;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  --series-5: #d55181; --series-6: #0ca30c; --series-7: #9085e9; --series-8: #e66767;
"""

BASE_CSS = """
:root {%s%s}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {%s} }
:root[data-theme="dark"] {%s}
*, *::before, *::after { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%%; }
body { margin: 0; background: var(--color-bg); color: var(--color-fg); padding: 32px 16px 64px; }
[hidden] { display: none !important; }
a { color: var(--series-1); }
.dash-select { font: inherit; font-size: var(--cds-font-size-caption); color: var(--color-fg); background: var(--color-panel);
  border: 1px solid var(--cds-border); border-radius: 6px; padding: 4px 8px; max-width: 100%%; }
details > summary { cursor: pointer; }
.sa-bar { font-family: var(--font-anthropic-sans); max-width: 1080px; margin: 0 auto 16px; display: flex; justify-content: flex-end; align-items: center; gap: 8px; flex-wrap: wrap; }
.sa-bar button { font: inherit; font-size: var(--cds-font-size-caption); color: var(--color-fg-muted); background: none;
  border: 1px solid var(--cds-border); border-radius: 6px; padding: 3px 10px; cursor: pointer; }
.sa-bar button:disabled { cursor: default; opacity: .6; }
.sa-refresh { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.sa-refresh-msg { font-family: var(--font-anthropic-sans); font-size: var(--cds-font-size-caption); color: var(--color-fg-muted); max-width: 60ch; text-align: right; }
.sa-refresh-msg.sa-bad { color: var(--color-bad); }
.sa-refresh button.sa-go { color: var(--color-fg); border-color: var(--color-fg-muted); }
""" % (TOKENS_COMMON, TOKENS_LIGHT, TOKENS_DARK, TOKENS_DARK)

# Stand-in for the hosted runtime's `dash` object: the subset the page uses.
SHIM = r"""
(() => {
  const RAW = JSON.parse(document.getElementById('dash-data').textContent);
  const store = {}, draws = new Map(), dataCbs = [];
  for (const [id, rows] of Object.entries(RAW)) store[id] = { status: 'ok', data: rows };
  const run = (name, fn) => document.querySelectorAll(`[data-draw="${CSS.escape(name)}"]`).forEach((el) => {
    try { fn(el); } catch (e) { console.error('draw ' + name, e); }
  });
  window.dash = {
    colors: [1, 2, 3, 4, 5, 6, 7, 8].map((i) => `var(--series-${i})`),
    data: (id) => store[id] || { status: 'error', data: null },
    draw(name, fn) { if (fn) draws.set(name, fn); const f = draws.get(name); if (f && dash._ready) run(name, f); },
    calc(id, spec) {
      try {
        const args = spec.inputs.map((i) => (store[i] && store[i].status === 'ok' ? store[i].data : null));
        store[id] = args.every((a) => a) ? { status: 'ok', data: spec.fn(...args) } : { status: 'error', data: null };
      } catch (e) { console.error('calc ' + id, e); store[id] = { status: 'error', data: null }; }
    },
    onData(fn) { dataCbs.push(fn); if (dash._ready) fn(); },
    _ready: false,
    _render() { draws.forEach((fn, name) => run(name, fn)); dataCbs.forEach((fn) => fn()); },
  };
})();
"""

BOOT = r"""
(() => {
  dash._ready = true; dash._render();
  let w = window.innerWidth, t = null;
  window.addEventListener('resize', () => {
    if (window.innerWidth === w) return; w = window.innerWidth;
    clearTimeout(t); t = setTimeout(() => dash._render(), 150);
  });
  const btn = document.getElementById('sa-theme'), root = document.documentElement;
  const dark = () => root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  const label = () => { btn.textContent = dark() ? 'Light mode' : 'Dark mode'; };
  try { const s = localStorage.getItem('sa-theme'); if (s) root.dataset.theme = s; } catch (e) {}
  label();
  btn.addEventListener('click', () => {
    root.dataset.theme = dark() ? 'light' : 'dark';
    try { localStorage.setItem('sa-theme', root.dataset.theme); } catch (e) {}
    label();
  });
})();
"""

# The "Refresh forecast" button. It appears only when the page is opened on claude.ai
# (where the `mcp` capability exists) and refresh.json names a scheduled task; opened
# as a local file the page shows no button. A click asks for confirmation, then fires
# the scheduled task through the viewer's "Claude Code Remote" connector; the task starts
# a cloud Claude session that downloads data, runs the pipeline and republishes the page.
REFRESH = r"""
(() => {
  const cfg = JSON.parse(document.getElementById('dash-refresh').textContent || 'null');
  const box = document.getElementById('sa-refresh');
  if (!cfg || !cfg.trigger_id || !box || !window.claude || typeof window.claude.use !== 'function') return;
  const S = cfg.connector || 'Claude Code Remote';
  const btn = document.createElement('button'), msg = document.createElement('span');
  btn.type = 'button'; btn.className = 'sa-go'; btn.textContent = 'Refresh forecast';
  msg.className = 'sa-refresh-msg'; msg.setAttribute('aria-live', 'polite');
  const say = (text, bad, link) => {
    msg.className = 'sa-refresh-msg' + (bad ? ' sa-bad' : '');
    msg.replaceChildren(document.createTextNode(text));
    if (link) { const a = document.createElement('a'); a.href = link; a.target = '_blank'; a.rel = 'noopener'; a.textContent = 'Open the session'; msg.append(' ', a, '.'); }
  };
  const when = (t) => { const d = t ? new Date(t) : null; return d && !isNaN(d) ? d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : ''; };
  const sessionUrl = (id) => (id ? 'https://claude.ai/code/' + encodeURIComponent(id) : null);
  const failCopy = (e) => {
    switch (e && e.code) {
      case 'server_not_connected': return 'Add the ' + S + ' connector in claude.ai Settings → Connectors, then try again.';
      case 'needs_reauth': return 'Reconnect the ' + S + ' connector in claude.ai Settings → Connectors, then try again.';
      case 'selection_required': return 'Choose which ' + S + ' connector this page should use, then try again.';
      case 'approval_required': case 'not_in_manifest': case 'consent_required': return 'Starting a refresh needs your approval for the ' + S + ' connector on this page.';
      case 'blocked_by_policy': return 'Your organization does not allow this page to start a refresh.';
      case 'tool_error': return 'The refresh could not be started: ' + (e.message || 'the scheduled task reported an error') + '.';
      case 'server_unavailable': case 'upstream_error': case 'cancelled': return 'No answer from claude.ai, so a refresh may or may not have started. Check “Last run” before trying again.';
      default: return 'The button is not available in this view.';
    }
  };
  let mcp = null, busy = false;
  const showLast = (p) => {
    const t = p && (p.trigger || p), r = t && t.last_run;
    if (!r) { say('No refresh has been run from this button yet.'); return null; }
    const done = !!r.finished_at, st = String(r.status || '').toLowerCase();
    if (!done) say('A refresh started ' + when(r.fired_at) + ' is still running (about an hour).', false, sessionUrl(r.session_id));
    else say('Last refresh ' + (st ? st.replace(/_/g, ' ') + ', ' : '') + 'finished ' + when(r.finished_at) + '.', /fail|error/.test(st), sessionUrl(r.session_id));
    return r;
  };
  const check = async () => {
    try { const res = await mcp.callTool(S, 'get_trigger', { trigger_id: cfg.trigger_id }, { cache: false }); return showLast(res.payload); }
    catch (e) { say(failCopy(e), true); return undefined; }
  };
  const last = document.createElement('button');
  last.type = 'button'; last.textContent = 'Last run';
  last.addEventListener('click', async () => { if (busy) return; busy = true; last.disabled = true; say('Checking…'); await check(); busy = false; last.disabled = false; });
  btn.addEventListener('click', async () => {
    if (busy) return;
    const ok = window.confirm('Start a refresh?\n\nA cloud Claude session will download the latest market data, rerun the pipeline (about an hour), commit the result to a new branch with a pull request, and update this page. It uses your Claude usage. If no new month of data has closed since the last forecast, it stops without changing anything.');
    if (!ok) return;
    busy = true; btn.disabled = true; last.disabled = true; say('Starting…');
    try {
      await mcp.callTool(S, 'fire_trigger', { trigger_id: cfg.trigger_id }, { cache: false });
      say('Refresh started. It takes about an hour; this page updates itself when it finishes. You will get a notification.');
    } catch (e) { say(failCopy(e), true); }
    busy = false; btn.disabled = false; last.disabled = false;
  });
  claude.use('mcp').then((m) => {
    if (!m) return;
    mcp = m; box.append(msg, last, btn); box.hidden = false;
  }, () => {});
})();
"""


def refresh_config(path: Path = REFRESH_CONFIG) -> dict | None:
    """The refresh button's settings (``trigger_id``, ``connector``), or None when there is no button."""
    if not path.exists():
        return None
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return cfg if cfg.get("trigger_id") else None


def read_rows(path: Path) -> list[dict]:
    """CSV rows with numeric cells as numbers and empty cells as null."""
    with path.open(newline="", encoding="utf-8") as fh:
        return [
            {k: (None if v == "" else float(v) if NUM.match(v) else v) for k, v in row.items()}
            for row in csv.DictReader(fh)
        ]


def page_source(html: str) -> str:
    """The hand-written part of the page (the whole file if it has no markers yet)."""
    if OPEN in html and CLOSE in html:
        return html.split(OPEN, 1)[1].split(CLOSE, 1)[0].strip("\n")
    return re.sub(r'^<meta name="dash-one-writer">\s*', "", html.strip("\n"))


def build(page: Path = PAGE, manifest: Path = MANIFEST, refresh: Path = REFRESH_CONFIG) -> Path:
    """Rewrite ``page`` in place around its hand-written part, with the current data inlined."""
    cfg = json.dumps(refresh_config(refresh)).replace("</", "<\\/")
    datasets = json.loads(manifest.read_text(encoding="utf-8"))
    data = {d["id"]: read_rows(manifest.parent / d["file"]) for d in datasets}
    src = page_source(page.read_text(encoding="utf-8"))
    title = re.search(r'<h1 id="title">(.*?)</h1>', src, re.S)
    desc = re.sub(r"\s+", " ", title.group(1)).strip().replace('"', "&quot;") if title else ""
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = "\n".join([
        "<!doctype html>",
        "<!-- Generated around the marked page section by python -m assetforecast.dashboard;",
        "     edit only between the page markers. -->",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Asset Class Forecast</title>",
        f'<meta name="description" content="{desc}">',
        f"<style>{BASE_CSS}</style>",
        f'<script id="dash-d3">{D3.read_text(encoding="utf-8")}</script>',
        f'<script type="application/json" id="dash-data">{payload}</script>',
        f'<script id="dash-runtime">{SHIM}</script>',
        f'<script type="application/json" id="dash-refresh">{cfg}</script>',
        "</head>",
        "<body>",
        '<div class="sa-bar"><div class="sa-refresh" id="sa-refresh" hidden></div>'
        '<button type="button" id="sa-theme">Dark mode</button></div>',
        OPEN,
        src,
        CLOSE,
        f'<script id="dash-boot">{BOOT}</script>',
        f'<script id="dash-refresh-ui">{REFRESH}</script>',
        "</body>",
        "</html>",
        "",
    ])
    page.write_text(html, encoding="utf-8")
    return page
