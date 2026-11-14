"""Make ``dashboard/index.html`` a self-contained page that opens offline in any browser.

The hand-written page (its prose, styles, ``data-*`` bindings and chart code) sits
between ``<!-- page -->`` and ``<!-- /page -->``. :func:`build` keeps that part
verbatim and rewrites everything around it: the document head, theme tokens,
the vendored d3 (``dashboard/vendor/``), every dataset in ``dashboard/datasets.json``
inlined as JSON, and a small stand-in for the ``dash`` runtime the page calls.
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
.sa-bar { max-width: 1080px; margin: 0 auto 16px; display: flex; justify-content: flex-end; }
.sa-bar button { font: inherit; font-size: var(--cds-font-size-caption); color: var(--color-fg-muted); background: none;
  border: 1px solid var(--cds-border); border-radius: 6px; padding: 3px 10px; cursor: pointer; }
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


def build(page: Path = PAGE, manifest: Path = MANIFEST) -> Path:
    """Rewrite ``page`` in place around its hand-written part, with the current data inlined."""
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
        "</head>",
        "<body>",
        '<div class="sa-bar"><button type="button" id="sa-theme">Dark mode</button></div>',
        OPEN,
        src,
        CLOSE,
        f'<script id="dash-boot">{BOOT}</script>',
        "</body>",
        "</html>",
        "",
    ])
    page.write_text(html, encoding="utf-8")
    return page
