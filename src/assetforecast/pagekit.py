"""Pieces the study pages and reports share: ``allocation_page`` and ``questionnaire_page``
(styles, browser helpers, inline JSON, the document wrapper, the command line), ``horizons``
(the markdown table, the document wrapper) and ``diversifiers`` (number formats, the markdown
table). Formatting only; nothing here computes a number."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TOKENS = """
:root {
  /* layout: one reading column; three answers on top, then evidence, the replay with its controls, and today's menu */
  --bg: #f9f9f7; --panel: #fcfcfb; --fg: #0b0b0b; --muted: #52514e; --line: #e1e0d9; --border: #c3c2b7;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --base: #8a897f;
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --serif: ui-serif, Georgia, Cambria, "Times New Roman", serif;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  color-scheme: dark; --bg: #0d0d0d; --panel: #1a1a19; --fg: #f0efec; --muted: #c3c2b7; --line: #2c2c2a;
  --border: #383835; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --base: #8a897f; } }
:root[data-theme="dark"] {
  color-scheme: dark; --bg: #0d0d0d; --panel: #1a1a19; --fg: #f0efec; --muted: #c3c2b7; --line: #2c2c2a;
  --border: #383835; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --base: #8a897f; }
"""

CSS = """
*, *::before, *::after { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font-family: var(--sans); font-size: 15px;
  line-height: 1.5; padding-block: 32px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main { max-width: 1080px; margin: 0 auto; display: flex; flex-direction: column; gap: 36px; }
section { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
h1 { font-family: var(--serif); font-size: 2rem; line-height: 1.15; margin: 0; font-weight: 600; text-wrap: balance; }
h2 { font-size: 1.25rem; margin: 0; font-weight: 600; text-wrap: balance; }
h3 { font-size: .9375rem; margin: 0; font-weight: 600; }
p { margin: 0; max-width: 68ch; }
.muted, .note { color: var(--muted); font-size: .8125rem; }
.label { border-left: 3px solid var(--border); padding: 2px 0 2px 12px; color: var(--muted); font-size: .8125rem; max-width: 80ch; }
.answers { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px 28px; }
.answers > div { min-width: 0; border-top: 2px solid var(--fg); padding-top: 10px; display: flex; flex-direction: column; gap: 6px; }
.answers .q { font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
.answers .verdict { font-family: var(--serif); font-size: 1.25rem; line-height: 1.2; }
.answers p { font-size: .875rem; color: var(--muted); }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px 32px; }
.two > div { min-width: 0; display: flex; flex-direction: column; gap: 8px; }
.controls { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: end; }
.controls label { display: flex; flex-direction: column; gap: 3px; font-size: .75rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: .04em; }
select { font: inherit; font-size: .875rem; color: var(--fg); background: var(--panel); border: 1px solid var(--border);
  border-radius: 6px; padding: 5px 8px; max-width: 100%; }
select:focus-visible, [tabindex]:focus-visible { outline: 2px solid var(--s1); outline-offset: 1px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: .8125rem; width: 100%; }
th { font-weight: 500; color: var(--muted); text-align: right; padding: 5px 8px; white-space: nowrap; border-bottom: 1px solid var(--border); }
td { text-align: right; padding: 5px 8px; white-space: nowrap; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; padding-left: 0; }
tr.base td { font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: center; font-size: .8125rem; color: var(--muted); }
.sw { display: inline-block; width: 11px; height: 11px; border-radius: 2px; vertical-align: -1px; margin-right: 6px; }
.ln { display: inline-block; width: 16px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--muted); font-size: 11px; font-family: var(--sans); font-variant-numeric: tabular-nums; }
svg .lab { fill: var(--fg); font-size: 12px; }
svg .grid { stroke: var(--line); stroke-width: 1; }
svg .zero { stroke: var(--border); stroke-width: 1; }
#tip { position: fixed; pointer-events: none; background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  padding: 7px 10px; font-size: .8125rem; box-shadow: 0 4px 16px rgba(0,0,0,.18); max-width: 280px; z-index: 5;
  font-variant-numeric: tabular-nums; }
#tip b { display: block; }
ul { margin: 0; padding-left: 20px; max-width: 72ch; }
li { margin-bottom: 4px; }
code { font-size: .875em; }
"""

# Browser helpers, spliced into each page's script: an SVG element, and the #tip
# tooltip shown on hover and on keyboard focus.
EL_JS = r"""  const el = (tag, attrs = {}, text) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  };
"""

TIP_JS = r"""  const tip = $('tip');
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
"""


def pct(v, d=1, sign=True) -> str:
    """A fraction as a percent, '+' on positives unless ``sign`` is false; blank when missing."""
    return "" if v is None or not np.isfinite(v) else f"{100 * v:{'+' if sign else ''}.{d}f}%"


def num(v, d=2, sign=True) -> str:
    """A number to ``d`` places, '+' on positives unless ``sign`` is false; blank when missing."""
    return "" if v is None or not np.isfinite(v) else f"{v:{'+' if sign else ''}.{d}f}"


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", double_precision=5))


def md_table(head, rows) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])


def inline_json(obj) -> str:
    """``obj`` as compact JSON that is safe inside a ``<script>`` element."""
    return json.dumps(obj, separators=(",", ":"), allow_nan=False, default=lambda o: None).replace("</", "<\\/")


def document(body: str, head: str = "", viewport: str = "width=device-width, initial-scale=1, viewport-fit=cover") -> str:
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="{viewport}">\n{head}</head>\n<body>\n{body}\n</body>\n</html>\n')


def write_page(page: Path, frag: str, fragment_path: Path | None = None) -> str:
    """Write ``frag`` as the whole document ``page``, and bare to ``fragment_path`` if given."""
    page.write_text(document(frag))
    if fragment_path is not None:
        Path(fragment_path).write_text(frag)
    return str(page)


def main(doc: str, prog: str, load, write_report, write_page, argv=None) -> int:
    """``prog [--fragment PATH]``: write the report and the page from one load of the tables."""
    ap = argparse.ArgumentParser(prog=prog, description=doc.split("\n")[0])
    ap.add_argument("--fragment", type=Path, default=None)
    a = ap.parse_args(argv)
    d = load()
    print(write_report(d), write_page(d, a.fragment))
    return 0
