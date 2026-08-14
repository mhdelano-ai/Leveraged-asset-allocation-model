"""Generator for docs/dashboard.html -- the live operating dashboard.

    python scripts/build_dashboard.py

Reuses the existing model rather than reimplementing it: lam.data.panels for
the Nasdaq-100 panel, lam.alloc.rules for the four-state rule (improved
variant, RuleParams(up_wild=1.0)), and lam.report.dashboard for both the JSON
payload and the page shell it is spliced into.

There is deliberately almost nothing in this file. The payload shaping is in
lam.report.dashboard so tests can drive it against a synthetic fixture with no
network, and the page itself is templates/dashboard.html so 46KB of HTML, CSS
and JavaScript can be edited as HTML, CSS and JavaScript. What is left here is
the part that genuinely belongs to a script: fetch, write, and say what
happened.

The page is a single self-contained file: history is baked in at generation
time (it never changes, so there is no reason to recompute it on every page
load), and only the last day or two of prices is fetched live, client-side,
against a Twelve Data API key the *reader* supplies and that this script never
sees. Re-run this whenever the model, its parameters, or the baked history
changes; the live segment updates itself on every open.
"""

from __future__ import annotations

import sys
from pathlib import Path

from lam.report.dashboard import load_and_build_payload, render

OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "dashboard.html"


def main() -> int:
    payload = load_and_build_payload()
    html = render(payload)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")

    head = payload["headline"]["strategy"]
    print(f"wrote {OUT_PATH} ({len(html):,} bytes)")
    print(f"  generated_at={payload['generated_at']}  "
          f"current_state={payload['current']['state']!r}")
    print(f"  strategy CAGR={head['cagr']:.4%}  max_dd={head['max_dd']:.4%}  "
          f"switches/yr={head['switches_per_year']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
