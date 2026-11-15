"""assetforecast.data.sources.margin_rates: reading Robinhood's page, and the committed schedule."""
from __future__ import annotations

import json

import pytest

from assetforecast.data.sources import margin_rates as M

PAGE = """<html><head><style>p {color: red}</style><script>var x = "Margin balance";</script></head><body>
<h1>Robinhood margin rates</h1><p>The following interest rates are applied to your full margin balance:</p>
<table><tr><th>Margin balance</th><th>Margin interest rate</th></tr>
<tr><td>Up to $50,000</td><td>5.25%</td></tr><tr><td>$50,000 up to $100,000</td><td>5.05%</td></tr>
<tr><td>$100,000 up to $1 million</td><td>4.75%</td></tr><tr><td>$1 million up to $10 million</td><td>4.5%</td></tr>
<tr><td>$50+ million</td><td>4.20%</td></tr></table>
<h2>Check your settings</h2><p>$3,000 * (5.25% / 360) = $0.44 per day</p><p>$2,000 * (5.25% / 360) = $0.29 per day</p>
<p>The margin interest rates shown are as of September 17, 2026 and might change at any time.</p></body></html>"""


def test_parse_reads_the_schedule():
    with pytest.raises(ValueError, match="complete schedule"):      # $10 million to $50 million is missing
        M.parse(PAGE)
    page = PAGE.replace("<tr><td>$50+", "<tr><td>$10 million up to $50 million</td><td>4.45%</td></tr><tr><td>$50+")
    p = M.parse(page)
    assert (p["as_of"], p["day_count"]) == ("2026-09-17", 360)
    assert [(t["from"], t["below"], t["rate"]) for t in p["tiers"]] == [
        (0, 50e3, 0.0525), (50e3, 100e3, 0.0505), (100e3, 1e6, 0.0475), (1e6, 10e6, 0.045), (10e6, 50e6, 0.0445),
        (50e6, None, 0.042)]
    assert M.parse(page.replace("5.25%</td>", "4.9%</td>"))["tiers"][0]["rate"] == 0.049


@pytest.mark.parametrize("old, new, msg", [
    ("Margin interest rate</th>", "Rate</th>", "table"),
    ("as of September 17, 2026", "current", "as of"),
    ("/ 360) = $0.29", "/ 365) = $0.29", "day count"),
    ("<td>5.05%</td>", "<td>55%</td>", "complete schedule"),
])
def test_parse_refuses_a_page_it_cannot_read(old, new, msg):
    page = PAGE.replace("<tr><td>$50+", "<tr><td>$10 million up to $50 million</td><td>4.45%</td></tr><tr><td>$50+")
    assert old in page
    with pytest.raises(ValueError, match=msg):
        M.parse(page.replace(old, new))


@pytest.mark.realdata
def test_committed_schedule_is_complete():
    s = json.loads(M.OUT.read_text())
    assert s["source"] == M.URL and s["as_of"] <= s["retrieved"] and s["day_count"] in (360, 365)
    t = s["tiers"]
    assert t[0]["from"] == 0 and t[-1]["below"] is None
    assert all(a["below"] == b["from"] for a, b in zip(t, t[1:])) and all(0 < x["rate"] < 0.3 for x in t)
