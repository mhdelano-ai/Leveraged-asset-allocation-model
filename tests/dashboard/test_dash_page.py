"""Every mark in dashboard/index.html names a dataset and column that exist and, where it selects
one row, selects exactly one; every verdict sentence group has a sentence for each status the data uses."""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from urllib.parse import unquote

import pandas as pd
import pytest

from assetforecast import paths
from assetforecast.dashboard import data as dd
from assetforecast.dashboard import page as page_mod

PAGE = paths.ROOT / "dashboard" / "index.html"
D = paths.DASHBOARD_DATA
pytestmark = pytest.mark.skipif(not (D / "verdicts.csv").exists(), reason="dashboard data not built")


class Marks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.els, self.stack = [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        a["_tag"], a["_parents"] = tag, [dict(p) for p in self.stack]
        self.els.append(a)
        if tag not in ("meta", "br", "img", "input", "option"):
            self.stack.append(a)

    def handle_endtag(self, tag):
        while self.stack:
            if self.stack.pop()["_tag"] == tag:
                break


@pytest.fixture(scope="module")
def page():
    html = page_mod.page_source(PAGE.read_text())
    p = Marks()
    p.feed(re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.S))
    return html, p.els


@pytest.fixture(scope="module")
def frames():
    man = json.loads(dd.MANIFEST.read_text())
    return {m["id"]: pd.read_csv(paths.ROOT / "dashboard" / m["file"], dtype=str, keep_default_na=False) for m in man}


def calcs(html):
    return set(re.findall(r"dash\.calc\('([\w-]+)'", html))


def test_page_is_self_contained_with_current_data(frames):
    html = PAGE.read_text()
    assert html.count(page_mod.OPEN) == 1 and html.count(page_mod.CLOSE) == 1
    assert not re.search(r"<(script|link)\b[^>]*\b(src|href)=", html), "the page must load nothing external"
    inlined = json.loads(re.search(r'<script type="application/json" id="dash-data">(.*?)</script>', html, re.S).group(1))
    assert set(inlined) == set(frames)
    for sid, df in frames.items():
        assert len(inlined[sid]) == len(df), sid


def test_every_source_exists(page, frames):
    html, els = page
    known = set(frames) | calcs(html)
    for e in els:
        for sid in e.get("data-source", "").split():
            assert sid in known, sid


def test_single_cell_marks_resolve_to_one_row(page, frames):
    _, els = page
    n = 0
    for e in els:
        if "data-field" not in e or e["_tag"] in ("th", "table") or "data-source" not in e:
            continue
        df = frames[e["data-source"].split()[0]]
        assert e["data-field"] in df.columns, (e["data-source"], e["data-field"])
        assert "data-where" in e or "data-row" in e, e
        if "data-where" in e:
            m = pd.Series(True, index=df.index)
            for kv in e["data-where"].split("&"):
                k, v = (unquote(x) for x in kv.split("="))
                assert k in df.columns, k
                m &= df[k] == v
            assert m.sum() == 1, (e["data-source"], e["data-where"], int(m.sum()))
            assert df.loc[m, e["data-field"]].iloc[0] != "", (e["data-source"], e["data-where"], e["data-field"])
        n += 1
    assert n > 100


def test_tables(page, frames):
    _, els = page
    tables = [e for e in els if e["_tag"] == "table" and "data-source" in e]
    assert len(tables) >= 8
    for t in tables:
        df = frames[t["data-source"]]
        for k in t["data-row-key"].split(","):
            assert k in df.columns
        assert not df.duplicated(t["data-row-key"].split(",")).any(), t["id"]
        ths = [e for e in els if e["_tag"] == "th" and any(p.get("id") == t["id"] for p in e["_parents"])]
        assert ths, t["id"]
        for th in ths:
            assert th["data-field"] in df.columns, (t["id"], th["data-field"])


def test_verdict_groups_cover_every_status(page, frames):
    _, els = page
    v = frames["verdicts"].set_index("question")
    groups = {}
    for e in els:
        if "data-when" in e:
            g = next(p for p in reversed(e["_parents"]) if "data-verdict" in p)
            groups.setdefault((g["data-verdict"], g.get("data-key", "status")), set()).add(e["data-when"])
            assert "hidden" in e, "alternatives are stored hidden; the page shows one"
    assert groups
    for (q, key), whens in groups.items():
        assert q in v.index, q
        assert v.loc[q, key] in whens, (q, key, v.loc[q, key], whens)


def test_replication_numbers_sit_next_to_the_label(page):
    """Each part of the page that shows replication figures also shows the replication label."""
    html = page[0]
    for sec in ("trust", "mean", "portfolio"):
        body = html.split(f'id="{sec}"', 1)[1].split("</section>", 1)[0]
        assert 'data-field="label"' in body, sec


def test_refresh_button_config(tmp_path):
    """The button's settings are inlined as JSON; without refresh.json the page carries null (no button)."""
    html = PAGE.read_text()
    cfg = json.loads(re.search(r'<script type="application/json" id="dash-refresh">(.*?)</script>', html, re.S).group(1))
    assert cfg == page_mod.refresh_config()
    assert 'id="sa-refresh" hidden' in html and 'id="dash-refresh-ui"' in html
    assert page_mod.refresh_config(tmp_path / "missing.json") is None
    (tmp_path / "r.json").write_text('{"trigger_id": ""}')
    assert page_mod.refresh_config(tmp_path / "r.json") is None
