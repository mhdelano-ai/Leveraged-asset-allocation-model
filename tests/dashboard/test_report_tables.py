"""The generated blocks committed in reports/REPORT.md and README.md match a fresh build."""
from __future__ import annotations

import importlib.util

import pytest

from assetforecast import paths

pytestmark = pytest.mark.skipif(not (paths.DASHBOARD_DATA / "verdicts.csv").exists(), reason="results not present")


def load():
    spec = importlib.util.spec_from_file_location("make_report_tables", paths.ROOT / "scripts" / "make_report_tables.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_committed_blocks_are_current():
    m = load()
    blocks = m.build_blocks()
    for path in m.TARGETS:
        text = path.read_text()
        assert m.MARK.search(text), path
        assert m.render(text, blocks) == text, f"{path.name} is stale: run python scripts/make_report_tables.py"


def test_unknown_block_is_an_error():
    m = load()
    with pytest.raises(KeyError):
        m.render("<!-- gen:nope -->x<!-- /gen:nope -->", {})
