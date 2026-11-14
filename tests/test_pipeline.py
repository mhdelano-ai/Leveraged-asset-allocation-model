"""The orchestration of ``python -m assetforecast.pipeline``, with every step stubbed."""
from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from assetforecast import pipeline as P


class Recorder:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at

    def __call__(self, cmd, cwd=None, env=None):
        self.calls.append((list(cmd), dict(env)))
        return SimpleNamespace(returncode=1 if self.fail_at and self.fail_at in " ".join(cmd) else 0)


def _modules(rec):
    return [c[c.index("-m") + 1] if "-m" in c else c[1] for c, _ in rec.calls]


@pytest.fixture
def full_store(monkeypatch):
    """Pretend results/full exists and is current."""
    import pandas as pd
    monkeypatch.setattr(P, "covariance_status", lambda path=P.PRIMARY_COV: {
        "exists": True, "first_origin": pd.Timestamp("1922-01-31"), "last_origin": pd.Timestamp("2026-10-31")})
    monkeypatch.setattr(P, "_last_complete_month", lambda: pd.Timestamp("2026-10-31"))


def test_order_and_commands_reuse_the_covariance_store(full_store, monkeypatch):
    monkeypatch.setenv("ASSETFORECAST_MAX_DATE", "2014-12-31")      # left in the shell: must be removed
    monkeypatch.setenv("ASSETFORECAST_DROP_TIERS", "D")
    monkeypatch.setenv("ASSETFORECAST_RESULTS", "/elsewhere")
    rec = Recorder()
    recs = P.run(P.Config(jobs=2), runner=rec, log=lambda *a: None)
    assert [r["step"] for r in recs] == list(P.STEP_NAMES)
    assert _modules(rec)[:6] == ["assetforecast.covhistory", "assetforecast.returns.features.build",
                                 "assetforecast.returns.run", "assetforecast.returns.distribution",
                                 "assetforecast.returns.evaluation.strategy", "assetforecast.forecast"]
    assert not recs[0]["ran"]                                          # results/full reused
    cmds = {m: c for m, (c, _) in zip(_modules(rec), rec.calls)}
    assert "--score" in cmds["assetforecast.returns.distribution"]
    assert cmds["assetforecast.returns.run"][-2:] == ["--jobs", "2"]
    assert cmds["assetforecast.forecast"][-2:] == ["--results", str(P.RESULTS_FULL)]
    for cmd, env in rec.calls:
        assert "--max-date" not in cmd
        assert "ASSETFORECAST_MAX_DATE" not in env
        assert "ASSETFORECAST_DROP_TIERS" not in env and "ASSETFORECAST_RESULTS" not in env
        assert env["OMP_NUM_THREADS"] == "1"


def test_max_date_reaches_every_step(full_store):
    rec = Recorder()
    recs = P.run(P.Config(max_date="2014-12-31"), runner=rec, log=lambda *a: None)
    assert recs[-1]["step"] == "dashboard" and not recs[-1]["ran"]   # never under --max-date
    assert len(rec.calls) == 6
    for cmd, env in rec.calls:
        assert env["ASSETFORECAST_MAX_DATE"] == "2014-12-31"
        i = cmd.index("--max-date")
        assert cmd[i + 1] == "2014-12-31"


def test_skip_rebuild_and_failure(full_store):
    rec = Recorder()
    P.run(P.Config(rebuild_covariance=True), skip=("strategy", "dashboard"), runner=rec, log=lambda *a: None)
    assert rec.calls[0][0][:2] == ["bash", str(P.RUN_ALL)]
    assert "assetforecast.returns.evaluation.strategy" not in _modules(rec)
    rec = Recorder(fail_at="assetforecast.returns.run")
    with pytest.raises(SystemExit, match="replication"):
        P.run(P.Config(), runner=rec, log=lambda *a: None)
    assert _modules(rec)[-1] == "assetforecast.returns.run"           # nothing after the failure
    with pytest.raises(ValueError, match="unknown steps"):
        P.run(P.Config(), skip=("nope",), runner=rec)


def test_missing_covariance_store_is_built(monkeypatch):
    monkeypatch.setattr(P, "covariance_status", lambda path=P.PRIMARY_COV: {"exists": False})
    cmd = P.STEPS[0].command(P.Config(jobs=4))
    assert cmd == ["bash", str(P.RUN_ALL), "4"]
    assert "missing" in P.STEPS[0].note(P.Config())


def test_stale_covariance_store_warns(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(P, "covariance_status", lambda path=P.PRIMARY_COV: {
        "exists": True, "first_origin": pd.Timestamp("1922-01-31"), "last_origin": pd.Timestamp("2026-09-30")})
    monkeypatch.setattr(P, "_last_complete_month", lambda: pd.Timestamp("2026-10-31"))
    assert "WARNING" in P.STEPS[0].note(P.Config())
    assert "WARNING" not in P.STEPS[0].note(P.Config(max_date="2014-12-31"))


def test_dashboard_step_runs_when_the_module_exists(full_store, monkeypatch):
    real = P.importlib.util.find_spec
    monkeypatch.setattr(P.importlib.util, "find_spec",
                        lambda name: object() if name == "assetforecast.dashboard" else real(name))
    rec = Recorder()
    P.run(P.Config(), runner=rec, log=lambda *a: None)
    assert _modules(rec)[-1] == "assetforecast.dashboard"


def test_step_processes_see_the_development_switch(full_store, monkeypatch, tmp_path):
    """Real subprocesses with the environment the pipeline builds: ``assetforecast.paths``
    in each step process is set up for the run (results/dev under --max-date)."""
    probe = ("import json, os, sys; from assetforecast import paths; "
             "json.dump({'env': os.environ.get('ASSETFORECAST_MAX_DATE'), 'dev': paths._DEV, "
             "'cov': str(paths.COV_HISTORY), 'ret': str(paths.RESULTS_RETURNS), "
             "'feat': str(paths.FEATURES)}, open(sys.argv[1], 'w'))")

    def runner(cmd, cwd=None, env=None):
        out = tmp_path / f"{len(list(tmp_path.iterdir()))}.json"
        return subprocess.run([sys.executable, "-c", probe, str(out)], cwd=cwd, env=env)

    monkeypatch.setenv("ASSETFORECAST_MAX_DATE", "1999-12-31")
    for md, dev in (("2014-12-31", True), (None, False)):
        for f in tmp_path.iterdir():
            f.unlink()
        P.run(P.Config(max_date=md), skip=("dashboard",), runner=runner, log=lambda *a: None)
        seen = [json.loads(f.read_text()) for f in tmp_path.iterdir()]
        assert len(seen) == 6
        for s in seen:
            assert s["env"] == md and s["dev"] is dev
            assert ("/results/dev/" in s["cov"]) is dev and ("/results/dev/" in s["ret"]) is dev
            assert ("/results/dev/" in s["feat"]) is dev
