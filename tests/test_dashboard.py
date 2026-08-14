"""Tests for the live dashboard: the payload builder, and the test that
matters most -- that the page's own JS state machine agrees with
``lam.alloc.rules.four_state_signal`` on the same data.

The payload tests use a small synthetic fixture and never touch the network or
the on-disk parquet cache (see ``lam.report.dashboard``'s module docstring for
why that split exists). The JS/Python agreement test and the static-page
checks below it drive the actual generated ``docs/dashboard.html`` through
Playwright and are skipped -- not failed -- when Playwright or its bundled
Chromium is unavailable, so a clean checkout without a browser installed still
passes `pytest -q` in full.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lam.alloc.rules import RuleParams, backtest_four_state, four_state_signal
from lam.report.dashboard import _days_in_current_state, build_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_HTML = REPO_ROOT / "docs" / "dashboard.html"
PW_CHROMIUM = Path("/opt/pw-browsers/chromium")


def _gbm(n: int = 3000, mu: float = 0.15, sigma: float = 0.30, seed: int = 0) -> pd.Series:
    """Daily lognormal returns -- enough history for the 200-day SMA to warm
    up and leave several years of monthly and trailing-window data besides."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n)
    dt = 1.0 / 252.0
    logs = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.normal(size=n)
    return pd.Series(np.expm1(logs), index=idx, name="asset")


def _fixture(n: int = 3000, seed: int = 0):
    """A no-dividend world: price_basis is exactly the compounded total-return
    curve, so this exercises the payload shape without also exercising the
    price-basis divergence -- that is covered on its own in
    ``tests/test_growth.py`` (``test_price_basis_changes_the_trend_reading...``).
    """
    index_tr = _gbm(n, seed=seed)
    rf = pd.Series(0.03, index=index_tr.index)
    price_basis = (1.0 + index_tr.fillna(0.0)).cumprod()
    return index_tr, rf, price_basis


# --------------------------------------------------------------------------
# Payload shape (pure, synthetic, no I/O)
# --------------------------------------------------------------------------

def test_payload_is_valid_json_with_no_nan_or_inf():
    index_tr, rf, price_basis = _fixture()
    payload = build_payload(index_tr, rf, price_basis, params=RuleParams(up_wild=1.0))
    # allow_nan=False raises on NaN/Infinity. This is a *stricter* check than
    # the page itself needs -- DATA is embedded as a JS object literal, not
    # parsed from a string, so a bare NaN would actually still run -- but a
    # NaN in this payload is always a bug (an un-trimmed warmup row), and this
    # is where it should be caught, not in the browser.
    json.dumps(payload, allow_nan=False)


def test_monthly_series_share_one_date_axis_and_span_the_input():
    index_tr, rf, price_basis = _fixture()
    payload = build_payload(index_tr, rf, price_basis)
    m = payload["monthly"]
    assert len(m["dates"]) == len(m["strategy_equity"]) == len(m["strategy_drawdown"])
    assert len(m["dates"]) == len(m["qqq_equity"]) == len(m["qqq_drawdown"])
    assert m["dates"] == sorted(m["dates"])
    assert m["dates"][0][:7] in {"2005-01", "2005-02"}
    # drawdown_series is <= 0 everywhere by construction.
    assert all(v <= 1e-9 for v in m["strategy_drawdown"])
    assert all(v <= 1e-9 for v in m["qqq_drawdown"])
    # And equity must be on the whole series' own $1-at-inception scale, not
    # pre-rebased the way the daily block deliberately is: the first monthly
    # point should equal the plain compounded return through that month-end.
    first_month_end = pd.Timestamp(m["dates"][0] + "-01") + pd.offsets.MonthEnd(0)
    expected = float((1.0 + index_tr.loc[:first_month_end]).prod())
    assert m["qqq_equity"][0] == pytest.approx(expected, rel=1e-4)


def test_daily_trailing_window_is_rebased_and_ends_at_generated_at():
    index_tr, rf, price_basis = _fixture()
    payload = build_payload(index_tr, rf, price_basis, trailing_months=18)
    d = payload["daily"]
    lengths = {len(d[k]) for k in
               ("dates", "strategy_equity", "qqq_equity", "pct_above_sma", "vol20", "state", "exposure")}
    assert len(lengths) == 1, f"daily block arrays have mismatched lengths: {lengths}"

    assert d["strategy_equity"][0] == pytest.approx(1.0)
    assert d["qqq_equity"][0] == pytest.approx(1.0)
    assert d["dates"] == sorted(d["dates"])
    assert d["dates"][-1] == payload["generated_at"]

    span_days = (pd.Timestamp(d["dates"][-1]) - pd.Timestamp(d["dates"][0])).days
    assert 500 <= span_days <= 580  # ~18 months, with slack for weekends/holidays


def test_headline_stats_present_and_finite():
    index_tr, rf, price_basis = _fixture()
    payload = build_payload(index_tr, rf, price_basis)
    for series in ("strategy", "qqq"):
        stats = payload["headline"][series]
        for key in ("cagr", "vol", "sharpe", "max_dd"):
            assert np.isfinite(stats[key]), f"{series}.{key} is not finite: {stats[key]}"
        assert stats["max_dd"] <= 0
    assert payload["headline"]["strategy"]["switches_per_year"] > 0
    assert "switches_per_year" not in payload["headline"]["qqq"]


def test_params_echo_the_improved_variant_by_default():
    index_tr, rf, price_basis = _fixture()
    payload = build_payload(index_tr, rf, price_basis)
    assert payload["params"]["up_wild"] == 1.0
    assert payload["params"]["sma_band"] == pytest.approx(0.01)
    assert payload["params"]["sma_window"] == 200


def test_current_block_matches_the_last_row_of_the_signal():
    index_tr, rf, price_basis = _fixture()
    params = RuleParams(up_wild=1.0)
    payload = build_payload(index_tr, rf, price_basis, params=params)
    _, sig = backtest_four_state(index_tr, rf, params, price_basis=price_basis)

    assert payload["current"]["state"] == sig["state"].iloc[-1]
    assert payload["current"]["exposure"] == pytest.approx(float(sig["exposure"].iloc[-1]))
    assert payload["current"]["date"] == str(sig.index[-1].date())


def test_days_in_current_state_counts_the_trailing_run():
    assert _days_in_current_state(pd.Series(["a", "b", "b", "c", "c", "c"])) == 3
    assert _days_in_current_state(pd.Series(["x"])) == 1
    assert _days_in_current_state(pd.Series(["a", "a", "a"])) == 3


@pytest.mark.network
def test_load_and_build_payload_smoke():
    """Integration smoke test for the generator's actual entry point.

    Skipped, not failed, when the parquet cache under data/cache/ (gitignored)
    isn't populated -- see the module docstring for why every other test here
    uses a synthetic fixture instead of this function.
    """
    from lam.report.dashboard import load_and_build_payload

    try:
        payload = load_and_build_payload()
    except Exception as exc:  # noqa: BLE001 - any I/O failure means "skip", not "fail"
        pytest.skip(f"load_and_build_payload needs the data cache/network: {exc}")
    json.dumps(payload, allow_nan=False)
    assert payload["panel"] == "N"
    assert payload["params"]["up_wild"] == 1.0


# --------------------------------------------------------------------------
# Playwright: the page against the real generated docs/dashboard.html
# --------------------------------------------------------------------------

def _launch_chromium():
    """A launched (playwright, browser) pair, or (None, None) if unavailable.

    Tries the environment's pre-provisioned Chromium first (fast, no download),
    then falls back to whatever Playwright would normally manage, so this
    degrades gracefully on a machine set up differently rather than only ever
    working in one specific sandbox.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None, None

    pw = sync_playwright().start()
    attempts = []
    if PW_CHROMIUM.exists():
        attempts.append({"executable_path": str(PW_CHROMIUM)})
    attempts.append({})
    for kwargs in attempts:
        try:
            return pw, pw.chromium.launch(**kwargs)
        except Exception:  # noqa: BLE001 - try the next option
            continue
    pw.stop()
    return None, None


def _agreement_fixture() -> pd.Series:
    """A ~965-day raw-close series exercising all four states, the single
    "warmup" row, and -- critically -- the sma_band hysteresis in both
    directions.

    The middle 16 days are hand-placed at known offsets from the SMA the
    preceding calm run leaves behind, alternating inside +/-1% (must hold the
    prior reading) and decisively outside it (must flip) on both sides --
    the same construction as
    test_growth.py::test_sma_band_holds_the_previous_state_inside_the_band.
    Built once, in Python; the exact same numbers go to both sides of the
    comparison in test_js_state_machine_matches_python.
    """
    rng = np.random.default_rng(2026)
    r1 = np.full(260, 0.0009)                    # calm uptrend -- warms up the SMA
    r3 = rng.normal(0.0060, 0.035, 230)           # noisy uptrend -- too wild to lever
    r4 = rng.normal(-0.0012, 0.006, 230)          # calm decline
    r5 = rng.normal(-0.0030, 0.035, 230)          # violent decline

    idx1 = pd.bdate_range("2000-01-03", periods=len(r1))
    price1 = (1.0 + pd.Series(r1, index=idx1)).cumprod() * 100.0
    sma_end = price1.rolling(200, min_periods=200).mean().iloc[-1]

    offsets = [0.005, -0.003, 0.012, -0.006, 0.002, -0.015, 0.006, -0.002,
               0.020, -0.030, -0.008, 0.040, 0.008, -0.002, 0.001, 0.060]
    idx2 = pd.bdate_range(idx1[-1] + pd.Timedelta(days=1), periods=len(offsets))
    price2 = pd.Series([sma_end * (1.0 + o) for o in offsets], index=idx2)

    idx3 = pd.bdate_range(idx2[-1] + pd.Timedelta(days=1), periods=len(r3))
    idx4 = pd.bdate_range(idx3[-1] + pd.Timedelta(days=1), periods=len(r4))
    idx5 = pd.bdate_range(idx4[-1] + pd.Timedelta(days=1), periods=len(r5))
    phase3 = (1.0 + pd.Series(r3, index=idx3)).cumprod() * price2.iloc[-1]
    phase4 = (1.0 + pd.Series(r4, index=idx4)).cumprod() * phase3.iloc[-1]
    phase5 = (1.0 + pd.Series(r5, index=idx5)).cumprod() * phase4.iloc[-1]

    day0 = pd.Series([100.0], index=[idx1[0] - pd.Timedelta(days=1)])
    closes = pd.concat([day0, price1, price2, phase3, phase4, phase5])
    return closes[~closes.index.duplicated(keep="first")].sort_index()


def test_js_state_machine_matches_python():
    """The test that matters most: feed one fixed price series through the
    page's own ``computeStateSeries`` (via Playwright) and through
    ``four_state_signal``, and assert identical states and exposures across
    the whole series -- band hysteresis included.

    This does not reimplement the JS logic in Python to compare against --
    that would only prove the reimplementation agrees with itself. It drives
    the exact function docs/dashboard.html calls at render time, inside the
    real generated page.

    Capable of failing: confirmed during development by temporarily setting
    the JS side's default `sma_band` to 0.02 (mismatched against Python's
    0.01) and observing this test fail on the choppy-window rows, then
    reverting -- see the session notes for the exact before/after.
    """
    if not DASHBOARD_HTML.exists():
        pytest.skip("docs/dashboard.html has not been generated -- run scripts/build_dashboard.py")
    pw, browser = _launch_chromium()
    if browser is None:
        pytest.skip("Playwright/Chromium is not available in this environment")

    closes = _agreement_fixture()
    dates = [d.strftime("%Y-%m-%d") for d in closes.index]
    values = [float(v) for v in closes.to_numpy()]

    try:
        page = browser.new_page()
        page.goto(DASHBOARD_HTML.as_uri())
        js_result = page.evaluate(
            "([dates, closes]) => computeStateSeries(dates, closes, {up_wild: 1.0})",
            [dates, values],
        )
        page.close()
    finally:
        browser.close()
        pw.stop()

    index_tr = closes.pct_change().dropna()
    price_basis = closes.reindex(index_tr.index)
    params = RuleParams(up_wild=1.0)
    sig = four_state_signal(index_tr, params, price_basis=price_basis)

    py_dates = [d.strftime("%Y-%m-%d") for d in sig.index]
    assert js_result["dates"] == py_dates, "date alignment differs between JS and Python"
    assert js_result["state"] == sig["state"].tolist(), "state sequence differs between JS and Python"
    np.testing.assert_allclose(
        np.asarray(js_result["exposure"], dtype=float),
        sig["exposure"].to_numpy(),
        rtol=0, atol=1e-9,
        err_msg="exposure sequence differs between JS and Python",
    )

    # The fixture must actually exercise the band's hold-previous-state path
    # -- otherwise the assertions above pass no matter what the JS band logic
    # does, which is exactly the "trivially passing" failure mode this test
    # has to avoid.
    unbanded = four_state_signal(index_tr, RuleParams(up_wild=1.0, sma_band=0.0), price_basis=price_basis)
    choppy_window = price_basis.index[260:276]  # the 16 hand-placed days
    diverged = (sig.loc[choppy_window, "state"] != unbanded.loc[choppy_window, "state"]).sum()
    assert diverged >= 3, (
        f"only {diverged} of 16 choppy-window days differ between banded and unbanded -- "
        "the fixture is not exercising sma_band, so agreement there would be a false positive"
    )


def test_dashboard_page_has_no_console_errors_or_overflow():
    """Static-page health check across both themes and phone/desktop widths.

    No API key is stored in this fresh browser context, so the page takes its
    "no key" path and never attempts a network fetch -- this is purely about
    the page rendering baked data cleanly, not about live-fetch behaviour.
    """
    if not DASHBOARD_HTML.exists():
        pytest.skip("docs/dashboard.html has not been generated -- run scripts/build_dashboard.py")
    pw, browser = _launch_chromium()
    if browser is None:
        pytest.skip("Playwright/Chromium is not available in this environment")

    failures = []
    try:
        for width in (390, 1400):
            for scheme in ("light", "dark"):
                page = browser.new_page(viewport={"width": width, "height": 1000}, color_scheme=scheme)
                errors = []
                page.on("pageerror", lambda exc: errors.append(str(exc)))
                page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
                page.goto(DASHBOARD_HTML.as_uri())
                page.wait_for_timeout(700)
                overflow = page.evaluate(
                    "document.documentElement.scrollWidth > document.documentElement.clientWidth"
                )
                page.close()
                if errors:
                    failures.append(f"{width}px {scheme}: console errors {errors!r}")
                if overflow:
                    failures.append(f"{width}px {scheme}: horizontal overflow")
    finally:
        browser.close()
        pw.stop()

    assert not failures, "; ".join(failures)
