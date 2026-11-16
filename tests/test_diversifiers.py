"""The diversifier study: bear markets, measures, verdicts, and the managed-futures loader."""
import numpy as np
import pandas as pd
import pytest

from assetforecast import diversifiers as dv
from assetforecast.data.sources import managed_futures as mfs


def _idx(n, start="1990-01-31"):
    return pd.date_range(start, periods=n, freq="ME")


def test_bear_markets_peak_to_trough():
    simple = [0.10, -0.10, -0.15, 0.05, -0.05, 0.30, 0.10, -0.25, 0.02]
    x = pd.Series(np.log1p(simple), index=_idx(len(simple)))
    ep = dv.bear_markets(x, 0.20)
    assert len(ep) == 2
    assert (ep["peak"][0], ep["trough"][0]) == (x.index[0], x.index[4])
    assert ep["stocks"][0] == pytest.approx(0.90 * 0.85 * 1.05 * 0.95 - 1)
    # the last fall never recovers 20%: closed at its low so far
    assert (ep["peak"][1], ep["trough"][1]) == (x.index[6], x.index[7])
    assert ep["stocks"][1] == pytest.approx(-0.25)


def test_episode_returns_need_full_coverage():
    x = pd.Series(np.log1p([0.10, -0.10, -0.15, 0.30]), index=_idx(4))
    c = pd.DataFrame({"a": [0.01, 0.02, 0.03, 0.0], "b": [np.nan, np.nan, 0.03, 0.0]}, index=x.index)
    ep = dv.episode_returns(x, c, 0.20)
    assert ep["a"][0] == pytest.approx(np.expm1(0.05)) and np.isnan(ep["b"][0])
    assert dv.bear_mask(x.index, ep).tolist() == [False, True, True, False]


def test_stats_recover_beta_alpha_and_capture():
    rng = np.random.default_rng(1)
    n = 480
    x = pd.Series(rng.normal(0.005, 0.045, n), index=_idx(n))
    y = np.log1p(0.5 * np.expm1(x) + 0.002 + rng.normal(0, 0.002, n))
    bear = pd.Series(x < -0.05, index=x.index)
    s = dv.stats(pd.Series(y, index=x.index), x, bear)
    assert s["n"] == n and s["corr_1m"] > 0.95
    assert s["beta"] == pytest.approx(0.5, abs=0.03)
    assert s["alpha"] == pytest.approx(0.024, abs=0.003) and s["alpha_t"] > 5
    assert s["capture"] == pytest.approx(0.5, abs=0.06)
    assert dv.verdict(s) == "stock-like"


@pytest.mark.parametrize("change, want", [
    ({}, "diversifier"), ({"alpha_t": 1.0}, "diversifier (unproven)"), ({"alpha": -0.01}, "diluter"),
    ({"roll_max": 0.6}, "unreliable"), ({"capture": 0.3}, "unreliable"), ({"corr_12m": 0.4}, "stock-like"),
    ({"n": 100}, "short"), ({"n": 10}, "")])
def test_verdict_rules(change, want):
    s = {"n": 400, "corr_1m": 0.1, "corr_12m": 0.1, "roll_max": 0.3, "capture": 0.0, "alpha": 0.02, "alpha_t": 2.0}
    assert dv.verdict({**s, **change}) == want


def test_measured_drag_is_the_gap_at_the_factor_volatility():
    rng = np.random.default_rng(2)
    t = pd.Series(rng.normal(0.01, 0.04, 240), index=_idx(240))
    d = dv.measured_drag(t, 0.5 * t - 0.001)          # half the risk, 1.2 points a year lost
    assert d["vol_ratio_funds_to_factor"] == pytest.approx(0.5)
    assert d["gap_at_fund_vol"] == pytest.approx(0.012) and d["drag"] == pytest.approx(0.024)


def test_mixes_cash_swap_scales_stocks():
    n = 120
    rng = np.random.default_rng(3)
    idx = _idx(n, "2007-03-31")
    x = pd.Series(rng.normal(0.005, 0.04, n), index=idx)
    cash = pd.Series(0.002, index=idx)
    cands = pd.DataFrame({"a": rng.normal(0.002, 0.02, n)}, index=idx)
    m = dv.mixes(x, cands, cash).query("window == 'since_2007'").set_index(["asset", "kind"])
    assert m.loc[("cash", "swap"), "vol"] == pytest.approx(0.8 * m.loc[("stocks only", "base"), "vol"], rel=0.03)
    assert m.loc[("a", "overlay"), "vol"] > m.loc[("a", "swap"), "vol"]
    assert set(m["n"]) == {n}


def test_month_returns_keep_complete_months_only():
    d = pd.bdate_range("2020-01-02", "2020-04-15")
    px = pd.Series(100.0, index=d)
    px.loc["2020-02-01":] = 110.0
    px.loc["2020-03-01":] = 99.0
    r = mfs.month_returns(px)
    assert list(r.index) == [pd.Timestamp("2020-02-29"), pd.Timestamp("2020-03-31")]   # April is incomplete
    assert r.tolist() == pytest.approx([0.10, -0.10])
    gap = px.drop(px.loc["2020-02-20":"2020-02-29"].index)                              # February's end is missing
    assert list(mfs.month_returns(gap).index) == []


def test_fund_composite_averages_the_funds_present():
    f = pd.DataFrame({"fund_A": [0.02, 0.04], "fund_B": [np.nan, 0.00]}, index=_idx(2))
    c = mfs.composite(f)
    assert c["funds_ret"].tolist() == pytest.approx([0.02, 0.02]) and c["n_funds"].tolist() == [1, 2]


@pytest.mark.realdata
def test_committed_tables_match_the_study():
    if not (dv.OUT / "summary.csv").exists() or not mfs.OUT.exists():
        pytest.skip("study not run")
    stocks, cands, _, _ = dv.load_inputs()
    got = dv.summary(stocks, cands).set_index(["window", "asset"])
    want = pd.read_csv(dv.OUT / "summary.csv").set_index(["window", "asset"])
    assert {"mf_tsmom", "mf_funds"} <= set(want.loc["own"].index)
    for c in ("corr_1m", "corr_12m", "capture", "alpha"):
        np.testing.assert_allclose(got[c], want[c], rtol=1e-4, atol=1e-6)
    assert (got["verdict"].fillna("") == want["verdict"].fillna("")).all()
