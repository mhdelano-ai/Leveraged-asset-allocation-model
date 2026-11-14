"""Tests for the ``tips`` loader: offline logic, processed files (realdata), downloads (network)."""
import numpy as np
import pandas as pd
import pytest

from assetforecast.data.sources import tips as T
from assetforecast.paths import ASSET_DIR, PREDICTOR_DIR

DAILY = ASSET_DIR / "tips_daily.parquet"
MONTHLY = ASSET_DIR / "tips_monthly.parquet"
PRED = PREDICTOR_DIR / "tips_real_yields.parquet"


# ------------------------------------------------------------------ offline logic
def _fund_frame():
    idx = pd.bdate_range("2001-01-02", "2001-03-30")
    rng = np.random.default_rng(0)
    close = 10 + np.cumsum(rng.normal(0, 0.01, len(idx)))
    df = pd.DataFrame({"close": close, "adj": np.nan, "div": 0.0}, index=idx)
    df.loc["2001-01-31", "div"] = 0.04          # month-end -> daily accrual
    df.loc["2001-02-28", "div"] = 0.05
    df.loc["2001-03-15", "div"] = 0.20          # mid-month -> ex-date
    df.loc["2001-03-15":, "close"] -= 0.20       # NAV drops on the ex-date
    return df


def test_accrual_preserves_monthly_income_and_removes_spike():
    df = _fund_frame()
    r, cls = T.accrual_total_return(df)
    assert set(cls["treated_as"]) == {"daily_accrual", "ex_date"}
    assert cls.loc["2001-03-15", "treated_as"] == "ex_date"
    # month-end day carries only its share of the income, not the whole distribution
    pr = df["close"].pct_change()
    assert abs(r["2001-01-31"] - pr["2001-01-31"]) < 0.04 / 10 / 5
    # the month's income is fully booked within the month (simple-sum check)
    feb = (r - pr).loc["2001-02"]
    inc = (0.05 / df["close"].shift(1)).loc["2001-02"].mean()
    assert feb.sum() == pytest.approx(inc, rel=0.02)
    # ex-date distribution offsets the NAV drop
    assert abs(r["2001-03-15"]) < 0.03


def test_reference_cpi_interpolation():
    cpi = pd.Series([100.0, 101.0, 102.0, 103.0], index=pd.period_range("2000-01", periods=4, freq="M"))
    d = pd.DatetimeIndex(["2000-04-01", "2000-04-16"])
    rc = T.reference_cpi(d, cpi)
    assert rc.iloc[0] == pytest.approx(100.0)                 # CPI(M-3) on day 1
    assert rc.iloc[1] == pytest.approx(100.0 + 15 / 30)      # halfway to CPI(M-2)


def test_gsw_zero_yield_short_and_long_limits():
    g = pd.DataFrame({"BETA0": [2.0], "BETA1": [-1.0], "BETA2": [0.0], "BETA3": [np.nan],
                      "TAU1": [1.0], "TAU2": [np.nan]})
    assert T.gsw_zero_yield(g, 1e-6).iloc[0] == pytest.approx(0.01, abs=1e-6)
    assert T.gsw_zero_yield(g, 1e4).iloc[0] == pytest.approx(0.02, abs=1e-4)


def test_build_monthly_compounds_and_drops_partial_months():
    idx = pd.bdate_range("2001-01-29", "2001-04-10")
    d = pd.DataFrame({"ret": 0.001, "source": "x", "tier": "A"}, index=idx)
    m = T.build_monthly(d, today=pd.Timestamp("2001-04-10"))
    assert list(m.index) == [pd.Timestamp("2001-02-28"), pd.Timestamp("2001-03-31")]
    assert m["ret"].iloc[0] == pytest.approx(1.001 ** 20 - 1)


# ------------------------------------------------------------------ processed files
@pytest.mark.realdata
def test_daily_file_contract():
    d = pd.read_parquet(DAILY)
    assert list(d.columns) == ["ret", "source", "tier"]
    assert d.index.name == "date" and d.index.is_monotonic_increasing and d.index.is_unique
    assert d["ret"].notna().all() and d["ret"].abs().max() < 0.08
    assert set(d["tier"]) <= {"A", "B"}
    assert d.index[0] >= pd.Timestamp("1997-01-01")      # TIPS did not exist earlier
    assert d.index[0] <= pd.Timestamp("1997-02-28")
    assert (d.index.dayofweek < 5).all()
    assert d.loc["2004-01-02":, "tier"].eq("A").all()


@pytest.mark.realdata
def test_monthly_equals_compounded_daily():
    d = pd.read_parquet(DAILY)
    m = pd.read_parquet(MONTHLY)
    assert list(m.columns) == ["ret", "source", "tier"]
    assert (m.index == m.index + pd.offsets.MonthEnd(0)).all()
    c = (1 + d["ret"]).groupby(pd.Grouper(freq="ME")).prod() - 1
    np.testing.assert_allclose(m["ret"].values, c.reindex(m.index).values, atol=1e-12)
    assert m.index[0] == pd.Timestamp("1997-02-28")
    # runs to the last complete month before the end of the daily data
    assert m.index[-1] >= (d.index[-1] - pd.offsets.MonthEnd(1)).normalize() - pd.Timedelta(days=1)


@pytest.mark.realdata
def test_known_episodes_look_right():
    d = pd.read_parquet(DAILY)["ret"]
    m = pd.read_parquet(MONTHLY)["ret"]
    vol = lambda a, b: d.loc[a:b].std() * np.sqrt(252)
    assert 0.12 < vol("2008-09", "2008-12") < 0.25          # 2008 Q4 liquidity episode
    assert vol("2020-03-01", "2020-03-31") > 0.15            # COVID dash for cash
    assert m.loc["2013"].add(1).prod() - 1 < -0.07           # 2013: TIP about -8.5%
    assert m.loc["2022"].add(1).prod() - 1 < -0.10           # 2022: TIP about -12%
    assert 0.015 < vol("1999", "1999") < 0.05                # quiet late-1990s
    assert 0.03 < vol("2010", "2025") < 0.08


@pytest.mark.realdata
def test_predictor_file_contract():
    p = pd.read_parquet(PRED)
    assert list(p.columns) == ["obs_date", "available_date", "series", "value"]
    assert set(p["series"]) == {"real_yield_5y", "real_yield_10y", "breakeven_5y", "breakeven_10y"}
    assert (p["available_date"] > p["obs_date"]).all()
    assert ((p["available_date"] - p["obs_date"]).dt.days >= 14).all()
    assert p["value"].between(-5, 8).all()
    assert p["obs_date"].min() <= pd.Timestamp("1999-01-31")
    r10 = p[p.series == "real_yield_10y"].set_index("obs_date")["value"]
    assert r10.loc["2021-06"].mean() < 0 < r10.loc["2024-06"].mean()   # negative real yields in 2021


# ------------------------------------------------------------------ network
@pytest.mark.network
def test_yahoo_chart_api_responds():
    r = T._get(T.YAHOO_URL.format(host="query1", sym="TIP"),
               params={"period1": 1704067200, "period2": 1706745600, "interval": "1d",
                       "events": "div,splits"}, tries=3)
    res = r.json()["chart"]["result"][0]
    assert len(res["timestamp"]) > 15 and "adjclose" in res["indicators"]


@pytest.mark.network
def test_gsw_tips_file_responds():
    r = T._get(T.GSW_TIPS_URL, tries=3, timeout=180)
    assert b"TIPSY10" in r.content and b"BKEVEN10" in r.content
