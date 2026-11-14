"""Tests for the us_equity track (us_large, us_small)."""
import numpy as np
import pandas as pd
import pytest

from volmodel.paths import ASSET_DIR
from volmodel.data.sources import us_equity as ue

ASSETS = ["us_large", "us_small"]


# ----------------------------------------------------------- offline, unit
def test_parse_french_daily_and_monthly():
    text = (
        "header line\n\n  Some Daily Section\n,A,B\n"
        "19260701,   0.09,  -99.99\n19260702,   1.00,   2.00\n\n"
        "  Second Section\n,A,B\n192607,   2.89,  -2.42\n"
    )
    d = ue.parse_french(text, "Daily Section")
    assert list(d.index) == [pd.Timestamp("1926-07-01"), pd.Timestamp("1926-07-02")]
    assert d.loc["1926-07-01", "A"] == pytest.approx(0.0009)
    assert np.isnan(d.loc["1926-07-01", "B"])
    m = ue.parse_french(text, "Second Section")
    assert m.index[0] == pd.Timestamp("1926-07-31")
    assert m.iloc[0, 0] == pytest.approx(0.0289)


def test_splice_later_piece_wins_and_no_gaps():
    i1 = pd.bdate_range("2000-01-03", periods=10)
    a = pd.Series(0.01, index=i1)
    b = pd.Series(0.02, index=i1[5:])
    out = ue._splice([(a, "a", "B"), (b, "b", "A")])
    assert out.index.equals(i1)
    assert (out.loc[i1[:5], "source"] == "a").all()
    assert (out.loc[i1[5:], "source"] == "b").all()
    assert (out.loc[i1[5:], "tier"] == "A").all()


# ------------------------------------------------- offline, processed files
def _load(asset, freq):
    p = ASSET_DIR / f"{asset}_{freq}.parquet"
    if not p.exists():
        pytest.skip(f"{p} not built")
    return pd.read_parquet(p)


@pytest.mark.realdata
@pytest.mark.parametrize("asset", ASSETS)
def test_format(asset):
    for freq in ("daily", "monthly"):
        df = _load(asset, freq)
        assert list(df.columns) == ["ret", "source", "tier"]
        assert df.index.name == "date"
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing and df.index.is_unique
        assert df["ret"].notna().all()
        assert df["tier"].isin(list("ABCD")).all()
        assert (df["ret"] > -0.5).all() and (df["ret"] < 1.0).all()
    m = _load(asset, "monthly")
    assert (m.index == m.index + pd.offsets.MonthEnd(0)).all()


@pytest.mark.realdata
@pytest.mark.parametrize("asset", ASSETS)
def test_daily_history_span(asset):
    d = _load(asset, "daily")
    assert d.index[0] == pd.Timestamp("1926-07-01")
    # carried to the present by the fund/ETF segment (French lags by months)
    assert d.index[-1] >= pd.Timestamp("2026-11-01")
    assert d.loc["2001":, "tier"].eq("A").all()
    # no weekends, no big holes (longest gap: 1933 bank holiday ~ 10 days, 9/11 ~ 6)
    assert (d.index.dayofweek < 6).all()
    assert d.index.to_series().diff().max() <= pd.Timedelta(days=12)
    assert d.loc["1953":].index.dayofweek.max() <= 4  # Saturday trading ended 1952


@pytest.mark.realdata
@pytest.mark.parametrize("asset", ASSETS)
def test_monthly_equals_compounded_daily(asset):
    d, m = _load(asset, "daily"), _load(asset, "monthly")
    c = (1 + d["ret"]).groupby(pd.Grouper(freq="ME")).prod() - 1
    common = m.index.intersection(c.index)
    assert len(common) > 1000
    assert np.allclose(m.loc[common, "ret"], c.loc[common], atol=1e-12)
    # only complete months: the last monthly row is not after the last daily month
    assert m.index[-1] <= d.index[-1] + pd.offsets.MonthEnd(0)


@pytest.mark.realdata
def test_us_large_monthly_extends_to_1871_tier_d():
    m = _load("us_large", "monthly")
    assert m.index[0] <= pd.Timestamp("1871-02-28")
    pre = m.loc[:"1926-06"]
    assert (pre["tier"] == "D").all() and len(pre) > 600
    assert (m.loc["1926-07":, "tier"] != "D").all()


@pytest.mark.realdata
def test_known_dates():
    L, S = _load("us_large", "daily")["ret"], _load("us_small", "daily")["ret"]
    assert L["1987-10-19"] < -0.17          # S&P 500 TR ~ -20.5%
    assert S["1987-10-19"] < -0.10
    assert L["1929-10-28"] < -0.10 and L["1929-10-29"] < -0.08
    assert L["2020-03-16"] < -0.10 and S["2020-03-16"] < -0.10
    assert L["2008-10-13"] > 0.10            # +11.6% S&P rally
    assert L["2008-10-15"] < -0.08
    oct08 = (1 + L.loc["2008-10"]).prod() - 1
    assert -0.20 < oct08 < -0.15             # S&P 500 TR Oct 2008: -16.8%
    # aligned to trading date: no return on a weekend, Black Monday is a Monday
    assert pd.Timestamp("1987-10-19").dayofweek == 0


@pytest.mark.realdata
@pytest.mark.parametrize("asset", ASSETS)
def test_no_splice_jumps(asset):
    """Returns on the first day of each segment must be unremarkable."""
    d = _load(asset, "daily")
    first = d.index[d["source"].ne(d["source"].shift())][1:]
    vol = d["ret"].rolling(63).std()
    for t in first:
        z = abs(d.loc[t, "ret"]) / vol.shift(1).loc[t]
        assert z < 4, (asset, t, d.loc[t, "ret"])


@pytest.mark.realdata
@pytest.mark.parametrize("asset,lo,hi", [("us_large", 0.12, 0.25), ("us_small", 0.14, 0.32)])
def test_vol_levels_reasonable(asset, lo, hi):
    d = _load(asset, "daily")["ret"]
    v = d.loc["1960":].std() * np.sqrt(252)
    assert lo < v < hi


@pytest.mark.realdata
def test_small_more_volatile_than_large_each_decade():
    # monthly, not daily: pre-2000 daily small-cap index returns are positively
    # autocorrelated (stale prices), which damps DAILY vol (e.g. 1980s), see report
    L, S = _load("us_large", "monthly")["ret"], _load("us_small", "monthly")["ret"]
    for dec in range(1930, 2030, 10):
        sl = slice(f"{dec}", f"{dec + 9}")
        assert S.loc[sl].std() > 0.95 * L.loc[sl].std(), dec


# ------------------------------------------------------------------ network
@pytest.mark.network
def test_yahoo_spy_live(tmp_path, monkeypatch):
    monkeypatch.setattr(ue, "CACHE", tmp_path)
    df = ue.yahoo_series("SPY", refresh=True)
    assert df.index[0] <= pd.Timestamp("1993-02-01")
    assert (df["div"] > 0).sum() > 100
    assert df.index[-1] >= pd.Timestamp.today().normalize() - pd.Timedelta(days=10)
