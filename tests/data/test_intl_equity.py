"""Tests for the ``intl_equity`` loader (intl_dev, em). Offline tests read
data/processed (``realdata``); download tests are marked ``network``."""
import numpy as np
import pandas as pd
import pytest

from assetforecast.data.sources import intl_equity as IE
from assetforecast.paths import ASSET_DIR

ASSETS = ["intl_dev", "em"]


def _load(asset, freq):
    f = ASSET_DIR / f"{asset}_{freq}.parquet"
    if not f.exists():
        pytest.skip(f"{f.name} not built")
    return pd.read_parquet(f)


@pytest.fixture(scope="module")
def data():
    return {a: (_load(a, "daily"), _load(a, "monthly")) for a in ASSETS}


@pytest.mark.realdata
@pytest.mark.parametrize("asset", ASSETS)
def test_format(data, asset):
    daily, monthly = data[asset]
    for df in (daily, monthly):
        assert list(df.columns) == ["ret", "source", "tier"]
        assert df.index.name == "date"
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_unique and df.index.is_monotonic_increasing
        assert df["ret"].notna().all() and df["ret"].gt(-1).all()
        assert set(df["tier"]) <= set("ABCD")
        assert df["source"].notna().all()
    assert (monthly.index == monthly.index + pd.offsets.MonthEnd(0)).all()
    assert (daily.index.dayofweek < 5).all()
    assert daily["ret"].abs().max() < 0.25
    assert monthly["ret"].between(-0.40, 0.40).all()


@pytest.mark.realdata
@pytest.mark.parametrize("asset", ASSETS)
def test_monthly_equals_compounded_daily(data, asset):
    daily, monthly = data[asset]
    m = (1 + daily["ret"]).groupby(pd.Grouper(freq="ME")).prod() - 1
    common = monthly.index.intersection(m.index)
    assert len(common) > 250
    assert np.allclose(monthly.loc[common, "ret"], m.loc[common], atol=1e-12)
    # monthly covers the whole daily era up to the last complete month
    assert monthly.index[-1] >= daily.index[-1] - pd.offsets.MonthEnd(1)
    assert monthly.index[-1] <= daily.index[-1] + pd.offsets.MonthEnd(0)


@pytest.mark.realdata
def test_spans_and_segments(data):
    d, m = data["intl_dev"]
    assert m.index[0] == pd.Timestamp("1975-01-31")
    assert d.index[0] <= pd.Timestamp("1986-01-03")
    e, n = data["em"]
    assert n.index[0] == pd.Timestamp("1989-07-31")
    assert e.index[0] == pd.Timestamp("1994-06-01")
    for daily, monthly in data.values():
        assert daily.index[-1] >= pd.Timestamp("2026-11-01")      # runs to (near) today
        # monthly series has no gaps
        full = pd.date_range(monthly.index[0], monthly.index[-1], freq="ME")
        assert monthly.index.equals(full)
        # splices are at month boundaries: each month single-source
        assert daily.groupby(daily.index.to_period("M"))["source"].nunique().max() == 1
    assert list(d["source"].unique()) == [IE.SOURCE["aqr_gxus"], IE.SOURCE["EFA"]]
    assert list(e["source"].unique()) == [IE.SOURCE["VEIEX"], IE.SOURCE["EEM"]]
    assert (d.loc["2001-09-01":, "tier"] == "A").all() and (d.loc[:"2001-08-31", "tier"] == "B").all()
    assert (m.loc[:"1985-12-31", "tier"] == "C").all() and (n.loc[:"1994-05-31", "tier"] == "C").all()


@pytest.mark.realdata
def test_daily_density(data):
    """No multi-week holes in the daily data."""
    for daily, _ in data.values():
        gaps = daily.index.to_series().diff().dt.days.dropna()
        assert gaps.max() <= 7          # longest: 9/11 closure (Mon 9/10 -> Mon 9/17)
        per_year = daily.groupby(daily.index.year).size()
        assert per_year.iloc[1:-1].min() >= 240


@pytest.mark.realdata
def test_known_episodes(data):
    d, m = data["intl_dev"]
    e, n = data["em"]
    comp = lambda df, lo, hi: float((1 + df["ret"].loc[lo:hi]).prod() - 1)  # noqa: E731
    assert comp(m, "1987-10", "1987-10") < -0.10
    assert comp(m, "1990-01", "1990-09") < -0.25                 # Japan
    assert comp(m, "2008-10", "2008-10") < -0.15
    assert comp(m, "2020-03", "2020-03") < -0.10
    assert comp(n, "1997-07", "1998-08") < -0.40                  # Asia / Russia
    assert comp(n, "1998-08", "1998-08") < -0.20
    assert comp(n, "2008-10", "2008-10") < -0.20
    assert comp(n, "2020-03", "2020-03") < -0.12
    assert d.loc["1987-10-20", "ret"] < -0.10                     # day after the US crash
    # sane long-run vol
    for daily, monthly in data.values():
        v = np.log1p(monthly["ret"]).std() * np.sqrt(12)
        assert 0.12 < v < 0.30


@pytest.mark.realdata
def test_async_convention_documented(data):
    """Index segment reacts to the previous US day; ETF segment is synchronous."""
    d, _ = data["intl_dev"]
    us_path = ASSET_DIR / "us_large_daily.parquet"
    if not us_path.exists():
        pytest.skip("us_large not built")
    us = pd.read_parquet(us_path)["ret"]
    df = pd.concat([d["ret"], us], axis=1, join="inner").dropna()
    df.columns = ["x", "u"]
    old, new = df.loc["1990":"2000"], df.loc["2010":"2025"]
    assert old.x.corr(old.u.shift(1)) > 0.2 and old.x.corr(old.u) < 0.4
    assert new.x.corr(new.u) > 0.75
    assert new.x.corr(new.u.shift(1)) < 0.05


# ----------------------------------------------------------------- unit
def test_to_monthly_drops_incomplete_month():
    idx = pd.bdate_range("2020-01-01", "2020-02-14")
    daily = pd.DataFrame({"ret": 0.001, "source": "x", "tier": "A"}, index=idx)
    m = IE.to_monthly(daily)
    assert list(m.index) == [pd.Timestamp("2020-01-31")]
    assert np.isclose(m["ret"].iloc[0], 1.001 ** len(idx[idx.month == 1]) - 1)


def test_vol_helpers():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 0.01, 20000))
    assert abs(IE._vol(x) / (0.01 * np.sqrt(252)) - 1) < 0.02
    assert abs(IE._vol(x, 3) / IE._vol(x) - 1) < 0.05


# ---------------------------------------------------------------- network
@pytest.mark.network
def test_french_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(IE, "CACHE", tmp_path)
    s = IE.french_intl_index()
    assert s.index[0] == pd.Timestamp("1975-01-31") and len(s) > 600
    f = IE.french_factors("Emerging_5_Factors_CSV.zip")
    assert f.index[0] == pd.Timestamp("1989-07-31") and {"Mkt-RF", "RF"} <= set(f.columns)


@pytest.mark.network
def test_yahoo_download(tmp_path, monkeypatch):
    monkeypatch.setattr(IE, "CACHE", tmp_path)
    r = IE.yahoo_returns("EFA")
    assert r.index[0] < pd.Timestamp("2001-09-01")
    assert r.index[-1] > pd.Timestamp.today() - pd.Timedelta(days=10)


@pytest.mark.network
def test_msci_download(tmp_path, monkeypatch):
    monkeypatch.setattr(IE, "CACHE", tmp_path)
    r = IE.msci_returns("891800", "END_OF_MONTH")
    assert r.index[0] <= pd.Timestamp("2001-01-31") and len(r) > 290


@pytest.mark.network
@pytest.mark.slow
def test_aqr_download(tmp_path, monkeypatch):
    monkeypatch.setattr(IE, "CACHE", tmp_path)
    r = IE.aqr_total("Global Ex USA")
    assert r.loc["1986-01":"1986-12"].size > 240
