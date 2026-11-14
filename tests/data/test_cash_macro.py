import numpy as np
import pandas as pd
import pytest

from assetforecast.paths import ASSET_DIR, PREDICTOR_DIR

CASH_D = ASSET_DIR / "cash_daily.parquet"
CASH_M = ASSET_DIR / "cash_monthly.parquet"
MACRO = PREDICTOR_DIR / "macro.parquet"

needs_files = pytest.mark.skipif(not (CASH_D.exists() and CASH_M.exists() and MACRO.exists()),
                                 reason="processed cash/macro files not built")


@pytest.fixture(scope="module")
def daily():
    return pd.read_parquet(CASH_D)


@pytest.fixture(scope="module")
def monthly():
    return pd.read_parquet(CASH_M)


@pytest.fixture(scope="module")
def macro():
    return pd.read_parquet(MACRO)


# ------------------------------------------------------------------ offline, real data
@pytest.mark.realdata
@needs_files
def test_cash_format(daily, monthly):
    for df in (daily, monthly):
        assert list(df.columns) == ["ret", "source", "tier"]
        assert df.index.name == "date" and df.index.is_monotonic_increasing and df.index.is_unique
        assert set(df["tier"]) <= set("ABCD")
    assert (monthly.index == monthly.index + pd.offsets.MonthEnd(0)).all()
    assert monthly.index[0] <= pd.Timestamp("1871-01-31")
    assert daily.index[0] == pd.Timestamp("1926-07-01")


@pytest.mark.realdata
@needs_files
def test_no_gaps_in_daily_cash(daily, monthly):
    assert daily["ret"].notna().all()
    assert monthly["ret"].notna().all()
    # no gaps longer than a 4-day weekend + bridge day (5 calendar days, e.g. Hurricane Sandy 2012,
    # 1950s-60s holiday closures), except the 1933 bank holiday and the 9/11 closure
    gaps = daily.index.to_series().diff().dt.days
    big = gaps[gaps > 5]
    allowed = {pd.Timestamp("1933-03-15"), pd.Timestamp("2001-09-17")}
    assert set(big.index) <= allowed, big
    # the daily series runs close to "now" (within a month of the latest monthly row)
    assert daily.index[-1] >= monthly.index[-1] - pd.Timedelta(days=31)
    # no month in the daily era is missing
    months = pd.date_range(daily.index[0], daily.index[-1], freq="ME")
    assert months.isin(daily.index + pd.offsets.MonthEnd(0)).all()
    # monthly has no missing months
    full = pd.date_range(monthly.index[0], monthly.index[-1], freq="ME")
    assert monthly.index.equals(full)


@pytest.mark.realdata
@needs_files
def test_daily_compounds_to_monthly(daily, monthly):
    m = (1 + daily["ret"]).groupby(daily.index + pd.offsets.MonthEnd(0)).prod() - 1
    assert np.allclose(m.values, monthly["ret"].reindex(m.index).values, atol=1e-12, rtol=0)


@pytest.mark.realdata
@needs_files
def test_cash_plausible(daily, monthly):
    ann = monthly["ret"] * 12
    assert ann.between(-0.01, 0.20).all()
    assert daily["ret"].between(-0.001, 0.003).all()
    # the splice to the yield-based tail is seamless: the first tail month is within 50bp/yr of the last French month
    fr = monthly[monthly["source"] == "french_rf"]["ret"]
    tail = monthly[monthly["source"] != "french_rf"].loc[fr.index[-1]:]["ret"]
    if len(tail):
        assert abs(tail.iloc[0] - fr.iloc[-1]) * 12 < 0.005


@pytest.mark.realdata
@needs_files
def test_macro_point_in_time(macro):
    assert list(macro.columns) == ["obs_date", "available_date", "series", "value"]
    assert (macro["available_date"] >= macro["obs_date"]).all()
    assert not macro.duplicated(["series", "obs_date", "available_date"]).any()
    assert macro["series"].str.startswith("macro_").all()
    assert np.isfinite(macro["value"]).all()
    # release-lagged series must not be available in their own month
    for s in ["macro_cpi_level", "macro_unrate", "macro_ip_yoy", "macro_sahm_recession"]:
        g = macro[macro["series"] == s]
        assert (g["available_date"] > g["obs_date"] + pd.Timedelta(days=10)).all(), s


@pytest.mark.realdata
@needs_files
def test_market_rate_averages_published_after_month_end(macro):
    """Monthly averages of daily market rates need the month's last daily print, which H.15
    publishes the next business day (H.10 dollar: weekly)."""
    for s in ["macro_short_rate", "macro_long_yield", "macro_term_spread", "macro_policy_rate",
              "macro_policy_rate_chg_3m", "macro_policy_rate_chg_12m", "macro_usd_broad_yoy"]:
        g = macro[macro["series"] == s]
        assert len(g)
        assert (g["available_date"] >= g["obs_date"] + pd.offsets.BDay(1)).all(), s
    g = macro[macro["series"] == "macro_usd_broad_yoy"]
    assert (g["available_date"] >= g["obs_date"] + pd.offsets.BDay(5)).all()


@pytest.mark.realdata
@needs_files
def test_macro_coverage(macro):
    first = macro.groupby("series")["obs_date"].min()
    assert first["macro_cpi_level"] <= pd.Timestamp("1871-12-31")
    assert first["macro_unrate"] <= pd.Timestamp("1948-01-31")
    last = macro.groupby("series")["obs_date"].max()
    assert (last >= pd.Timestamp("2025-06-30")).all(), last


# --------------------------------------------------------------------------- network
@pytest.mark.network
def test_download_french_rf():
    from assetforecast.data.sources.cash_macro import french_rf

    m, d = french_rf(refresh=True)
    assert m.index[0] == pd.Timestamp("1926-07-31") and d.index[0] == pd.Timestamp("1926-07-01")
    assert m.index[-1] > pd.Timestamp.today() - pd.Timedelta(days=200)


@pytest.mark.network
def test_download_h15_bill():
    from assetforecast.data.sources.cash_macro import dbnomics

    s = dbnomics("FED/H15/RIFSGFSW04_N.B", refresh=True)
    assert s.dropna().index[-1] > pd.Timestamp.today() - pd.Timedelta(days=30)


@pytest.mark.network
def test_download_nber_cp():
    from assetforecast.data.sources.cash_macro import nber

    s = nber("m13002", refresh=True)
    assert s.index[0] <= pd.Timestamp("1871-01-31") and s.loc["1871":"1919"].notna().all()
