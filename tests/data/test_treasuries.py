"""Tests for the Treasury track (ust_int, ust_long)."""
import numpy as np
import pandas as pd
import pytest

from assetforecast.data.sources import treasuries as T
from assetforecast.paths import ASSET_DIR, PREDICTOR_DIR


# ------------------------------------------------------------- bond math
def _flat(rate_pct: float, n: int = 1) -> pd.DataFrame:
    """NSS parameters for a flat continuously-compounded curve."""
    return pd.DataFrame({"BETA0": [rate_pct] * n, "BETA1": 0.0, "BETA2": 0.0, "BETA3": 0.0,
                         "TAU1": 1.0, "TAU2": np.nan})


def test_par_bond_price_hand_checked():
    # 2-year 6% semiannual bond: at 6% it is at par; at 8% it is
    # 3/1.04 + 3/1.04^2 + 3/1.04^3 + 103/1.04^4 = 96.3701...
    assert T.par_bond_price(0.06, 0.06, 2.0) == pytest.approx(1.0, abs=1e-12)
    hand = (3 / 1.04 + 3 / 1.04**2 + 3 / 1.04**3 + 103 / 1.04**4) / 100
    assert hand == pytest.approx(0.963701, abs=1e-6)
    assert T.par_bond_price(0.08, 0.06, 2.0) == pytest.approx(hand, abs=1e-12)
    # a quarter of a coupon period later, unchanged yield: value accrues at y/2 per period
    assert T.par_bond_price(0.06, 0.06, 2.0 - 0.125) == pytest.approx(1.03 ** 0.25, abs=1e-12)


def test_cmt_par_bond_return_is_carry_when_yield_unchanged():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    r = T.cmt_par_bond_returns(pd.Series([5.0, 5.0, 5.0], index=idx), 10.0)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(1.025 ** (2 / 365.25) - 1, rel=1e-10)
    assert r.iloc[2] == pytest.approx(1.025 ** (2 * 3 / 365.25) - 1, rel=1e-10)


def test_nss_flat_curve_and_par_yield():
    p = _flat(5.0)
    assert T.nss_zero(p, np.array([[1.0, 10.0, 30.0]]))[0] == pytest.approx([0.05] * 3)
    # flat 5% continuous = 2*(e^{0.025}-1) semiannual bond-equivalent at every maturity
    assert T.par_yield(p, 10.0)[0] == pytest.approx(2 * (np.exp(0.025) - 1), abs=1e-12)


def test_nss_reproduces_published_gsw_yields():
    # feds200628 row for 1990-01-02: parameters and the published SVENY10 / SVENPY10
    p = pd.DataFrame([{"BETA0": 7.55833430131143, "BETA1": 0.700439779754183,
                       "BETA2": -0.0021817248740233, "BETA3": 1.31874773285689,
                       "TAU1": 0.250007139834808, "TAU2": 9.09717327936404}])
    assert T.nss_zero(p, np.array([[10.0]]))[0, 0] * 100 == pytest.approx(7.936529377863, abs=1e-6)
    assert T.par_yield(p, 10.0)[0] * 100 == pytest.approx(8.07247462448, abs=1e-4)


def test_gsw_return_flat_curve_is_exact_carry_and_shift_matches_duration():
    idx = pd.to_datetime(["2021-03-01", "2021-03-02", "2021-03-03"])
    g = pd.concat([_flat(4.0), _flat(4.0), _flat(4.1)], ignore_index=True)
    g.index = idx
    r = T.gsw_par_bond_returns(g, 10.0)
    dt = 1 / 365.25
    # unchanged flat curve: return = exp(r dt) - 1 exactly
    assert r.iloc[1] == pytest.approx(np.exp(0.04 * dt) - 1, rel=1e-9)
    # +10bp parallel (continuous) shift: approx -D_mod * dy, with D_mod at the bond-equivalent yield
    y_be = 2 * (np.exp(0.02) - 1)
    dy_be = 2 * (np.exp(0.0205) - 1) - y_be
    d = T.par_mod_duration(y_be, 10.0)
    approx = np.exp(0.04 * dt) - 1 - d * dy_be
    assert r.iloc[2] == pytest.approx(approx, abs=3e-4)
    assert r.iloc[2] < -0.007


def test_par_mod_duration_matches_numerical_derivative():
    for y, m in [(0.05, 10.0), (0.02, 25.0), (0.12, 7.0)]:
        h = 1e-6
        num = -(T.par_bond_price(y + h, y, m) - T.par_bond_price(y - h, y, m)) / (2 * h)
        assert T.par_mod_duration(y, m) == pytest.approx(num, rel=1e-6)


def test_monthly_par_bond_returns_shape():
    idx = pd.date_range("1950-01-31", periods=4, freq="ME")
    r = T.monthly_par_bond_returns(pd.Series([3.0, 3.0, 3.5, 3.0], index=idx), 20.0)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(1.015 ** (2 / 12) - 1, rel=1e-9)
    assert r.iloc[2] < -0.05 and r.iloc[3] > 0.05


# ------------------------------------------------------- processed files
@pytest.mark.realdata
@pytest.mark.parametrize("asset", ["ust_int", "ust_long"])
def test_processed_asset_files(asset):
    d = pd.read_parquet(ASSET_DIR / f"{asset}_daily.parquet")
    m = pd.read_parquet(ASSET_DIR / f"{asset}_monthly.parquet")
    for df in (d, m):
        assert list(df.columns) == ["ret", "source", "tier"]
        assert df.index.name == "date"
        assert df.index.is_monotonic_increasing and df.index.is_unique
        assert df["ret"].notna().all()
        assert set(df["tier"]) <= set("ABCD")
    assert (m.index == m.index + pd.offsets.MonthEnd(0)).all()
    assert d.index[0] <= pd.Timestamp("1962-01-31")
    assert d.index[-1] >= pd.Timestamp("2026-11-01")
    assert m.index[-1] >= pd.Timestamp("2026-10-31")
    assert m.index[0] <= pd.Timestamp("1953-06-30" if asset == "ust_int" else "1919-12-31")
    assert d["ret"].abs().max() < 0.12
    assert (d.loc["2002-08-01":, "tier"] == "A").all()
    # monthly == compounded daily over the daily era
    comp = (1 + d["ret"]).groupby(d.index + pd.offsets.MonthEnd(0)).prod() - 1
    common = m.index.intersection(comp.index)[1:]
    assert len(common) > 700
    assert np.allclose(m.loc[common, "ret"], comp.loc[common], atol=1e-12)
    # months are contiguous: no gaps in the monthly history
    full = pd.date_range(m.index[0], m.index[-1], freq="ME")
    assert len(full) == len(m)


@pytest.mark.realdata
def test_known_episodes_look_right():
    lg = pd.read_parquet(ASSET_DIR / "ust_long_monthly.parquet")["ret"]
    it = pd.read_parquet(ASSET_DIR / "ust_int_monthly.parquet")["ret"]
    yr = lambda s, y: (1 + s.loc[str(y)]).prod() - 1
    assert yr(lg, 2022) < -0.25 and yr(it, 2022) < -0.10        # 2022 rate shock
    assert yr(lg, 1994) < -0.05                                 # 1994 bond massacre
    assert yr(lg, 2008) > 0.20                                  # 2008 flight to quality
    vol = lambda s, a, b: s.loc[a:b].std() * np.sqrt(12)
    assert vol(lg, "1979-10", "1982-12") > 2 * vol(lg, "1962", "1968")   # Volcker
    assert vol(lg, "1942", "1950") < 0.05                       # WWII peg


@pytest.mark.realdata
def test_predictor_file():
    p = pd.read_parquet(PREDICTOR_DIR / "treasury_curve.parquet")
    assert list(p.columns) == ["obs_date", "available_date", "series", "value"]
    assert (p["available_date"] >= p["obs_date"]).all()
    need = {"yield_2y", "yield_5y", "yield_10y", "yield_30y", "duration_ust_int", "duration_ust_long"}
    assert need <= set(p["series"])
    w = p.pivot(index="obs_date", columns="series", values="value")
    assert w["yield_10y"].dropna().between(0, 0.2).all()
    assert w["duration_ust_int"].dropna().between(4, 9).all()
    assert w["duration_ust_long"].dropna().between(6, 23).all()
    assert w["yield_10y"].dropna().index[-1] >= pd.Timestamp("2026-11-01")


@pytest.mark.realdata
def test_predictor_publication_lags():
    """H.15 CMT: next business day; anything priced off the GSW curve: 10 business days."""
    p = pd.read_parquet(PREDICTOR_DIR / "treasury_curve.parquet")
    lag = {s: g.set_index("obs_date")["available_date"] for s, g in p.groupby("series")}
    for s in ["yield_2y", "yield_5y", "yield_7y", "yield_10y", "yield_20y", "yield_30y"]:
        exp = lag[s].index + pd.offsets.BDay(T.H15_PUBLICATION_LAG_BDAYS)
        assert (lag[s].values == exp.values).all(), s
    for s in ["yield_ladder_ust_int", "yield_ladder_ust_long", "duration_ust_int_etf_est",
              "duration_ust_long_etf_est"]:
        exp = lag[s].index + pd.offsets.BDay(T.GSW_PUBLICATION_LAG_BDAYS)
        assert (lag[s].values == exp.values).all(), s
    # ladder durations: GSW lag inside the GSW era, H.15 lag outside
    d = lag["duration_ust_int"]
    assert d.loc["2010-06-01"] == pd.Timestamp("2010-06-01") + pd.offsets.BDay(10)
    assert d.loc["1965-06-01"] == pd.Timestamp("1965-06-01") + pd.offsets.BDay(1)
    assert T.GSW_PUBLICATION_LAG_BDAYS == 10 and T.H15_PUBLICATION_LAG_BDAYS >= 1


@pytest.mark.realdata
def test_yield_20y_predictor_starts_after_rebasing():
    p = pd.read_parquet(PREDICTOR_DIR / "treasury_curve.parquet")
    y = p[p.series == "yield_20y"].set_index("obs_date")["value"]
    assert y.index[0] == pd.Timestamp("1973-01-04")     # pre-rebasing (flower-bond era) level not published


@pytest.mark.realdata
def test_ief_2003_distributions_corrected():
    """AUDIT A-1: five inflated Yahoo IEF distributions in 2003 are corrected; TLT is untouched."""
    if not (T.RAW_DIR / "y_IEF.json").exists():
        pytest.skip("raw cache missing")
    h15 = {k: T.load_h15(k) for k in T.H15_CODES if not k.endswith("_m")}
    r, t = T.etf_total_return("IEF", h15)
    changed = t[t["action"].str.startswith("replaced")]
    assert [str(d.date()) for d in changed.index] == ["2003-07-01", "2003-08-01", "2003-09-02",
                                                      "2003-10-01", "2003-12-16"]
    assert (changed["used"] < 0.45).all()
    assert abs((1 + r.loc["2003"]).prod() - 1 - 0.022) < 0.01
    _, t2 = T.etf_total_return("TLT", h15)
    assert not t2["action"].str.startswith("replaced").any()


@pytest.mark.realdata
def test_reconstruction_matches_etfs_on_overlap():
    """Re-runs the build from the raw cache (no network if cached)."""
    if not (T.RAW_DIR / "feds200628.csv").exists():
        pytest.skip("raw cache missing")
    out = T.build(refresh=False)
    D = out["diag"]
    for a, ref in [("ust_int", "IEF"), ("ust_long", "TLT")]:
        dd, mm = D[f"{a}_gsw_vs_{ref}_D_all"], D[f"{a}_gsw_vs_{ref}_M_all"]
        assert 0.93 < dd["vol_ratio"] < 1.07 and dd["corr"] > 0.94
        assert 0.93 < mm["vol_ratio"] < 1.07 and mm["corr"] > 0.98
    # month-average yields damp volatility and induce autocorrelation
    dmp = D["ust_int_monthavg_vs_monthend_all"]
    assert dmp["vol_ratio"] < 0.9 and dmp["ac1_rec"] > dmp["ac1_ref"] + 0.1
    # NSS params reproduce the published zero yields
    g = out["series"]["gsw"]
    z = T.nss_zero(g, np.full((len(g), 1), 10.0))[:, 0] * 100
    assert np.nanmax(np.abs(z - g["SVENY10"])) < 1e-3


# --------------------------------------------------------------- network
@pytest.mark.network
def test_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "RAW_DIR", tmp_path)
    g = T.load_gsw(refresh=True)
    assert g.index[0] == pd.Timestamp("1961-06-14") and g.index[-1] > pd.Timestamp("2026-01-01")
    h = T.load_h15("cmt_10y", refresh=True)
    assert h.index[0] == pd.Timestamp("1962-01-02")
    y = T.load_yahoo("IEF", refresh=True)
    assert y.index[0] <= pd.Timestamp("2002-08-01") and len(y) > 5000
    n = T.load_nber("m13033b", refresh=True)
    assert n.index[0] == pd.Timestamp("1941-10-31")
