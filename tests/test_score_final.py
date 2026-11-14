"""Fast tests of the pure helpers in scripts/score_final.py (synthetic inputs only)."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_P = Path(__file__).resolve().parents[1] / "scripts" / "score_final.py"
_spec = importlib.util.spec_from_file_location("score_final", _P)
sf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sf)


def test_period_and_era_masks():
    o = pd.DatetimeIndex(["1925-06-30", "1930-01-31", "2013-12-31", "2014-01-31", "2025-10-31", "2025-11-30"])
    assert sf.period_mask(o, "dev").tolist() == [True, True, True, False, False, False]
    assert sf.period_mask(o, "holdout").tolist() == [False, False, False, True, True, False]
    assert sf.period_mask(o, "full").tolist() == [True, True, True, True, True, False]
    assert sf.era_mask(o, "pre-1926").tolist() == [True, False, False, False, False, False]
    assert sf.era_mask(o, "1926-1945").tolist() == [False, True, False, False, False, False]
    assert sf.era_mask(o, "2000-2013").tolist() == [False, False, True, False, False, False]  # holdout excluded
    with pytest.raises(ValueError):
        sf.period_mask(o, "other")


def test_family_of():
    assert sf.family_of("har_pooled_log_d") == "har"
    assert sf.family_of("pair_cov_ref") == "pairs"
    assert sf.family_of("cov:combo_cov_main") == "combo"


def _frame():
    o = pd.to_datetime(["2000-01-31", "2000-02-29"])
    rows = []
    for m, base in (("m1", 1.0), ("m2", 2.0)):
        for i, t in enumerate(o):
            for j, a in enumerate(["x", "y", "z"]):
                if m == "m2" and t == o[1] and a == "z":
                    continue                      # m2 misses one cell
                rows.append({"model": m, "origin": t, "asset": a, "v": base + 10 * i + j})
    return pd.DataFrame(rows), o


def test_to_cube_common_cells_and_pooling():
    df, o = _frame()
    oo, items, models, C = sf.to_cube(df, "v", "asset", items=["x", "y", "z"])
    assert C.shape == (2, 3, 2) and models == ["m1", "m2"]
    cells = sf.common_cells(C)
    assert cells.tolist() == [[True, True, True], [True, True, False]]
    L, cnt = sf.pool_cells(C, cells)
    assert cnt.tolist() == [3, 2]
    # origin 0: m1 mean of 1,2,3 = 2 ; origin 1 common cells x,y only: m1 (11+12)/2, m2 (12+13)/2
    np.testing.assert_allclose(L, [[2.0, 3.0], [11.5, 12.5]])
    with pytest.raises(ValueError):
        sf.to_cube(pd.concat([df, df.iloc[:1]]), "v", "asset")


def test_subset_cells_requires_all_members():
    cells = np.array([[True, True, True], [True, False, True], [False, True, True]])
    s = sf.subset_cells(cells, ["a", "b", "c"], ["a", "c"])
    assert s.tolist() == [[True, False, True], [True, False, True], [False, False, False]]
    assert not sf.subset_cells(cells, ["a", "b", "c"], []).any()


def test_kday_var_matches_definition():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 2))
    X[5, 0] = np.nan                                   # missing contributes 0
    var, G = sf.kday_var(X, 21)
    X0 = np.nan_to_num(X)
    sums = np.array([X0[m:m + 21].sum(axis=0) for m in range(60 - 21 + 1)])
    ref = sums.T @ sums / len(sums) * (60 / 21)
    np.testing.assert_allclose(G, ref)
    np.testing.assert_allclose(var, np.diag(ref))


def test_kday_var_scale_iid():
    # constant daily variance v: target is n_rows * v (any calendar length)
    rng = np.random.default_rng(0)
    for n in (252, 295):
        v = np.mean([sf.kday_var(rng.standard_normal((n, 1)) * 0.01, 21)[0][0] for _ in range(400)])
        assert abs(v / (n * 1e-4) - 1) < 0.05


def test_qlike_and_qlike_rel():
    assert sf.qlike_var(1.0, 1.0) == pytest.approx(0.0)
    assert sf.qlike_var(2.0, 1.0) == pytest.approx(2 - math.log(2) - 1)
    S = np.array([[0.04, 0.01], [0.01, 0.09]])
    # diagonal H equal to diag(S): qlike_rel = mean per-asset QLIKE = 0
    assert sf.qlike_rel_independent(np.diag(np.diag(S)), S) == pytest.approx(0.0, abs=1e-12)
    # H = S: log det(C) / N
    C = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    assert sf.qlike_rel_independent(S, S) == pytest.approx(np.linalg.slogdet(C)[1] / 2)


def test_block_bootstrap_mean():
    rng = np.random.default_rng(1)
    X = rng.random((200, 3))
    X[10, 1] = np.nan
    m, lo, hi = sf.block_bootstrap_mean(X, block=24, n_boot=500, seed=0)
    assert m == pytest.approx(np.nanmean(X))
    assert lo < m < hi
    const = np.full((50, 2), 0.3)
    m, lo, hi = sf.block_bootstrap_mean(const, n_boot=100)
    assert (m, lo, hi) == pytest.approx((0.3, 0.3, 0.3))
    assert np.isnan(sf.block_bootstrap_mean(np.full((5, 2), np.nan))[0])


def test_block_bootstrap_wider_for_persistent_series():
    rng = np.random.default_rng(2)
    e = rng.standard_normal(600)
    ar = np.zeros(600)
    for t in range(1, 600):
        ar[t] = 0.95 * ar[t - 1] + e[t]
    w_iid = np.diff(sf.block_bootstrap_mean(e, block=24, n_boot=800)[1:])[0]
    w_ar = np.diff(sf.block_bootstrap_mean(ar * e.std() / ar.std(), block=24, n_boot=800)[1:])[0]
    assert w_ar > 2 * w_iid


def test_log_ratio_summary():
    x = np.log(np.array([0.5, 1.0, 1.0, 2.0, np.nan]))
    s = sf.log_ratio_summary(x)
    assert s["n"] == 4 and s["median"] == pytest.approx(0.0)
    assert s["two_thirds_lo_pct"] < 0 < s["two_thirds_hi_pct"]
    assert sf.log_ratio_summary([])["n"] == 0


def test_noise_sd():
    nd = sf.noise_sd_logvol(252, n_sim=4000)
    assert 0.14 < nd < 0.19          # ~17% standard error of a realized 21-day vol
    assert sf.noise_sd_logvol(295, n_sim=4000) < nd + 0.01
    assert sf.noise_sd_logvol_proxy(12) == pytest.approx(0.5 * math.sqrt(0.18132), rel=1e-3)


def test_hac_mean_ci_and_compare():
    idx = pd.date_range("1990-01-31", periods=240, freq="ME")
    rng = np.random.default_rng(5)
    a = pd.Series(1.0 + 0.1 * rng.standard_normal(240), idx)
    b = a + 0.5
    m, lo, hi, neff = sf.hac_mean_ci(a)
    assert lo < m < hi and neff > 0
    assert np.isnan(sf.hac_mean_ci(a.iloc[:4])[1])
    c = sf.compare(a, b)
    assert c["mean_diff"] == pytest.approx(-0.5)
    assert c["dm_p"] < 1e-6
    assert c["skill_pct"] == pytest.approx(100 * (1 - a.mean() / b.mean()))
    assert c["skill_lo"] < c["skill_pct"] < c["skill_hi"]
    assert np.isnan(sf.compare(a.iloc[:3], b.iloc[:3])["dm_p"])
    assert np.isnan(sf.compare(a - 2, b, skill=True)["skill_pct"])   # negative losses: no skill ratio


def test_leaderboard_rows():
    idx = pd.date_range("2010-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(0)
    L = pd.DataFrame({"m": 0.5 + 0.1 * rng.random(60), "ref_a": 1.0 + 0.1 * rng.random(60),
                      "ref_b": 2.0 + 0.1 * rng.random(60)}, index=idx)
    rows = sf.leaderboard(L, {}, pd.Series(3, index=idx), ["ref_a", "ref_b"], "vol", "daily")
    df = pd.DataFrame(rows)
    assert set(df["period"]) == {"dev", "holdout", "full"}
    assert (df["ref"] == "ref_a").all()                      # best reference chosen by mean loss
    r = df[(df.model == "m") & (df.period == "dev") & (df.era == "all")].iloc[0]
    assert r["n_origins"] == 48 and r["n_cells"] == 144
    assert r["skill_pct"] > 40
    assert set(df.loc[df.era != "all", "era"]) == {"2000-2013"}
