import numpy as np
import pandas as pd
import pytest

from volmodel.data.synthetic import make_synthetic_panel
from volmodel.eval.harness import load_forecasts_long, run_model
from volmodel.eval.losses import (compute_losses, corr_losses, cov_losses, loss_matrix, scoreboard,
                                  vol_losses)
from volmodel.eval.targets import Targets, compute_targets
from volmodel.models.base import CorrForecaster, CovForecaster, VolForecaster


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    p = make_synthetic_panel(seed=7, start="1990-01-01", end="2003-12-31", monthly_only_years=3)
    T = compute_targets(p)
    rd = tmp_path_factory.mktemp("res")
    return p, T, rd


class OracleVol(VolForecaster):
    name = "t_oracle_vol"

    def __init__(self, T, mult=1.0):
        self.T, self.mult = T, mult

    def forecast_vol(self, view, assets):
        v = self.T.vol_at(view.asof_date)["vol"]
        return v.reindex(assets).fillna(0.15) * self.mult


class OracleCorr(CorrForecaster):
    name = "t_oracle_corr"

    def __init__(self, T):
        self.T = T

    def forecast_corr(self, view, assets):
        C = self.T.corr_matrix(view.asof_date, assets).fillna(0.0)
        return C


class OracleCov(CovForecaster):
    name = "t_oracle_cov"

    def __init__(self, T, k=1.0):
        self.T, self.k = T, k

    def forecast_cov(self, view, assets):
        o = view.asof_date
        comp = self.T.complete_daily_set(o)
        v = self.T.vol_at(o)["vol"].reindex(assets).fillna(0.15).to_numpy()
        C = self.T.corr_matrix(o, assets).to_numpy()
        inside = np.isin(assets, comp)
        C = np.where(np.outer(inside, inside), C, np.eye(len(assets)))
        return pd.DataFrame(C * np.outer(v, v) * self.k, index=assets, columns=assets)


def test_oracle_scores_zero(setup):
    p, T, rd = setup
    run_model(OracleVol(T), "vol", p, results_dir=rd)
    run_model(OracleCorr(T), "corr", p, results_dir=rd)
    run_model(OracleCov(T), "cov", p, results_dir=rd)
    L = compute_losses("vol", ["t_oracle_vol"], T, rd)
    assert len(L["asset"]) > 500 and L["asset"]["proxy"].any()
    assert np.abs(L["asset"][["qlike", "se_vol", "se_logvol", "log_err", "se_var_scaled"]].to_numpy()).max() < 1e-12
    assert np.abs(L["pooled"][["qlike", "se_vol_scaled"]].to_numpy()).max() < 1e-12
    Lc = compute_losses("corr", ["t_oracle_corr"], T, rd)
    assert len(Lc["pair"]) > 500 and np.abs(Lc["pair"]["se"]).max() < 1e-20
    Lv = compute_losses("cov", ["t_oracle_cov"], T, rd)
    pooled = Lv["pooled"]
    assert set(pooled["subset"]) >= {"daily", "stock_bond", "full12"}
    assert np.abs(pooled[["stein", "frob_scaled"]].to_numpy()).max() < 1e-9
    assert pooled["target_pd"].all()
    # the cov model's implied vols score zero against the vol targets on daily rows
    Lx = compute_losses("vol", ["cov:t_oracle_cov"], T, rd)
    assert np.abs(Lx["asset"]["qlike"]).max() < 1e-12


def test_stein_is_dimension_comparable(setup):
    p, T, rd = setup
    k = 1.3
    fc = load_forecasts_long("cov", "t_oracle_cov", rd)
    fc = fc.assign(cov=fc["cov"] * k)
    pooled = cov_losses(fc, T, "scaled")["pooled"]
    expected = 1 / k + np.log(k) - 1
    assert np.allclose(pooled["stein"], expected)
    assert pooled["dim"].nunique() >= 3                  # same value at every dimension


def test_qlike_minimized_by_true_variance_under_noisy_proxy():
    rng = np.random.default_rng(0)
    n, true_var = 20000, 0.04
    origins = pd.date_range("1900-01-31", periods=n, freq="ME")
    realized = true_var * rng.chisquare(5, n) / 5             # unbiased but very noisy proxy
    T = Targets(vol=pd.DataFrame({"origin": origins, "asset": "x", "vol": np.sqrt(realized),
                                  "var": realized, "proxy": True}),
                corr=pd.DataFrame(columns=["origin", "asset_i", "asset_j", "corr"]),
                fwd_ret=pd.DataFrame(columns=["origin", "asset", "ret"]), meta={"assets": ["x"]})
    mults = [0.7, 0.85, 1.0, 1.15, 1.3]
    q, mv, ml = [], [], []
    for m in mults:
        fc = pd.DataFrame({"origin": origins, "asset": "x", "vol": np.sqrt(true_var * m)})
        a = vol_losses(fc, T, f"m{m}", subsets={})["asset"]
        q.append(a["qlike"].mean())
        mv.append(a["se_var_scaled"].mean())
        ml.append(a["se_logvol"].mean())
    assert int(np.argmin(q)) == 2 and int(np.argmin(mv)) == 2
    # documented non-robustness: log-vol MSE prefers a downward-biased forecast under this noise
    assert int(np.argmin(ml)) < 2


def test_proxy_targets_still_rank_correct_above_biased(setup):
    p, T, _ = setup
    T_proxy = compute_targets(p, min_daily_frac=2.0)      # force the monthly proxy everywhere
    assert T_proxy.vol["proxy"].all()
    truth = T.vol[~T.vol["proxy"]][["origin", "asset", "vol"]]
    scores = {}
    for m in (0.8, 1.0, 1.25):
        L = vol_losses(truth.assign(vol=truth["vol"] * m), T_proxy, f"x{m}", subsets={})
        scores[m] = L["asset"][["qlike", "se_var_scaled"]].mean()
    for col in ("qlike", "se_var_scaled"):
        assert scores[1.0][col] < scores[0.8][col] and scores[1.0][col] < scores[1.25][col]


def test_corr_losses_and_scoreboard_common_origins(setup):
    p, T, rd = setup
    biased = OracleVol(T, mult=1.2)
    biased.name = "t_biased"
    run_model(biased, "vol", p, results_dir=rd, start="1998-01-01")
    L = compute_losses("vol", ["t_oracle_vol", "t_biased"], T, rd)
    sb = scoreboard(L, "vol", eras=["1999-12-31"])
    ov = sb["overall"]
    assert list(ov.index) == ["t_oracle_vol", "t_biased"]
    common = loss_matrix(L["pooled"], "qlike", subset="daily").dropna()
    assert (ov["n_origins"] == len(common)).all() and common.index.min() >= pd.Timestamp("1998-01-31")
    assert ov.loc["t_biased", "log_err"] == pytest.approx(np.log(1.2))
    assert list(sb["by_era"].columns) == ["start..1999-12", "2000-01..end"]
    assert sb["by_asset"].shape[0] == 2
    # corr: identity forecast vs realized gives err = -corr_r
    fc = load_forecasts_long("corr", "t_oracle_corr", rd)
    eye = fc.assign(corr=(fc["asset_i"] == fc["asset_j"]).astype(float))
    c = corr_losses(eye, T, "eye")
    assert np.allclose(c["pair"]["err"], -c["pair"]["corr_r"])
    d = c["pooled"][c["pooled"]["subset"] == "daily"]
    assert (d["n_pairs"] == d["dim"] * (d["dim"] - 1) // 2).all()
