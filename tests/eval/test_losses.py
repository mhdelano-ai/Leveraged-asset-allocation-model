import numpy as np
import pandas as pd
import pytest

from assetforecast.data.synthetic import make_synthetic_panel
from assetforecast.eval.harness import load_forecasts_long, run_model
from assetforecast.eval.losses import (DEFAULT_SUBSETS, _qlike_frob, compute_losses, corr_losses, cov_losses,
                                  loss_matrix, robustness_note, scoreboard, vol_losses)
from assetforecast.eval.targets import Targets, compute_targets
from assetforecast.models.base import CorrForecaster, CovForecaster, VolForecaster


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    p = make_synthetic_panel(seed=7, start="1988-01-01", end="2003-12-31", monthly_only_years=6)
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


def test_oracle_scores_perfectly(setup):
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
    assert list(pooled.columns[-3:]) == ["qlike_rel", "qlike", "frob_scaled"]
    assert np.abs(pooled["frob_scaled"]).max() < 1e-20
    # H = S: the un-normalized QLIKE is (log det S + N) / N, and qlike_rel = log det(C_r) / N
    # (<= 0 by Hadamard) - per-origin constants that do not depend on any model
    fc = load_forecasts_long("cov", "t_oracle_cov", rd)
    for r in pooled.sample(20, random_state=0).itertuples():
        have = set(fc.loc[fc["origin"] == r.origin, "asset_i"])
        members = have if r.subset == "daily" else set(DEFAULT_SUBSETS[r.subset])
        S = T.cov_array(r.origin, [a for a in T.complete_daily_set(r.origin) if a in have and a in members])
        assert len(S) == r.dim
        d = np.sqrt(np.diag(S))
        assert r.qlike == pytest.approx((np.linalg.slogdet(S)[1] + len(S)) / len(S), rel=1e-9, abs=1e-9)
        assert r.qlike_rel == pytest.approx(np.linalg.slogdet(S / np.outer(d, d))[1] / len(S), abs=1e-9)
    assert (pooled["qlike_rel"] <= 1e-12).all()
    # the cov model's implied vols score zero against the vol targets on daily rows
    Lx = compute_losses("vol", ["cov:t_oracle_cov"], T, rd)
    assert np.abs(Lx["asset"]["qlike"]).max() < 1e-12


def test_qlike_differences_are_dimension_comparable(setup):
    p, T, rd = setup
    k = 1.3
    fc = load_forecasts_long("cov", "t_oracle_cov", rd)
    base = cov_losses(fc, T, "base")["pooled"]
    pooled = cov_losses(fc.assign(cov=fc["cov"] * k), T, "scaled")["pooled"]
    expected = 1 / k + np.log(k) - 1
    assert np.allclose(pooled["qlike"] - base["qlike"], expected)
    assert np.allclose(pooled["qlike_rel"] - base["qlike_rel"], expected)
    assert pooled["dim"].nunique() >= 3                  # same value at every dimension


def test_qlike_rel_of_diagonal_forecast_is_mean_vol_qlike():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((40, 5))
    S = A.T @ A / 40
    h = np.diag(S) * rng.uniform(0.5, 2.0, 5)
    q, qr, _ = _qlike_frob(np.diag(h), S, np.ones(5))
    x = np.diag(S) / h
    assert qr == pytest.approx(np.mean(x - np.log(x) - 1), rel=1e-12)
    assert q - qr == pytest.approx((np.log(np.diag(S)).sum() + 5) / 5)


def test_true_expected_matrix_scores_lowest_on_average():
    """With a noisy-but-unbiased realized matrix (here ~15 effective observations
    for 8 assets, like a 12-month window of 21-day returns) the un-normalized
    QLIKE no longer scores the truth at zero, but E[loss] is minimized by
    E[S]; so is the fixed-scale Frobenius loss. Losses that invert or take
    log det of the realized matrix would not be usable here: it is often
    near-singular."""
    rng = np.random.default_rng(0)
    n, dof, reps = 8, 15, 4000
    vols = np.linspace(0.05, 0.25, n)
    B = rng.standard_normal((n, 2))
    C = B @ B.T + np.eye(n)
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    Sig = C * np.outer(vols, vols)
    Lc = np.linalg.cholesky(Sig)
    Z = rng.standard_normal((reps, dof, n)) @ Lc.T
    S_draws = np.einsum("rti,rtj->rij", Z, Z) / dof                    # E[S] = Sig
    cond = np.linalg.cond(S_draws)
    assert np.median(cond) > 2 * np.linalg.cond(Sig)                  # realized matrices are ill-conditioned
    cands = {"true": Sig, "low": 0.8 * Sig, "high": 1.25 * Sig,
             "corr_shrunk": (0.5 * C + 0.5 * np.eye(n)) * np.outer(vols, vols),
             "corr_extreme": (1.3 * C - 0.3 * np.eye(n)) * np.outer(vols, vols),
             "vols_tilted": C * np.outer(vols * np.linspace(0.85, 1.15, n), vols * np.linspace(0.85, 1.15, n))}
    for nm, H in cands.items():
        assert np.linalg.eigvalsh(H).min() > 0, nm
    res = {nm: np.mean([_qlike_frob(H, S, vols)[0:3:2] for S in S_draws[:1500]], axis=0)
           for nm, H in cands.items()}
    q = {nm: v[0] for nm, v in res.items()}
    fr = {nm: v[1] for nm, v in res.items()}
    assert min(q, key=q.get) == "true" and min(fr, key=fr.get) == "true"
    assert q["true"] > (np.linalg.slogdet(Sig)[1] + n) / n - 1e-9 - 0.01   # E qlike(true) = (log det Sig + N) / N
    assert q["true"] == pytest.approx((np.linalg.slogdet(Sig)[1] + n) / n, abs=0.01)


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
    # the non-robust metrics are flagged wherever the scoreboard is shown
    note = ov.attrs["note"]
    assert note == robustness_note("vol", list(ov.columns)) and "se_logvol" in note and "log_err" in note and "qlike " not in note
    # corr: identity forecast vs realized gives err = -corr_r
    fc = load_forecasts_long("corr", "t_oracle_corr", rd)
    eye = fc.assign(corr=(fc["asset_i"] == fc["asset_j"]).astype(float))
    c = corr_losses(eye, T, "eye")
    assert np.allclose(c["pair"]["err"], -c["pair"]["corr_r"])
    d = c["pooled"][c["pooled"]["subset"] == "daily"]
    assert (d["n_pairs"] == d["dim"] * (d["dim"] - 1) // 2).all()
