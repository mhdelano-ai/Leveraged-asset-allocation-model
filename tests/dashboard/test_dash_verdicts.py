"""The verdict rules flip correctly on synthetic inputs."""
from __future__ import annotations

import pandas as pd

from assetforecast.dashboard import verdicts as vd

FAM = ("m1", "m2", "m3")


def scores(r2: dict, spa: float, det=0.2, extra=()):
    rows = [(m, 12, "pooled", "holdout", "hist_mean", "r2_oos", v) for m, v in r2.items()]
    rows += [("*mean", 12, "pooled", "holdout", "hist_mean", "spa_p_mse", spa),
             ("hist_mean", 12, "pooled", "holdout", "none", "r2_detectable", det),
             ("hist_mean", 12, "pooled", "holdout", "none", "n_indep", 11.0), *extra]
    return pd.DataFrame(rows, columns=["model", "horizon", "asset", "period", "benchmark", "metric", "value"])


def test_mean_verdict_flips():
    assert vd.mean_verdict(scores({"m1": 0.03, "m2": -0.01, "m3": 0.0}, 0.01), FAM, 12, "holdout")["status"] == "pass"
    v = vd.mean_verdict(scores({"m1": 0.03, "m2": -0.01, "m3": 0.0}, 0.2), FAM, 12, "holdout")
    assert v["status"] == "weak" and v["best_model"] == "m1" and v["n_positive"] == 1
    assert vd.mean_verdict(scores({"m1": -0.03, "m2": -0.01, "m3": 0.0}, 0.01), FAM, 12, "holdout")["status"] == "none"
    assert vd.mean_verdict(scores({}, 0.5), FAM, 12, "holdout")["status"] == "na"
    # a model outside the family never decides the verdict
    assert vd.mean_verdict(scores({"m1": -0.01, "other": 0.5}, 0.001), FAM, 12, "holdout")["status"] == "none"


def test_dist_skill_verdict_flips():
    def sc(skill, spa):
        return pd.DataFrame([("m1", 3, "pooled", "holdout", "vol_ewma", "crps_skill", skill),
                             ("*dist", 3, "pooled", "holdout", "vol_ewma", "spa_p_crps", spa)],
                            columns=["model", "horizon", "asset", "period", "benchmark", "metric", "value"])
    assert vd.dist_skill_verdict(sc(0.01, 0.01), ("m1",), 3, "holdout")["status"] == "pass"
    assert vd.dist_skill_verdict(sc(0.01, 0.30), ("m1",), 3, "holdout")["status"] == "weak"
    assert vd.dist_skill_verdict(sc(-0.01, 0.01), ("m1",), 3, "holdout")["status"] == "none"


def test_calib_verdict_flips():
    def sc(c90, p):
        return pd.DataFrame([("d", 12, "pooled", "holdout", "none", "cover_90", c90, "assets"),
                             ("d", 12, "pooled", "holdout", "none", "calib_p", p, "assets"),
                             ("d", 12, "pooled", "holdout", "none", "cover_90", 0.5, "portfolios")],
                            columns=["model", "horizon", "asset", "period", "benchmark", "metric", "value", "target"])
    assert vd.calib_verdict(sc(0.905, 0.5), "d", "pooled", "holdout", target="assets")["status"] == "ok"
    v = vd.calib_verdict(sc(0.95, 0.5), "d", "pooled", "holdout", target="assets")
    assert v["status"] == "wide" and v["calib_test"] == "accepts"
    v = vd.calib_verdict(sc(0.85, 0.01), "d", "pooled", "holdout", target="assets")
    assert v["status"] == "narrow" and v["calib_test"] == "rejects"
    assert vd.calib_verdict(sc(0.9, 0.5), "d", "pooled", "holdout", target="portfolios")["status"] == "narrow"


def strat(rows):
    return pd.DataFrame(rows, columns=["family", "universe", "horizon", "period", "mean_model", "rule", "d_cer",
                                       "d_cer_p", "d_cer_q_bh"])


def test_strategy_verdict_flips():
    base = [("mean", "all", 12, "holdout", "m1", "mv", 0.01, 0.01, 0.02),
            ("mean", "all", 12, "holdout", "m2", "mv", -0.01, 0.3, 0.5),
            ("mean", "all", 12, "holdout", "m3", "mv", 0.002, 0.4, 0.5)]
    assert vd.strategy_verdict(strat(base), FAM, ("mv",), 12, "holdout")["status"] == "pass"
    weak = [r[:8] + (0.2,) for r in base]
    v = vd.strategy_verdict(strat(weak), FAM, ("mv",), 12, "holdout")
    assert v["status"] == "weak" and v["n_pos"] == 2 and v["n_sig_pos"] == 1 and v["n_q_pos"] == 0
    worse = [("mean", "all", 12, "holdout", m, "mv", -0.01, 0.01, 0.9) for m in FAM]
    v = vd.strategy_verdict(strat(worse), FAM, ("mv",), 12, "holdout")
    assert v["status"] == "worse" and v["n_sig_neg"] == 3
    # a gain that survives adjustment but only in another rule or universe does not count
    other = base + [("mean", "ex_hy", 12, "holdout", "m2", "mv", 0.05, 0.001, 0.001)]
    assert vd.strategy_verdict(strat([r[:8] + (0.9,) for r in other[:3]] + other[3:]), FAM, ("mv",), 12,
                               "holdout")["status"] == "weak"
    assert vd.combine_strategy([v, vd.strategy_verdict(strat(base), FAM, ("mv",), 12, "holdout")])["status"] == "pass"


def test_skill_and_expectations():
    assert vd.skill_verdict(7.8, 0.03) == "pass"
    assert vd.skill_verdict(7.8, 0.3) == "weak"
    assert vd.skill_verdict(-1.0, 0.01) == "none"
    assert vd.expectations_verdict(["met", "met", "met"])["status"] == "all"
    e = vd.expectations_verdict(["met", "met (point estimate; not significant)", "not met"])
    assert e["status"] == "some" and e["n_met"] == 2 and e["n_met_point_only"] == 1
    assert vd.expectations_verdict(["not met"])["status"] == "none"
    assert vd.combine_edges([{"status": "weak"}, {"status": "none"}])["status"] == "weak"
    assert vd.combine_edges([{"status": "none"}, {"status": "pass"}])["status"] == "pass"
