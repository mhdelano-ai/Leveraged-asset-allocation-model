"""The three-holding restatement: exact aggregation, composites, past losses."""
import numpy as np
import pandas as pd
import pytest

from assetforecast import threefund as tf
from assetforecast.universe import ASSET_IDS


def test_weights_sum_to_one_with_cash():
    W = tf.weight_matrix(list(ASSET_IDS))
    assert W.loc["world_stocks"].sum() == pytest.approx(1.0)
    assert W.loc["world_bonds"].sum() == pytest.approx(1.0 - tf.COMPOSITES["world_bonds"][tf.CASH])


def test_aggregate_is_w_cov_w():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, len(ASSET_IDS)))
    cov = pd.DataFrame(a.T @ a / 40, index=ASSET_IDS, columns=ASSET_IDS)
    A = tf.aggregate(cov)
    w = pd.Series(tf.COMPOSITES["world_stocks"])
    assert A.loc["world_stocks", "world_stocks"] == pytest.approx(w @ cov.loc[w.index, w.index] @ w)
    assert np.all(np.linalg.eigvalsh(A.to_numpy()) > 0)


def test_composite_renormalizes_over_present_components():
    idx = pd.date_range("2000-01-31", periods=3, freq="ME")
    simple = pd.DataFrame(0.0, index=idx, columns=list(ASSET_IDS))
    simple["us_large"], simple["us_small"], simple["intl_dev"], simple["em"] = 0.10, 0.10, 0.02, np.nan
    simple.loc[idx[0], "intl_dev"] = np.nan
    cash = pd.Series(0.001, index=idx)
    c = tf.composite_simple(simple, cash)
    assert c["world_stocks"].iloc[0] == pytest.approx(0.10)             # US only
    assert c["world_stocks"].iloc[1] == pytest.approx((0.62 * 0.10 + 0.28 * 0.02) / 0.90)
    assert c["world_bonds"].iloc[1] == pytest.approx(0.30 * 0.001)      # bonds at 0, bills at cash
    assert tf.composite_simple(simple, cash, full_mix=True)["world_stocks"].isna().all()


def test_past_losses_of_a_levered_fall():
    idx = pd.date_range("2000-01-31", periods=14, freq="ME")
    comp = pd.DataFrame({"world_stocks": [0.0] * 6 + [-0.2] + [0.0] * 7, "world_bonds": 0.0}, index=idx)
    cash = pd.Series(0.0, index=idx)
    one = tf.past_losses([1.0, 0.0], comp, cash, 0.0)
    two = tf.past_losses([2.0, 0.0], comp, cash, 0.012)
    assert one["max_drawdown"] == pytest.approx(-0.2) and one["worst_12m"] == pytest.approx(-0.2)
    assert two["max_drawdown"] < -0.40                                  # twice the fall, plus interest
