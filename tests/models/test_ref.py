import numpy as np
import pandas as pd
import pytest

from volmodel.data.synthetic import make_synthetic_panel
from volmodel.models import get_model, list_models
from volmodel.models.ref import nearest_corr, pairwise_corr


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(seed=3)


def test_registered():
    assert {"ref_longrun", "ref_last12m", "ref_last60m", "ref_blend"} <= set(list_models("vol"))
    assert {"ref_corr_longrun", "ref_corr_last120m", "ref_corr_zero"} <= set(list_models("corr"))


@pytest.mark.parametrize("name", ["ref_longrun", "ref_last12m", "ref_last60m", "ref_blend"])
def test_vol_models_valid_at_every_stage(panel, name):
    m = get_model("vol", name)
    for t in ["1979-12-31", "1990-06-30", "2010-12-31"]:
        view = panel.asof(t)
        assets = view.eligible()
        v = m.forecast_vol(view, assets)
        assert list(v.index) == assets and np.isfinite(v).all() and (v > 0).all()


@pytest.mark.parametrize("name", ["ref_corr_longrun", "ref_corr_last120m", "ref_corr_zero"])
def test_corr_models_positive_definite(panel, name):
    m = get_model("corr", name)
    for t in ["1979-12-31", "1990-06-30", "2010-12-31"]:
        view = panel.asof(t)
        assets = view.eligible()
        c = m.forecast_corr(view, assets).to_numpy()
        assert np.allclose(c, c.T) and np.allclose(np.diag(c), 1.0)
        assert np.linalg.eigvalsh(c).min() > 0


def test_longrun_recovers_true_vol(panel):
    view = panel.asof("2015-12-31")
    v = get_model("vol", "ref_longrun").forecast_vol(view, ["us_large", "ust_int"])
    assert 0.12 < v["us_large"] < 0.21 and 0.04 < v["ust_int"] < 0.08


def test_pairwise_corr_handles_unequal_history():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.standard_normal((200, 3)), columns=list("abc"))
    x["b"] = 0.8 * x["a"] + 0.6 * x["b"]
    x.loc[:150, "c"] = np.nan
    c = pairwise_corr(x)
    assert c[0, 1] == pytest.approx(0.8, abs=0.08)
    c2 = pairwise_corr(x.assign(c=x["c"].where(x.index > 185)))
    assert c2[0, 2] == 0.0                       # too little overlap: no claim
    assert np.linalg.eigvalsh(nearest_corr(c)).min() > 0
