import numpy as np
import pandas as pd

from assetforecast.data.synthetic import make_synthetic_panel
from assetforecast.universe import ASSET_IDS


def test_synthetic_panel_shapes():
    p = make_synthetic_panel(seed=1)
    assert list(p.daily.columns) == ASSET_IDS == list(p.monthly.columns)
    assert p.monthly.index[0] < p.daily.index[0]
    assert p.cash_daily.index.equals(p.daily.index)
    assert p.cash_monthly.index.equals(p.monthly.index)
    assert set(p.predictors.columns) == {"obs_date", "available_date", "series", "value"}


def test_monthly_equals_compounded_daily():
    p = make_synthetic_panel(seed=1, stagger=False)
    m = (1 + p.daily).groupby(pd.Grouper(freq="ME")).prod() - 1
    common = m.index[1:-1]
    assert np.allclose(m.loc[common], p.monthly.loc[common])


def test_asof_never_leaks():
    p = make_synthetic_panel(seed=1)
    t = pd.Timestamp("2000-06-30")
    v = p.asof(t)
    assert v.daily.index.max() <= t and v.monthly.index.max() <= t
    assert (v.predictors["available_date"] <= t).all()
    # a predictor observed at t but published later must be absent
    w = v.predictors_wide()
    assert w["synthetic_macro"].dropna().index.max() < t
    assert v.asof_date == t


def test_eligibility_grows():
    p = make_synthetic_panel(seed=1)
    assert len(p.asof("1980-12-31").eligible()) < len(p.asof("2010-12-31").eligible())
