"""Covariance models made by pairing a volatility model with a correlation
model, registered so they can be run and scored like any other.

    pair_cov_ref     ref_blend vols x ref_corr_last120m      (the reference matrix)
    pair_cov_single  statespace_2c_daily vols x mcorr_nls    (best single models)
"""
from ..base import Composite, get_model, register


def _pair(vol: str, corr: str):
    return lambda: Composite(get_model("vol", vol), get_model("corr", corr))


register("cov", "pair_cov_ref")(_pair("ref_blend", "ref_corr_last120m"))
register("cov", "pair_cov_single")(_pair("statespace_2c_daily", "mcorr_nls"))
