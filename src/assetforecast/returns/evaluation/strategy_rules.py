"""Portfolio rules for the economic-value test. Pre-declared; see ``strategy.py``.

Every rule maps one origin's inputs to target weights on the risky assets
(the rest is T-bills, negative = borrowing):

* ``mu``    expected one-month *simple* excess return per asset (NaN = not eligible)
* ``sigma`` one-month covariance of excess returns (rows/cols of ineligible
            assets may be NaN)

All rules are long-only. Weights of ineligible assets are 0.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class RuleSettings:
    gamma: float = 5.0            # relative risk aversion (rule and CER evaluation)
    max_weight: float = 1.0       # per-asset cap, fraction of capital
    max_gross: float = 2.0        # gross exposure cap (Reg-T style 50% margin -> 2x)
    vol_target: float = 0.10      # annualized ex-ante volatility target
    timing_cap: float = 1.5       # Campbell-Thompson (2008) bounds on a single-asset sleeve


RULES_MEAN = ("mv", "mv_vt", "timing")   # use expected returns
RULES_RISK = ("vt_rp",)                  # use only the risk model
RULES_STATIC = ("ew", "sixty_forty")     # use neither


def _sub(mu, sigma):
    idx = np.flatnonzero(np.isfinite(mu) & np.isfinite(np.diag(sigma)))
    S = sigma[np.ix_(idx, idx)]
    if not np.isfinite(S).all():
        raise ValueError("covariance has NaN entries among eligible assets")
    S = 0.5 * (S + S.T)
    return idx, mu[idx], S


def _solve(obj, jac, x0, bounds, cons):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = minimize(obj, x0, jac=jac, bounds=bounds, constraints=cons, method="SLSQP",
                     options={"ftol": 1e-12, "maxiter": 500})
    return r


def _feasible(w, s: RuleSettings, tol=1e-7):
    return (w >= -tol).all() and (w <= s.max_weight + tol).all() and w.sum() <= s.max_gross + tol


def mean_variance(mu, sigma, s: RuleSettings = RuleSettings()) -> np.ndarray:
    """max w'mu - gamma/2 w'Sigma w  s.t. 0 <= w_i <= max_weight, sum w <= max_gross."""
    N = len(mu)
    out = np.zeros(N)
    idx, m, S = _sub(mu, sigma)
    if len(idx) == 0 or (m <= 0).all():
        return out
    g = s.gamma
    n = len(idx)
    bounds = [(0.0, s.max_weight)] * n
    cons = [{"type": "ineq", "fun": lambda w: s.max_gross - w.sum(), "jac": lambda w: -np.ones(n)}]
    obj = lambda w: -(m @ w - 0.5 * g * w @ S @ w)
    jac = lambda w: -(m - g * S @ w)
    best = None
    for x0 in (np.clip(np.maximum(m, 0) / (g * np.diag(S)), 0, s.max_weight) * min(1.0, s.max_gross / n),
               np.zeros(n)):
        r = _solve(obj, jac, x0, bounds, cons)
        w = np.clip(r.x, 0, s.max_weight)
        if w.sum() > s.max_gross:
            w *= s.max_gross / w.sum()
        if best is None or obj(w) < obj(best) - 1e-14:
            best = w
        if r.success:
            break
    out[idx] = best
    return out


def mean_variance_voltarget(mu, sigma, s: RuleSettings = RuleSettings()) -> np.ndarray:
    """max w'mu  s.t. w'Sigma w <= (vol_target^2 / 12), 0 <= w_i <= max_weight, sum w <= max_gross.

    The direction of the mean-variance portfolio at a fixed ex-ante risk: the
    forecast decides *what* to hold, not *how much* risk to take."""
    N = len(mu)
    out = np.zeros(N)
    idx, m, S = _sub(mu, sigma)
    if len(idx) == 0 or (m <= 0).all():
        return out
    n = len(idx)
    v2 = s.vol_target ** 2 / 12.0
    bounds = [(0.0, s.max_weight)] * n
    cons = [{"type": "ineq", "fun": lambda w: s.max_gross - w.sum(), "jac": lambda w: -np.ones(n)},
            {"type": "ineq", "fun": lambda w: (v2 - w @ S @ w) / v2, "jac": lambda w: -2 * S @ w / v2}]
    obj = lambda w: -(m @ w) * 100.0
    jac = lambda w: -m * 100.0
    # start: inverse-vol portfolio on the positive-mean assets, scaled inside the risk budget
    pos = m > 0
    x0 = np.where(pos, 1.0 / np.sqrt(np.diag(S)), 0.0)
    x0 *= min(np.sqrt(v2 / (x0 @ S @ x0)), s.max_gross / x0.sum(), s.max_weight / x0.max()) * 0.99
    r = _solve(obj, jac, x0, bounds, cons)
    w = np.clip(r.x, 0, s.max_weight)
    if w.sum() > s.max_gross:
        w *= s.max_gross / w.sum()
    risk = w @ S @ w
    if risk > v2:
        w *= np.sqrt(v2 / risk)
    if not r.success and m @ w < m @ x0:
        w = x0
    out[idx] = w
    return out


def timing(mu, sigma, s: RuleSettings = RuleSettings()) -> np.ndarray:
    """Per-asset timing: 1/N of capital to each eligible asset, each sleeve
    holding the single-asset Campbell-Thompson weight mu_i / (gamma sigma_i^2)
    clipped to [0, timing_cap]. Risk per sleeve is proportional to the forecast
    Sharpe ratio (w_i sigma_i = SR_i / gamma); correlations are ignored."""
    N = len(mu)
    out = np.zeros(N)
    var = np.diag(sigma)
    ok = np.isfinite(mu) & np.isfinite(var)
    n = ok.sum()
    if n == 0:
        return out
    out[ok] = np.clip(mu[ok] / (s.gamma * var[ok]), 0.0, s.timing_cap) / n
    return out


def vol_target_risk_parity(mu, sigma, s: RuleSettings = RuleSettings()) -> np.ndarray:
    """Inverse-volatility weights scaled to the ex-ante vol target (full
    covariance), within the gross and per-asset caps. Ignores ``mu`` except
    to read the eligible set."""
    N = len(mu)
    out = np.zeros(N)
    idx, _, S = _sub(mu, sigma)
    if len(idx) == 0:
        return out
    w = 1.0 / np.sqrt(np.diag(S))
    v2 = s.vol_target ** 2 / 12.0
    k = min(np.sqrt(v2 / (w @ S @ w)), s.max_gross / w.sum(), s.max_weight / w.max())
    out[idx] = k * w
    return out


def equal_weight(mu, sigma=None, s: RuleSettings = RuleSettings()) -> np.ndarray:
    ok = np.isfinite(mu)
    out = np.zeros(len(mu))
    if ok.any():
        out[ok] = 1.0 / ok.sum()
    return out


RULE_FUNCS = {
    "mv": mean_variance,
    "mv_vt": mean_variance_voltarget,
    "timing": timing,
    "vt_rp": vol_target_risk_parity,
    "ew": equal_weight,
}
