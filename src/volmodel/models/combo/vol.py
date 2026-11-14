"""Volatility combinations of stored forecasts (see the package docstring).

Pipeline at origin ``t`` for the requested assets:

1. read every member's stored forecast at ``t`` and turn it into a VARIANCE;
2. average members within each group (a family, or each member on its own),
   arithmetically in variance;
3. combine the group forecasts across groups: equal weights (``mean``),
   trimmed mean (``trim``: drop the highest and lowest group per asset when
   there are at least 5 groups), ``median``, or weights learned at the
   estimation date ``E(t)`` (``stack``, ``invq``, ``best``);
4. optionally multiply by a level-calibration constant ``c(E)``.

Learned quantities (at ``E``, from the training cells of
:mod:`~volmodel.models.combo.train`, every origin weighted equally and the
assets of an origin equally, as in the pooled scoreboard):

* ``stack``: non-negative weights summing to one that minimize mean QLIKE
  of ``f = G w`` on the training cells, shrunk toward equal weights:
  ``w = lam w* + (1 - lam) / g``, ``lam = Y / (Y + 20)`` with ``Y`` the
  number of training years (20 years: lam = 0.5).
* ``invq``: ``w* ~ 1 / mean QLIKE`` of each group forecast, same shrinkage.
* ``best``: all weight on the group with the lowest past mean QLIKE (the
  "pick the past winner" baseline; not shrunk).
* calibration (``calibrate=True`` or ``"pooled"``): ``c_hat = mean(r / f)``
  (the QLIKE-optimal multiplicative constant), shrunk toward 1 the same way:
  ``c = 1 + lam (c_hat - 1)``. ``calibrate="group"`` (evaluated, not
  registered) estimates one constant per asset group (equity, rates, credit,
  real), each shrunk toward 1 with its own training years; a group with no
  training cells gets the pooled constant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..base import VolForecaster
from ...universe import BY_ID
from .store import POS, MissingComponentError, family_of, load, results_dir_in_use
from .train import estimation_date, training_data

SHRINK_YEARS = 20.0     # lam = Y / (Y + 20): half weight on the data after 20 years of training
TRIM_MIN_GROUPS = 5     # trimmed mean drops one group at each end only from 5 groups up
AGGS = ("mean", "trim", "median", "stack", "invq", "best")
LEARNED = ("stack", "invq", "best")


@dataclass(frozen=True)
class Estimate:
    E: pd.Timestamp
    groups: tuple
    weights: np.ndarray        # across groups
    scale: float               # calibration constant c (1 if not calibrating)
    scale_raw: float           # c_hat before shrinkage (nan if no data)
    lam: float                 # shrinkage weight on the data
    n_years: float
    n_cells: int
    past_qlike: np.ndarray     # mean past QLIKE of each group forecast (nan if no data)
    scale_by_group: tuple = () # ((asset group, c), ...) for calibrate="group"

    def scale_for(self, assets) -> np.ndarray:
        d = dict(self.scale_by_group)
        return np.array([d.get(BY_ID[a].group, self.scale) for a in assets])


def aggregate(X: np.ndarray, w: np.ndarray, how: str) -> np.ndarray:
    """Combine group forecasts ``X`` (n x g, variances) into n forecasts."""
    g = X.shape[1]
    if how in ("mean",) + LEARNED:
        return X @ w
    if how == "median":
        return np.median(X, axis=1)
    if how == "trim":
        if g < TRIM_MIN_GROUPS:
            return X.mean(axis=1)
        s = np.sort(X, axis=1)
        return s[:, 1:-1].mean(axis=1)
    raise ValueError(f"unknown aggregation {how!r}")


def _qlike(r, f):
    x = r / f
    return x - np.log(x) - 1.0


def stack_weights(X: np.ndarray, r: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Simplex weights minimizing sum omega * QLIKE(r, X w) (SLSQP from equal weights)."""
    g = X.shape[1]
    if g == 1:
        return np.ones(1)
    om = omega / omega.sum()

    def fun(w):
        f = X @ w
        return float(om @ (r / f + np.log(f))), (om * (1.0 / f - r / f ** 2)) @ X

    w0 = np.full(g, 1.0 / g)
    res = minimize(fun, w0, jac=True, method="SLSQP", bounds=[(0.0, 1.0)] * g,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                 "jac": lambda w: np.ones_like(w)}],
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x if np.all(np.isfinite(res.x)) else w0, 0.0, None)
    if w.sum() <= 0:
        return w0
    w = w / w.sum()
    # never accept a worse point than the start (SLSQP can stop early)
    return w if fun(w)[0] <= fun(w0)[0] + 1e-12 else w0


class VolCombo(VolForecaster):
    """Combination of stored vol forecasts. See the module docstring."""

    def __init__(self, members: list[str] | None = None, *, group: str = "family", agg: str = "mean",
                 calibrate: bool = False, exclude_families: tuple = ("combo",), results_dir=None,
                 shrink_years: float = SHRINK_YEARS):
        if calibrate is True:
            calibrate = "pooled"
        if calibrate not in (False, None, "pooled", "group"):
            raise ValueError("calibrate must be False, True/'pooled' or 'group'")
        if agg not in AGGS:
            raise ValueError(f"agg must be one of {AGGS}")
        if group not in ("family", "none"):
            raise ValueError("group must be 'family' or 'none'")
        self.members = list(members) if members is not None else None
        self.group, self.agg, self.calibrate = group, agg, (calibrate or False)
        self.exclude_families = tuple(exclude_families)
        self.results_dir = results_dir
        self.shrink_years = float(shrink_years)
        self._prep = None
        self._est: dict[pd.Timestamp, Estimate] = {}

    # ------------------------------------------------------------ setup
    def member_list(self) -> list[str]:
        if self.members is not None:
            return list(self.members)
        from ..base import list_models
        own = getattr(self, "name", "")
        return [m for m in list_models("vol") if family_of(m) not in self.exclude_families and m != own]

    def _groups(self, members):
        if self.group == "none":
            return [(m, [m]) for m in members]
        fams: dict[str, list[str]] = {}
        for m in members:
            fams.setdefault(family_of(m), []).append(m)
        return sorted(fams.items())

    def _prepare(self):
        root = results_dir_in_use(self.results_dir)
        members = self.member_list()
        if not members:
            raise MissingComponentError(f"combo {getattr(self, 'name', '?')!r} has no members")
        stores = {m: load("vol", m, root) for m in members}
        sig = (str(root),) + tuple((m, id(s)) for m, s in stores.items())
        if self._prep is not None and self._prep["sig"] == sig:
            return self._prep
        groups = self._groups(members)
        grid = stores[members[0]].origins
        for s in stores.values():
            grid = grid.union(s.origins)
        arrays = {m: s.on_grid(grid) for m, s in stores.items()}
        G = np.stack([np.mean([arrays[m] for m in mem], axis=0) for _, mem in groups], axis=1)  # O x g x N
        self._prep = dict(sig=sig, root=root, members=members, stores=stores, groups=groups,
                          grid=grid, arrays=arrays, G=G)
        self._est = {}
        return self._prep

    # -------------------------------------------------------- estimation
    def _missing(self, prep, origin, a):
        for m in prep["members"]:
            k = prep["grid"].get_indexer([origin])[0]
            if k < 0 or not np.isfinite(prep["arrays"][m][k, a]):
                return m
        return "?"

    def estimate(self, view, E) -> Estimate:
        prep = self._prepare()
        E = pd.Timestamp(E)
        hit = self._est.get(E)
        if hit is not None:
            return hit
        groups = tuple(g for g, _ in prep["groups"])
        g = len(groups)
        eq = np.full(g, 1.0 / g)
        need_fit = self.agg in LEARNED or self.calibrate
        tr = training_data(view, E) if need_fit else None
        if tr is None or not tr.cells.any():
            est = Estimate(E, groups, eq, 1.0, np.nan, 0.0, 0.0, 0, np.full(g, np.nan), ())
            self._est[E] = est
            return est
        rows = prep["grid"].get_indexer(tr.origins)
        cells = tr.cells.copy()
        oi, ai = np.nonzero(cells)
        if (rows[oi] < 0).any():
            k = int(np.argmax(rows[oi] < 0))
            o = tr.origins[oi[k]]
            raise MissingComponentError(
                f"combo {getattr(self, 'name', '?')!r}: component {self._missing(prep, o, ai[k])!r} has no "
                f"stored vol forecast at training origin {o.date()} (needed at estimation date {E.date()})")
        X = prep["G"][rows[oi], :, ai]                      # n x g
        bad = ~np.isfinite(X).all(axis=1)
        if bad.any():
            k = int(np.argmax(bad))
            o = tr.origins[oi[k]]
            asset = list(POS)[ai[k]]
            raise MissingComponentError(
                f"combo {getattr(self, 'name', '?')!r}: component {self._missing(prep, o, ai[k])!r} has no "
                f"stored vol forecast for {asset} at training origin {o.date()} "
                f"(needed at estimation date {E.date()})")
        r = tr.realized[oi, ai]
        n_per = cells.sum(axis=1)
        omega = 1.0 / n_per[oi]                             # each origin counts once
        Y = tr.n_years
        lam = Y / (Y + self.shrink_years)
        q = np.array([float(omega @ _qlike(r, X[:, j]) / omega.sum()) for j in range(g)])
        if self.agg == "stack":
            w = lam * stack_weights(X, r, omega) + (1 - lam) * eq
        elif self.agg == "invq":
            w_star = (1.0 / q) / (1.0 / q).sum()
            w = lam * w_star + (1 - lam) * eq
        elif self.agg == "best":
            w = np.zeros(g)
            w[int(np.argmin(q))] = 1.0
        else:
            w = eq
        c, c_raw, by_group = 1.0, np.nan, ()
        if self.calibrate:
            f = aggregate(X, w, self.agg)
            c_raw = float(omega @ (r / f) / omega.sum())
            c = 1.0 + lam * (c_raw - 1.0)
            if self.calibrate == "group":
                agrp = np.array([BY_ID[a].group for a in POS])[ai]
                out = []
                for grp in sorted(set(agrp)):
                    sel = agrp == grp
                    og = oi[sel]
                    n_g = np.bincount(og, minlength=cells.shape[0])[og]   # group cells at that origin
                    om_g = 1.0 / n_g
                    Y_g = len(np.unique(og)) / 12.0
                    lam_g = Y_g / (Y_g + self.shrink_years)
                    cg_raw = float(om_g @ (r[sel] / f[sel]) / om_g.sum())
                    out.append((grp, 1.0 + lam_g * (cg_raw - 1.0)))
                by_group = tuple(out)
        est = Estimate(E, groups, w, c, c_raw, lam, Y, int(len(r)), q, by_group)
        self._est[E] = est
        return est

    # ---------------------------------------------------------- forecast
    def forecast_vol(self, view, assets):
        prep = self._prepare()
        t = view.asof_date if view.asof_date is not None else view.monthly.index[-1]
        t = pd.Timestamp(t)
        per_member = {m: prep["stores"][m].at(t, assets) for m in prep["members"]}   # raises if missing
        Xt = np.stack([np.mean([per_member[m] for m in mem], axis=0) for _, mem in prep["groups"]], axis=1)
        if self.agg in LEARNED or self.calibrate:
            est = self.estimate(view, estimation_date(t))
            w = est.weights
            c = est.scale_for(assets) if self.calibrate == "group" else est.scale
        else:
            w, c = np.full(Xt.shape[1], 1.0 / Xt.shape[1]), 1.0
        v = aggregate(Xt, w, self.agg) * c
        return pd.Series(np.sqrt(v), index=list(assets))
