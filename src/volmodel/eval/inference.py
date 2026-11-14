"""Forecast-comparison inference for overlapping 12-month-ahead losses.

Every function here takes pandas loss series produced by the walk-forward
harness and answers "is this difference real?" honestly, given the structure
of the backtest:

* **Overlap.** Origins are monthly but each loss covers the next 12 months, so
  consecutive losses share 11 months of data. Loss series are at least MA(11)
  and, because volatility itself is persistent, usually much more persistent
  than that. 600 monthly origins hold roughly 50 independent years.
* **Heavy tails.** Crisis years dominate mean losses.
* **Unequal histories.** Assets/models start at different dates; pairwise
  comparisons use only the common sample.
* **Many models.** Pairwise p-values need a multiplicity adjustment; set-wise
  questions should use the Model Confidence Set.

Conventions
-----------
* ``losses``: DataFrame, index = month-end origin (``DatetimeIndex``; gaps and
  NaN allowed), columns = model names, lower is better.
* Panel input: long frame with columns ``origin, asset, model, loss``.
* ``h`` is the forecast horizon in origin steps (12 for monthly origins and a
  12-month window). It sets the minimum dependence the variance estimators
  must allow for.
* Monthly regularity: series are placed on a regular month-end grid between
  their first and last common observation. Interior gaps are treated by the
  "amplitude-modulated" approach (missing deviations set to zero, moments
  normalised by the number of observed points; Parzen 1963, Datta & Du 2012)
  for HAC estimators, and are simply skipped (series compressed) for the
  bootstraps. Both are fine for a few scattered gaps; a warning is emitted
  when more than 10% of the interior grid is missing.

Variance estimators (``kernel``)
--------------------------------
``"ewc"`` (default)
    Equal-weighted cosine (Müller 2007; Lazarus, Lu, Stock & Watson 2018).
    The long-run variance is the average of the first ``B`` squared cosine
    projections of the series. Under fixed-b asymptotics the t-statistic is
    exactly Student-t with ``B`` degrees of freedom and Wald statistics are
    F-distributed, so the reference distribution *pays* for the estimation
    noise in the variance. The estimator is always positive. Default
    ``B = ewc_terms(T, h)`` (see :func:`default_ewc_terms`), which was chosen
    by the Monte Carlo documented in ``tests/eval/test_inference.py``.
``"bartlett"``
    Newey-West with lag truncation ``bandwidth`` (default
    :func:`default_bandwidth`) and Kiefer-Vogelsang (2005) fixed-b critical
    values, expressed as a Student-t with matched 97.5% quantile.
``"truncated"``
    The textbook Diebold-Mariano rectangular kernel at lag ``h-1`` plus the
    Harvey-Leybourne-Newbold (1997) correction and ``t(T-1)`` reference. It
    assumes the loss differential is exactly MA(h-1); with persistent
    volatility it is not, and this variant over-rejects (see Monte Carlo).
    The rectangular kernel can produce a negative variance; when it does the
    estimate falls back to Bartlett at the same lag and a note is recorded.

Monte Carlo summary
-------------------
DGP (reproduced in ``tests/eval/test_inference.py::simulate_losses``): latent
monthly log-variance AR(1) with clipped t(5) shocks; monthly realized
variance = exp(x) times Gamma(4)/4 noise; target = mean of the next 12
months; two forecasts with log-errors that are AR(1) with a common component.
"base": phi = 0.95, error rho = 0.90 (loss-differential LRV/variance ~ 8);
"persistent": phi = 0.98, error rho = 0.97 (LRV/variance ~ 19, i.e. n_eff
~ T/19). QLIKE loss, H0 exact by symmetry, 500 replications (MC s.e. ~1pp at
5%). Rejection rates at 5% / 10% nominal:

=====================  ==========  ==========  ==========  ==========  ==========  ==========
variant                base T=240  base 480    base 1080   pers. 240   pers. 480   pers. 1080
=====================  ==========  ==========  ==========  ==========  ==========  ==========
classic DM+HLN (h-1)   6.4 / 12.4  5.2 / 13.2  5.8 / 11.2  8.8 / 17.6  11.4 / 21   11.4 / 20
NW bw=11, fixed-b      --          7.2 / 14.2  9.6 / 17.0  --          19.0 / 29.6 23.2 / 32.8
NW default bw (48)     4.0 / 12.4  5.4 / 12.2  5.8 / 11.2  5.4 / 12.8  6.2 / 15.0  7.4 / 15.6
EWC B=0.4T^(2/3)       --          3.4 / 8.8   5.8 / 11.4  --          10.2 / 18.6 11.2 / 21.6
**EWC default T/(4h)** 3.6 / 9.0   4.0 / 9.8   5.2 / 10.6  3.8 / 10.0  4.8 / 12.2  6.4 / 12.0
GK boot L=12           --          4.4 / 10.8  8.0 / 16.0  --          14.8 / 25.8 20.4 / 30.8
GK boot default L=36   1.0 / 6.6   3.4 / 8.6   4.6 / 9.4   3.2 / 6.6   4.4 / 12.0  6.4 / 14.6
stationary, no stud.   15.6 / 23.6 9.4 / 17.6  7.4 / 14.8  19.0 / 29.8 13.6 / 22.4 10.8 / 18.0
=====================  ==========  ==========  ==========  ==========  ==========  ==========

Power is low for differences of the size that separate reasonable models.
At 5% (default EWC): base T=480/1080 -- 6%/9% for a true 5% QLIKE skill,
12%/22% for 14% skill, 22%/52% for 22% skill, 39%/82% for 32% skill. In the
persistent design 15%/41% for 28% skill. A 40-model horse race will mostly
fail to separate models whose true mean losses differ by less than ~15-20%.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "DMResult",
    "LRVResult",
    "MCSResult",
    "MZResult",
    "PairwiseResult",
    "SkillResult",
    "InferenceWarning",
    "default_bandwidth",
    "default_ewc_terms",
    "default_block_length",
    "long_run_variance",
    "effective_sample_size",
    "block_bootstrap_indices",
    "dm_test",
    "dm_test_bootstrap",
    "panel_dm_test",
    "model_confidence_set",
    "mincer_zarnowitz",
    "adjust_pvalues",
    "pairwise_dm",
    "skill_score",
]

Kernel = Literal["ewc", "bartlett", "truncated"]
Alternative = Literal["two-sided", "less", "greater"]
BootstrapKind = Literal["circular", "stationary"]

#: Below this many effective observations p-values are flagged as fragile.
MIN_NEFF = 20.0
#: Below this many non-overlapping windows (T / h) results are flagged.
MIN_INDEPENDENT_WINDOWS = 15.0


class InferenceWarning(UserWarning):
    """Raised when a test result should not be trusted at face value."""


# ---------------------------------------------------------------------------
# Default tuning rules (chosen by Monte Carlo; see module docstring)
# ---------------------------------------------------------------------------


def default_ewc_terms(n: int, h: int = 12) -> int:
    """Default number of cosine terms ``B`` (= t degrees of freedom) for the EWC estimator.

    ``B = clip(floor(n / (4 h)), 4, floor(0.4 n^(2/3)))``; 10 for a 40-year
    monthly backtest, 22 for 90 years.

    The Lazarus-Lu-Stock-Watson rule ``0.4 n^(2/3)`` is tuned for AR(1)-type
    persistence and over-rejects here (Monte Carlo: 10-11% at 5% nominal in
    the persistent design). With a 12-step overlap plus persistent volatility
    and persistent model errors, the loss differential's spectrum is
    concentrated at periods longer than several years; the highest cosine
    frequency used, ``B / (2n)`` cycles per month, must stay below that.
    ``B = n/(4h)`` keeps every cosine term's period >= ``8h`` months (8 years).
    The t(B) reference then honestly reflects that the sample holds only about
    ``B`` independent pieces of information about the mean differential.
    """
    if n < 2:
        return 1
    upper = max(4, int(math.floor(0.4 * n ** (2.0 / 3.0))))
    b = int(math.floor(n / (4.0 * max(h, 1))))
    return int(min(max(b, 4), upper, max(n - 1, 1)))


def default_bandwidth(n: int, h: int = 12) -> int:
    """Default Newey-West lag truncation: ``max(4h, ceil(1.3 sqrt(n)))``, capped at ``n-1``.

    ``1.3 sqrt(n)`` is the Lazarus et al. (2018) rule; it is too short here
    (the overlap alone needs lag ``h-1`` with substantial weight, and
    volatility persistence extends much further), so the floor ``4h`` (48
    months) dominates for any realistic backtest length. Used with
    Kiefer-Vogelsang fixed-b critical values.
    """
    return int(min(max(4 * h, math.ceil(1.3 * math.sqrt(max(n, 1)))), max(n - 1, 0)))


def default_block_length(n: int | None = None, h: int = 12) -> int:
    """Default bootstrap (mean) block length: ``max(3h, ceil(n^(1/3)))`` -> 36 months for h=12.

    A block must contain the overlap (``h``) and a good part of the
    volatility persistence. In the Monte Carlo a 12-month block over-rejects
    badly (15-20% at 5% nominal in the persistent design), 24 months
    moderately, 36 months is close to the EWC default. Longer blocks mean
    fewer distinct blocks (30 in a 90-year sample), so do not go much higher.
    """
    base = 3 * h
    if n is not None and n > 0:
        base = max(base, int(math.ceil(n ** (1.0 / 3.0))))
    return int(base)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _to_grid(frame: pd.DataFrame | pd.Series) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Place rows observed in *all* columns on a regular grid.

    Returns ``(values, mask, index)``. ``values`` has shape (n_grid, k) with NaN
    where unobserved; ``mask`` is True where every column is observed. The grid
    runs from the first to the last fully observed row. For a DatetimeIndex
    the grid is month-end; any other index is treated as already regular.
    """
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    frame = frame.astype(float)
    full = frame.notna().all(axis=1) & np.isfinite(frame).all(axis=1)
    if not full.any():
        return np.empty((0, frame.shape[1])), np.zeros(0, bool), frame.index[:0]
    obs = frame.loc[full]
    if isinstance(frame.index, pd.DatetimeIndex):
        obs = obs.sort_index()
        per = obs.index.to_period("M")
        if per.has_duplicates:
            raise ValueError("duplicate months in the loss index")
        grid = pd.period_range(per[0], per[-1], freq="M")
        vals = obs.set_axis(per).reindex(grid).to_numpy()
        idx = grid.to_timestamp(how="end").normalize()
    else:
        first, last = np.flatnonzero(full.to_numpy())[[0, -1]]
        sub = frame.iloc[first : last + 1].where(full.iloc[first : last + 1], np.nan)
        vals = sub.to_numpy()
        idx = sub.index
    mask = np.isfinite(vals).all(axis=1)
    return vals, mask, idx


def _gap_warning(mask: np.ndarray, what: str) -> None:
    if mask.size and (1.0 - mask.mean()) > 0.10:
        warnings.warn(
            f"{what}: {100 * (1 - mask.mean()):.0f}% of the interior monthly grid is missing; "
            "HAC treats gaps by amplitude modulation and bootstraps compress the series. "
            "Both assume the gaps are unrelated to the losses.",
            InferenceWarning,
            stacklevel=3,
        )


def _autocov_am(u: np.ndarray, mask: np.ndarray, max_lag: int) -> np.ndarray:
    """Amplitude-modulated autocovariances gamma_0..gamma_max_lag (divided by n_obs)."""
    z = np.where(mask, u, 0.0)
    n_obs = mask.sum()
    n = z.size
    max_lag = min(max_lag, n - 1)
    # FFT autocovariance for speed on long series
    nfft = 1 << int(math.ceil(math.log2(2 * n + 1)))
    f = np.fft.rfft(z, nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[: max_lag + 1]
    return ac / n_obs


def _ewc_lrv(u: np.ndarray, mask: np.ndarray, B: int) -> float:
    z = np.where(mask, u, 0.0)
    n = z.size
    n_obs = mask.sum()
    t = (np.arange(n) + 0.5) / n
    j = np.arange(1, B + 1)[:, None]
    lam = np.sqrt(2.0 / n_obs) * (np.cos(np.pi * j * t[None, :]) @ z)
    return float(np.mean(lam**2))


def _kv_bartlett_cv(b: float) -> float:
    """Kiefer & Vogelsang (2005) fixed-b 97.5% critical value for the Bartlett t-test."""
    b = min(max(b, 0.0), 1.0)
    return 1.9600 + 2.9694 * b + 0.4160 * b**2 - 0.5324 * b**3


def _t_df_matching(cv: float) -> float:
    """Degrees of freedom nu such that t_nu^{-1}(0.975) = cv (cv > 1.96)."""
    if cv <= 1.9600001:
        return np.inf
    lo, hi = 1.0, 1e7
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if stats.t.ppf(0.975, mid) > cv:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class LRVResult:
    """Long-run variance of a (possibly gappy) series.

    ``lrv`` is the long-run variance sigma^2 such that Var(mean) ~= lrv / n_obs.
    ``df`` is the degrees of freedom of the Student-t reference for a
    t-statistic built on it (``inf`` = normal). ``n_eff = n_obs * gamma0 / lrv``
    (capped at ``n_obs``) is the number of iid observations that would give the
    same precision for the mean.
    """

    lrv: float
    gamma0: float
    df: float
    kernel: str
    bandwidth: float
    n_obs: int
    n_grid: int
    n_eff: float
    fallback: bool = False


def _lrv_core(
    u: np.ndarray,
    mask: np.ndarray,
    *,
    kernel: Kernel,
    h: int,
    bandwidth: int | None,
    ewc_terms: int | None,
) -> LRVResult:
    n_obs = int(mask.sum())
    n = u.size
    if n_obs < 3:
        raise ValueError(f"need at least 3 observations, got {n_obs}")
    u = np.where(mask, u - u[mask].mean(), 0.0)
    gamma0 = float(np.sum(u[mask] ** 2) / n_obs)
    fallback = False
    if kernel == "ewc":
        B = int(ewc_terms) if ewc_terms is not None else default_ewc_terms(n_obs, h)
        B = max(1, min(B, n - 1))
        lrv = _ewc_lrv(u, mask, B)
        df = float(B)
        bw = float(B)
    elif kernel in ("bartlett", "truncated"):
        if kernel == "truncated":
            M = int(bandwidth) if bandwidth is not None else max(h - 1, 0)
        else:
            M = int(bandwidth) if bandwidth is not None else default_bandwidth(n_obs, h)
        M = max(0, min(M, n - 1))
        g = _autocov_am(u, mask, M)
        k = np.arange(1, M + 1)
        if kernel == "truncated":
            lrv = float(g[0] + 2.0 * g[1:].sum())
            df = float(n_obs - 1)
            if lrv <= 0:
                fallback = True
                w = 1.0 - k / (M + 1.0)
                lrv = float(g[0] + 2.0 * np.sum(w * g[1:]))
        else:
            w = 1.0 - k / (M + 1.0)
            lrv = float(g[0] + 2.0 * np.sum(w * g[1:]))
            df = _t_df_matching(_kv_bartlett_cv((M + 1.0) / n_obs))
        bw = float(M)
    else:
        raise ValueError(f"unknown kernel {kernel!r}")
    lrv = max(lrv, 1e-300)
    n_eff = float(min(n_obs, n_obs * gamma0 / lrv)) if gamma0 > 0 else float(n_obs)
    return LRVResult(lrv, gamma0, df, kernel, bw, n_obs, n, n_eff, fallback)


def long_run_variance(
    x: pd.Series | np.ndarray,
    *,
    kernel: Kernel = "ewc",
    h: int = 12,
    bandwidth: int | None = None,
    ewc_terms: int | None = None,
) -> LRVResult:
    """Long-run (HAC) variance of a monthly series, gaps allowed.

    Parameters
    ----------
    x : Series with a month-end DatetimeIndex (gaps/NaN allowed) or 1-d array
        (treated as regular, NaN = gap).
    kernel : ``"ewc"`` (default), ``"bartlett"`` or ``"truncated"``.
    h : overlap horizon; drives the default bandwidth / number of cosine terms.
    bandwidth : lag truncation for ``bartlett``/``truncated``.
    ewc_terms : number of cosine terms ``B`` for ``ewc``.
    """
    s = pd.Series(np.asarray(x, float)) if not isinstance(x, pd.Series) else x
    vals, mask, _ = _to_grid(s)
    return _lrv_core(vals[:, 0], mask, kernel=kernel, h=h, bandwidth=bandwidth, ewc_terms=ewc_terms)


def effective_sample_size(
    x: pd.Series | np.ndarray,
    *,
    h: int = 12,
    kernel: Kernel = "ewc",
    bandwidth: int | None = None,
    ewc_terms: int | None = None,
) -> pd.Series:
    """How many independent observations a loss (or loss-differential) series is worth.

    Returns a Series with

    * ``n_obs`` - observed origins;
    * ``n_nonoverlap`` - ``n_obs / h``, the number of disjoint forecast windows
      (an upper bound on information when losses are MA(h-1) with no other
      persistence);
    * ``n_eff`` - ``n_obs * gamma0 / LRV`` from the chosen HAC estimator, i.e.
      the iid sample size with the same variance of the mean. Capped at
      ``n_obs``. Noisy: the EWC estimate has only ``df`` degrees of freedom.
    * ``acf1``, ``acf12`` - sample autocorrelations at lag 1 and ``h``;
    * ``df`` - degrees of freedom of the HAC estimate.

    Typical values for 12-month QLIKE loss differentials: ``n_eff`` between
    ``n_obs/30`` and ``n_obs/10``, i.e. *fewer* than the number of disjoint
    years once volatility persistence is accounted for.
    """
    s = pd.Series(np.asarray(x, float)) if not isinstance(x, pd.Series) else x
    vals, mask, _ = _to_grid(s)
    r = _lrv_core(vals[:, 0], mask, kernel=kernel, h=h, bandwidth=bandwidth, ewc_terms=ewc_terms)
    u = np.where(mask, vals[:, 0] - np.nanmean(vals[mask, 0]), 0.0)
    g = _autocov_am(u, mask, max(h, 1))
    acf = g / g[0] if g[0] > 0 else np.full_like(g, np.nan)
    return pd.Series(
        {
            "n_obs": r.n_obs,
            "n_nonoverlap": r.n_obs / max(h, 1),
            "n_eff": r.n_eff,
            "acf1": float(acf[1]) if acf.size > 1 else np.nan,
            f"acf{h}": float(acf[h]) if acf.size > h else np.nan,
            "df": r.df,
            "kernel": r.kernel,
        }
    )


# ---------------------------------------------------------------------------
# Block bootstrap indices
# ---------------------------------------------------------------------------


def block_bootstrap_indices(
    n: int,
    n_boot: int,
    block_length: float,
    *,
    kind: BootstrapKind = "stationary",
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Return an (n_boot, n) array of resampling indices.

    ``"circular"``: Politis & Romano (1992) fixed-length blocks wrapped around
    the end. ``"stationary"``: Politis & Romano (1994) geometric block lengths
    with mean ``block_length``, wrapped. Both keep every observation equally
    likely to be drawn, so the bootstrap mean is unbiased for the sample mean.
    """
    rng = np.random.default_rng(rng)
    if n <= 0:
        return np.empty((n_boot, 0), dtype=np.int64)
    if kind == "circular":
        L = int(max(1, round(block_length)))
        k = int(math.ceil(n / L))
        starts = rng.integers(0, n, size=(n_boot, k))
        idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % n
        return idx.reshape(n_boot, k * L)[:, :n]
    if kind == "stationary":
        p = 1.0 / max(float(block_length), 1.0)
        new = rng.random((n_boot, n)) < p
        new[:, 0] = True
        rand_start = rng.integers(0, n, size=(n_boot, n))
        pos = np.arange(n)[None, :]
        last = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
        start_val = np.take_along_axis(rand_start, last, axis=1)
        return (start_val + pos - last) % n
    raise ValueError(f"unknown bootstrap kind {kind!r}")


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DMResult:
    """Result of a test of equal predictive accuracy.

    ``mean_diff`` = mean of ``loss_a - loss_b`` (negative: ``a`` is better).
    ``se`` is the standard error of ``mean_diff``; ``stat`` the test statistic
    (``mean_diff / se``, times the HLN factor for the classic variant); ``df``
    the degrees of freedom of the Student-t reference (``inf`` = normal,
    ``nan`` for bootstrap p-values). ``n_eff`` is the effective number of
    independent observations behind the mean. ``notes`` lists caveats; any
    note starting with ``"WARNING"`` was also emitted as an
    :class:`InferenceWarning`.
    """

    mean_diff: float
    se: float
    stat: float
    pvalue: float
    df: float
    n_obs: int
    n_eff: float
    method: str
    alternative: str
    bandwidth: float
    start: object = None
    end: object = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_series(self) -> pd.Series:
        d = self.to_dict()
        d["notes"] = "; ".join(self.notes)
        return pd.Series(d)


def _pvalue_t(stat: float, df: float, alternative: Alternative) -> float:
    dist = stats.norm if not np.isfinite(df) else stats.t(df)
    if alternative == "two-sided":
        return float(2.0 * dist.sf(abs(stat)))
    if alternative == "less":  # H1: mean_diff < 0 (a better)
        return float(dist.cdf(stat))
    if alternative == "greater":
        return float(dist.sf(stat))
    raise ValueError(f"unknown alternative {alternative!r}")


def _reliability_notes(n_obs: int, n_eff: float, h: int, emit: bool) -> list[str]:
    notes = []
    if n_obs / max(h, 1) < MIN_INDEPENDENT_WINDOWS:
        notes.append(
            f"WARNING: only {n_obs / h:.1f} non-overlapping {h}-step windows; "
            "p-values are unreliable at this length"
        )
    elif n_eff < MIN_NEFF:
        notes.append(
            f"WARNING: effective sample size {n_eff:.1f} < {MIN_NEFF:.0f}; "
            "HAC variance is imprecise and heavy tails can dominate"
        )
    if emit:
        for n in notes:
            warnings.warn(n, InferenceWarning, stacklevel=4)
    return notes


def _pair_frame(loss_a, loss_b) -> pd.DataFrame:
    if isinstance(loss_a, pd.Series) and isinstance(loss_b, pd.Series):
        return pd.concat({"a": loss_a, "b": loss_b}, axis=1)
    return pd.DataFrame({"a": np.asarray(loss_a, float), "b": np.asarray(loss_b, float)})


def dm_test(
    loss_a: pd.Series | np.ndarray,
    loss_b: pd.Series | np.ndarray,
    *,
    h: int = 12,
    kernel: Kernel = "ewc",
    bandwidth: int | None = None,
    ewc_terms: int | None = None,
    hln: bool | None = None,
    alternative: Alternative = "two-sided",
    warn: bool = True,
) -> DMResult:
    """Diebold-Mariano test of equal expected loss, HAC version.

    H0: E[loss_a - loss_b] = 0. ``alternative="less"`` tests H1: ``a`` has
    lower expected loss.

    Assumptions: the loss differential is covariance-stationary with a finite
    long-run variance and the losses come from forecasts whose estimation
    error does not vanish (Giacomini-White view: we test the forecasting
    *methods* as implemented, with their rolling/expanding estimation). Only
    origins where both losses are observed are used.

    Variants (see module docstring): ``kernel="ewc"`` (default, fixed-b
    t(B)); ``"bartlett"`` (Newey-West + Kiefer-Vogelsang fixed-b); and
    ``"truncated"`` (classic DM with rectangular kernel at ``h-1``; with
    ``hln=True``, the default for this kernel, the statistic is multiplied by
    ``sqrt((T + 1 - 2h + h(h-1)/T) / T)`` and compared with ``t(T-1)``, as in
    Harvey, Leybourne & Newbold 1997). The classic variant over-rejects with
    persistent volatility; it is provided for reference only.

    Returns a :class:`DMResult`.
    """
    frame = _pair_frame(loss_a, loss_b)
    vals, mask, idx = _to_grid(frame)
    _gap_warning(mask, "dm_test")
    d = vals[:, 0] - vals[:, 1]
    r = _lrv_core(d, mask, kernel=kernel, h=h, bandwidth=bandwidth, ewc_terms=ewc_terms)
    n = r.n_obs
    dbar = float(np.mean(d[mask]))
    se = math.sqrt(r.lrv / n)
    stat = dbar / se if se > 0 else (0.0 if dbar == 0 else math.copysign(np.inf, dbar))
    notes: list[str] = []
    use_hln = (kernel == "truncated") if hln is None else hln
    method = f"dm_{kernel}"
    if use_hln:
        factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
        if factor > 0:
            stat *= math.sqrt(factor)
        method += "_hln"
    if r.fallback:
        notes.append("rectangular kernel gave a negative variance; fell back to Bartlett at the same lag")
    if kernel == "truncated":
        notes.append(
            "classic DM assumes the differential is MA(h-1); persistent volatility makes it over-reject"
        )
    notes += _reliability_notes(n, r.n_eff, h, warn)
    return DMResult(
        mean_diff=dbar,
        se=se,
        stat=float(stat),
        pvalue=_pvalue_t(stat, r.df, alternative),
        df=r.df,
        n_obs=n,
        n_eff=r.n_eff,
        method=method,
        alternative=alternative,
        bandwidth=r.bandwidth,
        start=idx[mask][0] if n else None,
        end=idx[mask][-1] if n else None,
        notes=tuple(notes),
    )


def _gk_sigma2(d: np.ndarray, L: int) -> float:
    """Gotze-Kunsch overlapping-block variance estimator (circular), = L * var(block means)."""
    n = d.size
    c = np.concatenate([d, d[: L - 1]]) if L > 1 else d
    cs = np.concatenate([[0.0], np.cumsum(c)])
    bm = (cs[L : L + n] - cs[:n]) / L
    return float(L * np.mean((bm - d.mean()) ** 2))


def dm_test_bootstrap(
    loss_a: pd.Series | np.ndarray,
    loss_b: pd.Series | np.ndarray,
    *,
    h: int = 12,
    block_length: int | None = None,
    n_boot: int = 2000,
    bootstrap: BootstrapKind = "circular",
    studentize: bool = True,
    alternative: Alternative = "two-sided",
    seed: int | np.random.Generator | None = None,
    warn: bool = True,
) -> DMResult:
    """Block-bootstrap test of equal expected loss.

    The loss differential (common observed origins, gaps compressed) is
    resampled in blocks; the bootstrap world is recentred at the sample mean
    so it satisfies H0.

    * ``studentize=True`` (default, requires ``bootstrap="circular"``): the
      Götze-Künsch studentized block bootstrap. The statistic is
      ``sqrt(n) dbar / sigma_L`` with ``sigma_L^2`` the overlapping-block
      variance; in each replicate the variance is recomputed from the
      replicate's own blocks. This captures skewness from heavy-tailed
      crisis years that a t-reference ignores.
    * ``studentize=False``: percentile test on the centred bootstrap mean;
      supports the stationary bootstrap.

    ``block_length`` defaults to :func:`default_block_length` (36 months for
    h=12). Blocks shorter than ~3h behave like an under-sized HAC bandwidth
    and over-reject (L=12: 15-20% rejections at 5% nominal in the persistent
    design). The non-studentized variant over-rejects in 20-40 year samples
    (10-19% at 5%) and is provided for comparison only. ``se`` is the
    bootstrap standard deviation of the mean.
    """
    frame = _pair_frame(loss_a, loss_b)
    vals, mask, idx = _to_grid(frame)
    _gap_warning(mask, "dm_test_bootstrap")
    d = (vals[:, 0] - vals[:, 1])[mask]
    n = d.size
    if n < 3:
        raise ValueError("need at least 3 common observations")
    L = int(block_length) if block_length is not None else default_block_length(n, h)
    L = max(1, min(L, n))
    rng = np.random.default_rng(seed)
    dbar = float(d.mean())
    notes: list[str] = []
    if studentize and bootstrap != "circular":
        raise ValueError("studentized bootstrap requires bootstrap='circular'")
    if studentize:
        k = int(math.ceil(n / L))
        starts = rng.integers(0, n, size=(n_boot, k))
        idxb = (starts[:, :, None] + np.arange(L)[None, None, :]) % n  # (B, k, L)
        blocks = d[idxb].mean(axis=2)  # block means (B, k)
        mstar = blocks.mean(axis=1)
        s2star = L * np.mean((blocks - mstar[:, None]) ** 2, axis=1)
        nstar = k * L
        tstar = math.sqrt(nstar) * (mstar - dbar) / np.sqrt(np.maximum(s2star, 1e-300))
        s2 = _gk_sigma2(d, L)
        se = math.sqrt(s2 / n)
        stat = dbar / se if se > 0 else 0.0
        boot = tstar
        sd_boot = float(np.std(mstar))
        method = f"dm_boot_gk_circular_L{L}"
    else:
        ib = block_bootstrap_indices(n, n_boot, L, kind=bootstrap, rng=rng)
        mstar = d[ib].mean(axis=1)
        boot = mstar - dbar
        stat_raw = dbar
        sd_boot = float(np.std(mstar))
        se = sd_boot
        stat = stat_raw / se if se > 0 else 0.0
        boot = boot / se if se > 0 else boot
        method = f"dm_boot_{bootstrap}_L{L}"
    if alternative == "two-sided":
        p = float((1 + np.sum(np.abs(boot) >= abs(stat))) / (n_boot + 1))
    elif alternative == "less":
        p = float((1 + np.sum(boot <= stat)) / (n_boot + 1))
    elif alternative == "greater":
        p = float((1 + np.sum(boot >= stat)) / (n_boot + 1))
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    n_eff = float(min(n, np.var(d) / max(sd_boot**2, 1e-300)))
    if L < 2 * h:
        notes.append(f"block length {L} < 2h={2 * h}: expect over-rejection")
    if not studentize:
        notes.append("non-studentized block bootstrap over-rejects in samples under ~50 years")
    notes += _reliability_notes(n, n_eff, h, warn)
    return DMResult(
        mean_diff=dbar,
        se=float(se),
        stat=float(stat),
        pvalue=p,
        df=np.nan,
        n_obs=n,
        n_eff=n_eff,
        method=method,
        alternative=alternative,
        bandwidth=float(L),
        start=idx[mask][0],
        end=idx[mask][-1],
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Panel DM
# ---------------------------------------------------------------------------


def _panel_diff(panel: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    need = {"origin", "asset", "model", "loss"}
    missing = need - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing columns {sorted(missing)}")
    sub = panel[panel["model"].isin([model_a, model_b])]
    wide = sub.pivot_table(index=["origin", "asset"], columns="model", values="loss", aggfunc="first")
    for m in (model_a, model_b):
        if m not in wide.columns:
            raise KeyError(f"model {m!r} not in panel")
    d = (wide[model_a] - wide[model_b]).dropna()
    return d.unstack("asset").sort_index()  # origin x asset, NaN where unavailable


def panel_dm_test(
    panel: pd.DataFrame,
    model_a: str,
    model_b: str,
    *,
    h: int = 12,
    weights: Literal["obs", "date", "asset"] = "obs",
    method: Literal["hac", "bootstrap"] = "hac",
    kernel: Kernel = "ewc",
    bandwidth: int | None = None,
    ewc_terms: int | None = None,
    block_length: int | None = None,
    n_boot: int = 2000,
    alternative: Alternative = "two-sided",
    seed: int | np.random.Generator | None = None,
    warn: bool = True,
) -> DMResult:
    """Pooled Diebold-Mariano test across assets (unbalanced panels allowed).

    The pooled differential is a weighted mean over (origin, asset) cells
    where *both* models have a loss:

    * ``weights="obs"`` (default): every available cell equally weighted, so
      dates with more assets count more;
    * ``weights="date"``: each origin's cross-sectional mean counts once (each
      date weighted equally, assets equally within a date);
    * ``weights="asset"``: each asset's time-series mean counts once
      (assets with short histories get the same weight as long ones).

    Inference (``method="hac"``) is Driscoll-Kraay: the weighted score of
    each origin, ``s_t = sum_a w_{a,t} (d_{a,t} - dbar)``, aggregates all
    assets at a date so arbitrary cross-asset correlation within (and across
    nearby) dates is allowed; the HAC estimator over time then handles the
    12-month overlap. Thus this is "cluster by time + HAC over time". There is
    no independence assumption across assets: 12 highly correlated assets add
    little information, and ``n_eff`` reports that honestly in units of
    *dates*. ``method="bootstrap"`` resamples whole cross-sections (dates) in
    circular blocks, studentized-free percentile test.

    Assets with different loss scales (e.g. MSE on variances) are weighted
    implicitly by scale; prefer scale-free losses (QLIKE) or normalize first.
    """
    D = _panel_diff(panel, model_a, model_b)
    if isinstance(D.index, pd.DatetimeIndex):
        D.index = pd.DatetimeIndex(D.index)
    A = D.to_numpy(float)
    obs = np.isfinite(A)
    n_cells = int(obs.sum())
    if n_cells < 3:
        raise ValueError("need at least 3 common (origin, asset) cells")
    if weights == "obs":
        W = obs / n_cells
    elif weights == "date":
        nt = obs.sum(axis=1, keepdims=True)
        W = np.where(obs, 1.0 / np.maximum(nt, 1), 0.0)
        W /= (nt[:, 0] > 0).sum()
    elif weights == "asset":
        na = obs.sum(axis=0, keepdims=True)
        W = np.where(obs, 1.0 / np.maximum(na, 1), 0.0)
        W /= (na[0] > 0).sum()
    else:
        raise ValueError(f"unknown weights {weights!r}")
    A0 = np.where(obs, A, 0.0)
    dbar = float(np.sum(W * A0))
    # per-date score scaled so that dbar - mu = mean_t z_t
    has = obs.any(axis=1)
    T_dates = int(has.sum())
    z = T_dates * np.sum(W * np.where(obs, A0 - dbar, 0.0), axis=1)  # length n_dates
    # put the date series on the monthly grid
    zs = pd.Series(np.where(has, z, np.nan), index=D.index)
    vals, mask, idx = _to_grid(zs)
    _gap_warning(mask, "panel_dm_test")
    zz = vals[:, 0]
    notes: list[str] = [f"{D.shape[1]} assets, {n_cells} cells, {T_dates} dates, weights={weights}"]
    if method == "hac":
        r = _lrv_core(zz, mask, kernel=kernel, h=h, bandwidth=bandwidth, ewc_terms=ewc_terms)
        se = math.sqrt(r.lrv / r.n_obs)
        stat = dbar / se if se > 0 else 0.0
        p = _pvalue_t(stat, r.df, alternative)
        df, bw, n_eff = r.df, r.bandwidth, r.n_eff
        meth = f"panel_dk_{kernel}"
        # cross-sectional information: n_eff in date units relative to all cells
        if r.fallback:
            notes.append("rectangular kernel negative; Bartlett fallback")
    elif method == "bootstrap":
        L = int(block_length) if block_length is not None else default_block_length(T_dates, h)
        rng = np.random.default_rng(seed)
        Ad = A0[has]
        Wd = W[has]
        ib = block_bootstrap_indices(T_dates, n_boot, L, kind="circular", rng=rng)
        # re-weight within each replicate so weights sum to one
        num = (Wd * Ad)[ib].sum(axis=(1, 2))
        den = Wd[ib].sum(axis=(1, 2))
        mstar = num / den
        se = float(np.std(mstar))
        stat = dbar / se if se > 0 else 0.0
        boot = (mstar - dbar) / se if se > 0 else mstar - dbar
        if alternative == "two-sided":
            p = float((1 + np.sum(np.abs(boot) >= abs(stat))) / (n_boot + 1))
        elif alternative == "less":
            p = float((1 + np.sum(boot <= stat)) / (n_boot + 1))
        else:
            p = float((1 + np.sum(boot >= stat)) / (n_boot + 1))
        df, bw = np.nan, float(L)
        n_eff = float(min(T_dates, np.var(z[has]) / max(se**2, 1e-300)))
        meth = f"panel_boot_circular_L{L}"
    else:
        raise ValueError(f"unknown method {method!r}")
    notes += _reliability_notes(T_dates, n_eff, h, warn)
    return DMResult(
        mean_diff=dbar,
        se=float(se),
        stat=float(stat),
        pvalue=float(p),
        df=float(df),
        n_obs=T_dates,
        n_eff=float(n_eff),
        method=meth,
        alternative=alternative,
        bandwidth=float(bw),
        start=idx[mask][0],
        end=idx[mask][-1],
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Model Confidence Set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCSResult:
    """Model Confidence Set (Hansen, Lunde & Nason 2011).

    ``included``: models in the MCS at level ``alpha`` (MCS p-value >= alpha).
    ``excluded``: the rest, in elimination order. ``pvalues``: MCS p-value per
    model (Series sorted in elimination order, last survivor has p = 1).
    ``elimination_order``: all models in the order the sequential procedure
    removes them (the last one is the model it never eliminates).
    ``mean_loss``: mean loss on the common sample used.
    """

    included: list[str]
    excluded: list[str]
    pvalues: pd.Series
    elimination_order: list[str]
    alpha: float
    statistic: str
    block_length: float
    bootstrap: str
    n_boot: int
    n_obs: int
    mean_loss: pd.Series
    dropped_models: list[str] = field(default_factory=list)
    notes: tuple[str, ...] = ()

    def summary(self) -> pd.DataFrame:
        out = pd.DataFrame({"mean_loss": self.mean_loss, "mcs_pvalue": self.pvalues})
        out["in_mcs"] = out["mcs_pvalue"] >= self.alpha
        out["elimination_rank"] = pd.Series(
            {m: i + 1 for i, m in enumerate(self.elimination_order)}
        )
        return out.sort_values("mcs_pvalue", ascending=False)


def _mcs_step_stats(statistic, mu, C, a, scale):
    """Studentized MCS statistics for alive set ``a``.

    ``mu`` (..., m) mean losses (or centred bootstrap deviations), ``C`` (..., m, m)
    long-run covariance (times 1/n via ``scale``). Returns (stat, per-model
    elimination score).
    """
    ma = mu[..., a]
    Ca = C[..., a[:, None], a[None, :]]
    if statistic == "R":
        d = ma[..., :, None] - ma[..., None, :]
        dg = np.diagonal(Ca, axis1=-2, axis2=-1)
        v = dg[..., :, None] + dg[..., None, :] - 2.0 * Ca
        sd = np.sqrt(np.maximum(v, 0.0) * scale)
        t = np.divide(d, sd, out=np.zeros_like(d), where=sd > 0)
        return np.max(np.abs(t), axis=(-1, -2)), np.max(t, axis=-1)
    k = len(a)
    d = ma - ma.mean(axis=-1, keepdims=True)
    rs = Ca.sum(axis=-1)
    v = np.diagonal(Ca, axis1=-2, axis2=-1) - 2.0 * rs / k + rs.sum(axis=-1, keepdims=True) / k**2
    sd = np.sqrt(np.maximum(v, 0.0) * scale)
    t = np.divide(d, sd, out=np.zeros_like(d), where=sd > 0)
    return np.max(t, axis=-1), t


def model_confidence_set(
    losses: pd.DataFrame,
    alpha: float = 0.10,
    *,
    statistic: Literal["R", "max"] = "R",
    variance: Literal["gk", "bootstrap"] = "gk",
    block_length: int | None = None,
    n_boot: int = 2000,
    bootstrap: BootstrapKind | None = None,
    h: int = 12,
    min_coverage: float = 0.0,
    seed: int | np.random.Generator | None = None,
    warn: bool = True,
) -> MCSResult:
    """Model Confidence Set by block bootstrap (Hansen, Lunde & Nason 2011).

    Sequentially tests "all models in the current set have equal expected
    loss" and, on rejection, eliminates the worst model. The MCS p-value of a
    model is the largest test p-value up to its elimination; models with
    p-value >= ``alpha`` form the set, which contains the best model(s) with
    probability >= 1 - alpha (asymptotically).

    Parameters
    ----------
    losses : origins x models; lower is better.
    alpha : MCS level.
    statistic : ``"R"`` (default) - range statistic max_{i,j} |t_ij| over
        pairwise studentized differences; ``"max"`` - max_i t_i with each
        model compared with the set average.
    variance : how the differences are studentized.

        ``"gk"`` (default): Götze-Künsch studentized circular block
        bootstrap. The sample statistic uses the overlapping-block long-run
        covariance of the losses; each bootstrap replicate recomputes the
        covariance from its own blocks, so the bootstrap distribution
        includes the noise of the variance estimate. Requires
        ``bootstrap="circular"`` (the default for this choice).

        ``"bootstrap"``: the original HLN (2011) / ``arch.bootstrap.MCS``
        recipe - variances are the bootstrap variances of the mean
        differences and the bootstrap statistic reuses them (stationary
        bootstrap by default). **Over-rejects badly here**: with 10 equally
        good models and 40 years of 12-month overlapping losses it excludes
        at least one model ~40-45% of the time at alpha = 0.10 (arch's
        implementation does the same). Kept for comparison.
    block_length : block length in months (mean length for stationary).
        Default :func:`default_block_length` (36 for h=12). Values below
        ``h`` are warned about.
    bootstrap : ``"circular"`` or ``"stationary"``; default circular for
        ``variance="gk"`` and stationary for ``variance="bootstrap"``.
    min_coverage : the MCS needs a common sample. Models observed on fewer
        than this fraction of origins (relative to the best-covered model)
        are dropped before forming it (reported in ``dropped_models``). With
        the default 0 nothing is dropped and the common sample is the
        intersection of all models' histories, which can be short.

    Notes
    -----
    Monte Carlo (10 models, see ``tests/eval/test_inference.py``): with the
    default settings the set keeps all 10 equally good models in roughly
    85-90% of samples at alpha = 0.10 and contains the best of 10 graded models
    in >= 95%; models that are 30-40% worse in QLIKE are excluded in most
    samples only with 90 years of data. See the report for the full table.
    """
    if not isinstance(losses, pd.DataFrame):
        raise TypeError("losses must be a DataFrame (origins x models)")
    if variance not in ("gk", "bootstrap"):
        raise ValueError(f"unknown variance {variance!r}")
    if statistic not in ("R", "max"):
        raise ValueError(f"unknown statistic {statistic!r}")
    if bootstrap is None:
        bootstrap = "circular" if variance == "gk" else "stationary"
    if variance == "gk" and bootstrap != "circular":
        raise ValueError("variance='gk' requires bootstrap='circular'")
    L0 = losses.astype(float)
    dropped: list[str] = []
    if min_coverage > 0:
        cov = L0.notna().sum()
        keep = cov >= min_coverage * cov.max()
        dropped = list(cov.index[~keep])
        L0 = L0.loc[:, keep]
    vals, mask, idx = _to_grid(L0)
    X = vals[mask]
    n, m = X.shape
    names = list(L0.columns)
    notes: list[str] = []
    n_full = int(L0.notna().any(axis=1).sum())
    if n < n_full:
        notes.append(f"common sample uses {n} of {n_full} origins")
    if n < 3:
        raise ValueError("fewer than 3 origins where all models are observed")
    mu = X.mean(axis=0)
    if m < 2:
        pv = pd.Series(1.0, index=names)
        return MCSResult(names, [], pv, names, alpha, statistic, np.nan, bootstrap, 0, n, pd.Series(mu, index=names), dropped, tuple(notes))
    Lb = int(block_length) if block_length is not None else default_block_length(n, h)
    Lb = max(1, min(Lb, n))
    if Lb < h:
        notes.append(f"WARNING: block length {Lb} < h={h}; the MCS will be too small")
        if warn:
            warnings.warn(notes[-1], InferenceWarning, stacklevel=2)
    rng = np.random.default_rng(seed)
    Xc = X - mu
    if variance == "gk":
        k = int(math.ceil(n / Lb))
        ext = np.concatenate([Xc, Xc[: Lb - 1]], axis=0) if Lb > 1 else Xc
        cs = np.concatenate([np.zeros((1, m)), np.cumsum(ext, axis=0)], axis=0)
        obm = (cs[Lb : Lb + n] - cs[:n]) / Lb  # overlapping block means (n, m), mean zero
        C = Lb * (obm.T @ obm) / n  # GK long-run covariance
        starts = rng.integers(0, n, size=(n_boot, k))
        bm = obm[starts]  # (B, k, m) block means of centred data
        dev = bm.mean(axis=1)  # (B, m) = bootstrap mean - sample mean
        bmc = bm - dev[:, None, :]
        Cb = Lb * np.einsum("bki,bkj->bij", bmc, bmc) / k
        scale, scale_b = 1.0 / n, 1.0 / (k * Lb)
    else:
        ib = block_bootstrap_indices(n, n_boot, Lb, kind=bootstrap, rng=rng)
        dev = np.empty((n_boot, m))
        step = max(1, int(2e7 // max(n * m, 1)))
        for s0 in range(0, n_boot, step):
            dev[s0 : s0 + step] = Xc[ib[s0 : s0 + step]].mean(axis=1)
        C = dev.T @ dev / n_boot  # bootstrap covariance of the means
        Cb = C[None]
        scale = scale_b = 1.0
    alive = list(range(m))
    order: list[int] = []
    step_p: list[float] = []
    while len(alive) > 1:
        a = np.array(alive)
        t_obs, score = _mcs_step_stats(statistic, mu, C, a, scale)
        t_boot, _ = _mcs_step_stats(statistic, dev, Cb, a, scale_b)
        p = float((1 + np.sum(t_boot >= t_obs)) / (n_boot + 1))
        step_p.append(p)
        elim_local = int(np.argmax(score))
        order.append(alive[elim_local])
        alive.pop(elim_local)
    order.append(alive[0])
    step_p.append(1.0)
    mcs_p = np.maximum.accumulate(np.array(step_p))
    elim_names = [names[i] for i in order]
    pvals = pd.Series(mcs_p, index=elim_names, name="mcs_pvalue")
    included = [nm for nm in elim_names if pvals[nm] >= alpha]
    excluded = [nm for nm in elim_names if pvals[nm] < alpha]
    if n / max(h, 1) < MIN_INDEPENDENT_WINDOWS:
        notes.append(f"WARNING: only {n / h:.1f} non-overlapping windows in the common sample")
        if warn:
            warnings.warn(notes[-1], InferenceWarning, stacklevel=2)
    return MCSResult(
        included=included,
        excluded=excluded,
        pvalues=pvals,
        elimination_order=elim_names,
        alpha=alpha,
        statistic=statistic,
        block_length=float(Lb),
        bootstrap=f"{bootstrap}/{variance}",
        n_boot=n_boot,
        n_obs=n,
        mean_loss=pd.Series(mu, index=names),
        dropped_models=dropped,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Mincer-Zarnowitz
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MZResult:
    """Mincer-Zarnowitz regression ``y = a + b f + e``.

    ``params``/``se``/``tstat`` are Series indexed ``["intercept", "slope"]``;
    ``t`` tests for the slope are against 1 (``tstat["slope"]`` is
    ``(b - 1)/se``). ``wald``/``wald_pvalue`` test (a, b) = (0, 1) jointly with
    an F(2, df-1) reference under EWC (exact under fixed-b) or chi2(2)/2
    scaling for other kernels.
    """

    params: pd.Series
    se: pd.Series
    tstat: pd.Series
    pvalue: pd.Series
    wald: float
    wald_pvalue: float
    df: float
    r2: float
    n_obs: int
    n_eff: float
    log: bool
    kernel: str
    notes: tuple[str, ...] = ()


def _hac_matrix(scores: np.ndarray, mask: np.ndarray, *, kernel: Kernel, h: int, bandwidth, ewc_terms):
    """HAC long-run covariance of a (n, k) score matrix with amplitude modulation."""
    n, k = scores.shape
    n_obs = int(mask.sum())
    U = np.where(mask[:, None], scores, 0.0)
    if kernel == "ewc":
        B = int(ewc_terms) if ewc_terms is not None else default_ewc_terms(n_obs, h)
        t = (np.arange(n) + 0.5) / n
        C = np.cos(np.pi * np.arange(1, B + 1)[:, None] * t[None, :])
        lam = np.sqrt(2.0 / n_obs) * (C @ U)  # (B, k)
        return lam.T @ lam / B, float(B)
    M = int(bandwidth) if bandwidth is not None else (default_bandwidth(n_obs, h) if kernel == "bartlett" else h - 1)
    M = min(M, n - 1)
    S = U.T @ U / n_obs
    for j in range(1, M + 1):
        G = U[j:].T @ U[:-j] / n_obs
        w = 1.0 - j / (M + 1.0) if kernel == "bartlett" else 1.0
        S += w * (G + G.T)
    if kernel == "bartlett":
        df = _t_df_matching(_kv_bartlett_cv((M + 1.0) / n_obs))
    else:
        df = float(n_obs - 1)
        if np.any(np.linalg.eigvalsh(S) <= 0):
            S = _hac_matrix(scores, mask, kernel="bartlett", h=h, bandwidth=M, ewc_terms=None)[0]
    return S, df


def mincer_zarnowitz(
    realized: pd.Series,
    forecast: pd.Series,
    *,
    log: bool = False,
    h: int = 12,
    kernel: Kernel = "ewc",
    bandwidth: int | None = None,
    ewc_terms: int | None = None,
    warn: bool = True,
) -> MZResult:
    """Mincer-Zarnowitz efficiency regression with HAC standard errors.

    Regresses ``realized`` on ``forecast`` (aligned by origin; both must be
    positive when ``log=True``). Use variances (or vols) in levels for the
    unbiasedness test: with a noisy but *unbiased* realized proxy, an optimal
    forecast has (a, b) = (0, 1). In logs, Jensen's inequality and proxy noise
    shift the intercept: E[log RV | f] = log f - c with c > 0, so in logs read
    the **slope** (b = 1: forecasts move one-for-one with outcomes; b < 1:
    forecasts too dispersed) and treat the joint test as a test of
    "log-unbiasedness", which an otherwise good forecast may fail. Levels
    regressions are dominated by crisis years (heavy tails): the log version
    is the more robust slope diagnostic.

    Standard errors: HAC on the regression scores x_t e_t with the chosen
    kernel (default EWC, t(B)/F(2, B-1) references, Lazarus et al. 2018).
    """
    frame = pd.concat({"y": realized, "f": forecast}, axis=1) if isinstance(realized, pd.Series) else pd.DataFrame(
        {"y": np.asarray(realized, float), "f": np.asarray(forecast, float)}
    )
    frame = frame.astype(float)
    notes: list[str] = []
    if log:
        bad = (frame <= 0).any(axis=1)
        if bad.any():
            notes.append(f"dropped {int(bad.sum())} non-positive rows for log regression")
        frame = np.log(frame.where(~bad))
    vals, mask, idx = _to_grid(frame)
    _gap_warning(mask, "mincer_zarnowitz")
    y = np.where(mask, vals[:, 0], 0.0)
    f = np.where(mask, vals[:, 1], 0.0)
    n = int(mask.sum())
    if n < 5:
        raise ValueError("need at least 5 observations")
    X = np.column_stack([np.ones_like(f), f])[mask]
    yy = y[mask]
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ yy
    e = yy - X @ beta
    scores = np.zeros((vals.shape[0], 2))
    scores[mask] = X * e[:, None]
    S, df = _hac_matrix(scores, mask, kernel=kernel, h=h, bandwidth=bandwidth, ewc_terms=ewc_terms)
    Q = XtX_inv * n
    V = Q @ S @ Q / n
    se = np.sqrt(np.diag(V))
    r = beta - np.array([0.0, 1.0])
    t = np.array([beta[0] / se[0], r[1] / se[1]])
    pv = np.array([_pvalue_t(ti, df, "two-sided") for ti in t])
    W = float(r @ np.linalg.solve(V, r))
    if np.isfinite(df) and df > 2:
        F = W * (df - 2 + 1) / (df * 2)  # Hotelling-type small-sample scaling
        wald_p = float(stats.f.sf(F, 2, df - 1))
    else:
        wald_p = float(stats.chi2.sf(W, 2))
    r2 = float(1 - np.sum(e**2) / np.sum((yy - yy.mean()) ** 2))
    n_eff = _lrv_core(np.where(mask, y - f, 0.0), mask, kernel=kernel, h=h, bandwidth=bandwidth, ewc_terms=ewc_terms).n_eff
    notes += _reliability_notes(n, n_eff, h, warn)
    if log:
        notes.append("log regression: intercept < 0 expected from Jensen/proxy noise; judge the slope")
    names = ["intercept", "slope"]
    return MZResult(
        params=pd.Series(beta, index=names),
        se=pd.Series(se, index=names),
        tstat=pd.Series(t, index=names),
        pvalue=pd.Series(pv, index=names),
        wald=W,
        wald_pvalue=wald_p,
        df=df,
        r2=r2,
        n_obs=n,
        n_eff=float(n_eff),
        log=log,
        kernel=kernel,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------


def adjust_pvalues(p: pd.Series | Sequence[float], method: Literal["holm", "bonferroni", "bh", "none"] = "holm"):
    """Multiplicity-adjusted p-values (NaN ignored).

    ``"holm"`` (default) controls the family-wise error rate under arbitrary
    dependence - appropriate here because all comparisons share the same
    crisis years and are strongly dependent. ``"bh"`` (Benjamini-Hochberg)
    controls the false discovery rate under positive dependence (PRDS);
    ``"bonferroni"`` is the most conservative.
    """
    s = pd.Series(p, dtype=float)
    ok = s.notna()
    q = s[ok].to_numpy()
    k = q.size
    out = np.full(k, np.nan)
    if k:
        order = np.argsort(q)
        qs = q[order]
        if method == "none":
            adj = qs
        elif method == "bonferroni":
            adj = np.minimum(qs * k, 1.0)
        elif method == "holm":
            adj = np.minimum(np.maximum.accumulate(qs * (k - np.arange(k))), 1.0)
        elif method == "bh":
            adj = np.minimum.accumulate((qs * k / np.arange(1, k + 1))[::-1])[::-1]
            adj = np.minimum(adj, 1.0)
        else:
            raise ValueError(f"unknown method {method!r}")
        out[order] = adj
    res = s.copy()
    res[ok] = out
    return res


@dataclass(frozen=True)
class PairwiseResult:
    """All-pairs DM comparison.

    Matrices are models x models, entry (i, j) refers to ``loss_i - loss_j``
    (negative ``mean_diff``/``stat``: row model better). ``pvalue`` is
    unadjusted; ``pvalue_adj`` is adjusted across the m(m-1)/2 distinct pairs.
    ``vs_benchmark`` (if a benchmark was given) has one row per model with
    the comparison against the benchmark, p-values adjusted across the m-1
    benchmark comparisons only, and the skill (percentage loss reduction).
    """

    mean_diff: pd.DataFrame
    stat: pd.DataFrame
    pvalue: pd.DataFrame
    pvalue_adj: pd.DataFrame
    n_obs: pd.DataFrame
    vs_benchmark: pd.DataFrame | None
    adjust: str


def pairwise_dm(
    losses: pd.DataFrame,
    *,
    benchmark: str | None = None,
    adjust: Literal["holm", "bonferroni", "bh", "none"] = "holm",
    h: int = 12,
    kernel: Kernel = "ewc",
    bandwidth: int | None = None,
    ewc_terms: int | None = None,
    models: Iterable[str] | None = None,
) -> PairwiseResult:
    """Pairwise DM tests among all models (each pair on its own common sample).

    Each pair uses its own overlap of origins, so different cells can rest on
    different periods - compare ``n_obs`` before reading across a row.
    Reliability warnings are suppressed per pair and collected in the
    ``vs_benchmark`` notes column instead.
    """
    cols = list(models) if models is not None else list(losses.columns)
    m = len(cols)
    md = pd.DataFrame(np.nan, index=cols, columns=cols)
    st, pv, no = md.copy(), md.copy(), md.copy()
    pair_p = {}
    for a, b in combinations(cols, 2):
        try:
            r = dm_test(losses[a], losses[b], h=h, kernel=kernel, bandwidth=bandwidth, ewc_terms=ewc_terms, warn=False)
        except ValueError:
            continue
        md.loc[a, b], md.loc[b, a] = r.mean_diff, -r.mean_diff
        st.loc[a, b], st.loc[b, a] = r.stat, -r.stat
        pv.loc[a, b] = pv.loc[b, a] = r.pvalue
        no.loc[a, b] = no.loc[b, a] = r.n_obs
        pair_p[(a, b)] = r.pvalue
    adj = adjust_pvalues(pd.Series(list(pair_p.values()), dtype=float), adjust)
    pa = md * np.nan
    for (a, b), q in zip(pair_p, adj):
        pa.loc[a, b] = pa.loc[b, a] = q
    vsb = None
    if benchmark is not None:
        if benchmark not in losses.columns:
            raise KeyError(benchmark)
        rows = {}
        for a in cols:
            if a == benchmark:
                continue
            try:
                r = dm_test(losses[a], losses[benchmark], h=h, kernel=kernel, bandwidth=bandwidth, ewc_terms=ewc_terms, warn=False)
            except ValueError:
                continue
            common = losses[[a, benchmark]].dropna()
            mb = common[benchmark].mean()
            rows[a] = {
                "mean_loss": common[a].mean(),
                "bench_loss": mb,
                "mean_diff": r.mean_diff,
                "skill_pct": 100.0 * (1.0 - common[a].mean() / mb) if mb != 0 else np.nan,
                "stat": r.stat,
                "pvalue": r.pvalue,
                "n_obs": r.n_obs,
                "n_eff": r.n_eff,
                "df": r.df,
                "notes": "; ".join(n for n in r.notes if n.startswith("WARNING")),
            }
        vsb = pd.DataFrame.from_dict(rows, orient="index")
        if len(vsb):
            vsb["pvalue_adj"] = adjust_pvalues(vsb["pvalue"], adjust).to_numpy()
            vsb = vsb.sort_values("mean_diff")
    return PairwiseResult(md, st, pv, pa, no, vsb, adjust)


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillResult:
    """Percentage reduction in mean loss versus a benchmark.

    ``skill = 100 * (1 - mean(loss_model) / mean(loss_bench))`` on the common
    sample: positive means the model's average loss is lower. ``ci_low``/
    ``ci_high`` is the two-sided ``level`` confidence interval.
    """

    skill: float
    ci_low: float
    ci_high: float
    level: float
    method: str
    n_obs: int
    n_eff: float
    mean_loss: float
    bench_loss: float
    notes: tuple[str, ...] = ()


def skill_score(
    loss_model: pd.Series | np.ndarray,
    loss_bench: pd.Series | np.ndarray,
    *,
    level: float = 0.90,
    method: Literal["fieller", "bootstrap", "hac"] = "fieller",
    h: int = 12,
    block_length: int | None = None,
    n_boot: int = 2000,
    bootstrap: BootstrapKind = "circular",
    kernel: Kernel = "ewc",
    ewc_terms: int | None = None,
    seed: int | np.random.Generator | None = None,
    warn: bool = True,
) -> SkillResult:
    """Skill = percentage reduction in mean loss vs a benchmark, with a CI.

    ``skill = 100 (1 - R)``, ``R = mean(l_model) / mean(l_bench)`` on the
    common sample of origins.

    ``method``:

    * ``"fieller"`` (default): invert the HAC t-test of
      ``E[l_model - R0 l_bench] = 0`` over R0 (Fieller 1954), with the joint
      long-run covariance of the two losses from ``kernel`` (default EWC,
      t(B) quantiles). Exact treatment of the ratio, no linearization. If the
      benchmark mean is itself not significantly different from zero at the
      chosen level the set is unbounded; ``ci_low``/``ci_high`` are then
      -inf/+inf and a warning is emitted.
    * ``"hac"``: delta method, ``z_t = (l_m,t - R l_b,t) / mean(l_b)``, HAC
      variance, t(df) quantiles. Symmetric; under-covers (80-84% for a
      nominal 90% interval in the Monte Carlo).
    * ``"bootstrap"``: circular/stationary block bootstrap of origins (both
      series jointly), percentile interval. Under-covers badly with
      persistent losses (64-81% for nominal 90%), because the block
      bootstrap understates the variance and the ratio is skewed.

    Monte Carlo coverage of nominal 90% intervals (true skill 14-18%):
    Fieller-EWC 90-92% for 40 and 90 years in both designs.

    Assumes losses are non-negative with a positive benchmark mean (QLIKE,
    MSE). Skill is a ratio of means, so it is dominated by the periods with
    the largest benchmark losses (crisis years); report it alongside the DM
    test, not instead of it.
    """
    frame = _pair_frame(loss_model, loss_bench)
    vals, mask, idx = _to_grid(frame)
    _gap_warning(mask, "skill_score")
    lm, lb = vals[mask, 0], vals[mask, 1]
    n = lm.size
    if n < 3:
        raise ValueError("need at least 3 common observations")
    mm, mb = lm.mean(), lb.mean()
    if mb <= 0:
        raise ValueError("benchmark mean loss must be positive")
    R = mm / mb
    skill = 100.0 * (1.0 - R)
    a = 1.0 - level
    notes: list[str] = []
    z = (vals[:, 0] - R * vals[:, 1]) / mb
    lr = _lrv_core(np.where(mask, z, 0.0), mask, kernel=kernel, h=h, bandwidth=None, ewc_terms=ewc_terms)
    n_eff = lr.n_eff
    if method == "fieller":
        X = np.column_stack(
            [np.where(mask, vals[:, 0] - mm, 0.0), np.where(mask, vals[:, 1] - mb, 0.0)]
        )
        S, df = _hac_matrix(X, mask, kernel=kernel, h=h, bandwidth=None, ewc_terms=ewc_terms)
        q = stats.t.ppf(1 - a / 2, df) if np.isfinite(df) else stats.norm.ppf(1 - a / 2)
        qa = q * q / n
        A = mb * mb - qa * S[1, 1]
        Bq = -2.0 * (mm * mb - qa * S[0, 1])
        Cq = mm * mm - qa * S[0, 0]
        disc = Bq * Bq - 4 * A * Cq
        if A <= 0 or disc < 0:
            lo, hi = -np.inf, np.inf
            notes.append(
                "WARNING: Fieller set unbounded - the benchmark's mean loss is not "
                "precisely estimated; skill is not identified at this level"
            )
            if warn:
                warnings.warn(notes[-1], InferenceWarning, stacklevel=2)
        else:
            r1 = (-Bq - math.sqrt(disc)) / (2 * A)
            r2 = (-Bq + math.sqrt(disc)) / (2 * A)
            lo, hi = 100.0 * (1.0 - r2), 100.0 * (1.0 - r1)
        meth = f"fieller_{kernel}"
    elif method == "bootstrap":
        L = int(block_length) if block_length is not None else default_block_length(n, h)
        ib = block_bootstrap_indices(n, n_boot, L, kind=bootstrap, rng=np.random.default_rng(seed))
        Rb = lm[ib].mean(axis=1) / lb[ib].mean(axis=1)
        sk = 100.0 * (1.0 - Rb)
        lo, hi = np.quantile(sk, [a / 2, 1 - a / 2])
        meth = f"bootstrap_{bootstrap}_L{L}_percentile"
    elif method == "hac":
        se = math.sqrt(lr.lrv / lr.n_obs)
        q = stats.t.ppf(1 - a / 2, lr.df) if np.isfinite(lr.df) else stats.norm.ppf(1 - a / 2)
        lo, hi = skill - 100 * q * se, skill + 100 * q * se
        meth = f"delta_{kernel}"
    else:
        raise ValueError(f"unknown method {method!r}")
    notes += _reliability_notes(n, n_eff, h, warn)
    return SkillResult(float(skill), float(lo), float(hi), level, meth, n, float(n_eff), float(mm), float(mb), tuple(notes))
