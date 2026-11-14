# GARCH family (`garch_*`, kind `vol`)

Package `src/assetforecast/models/garch/` (`core.py` numerics, `prep.py` inputs on
the 21-day scale, `models.py` forecasters), tests `tests/models/test_garch.py`.
All numbers below are development results (`ASSETFORECAST_MAX_DATE=2014-12-31`,
origins to 2013-12 with observed targets, `results/dev/`). Nothing after 2014
was looked at.

## The question and the design

At a 12-month horizon a GARCH(1,1) has mean-reverted almost completely to its
long-run variance, so a GARCH forecast of the 12-month average variance is
mostly (a) the long-run level and (b) any slowly moving component. The family
therefore spans: plain baselines (GARCH(1,1), GJR), models with a slow
component (Engle-Lee component GARCH, GARCH-MIDAS, RiskMetrics-2006 long
memory), and three sampling frequencies (daily, 5-day, monthly).

Common to every member:

* **Returns**: log excess returns, zero mean (uncentered, like the target).
* **Innovations**: standardized Student-t, nu in [2.5, 60] estimated (1929-33,
  1987-10-19, the 1980 gold days and the 2008 ETF reversals are in the
  sample; t innovations keep them from dominating the parameter estimates).
* **Variance targeting**: the long-run variance is not a free parameter; it is
  the mean square of the estimation/filtering window. omega = (1 - p) vbar
  (GARCH), omega = (1 - rho) vbar (component GARCH).
* **Re-estimation once a year**: parameters are estimated on data through the
  last December 31 on or before the origin and cached under a hash of the
  estimation data; between re-estimations the state (and vbar) is filtered
  with fixed parameters up to the origin. Cold and warm calls give identical
  results (tested).
* **Forecast** = SUM of the multi-step conditional variances over the next
  12 months' steps, in closed form. H = the asset's own number of trading days
  in the trailing 12 months (about 295 before 1952, 252 after), H/5 for the
  5-day model, 12 for monthly models. Annualized vol = sqrt(sum).
* **Robustness**: bounded parameters (L-BFGS-B box, orderings built into the
  parameterization), three conventional starting values, best finite optimum
  kept; if every start fails, the first start (a conventional
  parameterization) is used. A daily/5-day model with fewer than 3 years of
  data in its window at the last re-estimation, or whose daily data stop more
  than a month before the origin, uses the pooled monthly GARCH(1,1) for that
  asset; if that fails, 12 x mean squared monthly return of the last 5 years;
  then 15%. The harness validated every one of the 1,116 stored origins of
  every model (no NaN, no failure). All 168 parameter fits inspected for the
  report converged (`ok`); none fell back to defaults.

### Mapping daily returns to the 21-day scale (`prep.py`)

Daily variance depends on the pricing convention (smoothed NAVs too low,
ETFs of foreign/illiquid assets too high) and changes at splices. Every daily
return is rescaled BEFORE fitting by a causal, segment-local variance ratio

    VR_k[i] = EWmean(s21^2 / 21) / EWmean(s_k^2 / k)        (rows <= i, current segment)
    log VR*_k = w log VR_k,  w = n_eff / (n_eff + n0),  n0 = 2*21 / sd_k^2
    r_adj[i] = r[i] * sqrt(VR*_1[i-1])

s_m are overlapping m-row sums (NaN day = 0, as in the target), exponential
weights with a 3-year half-life, n_eff the Kish effective number of 21-day
sums, 2*21/n the iid sampling variance of a log variance ratio, sd_1 = 0.30
and sd_5 = 0.15 the cross-segment spread of the log variance ratio implied by
`reports/data/horizon_evidence.md` (1-day/21-day vol ratios 0.69-1.20; 5-day
0.84-1.08); log VR is clipped to [log 1/3, log 3]. The EW estimate restarts at
every change of `view.tiers` (no pre-splice scale crosses a tier change);
source changes that keep the tier (VGSIX -> VNQ, VEIEX -> EEM, GSW -> IEF) are
absorbed within a few years by the half-life. So the fitted series is "daily
returns on the 21-day convention", and the models' parameters are not
estimated across a level break. The 5-day model uses the same machinery with
k = 5: non-overlapping 5-row sums anchored at the origin (estimation: at the
re-estimation date), each scaled by sqrt(VR*_5) known before the block.

VR* actually applied (21-day/1-day variance ratio, year-end):

| asset | 1940 | 1970 | 1990 | 2000 | 2008 | 2014 |
|---|---|---|---|---|---|---|
| us_large | 1.17 | 1.48 | 0.98 | 0.69 | 0.69 | 0.78 |
| us_small | 1.46 | 2.17 | 1.96 | 0.87 | 0.80 | 0.82 |
| intl_dev |  |  | 1.32 | 1.16 | 0.72 | 0.75 |
| em |  |  |  | 1.46 | 0.68 | 0.73 |
| ust_int |  | 1.89 | 1.34 | 1.09 | 1.10 | 0.93 |
| ust_long |  | 1.44 | 1.24 | 1.01 | 1.21 | 0.86 |
| tips |  |  |  | 0.93 | 0.97 | 0.96 |
| ig_credit |  |  | 1.25 | 0.92 | 0.76 | 0.90 |
| hy_credit |  |  | 1.81 | 1.74 | 0.94 | 1.02 |
| reits |  |  | 1.33 | 1.45 | 0.69 | 0.72 |
| gold |  |  | 0.99 | 1.03 | 0.96 | 0.90 |
| commodities |  |  |  | 1.08 | 1.22 | 1.05 |

The signs match the data reports: smoothed NAVs and 1960s-80s bond
reconstructions get VR > 1 (hy_credit VWEHX 1.8, us_small pre-2000 2.0,
reits FRESX/VGSIX 1.3-1.5), ETFs get VR < 1 (EFA, EEM, VNQ ~0.7). us_large
SPY at 0.69-0.78 reflects negative daily autocorrelation of the S&P in the
2000s, a real feature of 21-day risk that a raw daily GARCH would miss.

## Models

Notation: e = VR-adjusted return of one step, s2 = conditional variance, vbar
= window mean of e^2, H = steps in the next 12 months, G(p, H) = (1 - p^H)/(1 - p).

1. **`garch_d_garch11`** - daily GARCH(1,1)-t, 20-year window.
   s2[t+1] = (1-p) vbar + a e[t]^2 + b s2[t], p = a + b.
   Forecast: H vbar + (s2[t+1] - vbar) G(p, H).
2. **`garch_d_gjr`** - daily GJR-GARCH(1,1)-t: adds g e[t]^2 1[e[t] < 0];
   p = a + g/2 + b. Same forecast formula.
3. **`garch_d_cgarch`** - daily Engle-Lee (1999) component GARCH-t.
   q[t+1] = (1-rho) vbar + rho q[t] + phi (e[t]^2 - s2[t]);
   s2[t+1] = q[t+1] + a (e[t]^2 - q[t]) + b (s2[t] - q[t]).
   Constraints 1 > rho > a+b, b > phi > 0 by construction; rho half-life
   bounded to [3 months, 50 years].
   Forecast: H vbar + (q[t+1] - vbar) G(rho, H) + (s2[t+1] - q[t+1]) G(a+b, H).
4. **`garch_d_midas`** - GARCH-MIDAS (Engle, Ghysels & Sohn 2013), rolling
   form: e = sqrt(tau_m g) z; g[t+1] = (1-p) + a e[t]^2/tau + b g[t];
   log tau_m = log vbar + m0 + theta (log X_{m-1} - log vbar), X = beta-lag
   weighted average (K = 36 months, single-parameter beta weights
   (1 - (k-0.5)/K)^(w-1)) of the per-day realized variance of the previous
   months. tau for the next month is known at the origin; it is held
   constant over the year (standard practice). Forecast:
   tau_next (H + (g[t+1] - 1) G(p, H)).
5. **`garch_d_rm2006`** - RiskMetrics-2006 long-memory ARCH (Zumbach 2006):
   15 EWMAs of e^2 with tau_k = 4 sqrt(2)^(k-1) days (4..512), weights
   w_k ~ 1 - ln tau_k / ln 1560. All constants are the published ones,
   nothing is estimated. Forecast: w' sum_{h<H} M^h x[t+1],
   M = diag(mu) + (1-mu) w' (iterated expectations of the linear system).
   It has no fixed long-run level: the level is a log-weighted mix of
   EWMAs out to ~2 years.
6. **`garch_w_cgarch`** - component GARCH-t on non-overlapping VR5-adjusted
   5-day returns, 20-year window.
7. **`garch_m_garch11`** - monthly GARCH(1,1)-t with (a, b, nu) common to
   all assets (pooled likelihood over every asset with >= 36 months at the
   re-estimation date, all history) and asset-specific variance targets
   (all-history mean square). Convention-free; works for every asset from
   its 36th month. Also the fallback of all daily/5-day models.
8. **`garch_m_cgarch`** - monthly component GARCH-t, pooled the same way.

Fixed hyperparameters and why: 20-year window (round number; several
volatility cycles, but a 1990s fit does not see the 1930s); 3 years minimum
(below that a GARCH is barely identified); re-estimation each December
(protocol: at most once a year); 36-month MIDAS lag window (EGS 2013 use about
3 years); RM2006 constants (published); VR half-life 3 years and priors (see
above); t lower bound 2.5 (finite variance with margin); starting values at
conventional half-lives (1 month, 3 months, 1 year; component: long 6 months
/ 2 / 8 years, transitory 1 week / 1 month). No value was changed after any
score was seen.

## Estimated persistence

Median across assets of the estimates at four re-estimation dates (half-lives
in MONTHS; for component models "long" = rho, "short" = a + b):

| model | E | persistence | half-life long | half-life short | nu |
|---|---|---|---|---|---|
| garch_d_garch11 | 1955 | 0.985 | 2.3 | | 3.9 |
| garch_d_garch11 | 1985 | 0.992 | 4.5 | | 5.3 |
| garch_d_garch11 | 2000 | 0.988 | 2.8 | | 5.3 |
| garch_d_garch11 | 2013 | 0.994 | 5.8 | | 8.0 |
| garch_d_gjr | 2013 | 0.993 | 4.9 | | 8.0 |
| garch_d_cgarch | 1955 | 0.997 | 16.6 | 0.39 | 3.9 |
| garch_d_cgarch | 1985 | 0.997 | 13.3 | 0.37 | 5.3 |
| garch_d_cgarch | 2000 | 0.991 | 3.6 | 0.15 | 5.3 |
| garch_d_cgarch | 2013 | 0.997 | 12.8 | 0.22 | 7.4 |
| garch_d_midas (g) | 2013 | 0.982 | 1.8 | | 7.3 |
| garch_w_cgarch | 1955 | 0.989 | 18.0 | 0.08 | 5.5 |
| garch_w_cgarch | 2013 | 0.991 | 17.5 | 1.35 | 7.4 |
| garch_m_garch11 (pooled) | 1955 | 0.969 | 22.3 | | 6.3 |
| garch_m_garch11 (pooled) | 2013 | 0.992 | 82.7 | | 7.0 |
| garch_m_cgarch (pooled) | 1955 | 0.973 | 25.6 | 0.49 | 6.4 |
| garch_m_cgarch (pooled) | 1985 | 0.992 | 86.5 | 0.41 | 6.8 |
| garch_m_cgarch (pooled) | 2013 | 0.998 | 417 | 11.2 | 6.9 |

Daily component GARCH, 2013, by asset (long half-life, months): us_large 4.2,
us_small 12.7, intl_dev 11.7, em 15.4, ust_int 5.1, ust_long 7.4, tips 54,
ig_credit 3.9, hy_credit 46, reits 22, gold 12.9, commodities 27.7. GARCH-MIDAS
(median): theta 0.48-0.72 (a 1% higher smoothed RV raises the long-run
component by ~0.6%: partial mean reversion of the level), beta-lag w 4-8
(recent-weighted), m0 ~0.

Reading: daily GARCH(1,1) shocks have half-lives of 2-6 months, so after 12
months its forecast is ~80-90% the 20-year mean. The daily component model
finds a long-run component with a half-life of about one year (median 13
months), which is exactly the horizon-relevant persistence. Monthly data say
volatility is much more persistent than daily data imply (monthly GARCH half-
life 2-7 years; by 2000 the pooled monthly component model degenerates into a
"GARCH(1,1) with 1-year half-life on top of a near-constant 30-year level" -
the two components swap roles, i.e. the monthly data cannot separate them).

## Results

QLIKE (lower is better), common origins, means over origins of the
equal-weighted mean over assets. Skill = 100 (1 - mean loss / mean loss of the
benchmark), 90% Fieller interval with the EWC long-run variance (the package
default), DM = two-sided Diebold-Mariano p-value (EWC).


#### Subset `daily`: overall

| model | n_origins | first | last | qlike | se_var_scaled | log_err |
|---|---|---|---|---|---|---|
| garch_d_rm2006 | 1051 | 1926-06 | 2013-12 | 0.400 | 4.958 | -0.003 |
| garch_m_cgarch | 1051 | 1926-06 | 2013-12 | 0.424 | 4.818 | 0.074 |
| garch_m_garch11 | 1051 | 1926-06 | 2013-12 | 0.437 | 4.912 | 0.066 |
| garch_w_cgarch | 1051 | 1926-06 | 2013-12 | 0.442 | 4.709 | 0.026 |
| garch_d_cgarch | 1051 | 1926-06 | 2013-12 | 0.446 | 4.922 | 0.036 |
| ref_last12m | 1051 | 1926-06 | 2013-12 | 0.505 | 5.745 | -0.019 |
| garch_d_midas | 1051 | 1926-06 | 2013-12 | 0.507 | 6.160 | 0.054 |
| ref_last60m | 1051 | 1926-06 | 2013-12 | 0.520 | 6.433 | 0.070 |
| garch_d_garch11 | 1051 | 1926-06 | 2013-12 | 0.537 | 5.363 | 0.080 |
| garch_d_gjr | 1051 | 1926-06 | 2013-12 | 0.551 | 5.519 | 0.086 |
| ref_blend | 1051 | 1926-06 | 2013-12 | 0.567 | 5.858 | 0.101 |
| ref_longrun | 1051 | 1926-06 | 2013-12 | 0.837 | 6.819 | 0.131 |

#### Subset `daily`: QLIKE by era

| model | start..1945-12 | 1946-01..1979-12 | 1980-01..1999-12 | 2000-01..end |
|---|---|---|---|---|
| garch_d_rm2006 | 0.448 | 0.488 | 0.247 | 0.339 |
| garch_m_cgarch | 0.584 | 0.389 | 0.226 | 0.569 |
| garch_m_garch11 | 0.627 | 0.388 | 0.229 | 0.584 |
| garch_w_cgarch | 0.601 | 0.457 | 0.246 | 0.463 |
| garch_d_cgarch | 0.597 | 0.496 | 0.247 | 0.395 |
| ref_last12m | 0.418 | 0.599 | 0.313 | 0.673 |
| garch_d_midas | 0.800 | 0.524 | 0.294 | 0.361 |
| ref_last60m | 0.792 | 0.410 | 0.280 | 0.749 |
| garch_d_garch11 | 0.755 | 0.607 | 0.270 | 0.443 |
| garch_d_gjr | 0.789 | 0.612 | 0.269 | 0.475 |
| ref_blend | 0.958 | 0.471 | 0.313 | 0.618 |
| ref_longrun | 1.475 | 0.720 | 0.576 | 0.600 |
| n_origins | 235.000 | 408.000 | 240.000 | 168.000 |

#### Subset `daily`: QLIKE by asset

| model | us_large | us_small | intl_dev | em | ust_int | ust_long | tips | ig_credit | hy_credit | reits | gold | commodities |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| garch_d_rm2006 | 0.303 | 0.361 | 0.268 | 0.308 | 0.500 | 0.451 | 0.414 | 0.261 | 0.527 | 0.429 | 0.370 | 0.248 |
| garch_m_cgarch | 0.361 | 0.351 | 0.394 | 0.367 | 0.393 | 0.464 | 0.445 | 0.504 | 0.844 | 0.566 | 0.288 | 0.271 |
| garch_m_garch11 | 0.373 | 0.352 | 0.402 | 0.383 | 0.398 | 0.487 | 0.472 | 0.562 | 0.814 | 0.554 | 0.288 | 0.263 |
| garch_w_cgarch | 0.316 | 0.361 | 0.287 | 0.327 | 0.521 | 0.494 | 0.534 | 0.335 | 0.851 | 0.469 | 0.401 | 0.284 |
| garch_d_cgarch | 0.296 | 0.391 | 0.262 | 0.282 | 0.519 | 0.536 | 0.518 | 0.275 | 0.768 | 0.395 | 0.392 | 0.263 |
| ref_last12m | 0.390 | 0.402 | 0.517 | 0.473 | 0.712 | 0.678 | 0.508 | 0.616 | 1.062 | 0.592 | 0.420 | 0.406 |
| garch_d_midas | 0.343 | 0.453 | 0.266 | 0.292 | 0.591 | 0.554 | 0.295 | 0.302 | 0.657 | 0.570 | 0.415 | 0.257 |
| ref_last60m | 0.496 | 0.403 | 0.513 | 0.410 | 0.531 | 0.513 | 0.565 | 0.446 | 1.362 | 0.603 | 0.401 | 0.415 |
| garch_d_garch11 | 0.344 | 0.483 | 0.264 | 0.329 | 0.588 | 0.570 | 0.587 | 0.288 | 0.821 | 0.551 | 0.465 | 0.261 |
| garch_d_gjr | 0.363 | 0.496 | 0.263 | 0.331 | 0.573 | 0.563 | 0.587 | 0.277 | 0.977 | 0.558 | 0.465 | 0.261 |
| ref_blend | 0.470 | 0.414 | 0.380 | 0.349 | 0.517 | 0.645 | 0.562 | 0.466 | 1.070 | 0.660 | 0.374 | 0.308 |
| ref_longrun | 0.662 | 0.588 | 0.298 | 0.305 | 0.646 | 1.171 | 0.584 | 0.712 | 0.927 | 0.837 | 0.499 | 0.362 |

#### Subset `daily`: skill vs best reference `ref_last12m` (90% Fieller-EWC CI, DM two-sided)

| model | skill_% | ci90_low | ci90_high | DM_p |
|---|---|---|---|---|
| garch_d_garch11 | -6.328 | -53.662 | 23.953 | 0.751 |
| garch_d_gjr | -9.151 | -58.200 | 22.308 | 0.655 |
| garch_d_cgarch | 11.723 | -22.200 | 35.071 | 0.486 |
| garch_d_midas | -0.464 | -50.040 | 35.114 | 0.984 |
| garch_d_rm2006 | 20.752 | -2.150 | 35.658 | 0.124 |
| garch_w_cgarch | 12.444 | -21.186 | 35.778 | 0.460 |
| garch_m_garch11 | 13.530 | -21.473 | 38.491 | 0.445 |
| garch_m_cgarch | 16.011 | -16.027 | 38.285 | 0.337 |

#### Subset `daily`: within-family comparisons

| model | vs | skill_% | ci90_low | ci90_high | DM_p |
|---|---|---|---|---|---|
| garch_d_cgarch | garch_d_garch11 | 16.977 | 10.592 | 24.057 | 0.001 |
| garch_d_rm2006 | garch_d_garch11 | 25.469 | 7.556 | 39.158 | 0.037 |
| garch_d_midas | garch_d_garch11 | 5.515 | -5.051 | 20.693 | 0.403 |
| garch_d_gjr | garch_d_garch11 | -2.655 | -4.724 | -0.436 | 0.059 |
| garch_m_cgarch | garch_m_garch11 | 2.869 | -1.188 | 5.288 | 0.196 |
| garch_d_garch11 | garch_m_garch11 | -22.966 | -52.780 | -2.367 | 0.067 |
| garch_d_cgarch | garch_m_cgarch | -5.105 | -28.187 | 13.561 | 0.648 |
| garch_w_cgarch | garch_m_cgarch | -4.247 | -19.344 | 8.928 | 0.580 |
| garch_d_cgarch | garch_w_cgarch | -0.823 | -9.748 | 7.109 | 0.858 |

#### Subset `all`: overall

| model | n_origins | first | last | qlike | se_var_scaled | log_err |
|---|---|---|---|---|---|---|
| garch_d_rm2006 | 1104 | 1922-01 | 2013-12 | 0.413 | 3.225 | 0.088 |
| garch_m_cgarch | 1104 | 1922-01 | 2013-12 | 0.428 | 3.084 | 0.119 |
| garch_m_garch11 | 1104 | 1922-01 | 2013-12 | 0.435 | 3.150 | 0.113 |
| garch_d_cgarch | 1104 | 1922-01 | 2013-12 | 0.436 | 3.051 | 0.101 |
| garch_w_cgarch | 1104 | 1922-01 | 2013-12 | 0.437 | 2.989 | 0.097 |
| garch_d_midas | 1104 | 1922-01 | 2013-12 | 0.463 | 3.912 | 0.119 |
| garch_d_garch11 | 1104 | 1922-01 | 2013-12 | 0.481 | 3.213 | 0.117 |
| garch_d_gjr | 1104 | 1922-01 | 2013-12 | 0.488 | 3.271 | 0.120 |
| ref_blend | 1104 | 1922-01 | 2013-12 | 0.537 | 3.480 | 0.148 |
| ref_last60m | 1104 | 1922-01 | 2013-12 | 0.557 | 3.765 | 0.105 |
| ref_last12m | 1104 | 1922-01 | 2013-12 | 0.579 | 3.706 | -0.009 |
| ref_longrun | 1104 | 1922-01 | 2013-12 | 0.699 | 3.979 | 0.191 |

#### Subset `all`: QLIKE by era

| model | start..1945-12 | 1946-01..1979-12 | 1980-01..1999-12 | 2000-01..end |
|---|---|---|---|---|
| garch_d_rm2006 | 0.473 | 0.491 | 0.258 | 0.339 |
| garch_m_cgarch | 0.511 | 0.423 | 0.236 | 0.569 |
| garch_m_garch11 | 0.527 | 0.422 | 0.241 | 0.584 |
| garch_d_cgarch | 0.520 | 0.499 | 0.256 | 0.395 |
| garch_w_cgarch | 0.520 | 0.476 | 0.255 | 0.463 |
| garch_d_midas | 0.588 | 0.514 | 0.298 | 0.361 |
| garch_d_garch11 | 0.569 | 0.554 | 0.276 | 0.443 |
| garch_d_gjr | 0.580 | 0.554 | 0.275 | 0.475 |
| ref_blend | 0.729 | 0.515 | 0.288 | 0.618 |
| ref_last60m | 0.726 | 0.528 | 0.270 | 0.749 |
| ref_last12m | 0.555 | 0.701 | 0.336 | 0.673 |
| ref_longrun | 0.896 | 0.725 | 0.487 | 0.600 |
| n_origins | 288.000 | 408.000 | 240.000 | 168.000 |

#### Subset `all`: QLIKE by asset

| model | us_large | us_small | intl_dev | em | ust_int | ust_long | tips | ig_credit | hy_credit | reits | gold | commodities |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| garch_d_rm2006 | 0.290 | 0.361 | 0.293 | 0.342 | 0.482 | 0.483 | 0.414 | 0.489 | 0.507 | 0.371 | 0.370 | 0.301 |
| garch_m_cgarch | 0.345 | 0.351 | 0.385 | 0.381 | 0.386 | 0.491 | 0.445 | 0.580 | 0.805 | 0.468 | 0.288 | 0.305 |
| garch_m_garch11 | 0.357 | 0.352 | 0.397 | 0.410 | 0.390 | 0.503 | 0.472 | 0.600 | 0.776 | 0.458 | 0.288 | 0.305 |
| garch_d_cgarch | 0.284 | 0.391 | 0.288 | 0.318 | 0.499 | 0.531 | 0.518 | 0.494 | 0.733 | 0.348 | 0.392 | 0.305 |
| garch_w_cgarch | 0.303 | 0.361 | 0.308 | 0.360 | 0.501 | 0.508 | 0.534 | 0.516 | 0.811 | 0.399 | 0.401 | 0.311 |
| garch_d_midas | 0.329 | 0.453 | 0.291 | 0.327 | 0.564 | 0.542 | 0.295 | 0.504 | 0.628 | 0.469 | 0.415 | 0.304 |
| garch_d_garch11 | 0.329 | 0.483 | 0.290 | 0.361 | 0.562 | 0.551 | 0.587 | 0.499 | 0.782 | 0.456 | 0.465 | 0.305 |
| garch_d_gjr | 0.347 | 0.496 | 0.289 | 0.363 | 0.549 | 0.547 | 0.587 | 0.495 | 0.929 | 0.461 | 0.465 | 0.305 |
| ref_blend | 0.450 | 0.414 | 0.330 | 0.341 | 0.503 | 0.618 | 0.562 | 0.689 | 1.030 | 0.551 | 0.374 | 0.376 |
| ref_last60m | 0.477 | 0.403 | 0.437 | 0.396 | 0.518 | 0.580 | 0.565 | 0.722 | 1.304 | 0.509 | 0.401 | 0.444 |
| ref_last12m | 0.374 | 0.402 | 0.540 | 0.534 | 0.672 | 0.791 | 0.508 | 0.721 | 1.010 | 0.509 | 0.420 | 0.457 |
| ref_longrun | 0.632 | 0.588 | 0.263 | 0.301 | 0.618 | 0.931 | 0.584 | 0.909 | 0.895 | 0.680 | 0.499 | 0.480 |

#### Subset `all`: skill vs best reference `ref_blend` (90% Fieller-EWC CI, DM two-sided)

| model | skill_% | ci90_low | ci90_high | DM_p |
|---|---|---|---|---|
| garch_d_garch11 | 10.486 | -2.137 | 19.117 | 0.153 |
| garch_d_gjr | 9.109 | -2.637 | 17.196 | 0.176 |
| garch_d_cgarch | 18.793 | 6.357 | 27.433 | 0.029 |
| garch_d_midas | 13.758 | -0.027 | 24.276 | 0.101 |
| garch_d_rm2006 | 23.161 | 7.067 | 34.172 | 0.035 |
| garch_w_cgarch | 18.561 | 8.120 | 25.981 | 0.016 |
| garch_m_garch11 | 19.036 | 7.951 | 28.692 | 0.014 |
| garch_m_cgarch | 20.352 | 9.376 | 29.551 | 0.011 |

#### Subset `all`: within-family comparisons

| model | vs | skill_% | ci90_low | ci90_high | DM_p |
|---|---|---|---|---|---|
| garch_d_cgarch | garch_d_garch11 | 9.280 | 6.108 | 12.390 | 0.000 |
| garch_d_rm2006 | garch_d_garch11 | 14.160 | 5.538 | 21.605 | 0.016 |
| garch_d_midas | garch_d_garch11 | 3.656 | -2.703 | 10.725 | 0.338 |
| garch_d_gjr | garch_d_garch11 | -1.538 | -3.089 | 0.206 | 0.140 |
| garch_m_cgarch | garch_m_garch11 | 1.625 | -0.335 | 3.059 | 0.155 |
| garch_d_garch11 | garch_m_garch11 | -10.561 | -29.704 | 2.965 | 0.206 |
| garch_d_cgarch | garch_m_cgarch | -1.958 | -19.075 | 10.611 | 0.811 |
| garch_w_cgarch | garch_m_cgarch | -2.248 | -13.420 | 6.079 | 0.673 |
| garch_d_cgarch | garch_w_cgarch | 0.284 | -5.768 | 5.530 | 0.930 |


## Findings

* **Against the references.** On `all` (1922-2013, includes monthly-proxy
  targets) every GARCH member beats the best reference (`ref_blend`, QLIKE
  0.537), skill 9-23%; for five of eight the 90% CI excludes 0 (d_cgarch,
  d_rm2006, w_cgarch, m_garch11, m_cgarch; d_midas's lower bound is -0.0). On `daily`
  (1926-2013) the best reference is `ref_last12m` (0.505); the best member,
  `garch_d_rm2006` (0.400), has skill 20.8% [-2.2, 35.7], DM p = 0.12: not
  significant. `garch_d_garch11` (0.537), `garch_d_gjr` (0.551) and
  `garch_d_midas` (0.507) do not beat it.
* **Long-run component vs plain GARCH (yes, at this horizon).** Daily
  component GARCH beats daily GARCH(1,1) by 17.0% [10.6, 24.1] on `daily`
  (DM p = 0.001) and 9.3% [6.1, 12.4] on `all`; RM2006 long memory beats it by
  25.5% [7.6, 39.2]. GARCH-MIDAS gains only 5.5% [-5.1, 20.7] (n.s.). At the
  monthly frequency the component adds little (2.9% [-1.2, 5.3]) because the
  monthly GARCH(1,1) is already very persistent (half-life years). The
  asymmetric GJR is slightly WORSE than symmetric GARCH (-2.7% [-4.7, -0.4]):
  leverage matters for days, not for the 12-month average.
* **Daily-fitted vs monthly-fitted (it depends on the model and the era).**
  Plain daily GARCH(1,1) loses to monthly GARCH(1,1) (-23% [-53, -2] on
  `daily`): its 2-6 month half-lives throw away the slow persistence that
  monthly data reveal. With a slow component the daily model draws level:
  d_cgarch vs m_cgarch -5.1% [-28, 14], w_cgarch vs m_cgarch -4.2% [-19, 9],
  d_cgarch vs w_cgarch -0.8% (all n.s.). By era the picture is sharp: daily
  models win clearly 2000-2013 (d_cgarch 0.395, rm2006 0.339 vs m_cgarch
  0.569) and lose 1926-1979 (d_cgarch 0.597/0.496 vs m_cgarch 0.584/0.389).
  RM2006, which is daily but has no estimated level and an effective memory of
  ~2 years, is the only daily model that does well in every era.
* **Where it loses.** (1) 1926-1945: `ref_last12m` (0.418) beats every
  member except RM2006 (0.448); daily models have <20 years of daily history
  then, and the 1929-33 regime shift is followed better by a 12-month window
  than by a level estimated over the window. (2) Treasuries, daily: d_cgarch
  0.52/0.54 vs m_cgarch 0.39/0.46 on ust_int/ust_long; the 1962-c.1990 daily
  bond series are yield reconstructions with VR up to 2.7 (clipped/shrunk),
  and the adjustment is noisy there. (3) hy_credit everywhere 0.53-0.85
  (2008 spike from the lowest-vol asset; smoothed VWEHX NAV with VR 1.8 then
  HYG) - but every reference is worse (0.93-1.36). (4) Gold and us_small:
  the monthly models are best (0.29 and 0.35).
* **Bias.** log_err of the daily component and RM2006 models is close to the
  +0.03 Jensen offset a perfect forecast has (0.03-0.04 / -0.003), i.e. the
  VR mapping gets the level right on average; monthly models over-forecast
  slightly more (+0.07).

## Tried and dropped

Nothing was tried against the scoreboard and dropped: the eight models were
specified before the first backtest and none was changed after a score was
seen. One quick `--step 6` run preceded the full run with the identical code
(QLIKE `daily`: rm2006 0.390, m_cgarch 0.409, m_garch11 0.415, w_cgarch
0.432, d_cgarch 0.433, d_midas 0.476, d_garch11 0.510, d_gjr 0.532; same
ranking as the full run). Implementation decisions made before any scoring
and not revisited: EGARCH was not included (its log-variance recursion is
not linear in e^2, so it cannot use the fast `lfilter` path and would cost
~10x more per fit); FIGARCH (in `arch`) was replaced by RM2006 as the
long-memory member for cost (a full FIGARCH fit per asset-year with
1,000-lag truncation would not fit the 10-minute budget); fitting raw
(unadjusted) daily returns was rejected a priori on the evidence in
`horizon_evidence.md`.

## Run time

Full monthly run to 2014 (1,116 origins with >= 2 eligible assets),
single process, with 2 jobs on a 4-core machine shared with ~10 other jobs
(load average 10-17), so these are upper bounds; unloaded they are roughly
2-3x faster: d_cgarch 451 s, d_midas 401 s, w_cgarch 252 s, m_cgarch 166 s,
d_gjr 136 s, d_garch11 104 s, d_rm2006 53 s, m_garch11 28 s. Cost is
dominated by the yearly re-estimations (about 0.1-0.4 s per asset-year for
daily models); a warm call (filter only) takes ~60 ms for 12 assets. All
variance recursions are linear in e^2 for fixed parameters and run through
`scipy.signal.lfilter` (the 2-D component model is diagonalized; its
transition matrix always has real eigenvalues). Note: the optimizer is ~40x
slower without `OMP_NUM_THREADS=1`.

## Tests

`tests/models/test_garch.py` (33 tests, ~2 minutes; the 24 per-model
contract tests are marked `slow`): GARCH(1,1) estimates match `arch` 9.0
(alpha, beta within 0.01); component filter equals a plain loop and the
closed-form 12-month sum equals iterated expectations; estimation never
raises on degenerate data; RM2006 weights; the VR adjustment detects a
smoothed series (true log VR 0.69 -> 0.25+ after shrinkage), resets to 0 at a
tier change and stays near 0 for iid data; blocks are anchored at the
origin; every registered model gives finite positive vols for all eligible
assets at 1979 (monthly-only), 1990 and 2010 on the synthetic panel; cold =
warm = reverse-order answers; altering data after t (and warming the cache
on the altered future first) does not change the forecast at t; the daily
model equals the monthly fallback in the monthly-only era. All pass, as does
`tests/test_contract.py`.

## Caveats

* Statistical power is low (`daily`: about 1,050 overlapping monthly origins). Differences of 5-10% between members are noise.
* The daily models' level in 1962-1990 Treasuries and pre-2000 small caps
  depends on a VR that is itself noisy (log VR SE ~0.14 at full weight);
  shrinkage toward 1 limits but does not remove that.
* Tier-preserving splices (VGSIX -> VNQ in 2004: VR 1.45 -> 0.69) are
  absorbed over ~3 years, during which the daily scale is partly wrong.
* GARCH-MIDAS holds tau constant over the year; the in-sample theta < 1
  provides the only mean reversion of the level.
* Monthly component GARCH is weakly identified (components swap roles after
  2000); its forecasts are then close to monthly GARCH(1,1).
* Variance targeting on a 20-year window means the daily GARCH(1,1)/GJR
  12-month forecast is essentially a 20-year average: their poor 1926-1945 and
  1946-1979 scores are mostly that window, not the GARCH dynamics.
