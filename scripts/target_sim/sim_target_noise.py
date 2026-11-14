"""Item 5: standard error and bias of the realized 12-month vol, 21-day vs 1-day definition."""
import numpy as np

rng = np.random.default_rng(42)
D, K = 252, 21
SIG = 0.16


def estimators(X, mu_daily=0.0):
    """X: reps x D daily log returns. Returns dict of annualized variance estimates per rep."""
    reps = X.shape[0]
    c = np.concatenate([np.zeros((reps, 1)), np.cumsum(X, axis=1)], axis=1)
    S = c[:, K:] - c[:, :-K]                                       # reps x (D-K+1) overlapping sums
    out = {
        "1d": (X ** 2).mean(axis=1) * 252,
        "21d": (S ** 2).mean(axis=1) * 252 / K,
        "21d_centered": ((S - S.mean(axis=1, keepdims=True)) ** 2).mean(axis=1) * 252 / K,
        "monthly_proxy": (X.reshape(reps, 12, 21).sum(axis=2) ** 2).mean(axis=1) * 12,
    }
    return out


def report(name, est, true_var):
    print(f"\n== {name} ==")
    print(f"{'estimator':16s} {'SE(vol)/vol':>12s} {'bias var':>9s} {'bias vol':>9s} {'n_eff':>6s}")
    for k, v in est.items():
        tv = true_var if np.ndim(true_var) == 0 else true_var
        r = v / tv                                                  # ratio to the window's true var
        vol_r = np.sqrt(r)
        n_eff = 2 / np.var(r)
        print(f"{k:16s} {vol_r.std():12.4f} {r.mean() - 1:+9.4f} {vol_r.mean() - 1:+9.4f} {n_eff:6.1f}")


# ---- i.i.d. normal, mean zero
reps = 40000
X = rng.standard_normal((reps, D)) * SIG / np.sqrt(252)
report("iid normal, mean 0", estimators(X), SIG ** 2)

# ---- i.i.d. normal with Sharpe 0.5 (mu = 0.5 * sigma per year)
mu = 0.5 * SIG / 252
report("iid normal, Sharpe 0.5 (true var = sigma^2, uncentered)", estimators(X + mu), SIG ** 2)
print(f"theory: 21-day uncentered relative var bias = K*SR^2/252 = {K * 0.25 / 252:.4f}; 1-day = {0.25 / 252:.4f}")
for sr in (0.3, 1.0):
    est = estimators(X + sr * SIG / 252)
    print(f"  SR {sr}: 21d var bias {est['21d'].mean() / SIG**2 - 1:+.4f}, monthly proxy {est['monthly_proxy'].mean() / SIG**2 - 1:+.4f}")

# ---- i.i.d. Student-t(5) (fat tails, no vol clustering)
t = rng.standard_t(5, size=(reps, D)) / np.sqrt(5 / 3) * SIG / np.sqrt(252)
report("iid Student-t(5)", estimators(t), SIG ** 2)

# ---- persistent stochastic volatility (synthetic-panel generator: log-vol AR(1), phi=0.995, vov=0.035/day)
years = 20000
n = years * D
phi, vov = 0.995, 0.035
e = rng.standard_normal(n) * vov
h = np.empty(n)
h[0] = 0.0
# AR(1) via loop in chunks (vectorized filter)
from scipy.signal import lfilter
h = lfilter([1.0], [1.0, -phi], e)
sv = np.exp(h - h.var() / 2)
sigma_t = SIG / np.sqrt(252) * sv / np.sqrt(np.mean(sv ** 2))
Xs = (rng.standard_normal(n) * sigma_t).reshape(years, D)
iv = (sigma_t.reshape(years, D) ** 2).mean(axis=1) * 252          # window's integrated variance
report("persistent SV, error vs the window's integrated variance", estimators(Xs), iv)
# for the 21-day target, the expected value given the window's path is the mean over sums of the
# 21-day integrated variances (excludes edge days) - closer to what is measured; report too
c = np.concatenate([np.zeros((years, 1)), np.cumsum(sigma_t.reshape(years, D) ** 2, axis=1)], axis=1)
iv21 = (c[:, K:] - c[:, :-K]).mean(axis=1) * 252 / K
est = estimators(Xs)
r = est["21d"] / iv21
print(f"  21d vs its own edge-weighted IV: SE(vol) {np.sqrt(r).std():.4f}, n_eff {2 / np.var(r):.1f}")
print(f"  unconditional dispersion of the window IV itself (vol): {np.sqrt(iv / SIG**2).std():.3f}")

# ---- persistent SV with Student-t(5) innovations
Xt = (rng.standard_t(5, size=n) / np.sqrt(5 / 3) * sigma_t).reshape(years, D)
report("persistent SV + t(5), error vs integrated variance", estimators(Xt), iv)

# ---- GARCH(1,1) daily, persistence 0.99 (alpha .08 beta .91)
a, b = 0.08, 0.91
om = (SIG ** 2 / 252) * (1 - a - b)
n2 = 5000 * D
z = rng.standard_normal(n2)
x = np.empty(n2)
s2 = np.empty(n2)
s2[0] = SIG ** 2 / 252
for i in range(n2):
    if i:
        s2[i] = om + a * x[i - 1] ** 2 + b * s2[i - 1]
    x[i] = np.sqrt(s2[i]) * z[i]
ivg = s2.reshape(-1, D).mean(axis=1) * 252
report("GARCH(1,1) a=.08 b=.91, error vs integrated variance", estimators(x.reshape(-1, D)), ivg)
