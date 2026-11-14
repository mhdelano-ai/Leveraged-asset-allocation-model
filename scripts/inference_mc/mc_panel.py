"""Panel DM size: 12 assets, common vol factor, unbalanced starts. usage: python mc_panel.py REPS"""
import sys, warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
from assetforecast.eval import inference as inf

warnings.simplefilter("ignore")
H = 12


def ar1(n, phi, rng, shocks):
    x = np.empty(n)
    x[0] = 0
    for t in range(1, n):
        x[t] = phi * x[t - 1] + shocks[t]
    return x


def sim_panel(T, n_assets, rng, phi=0.95, rho_e=0.9, sd_x=0.7, share=0.6, sig=(0.3, 0.3), burn=200):
    N = burn + T + H
    sh = sd_x * np.sqrt(1 - phi**2)
    zc = np.clip(rng.standard_t(5, N) / np.sqrt(5 / 3), -5, 5)
    xc = ar1(N, phi, rng, sh * zc)
    idx = np.arange(burn, burn + T)
    k = np.arange(1, H + 1)
    se = np.sqrt(1 - rho_e**2)
    ec = ar1(T, rho_e, rng, se * rng.standard_normal(T))  # common model-error factor
    rows = []
    starts = np.linspace(0, int(0.6 * T), n_assets).astype(int)  # unequal histories
    for a in range(n_assets):
        zi = np.clip(rng.standard_t(5, N) / np.sqrt(5 / 3), -5, 5)
        x = np.sqrt(share) * xc + np.sqrt(1 - share) * ar1(N, phi, rng, sh * zi)
        v = np.exp(x) * rng.gamma(4.0, 0.25, N)
        cs = np.concatenate([[0.0], np.cumsum(v)])
        Y = (cs[idx + H + 1] - cs[idx + 1]) / H
        F = np.mean(np.exp(phi ** k[None, :] * x[idx][:, None] + 0.5 * sd_x**2 * (1 - phi ** (2 * k))[None, :]), axis=1)
        for j, s in enumerate(sig):
            ei = ar1(T, rho_e, rng, se * rng.standard_normal(T))
            e = s * (np.sqrt(0.5) * ec + np.sqrt(0.5) * ei)
            f = F * np.exp(e)
            r = Y / f
            rows.append(pd.DataFrame({"origin": np.arange(T), "asset": a, "model": "ab"[j], "loss": r - np.log(r) - 1}).iloc[starts[a]:])
    return pd.concat(rows, ignore_index=True)


def one(args):
    seed, T = args
    rng = np.random.default_rng(seed)
    P = sim_panel(T, 12, rng)
    P["origin"] = pd.date_range("1930-01-31", periods=T, freq="ME")[P["origin"].to_numpy()]
    out = {
        "dk_ewc_default": inf.panel_dm_test(P, "a", "b", warn=False).pvalue,
        "dk_ewc_dateweights": inf.panel_dm_test(P, "a", "b", weights="date", warn=False).pvalue,
        "dk_boot_default": inf.panel_dm_test(P, "a", "b", method="bootstrap", n_boot=399, seed=seed, warn=False).pvalue,
    }
    # naive: pool all cells as iid
    w = P.pivot_table(index=["origin", "asset"], columns="model", values="loss").dropna()
    d = (w["a"] - w["b"]).to_numpy()
    from scipy import stats
    out["naive_iid_pooled"] = 2 * stats.norm.sf(abs(d.mean() / (d.std() / np.sqrt(d.size))))
    # cluster by time only (no HAC): bandwidth 0 on date scores
    out["cluster_time_only"] = inf.panel_dm_test(P, "a", "b", kernel="bartlett", bandwidth=0, warn=False).pvalue
    return out


if __name__ == "__main__":
    reps = int(sys.argv[1])
    with Pool(4) as pool:
        for T in (480, 1080):
            res = pool.map(one, [(i * 13 + T, T) for i in range(reps)])
            print("panel null T", T, flush=True)
            for k in res[0]:
                p = np.array([r[k] for r in res])
                print(f"   {k:22s} {np.mean(p<0.05):.3f} {np.mean(p<0.10):.3f}", flush=True)
