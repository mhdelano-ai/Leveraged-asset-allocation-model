"""MC for MCS coverage and skill CI coverage. usage: python mc_mcs_skill.py REPS"""
import sys, warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
from simdgp import simulate_losses
from assetforecast.eval import inference as inf

warnings.simplefilter("ignore")
SCEN = {"base": dict(phi=0.95, rho_e=0.9), "persistent": dict(phi=0.98, rho_e=0.97)}
SIG10 = [0.30, 0.33, 0.33, 0.36, 0.36, 0.38, 0.40, 0.40, 0.45, 0.50]
EQ10 = [0.30] * 10


def mcs_one(args):
    seed, T, scen, design, L = args
    rng = np.random.default_rng(seed)
    sig = SIG10 if design == "graded" else EQ10
    X = simulate_losses(T, sig, rng, **SCEN[scen])
    df = pd.DataFrame(X, columns=[f"m{i}" for i in range(10)])
    out = {}
    var, L = L
    r = inf.model_confidence_set(df, 0.10, block_length=L, variance=var, n_boot=500, seed=seed + 1, warn=False)
    for a in (0.10, 0.25):
        inc = [k for k, p in r.pvalues.items() if p >= a]
        out[a] = ("m0" in inc, len(inc), "m9" in inc)
    return out


def skill_one(args):
    seed, T, scen, true_skill = args
    rng = np.random.default_rng(seed)
    X = simulate_losses(T, [0.30, 0.36], rng, **SCEN[scen])
    out = {}
    for name, kw in {
        "fieller_default": dict(),
        "delta_hac_ewc": dict(method="hac"),
        "boot_L12": dict(method="bootstrap", block_length=12),
        "boot_L36": dict(method="bootstrap", block_length=36),
    }.items():
        r = inf.skill_score(X[:, 0], X[:, 1], level=0.90, n_boot=500, seed=seed + 3, warn=False, **kw)
        out[name] = (r.ci_low <= true_skill <= r.ci_high, r.ci_high - r.ci_low)
    return out


def true_skill(scen, n=400, T=5000):
    rng = np.random.default_rng(12345)
    X = np.vstack([simulate_losses(T, [0.30, 0.36], rng, **SCEN[scen]) for _ in range(n)])
    return 100 * (1 - X[:, 0].mean() / X[:, 1].mean())


if __name__ == "__main__":
    reps = int(sys.argv[1])
    with Pool(4) as pool:
        for scen in SCEN:
            ts = true_skill(scen)
            print(f"true skill {scen}: {ts:.2f}%", flush=True)
            for T in (480, 1080):
                res = pool.map(skill_one, [(77 * i + T, T, scen, ts) for i in range(reps)])
                for name in res[0]:
                    cov = np.mean([r[name][0] for r in res])
                    w = np.median([r[name][1] for r in res])
                    print(f"  skill {scen} T={T} {name:14s} coverage90={cov:.3f} median_width={w:.1f}pp", flush=True)
        for scen in SCEN:
            for T in (480, 1080):
                for design in ("graded", "equal"):
                    for L in (("gk", 36), ("gk", 24), ("bootstrap", 36)):
                        res = pool.map(mcs_one, [(31 * i + T, T, scen, design, L) for i in range(reps)])
                        for a in (0.10, 0.25):
                            inc = np.mean([r[a][0] for r in res])
                            size = np.mean([r[a][1] for r in res])
                            worst = np.mean([r[a][2] for r in res])
                            allin = np.mean([r[a][1] == 10 for r in res])
                            print(f"  MCS {scen} T={T} {design} L={L} alpha={a}: P(best in)={inc:.3f} "
                                  f"mean size={size:.2f} P(worst in)={worst:.3f} P(all in)={allin:.3f}", flush=True)
