"""Confirm final defaults incl. short (20y) samples. usage: python mc_dm3.py REPS"""
import sys, warnings
import numpy as np
from multiprocessing import Pool
from simdgp import simulate_losses
from volmodel.eval import inference as inf
import mc_dm
warnings.simplefilter("ignore")
V = ["classic", "nw_default", "ewc_default", "boot_gk_default", "boot_stat_default_nostud"]
def one(args):
    seed, T, sig_b, loss, scen = args
    rng = np.random.default_rng(seed)
    L = simulate_losses(T, [0.3, sig_b], rng, loss=loss, **mc_dm.SCEN[scen])
    a, b = L[:, 0], L[:, 1]
    return {
        "classic": inf.dm_test(a, b, kernel="truncated", warn=False).pvalue,
        "nw_default": inf.dm_test(a, b, kernel="bartlett", warn=False).pvalue,
        "ewc_default": inf.dm_test(a, b, warn=False).pvalue,
        "boot_gk_default": inf.dm_test_bootstrap(a, b, n_boot=399, seed=seed + 7, warn=False).pvalue,
        "boot_stat_default_nostud": inf.dm_test_bootstrap(a, b, n_boot=399, seed=seed + 7, bootstrap="stationary", studentize=False, warn=False).pvalue,
    }
if __name__ == "__main__":
    reps = int(sys.argv[1])
    with Pool(4) as pool:
        for scen in ["base", "persistent"]:
            for T in [240, 480, 1080]:
                ps = pool.map(one, [(1000 * i + T + 29, T, 0.3, "qlike", scen) for i in range(reps)])
                print(scen, "qlike", T, "null", flush=True)
                for name in V:
                    p = np.array([q[name] for q in ps])
                    print(f"   {name:26s} {np.mean(p<0.05):.3f} {np.mean(p<0.10):.3f}", flush=True)
