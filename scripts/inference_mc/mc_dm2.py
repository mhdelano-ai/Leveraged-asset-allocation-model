"""Targeted remainder + power curve. usage: python mc_dm2.py REPS"""
import sys, warnings
import numpy as np
from multiprocessing import Pool
from simdgp import simulate_losses
from assetforecast.eval import inference as inf
import mc_dm
warnings.simplefilter("ignore")
mc_dm.VARIANTS = {
    "classic_trunc_hln": dict(fn="hac", kernel="truncated"),
    "nw_default_fixedb": dict(fn="hac", kernel="bartlett"),
    "ewc_T/2h": dict(fn="hac", kernel="ewc", ewc="t2h"),
    "ewc_T/4h": dict(fn="hac", kernel="ewc", ewc="t4h"),
    "boot_gk_L24": dict(fn="boot", L=24),
    "boot_gk_L36": dict(fn="boot", L=36),
}
def one(args):
    seed, T, sig_b, loss, scen = args
    rng = np.random.default_rng(seed)
    L = simulate_losses(T, [0.3, sig_b], rng, loss=loss, **mc_dm.SCEN[scen])
    out = {}
    for name, v in mc_dm.VARIANTS.items():
        if v["fn"] == "hac":
            kw = {}
            if v.get("ewc") == "t2h": kw["ewc_terms"] = min(max(4, T // 24), int(0.4*T**(2/3)))
            if v.get("ewc") == "t4h": kw["ewc_terms"] = max(4, T // 48)
            r = inf.dm_test(L[:, 0], L[:, 1], kernel=v["kernel"], warn=False, **kw)
        else:
            r = inf.dm_test_bootstrap(L[:, 0], L[:, 1], block_length=v["L"], n_boot=399, seed=seed + 7, warn=False)
        out[name] = r.pvalue
    return out
cells = [("persistent","mse",480,0.3),("persistent","mse",1080,0.3),
         ("persistent_gauss","qlike",480,0.3),("persistent_gauss","qlike",1080,0.3)]
for scen in ["base","persistent"]:
    for T in [480,1080]:
        for s in [0.40,0.45]:
            cells.append((scen,"qlike",T,s))
if __name__ == "__main__":
    reps = int(sys.argv[1])
    with Pool(4) as pool:
        for scen, loss, T, s in cells:
            ps = pool.map(one, [(1000 * i + T + 17, T, s, loss, scen) for i in range(reps)])
            print(scen, loss, T, "sig_b", s, flush=True)
            for name in mc_dm.VARIANTS:
                p = np.array([q[name] for q in ps])
                print(f"   {name:24s} {np.mean(p<0.05):.3f} {np.mean(p<0.10):.3f}", flush=True)
