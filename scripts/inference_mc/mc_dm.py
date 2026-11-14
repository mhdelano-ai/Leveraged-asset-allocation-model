"""Monte Carlo size/power of DM variants. usage: python mc_dm.py REPS"""
import sys, warnings, json
import numpy as np
from multiprocessing import Pool
from simdgp import simulate_losses
from volmodel.eval import inference as inf

warnings.simplefilter("ignore")
H = 12

VARIANTS = {
    "classic_trunc_hln": dict(fn="hac", kernel="truncated"),
    "nw_h-1_fixedb": dict(fn="hac", kernel="bartlett", bandwidth=11),
    "nw_default_fixedb": dict(fn="hac", kernel="bartlett"),
    "ewc_llsw_0.4T^2/3": dict(fn="hac", kernel="ewc", ewc="llsw"),
    "ewc_default_T/2h": dict(fn="hac", kernel="ewc"),
    "ewc_T/4h": dict(fn="hac", kernel="ewc", ewc="t4h"),
    "boot_gk_L12": dict(fn="boot", L=12),
    "boot_gk_L24": dict(fn="boot", L=24),
    "boot_gk_L36": dict(fn="boot", L=36),
    "boot_stat_L24_nostud": dict(fn="boot", L=24, stud=False),
}

SCEN = {
    "base": dict(phi=0.95, rho_e=0.9),
    "persistent": dict(phi=0.98, rho_e=0.97),
    "persistent_gauss": dict(phi=0.98, rho_e=0.97, nu=None),
}


def one(args):
    seed, T, sig_b, loss, scen = args
    rng = np.random.default_rng(seed)
    L = simulate_losses(T, [0.3, sig_b], rng, loss=loss, **SCEN[scen])
    out = {}
    for name, v in VARIANTS.items():
        if v["fn"] == "hac":
            kw = {}
            if v.get("ewc") == "llsw":
                kw["ewc_terms"] = int(0.4 * T ** (2 / 3))
            elif v.get("ewc") == "t4h":
                kw["ewc_terms"] = max(4, T // (4 * H))
            r = inf.dm_test(L[:, 0], L[:, 1], h=H, kernel=v["kernel"], bandwidth=v.get("bandwidth"), warn=False, **kw)
        else:
            stud = v.get("stud", True)
            r = inf.dm_test_bootstrap(L[:, 0], L[:, 1], h=H, block_length=v["L"], n_boot=399,
                                      bootstrap="circular" if stud else "stationary", studentize=stud,
                                      seed=seed + 7, warn=False)
        out[name] = r.pvalue
    return out


if __name__ == "__main__":
    reps = int(sys.argv[1])
    res = {}
    with Pool(4) as pool:
        for scen in SCEN:
            for loss in ["qlike", "mse"]:
                for T in [480, 1080]:
                    for sig_b, lab in [(0.3, "null"), (0.33, "alt33"), (0.36, "alt36")]:
                        if loss == "mse" and lab == "alt" and scen == "persistent":
                            pass
                        ps = pool.map(one, [(1000 * i + T, T, sig_b, loss, scen) for i in range(reps)])
                        for name in VARIANTS:
                            p = np.array([q[name] for q in ps])
                            res[(scen, loss, T, lab, name)] = (np.mean(p < 0.05), np.mean(p < 0.10))
                        print(scen, loss, T, lab, flush=True)
                        for name in VARIANTS:
                            print(f"   {name:24s} {res[(scen, loss, T, lab, name)][0]:.3f} {res[(scen, loss, T, lab, name)][1]:.3f}", flush=True)
    json.dump({"|".join(map(str, k)): v for k, v in res.items()}, open(f"mc_dm_{reps}.json", "w"), indent=1)
