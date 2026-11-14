import numpy as np, warnings
from multiprocessing import Pool
from assetforecast.eval import inference as inf
warnings.simplefilter("ignore")
def sim(T, rng, phi, h=12, sd_x=0.7, burn=200):
    N=burn+T+h; sh=sd_x*np.sqrt(1-phi**2); z=rng.standard_normal(N); x=np.empty(N); x[0]=0
    for t in range(1,N): x[t]=phi*x[t-1]+sh*z[t]
    v=np.exp(x)*rng.gamma(4.0,0.25,N); cs=np.concatenate([[0.0],np.cumsum(v)]); idx=np.arange(burn,burn+T)
    Y=(cs[idx+h+1]-cs[idx+1])/h; k=np.arange(1,h+1)
    F=np.mean(np.exp(phi**k[None,:]*x[idx][:,None]+0.5*sd_x**2*(1-phi**(2*k))[None,:]),axis=1)
    return Y,F
def one(a):
    seed,T,phi=a; rng=np.random.default_rng([seed,T,int(phi*100)]); Y,F=sim(T,rng,phi)
    r=inf.mincer_zarnowitz(Y,F,warn=False); rn=inf.mincer_zarnowitz(Y,F,weighting="ols",warn=False)
    rl=inf.mincer_zarnowitz(Y,F,log=True,warn=False)
    return r.wald_pvalue, r.pvalue["slope"], rn.wald_pvalue, rl.pvalue["slope"], rl.params["slope"]
if __name__=="__main__":
    with Pool(4) as p:
        for phi in (0.95,0.98):
            for T in (480,1080):
                r=np.array(p.map(one,[(i,T,phi) for i in range(1000)]))
                print(f"phi={phi} T={T}: levels GLS joint {np.mean(r[:,0]<0.05):.3f}/{np.mean(r[:,0]<0.10):.3f}  slope {np.mean(r[:,1]<0.05):.3f}/{np.mean(r[:,1]<0.10):.3f}  levels OLS joint {np.mean(r[:,2]<0.05):.3f}/{np.mean(r[:,2]<0.10):.3f}  log slope=1 {np.mean(r[:,3]<0.05):.3f}/{np.mean(r[:,3]<0.10):.3f} mean log slope {r[:,4].mean():.3f}",flush=True)
