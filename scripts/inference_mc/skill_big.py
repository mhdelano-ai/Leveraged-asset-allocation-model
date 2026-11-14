import numpy as np, warnings
from multiprocessing import Pool
from simdgp import simulate_losses
from volmodel.eval import inference as inf
warnings.simplefilter("ignore")
SC={"base":dict(),"persistent":dict(phi=0.98,rho_e=0.97)}
TRUE={"base":14.05,"persistent":18.02}
def one(a):
    seed,T,scen=a; rng=np.random.default_rng([seed,T,len(scen)])
    X=simulate_losses(T,[0.30,0.36],rng,**SC[scen]); out={}
    for m in ("fieller","hac"):
        s=inf.skill_score(X[:,0],X[:,1],method=m,warn=False); out[m]=(s.ci_low<=TRUE[scen]<=s.ci_high, s.ci_high<TRUE[scen])
    s=inf.skill_score(X[:,0],X[:,1],method="fieller",level=0.8,warn=False); out["fieller80"]=(s.ci_low<=TRUE[scen]<=s.ci_high, s.ci_high<TRUE[scen])
    return out
if __name__=="__main__":
    with Pool(4) as p:
        for scen in SC:
            for T in (240,480,1080):
                r=p.map(one,[(i,T,scen) for i in range(1500)])
                for k in r[0]:
                    print(scen,T,k,"coverage",np.mean([x[k][0] for x in r]).round(3),"P(miss high: CI below truth)",np.mean([x[k][1] for x in r]).round(3),flush=True)
