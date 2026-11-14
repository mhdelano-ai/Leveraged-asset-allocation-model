import numpy as np
rng=np.random.default_rng(1); D,K,reps=252,21,40000
def s21(X):
    c=np.concatenate([np.zeros(X.shape[:1]+(1,)+X.shape[2:]),np.cumsum(X,axis=1)],axis=1)
    return c[:,K:]-c[:,:-K]
X=rng.standard_normal((reps,D))
r=(s21(X)**2).mean(1)/K   # ratio to true var
v=np.sqrt(r)
print("se_vol minimizer E[s]/s-1: %.4f"%(v.mean()-1))
print("se_logvol minimizer exp(E log s)/s-1: %.4f"%(np.exp(np.log(v).mean())-1))
print("mean log_err of a perfect forecast -E log(s_r/s): %+.4f"%(-np.log(v).mean()))
print("portfolio log_ratio of perfect forecast E[0.5 log RV/PV]: %+.4f"%(np.log(v).mean()))
for rho in (0.0,0.3,0.5,0.8):
    Z=rng.standard_normal((reps,D,2)); Z[:,:,1]=rho*Z[:,:,0]+np.sqrt(1-rho**2)*Z[:,:,1]
    S=s21(Z); c=(S[:,:,0]*S[:,:,1]).mean(1)/np.sqrt((S[:,:,0]**2).mean(1)*(S[:,:,1]**2).mean(1))
    print(f"rho {rho}: E[corr_r]-rho {c.mean()-rho:+.4f}  SE {c.std():.3f}   (1-day SE {(1-rho**2)/np.sqrt(252):.3f})")
