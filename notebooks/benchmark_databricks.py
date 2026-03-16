# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: R2VF Territory Clustering vs Manual Quintile Banding

# COMMAND ----------

%pip install "torch" --index-url https://download.pytorch.org/whl/cpu --quiet

# COMMAND ----------

%pip install "statsmodels==0.14.4" "insurance-glm-tools" --no-deps --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

try:
    from insurance_glm_tools.cluster import FactorClusterer
    print("FactorClusterer OK")
    HAS_CLUSTERER = True
except Exception as e:
    import traceback
    print(f"Import failed: {e}")
    print(traceback.format_exc()[:500])
    HAS_CLUSTERER = False

RNG = np.random.default_rng(42)
print(f"numpy: {np.__version__}")

# COMMAND ----------

# Data generation
N=20_000; ND=30
TB={0:list(range(0,4)),1:list(range(4,9)),2:list(range(9,15)),3:list(range(15,20)),4:list(range(20,24)),5:list(range(24,28)),6:list(range(28,30))}
TM={0:.7,1:.85,2:1.,3:1.15,4:1.35,5:1.6,6:2.}
d2b={d:b for b,ds in TB.items() for d in ds}; d2m={d:TM[b] for b,ds in TB.items() for d in ds}
DN=[f"D{i:02d}" for i in range(ND)]
ew=np.ones(ND); ew[20:28]=3.5; ew[0:4]=0.6; dp=ew/ew.sum()
VM={a:1+.02*max(0,a-5) for a in range(16)}
DB=["young","standard","mature","senior"]; DM={"young":2.2,"standard":1,"mature":.85,"senior":1.1}
di=RNG.choice(ND,size=N,p=dp); va=RNG.integers(0,16,size=N)
db=RNG.choice(DB,size=N,p=[.12,.6,.2,.08]); ex=RNG.uniform(.3,1,size=N)
tf=np.array([.08*d2m[d]*VM[v]*DM[b] for d,v,b in zip(di,va,db)])
cl=RNG.poisson(tf*ex)
df=pd.DataFrame({"district":[DN[i] for i in di],"district_num":di.astype(int),"veh_age":va.astype(int),
                  "driver_band":db,"exposure":ex,"claims":cl.astype(int),"true_band":[d2b[d] for d in di]})
mtr=RNG.random(N)<.70; df_tr=df[mtr].copy().reset_index(drop=True); df_te=df[~mtr].copy().reset_index(drop=True)
print(f"Train:{len(df_tr):,} Test:{len(df_te):,}")

# COMMAND ----------

def pdev(y,mu): return float(2*np.where(np.asarray(y)>0,np.asarray(y)*np.log(np.maximum(np.asarray(y),1e-10)/np.maximum(np.asarray(mu),1e-10))-(np.asarray(y)-np.asarray(mu)),np.asarray(mu)).sum())
def pll(y,mu): return float(np.where(np.asarray(y)>0,np.asarray(y)*np.log(np.maximum(np.asarray(mu),1e-10))-np.asarray(mu),-np.asarray(mu)).sum())
def bicscore(ll,k,n): return float(-2*ll+k*np.log(n))
def arivt(df,dc,pc):
    s=df.groupby(dc).agg(t=("true_band","first"),p=((pc),"first")).reset_index()
    return float(adjusted_rand_score(s["t"],s["p"]))

# Baseline: manual quintile banding
t0=time.perf_counter()
dd=pd.get_dummies(df_tr["district"],prefix="d",drop_first=True,dtype=float)
bd=pd.get_dummies(df_tr["driver_band"],prefix="b",drop_first=True,dtype=float)
Xr=pd.concat([df_tr[["veh_age"]].reset_index(drop=True),dd.reset_index(drop=True),bd.reset_index(drop=True)],axis=1)
yr=df_tr["claims"].values.astype(float)/np.maximum(df_tr["exposure"].values,1e-10)
g0=PoissonRegressor(alpha=0,max_iter=500); g0.fit(Xr.values,yr,sample_weight=df_tr["exposure"].values)
traw=time.perf_counter()-t0
dc2={"D00":0.}
for c in [x for x in Xr.columns if x.startswith("d_")]: dc2[c[2:]]=float(g0.coef_[list(Xr.columns).index(c)])
cs=pd.Series(dc2).sort_values(); lb=pd.qcut(cs,q=5,labels=False,duplicates="drop")
d2b2=dict(zip(cs.index,lb.values.tolist()))
df_tr["ter"]=df_tr["district"].map(d2b2); df_te["ter"]=df_te["district"].map(d2b2)
t0r=time.perf_counter()
md=pd.get_dummies(df_tr["ter"].astype(str),prefix="m",drop_first=True,dtype=float)
bd2=pd.get_dummies(df_tr["driver_band"],prefix="b",drop_first=True,dtype=float)
Xm=pd.concat([df_tr[["veh_age"]].reset_index(drop=True),md.reset_index(drop=True),bd2.reset_index(drop=True)],axis=1)
g1=PoissonRegressor(alpha=0,max_iter=500); g1.fit(Xm.values,yr,sample_weight=df_tr["exposure"].values)
ttb=traw+(time.perf_counter()-t0r)
mte=pd.get_dummies(df_te["ter"].fillna(-1).astype(str),prefix="m",dtype=float)
bte=pd.get_dummies(df_te["driver_band"],prefix="b",dtype=float)
for c in md.columns:
    if c not in mte.columns: mte[c]=0.
for c in bd2.columns:
    if c not in bte.columns: bte[c]=0.
Xmte=pd.concat([df_te[["veh_age"]].reset_index(drop=True),mte[md.columns].reset_index(drop=True),bte[bd2.columns].reset_index(drop=True)],axis=1)
mu_m=g1.predict(Xmte.values)*df_te["exposure"].values
dev_m=pdev(df_te["claims"].values,mu_m)
bic_m=bicscore(pll(df_tr["claims"].values,g1.predict(Xm.values)*df_tr["exposure"].values),Xm.shape[1]+1,len(df_tr))
df_te["mg"]=df_te["ter"].fillna(-1).astype(int); ari_m=arivt(df_te,"district","mg")
dpbm=dev_m/max(ND-5,1)
print(f"Manual: dev={dev_m:.2f}, BIC={bic_m:.2f}, ARI={ari_m:.4f}, t={ttb:.2f}s")

# COMMAND ----------

if HAS_CLUSTERER:
    # Correct FactorClusterer API: ordinal_factors is a fit() param, not __init__
    def bfc(df_in):
        bd=pd.get_dummies(df_in["driver_band"],prefix="b",drop_first=True,dtype=float)
        return pd.concat([df_in[["district_num","veh_age"]].reset_index(drop=True),bd.reset_index(drop=True)],axis=1)
    Xtr=bfc(df_tr); Xte2=bfc(df_te)
    ytr=df_tr["claims"].values.astype(float); yte2=df_te["claims"].values.astype(float)
    etr=df_tr["exposure"].values; ete2=df_te["exposure"].values

    t0=time.perf_counter()
    fc=FactorClusterer(family="poisson", lambda_="bic", min_exposure=100)
    # Pass ordinal_factors to fit(), not __init__
    fc.fit(Xtr, ytr, exposure=etr, ordinal_factors=["district_num"])
    tfit=time.perf_counter()-t0

    # Get transformed X and refit unpenalised GLM
    Xtr_tf = fc.transform(Xtr)
    Xte2_tf = fc.transform(Xte2)

    # refit_glm returns a statsmodels GLMResults
    glm_result = fc.refit_glm(Xtr, ytr, exposure=etr)

    # Build prediction design matrix for test set using transformed X
    Xte2_refit_mat, _ = __import__('insurance_glm_tools.cluster.backends', fromlist=['build_refit_matrix']).build_refit_matrix(
        Xte2, fc._factor_group_maps, fc._ordinal_factors) if hasattr(fc, '_factor_group_maps') else (None, None)

    # Alternative: use refit GLM to predict from train design
    # The refit result has .predict() that takes the design matrix
    # Build test design matrix the same way as training
    from insurance_glm_tools.cluster.backends import build_refit_matrix, fit_poisson_refit
    Xte_refit, _ = build_refit_matrix(Xte2, fc._factor_group_maps, fc._ordinal_factors)
    offset_te = np.log(np.maximum(ete2, 1e-10))
    mu_r = glm_result.predict(Xte_refit, offset=offset_te)
    tfc = time.perf_counter() - t0

    dev_r=pdev(yte2,mu_r)
    # Number of distinct territory bands
    Xtr_trans = fc.transform(Xtr)
    nbr = int(Xtr_trans["district_num"].nunique())
    # BIC from train fit
    Xtr_refit, _ = build_refit_matrix(Xtr, fc._factor_group_maps, fc._ordinal_factors)
    offset_tr = np.log(np.maximum(etr, 1e-10))
    mu_r_tr = glm_result.predict(Xtr_refit, offset=offset_tr)
    bic_r=bicscore(pll(ytr,mu_r_tr), glm_result.df_model+1, len(df_tr))

    # ARI
    band_map = fc.level_map("district_num")
    df_te["r2vf_band"] = band_map.apply(df_te["district_num"])
    le=LabelEncoder(); df_te["rg"]=le.fit_transform(df_te["r2vf_band"].astype(str))
    ari_r=arivt(df_te,"district","rg")
    dpbr=dev_r/max(ND-nbr,1)
    print(f"R2VF: bands={nbr}, dev={dev_r:.2f}, BIC={bic_r:.2f}, ARI={ari_r:.4f}, t={tfc:.2f}s")
else:
    nbr=0; dev_r=None; bic_r=None; ari_r=None; tfc=0.; dpbr=None

# COMMAND ----------

import json as _j
dbutils.notebook.exit(_j.dumps({"n_bands_manual":5,"n_bands_r2vf":int(nbr),"dev_manual_test":float(dev_m),"dev_r2vf_test":dev_r,
    "bic_manual":float(bic_m),"bic_r2vf":bic_r,"ari_manual":float(ari_m),"ari_r2vf":ari_r,"has_r2vf":HAS_CLUSTERER,
    "dev_per_band_manual":float(dpbm),"dev_per_band_r2vf":dpbr,"total_time_baseline":float(ttb),"total_time_r2vf":float(tfc)}))
