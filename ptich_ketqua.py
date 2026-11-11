# -*- coding: utf-8 -*-
"""Phan tich ket qua ConsisXAI - Credit (crx).

Chay tu thu muc goc ConsisXAI:
    python phan_tich_ket_qua.py

Cac lua chon phuong phap duoc ghi ro:
- ratio_count: so feature giao nhau / K.
- ratio_score: tong diem XAI tren feature giao nhau / K.
- Diem top-K duoc Min-Max scale rieng trong top-K (lua chon cua nhom).
- AIC/BIC dung 1-ratio va kmiss=K-intersection.
- Bootstrap SHAP lay mau lai hang SHAP; Perm lay mau lai cac lan lap Perm.
- Nguong mo phong lay uniform [min(means), max(means)].
"""
from pathlib import Path
import ast, json
import numpy as np
import pandas as pd
from math import log2

BASE = Path("uci_logs")
OUT = Path("ketqua_nhom")
OUT.mkdir(parents=True, exist_ok=True)
N_TRAIN = 552
B = 2000
N_SIM = 1000
MAX_ATTEMPTS = 100
METHODS = [
    "embedded_logit", "embedded_xgboost", "embedded_rf", "embedded_gbm",
    "Information_gain_entropy", "Information_gain_gini", "IV", "Chi/ANOVA"
]
MODELS = ["logit", "xgboost"]

def need(path):
    if not path.exists():
        raise FileNotFoundError(f"Thieu tep {path.resolve()}. Hay chay script tu thu muc goc ConsisXAI.")

# 1. Doc du lieu va scores
paths = [
    BASE/"preprocessed_not_discretized_dataset_crx.csv",
    BASE/"features_scores_crx.csv",
    BASE/"reds_crx_python.json",
    BASE/"core_crx_python.json",
]
for m in MODELS:
    paths += [BASE/f"shap_values_crx_{m}.csv",
              BASE/f"permutation_importance_crx_{m}_training.csv"]
for p in paths: need(p)

df = pd.read_csv(BASE/"preprocessed_not_discretized_dataset_crx.csv")
feats = [c for c in df.columns if c != "+"]
fs = pd.read_csv(BASE/"features_scores_crx.csv", sep=";")
missing = [m for m in METHODS if m not in fs.columns]
if "feature" not in fs.columns or missing:
    raise KeyError(f"features_scores_crx.csv thieu cot feature/phuong phap: {missing}")
sc = (fs[fs.feature.isin(feats)].drop_duplicates("feature")
      .set_index("feature")[METHODS].astype(float).reindex(feats))
if sc.isna().any().any():
    raise ValueError("features_scores_crx.csv co diem thieu/NaN tren cac feature.")
means = sc.mean()

def make_reducts(th):
    return {m: [f for f in feats if sc.loc[f,m] >= th] for m in METHODS}

# 2. Reduct/core da luu va theo Algorithm 3.1
saved_idx = json.loads((BASE/"reds_crx_python.json").read_text(encoding="utf-8"))
core_idx = json.loads((BASE/"core_crx_python.json").read_text(encoding="utf-8"))
R_saved = {}
for m in METHODS:
    if m not in saved_idx: raise KeyError(f"Khong co {m} trong reds_crx_python.json")
    R_saved[m] = [feats[int(i)] for i in saved_idx[m]]
core_saved = [feats[int(i)] for i in core_idx]
th_min = float(means.min())
R_alg = make_reducts(th_min)
core_alg = sorted(set.intersection(*(set(v) for v in R_alg.values())))
same = all(set(R_saved[m]) == set(R_alg[m]) for m in METHODS)
short = min(METHODS, key=lambda m: len(R_saved[m]))
RDS = R_saved[short]

# Kiem tra mot nguong chung co the tao ra cac reduct da luu hay khong.
lower, upper = [], []
for m in METHODS:
    selected = R_saved[m]
    excluded = [f for f in feats if f not in selected]
    if excluded: lower.append(float(sc.loc[excluded,m].max()))
    if selected: upper.append(float(sc.loc[selected,m].min()))
lo = max(lower) if lower else float("-inf")
hi = min(upper) if upper else float("inf")
interval_ok = lo < hi

print("\\n=== REDUCT / CORE ===")
print("Threshold min(means):", th_min)
print("Reduct da luu:", {m:len(R_saved[m]) for m in METHODS})
print("Reduct ngan nhat:", short, len(RDS), RDS)
print("Core da luu:", core_saved)
print("Reduct Algorithm 3.1:", {m:len(R_alg[m]) for m in METHODS})
print("Core Algorithm 3.1:", core_alg)
print("Saved reducts == Algorithm 3.1:", same)
print(f"Khoang nguong chung cho reduct da luu: ({lo:.6f}, {hi:.6f}] ; hop le={interval_ok}")

# 3. Doc SHAP va Permutation
def load_shap(m):
    d = pd.read_csv(BASE/f"shap_values_crx_{m}.csv", sep=";")
    miss = [f for f in feats if f not in d.columns]
    if miss: raise KeyError(f"SHAP {m} thieu feature: {miss}")
    d = d[feats].apply(pd.to_numeric, errors="coerce")
    if d.isna().any().any(): raise ValueError(f"SHAP {m} co NaN/gia tri khong phai so.")
    return d

def parse_array(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        a = np.asarray(x, dtype=float)
    else:
        s = str(x).strip()
        try:
            a = np.asarray(ast.literal_eval(s), dtype=float)
        except (ValueError, SyntaxError):
            a = np.fromstring(s.strip("[]").replace(",", " "), sep=" ")
    a = np.asarray(a, dtype=float).reshape(-1)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError(f"Khong parse duoc permutation array: {x!r}")
    return a

def load_perm(m):
    p = pd.read_csv(BASE/f"permutation_importance_crx_{m}_training.csv", sep=";")
    req = ["Feature", "importances(mean)", "importances"]
    if any(c not in p.columns for c in req):
        raise KeyError(f"Permutation {m} can cac cot {req}")
    p = p.drop_duplicates("Feature").set_index("Feature").reindex(feats)
    if p[req[1:]].isna().any().any():
        raise ValueError(f"Permutation {m} thieu feature/importance.")
    p["importances(mean)"] = pd.to_numeric(p["importances(mean)"], errors="coerce")
    if p["importances(mean)"].isna().any():
        raise ValueError(f"Permutation {m}: importances(mean) khong hop le.")
    p["arr"] = p["importances"].apply(parse_array)
    if p["arr"].map(len).nunique() != 1:
        raise ValueError(f"Permutation {m}: so lan lap khong dong nhat.")
    return p

SHAP = {m:load_shap(m) for m in MODELS}
PERM = {m:load_perm(m) for m in MODELS}
scores = {}
for m in MODELS:
    scores[(m,"SHAP")] = SHAP[m].abs().mean(axis=0)
    scores[(m,"Perm")] = PERM[m]["importances(mean)"]

def topk_scaled(series, k):
    """Top-K voi Min-Max scale trong top-K; day la lua chon cua nhom."""
    if k <= 0: return {}
    top = series.sort_values(ascending=False, kind="stable").head(k)
    v = top.to_numpy(dtype=float)
    span = v.max()-v.min()
    scaled = (v-v.min())/span if span > 0 else np.zeros_like(v)
    return dict(zip(top.index.tolist(), scaled.tolist()))

def ratios(top, ref):
    k = len(ref)
    if k == 0: return float("nan"), float("nan"), 0
    inter = [f for f in ref if f in top]
    return len(inter)/k, sum(float(top[f]) for f in inter)/k, len(inter)

def abic(n, ratio, kmiss, kind):
    """AIC/BIC voi complement ratio va kmiss=K-intersection."""
    if not np.isfinite(ratio) or n <= 0 or kmiss < 0: return float("nan")
    complement = 1.0-float(ratio)
    if complement <= 0: return float("nan")
    fit = -2*log2(complement)
    kind = kind.upper()
    if kind == "AIC": return fit + 2*kmiss
    if kind == "BIC": return fit + kmiss*log2(n)
    raise ValueError("kind phai la AIC hoac BIC")

# 4. Ket qua chinh: reduct ngan nhat da luu va core da luu
rows = []
for (m,x), s in scores.items():
    for set_name, ref in (("reduct",RDS),("core",core_saved)):
        k = len(ref); top = topk_scaled(s,k)
        cnt, sco, inter = ratios(top,ref); kmiss = k-inter
        rows.append(dict(model=m,xai=x,set=set_name,K=k,intersection=inter,missing=kmiss,
            ratio_count=cnt,ratio_score=sco,
            AIC_count=abic(N_TRAIN,cnt,kmiss,"AIC"),
            BIC_count=abic(N_TRAIN,cnt,kmiss,"BIC"),
            AIC_score=abic(N_TRAIN,sco,kmiss,"AIC"),
            BIC_score=abic(N_TRAIN,sco,kmiss,"BIC"),
            top_features=",".join(top)))
res = pd.DataFrame(rows)
print("\\n=== KET QUA CHINH ===")
pd.set_option("display.width",250)
print(res.drop(columns="top_features").round(4).to_string(index=False))
res.to_csv(OUT/"crx_results.csv",index=False)

# 5. Bootstrap: SHAP theo hang mau; Perm theo cac lan lap da luu
print("\\n=== BOOTSTRAP ===")
rng = np.random.default_rng(0)
boot = []
K = len(RDS)
for m in MODELS:
    S = SHAP[m].abs().to_numpy()
    P = np.stack(PERM[m]["arr"].to_numpy())  # feature x repeat
    for b in range(B):
        ix = rng.integers(0,len(S),size=len(S))
        sh = pd.Series(S[ix].mean(axis=0),index=feats)
        jx = rng.integers(0,P.shape[1],size=P.shape[1])
        pm = pd.Series(P[:,jx].mean(axis=1),index=feats)
        out = {}
        for x,s in (("SHAP",sh),("Perm",pm)):
            top = topk_scaled(s,K); cnt,sco,inter = ratios(top,RDS); kmiss=K-inter
            out[x] = (cnt,sco,inter,abic(N_TRAIN,sco,kmiss,"AIC"),abic(N_TRAIN,sco,kmiss,"BIC"))
        row = {"model":m,"bootstrap_iteration":b}
        for x,vals in out.items():
            for i,n in enumerate(["count","score","intersection","AIC","BIC"]):
                row[f"{x}_{n}"] = vals[i]
        boot.append(row)
bt = pd.DataFrame(boot)
summary = []
for m,g in bt.groupby("model"):
    summary.append(dict(model=m,
        SHAP_ratio_mean=g.SHAP_score.mean(),SHAP_ratio_sd=g.SHAP_score.std(),
        Perm_ratio_mean=g.Perm_score.mean(),Perm_ratio_sd=g.Perm_score.std(),
        SHAP_intersection_mean=g.SHAP_intersection.mean(),
        Perm_intersection_mean=g.Perm_intersection.mean(),
        P_SHAP_ratio_gt_Perm=(g.SHAP_score>g.Perm_score).mean(),
        P_equal_ratio=(g.SHAP_score==g.Perm_score).mean(),
        P_SHAP_AIC_lt_Perm=(g.SHAP_AIC<g.Perm_AIC).mean(),
        P_SHAP_BIC_lt_Perm=(g.SHAP_BIC<g.Perm_BIC).mean(),
        SHAP_AIC_mean=g.SHAP_AIC.mean(),Perm_AIC_mean=g.Perm_AIC.mean(),
        SHAP_BIC_mean=g.SHAP_BIC.mean(),Perm_BIC_mean=g.Perm_BIC.mean()))
bt_summary = pd.DataFrame(summary)
print(bt_summary.round(4).T.to_string())
bt.to_csv(OUT/"crx_bootstrap_raw.csv",index=False)
bt_summary.to_csv(OUT/"crx_bootstrap_summary.csv",index=False)

# 6. Do nhay voi nguong
print("\\n=== DO NHAY NGUONG ===")
variants = {
    "nguong_lan_chay_da_luu_0.0155":0.0155,
    "min_trung_binh_Algorithm_3.1":float(means.min()),
    "trung_vi_cac_trung_binh":float(means.median()),
    "trung_binh_cac_trung_binh":float(means.mean())
}
tr = []
for name,th in variants.items():
    Rv = make_reducts(th)
    sm = min(METHODS,key=lambda m:len(Rv[m]))
    ref = Rv[sm]
    row = dict(variant=name,threshold=th,K=len(ref),shortest_method=sm)
    if ref:
        for m in MODELS:
            for x in ["SHAP","Perm"]:
                top=topk_scaled(scores[(m,x)],len(ref))
                cnt,sco,inter=ratios(top,ref); kmiss=len(ref)-inter
                key=f"{m}_{x}"
                row[f"{key}_intersection"]=inter
                row[f"{key}_ratio_count"]=cnt
                row[f"{key}_ratio_score"]=sco
                row[f"{key}_AIC_score"]=abic(N_TRAIN,sco,kmiss,"AIC")
    tr.append(row)
trd=pd.DataFrame(tr)
print(trd.T.to_string())
trd.to_csv(OUT/"crx_threshold_sensitivity.csv",index=False)

# 7. Mo phong nguong ngau nhien: uniform trong [tmin,tmax], gioi han so lan thu
print("\\n=== MO PHONG NGUONG NGAU NHIEN ===")
def get_threshold(tmin,tmax,rng):
    return float(rng.uniform(tmin,tmax))

tmin,tmax=float(means.min()),float(means.max())
rng_threshold=np.random.default_rng(1)
sim=[]
for run in range(N_SIM):
    th=tmin
    Rv=make_reducts(th)
    co=set.intersection(*(set(v) for v in Rv.values()))
    attempts=0
    while len(co)==0 and attempts<MAX_ATTEMPTS:
        th=get_threshold(tmin,tmax,rng_threshold)
        Rv=make_reducts(th)
        co=set.intersection(*(set(v) for v in Rv.values()))
        attempts+=1
    sm=min(METHODS,key=lambda m:len(Rv[m]))
    ref=Rv[sm]
    row=dict(run=run,threshold=th,K=len(ref),core_size=len(co),
             attempts=attempts,core_found=(len(co)>0))
    if ref:
        for m in MODELS:
            for x in ["SHAP","Perm"]:
                top=topk_scaled(scores[(m,x)],len(ref))
                cnt,sco,inter=ratios(top,ref); kmiss=len(ref)-inter
                key=f"{m}_{x}"
                row[f"{key}_intersection"]=inter
                row[f"{key}_ratio_count"]=cnt
                row[f"{key}_ratio_score"]=sco
                row[f"{key}_AIC_score"]=abic(N_TRAIN,sco,kmiss,"AIC")
    sim.append(row)
sd=pd.DataFrame(sim)
sd.to_csv(OUT/"crx_random_threshold.csv",index=False)
valid=sd[sd.K>0]
print("So lan khong tim duoc core:",int((~sd.core_found).sum()))
if not valid.empty:
    print("K min/median/max/nunique:",valid.K.min(),valid.K.median(),valid.K.max(),valid.K.nunique())
    print("Phan bo K:",valid.K.value_counts().sort_index().to_dict())
    print("Nguong min/median/max:",valid.threshold.min(),valid.threshold.median(),valid.threshold.max())
    for m in MODELS:
        a=valid[f"{m}_SHAP_ratio_score"]; b=valid[f"{m}_Perm_ratio_score"]
        aa=valid[f"{m}_SHAP_AIC_score"]; bb=valid[f"{m}_Perm_AIC_score"]
        print(m, "SHAP mean/sd",a.mean(),a.std(),"Perm mean/sd",b.mean(),b.std(),
              "P(SHAP>Perm)",(a>b).mean(),"P(equal)",(a==b).mean(),
              "P(AIC SHAP<Perm)",(aa<bb).mean())
else:
    print("Khong co reduct hop le trong cac lan mo phong.")

# 8. Luu JSON metadata
metadata = dict(
    threshold_min_mean=th_min,
    reducts_saved=R_saved,
    reducts_algorithm_3_1=R_alg,
    shortest_method_saved=short,
    shortest_reduct_saved=RDS,
    core_saved=core_saved,
    core_algorithm_3_1=core_alg,
    saved_reducts_match_algorithm_3_1=bool(same),
    saved_reduct_threshold_interval=dict(lower_exclusive=lo,upper_inclusive=hi,exists=bool(interval_ok)),
    xai_topk_normalization="Min-Max within selected top-K; group analysis choice",
    ratio_definitions=dict(ratio_count="intersection_count/K",ratio_score="sum_scaled_XAI_scores_on_intersection/K"),
    aic_bic=dict(ratio_complement="1-ratio",parameter_count="K-intersection",N=N_TRAIN),
    bootstrap=dict(B=B,SHAP="resample SHAP rows",Permutation="resample saved permutation repeats")
)
(OUT/"crx_reduct_comparison.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")

# 9. Bieu do
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})

if not valid.empty:
    fig,ax=plt.subplots(1,2,figsize=(8.6,3.2))
    vc=valid.K.value_counts().sort_index()
    pct=vc.values/len(valid)*100
    ax[0].bar([str(k) for k in vc.index],pct)
    for i,c in enumerate(vc.values):
        ax[0].text(i,pct[i]+0.5,f"{c} lan",ha="center")
    ax[0].set_xlabel("Do dai reduct ngan nhat K")
    ax[0].set_ylabel("Ty le trong cac lan mo phong hop le (%)")
    ax[0].set_title("(a) Reduct theo nguong ngau nhien")
    for m,label in (("logit","Logit"),("xgboost","XGBoost")):
        delta=(valid[f"{m}_SHAP_AIC_score"]-valid[f"{m}_Perm_AIC_score"]).dropna()
        ax[1].hist(delta,bins=20,alpha=.6,label=label)
    ax[1].axvline(0,color="black",lw=1)
    ax[1].set_xlabel("AIC(SHAP) - AIC(Perm)")
    ax[1].set_ylabel("So lan mo phong")
    ax[1].legend()
    ax[1].set_title("(b) Chenh lech AIC")
    fig.tight_layout()
    fig.savefig(OUT/"fig4_3.png",dpi=200)
    plt.close(fig)

fig,ax=plt.subplots(2,2,figsize=(8.6,5.4))
for j,m in enumerate(MODELS):
    g=bt[bt.model==m]
    label="Logit" if m=="logit" else "XGBoost"
    ax[0,j].hist(g.SHAP_score.dropna(),bins=30,alpha=.6,label="SHAP")
    ax[0,j].hist(g.Perm_score.dropna(),bins=30,alpha=.6,label="Permutation")
    ax[0,j].set_title(f"{label}: reduct ratio")
    ax[0,j].set_xlabel("Ratio theo diem XAI")
    ax[0,j].set_ylabel("So mau bootstrap")
    ax[0,j].legend()
    ax[1,j].hist(g.SHAP_AIC.dropna(),bins=30,alpha=.6,label="SHAP")
    ax[1,j].hist(g.Perm_AIC.dropna(),bins=30,alpha=.6,label="Permutation")
    ax[1,j].set_title(f"{label}: AIC")
    ax[1,j].set_xlabel("AIC")
    ax[1,j].set_ylabel("So mau bootstrap")
    ax[1,j].legend()
fig.tight_layout()
fig.savefig(OUT/"fig4_4.png",dpi=200)
plt.close(fig)

print("\\nHOAN TAT. Ket qua tai:",OUT.resolve())
