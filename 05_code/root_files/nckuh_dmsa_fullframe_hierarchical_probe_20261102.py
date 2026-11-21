"""Hierarchical affected-kidney-count and laterality modeling on clean full-frame features."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import nckuh_dmsa_fullframe_clean_probe_20261102 as F
import nckuh_dmsa_fullframe_deep_probe_20261102 as D
import nckuh_dmsa_fullframe_patch_mil_20261102 as M
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json

ROOT=F.ROOT
OUT=ROOT/"02_experiments/NCKUH_DMSA_fullframe_hierarchical_probe_20261102"
GROUPS=("affected_count_then_laterality","four_state_powerset",
        "affected_count_then_laterality_plus_deep","four_state_powerset_plus_deep",
        "affected_count_then_laterality_plus_patch_deep")
REFERENCE="deep_concat_plus_clean_radiomics"
CODE={"N":0,"L":1,"R":2,"B":3}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"protocol.json").exists(): raise FileExistsError("Preserve registered hierarchical protocol")
    F.require_lock();D.require_lock();M.require_lock()
    inputs=[F.OUT/"patient_assignments.csv",D.OUT/"radimagenet_gray.npy",D.OUT/"radimagenet_color.npy",D.OUT/"usf_gray.npy",
            D.OUT/"development_predictions.csv",M.OUT/"development_predictions.csv",Path(__file__)]
    protocol={"created_utc":datetime.now(timezone.utc).isoformat(),
        "cohort":"Same clean-full-frame111 patients,212 imaged kidneys,418 images and exact70/20/21 roles",
        "representation":"Frozen three-encoder patient pair preserving left/right and longitudinal/transverse evidence",
        "hierarchy":"Multinomial affected-kidney count N=0,L/R=1,B=2 from symmetric sum/absolute-difference features, plus L-versus-R head fitted only to unilateral training patients from signed difference/sum features",
        "marginals":"left=P(count2)+P(count1)*P(left|unilateral); right=P(count2)+P(count1)*(1-P(left|unilateral))",
        "control":"Four-state N/L/R/B label-powerset LR with left/right swap augmentation and symmetrized inference",
        "preprocessing":"Train-only median imputation, standardization, whitened PCA64 count/four-state and PCA16 laterality; balanced LR C0.1",
        "fusion":"Fixed equal probability averages with completed deep-radiomics and patch-deep branches",
        "prohibited":"No opposite diagnosis or DMSA code as inference input, no development-selected weight/threshold, no exploratory-test scoring",
        "target":"Development AUROC>0.8 and AP>0.8; later locked confirmation required",
        "groups":list(GROUPS),"inputs":{str(p):digest(p) for p in inputs}}
    write_json(OUT/"protocol.json",protocol);write_json(OUT/"run_lock.json",{"source_sha256":digest(__file__),"protocol_sha256":digest(OUT/"protocol.json"),"inputs":{str(p):digest(p) for p in inputs}})
    print("Registered hierarchical affected-count/laterality probe",flush=True)


def require_lock():
    lock=json.loads((OUT/"run_lock.json").read_text(encoding="utf-8"));assert digest(__file__)==lock["source_sha256"];assert digest(OUT/"protocol.json")==lock["protocol_sha256"]
    for name,sha in lock["inputs"].items():assert digest(Path(name))==sha,name


def swap_pairs(values):
    half=values.shape[1]//2;assert values.shape[1]%2==0
    return np.c_[values[:,half:],values[:,:half]]


def preprocessing(components,seed):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler(),PCA(components,whiten=True,random_state=seed))


def logistic(seed,multiclass=False):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=.1,class_weight="balanced",solver="lbfgs" if multiclass else "liblinear",max_iter=3000,random_state=seed)


def marginal_scores(count_model,count_x,side_model,side_x):
    count_order=[list(count_model.classes_).index(i) for i in (0,1,2)]
    count=count_model.predict_proba(count_x)[:,count_order]
    left_index=list(side_model.classes_).index(0)
    left_given=side_model.predict_proba(side_x)[:,left_index]
    left=count[:,2]+count[:,1]*left_given;right=count[:,2]+count[:,1]*(1-left_given)
    return left,right


def powerset_scores(model,preprocessor,raw_values):
    order=[list(model.classes_).index(i) for i in range(4)]
    direct=model.predict_proba(preprocessor.transform(raw_values))[:,order]
    swapped=model.predict_proba(preprocessor.transform(swap_pairs(raw_values)))[:,order][:,[0,2,1,3]]
    probability=(direct+swapped)/2
    return probability[:,1]+probability[:,3],probability[:,2]+probability[:,3]


def run():
    require_lock()
    if (OUT/"development_predictions.csv").exists():raise FileExistsError("Preserve completed hierarchical predictions")
    patients,kidneys,images=F.matched_reference();labels=kidneys.kidney_APN.to_numpy(int)
    image=np.concatenate([np.load(D.OUT/name) for name in ("radimagenet_gray.npy","radimagenet_color.npy","usf_gray.npy")],axis=1)
    pairs=F.patient_pair_features(image,images,patients);half=pairs.shape[1]//2;left,right=pairs[:,:half],pairs[:,half:]
    count_values=np.c_[left+right,np.abs(left-right)];side_values=np.c_[left-right,left+right]
    code=patients.DMSA_APN.map(CODE).to_numpy(int);count=np.asarray([0 if c==0 else 2 if c==3 else 1 for c in code]);side=np.asarray([0 if c==1 else 1 if c==2 else -1 for c in code])
    assignments=pd.read_csv(F.OUT/"patient_assignments.csv")
    deep=pd.read_csv(D.OUT/"development_predictions.csv");deep=deep[deep.group.eq(REFERENCE)].set_index(["seed","kidney_alias"]).score
    patch=pd.read_csv(M.OUT/"development_predictions.csv");patch=patch[patch.group.eq("gated_attention_plus_deep_radiomics")].set_index(["seed","kidney_alias"]).score
    records=[];predictions=[];(OUT/"models").mkdir(exist_ok=True)
    patient_lookup={alias:i for i,alias in enumerate(patients.patient_alias)}
    for seed in F.SEEDS:
        mapping=assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train=np.flatnonzero(patients.patient_alias.map(mapping).eq("train"));dev_pat=np.flatnonzero(patients.patient_alias.map(mapping).eq("development"))
        count_pre=preprocessing(64,seed).fit(count_values[train]);count_model=logistic(seed,True).fit(count_pre.transform(count_values[train]),count[train])
        unilateral=train[side[train]>=0];side_pre=preprocessing(16,seed).fit(side_values[unilateral]);side_model=logistic(seed).fit(side_pre.transform(side_values[unilateral]),side[unilateral])
        left_h,right_h=marginal_scores(count_model,count_pre.transform(count_values),side_model,side_pre.transform(side_values))
        augmented=np.r_[pairs[train],swap_pairs(pairs[train])];augmented_code=np.r_[code[train],np.asarray([0 if c==0 else 2 if c==1 else 1 if c==2 else 3 for c in code[train]])]
        power_pre=preprocessing(64,seed).fit(augmented);power_model=logistic(seed,True).fit(power_pre.transform(augmented),augmented_code)
        left_p,right_p=powerset_scores(power_model,power_pre,pairs)
        joblib.dump({"count_pre":count_pre,"count_model":count_model,"side_pre":side_pre,"side_model":side_model,"powerset_pre":power_pre,"powerset_model":power_model},OUT/"models"/f"{seed}_hierarchical.joblib",compress=3)
        dev=np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"));base_h=[];base_p=[]
        for index in dev:
            patient=patient_lookup[kidneys.patient_alias.iloc[index]];is_left=kidneys.side.iloc[index]=="left"
            base_h.append(left_h[patient] if is_left else right_h[patient]);base_p.append(left_p[patient] if is_left else right_p[patient])
        base_h=np.asarray(base_h);base_p=np.asarray(base_p);aliases=kidneys.kidney_alias.iloc[dev]
        deep_score=np.asarray([deep.loc[(seed,a)] for a in aliases]);patch_score=np.asarray([patch.loc[(seed,a)] for a in aliases])
        scores={GROUPS[0]:base_h,GROUPS[1]:base_p,GROUPS[2]:(base_h+deep_score)/2,GROUPS[3]:(base_p+deep_score)/2,GROUPS[4]:(base_h+patch_score)/2}
        for group in GROUPS:
            result=metric(labels[dev],scores[group]);records.append({"seed":seed,"role":"development","group":group,"patients":20,"kidneys":len(dev),**result})
            for local,index in enumerate(dev):predictions.append({"seed":seed,"group":group,"patient_alias":kidneys.patient_alias.iloc[index],"kidney_alias":kidneys.kidney_alias.iloc[index],"DMSA_APN":kidneys.DMSA_APN.iloc[index],"side":kidneys.side.iloc[index],"label":int(labels[index]),"score":float(scores[group][local])})
        print(f"Completed hierarchical draw {seed}",flush=True)
    pd.DataFrame(records).to_csv(OUT/"development_metrics.csv",index=False);pd.DataFrame(predictions).to_csv(OUT/"development_predictions.csv",index=False)
    summary=pd.DataFrame(records).groupby("group")[["auroc","ap"]].mean().sort_values(["auroc","ap"],ascending=False);summary.to_csv(OUT/"development_summary.csv")
    achieved=summary[(summary.auroc>.8)&(summary.ap>.8)].index.tolist();write_json(OUT/"development_decision.json",{"groups_above_both_development_targets":achieved,"test_scored":False,"goal_achieved":False,"reason":"Development-only subgroup cannot complete full objective"});print(summary.to_string(),flush=True)


def bootstrap():
    require_lock();frame=pd.read_csv(OUT/"development_predictions.csv");previous=pd.read_csv(D.OUT/"development_predictions.csv");reference=previous[previous.group.eq(REFERENCE)]
    ci,paired,samples=cluster_intervals(pd.concat([frame,reference],ignore_index=True),list(GROUPS)+[REFERENCE],F.SEEDS,REFERENCE)
    ci.to_csv(OUT/"development_confidence_intervals.csv",index=False);paired.to_csv(OUT/"development_paired_intervals.csv",index=False);np.save(OUT/"development_patient_cluster_bootstrap.npy",samples)
    write_json(OUT/"completion_audit.json",{"completed_utc":datetime.now(timezone.utc).isoformat(),"patients":111,"kidneys":212,"images":418,"test_scored":False,"scored_unit":"kidney","bootstrap_cluster":"patient"});print("Hierarchical intervals completed",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=["prepare","run","bootstrap"]);args=parser.parse_args();globals()[args.stage]()
