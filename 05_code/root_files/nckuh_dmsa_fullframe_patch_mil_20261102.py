"""Local-patch MIL on clean archived NCKUH full-frame kidney ROIs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

import nckuh_dmsa_fullframe_clean_probe_20261102 as F
import nckuh_dmsa_fullframe_deep_probe_20261102 as D
import nckuh_dmsa_fullframe_rank_probe_20261102 as R
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_patch_mil_20261102 import PatchBagModel
from nckuh_only_study_20261102 import digest, metric, write_json
from nckuh_ultrasound_probe_20261102 import WEIGHT

ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullframe_patch_mil_20261102"
ARMS = ("mean_pool", "gated_attention", "gated_global_rank", "gated_global_and_unilateral_rank")
GROUPS = ARMS + tuple(f"{arm}_plus_deep_radiomics" for arm in ARMS)
REFERENCE = "deep_concat_plus_clean_radiomics"
EPOCHS = (10, 20, 40, 80)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered clean patch-MIL protocol")
    F.require_lock(); D.require_lock()
    inputs = [F.OUT / "patient_assignments.csv", F.OUT / "deidentified_image_manifest.csv",
              F.OUT / "archive_extraction_audit.json", D.OUT / "development_predictions.csv",
              WEIGHT, Path(__file__)]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same clean-full-frame111 patients,212 kidneys,418 images and exact70/20/21 roles",
        "patches": "Frozen USF-MAE ViT-B16 patch tokens from grayscale clean ROI; retain patches with at least80% kidney-mask coverage; no positive patch labels",
        "preprocessing": "L2 normalize; sample64 patches per fitting patient; fit whitened randomized PCA32 only on fitting patients; compact each kidney bag to at most128 deterministic evenly spaced patches",
        "models": "Capacity-matched mean pooling control and Ilse-style gated attention; balanced kidney BCE; optional global RankNet0.2 and same-patient unilateral RankNet0.5",
        "selection": "Within each70-patient training role, fixed55/15 stratified inner split selects10/20/40/80 epochs by mean(AUROC,AP), tie lower epoch; reset/refit all70",
        "fusion": "Fixed equal-probability average with completed deep-concat plus clean-radiomics branch",
        "prohibited": "No patch diagnosis, no development-selected loss/fusion weight, no test scoring; attention is contribution, not lesion truth",
        "target": "Development AUROC>0.8 and AP>0.8; later locked confirmation required",
        "arms": list(ARMS), "groups": list(GROUPS), "epochs": list(EPOCHS),
        "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {"source_sha256":digest(__file__),
        "protocol_sha256":digest(OUT / "protocol.json"),
        "inputs":{str(path):digest(path) for path in inputs}})
    print("Registered clean full-frame local-patch MIL", flush=True)


def require_lock() -> None:
    lock=json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__)==lock["source_sha256"]
    assert digest(OUT / "protocol.json")==lock["protocol_sha256"]
    for name,expected in lock["inputs"].items(): assert digest(Path(name))==expected,name


def encode() -> None:
    require_lock()
    if (OUT / "patch_cache_audit.json").exists():
        raise FileExistsError("Preserve completed clean patch cache")
    import timm
    _,_,images=F.matched_reference()
    checkpoint=torch.load(WEIGHT,map_location="cpu",weights_only=True)
    model=timm.create_model("vit_base_patch16_224",pretrained=False,num_classes=0)
    keys=set(model.state_dict()); encoder={k:v for k,v in checkpoint.items() if k in keys}
    assert set(encoder)==keys and len(keys)==150
    model.load_state_dict(encoder,strict=True); model=model.cuda().eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device="cuda").view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device="cuda").view(1,3,1,1)
    patches=[]; offsets=[0]; selected_counts=[]
    with torch.inference_mode():
        for start in range(0,len(images),32):
            canvases=[]; selections=[]
            for ident in images.image_id.iloc[start:start+32]:
                gray=cv2.imread(str(F.OUT/"percentile"/f"{ident}.png"),0)
                mask=cv2.imread(str(F.OUT/"masks"/f"{ident}.png"),0)>0
                canvas,binary=D.tight_canvas(gray,mask)
                coverage=binary.reshape(14,16,14,16).mean((1,3)).ravel()
                selected=coverage>=.8
                if not selected.any(): selected[np.argmax(coverage)]=True
                canvases.append(canvas); selections.append(selected)
            tensor=torch.from_numpy(np.stack(canvases).transpose(0,3,1,2).copy()).cuda().float()/255
            tokens=model.forward_features((tensor-mean)/std)[:,1:].cpu().numpy()
            for token,selected in zip(tokens,selections):
                current=token[selected].astype(np.float16)
                patches.append(current); selected_counts.append(len(current)); offsets.append(offsets[-1]+len(current))
    flat=np.concatenate(patches)
    np.save(OUT/"usf_clean_roi_patches_float16.npy",flat)
    np.save(OUT/"patch_offsets.npy",np.asarray(offsets,np.int64))
    write_json(OUT/"patch_cache_audit.json",{
        "completed_utc":datetime.now(timezone.utc).isoformat(),"images":418,
        "patches":len(flat),"shape":list(flat.shape),
        "selected_per_image_min_median_max":[int(np.min(selected_counts)),float(np.median(selected_counts)),int(np.max(selected_counts))],
        "labels_used":[],"encoder_frozen":True,
        "patch_sha256":digest(OUT/"usf_clean_roi_patches_float16.npy"),
        "offset_sha256":digest(OUT/"patch_offsets.npy")})
    print(f"Cached {len(flat)} clean in-mask USF patches",flush=True)


def patch_data(images,kidneys):
    raw=np.load(OUT/"usf_clean_roi_patches_float16.npy",mmap_mode="r")
    offsets=np.load(OUT/"patch_offsets.npy")
    assert len(offsets)==len(images)+1 and offsets[-1]==len(raw)
    values=np.asarray(raw,dtype=np.float32); values/=np.linalg.norm(values,axis=1,keepdims=True).clip(1e-8)
    image_index=np.repeat(np.arange(len(images)),np.diff(offsets))
    patch_aliases=images.patient_alias.to_numpy()[image_index]
    patch_kidneys=images.kidney_alias.to_numpy()[image_index]
    groups=[np.flatnonzero(patch_kidneys==alias) for alias in kidneys.kidney_alias]
    assert all(len(group)>0 for group in groups)
    return values,patch_aliases,groups


def prepare_bags(values,patch_aliases,groups,train_aliases,seed):
    from sklearn.decomposition import PCA
    rng=np.random.default_rng(seed+7000); sample=[]
    for alias in sorted(train_aliases):
        available=np.flatnonzero(patch_aliases==alias)
        sample.extend(rng.choice(available,64,replace=len(available)<64))
    sample=np.asarray(sample,np.int64)
    pca=PCA(32,whiten=True,random_state=seed,svd_solver="randomized").fit(values[sample])
    transformed=pca.transform(values).astype(np.float32)
    compact=[]
    for group in groups:
        if len(group)>128:
            group=group[np.linspace(0,len(group)-1,128).round().astype(int)]
        compact.append(group)
    width=max(map(len,compact)); bags=np.zeros((len(compact),width,32),np.float32); mask=np.zeros((len(compact),width),bool)
    for index,group in enumerate(compact):
        bags[index,:len(group)]=transformed[group]; mask[index,:len(group)]=True
    return torch.from_numpy(bags).cuda(),torch.from_numpy(mask).cuda(),{"pca":pca,"sample":sample}


def train_model(arm,seed,bags,mask,labels,fit,pairs,epochs,evaluation):
    torch.manual_seed(seed+37)
    mode="mean" if arm==ARMS[0] else "gated"
    model=PatchBagModel(mode).cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    target=labels[fit]; weights=torch.where(target>0,len(target)/(2*target.sum()),len(target)/(2*(len(target)-target.sum())))
    history={}
    for epoch in range(1,max(epochs)+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits,_=model(bags[fit],mask[fit])
        loss=(nn.functional.binary_cross_entropy_with_logits(logits,target,reduction="none")*weights).mean()
        if arm in (ARMS[2],ARMS[3]): loss=loss+.2*R.rank_loss(logits,target)
        if arm==ARMS[3]: loss=loss+.5*R.unilateral_rank_loss(logits,pairs)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
        if epoch in epochs and evaluation is not None:
            model.eval()
            with torch.inference_mode(): history[epoch]=torch.sigmoid(model(bags[evaluation],mask[evaluation])[0]).cpu().numpy()
    return model,history


def run() -> None:
    from sklearn.model_selection import train_test_split
    require_lock()
    if (OUT/"development_predictions.csv").exists(): raise FileExistsError("Preserve completed clean patch-MIL predictions")
    patients,kidneys,images=F.matched_reference(); y=torch.from_numpy(kidneys.kidney_APN.to_numpy(np.float32)).cuda()
    values,patch_aliases,groups=patch_data(images,kidneys)
    assignments=pd.read_csv(F.OUT/"patient_assignments.csv")
    reference=pd.read_csv(D.OUT/"development_predictions.csv"); reference=reference[reference.group.eq(REFERENCE)].set_index(["seed","kidney_alias"]).score
    records=[]; predictions=[]; selections=[]; (OUT/"models").mkdir(exist_ok=True)
    for seed in F.SEEDS:
        mapping=assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        outer_train=np.flatnonzero(kidneys.patient_alias.map(mapping).eq("train")); dev=np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        train_patients=patients[patients.patient_alias.isin(mapping[mapping.eq("train")].index)].reset_index(drop=True)
        fit_patient,select_patient=train_test_split(np.arange(len(train_patients)),train_size=55,random_state=seed+4000,stratify=train_patients.DMSA_APN)
        fit_alias=train_patients.patient_alias.iloc[fit_patient].to_numpy(); select_alias=train_patients.patient_alias.iloc[select_patient].to_numpy()
        fit=np.flatnonzero(kidneys.patient_alias.isin(fit_alias)); select=np.flatnonzero(kidneys.patient_alias.isin(select_alias))
        inner_bags,inner_mask,inner_proof=prepare_bags(values,patch_aliases,groups,fit_alias,seed)
        selected={}
        for arm in ARMS:
            pairs=R.unilateral_pairs(fit,kidneys,kidneys.kidney_APN.to_numpy(int))
            _,history=train_model(arm,seed,inner_bags,inner_mask,y,fit,pairs,EPOCHS,select)
            choices=[]
            for epoch,score in history.items():
                result=metric(kidneys.kidney_APN.to_numpy(int)[select],score)
                choices.append((-(result["auroc"]+result["ap"])/2,epoch,result))
            _,chosen,result=min(choices,key=lambda item:(item[0],item[1])); selected[arm]=chosen
            selections.append({"seed":seed,"arm":arm,"selected_epoch":chosen,"inner_auroc":result["auroc"],"inner_ap":result["ap"]})
        del inner_bags,inner_mask; torch.cuda.empty_cache()
        train_alias=patients.patient_alias[patients.patient_alias.map(mapping).eq("train")].to_numpy()
        bags,mask,proof=prepare_bags(values,patch_aliases,groups,train_alias,seed)
        joblib.dump({"inner":inner_proof,"full":proof},OUT/"models"/f"{seed}_preprocess.joblib",compress=3)
        scores={}
        for arm in ARMS:
            pairs=R.unilateral_pairs(outer_train,kidneys,kidneys.kidney_APN.to_numpy(int))
            model,_=train_model(arm,seed,bags,mask,y,outer_train,pairs,(selected[arm],),None)
            model.eval()
            with torch.inference_mode():
                logits,attention=model(bags[dev],mask[dev]); scores[arm]=torch.sigmoid(logits).cpu().numpy()
            torch.save({"state_dict":model.state_dict(),"arm":arm,"epoch":selected[arm],"seed":seed},OUT/"models"/f"{seed}_{arm}.pth")
            np.savez_compressed(OUT/f"{seed}_{arm}_development_attention.npz",attention=attention.cpu().numpy(),mask=mask[dev].cpu().numpy())
        ref=np.asarray([reference.loc[(seed,alias)] for alias in kidneys.kidney_alias.iloc[dev]])
        for arm in ARMS: scores[f"{arm}_plus_deep_radiomics"]=(scores[arm]+ref)/2
        labels=kidneys.kidney_APN.to_numpy(int)
        for group in GROUPS:
            result=metric(labels[dev],scores[group]); records.append({"seed":seed,"role":"development","group":group,"patients":20,"kidneys":len(dev),**result})
            for local,index in enumerate(dev):
                predictions.append({"seed":seed,"group":group,"patient_alias":kidneys.patient_alias.iloc[index],"kidney_alias":kidneys.kidney_alias.iloc[index],"DMSA_APN":kidneys.DMSA_APN.iloc[index],"side":kidneys.side.iloc[index],"label":int(labels[index]),"score":float(scores[group][local])})
        print(f"Completed clean patch-MIL draw {seed}",flush=True)
        del bags,mask; torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(OUT/"development_metrics.csv",index=False); pd.DataFrame(predictions).to_csv(OUT/"development_predictions.csv",index=False)
    pd.DataFrame(selections).to_csv(OUT/"inner_epoch_selections.csv",index=False)
    summary=pd.DataFrame(records).groupby("group")[["auroc","ap"]].mean().sort_values(["auroc","ap"],ascending=False); summary.to_csv(OUT/"development_summary.csv")
    achieved=summary[(summary.auroc>.8)&(summary.ap>.8)].index.tolist(); write_json(OUT/"development_decision.json",{"groups_above_both_development_targets":achieved,"test_scored":False,"goal_achieved":False,"reason":"Development-only subgroup cannot complete full objective"})
    print(summary.to_string(),flush=True)


def bootstrap() -> None:
    require_lock(); frame=pd.read_csv(OUT/"development_predictions.csv"); previous=pd.read_csv(D.OUT/"development_predictions.csv"); reference=previous[previous.group.eq(REFERENCE)]
    ci,paired,samples=cluster_intervals(pd.concat([frame,reference],ignore_index=True),list(GROUPS)+[REFERENCE],F.SEEDS,REFERENCE)
    ci.to_csv(OUT/"development_confidence_intervals.csv",index=False); paired.to_csv(OUT/"development_paired_intervals.csv",index=False); np.save(OUT/"development_patient_cluster_bootstrap.npy",samples)
    write_json(OUT/"completion_audit.json",{"completed_utc":datetime.now(timezone.utc).isoformat(),"patients":111,"kidneys":212,"images":418,"test_scored":False,"scored_unit":"kidney","bootstrap_cluster":"patient"})
    print("Clean patch-MIL intervals completed",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("stage",choices=["prepare","encode","run","bootstrap"]); args=parser.parse_args(); globals()[args.stage]()
