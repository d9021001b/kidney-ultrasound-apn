"""Two fixed training-only augmentation conditions for the development-selected texture family."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import logging
from pathlib import Path
import shutil

import cv2
import joblib
import numpy as np
import pandas as pd

import renal_dinov3_native_probe as N
import renal_interpretable_probe as I
import renal_local_features as L
import renal_random_validation as R
import renal_voting_validation as V

OUT=R.ROOT/"CYCH_texture_augmentation_20260913"
VARIANTS=[(0,1.0),(-5,1.0),(5,1.0),(0,.9),(-5,.9),(5,.9),(0,1.1),(-5,1.1),(5,1.1)]
GROUPS={"AUG-none":[0],"AUG-geometry":[0,1,2],"AUG-geometry-gain":list(range(9))}


def prepare():
    prior=json.loads((I.OUT/"development_decision.json").read_text())
    assert prior["selected"]=="I3-direction-original" and prior["vote"]["promoted"]
    OUT.mkdir(exist_ok=True)
    for sub in ("source","features","models","figures_600dpi"):
        (OUT/sub).mkdir(exist_ok=True)
    protocol=dict(
        stage="Planned conservative augmentation after a feature family passes development. Prior exploratory-test decline is known; this is further exploration, not independent confirmation.",
        source_selection="I3-direction-original selected using development only; fixed same25 PyRadiomics descriptors and50 patient mean/std features.",
        source_decision_sha256=R.sha256(I.OUT/"development_decision.json"),variants=VARIANTS,groups=GROUPS,
        transform="Angles0,-5,+5 degrees on jointly padded image/mask, image bilinear and mask nearest. Gain0.9,1,1.1 multiplies normalized grayscale, clip[0,255], without renormalizing. Unmodified inputs and small5-degree limit checked on fixed I0001/I0002; preservation of microtexture is not a clinical guarantee. Calipers retained. No GAN.",
        masks="Same13x13 erosion and fixed3x3 fallback as I3; compute after common image/mask rotation. First-order+coif1 GLRLM exactly I3 settings.",
        partition="All ten fixed58/20/20 patient assignments unchanged. Augmented arrays may be precomputed without labels but only training-patient rows enter fit. Development and test use original-variant0 only.",
        patient_unit="All images belonging to a patient transformed together per variant, then image mean/std. Equal number of variants per training patient; sample_weight=1/K so total LR weight per patient remains1. No synthetic patient count.",
        head="Fixed all-feature L2 LR C=.1,balanced,liblinear; imputer/constant-removal/scaler fit training variants only. Original control has1 variant.",
        gate="Against matched unaugmented I3: development mean AUROC improvement>=.02,AP>=.01,and>=6/10 joint wins. At most one eligible winner by mean(AUROC+AP)/2 then group ID. Only then evaluate one fixed RAD+DN+augmented feature vote; promotion relative original pair requires .03AUROC/.02AP and>=6 joint wins.",
        limitation="All98patients previously explored. Labels/clinical contradictions unchanged; display-image augmentations not simulated acquisition physics. No test-driven parameter, seed or threshold tuning.",
        references=I.REFERENCES)
    protocol=json.loads(json.dumps(protocol))
    path=OUT/"protocol.json"
    if path.exists():
        assert json.loads(path.read_text())==protocol
    else:
        R.write_json(path,protocol)
    lock=OUT/"protocol_locked.json"
    if not lock.exists():
        R.write_json(lock,dict(utc=datetime.now(timezone.utc).isoformat(),sha256=R.sha256(path),source_sha256=R.sha256(Path(__file__))))
    for name in ("patients.csv","image_manifest.csv","random_patient_assignments.csv"):
        shutil.copy2(I.OUT/name,OUT/name)
    shutil.copy2(__file__,OUT/"source"/Path(__file__).name)
    print("Locked two training-only augmentation arms and weight-matched original control",flush=True)


def require_lock():
    lock=json.loads((OUT/"protocol_locked.json").read_text())
    assert lock["sha256"]==R.sha256(OUT/"protocol.json") and lock["source_sha256"]==R.sha256(Path(__file__))


def augmented_input(gray,mask,angle,gain):
    im,om,_,_=I.rotate_native(gray,mask,False)
    side=len(im)
    if angle:
        matrix=cv2.getRotationMatrix2D((side/2,side/2),angle,1)
        im=cv2.warpAffine(im,matrix,(side,side),flags=cv2.INTER_LINEAR)
        om=cv2.warpAffine(om,matrix,(side,side),flags=cv2.INTER_NEAREST)
    im=np.clip(im*gain,0,255)*om
    selected=cv2.erode(om,np.ones((13,13),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    fallback=bool(selected.sum()<64)
    if fallback:
        selected=cv2.erode(om,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    assert selected.sum()>=64
    return im.astype(np.float32),selected,fallback


def extract():
    require_lock()
    import radiomics
    import SimpleITK as sitk
    from radiomics import featureextractor
    radiomics.setVerbosity(logging.ERROR)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    ext=featureextractor.RadiomicsFeatureExtractor(binWidth=16,normalize=False,distances=[1,2],force2D=False,minimumROIDimensions=2)
    ext.disableAllImageTypes()
    ext.enableImageTypeByName("Original")
    ext.enableImageTypeByName("Wavelet",customArgs={"wavelet":"coif1","level":1})
    ext.disableAllFeatures()
    ext.enableFeaturesByName(**I.RAD_FEATURES)
    df,patients=I.read("image_manifest.csv"),I.read("patients.csv")
    original=pd.read_csv(I.OUT/"features/I3-direction-original_image.csv")
    names=list(original.columns[1:])
    features=np.empty((9,len(df),len(names)),np.float32)
    audit=[]
    for i,row in enumerate(df.itertuples()):
        assert original.image_id.iloc[i]==row.image_id
        gray=cv2.imread(str(L.OUT/"images/percentile"/f"{row.image_id}.png"),0)
        mp=L.OUT/"masks"/f"{row.image_id}.png"
        assert R.sha256(mp)==row.mask_sha256
        mask=(cv2.imread(str(mp),0)>0).astype(np.uint8)
        for j,(angle,gain) in enumerate(VARIANTS):
            im,selected,fallback=augmented_input(gray,mask,angle,gain)
            result=ext.execute(sitk.GetImageFromArray(im),sitk.GetImageFromArray(selected))
            values=np.array([float(result[n]) for n in names],np.float32)
            assert np.isfinite(values).all()
            features[j,i]=values
            audit.append(dict(image_id=row.image_id,variant=j,angle=angle,gain=gain,erosion_fallback=fallback))
        assert np.allclose(features[0,i],original[names].iloc[i].to_numpy(float),rtol=1e-5,atol=1e-5)
        if i%40==0:
            print(f"Augmented texture features {i+1}/{len(df)}",flush=True)
    np.save(OUT/"features/image_variants.npy",features)
    np.save(OUT/"features/patient_variants.npy",np.stack([R.aggregate(f,df,patients) for f in features]))
    pd.DataFrame(audit).to_csv(OUT/"augmentation_input_audit.csv",index=False)
    R.write_json(OUT/"feature_names.json",names)


def evaluate():
    require_lock()
    patients,split=I.read("patients.csv"),I.read("random_patient_assignments.csv")
    arrays=np.load(OUT/"features/patient_variants.npy")
    labels=patients.label.to_numpy(int)
    scores,records,predictions,audit={},[],[],[]
    for seed in R.SEEDS:
        ix=N.indices_for(patients,split,seed)
        scores[seed]={}
        for group,variants in GROUPS.items():
            train=np.concatenate([arrays[v,ix["train"]] for v in variants])
            y=np.tile(labels[ix["train"]],len(variants))
            weights=np.full(len(train),1/len(variants))
            model=N.full_head(seed)
            model.fit(train,y,logisticregression__sample_weight=weights)
            score=model.predict_proba(arrays[0])[:,1]
            joblib.dump(model,OUT/"models"/f"{group}_{seed}.joblib",compress=3)
            scores[seed][group]=score
            records.append(dict(seed=seed,group=group,role="development_validation",**V.score_metrics(labels[ix["validation"]],score[ix["validation"]])))
            audit.append(dict(seed=seed,group=group,training_patients=patients.patient_alias.iloc[ix["train"]].tolist(),variants=variants,
                effective_total_sample_weight=float(weights.sum()),fitted_rows=len(train),evaluation_variant=0))
    md=pd.DataFrame(records).set_index(["group","seed"])
    candidates=[]
    for group in list(GROUPS)[1:]:
        delta=md.loc[group,["auroc","average_precision"]]-md.loc["AUG-none",["auroc","average_precision"]]
        wins=int(delta.gt(0).all(axis=1).sum())
        candidates.append(dict(group=group,delta_auroc=float(delta.auroc.mean()),delta_ap=float(delta.average_precision.mean()),joint_wins=wins,
            eligible=bool(delta.auroc.mean()>=.02-1e-12 and delta.average_precision.mean()>=.01-1e-12 and wins>=6),
            mean_score=float(md.loc[group,["auroc","average_precision"]].mean().mean())))
    eligible=sorted([c for c in candidates if c["eligible"]],key=lambda c:(-c["mean_score"],c["group"]))
    winner=eligible[0]["group"] if eligible else None
    decision=dict(selection_role="development_validation",candidates=candidates,selected=winner)
    baseline=pd.read_csv(I.OUT/"patient_predictions_by_role.csv")
    for seed in R.SEEDS:
        for group in ("PAIR","RAD","DN-A"):
            frame=baseline[(baseline.seed==seed)&(baseline.group==group)].set_index("patient_alias")
            scores[seed][group]=np.array([frame.loc[p,"score"] if p in frame.index else np.nan for p in patients.patient_alias])
        if winner:
            scores[seed]["VOTE"]=(scores[seed]["RAD"]+scores[seed]["DN-A"]+scores[seed][winner])/3
    if winner:
        differences=[]
        for seed in R.SEEDS:
            ix=N.indices_for(patients,split,seed)["validation"]
            v=V.score_metrics(labels[ix],scores[seed]["VOTE"][ix])
            b=V.score_metrics(labels[ix],scores[seed]["PAIR"][ix])
            differences.append([v["auroc"]-b["auroc"],v["average_precision"]-b["average_precision"]])
        delta=np.array(differences)
        decision["vote"]=dict(delta_auroc=float(delta[:,0].mean()),delta_ap=float(delta[:,1].mean()),joint_wins=int((delta>0).all(1).sum()),
            promoted=bool(delta[:,0].mean()>=.03 and delta[:,1].mean()>=.02 and (delta>0).all(1).sum()>=6))
    R.write_json(OUT/"development_decision.json",decision)
    for seed in R.SEEDS:
        ix=N.indices_for(patients,split,seed)
        for group,score in scores[seed].items():
            for role,label in (("validation","development_validation"),("test","exploratory_test")):
                assert np.isfinite(score[ix[role]]).all()
                if role=="test" or group not in GROUPS:
                    records.append(dict(seed=seed,group=group,role=label,**V.score_metrics(labels[ix[role]],score[ix[role]])))
                for i in ix[role]:
                    predictions.append(dict(seed=seed,group=group,role=label,patient_alias=patients.patient_alias.iloc[i],label=int(labels[i]),score=float(score[i])))
    pd.DataFrame(records).to_csv(OUT/"metrics_by_role_per_draw.csv",index=False)
    N.summarize(pd.DataFrame(records)).to_csv(OUT/"summary_by_role.csv",index=False)
    pd.DataFrame(predictions).to_csv(OUT/"patient_predictions_by_role.csv",index=False)
    R.write_json(OUT/"fit_audit.json",audit)
    print(N.summarize(pd.DataFrame(records))[["role","group","auroc_mean","average_precision_mean"]].to_string(index=False),flush=True)
    print(json.dumps(decision),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","extract","evaluate","all"))
    args=parser.parse_args()
    for stage in (("prepare","extract","evaluate") if args.stage=="all" else (args.stage,)):
        globals()[stage]()
