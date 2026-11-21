"""Native ROI Saab features, reused medical encoders and fixed patient-level votes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import warnings

os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
import cv2
import joblib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

import renal_random_validation as R
import renal_enhancement_validation as E
import renal_voting_validation as V

ROOT=R.ROOT
OUT=Path(os.environ.get("CYCH_LOCAL_DIR",str(ROOT/"CYCH_local_features_20260913")))
KERNELS=(5,9)
MAX_PATCHES=2048
TRAIN_PATCHES_PER_PATIENT=512
GROUPS=("radiomics_percentile","radimagenet_percentile","green_original",
        "resnet_percentile","panecho_percentile","green_roi_reused","green_roi_fitted")
VOTES={
    "vote_strong_pair":("radiomics_percentile","radimagenet_percentile"),
    "vote_radiomics_dense_resnet":("radiomics_percentile","radimagenet_percentile","resnet_percentile"),
    "vote_radiomics_dense_panecho":("radiomics_percentile","radimagenet_percentile","panecho_percentile"),
    "vote_radiomics_dense_green_reused":("radiomics_percentile","radimagenet_percentile","green_roi_reused"),
    "vote_radiomics_dense_green_fitted":("radiomics_percentile","radimagenet_percentile","green_roi_fitted"),
}
PRIMARY="vote_radiomics_dense_resnet"
LABELS=dict(radiomics_percentile="ROI-normalized radiomics",
    radimagenet_percentile="RadImageNet DenseNet-121",green_original="Original Green features",
    resnet_percentile="RadImageNet ResNet-50",panecho_percentile="PanEcho frame encoder",
    green_roi_reused="Native ROI, reused Saab filters",green_roi_fitted="Native ROI, training-fitted Saab",
    vote_strong_pair="Radiomics + DenseNet",
    vote_radiomics_dense_resnet="Pair + ResNet (primary)",
    vote_radiomics_dense_panecho="Pair + PanEcho",
    vote_radiomics_dense_green_reused="Pair + reused native Saab",
    vote_radiomics_dense_green_fitted="Pair + fitted native Saab")
REFERENCES=[
    dict(citation="Mei X, et al. RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning. Radiology: Artificial Intelligence. 2022. doi:10.1148/ryai.210315.",
         url="https://github.com/BMEII-AI/RadImageNet",scope="Existing medical DenseNet-121 and ResNet-50 weights; recorded local loaders and preprocessing reused."),
    dict(citation="Holste G, et al. Complete AI-Enabled Echocardiography Interpretation with Multitask Deep Learning. JAMA. 2025.",
         url="https://github.com/CarDS-Yale/PanEcho",scope="Only the frozen 2D ConvNeXt-Tiny image encoder is transferred to static renal images. Not the full video model, its cardiac tasks, or a renal-trained checkpoint. ImageNet normalization follows the official repository."),
    dict(citation="Chen Y, Rouhsedaghat M, You S, Rao R, Kuo CCJ. PixelHop++: A Small Successive-Subspace-Learning-Based Model for Image Classification. ICIP. 2020.",
         url="https://arxiv.org/abs/2002.03141",scope="Source for Saab/PixelHop++ principles. Native 5/9-pixel single-stage descriptors here are a clearly specified adaptation using the existing fit_saab_unit, not the full PixelHop++ cascade or a newly named model."),
    dict(citation="Guo C, Pleiss G, Sun Y, Weinberger KQ. On Calibration of Modern Neural Networks. ICML. 2017;70:1321-1330.",
         url="https://proceedings.mlr.press/v70/guo17a.html",scope="Positive temperature scaling. This experiment adds fixed bounds [0.5,5] and a 0.01(log T)^2 penalty for a small calibration sample; not an exact replication or a guarantee of calibrated risk."),
    dict(citation="scikit-learn 1.8. LogisticRegression and VotingClassifier.",
         url="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.VotingClassifier.html",scope="Fixed L2 logistic heads and arithmetic probability means. No learned voting weights or new meta-model."),
    dict(citation="He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. CVPR. 2016.",
         url="https://arxiv.org/abs/1512.03385",scope="ResNet architecture, distinct from the medical pretraining provenance."),
    dict(citation="Liu Z, et al. A ConvNet for the 2020s. CVPR. 2022.",
         url="https://arxiv.org/abs/2201.03545",scope="ConvNeXt architecture underlying the reused PanEcho frame encoder."),
    dict(citation="scikit-learn 1.8. LogisticRegression.",
         url="https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html",scope="Exact regularized classifier API and parameter definitions."),
]


def read(name):
    return pd.read_csv(OUT/name,dtype={"patient_id":str})


def require_lock():
    if json.loads((OUT/"protocol_locked.json").read_text())["sha256"]!=R.sha256(OUT/"protocol.json"):
        raise RuntimeError("Protocol hash mismatch")


def patch_coordinates(mask,k,image_id):
    """Top-left coordinates of windows wholly inside the mask; no interpolation."""
    mask=np.asarray(mask,dtype=np.uint8)
    if mask.ndim!=2 or k%2!=1 or min(mask.shape)<k:
        raise ValueError("Invalid mask/window geometry")
    radius=k//2
    valid=cv2.erode(mask,np.ones((k,k),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    yy,xx=np.nonzero(valid[radius:-radius,radius:-radius])
    total=len(yy)
    if not total:
        raise ValueError(f"No wholly intrarenal {k}x{k} window: {image_id}")
    if total>MAX_PATCHES:
        seed=int(hashlib.sha256(f"{image_id}:k{k}:20260913".encode()).hexdigest()[:8],16)
        selected=np.sort(np.random.default_rng(seed).choice(total,MAX_PATCHES,replace=False))
        yy,xx=yy[selected],xx[selected]
    return np.column_stack([yy,xx]).astype(np.int32),total


def get_patches(gray,mask,k,image_id):
    coords,total=patch_coordinates(mask,k,image_id)
    windows=sliding_window_view(gray,(k,k))
    patches=windows[coords[:,0],coords[:,1]].reshape(len(coords),k*k).astype(np.float32)/255.
    return patches,coords,total


def prepare():
    OUT.mkdir(exist_ok=True)
    for sub in ("features","models","images/percentile","masks","patch_coordinates","source","qa","figures_600dpi"):
        (OUT/sub).mkdir(parents=True,exist_ok=True)
    for name in ("image_manifest.csv","patients.csv","random_patient_assignments.csv"):
        source=R.OUT/name
        target=OUT/name
        if target.exists() and R.sha256(target)!=R.sha256(source):
            raise RuntimeError(f"Cohort drift: {name}")
        shutil.copy2(source,target)
    df=read("image_manifest.csv")
    pd.testing.assert_frame_equal(df,pd.read_csv(E.OUT/"image_manifest.csv",dtype={"patient_id":str}))
    from radimagenet_backbones import RADIMAGENET_DIR,PANECHO_PT
    checkpoints=[RADIMAGENET_DIR/"DenseNet121.pt",RADIMAGENET_DIR/"ResNet50.pt",PANECHO_PT]
    protocol=dict(
        design="Further retrospective exploration of the same previously explored 98 patients/349 images. Ten patient-random draws. No external or untouched-cohort claim.",
        seeds=R.SEEDS,roles=dict(train=58,calibration=20,test=20),
        assignments="Original train/validation/test assignments unchanged. Original validation now used only for fixed-procedure probability calibration, not feature/head/ensemble selection.",
        groups=list(GROUPS),votes=VOTES,primary=PRIMARY,
        preprocessing="Original segmentation masks unchanged; existing ROI P1-P99 grayscale normalization reused. No caliper removal, inpainting, pseudo-lesion labels, new augmentation or clinical covariates.",
        head="Same fixed LR setting as prior voting: train-only median imputer, standardization, ANOVA top32, L2 LR C=.1/liblinear/balanced/max_iter2000. Fit on 58 patients only.",
        encoders="Frozen existing RadImageNet DenseNet-121/ResNet-50 and PanEcho 2D ConvNeXt-Tiny. No encoder or segmentation retraining. PanEcho is an echocardiographic domain-transfer experiment, not a renal-specific or full-video model.",
        encoder_input="Same existing 224x224 masked grayscale canvas. RadImageNet x/127.5-1; PanEcho ImageNet mean/std. No Test-time augmentation.",
        local_support=dict(kernels=list(KERNELS),max_patches_per_image=MAX_PATCHES,training_patches_per_patient_per_kernel=TRAIN_PATCHES_PER_PATIENT,
            sampling="Wholly in-mask native-pixel windows only; all valid windows if <=2048, otherwise deterministic uniform sample without replacement, SHA256(image_id,k,20260913) seed. No resizing or interpolation of local patches."),
        reused_saab="Existing first-hop 5x5 unit/bias from each draw's training-fitted GreenExtractor. All25 channels, local mean/std/P10/P50/P90; not the old full three-hop cascade.",
        fitted_saab="Existing fit_saab_unit independently fits 5x5 and9x9 DC-anchored PCA bases to 512 patches per training patient per kernel, balanced over their images. All25+81 channels retained. AC bias fixed at k, a norm bound for [0,1] inputs; no cascade/ReLU or cross-entropy feature tree. Local mean/std/P10/P50/P90, then patient image mean/std.",
        feature_dimensions=dict(green_roi_reused_image=125,green_roi_fitted_image=530,green_roi_reused_patient=250,green_roi_fitted_patient=1060),
        calibration="After each fixed single/vote, one positive temperature fitted using all20 calibration patients. Optimize mean binary log loss+0.01(log T)^2, T in [0.5,5]; no intercept, no sign flip, no member reweighting. Primary also has a separately reported temperature fit on the prior ten calibration patients, with identical input predictions.",
        ranking="Positive temperature is monotone, preserving AUROC/AP and the0.5 decision threshold. Calibration quality checked with held-out Brier/log loss, not claimed as improved discrimination.",
        comparison="Seven fixed single LR heads and five fixed equal-probability votes; all native and temperature20 results, plus primary temperature10. No validation-selected or test-selected winning workflow.",
        limitations="No physical pixel spacing; native pixel scales can remain acquisition-dependent. Calipers inside masks remain. Old segmentation training overlap remains unresolved. All follow-up results are exploratory.",
        cohort_hashes={name:R.sha256(OUT/name) for name in ("image_manifest.csv","patients.csv","random_patient_assignments.csv")},
        checkpoints=[dict(path=str(p),sha256=R.sha256(p),bytes=p.stat().st_size) for p in checkpoints])
    path=OUT/"protocol.json"
    # JSON round-trip normalizes vote tuples to lists before comparing an existing lock.
    protocol=json.loads(json.dumps(protocol))
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise RuntimeError("Protocol changed: use a different output directory")
    R.write_json(path,protocol)
    if not (OUT/"protocol_locked.json").exists():
        R.write_json(OUT/"protocol_locked.json",dict(utc=datetime.now(timezone.utc).isoformat(),sha256=R.sha256(path)))
    R.write_json(OUT/"references.json",REFERENCES)
    shutil.copy2(V.OUT/"inherited_method_references.json",OUT/"inherited_method_references.json")
    shutil.copy2(V.OUT/"voting_patient_assignments.csv",OUT/"prior_voting_patient_assignments.csv")
    shutil.copy2(E.OUT/"qa/visual_review_selection.csv",OUT/"qa/visual_review_selection.csv")
    audit=[]
    for row in df.itertuples():
        image_path=OUT/"images/percentile"/f"{row.image_id}.png"
        mask_path=OUT/"masks"/f"{row.image_id}.png"
        shutil.copy2(E.OUT/"images/percentile"/image_path.name,image_path)
        shutil.copy2(E.OUT/"masks"/mask_path.name,mask_path)
        assert R.sha256(mask_path)==row.mask_sha256
        gray=cv2.imread(str(image_path),0)
        mask=cv2.imread(str(mask_path),0)>0
        assert np.all(gray[~mask]==0)
        for k in KERNELS:
            coords,n=patch_coordinates(mask,k,row.image_id)
            path=OUT/"patch_coordinates"/f"{row.image_id}_k{k}.npy"
            np.save(path,coords)
            audit.append(dict(image_id=row.image_id,kernel=k,height=gray.shape[0],width=gray.shape[1],
                valid_windows=n,sampled_windows=len(coords),image_sha256=R.sha256(image_path),mask_sha256=R.sha256(mask_path),
                coordinate_sha256=R.sha256(path)))
    pd.DataFrame(audit).to_csv(OUT/"native_patch_audit.csv",index=False)
    sources=[]
    for group in ("radiomics_percentile","radimagenet_percentile"):
        src=V.OUT/"features"/f"{group}.npy"
        dst=OUT/"features"/src.name
        shutil.copy2(src,dst)
        sources.append(dict(source=str(src),sha256=R.sha256(src),destination=str(dst.relative_to(OUT))))
    for seed in R.SEEDS:
        src=V.OUT/"features"/f"green_original_{seed}.npy"
        shutil.copy2(src,OUT/"features"/src.name)
        sources.append(dict(source=str(src),sha256=R.sha256(src),destination="features/"+src.name))
    R.write_json(OUT/"features/inherited_sources.json",sources)
    R.write_json(OUT/"original_manuscript_preservation.json",dict(
        path=str(ROOT/"yolo_aug_triEMR_PCA_SVM_renal_2022_v2.docx"),
        sha256=R.sha256(ROOT/"yolo_aug_triEMR_PCA_SVM_renal_2022_v2.docx")))
    print("Protocol locked; 349 native ROI images, patch coordinates and inherited features ready",flush=True)


def model_digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def encoders():
    require_lock()
    import torch
    from radimagenet_backbones import radimagenet_resnet50,panecho_convnext_tiny,RADIMAGENET_DIR,PANECHO_PT
    torch.set_num_threads(4)
    df=read("image_manifest.csv")
    patients=read("patients.csv")
    images=[]
    for row in df.itertuples():
        gray=cv2.imread(str(OUT/"images/percentile"/f"{row.image_id}.png"),0)
        mask=(cv2.imread(str(OUT/"masks"/f"{row.image_id}.png"),0)>0).astype(np.uint8)
        images.append(R.canvas_image(gray,mask,0,"gray",224))
    values=[]
    for name,loader,checkpoint,dim,keys in (("resnet_percentile",radimagenet_resnet50,RADIMAGENET_DIR/"ResNet50.pt",2048,318),
            ("panecho_percentile",panecho_convnext_tiny,PANECHO_PT,768,180)):
        model=loader().cuda().eval()
        for p in model.parameters():
            p.requires_grad_(False)
        before=model_digest(model)
        features=[]
        with torch.inference_mode():
            for start in range(0,len(images),32):
                x=torch.from_numpy(np.stack(images[start:start+32]).transpose(0,3,1,2).copy()).cuda().float()/255.
                if name.startswith("resnet"):
                    x=(x-.5)/.5
                else:
                    mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
                    std=x.new_tensor([.229,.224,.225])[None,:,None,None]
                    x=(x-mean)/std
                features.append(model(x).cpu().numpy())
        f=np.concatenate(features)
        assert f.shape==(349,dim) and np.isfinite(f).all()
        after=model_digest(model)
        assert before==after
        np.save(OUT/"features"/f"{name}_image.npy",f)
        np.save(OUT/"features"/f"{name}.npy",R.aggregate(f,df,patients))
        values.append(dict(group=name,checkpoint=str(checkpoint),sha256=R.sha256(checkpoint),dimension=dim,
            matched_backbone_keys=keys,trainable_parameters=0,state_before=before,state_after=after,
            normalization="x/127.5-1" if name.startswith("resnet") else "ImageNet mean/std",
            image_features_sha256=R.sha256(OUT/"features"/f"{name}_image.npy")))
        print(f"Reused frozen {name}: {f.shape}, state unchanged",flush=True)
        del model
        torch.cuda.empty_cache()
    R.write_json(OUT/"features/encoder_provenance.json",values)


def load_patch_cache():
    df=read("image_manifest.csv")
    cache={k:{} for k in KERNELS}
    for row in df.itertuples():
        gray=cv2.imread(str(OUT/"images/percentile"/f"{row.image_id}.png"),0)
        for k in KERNELS:
            coords=np.load(OUT/"patch_coordinates"/f"{row.image_id}_k{k}.npy")
            windows=sliding_window_view(gray,(k,k))
            cache[k][row.image_id]=windows[coords[:,0],coords[:,1]].reshape(len(coords),k*k).astype(np.float32)/255.
    return cache


def training_patches(cache,df,train_patients,seed,k):
    rng=np.random.default_rng(seed+k*10000)
    chunks,audit=[],[]
    for patient in sorted(train_patients):
        ids=sorted(df.loc[df.patient_id.eq(patient),"image_id"])
        if not ids:
            raise ValueError("Training patient has no images")
        base,remainder=divmod(TRAIN_PATCHES_PER_PATIENT,len(ids))
        for j,image_id in enumerate(ids):
            n=base+(j<remainder)
            patches=cache[k][image_id]
            ix=rng.choice(len(patches),size=n,replace=n>len(patches))
            chunks.append(patches[ix])
            audit.append(dict(patient_id=patient,image_id=image_id,kernel=k,n_patches=n,
                sampled_patch_indices=ix.tolist()))
    return np.concatenate(chunks),audit


def local_statistics(patches,unit,bias):
    # Keep the existing uncentered Saab response convention; DC is unbiased.
    responses=patches.astype(np.float64)@unit.W
    responses[:,1:]+=bias
    summaries=np.stack([responses.mean(0),responses.std(0),*np.quantile(responses,[.1,.5,.9],axis=0)],axis=1)
    return summaries.reshape(-1).astype(np.float32)


def green():
    require_lock()
    from green_learning_cv import fit_saab_unit
    df=read("image_manifest.csv")
    splits=read("random_patient_assignments.csv")
    patients=read("patients.csv")
    cache=load_patch_cache()
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            train=sorted(splits.loc[splits.seed.eq(seed)&splits.role.eq("train"),"patient_id"])
            previous=R.OUT/"models"/f"green_extractor_{seed}.joblib"
            extractor=joblib.load(previous)
            expected=set(df.loc[df.patient_id.isin(train),"image_id"])
            assert extractor.fit_paths==expected
            before=joblib.hash(extractor)
            fitted,fit_audit={},[]
            for k in KERNELS:
                samples,items=training_patches(cache,df,train,seed,k)
                fitted[k]=fit_saab_unit(samples)
                fit_audit.extend(items)
            local,reused=[],[]
            for image_id in df.image_id:
                local.append(np.concatenate([local_statistics(cache[k][image_id],fitted[k],k) for k in KERNELS]))
                reused.append(local_statistics(cache[5][image_id],extractor.hops[0].units[0],extractor.hops[0].d))
            assert before==joblib.hash(extractor)
            for group,features in (("green_roi_reused",np.stack(reused)),("green_roi_fitted",np.stack(local))):
                np.save(OUT/"features"/f"{group}_{seed}_image.npy",features)
                np.save(OUT/"features"/f"{group}_{seed}.npy",R.aggregate(features,df,patients))
            model_path=OUT/"models"/f"native_saab_{seed}.joblib"
            joblib.dump(dict(fitted=fitted,reused_unit=extractor.hops[0].units[0],reused_bias=extractor.hops[0].d,
                train_patients=train,fit_images=sorted(expected)),model_path,compress=3)
            R.write_json(OUT/"features"/f"native_saab_{seed}_provenance.json",dict(seed=seed,train_patients=train,
                fit_images=sorted(expected),training_patches=fit_audit,old_extractor=str(previous),old_extractor_sha256=R.sha256(previous),
                old_state_before=before,old_state_after=joblib.hash(extractor),new_model_sha256=R.sha256(model_path),
                feature_dimensions=dict(reused=250,fitted=1060)))
            print(f"{seed}: native Saab features; 58 training patients x512 patches/scale, old bank reused unchanged",flush=True)


def temperature(probability,t):
    p=np.asarray(probability,dtype=np.float64)
    if not np.isfinite(t) or t<=0 or not np.isfinite(p).all() or np.any((p<=0)|(p>=1)):
        raise ValueError("Temperature must be positive and probabilities strictly inside (0,1)")
    return expit(logit(p)/t)


def fit_temperature(probability,y):
    p=np.asarray(probability,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    if len(p)!=len(y) or set(y)!={0.,1.}:
        raise ValueError("Calibration requires matching binary labels from both classes")
    z=logit(p)
    temperature(p,1.)
    def objective(log_t):
        score=z/np.exp(log_t)
        return float(np.mean(np.logaddexp(0,score)-y*score)+.01*log_t*log_t)
    result=minimize_scalar(objective,bounds=(np.log(.5),np.log(5.)),method="bounded",options={"xatol":1e-10})
    if not result.success:
        raise RuntimeError("Temperature optimization failed")
    candidates=[float(result.x),float(np.log(.5)),0.,float(np.log(5.))]
    best=min(candidates,key=objective)
    return dict(temperature=float(np.exp(best)),objective=objective(best),identity_objective=objective(0.),
                boundary=bool(abs(best-np.log(.5))<1e-7 or abs(best-np.log(5.))<1e-7),n_calibration=len(y))


def load_features(seed):
    arrays={}
    for group in GROUPS:
        suffix=f"_{seed}" if group.startswith("green") else ""
        arrays[group]=np.load(OUT/"features"/f"{group}{suffix}.npy")
        assert arrays[group].shape[0]==98 and np.isfinite(arrays[group]).all()
    return arrays


def fixed_votes(probabilities):
    result={k:np.asarray(v,dtype=float) for k,v in probabilities.items()}
    for name,members in VOTES.items():
        result[name]=np.column_stack([probabilities[k] for k in members]).mean(1)
    return result


def evaluate():
    require_lock()
    patients=read("patients.csv")
    splits=read("random_patient_assignments.csv")
    previous_path=OUT/"prior_voting_patient_assignments.csv"
    if not previous_path.exists():
        previous_path=V.OUT/"voting_patient_assignments.csv"
    previous_splits=pd.read_csv(previous_path,dtype={"patient_id":str})
    y=patients.label.to_numpy()
    results,predictions,calibrators,heads,member_rows,diversity=[],[],[],[],[],[]
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            per=splits[splits.seed.eq(seed)]
            ix={r:np.flatnonzero(patients.patient_id.isin(per.loc[per.role.eq(r),"patient_id"])) for r in ("train","validation","test")}
            cal10=np.flatnonzero(patients.patient_id.isin(previous_splits.loc[previous_splits.seed.eq(seed)&previous_splits.role.eq("calibration"),"patient_id"]))
            assert set(cal10)<=set(ix["validation"])
            arrays=load_features(seed)
            probabilities={}
            for group,x in arrays.items():
                model=V.make_head("lr",seed,x.shape[1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore",UserWarning)
                    model.fit(x[ix["train"]],y[ix["train"]])
                probabilities[group]=model.predict_proba(x)[:,1]
                path=OUT/"models"/f"{group}_lr_{seed}.joblib"
                joblib.dump(model,path,compress=3)
                heads.append(dict(seed=seed,group=group,file=str(path.relative_to(OUT)),sha256=R.sha256(path),
                    train_patients=patients.patient_id.iloc[ix["train"]].tolist(),selected_indices=model[2].get_support(indices=True).tolist(),
                    scaler_n=int(model[1].n_samples_seen_)))
                for role in ("validation","test"):
                    for i in ix[role]:
                        member_rows.append(dict(seed=seed,role=role,patient_id=patients.patient_id.iloc[i],group=group,label=int(y[i]),probability=float(probabilities[group][i])))
            all_scores=fixed_votes(probabilities)
            for name,p in all_scores.items():
                cal=fit_temperature(p[ix["validation"]],y[ix["validation"]])
                calibrators.append(dict(seed=seed,workflow=name,role="temperature20",patients=patients.patient_id.iloc[ix["validation"]].tolist(),**cal))
                arms={"native":p[ix["test"]],"temperature20":temperature(p[ix["test"]],cal["temperature"])}
                if name==PRIMARY:
                    c=fit_temperature(p[cal10],y[cal10])
                    calibrators.append(dict(seed=seed,workflow=name,role="temperature10",patients=patients.patient_id.iloc[cal10].tolist(),**c))
                    arms["temperature10"]=temperature(p[ix["test"]],c["temperature"])
                for arm,scores in arms.items():
                    result=V.score_metrics(y[ix["test"]],scores)
                    model=name+"__"+arm
                    results.append(dict(seed=seed,model=model,workflow=name,probability_method=arm,**result))
                    for i,score in zip(ix["test"],scores):
                        predictions.append(dict(seed=seed,model=model,patient_id=patients.patient_id.iloc[i],label=int(y[i]),score=float(score),prediction=int(score>.5)))
            val=ix["validation"]
            for i,a in enumerate(GROUPS):
                for b in GROUPS[i+1:]:
                    pa,pb=probabilities[a][val],probabilities[b][val]
                    ea,eb=(pa>.5)!=y[val],(pb>.5)!=y[val]
                    corr=float(np.corrcoef(pa,pb)[0,1]) if pa.std()>1e-12 and pb.std()>1e-12 else None
                    diversity.append(dict(seed=seed,member_a=a,member_b=b,n=20,role="validation",probability_correlation=corr,
                        disagreement=float(np.mean((pa>.5)!=(pb>.5))),double_fault=float(np.mean(ea&eb))))
            pd.DataFrame(results).to_csv(OUT/"metrics_per_draw.csv",index=False)
            pd.DataFrame(predictions).to_csv(OUT/"patient_test_predictions.csv",index=False)
            pd.DataFrame(member_rows).to_csv(OUT/"member_predictions_by_role.csv",index=False)
            pd.DataFrame(diversity).to_csv(OUT/"development_diversity.csv",index=False)
            R.write_json(OUT/"calibration_parameters.json",calibrators)
            R.write_json(OUT/"head_provenance.json",heads)
            item=next(v for v in reversed(results) if v["model"]==PRIMARY+"__native")
            print(f"{seed}: primary fixed three-member vote ROC={item['auroc']:.4f} AP={item['average_precision']:.4f}",flush=True)
    summarize()


def summarize():
    m=pd.read_csv(OUT/"metrics_per_draw.csv")
    rows=[]
    for name,g in m.groupby("model",sort=False):
        r=dict(model=name,workflow=g.workflow.iloc[0],probability_method=g.probability_method.iloc[0],draws=len(g),
            joint_target_draws=int(((g.auroc>.8)&(g.average_precision>.8)).sum()))
        for field in ("auroc","average_precision","auprc_trapezoid","brier","log_loss","sensitivity","specificity"):
            for stat in ("mean","std","min","max"):
                r[f"{field}_{stat}"]=float(getattr(g[field],stat)())
        rows.append(r)
    s=pd.DataFrame(rows)
    s.to_csv(OUT/"summary.csv",index=False)
    comparisons=[]
    pairs=[("green_roi_reused__native","green_original__native"),("green_roi_fitted__native","green_original__native"),
           ("green_roi_fitted__native","green_roi_reused__native"),
           ("vote_strong_pair__native","radiomics_percentile__native"),
           ("vote_strong_pair__native","radimagenet_percentile__native"),
           *[(name+"__native","vote_strong_pair__native") for name in VOTES if name!="vote_strong_pair"],
           *[(name+"__temperature20",name+"__native") for name in (*GROUPS,*VOTES)],
           (PRIMARY+"__temperature20",PRIMARY+"__temperature10")]
    for candidate,baseline in pairs:
        a=m[m.model.eq(candidate)].set_index("seed")
        b=m[m.model.eq(baseline)].set_index("seed").loc[a.index]
        r=dict(candidate=candidate,baseline=baseline,draws=len(a),joint_ranking_improved_draws=int(((a.auroc>b.auroc)&(a.average_precision>b.average_precision)).sum()))
        for field in ("auroc","average_precision","brier","log_loss"):
            r[f"delta_{field}_mean"]=float((a[field]-b[field]).mean())
        comparisons.append(r)
    pd.DataFrame(comparisons).to_csv(OUT/"paired_comparisons.csv",index=False)
    print(s.loc[s.probability_method.eq("native"),["workflow","auroc_mean","average_precision_mean","joint_target_draws"]].to_string(index=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["prepare","encoders","green","evaluate","all"])
    args=parser.parse_args()
    for stage in (["prepare","encoders","green","evaluate"] if args.stage=="all" else [args.stage]):
        globals()[stage]()
