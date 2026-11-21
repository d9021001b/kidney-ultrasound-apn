"""Literature-motivated, explicitly defined B-mode feature adaptations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil

import cv2
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import renal_dinov3_native_probe as N
import renal_local_features as L
import renal_random_validation as R
import renal_voting_validation as V

OUT = R.ROOT / "CYCH_interpretable_features_20260913"
GROUPS = ("I1-global", "I2-spatial", "I3-direction-original", "I4-direction-aligned")
RAD_FEATURES = {
    "firstorder": ["Entropy", "Variance"],
    "glrlm": ["RunLengthNonUniformityNormalized", "GrayLevelNonUniformityNormalized", "RunEntropy"],
}
REFERENCES = [
    dict(id="bandara2022", citation="Bandara MS, et al. Ultrasound Based Radiomics Features of Chronic Kidney Disease. Academic Radiology. 2022;29:229-235. doi:10.1016/j.acra.2021.01.006.",
         url="https://pubmed.ncbi.nlm.nih.gov/33589307/", scope="B-mode CKD vs healthy; directional wavelet GLRLM clue, not a CYCH inflammation replication."),
    dict(id="qin2023", citation="Qin X, et al. Noninvasive Evaluation of Lupus Nephritis Activity Using a Radiomics Machine Learning Model Based on Ultrasound. J Inflamm Res. 2023;16:433-441. doi:10.2147/JIR.S398399.",
         url="https://pmc.ncbi.nlm.nih.gov/articles/PMC9904229/", scope="Biopsy-defined lupus activity; texture heterogeneity hypothesis, not proof of this study label or localized lesions."),
    dict(id="moghazi2005", citation="Moghazi S, et al. Correlation of renal histopathology with sonographic findings. Kidney Int. 2005;67:1515-1520. doi:10.1111/j.1523-1755.2005.00230.x.",
         url="https://pubmed.ncbi.nlm.nih.gov/15780105/", scope="Echogenicity associated with several histological processes; no inflammation-specific grayscale threshold. True cortex/liver ratios unavailable here."),
    dict(id="fiegler1983", citation="Fiegler W. Ultrasound in acute renal inflammatory lesions. Eur J Radiol. 1983;3:354-357.",
         url="https://pubmed.ncbi.nlm.nih.gov/6653572/", scope="Clinical clue for local/global echotexture; mild inflammatory disease may appear normal."),
    dict(id="pyradiomics", citation="van Griethuysen JJM, et al. Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Res. 2017;77:e104-e107. doi:10.1158/0008-5472.CAN-17-0339.",
         url="https://pyradiomics.readthedocs.io/en/v3.0.1/features.html", scope="Standard first-order and GLRLM definitions; coif1 inherited from the existing project, not claimed as Bandara's exact implementation."),
]


def read(name):
    return pd.read_csv(OUT / name, dtype={"patient_id": str})


def require_lock():
    lock = json.loads((OUT / "protocol_locked.json").read_text())
    assert lock["sha256"] == R.sha256(OUT / "protocol.json")
    assert lock["source_sha256"] == R.sha256(Path(__file__))


def prepare():
    OUT.mkdir(exist_ok=True)
    for sub in ("features", "models", "source", "qa", "aligned_images", "aligned_masks", "figures_600dpi"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv"):
        source, target = L.OUT / name, OUT / name
        if target.exists():
            assert R.sha256(source) == R.sha256(target)
        else:
            shutil.copy2(source, target)
    protocol = dict(cohort="Same98 previously explored patients,349 first-exam images; unchanged masks and P1-P99 normalization.",
        seeds=R.SEEDS, roles=dict(train=58, development=20, exploratory_test=20), groups=GROUPS,
        global_features="Normalized ROI intensity mean,std,P10,median,P90,IQR,16-bin entropy,fraction<.25,fraction>.75 (9 image features).",
        spatial_features="Global9 + 9x9 and17x17 native wholly in-mask windows, at most2048/image/size, deterministic L.patch_coordinates. Window mean/std/16-bin entropy summarized by mean,std,P10,P50,P90. Three equal-width normalized-distance-to-boundary bands: intensity mean/std/entropy plus outer-minus-inner mean. Distance bands are geometry, NOT cortex/medulla/sinus labels. Missing bands -> NaN, train-only imputer.",
        directional_features="Original plus4 coif1 level1 stationary-wavelet images;5 fixed PyRadiomics features/image type =25. Each unaligned/aligned condition uses native pixels and13x13 square mask erosion to avoid filter support crossing original kidney boundary. If fewer than64 pixels survive, fixed3x3 erosion fallback, QA flag. Whole image multiplied by original mask only, not eroded mask, before transform.",
        alignment="Mask covariance principal axis rotated vertical, angle normalized[-90,90); centered square ceil(hypot(h,w))+8, image bilinear, mask nearest. Original control uses identical canvas with0deg. No resizing, label dependence or physical length claim. Rotation interpolation is an acknowledged confound.",
        radiomics=dict(binWidth=16, normalize=False, features=RAD_FEATURES, wavelet="coif1", level=1, force2D=False),
        aggregation="Each image feature -> patient mean and population std across original images. Equal patient weight. No count or absolute pixel area predictor.",
        head="Fixed train-only imputer,constant removal,standardizer,all-feature L2 LR C=.1/liblinear/balanced/max_iter2000. No feature or parameter search.",
        gate="Single features must have development mean AUROC/AP >= original DenseNet minus.02. One eligible winner by mean(AUROC+AP)/2 then group ID; equal probability mean with original RAD+DN. Promote only if AUROC delta>=.03,AP>=.02,joint improvements>=6/10. No test-driven choice.",
        limitations="Literature-motivated adaptation, not paper replication. B-mode displayed intensity is not RF backscatter. No physical spacing, cortex annotation, liver reference, Doppler, contrast time series, shear-wave elastography or biopsy-site registration. Existing calipers remain. No diagnosis by naked-eye bright/dark region.",
        references=REFERENCES, inputs={n:R.sha256(OUT/n) for n in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv")})
    protocol = json.loads(json.dumps(protocol))
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        R.write_json(path, protocol)
    if not (OUT / "protocol_locked.json").exists():
        R.write_json(OUT / "protocol_locked.json", dict(utc=datetime.now(timezone.utc).isoformat(),sha256=R.sha256(path),source_sha256=R.sha256(Path(__file__))))
    shutil.copy2(__file__, OUT / "source" / Path(__file__).name)
    R.write_json(OUT / "references.json", REFERENCES)
    print("Locked four interpretable-feature configurations before extraction/evaluation", flush=True)


def entropy(values):
    bins = np.minimum((np.asarray(values)*16).astype(int), 15)
    counts = np.bincount(bins.ravel(), minlength=16).astype(float)
    p = counts[counts > 0]/max(counts.sum(), 1)
    return float(-np.sum(p*np.log2(p)))


def histogram_features(values):
    q10, q25, q50, q75, q90 = np.percentile(values, [10,25,50,75,90])
    return dict(mean=float(values.mean()), std=float(values.std()), p10=q10, median=q50, p90=q90,
                iqr=q75-q25, entropy16=entropy(values), dark_fraction=float((values < .25).mean()),
                bright_fraction=float((values > .75).mean()))


def heterogeneity(gray, mask, image_id):
    global_f = histogram_features(gray[mask > 0].astype(float)/255)
    spatial = {f"global_{k}":v for k,v in global_f.items()}
    for size in (9,17):
        patches, coords, total = L.get_patches(gray, mask, size, image_id)
        np.save(OUT / "qa" / f"{image_id}_k{size}_coordinates.npy", coords)
        values = dict(mean=patches.mean(1), std=patches.std(1), entropy16=np.array([entropy(p) for p in patches]))
        for name, array in values.items():
            for stat, val in zip(("mean", "std", "p10", "p50", "p90"),
                                 [array.mean(), array.std(), *np.percentile(array,[10,50,90])]):
                spatial[f"window{size}_{name}_{stat}"] = float(val)
    # Padding makes distance to the boundary well-defined even at a native crop edge.
    dist = cv2.distanceTransform(np.pad(mask,1), cv2.DIST_L2,5)[1:-1,1:-1]
    dist /= dist.max()
    for band in range(3):
        selected = (mask > 0)&(dist > band/3)&(dist <= (band+1)/3+1e-7)
        values = gray[selected].astype(float)/255
        for name, val in dict(mean=values.mean() if len(values) else np.nan,
                             std=values.std() if len(values) else np.nan,
                             entropy16=entropy(values) if len(values) else np.nan).items():
            spatial[f"distance_band{band}_{name}"] = val
    spatial["distance_outer_minus_inner_mean"] = spatial["distance_band0_mean"]-spatial["distance_band2_mean"]
    return global_f, spatial


def axis_angle(mask):
    yy, xx = np.nonzero(mask)
    eigenvalues, vectors = np.linalg.eigh(np.cov(np.stack([xx,yy])))
    vx, vy = vectors[:, np.argmax(eigenvalues)]
    return float(np.degrees(np.arctan2(vy,vx)))


def rotate_native(gray, mask, aligned):
    side = int(np.ceil(np.hypot(*gray.shape)))+8
    pad = np.zeros((side,side), np.float32)
    pm = np.zeros((side,side), np.uint8)
    y, x = (side-gray.shape[0])//2, (side-gray.shape[1])//2
    pad[y:y+gray.shape[0],x:x+gray.shape[1]] = gray
    pm[y:y+gray.shape[0],x:x+gray.shape[1]] = mask
    angle = (axis_angle(mask)-90+90)%180-90 if aligned else 0.0
    matrix = cv2.getRotationMatrix2D((side/2,side/2),angle,1)
    image = cv2.warpAffine(pad,matrix,(side,side),flags=cv2.INTER_LINEAR)
    om = cv2.warpAffine(pm,matrix,(side,side),flags=cv2.INTER_NEAREST)
    image *= om
    selected = cv2.erode(om,np.ones((13,13),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    fallback = bool(selected.sum() < 64)
    if fallback:
        selected = cv2.erode(om,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    assert selected.sum() >= 64
    return image, om, selected, dict(rotation_degrees=angle, erosion_fallback=fallback,
        retained_fraction=float(selected.sum()/om.sum()), matrix=matrix.tolist(), padded_side=side)


def extract():
    require_lock()
    import SimpleITK as sitk
    import radiomics
    from radiomics import featureextractor
    radiomics.setVerbosity(logging.ERROR)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    ext = featureextractor.RadiomicsFeatureExtractor(binWidth=16, normalize=False, distances=[1,2],
        force2D=False, minimumROIDimensions=2)
    ext.disableAllImageTypes()
    ext.enableImageTypeByName("Original")
    ext.enableImageTypeByName("Wavelet", customArgs={"wavelet":"coif1", "level":1})
    ext.disableAllFeatures()
    ext.enableFeaturesByName(**RAD_FEATURES)
    df, patients = read("image_manifest.csv"), read("patients.csv")
    outputs, audit = {g:[] for g in GROUPS}, []
    for i,row in enumerate(df.itertuples()):
        ip, mp = L.OUT/"images/percentile"/f"{row.image_id}.png", L.OUT/"masks"/f"{row.image_id}.png"
        assert R.sha256(mp) == row.mask_sha256
        gray, mask = cv2.imread(str(ip),0), (cv2.imread(str(mp),0)>0).astype(np.uint8)
        global_f, spatial_f = heterogeneity(gray,mask,row.image_id)
        outputs[GROUPS[0]].append(dict(image_id=row.image_id,**global_f))
        outputs[GROUPS[1]].append(dict(image_id=row.image_id,**spatial_f))
        for aligned in (False,True):
            im, om, selected, qa = rotate_native(gray,mask,aligned)
            result = ext.execute(sitk.GetImageFromArray(im),sitk.GetImageFromArray(selected))
            values = {k:float(v) for k,v in result.items() if not k.startswith("diagnostics")}
            assert len(values) == 25 and np.isfinite(list(values.values())).all()
            group = GROUPS[3] if aligned else GROUPS[2]
            outputs[group].append(dict(image_id=row.image_id,**values))
            audit.append(dict(image_id=row.image_id,group=group,image_sha256=R.sha256(ip),mask_sha256=R.sha256(mp),**qa))
            if aligned:
                cv2.imwrite(str(OUT/"aligned_images"/f"{row.image_id}.png"),np.rint(im).astype(np.uint8))
                cv2.imwrite(str(OUT/"aligned_masks"/f"{row.image_id}.png"),selected*255)
        if i%40 == 0:
            print(f"Interpretable features {i+1}/{len(df)}",flush=True)
    dictionary=[]
    for group,rows in outputs.items():
        frame=pd.DataFrame(rows)
        frame.to_csv(OUT/"features"/f"{group}_image.csv",index=False)
        features=frame.drop(columns="image_id").to_numpy(np.float32)
        np.save(OUT/"features"/f"{group}_image.npy",features)
        np.save(OUT/"features"/f"{group}.npy",R.aggregate(features,df,patients))
        for agg in ("image_mean","image_std"):
            for name in frame.columns[1:]:
                dictionary.append(dict(group=group,patient_feature=f"{agg}:{name}",image_feature=name,patient_aggregation=agg))
    pd.DataFrame(dictionary).to_csv(OUT/"feature_dictionary.csv",index=False)
    R.write_json(OUT/"extraction_audit.json",dict(pyradiomics=radiomics.__version__,images=audit))
    print("All four interpretable image/patient feature matrices saved",flush=True)


def evaluate():
    require_lock()
    patients, split = read("patients.csv"),read("random_patient_assignments.csv")
    y=patients.label.to_numpy(int)
    arrays={g:np.load(OUT/"features"/f"{g}.npy") for g in GROUPS}
    arrays.update({g:np.load(L.OUT/"features"/f"{f}.npy") for g,f in (("RAD","radiomics_percentile"),("DN-A","radimagenet_percentile"))})
    scores,records,predictions,coefficients,fit_audit={},[],[],[],[]
    dictionary=pd.read_csv(OUT/"feature_dictionary.csv")
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            idx=N.indices_for(patients,split,seed)
            scores[seed]={}
            for group,x in arrays.items():
                model=V.make_head("lr",seed,x.shape[1]) if group in ("RAD","DN-A") else N.full_head(seed)
                model.fit(x[idx["train"]],y[idx["train"]])
                score=model.predict_proba(x)[:,1]
                scores[seed][group]=score
                path=OUT/"models"/f"{group}_{seed}.joblib"
                joblib.dump(model,path,compress=3)
                assert np.allclose(joblib.load(path).predict_proba(x)[:,1],score,atol=1e-12)
                fit_audit.append(dict(seed=seed,group=group,sha256=R.sha256(path),train_patients=patients.patient_alias.iloc[idx["train"]].tolist()))
                if group in GROUPS:
                    names=dictionary[dictionary.group.eq(group)].patient_feature.to_numpy()
                    assert len(model[0].statistics_)==len(names) and np.isfinite(model[0].statistics_).all()
                    selected=model[1].get_support()
                    for name,coef in zip(names[selected],model[-1].coef_[0]):
                        coefficients.append(dict(seed=seed,group=group,feature=name,standardized_coefficient=float(coef)))
            scores[seed]["PAIR"]=(scores[seed]["RAD"]+scores[seed]["DN-A"])/2
            for group,score in scores[seed].items():
                records.append(dict(seed=seed,group=group,role="development_validation",**V.score_metrics(y[idx["validation"]],score[idx["validation"]])))
    configs={g:(g,"full","DN-A") for g in GROUPS}
    decision=N.gate_decision(pd.DataFrame(records),configs)
    winner=decision["selected"]
    if winner:
        for seed in R.SEEDS:
            ix=N.indices_for(patients,split,seed)["validation"]
            scores[seed]["VOTE"]=(scores[seed]["RAD"]+scores[seed]["DN-A"]+scores[seed][winner])/3
            records.append(dict(seed=seed,group="VOTE",role="development_validation",**V.score_metrics(y[ix],scores[seed]["VOTE"][ix])))
        md=pd.DataFrame(records).set_index(["group","seed"])
        delta=md.loc["VOTE",["auroc","average_precision"]]-md.loc["PAIR",["auroc","average_precision"]]
        wins=int(delta.gt(0).all(axis=1).sum())
        decision["vote"]=dict(delta_auroc=float(delta.auroc.mean()),delta_ap=float(delta.average_precision.mean()),joint_wins=wins,
            promoted=bool(delta.auroc.mean()>=.03-1e-12 and delta.average_precision.mean()>=.02-1e-12 and wins>=6))
    R.write_json(OUT/"development_decision.json",decision)
    for seed in R.SEEDS:
        ix=N.indices_for(patients,split,seed)
        for group,score in scores[seed].items():
            records.append(dict(seed=seed,group=group,role="exploratory_test",**V.score_metrics(y[ix["test"]],score[ix["test"]])))
            for role,label in (("validation","development_validation"),("test","exploratory_test")):
                for i in ix[role]:
                    predictions.append(dict(seed=seed,group=group,role=label,patient_alias=patients.patient_alias.iloc[i],label=int(y[i]),score=float(score[i])))
    metrics=pd.DataFrame(records)
    metrics.to_csv(OUT/"metrics_by_role_per_draw.csv",index=False)
    summary=N.summarize(metrics)
    summary.to_csv(OUT/"summary_by_role.csv",index=False)
    pd.DataFrame(predictions).to_csv(OUT/"patient_predictions_by_role.csv",index=False)
    pd.DataFrame(coefficients).to_csv(OUT/"training_coefficients.csv",index=False)
    R.write_json(OUT/"fit_audit.json",fit_audit)
    print(summary[["role","group","auroc_mean","average_precision_mean"]].to_string(index=False),flush=True)
    print(json.dumps(decision),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","extract","evaluate","all"))
    args=parser.parse_args()
    for stage in (("prepare","extract","evaluate") if args.stage=="all" else (args.stage,)):
        globals()[stage]()
