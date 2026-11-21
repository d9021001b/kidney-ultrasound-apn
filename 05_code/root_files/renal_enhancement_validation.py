"""Fixed-protocol kidney-ROI enhancement ablation on the existing patient draws.

This is further retrospective exploration, not a new independent test cohort.
Run prepare, features, evaluate, report in that order, or run all.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import renal_random_validation as R

ROOT = R.ROOT
BASE = R.OUT
OUT = Path(os.environ.get("CYCH_ENHANCEMENT_DIR", str(ROOT / "CYCH_roi_enhancement_20260913")))
MODES = ("original", "percentile", "clahe", "bilateral", "bilateral_clahe", "nlm", "nlm_clahe")
GREEN_MODES = ("original", "clahe", "bilateral_clahe")
FAMILIES = ("radiomics", "radimagenet", "radimagenet_aug", "fusion_aug")
LABELS = {
    "original": "Original", "percentile": "ROI P1-P99", "clahe": "CLAHE",
    "bilateral": "Bilateral", "bilateral_clahe": "Bilateral + CLAHE",
    "nlm": "NLM", "nlm_clahe": "NLM + CLAHE",
}
REFERENCES = [
    dict(id="ultrasound_enhancement", citation="Singh P, Mukundan R, De Ryke R. Feature Enhancement in Medical Ultrasound Videos Using Contrast-Limited Adaptive Histogram Equalization. J Digit Imaging. 2020;33:273-285.",
         url="https://pmc.ncbi.nlm.nih.gov/articles/PMC7064707/", doi="10.1007/s10278-019-00211-5",
         scope="Supports investigating denoising followed by contrast enhancement, not renal inflammation accuracy. Their trained CNN and Rayleigh/Lanczos CLAHE are NOT replicated here."),
    dict(id="bilateral", citation="Tomasi C, Manduchi R. Bilateral Filtering for Gray and Color Images. ICCV. 1998:839-846.",
         url="https://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf", doi="10.1109/ICCV.1998.710815",
         scope="Established edge-preserving local filter; OpenCV implementation used."),
    dict(id="nlm", citation="Buades A, Coll B, Morel JM. Non-Local Means Denoising. Image Processing On Line. 2011;1:208-212.",
         url="https://doi.org/10.5201/ipol.2011.bcm_nlm", doi="10.5201/ipol.2011.bcm_nlm",
         scope="OpenCV fast non-local means used. Additive Gaussian assumption is not a verified noise model for these processed B-mode frames."),
    dict(id="clahe_implementation", citation="OpenCV. Histograms - 2: Histogram Equalization.",
         url="https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html",
         scope="Conventional contrast-limited tiled equalization with bilinear interpolation, not the modified ultrasound-paper algorithm."),
    dict(id="nlm_implementation", citation="OpenCV. Denoising.",
         url="https://docs.opencv.org/4.x/d1/d79/group__photo__denoise.html",
         scope="Documents h, search/template window sizes, expected Gaussian white noise and detail loss at high h."),
]


def read_manifest():
    return pd.read_csv(OUT / "image_manifest.csv", dtype={"patient_id": str})


def nearest_roi_extension(gray, mask):
    """Supply boundary values derived solely from real in-ROI pixels to library filters."""
    mask = np.asarray(mask, dtype=bool)
    gray = np.asarray(gray)
    if gray.ndim != 2 or gray.shape != mask.shape or not mask.any():
        raise ValueError("A nonempty same-size two-dimensional ROI is required")
    if mask.all():
        return gray.copy()
    indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return gray[tuple(indices)].copy()


def enhance(gray, mask, mode):
    if mode not in MODES:
        raise ValueError(mode)
    if np.asarray(gray).dtype != np.uint8:
        raise ValueError("Expected native uint8 B-mode intensities")
    mask = np.asarray(mask, dtype=bool)
    ext = nearest_roi_extension(gray, mask)
    if mode == "original" or np.ptp(gray[mask]) == 0:
        return np.where(mask, gray, 0).astype(np.uint8)
    if mode == "percentile":
        lo, hi = np.percentile(gray[mask], [1, 99])
        result = ext if hi <= lo else np.rint(np.clip((ext.astype(float)-lo)/(hi-lo), 0, 1)*255).astype(np.uint8)
    else:
        result = ext
        if mode.startswith("bilateral"):
            result = cv2.bilateralFilter(result, d=5, sigmaColor=20, sigmaSpace=2)
        if mode.startswith("nlm"):
            result = cv2.fastNlMeansDenoising(result, None, h=8, templateWindowSize=7, searchWindowSize=21)
        if mode.endswith("clahe"):
            result = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(result)
    return np.where(mask, result, 0).astype(np.uint8)


def prepare():
    OUT.mkdir(exist_ok=True)
    for sub in ("images", "masks", "features", "models", "qa", "figures_600dpi", "source"):
        (OUT / sub).mkdir(exist_ok=True)
    for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv"):
        dest = OUT / name
        if dest.exists() and R.sha256(dest) != R.sha256(BASE / name):
            raise RuntimeError(f"Refusing to change an established cohort: {name}")
        shutil.copy2(BASE / name, dest)
    protocol = dict(
        design="Further retrospective internal patient-level random-sampling exploration on an already explored cohort.",
        base_run=str(BASE), seeds=R.SEEDS, patients=98, images=349,
        roles=dict(train=58, validation=20, test=20), modes=list(MODES),
        operations=dict(percentile="Within-image ROI percentiles 1 and 99; linear mapping to [0,255]",
            clahe=dict(clipLimit=2.0, tileGridSize=[4,4], distribution="uniform", interpolation="bilinear"),
            bilateral=dict(d=5, sigmaColor=20, sigmaSpace=2),
            nlm=dict(h=8, templateWindowSize=7, searchWindowSize=21)),
        boundary="Nearest in-ROI pixel extension before library filtering; final exterior exactly zero. Outside original pixels cannot influence outputs. Extended pixels participate in CLAHE tile histograms; not a custom masked-histogram CLAHE.",
        constant_roi="Unchanged to avoid manufacturing texture from a constant field.",
        geometric_changes=False, new_segmentation_training=False, labels_unchanged=True,
        pretrained="Existing RadImageNet DenseNet-121; frozen parameters and BatchNorm; x/127.5-1",
        radiomics="Existing 457 selected true 2D PyRadiomics descriptors recomputed for each condition, including four coif1 wavelet bands.",
        green="Original training-only fitted GreenExtractor retained per draw; transform original, CLAHE and bilateral+CLAHE inputs without refitting. A fixed-extractor sensitivity analysis, not enhanced-domain Green refitting.",
        families=list(FAMILIES), green_modes=list(GREEN_MODES),
        augmentation="For augmented arms, matched -2/0/+2 degree images and masks in training only; radiomics and Green rows repeated while pretrained features vary. Validation/test use deterministic angle 0.",
        model_selection="16 original SVM candidates; train-only imputer/scaler/ANOVA; validation mean(AUROC,AP), no refit on validation. Family-level enhancement selection uses validation only; ties use protocol order.",
        extra_arm="Original radiomics plus original and CLAHE pretrained features, with the same train-only rotations.",
        primary="validation_selected_fusion_aug", metrics=["AUROC", "average precision (AP)", "trapezoidal PR area"],
        interpretation="All fixed arms and all ten draws reported; no new independent-cohort or confidence-interval claim. Visual quality cannot establish pathology or diagnostic gain.",
        source_hashes={name: R.sha256(BASE/name) for name in ("image_manifest.csv", "random_patient_assignments.csv")},
    )
    p = OUT / "protocol.json"
    if p.exists() and json.loads(p.read_text()) != protocol:
        raise RuntimeError("Protocol changed; create a separate run directory")
    R.write_json(p, protocol)
    stamp = OUT / "protocol_locked.json"
    if not stamp.exists():
        R.write_json(stamp, dict(utc=datetime.now(timezone.utc).isoformat(), sha256=R.sha256(p),
                                note="Settings locked before new classifier evaluation, but after historical results were known."))
    R.write_json(OUT / "references.json", REFERENCES)
    df = read_manifest()
    records = []
    for mode in MODES:
        (OUT / "images" / mode).mkdir(exist_ok=True)
    for row in df.itertuples():
        ident = row.image_id
        gray = cv2.imread(str(BASE / "native" / f"{ident}.png"), 0)
        mask = cv2.imread(str(BASE / "masks" / f"{ident}.png"), 0) > 0
        assert R.sha256(BASE / "masks" / f"{ident}.png") == row.mask_sha256
        shutil.copy2(BASE / "masks" / f"{ident}.png", OUT / "masks" / f"{ident}.png")
        for mode in MODES:
            arr = enhance(gray, mask, mode)
            path = OUT / "images" / mode / f"{ident}.png"
            assert cv2.imwrite(str(path), arr)
            original = gray[mask].astype(float)
            output = arr[mask].astype(float)
            # These describe pixel transformations, not signal-to-noise or diagnostic quality.
            records.append(dict(image_id=ident, mode=mode, image_sha256=R.sha256(path),
                mask_sha256=row.mask_sha256, roi_pixels=int(mask.sum()),
                roi_mean=float(output.mean()), roi_std=float(output.std()),
                mean_absolute_change=float(np.mean(np.abs(output-original))),
                changed_fraction=float(np.mean(output != original)),
                zero_fraction=float(np.mean(output == 0)), white_fraction=float(np.mean(output == 255))))
    pd.DataFrame(records).to_csv(OUT / "image_transform_audit.csv", index=False)
    selection = pd.read_csv(BASE / "qa/visual_review_selection.csv").head(6)
    selection[["image_id", "patient_alias", "view"]].to_csv(OUT / "qa/visual_review_selection.csv", index=False)
    print(f"Prepared {len(df)} images x {len(MODES)} conditions; protocol locked", flush=True)
    visual_montage()


def visual_montage():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selection = pd.read_csv(OUT / "qa/visual_review_selection.csv")
    fig, axes = plt.subplots(len(selection), len(MODES), figsize=(12.6, 10.6), squeeze=False)
    for r, row in enumerate(selection.itertuples()):
        for c, mode in enumerate(MODES):
            arr = cv2.imread(str(OUT / "images" / mode / f"{row.image_id}.png"), 0)
            axes[r,c].imshow(arr, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axes[r,c].set_xticks([])
            axes[r,c].set_yticks([])
            for spine in axes[r,c].spines.values():
                spine.set_visible(False)
            if r == 0:
                axes[r,c].set_title(LABELS[mode], fontsize=9)
            if c == 0:
                axes[r,c].set_ylabel(row.image_id, fontsize=9)
    fig.subplots_adjust(left=.055, right=.99, top=.93, bottom=.07, hspace=.12, wspace=.08)
    fig.suptitle("Fixed-mask kidney enhancement: identical images and grayscale display range", fontsize=13, y=.982)
    fig.text(.055, .035, "Systematic examples selected before new classification. Calipers remain inside some masks. No lesion diagnosis is assigned.", fontsize=9)
    export_figure(fig, "Figure_E1_roi_enhancement")
    fig.savefig(OUT / "qa/enhancement_preview.png", dpi=140)
    plt.close(fig)


def review_extremes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    audit = pd.read_csv(OUT / "image_transform_audit.csv")
    largest = audit[audit["mode"].eq("clahe")].nlargest(4,"mean_absolute_change").copy()
    largest["reason"] = "Largest CLAHE intensity change"
    smallest = audit[audit["mode"].eq("original")].nsmallest(4,"roi_pixels").copy()
    smallest["reason"] = "Smallest native ROI pixel count"
    selected = pd.concat([largest,smallest]).drop_duplicates("image_id")
    selected[["image_id","reason","roi_pixels","mean_absolute_change"]].to_csv(OUT / "qa/extreme_review_selection.csv",index=False)
    modes = ["original","clahe","bilateral_clahe","nlm_clahe"]
    fig,axes = plt.subplots(len(selected),4,figsize=(10,2*len(selected)),squeeze=False)
    for r,row in enumerate(selected.itertuples()):
        for c,mode in enumerate(modes):
            image=cv2.imread(str(OUT/"images"/mode/f"{row.image_id}.png"),0)
            axes[r,c].imshow(image,cmap="gray",vmin=0,vmax=255,interpolation="nearest")
            axes[r,c].set_xticks([])
            axes[r,c].set_yticks([])
            if r==0:
                axes[r,c].set_title(LABELS[mode],fontsize=10)
            if c==0:
                axes[r,c].set_ylabel(row.image_id,fontsize=10)
    fig.suptitle("Label-blinded technical stress cases; no lesion diagnosis",fontsize=13,y=.995)
    fig.tight_layout(rect=(0,0,1,.98))
    fig.savefig(OUT/"qa/extreme_review.png",dpi=140)
    plt.close(fig)


def audit_legacy_marker_rule():
    from recrop_with_yolo import polygon_mask, bbox_of
    records=[]
    for row in read_manifest().itertuples():
        bgr=cv2.imread(row.clean_path)
        h,w=bgr.shape[:2]
        if row.mask_source == "existing_yolo":
            full=cv2.imread(str(BASE/"predicted_masks"/f"{row.image_id}.png"),0)//255
        else:
            full=polygon_mask(ROOT/"cych_image_label"/Path(row.filename).with_suffix(".txt"),h,w)
        x0,y0,x1,y1=bbox_of(full)
        x0,y0,x1,y1=max(0,x0-4),max(0,y0-4),min(w,x1+4),min(h,y1+4)
        crop=bgr[y0:y1,x0:x1]
        mask=cv2.imread(str(OUT/"masks"/f"{row.image_id}.png"),0)>0
        np.testing.assert_array_equal(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),
                                      cv2.imread(str(BASE/"native"/f"{row.image_id}.png"),0))
        color=(crop.max(2).astype(np.int16)-crop.min(2).astype(np.int16))>8
        if float(color[mask].mean())>.9:
            cv2.imwrite(str(OUT/"qa"/f"color_rule_stress_{row.image_id}.png"),crop*mask[...,None])
        records.append(dict(image_id=row.image_id,roi_pixels=int(mask.sum()),
                            color_rule_pixels=int((color&mask).sum()),
                            color_rule_fraction=float(color[mask].mean())))
    result=pd.DataFrame(records)
    result.to_csv(OUT/"qa/legacy_color_rule_audit.csv",index=False)
    R.write_json(OUT/"qa/legacy_color_rule_summary.json",dict(
        rule="Existing green_learning_cv.preprocess_one uses max(R,G,B)-min(R,G,B)>8 followed by Telea inpainting.",
        evaluation="Color-rule coverage on original clean RGB pixels inside the exact current ROI. No inpainting applied and no ground-truth marker recall claimed.",
        images=len(result),images_with_no_flagged_pixels=int(result.color_rule_pixels.eq(0).sum()),
        median_fraction=float(result.color_rule_fraction.median()),max_fraction=float(result.color_rule_fraction.max()),
        limitation="Achromatic white/black calipers have equal RGB channels and cannot be detected by this rule. Flagged color is not necessarily a caliper."))
    print(result.color_rule_fraction.describe().to_string(),flush=True)


def export_figure(fig, name):
    from PIL import Image
    path = OUT / "figures_600dpi" / f"{name}.png"
    fig.savefig(path, dpi=600, facecolor="white")
    with Image.open(path) as im:
        im.save(path.with_suffix(".tiff"), compression="tiff_lzw", dpi=(600,600))


def features():
    from concurrent.futures import ProcessPoolExecutor
    import torch
    from radimagenet_backbones import radimagenet_densenet121, RADIMAGENET_DIR
    df = read_manifest()
    for mode in MODES:
        path = OUT / "features" / f"radiomics_{mode}.csv"
        if path.exists():
            continue
        if mode == "original":
            shutil.copy2(BASE / "features/radiomics.csv", path)
            continue
        todo = [dict(image_id=row.image_id,
                     gray_path=str(OUT / "images" / mode / f"{row.image_id}.png"),
                     mask_path=str(OUT / "masks" / f"{row.image_id}.png")) for row in df.itertuples()]
        values = []
        with ProcessPoolExecutor(max_workers=6) as pool:
            for i, value in enumerate(pool.map(R.radiomics_one, todo)):
                values.append(value)
                if i % 100 == 0:
                    print(f"Radiomics {mode}: {i+1}/{len(todo)}", flush=True)
        pd.DataFrame(values).to_csv(path, index=False)
    torch.set_num_threads(4)
    model = radimagenet_densenet121().cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    R.write_json(OUT / "features/pretrained_provenance.json", dict(
        path=str(RADIMAGENET_DIR/"DenseNet121.pt"), sha256=R.sha256(RADIMAGENET_DIR/"DenseNet121.pt"),
        matched_backbone_keys=725, trainable_parameters=0, batchnorm="frozen eval", normalization="x/127.5-1"))
    first = df.iloc[0]
    gray = cv2.imread(str(OUT / "images/original" / f"{first.image_id}.png"), 0)
    mask = cv2.imread(str(OUT / "masks" / f"{first.image_id}.png"), 0)//255
    with torch.inference_mode():
        arr = R.canvas_image(gray, mask)
        x = torch.from_numpy(arr.transpose(2,0,1).copy()[None]).cuda().float()/127.5-1
        actual = model(x).cpu().numpy()[0]
    cached = np.load(BASE / "features/masked.npy")[0,0]
    np.testing.assert_allclose(actual, cached, rtol=2e-4, atol=2e-5)
    R.write_json(OUT / "features/baseline_reinference_check.json", dict(image_id=first.image_id,
        dimension=len(actual), max_absolute_difference=float(np.max(np.abs(actual-cached))),
        note="Fresh inference from exact original ROI reproduces reused embedding within GPU batch numerical tolerance."))
    for mode in MODES:
        path = OUT / "features" / f"radimagenet_{mode}.npy"
        if path.exists():
            continue
        if mode == "original":
            shutil.copy2(BASE / "features/masked.npy", path)
            continue
        embeddings = []
        for angle in R.ANGLES:
            batch, collected = [], []
            for i, row in enumerate(df.itertuples()):
                gray = cv2.imread(str(OUT / "images" / mode / f"{row.image_id}.png"), 0)
                mask = cv2.imread(str(OUT / "masks" / f"{row.image_id}.png"), 0)//255
                batch.append(R.canvas_image(gray, mask, angle))
                if len(batch) == 32 or i == len(df)-1:
                    with torch.inference_mode():
                        x = torch.from_numpy(np.stack(batch).transpose(0,3,1,2).copy()).cuda().float()/127.5-1
                        collected.append(model(x).cpu().numpy())
                    batch = []
            embeddings.append(np.concatenate(collected))
            print(f"RadImageNet {mode} angle={angle}: {embeddings[-1].shape}", flush=True)
        np.save(path, np.stack(embeddings))
    assert all(not p.requires_grad for p in model.parameters())


def arm_matrices(df, patients):
    fixed = {}
    for mode in MODES:
        rad = pd.read_csv(OUT / "features" / f"radiomics_{mode}.csv").set_index("image_id").loc[df.image_id]
        rad = rad.loc[:, ~rad.columns.str.endswith(("_Energy", "_TotalEnergy"))]
        assert rad.shape == (349,457) and np.isfinite(rad.to_numpy()).all()
        radiomics = R.aggregate(rad.to_numpy(), df, patients)[None]
        deep = np.stack([R.aggregate(x, df, patients) for x in np.load(OUT / "features" / f"radimagenet_{mode}.npy")])
        fixed[f"{mode}__radiomics"] = radiomics
        fixed[f"{mode}__radimagenet"] = deep[:1]
        fixed[f"{mode}__radimagenet_aug"] = deep
        fixed[f"{mode}__fusion_aug"] = np.concatenate([np.repeat(radiomics,3,axis=0), deep], axis=-1)
    fixed["original_and_clahe__fusion_aug"] = np.concatenate([
        fixed["original__fusion_aug"], fixed["clahe__radimagenet_aug"]], axis=-1)
    return fixed


def green_matrices(seed, df, patients, train):
    import joblib
    out = {}
    extractor = None
    for mode in GREEN_MODES:
        if mode == "original":
            gx = np.load(BASE / "features" / f"green_{seed}.npy")
        else:
            path = OUT / "features" / f"green_{mode}_{seed}.npy"
            if path.exists():
                gx = np.load(path)
            else:
                if extractor is None:
                    extractor = joblib.load(BASE / "models" / f"green_extractor_{seed}.joblib")
                    expected = set(df.loc[df.patient_id.isin(patients.patient_id.iloc[train]), "image_id"])
                    assert extractor.fit_paths == expected, "Green extractor training patients do not match this draw"
                before = extractor._fingerprint_state()
                inputs = []
                for row in df.itertuples():
                    gray = cv2.imread(str(OUT / "images" / mode / f"{row.image_id}.png"), 0)
                    mask = cv2.imread(str(OUT / "masks" / f"{row.image_id}.png"), 0)//255
                    inputs.append(R.canvas_image(gray, mask, size=96)[...,0]/255.)
                gx, diagnostic = extractor.transform(np.asarray(inputs,dtype=np.float32))
                assert before == extractor._fingerprint_state()
                np.save(path,gx)
                R.write_json(OUT / "features" / f"green_{mode}_{seed}_provenance.json", dict(
                    extractor_path=str(BASE / "models" / f"green_extractor_{seed}.joblib"),
                    extractor_sha256=R.sha256(BASE / "models" / f"green_extractor_{seed}.joblib"),
                    unchanged_fingerprint=before, original_fit=extractor.diag, transform=diagnostic,
                    actual_fit_images=sorted(extractor.fit_paths),
                    expected_train_images=df.loc[df.patient_id.isin(patients.patient_id.iloc[train]),"image_id"].tolist()))
                print(f"Green frozen transform {mode} {seed}: {gx.shape}", flush=True)
        out[mode] = R.aggregate(gx, df, patients)[None]
    return out


def evaluate():
    import joblib
    from threadpoolctl import threadpool_limits
    assert json.loads((OUT / "protocol_locked.json").read_text())["sha256"] == R.sha256(OUT / "protocol.json")
    df = read_manifest()
    patients = pd.read_csv(OUT / "patients.csv", dtype={"patient_id":str})
    split = pd.read_csv(OUT / "random_patient_assignments.csv", dtype={"patient_id":str})
    y = patients.label.to_numpy()
    fixed = arm_matrices(df, patients)
    all_metrics, all_predictions, all_tuning, all_selection = [], [], [], []
    start = time.monotonic()
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            per = split[split.seed.eq(seed)]
            ix = {role: np.flatnonzero(patients.patient_id.isin(per.loc[per.role.eq(role),"patient_id"]))
                  for role in ("train","validation","test")}
            train,val,test = ix["train"],ix["validation"],ix["test"]
            arms = fixed.copy()
            for mode, green in green_matrices(seed,df,patients,train).items():
                arms[f"{mode}__green_frozen"] = green
                arms[f"{mode}__green_fusion_aug"] = np.concatenate([np.repeat(green,3,axis=0), fixed[f"{mode}__fusion_aug"]],axis=-1)
            current = {}
            for arm,X in arms.items():
                scores,best,tuning = R.fit_arm(X,y,train,val,test)
                m = dict(seed=seed,model=arm,**R.metrics(y[test],scores),validation_score=best[0],**best[2])
                all_metrics.append(m)
                current[arm] = (scores,best,m)
                all_tuning.append(dict(seed=seed,model=arm,candidates=tuning))
                joblib.dump(best[1],OUT / "models" / f"{arm}_{seed}.joblib",compress=3)
                for p,truth,score in zip(patients.patient_id.iloc[test],y[test],scores):
                    all_predictions.append(dict(seed=seed,model=arm,patient_id=p,label=int(truth),score=float(score)))
            for family,modes in [(f,MODES) for f in FAMILIES]+[("green_frozen",GREEN_MODES),("green_fusion_aug",GREEN_MODES)]:
                eligible = [f"{mode}__{family}" for mode in modes]
                chosen = max(eligible,key=lambda arm: current[arm][1][0])
                scores,best,m = current[chosen]
                name = f"validation_selected_{family}"
                all_metrics.append(dict(m,model=name,selected_arm=chosen))
                all_selection.append(dict(seed=seed,selector=name,selected_arm=chosen,
                    selection_metric="validation mean AUROC/AP only",candidate_scores={arm:current[arm][1][0] for arm in eligible}))
                for p,truth,score in zip(patients.patient_id.iloc[test],y[test],scores):
                    all_predictions.append(dict(seed=seed,model=name,patient_id=p,label=int(truth),score=float(score)))
            pd.DataFrame(all_metrics).to_csv(OUT / "metrics_per_draw.csv",index=False)
            pd.DataFrame(all_predictions).to_csv(OUT / "patient_test_predictions.csv",index=False)
            R.write_json(OUT / "validation_tuning.json",all_tuning)
            R.write_json(OUT / "validation_selection.json",all_selection)
            primary = next(m for m in reversed(all_metrics) if m["model"] == "validation_selected_fusion_aug")
            print(f"Completed seed {seed}: {len(arms)} fitted arms; primary ROC={primary['auroc']:.4f} AP={primary['average_precision']:.4f}; elapsed {(time.monotonic()-start)/60:.1f} min",flush=True)
    summarize()


def summarize():
    result = pd.read_csv(OUT / "metrics_per_draw.csv")
    summaries = []
    for model,g in result.groupby("model",sort=False):
        rec = dict(model=model,draws=len(g),joint_target_draws=int(((g.auroc>.8)&(g.average_precision>.8)).sum()))
        for field in ("auroc","average_precision","auprc_trapezoid"):
            for stat in ("mean","std","min","max"):
                rec[f"{field}_{stat}"] = float(getattr(g[field],stat)())
        summaries.append(rec)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv",index=False)
    comparisons = []
    for family in FAMILIES:
        base = result[result.model.eq(f"original__{family}")].set_index("seed")
        for mode in MODES[1:]:
            new = result[result.model.eq(f"{mode}__{family}")].set_index("seed").loc[base.index]
            dr,dp = new.auroc-base.auroc,new.average_precision-base.average_precision
            comparisons.append(dict(family=family,mode=mode,draws=len(new),delta_auroc_mean=float(dr.mean()),
                delta_ap_mean=float(dp.mean()),auroc_improved_draws=int((dr>0).sum()),
                ap_improved_draws=int((dp>0).sum()),joint_improved_draws=int(((dr>0)&(dp>0)).sum())))
    pd.DataFrame(comparisons).to_csv(OUT / "paired_enhancement_comparisons.csv",index=False)
    print(pd.DataFrame(summaries)[["model","auroc_mean","average_precision_mean","joint_target_draws"]].to_string(index=False),flush=True)


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    inherited = [r for r in json.loads((BASE / "references.json").read_text()) if 4 <= r["number"] <= 10]
    R.write_json(OUT / "inherited_method_references.json", inherited)
    s = pd.read_csv(OUT / "summary.csv").set_index("model")
    m = pd.read_csv(OUT / "metrics_per_draw.csv")
    fig,axes = plt.subplots(1,2,figsize=(10,5.7))
    colors = ["#444444","#217a66","#b13f60","#217a66","#b13f60","#217a66","#b13f60"]
    for ax,field,title in zip(axes,["auroc","average_precision"],["AUROC","Average precision (AP)"]):
        for i,mode in enumerate(MODES):
            values = m.loc[m.model.eq(f"{mode}__fusion_aug"),field].to_numpy()
            ax.scatter(values,i+np.linspace(-.12,.12,len(values)),s=17,color=colors[i],alpha=.65)
            ax.plot(values.mean(),i,marker="D",color="black",markersize=5)
        ax.axvline(.8,ls="--",color="#b13f60",lw=1)
        ax.set_xlim(0,1)
        ax.set_ylim(6.6,-.6)
        ax.set_yticks(range(7),[LABELS[k] for k in MODES],fontsize=9)
        ax.set_title(title,fontsize=12)
        ax.grid(axis="x",alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("ROI enhancement: radiomics + frozen RadImageNet + training rotations",fontsize=12)
    fig.subplots_adjust(left=.16,right=.98,top=.88,bottom=.18,wspace=.55)
    fig.text(.06,.06,"Dots: all 10 matched patient-random draws. Diamonds: means. Repeated patients are not independent cohorts.\nDashed line: 0.80 target. Post-hoc fixed-arm comparisons are exploratory, not confirmatory evidence.",fontsize=9)
    export_figure(fig,"Figure_E2_enhancement_results")
    fig.savefig(OUT / "qa/results_preview.png",dpi=140)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    normalization_plot=[]
    families=[("radiomics","Radiomics"),("radimagenet","Frozen RadImageNet"),("fusion_aug","Fusion + train rotations")]
    for ax,field,title in zip(axes,["auroc","average_precision"],["Change in AUROC","Change in AP"]):
        for i,(family,label) in enumerate(families):
            a=m[m.model.eq(f"original__{family}")].set_index("seed").sort_index()
            b=m[m.model.eq(f"percentile__{family}")].set_index("seed").loc[a.index]
            delta=b[field]-a[field]
            ax.scatter(delta,i+np.linspace(-.12,.12,len(delta)),color="#217a66",s=25,alpha=.75)
            ax.scatter(delta.mean(),i,marker="D",color="black",s=30)
            for seed,value in delta.items():
                normalization_plot.append(dict(family=family,metric=field,seed=int(seed),
                    original=float(a.loc[seed,field]),normalized=float(b.loc[seed,field]),delta=float(value)))
        ax.axvline(0,color="#444444",lw=1,ls="--")
        ax.set_xlim(-.45,.45)
        ax.set_ylim(2.6,-.6)
        ax.set_yticks(range(3),[v[1] for v in families],fontsize=9)
        ax.set_title(title,fontsize=12)
        ax.grid(axis="x",alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Paired effects of ROI P1-P99 intensity normalization",fontsize=13)
    fig.subplots_adjust(left=.18,right=.99,top=.82,bottom=.23,wspace=.68)
    fig.text(.06,.065,"Dots: all ten matched patient-random draws. Diamonds: mean changes. Zero: no change.\nThis post-hoc contrast is exploratory; the validation-selected primary workflow is reported separately.",fontsize=9)
    export_figure(fig,"Figure_E3_paired_normalization")
    fig.savefig(OUT/"qa/normalization_preview.png",dpi=140)
    plt.close(fig)
    pd.DataFrame(normalization_plot).to_csv(OUT/"paired_normalization_plot_data.csv",index=False)
    primary=s.loc["validation_selected_fusion_aug"]
    original=s.loc["original__fusion_aug"]
    rows=["| Condition | Mean AUROC | Mean AP | Both > 0.80 draws |","|---|---:|---:|---:|"]
    for mode in MODES:
        row=s.loc[f"{mode}__fusion_aug"]
        rows.append(f"| {LABELS[mode]} | {row.auroc_mean:.4f} | {row.average_precision_mean:.4f} | {int(row.joint_target_draws)}/10 |")
    rows.append(f"| Validation-selected condition | {primary.auroc_mean:.4f} | {primary.average_precision_mean:.4f} | {int(primary.joint_target_draws)}/10 |")
    normalization_rows=["| Feature family | Original AUROC / AP | ROI P1-P99 AUROC / AP |", "|---|---:|---:|"]
    for family in ("radiomics","radimagenet","radimagenet_aug","fusion_aug"):
        a,b=s.loc[f"original__{family}"],s.loc[f"percentile__{family}"]
        normalization_rows.append(f"| {family} | {a.auroc_mean:.4f} / {a.average_precision_mean:.4f} | {b.auroc_mean:.4f} / {b.average_precision_mean:.4f} |")
    text = "\n".join([
        "# Kidney-ROI enhancement: completed exploratory comparison", "",
        "Seven deterministic processing conditions were evaluated on the same 349 images / 98 patients and the same ten 58/20/20 patient partitions as the previous manuscript. No patient or image was removed to improve scores.", "",
        f"The protocol-designated primary workflow chooses preprocessing using only each draw's validation patients. It obtained mean AUROC {primary.auroc_mean:.4f} and mean AP {primary.average_precision_mean:.4f}, compared with {original.auroc_mean:.4f} and {original.average_precision_mean:.4f} for the unchanged original-image fusion arm.", "",
        "## Like-for-like fusion comparison", "", *rows, "",
        "The table fixes radiomics + frozen RadImageNet + training-only rotations. The complete summary includes radiomics alone, pretrained features with/without rotation, original-plus-CLAHE feature fusion, and fixed-Green-extractor sensitivity arms. All individual draws and predictions are retained.", "",
        "## ROI intensity normalization signal", "", *normalization_rows, "",
        "ROI P1-P99 normalization improved the descriptive means in these paired comparisons, with larger gains for radiomics alone than for the previously best fusion arm. This interpretation was made after all fixed arms completed. It is a candidate for future development, not an independently confirmed superior preprocessing policy. The validation-selected primary workflow above remains the prespecified selection result; its modest performance must not be replaced by a test-ranked winner.", "",
        "## What changed", "",
        "- Original: unchanged intensities with the established kidney mask.",
        "- ROI P1-P99: within-image ROI percentile intensity normalization.",
        "- CLAHE: conventional OpenCV clip limit 2.0 and 4-by-4 tile grid.",
        "- Bilateral: edge-preserving filter with diameter 5, intensity sigma 20, spatial sigma 2 pixels.",
        "- NLM: non-local means with h=8, template 7 and search 21 pixels.",
        "- Combination conditions apply denoising before CLAHE, not the reverse.",
        "- Filters receive nearest-ROI boundary extension and outputs are remasked. No outside original intensity can influence enhancement. Synthetic extension pixels do participate in library CLAHE histograms; this is not a custom mask-weighted CLAHE implementation.",
        "- Existing per-draw training-fitted Green extractors are frozen and reused, rather than refitted to all enhanced images. This isolates input processing but does not evaluate Green refitting in the enhanced domain.", "",
        "## Interpretation and remaining work", "",
        "This is additional retrospective exploration after previous cohort-wide results were known. Repeated patient appearances must not be treated as independent samples, and ranking fixed arms after observing test scores is exploratory. The predeclared validation-selected workflow is reported separately to expose the cost of preprocessing selection.", "",
        "Brighter or smoother images do not prove improved diagnosis. No clean-image reference is available, so image-change statistics are not called true SNR/PSNR gains. Calipers within the renal mask remain and may be amplified. There is no verified cortex/sinus or scar annotation; bright central tissue cannot simply be removed and called anatomical sinus segmentation. Any morphology or lesion label requires independent clinical review.", "",
        "The implemented bilateral and NLM comparisons are conservative established-library baselines. They are not SRAD, GAN restoration, or an exact replication of Singh et al.'s trained CNN and modified Rayleigh/Lanczos CLAHE. NLM's Gaussian noise assumption is unverified for the supplied processed B-mode frames. No generative pixels are used.", "",
        "Next priorities: retain ROI percentile normalization as a candidate alongside the original image; reviewer-annotated caliper masks and exclusion sensitivity without invented tissue; multiscale local texture and patient-level multiple-instance learning; then training-only adaptation of the existing medical backbone. Further enhancement tuning must use development patients, not repeatedly select settings by these test scores. See NEXT_EXPERIMENTS.md for literature-linked hypotheses and qa/visual_observations.md for the technical spot checks, including why the historical color-only marker detector was not reused blindly.", "",
        "## Reproduction", "", "```powershell",
        "& 'C:\\Users\\udoo_w2\\AppData\\Local\\Programs\\Python\\Python311\\python.exe' renal_enhancement_validation.py all",
        "& 'C:\\Users\\udoo_w2\\AppData\\Local\\Programs\\Python\\Python311\\python.exe' -m unittest test_renal_enhancement_validation -v",
        "```", "", "The base run and large medical/segmentation checkpoints remain at the recorded local paths. Source copies, environment versions, masks, enhanced PNGs, features, fitted SVMs, audit files, and 600-dpi PNG/TIFF figures are included. Patient identifiers are present; do not publish this bundle as deidentified data.", "",
        "## References", "", *[f"- {r['citation']} {r['url']} Scope: {r['scope']}" for r in REFERENCES], "",
        "## Inherited model definitions and references", "",
        "RadImageNet is a radiologic pretraining dataset; DenseNet-121 is the reused convolutional backbone. PyRadiomics is the feature-extraction implementation. The local GreenExtractor is the existing Saab/PixelHop++-adapted feedforward extractor, not a new named model. SVM denotes support vector machine. ANOVA denotes analysis of variance; univariate ANOVA F scores rank training features. AP denotes average precision, not a pooled PR curve over duplicated patients.", "",
        *[f"- {r['citation']} {r['url']}" for r in inherited], "",
    ])
    (OUT / "RESULTS.md").write_text(text,encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["prepare","features","evaluate","report","review_extremes","audit_legacy_marker_rule","all"])
    args=parser.parse_args()
    stages=["prepare","review_extremes","audit_legacy_marker_rule","features","evaluate","report"] if args.stage=="all" else [args.stage]
    for stage in stages:
        globals()[stage]()
