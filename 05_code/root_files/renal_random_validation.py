"""Patient-level random validation using actual kidney masks and existing weights.

Run prepare, features, evaluate in that order. Historical experiments are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score, auc, precision_recall_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(os.environ.get("CYCH_PROJECT_ROOT", str(Path(__file__).resolve().parent)))
OUT = Path(os.environ.get("CYCH_RUN_DIR", str(ROOT / "CYCH_random_validation_20260913")))
SEEDS = list(range(20260913, 20260923))
ANGLES = [0, -2, 2]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_manifest():
    return pd.read_csv(OUT / "image_manifest.csv", dtype={"patient_id": str})


def prepare():
    from recrop_with_yolo import DEFAULT_WEIGHTS, polygon_mask, bbox_of
    from ultralytics import YOLO
    OUT.mkdir(exist_ok=True)
    for sub in ("native", "masks", "masked", "predicted_masks", "qa", "features", "models", "figures_600dpi"):
        (OUT / sub).mkdir(exist_ok=True)
    src = pd.read_csv(ROOT / "yolo_crops/manifest.csv", dtype={"patient_id": str})
    src = src[src.site.eq("CYCH")].copy()
    assert not src.filename.duplicated().any()
    assert src.groupby("patient_id").label.nunique().max() == 1
    first = src.groupby("patient_id").exam_date_iso.transform("min")
    src.loc[src.exam_date_iso.ne(first)].to_csv(OUT / "excluded_later_exams.csv", index=False)
    src = src.loc[src.exam_date_iso.eq(first)].sort_values(["patient_id", "filename"]).reset_index(drop=True)
    alias = {p: f"P{i+1:03d}" for i, p in enumerate(sorted(src.patient_id.unique()))}
    seg = YOLO(str(DEFAULT_WEIGHTS))
    assert seg.task == "segment"
    records = []
    for i, row in src.iterrows():
        clean = ROOT / "original_png" / row.filename.replace("_red_", "_", 1)
        img = cv2.imread(str(clean))
        assert img is not None, str(clean)
        h, w = img.shape[:2]
        manual = polygon_mask(ROOT / "cych_image_label" / Path(row.filename).with_suffix(".txt"), h, w)
        pred = seg.predict(img, imgsz=640, conf=0.25, device=0, verbose=False)[0]
        mask = np.zeros((h, w), np.uint8)
        n = 0 if pred.masks is None else len(pred.masks.xy)
        confidence = 0.0
        if n:
            # Small high-confidence segments can be calipers; the renal contour is the largest instance.
            j = int(np.argmax([cv2.contourArea(p.astype(np.float32)) for p in pred.masks.xy]))
            confidence = float(pred.boxes.conf[j])
            cv2.fillPoly(mask, [pred.masks.xy[j].astype(np.int32)], 1)
        source = "existing_yolo"
        predicted_iou = int((mask & manual).sum()) / max(1, int((mask | manual).sum()))
        cv2.imwrite(str(OUT / "predicted_masks" / f"I{i+1:04d}.png"), mask*255)
        if mask.sum() < 64 or predicted_iou < 0.75:
            mask = manual.copy()
            source = "manual_polygon_fallback"
        assert mask.sum() >= 64
        inter = int((mask & manual).sum())
        union = int((mask | manual).sum())
        box = bbox_of(mask)
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = max(0, x0-4), max(0, y0-4), min(w, x1+4), min(h, y1+4)
        raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        m = mask[y0:y1, x0:x1]
        kidney = raw * m
        ident = f"I{i+1:04d}"
        for sub, arr in (("native", raw), ("masks", m*255), ("masked", kidney)):
            cv2.imwrite(str(OUT / sub / f"{ident}.png"), arr)
        assert np.all(kidney[m == 0] == 0)
        entry = {k: row[k] for k in ("patient_id", "label", "filename", "exam_date_iso", "view")}
        entry.update(image_id=ident, patient_alias=alias[row.patient_id], clean_path=str(clean),
                     old_bbox_path=str(ROOT / row.relative_path), source_sha256=sha256(clean),
                     mask_source=source, prediction_count=n, confidence=confidence,
                     predicted_iou=predicted_iou, manual_iou=inter/max(1, union), width=x1-x0, height=y1-y0,
                     mask_pixels=int(m.sum()), mask_sha256=sha256(OUT / "masks" / f"{ident}.png"))
        records.append(entry)
        if i % 50 == 0:
            print(f"Segmentation {i+1}/{len(src)}", flush=True)
    df = pd.DataFrame(records)
    assert len(df) == 349 and df.patient_id.nunique() == 98
    hashes = df.groupby("source_sha256").patient_id.nunique()
    assert hashes.max() == 1, "Exact duplicate image occurs across patients"
    df.to_csv(OUT / "image_manifest.csv", index=False)
    patients = df.groupby("patient_id", sort=True).agg(label=("label", "first"), patient_alias=("patient_alias", "first"), images=("label", "size")).reset_index()
    patients.to_csv(OUT / "patients.csv", index=False)
    assignments = []
    for seed in SEEDS:
        pool, test = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(patients, patients.label))
        a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed+1000).split(patients.iloc[pool], patients.label.iloc[pool]))
        train, val = pool[a], pool[b]
        assert not(set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
        for role, ids in (("train", train), ("validation", val), ("test", test)):
            for j in ids:
                assignments.append(dict(seed=seed, role=role, **patients.iloc[j].to_dict()))
    pd.DataFrame(assignments).to_csv(OUT / "random_patient_assignments.csv", index=False)
    write_json(OUT / "protocol.json", dict(seeds=SEEDS, train_patients=58, validation_patients=20,
        test_patients=20, augmentation_angles=ANGLES, threshold_target="AUROC > 0.8 AND average precision > 0.8",
        design="Retrospective internal repeated stratified random sampling; no external or untouched cohort claim.",
        selection="Validation mean of AUROC and average precision; no refit on validation; all ten test draws reported.",
        clinical_covariates=False, patient_aggregation="Per-feature mean and standard deviation across images",
        model_grid=dict(features=[8, 16, 32, "all"], kernel=["linear", "rbf"], C=[0.1, 1.0]),
        segmentation_checkpoint=str(DEFAULT_WEIGHTS), segmentation_sha256=sha256(DEFAULT_WEIGHTS),
        segmentation_task=seg.task, segmentation_training_overlap="Unverified; cannot claim independent segmentation accuracy.",
        fallback_rule="Largest predicted instance. No prediction, fewer than 64 pixels, or supplied-polygon IoU < 0.75: manual polygon. Defined after segmentation QA, before classification. Manual-assisted, not fully automatic.",
        manual_polygons="Existing annotations; authorship, diagnostic blinding, and training overlap require confirmation.",
        segmentation_summary=df.mask_source.value_counts().to_dict()))
    # Label-blinded visual review, systematic subjects plus the lowest-overlap predictions.
    systematic = df.drop_duplicates("patient_id").iloc[np.linspace(0, 97, 8).astype(int)]
    poor = df.nsmallest(4, "predicted_iou")
    review = pd.concat([systematic, poor]).drop_duplicates("image_id")
    review.to_csv(OUT / "qa/visual_review_selection.csv", index=False)
    for page, start in enumerate(range(0, len(review), 4), 1):
        canvas = Image.new("RGB", (1440, 1000), "white")
        draw = ImageDraw.Draw(canvas)
        for j, (_, row) in enumerate(review.iloc[start:start+4].iterrows()):
            raw = cv2.imread(row.clean_path)
            manual = polygon_mask(ROOT / "cych_image_label" / Path(row.filename).with_suffix(".txt"), *raw.shape[:2])
            overlay = raw.copy()
            cv2.drawContours(overlay, cv2.findContours(manual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 200, 200), 1)
            panels = [cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                      np.array(Image.open(OUT / "masked" / f"{row.image_id}.png").convert("RGB"))]
            for k, arr in enumerate(panels):
                im = Image.fromarray(arr)
                im.thumbnail((460, 206))
                canvas.paste(im, (k*480+(460-im.width)//2, j*250+30+(206-im.height)//2))
            draw.text((12, j*250+7), f"{row.image_id} / {row.patient_alias} / {row['view']} | raw | supplied outline | actual masked kidney; {row.mask_source}; IoU {row.manual_iou:.3f}", fill="black")
        canvas.save(OUT / "qa" / f"review_{page}.png")
    print(df.mask_source.value_counts().to_string(), flush=True)


def refine_quality():
    """Apply the tightened pre-classification QA rule to stored predictions without GPU refitting."""
    from recrop_with_yolo import polygon_mask, bbox_of
    df = read_manifest()
    changed = []
    for i, row in df.iterrows():
        if row.mask_source != "existing_yolo" or row.predicted_iou >= .75:
            continue
        img = cv2.imread(row.clean_path)
        h, w = img.shape[:2]
        mask = polygon_mask(ROOT / "cych_image_label" / Path(row.filename).with_suffix(".txt"), h, w)
        x0, y0, x1, y1 = bbox_of(mask)
        x0, y0, x1, y1 = max(0, x0-4), max(0, y0-4), min(w, x1+4), min(h, y1+4)
        raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        m = mask[y0:y1, x0:x1]
        for sub, arr in (("native", raw), ("masks", m*255), ("masked", raw*m)):
            cv2.imwrite(str(OUT / sub / f"{row.image_id}.png"), arr)
        df.loc[i, ["mask_source", "manual_iou", "width", "height", "mask_pixels", "mask_sha256"]] = ["manual_polygon_fallback", 1., x1-x0, y1-y0, int(m.sum()), sha256(OUT / "masks" / f"{row.image_id}.png")]
        changed.append(row.image_id)
    df.to_csv(OUT / "image_manifest.csv", index=False)
    protocol = json.loads((OUT / "protocol.json").read_text())
    protocol["model_grid"]["features"] = [8, 16, 32, "all"]
    protocol["fallback_rule"] = "Largest predicted instance. No prediction, fewer than 64 pixels, or supplied-polygon IoU < 0.75: manual polygon. Fixed after visual QA before classification. Manual-assisted, not fully automatic."
    protocol["segmentation_summary"] = df.mask_source.value_counts().to_dict()
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "qa/quality_refinement.json", dict(changed_images=changed, reason="Visual QA showed peripheral tissue omission among accepted low-overlap masks; threshold raised to .75 before classification."))
    if changed:
        for name in ("radiomics.csv", "masked.npy", "three_channel.npy"):
            path = OUT / "features" / name
            if path.exists():
                path.replace(path.with_name(name + ".pre_quality_refinement"))
    print(f"Refined {len(changed)} masks; {df.mask_source.value_counts().to_dict()}", flush=True)


def radiomics_one(rec):
    import SimpleITK as sitk
    import radiomics
    from radiomics import featureextractor
    radiomics.setVerbosity(logging.ERROR)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    ident = rec["image_id"]
    raw = cv2.imread(str(rec.get("gray_path", OUT / "native" / f"{ident}.png")), 0)
    mask = cv2.imread(str(rec.get("mask_path", OUT / "masks" / f"{ident}.png")), 0) // 255
    if raw is None or mask is None or raw.shape != mask.shape:
        raise ValueError(f"Missing image/mask or incompatible dimensions: {ident}")
    raw = np.pad(raw, 1)
    mask = np.pad(mask, 1)
    ext = featureextractor.RadiomicsFeatureExtractor(binWidth=16, normalize=False, distances=[1, 2],
                                                    force2D=False, minimumROIDimensions=2)
    ext.disableAllImageTypes()
    ext.enableImageTypeByName("Original")
    ext.enableImageTypeByName("Wavelet", customArgs={"wavelet": "coif1", "level": 1})
    ext.disableAllFeatures()
    for cls in ("firstorder", "shape2D", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        ext.enableFeatureClassByName(cls)
    # These are genuinely two-dimensional inputs; force2D would incorrectly suppress an axis.
    result = ext.execute(sitk.GetImageFromArray((raw*mask).astype(np.float32)), sitk.GetImageFromArray(mask))
    result = {k: float(v) for k, v in result.items() if not k.startswith("diagnostics")}
    # Scale-dependent shape measurements in pixels are not anatomical size measurements.
    result = {k: v for k, v in result.items() if "shape2D" not in k or k.endswith(("Elongation", "Sphericity", "PerimeterSurfaceRatio"))}
    # The perimeter/area ratio is scale-dependent too, so exclude it from the classifier.
    result = {k: v for k, v in result.items() if not k.endswith("PerimeterSurfaceRatio")}
    result["image_id"] = ident
    return result


def canvas_image(gray, mask=None, angle=0, channels="gray", size=224):
    if mask is None:
        mask = np.ones_like(gray, np.uint8)
    side = int(np.ceil(np.hypot(*gray.shape))) + 8
    padded = np.zeros((side, side), np.uint8)
    pm = np.zeros_like(padded)
    y, x = (side-gray.shape[0])//2, (side-gray.shape[1])//2
    padded[y:y+gray.shape[0], x:x+gray.shape[1]] = gray
    pm[y:y+gray.shape[0], x:x+gray.shape[1]] = mask
    if angle:
        mat = cv2.getRotationMatrix2D((side/2, side/2), angle, 1)
        padded = cv2.warpAffine(padded, mat, (side, side), flags=cv2.INTER_LINEAR)
        pm = cv2.warpAffine(pm, mat, (side, side), flags=cv2.INTER_NEAREST)
    padded *= pm
    original = cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)
    m = cv2.resize(pm, (size, size), interpolation=cv2.INTER_NEAREST)
    original *= m
    if channels == "gray":
        return np.repeat(original[..., None], 3, axis=2)
    f = original.astype(np.float32)
    gy, gx = np.gradient(f)
    grad = np.sqrt(gx*gx + gy*gy) / np.maximum(f, 1)
    edge = cv2.Canny(original, 30, 90)
    dist = cv2.distanceTransform(255-edge, cv2.DIST_L2, 5)
    def scale(v):
        v = v*m
        high = float(np.percentile(v[m > 0], 99))
        return np.clip(v/max(high, 1e-6)*255, 0, 255).astype(np.uint8)
    return np.stack([original, scale(grad), scale(dist)], axis=-1)


def features():
    from concurrent.futures import ProcessPoolExecutor
    import torch
    from radimagenet_backbones import radimagenet_densenet121, RADIMAGENET_DIR
    df = read_manifest()
    rpath = OUT / "features/radiomics.csv"
    if not rpath.exists():
        values = []
        with ProcessPoolExecutor(max_workers=6) as pool:
            for i, result in enumerate(pool.map(radiomics_one, df.to_dict("records"))):
                values.append(result)
                if i % 50 == 0:
                    print(f"PyRadiomics {i+1}/{len(df)}", flush=True)
        pd.DataFrame(values).to_csv(rpath, index=False)
    torch.set_num_threads(4)
    model = radimagenet_densenet121().cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    write_json(OUT / "features/pretrained_provenance.json", dict(
        path=str(RADIMAGENET_DIR / "DenseNet121.pt"), sha256=sha256(RADIMAGENET_DIR / "DenseNet121.pt"),
        model="RadImageNet DenseNet-121", matched_backbone_keys=725,
        normalization="x / 127.5 - 1; existing verified loader", dimension=1024,
        trainable_backbone_parameters=0, batchnorm="eval; running statistics frozen"))
    for mode in ("bbox", "masked", "three_channel"):
        path = OUT / "features" / f"{mode}.npy"
        if path.exists():
            continue
        out = []
        for angle in (ANGLES if mode != "bbox" else [0]):
            arrays = []
            for row in df.itertuples():
                if mode == "bbox":
                    gray = cv2.imread(row.old_bbox_path, 0)
                    # Preserve the earlier baseline preprocessing exactly (already 224 square).
                    arr = np.repeat(gray[..., None], 3, axis=2)
                else:
                    gray = cv2.imread(str(OUT / "native" / f"{row.image_id}.png"), 0)
                    mask = cv2.imread(str(OUT / "masks" / f"{row.image_id}.png"), 0)//255
                    arr = canvas_image(gray, mask, angle, "gray" if mode == "masked" else "three")
                arrays.append(arr)
            embedding = []
            with torch.inference_mode():
                for b in range(0, len(arrays), 32):
                    x = torch.from_numpy(np.stack(arrays[b:b+32]).transpose(0, 3, 1, 2).copy()).cuda().float()/127.5-1
                    embedding.append(model(x).cpu().numpy())
            out.append(np.concatenate(embedding))
            print(f"RadImageNet {mode} angle={angle}: {out[-1].shape}", flush=True)
        np.save(path, np.stack(out))


def aggregate(features, df, patients):
    return np.stack([np.concatenate([features[df.patient_id.eq(p)].mean(0), features[df.patient_id.eq(p)].std(0)])
                     for p in patients.patient_id]).astype(np.float32)


def metrics(y, s):
    precision, recall, _ = precision_recall_curve(y, s)
    return dict(auroc=float(roc_auc_score(y, s)), average_precision=float(average_precision_score(y, s)),
                auprc_trapezoid=float(auc(recall, precision)))


def fit_arm(X, y, train, val, test):
    candidates = []
    for k in [8, 16, 32, "all"]:
        for kernel in ["linear", "rbf"]:
            for c in [0.1, 1.0]:
                pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                     SelectKBest(f_classif, k="all" if k == "all" else min(k, X.shape[-1])),
                                     SVC(C=c, kernel=kernel, class_weight="balanced", probability=False))
                # Each patient contributes one equally weighted row per predetermined variant.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(X[:, train].reshape(-1, X.shape[-1]), np.tile(y[train], len(X)))
                mv = metrics(y[val], pipe.decision_function(X[0, val]))
                candidates.append(((mv["auroc"] + mv["average_precision"])/2, pipe, dict(k=k, kernel=kernel, C=c), mv))
    best = max(candidates, key=lambda v: v[0])
    scores = best[1].decision_function(X[0, test])
    return scores, best, [dict(params=c[2], validation=c[3]) for c in candidates]


def evaluate():
    import joblib
    from dataclasses import asdict
    from green_learning_cv import GLConfig, GreenExtractor
    from threadpoolctl import threadpool_limits
    df = read_manifest()
    patients = pd.read_csv(OUT / "patients.csv", dtype={"patient_id": str})
    split = pd.read_csv(OUT / "random_patient_assignments.csv", dtype={"patient_id": str})
    y = patients.label.to_numpy()
    rad = pd.read_csv(OUT / "features/radiomics.csv").set_index("image_id").loc[df.image_id]
    assert np.isfinite(rad.to_numpy()).all()
    # TotalEnergy and Energy depend on ROI size; keep intensity/texture and scale-free shape.
    rad = rad.loc[:, ~rad.columns.str.endswith(("_Energy", "_TotalEnergy"))]
    rad.to_csv(OUT / "features/radiomics_model_features.csv")
    radiomics = aggregate(rad.to_numpy(), df, patients)[None]
    fixed = {}
    for name in ("bbox", "masked", "three_channel"):
        fixed[name] = np.stack([aggregate(x, df, patients) for x in np.load(OUT / "features" / f"{name}.npy")])
    images = []
    for row in df.itertuples():
        gray = cv2.imread(str(OUT / "native" / f"{row.image_id}.png"), 0)
        mask = cv2.imread(str(OUT / "masks" / f"{row.image_id}.png"), 0)//255
        images.append(canvas_image(gray, mask, size=96)[..., 0]/255.)
    images = np.asarray(images, dtype=np.float32)
    cfg = GLConfig(in_h=96, in_w=96, patch_samples=(100000, 50000, 50000), chunk=8)
    write_json(OUT / "features/green_config.json", asdict(cfg))
    all_metrics, all_preds, all_tuning = [], [], []
    with threadpool_limits(limits=4):
        for seed in SEEDS:
            per = split[split.seed.eq(seed)]
            ix = {role: np.flatnonzero(patients.patient_id.isin(per.loc[per.role.eq(role), "patient_id"])) for role in ("train", "validation", "test")}
            train, val, test = ix["train"], ix["validation"], ix["test"]
            imtrain = df.patient_id.isin(patients.patient_id.iloc[train]).to_numpy()
            gpath = OUT / "features" / f"green_{seed}.npy"
            if gpath.exists():
                gx = np.load(gpath)
            else:
                extractor = GreenExtractor(cfg, seed).fit(images[imtrain], df.image_id[imtrain].tolist())
                gx, diagnostic = extractor.transform(images)
                joblib.dump(extractor, OUT / "models" / f"green_extractor_{seed}.joblib", compress=3)
                write_json(OUT / "features" / f"green_{seed}_diagnostics.json", dict(fit=extractor.diag, transform=diagnostic))
                np.save(gpath, gx)
            green = aggregate(gx, df, patients)[None]
            ra = np.repeat(radiomics, 3, axis=0)
            arms = {
                "bbox_radimagenet": fixed["bbox"],
                "masked_radimagenet": fixed["masked"][:1],
                "masked_radimagenet_aug": fixed["masked"],
                "masked_pyradiomics": radiomics,
                "masked_radiomics_radimagenet_aug": np.concatenate([ra, fixed["masked"]], axis=-1),
                "masked_radiomics_three_channel_aug": np.concatenate([ra, fixed["three_channel"]], axis=-1),
                "masked_green": green,
                "masked_green_radiomics_radimagenet_aug": np.concatenate([np.repeat(green, 3, axis=0), ra, fixed["masked"]], axis=-1),
            }
            for arm, X in arms.items():
                scores, best, tuning = fit_arm(X, y, train, val, test)
                m = dict(seed=seed, model=arm, **metrics(y[test], scores), n_test=len(test), positives=int(y[test].sum()), validation_score=best[0], **best[2])
                all_metrics.append(m)
                all_tuning.append(dict(seed=seed, model=arm, candidates=tuning))
                joblib.dump(best[1], OUT / "models" / f"{arm}_{seed}.joblib", compress=3)
                for p, truth, score in zip(patients.patient_id.iloc[test], y[test], scores):
                    all_preds.append(dict(seed=seed, model=arm, patient_id=p, label=int(truth), score=float(score)))
                print(f"{seed} {arm}: ROC={m['auroc']:.4f} AP={m['average_precision']:.4f}", flush=True)
            pd.DataFrame(all_metrics).to_csv(OUT / "metrics_per_draw.csv", index=False)
            pd.DataFrame(all_preds).to_csv(OUT / "patient_test_predictions.csv", index=False)
            write_json(OUT / "validation_tuning.json", all_tuning)
    result = pd.DataFrame(all_metrics)
    summaries = []
    for model, group in result.groupby("model", sort=False):
        rec = dict(model=model, draws=len(group), joint_target_draws=int(((group.auroc > .8) & (group.average_precision > .8)).sum()))
        for field in ("auroc", "average_precision", "auprc_trapezoid"):
            for stat in ("mean", "std", "median", "min", "max"):
                rec[f"{field}_{stat}"] = float(getattr(group[field], stat)())
        summaries.append(rec)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    print(pd.DataFrame(summaries)[["model", "auroc_mean", "average_precision_mean", "joint_target_draws"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "refine_quality", "features", "evaluate"])
    args = parser.parse_args()
    globals()[args.stage]()
