"""Geometric-region and view-preserving renal ultrasound texture pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

import nckuh_dmsa_label_powerset_20261102 as P
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_geometric_texture_20261102"
FEATURES = ROOT / "02_experiments/CYCH_honest_reevaluation_20261003/feat_NCKUH"
SEEDS = [20261102, 20261103, 20261104]
GROUPS = [
    "geometry_anova32_lr",
    "geometric_texture_anova64_lr",
    "combined_anova64_lr",
    "combined_anova128_lr",
    "combined_l1_lr",
    "combined_rbf_svm",
    "combined_xgboost",
]


def prepare():
    P.require_lock()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered protocol")
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "motivation": "APN B-mode findings can include enlargement, altered echogenicity, focal heterogeneity and loss of corticomedullary differentiation. Test explicit geometric-region/view-preserving descriptors after whole-mask pooled representations plateaued",
        "references": [
            "https://pubmed.ncbi.nlm.nih.gov/29382187/",
            "https://pubmed.ncbi.nlm.nih.gov/12069464/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC8305820/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC12385666/",
        ],
        "cohort": "Same 210 source-code-available patients, 414 imaged kidneys and 820 images; 54 missing-reference patients excluded from supervised fitting/scoring",
        "regions": "Three normalized Euclidean distance-to-boundary bands and three principal-long-axis thirds inside the supplied whole-kidney mask. These are geometric regions, not claimed cortex/medulla or lesion annotations",
        "features": "P1-P99 masked image; shape, 16-bin intensity histogram, robust intensity statistics, gradient, LoG scales1/2, uniform LBP, and four-orientation Gabor magnitude. No labels enter image feature extraction",
        "views": "Within each kidney preserve longitudinal and transverse descriptors, their difference, overall mean/SD and availability; retain ordered own/opposite kidney pair",
        "combination": "Compare new descriptors alone with the already locked 1930-dimensional paired PyRadiomics/directional representation. Geometry-only is a negative/acquisition control",
        "heads": "Training-only median imputation, scaling/variance filtering and fixed ANOVA64/128 balanced LR C=.01; L1 LR C=.05, RBF-SVM C=1, and shallow regularized XGBoost are fixed family controls, not tuned on development",
        "split": "Same three R/L/B/N-stratified patient draws, 126 train / 42 development / 42 unscored exploratory test. No k-fold and no patient crosses roles",
        "gate": "Advance only if mean AUROC>=.70 and AP>=.78 with both metrics improving over BCE-Adam paired handcrafted reference by >=.03/.02 in at least2/3 draws, or both means>.8 with both>.8 in at least2/3 draws",
        "uncertainty": "10000 R/L/B/N-stratified patient-cluster fixed-prediction resamples over three development draws",
        "caveats": "Ultrasound gray level is acquisition-dependent; geometric bands are not anatomy; calipers/text overlays remain and are not disease evidence; development results remain exploratory",
    }
    write_json(OUT / "protocol.json", protocol)
    paths = [
        Path(__file__), P.SOURCE, P.BASE / "patient_assignments.csv", P.OUT / "handcrafted_pairs.npy",
        P.AUDIT / "patient_label_linkage_private.csv", P.AUDIT / "kidney_label_linkage_private.csv",
        P.AUDIT / "image_labels_dmsa_private.csv", FEATURES / "image_manifest.csv",
    ]
    write_json(OUT / "run_lock.json", {
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in paths},
    })
    print("Registered geometric-region/view-preserving texture pilot", flush=True)


def require_lock():
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(ROOT / name) == expected, name


def distribution_features(values, bins=16):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.full(8 + bins, np.nan, dtype=np.float32)
    hist = np.histogram(values, bins=bins, range=(0, 255))[0].astype(float)
    hist /= max(hist.sum(), 1)
    nz = hist[hist > 0]
    entropy = -(nz * np.log2(nz)).sum()
    stats = [values.mean(), values.std(), *np.percentile(values, [10, 25, 50, 75, 90]), entropy]
    return np.r_[stats, hist].astype(np.float32)


def field_features(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.full(4, np.nan, dtype=np.float32)
    return np.r_[values.mean(), values.std(), np.percentile(values, [50, 90])].astype(np.float32)


def descriptor(gray, mask):
    from skimage.feature import local_binary_pattern
    from skimage.filters import gabor
    mask = mask.astype(bool)
    y, x = np.nonzero(mask)
    area = float(mask.sum())
    y0, y1, x0, x1 = y.min(), y.max()+1, x.min(), x.max()+1
    h, w = y1-y0, x1-x0
    contour = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    largest = max(contour, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, True)
    hull_area = max(float(cv2.contourArea(cv2.convexHull(largest))), 1.0)
    centered = np.c_[x-x.mean(), y-y.mean()]
    eigen = np.linalg.eigvalsh(np.cov(centered, rowvar=False))
    eigen = np.maximum(eigen, 1e-8)
    shape = np.array([
        area/(h*w), w/max(h, 1), perimeter/np.sqrt(area), 4*np.pi*area/max(perimeter**2, 1),
        area/hull_area, np.sqrt(eigen[1]/eigen[0]), eigen[1]/area, eigen[0]/area,
    ], dtype=np.float32)
    image = gray.astype(np.float32)
    gx = ndimage.sobel(image, axis=1); gy = ndimage.sobel(image, axis=0)
    gradient = np.hypot(gx, gy)
    logs = [np.abs(ndimage.gaussian_laplace(image, sigma=s)) for s in (1, 2)]
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    gabors = []
    for theta in (0, np.pi/4, np.pi/2, 3*np.pi/4):
        real, imag = gabor(image/255., frequency=.15, theta=theta)
        gabors.append(np.hypot(real, imag))
    distance = ndimage.distance_transform_edt(mask)
    normalized = distance / max(float(distance.max()), 1.0)
    regions = [mask & (normalized <= 1/3), mask & (normalized > 1/3) & (normalized <= 2/3),
               mask & (normalized > 2/3)]
    features = [shape]
    for region in regions:
        parts = [distribution_features(image[region]), field_features(gradient[region])]
        parts.extend(field_features(field[region]) for field in logs)
        lbp_hist = np.bincount(lbp[region].astype(int), minlength=10)[:10].astype(float)
        lbp_hist /= max(lbp_hist.sum(), 1)
        parts.append(lbp_hist.astype(np.float32))
        parts.extend(field_features(field[region])[:2] for field in gabors)
        features.append(np.concatenate(parts))
    principal = np.linalg.eigh(np.cov(centered, rowvar=False))[1][:, -1]
    projection = centered @ principal
    cuts = np.linspace(projection.min(), projection.max(), 4)
    projected = np.full(mask.shape, np.nan, dtype=np.float32); projected[mask] = projection
    for i in range(3):
        region = mask & (projected >= cuts[i]) & (projected <= cuts[i+1] if i == 2 else projected < cuts[i+1])
        features.append(np.r_[distribution_features(image[region]), field_features(gradient[region]),
                              np.bincount(lbp[region].astype(int), minlength=10)[:10] / max(region.sum(), 1)])
    full = np.concatenate(features).astype(np.float32)
    assert full.shape == (284,) and np.isfinite(full).all()
    return full, shape


def extract():
    require_lock()
    if (OUT / "image_geometric_texture.npy").exists():
        raise FileExistsError("Preserve extracted descriptors")
    _, _, images = P.load_reference()
    values, geometry = [], []
    for i, ident in enumerate(images.image_id):
        gray = cv2.imread(str(FEATURES / "percentile" / f"{ident}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(FEATURES / "masks" / f"{ident}.png"), cv2.IMREAD_GRAYSCALE) > 0
        current, shape = descriptor(gray, mask)
        values.append(current); geometry.append(shape)
        if (i+1) % 100 == 0: print(f"Extracted {i+1}/{len(images)} images", flush=True)
    values, geometry = np.stack(values), np.stack(geometry)
    np.save(OUT / "image_geometric_texture.npy", values)
    np.save(OUT / "image_geometry.npy", geometry)
    write_json(OUT / "feature_provenance.json", {
        "images": len(images), "geometric_texture_shape": list(values.shape), "geometry_shape": list(geometry.shape),
        "finite": bool(np.isfinite(values).all()), "label_fields_used": [],
        "geometric_texture_sha256": digest(OUT / "image_geometric_texture.npy"),
        "geometry_sha256": digest(OUT / "image_geometry.npy"),
    })


def view_preserving_pairs(image_values, images, patients):
    width = image_values.shape[1]
    sides = []
    for alias in patients.patient_alias:
        patient_sides = []
        for side in ("left", "right"):
            subset = images[(images.patient_alias == alias) & (images.side == side)]
            rows = subset.index.to_numpy()
            if len(rows):
                overall_mean = image_values[rows].mean(0); overall_sd = image_values[rows].std(0)
            else:
                overall_mean = overall_sd = np.full(width, np.nan)
            parts = [overall_mean, overall_sd]
            available = [float(len(rows) > 0)]
            typed = []
            for suffix in ("L", "T"):
                selected = subset.index[subset.view.str.endswith(suffix)].to_numpy()
                typed.append(image_values[selected].mean(0) if len(selected) else np.full(width, np.nan))
                available.append(float(len(selected) > 0))
            parts.extend(typed); parts.append(typed[0]-typed[1]); parts.append(np.asarray(available))
            patient_sides.append(np.concatenate(parts))
        sides.append(np.r_[patient_sides[0], patient_sides[1]])
    return np.asarray(sides, dtype=np.float32)


def oriented_kidneys(patient_pairs, patients, kidneys):
    lookup = {alias: i for i, alias in enumerate(patients.patient_alias)}
    return np.stack([patient_pairs[lookup[row.patient_alias]] if row.side == "left"
                     else P.swap_pair(patient_pairs[lookup[row.patient_alias]:lookup[row.patient_alias]+1])[0]
                     for row in kidneys.itertuples(index=False)])


def fit_predict(group, x, train, dev, y, seed):
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if "anova32" in group: k = 32
    elif "anova64" in group: k = 64
    else: k = 128
    transform = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                              VarianceThreshold(), SelectKBest(f_classif, k=k))
    z_train = transform.fit_transform(x[train], y[train]); z_dev = transform.transform(x[dev])
    if group.endswith("l1_lr"):
        model = LogisticRegression(C=.05, penalty="l1", class_weight="balanced", solver="liblinear",
                                   max_iter=3000, random_state=seed)
        model.fit(z_train, y[train]); score = model.predict_proba(z_dev)[:, 1]
    elif group.endswith("rbf_svm"):
        from sklearn.svm import SVC
        model = SVC(C=1, gamma="scale", class_weight="balanced", probability=False, random_state=seed)
        model.fit(z_train, y[train]); decision = model.decision_function(z_dev)
        score = 1/(1+np.exp(-decision))
    elif group.endswith("xgboost"):
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=200, max_depth=2, learning_rate=.03, subsample=.8,
                              colsample_bytree=.5, min_child_weight=5, reg_lambda=10, objective="binary:logistic",
                              eval_metric="logloss", n_jobs=4, random_state=seed)
        target = y[train]; weight = np.where(target == 1, len(target)/(2*target.sum()),
                                             len(target)/(2*(len(target)-target.sum())))
        model.fit(z_train, target, sample_weight=weight); score = model.predict_proba(z_dev)[:, 1]
    else:
        model = LogisticRegression(C=.01, class_weight="balanced", solver="liblinear",
                                   max_iter=3000, random_state=seed)
        model.fit(z_train, y[train]); score = model.predict_proba(z_dev)[:, 1]
    return score, transform, model


def run():
    import joblib
    from threadpoolctl import threadpool_limits
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed pilot")
    patients, kidneys, images = P.load_reference()
    new = np.load(OUT / "image_geometric_texture.npy")
    geometry = np.load(OUT / "image_geometry.npy")
    assert new.shape == (1027, 284) and geometry.shape == (1027, 8)
    new_pairs = view_preserving_pairs(new, images, patients)
    geometry_pairs = view_preserving_pairs(geometry, images, patients)
    locked_pairs = np.load(P.OUT / "handcrafted_pairs.npy")
    assert locked_pairs.shape == (210, 1930)
    matrices = {
        "geometry": oriented_kidneys(geometry_pairs, patients, kidneys),
        "new": oriented_kidneys(new_pairs, patients, kidneys),
        "combined": np.c_[oriented_kidneys(locked_pairs, patients, kidneys),
                           oriented_kidneys(new_pairs, patients, kidneys)],
    }
    y = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(P.BASE / "patient_assignments.csv")
    outputs, metrics_rows = [], []
    (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in SEEDS:
            mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
            roles = kidneys.patient_alias.map(mapping)
            train = np.flatnonzero(roles.eq("train")); dev = np.flatnonzero(roles.eq("development"))
            assert kidneys.patient_alias.iloc[train].nunique() == 126 and kidneys.patient_alias.iloc[dev].nunique() == 42
            for group in GROUPS:
                family = "geometry" if group.startswith("geometry_") else ("new" if group.startswith("geometric_texture_") else "combined")
                score, transform, model = fit_predict(group, matrices[family], train, dev, y, seed)
                joblib.dump({"transform": transform, "model": model, "family": family,
                             "train_patient_aliases": sorted(kidneys.patient_alias.iloc[train].unique())},
                            OUT / "models" / f"{seed}_{group}.joblib", compress=3)
                current = metric(y[dev], score)
                metrics_rows.append({"seed": seed, "group": group, "patients": 42,
                                     "kidneys": len(dev), **current})
                for i, value in zip(dev, score):
                    row = kidneys.iloc[i]
                    outputs.append({"seed": seed, "group": group, "patient_alias": row.patient_alias,
                                    "kidney_alias": row.kidney_alias, "DMSA_APN": row.DMSA_APN,
                                    "side": row.side, "label": int(y[i]), "score": float(value)})
            print(f"Completed geometric texture draw {seed}", flush=True)
    predictions = pd.DataFrame(outputs); metrics_frame = pd.DataFrame(metrics_rows)
    predictions.to_csv(OUT / "development_predictions.csv", index=False)
    metrics_frame.to_csv(OUT / "development_metrics.csv", index=False)
    summary = metrics_frame.groupby("group")[["auroc", "ap", "auprc", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]; summary.to_csv(OUT / "development_summary.csv")
    ci, differences, _ = cluster_intervals(predictions, GROUPS, SEEDS, "combined_anova64_lr", 10000)
    ci.to_csv(OUT / "development_cluster_ci.csv", index=False)
    differences.to_csv(OUT / "development_cluster_differences.csv", index=False)
    print(summary[["auroc_mean", "ap_mean"]].to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "extract", "run"])
    args = parser.parse_args()
    {"prepare": prepare, "extract": extract, "run": run}[args.command]()
