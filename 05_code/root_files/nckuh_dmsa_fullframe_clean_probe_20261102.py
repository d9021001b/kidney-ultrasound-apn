"""Clean full-frame ROI probe using the archived NCKUH image+mask panels."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

import cv2
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_kidney_baseline_20261102 import AUDIT, FEATURES, ROOT, SOURCE, load_reference
from nckuh_only_study_20261102 import digest, metric, write_json

OUT = ROOT / "02_experiments/NCKUH_DMSA_fullframe_clean_probe_20261102"
ARCHIVE = ROOT / "03_data_models/ki-data/test_predict-e.zip"
SEEDS = (20261102, 20261103, 20261104)
GROUPS = (
    "current_crop_radiomics_anova64_lr",
    "clean_fullframe_radiomics_anova64_lr",
    "clean_fullframe_radiomics_anova128_lr",
    "clean_fullframe_radiomics_anova128_xgboost",
    "clean_fullframe_color_anova32_lr",
    "clean_fullframe_radiomics_color_anova128_lr",
    "clean_fullframe_stack_xgboost_plus_current_lr",
)
REFERENCE = GROUPS[0]
INPUTS = [
    SOURCE,
    ARCHIVE,
    AUDIT / "audit_summary.json",
    AUDIT / "patient_label_linkage_private.csv",
    AUDIT / "kidney_label_linkage_private.csv",
    AUDIT / "image_labels_dmsa_private.csv",
    FEATURES / "radiomics_percentile_image.csv",
    Path(__file__),
]


def archive_key(name: str) -> tuple[str, str, str] | None:
    tokens = PurePosixPath(name).stem.split("_")
    if len(tokens) < 3:
        return None
    view = re.match(r"(LL|LT|RL|RT)", tokens[2].upper())
    if not view:
        return None
    return tokens[0], re.sub(r"\D", "", tokens[1])[:8], view.group(1)


def archive_index() -> dict[tuple[str, str, str], list[str]]:
    index: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    with zipfile.ZipFile(ARCHIVE) as handle:
        for name in handle.namelist():
            key = archive_key(name)
            if key is None:
                continue
            index[key].append(name)
    return dict(index)


def matched_reference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients, kidneys, images = load_reference()
    images = images[images.kidney_APN.notna()].copy()
    images["date_key"] = images.exam_date_raw.astype(str).str.replace(r"\D", "", regex=True).str[:8]
    available = set(archive_index())
    images = images[[((row.patient_id, row.date_key, row.view) in available)
                     for row in images.itertuples(index=False)]].reset_index(drop=True)
    kidney_aliases = set(images.kidney_alias)
    patient_aliases = set(images.patient_alias)
    kidneys = kidneys[kidneys.kidney_alias.isin(kidney_aliases)].reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(patient_aliases)].reset_index(drop=True)
    assert len(patients) == 111 and len(kidneys) == 212 and len(images) == 418
    assert images.patient_alias.nunique() == 111 and images.kidney_alias.nunique() == 212
    return patients, kidneys, images


def split_archive_panel(encoded: bytes) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[1] % 2:
        raise ValueError("Archive panel must be an even-width image")
    half = image.shape[1] // 2
    left = image[:, :half]
    right = cv2.cvtColor(image[:, half:], cv2.COLOR_BGR2GRAY)
    binary = (right > 127).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        raise ValueError("Archive panel has no kidney mask")
    selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == selected).astype(np.uint8)
    if mask.sum() < 1024:
        raise ValueError("Archive kidney mask is too small")
    return left, mask


def crop_masked(image: np.ndarray, mask: np.ndarray, margin: int = 8) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.nonzero(mask)
    x0, x1 = max(0, int(xx.min()) - margin), min(mask.shape[1], int(xx.max()) + 1 + margin)
    y0, y1 = max(0, int(yy.min()) - margin), min(mask.shape[0], int(yy.max()) + 1 + margin)
    return image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def percentile_gray(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = mask.astype(bool)
    indices = distance_transform_edt(~binary, return_distances=False, return_indices=True)
    extended = gray[tuple(indices)]
    lo, hi = np.percentile(gray[binary], [1, 99])
    normalized = extended if hi <= lo else np.rint(np.clip((extended.astype(float) - lo) / (hi - lo), 0, 1) * 255)
    return np.where(binary, normalized, 0).astype(np.uint8)


def color_descriptor(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(bool)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    selected_rgb = rgb[binary]
    chroma = selected_rgb.max(1) - selected_rgb.min(1)
    saturation = hsv[..., 1][binary]
    hue = hsv[..., 0][binary] * (2 * np.pi / 180)
    weight = saturation / 255
    features = []
    for values in [selected_rgb[:, 0], selected_rgb[:, 1], selected_rgb[:, 2], chroma,
                   saturation, lab[..., 1][binary], lab[..., 2][binary]]:
        features.extend([values.mean(), values.std(), *np.percentile(values, [10, 50, 90])])
    features.extend([(chroma > t).mean() for t in (5, 15, 30, 60)])
    features.extend([(saturation > t).mean() for t in (10, 25, 50, 100)])
    features.extend([(np.cos(hue) * weight).mean(), (np.sin(hue) * weight).mean()])
    return np.asarray(features, dtype=np.float32)


def prepare() -> None:
    from sklearn.model_selection import train_test_split
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered full-frame protocol")
    patients, kidneys, images = matched_reference()
    assignments = []
    for seed in SEEDS:
        train, remainder = train_test_split(np.arange(len(patients)), train_size=70,
                                            random_state=seed, stratify=patients.DMSA_APN)
        development, test = train_test_split(remainder, train_size=20, random_state=seed + 1000,
                                              stratify=patients.DMSA_APN.to_numpy()[remainder])
        for role, indices in (("train", train), ("development", development), ("exploratory_test", test)):
            for index in indices:
                assignments.append({"seed": seed, "patient_alias": patients.patient_alias.iloc[index],
                                    "role": role, "DMSA_APN": patients.DMSA_APN.iloc[index]})
    pd.DataFrame(assignments).to_csv(OUT / "patient_assignments.csv", index=False)
    images[["image_id", "patient_alias", "kidney_alias", "side", "view", "kidney_APN", "DMSA_APN"]].to_csv(
        OUT / "deidentified_image_manifest.csv", index=False)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "Compare current caliper-containing crops with clean full-frame-derived kidney ROIs for side-specific APN",
        "availability_defined_cohort": "111 source-code-available patients, 212 matched kidneys, 418 matched views from the archived full-frame image+mask ZIP",
        "source_distribution": patients.DMSA_APN.value_counts().sort_index().to_dict(),
        "split": "Three fixed patient-stratified draws, 70 train / 20 development / 21 exploratory test; no k-fold",
        "archive_handling": "Match only patient/date/view internally; ignore every filename suffix after LL/LT/RL/RT. Save deidentified image_id outputs only",
        "roi": "Left panel is the full ultrasound frame. The largest connected white component in the aligned right panel is the kidney mask; crop mask box plus8 pixels",
        "features": "Same grayscale P1-P99 PyRadiomics definition as the current path plus fixed within-mask RGB/HSV/Lab color descriptors",
        "models": "Ordered own/opposite kidney and longitudinal/transverse context; train-only imputation/scaling/variance filter/ANOVA; fixed balanced LR or shallow XGBoost",
        "prohibited": "No filename diagnosis suffix, DMSA field, MRN, dates, source rows, test-derived weight/threshold, or test prediction",
        "target": "Development AUROC>0.8 and AP>0.8; full goal remains unproven without later locked confirmation",
        "seeds": list(SEEDS), "groups": list(GROUPS),
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in INPUTS},
        "assignments_sha256": digest(OUT / "patient_assignments.csv"),
        "manifest_sha256": digest(OUT / "deidentified_image_manifest.csv"),
        "protocol_sha256": digest(OUT / "protocol.json"),
    })
    print("Registered clean full-frame availability-defined cohort", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(OUT / "patient_assignments.csv") == lock["assignments_sha256"]
    assert digest(OUT / "deidentified_image_manifest.csv") == lock["manifest_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(ROOT / name) == expected, name


def extract() -> None:
    require_lock()
    if (OUT / "archive_extraction_audit.json").exists():
        raise FileExistsError("Preserve completed full-frame extraction")
    _, _, images = matched_reference()
    index = archive_index()
    for sub in ("clean_rgb", "masks", "percentile"):
        (OUT / sub).mkdir(exist_ok=True)
    descriptors, audit = [], []
    with zipfile.ZipFile(ARCHIVE) as handle:
        for row in images.itertuples(index=False):
            key = (row.patient_id, row.date_key, row.view)
            members = index[key]
            if len(members) != 1:
                raise ValueError(f"Matched study key is not unique: {row.image_id}")
            left, mask = split_archive_panel(handle.read(members[0]))
            roi, roi_mask = crop_masked(left, mask)
            gray = percentile_gray(roi, roi_mask)
            cv2.imwrite(str(OUT / "clean_rgb" / f"{row.image_id}.png"), roi)
            cv2.imwrite(str(OUT / "masks" / f"{row.image_id}.png"), roi_mask * 255)
            cv2.imwrite(str(OUT / "percentile" / f"{row.image_id}.png"), gray)
            descriptor = color_descriptor(roi, roi_mask)
            descriptors.append({"image_id": row.image_id, **{f"color_{i:02d}": float(v) for i, v in enumerate(descriptor)}})
            audit.append({"image_id": row.image_id, "height": left.shape[0], "width": left.shape[1],
                          "mask_pixels": int(mask.sum()), "roi_height": roi.shape[0], "roi_width": roi.shape[1],
                          "color_fraction_chroma_gt15": float(descriptor[36])})
    pd.DataFrame(descriptors).to_csv(OUT / "color_features.csv", index=False)
    pd.DataFrame(audit).to_csv(OUT / "deidentified_extraction_qc.csv", index=False)

    sys.path.insert(0, str(ROOT / "02_experiments/CYCH_honest_reevaluation_20261003"))
    from extract_site import _radiomics_worker, run_pool
    jobs = [(ident, OUT / "percentile" / f"{ident}.png", OUT / "masks" / f"{ident}.png")
            for ident in images.image_id]
    radiomics = run_pool(_radiomics_worker, jobs, processes=8)
    radiomics.to_csv(OUT / "radiomics_clean_fullframe.csv")
    write_json(OUT / "archive_extraction_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": digest(ARCHIVE),
        "patients": 111, "kidneys": 212, "images": 418,
        "saved_clean_rgb": len(list((OUT / "clean_rgb").glob("*.png"))),
        "saved_masks": len(list((OUT / "masks").glob("*.png"))),
        "saved_percentile": len(list((OUT / "percentile").glob("*.png"))),
        "filename_suffix_used_as_predictor": False,
        "labels_used_in_feature_extraction": [],
        "radiomics_sha256": digest(OUT / "radiomics_clean_fullframe.csv"),
        "color_features_sha256": digest(OUT / "color_features.csv"),
    })
    print("Extracted 414 clean full-frame ROIs and features", flush=True)


def patient_pair_features(image_values: np.ndarray, images: pd.DataFrame, patients: pd.DataFrame) -> np.ndarray:
    width = image_values.shape[1]
    output = []
    for alias in patients.patient_alias:
        sides = []
        for side in ("left", "right"):
            subset = images[(images.patient_alias == alias) & (images.side == side)]
            rows = subset.index.to_numpy()
            mean = image_values[rows].mean(0) if len(rows) else np.full(width, np.nan)
            std = image_values[rows].std(0) if len(rows) else np.full(width, np.nan)
            views, flags = [], [float(len(rows) > 0)]
            for suffix in ("L", "T"):
                selected = subset.index[subset.view.str.endswith(suffix)].to_numpy()
                views.append(image_values[selected].mean(0) if len(selected) else np.full(width, np.nan))
                flags.append(float(len(selected) > 0))
            sides.append(np.concatenate([mean, std, views[0], views[1], views[0] - views[1], flags]))
        output.append(np.concatenate(sides))
    return np.asarray(output, dtype=np.float32)


def orient_kidneys(patient_pairs: np.ndarray, patients: pd.DataFrame, kidneys: pd.DataFrame) -> np.ndarray:
    half = patient_pairs.shape[1] // 2
    swapped = np.c_[patient_pairs[:, half:], patient_pairs[:, :half]]
    lookup = {alias: i for i, alias in enumerate(patients.patient_alias)}
    return np.stack([patient_pairs[lookup[row.patient_alias]] if row.side == "left"
                     else swapped[lookup[row.patient_alias]] for row in kidneys.itertuples(index=False)])


def fit_predict(values, train, dev, labels, seed, k, xgboost=False):
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    transform = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                              VarianceThreshold(), SelectKBest(f_classif, k=k))
    x_train = transform.fit_transform(values[train], labels[train])
    x_dev = transform.transform(values[dev])
    if xgboost:
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=200, max_depth=2, learning_rate=.03, subsample=.8,
                              colsample_bytree=.5, min_child_weight=5, reg_lambda=10,
                              objective="binary:logistic", eval_metric="logloss", n_jobs=4,
                              random_state=seed)
        target = labels[train]
        weight = np.where(target == 1, len(target)/(2*target.sum()), len(target)/(2*(len(target)-target.sum())))
        model.fit(x_train, target, sample_weight=weight)
    else:
        model = LogisticRegression(C=.01, class_weight="balanced", solver="liblinear",
                                   max_iter=3000, random_state=seed)
        model.fit(x_train, labels[train])
    return model.predict_proba(x_dev)[:, 1], transform, model


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed clean full-frame predictions")
    patients, kidneys, images = matched_reference()
    labels = kidneys.kidney_APN.to_numpy(int)
    current = pd.read_csv(FEATURES / "radiomics_percentile_image.csv").set_index("image_id").reindex(images.image_id)
    clean = pd.read_csv(OUT / "radiomics_clean_fullframe.csv").set_index("image_id").reindex(images.image_id)
    current = current[[c for c in current if not c.endswith(("_Energy", "_TotalEnergy"))]].to_numpy(float)
    clean = clean[[c for c in clean if not c.endswith(("_Energy", "_TotalEnergy"))]].to_numpy(float)
    color = pd.read_csv(OUT / "color_features.csv").set_index("image_id").reindex(images.image_id).to_numpy(float)
    assert current.shape == clean.shape and current.shape[0] == color.shape[0] == 418
    oriented = {
        "current": orient_kidneys(patient_pair_features(current, images, patients), patients, kidneys),
        "clean": orient_kidneys(patient_pair_features(clean, images, patients), patients, kidneys),
        "color": orient_kidneys(patient_pair_features(color, images, patients), patients, kidneys),
    }
    oriented["clean_color"] = np.concatenate([oriented["clean"], oriented["color"]], axis=1)
    assignments = pd.read_csv(OUT / "patient_assignments.csv")
    predictions, metrics = [], []
    (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in SEEDS:
            roles = kidneys.patient_alias.map(assignments[assignments.seed.eq(seed)].set_index("patient_alias").role)
            train = np.flatnonzero(roles.eq("train")); dev = np.flatnonzero(roles.eq("development"))
            assert kidneys.patient_alias.iloc[train].nunique() == 70 and kidneys.patient_alias.iloc[dev].nunique() == 20
            specs = {
                GROUPS[0]: (oriented["current"], 64, False),
                GROUPS[1]: (oriented["clean"], 64, False),
                GROUPS[2]: (oriented["clean"], 128, False),
                GROUPS[3]: (oriented["clean"], 128, True),
                GROUPS[4]: (oriented["color"], 32, False),
                GROUPS[5]: (oriented["clean_color"], 128, False),
            }
            scores = {}
            for group, (values, k, use_xgb) in specs.items():
                scores[group], transform, model = fit_predict(values, train, dev, labels, seed, k, use_xgb)
                joblib.dump({"transform": transform, "model": model}, OUT / "models" / f"{seed}_{group}.joblib", compress=3)
            scores[GROUPS[6]] = (scores[GROUPS[3]] + scores[GROUPS[0]]) / 2
            for group in GROUPS:
                result = metric(labels[dev], scores[group])
                metrics.append({"seed": seed, "role": "development", "group": group,
                                "patients": 20, "kidneys": len(dev), **result})
                for local, index in enumerate(dev):
                    predictions.append({"seed": seed, "group": group,
                                        "patient_alias": kidneys.patient_alias.iloc[index],
                                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                                        "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                                        "side": kidneys.side.iloc[index], "label": int(labels[index]),
                                        "score": float(scores[group][local])})
            print(f"Completed clean full-frame draw {seed}", flush=True)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap"]].mean().sort_values(["auroc", "ap"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc > .8) & (summary.ap > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {
        "groups_above_both_development_targets": achieved,
        "test_scored": False,
        "goal_achieved": False,
        "reason": "Development-only availability-defined subgroup cannot complete the full objective",
    })
    print(summary.to_string(), flush=True)


def bootstrap() -> None:
    require_lock()
    frame = pd.read_csv(OUT / "development_predictions.csv")
    ci, paired, samples = cluster_intervals(frame, list(GROUPS), SEEDS, REFERENCE)
    ci.to_csv(OUT / "development_confidence_intervals.csv", index=False)
    paired.to_csv(OUT / "development_paired_intervals.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    write_json(OUT / "completion_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "patients": 111, "kidneys": 212, "images": 418,
        "test_scored": False, "scored_unit": "kidney", "bootstrap_cluster": "patient",
    })
    print("Clean full-frame intervals completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "extract", "run", "bootstrap"])
    args = parser.parse_args()
    {"prepare": prepare, "extract": extract, "run": run, "bootstrap": bootstrap}[args.stage]()
