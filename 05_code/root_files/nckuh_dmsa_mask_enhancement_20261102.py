"""Development-only post-segmentation enhancement pilot for corrected NCKUH DMSA labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_kidney_baseline_20261102 import (
    AUDIT, FEATURES, OUT as BASE, ROOT, SEEDS, SOURCE, load_reference,
)
from nckuh_only_study_20261102 import digest, metric, write_json

OUT = ROOT / "02_experiments/NCKUH_DMSA_mask_enhancement_20261102"
PILOT_SEEDS = SEEDS[:3]
VARIANTS = ("base", "clahe", "bilateral", "unsharp")
REFERENCE = "BCE_Adam"
GROUPS = (
    "base_anova64_lr",
    "clahe_anova64_lr",
    "bilateral_anova64_lr",
    "unsharp_anova64_lr",
    "mean4_anova64_lr",
    "delta_anova64_lr",
    "stack_anova64_lr",
    "stack_anova128_lr",
    "stack_anova128_xgboost",
    "stack_anova128_lr_plus_bce_adam",
    "stack_anova128_xgboost_plus_bce_adam",
)
BASELINE = ROOT / "02_experiments/NCKUH_DMSA_metric_optimization_20261102"
INPUTS = [
    SOURCE,
    AUDIT / "audit_summary.json",
    AUDIT / "patient_label_linkage_private.csv",
    AUDIT / "kidney_label_linkage_private.csv",
    AUDIT / "image_labels_dmsa_private.csv",
    FEATURES / "image_manifest.csv",
    BASE / "patient_assignments.csv",
    BASELINE / "pilot_predictions.csv",
    Path(__file__),
]


def nearest_extension(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    binary = np.asarray(mask, dtype=bool)
    if binary.all():
        return gray.copy()
    indices = distance_transform_edt(~binary, return_distances=False, return_indices=True)
    return gray[tuple(indices)].copy()


def robust_base(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binary = np.asarray(mask, dtype=bool)
    extended = nearest_extension(gray, binary)
    lo, hi = np.percentile(gray[binary], [1, 99])
    if hi <= lo:
        normalized = extended.astype(np.uint8)
    else:
        normalized = np.rint(np.clip((extended.astype(float) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return normalized, np.where(binary, normalized, 0).astype(np.uint8)


def enhancement_variants(gray: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    binary = np.asarray(mask, dtype=bool)
    extended, base = robust_base(gray, binary)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(extended)
    bilateral = cv2.bilateralFilter(extended, d=5, sigmaColor=25, sigmaSpace=3)
    blur = cv2.GaussianBlur(extended, (0, 0), sigmaX=1.2)
    unsharp = cv2.addWeighted(extended, 1.5, blur, -0.5, 0)
    outputs = {
        "base": base,
        "clahe": np.where(binary, clahe, 0).astype(np.uint8),
        "bilateral": np.where(binary, bilateral, 0).astype(np.uint8),
        "unsharp": np.where(binary, unsharp, 0).astype(np.uint8),
    }
    assert all(v.shape == gray.shape and v.dtype == np.uint8 for v in outputs.values())
    assert all(not np.any(v[~binary]) for v in outputs.values())
    return outputs


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve the registered enhancement protocol")
    load_reference()
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "Test whether deterministic post-segmentation ultrasound enhancement reveals complementary kidney-level APN signal",
        "cohort": "210 source-code-available patients, 414 imaged kidneys, 820 images; 54 unavailable DMSA outcomes excluded",
        "unit": "One score per imaged kidney; all patient roles remain grouped",
        "roles": "Existing fixed first three 126 train / 42 development / 42 unscored exploratory-test patient draws",
        "input": "Existing native kidney crop and fixed whole-kidney mask; no segmentation retraining or reference-informed mask change",
        "variants": {
            "base": "Within-mask P1-P99 robust normalization",
            "clahe": "Base extended image, OpenCV CLAHE clipLimit2 tile8x8, then mask outside zero",
            "bilateral": "Base extended image, bilateral d5 sigmaColor25 sigmaSpace3, then mask outside zero",
            "unsharp": "Base extended image plus 0.5 high-frequency residual from Gaussian sigma1.2, then mask outside zero",
        },
        "radiomics": "Unchanged PyRadiomics original+coif1 feature definition; Energy and TotalEnergy excluded; image labels never enter extraction",
        "context": "Preserve longitudinal/transverse views and ordered own/opposite kidney context; opposite diagnosis is never an input",
        "models": "Train-only median imputation, standardization, variance filtering and fixed ANOVA64/128; balanced LR C.01 or fixed shallow XGBoost",
        "fusion": "Two preregistered arithmetic probability averages with the existing same-draw paired-handcrafted BCE-Adam development prediction",
        "prohibited": "No DMSA field, MRN, dates, source row, prior label, test-derived model/weight/threshold, or best-seed reporting",
        "gate": "AUROC>0.8 and AP>0.8 as development point estimates; no test scoring unless a later separate method lock is made",
        "seeds": list(PILOT_SEEDS),
        "groups": list(GROUPS),
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in INPUTS},
        "protocol_sha256": digest(OUT / "protocol.json"),
    })
    print("Registered deterministic mask-enhancement pilot", flush=True)


def require_lock(source_override: Path | None = None) -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        path = ROOT / name
        if source_override is not None and path.resolve() == Path(__file__).resolve():
            path = source_override
        assert digest(path) == expected, name


def build_features() -> None:
    require_lock()
    if (OUT / "feature_audit.json").exists():
        raise FileExistsError("Preserve completed enhancement features")
    _, _, images = load_reference()
    images = images[images.kidney_APN.notna()].reset_index(drop=True)
    assert len(images) == 820
    for variant in VARIANTS:
        (OUT / "images" / variant).mkdir(parents=True, exist_ok=True)
    max_base_difference = 0
    for row in images.itertuples(index=False):
        gray = cv2.imread(str(FEATURES / "native" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(FEATURES / "masks" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE) > 0
        if gray is None or mask is None or gray.shape != mask.shape or not mask.any():
            raise ValueError(f"Invalid native image/mask: {row.image_id}")
        variants = enhancement_variants(gray, mask)
        existing = cv2.imread(str(FEATURES / "percentile" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE)
        max_base_difference = max(max_base_difference, int(np.abs(variants["base"].astype(int) - existing.astype(int)).max()))
        for variant, array in variants.items():
            if not cv2.imwrite(str(OUT / "images" / variant / f"{row.image_id}.png"), array):
                raise IOError(row.image_id)
    assert max_base_difference == 0

    sys.path.insert(0, str(ROOT / "02_experiments/CYCH_honest_reevaluation_20261003"))
    from extract_site import _radiomics_worker, run_pool
    for variant in VARIANTS:
        jobs = [(ident, OUT / "images" / variant / f"{ident}.png", FEATURES / "masks" / f"{ident}.png")
                for ident in images.image_id]
        frame = run_pool(_radiomics_worker, jobs, processes=8)
        frame.to_csv(OUT / f"radiomics_{variant}.csv")
        print(f"Radiomics complete: {variant} {frame.shape}", flush=True)
    write_json(OUT / "feature_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "images": len(images),
        "variants": list(VARIANTS),
        "base_max_abs_difference_from_existing": max_base_difference,
        "outside_mask_nonzero_pixels": 0,
        "feature_files": {v: {"sha256": digest(OUT / f"radiomics_{v}.csv")}
                          for v in VARIANTS},
        "labels_used_in_extraction": [],
    })


def patient_pair_features(image_values: np.ndarray, images: pd.DataFrame, patients: pd.DataFrame) -> np.ndarray:
    width = image_values.shape[1]
    patient_pairs = []
    for alias in patients.patient_alias:
        sides = []
        for side in ("left", "right"):
            subset = images[(images.patient_alias == alias) & (images.side == side)]
            rows = subset.index.to_numpy()
            overall_mean = image_values[rows].mean(0) if len(rows) else np.full(width, np.nan)
            overall_sd = image_values[rows].std(0) if len(rows) else np.full(width, np.nan)
            views = []
            flags = [float(len(rows) > 0)]
            for suffix in ("L", "T"):
                selected = subset.index[subset.view.str.endswith(suffix)].to_numpy()
                views.append(image_values[selected].mean(0) if len(selected) else np.full(width, np.nan))
                flags.append(float(len(selected) > 0))
            sides.append(np.concatenate([overall_mean, overall_sd, views[0], views[1], views[0] - views[1], flags]))
        patient_pairs.append(np.concatenate(sides))
    return np.asarray(patient_pairs, dtype=np.float32)


def swap_pair(values: np.ndarray) -> np.ndarray:
    half = values.shape[1] // 2
    assert values.shape[1] % 2 == 0
    return np.c_[values[:, half:], values[:, :half]]


def orient_kidneys(patient_pairs: np.ndarray, patients: pd.DataFrame, kidneys: pd.DataFrame) -> np.ndarray:
    lookup = {alias: i for i, alias in enumerate(patients.patient_alias)}
    swapped = swap_pair(patient_pairs)
    return np.stack([patient_pairs[lookup[row.patient_alias]] if row.side == "left"
                     else swapped[lookup[row.patient_alias]] for row in kidneys.itertuples(index=False)])


def fit_predict(values: np.ndarray, train: np.ndarray, dev: np.ndarray, y: np.ndarray,
                seed: int, k: int, xgboost: bool = False):
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    transform = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                              VarianceThreshold(), SelectKBest(f_classif, k=k))
    z_train = transform.fit_transform(values[train], y[train])
    z_dev = transform.transform(values[dev])
    if xgboost:
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=200, max_depth=2, learning_rate=.03, subsample=.8,
                              colsample_bytree=.5, min_child_weight=5, reg_lambda=10,
                              objective="binary:logistic", eval_metric="logloss", n_jobs=4,
                              random_state=seed)
        target = y[train]
        weight = np.where(target == 1, len(target) / (2 * target.sum()),
                          len(target) / (2 * (len(target) - target.sum())))
        model.fit(z_train, target, sample_weight=weight)
    else:
        model = LogisticRegression(C=.01, class_weight="balanced", solver="liblinear",
                                   max_iter=3000, random_state=seed)
        model.fit(z_train, y[train])
    return model.predict_proba(z_dev)[:, 1], transform, model


def run() -> None:
    require_lock()
    if (OUT / "pilot_predictions.csv").exists():
        raise FileExistsError("Preserve completed enhancement predictions")
    patients, kidneys, images_all = load_reference()
    images = images_all[images_all.kidney_APN.notna()].reset_index(drop=True)
    y = kidneys.kidney_APN.to_numpy(int)
    frames = {}
    for variant in VARIANTS:
        frame = pd.read_csv(OUT / f"radiomics_{variant}.csv").set_index("image_id").reindex(images.image_id)
        frame = frame[[c for c in frame if not c.endswith(("_Energy", "_TotalEnergy"))]]
        frames[variant] = frame.to_numpy(float)
    assert len({x.shape for x in frames.values()}) == 1
    pairs = {v: patient_pair_features(x, images, patients) for v, x in frames.items()}
    oriented = {v: orient_kidneys(x, patients, kidneys) for v, x in pairs.items()}
    image_mean = np.mean(np.stack(list(frames.values())), axis=0)
    oriented["mean4"] = orient_kidneys(patient_pair_features(image_mean, images, patients), patients, kidneys)
    image_delta = np.concatenate([frames["base"]] + [frames[v] - frames["base"] for v in VARIANTS[1:]], axis=1)
    oriented["delta"] = orient_kidneys(patient_pair_features(image_delta, images, patients), patients, kidneys)
    oriented["stack"] = np.concatenate([oriented[v] for v in VARIANTS], axis=1)

    assignments = pd.read_csv(BASE / "patient_assignments.csv")
    reference = pd.read_csv(BASELINE / "pilot_predictions.csv")
    reference = reference[(reference.group == REFERENCE) & reference.seed.isin(PILOT_SEEDS)]
    reference_lookup = reference.set_index(["seed", "kidney_alias"]).score
    (OUT / "models").mkdir(exist_ok=True)
    predictions, metrics = [], []
    with threadpool_limits(limits=4):
        for seed in PILOT_SEEDS:
            roles = kidneys.patient_alias.map(assignments[assignments.seed.eq(seed)].set_index("patient_alias").role)
            train = np.flatnonzero(roles.eq("train")); dev = np.flatnonzero(roles.eq("development"))
            assert kidneys.patient_alias.iloc[train].nunique() == 126
            assert kidneys.patient_alias.iloc[dev].nunique() == 42
            specifications = {
                "base_anova64_lr": (oriented["base"], 64, False),
                "clahe_anova64_lr": (oriented["clahe"], 64, False),
                "bilateral_anova64_lr": (oriented["bilateral"], 64, False),
                "unsharp_anova64_lr": (oriented["unsharp"], 64, False),
                "mean4_anova64_lr": (oriented["mean4"], 64, False),
                "delta_anova64_lr": (oriented["delta"], 64, False),
                "stack_anova64_lr": (oriented["stack"], 64, False),
                "stack_anova128_lr": (oriented["stack"], 128, False),
                "stack_anova128_xgboost": (oriented["stack"], 128, True),
            }
            scores = {}
            for group, (values, k, use_xgb) in specifications.items():
                scores[group], transform, model = fit_predict(values, train, dev, y, seed, k, use_xgb)
                joblib.dump({"transform": transform, "model": model}, OUT / "models" / f"{seed}_{group}.joblib", compress=3)
            ref = np.asarray([reference_lookup.loc[(seed, alias)] for alias in kidneys.kidney_alias.iloc[dev]])
            scores["stack_anova128_lr_plus_bce_adam"] = (scores["stack_anova128_lr"] + ref) / 2
            scores["stack_anova128_xgboost_plus_bce_adam"] = (scores["stack_anova128_xgboost"] + ref) / 2
            for group in GROUPS:
                result = metric(y[dev], scores[group])
                metrics.append({"seed": seed, "role": "development", "group": group,
                                "patients": 42, "kidneys": len(dev), **result})
                for local, index in enumerate(dev):
                    predictions.append({"seed": seed, "group": group,
                                        "patient_alias": kidneys.patient_alias.iloc[index],
                                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                                        "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                                        "side": kidneys.side.iloc[index], "label": int(y[index]),
                                        "score": float(scores[group][local])})
            print(f"Completed enhancement draw {seed}", flush=True)
    prediction_frame = pd.DataFrame(predictions)
    metric_frame = pd.DataFrame(metrics)
    prediction_frame.to_csv(OUT / "pilot_predictions.csv", index=False)
    metric_frame.to_csv(OUT / "pilot_metrics.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap"]].mean().sort_values(["auroc", "ap"], ascending=False)
    summary.to_csv(OUT / "pilot_summary.csv")
    achieved = summary[(summary.auroc > .8) & (summary.ap > .8)].index.tolist()
    write_json(OUT / "pilot_decision.json", {
        "development_groups_above_both_targets": achieved,
        "goal_achieved": False,
        "test_scored": False,
        "reason": "Development target alone cannot establish the full goal; no exploratory-test data were scored",
    })
    print(summary.to_string(), flush=True)


def bootstrap() -> None:
    # Training used the locked source copy. The live source differs only by this
    # UTF-8 audit-output fix for Windows paths containing Chinese characters.
    locked_source = OUT / "locked_source_before_bootstrap_encoding_fix.py"
    require_lock(source_override=locked_source)
    frame = pd.read_csv(OUT / "pilot_predictions.csv")
    baseline = pd.read_csv(BASELINE / "pilot_predictions.csv")
    baseline = baseline[(baseline.group == REFERENCE) & baseline.seed.isin(PILOT_SEEDS)]
    groups = list(GROUPS) + [REFERENCE]
    ci, paired, samples = cluster_intervals(pd.concat([frame, baseline], ignore_index=True),
                                            groups, PILOT_SEEDS, REFERENCE)
    ci.to_csv(OUT / "pilot_confidence_intervals.csv", index=False)
    paired.to_csv(OUT / "pilot_paired_intervals.csv", index=False)
    np.save(OUT / "pilot_cluster_bootstrap.npy", samples)
    write_json(OUT / "completion_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "patients": 210, "imaged_kidneys": 414, "images": 820,
        "test_scored": False, "scored_unit": "kidney", "bootstrap_cluster": "patient",
        "source_unchanged": digest(SOURCE) == json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))["inputs"][str(SOURCE.relative_to(ROOT)).replace("\\", "/")],
        "training_source_sha256": digest(locked_source),
        "bootstrap_source_revision": "UTF-8 read fix only; model predictions were not recomputed",
    })
    print("Enhancement intervals and completion audit written", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "features", "run", "bootstrap"])
    args = parser.parse_args()
    {"prepare": prepare, "features": build_features, "run": run, "bootstrap": bootstrap}[args.stage]()
