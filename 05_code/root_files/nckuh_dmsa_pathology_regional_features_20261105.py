"""Pathology-motivated regional features inside fixed kidney contours.

The experiment quantifies focal echogenicity, local heterogeneity, contour-derived
radial bands, and pole-versus-mid-kidney variation.  The bands are geometric
measurement regions, not asserted anatomical cortex or medulla labels.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_fullframe_20261103"
REFERENCE = ROOT / "02_experiments/NCKUH_DMSA_uricads_official_frozen_20261105/development_predictions.csv"
OUT = ROOT / "02_experiments/NCKUH_DMSA_pathology_regional_features_20261105"
SEEDS = (20261102, 20261103, 20261104)
CURRENT_GROUP = "current_tile_usfm_reference"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"
STATE_CODE = {"N": 0, "L": 1, "R": 2, "B": 3}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _extended_gray(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return gray[tuple(indices)].astype(np.float32) / 255.0


def _distribution(values: np.ndarray) -> list[float]:
    if values.size < 8:
        return [np.nan] * 7
    q = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return [float(values.mean()), float(values.std()), *map(float, q)]


def _component_features(binary: np.ndarray, mask: np.ndarray) -> list[float]:
    selected = (binary & mask).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(selected, 8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float) if count > 1 else np.empty(0)
    denominator = max(float(mask.sum()), 1.0)
    retained = areas[areas >= max(4.0, denominator * 0.001)]
    return [
        float(selected.sum() / denominator),
        float(areas.max() / denominator) if areas.size else 0.0,
        float(len(retained)),
        float(retained.sum() / denominator) if retained.size else 0.0,
    ]


def image_features(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Extract finite, mask-constrained descriptors from one renal ROI."""
    if gray.ndim != 2 or mask.shape != gray.shape:
        raise ValueError("Expected shape-matched two-dimensional image and mask")
    mask = mask.astype(bool)
    if mask.sum() < 256:
        raise ValueError("Kidney mask is too small")

    filled = _extended_gray(gray, mask)
    gx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gx, gy)
    laplacian = np.abs(cv2.Laplacian(filled, cv2.CV_32F, ksize=3))
    mean9 = cv2.boxFilter(filled, cv2.CV_32F, (9, 9), normalize=True)
    square9 = cv2.boxFilter(filled * filled, cv2.CV_32F, (9, 9), normalize=True)
    local_std = np.sqrt(np.maximum(square9 - mean9 * mean9, 0.0))
    dog = np.abs(cv2.GaussianBlur(filled, (0, 0), 1.0) - cv2.GaussianBlur(filled, (0, 0), 3.0))

    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2.0)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / (jxx + jyy + 1e-6)

    gabor_responses = []
    for wavelength, sigma in ((5.0, 2.5), (9.0, 4.0)):
        orientation = []
        for theta in np.linspace(0.0, np.pi, 4, endpoint=False):
            kernel = cv2.getGaborKernel((15, 15), sigma, theta, wavelength, 0.6, 0.0, ktype=cv2.CV_32F)
            kernel -= kernel.mean()
            kernel /= np.abs(kernel).sum() + 1e-8
            orientation.append(np.abs(cv2.filter2D(filled, cv2.CV_32F, kernel)))
        gabor_responses.append(np.max(np.stack(orientation), axis=0))
    gabor_fine, gabor_coarse = gabor_responses

    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    radial = distance / max(float(distance.max()), 1e-6)
    yy, xx = np.nonzero(mask)
    coordinates = np.c_[yy - yy.mean(), xx - xx.mean()].astype(np.float64)
    _, _, vectors = np.linalg.svd(coordinates, full_matrices=False)
    longitudinal = np.zeros(mask.shape, np.float32)
    projection = coordinates @ vectors[0]
    longitudinal[yy, xx] = np.abs(projection) / max(float(np.abs(projection).max()), 1e-6)

    regions = {
        "whole": mask,
        "outer": mask & (radial <= 1.0 / 3.0),
        "middle": mask & (radial > 1.0 / 3.0) & (radial <= 2.0 / 3.0),
        "central": mask & (radial > 2.0 / 3.0),
        "poles": mask & (longitudinal >= 2.0 / 3.0),
        "midlong": mask & (longitudinal <= 1.0 / 3.0),
        "outer_poles": mask & (radial <= 1.0 / 3.0) & (longitudinal >= 2.0 / 3.0),
    }
    maps = {
        "intensity": filled,
        "gradient": gradient,
        "laplacian": laplacian,
        "local_std": local_std,
        "dog": dog,
        "coherence": coherence,
        "gabor_fine": gabor_fine,
        "gabor_coarse": gabor_coarse,
    }
    values: list[float] = []
    names: list[str] = []
    distribution_names = ("mean", "std", "p10", "p25", "p50", "p75", "p90")
    for region_name, region in regions.items():
        for map_name, image in maps.items():
            values.extend(_distribution(image[region]))
            names.extend(f"{region_name}_{map_name}_{suffix}" for suffix in distribution_names)

    contour = max(cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    area = float(mask.sum())
    perimeter = float(cv2.arcLength(contour, True))
    hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
    _, _, width, height = cv2.boundingRect(contour)
    covariance = np.cov(coordinates, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    geometry = [
        area / max(float(mask.size), 1.0),
        perimeter * perimeter / max(4.0 * np.pi * area, 1e-6),
        area / hull_area,
        float(max(width, height) / max(min(width, height), 1)),
        float(np.sqrt(max(1.0 - eigenvalues[1] / max(eigenvalues[0], 1e-8), 0.0))),
        float(distance.max() / max(np.sqrt(area), 1e-6)),
    ]
    values.extend(geometry)
    names.extend(("mask_fill", "compactness", "solidity", "axis_ratio", "eccentricity", "normalized_max_depth"))

    inside = filled[mask]
    median = float(np.median(inside))
    mad = max(float(np.median(np.abs(inside - median))) * 1.4826, 1e-3)
    zscore = (filled - median) / mad
    for label, binary in (("dark", zscore <= -2.0), ("bright", zscore >= 2.0)):
        values.extend(_component_features(binary, mask))
        names.extend((f"{label}_fraction", f"{label}_largest_component", f"{label}_component_count", f"{label}_retained_fraction"))
        values.extend([
            float((binary & regions["outer"]).sum() / max((binary & mask).sum(), 1)),
            float((binary & regions["poles"]).sum() / max((binary & mask).sum(), 1)),
        ])
        names.extend((f"{label}_outer_share", f"{label}_pole_share"))

    array = np.asarray(values, np.float32)
    array[~np.isfinite(array)] = np.nan
    return array, names


def load_reference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference as load

    patients, kidneys, images = load()
    images = images[images.kidney_APN.notna()].copy().reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].copy().reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].copy().reset_index(drop=True)
    if (len(patients), len(kidneys), len(images)) != (210, 414, 820):
        raise ValueError("Unexpected analytic cohort")
    return patients, kidneys, images


def extract_features(images: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    rows = []
    names: list[str] | None = None
    for index, identifier in enumerate(images.image_id):
        gray = cv2.imread(str(SOURCE / "percentile" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(SOURCE / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None:
            raise FileNotFoundError(identifier)
        current, current_names = image_features(gray, mask > 0)
        if names is None:
            names = current_names
        elif names != current_names:
            raise ValueError("Feature ordering changed")
        rows.append(current)
        if (index + 1) % 100 == 0:
            print(f"Regional features {index + 1}/{len(images)}", flush=True)
    matrix = np.stack(rows)
    if not np.isfinite(matrix).any(axis=0).all():
        raise ValueError("Entirely nonfinite feature column")
    return matrix, names or []


def fit_head(values: np.ndarray, labels: np.ndarray, selected: np.ndarray, seed: int, family: str):
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    counts = np.bincount(labels[selected].astype(int))
    folds = min(4, int(counts[counts > 0].min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    if family == "lr":
        pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("reduce", SelectKBest(f_classif)),
            ("model", LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000, random_state=seed)),
        ])
        parameters = {
            "reduce__k": [16, 32, 64, 96],
            "model__C": [0.003, 0.01, 0.03, 0.1, 0.3],
            "model__penalty": ["l1", "l2"],
        }
    elif family == "rbf":
        pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("reduce", PCA(whiten=True, random_state=seed)),
            ("model", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed)),
        ])
        parameters = {
            "reduce__n_components": [8, 16, 24],
            "model__C": [0.1, 0.3, 1.0, 3.0],
            "model__gamma": ["scale", 0.01, 0.03],
        }
    else:
        raise ValueError(family)
    search = GridSearchCV(pipeline, parameters, scoring="roc_auc", cv=cv, n_jobs=1, refit=True, error_score="raise")
    search.fit(values[selected], labels[selected])
    return search


def state_probabilities(symmetric: np.ndarray, antisymmetric: np.ndarray, states: np.ndarray, train: np.ndarray, seed: int, family: str):
    any_label = (states != STATE_CODE["N"]).astype(int)
    any_model = fit_head(symmetric, any_label, train, seed, family)
    affected = train[states[train] != STATE_CODE["N"]]
    bilateral_label = (states == STATE_CODE["B"]).astype(int)
    bilateral_model = fit_head(symmetric, bilateral_label, affected, seed + 101, family)
    unilateral = train[np.isin(states[train], [STATE_CODE["L"], STATE_CODE["R"]])]
    left_label = (states == STATE_CODE["L"]).astype(int)
    laterality_model = fit_head(antisymmetric, left_label, unilateral, seed + 202, family)

    p_any = any_model.predict_proba(symmetric)[:, 1]
    p_bilateral = bilateral_model.predict_proba(symmetric)[:, 1]
    p_left = laterality_model.predict_proba(antisymmetric)[:, 1]
    probability = np.c_[
        1.0 - p_any,
        p_any * (1.0 - p_bilateral) * p_left,
        p_any * (1.0 - p_bilateral) * (1.0 - p_left),
        p_any * p_bilateral,
    ]
    probability /= probability.sum(1, keepdims=True)
    return probability, {"any": any_model, "bilateral_given_any": bilateral_model, "left_given_unilateral": laterality_model}


def centered_laterality(reference: np.ndarray, candidate: np.ndarray, aliases: pd.Series, weight: float = 0.25) -> np.ndarray:
    frame = pd.DataFrame({"alias": aliases.to_numpy(), "reference": reference, "candidate": candidate})
    centered = frame.candidate - frame.groupby("alias").candidate.transform("mean")
    return np.clip(frame.reference + weight * centered, 0.0, 1.0).to_numpy()


def run() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve completed or partial experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "APN-related focal echogenicity, heterogeneous parenchyma, loss of internal differentiation, and swelling are diluted by whole-kidney pooling",
        "cohort": "210 patients, 414 kidneys, 820 contour-first images",
        "features": "Mask-only intensity, gradient, Laplacian, local variance, difference-of-Gaussians, structure coherence, and multi-orientation Gabor summaries in contour-derived radial and longitudinal regions",
        "region_caution": "Radial regions are reproducible geometric measurement bands and are not cortex/medulla ground truth",
        "model": "Training-only hierarchical N versus any, bilateral versus unilateral, and left versus right heads; LR feature-selection and PCA-RBF families are predeclared",
        "fusion": "Candidate alone; fixed 20% or 25% candidate contribution to the unchanged official URI-CADS quarter-fusion reference; fixed laterality-centered sensitivity",
        "roles": "Exact three 126/42/42 patient assignments; development only; exploratory test unopened",
        "target": "Mean development AUROC > 0.8 and AP > 0.8 before locked confirmation",
        "literature_motivation": [
            "Renal enlargement and cortical echo changes in pediatric APN",
            "Focal hyper/hypoechoic lesions and loss of corticomedullary differentiation",
            "DMSA-corresponding focal cortical/perfusion defects",
        ],
        "inputs": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (Path(__file__), SOURCE / "patient_assignments.csv", SOURCE / "extraction_audit.json", REFERENCE)
        },
    }
    write_json(OUT / "protocol.json", protocol)

    patients, kidneys, images = load_reference()
    features, feature_names = extract_features(images)
    np.save(OUT / "pathology_regional_features.npy", features)
    write_json(OUT / "feature_names.json", feature_names)
    write_json(OUT / "feature_audit.json", {
        "images": len(images),
        "features": features.shape[1],
        "finite_fraction": float(np.isfinite(features).mean()),
        "labels_used_in_extraction": [],
        "feature_sha256": digest(OUT / "pathology_regional_features.npy"),
    })

    pairs = base.patient_pair_features(features, images, patients)
    symmetric, antisymmetric = hierarchy.structured_views(pairs)
    states = patients.DMSA_APN.map(STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(SOURCE / "patient_assignments.csv")
    references = pd.read_csv(REFERENCE)
    references = references[references.group.isin([CURRENT_GROUP, URI_GROUP])]
    references = references.set_index(["seed", "group", "kidney_alias"], verify_integrity=True)

    records: list[dict] = []
    metrics: list[dict] = []
    selections: list[dict] = []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train = np.flatnonzero(patients.patient_alias.map(mapping).eq("train"))
        development = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        target = kidneys.iloc[development]
        if len(train) != 126 or target.patient_alias.nunique() != 42:
            raise ValueError("Unexpected patient split")
        current = references.loc[[(seed, CURRENT_GROUP, alias) for alias in target.kidney_alias]].score.to_numpy(float)
        uri = references.loc[[(seed, URI_GROUP, alias) for alias in target.kidney_alias]].score.to_numpy(float)

        scores = {
            "current_tile_usfm_reference": current,
            "official_uri_fixed_quarter_reference": uri,
        }
        for family in ("lr", "rbf"):
            probability, models = state_probabilities(symmetric, antisymmetric, states, train, seed, family)
            candidate = hierarchy.kidney_scores(probability, patients, kidneys)[development]
            stem = f"pathology_regional_{family}"
            scores.update({
                stem: candidate,
                f"{stem}_quarter_plus_uri": 0.25 * candidate + 0.75 * uri,
                f"{stem}_fifth_plus_uri": 0.20 * candidate + 0.80 * uri,
                f"{stem}_laterality_plus_uri": centered_laterality(uri, candidate, target.patient_alias, 0.25),
            })
            joblib.dump(models, OUT / "models" / f"{seed}_{family}.joblib", compress=3)
            for head, model in models.items():
                selections.append({
                    "seed": seed,
                    "family": family,
                    "head": head,
                    "inner_cv_auroc": float(model.best_score_),
                    **{key: str(value) for key, value in model.best_params_.items()},
                })

        for group, score in scores.items():
            result = hierarchy.metric(labels[development], score)
            metrics.append({"seed": seed, "role": "development", "group": group, "patients": 42, "kidneys": len(development), **result})
            for local, index in enumerate(development):
                records.append({
                    "seed": seed,
                    "group": group,
                    "patient_alias": kidneys.patient_alias.iloc[index],
                    "kidney_alias": kidneys.kidney_alias.iloc[index],
                    "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                    "side": kidneys.side.iloc[index],
                    "label": int(labels[index]),
                    "score": float(score[local]),
                })
        print(f"Completed pathology-regional draw {seed}", flush=True)

    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(records)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "training_inner_cv_selections.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    groups = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(prediction_frame, groups, SEEDS, "official_uri_fixed_quarter_reference")
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(OUT / "decision.json", {
        "groups_above_both_development_targets": achieved,
        "promotion": bool(achieved),
        "reference": "official_uri_fixed_quarter_reference",
        "exploratory_test_rows_scored": 0,
    })
    write_json(OUT / "integrity_audit.json", {
        "patients": len(patients),
        "kidneys": len(kidneys),
        "images": len(images),
        "prediction_rows": len(prediction_frame),
        "development_groups": len(groups),
        "exploratory_test_rows_scored": 0,
        "finite_scores": bool(np.isfinite(prediction_frame.score).all()),
        "unit_interval_scores": bool(prediction_frame.score.between(0.0, 1.0).all()),
    })
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    run()
