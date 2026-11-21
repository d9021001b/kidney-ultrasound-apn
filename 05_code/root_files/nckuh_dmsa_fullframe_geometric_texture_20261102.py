"""Stable geometric-region texture descriptors on clean NCKUH renal ROIs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy import ndimage
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullframe_clean_probe_20261102 as F
import nckuh_dmsa_fullframe_deep_probe_20261102 as D
import nckuh_dmsa_fullframe_hierarchical_probe_20261102 as H
import nckuh_dmsa_fullframe_relative_geometry_probe_20261102 as G
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullframe_geometric_texture_20261102"
REFERENCE = "deep_concat_plus_clean_radiomics"
POWERSET_REFERENCE = "four_state_powerset_plus_deep"
GEOMETRY_REFERENCE = "relative_frame_geometry_plus_deep_radiomics"
GROUPS = (
    "clean_geometric_texture_anova64_lr",
    "clean_geometric_texture_relative_geometry_anova64_lr",
    "clean_geometric_texture_plus_deep_radiomics",
    "clean_geometric_texture_geometry_plus_deep_radiomics",
    "early_concat_deep_radiomics_geometric_texture_geometry_anova128_lr",
    "fixed_powerset_geometry_texture_three_source_fusion",
)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered full-frame geometric-texture protocol")
    F.require_lock(); D.require_lock(); G.require_lock(); H.require_lock()
    inputs = [
        F.OUT / "patient_assignments.csv",
        F.OUT / "deidentified_image_manifest.csv",
        F.OUT / "radiomics_clean_fullframe.csv",
        F.OUT / "archive_extraction_audit.json",
        D.OUT / "radimagenet_gray.npy",
        D.OUT / "radimagenet_color.npy",
        D.OUT / "usf_gray.npy",
        D.OUT / "development_predictions.csv",
        G.OUT / "relative_geometry.csv",
        G.OUT / "development_predictions.csv",
        H.OUT / "development_predictions.csv",
        Path(__file__),
    ]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same availability-defined 111 patients, 212 kidneys, 418 clean full-frame-derived views and exact 70/20/21 patient roles",
        "motivation": "The completed error decomposition localized the largest AUROC deficit to bilateral APN versus bilateral normal kidneys. These cases require absolute parenchymal and size evidence rather than contralateral ranking",
        "regions": "Three normalized distance-to-boundary bands plus sign-invariant principal-long-axis end mean, center and end absolute-difference summaries. Regions are geometric and are not called cortex, medulla or lesions",
        "texture": "P1-P99 renal grayscale; intensity distribution, gradient magnitude, LoG scales 1/2, uniform LBP and four-orientation Gabor magnitude. Pixels outside the contour are nearest-renal-pixel extended before spatial filters to suppress artificial black-mask boundary responses",
        "representation": "Preserve kidney side and longitudinal/transverse views, view difference, image mean/SD and availability; orient each scored kidney as own kidney followed by contralateral context",
        "models": "Training-only median imputation, scaling, variance filtering, fixed ANOVA64 or ANOVA128 and balanced LR C=.01",
        "fusion": "Only preregistered equal-probability averages: texture with deep-radiomics; texture-geometry with deep-radiomics; and one three-source average of completed powerset-deep, completed geometry-deep and the new texture branch. No development-selected weight",
        "references": [
            "https://pubmed.ncbi.nlm.nih.gov/29382187/",
            "https://pubmed.ncbi.nlm.nih.gov/12069464/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC8305820/",
        ],
        "prohibited": "No DMSA code, diagnosis suffix, age, sex, OCR text, development-selected feature count/weight/threshold, or exploratory-test scoring",
        "target": "Development AUROC>0.8 and AP>0.8; independent locked confirmation is still required",
        "groups": list(GROUPS),
        "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "source_sha256": digest(__file__),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs},
    })
    print("Registered clean full-frame geometric-region texture probe", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(Path(name)) == expected, name


def distribution_features(values: np.ndarray, bins: int = 16) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.full(8 + bins, np.nan, dtype=np.float32)
    histogram = np.histogram(values, bins=bins, range=(0, 255))[0].astype(float)
    histogram /= max(histogram.sum(), 1)
    nonzero = histogram[histogram > 0]
    entropy = -(nonzero * np.log2(nonzero)).sum()
    statistics = [values.mean(), values.std(), *np.percentile(values, [10, 25, 50, 75, 90]), entropy]
    return np.r_[statistics, histogram].astype(np.float32)


def field_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.full(4, np.nan, dtype=np.float32)
    return np.r_[values.mean(), values.std(), np.percentile(values, [50, 90])].astype(np.float32)


def feature_names() -> list[str]:
    distribution = ["mean", "sd", "p10", "p25", "p50", "p75", "p90", "entropy"]
    distribution += [f"hist_{index:02d}" for index in range(16)]
    field = ["mean", "sd", "p50", "p90"]
    names = ["mask_fill", "bbox_aspect", "perimeter_sqrt_area", "circularity",
             "solidity", "axis_ratio", "major_variance_per_area", "minor_variance_per_area"]
    for band in ("outer", "middle", "inner"):
        names.extend(f"distance_{band}_intensity_{name}" for name in distribution)
        names.extend(f"distance_{band}_gradient_{name}" for name in field)
        for scale in (1, 2):
            names.extend(f"distance_{band}_log{scale}_{name}" for name in field)
        names.extend(f"distance_{band}_lbp_{index:02d}" for index in range(10))
        for angle in (0, 45, 90, 135):
            names.extend(f"distance_{band}_gabor{angle}_{name}" for name in ("mean", "sd"))
    longitudinal_base = [f"intensity_{name}" for name in distribution]
    longitudinal_base += [f"gradient_{name}" for name in field]
    longitudinal_base += [f"lbp_{index:02d}" for index in range(10)]
    for summary in ("ends_mean", "center", "ends_absdiff"):
        names.extend(f"longaxis_{summary}_{name}" for name in longitudinal_base)
    assert len(names) == 284
    return names


def _region_texture(image: np.ndarray, region: np.ndarray, gradient: np.ndarray,
                    logs: list[np.ndarray], lbp: np.ndarray, gabors: list[np.ndarray]) -> np.ndarray:
    parts = [distribution_features(image[region]), field_features(gradient[region])]
    parts.extend(field_features(field[region]) for field in logs)
    lbp_histogram = np.bincount(lbp[region].astype(int), minlength=10)[:10].astype(float)
    lbp_histogram /= max(lbp_histogram.sum(), 1)
    parts.append(lbp_histogram.astype(np.float32))
    parts.extend(field_features(field[region])[:2] for field in gabors)
    output = np.concatenate(parts).astype(np.float32)
    assert output.shape == (54,)
    return output


def _longitudinal_texture(image: np.ndarray, region: np.ndarray,
                          gradient: np.ndarray, lbp: np.ndarray) -> np.ndarray:
    histogram = np.bincount(lbp[region].astype(int), minlength=10)[:10].astype(float)
    histogram /= max(histogram.sum(), 1)
    output = np.r_[distribution_features(image[region]), field_features(gradient[region]), histogram]
    assert output.shape == (38,)
    return output.astype(np.float32)


def descriptor(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from skimage.feature import local_binary_pattern
    from skimage.filters import gabor

    binary = mask.astype(bool)
    if gray.ndim != 2 or gray.shape != binary.shape or binary.sum() < 1024:
        raise ValueError("Invalid clean kidney ROI")
    y, x = np.nonzero(binary)
    area = float(binary.sum())
    y0, y1, x0, x1 = y.min(), y.max() + 1, x.min(), x.max() + 1
    height, width = y1 - y0, x1 - x0
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(contour, True))
    hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
    centered = np.c_[x - x.mean(), y - y.mean()]
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    shape = np.asarray([
        area / (height * width), width / max(height, 1), perimeter / np.sqrt(area),
        4 * np.pi * area / max(perimeter ** 2, 1), area / hull_area,
        np.sqrt(eigenvalues[1] / eigenvalues[0]), eigenvalues[1] / area, eigenvalues[0] / area,
    ], dtype=np.float32)

    nearest = ndimage.distance_transform_edt(~binary, return_distances=False, return_indices=True)
    image = gray[tuple(nearest)].astype(np.float32)
    gradient = np.hypot(ndimage.sobel(image, axis=1), ndimage.sobel(image, axis=0))
    logs = [np.abs(ndimage.gaussian_laplace(image, sigma=scale)) for scale in (1, 2)]
    lbp = local_binary_pattern(image.astype(np.uint8), 8, 1, method="uniform")
    gabors = []
    for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
        real, imaginary = gabor(image / 255.0, frequency=.15, theta=theta)
        gabors.append(np.hypot(real, imaginary))

    distance = ndimage.distance_transform_edt(binary)
    normalized_distance = distance / max(float(distance.max()), 1.0)
    distance_regions = [
        binary & (normalized_distance <= 1 / 3),
        binary & (normalized_distance > 1 / 3) & (normalized_distance <= 2 / 3),
        binary & (normalized_distance > 2 / 3),
    ]
    features = [shape]
    features.extend(_region_texture(image, region, gradient, logs, lbp, gabors)
                    for region in distance_regions)

    principal_axis = eigenvectors[:, -1]
    projection = centered @ principal_axis
    cuts = np.linspace(projection.min(), projection.max(), 4)
    projected = np.full(binary.shape, np.nan, dtype=np.float32)
    projected[binary] = projection
    longitudinal_regions = []
    for index in range(3):
        upper = projected <= cuts[index + 1] if index == 2 else projected < cuts[index + 1]
        longitudinal_regions.append(binary & (projected >= cuts[index]) & upper)
    longitudinal = [_longitudinal_texture(image, region, gradient, lbp)
                    for region in longitudinal_regions]
    features.extend([(longitudinal[0] + longitudinal[2]) / 2,
                     longitudinal[1], np.abs(longitudinal[0] - longitudinal[2])])

    output = np.concatenate(features).astype(np.float32)
    assert output.shape == (284,) and np.isfinite(output).all()
    return output, shape


def extract() -> None:
    require_lock()
    if (OUT / "image_geometric_texture.npy").exists():
        raise FileExistsError("Preserve completed full-frame geometric-texture extraction")
    _, _, images = F.matched_reference()
    values, shapes = [], []
    for index, identifier in enumerate(images.image_id):
        gray = cv2.imread(str(F.OUT / "percentile" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(F.OUT / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE)
        current, shape = descriptor(gray, mask > 0)
        values.append(current); shapes.append(shape)
        if (index + 1) % 100 == 0:
            print(f"Extracted clean geometric texture {index + 1}/{len(images)}", flush=True)
    values, shapes = np.stack(values), np.stack(shapes)
    np.save(OUT / "image_geometric_texture.npy", values)
    np.save(OUT / "image_shape_control.npy", shapes)
    write_json(OUT / "feature_names.json", {"feature_names": feature_names()})
    write_json(OUT / "feature_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "images": len(images), "features_per_image": values.shape[1],
        "geometric_texture_shape": list(values.shape), "shape_control_shape": list(shapes.shape),
        "finite": bool(np.isfinite(values).all()), "label_fields_used": [],
        "outside_mask_spatial_filter_policy": "nearest renal pixel extension",
        "long_axis_sign_policy": "ends mean, center, ends absolute difference",
        "geometric_texture_sha256": digest(OUT / "image_geometric_texture.npy"),
        "shape_control_sha256": digest(OUT / "image_shape_control.npy"),
    })
    print("Extracted 418 clean full-frame geometric-region descriptors", flush=True)


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed full-frame geometric-texture predictions")
    patients, kidneys, images = F.matched_reference()
    labels = kidneys.kidney_APN.to_numpy(int)
    texture = np.load(OUT / "image_geometric_texture.npy")
    assert texture.shape == (418, 284)
    geometry_frame = pd.read_csv(G.OUT / "relative_geometry.csv").set_index("image_id").reindex(images.image_id)
    geometry = geometry_frame[[column for column in geometry_frame if column.startswith("geometry_")]].to_numpy(float)
    clean_frame = pd.read_csv(F.OUT / "radiomics_clean_fullframe.csv").set_index("image_id").reindex(images.image_id)
    clean = clean_frame[[column for column in clean_frame if not column.endswith(("_Energy", "_TotalEnergy"))]].to_numpy(float)
    deep = np.concatenate([np.load(D.OUT / name) for name in
                           ("radimagenet_gray.npy", "radimagenet_color.npy", "usf_gray.npy")], axis=1)
    oriented_texture = F.orient_kidneys(F.patient_pair_features(texture, images, patients), patients, kidneys)
    oriented_geometry = F.orient_kidneys(F.patient_pair_features(geometry, images, patients), patients, kidneys)
    oriented_clean = F.orient_kidneys(F.patient_pair_features(clean, images, patients), patients, kidneys)
    oriented_deep = F.orient_kidneys(F.patient_pair_features(deep, images, patients), patients, kidneys)
    texture_geometry = np.c_[oriented_texture, oriented_geometry]
    early = np.c_[oriented_deep, oriented_clean, oriented_texture, oriented_geometry]

    assignments = pd.read_csv(F.OUT / "patient_assignments.csv")
    deep_predictions = pd.read_csv(D.OUT / "development_predictions.csv")
    deep_reference = deep_predictions[deep_predictions.group.eq(REFERENCE)].set_index(["seed", "kidney_alias"]).score
    geometry_predictions = pd.read_csv(G.OUT / "development_predictions.csv")
    geometry_reference = geometry_predictions[geometry_predictions.group.eq(GEOMETRY_REFERENCE)].set_index(["seed", "kidney_alias"]).score
    powerset_predictions = pd.read_csv(H.OUT / "development_predictions.csv")
    powerset_reference = powerset_predictions[powerset_predictions.group.eq(POWERSET_REFERENCE)].set_index(["seed", "kidney_alias"]).score

    records, predictions = [], []
    (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in F.SEEDS:
            roles = kidneys.patient_alias.map(assignments[assignments.seed.eq(seed)].set_index("patient_alias").role)
            train, development = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
            aliases = kidneys.kidney_alias.iloc[development]
            deep_score = np.asarray([deep_reference.loc[(seed, alias)] for alias in aliases])
            geometry_score = np.asarray([geometry_reference.loc[(seed, alias)] for alias in aliases])
            powerset_score = np.asarray([powerset_reference.loc[(seed, alias)] for alias in aliases])
            scores, models = {}, {}
            texture_score, transform, model = F.fit_predict(
                oriented_texture, train, development, labels, seed, 64)
            scores[GROUPS[0]] = texture_score; models[GROUPS[0]] = (transform, model)
            combined_score, transform, model = F.fit_predict(
                texture_geometry, train, development, labels, seed, 64)
            scores[GROUPS[1]] = combined_score; models[GROUPS[1]] = (transform, model)
            scores[GROUPS[2]] = (texture_score + deep_score) / 2
            scores[GROUPS[3]] = (combined_score + deep_score) / 2
            early_score, transform, model = F.fit_predict(early, train, development, labels, seed, 128)
            scores[GROUPS[4]] = early_score; models[GROUPS[4]] = (transform, model)
            scores[GROUPS[5]] = (powerset_score + geometry_score + texture_score) / 3
            joblib.dump(models, OUT / "models" / f"{seed}_geometric_texture.joblib", compress=3)
            for group in GROUPS:
                result = metric(labels[development], scores[group])
                records.append({"seed": seed, "role": "development", "group": group,
                                "patients": 20, "kidneys": len(development), **result})
                for local, row_index in enumerate(development):
                    predictions.append({"seed": seed, "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[row_index],
                        "kidney_alias": kidneys.kidney_alias.iloc[row_index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[row_index],
                        "side": kidneys.side.iloc[row_index], "label": int(labels[row_index]),
                        "score": float(scores[group][local])})
            print(f"Completed clean geometric-texture draw {seed}", flush=True)

    metric_frame = pd.DataFrame(records)
    prediction_frame = pd.DataFrame(predictions)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap"]].mean().sort_values(["auroc", "ap"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc > .8) & (summary.ap > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {
        "groups_above_both_development_targets": achieved,
        "test_scored": False, "goal_achieved": False,
        "reason": "Development-only availability subgroup cannot complete the full objective",
    })
    print(summary.to_string(), flush=True)


def bootstrap() -> None:
    require_lock()
    frame = pd.read_csv(OUT / "development_predictions.csv")
    previous = pd.read_csv(D.OUT / "development_predictions.csv")
    reference = previous[previous.group.eq(REFERENCE)]
    ci, paired, samples = cluster_intervals(
        pd.concat([frame, reference], ignore_index=True), list(GROUPS) + [REFERENCE], F.SEEDS, REFERENCE)
    ci.to_csv(OUT / "development_confidence_intervals.csv", index=False)
    paired.to_csv(OUT / "development_paired_intervals.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    write_json(OUT / "completion_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "patients": 111, "kidneys": 212, "images": 418,
        "test_scored": False, "scored_unit": "kidney", "bootstrap_cluster": "patient",
    })
    print("Clean full-frame geometric-texture intervals completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "extract", "run", "bootstrap"])
    arguments = parser.parse_args()
    globals()[arguments.stage]()
