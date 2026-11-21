from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "CYCH_random_validation_20260913"
OUT = ROOT / "CYCH_literature_spatial_focality_development_20260914"
MANIFEST_PATH = SOURCE / "image_manifest.csv"
ASSIGNMENTS_PATH = (
    ROOT
    / "CYCH_directional_robustness_selection_20260914"
    / "random_patient_assignments_deidentified.csv"
)
PAIR_PREDICTIONS_PATH = (
    ROOT / "CYCH_dinov3_native_probe_20260913" / "patient_predictions_by_role.csv"
)
I3_PREDICTIONS_PATH = (
    ROOT / "CYCH_directional_robustness_selection_20260914" / "development_predictions.csv"
)
PROTOCOL_PATH = OUT / "protocol.json"
REFERENCES_PATH = OUT / "references.json"
LOCK_PATH = OUT / "protocol_lock.json"
SCRIPT_COPY = OUT / "source" / Path(__file__).name

SEEDS = list(range(20260913, 20260923))
FEATURE_SETS = {
    "SPATIAL-NATIVE": ("shape_", "native_"),
    "SPATIAL-CLAHE": ("shape_", "clahe_"),
    "SPATIAL-FUSED": ("shape_", "native_", "clahe_"),
}
K_VALUES = [5, 10, 20, 30]
C_VALUES = [0.1, 1.0, 10.0]


REFERENCES = [
    {
        "finding": "B-mode findings of acute pyelonephritis",
        "citation": "Sim, Ultrasonography, 2018",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6177687/",
        "use_here": (
            "Motivates measurements of heterogeneous echogenicity, focal abnormalities, "
            "renal contour change, and loss of corticomedullary differentiation."
        ),
    },
    {
        "finding": "Focal echogenic lesions and cortical bulging in complicated APN",
        "citation": "An update on pyelonephritis: role of contrast-enhanced ultrasound",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10247636/",
        "use_here": (
            "Motivates label-free focal bright/dark component and contour descriptors. "
            "The present data are B-mode only and cannot reproduce perfusion findings."
        ),
    },
    {
        "method": "Contour-derived renal structure combined with deep learning",
        "citation": "Combined application of deep learning and conventional computer vision, 2025",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12457927/",
        "use_here": (
            "Supports combining kidney-contour structure with image representation. The "
            "published work used manually delineated sinus contours and a CKD endpoint; "
            "this experiment does not claim its geometry-derived zones are cortex or sinus."
        ),
    },
    {
        "method": "Anatomical-level ultrasound segmentation and radiomics",
        "citation": "Auto-segmentation ultrasound-based radiomics for diabetic kidney disease, 2022",
        "url": "https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2022.876967/full",
        "use_here": (
            "Supports the value of separating renal parenchyma and sinus before radiomics, "
            "while underscoring that clinical anatomy masks are required for anatomical claims."
        ),
    },
    {
        "method": "Deep and local texture feature fusion in renal ultrasound",
        "citation": "Deep learning radiomics based on ultrasound images for CKD, 2024",
        "doi": "10.1111/nep.14376",
        "url": "https://onlinelibrary.wiley.com/doi/10.1111/nep.14376",
        "use_here": "Motivates combining deep features with local GLCM and gradient descriptors.",
    },
]


PROTOCOL = {
    "title": "Development-only literature-guided spatial and focality feature experiment",
    "status": "locked before feature extraction and model fitting",
    "cohort": {
        "patients": 98,
        "images": 349,
        "unit_of_split_and_evaluation": "patient",
        "fixed_seeds": SEEDS,
        "per_seed": {"train": 58, "development_validation": 20, "unused_test": 20},
    },
    "hard_constraints": [
        "Every image must use the existing kidney segmentation mask.",
        "All image features and geometric zones are created without outcome labels.",
        "No geometry-derived zone is called cortex, medulla, sinus, scar, or inflammatory focus.",
        "Classifier fitting and hyperparameter selection use the 58 outer-training patients only.",
        "Only fixed development-validation rows are scored; test rows are not loaded or evaluated.",
        "No result is described as cross-validation performance or independent confirmation.",
    ],
    "preprocessing": {
        "native": "within-mask 1st/99th percentile clipping followed by [0,1] scaling",
        "clahe": "fixed CLAHE with clipLimit 2.0 and 8x8 tiles",
        "coordinates": (
            "Mask-pixel coordinates projected onto the kidney mask principal axes and "
            "normalized by the maximum absolute in-mask coordinate."
        ),
        "radial_depth": "Euclidean distance to the mask boundary divided by its in-mask maximum",
    },
    "features": {
        "zones": [
            "whole kidney mask",
            "boundary-depth <=0.25",
            "boundary-depth 0.25-0.55",
            "boundary-depth >0.55",
            "absolute long-axis coordinate >0.55 (two poles combined)",
            "absolute long-axis coordinate <=0.55 (mid section)",
        ],
        "intensity": "quantiles, dispersion, skewness, kurtosis, entropy, and bright/dark fractions",
        "local_structure": "Sobel magnitude, absolute Laplacian, and 7x7 local standard deviation",
        "texture": (
            "Masked 16-level co-occurrence matrices in four directions, summarized by "
            "contrast, homogeneity, energy, entropy, and correlation means and ranges"
        ),
        "focality": (
            "Connected-component count and largest-component fraction for within-kidney "
            "intensity values below the 10th or above the 90th percentile"
        ),
        "shape": "scale-free aspect ratio, extent, solidity, circularity, eccentricity, and roughness",
        "interpretation": (
            "Literature-guided engineered measurements. They are reproducible image statistics, "
            "not validated anatomical compartments or lesion annotations."
        ),
    },
    "patient_aggregation": (
        "Per-feature patient mean, SD, minimum, maximum, and separate LL/LT/RL/RT means, "
        "with view-presence indicators."
    ),
    "model": {
        "classifier": "RBF SVC with balanced class weights and probability output",
        "pipeline": "median imputation, standardization, ANOVA F selection, RBF SVC",
        "inner_selection": "four-fold stratified CV within each outer-training set",
        "inner_objective": "mean of AUROC and average precision",
        "feature_counts": K_VALUES,
        "C": C_VALUES,
        "gamma": "scale",
    },
    "registered_ensembles": {
        "pair_only": ["PAIR75+25%SPATIAL", "PAIR50+50%SPATIAL"],
        "pair_i3_spatial": [
            "PAIR60+20%I3+20%SPATIAL",
            "PAIR50+25%I3+25%SPATIAL",
        ],
        "weights_apply_to_each_of": list(FEATURE_SETS),
    },
    "candidate_gate_vs_fixed_pair": {
        "mean_auroc_gain_at_least": 0.03,
        "mean_average_precision_gain_at_least": 0.02,
        "seeds_with_both_metrics_improved_at_least": 6,
        "action": "A separate pipeline lock is required before any test evaluation.",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def lock_protocol() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SCRIPT_COPY.parent.mkdir(parents=True, exist_ok=True)
    write_json(PROTOCOL_PATH, PROTOCOL)
    write_json(REFERENCES_PATH, REFERENCES)
    shutil.copy2(Path(__file__), SCRIPT_COPY)
    write_json(
        LOCK_PATH,
        {
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "references_sha256": sha256(REFERENCES_PATH),
            "script_sha256": sha256(Path(__file__)),
            "manifest_sha256": sha256(MANIFEST_PATH),
            "assignments_sha256": sha256(ASSIGNMENTS_PATH),
            "fixed_pair_predictions_sha256": sha256(PAIR_PREDICTIONS_PATH),
            "i3_predictions_sha256": sha256(I3_PREDICTIONS_PATH),
        },
    )
    print(f"Locked protocol: {LOCK_PATH}", flush=True)


def verify_lock() -> None:
    if not LOCK_PATH.exists():
        raise RuntimeError("Run --lock-only before feature extraction or fitting")
    expected = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    paths = {
        "protocol_sha256": PROTOCOL_PATH,
        "references_sha256": REFERENCES_PATH,
        "script_sha256": Path(__file__),
        "manifest_sha256": MANIFEST_PATH,
        "assignments_sha256": ASSIGNMENTS_PATH,
        "fixed_pair_predictions_sha256": PAIR_PREDICTIONS_PATH,
        "i3_predictions_sha256": I3_PREDICTIONS_PATH,
    }
    changed = [key for key, path in paths.items() if sha256(path) != expected[key]]
    if changed:
        raise RuntimeError(f"Locked inputs changed: {changed}")


def crop_to_mask(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("Empty kidney mask")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def robust_channel(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = image[mask].astype(np.float32)
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        high = low + 1.0
    channel = np.clip(image.astype(np.float32), low, high)
    channel = ((channel - low) / (high - low)).astype(np.float32, copy=False)
    channel[~mask] = float(np.median(channel[mask]))
    return channel


def clahe_channel(native: np.ndarray, mask: np.ndarray) -> np.ndarray:
    work = np.round(np.clip(native, 0, 1) * 255).astype(np.uint8)
    work[~mask] = int(np.median(work[mask]))
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(work)
    channel = enhanced.astype(np.float32) / 255.0
    channel[~mask] = float(np.median(channel[mask]))
    return channel


def geometry_maps(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    ys, xs = np.where(mask)
    points = np.column_stack([xs, ys]).astype(np.float64)
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    minor = np.array([-major[1], major[0]])
    u_values = centered @ major
    v_values = centered @ minor
    u_scale = max(float(np.max(np.abs(u_values))), 1.0)
    v_scale = max(float(np.max(np.abs(v_values))), 1.0)
    u_values /= u_scale
    v_values /= v_scale
    u_map = np.zeros(mask.shape, dtype=np.float32)
    v_map = np.zeros(mask.shape, dtype=np.float32)
    u_map[ys, xs] = u_values
    v_map[ys, xs] = v_values
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    distance /= max(float(distance[mask].max()), 1e-8)
    zones = {
        "all": mask,
        "boundary": mask & (distance <= 0.25),
        "middle": mask & (distance > 0.25) & (distance <= 0.55),
        "core": mask & (distance > 0.55),
        "poles": mask & (np.abs(u_map) > 0.55),
        "midlong": mask & (np.abs(u_map) <= 0.55),
    }
    return u_map, distance.astype(np.float32), zones


def entropy(values: np.ndarray, bins: int = 32) -> float:
    histogram = np.histogram(values, bins=bins, range=(0.0, 1.0))[0].astype(np.float64)
    histogram = histogram[histogram > 0]
    if histogram.size <= 1:
        return 0.0
    probabilities = histogram / histogram.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def distribution_features(values: np.ndarray) -> dict[str, float]:
    if values.size < 8:
        return {key: np.nan for key in ["mean", "std", "q10", "q25", "median", "q75", "q90", "iqr", "skew", "kurtosis", "entropy"]}
    mean = float(values.mean())
    std = float(values.std())
    q10, q25, median, q75, q90 = np.percentile(values, [10, 25, 50, 75, 90])
    centered = values - mean
    if std > 1e-8:
        skew = float(np.mean(centered**3) / std**3)
        kurtosis = float(np.mean(centered**4) / std**4 - 3.0)
    else:
        skew = 0.0
        kurtosis = 0.0
    return {
        "mean": mean,
        "std": std,
        "q10": float(q10),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "q90": float(q90),
        "iqr": float(q75 - q25),
        "skew": skew,
        "kurtosis": kurtosis,
        "entropy": entropy(values),
    }


def masked_glcm(channel: np.ndarray, region: np.ndarray) -> dict[str, float]:
    quantized = np.minimum((channel * 16).astype(np.int16), 15)
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    records: dict[str, list[float]] = {
        "contrast": [],
        "homogeneity": [],
        "energy": [],
        "entropy": [],
        "correlation": [],
    }
    levels = np.arange(16, dtype=np.float64)
    ii, jj = np.meshgrid(levels, levels, indexing="ij")
    for dy, dx in directions:
        y0a, y1a = max(0, -dy), region.shape[0] - max(0, dy)
        x0a, x1a = max(0, -dx), region.shape[1] - max(0, dx)
        y0b, y1b = max(0, dy), region.shape[0] - max(0, -dy)
        x0b, x1b = max(0, dx), region.shape[1] - max(0, -dx)
        valid = region[y0a:y1a, x0a:x1a] & region[y0b:y1b, x0b:x1b]
        if int(valid.sum()) < 32:
            for values in records.values():
                values.append(np.nan)
            continue
        first = quantized[y0a:y1a, x0a:x1a][valid]
        second = quantized[y0b:y1b, x0b:x1b][valid]
        matrix = np.zeros((16, 16), dtype=np.float64)
        np.add.at(matrix, (first, second), 1)
        np.add.at(matrix, (second, first), 1)
        matrix /= matrix.sum()
        records["contrast"].append(float(np.sum(matrix * (ii - jj) ** 2)))
        records["homogeneity"].append(float(np.sum(matrix / (1.0 + (ii - jj) ** 2))))
        records["energy"].append(float(np.sqrt(np.sum(matrix**2))))
        positive = matrix[matrix > 0]
        records["entropy"].append(float(-(positive * np.log2(positive)).sum()))
        pi = matrix.sum(axis=1)
        pj = matrix.sum(axis=0)
        mi = float(np.sum(levels * pi))
        mj = float(np.sum(levels * pj))
        si = float(np.sqrt(np.sum((levels - mi) ** 2 * pi)))
        sj = float(np.sqrt(np.sum((levels - mj) ** 2 * pj)))
        correlation = float(np.sum((ii - mi) * (jj - mj) * matrix) / (si * sj + 1e-12))
        records["correlation"].append(correlation)
    output: dict[str, float] = {}
    for name, values in records.items():
        array = np.asarray(values, dtype=float)
        output[f"glcm_{name}_mean"] = float(np.nanmean(array))
        output[f"glcm_{name}_range"] = float(np.nanmax(array) - np.nanmin(array))
    return output


def component_features(binary: np.ndarray, mask_pixels: int) -> tuple[float, float]:
    binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0, dtype=int)
    retained = areas[areas >= 4]
    component_count = float(len(retained))
    largest_fraction = float(retained.max() / max(mask_pixels, 1)) if retained.size else 0.0
    return component_count, largest_fraction


def shape_features(mask: np.ndarray) -> dict[str, float]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    area = max(float(cv2.contourArea(contour)), 1.0)
    perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    hull_perimeter = max(float(cv2.arcLength(hull, True)), 1.0)
    x, y, width, height = cv2.boundingRect(contour)
    points = contour[:, 0, :].astype(np.float64)
    covariance = np.cov(points, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1e-8)
    return {
        "shape_aspect_ratio": float(max(width, height) / max(min(width, height), 1)),
        "shape_extent": float(area / max(width * height, 1)),
        "shape_solidity": float(area / hull_area),
        "shape_circularity": float(4.0 * np.pi * area / perimeter**2),
        "shape_eccentricity": float(np.sqrt(1.0 - eigenvalues.min() / eigenvalues.max())),
        "shape_roughness": float(perimeter / hull_perimeter),
    }


def channel_features(channel: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float]:
    u_map, distance, zones = geometry_maps(mask)
    sobel_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    laplacian = np.abs(cv2.Laplacian(channel, cv2.CV_32F, ksize=3))
    local_mean = cv2.blur(channel, (7, 7))
    local_square_mean = cv2.blur(channel**2, (7, 7))
    local_std = np.sqrt(np.maximum(local_square_mean - local_mean**2, 0.0))
    global_values = channel[mask]
    low, high = np.percentile(global_values, [10, 90])
    output: dict[str, float] = {}
    zone_summaries: dict[str, dict[str, float]] = {}
    for zone_name, region in zones.items():
        summary = distribution_features(channel[region])
        zone_summaries[zone_name] = summary
        summary["pixel_fraction"] = float(region.sum() / max(mask.sum(), 1))
        summary["dark_fraction"] = float(np.mean(channel[region] <= low)) if region.any() else np.nan
        summary["bright_fraction"] = float(np.mean(channel[region] >= high)) if region.any() else np.nan
        for map_name, values in [("gradient", gradient), ("laplacian", laplacian), ("local_std", local_std)]:
            selected = values[region]
            summary[f"{map_name}_mean"] = float(np.mean(selected)) if selected.size else np.nan
            summary[f"{map_name}_q90"] = float(np.percentile(selected, 90)) if selected.size else np.nan
        if zone_name in {"all", "boundary", "middle", "core"}:
            summary.update(masked_glcm(channel, region))
        output.update({f"{prefix}_{zone_name}_{key}": value for key, value in summary.items()})
    for inner_name, outer_name in [("core", "boundary"), ("middle", "boundary"), ("poles", "midlong")]:
        for metric in ["mean", "std", "iqr", "entropy", "gradient_mean", "local_std_mean", "bright_fraction", "dark_fraction"]:
            first = zone_summaries[inner_name].get(metric, np.nan)
            second = zone_summaries[outer_name].get(metric, np.nan)
            output[f"{prefix}_{inner_name}_minus_{outer_name}_{metric}"] = float(first - second)
            output[f"{prefix}_{inner_name}_vs_{outer_name}_{metric}"] = float(
                (first - second) / (abs(first) + abs(second) + 1e-8)
            )
    if np.std(distance[mask]) > 1e-8:
        output[f"{prefix}_radial_intensity_correlation"] = float(np.corrcoef(distance[mask], channel[mask])[0, 1])
        output[f"{prefix}_radial_gradient_correlation"] = float(np.corrcoef(distance[mask], gradient[mask])[0, 1])
    else:
        output[f"{prefix}_radial_intensity_correlation"] = 0.0
        output[f"{prefix}_radial_gradient_correlation"] = 0.0
    negative_pole = mask & (u_map < -0.55)
    positive_pole = mask & (u_map > 0.55)
    for metric_name, values in [("intensity", channel), ("gradient", gradient), ("local_std", local_std)]:
        first = float(np.mean(values[negative_pole])) if negative_pole.any() else np.nan
        second = float(np.mean(values[positive_pole])) if positive_pole.any() else np.nan
        output[f"{prefix}_pole_absdiff_{metric_name}"] = float(abs(first - second))
    for name, binary in [("dark", mask & (channel <= low)), ("bright", mask & (channel >= high))]:
        count, largest = component_features(binary, int(mask.sum()))
        output[f"{prefix}_{name}_component_count"] = count
        output[f"{prefix}_{name}_largest_component_fraction"] = largest
    return output


def extract_image_features() -> pd.DataFrame:
    manifest = pd.read_csv(
        MANIFEST_PATH,
        usecols=["image_id", "patient_alias", "view"],
        dtype=str,
    )
    rows: list[dict[str, object]] = []
    for position, row in enumerate(manifest.itertuples(index=False), start=1):
        image = cv2.imread(str(SOURCE / "native" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE)
        mask_image = cv2.imread(str(SOURCE / "masks" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask_image is None:
            raise FileNotFoundError(f"Missing image or mask for {row.image_id}")
        mask = mask_image > 0
        image, mask = crop_to_mask(image, mask)
        native = robust_channel(image, mask)
        clahe = clahe_channel(native, mask)
        features: dict[str, object] = {
            "image_id": row.image_id,
            "patient_alias": row.patient_alias,
            "view": row.view,
        }
        features.update(shape_features(mask))
        features.update(channel_features(native, mask, "native"))
        features.update(channel_features(clahe, mask, "clahe"))
        rows.append(features)
        if position % 50 == 0 or position == len(manifest):
            print(f"Extracted spatial features {position}/{len(manifest)}", flush=True)
    result = pd.DataFrame(rows)
    if result.shape[0] != 349 or result["patient_alias"].nunique() != 98:
        raise RuntimeError(f"Unexpected image feature table: {result.shape}")
    return result


def aggregate_patients(image_features: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        column
        for column in image_features.columns
        if column.startswith(("shape_", "native_", "clahe_"))
    ]
    rows: list[dict[str, object]] = []
    for patient_alias, group in image_features.groupby("patient_alias", sort=True):
        row: dict[str, object] = {"patient_alias": patient_alias}
        values = group[feature_columns]
        for statistic, series in [
            ("all_mean", values.mean()),
            ("all_std", values.std(ddof=0)),
            ("all_min", values.min()),
            ("all_max", values.max()),
        ]:
            row.update({f"{statistic}__{key}": float(value) for key, value in series.items()})
        for view in ("LL", "LT", "RL", "RT"):
            subset = group.loc[group["view"] == view, feature_columns]
            row[f"view_present__{view}"] = int(not subset.empty)
            series = subset.mean() if not subset.empty else pd.Series(np.nan, index=feature_columns)
            row.update({f"{view}_mean__{key}": float(value) for key, value in series.items()})
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.shape[0] != 98:
        raise RuntimeError(f"Expected 98 patients, found {result.shape[0]}")
    return result


def load_development_predictions(path: Path, group: str, output_name: str) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["role"] != "development_validation" or row["group"] != group:
                continue
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "patient_alias": row["patient_alias"],
                    "label": int(row["label"]),
                    output_name: float(row["score"]),
                }
            )
    result = pd.DataFrame(rows)
    if result.shape[0] != 200 or sorted(result["seed"].unique()) != SEEDS:
        raise RuntimeError(f"Unexpected {group} development predictions: {result.shape}")
    return result


def build_model(k: int, c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k)),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=c_value,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=seed,
                ),
            ),
        ]
    )


def select_inner_model(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[int, float, float]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    records = []
    for k in K_VALUES:
        for c_value in C_VALUES:
            fold_scores = []
            for train_index, validation_index in splitter.split(x, y):
                model = build_model(k, c_value, seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x[train_index], y[train_index])
                score = model.predict_proba(x[validation_index])[:, 1]
                fold_scores.append(
                    (
                        roc_auc_score(y[validation_index], score)
                        + average_precision_score(y[validation_index], score)
                    )
                    / 2.0
                )
            records.append((float(np.mean(fold_scores)), -k, -c_value, k, c_value))
    best = max(records)
    return int(best[3]), float(best[4]), float(best[0])


def metric_row(seed: int, group: str, y: np.ndarray, score: np.ndarray) -> dict[str, object]:
    return {
        "seed": seed,
        "group": group,
        "role": "development_validation",
        "n": len(y),
        "auroc": roc_auc_score(y, score),
        "average_precision": average_precision_score(y, score),
    }


def run_development(patient_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments = pd.read_csv(
        ASSIGNMENTS_PATH,
        usecols=["seed", "role", "patient_alias", "label"],
        dtype={"patient_alias": str},
    )
    assignments = assignments.loc[assignments["role"].isin(["train", "validation"])].copy()
    pair = load_development_predictions(PAIR_PREDICTIONS_PATH, "PAIR", "pair_score")
    i3 = load_development_predictions(I3_PREDICTIONS_PATH, "I3-ALL", "i3_score")
    all_feature_columns = [column for column in patient_features if column != "patient_alias"]
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        split = assignments.loc[assignments["seed"] == seed]
        train = split.loc[split["role"] == "train", ["patient_alias", "label"]]
        development = split.loc[split["role"] == "validation", ["patient_alias", "label"]]
        if len(train) != 58 or len(development) != 20:
            raise RuntimeError(f"Unexpected split size for seed {seed}")
        train_frame = train.merge(patient_features, on="patient_alias", validate="one_to_one")
        development_frame = development.merge(patient_features, on="patient_alias", validate="one_to_one")
        baseline = development.merge(
            pair.loc[pair["seed"] == seed],
            on=["patient_alias", "label"],
            validate="one_to_one",
        ).merge(
            i3.loc[i3["seed"] == seed],
            on=["seed", "patient_alias", "label"],
            validate="one_to_one",
        )
        if list(development_frame["patient_alias"]) != list(baseline["patient_alias"]):
            raise RuntimeError(f"Patient order mismatch for seed {seed}")
        y_train = train_frame["label"].to_numpy(dtype=int)
        y_development = development_frame["label"].to_numpy(dtype=int)
        pair_score = baseline["pair_score"].to_numpy(dtype=float)
        i3_score = baseline["i3_score"].to_numpy(dtype=float)
        metric_rows.append(metric_row(seed, "PAIR", y_development, pair_score))
        metric_rows.append(
            metric_row(seed, "PAIR+I3-REFERENCE", y_development, (2 * pair_score + i3_score) / 3)
        )
        for feature_set, prefixes in FEATURE_SETS.items():
            columns = [
                column
                for column in all_feature_columns
                if column.startswith("view_present__")
                or any(f"__{prefix}" in column for prefix in prefixes)
            ]
            x_train = train_frame[columns].to_numpy(dtype=float)
            x_development = development_frame[columns].to_numpy(dtype=float)
            k, c_value, inner_score = select_inner_model(x_train, y_train, seed)
            model = build_model(k, c_value, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_train, y_train)
            score = model.predict_proba(x_development)[:, 1]
            metric_rows.append(metric_row(seed, feature_set, y_development, score))
            tuning_rows.append(
                {
                    "seed": seed,
                    "feature_set": feature_set,
                    "candidate_variables": len(columns),
                    "selected_k": k,
                    "selected_C": c_value,
                    "inner_cv_objective": inner_score,
                    "test_accessed": False,
                }
            )
            ensembles = {
                f"PAIR75+25%{feature_set}": 0.75 * pair_score + 0.25 * score,
                f"PAIR50+50%{feature_set}": 0.50 * pair_score + 0.50 * score,
                f"PAIR60+20%I3+20%{feature_set}": 0.60 * pair_score + 0.20 * i3_score + 0.20 * score,
                f"PAIR50+25%I3+25%{feature_set}": 0.50 * pair_score + 0.25 * i3_score + 0.25 * score,
            }
            score_groups = {feature_set: score, **ensembles}
            for group_name, group_score in score_groups.items():
                if group_name != feature_set:
                    metric_rows.append(metric_row(seed, group_name, y_development, group_score))
                for patient_alias, label, value in zip(
                    development_frame["patient_alias"], y_development, group_score
                ):
                    prediction_rows.append(
                        {
                            "seed": seed,
                            "role": "development_validation",
                            "group": group_name,
                            "patient_alias": patient_alias,
                            "label": int(label),
                            "score": float(value),
                        }
                    )
        print(f"Completed development seed {seed}", flush=True)
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), pd.DataFrame(tuning_rows)


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        metrics.groupby("group", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_auroc=("auroc", "mean"),
            sd_auroc=("auroc", "std"),
            min_auroc=("auroc", "min"),
            max_auroc=("auroc", "max"),
            mean_average_precision=("average_precision", "mean"),
            sd_average_precision=("average_precision", "std"),
            min_average_precision=("average_precision", "min"),
            max_average_precision=("average_precision", "max"),
        )
        .sort_values(["mean_auroc", "mean_average_precision"], ascending=False)
    )
    baseline = metrics.loc[metrics["group"] == "PAIR", ["seed", "auroc", "average_precision"]].rename(
        columns={"auroc": "pair_auroc", "average_precision": "pair_average_precision"}
    )
    candidates = metrics.loc[
        metrics["group"].str.startswith("PAIR")
        & ~metrics["group"].isin(["PAIR", "PAIR+I3-REFERENCE"])
    ]
    gate_rows = []
    for group, frame in candidates.groupby("group"):
        comparison = frame.merge(baseline, on="seed", validate="one_to_one")
        delta_auc = comparison["auroc"] - comparison["pair_auroc"]
        delta_ap = comparison["average_precision"] - comparison["pair_average_precision"]
        simultaneous = int(((delta_auc > 0) & (delta_ap > 0)).sum())
        mean_auc_gain = float(delta_auc.mean())
        mean_ap_gain = float(delta_ap.mean())
        gate_rows.append(
            {
                "group": group,
                "mean_auroc_gain": mean_auc_gain,
                "mean_average_precision_gain": mean_ap_gain,
                "seeds_both_improved": simultaneous,
                "passes_registered_gate": bool(
                    mean_auc_gain >= 0.03 and mean_ap_gain >= 0.02 and simultaneous >= 6
                ),
                "exploratory_test_authorized": False,
            }
        )
    gate = pd.DataFrame(gate_rows).sort_values(
        ["passes_registered_gate", "mean_auroc_gain", "mean_average_precision_gain"],
        ascending=False,
    )
    return summary, gate


def write_readme(summary: pd.DataFrame, gate: pd.DataFrame, patient_features: pd.DataFrame) -> None:
    baseline = summary.loc[summary["group"] == "PAIR"].iloc[0]
    i3_reference = summary.loc[summary["group"] == "PAIR+I3-REFERENCE"].iloc[0]
    standalone = summary.loc[summary["group"].isin(FEATURE_SETS)].iloc[0]
    ensemble = summary.loc[
        summary["group"].str.startswith("PAIR")
        & ~summary["group"].isin(["PAIR", "PAIR+I3-REFERENCE"])
    ].iloc[0]
    passed = gate.loc[gate["passes_registered_gate"]]
    decision = (
        f"Passing registered candidates: {', '.join(passed['group'])}. A separate lock is required before test access."
        if not passed.empty
        else "No registered candidate passed; the exploratory internal test remains unopened."
    )
    text = f"""# Literature-guided spatial and focality development experiment

## Scope

This label-free feature screen quantifies spatially localized B-mode appearance inside the
existing kidney masks. Geometry-derived boundary, middle, core, pole, and mid-kidney zones
are normalized image coordinates; they are not clinical cortex, medulla, sinus, scar, or
inflammatory-focus annotations.

Only the ten fixed 20-patient development-validation sets were evaluated. Model selection
used the corresponding 58 training patients. No `test` row or exploratory-test prediction
was loaded.

## Features

- {patient_features.shape[1] - 1:,} patient-level variables from 349 segmented images.
- Native robust intensity and fixed CLAHE channels.
- Zone-wise intensity, gradient, Laplacian, local-variance, and masked GLCM descriptors.
- Scale-free contour shape and focal bright/dark connected-component descriptors.
- View-aware LL/LT/RL/RT aggregation.

## Development results

- Fixed pair: AUROC {baseline.mean_auroc:.3f}, AP {baseline.mean_average_precision:.3f}.
- Existing pair plus I3 reference: AUROC {i3_reference.mean_auroc:.3f}, AP {i3_reference.mean_average_precision:.3f}.
- Best standalone spatial set: {standalone.group}, AUROC {standalone.mean_auroc:.3f}, AP {standalone.mean_average_precision:.3f}.
- Best registered spatial ensemble: {ensemble.group}, AUROC {ensemble.mean_auroc:.3f}, AP {ensemble.mean_average_precision:.3f}.

Registered gate versus the fixed pair: mean AUROC gain >=0.03, mean AP gain >=0.02,
and both metrics improve in at least 6/10 draws. {decision}

## Interpretation boundary

This is a literature-guided adaptation for the current B-mode images, not an exact
replication of the cited CKD or APN studies. The 98 patients were previously explored;
development performance is not independent confirmation and is not reported as k-fold.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    if args.lock_only:
        lock_protocol()
        return
    verify_lock()
    image_path = OUT / "image_features_deidentified.csv"
    patient_path = OUT / "patient_features_deidentified.csv"
    if args.reuse_features and image_path.exists() and patient_path.exists():
        image_features = pd.read_csv(image_path)
        patient_features = pd.read_csv(patient_path)
    else:
        image_features = extract_image_features()
        patient_features = aggregate_patients(image_features)
        image_features.to_csv(image_path, index=False)
        patient_features.to_csv(patient_path, index=False)
    metrics, predictions, tuning = run_development(patient_features)
    summary, gate = summarize(metrics)
    metrics.to_csv(OUT / "metrics_per_seed.csv", index=False)
    predictions.to_csv(OUT / "development_predictions.csv", index=False)
    tuning.to_csv(OUT / "selected_hyperparameters.csv", index=False)
    summary.to_csv(OUT / "summary_development.csv", index=False)
    gate.to_csv(OUT / "candidate_gate.csv", index=False)
    write_json(
        OUT / "run_audit.json",
        {
            "images": int(len(image_features)),
            "patients": int(patient_features["patient_alias"].nunique()),
            "roles_evaluated": ["train", "development_validation"],
            "test_role_evaluated": False,
            "exploratory_test_predictions_loaded": False,
            "passing_candidates": gate.loc[gate["passes_registered_gate"], "group"].tolist(),
            "output_hashes": {
                path.name: sha256(path)
                for path in [
                    image_path,
                    patient_path,
                    OUT / "metrics_per_seed.csv",
                    OUT / "development_predictions.csv",
                    OUT / "selected_hyperparameters.csv",
                    OUT / "summary_development.csv",
                    OUT / "candidate_gate.csv",
                ]
            },
        },
    )
    write_readme(summary, gate, patient_features)
    print(summary.to_string(index=False))
    print("\nCandidate gate:")
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
