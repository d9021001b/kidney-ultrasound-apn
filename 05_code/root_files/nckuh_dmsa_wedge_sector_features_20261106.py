"""Rotation-invariant focal wedge and contour-bulge features for NCKUH APN."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_fullframe_20261103"
OUT = ROOT / "02_experiments/NCKUH_DMSA_wedge_sector_features_20261106"
REFERENCE = uri.OUT / "development_predictions.csv"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"
SECTORS = 12


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def robust_z(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = image[mask]
    median = float(np.median(values))
    scale = max(float(np.median(np.abs(values - median))) * 1.4826, 1e-4)
    return (image - median) / scale


def summary(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [0.0] * 7
    return [
        float(finite.mean()),
        float(finite.std()),
        float(finite.min()),
        float(finite.max()),
        float(finite.max() - finite.min()),
        float(np.quantile(finite, 0.10)),
        float(np.quantile(finite, 0.90)),
    ]


def circular_run(flags: np.ndarray) -> float:
    flags = np.asarray(flags, dtype=bool)
    if not flags.any():
        return 0.0
    if flags.all():
        return 1.0
    doubled = np.r_[flags, flags]
    best = current = 0
    for value in doubled:
        current = current + 1 if value else 0
        best = max(best, current)
    return float(min(best, len(flags)) / len(flags))


def sector_values(
    image: np.ndarray, sector: np.ndarray, region: np.ndarray
) -> np.ndarray:
    values = []
    for index in range(SECTORS):
        selected = region & (sector == index)
        values.append(float(image[selected].mean()) if selected.sum() >= 8 else np.nan)
    return np.asarray(values, dtype=np.float32)


def component_morphology(
    binary: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    angle: np.ndarray,
    longitudinal: np.ndarray,
) -> list[float]:
    selected = (binary & mask).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(selected, 8)
    if count <= 1:
        return [0.0] * 9
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas) + 1)
    component = labels == label
    area = float(component.sum())
    denominator = max(float(mask.sum()), 1.0)
    current_depth = depth[component]
    current_angle = angle[component]
    current_longitudinal = longitudinal[component]
    concentration = np.hypot(
        np.cos(current_angle).mean(), np.sin(current_angle).mean()
    )
    contours = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    contour = max(contours, key=cv2.contourArea)
    contour_area = max(float(cv2.contourArea(contour)), 1.0)
    hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
    covariance = np.cov(np.c_[np.nonzero(component)], rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    eccentricity = float(
        np.sqrt(max(1.0 - eigenvalues[-1] / max(eigenvalues[0], 1e-8), 0.0))
    )
    return [
        area / denominator,
        float(current_depth.mean()),
        float(current_depth.max() - current_depth.min()),
        float((current_depth <= 0.20).mean()),
        float(concentration),
        float(np.abs(current_longitudinal).mean()),
        contour_area / hull_area,
        eccentricity,
        float(len(contours)),
    ]


def shape_profile_features(
    mask: np.ndarray,
    longitudinal: np.ndarray,
    transverse: np.ndarray,
    angle: np.ndarray,
) -> tuple[list[float], list[str]]:
    selected_long = longitudinal[mask]
    selected_transverse = transverse[mask]
    long_scale = max(float(np.abs(selected_long).max()), 1e-6)
    transverse_scale = max(float(np.abs(selected_transverse).max()), 1e-6)
    long_unit = longitudinal / long_scale
    transverse_unit = transverse / transverse_scale
    centers = np.linspace(-11 / 12, 11 / 12, 12)
    bins = np.clip(((long_unit + 1.0) * 6.0).astype(int), 0, 11)
    widths = np.full(12, np.nan, dtype=np.float32)
    for index in range(12):
        values = transverse_unit[mask & (bins == index)]
        if values.size >= 8:
            widths[index] = values.max() - values.min()
    finite = np.isfinite(widths)
    if finite.sum() >= 4:
        widths[~finite] = np.interp(centers[~finite], centers[finite], widths[finite])
    else:
        widths[~finite] = 0.0
    fitted = np.polyval(np.polyfit(centers, widths, 2), centers)
    residual = widths - fitted
    width_features = [
        float(residual.max()),
        float(residual.min()),
        float(residual.std()),
        float(abs(centers[int(np.argmax(residual))])),
        float(np.mean(np.abs(widths - widths[::-1]))),
        float((widths[:2].mean() + widths[-2:].mean()) / max(2 * widths[5:7].mean(), 1e-6)),
    ]
    width_names = [
        "width_positive_bulge",
        "width_negative_residual",
        "width_residual_std",
        "width_bulge_longitudinal_location",
        "width_mirror_asymmetry",
        "width_pole_to_mid_ratio",
    ]

    rho = np.sqrt(long_unit * long_unit + transverse_unit * transverse_unit)
    angular_bins = np.floor(((angle + np.pi) / (2 * np.pi)) * 24).astype(int) % 24
    radius = np.full(24, np.nan, dtype=np.float32)
    for index in range(24):
        values = rho[mask & (angular_bins == index)]
        if values.size:
            radius[index] = values.max()
    radius[~np.isfinite(radius)] = float(np.nanmedian(radius))
    smooth = (
        np.roll(radius, 2)
        + 2 * np.roll(radius, 1)
        + 3 * radius
        + 2 * np.roll(radius, -1)
        + np.roll(radius, -2)
    ) / 9.0
    radial_residual = radius - smooth
    radius_features = [
        float(radial_residual.max()),
        float(radial_residual.min()),
        float(radial_residual.std()),
        float(np.quantile(np.abs(radial_residual), 0.95)),
        float(radius.max() / max(radius.min(), 1e-6)),
    ]
    radius_names = [
        "radius_positive_bulge",
        "radius_negative_residual",
        "radius_residual_std",
        "radius_abs_residual_p95",
        "radius_max_to_min",
    ]
    return width_features + radius_features, width_names + radius_names


def image_features(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if gray.ndim != 2 or gray.shape != mask.shape:
        raise ValueError("Expected shape-matched grayscale image and mask")
    mask = mask.astype(bool)
    if mask.sum() < 256:
        raise ValueError("Kidney mask is too small")
    image = gray.astype(np.float32) / 255.0
    yy, xx = np.nonzero(mask)
    coordinates = np.c_[yy - yy.mean(), xx - xx.mean()].astype(np.float64)
    _, _, vectors = np.linalg.svd(coordinates, full_matrices=False)
    longitudinal = np.zeros(mask.shape, np.float32)
    transverse = np.zeros(mask.shape, np.float32)
    longitudinal[yy, xx] = coordinates @ vectors[0]
    transverse[yy, xx] = coordinates @ vectors[1]
    long_scale = max(float(longitudinal[mask].std()), 1e-6)
    transverse_scale = max(float(transverse[mask].std()), 1e-6)
    angle = np.arctan2(transverse / transverse_scale, longitudinal / long_scale)
    sector = np.floor(((angle + np.pi) / (2 * np.pi)) * SECTORS).astype(int) % SECTORS
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    depth = distance / max(float(distance.max()), 1e-6)
    outer = mask & (depth <= 1.0 / 3.0)
    inner = mask & (depth > 1.0 / 3.0)

    extended = image.copy()
    extended[~mask] = float(np.median(image[mask]))
    mean9 = cv2.boxFilter(extended, cv2.CV_32F, (9, 9), normalize=True)
    square9 = cv2.boxFilter(extended * extended, cv2.CV_32F, (9, 9), normalize=True)
    local_std = np.sqrt(np.maximum(square9 - mean9 * mean9, 0.0))
    gx = cv2.Sobel(extended, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(extended, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gx, gy)
    laplacian = np.abs(cv2.Laplacian(extended, cv2.CV_32F, ksize=3))
    maps = {
        "intensity": robust_z(image, mask),
        "local_std": robust_z(local_std, mask),
        "gradient": robust_z(gradient, mask),
        "laplacian": robust_z(laplacian, mask),
    }
    values: list[float] = []
    names: list[str] = []
    summary_names = ("mean", "std", "min", "max", "range", "p10", "p90")
    for map_name, current in maps.items():
        whole_sector = sector_values(current, sector, mask)
        outer_sector = sector_values(current, sector, outer)
        inner_sector = sector_values(current, sector, inner)
        contrasts = outer_sector - inner_sector
        for region_name, current_values in (
            ("whole", whole_sector),
            ("outer", outer_sector),
            ("inner", inner_sector),
            ("outer_minus_inner", contrasts),
        ):
            values.extend(summary(current_values))
            names.extend(
                f"{map_name}_{region_name}_sector_{suffix}" for suffix in summary_names
            )

    intensity_z = maps["intensity"]
    morphology_names = (
        "area_fraction",
        "mean_depth",
        "depth_span",
        "outer_touch_fraction",
        "angular_concentration",
        "absolute_longitudinal_location",
        "solidity",
        "eccentricity",
        "contour_count",
    )
    for label, binary in (("bright", intensity_z >= 2.0), ("dark", intensity_z <= -2.0)):
        whole_fraction = sector_values(binary.astype(np.float32), sector, mask)
        outer_fraction = sector_values(binary.astype(np.float32), sector, outer)
        inner_fraction = sector_values(binary.astype(np.float32), sector, inner)
        for region_name, fractions in (
            ("whole", whole_fraction),
            ("outer", outer_fraction),
            ("inner", inner_fraction),
        ):
            values.extend(summary(fractions))
            names.extend(
                f"{label}_{region_name}_sector_{suffix}" for suffix in summary_names
            )
        values.extend(
            [
                circular_run(whole_fraction >= 0.05),
                circular_run(outer_fraction >= 0.05),
                circular_run(inner_fraction >= 0.05),
            ]
        )
        names.extend(
            (
                f"{label}_whole_contiguous_sector_run",
                f"{label}_outer_contiguous_sector_run",
                f"{label}_inner_contiguous_sector_run",
            )
        )
        values.extend(
            component_morphology(
                binary, mask, depth, angle, longitudinal / max(float(np.abs(longitudinal[mask]).max()), 1e-6)
            )
        )
        names.extend(f"{label}_component_{name}" for name in morphology_names)

    shape_values, shape_names = shape_profile_features(
        mask, longitudinal, transverse, angle
    )
    values.extend(shape_values)
    names.extend(shape_names)
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite wedge-sector feature")
    return array, names


def extract(images: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
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
            print(f"Wedge-sector features {index + 1}/{len(images)}", flush=True)
    return np.stack(rows), names or []


def run() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Preserve completed or partial wedge experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    patients, kidneys, images = cohort.load_reference()
    features, feature_names = extract(images)
    np.save(OUT / "wedge_sector_features.npy", features)
    write_json(OUT / "feature_names.json", feature_names)
    pairs = base.patient_pair_features(features, images, patients)
    symmetric, antisymmetric = hierarchy.structured_views(pairs)
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    references = pd.read_csv(REFERENCE)
    references = references[references.group.eq(URI_GROUP)].set_index(
        ["seed", "kidney_alias"], verify_integrity=True
    )
    metrics: list[dict] = []
    predictions: list[dict] = []
    selections: list[dict] = []
    for seed in base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        patient_roles = patients.patient_alias.map(mapping)
        kidney_roles = kidneys.patient_alias.map(mapping)
        train = np.flatnonzero(patient_roles.eq("train"))
        development = np.flatnonzero(kidney_roles.eq("development"))
        target = kidneys.iloc[development]
        fixed_uri = references.loc[
            [(seed, alias) for alias in target.kidney_alias], "score"
        ].to_numpy(float)
        probability, models = hierarchy.state_probabilities(
            symmetric, antisymmetric, states, train, seed
        )
        score = hierarchy.kidney_scores(probability, patients, kidneys)[development]
        draw_scores = {
            "official_uri_fixed_reference": fixed_uri,
            "wedge_sector": score,
            "wedge_sector_20pct_plus_uri": 0.20 * score + 0.80 * fixed_uri,
            "wedge_sector_25pct_plus_uri": 0.25 * score + 0.75 * fixed_uri,
        }
        joblib.dump(models, OUT / "models" / f"{seed}_wedge_sector.joblib", compress=3)
        for head, model in models.items():
            selections.append(
                {
                    "seed": seed,
                    "head": head,
                    "inner_cv_auroc": float(model.best_score_),
                    **model.best_params_,
                }
            )
        for group, current_score in draw_scores.items():
            result = base.metric(labels[development], current_score)
            metrics.append(
                {
                    "seed": seed,
                    "role": "development",
                    "group": group,
                    "patients": target.patient_alias.nunique(),
                    "kidneys": len(target),
                    **result,
                }
            )
            for local, index in enumerate(development):
                predictions.append(
                    {
                        "seed": seed,
                        "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[index],
                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                        "side": kidneys.side.iloc[index],
                        "label": int(labels[index]),
                        "score": float(current_score[local]),
                    }
                )
        print(f"Wedge-sector development draw {seed} complete", flush=True)

    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    summary_frame = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary_frame.columns = ["_".join(column) for column in summary_frame.columns]
    summary_frame = summary_frame.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    summary_frame.to_csv(OUT / "development_summary.csv")
    pd.DataFrame(selections).to_csv(OUT / "training_inner_cv_selections.csv", index=False)
    groups = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups, base.SEEDS, "official_uri_fixed_reference"
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary_frame[
        (summary_frame.auroc_mean > 0.8) & (summary_frame.ap_mean > 0.8)
    ].index.tolist()
    write_json(
        OUT / "protocol_and_decision.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "hypothesis": "Focal APN can appear as a wedge-shaped hyper/hypoechoic cortical lesion with local bulging; whole-region summaries dilute a single affected sector.",
            "features": "Rotation/reflection-invariant summaries of 12 angular sectors, contour-depth contrasts, bright/dark component radial morphology, longitudinal width residuals, and angular contour-radius residuals.",
            "feature_count": len(feature_names),
            "literature": [
                "PMID 17987762: focal lesions were commonly hyperechoic, upper-pole, and wedge-shaped",
                "DOI 10.1371/journal.pone.0284016: grayscale echotexture changes are nonspecific and focal lesions may be missed",
            ],
            "groups_above_both_development_targets": achieved,
            "promotion": bool(achieved),
            "exploratory_test_rows_scored": 0,
        },
    )
    print(summary_frame.to_string(), flush=True)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=4):
        run()
