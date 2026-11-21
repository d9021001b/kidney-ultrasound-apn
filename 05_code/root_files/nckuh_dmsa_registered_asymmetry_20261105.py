"""Contour-registered bilateral residual features for NCKUH APN laterality."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_directional as directional
import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_uri_contralateral_local_20261105 as local_study
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_registered_asymmetry_20261105"
REFERENCE = (
    ROOT
    / "02_experiments/NCKUH_DMSA_uri_contralateral_local_20261105/development_predictions.csv"
)
CURRENT_GROUP = "current_tile_usfm_reference"
URI_GROUP = "official_uri_fixed_quarter_reference"
SEEDS = (20261102, 20261103, 20261104)
GRID = 6
SIZE = 192
FLIPS = ("none", "horizontal", "vertical", "both")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def apply_flip(array: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return array.copy()
    if mode == "horizontal":
        return np.fliplr(array).copy()
    if mode == "vertical":
        return np.flipud(array).copy()
    if mode == "both":
        return np.flip(array, axis=(0, 1)).copy()
    raise ValueError(mode)


def canonical_canvas(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, aligned, _ = directional.matched_canvases(gray, mask.astype(np.uint8))
    image, binary = aligned
    canvas, canvas_mask = base.tight_canvas(image, binary, size=SIZE)
    return canvas[..., 0].astype(np.float32) / 255.0, canvas_mask.astype(np.uint8)


def register_by_contour(
    fixed_mask: np.ndarray,
    moving_image: np.ndarray,
    moving_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    fixed = distance_transform_edt(fixed_mask).astype(np.float32)
    moving = distance_transform_edt(moving_mask).astype(np.float32)
    if fixed.max() > 0:
        fixed /= fixed.max()
    if moving.max() > 0:
        moving /= moving.max()
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
    try:
        cv2.findTransformECC(
            fixed,
            moving,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
    except cv2.error:
        warp = np.eye(2, 3, dtype=np.float32)
    registered_image = cv2.warpAffine(
        moving_image,
        warp,
        (fixed_mask.shape[1], fixed_mask.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )
    registered_mask = cv2.warpAffine(
        moving_mask,
        warp,
        (fixed_mask.shape[1], fixed_mask.shape[0]),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(np.uint8)
    registered_image *= registered_mask
    union = np.logical_or(fixed_mask, registered_mask).sum()
    overlap = np.logical_and(fixed_mask, registered_mask).sum()
    iou = float(overlap / union) if union else 0.0
    return registered_image, registered_mask, iou


def masked_pool(values: np.ndarray, mask: np.ndarray, grid: int = GRID) -> np.ndarray:
    weighted = cv2.resize(
        (values * mask).astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA
    )
    coverage = cv2.resize(mask.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
    return np.divide(weighted, coverage, out=np.zeros_like(weighted), where=coverage >= 0.05).ravel()


def descriptor(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    image = image.astype(np.float32) * mask
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    mean = cv2.GaussianBlur(image, (0, 0), 2.0)
    mean_square = cv2.GaussianBlur(image * image, (0, 0), 2.0)
    local_std = np.sqrt(np.maximum(mean_square - mean * mean, 0))
    coverage = cv2.resize(mask.astype(np.float32), (GRID, GRID), interpolation=cv2.INTER_AREA).ravel()
    pixels = image[mask > 0]
    y, x = np.nonzero(mask)
    covariance = np.cov(np.stack([x, y])) if len(x) > 1 else np.eye(2)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1e-8)
    bbox_height = max(1, int(y.max() - y.min() + 1))
    bbox_width = max(1, int(x.max() - x.min() + 1))
    globals_ = np.asarray(
        [
            mask.mean(),
            bbox_width / SIZE,
            bbox_height / SIZE,
            np.sqrt(eigenvalues[-1]) / SIZE,
            np.sqrt(eigenvalues[0]) / SIZE,
            np.sqrt(eigenvalues[-1] / eigenvalues[0]),
            pixels.mean(),
            pixels.std(),
            *np.quantile(pixels, [0.1, 0.5, 0.9]),
            gradient[mask > 0].mean(),
            local_std[mask > 0].mean(),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            masked_pool(image, mask),
            masked_pool(gradient, mask),
            masked_pool(local_std, mask),
            coverage,
            globals_,
        ]
    ).astype(np.float32)


def registered_pair_features(
    left: tuple[np.ndarray, np.ndarray] | None,
    right: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, list[float]]:
    width = 4 * (4 * GRID * GRID + 13) + 1
    if left is None or right is None:
        availability = float(left is not None) - float(right is not None)
        return np.r_[np.zeros(width - 1, np.float32), availability], [0.0] * 4
    left_image, left_mask = left
    left_descriptor = descriptor(left_image, left_mask)
    values, quality = [], []
    for mode in FLIPS:
        moving_image = apply_flip(right[0], mode)
        moving_mask = apply_flip(right[1], mode)
        registered_image, registered_mask, iou = register_by_contour(
            left_mask, moving_image, moving_mask
        )
        values.append((left_descriptor - descriptor(registered_image, registered_mask)) * iou)
        quality.append(iou)
    return np.r_[np.concatenate(values), 0.0].astype(np.float32), quality


def load_images(images: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output = {}
    for identifier in images.image_id:
        gray = cv2.imread(str(base.OUT / "percentile" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(base.OUT / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None:
            raise FileNotFoundError(identifier)
        output[identifier] = canonical_canvas(gray, mask > 0)
    return output


def build_patient_features(
    patients: pd.DataFrame, images: pd.DataFrame, canvases: dict[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, pd.DataFrame]:
    rows, audit = [], []
    for patient in patients.patient_alias:
        group = images[images.patient_alias.eq(patient)]
        lookup = {row.view: row.image_id for row in group.itertuples(index=False)}
        patient_values = []
        record = {"patient_alias": patient}
        for plane, left_view, right_view in (
            ("longitudinal", "LL", "RL"),
            ("transverse", "LT", "RT"),
        ):
            values, quality = registered_pair_features(
                canvases.get(lookup.get(left_view, "")),
                canvases.get(lookup.get(right_view, "")),
            )
            patient_values.append(values)
            record[f"{plane}_left_present"] = left_view in lookup
            record[f"{plane}_right_present"] = right_view in lookup
            for mode, iou in zip(FLIPS, quality):
                record[f"{plane}_{mode}_contour_iou"] = iou
        rows.append(np.concatenate(patient_values))
        audit.append(record)
    result = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite registered residual feature")
    return result, pd.DataFrame(audit)


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve registered experiment: {OUT}")
    OUT.mkdir(parents=True)
    patients, _, images = local_study.load_reference()
    images[["image_id", "patient_alias", "view"]].to_csv(
        OUT / "encoding_manifest.csv", index=False
    )
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "Improve unilateral affected-side ranking without changing the any-APN or bilateral URI-CADS heads",
        "cohort": "210 patients, 414 imaged kidneys, 820 contour-first images",
        "registration": "Principal-axis canonicalization to 192x192, four fixed reflection hypotheses, and outcome-blind Euclidean ECC fitted only to kidney-contour distance maps",
        "features": "Signed left-minus-registered-right 6x6 intensity, gradient, local-variation and contour-coverage grids plus compact shape/intensity summaries for matched longitudinal and transverse views",
        "model": "Frozen URI-CADS global any/bilateral heads; training-only PCA-logistic laterality head using registered signed residuals",
        "roles": "Unchanged three 126/42/42 patient random assignments; development scored, exploratory test unopened",
        "registered_fusions": "Candidate alone and fixed 20%/25% contributions to current and official URI-CADS references",
        "literature": [
            "https://pubmed.ncbi.nlm.nih.gov/3513490/",
            "https://pubmed.ncbi.nlm.nih.gov/12069464/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC7566852/",
            "https://pubmed.ncbi.nlm.nih.gov/15132504/",
        ],
        "labels_used_in_encoding": [],
        "source_sha256": digest(Path(__file__)),
        "manifest_rows": len(images),
        "patients": len(patients),
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(
        OUT / "run_lock.json",
        {
            "protocol_sha256": digest(OUT / "protocol.json"),
            "manifest_sha256": digest(OUT / "encoding_manifest.csv"),
            "source_sha256": digest(Path(__file__)),
        },
    )
    print("Registered contour-guided bilateral residual experiment", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    for path, key in (
        (OUT / "protocol.json", "protocol_sha256"),
        (OUT / "encoding_manifest.csv", "manifest_sha256"),
        (Path(__file__), "source_sha256"),
    ):
        if digest(path) != lock[key]:
            raise ValueError(f"Registered input changed: {path}")


def extract() -> None:
    require_lock()
    target = OUT / "registered_residual_patient_features.npy"
    if target.exists():
        raise FileExistsError(f"Preserve completed features: {target}")
    patients, _, images = local_study.load_reference()
    manifest = pd.read_csv(OUT / "encoding_manifest.csv")
    if manifest.image_id.tolist() != images.image_id.tolist():
        raise ValueError("Image order changed")
    features, audit = build_patient_features(patients, images, load_images(images))
    np.save(target, features)
    audit.to_csv(OUT / "registration_quality.csv", index=False)
    write_json(
        OUT / "encoder_audit.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "patients": len(patients),
            "images": len(images),
            "shape": list(features.shape),
            "finite": bool(np.isfinite(features).all()),
            "labels_used_in_encoding": [],
            "feature_sha256": digest(target),
            "median_contour_iou": float(
                audit.filter(like="contour_iou").replace(0, np.nan).stack().median()
            ),
        },
    )
    print(f"Extracted registered residual features {features.shape}", flush=True)


def evaluate() -> None:
    require_lock()
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve completed development evaluation")
    features_path = OUT / "registered_residual_patient_features.npy"
    encoder = json.loads((OUT / "encoder_audit.json").read_text(encoding="utf-8"))
    if digest(features_path) != encoder["feature_sha256"]:
        raise ValueError("Feature hash mismatch")
    registered = np.load(features_path)
    global_embedding = np.load(uri.OUT / "official_fold_embeddings.npy").reshape(820, -1)
    patients, kidneys, images = local_study.load_reference()
    global_views = hierarchy.structured_views(
        base.patient_pair_features(global_embedding, images, patients)
    )
    combined = np.c_[global_views[1], registered]
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    references = pd.read_csv(REFERENCE)
    references = references[
        references.group.isin([CURRENT_GROUP, URI_GROUP])
    ].set_index(["seed", "group", "kidney_alias"], verify_integrity=True)
    (OUT / "models").mkdir()

    metrics, predictions, selections = [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train = np.flatnonzero(patients.patient_alias.map(mapping).eq("train"))
        development = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        target = kidneys.iloc[development]
        current = references.loc[
            [(seed, CURRENT_GROUP, alias) for alias in target.kidney_alias]
        ].score.to_numpy(float)
        fixed_uri = references.loc[
            [(seed, URI_GROUP, alias) for alias in target.kidney_alias]
        ].score.to_numpy(float)
        probability_sets = {}
        probability_sets["registered_residual_laterality"], residual_models = (
            local_study.hybrid_probability(global_views, registered, states, train, seed)
        )
        probability_sets["registered_plus_global_laterality"], combined_models = (
            local_study.hybrid_probability(global_views, combined, states, train, seed)
        )
        joblib.dump(
            {"residual": residual_models, "combined": combined_models},
            OUT / "models" / f"{seed}_registered_asymmetry.joblib",
            compress=3,
        )
        for source, models in (("residual", residual_models), ("combined", combined_models)):
            for head, model in models.items():
                selections.append(
                    {
                        "seed": seed,
                        "source": source,
                        "head": head,
                        "inner_cv_auroc": float(model.best_score_),
                        **model.best_params_,
                    }
                )
        scores = {CURRENT_GROUP: current, URI_GROUP: fixed_uri}
        for name, probability in probability_sets.items():
            candidate = hierarchy.kidney_scores(probability, patients, kidneys)[development]
            scores[name] = candidate
            for fraction, suffix in ((0.20, "fifth"), (0.25, "quarter")):
                scores[f"{name}_{suffix}_plus_current"] = fraction * candidate + (1 - fraction) * current
                scores[f"{name}_{suffix}_plus_uri"] = fraction * candidate + (1 - fraction) * fixed_uri
        for group, score in scores.items():
            metrics.append(
                {
                    "seed": seed,
                    "role": "development",
                    "group": group,
                    "patients": target.patient_alias.nunique(),
                    "kidneys": len(target),
                    **hierarchy.metric(labels[development], score),
                }
            )
            for local_index, index in enumerate(development):
                predictions.append(
                    {
                        "seed": seed,
                        "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[index],
                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                        "side": kidneys.side.iloc[index],
                        "label": int(labels[index]),
                        "score": float(score[local_index]),
                    }
                )
        print(f"Completed registered asymmetry draw {seed}", flush=True)

    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "training_inner_cv_selections.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    groups = prediction_frame.group.drop_duplicates().tolist()
    intervals, differences, samples = cluster_intervals(
        prediction_frame, groups, SEEDS, URI_GROUP
    )
    intervals.to_csv(OUT / "development_ci.csv", index=False)
    differences.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(
        OUT / "decision.json",
        {
            "groups_above_both_development_targets": achieved,
            "promotion": bool(achieved),
            "reference": URI_GROUP,
            "exploratory_test_rows_scored": 0,
        },
    )
    write_json(
        OUT / "integrity_audit.json",
        {
            "status": "pass",
            "patients": len(patients),
            "kidneys": len(kidneys),
            "images": len(images),
            "features": int(registered.shape[1]),
            "development_groups": prediction_frame.group.nunique(),
            "finite_scores": bool(np.isfinite(prediction_frame.score).all()),
            "unit_interval_scores": bool(prediction_frame.score.between(0, 1).all()),
            "exploratory_test_rows_scored": 0,
            "labels_used_in_encoding": [],
        },
    )
    print(summary.to_string(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "extract", "evaluate"))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        globals()[args.stage]()


if __name__ == "__main__":
    main()
