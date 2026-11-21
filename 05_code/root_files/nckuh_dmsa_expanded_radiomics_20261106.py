"""Evaluate an expanded, mask-constrained radiomics bank for NCKUH APN."""
from __future__ import annotations

import hashlib
import json
import logging
import warnings
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_fullframe_20261103"
REFERENCE = uri.OUT / "development_predictions.csv"
OUT = ROOT / "02_experiments/NCKUH_DMSA_expanded_radiomics_20261106"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"
IMAGE_TYPES = {
    "Original": {},
    "Wavelet": {"wavelet": "coif1", "level": 1},
    "LoG": {"sigma": [1.0, 2.0, 3.0]},
    "Gradient": {},
    "LBP2D": {"lbp2DRadius": 1, "lbp2DSamples": 8, "lbp2DMethod": "uniform"},
    "Square": {},
    "SquareRoot": {},
    "Logarithm": {},
    "Exponential": {},
}
FEATURE_CLASSES = ("firstorder", "shape2D", "glcm", "glrlm", "glszm", "gldm", "ngtdm")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def nearest_extension(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if gray.shape != binary.shape or not binary.any():
        raise ValueError("Expected a nonempty shape-matched kidney mask")
    if binary.all():
        return gray.copy()
    indices = distance_transform_edt(~binary, return_distances=False, return_indices=True)
    return gray[tuple(indices)].copy()


def build_extractor():
    import radiomics
    from radiomics import featureextractor

    radiomics.setVerbosity(logging.ERROR)
    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=16,
        normalize=False,
        distances=[1, 2],
        force2D=True,
        minimumROIDimensions=2,
        padDistance=5,
    )
    extractor.disableAllImageTypes()
    for name, settings in IMAGE_TYPES.items():
        extractor.enableImageTypeByName(name, customArgs=settings)
    extractor.disableAllFeatures()
    for name in FEATURE_CLASSES:
        extractor.enableFeatureClassByName(name)
    return extractor


_EXTRACTOR = None


def extract_one(job: tuple[str, str, str]) -> tuple[str, dict[str, float]]:
    global _EXTRACTOR
    import SimpleITK as sitk

    identifier, gray_path, mask_path = job
    gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gray is None or mask is None or gray.shape != mask.shape:
        raise ValueError(f"Unreadable image-mask pair: {identifier}")
    binary = mask > 0
    extended = nearest_extension(gray, binary)
    extended = np.pad(extended, 8, mode="edge")
    padded_mask = np.pad(binary.astype(np.uint8), 8)
    if _EXTRACTOR is None:
        _EXTRACTOR = build_extractor()
    with warnings.catch_warnings():
        # The float values remain exact integers; float avoids PyRadiomics uint8
        # bin-bound overflow while making skimage's generic LBP warning irrelevant.
        warnings.filterwarnings(
            "ignore", message="Applying `local_binary_pattern` to floating-point images"
        )
        result = _EXTRACTOR.execute(
            sitk.GetImageFromArray(extended.astype(np.float32)),
            sitk.GetImageFromArray(padded_mask),
        )
    values = {
        key: float(value)
        for key, value in result.items()
        if not key.startswith("diagnostics")
    }
    if not values or not np.isfinite(np.fromiter(values.values(), dtype=float)).all():
        raise ValueError(f"Nonfinite expanded radiomics: {identifier}")
    return identifier, values


def extract(processes: int = 8) -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Preserve completed or partial experiment: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    patients, kidneys, images = cohort.load_reference()
    jobs = [
        (
            row.image_id,
            str(SOURCE / "percentile" / f"{row.image_id}.png"),
            str(SOURCE / "masks" / f"{row.image_id}.png"),
        )
        for row in images.itertuples(index=False)
    ]
    with Pool(processes=processes) as pool:
        rows = pool.map(extract_one, jobs, chunksize=2)
    names = list(rows[0][1])
    if any(list(values) != names for _, values in rows):
        raise ValueError("Feature columns changed across images")
    matrix = np.asarray([[values[name] for name in names] for _, values in rows], dtype=np.float32)
    if matrix.shape[0] != 820 or not np.isfinite(matrix).all():
        raise ValueError(f"Unexpected expanded radiomics matrix: {matrix.shape}")
    np.save(OUT / "expanded_radiomics_image.npy", matrix)
    write_json(OUT / "feature_names.json", names)
    write_json(
        OUT / "extraction_audit.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "patients": len(patients),
            "kidneys": len(kidneys),
            "images": len(images),
            "features": matrix.shape[1],
            "finite_fraction": float(np.isfinite(matrix).mean()),
            "image_types": IMAGE_TYPES,
            "feature_classes": FEATURE_CLASSES,
            "input": "Contour-first P1-P99 kidney ROI with nearest-kidney extension for filtering",
            "labels_used_in_extraction": [],
            "feature_sha256": digest(OUT / "expanded_radiomics_image.npy"),
        },
    )
    print(f"Extracted {matrix.shape[1]} expanded radiomics features for 820 images", flush=True)


def fit_head(values: np.ndarray, labels: np.ndarray, selected: np.ndarray, seed: int):
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(f_classif)),
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=4000,
                    random_state=seed,
                ),
            ),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {"select__k": [32, 64, 128], "lr__C": [0.01, 0.1, 1.0]},
        scoring="roc_auc",
        cv=StratifiedKFold(4, shuffle=True, random_state=seed),
        n_jobs=1,
        refit=True,
    )
    search.fit(values[selected], labels[selected])
    return search


def state_probabilities(
    symmetric: np.ndarray,
    antisymmetric: np.ndarray,
    states: np.ndarray,
    train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    any_label = (states != hierarchy.STATE_CODE["N"]).astype(int)
    any_model = fit_head(symmetric, any_label, train, seed)
    affected = train[states[train] != hierarchy.STATE_CODE["N"]]
    bilateral_label = (states == hierarchy.STATE_CODE["B"]).astype(int)
    bilateral_model = fit_head(symmetric, bilateral_label, affected, seed + 101)
    unilateral = train[np.isin(states[train], [hierarchy.STATE_CODE["L"], hierarchy.STATE_CODE["R"]])]
    left_label = (states == hierarchy.STATE_CODE["L"]).astype(int)
    left_model = fit_head(antisymmetric, left_label, unilateral, seed + 202)
    p_any = any_model.predict_proba(symmetric)[:, 1]
    p_bilateral = bilateral_model.predict_proba(symmetric)[:, 1]
    p_left = left_model.predict_proba(antisymmetric)[:, 1]
    probability = np.c_[
        1.0 - p_any,
        p_any * (1.0 - p_bilateral) * p_left,
        p_any * (1.0 - p_bilateral) * (1.0 - p_left),
        p_any * p_bilateral,
    ]
    probability /= probability.sum(1, keepdims=True)
    return probability, {
        "any": any_model,
        "bilateral_given_any": bilateral_model,
        "left_given_unilateral": left_model,
    }


def direct_scores(
    values: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, object]:
    model = fit_head(values, labels, train, seed + 303)
    return model.predict_proba(values)[:, 1], model


def evaluate() -> None:
    feature_path = OUT / "expanded_radiomics_image.npy"
    if not feature_path.is_file():
        raise FileNotFoundError("Run the extract stage first")
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed expanded-radiomics evaluation")
    (OUT / "models").mkdir(exist_ok=True)
    patients, kidneys, images = cohort.load_reference()
    matrix = np.load(feature_path)
    pairs = base.patient_pair_features(matrix, images, patients)
    symmetric, antisymmetric = hierarchy.structured_views(pairs)
    oriented = base.orient_kidneys(pairs, patients, kidneys)
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
        train_patients = np.flatnonzero(patient_roles.eq("train"))
        train_kidneys = np.flatnonzero(kidney_roles.eq("train"))
        development = np.flatnonzero(kidney_roles.eq("development"))
        target = kidneys.iloc[development]
        fixed_uri = references.loc[
            [(seed, alias) for alias in target.kidney_alias], "score"
        ].to_numpy(float)
        probability, state_models = state_probabilities(
            symmetric, antisymmetric, states, train_patients, seed
        )
        state_score = hierarchy.kidney_scores(probability, patients, kidneys)
        direct_score, direct_model = direct_scores(oriented, labels, train_kidneys, seed)
        mean_score = (state_score + direct_score) / 2.0
        candidates = {
            "official_uri_fixed_reference": fixed_uri,
            "expanded_radiomics_state": state_score[development],
            "expanded_radiomics_direct": direct_score[development],
            "expanded_radiomics_state_direct_mean": mean_score[development],
        }
        for name in tuple(candidates)[1:]:
            candidate = candidates[name]
            candidates[f"{name}_20pct_plus_uri"] = 0.20 * candidate + 0.80 * fixed_uri
            candidates[f"{name}_25pct_plus_uri"] = 0.25 * candidate + 0.75 * fixed_uri
        joblib.dump(
            {"state": state_models, "direct": direct_model},
            OUT / "models" / f"{seed}_expanded_radiomics.joblib",
            compress=3,
        )
        for family, models in (("state", state_models), ("direct", {"direct": direct_model})):
            for head, model in models.items():
                selections.append(
                    {
                        "seed": seed,
                        "family": family,
                        "head": head,
                        "inner_cv_auroc": float(model.best_score_),
                        **model.best_params_,
                    }
                )
        for group, score in candidates.items():
            result = base.metric(labels[development], score)
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
                        "score": float(score[local]),
                    }
                )
        print(f"Expanded radiomics development draw {seed} complete", flush=True)
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv")
    pd.DataFrame(selections).to_csv(OUT / "training_inner_cv_selections.csv", index=False)
    groups = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups, base.SEEDS, "official_uri_fixed_reference"
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(
        OUT / "decision.json",
        {
            "groups_above_both_development_targets": achieved,
            "promotion": bool(achieved),
            "reference": "official_uri_fixed_reference",
            "exploratory_test_rows_scored": 0,
        },
    )
    write_json(
        OUT / "integrity_audit.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "patients": len(patients),
            "kidneys": len(kidneys),
            "images": len(images),
            "split_unit": "patient",
            "score_unit": "kidney",
            "feature_extraction_read_labels": False,
            "exploratory_test_rows_scored": 0,
            "inputs": {
                str(feature_path.relative_to(ROOT)): digest(feature_path),
                str((base.OUT / "patient_assignments.csv").relative_to(ROOT)): digest(
                    base.OUT / "patient_assignments.csv"
                ),
                str(REFERENCE.relative_to(ROOT)): digest(REFERENCE),
            },
        },
    )
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "evaluate"))
    parser.add_argument("--processes", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.stage == "extract":
        extract(arguments.processes)
    else:
        evaluate()
