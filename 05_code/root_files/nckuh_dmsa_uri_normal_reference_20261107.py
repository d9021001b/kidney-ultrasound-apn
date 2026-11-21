"""Training-only normal-reference anomaly scores on official URI-CADS features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_uri_normal_reference_20261107"
REFERENCE = uri.OUT / "development_predictions.csv"
REFERENCE_GROUP = "official_uri_embedding_folds_quarter_plus_current"
SEEDS = tuple(base.SEEDS)
STATE_CODE = hierarchy.STATE_CODE


@dataclass(frozen=True)
class Configuration:
    source: str
    components: int
    normal_pool: str
    aggregation: str

    @property
    def name(self) -> str:
        return f"{self.source}_pca{self.components}_{self.normal_pool}_{self.aggregation}"


CONFIGURATIONS = tuple(
    [
        Configuration(source, components, normal_pool, aggregation)
        for source, components, normal_pool, aggregation in product(
            ("embedding",), (16, 32), ("strict_normal_patient", "negative_kidney"), ("mean", "max")
        )
    ]
    + [
        Configuration(source, components, normal_pool, aggregation)
        for source, components, normal_pool, aggregation in product(
            ("pathology",), (4, 8), ("strict_normal_patient", "negative_kidney"), ("mean", "max")
        )
    ]
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    q25, q75 = np.percentile(values, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(values))
    return median, max(scale, 1e-6)


def fit_normal_model(
    values: np.ndarray,
    normal_indices: np.ndarray,
    components: int,
    seed: int,
) -> dict:
    normal = np.asarray(values[normal_indices], dtype=np.float64)
    if len(normal) <= components + 2:
        raise ValueError(f"Only {len(normal)} normal images for {components} PCA components")
    scaler = StandardScaler().fit(normal)
    normal_scaled = scaler.transform(normal)
    pca = PCA(
        n_components=components,
        whiten=False,
        svd_solver="randomized",
        random_state=seed,
    ).fit(normal_scaled)
    normal_projected = pca.transform(normal_scaled)
    covariance = LedoitWolf(assume_centered=False).fit(normal_projected)
    mahalanobis = np.sqrt(np.maximum(covariance.mahalanobis(normal_projected), 0.0))
    reconstruction = pca.inverse_transform(normal_projected)
    residual = np.sqrt(np.mean(np.square(normal_scaled - reconstruction), axis=1))
    mahalanobis_location, mahalanobis_scale = robust_location_scale(mahalanobis)
    residual_location, residual_scale = robust_location_scale(residual)
    return {
        "scaler": scaler,
        "pca": pca,
        "covariance": covariance,
        "mahalanobis_location": mahalanobis_location,
        "mahalanobis_scale": mahalanobis_scale,
        "residual_location": residual_location,
        "residual_scale": residual_scale,
        "normal_images": int(len(normal_indices)),
    }


def normal_anomaly_score(model: dict, values: np.ndarray) -> np.ndarray:
    scaled = model["scaler"].transform(np.asarray(values, dtype=np.float64))
    projected = model["pca"].transform(scaled)
    mahalanobis = np.sqrt(np.maximum(model["covariance"].mahalanobis(projected), 0.0))
    reconstruction = model["pca"].inverse_transform(projected)
    residual = np.sqrt(np.mean(np.square(scaled - reconstruction), axis=1))
    mahalanobis_z = (
        mahalanobis - model["mahalanobis_location"]
    ) / model["mahalanobis_scale"]
    residual_z = (residual - model["residual_location"]) / model["residual_scale"]
    return 0.70 * mahalanobis_z + 0.30 * residual_z


def aggregate_images(
    image_score: np.ndarray,
    image_kidney_index: np.ndarray,
    kidney_count: int,
    method: str,
) -> np.ndarray:
    output = np.full(kidney_count, np.nan, dtype=float)
    for kidney_index in range(kidney_count):
        selected = image_score[image_kidney_index == kidney_index]
        if not len(selected):
            continue
        output[kidney_index] = float(np.mean(selected) if method == "mean" else np.max(selected))
    return output


def image_feature_sets() -> dict[str, np.ndarray]:
    embeddings = np.load(uri.OUT / "official_fold_embeddings.npy", mmap_mode="r")
    pathology = np.load(uri.OUT / "official_fold_pathology_scores.npy", mmap_mode="r")
    return {
        "embedding": np.asarray(embeddings.mean(1), dtype=np.float32),
        "pathology": np.asarray(
            np.concatenate([pathology.mean(1), pathology.std(1)], axis=1),
            dtype=np.float32,
        ),
    }


def fit_configuration(
    configuration: Configuration,
    values: np.ndarray,
    patients: pd.DataFrame,
    kidneys: pd.DataFrame,
    images: pd.DataFrame,
    image_kidney_index: np.ndarray,
    selected_patients: np.ndarray,
    seed: int,
) -> dict:
    patient_aliases = set(patients.patient_alias.iloc[selected_patients])
    selected_images = images.patient_alias.isin(patient_aliases).to_numpy()
    if configuration.normal_pool == "strict_normal_patient":
        normal_mask = selected_images & images.DMSA_APN.eq("N").to_numpy()
    else:
        normal_mask = selected_images & images.kidney_APN.eq(0).to_numpy()
    normal_indices = np.flatnonzero(normal_mask)
    model = fit_normal_model(values, normal_indices, configuration.components, seed)
    image_score = normal_anomaly_score(model, values)
    kidney_score = aggregate_images(
        image_score, image_kidney_index, len(kidneys), configuration.aggregation,
    )
    selected_kidneys = kidneys.patient_alias.isin(patient_aliases).to_numpy()
    if not np.isfinite(kidney_score[selected_kidneys]).all():
        raise ValueError("Selected training kidney is missing an anomaly score")
    calibrator = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
    ).fit(
        kidney_score[selected_kidneys, None],
        kidneys.kidney_APN.to_numpy(int)[selected_kidneys],
    )
    return {
        "configuration": configuration,
        "normal_model": model,
        "calibrator": calibrator,
        "image_score": image_score,
        "kidney_score": kidney_score,
    }


def calibrated_probability(fitted: dict) -> np.ndarray:
    score = fitted["kidney_score"]
    if not np.isfinite(score).all():
        raise ValueError("A kidney is missing from the anomaly representation")
    return fitted["calibrator"].predict_proba(score[:, None])[:, 1]


def nested_select(
    features: dict[str, np.ndarray],
    patients: pd.DataFrame,
    kidneys: pd.DataFrame,
    images: pd.DataFrame,
    image_kidney_index: np.ndarray,
    train: np.ndarray,
    states: np.ndarray,
    seed: int,
) -> tuple[dict, dict, list[dict]]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 907)
    records = []
    for configuration in CONFIGURATIONS:
        fold_metrics = []
        for fold, (local_fit, local_validation) in enumerate(splitter.split(train, states[train])):
            fit_patients = train[local_fit]
            validation_patients = train[local_validation]
            fitted = fit_configuration(
                configuration,
                features[configuration.source],
                patients,
                kidneys,
                images,
                image_kidney_index,
                fit_patients,
                seed + fold,
            )
            probability = calibrated_probability(fitted)
            validation_aliases = set(patients.patient_alias.iloc[validation_patients])
            validation_kidneys = kidneys.patient_alias.isin(validation_aliases).to_numpy()
            fold_metrics.append(hierarchy.metric(
                kidneys.kidney_APN.to_numpy(int)[validation_kidneys],
                probability[validation_kidneys],
            ))
        auroc = float(np.mean([row["auroc"] for row in fold_metrics]))
        ap = float(np.mean([row["ap"] for row in fold_metrics]))
        records.append({
            "configuration": configuration.name,
            "source": configuration.source,
            "components": configuration.components,
            "normal_pool": configuration.normal_pool,
            "aggregation": configuration.aggregation,
            "inner_auroc": auroc,
            "inner_ap": ap,
            "objective": (auroc + ap) / 2.0,
        })
    selection = pd.DataFrame(records).sort_values(
        ["objective", "inner_auroc", "configuration"],
        ascending=[False, False, True],
    ).iloc[0].to_dict()
    configuration = next(row for row in CONFIGURATIONS if row.name == selection["configuration"])
    fitted = fit_configuration(
        configuration,
        features[configuration.source],
        patients,
        kidneys,
        images,
        image_kidney_index,
        train,
        seed,
    )
    return fitted, selection, records


def run() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve completed or partial experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    patients, kidneys, images = cohort.load_reference()
    features = image_feature_sets()
    states = patients.DMSA_APN.map(STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    reference = pd.read_csv(REFERENCE)
    reference = reference[reference.group.eq(REFERENCE_GROUP)].set_index(
        ["seed", "kidney_alias"], verify_integrity=True,
    )
    kidney_lookup = {alias: index for index, alias in enumerate(kidneys.kidney_alias)}
    image_kidney_index = images.kidney_alias.map(kidney_lookup).to_numpy(int)
    predictions, metrics_rows, selection_rows, inner_rows = [], [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        patient_roles = patients.patient_alias.map(mapping)
        kidney_roles = kidneys.patient_alias.map(mapping)
        train = np.flatnonzero(patient_roles.eq("train"))
        development = np.flatnonzero(kidney_roles.eq("development"))
        target = kidneys.iloc[development]
        fixed = reference.loc[
            [(seed, alias) for alias in target.kidney_alias], "score"
        ].to_numpy(float)
        fitted, selection, records = nested_select(
            features,
            patients,
            kidneys,
            images,
            image_kidney_index,
            train,
            states,
            seed,
        )
        anomaly = calibrated_probability(fitted)[development]
        draw_scores = {
            "official_uri_fixed_reference": fixed,
            "normal_reference_selected": anomaly,
            "normal_reference_10pct_plus_uri": 0.10 * anomaly + 0.90 * fixed,
            "normal_reference_20pct_plus_uri": 0.20 * anomaly + 0.80 * fixed,
            "normal_reference_25pct_plus_uri": 0.25 * anomaly + 0.75 * fixed,
        }
        export_model = {
            key: value for key, value in fitted.items() if key not in {"image_score", "kidney_score"}
        }
        joblib.dump(export_model, OUT / "models" / f"{seed}_selected.joblib", compress=3)
        selection_rows.append({
            "seed": seed,
            **selection,
            "normal_images_refit": fitted["normal_model"]["normal_images"],
        })
        inner_rows.extend({"seed": seed, **row} for row in records)
        for group, score in draw_scores.items():
            current = hierarchy.metric(labels[development], score)
            metrics_rows.append({
                "seed": seed,
                "role": "development",
                "group": group,
                "patients": target.patient_alias.nunique(),
                "kidneys": len(target),
                **current,
            })
            for row, current_score in zip(target.itertuples(index=False), score):
                predictions.append({
                    "seed": seed,
                    "group": group,
                    "patient_alias": row.patient_alias,
                    "kidney_alias": row.kidney_alias,
                    "DMSA_APN": row.DMSA_APN,
                    "side": row.side,
                    "label": int(row.kidney_APN),
                    "score": float(current_score),
                })
        print(f"URI normal-reference development draw {seed} complete", flush=True)
    metric_frame = pd.DataFrame(metrics_rows)
    prediction_frame = pd.DataFrame(predictions)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(OUT / "training_inner_selections.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(OUT / "training_inner_grid.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv")
    groups = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups, SEEDS, "official_uri_fixed_reference",
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(OUT / "decision.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "groups_above_both_development_targets": achieved,
        "promotion": bool(achieved),
        "exploratory_test_rows_scored": 0,
        "selection": "Four-fold patient-level inner CV within each 126-patient training role",
        "normal_reference": "Training-only strict-normal patients or DMSA-negative kidneys",
        "anomaly_score": "0.70 robust Mahalanobis z + 0.30 PCA residual z",
        "reference": REFERENCE_GROUP,
    })
    write_json(OUT / "integrity_audit.json", {
        "patients": len(patients),
        "kidneys": len(kidneys),
        "images": len(images),
        "feature_shapes": {name: list(values.shape) for name, values in features.items()},
        "prediction_roles": sorted(metric_frame.role.unique()),
        "exploratory_test_rows_scored": 0,
        "scores_finite": bool(np.isfinite(prediction_frame.score).all()),
        "scores_in_unit_interval": bool(prediction_frame.score.between(0, 1).all()),
        "development_patients_per_draw": sorted(metric_frame.patients.unique().tolist()),
    })
    print(summary[["auroc_mean", "ap_mean", "brier_mean"]].to_string(), flush=True)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=4):
        run()
