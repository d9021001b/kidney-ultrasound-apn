"""Image-instance MIL over frozen official URI-CADS embeddings for NCKUH APN."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_uri_image_mil_20261106"
REFERENCE = uri.OUT / "development_predictions.csv"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"
POOLERS = ("mean_probability", "mean_logit", "max_probability", "softmax_logit")
REGULARIZATION = (0.01, 0.1, 1.0)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(value)))


def pool_values(probability: np.ndarray, method: str) -> float:
    probability = np.asarray(probability, dtype=float)
    if not len(probability) or not np.isfinite(probability).all():
        raise ValueError("Expected finite nonempty instance probabilities")
    if method == "mean_probability":
        return float(probability.mean())
    if method == "mean_logit":
        return float(sigmoid(logit(probability).mean()))
    if method == "max_probability":
        return float(probability.max())
    if method == "softmax_logit":
        values = logit(probability)
        weights = np.exp(2.0 * (values - values.max()))
        return float(sigmoid(np.average(values, weights=weights)))
    raise ValueError(f"Unknown pooling method: {method}")


def pool_kidneys(
    image_probability: np.ndarray,
    images: pd.DataFrame,
    kidneys: pd.DataFrame,
    method: str,
) -> np.ndarray:
    lookup = {
        alias: pool_values(image_probability[rows.index.to_numpy()], method)
        for alias, rows in images.groupby("kidney_alias", sort=False)
    }
    if not set(kidneys.kidney_alias).issubset(lookup):
        raise ValueError("A scored kidney has no image instance")
    return np.asarray([lookup[alias] for alias in kidneys.kidney_alias], dtype=float)


def image_weights(images: pd.DataFrame, selected_patients: set[str]) -> np.ndarray:
    chosen = images.patient_alias.isin(selected_patients)
    counts = images.loc[chosen].groupby("patient_alias").size()
    return images.loc[chosen].patient_alias.map(1.0 / counts).to_numpy(float)


def fit_image_head(
    values: np.ndarray,
    labels: np.ndarray,
    images: pd.DataFrame,
    patient_indices: np.ndarray,
    patients: pd.DataFrame,
    regularization: float,
    seed: int,
):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    aliases = set(patients.patient_alias.iloc[patient_indices])
    selected = np.flatnonzero(images.patient_alias.isin(aliases))
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(32, whiten=True, random_state=seed)),
            (
                "lr",
                LogisticRegression(
                    C=regularization,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=4000,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(
        values[selected],
        labels[selected],
        lr__sample_weight=image_weights(images, aliases),
    )
    return pipeline


def select_configuration(
    values: np.ndarray,
    image_labels: np.ndarray,
    images: pd.DataFrame,
    patients: pd.DataFrame,
    kidneys: pd.DataFrame,
    train_patients: np.ndarray,
    seed: int,
) -> tuple[float, str, pd.DataFrame]:
    from sklearn.model_selection import StratifiedKFold

    train_aliases = set(patients.patient_alias.iloc[train_patients])
    train_kidneys = np.flatnonzero(kidneys.patient_alias.isin(train_aliases))
    codes = patients.DMSA_APN.to_numpy()[train_patients]
    folds = StratifiedKFold(4, shuffle=True, random_state=seed)
    records: list[dict] = []
    best: tuple[float, float, str] | None = None
    for regularization in REGULARIZATION:
        oof = np.full(len(images), np.nan, dtype=float)
        for fold, (inner_train, inner_validation) in enumerate(
            folds.split(train_patients, codes), start=1
        ):
            fit_patients = train_patients[inner_train]
            validation_aliases = set(patients.patient_alias.iloc[train_patients[inner_validation]])
            validation_images = np.flatnonzero(images.patient_alias.isin(validation_aliases))
            model = fit_image_head(
                values,
                image_labels,
                images,
                fit_patients,
                patients,
                regularization,
                seed + fold,
            )
            oof[validation_images] = model.predict_proba(values[validation_images])[:, 1]
        selected_images = np.flatnonzero(images.patient_alias.isin(train_aliases))
        if not np.isfinite(oof[selected_images]).all():
            raise ValueError("Incomplete patient-level OOF image predictions")
        # Non-training bags are ignored by the selection metric; a neutral value
        # keeps the all-kidney pooling helper finite without exposing their labels.
        oof[~images.patient_alias.isin(train_aliases).to_numpy()] = 0.5
        for method in POOLERS:
            kidney_score = pool_kidneys(oof, images, kidneys, method)
            result = base.metric(kidneys.kidney_APN.to_numpy(int)[train_kidneys], kidney_score[train_kidneys])
            selection_score = (result["auroc"] + result["ap"]) / 2.0
            records.append(
                {
                    "seed": seed,
                    "C": regularization,
                    "pooling": method,
                    "oof_auroc": result["auroc"],
                    "oof_ap": result["ap"],
                    "selection_score": selection_score,
                }
            )
            candidate = (selection_score, -regularization, method)
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    selected = max(records, key=lambda row: (row["selection_score"], -row["C"], row["pooling"]))
    return float(selected["C"]), str(selected["pooling"]), pd.DataFrame(records)


def run() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve completed URI image-MIL experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    patients, kidneys, images = cohort.load_reference()
    embeddings = np.load(uri.OUT / "official_fold_embeddings.npy")
    pathology = np.load(uri.OUT / "official_fold_pathology_scores.npy")
    if embeddings.shape != (820, 5, 2048) or pathology.shape != (820, 5, 7):
        raise ValueError("Unexpected official URI-CADS feature shape")
    view = images["view"].str.endswith("T").to_numpy(float)[:, None]
    values = np.c_[
        embeddings.mean(1),
        pathology.mean(1),
        pathology.std(1),
        view,
    ].astype(np.float32)
    kidney_label = kidneys.set_index("kidney_alias").kidney_APN
    image_labels = images.kidney_alias.map(kidney_label).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    references = pd.read_csv(REFERENCE)
    references = references[references.group.eq(URI_GROUP)].set_index(
        ["seed", "kidney_alias"], verify_integrity=True
    )
    metrics: list[dict] = []
    predictions: list[dict] = []
    selection_rows: list[pd.DataFrame] = []
    for seed in base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        patient_roles = patients.patient_alias.map(mapping)
        kidney_roles = kidneys.patient_alias.map(mapping)
        train_patients = np.flatnonzero(patient_roles.eq("train"))
        development = np.flatnonzero(kidney_roles.eq("development"))
        regularization, pooling, selection = select_configuration(
            values,
            image_labels,
            images,
            patients,
            kidneys,
            train_patients,
            seed,
        )
        selection["selected"] = (
            selection.C.eq(regularization) & selection.pooling.eq(pooling)
        )
        selection_rows.append(selection)
        model = fit_image_head(
            values,
            image_labels,
            images,
            train_patients,
            patients,
            regularization,
            seed,
        )
        image_probability = model.predict_proba(values)[:, 1]
        mil_score = pool_kidneys(image_probability, images, kidneys, pooling)[development]
        target = kidneys.iloc[development]
        fixed_uri = references.loc[
            [(seed, alias) for alias in target.kidney_alias], "score"
        ].to_numpy(float)
        draw_scores = {
            "official_uri_fixed_reference": fixed_uri,
            "official_uri_image_mil": mil_score,
            "official_uri_image_mil_20pct_plus_reference": 0.20 * mil_score + 0.80 * fixed_uri,
            "official_uri_image_mil_25pct_plus_reference": 0.25 * mil_score + 0.75 * fixed_uri,
        }
        joblib.dump(
            {"model": model, "pooling": pooling, "C": regularization},
            OUT / "models" / f"{seed}_uri_image_mil.joblib",
            compress=3,
        )
        for group, score in draw_scores.items():
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
        print(
            f"URI image-MIL draw {seed}: C={regularization:g}, pooling={pooling}",
            flush=True,
        )
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv")
    pd.concat(selection_rows, ignore_index=True).to_csv(
        OUT / "training_oof_selections.csv", index=False
    )
    groups = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups, base.SEEDS, "official_uri_fixed_reference"
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(
        OUT / "protocol_and_decision.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "motivation": "Preserve focal APN evidence by selecting a training-only image-instance pooling rule instead of fixed mean/SD aggregation",
            "representation": "Frozen five-fold official URI-CADS embedding mean, pathology-score mean/SD, and longitudinal/transverse indicator",
            "selection": "Four-fold patient-level OOF selection of C and pooling within each outer training role",
            "poolers": POOLERS,
            "regularization": REGULARIZATION,
            "patients": len(patients),
            "kidneys": len(kidneys),
            "images": len(images),
            "split_unit": "patient",
            "score_unit": "kidney",
            "groups_above_both_development_targets": achieved,
            "promotion": bool(achieved),
            "exploratory_test_rows_scored": 0,
            "inputs": {
                str((uri.OUT / "official_fold_embeddings.npy").relative_to(ROOT)): digest(
                    uri.OUT / "official_fold_embeddings.npy"
                ),
                str((uri.OUT / "official_fold_pathology_scores.npy").relative_to(ROOT)): digest(
                    uri.OUT / "official_fold_pathology_scores.npy"
                ),
                str((base.OUT / "patient_assignments.csv").relative_to(ROOT)): digest(
                    base.OUT / "patient_assignments.csv"
                ),
            },
        },
    )
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=4):
        run()
