"""View-specialized official URI-CADS ensemble for kidney-level NCKUH APN."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_uri_image_mil_20261106 as image_mil
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_uri_view_ensemble_20261106"
REFERENCE = uri.OUT / "development_predictions.csv"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"
REGULARIZATION = (0.01, 0.1, 1.0)
LONGITUDINAL_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def fit_plane_head(
    values: np.ndarray,
    labels: np.ndarray,
    images: pd.DataFrame,
    patients: pd.DataFrame,
    patient_indices: np.ndarray,
    plane: str,
    regularization: float,
    seed: int,
):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    aliases = set(patients.patient_alias.iloc[patient_indices])
    selected = np.flatnonzero(
        images.patient_alias.isin(aliases) & images.plane.eq(plane)
    )
    counts = images.iloc[selected].groupby("patient_alias").size()
    weights = images.patient_alias.iloc[selected].map(1.0 / counts).to_numpy(float)
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
    pipeline.fit(values[selected], labels[selected], lr__sample_weight=weights)
    return pipeline


def plane_kidney_scores(
    image_probability: np.ndarray,
    images: pd.DataFrame,
    kidneys: pd.DataFrame,
    plane: str,
) -> np.ndarray:
    selected = images[images.plane.eq(plane)]
    lookup = {
        alias: float(image_mil.sigmoid(image_mil.logit(image_probability[rows.index]).mean()))
        for alias, rows in selected.groupby("kidney_alias", sort=False)
    }
    return np.asarray([lookup.get(alias, np.nan) for alias in kidneys.kidney_alias], dtype=float)


def combine_views(longitudinal: np.ndarray, transverse: np.ndarray, weight: float) -> np.ndarray:
    longitudinal = np.asarray(longitudinal, dtype=float)
    transverse = np.asarray(transverse, dtype=float)
    if longitudinal.shape != transverse.shape:
        raise ValueError("View score arrays must have matching shapes")
    output = np.full(longitudinal.shape, np.nan, dtype=float)
    both = np.isfinite(longitudinal) & np.isfinite(transverse)
    output[both] = image_mil.sigmoid(
        weight * image_mil.logit(longitudinal[both])
        + (1.0 - weight) * image_mil.logit(transverse[both])
    )
    only_longitudinal = np.isfinite(longitudinal) & ~np.isfinite(transverse)
    only_transverse = ~np.isfinite(longitudinal) & np.isfinite(transverse)
    output[only_longitudinal] = longitudinal[only_longitudinal]
    output[only_transverse] = transverse[only_transverse]
    if not np.isfinite(output).all():
        raise ValueError("A kidney has neither longitudinal nor transverse evidence")
    return output


def oof_plane_predictions(
    values: np.ndarray,
    labels: np.ndarray,
    images: pd.DataFrame,
    patients: pd.DataFrame,
    train_patients: np.ndarray,
    plane: str,
    regularization: float,
    seed: int,
) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold

    codes = patients.DMSA_APN.to_numpy()[train_patients]
    folds = StratifiedKFold(4, shuffle=True, random_state=seed)
    probability = np.full(len(images), np.nan, dtype=float)
    for fold, (inner_train, inner_validation) in enumerate(
        folds.split(train_patients, codes), start=1
    ):
        model = fit_plane_head(
            values,
            labels,
            images,
            patients,
            train_patients[inner_train],
            plane,
            regularization,
            seed + fold,
        )
        validation_aliases = set(patients.patient_alias.iloc[train_patients[inner_validation]])
        selected = np.flatnonzero(
            images.patient_alias.isin(validation_aliases) & images.plane.eq(plane)
        )
        probability[selected] = model.predict_proba(values[selected])[:, 1]
    return probability


def select_configuration(
    values: np.ndarray,
    image_labels: np.ndarray,
    images: pd.DataFrame,
    patients: pd.DataFrame,
    kidneys: pd.DataFrame,
    train_patients: np.ndarray,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    train_aliases = set(patients.patient_alias.iloc[train_patients])
    train_kidneys = np.flatnonzero(kidneys.patient_alias.isin(train_aliases))
    cache: dict[tuple[str, float], np.ndarray] = {}
    for plane in ("longitudinal", "transverse"):
        for regularization in REGULARIZATION:
            probability = oof_plane_predictions(
                values,
                image_labels,
                images,
                patients,
                train_patients,
                plane,
                regularization,
                seed + (0 if plane == "longitudinal" else 100),
            )
            cache[(plane, regularization)] = plane_kidney_scores(
                probability, images, kidneys, plane
            )
            outside_train = ~kidneys.patient_alias.isin(train_aliases).to_numpy()
            cache[(plane, regularization)][outside_train] = 0.5
    rows: list[dict] = []
    labels = kidneys.kidney_APN.to_numpy(int)
    for c_longitudinal in REGULARIZATION:
        for c_transverse in REGULARIZATION:
            for weight in LONGITUDINAL_WEIGHTS:
                score = combine_views(
                    cache[("longitudinal", c_longitudinal)],
                    cache[("transverse", c_transverse)],
                    weight,
                )
                result = base.metric(labels[train_kidneys], score[train_kidneys])
                rows.append(
                    {
                        "seed": seed,
                        "C_longitudinal": c_longitudinal,
                        "C_transverse": c_transverse,
                        "longitudinal_weight": weight,
                        "oof_auroc": result["auroc"],
                        "oof_ap": result["ap"],
                        "selection_score": (result["auroc"] + result["ap"]) / 2.0,
                    }
                )
    selected = max(
        rows,
        key=lambda row: (
            row["selection_score"],
            -abs(row["longitudinal_weight"] - 0.5),
            -row["C_longitudinal"],
            -row["C_transverse"],
        ),
    )
    frame = pd.DataFrame(rows)
    frame["selected"] = (
        frame.C_longitudinal.eq(selected["C_longitudinal"])
        & frame.C_transverse.eq(selected["C_transverse"])
        & frame.longitudinal_weight.eq(selected["longitudinal_weight"])
    )
    return selected, frame


def run() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve completed URI view experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    patients, kidneys, images = cohort.load_reference()
    images = images.copy()
    images["plane"] = np.where(images["view"].str.endswith("L"), "longitudinal", "transverse")
    embeddings = np.load(uri.OUT / "official_fold_embeddings.npy")
    pathology = np.load(uri.OUT / "official_fold_pathology_scores.npy")
    values = np.c_[embeddings.mean(1), pathology.mean(1), pathology.std(1)].astype(np.float32)
    image_labels = images.kidney_alias.map(
        kidneys.set_index("kidney_alias").kidney_APN
    ).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    references = pd.read_csv(REFERENCE)
    references = references[references.group.eq(URI_GROUP)].set_index(
        ["seed", "kidney_alias"], verify_integrity=True
    )
    metrics: list[dict] = []
    predictions: list[dict] = []
    selections: list[pd.DataFrame] = []
    for seed in base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        patient_roles = patients.patient_alias.map(mapping)
        kidney_roles = kidneys.patient_alias.map(mapping)
        train_patients = np.flatnonzero(patient_roles.eq("train"))
        development = np.flatnonzero(kidney_roles.eq("development"))
        selected, selection = select_configuration(
            values, image_labels, images, patients, kidneys, train_patients, seed
        )
        selections.append(selection)
        all_image_probability: dict[str, np.ndarray] = {}
        models = {}
        for plane, key in (
            ("longitudinal", "C_longitudinal"),
            ("transverse", "C_transverse"),
        ):
            model = fit_plane_head(
                values,
                image_labels,
                images,
                patients,
                train_patients,
                plane,
                float(selected[key]),
                seed,
            )
            selected_images = np.flatnonzero(images.plane.eq(plane))
            probability = np.full(len(images), np.nan, dtype=float)
            probability[selected_images] = model.predict_proba(values[selected_images])[:, 1]
            all_image_probability[plane] = probability
            models[plane] = model
        score = combine_views(
            plane_kidney_scores(
                all_image_probability["longitudinal"], images, kidneys, "longitudinal"
            ),
            plane_kidney_scores(
                all_image_probability["transverse"], images, kidneys, "transverse"
            ),
            float(selected["longitudinal_weight"]),
        )[development]
        target = kidneys.iloc[development]
        fixed_uri = references.loc[
            [(seed, alias) for alias in target.kidney_alias], "score"
        ].to_numpy(float)
        draw_scores = {
            "official_uri_fixed_reference": fixed_uri,
            "official_uri_view_ensemble": score,
            "official_uri_view_ensemble_20pct_plus_reference": 0.20 * score + 0.80 * fixed_uri,
            "official_uri_view_ensemble_25pct_plus_reference": 0.25 * score + 0.75 * fixed_uri,
        }
        joblib.dump(
            {"models": models, "selection": selected},
            OUT / "models" / f"{seed}_uri_view_ensemble.joblib",
            compress=3,
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
        print(
            "URI view draw "
            f"{seed}: C_L={selected['C_longitudinal']:g}, "
            f"C_T={selected['C_transverse']:g}, w_L={selected['longitudinal_weight']:g}",
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
    pd.concat(selections, ignore_index=True).to_csv(
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
    bag_planes = images.groupby("kidney_alias").plane.nunique()
    write_json(
        OUT / "protocol_and_decision.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "motivation": "Fit separate frozen-URI heads for longitudinal and transverse evidence before kidney fusion",
            "selection": "Four-fold patient OOF selection of two C values and the longitudinal logit weight",
            "regularization": REGULARIZATION,
            "longitudinal_weights": LONGITUDINAL_WEIGHTS,
            "patients": len(patients),
            "kidneys": len(kidneys),
            "images": len(images),
            "kidneys_with_both_planes": int((bag_planes == 2).sum()),
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
            },
        },
    )
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=4):
        run()
