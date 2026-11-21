"""Nested LightGBM four-state heads on official URI-CADS representations."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_uri_lightgbm_state_20261107"
REFERENCE = uri.OUT / "development_predictions.csv"
REFERENCE_GROUP = "official_uri_embedding_folds_quarter_plus_current"
SEEDS = tuple(base.SEEDS)
STATE_CODE = hierarchy.STATE_CODE
CONFIGURATIONS = tuple(product((8, 16, 32), (3, 7)))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def state_kidney_rows(states: np.ndarray) -> np.ndarray:
    """Return left/right binary kidney labels for N/L/R/B patient states."""
    left = np.isin(states, [STATE_CODE["L"], STATE_CODE["B"]]).astype(int)
    right = np.isin(states, [STATE_CODE["R"], STATE_CODE["B"]]).astype(int)
    return np.column_stack([left, right])


def probability_kidney_rows(probability: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map arbitrary class-column order to left/right APN probabilities."""
    aligned = np.zeros((len(probability), 4), dtype=float)
    for column, label in enumerate(classes):
        aligned[:, int(label)] = probability[:, column]
    return np.column_stack([
        aligned[:, STATE_CODE["L"]] + aligned[:, STATE_CODE["B"]],
        aligned[:, STATE_CODE["R"]] + aligned[:, STATE_CODE["B"]],
    ])


def metric_from_states(states: np.ndarray, score: np.ndarray) -> dict[str, float]:
    labels = state_kidney_rows(states).reshape(-1)
    values = score.reshape(-1)
    return {
        "auroc": float(roc_auc_score(labels, values)),
        "ap": float(average_precision_score(labels, values)),
    }


def make_pipeline(seed: int, components: int, leaves: int) -> Pipeline:
    from lightgbm import LGBMClassifier

    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=components, whiten=True, random_state=seed)),
        ("model", LGBMClassifier(
            objective="multiclass", num_class=4, n_estimators=240,
            learning_rate=0.025, num_leaves=leaves, max_depth=2 if leaves == 3 else 3,
            min_child_samples=12, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=3.0,
            class_weight="balanced", random_state=seed, n_jobs=4, verbosity=-1,
        )),
    ])


def nested_select(values: np.ndarray, states: np.ndarray, train: np.ndarray, seed: int):
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 700)
    records = []
    for components, leaves in CONFIGURATIONS:
        fold_metrics = []
        for fold, (local_train, local_validation) in enumerate(splitter.split(train, states[train])):
            fit = train[local_train]
            validation = train[local_validation]
            model = make_pipeline(seed + fold, components, leaves)
            model.fit(values[fit], states[fit])
            probability = model.predict_proba(values[validation])
            score = probability_kidney_rows(probability, model.named_steps["model"].classes_)
            fold_metrics.append(metric_from_states(states[validation], score))
        auroc = float(np.mean([row["auroc"] for row in fold_metrics]))
        ap = float(np.mean([row["ap"] for row in fold_metrics]))
        records.append({
            "components": components, "leaves": leaves,
            "inner_auroc": auroc, "inner_ap": ap, "objective": (auroc + ap) / 2.0,
        })
    selection = pd.DataFrame(records).sort_values(
        ["objective", "inner_auroc", "components", "leaves"],
        ascending=[False, False, True, True],
    ).iloc[0].to_dict()
    model = make_pipeline(seed, int(selection["components"]), int(selection["leaves"]))
    model.fit(values[train], states[train])
    return model, selection, records


def representations(images: pd.DataFrame, patients: pd.DataFrame) -> dict[str, np.ndarray]:
    embeddings = np.load(uri.OUT / "official_fold_embeddings.npy")
    pathology = np.load(uri.OUT / "official_fold_pathology_scores.npy")
    mean_sd = np.concatenate([
        embeddings.mean(1), embeddings.std(1), pathology.mean(1), pathology.std(1),
    ], axis=1)
    all_folds = np.concatenate([
        embeddings.reshape(len(embeddings), -1), pathology.reshape(len(pathology), -1),
    ], axis=1)
    return {
        "uri_fold_mean_sd": hierarchy.structured_views(
            base.patient_pair_features(mean_sd, images, patients)
        )[0],
        "uri_all_folds": hierarchy.structured_views(
            base.patient_pair_features(all_folds, images, patients)
        )[0],
    }


def run() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve completed or partial experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    patients, kidneys, images = cohort.load_reference()
    feature_sets = representations(images, patients)
    states = patients.DMSA_APN.map(STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    reference = pd.read_csv(REFERENCE)
    reference = reference[reference.group.eq(REFERENCE_GROUP)].set_index(
        ["seed", "kidney_alias"], verify_integrity=True,
    )
    patient_lookup = {alias: index for index, alias in enumerate(patients.patient_alias)}
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
        draw_scores = {"official_uri_fixed_reference": fixed}
        for source, values in feature_sets.items():
            model, selection, records = nested_select(values, states, train, seed)
            probability = model.predict_proba(values)
            patient_scores = probability_kidney_rows(
                probability, model.named_steps["model"].classes_,
            )
            score = np.asarray([
                patient_scores[patient_lookup[row.patient_alias], 0 if row.side == "left" else 1]
                for row in target.itertuples(index=False)
            ])
            group = f"lightgbm_state_{source}"
            draw_scores[group] = score
            draw_scores[f"{group}_20pct_plus_uri"] = 0.20 * score + 0.80 * fixed
            draw_scores[f"{group}_25pct_plus_uri"] = 0.25 * score + 0.75 * fixed
            joblib.dump(model, OUT / "models" / f"{seed}_{source}.joblib", compress=3)
            selection_rows.append({"seed": seed, "source": source, **selection})
            inner_rows.extend({"seed": seed, "source": source, **row} for row in records)
        for group, score in draw_scores.items():
            current = hierarchy.metric(labels[development], score)
            metrics_rows.append({
                "seed": seed, "role": "development", "group": group,
                "patients": target.patient_alias.nunique(), "kidneys": len(target), **current,
            })
            for row, current_score in zip(target.itertuples(index=False), score):
                predictions.append({
                    "seed": seed, "group": group, "patient_alias": row.patient_alias,
                    "kidney_alias": row.kidney_alias, "DMSA_APN": row.DMSA_APN,
                    "side": row.side, "label": int(row.kidney_APN), "score": float(current_score),
                })
        print(f"URI LightGBM development draw {seed} complete", flush=True)
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
        "reference": REFERENCE_GROUP,
    })
    write_json(OUT / "integrity_audit.json", {
        "patients": len(patients), "kidneys": len(kidneys), "images": len(images),
        "feature_shapes": {name: list(values.shape) for name, values in feature_sets.items()},
        "prediction_roles": sorted(metric_frame.role.unique()),
        "exploratory_test_rows_scored": 0,
        "scores_finite": bool(np.isfinite(prediction_frame.score).all()),
        "scores_in_unit_interval": bool(prediction_frame.score.between(0, 1).all()),
    })
    print(summary[["auroc_mean", "ap_mean", "brier_mean"]].to_string(), flush=True)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=4):
        run()
