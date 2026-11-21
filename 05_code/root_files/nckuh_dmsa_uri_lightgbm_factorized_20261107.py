"""Nested factorized LightGBM heads on official URI-CADS representations."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_uri_lightgbm_state_20261107 as state_control
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_uri_lightgbm_factorized_20261107"
REFERENCE = uri.OUT / "development_predictions.csv"
REFERENCE_GROUP = "official_uri_embedding_folds_quarter_plus_current"
SEEDS = tuple(base.SEEDS)
STATE_CODE = hierarchy.STATE_CODE
CONFIGURATIONS = state_control.CONFIGURATIONS


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def compose_probability(p_any: np.ndarray, p_bilateral: np.ndarray, p_left: np.ndarray) -> np.ndarray:
    p_n = 1.0 - p_any
    p_b = p_any * p_bilateral
    p_unilateral = p_any * (1.0 - p_bilateral)
    p_l = p_unilateral * p_left
    p_r = p_unilateral * (1.0 - p_left)
    probability = np.column_stack([p_n, p_l, p_r, p_b])
    probability /= probability.sum(1, keepdims=True)
    return probability


def binary_model(seed: int, leaves: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="binary", n_estimators=240, learning_rate=0.025,
        num_leaves=leaves, max_depth=2 if leaves == 3 else 3,
        min_child_samples=10, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=3.0,
        class_weight="balanced", random_state=seed, n_jobs=4, verbosity=-1,
    )


def fit_models(
    values: np.ndarray, states: np.ndarray, selected: np.ndarray,
    seed: int, components: int, leaves: int,
):
    preprocess = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=components, whiten=True, random_state=seed)),
    ])
    transformed = preprocess.fit_transform(values[selected])
    selected_states = states[selected]
    any_model = binary_model(seed, leaves)
    any_model.fit(transformed, (selected_states != STATE_CODE["N"]).astype(int))
    affected = selected_states != STATE_CODE["N"]
    bilateral_model = binary_model(seed + 101, leaves)
    bilateral_model.fit(
        transformed[affected], (selected_states[affected] == STATE_CODE["B"]).astype(int),
    )
    unilateral = np.isin(selected_states, [STATE_CODE["L"], STATE_CODE["R"]])
    laterality_model = binary_model(seed + 202, leaves)
    laterality_model.fit(
        transformed[unilateral], (selected_states[unilateral] == STATE_CODE["L"]).astype(int),
    )
    return {
        "preprocess": preprocess, "any": any_model,
        "bilateral_given_any": bilateral_model, "left_given_unilateral": laterality_model,
    }


def predict_probability(models: dict, values: np.ndarray) -> np.ndarray:
    transformed = models["preprocess"].transform(values)
    p_any = models["any"].predict_proba(transformed)[:, 1]
    p_bilateral = models["bilateral_given_any"].predict_proba(transformed)[:, 1]
    p_left = models["left_given_unilateral"].predict_proba(transformed)[:, 1]
    return compose_probability(p_any, p_bilateral, p_left)


def nested_select(values: np.ndarray, states: np.ndarray, train: np.ndarray, seed: int):
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 700)
    records = []
    for components, leaves in CONFIGURATIONS:
        fold_metrics = []
        for fold, (local_train, local_validation) in enumerate(splitter.split(train, states[train])):
            fit = train[local_train]
            validation = train[local_validation]
            models = fit_models(values, states, fit, seed + fold, components, leaves)
            probability = predict_probability(models, values[validation])
            score = state_control.probability_kidney_rows(probability, np.arange(4))
            fold_metrics.append(state_control.metric_from_states(states[validation], score))
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
    models = fit_models(
        values, states, train, seed,
        int(selection["components"]), int(selection["leaves"]),
    )
    return models, selection, records


def run() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve completed or partial experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    patients, kidneys, images = cohort.load_reference()
    feature_sets = state_control.representations(images, patients)
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
            models, selection, records = nested_select(values, states, train, seed)
            probability = predict_probability(models, values)
            patient_scores = state_control.probability_kidney_rows(probability, np.arange(4))
            score = np.asarray([
                patient_scores[patient_lookup[row.patient_alias], 0 if row.side == "left" else 1]
                for row in target.itertuples(index=False)
            ])
            group = f"lightgbm_factorized_{source}"
            draw_scores[group] = score
            draw_scores[f"{group}_20pct_plus_uri"] = 0.20 * score + 0.80 * fixed
            draw_scores[f"{group}_25pct_plus_uri"] = 0.25 * score + 0.75 * fixed
            joblib.dump(models, OUT / "models" / f"{seed}_{source}.joblib", compress=3)
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
        print(f"URI factorized LightGBM development draw {seed} complete", flush=True)
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
        "promotion": bool(achieved), "exploratory_test_rows_scored": 0,
        "factorization": "P(any APN) x P(bilateral|any) x P(left|unilateral)",
        "selection": "Four-fold patient-level inner CV within each 126-patient training role",
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
