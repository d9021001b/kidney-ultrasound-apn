"""Evaluate a patient-structured hierarchical APN model on NCKUH development draws."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_fullframe_20261103"
OUT = ROOT / "02_experiments/NCKUH_DMSA_hierarchical_paired_20261103"
SEEDS = (20261102, 20261103, 20261104)
STATE_CODE = {"N": 0, "L": 1, "R": 2, "B": 3}
GROUPS = {
    "rad": "hierarchical_radimagenet_paired",
    "deep": "hierarchical_radimagenet_dinov3_paired",
}
REFERENCE_GROUP = "fullcohort_rad_color_image_plus_patient_direct"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def structured_views(patient_pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return side-invariant disease features and signed left-right features."""
    half = patient_pairs.shape[1] // 2
    left, right = patient_pairs[:, :half], patient_pairs[:, half:]
    symmetric = np.c_[(left + right) / 2.0, np.abs(left - right)]
    antisymmetric = left - right
    return symmetric.astype(np.float32), antisymmetric.astype(np.float32)


def fit_head(values: np.ndarray, labels: np.ndarray, selected: np.ndarray, seed: int):
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold()),
        ("scale", StandardScaler()),
        ("pca", PCA(whiten=True, random_state=seed)),
        ("lr", LogisticRegression(
            class_weight="balanced", solver="liblinear", max_iter=4000, random_state=seed,
        )),
    ])
    search = GridSearchCV(
        pipeline,
        {
            "pca__n_components": [8, 16, 32],
            "lr__C": [0.01, 0.1, 1.0],
        },
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=seed),
        n_jobs=max(1, int(os.environ.get("NCKUH_GRID_N_JOBS", "1"))),
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
):
    any_label = (states != STATE_CODE["N"]).astype(int)
    any_model = fit_head(symmetric, any_label, train, seed)

    affected_train = train[states[train] != STATE_CODE["N"]]
    bilateral_label = (states == STATE_CODE["B"]).astype(int)
    bilateral_model = fit_head(symmetric, bilateral_label, affected_train, seed + 101)

    unilateral_train = train[np.isin(states[train], [STATE_CODE["L"], STATE_CODE["R"]])]
    left_label = (states == STATE_CODE["L"]).astype(int)
    laterality_model = fit_head(antisymmetric, left_label, unilateral_train, seed + 202)

    p_any = any_model.predict_proba(symmetric)[:, 1]
    p_bilateral_given_any = bilateral_model.predict_proba(symmetric)[:, 1]
    p_left_given_unilateral = laterality_model.predict_proba(antisymmetric)[:, 1]

    p_n = 1.0 - p_any
    p_b = p_any * p_bilateral_given_any
    p_unilateral = p_any * (1.0 - p_bilateral_given_any)
    p_l = p_unilateral * p_left_given_unilateral
    p_r = p_unilateral * (1.0 - p_left_given_unilateral)
    probability = np.c_[p_n, p_l, p_r, p_b]
    probability /= probability.sum(axis=1, keepdims=True)
    models = {
        "any": any_model,
        "bilateral_given_any": bilateral_model,
        "left_given_unilateral": laterality_model,
    }
    return probability, models


def kidney_scores(probability: np.ndarray, patients: pd.DataFrame, kidneys: pd.DataFrame) -> np.ndarray:
    lookup = {alias: index for index, alias in enumerate(patients.patient_alias)}
    left = probability[:, STATE_CODE["L"]] + probability[:, STATE_CODE["B"]]
    right = probability[:, STATE_CODE["R"]] + probability[:, STATE_CODE["B"]]
    return np.asarray([
        left[lookup[row.patient_alias]] if row.side == "left" else right[lookup[row.patient_alias]]
        for row in kidneys.itertuples(index=False)
    ])


def metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
    }


def run() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve completed experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()

    sys.path.insert(0, str(Path(__file__).parent))
    from nckuh_dmsa_fullcohort_fullframe_20261103 import patient_pair_features
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference

    patients, kidneys, images = load_reference()
    images = images[images.kidney_APN.notna()].copy().reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].copy().reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].copy().reset_index(drop=True)
    assert (len(patients), len(kidneys), len(images)) == (210, 414, 820)

    rad = np.load(SOURCE / "radimagenet_color.npy")
    deep = np.c_[rad, np.load(SOURCE / "dinov3_class.npy")]
    feature_sets = {
        "rad": structured_views(patient_pair_features(rad, images, patients)),
        "deep": structured_views(patient_pair_features(deep, images, patients)),
    }
    states = patients.DMSA_APN.map(STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(SOURCE / "patient_assignments.csv")
    reference = pd.read_csv(SOURCE / "instance_development_predictions.csv")
    reference = reference[reference.group.eq(REFERENCE_GROUP)].set_index(["seed", "kidney_alias"]).score

    predictions: list[dict] = []
    metrics: list[dict] = []
    selections: list[dict] = []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        patient_roles = patients.patient_alias.map(mapping)
        kidney_roles = kidneys.patient_alias.map(mapping)
        train = np.flatnonzero(patient_roles.eq("train"))
        development = np.flatnonzero(kidney_roles.eq("development"))
        assert len(train) == 126
        assert kidneys.patient_alias.iloc[development].nunique() == 42
        assert not kidney_roles.iloc[development].eq("exploratory_test").any()

        reference_score = np.asarray([
            reference.loc[(seed, alias)] for alias in kidneys.kidney_alias.iloc[development]
        ])
        for source, (symmetric, antisymmetric) in feature_sets.items():
            probability, models = state_probabilities(symmetric, antisymmetric, states, train, seed)
            score = kidney_scores(probability, patients, kidneys)[development]
            group = GROUPS[source]
            candidate_scores = {
                group: score,
                f"{group}_plus_instance_equal": (score + reference_score) / 2.0,
                f"{group}_quarter_plus_instance": score * 0.25 + reference_score * 0.75,
            }
            joblib.dump(models, OUT / "models" / f"{seed}_{source}.joblib", compress=3)
            for head, model in models.items():
                selections.append({
                    "seed": seed,
                    "source": source,
                    "head": head,
                    "inner_cv_auroc": float(model.best_score_),
                    **model.best_params_,
                })
            for candidate_group, candidate_score in candidate_scores.items():
                result = metric(labels[development], candidate_score)
                metrics.append({
                    "seed": seed,
                    "role": "development",
                    "group": candidate_group,
                    "patients": 42,
                    "kidneys": len(development),
                    **result,
                })
                for local, index in enumerate(development):
                    predictions.append({
                        "seed": seed,
                        "group": candidate_group,
                        "patient_alias": kidneys.patient_alias.iloc[index],
                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                        "side": kidneys.side.iloc[index],
                        "label": int(labels[index]),
                        "score": float(candidate_score[local]),
                    })
        print(f"Completed hierarchical draw {seed}", flush=True)

    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    selection_frame = pd.DataFrame(selections)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    selection_frame.to_csv(OUT / "training_inner_cv_selections.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(OUT / "decision.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "groups_above_both_development_targets": achieved,
        "exploratory_test_scored": False,
        "split_unit": "patient",
        "selection_unit": "kidney",
        "inner_selection": "Four-fold stratified CV within the 126 training patients only",
        "state_factorization": "P(any APN) x P(bilateral|any) x P(left|unilateral)",
        "reference_group": REFERENCE_GROUP,
    })
    write_json(OUT / "integrity_audit.json", {
        "patients": len(patients),
        "kidneys": len(kidneys),
        "images": len(images),
        "prediction_rows": len(prediction_frame),
        "prediction_roles": ["development"],
        "exploratory_test_rows_scored": 0,
        "probability_scores_finite": bool(np.isfinite(prediction_frame.score).all()),
        "probability_scores_in_unit_interval": bool(prediction_frame.score.between(0, 1).all()),
    })
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    run()
