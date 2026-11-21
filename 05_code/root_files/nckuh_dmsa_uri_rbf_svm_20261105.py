"""PCA-RBF-SVM state heads on official URI-CADS fold embeddings."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_uri_rbf_svm_20261105"
SEEDS = (20261102, 20261103, 20261104)
CURRENT_GROUP = "current_tile_usfm_reference"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_reference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference as load

    patients, kidneys, images = load()
    images = images[images.kidney_APN.notna()].copy().reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].copy().reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].copy().reset_index(drop=True)
    if (len(patients), len(kidneys), len(images)) != (210, 414, 820):
        raise ValueError("Unexpected cohort")
    return patients, kidneys, images


def fit_head(values: np.ndarray, labels: np.ndarray, selected: np.ndarray, seed: int):
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold()),
        ("scale", StandardScaler()),
        ("pca", PCA(whiten=True, random_state=seed)),
        ("svm", SVC(
            kernel="rbf", probability=True, class_weight="balanced",
            random_state=seed, cache_size=2048,
        )),
    ])
    search = GridSearchCV(
        pipeline,
        {
            "pca__n_components": [8, 16, 32],
            "svm__C": [0.1, 1.0, 10.0],
            "svm__gamma": ["scale", 0.01],
        },
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=seed),
        n_jobs=1,
        refit=True,
    )
    search.fit(values[selected], labels[selected])
    return search


def state_probabilities(symmetric, antisymmetric, states, train, seed):
    any_label = (states != hierarchy.STATE_CODE["N"]).astype(int)
    any_model = fit_head(symmetric, any_label, train, seed)
    affected = train[states[train] != hierarchy.STATE_CODE["N"]]
    bilateral_label = (states == hierarchy.STATE_CODE["B"]).astype(int)
    bilateral_model = fit_head(symmetric, bilateral_label, affected, seed + 101)
    unilateral = train[np.isin(states[train], [hierarchy.STATE_CODE["L"], hierarchy.STATE_CODE["R"]])]
    left_label = (states == hierarchy.STATE_CODE["L"]).astype(int)
    laterality_model = fit_head(antisymmetric, left_label, unilateral, seed + 202)

    p_any = any_model.predict_proba(symmetric)[:, 1]
    p_bilateral = bilateral_model.predict_proba(symmetric)[:, 1]
    p_left = laterality_model.predict_proba(antisymmetric)[:, 1]
    probability = np.c_[
        1.0 - p_any,
        p_any * (1.0 - p_bilateral) * p_left,
        p_any * (1.0 - p_bilateral) * (1.0 - p_left),
        p_any * p_bilateral,
    ]
    probability /= probability.sum(axis=1, keepdims=True)
    return probability, {
        "any": any_model,
        "bilateral_given_any": bilateral_model,
        "left_given_unilateral": laterality_model,
    }


def run() -> None:
    if OUT.exists():
        files = [path for path in OUT.rglob("*") if path.is_file()]
        if files:
            raise FileExistsError(f"Preserve completed or nonempty partial experiment: {OUT}")
    else:
        OUT.mkdir(parents=True)
    (OUT / "models").mkdir(exist_ok=True)
    patients, kidneys, images = load_reference()
    embedding = np.load(uri.OUT / "official_fold_embeddings.npy")
    if embedding.shape != (820, 5, 2048) or not np.isfinite(embedding).all():
        raise ValueError("Unexpected official URI-CADS embeddings")
    symmetric, antisymmetric = hierarchy.structured_views(
        base.patient_pair_features(embedding.reshape(820, -1), images, patients)
    )
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    fixed = pd.read_csv(uri.OUT / "development_predictions.csv")
    fixed = fixed[fixed.group.isin([CURRENT_GROUP, URI_GROUP])].set_index(
        ["seed", "group", "kidney_alias"], verify_integrity=True
    )

    metrics, predictions, selections = [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train = np.flatnonzero(patients.patient_alias.map(mapping).eq("train"))
        development = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        target = kidneys.iloc[development]
        if len(train) != 126 or target.patient_alias.nunique() != 42:
            raise ValueError("Unexpected split")
        probability, models = state_probabilities(
            symmetric, antisymmetric, states, train, seed
        )
        candidate = hierarchy.kidney_scores(probability, patients, kidneys)[development]
        current = fixed.loc[
            [(seed, CURRENT_GROUP, alias) for alias in target.kidney_alias]
        ].score.to_numpy(float)
        fixed_uri = fixed.loc[
            [(seed, URI_GROUP, alias) for alias in target.kidney_alias]
        ].score.to_numpy(float)
        scores = {
            "current_tile_usfm_reference": current,
            "official_uri_fixed_quarter_reference": fixed_uri,
            "official_uri_embedding_folds_rbf_svm": candidate,
            "official_uri_embedding_folds_rbf_quarter_plus_current": 0.25 * candidate + 0.75 * current,
            "official_uri_embedding_folds_rbf_fifth_plus_current": 0.20 * candidate + 0.80 * current,
        }
        joblib.dump(models, OUT / "models" / f"{seed}_uri_rbf_svm.joblib", compress=3)
        for head, model in models.items():
            selections.append({
                "seed": seed,
                "head": head,
                "inner_cv_auroc": float(model.best_score_),
                **model.best_params_,
            })
        for group, score in scores.items():
            metrics.append({
                "seed": seed,
                "role": "development",
                "group": group,
                "patients": 42,
                "kidneys": len(development),
                **hierarchy.metric(labels[development], score),
            })
            for local, index in enumerate(development):
                predictions.append({
                    "seed": seed,
                    "group": group,
                    "patient_alias": kidneys.patient_alias.iloc[index],
                    "kidney_alias": kidneys.kidney_alias.iloc[index],
                    "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                    "side": kidneys.side.iloc[index],
                    "label": int(labels[index]),
                    "score": float(score[local]),
                })
        print(f"Completed URI RBF-SVM draw {seed}", flush=True)

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
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups, SEEDS, "official_uri_fixed_quarter_reference"
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(OUT / "protocol_and_decision.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "motivation": "Apply the original-study PCA-RBF-SVM classifier family to the strongest current official URI-CADS representation rather than adding another encoder",
        "representation": "Five official URI-CADS whole-kidney fold embeddings concatenated before patient-pair aggregation",
        "state_factorization": "P(any APN) x P(bilateral|any) x P(left|unilateral)",
        "selection": "Four-fold stratified CV inside each 126-patient training role; PCA 8/16/32, C 0.1/1/10, gamma scale/0.01",
        "groups_above_both_development_targets": achieved,
        "exploratory_test_rows_scored": 0,
    })
    write_json(OUT / "integrity_audit.json", {
        "patients": len(patients),
        "kidneys": len(kidneys),
        "images": len(images),
        "prediction_rows": len(prediction_frame),
        "finite_scores": bool(np.isfinite(prediction_frame.score).all()),
        "unit_interval_scores": bool(prediction_frame.score.between(0, 1).all()),
        "exploratory_test_rows_scored": 0,
    })
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
