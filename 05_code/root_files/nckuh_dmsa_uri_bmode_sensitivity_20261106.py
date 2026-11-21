"""Outcome-blind URI-CADS sensitivity excluding three confirmed Doppler frames."""
from __future__ import annotations

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
OUT = ROOT / "02_experiments/NCKUH_DMSA_uri_bmode_sensitivity_20261106"
DOPPLER_IMAGE_IDS = ("I0619", "I0776", "I0942")
URI_SOURCE = "official_uri_embedding_folds"
CURRENT_GROUP = "current_tile_usfm_reference"
REFERENCE_GROUP = "official_uri_embedding_folds_quarter_plus_current"
CANDIDATE_GROUP = "official_uri_embedding_folds_bmode_only_quarter_plus_current"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def bmode_rows(
    values: np.ndarray, images: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    if len(values) != len(images):
        raise ValueError("Image representation and manifest lengths differ")
    present = set(images.image_id) & set(DOPPLER_IMAGE_IDS)
    if present != set(DOPPLER_IMAGE_IDS):
        raise ValueError(f"Missing registered Doppler images: {set(DOPPLER_IMAGE_IDS) - present}")
    keep = ~images.image_id.isin(DOPPLER_IMAGE_IDS).to_numpy()
    return values[keep], images.loc[keep].reset_index(drop=True)


def run() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Preserve completed or partial sensitivity: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    patients, kidneys, images = cohort.load_reference()
    values = uri.representations()[URI_SOURCE]
    filtered_values, filtered_images = bmode_rows(values, images)
    patient_pairs = base.patient_pair_features(
        filtered_values, filtered_images, patients
    )
    symmetric, antisymmetric = hierarchy.structured_views(patient_pairs)
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    original = pd.read_csv(uri.OUT / "development_predictions.csv")
    reference = original[original.group.eq(REFERENCE_GROUP)].set_index(
        ["seed", "kidney_alias"], verify_integrity=True
    )
    current = original[original.group.eq(CURRENT_GROUP)].set_index(
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
        keys = [(seed, alias) for alias in target.kidney_alias]
        fixed_score = reference.loc[keys, "score"].to_numpy(float)
        current_score = current.loc[keys, "score"].to_numpy(float)
        probability, models = hierarchy.state_probabilities(
            symmetric, antisymmetric, states, train, seed
        )
        bmode_score = hierarchy.kidney_scores(probability, patients, kidneys)[development]
        candidate = 0.25 * bmode_score + 0.75 * current_score
        draw_scores = {
            "official_uri_fixed_reference": fixed_score,
            "official_uri_embedding_folds_bmode_only": bmode_score,
            CANDIDATE_GROUP: candidate,
        }
        joblib.dump(models, OUT / "models" / f"{seed}_{URI_SOURCE}.joblib", compress=3)
        for head, model in models.items():
            selections.append(
                {
                    "seed": seed,
                    "head": head,
                    "inner_cv_auroc": float(model.best_score_),
                    **model.best_params_,
                }
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
        print(f"B-mode sensitivity draw {seed} complete", flush=True)

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
        OUT / "protocol_and_decision.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "motivation": "Remove confirmed Color Doppler frames from the B-mode URI-CADS representation because APN perfusion information and color overlays define a different acquisition modality.",
            "selection_rule": "Outcome-blind visual confirmation after dual red/blue interior-color screening; no APN labels or model scores were consulted.",
            "excluded_image_ids": list(DOPPLER_IMAGE_IDS),
            "excluded_patients": sorted(
                images.loc[images.image_id.isin(DOPPLER_IMAGE_IDS), "patient_alias"].tolist()
            ),
            "images_before": len(images),
            "images_after": len(filtered_images),
            "patients_retained": len(patients),
            "kidneys_retained": len(kidneys),
            "fixed_fusion": "25% filtered URI-CADS plus unchanged 75% current reference",
            "groups_above_both_development_targets": achieved,
            "promotion": bool(achieved),
            "exploratory_test_rows_scored": 0,
        },
    )
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=4):
        run()
