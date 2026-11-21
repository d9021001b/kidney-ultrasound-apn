"""Audit fixed URI-CADS contralateral-local development predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nckuh_only_study_20261102 import weighted_metrics_batch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_uri_contralateral_local_20261105"
ASSIGNMENTS = (
    ROOT
    / "02_experiments/NCKUH_DMSA_fullcohort_fullframe_20261103/patient_assignments.csv"
)
REFERENCE = "official_uri_fixed_quarter_reference"
SEEDS = (20261102, 20261103, 20261104)


def paired_values(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return affected-minus-unaffected gaps and affected/unaffected scores."""
    rows = []
    for _, patient in frame.groupby("patient_alias", sort=True):
        if len(patient) == 1:
            continue
        if len(patient) != 2 or set(patient.label.astype(int)) != {0, 1}:
            raise ValueError("A complete unilateral pair must contain one kidney per label")
        positive = float(patient.loc[patient.label.astype(int).eq(1), "score"].iloc[0])
        negative = float(patient.loc[patient.label.astype(int).eq(0), "score"].iloc[0])
        rows.append((positive - negative, positive, negative))
    if not rows:
        raise ValueError("No complete unilateral pair")
    values = np.asarray(rows, dtype=float)
    return values[:, 0], values[:, 1], values[:, 2]


def point_metrics(frame: pd.DataFrame) -> dict[str, float]:
    gaps, _, _ = paired_values(frame)
    counts = frame.groupby("patient_alias").size()
    complete = frame[frame.patient_alias.isin(counts[counts.eq(2)].index)]
    return {
        "affected_side_auroc": float(
            roc_auc_score(complete.label.astype(int), complete.score)
        ),
        "paired_correct_fraction": float(np.mean((gaps > 0) + 0.5 * (gaps == 0))),
        "mean_affected_minus_unaffected": float(gaps.mean()),
    }


def bootstrap_intervals(
    predictions: pd.DataFrame, groups: list[str], replicates: int = 10000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unilateral = predictions[predictions.DMSA_APN.isin(["L", "R"])].copy()
    aliases = sorted(unilateral.patient_alias.unique())
    alias_index = {alias: index for index, alias in enumerate(aliases)}
    codes = unilateral.groupby("patient_alias").DMSA_APN.first().reindex(aliases)
    strata = [np.flatnonzero(codes.to_numpy() == code) for code in ("L", "R")]
    if not all(len(values) for values in strata):
        raise ValueError("Both unilateral strata are required")

    point = np.zeros((len(groups), 3), dtype=float)
    seed_data = []
    for seed in SEEDS:
        by_seed = unilateral[unilateral.seed.eq(seed)]
        current = []
        for group_index, group in enumerate(groups):
            frame = by_seed[by_seed.group.eq(group)].copy()
            metrics = point_metrics(frame)
            point[group_index] += np.asarray(list(metrics.values()))
            counts = frame.groupby("patient_alias").size()
            complete = frame[frame.patient_alias.isin(counts[counts.eq(2)].index)]
            key = complete.groupby("kidney_alias", sort=True).first()
            clusters = np.asarray([alias_index[a] for a in key.patient_alias], dtype=int)
            y = key.label.to_numpy(int)
            scores = key.score.to_numpy(float)
            pair = []
            for alias, patient in frame.groupby("patient_alias", sort=True):
                if len(patient) == 1:
                    continue
                gaps, _, _ = paired_values(patient)
                pair.append((alias_index[alias], float(gaps[0])))
            current.append((clusters, y, scores, np.asarray(pair, dtype=float)))
        seed_data.append(current)
    point /= len(SEEDS)

    rng = np.random.default_rng(20261105)
    samples = []
    while sum(len(batch) for batch in samples) < replicates:
        size = min(500, replicates - sum(len(batch) for batch in samples))
        sampled = np.concatenate(
            [rng.choice(stratum, (size, len(stratum)), replace=True) for stratum in strata],
            axis=1,
        )
        counts = np.zeros((size, len(aliases)), dtype=float)
        np.add.at(counts, (np.arange(size)[:, None], sampled), 1)
        values = np.zeros((size, len(groups), 3), dtype=float)
        valid = np.ones(size, dtype=bool)
        for current in seed_data:
            for group_index, (clusters, y, scores, pairs) in enumerate(current):
                kidney_weights = counts[:, clusters]
                has_both = (kidney_weights[:, y == 0].sum(1) > 0) & (
                    kidney_weights[:, y == 1].sum(1) > 0
                )
                valid &= has_both
                values[:, group_index, 0] += weighted_metrics_batch(
                    y, scores, kidney_weights
                )[:, 0]
                pair_clusters = pairs[:, 0].astype(int)
                gaps = pairs[:, 1]
                pair_weights = counts[:, pair_clusters]
                denominator = pair_weights.sum(1)
                valid &= denominator > 0
                correct = (gaps > 0).astype(float) + 0.5 * (gaps == 0)
                values[:, group_index, 1] += (pair_weights * correct).sum(1) / denominator
                values[:, group_index, 2] += (pair_weights * gaps).sum(1) / denominator
        samples.append(values[valid] / len(SEEDS))
    samples_array = np.concatenate(samples)[:replicates]

    metric_names = (
        "affected_side_auroc",
        "paired_correct_fraction",
        "mean_affected_minus_unaffected",
    )
    rows, differences = [], []
    reference_index = groups.index(REFERENCE)
    for group_index, group in enumerate(groups):
        for metric_index, metric in enumerate(metric_names):
            low, high = np.quantile(samples_array[:, group_index, metric_index], [0.025, 0.975])
            rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "mean": point[group_index, metric_index],
                    "low": low,
                    "high": high,
                    "distinct_unilateral_patients": len(aliases),
                    "draws": len(SEEDS),
                    "replicates": replicates,
                }
            )
            if group != REFERENCE:
                delta = (
                    samples_array[:, group_index, metric_index]
                    - samples_array[:, reference_index, metric_index]
                )
                delta_low, delta_high = np.quantile(delta, [0.025, 0.975])
                differences.append(
                    {
                        "group": group,
                        "reference": REFERENCE,
                        "metric": metric,
                        "difference": point[group_index, metric_index]
                        - point[reference_index, metric_index],
                        "low": delta_low,
                        "high": delta_high,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(differences)


def run() -> None:
    predictions = pd.read_csv(OUT / "development_predictions.csv")
    assignments = pd.read_csv(ASSIGNMENTS)
    groups = predictions.group.drop_duplicates().tolist()
    if REFERENCE not in groups:
        raise ValueError("Missing fixed URI-CADS reference")
    if predictions.duplicated(["seed", "group", "kidney_alias"]).any():
        raise ValueError("Duplicate kidney prediction")
    for seed in SEEDS:
        role = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        observed = predictions[predictions.seed.eq(seed)].patient_alias.map(role)
        if not observed.eq("development").all():
            raise ValueError("Nondevelopment prediction detected")

    features = np.load(OUT / "uri_contralateral_local_features.npy", mmap_mode="r")
    dictionary = pd.read_csv(OUT / "feature_dictionary.csv")
    encoder = json.loads((OUT / "encoder_audit.json").read_text(encoding="utf-8"))
    if features.shape != (820, 215) or len(dictionary) != 215:
        raise ValueError("Unexpected local-feature geometry")
    if not np.isfinite(features).all() or encoder.get("labels_used_in_encoding") != []:
        raise ValueError("Invalid or label-dependent encoding")

    detail = []
    unilateral = predictions[predictions.DMSA_APN.isin(["L", "R"])]
    for seed in SEEDS:
        for group in groups:
            metrics = point_metrics(
                unilateral[
                    unilateral.seed.eq(seed) & unilateral.group.eq(group)
                ]
            )
            detail.append({"seed": seed, "group": group, **metrics})
    pd.DataFrame(detail).to_csv(OUT / "unilateral_side_metrics.csv", index=False)
    intervals, differences = bootstrap_intervals(predictions, groups)
    intervals.to_csv(OUT / "unilateral_side_intervals.csv", index=False)
    differences.to_csv(OUT / "unilateral_side_paired_intervals.csv", index=False)
    audit = {
        "status": "pass",
        "images": int(features.shape[0]),
        "features": int(features.shape[1]),
        "groups": len(groups),
        "draws": len(SEEDS),
        "development_prediction_rows": len(predictions),
        "exploratory_test_rows": 0,
        "unilateral_patients_without_complete_pair": sorted(
            predictions[
                predictions.DMSA_APN.isin(["L", "R"])
                & predictions.group.eq(REFERENCE)
            ]
            .groupby("patient_alias")
            .kidney_alias.nunique()
            .loc[lambda values: values.lt(2)]
            .index.tolist()
        ),
        "labels_used_in_encoding": [],
        "decision": "Do not promote; no candidate improved mean AUROC over the fixed URI-CADS reference.",
    }
    (OUT / "integrity_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(
        intervals[intervals.metric.eq("affected_side_auroc")]
        .sort_values("mean", ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    run()
