"""Label-free URI-CADS APN-compatible semantic score control."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_uricads_semantic_score_20261108"
REFERENCE = uri.OUT / "development_predictions.csv"
REFERENCE_GROUP = "official_uri_embedding_folds_quarter_plus_current"
CLASS_INDICES = (0, 1, 2, 6)
CLASS_NAMES = (
    "pathological",
    "poor_corticomedullary_differentiation",
    "hyperechogenicity",
    "other_local_lesion",
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def semantic_image_scores(scores: np.ndarray) -> dict[str, np.ndarray]:
    """Combine five-fold URI probabilities without using NCKUH outcomes."""
    if scores.ndim != 3 or scores.shape[1:] != (5, 7):
        raise ValueError(f"Unexpected URI score shape: {scores.shape}")
    selected = np.clip(scores[:, :, CLASS_INDICES], 0.0, 1.0)
    values = {
        "semantic_mean": selected.mean(axis=(1, 2)),
        "semantic_max": selected.max(axis=2).mean(axis=1),
        "semantic_union": (1.0 - np.prod(1.0 - selected, axis=2)).mean(axis=1),
    }
    if not all(np.isfinite(value).all() and ((value >= 0) & (value <= 1)).all() for value in values.values()):
        raise ValueError("Invalid semantic probability")
    return values


def aggregate_kidneys(
    image_score: np.ndarray, images: pd.DataFrame, kidneys: pd.DataFrame, mode: str
) -> np.ndarray:
    values = pd.Series(image_score, index=images.kidney_alias)
    if mode == "mean":
        grouped = values.groupby(level=0).mean()
    elif mode == "max":
        grouped = values.groupby(level=0).max()
    else:
        raise ValueError(mode)
    result = kidneys.kidney_alias.map(grouped).to_numpy(float)
    if not np.isfinite(result).all():
        raise ValueError("Missing kidney semantic score")
    return result


def evaluate() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve completed semantic control: {OUT}")
    OUT.mkdir(parents=True)
    patients, kidneys, images = cohort.load_reference()
    raw = np.load(uri.OUT / "official_fold_pathology_scores.npy")
    image_scores = semantic_image_scores(raw)
    kidney_scores = {
        f"{name}_kidney_{mode}": aggregate_kidneys(values, images, kidneys, mode)
        for name, values in image_scores.items()
        for mode in ("mean", "max")
    }
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    reference = pd.read_csv(REFERENCE)
    reference = reference[reference.group.eq(REFERENCE_GROUP)].set_index(
        ["seed", "kidney_alias"], verify_integrity=True
    )
    predictions, metrics = [], []
    for seed in base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        development = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        target = kidneys.iloc[development]
        fixed = reference.loc[
            [(seed, alias) for alias in target.kidney_alias], "score"
        ].to_numpy(float)
        groups = {"official_uri_fixed_reference": fixed}
        for name, values in kidney_scores.items():
            score = values[development]
            groups[name] = score
            groups[f"{name}_20pct_plus_uri"] = 0.20 * score + 0.80 * fixed
        for group, score in groups.items():
            current = base.metric(target.kidney_APN.to_numpy(int), score)
            metrics.append(
                {
                    "seed": seed,
                    "role": "development",
                    "group": group,
                    "patients": target.patient_alias.nunique(),
                    "kidneys": len(target),
                    **current,
                }
            )
            for row, value in zip(target.itertuples(index=False), score):
                predictions.append(
                    {
                        "seed": seed,
                        "group": group,
                        "patient_alias": row.patient_alias,
                        "kidney_alias": row.kidney_alias,
                        "DMSA_APN": row.DMSA_APN,
                        "side": row.side,
                        "label": int(row.kidney_APN),
                        "score": float(value),
                    }
                )
        print(f"URI semantic control draw {seed} complete", flush=True)
    prediction_frame = pd.DataFrame(predictions)
    metric_frame = pd.DataFrame(metrics)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv")
    groups = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups, base.SEEDS, "official_uri_fixed_reference"
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[
        summary.auroc_mean.gt(0.8) & summary.ap_mean.gt(0.8)
    ].index.tolist()
    write_json(
        OUT / "decision.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "class_indices": list(CLASS_INDICES),
            "class_names": list(CLASS_NAMES),
            "score_formulas": ["mean", "maximum", "probabilistic union"],
            "kidney_aggregation": ["mean across views", "maximum across views"],
            "labels_used_in_score_construction": [],
            "groups_above_both_development_targets": achieved,
            "promotion": bool(achieved),
            "exploratory_test_rows_scored": 0,
        },
    )
    print(summary[["auroc_mean", "ap_mean", "brier_mean"]].to_string(), flush=True)


if __name__ == "__main__":
    evaluate()
