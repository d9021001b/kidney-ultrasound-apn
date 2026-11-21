"""Patient-structured four-state APN model using tile and USFM evidence."""

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
import nckuh_dmsa_multiquery_mil_20261103 as cohort
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
TILE = ROOT / "02_experiments/NCKUH_DMSA_native_tile_probe_20261103"
USFM = ROOT / "02_experiments/NCKUH_DMSA_usfm_frozen_20261104"
FIXED = ROOT / "02_experiments/NCKUH_DMSA_tile_replacement_20261103"
OUT = ROOT / "02_experiments/NCKUH_DMSA_hierarchical_tile_usfm_20261104"
GROUPS = {
    "tile": "hierarchical_tile",
    "usfm": "hierarchical_usfm",
    "tile_usfm": "hierarchical_tile_usfm",
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def image_representations(images: pd.DataFrame) -> dict[str, np.ndarray]:
    with np.load(TILE / "tile_features.npz") as bundle:
        tile = bundle["mean"].astype(np.float32)
    usfm = np.load(USFM / "usfm_gap.npy").astype(np.float32)
    if tile.shape != (len(images), 1024) or usfm.shape != (len(images), 768):
        raise ValueError("Unexpected image representation geometry")
    if not np.isfinite(tile).all() or not np.isfinite(usfm).all():
        raise ValueError("Nonfinite image representation")
    return {"tile": tile, "usfm": usfm, "tile_usfm": np.c_[tile, usfm]}


def run() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve completed experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()

    patients, kidneys, images = cohort.load_reference()
    representations = image_representations(images)
    structured = {
        name: hierarchy.structured_views(base.patient_pair_features(values, images, patients))
        for name, values in representations.items()
    }
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    fixed = pd.read_csv(FIXED / "development_predictions.csv")
    fixed = fixed[fixed.group.eq("replace_with_tile_two_heads")].set_index(
        ["seed", "kidney_alias"], verify_integrity=True
    )

    metrics, predictions, selections = [], [], []
    for seed in base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        patient_roles = patients.patient_alias.map(mapping)
        kidney_roles = kidneys.patient_alias.map(mapping)
        train = np.flatnonzero(patient_roles.eq("train"))
        development = np.flatnonzero(kidney_roles.eq("development"))
        target = kidneys.iloc[development]
        if len(train) != 126 or target.patient_alias.nunique() != 42:
            raise ValueError("Unexpected patient split")
        fixed_rows = fixed.loc[[(seed, alias) for alias in target.kidney_alias]]
        np.testing.assert_array_equal(fixed_rows.label, labels[development])
        reference = fixed_rows.score.to_numpy()

        draw_scores = {"fixed_chain_reference": reference}
        for source, (symmetric, antisymmetric) in structured.items():
            probability, models = hierarchy.state_probabilities(
                symmetric, antisymmetric, states, train, seed
            )
            score = hierarchy.kidney_scores(probability, patients, kidneys)[development]
            group = GROUPS[source]
            draw_scores.update({
                group: score,
                f"{group}_quarter_plus_fixed": 0.25 * score + 0.75 * reference,
                f"{group}_equal_plus_fixed": 0.50 * score + 0.50 * reference,
            })
            joblib.dump(models, OUT / "models" / f"{seed}_{source}.joblib", compress=3)
            for head, model in models.items():
                selections.append({
                    "seed": seed, "source": source, "head": head,
                    "inner_cv_auroc": float(model.best_score_), **model.best_params_,
                })

        for group, score in draw_scores.items():
            result = hierarchy.metric(labels[development], score)
            metrics.append({
                "seed": seed, "role": "development", "group": group,
                "patients": 42, "kidneys": len(development), **result,
            })
            for local, index in enumerate(development):
                predictions.append({
                    "seed": seed, "group": group,
                    "patient_alias": kidneys.patient_alias.iloc[index],
                    "kidney_alias": kidneys.kidney_alias.iloc[index],
                    "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                    "side": kidneys.side.iloc[index], "label": int(labels[index]),
                    "score": float(score[local]),
                })
        print(f"Completed hierarchical tile/USFM draw {seed}", flush=True)

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
        prediction_frame, groups, base.SEEDS, "fixed_chain_reference"
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(OUT / "protocol_and_decision.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "motivation": "Target the known bilateral-APN versus normal deficit while preserving unilateral laterality",
        "state_factorization": "P(any APN) x P(bilateral|any) x P(left|unilateral)",
        "representations": {
            "tile": "Mean frozen RadImageNet descriptor over four overlapping contour-constrained image tiles",
            "usfm": "Mean final patch-token descriptor from the frozen official full USFM encoder",
            "tile_usfm": "Concatenation before patient-level symmetric and antisymmetric construction",
        },
        "selection": "Four-fold stratified CV inside each 126-patient training role only",
        "roles": "Three fixed 126-patient training and 42-patient development roles",
        "groups_above_both_development_targets": achieved,
        "exploratory_test_scored": False,
    })
    write_json(OUT / "integrity_audit.json", {
        "patients": len(patients), "kidneys": len(kidneys), "images": len(images),
        "prediction_rows": len(prediction_frame), "prediction_groups": len(groups),
        "exploratory_test_rows_scored": 0,
        "scores_finite": bool(np.isfinite(prediction_frame.score).all()),
        "scores_in_unit_interval": bool(prediction_frame.score.between(0, 1).all()),
    })
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()

