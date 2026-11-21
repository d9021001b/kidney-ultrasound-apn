"""Audit the completed contour-registered bilateral residual experiment."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import audit_nckuh_dmsa_uri_contralateral_local_20261105 as common
import nckuh_dmsa_registered_asymmetry_20261105 as study


REFERENCE = study.URI_GROUP


def run() -> None:
    predictions = pd.read_csv(study.OUT / "development_predictions.csv")
    groups = predictions.group.drop_duplicates().tolist()
    if REFERENCE not in groups:
        raise ValueError("Missing fixed URI-CADS reference")
    common.REFERENCE = REFERENCE
    details = []
    unilateral = predictions[predictions.DMSA_APN.isin(["L", "R"])]
    for seed in study.SEEDS:
        for group in groups:
            metrics = common.point_metrics(
                unilateral[unilateral.seed.eq(seed) & unilateral.group.eq(group)]
            )
            details.append({"seed": seed, "group": group, **metrics})
    pd.DataFrame(details).to_csv(study.OUT / "unilateral_side_metrics.csv", index=False)
    intervals, differences = common.bootstrap_intervals(predictions, groups)
    intervals.to_csv(study.OUT / "unilateral_side_intervals.csv", index=False)
    differences.to_csv(study.OUT / "unilateral_side_paired_intervals.csv", index=False)

    features = np.load(study.OUT / "registered_residual_patient_features.npy", mmap_mode="r")
    encoder = json.loads((study.OUT / "encoder_audit.json").read_text(encoding="utf-8"))
    assignments = pd.read_csv(study.base.OUT / "patient_assignments.csv")
    if features.shape != (210, 1258) or not np.isfinite(features).all():
        raise ValueError("Invalid registered feature output")
    if encoder.get("labels_used_in_encoding") != []:
        raise ValueError("Labels entered feature encoding")
    for seed in study.SEEDS:
        role = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        observed = predictions[predictions.seed.eq(seed)].patient_alias.map(role)
        if not observed.eq("development").all():
            raise ValueError("Nondevelopment prediction detected")
    audit = {
        "status": "pass",
        "patients": 210,
        "features": 1258,
        "groups": len(groups),
        "draws": len(study.SEEDS),
        "development_prediction_rows": len(predictions),
        "exploratory_test_rows": 0,
        "labels_used_in_encoding": [],
        "decision": "Do not promote; no registered residual configuration improved both AUROC and AP over the fixed URI-CADS reference.",
    }
    (study.OUT / "posthoc_integrity_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(
        intervals[intervals.metric.eq("affected_side_auroc")]
        .sort_values("mean", ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    run()
