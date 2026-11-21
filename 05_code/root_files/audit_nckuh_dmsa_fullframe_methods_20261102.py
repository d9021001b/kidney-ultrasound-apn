"""Cross-experiment integrity audit for clean full-frame and TIFF measurement probes."""
from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import nckuh_dmsa_fullframe_clean_probe_20261102 as clean
import nckuh_dmsa_fullframe_deep_probe_20261102 as deep
import nckuh_dmsa_tiff_measurement_probe_20261102 as measure
from nckuh_only_study_20261102 import digest, write_json


def audit_one(name, out, groups, seeds, assignments_path, kidneys, prediction_name,
              metric_name, expected_models, expected_patients, expected_kidneys, expected_images):
    predictions = pd.read_csv(out / prediction_name)
    metrics = pd.read_csv(out / metric_name)
    assignments = pd.read_csv(assignments_path)
    labels = kidneys.set_index("kidney_alias").kidney_APN.astype(int)
    assert set(predictions.group) == set(groups)
    assert set(predictions.seed) == set(seeds)
    assert not predictions.duplicated(["seed", "group", "kidney_alias"]).any()
    assert predictions.label.equals(predictions.kidney_alias.map(labels))
    assert predictions.score.between(0, 1).all() and np.isfinite(predictions.score).all()
    recomputed = []
    for (seed, group), frame in predictions.groupby(["seed", "group"], sort=False):
        dev = set(assignments[(assignments.seed == seed) & (assignments.role == "development")].patient_alias)
        test = set(assignments[(assignments.seed == seed) & (assignments.role == "exploratory_test")].patient_alias)
        assert set(frame.patient_alias) == dev
        assert not set(frame.patient_alias).intersection(test)
        recomputed.append({"seed": seed, "group": group,
                           "auroc": roc_auc_score(frame.label, frame.score),
                           "ap": average_precision_score(frame.label, frame.score)})
    recomputed = pd.DataFrame(recomputed).set_index(["seed", "group"]).sort_index()
    saved = metrics.set_index(["seed", "group"])[["auroc", "ap"]].sort_index()
    maximum_difference = float(np.abs(recomputed.to_numpy() - saved.to_numpy()).max())
    assert maximum_difference < 1e-12
    summary_path = out / ("development_summary.csv" if (out / "development_summary.csv").exists() else "pilot_summary.csv")
    summary = pd.read_csv(summary_path)
    target = summary[(summary.auroc > .8) & (summary.ap > .8)].group.tolist()
    assert not target
    models = len(list((out / "models").glob("*.joblib")))
    assert models == expected_models
    assert (out / "development_confidence_intervals.csv").is_file()
    lock = json.loads((out / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(out / "protocol.json") == lock["protocol_sha256"]
    result = {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "status": "pass",
        "experiment": name, "patients": expected_patients, "kidneys": expected_kidneys,
        "images": expected_images, "seeds": list(seeds), "groups": len(groups),
        "prediction_rows": len(predictions), "saved_models": models,
        "duplicate_seed_group_kidney_rows": 0, "exploratory_test_aliases_scored": 0,
        "label_mismatches": 0, "max_recomputed_metric_difference": maximum_difference,
        "development_groups_above_both_targets": target,
    }
    write_json(out / "integrity_audit.json", result)
    return result


def main():
    _, clean_kidneys, _ = clean.matched_reference()
    _, measurement_kidneys, _ = measure.matched_tiff_reference()
    results = [
        audit_one("clean_fullframe_radiomics", clean.OUT, clean.GROUPS, clean.SEEDS,
                  clean.OUT / "patient_assignments.csv", clean_kidneys,
                  "development_predictions.csv", "development_metrics.csv", 18, 111, 212, 418),
        audit_one("clean_fullframe_frozen_encoders", deep.OUT, deep.GROUPS, clean.SEEDS,
                  clean.OUT / "patient_assignments.csv", clean_kidneys,
                  "development_predictions.csv", "development_metrics.csv", 12, 111, 212, 418),
        audit_one("tiff_physical_measurement", measure.OUT, measure.GROUPS, clean.SEEDS,
                  measure.OUT / "patient_assignments.csv", measurement_kidneys,
                  "development_predictions.csv", "development_metrics.csv", 15, 81, 159, 310),
    ]
    write_json(clean.ROOT / "04_quality_archive/NCKUH_fullframe_methods_integrity_20261102.json",
               {"completed_utc": datetime.now(timezone.utc).isoformat(), "experiments": results})
    print("Full-frame and TIFF method integrity audits passed", flush=True)


if __name__ == "__main__":
    main()
