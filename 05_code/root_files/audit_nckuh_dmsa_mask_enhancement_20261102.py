"""Integrity audit for the completed NCKUH mask-enhancement development pilot."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import nckuh_dmsa_mask_enhancement_20261102 as study
from nckuh_only_study_20261102 import digest, write_json


def main() -> None:
    out = study.OUT
    predictions = pd.read_csv(out / "pilot_predictions.csv")
    metrics = pd.read_csv(out / "pilot_metrics.csv")
    assignments = pd.read_csv(study.BASE / "patient_assignments.csv")
    _, kidneys, images_all = study.load_reference()
    images = images_all[images_all.kidney_APN.notna()].reset_index(drop=True)
    label_by_kidney = kidneys.set_index("kidney_alias").kidney_APN.astype(int)

    assert set(predictions.group) == set(study.GROUPS)
    assert set(predictions.seed) == set(study.PILOT_SEEDS)
    assert not predictions.duplicated(["seed", "group", "kidney_alias"]).any()
    assert predictions.label.equals(predictions.kidney_alias.map(label_by_kidney))
    assert np.isfinite(predictions.score).all() and predictions.score.between(0, 1).all()

    recomputed = []
    for (seed, group), frame in predictions.groupby(["seed", "group"], sort=False):
        dev_aliases = set(assignments[(assignments.seed == seed) & (assignments.role == "development")].patient_alias)
        test_aliases = set(assignments[(assignments.seed == seed) & (assignments.role == "exploratory_test")].patient_alias)
        assert set(frame.patient_alias) == dev_aliases
        assert not set(frame.patient_alias).intersection(test_aliases)
        recomputed.append({"seed": seed, "group": group,
                           "auroc": roc_auc_score(frame.label, frame.score),
                           "ap": average_precision_score(frame.label, frame.score)})
    recomputed = pd.DataFrame(recomputed).set_index(["seed", "group"]).sort_index()
    saved = metrics.set_index(["seed", "group"])[["auroc", "ap"]].sort_index()
    max_metric_difference = float(np.abs(recomputed.to_numpy() - saved.to_numpy()).max())
    assert max_metric_difference < 1e-12

    lock = json.loads((out / "run_lock.json").read_text(encoding="utf-8"))
    locked_source = out / "locked_source_before_bootstrap_encoding_fix.py"
    source_key = str(Path(study.__file__).resolve().relative_to(study.ROOT)).replace("\\", "/")
    assert digest(locked_source) == lock["inputs"][source_key]
    assert digest(out / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        if name == source_key:
            continue
        assert digest(study.ROOT / name) == expected, name

    feature_rows = {}
    for variant in study.VARIANTS:
        frame = pd.read_csv(out / f"radiomics_{variant}.csv")
        assert frame.image_id.tolist() == images.image_id.tolist()
        assert len(frame) == 820
        feature_rows[variant] = len(frame)
    outside_nonzero = 0
    for row in images.itertuples(index=False):
        mask = cv2.imread(str(study.FEATURES / "masks" / f"{row.image_id}.png"), 0) > 0
        for variant in study.VARIANTS:
            image = cv2.imread(str(out / "images" / variant / f"{row.image_id}.png"), 0)
            outside_nonzero += int(np.count_nonzero(image[~mask]))
    assert outside_nonzero == 0

    model_count = len(list((out / "models").glob("*.joblib")))
    assert model_count == 27
    summary = pd.read_csv(out / "pilot_summary.csv")
    target_groups = summary[(summary.auroc > .8) & (summary.ap > .8)].group.tolist()
    assert not target_groups
    ci = pd.read_csv(out / "pilot_confidence_intervals.csv")
    assert set(ci.group) == set(study.GROUPS) | {study.REFERENCE}

    write_json(out / "integrity_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "patients": 210,
        "imaged_kidneys": 414,
        "labeled_images": 820,
        "seeds": list(study.PILOT_SEEDS),
        "groups": len(study.GROUPS),
        "prediction_rows": len(predictions),
        "duplicate_seed_group_kidney_rows": 0,
        "exploratory_test_aliases_scored": 0,
        "label_mismatches": 0,
        "max_recomputed_metric_difference": max_metric_difference,
        "feature_rows": feature_rows,
        "outside_mask_nonzero_pixels": outside_nonzero,
        "saved_models": model_count,
        "development_groups_above_both_targets": target_groups,
        "locked_training_source_sha256": digest(locked_source),
    })
    print("Mask-enhancement integrity audit passed", flush=True)


if __name__ == "__main__":
    main()
