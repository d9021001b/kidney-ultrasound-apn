"""Create the fixed segmented CYCH 80/20 + 5-fold patient protocol.

The input is the existing segmentation handoff, not the raw ultrasound frames.
The holdout is selected once at the patient level and is never used for model
selection. The development patients receive deterministic five-fold labels.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "segmented_protocol_80_20"
SEGMENTATION_MANIFEST = ROOT / "yolo_crops" / "manifest.csv"
LABEL_CSV = ROOT / "CYCH" / "CYCH.csv"
SEED = 42
HOLDOUT_FRACTION = 0.20
FOLDS = 5


def read_label_map() -> dict[str, int]:
    with LABEL_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        result = {}
        for row in rows:
            result[row["filename"]] = int(float(row["label"]))
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = read_label_map()
    seg = pd.read_csv(SEGMENTATION_MANIFEST, dtype=str)
    seg = seg.loc[seg["site"].eq("CYCH")].copy()
    if len(seg) != 353:
        raise ValueError(f"Expected 353 CYCH segmentation rows, found {len(seg)}")
    seg["label"] = seg["label"].astype(int)
    seg["patient_id"] = seg["patient_id"].astype(str)
    seg["exam_date_iso"] = seg["exam_date_iso"].fillna("").astype(str)
    seg["logical_filename"] = seg["filename"].str.replace("_red_", "_", regex=False)
    if not seg["logical_filename"].map(labels).eq(seg["label"]).all():
        raise ValueError("Segmentation manifest labels do not exactly match CYCH.csv")
    if seg["relative_path"].map(lambda p: (ROOT / p).exists()).eq(False).any():
        missing = seg.loc[~seg["relative_path"].map(lambda p: (ROOT / p).exists()), "relative_path"].tolist()
        raise FileNotFoundError(missing[:5])
    if Counter(seg["roi_source"]) != Counter({"yolo": 309, "physician_polygon": 44}):
        raise ValueError("Unexpected ROI source counts")

    # Match the experiment runner: retain only the first exam per patient.
    seg = seg.sort_values(["patient_id", "exam_date_iso", "filename"]).reset_index(drop=True)
    first_exam = seg.groupby("patient_id", sort=False)["exam_date_iso"].transform("min")
    keep = seg["exam_date_iso"].eq(first_exam)
    dropped = seg.loc[~keep].copy()
    data = seg.loc[keep].copy().reset_index(drop=True)

    patient_table = (
        data.groupby("patient_id", as_index=False)
        .agg(label=("label", "first"), exam_date_iso=("exam_date_iso", "first"))
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    if patient_table["label"].nunique() != 2:
        raise ValueError("The protocol requires both patient classes")
    if data.groupby("patient_id")["label"].nunique().max() != 1:
        raise ValueError("A patient has inconsistent image labels")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=HOLDOUT_FRACTION, random_state=SEED)
    dev_idx, holdout_idx = next(splitter.split(patient_table["patient_id"], patient_table["label"]))
    patient_table["split"] = "holdout"
    patient_table.loc[dev_idx, "split"] = "development"
    patient_table["fold"] = -1

    dev = patient_table.loc[patient_table["split"].eq("development")].copy().reset_index()
    fold_splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, val_idx) in enumerate(fold_splitter.split(dev["patient_id"], dev["label"])):
        patient_table.loc[dev.loc[val_idx, "index"], "fold"] = fold
    if (patient_table.loc[patient_table["split"].eq("development"), "fold"] < 0).any():
        raise AssertionError("Some development patient has no fold")

    fold_by_patient = patient_table.set_index("patient_id")[["split", "fold"]]
    data = data.join(fold_by_patient, on="patient_id", rsuffix="_patient")
    data["split"] = data["split"].astype(str)
    data["fold"] = data["fold"].astype(int)
    data["protocol_seed"] = SEED
    data["protocol"] = "patient-level 80% development / 20% external holdout; 5-fold CV within development"
    data.to_csv(OUT / "label_manifest.csv", index=False, encoding="utf-8-sig")
    dropped.to_csv(OUT / "dropped_repeat_exam_rows.csv", index=False, encoding="utf-8-sig")
    patient_table.to_csv(OUT / "patient_split.csv", index=False, encoding="utf-8-sig")

    counts = {
        "segmented_rows_before_repeat_exam_rule": int(len(seg)),
        "segmented_rows_after_repeat_exam_rule": int(len(data)),
        "dropped_repeat_exam_rows": int(len(dropped)),
        "patients": int(len(patient_table)),
        "patients_positive": int(patient_table["label"].sum()),
        "patients_negative": int((patient_table["label"] == 0).sum()),
        "development_patients": int((patient_table["split"] == "development").sum()),
        "holdout_patients": int((patient_table["split"] == "holdout").sum()),
        "development_images": int((data["split"] == "development").sum()),
        "holdout_images": int((data["split"] == "holdout").sum()),
        "roi_source_counts": {str(k): int(v) for k, v in data["roi_source"].value_counts().items()},
        "development_fold_patients": {
            str(k): int(v) for k, v in patient_table.loc[patient_table["split"] == "development", "fold"].value_counts().sort_index().items()
        },
    }
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(SEGMENTATION_MANIFEST),
        "label_source": str(LABEL_CSV),
        "seed": SEED,
        "holdout_fraction": HOLDOUT_FRACTION,
        "development_fraction": 1.0 - HOLDOUT_FRACTION,
        "development_folds": FOLDS,
        "split_unit": "patient_id",
        "repeat_exam_rule": "retain first exam_date_iso per patient",
        "augmentation": "ImageNet-pretrained CNN arm uses realistic affine, rotation, flip, crop, jitter and blur transforms on development training images only",
        "checkpoint_policy": "ImageNet public weights allowed; checkpoints trained on CYCH/NCKUH labels are excluded from held-out evaluation",
        "counts": counts,
    }
    (OUT / "protocol_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# Segmented CYCH 80/20 Protocol",
        "",
        "This protocol is frozen at the patient level before model selection.",
        "",
        f"- Input: `yolo_crops/manifest.csv`; only segmented kidney ROI crops are used.",
        f"- Patients: {counts['patients']} total; {counts['patients_positive']} positive and {counts['patients_negative']} negative.",
        f"- Development set: {counts['development_patients']} patients ({counts['development_images']} images), with stratified 5-fold patient-level CV.",
        f"- External holdout: {counts['holdout_patients']} patients ({counts['holdout_images']} images), untouched during model selection.",
        f"- Repeat exam handling: {counts['dropped_repeat_exam_rows']} later-exam rows removed, matching the existing runner's first-exam rule.",
        f"- ROI sources after filtering: {counts['roi_source_counts']}.",
        "- Augmentation is applied only inside development training folds; no augmented image is copied into validation or holdout sets.",
        "- Label-trained local checkpoints are excluded because their use would leak CYCH/NCKUH labels into evaluation.",
        "",
        "## Files",
        "",
        "- `label_manifest.csv`: image-level labels, split and development fold.",
        "- `patient_split.csv`: one row per patient with frozen split assignment.",
        "- `dropped_repeat_exam_rows.csv`: rows excluded by the repeat-exam rule.",
        "- `protocol_config.json`: machine-readable settings and counts.",
    ]
    (OUT / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
