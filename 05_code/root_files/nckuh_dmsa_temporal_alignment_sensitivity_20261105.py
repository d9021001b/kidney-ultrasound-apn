"""Fixed-prediction sensitivity to ultrasound/admission temporal alignment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import nckuh_dmsa_fullcohort_fullframe_20261103 as base


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_temporal_alignment_sensitivity_20261105"
WORKBOOK = ROOT / "2004-2019-NCKUH-亞大-20210723.xlsx"
PREDICTIONS = ROOT / "02_experiments/NCKUH_DMSA_uricads_official_frozen_20261105/development_predictions.csv"
GROUPS = (
    "current_tile_usfm_reference",
    "official_uri_embedding_folds_quarter_plus_current",
)
MAX_ABS_GAP_DAYS = 14


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def admission_table() -> pd.DataFrame:
    records = []
    for sheet_name in ("2007-2012", "2014-2019"):
        frame = pd.read_excel(WORKBOOK, sheet_name=sheet_name)
        admission_column = frame.columns[4]
        for _, row in frame.iterrows():
            records.append(
                {
                    "normalized_patient_id": base.normalize_patient_id(row.iloc[0]),
                    "admission_date": pd.to_datetime(row[admission_column], errors="coerce"),
                }
            )
    result = pd.DataFrame(records).drop_duplicates("normalized_patient_id")
    if result.normalized_patient_id.duplicated().any():
        raise ValueError("Duplicate normalized patient identifiers")
    return result


def temporal_table() -> pd.DataFrame:
    patients = base.labeled_images().drop_duplicates("patient_alias")[
        ["patient_alias", "normalized_patient_id", "exam_date_raw"]
    ].copy()
    patients = patients.merge(admission_table(), on="normalized_patient_id", how="left", validate="one_to_one")
    patients["exam_date"] = pd.to_datetime(
        patients.exam_date_raw, format="%Y%m%d", errors="coerce"
    )
    patients["gap_days"] = (patients.exam_date - patients.admission_date).dt.days
    patients["temporally_aligned"] = patients.gap_days.abs().le(MAX_ABS_GAP_DAYS)
    return patients[["patient_alias", "gap_days", "temporally_aligned"]]


def metrics(frame: pd.DataFrame) -> dict:
    return {
        "kidneys": len(frame),
        "patients": int(frame.patient_alias.nunique()),
        "auroc": float(roc_auc_score(frame.label, frame.score)),
        "ap": float(average_precision_score(frame.label, frame.score)),
        "brier": float(brier_score_loss(frame.label, frame.score)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("protocol.json", "development_metrics.csv", "development_summary.csv"):
        if (OUT / name).exists():
            raise FileExistsError(f"Preserve completed temporal sensitivity: {name}")

    temporal = temporal_table()
    temporal.to_csv(OUT / "deidentified_temporal_alignment.csv", index=False)
    gap = temporal.gap_days.dropna()
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "Test a source-defined plus/minus 14-day ultrasound/admission alignment rule without changing saved model scores",
        "cohort": "The unchanged 210-patient NCKUH cohort and fixed three development draws",
        "rule": "Keep patients with an absolute ultrasound-to-admission gap of at most 14 days; missing admission or exam date is not aligned",
        "model_handling": "No retraining, refitting, score recalibration, fusion-weight change, or new model selection",
        "evaluation": "Saved development predictions only; exploratory-test rows remain unscored",
        "groups": list(GROUPS),
        "inputs": {str(path.relative_to(ROOT)): digest(path) for path in (WORKBOOK, PREDICTIONS, Path(__file__))},
    }
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    predictions = pd.read_csv(PREDICTIONS)
    predictions = predictions[predictions.group.isin(GROUPS)].copy()
    predictions = predictions.merge(temporal, on="patient_alias", how="left", validate="many_to_one")
    rows = []
    for (seed, group), frame in predictions.groupby(["seed", "group"]):
        original = metrics(frame)
        aligned = metrics(frame[frame.temporally_aligned.eq(True)])
        for analysis, values in (("all_fixed_development", original), ("aligned_14d_sensitivity", aligned)):
            rows.append({"seed": seed, "group": group, "analysis": analysis, **values})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "development_metrics.csv", index=False)
    summary = (
        result.groupby(["group", "analysis"], as_index=False)
        .agg(
            kidneys_mean=("kidneys", "mean"),
            patients_mean=("patients", "mean"),
            auroc_mean=("auroc", "mean"),
            ap_mean=("ap", "mean"),
            brier_mean=("brier", "mean"),
        )
    )
    summary.to_csv(OUT / "development_summary.csv", index=False)

    decision = {
        "patients": len(temporal),
        "date_gap_available": int(temporal.gap_days.notna().sum()),
        "aligned_within_14_days": int(temporal.temporally_aligned.sum()),
        "missing_gap": int(temporal.gap_days.isna().sum()),
        "outside_14_days": int(temporal.gap_days.abs().gt(MAX_ABS_GAP_DAYS).sum()),
        "median_gap_days": float(gap.median()),
        "max_gap_days": float(gap.max()),
        "exploratory_test_scored": False,
        "decision": "Temporal sensitivity does not improve the fixed reference or URI-CADS candidate; retain original fixed development comparison and audit the two unmatched patients",
    }
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
