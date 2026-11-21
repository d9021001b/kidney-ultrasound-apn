"""Validate blinded clinician review and export split-safe training annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


ROOT = Path(r"C:\Users\udoo_w2\Desktop\work_kidney_inflammation")
EXPERIMENT = ROOT / "02_experiments/NCKUH_unilateral_visibility_annotation_20261103"
DEFAULT_WORKBOOK = EXPERIMENT / "blinded_packet/annotation_form_clinician_20261105.xlsx"
PRIVATE_KEY = EXPERIMENT / "private_key/case_to_DMSA_key_PRIVATE.csv"
ASSIGNMENTS = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_fullframe_20261103/patient_assignments.csv"
STATUS = EXPERIMENT / "clinician_annotation_return_status.json"
PRIVATE_OUTPUT = EXPERIMENT / "returned_private"

YES_NO_UNCERTAIN = {"yes", "no", "uncertain"}
FINDINGS = {
    "none",
    "focal hyperechoic",
    "focal hypoechoic",
    "swelling",
    "reduced corticomedullary differentiation",
    "perfusion abnormality",
    "collecting-system abnormality",
    "other",
    "uncertain",
}
REGIONS = {"upper pole", "middle", "lower pole", "diffuse", "none", "uncertain"}
EXPECTED_COLUMNS = [
    "case_id",
    "left_laterality_confirmed",
    "right_laterality_confirmed",
    "left_visible_abnormality",
    "right_visible_abnormality",
    "left_finding_type",
    "right_finding_type",
    "left_region",
    "right_region",
    "left_confidence_1_to_5",
    "right_confidence_1_to_5",
    "image_quality_exclusion",
    "reviewer_notes",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def normalize(value) -> str:
    return "" if value is None else str(value).strip().lower()


def read_review(path: Path) -> tuple[pd.DataFrame, str, str]:
    workbook = load_workbook(path, data_only=True, read_only=False)
    if workbook.sheetnames[:2] != ["標註表", "填寫說明"]:
        raise ValueError("Unexpected clinician workbook sheets")
    sheet = workbook["標註表"]
    headers = [str(sheet.cell(4, column).value).split("\n", 1)[0] for column in range(1, 14)]
    if headers != EXPECTED_COLUMNS:
        raise ValueError("Unexpected clinician workbook columns")
    rows = [
        [sheet.cell(row, column).value for column in range(1, 14)]
        for row in range(5, 88)
    ]
    frame = pd.DataFrame(rows, columns=headers)
    expected_ids = [f"U{index:03d}" for index in range(1, 84)]
    if frame.case_id.tolist() != expected_ids:
        raise ValueError("The 83 blinded case identifiers changed")
    return frame, normalize(sheet["B2"].value), normalize(sheet["D2"].value)


def completion(frame: pd.DataFrame) -> dict:
    required = [
        "left_laterality_confirmed",
        "right_laterality_confirmed",
        "left_visible_abnormality",
        "right_visible_abnormality",
        "image_quality_exclusion",
    ]
    completed = frame[required].apply(lambda column: column.map(normalize).ne("")).all(axis=1)
    return {
        "cases": len(frame),
        "complete_required_cases": int(completed.sum()),
        "remaining_required_cases": int((~completed).sum()),
    }


def validate_completed(frame: pd.DataFrame, reviewer: str, review_date: str) -> pd.DataFrame:
    if not reviewer or not review_date:
        raise ValueError("Reviewer code and review date are required before finalization")
    normalized = frame.copy()
    text_columns = [column for column in EXPECTED_COLUMNS if column != "case_id"]
    for column in text_columns:
        normalized[column] = normalized[column].map(normalize)

    for column in (
        "left_laterality_confirmed",
        "right_laterality_confirmed",
        "left_visible_abnormality",
        "right_visible_abnormality",
        "image_quality_exclusion",
    ):
        invalid = ~normalized[column].isin(YES_NO_UNCERTAIN)
        if invalid.any():
            raise ValueError(f"Invalid or missing {column}: {normalized.loc[invalid, 'case_id'].tolist()}")

    for side in ("left", "right"):
        visible = normalized[f"{side}_visible_abnormality"]
        finding = normalized[f"{side}_finding_type"]
        region = normalized[f"{side}_region"]
        confidence = pd.to_numeric(normalized[f"{side}_confidence_1_to_5"], errors="coerce")
        invalid_finding = ~finding.isin(FINDINGS)
        invalid_region = ~region.isin(REGIONS)
        invalid_confidence = ~confidence.between(1, 5, inclusive="both")
        if invalid_finding.any() or invalid_region.any() or invalid_confidence.any():
            bad = invalid_finding | invalid_region | invalid_confidence
            raise ValueError(f"Incomplete {side} detail: {normalized.loc[bad, 'case_id'].tolist()}")
        contradictory = (visible == "yes") & ((finding == "none") | (region == "none"))
        if contradictory.any():
            raise ValueError(
                f"Visible abnormality cannot use none for {side}: "
                f"{normalized.loc[contradictory, 'case_id'].tolist()}"
            )
        normalized[f"{side}_confidence_1_to_5"] = confidence.astype(int)
    return normalized


def finalize(frame: pd.DataFrame, reviewer: str, review_date: str, source: Path) -> dict:
    validated = validate_completed(frame, reviewer, review_date)
    key = pd.read_csv(PRIVATE_KEY, usecols=["case_id", "patient_alias"], dtype=str)
    if len(key) != 83 or key.case_id.nunique() != 83:
        raise ValueError("Unexpected private linkage key")
    linked = validated.merge(key, on="case_id", how="left", validate="one_to_one")
    if linked.patient_alias.isna().any():
        raise ValueError("Unlinked blinded cases")

    assignments = pd.read_csv(
        ASSIGNMENTS,
        usecols=["seed", "patient_alias", "role"],
        dtype={"patient_alias": str, "role": str},
    )
    seeds = sorted(assignments.seed.unique().tolist())
    PRIVATE_OUTPUT.mkdir(parents=True, exist_ok=True)
    exported = {}
    annotation_columns = [
        "case_id",
        "patient_alias",
        *[column for column in EXPECTED_COLUMNS if column != "case_id"],
    ]
    for seed in seeds:
        roles = assignments[assignments.seed.eq(seed)]
        training = linked.merge(roles, on="patient_alias", how="inner", validate="many_to_one")
        training = training[training.role.eq("train")][annotation_columns].copy()
        output = PRIVATE_OUTPUT / f"training_visibility_annotations_seed_{int(seed)}_PRIVATE.csv"
        training.to_csv(output, index=False)
        exported[str(int(seed))] = {
            "training_cases": len(training),
            "sha256": digest(output),
        }

    audit = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_split_safe_auxiliary_training",
        "source_workbook_sha256": digest(source),
        "reviewer_code_recorded": bool(reviewer),
        "review_date_recorded": bool(review_date),
        "cases": len(validated),
        "DMSA_values_read": False,
        "development_or_test_annotations_exported": False,
        "training_exports": exported,
    }
    (PRIVATE_OUTPUT / "ingestion_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    frame, reviewer, review_date = read_review(args.workbook)
    state = completion(frame)
    if args.finalize:
        if state["remaining_required_cases"]:
            raise SystemExit(
                f"Clinician review incomplete: {state['remaining_required_cases']} cases remain"
            )
        state = finalize(frame, reviewer, review_date, args.workbook)
    else:
        state.update(
            {
                "checked_utc": datetime.now(timezone.utc).isoformat(),
                "status": (
                    "ready_to_finalize"
                    if state["remaining_required_cases"] == 0
                    else "awaiting_clinician_review"
                ),
                "source_workbook_sha256": digest(args.workbook),
                "DMSA_values_read": False,
                "private_linkage_key_read": False,
            }
        )
        STATUS.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
