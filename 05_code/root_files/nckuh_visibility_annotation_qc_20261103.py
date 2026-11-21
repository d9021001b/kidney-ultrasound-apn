"""Blind QC and adjudication preparation for two NCKUH visibility reviews."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


REQUIRED = (
    "case_id", "left_laterality_confirmed", "right_laterality_confirmed",
    "left_visible_abnormality", "right_visible_abnormality",
    "left_finding_type", "right_finding_type", "left_region", "right_region",
    "left_confidence_1_to_5", "right_confidence_1_to_5", "image_quality_exclusion", "reviewer_notes",
)
COMPARABLE = tuple(column for column in REQUIRED if column not in {"case_id", "reviewer_notes"})


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != REQUIRED:
        raise ValueError("Submission columns do not exactly match the blinded annotation form")
    output = frame.copy()
    output["case_id"] = output.case_id.astype(str).str.strip()
    if output.case_id.duplicated().any() or output.case_id.eq("").any():
        raise ValueError("case_id must be nonempty and unique")
    for column in COMPARABLE:
        output[column] = output[column].fillna("").astype(str).str.strip().str.lower()
        if output[column].eq("").any():
            raise ValueError(f"Incomplete blinded review: {column}")
    return output.sort_values("case_id").reset_index(drop=True)


def read_submission(path: Path, expected_ids: set[str]) -> pd.DataFrame:
    frame = normalize(pd.read_csv(path, encoding="utf-8-sig"))
    if set(frame.case_id) != expected_ids:
        raise ValueError("Submission case IDs differ from the blinded packet")
    return frame


def agreement(first: pd.DataFrame, second: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if first.case_id.tolist() != second.case_id.tolist():
        raise ValueError("Review case order/IDs differ")
    rows, adjudication = [], []
    for column in COMPARABLE:
        left, right = first[column], second[column]
        exact = left.eq(right)
        rows.append({"field": column, "cases": len(left), "exact_agreement": float(exact.mean()),
                     "cohen_kappa": float(cohen_kappa_score(left, right)), "disagreements": int((~exact).sum())})
    mismatch = pd.DataFrame({"case_id": first.case_id})
    consensus = pd.DataFrame({"case_id": first.case_id})
    for column in COMPARABLE:
        mismatch[f"reviewer_1_{column}"] = first[column]
        mismatch[f"reviewer_2_{column}"] = second[column]
        mismatch[f"adjudicated_{column}"] = ""
        consensus[column] = first[column].where(first[column].eq(second[column]), "")
    needs = pd.concat([first[column].ne(second[column]) for column in COMPARABLE], axis=1).any(axis=1)
    consensus["needs_adjudication"] = needs
    return pd.DataFrame(rows), mismatch.loc[needs].reset_index(drop=True), consensus


def run(first_path: Path, second_path: Path, packet_form: Path, out: Path) -> None:
    template = pd.read_csv(packet_form, encoding="utf-8-sig")
    expected = set(template.case_id.astype(str))
    first, second = read_submission(first_path, expected), read_submission(second_path, expected)
    metrics, adjudication, consensus = agreement(first, second)
    out.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(out / "blinded_agreement.csv", index=False)
    adjudication.to_csv(out / "blinded_adjudication_template.csv", index=False)
    consensus.to_csv(out / "blinded_consensus_pending_adjudication.csv", index=False)
    (out / "blinded_qc.json").write_text(json.dumps({
        "reviewers": 2, "cases": len(first), "fields": len(COMPARABLE),
        "cases_requiring_adjudication": len(adjudication),
        "contains_DMSA": False, "contains_patient_alias": False,
    }, indent=2), encoding="utf-8")
    print(f"QC passed: {len(first)} blinded cases; {len(adjudication)} need adjudication")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviewer_1", type=Path)
    parser.add_argument("reviewer_2", type=Path)
    parser.add_argument("--packet-form", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.reviewer_1, args.reviewer_2, args.packet_form, args.out)
