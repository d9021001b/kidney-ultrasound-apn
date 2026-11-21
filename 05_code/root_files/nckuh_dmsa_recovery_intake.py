"""Prepare and validate a private intake for unresolved NCKUH outcomes."""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "02_experiments"
SOURCE = ROOT / "NCKUH_DMSA_label_audit_20261102"
OUT = ROOT / "NCKUH_DMSA_recovery_intake_20261103"
STATUS = {"report_available", "not_performed", "report_not_retrieved", "indeterminate"}
FIELDS = ("report_status", "dmsa_scan_date_iso", "source_reference", "dmsa_apn_r_l_b_n",
          "episode_link_confirmed", "reference_interpretation", "review_note")


def validate(submission, template):
    if list(submission.columns) != list(template.columns):
        raise ValueError("Intake schema mismatch")
    stable = [c for c in template if c not in FIELDS]
    pd.testing.assert_frame_equal(submission[stable].sort_values("patient_alias").reset_index(drop=True),
                                  template[stable].sort_values("patient_alias").reset_index(drop=True))
    ready = []
    for row in submission.itertuples(index=False):
        status, code = row.report_status.strip(), row.dmsa_apn_r_l_b_n.strip().upper()
        if not status and not any(str(getattr(row, f)).strip() for f in FIELDS):
            continue
        if status not in STATUS:
            raise ValueError("Invalid report status")
        if status != "report_available":
            if code:
                raise ValueError("A missing or indeterminate report cannot supply an APN code")
            continue
        date = pd.to_datetime(row.dmsa_scan_date_iso, format="%Y-%m-%d", errors="coerce")
        if pd.isna(date) or date.date().isoformat() != row.dmsa_scan_date_iso or not row.source_reference.strip():
            raise ValueError("Available reports need a valid ISO date and source reference")
        if row.reference_interpretation not in {"acute_apn", "no_apn", "scar_only", "indeterminate"}:
            raise ValueError("Invalid reference interpretation")
        if row.episode_link_confirmed not in {"yes", "no", "uncertain"}:
            raise ValueError("Report needs episode linkage status")
        expected = {"acute_apn": {"R", "L", "B"}, "no_apn": {"N"}}
        if code not in expected.get(row.reference_interpretation, {""}):
            raise ValueError("Code and report interpretation disagree")
        if code and row.episode_link_confirmed == "yes":
            ready.append(row.patient_alias)
    return ready


def build():
    if OUT.exists():
        raise FileExistsError(OUT)
    patients = pd.read_csv(SOURCE / "patient_label_linkage_private.csv", dtype=str, keep_default_na=False)
    missing = patients[patients.DMSA_APN.eq("")].copy()
    priorities = {"main_sheet_DMSA_APN_blank": "check_diagnostic_sheet_and_report",
                  "sheet2_only_no_DMSA_report": "search_for_original_report"}
    assert missing.source_status.value_counts().to_dict() == {
        "main_sheet_DMSA_APN_blank": 3, "sheet2_only_no_DMSA_report": 51}
    assert len(patients) == 264 and missing.patient_alias.is_unique
    result = missing[["patient_alias", "source_rows", "image_exam_date", "imaged_kidneys", "image_count"]].assign(
        review_route=missing.source_status.map(priorities))
    result = result.sort_values(["review_route", "patient_alias"])
    for field in FIELDS:
        result[field] = ""
    assert sum(result.imaged_kidneys.astype(int)) == 107 and sum(result.image_count.astype(int)) == 207
    assert validate(result, result) == []
    OUT.mkdir()
    result.to_csv(OUT / "DMSA_recovery_form_INTERNAL.csv", index=False, encoding="utf-8-sig")
    (OUT / "intake_audit.json").write_text(json.dumps({
        "diagnostic_sheet_blank": 3, "no_supplied_report": 51,
        "patients": 54, "imaged_kidneys": 107, "images": 207,
        "new_dmsa_outcomes": 0, "source_workbook_modified": False,
    }, indent=2), encoding="utf-8")
    print("54 unresolved patients, 107 kidneys and 207 images; zero outcomes assigned")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.validate:
        template = pd.read_csv(OUT / "DMSA_recovery_form_INTERNAL.csv", dtype=str, keep_default_na=False)
        reviewed = pd.read_csv(args.validate, dtype=str, keep_default_na=False)
        print(f"Episode-linked, source-documented candidates: {len(validate(reviewed, template))}")
    else:
        build()
