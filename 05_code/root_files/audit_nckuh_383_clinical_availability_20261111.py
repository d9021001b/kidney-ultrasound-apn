"""Audit pre-index clinical field availability without exporting identifiers."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_revised_383_clinical_audit_20261111"
WB = ROOT / "2004-2019-NCKUH-亞大.xlsx"
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"


def nonempty(value):
    return value is not None and str(value).strip() != ""


def patient_key(value):
    return str(value).strip().lstrip("0") or "0"


def sex_code(value):
    return {"1": "M", "2": "F"}.get(str(value).strip(), str(value).strip())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    book = load_workbook(WB, read_only=True, data_only=True)
    roster = list(book["完整資料彙整"].values)[1:]
    identifiers = {patient_key(row[0]) for row in roster if nonempty(row[0])}
    sex_by_id = defaultdict(set)
    source_coverage = {}
    for sheet_name in ("2007-2012", "2014-2019", "工作表2"):
        rows = list(book[sheet_name].values)[1:]
        matched = set()
        for row in rows:
            pid = row[0]
            if not nonempty(pid) or patient_key(pid) not in identifiers:
                continue
            pid = patient_key(pid)
            matched.add(pid)
            if nonempty(row[3]):
                sex_by_id[pid].add(sex_code(row[3]))
        source_coverage[sheet_name] = len(matched)
    echo_rows = list(book["成大51位DMSA報告"].values)[1:]
    echo_matched = {patient_key(row[0]) for row in echo_rows
                    if nonempty(row[0]) and patient_key(row[0]) in identifiers and nonempty(row[3])}
    assignments = pd.read_csv(REGISTRY / "patient_assignments.csv")
    linkage = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype=str)
    roles = assignments.merge(linkage, on="patient_alias", validate="one_to_one")
    age_by_id = {patient_key(row[0]): row[4] for row in roster if nonempty(row[0])}
    details = []
    for role, part in roles.groupby("role"):
        ids = set(part.patient_id.astype(str).map(patient_key))
        details.append({"role": role, "patients": len(ids),
                        "age_present": sum(nonempty(age_by_id.get(pid)) for pid in ids),
                        "sex_present": sum(len(sex_by_id[pid]) == 1 for pid in ids),
                        "sex_conflict": sum(len(sex_by_id[pid]) > 1 for pid in ids),
                        "echo_text_present": len(ids & echo_matched)})
    summary = {
        "workbook": WB.name,
        "roster_patients": len(identifiers),
        "sex_source_matches": source_coverage,
        "sex_coverage": Counter("missing" if not sex_by_id[pid] else
                                "conflicting" if len(sex_by_id[pid]) > 1 else
                                "present" for pid in identifiers),
        "echo_text_patients": len(echo_matched),
        "registry_ids_matching_roster": len(set(roles.patient_id.astype(str).map(patient_key)) & identifiers),
        "roles": sorted(details, key=lambda x: x["role"]),
        "available_preindex_predictor_candidates": ["age", "sex"],
        "timing_unverified_text": ["echo"],
        "outcome_or_postindex_not_predictors": ["DMSA_APN", "DMSA_RF", "DMSA_LF", "DMSA_YN", "DMSA", "VCUG"],
        "not_supplied_as_patient_values": ["CRP", "urinalysis", "WBC", "fever", "antibiotics"],
        "note": "Admission date and DMSA exam time are for episode/timing QA only. Echo text needs acquisition-time and leakage review before use."
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=dict), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, default=dict))


if __name__ == "__main__":
    main()
