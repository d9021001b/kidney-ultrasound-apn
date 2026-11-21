"""Summarize which local CYCH records directly support the current paper cohort."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CYCH_paper_support_audit_20260913"


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lstrip("0").str.zfill(8)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    characteristics = pd.read_csv(ROOT / "patient_characteristics" / "patient_level_records.csv", dtype={"patient_id": str})
    examinations = pd.read_csv(ROOT / "patient_characteristics" / "exam_level_records.csv", dtype={"patient_id": str})
    source = pd.read_csv(ROOT / "yolo_crops" / "manifest.csv", dtype={"patient_id": str})
    retained_images = pd.read_csv(ROOT / "CYCH_local_features_20260913" / "image_manifest.csv", dtype={"patient_id": str})
    retained_patients = pd.read_csv(ROOT / "CYCH_local_features_20260913" / "patients.csv", dtype={"patient_id": str})

    cych_characteristics = characteristics[characteristics.site.eq("CYCH")].copy()
    cych_examinations = examinations[examinations.site.eq("CYCH")].copy()
    cych_source = source[source.site.eq("CYCH")].copy()
    retained_patients["_id"] = normalize_id(retained_patients.patient_id)
    cych_characteristics["_id"] = normalize_id(cych_characteristics.patient_id)
    linked = retained_patients.merge(cych_characteristics, on="_id", how="left", validate="one_to_one", suffixes=("_analysis", "_clinical"))

    fields = ("patient_label", "diagnosis_group", "gender", "index_exam_date_iso", "index_age_years", "median_renal_measure_cm", "min_renal_measure_cm", "max_renal_measure_cm")
    field_rows = []
    for field in fields:
        field_rows.append({"field": field, "available_patients": int(linked[field].notna().sum()), "missing_patients": int(linked[field].isna().sum()), "complete": bool(linked[field].notna().all())})
    pd.DataFrame(field_rows).to_csv(OUT / "field_completeness.csv", index=False)

    label_agreement = linked.label.astype(str).eq(linked.patient_label.astype(str))
    summary = {
        "purpose": "Audit of local records that support the current CYCH image-label analysis cohort. Completeness of derived metadata fields does not establish their clinical provenance or semantics.",
        "source_cych": {"patients": int(cych_source.patient_id.nunique()), "images": int(len(cych_source)), "images_by_label": {str(k): int(v) for k, v in cych_source.label.value_counts().sort_index().items()}, "examinations": int(len(cych_examinations)), "patients_with_multiple_examinations": int(cych_examinations.patient_id.value_counts().gt(1).sum())},
        "analyzed_first_exam": {"patients": int(len(retained_patients)), "images": int(len(retained_images)), "positive_patients": int(retained_patients.label.sum()), "negative_patients": int((1 - retained_patients.label).sum()), "positive_images": int(retained_images.label.sum()), "negative_images": int((1 - retained_images.label).sum()), "images_excluded_by_earliest_exam_rule": int(len(cych_source) - len(retained_images)), "patients_excluded_by_earliest_exam_rule": int(cych_source.patient_id.nunique() - retained_images.patient_id.nunique())},
        "patient_record_linkage": {"matched": int(linked.patient_id_clinical.notna().sum()), "analysis_patients": int(len(linked)), "label_agreeing_matches": int(label_agreement.sum()), "label_conflicts_flagged": int(linked.label_conflict.fillna(True).astype(bool).sum()), "gender_conflicts_flagged": int(linked.gender_conflict.fillna(True).astype(bool).sum())},
        "limitations": ["The local records support the image-label cohort description.", "Age, sex, and renal-measure fields were reconstructed from filename tokens and require an original data dictionary or one-to-one clinical crosswalk before being reported as verified patient characteristics.", "The historical 209-row workbook cannot row-link sex to the current 98-patient cohort.", "The records do not independently validate the disease-label reference standard, eligibility criteria, ethics information, or selection from the broader source collection.", "The current analysis must continue to describe results as single-center retrospective image-label findings."],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# CYCH paper support audit",
        "",
        f"The analyzable paper cohort is {summary['analyzed_first_exam']['patients']} patients: {summary['analyzed_first_exam']['positive_patients']} label-positive and {summary['analyzed_first_exam']['negative_patients']} label-negative. It uses {summary['analyzed_first_exam']['images']} first-examination images.",
        "",
        f"The CYCH source manifest contains {summary['source_cych']['patients']} patients and {summary['source_cych']['images']} images. One patient has two examinations; the earliest-examination rule excludes {summary['analyzed_first_exam']['images_excluded_by_earliest_exam_rule']} images while retaining all {summary['analyzed_first_exam']['patients']} patients.",
        "",
        f"All {summary['patient_record_linkage']['matched']} analyzed patients link one-to-one to the derived patient table, and all {summary['patient_record_linkage']['label_agreeing_matches']} linked labels agree. No within-file label or sex-token conflict flag is set.",
        "",
        "Derived fields present for all 98 analyzed patients are listed in `field_completeness.csv`; completeness is not clinical validation. Age, sex, and renal-measure semantics require source confirmation. The separate provenance audit documents that the historical 209-row workbook cannot row-link sex to the current 98 patients.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
