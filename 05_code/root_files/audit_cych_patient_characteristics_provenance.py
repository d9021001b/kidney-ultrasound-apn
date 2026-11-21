from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "CYCH_feature_research_results_20260913" / "image_filename_manifest.csv"
PATIENT_RECORDS = ROOT / "patient_characteristics" / "patient_level_records.csv"
EXAM_RECORDS = ROOT / "patient_characteristics" / "exam_level_records.csv"
EXCEL = ROOT / "ki-data" / "patient_id_and_gender-cych20220412.xlsx"
OUT = ROOT / "CYCH_patient_characteristics_provenance_20260915"

FILENAME_PATTERN = re.compile(
    r"^(?P<patient_id>\d+)_(?P<date>\d{8})_red_"
    r"(?P<sex_code>[12])_(?P<numeric_1>[0-9.]+)_(?P<numeric_2>[0-9.]+)_"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def normalize_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    match = re.match(r"^\s*(\d+)", str(value))
    if not match:
        return None
    return match.group(1).lstrip("0") or "0"


def summarize_numeric(values: pd.Series) -> str:
    array = values.astype(float).to_numpy()
    q1, median, q3 = np.percentile(array, [25, 50, 75])
    return f"{median:.2f} ({q1:.2f}-{q3:.2f})"


def count_percent(count: int, denominator: int) -> str:
    return f"{count} ({100.0 * count / denominator:.1f})"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST, dtype={"patient_id": str})
    parsed = manifest["filename"].str.extract(FILENAME_PATTERN)
    if parsed.isna().any(axis=None):
        raise AssertionError("At least one retained filename did not match the fixed metadata pattern.")

    parsed["patient_id"] = parsed["patient_id"].str.lstrip("0").replace("", "0")
    manifest_ids = manifest["patient_id"].str.lstrip("0").replace("", "0")
    if not parsed["patient_id"].equals(manifest_ids):
        raise AssertionError("Filename and manifest patient identifiers disagree.")

    parsed["label"] = manifest["label"].astype(int)
    parsed["sex_code"] = parsed["sex_code"].astype(int)
    parsed["candidate_age_years"] = parsed["numeric_1"].astype(float)
    parsed["candidate_renal_measure_cm"] = parsed["numeric_2"].astype(float)

    consistency = parsed.groupby("patient_id", sort=True).agg(
        images=("patient_id", "size"),
        labels=("label", "nunique"),
        dates=("date", "nunique"),
        sex_codes=("sex_code", "nunique"),
        candidate_ages=("candidate_age_years", "nunique"),
        candidate_renal_measures=("candidate_renal_measure_cm", "nunique"),
    )
    conflict_counts = {
        column: int((consistency[column] > 1).sum())
        for column in (
            "labels",
            "dates",
            "sex_codes",
            "candidate_ages",
            "candidate_renal_measures",
        )
    }
    if any(conflict_counts.values()):
        raise AssertionError(f"Within-patient metadata conflicts detected: {conflict_counts}")

    patients = parsed.drop_duplicates("patient_id").copy()
    patients["group"] = patients["label"].map({0: "Negative", 1: "Positive"})
    patients["candidate_sex"] = patients["sex_code"].map({1: "Male", 2: "Female"})

    records = pd.read_csv(PATIENT_RECORDS, dtype={"patient_id": str})
    records = records[records["site"].eq("CYCH")].copy()
    records["patient_id_norm"] = records["patient_id"].str.lstrip("0").replace("", "0")
    linked = patients.merge(
        records,
        left_on="patient_id",
        right_on="patient_id_norm",
        how="left",
        validate="one_to_one",
        suffixes=("_filename", "_record"),
    )
    linked_checks = {
        "matched_patients": int(linked["patient_id_record"].notna().sum()),
        "label_agreement": int(
            linked["label"].eq(linked["patient_label"].astype(int)).sum()
        ),
        "sex_token_agreement": int(
            linked["candidate_sex"].eq(linked["gender"]).sum()
        ),
        "numeric_1_record_agreement": int(
            np.isclose(
                linked["candidate_age_years"].astype(float),
                linked["index_age_years"].astype(float),
                equal_nan=False,
            ).sum()
        ),
        "numeric_2_patient_summary_agreement": int(
            np.isclose(
                linked["candidate_renal_measure_cm"].astype(float),
                linked["median_renal_measure_cm"].astype(float),
                equal_nan=False,
            ).sum()
        ),
    }

    exams = pd.read_csv(EXAM_RECORDS, dtype={"patient_id": str})
    exams = exams[exams["site"].eq("CYCH")].copy()
    exams["patient_id_norm"] = exams["patient_id"].str.lstrip("0").replace("", "0")
    exams["exam_date_iso"] = exams["exam_date_iso"].astype(str)
    index_exams = (
        exams.sort_values(["patient_id_norm", "exam_date_iso"])
        .drop_duplicates("patient_id_norm")
        .loc[:, ["patient_id_norm", "age_years", "median_renal_measure_cm"]]
    )
    linked_index = patients.merge(
        index_exams,
        left_on="patient_id",
        right_on="patient_id_norm",
        how="left",
        validate="one_to_one",
    )
    linked_checks["numeric_1_first_exam_agreement"] = int(
        np.isclose(
            linked_index["candidate_age_years"].astype(float),
            linked_index["age_years"].astype(float),
            equal_nan=False,
        ).sum()
    )
    linked_checks["numeric_2_first_exam_agreement"] = int(
        np.isclose(
            linked_index["candidate_renal_measure_cm"].astype(float),
            linked_index["median_renal_measure_cm"].astype(float),
            equal_nan=False,
        ).sum()
    )

    excel = pd.read_excel(EXCEL, sheet_name="in", header=None).iloc[2:].copy()
    excel_sets = {
        f"column_{column}": set(filter(None, excel[column].map(normalize_id)))
        for column in (0, 2, 4, 5)
    }
    current_ids = set(patients["patient_id"])
    excel_overlap = {
        key: {
            "unique_ids": len(values),
            "current_98_overlap": len(current_ids & values),
        }
        for key, values in excel_sets.items()
    }
    excel_gender = excel[1].astype(str).str.strip().str.upper()

    table_rows: list[dict[str, str]] = []
    groups = [
        ("Negative", patients[patients["label"].eq(0)]),
        ("Positive", patients[patients["label"].eq(1)]),
        ("Total", patients),
    ]

    def row(characteristic: str, value_fn) -> None:
        output = {"Characteristic": characteristic}
        for name, frame in groups:
            output[name] = value_fn(frame)
        table_rows.append(output)

    row("Patients, n", lambda frame: str(len(frame)))
    row(
        "Candidate age, years, median (IQR)",
        lambda frame: summarize_numeric(frame["candidate_age_years"]),
    )
    row(
        "Candidate age, years, range",
        lambda frame: (
            f"{frame['candidate_age_years'].min():.2f}-"
            f"{frame['candidate_age_years'].max():.2f}"
        ),
    )
    row(
        "Candidate male sex, n (%)",
        lambda frame: count_percent(int(frame["candidate_sex"].eq("Male").sum()), len(frame)),
    )
    row(
        "Candidate female sex, n (%)",
        lambda frame: count_percent(int(frame["candidate_sex"].eq("Female").sum()), len(frame)),
    )
    row(
        "Candidate renal measurement, cm, median (IQR)",
        lambda frame: summarize_numeric(frame["candidate_renal_measure_cm"]),
    )
    pd.DataFrame(table_rows).to_csv(OUT / "candidate_table1_aggregate.csv", index=False)

    audit = {
        "purpose": (
            "Aggregate, identifier-free audit of filename-encoded candidate patient "
            "characteristics. It does not clinically validate the metadata semantics."
        ),
        "sources": {
            "analysis_manifest": {"sha256": sha256(MANIFEST)},
            "derived_patient_records": {"sha256": sha256(PATIENT_RECORDS)},
            "derived_exam_records": {"sha256": sha256(EXAM_RECORDS)},
            "historical_209_row_workbook": {"sha256": sha256(EXCEL)},
        },
        "analysis_cohort": {
            "patients": int(len(patients)),
            "images": int(len(parsed)),
            "positive_patients": int(patients["label"].sum()),
            "negative_patients": int((1 - patients["label"]).sum()),
            "filenames_parsed": int(len(parsed)),
        },
        "within_patient_conflicts": conflict_counts,
        "derived_record_reconstruction": linked_checks,
        "candidate_aggregate_flags": {
            "male": int(patients["candidate_sex"].eq("Male").sum()),
            "female": int(patients["candidate_sex"].eq("Female").sum()),
            "candidate_age_over_11_years": int(
                patients["candidate_age_years"].gt(11).sum()
            ),
            "candidate_age_max_years": float(patients["candidate_age_years"].max()),
            "interpretation": (
                "Nineteen filename-derived candidate ages exceed the historical draft's "
                "0-11-year criterion, further demonstrating that historical cohort text "
                "must not be transferred to the current 98-patient analysis."
            ),
        },
        "historical_workbook": {
            "data_rows": int(len(excel)),
            "gender_marginal": {
                "male": int(excel_gender.eq("M").sum()),
                "female": int(excel_gender.eq("F").sum()),
            },
            "column_id_overlap": excel_overlap,
            "interpretation": (
                "The workbook's gender column is paired with column 0, which has no "
                "patient overlap with the current 98-patient CYCH analysis. Column 5 "
                "overlaps the current cohort but is a separate list with no row-wise "
                "gender linkage. It therefore cannot verify current-cohort sex."
            ),
        },
        "semantic_boundary": {
            "supported": [
                "All 349 retained filenames follow one metadata-token pattern.",
                "The tokens are internally consistent within each of 98 patients.",
                "The derived index-examination table exactly reconstructs those filename tokens.",
                "The original 2022 draft states that age and sex came from EMR and renal size came from image calibration.",
            ],
            "not_supported_without_author_source_confirmation": [
                "That filename code 1 means male and code 2 means female for the current cohort.",
                "That numeric token 1 is age in years and numeric token 2 is a renal measurement in centimeters.",
                "That the historical 209-row workbook can be row-linked to the current 98 patients.",
                "That the current candidate values are verified clinical characteristics suitable for final Table 1.",
            ],
        },
    }
    (OUT / "provenance_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    readme = f"""# CYCH patient-characteristics provenance audit

This is an aggregate, identifier-free audit for the 98-patient analysis cohort. It does not establish clinical truth for filename-encoded metadata.

## What is reproducible

- All {len(parsed)} retained first-examination filenames from {len(patients)} patients match the same metadata-token pattern.
- Within the retained examination, no patient has conflicting label, date, sex-code, first numeric token, or second numeric token across images.
- The derived patient/examination tables link all {linked_checks['matched_patients']} patients. The index-examination table reproduces all {linked_checks['numeric_1_first_exam_agreement']} first and {linked_checks['numeric_2_first_exam_agreement']} second numeric tokens. The patient-level all-examination summary differs for one patient because it aggregates two examinations. This is reconstruction consistency, not independent verification.
- Candidate aggregate values are in `candidate_table1_aggregate.csv`; they contain no patient identifiers.

## Why these values are not yet final Table 1 data

The 2022 source manuscript states that age and sex came from EMR and renal size came from image calibration, but the project does not contain the 98-patient source crosswalk that proves the filename-token semantics. An older script labels the two numeric tokens as `long_axis` and `short_axis`, while the later cohort table labels them as age and renal measurement. That semantic conflict must be resolved from the original data dictionary or export code.

The workbook `patient_id_and_gender-cych20220412.xlsx` has 209 data rows and a marginal of {int(excel_gender.eq('M').sum())} male and {int(excel_gender.eq('F').sum())} female entries. Its row-linked gender table has zero overlap with the current 98 patients. A separate workbook list overlaps {excel_overlap['column_5']['current_98_overlap']} of 98 current patients, but it is not row-linked to the gender column. The workbook therefore cannot verify current-cohort sex.

The candidate age token ranges to {patients['candidate_age_years'].max():.2f} years and exceeds 11 years in {int(patients['candidate_age_years'].gt(11).sum())} patients. This conflicts with transferring the historical draft's 0-11-year eligibility wording to the current cohort and is another reason to require the source crosswalk.

## Author action

Obtain the original filename data dictionary or export script and a deidentified 98-patient crosswalk showing patient alias, sex, age at index examination, and renal measurement definition/unit. After one-to-one verification, the candidate aggregate rows may be transferred into Table 1 and converted from pending red text to approved black text. Until then, the journal manuscript should state that clinical covariates were excluded and should not present these candidate values as verified demographics.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
