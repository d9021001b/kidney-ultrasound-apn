"""Link the newly supplied clinical workbook to NCKUH images without editing it."""
from collections import Counter, defaultdict
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re

import openpyxl
from openpyxl.utils.datetime import from_excel
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "2004-2019-NCKUH-亞大-20210723.xlsx"
BASE = ROOT / "02_experiments/NCKUH_only_study_20261102"
OUT = ROOT / "02_experiments/NCKUH_DMSA_label_audit_20261102"
MAIN = ["2007-2012", "2014-2019"]
MAP = {"R": (0, 1), "L": (1, 0), "B": (1, 1), "N": (0, 0)}


def comparable_gender(value):
    # All 27 cross-sheet paired encodings support 1/M and 2/F; retain raw values.
    return {"1": "M", "2": "F"}.get(str(value).strip().upper(), str(value).strip().upper())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def patient_key(value):
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        if float(value).is_integer():
            value = str(int(value))
        else:
            raise ValueError("Noninteger patient identifier")
    value = str(value).strip()
    if not re.fullmatch(r"\d+", value):
        raise ValueError("Nondigit patient identifier")
    return value.lstrip("0") or "0"


def code(value):
    if value is None or str(value).strip() == "":
        return None
    value = str(value).strip().upper()
    value = {"BIL": "B", "BILATERAL": "B"}.get(value, value)
    if value not in MAP:
        raise ValueError("Unrecognized DMSA_APN code: " + value)
    return value


def date_value(value, epoch):
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if isinstance(value, (float, int)) and 20000 <= value <= 60000:
        return from_excel(value, epoch).date().isoformat()
    if text.endswith(".0"):
        text = text[:-2]
    try:
        if re.fullmatch(r"\d{8}", text):
            return datetime.strptime(text, "%Y%m%d").date().isoformat()
        if re.fullmatch(r"\d{7}", text):
            return date(int(text[:3])+1911, int(text[3:5]), int(text[5:7])).isoformat()
        match = re.fullmatch(r"(\d{2,4})[-/.](\d{1,2})[-/.](\d{1,2})(?:\s.*)?", text)
        if match:
            y, m, d = map(int, match.groups())
            return date(y+1911 if y < 1911 else y, m, d).isoformat()
    except ValueError:
        return None
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = sha256(SOURCE)
    patients = pd.read_csv(BASE / "patients_private.csv", dtype={"patient_id": str})
    manifest = pd.read_csv(BASE / "image_manifest_private.csv", dtype={"patient_id": str})
    patients["key"] = patients.patient_id.map(patient_key)
    manifest["key"] = manifest.patient_id.map(patient_key)
    used = set(patients.key)
    workbook = openpyxl.load_workbook(SOURCE, read_only=True, data_only=True)
    records, summaries = [], []
    all_flags, main_ids, sheet2_ids = set(), set(), set()
    for name in [*MAIN, "工作表2"]:
        sheet = workbook[name]
        values = list(sheet.iter_rows(values_only=True))
        header = [str(v).strip() if v is not None else "" for v in values[0]]
        marker = header.index("成大264位病歷號")
        ident = header.index("id") if "id" in header else header.index("病歷號碼")
        diagnosis = header.index("DMSA_APN") if "DMSA_APN" in header else None
        admission = header.index("admission_date") if "admission_date" in header else header.index("住院日")
        selected = []
        for row_number, row in enumerate(values[1:], 2):
            if str(row[marker]).strip() != "是":
                continue
            key = patient_key(row[ident])
            if key not in used:
                raise ValueError("Flagged patient outside the image cohort")
            apn = code(row[diagnosis]) if diagnosis is not None else None
            record = dict(key=key, sheet=name, excel_row=row_number, source_id_text=str(row[ident]),
                          raw_code=str(row[diagnosis]) if diagnosis is not None and row[diagnosis] is not None else "",
                          dmsa_apn=apn, admission_date_raw=str(row[admission]),
                          admission_date=date_value(row[admission], workbook.epoch),
                          birth_date=date_value(row[header.index("birth")], workbook.epoch),
                          gender_raw=str(row[header.index("gender")]).strip(),
                          age_raw=str(row[header.index("age")]) if "age" in header else "",
                          has_dmsa_column=diagnosis is not None)
            records.append(record); selected.append(record); all_flags.add(key)
            (main_ids if name in MAIN else sheet2_ids).add(key)
        summaries.append(dict(sheet=name, flagged_rows=len(selected), distinct_patient_ids=len({r["key"] for r in selected}),
                              code_rows=dict(Counter(r["dmsa_apn"] or "missing" for r in selected))))
    assert all_flags == used and len(used) == 264
    grouped = defaultdict(list)
    for row in records:
        grouped[row["key"]].append(row)
    output = []
    for patient in patients.itertuples(index=False):
        linked = grouped[patient.key]
        main_rows = [r for r in linked if r["sheet"] in MAIN]
        codes = sorted({r["dmsa_apn"] for r in main_rows if r["dmsa_apn"] is not None})
        if len(codes) > 1:
            status, apn = "conflicting_clinical_codes", None
        elif len(codes) == 1:
            status, apn = "source_side_code_available", codes[0]
        elif main_rows:
            status, apn = "main_sheet_DMSA_APN_blank", None
        else:
            status, apn = "sheet2_only_no_DMSA_report", None
        y = None if apn is None else int(apn != "N")
        left, right = MAP[apn] if apn is not None else (None, None)
        images = manifest[manifest.key.eq(patient.key)]
        exam_dates = sorted({parsed for value in images.exam_date_iso.dropna() if (parsed := date_value(value, workbook.epoch)) is not None})
        exam = date.fromisoformat(exam_dates[0]) if len(exam_dates) == 1 else None
        candidate_days = [abs((date.fromisoformat(r["admission_date"])-exam).days) for r in main_rows if r["admission_date"] and exam is not None]
        nonblank_dates = sorted({r["admission_date"] for r in main_rows if r["admission_date"]})
        birth_dates = sorted({r["birth_date"] for r in linked if r["birth_date"]})
        genders = sorted({r["gender_raw"] for r in linked if r["gender_raw"] not in ["None", "", "nan"]})
        comparable_genders = sorted({comparable_gender(value) for value in genders})
        output.append(dict(patient_id=patient.patient_id, patient_alias=patient.patient_alias, source_status=status,
                           DMSA_APN=apn, patient_APN=y, left_APN=left, right_APN=right,
                           previous_label=int(patient.label), previous_patient_label_disagrees=None if y is None else int(y != int(patient.label)),
                           source_rows=";".join(f"{r['sheet']}!{r['excel_row']}" for r in linked),
                           primary_source_rows=";".join(f"{r['sheet']}!{r['excel_row']}" for r in main_rows),
                           clinical_admission_dates=";".join(nonblank_dates), image_exam_date=";".join(exam_dates),
                           image_exam_date_ambiguous_or_missing=int(len(exam_dates) != 1),
                           nearest_admission_gap_days=min(candidate_days) if candidate_days else None,
                           birth_date=";".join(birth_dates), gender_source_values=";".join(genders),
                           birth_date_conflict=int(len(birth_dates) > 1), gender_raw_encoding_difference=int(len(genders) > 1),
                           gender_conflict=int(len(comparable_genders) > 1),
                           image_count=len(images), imaged_kidneys=int(images.side.nunique())))
    linked = pd.DataFrame(output)
    linked.to_csv(OUT / "patient_label_linkage_private.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(records).to_csv(OUT / "source_row_linkage_private.csv", index=False, encoding="utf-8-sig")
    long = []
    for patient in output:
        images = manifest[manifest.patient_alias.eq(patient["patient_alias"])]
        for side in sorted(images.side.unique()):
            subset = images[images.side.eq(side)]
            label = patient["left_APN"] if side == "left" else patient["right_APN"]
            long.append(dict(patient_id=patient["patient_id"], patient_alias=patient["patient_alias"], side=side,
                             kidney_alias=patient["patient_alias"]+("_L" if side == "left" else "_R"),
                             kidney_APN=label, DMSA_APN=patient["DMSA_APN"], source_status=patient["source_status"],
                             previous_patient_label=patient["previous_label"], image_count=len(subset),
                             image_ids=";".join(subset.image_id), views=";".join(subset["view"]),
                             source_rows=patient["primary_source_rows"] or patient["source_rows"], image_exam_date=patient["image_exam_date"],
                             nearest_admission_gap_days=patient["nearest_admission_gap_days"]))
    kidneys = pd.DataFrame(long)
    kidneys.to_csv(OUT / "kidney_label_linkage_private.csv", index=False, encoding="utf-8-sig")
    image_labels = manifest.rename(columns={"label": "previous_patient_label"}).merge(
        kidneys[["patient_alias", "side", "kidney_alias", "kidney_APN", "DMSA_APN", "source_status", "source_rows"]],
        on=["patient_alias", "side"], validate="many_to_one", how="left", sort=False)
    assert image_labels.image_id.tolist() == manifest.image_id.tolist()
    assert image_labels.source_status.notna().all()
    image_labels.to_csv(OUT / "image_labels_dmsa_private.csv", index=False, encoding="utf-8-sig")
    available = linked[linked.source_status.eq("source_side_code_available")]
    known_kidneys = kidneys[kidneys.kidney_APN.notna()]
    known_images = int(known_kidneys.image_count.sum())
    gaps = available.nearest_admission_gap_days.dropna()
    report = dict(source=str(SOURCE), source_sha256=before, source_unchanged=before == sha256(SOURCE),
                  sheets=summaries, flagged_distinct_patients=len(all_flags), exactly_matches_current_264=True,
                  main_sheet_unique_patients=len(main_ids), sheet2_unique_selected_patients=len(sheet2_ids),
                  sheet2_also_in_main=len(sheet2_ids & main_ids), sheet2_only_without_DMSA=len(sheet2_ids-main_ids),
                  source_status_counts=linked.source_status.value_counts().to_dict(),
                  patient_DMSA_code_counts=available.DMSA_APN.value_counts().to_dict(),
                  known_patient_APN_counts=available.patient_APN.value_counts().to_dict(),
                  known_patient_label_disagreements=int(available.previous_patient_label_disagrees.sum()),
                  patients_without_side_code_previous_label_counts=linked[linked.source_status.ne("source_side_code_available")].previous_label.value_counts().to_dict(),
                  imaged_kidneys_total=len(kidneys), imaged_kidneys_with_side_code=len(known_kidneys),
                  kidney_APN_counts=known_kidneys.kidney_APN.value_counts().to_dict(), images_with_side_code=known_images,
                  known_kidneys_previously_inheriting_wrong_patient_label=int((known_kidneys.kidney_APN != known_kidneys.previous_patient_label).sum()),
                  images_previously_inheriting_wrong_patient_label=int(known_kidneys.loc[known_kidneys.kidney_APN != known_kidneys.previous_patient_label, "image_count"].sum()),
                  nearest_admission_gap_days=dict(available=int(len(gaps)), exact=int(gaps.eq(0).sum()), within7=int(gaps.le(7).sum()),
                      within30=int(gaps.le(30).sum()), within90=int(gaps.le(90).sum()), over365=int(gaps.gt(365).sum()), median=float(gaps.median()) if len(gaps) else None),
                  gender_values_counts=available.gender_source_values.value_counts().to_dict(),
                  demographic_conflicts=dict(birth_date=int(linked.birth_date_conflict.sum()), gender=int(linked.gender_conflict.sum())),
                  gender_raw_encoding_difference_patients=int(linked.gender_raw_encoding_difference.sum()),
                  gender_comparison_basis="Observed paired encodings in 27 linked records support 1/M and 2/F. This is a cross-sheet comparison rule, not a separately confirmed institutional sex-code dictionary. Sex is not used as a model predictor.",
                  ambiguous_or_missing_image_exam_date_patients=int(linked.image_exam_date_ambiguous_or_missing.sum()),
                  temporal_boundary="Patient-key linkage and admission proximity are audited, not independent adjudication of the ultrasound-to-DMSA episode or scan timing",
                  prediction_exclusion="DMSA_APN is the reference outcome; DMSA_RF, DMSA_LF and DMSA_YN are not model predictors",
                  missing_label_policy="No DMSA report or blank APN field means unavailable, never APN negative. Preserve original264 as provenance but exclude unknown reference outcomes from supervised evaluation.",
                  revision_of_old_experiments="All prior264-patient labels/metrics remain historical provisional experiments and are not validated under the new DMSA side-specific reference")
    assert report["source_unchanged"]
    (OUT / "audit_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    packet = dict(summary=report,
                  patients=json.loads(linked.to_json(orient="records", force_ascii=False)),
                  kidneys=json.loads(kidneys.to_json(orient="records", force_ascii=False)))
    (OUT / "workbook_input_private.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ["source", "gender_values_counts", "sheets"]}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
