"""Audit the revised NCKUH clinical roster against local raw image folders.

Only aggregate counts are emitted. The source workbook and clinical identifiers
remain local and are never copied into the output.
"""

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import csv
import json
import re

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "2004-2019-NCKUH-亞大.xlsx"
IMAGE_ROOT = ROOT / "03_data_models/ki-data"
OLD_LINKAGE = ROOT / "02_experiments/NCKUH_DMSA_label_audit_20261102/patient_label_linkage_private.csv"
OUT = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
FOLDERS = ("1", "2", "3", "NCKUH3", "test_predict")
EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".dcm"}
VIEW_RE = re.compile(r"_(LL|LT|RL|RT)(?:$|[- (])", re.IGNORECASE)


def main():
    wb = load_workbook(SOURCE, read_only=True, data_only=True)
    ws = wb["完整資料彙整"]
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    assert header[5] == "DMSA_APN" and header[6] == "對應影像檔"
    patients = {}
    for row in rows:
        if row[0] is None:
            continue
        patient_id = str(row[0]).strip()
        assert patient_id not in patients
        code = str(row[5]).strip().upper()
        assert code in {"R", "L", "B", "N"}
        folders = {x.strip() for x in str(row[6]).split(",")}
        assert folders and folders <= set(FOLDERS)
        patients[patient_id] = (code, folders)
    assert len(patients) == 383

    old = {}
    with OLD_LINKAGE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            old[row["patient_id"]] = row

    files = defaultdict(lambda: defaultdict(list))
    suffix_counts = Counter()
    for folder in FOLDERS:
        for path in (IMAGE_ROOT / folder).rglob("*"):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            suffix_counts[path.suffix.lower()] += 1
            patient_id = path.name.split("_", 1)[0]
            files[folder][patient_id].append(path)

    clinical = Counter()
    image_status = Counter()
    folder_status = Counter()
    old_status = Counter()
    code_disagreement = Counter()
    side_coverage = Counter()
    imaged_kidney_labels = Counter()
    image_parse = Counter()
    matched_images = 0
    unmatched_directory_ids = set()
    for patient_id, (code, folders) in patients.items():
        clinical[code] += 1
        counts = {folder: len(files[folder].get(patient_id, [])) for folder in folders}
        total = sum(counts.values())
        image_status["has_image" if total else "no_image_in_listed_folder"] += 1
        matched_images += total
        sides = set()
        for folder in folders:
            for path in files[folder].get(patient_id, []):
                match = VIEW_RE.search(path.stem)
                if match:
                    side = "left" if match.group(1).upper().startswith("L") else "right"
                    sides.add(side)
                    image_parse["recognized_LL_LT_RL_RT"] += 1
                else:
                    image_parse["unrecognized_view"] += 1
        if sides == {"left", "right"}:
            side_coverage["both_sides"] += 1
        elif sides:
            side_coverage[f"{next(iter(sides))}_only"] += 1
        else:
            side_coverage["no_parsed_side"] += 1
        for side in sides:
            positive = side[0].upper() in code or code == "B"
            imaged_kidney_labels["APN" if positive else "non_APN"] += 1
        for folder, count in counts.items():
            folder_status[f"{folder}:has_image" if count else f"{folder}:no_image"] += 1
        previous = old.get(patient_id)
        old_status["in_old_264" if previous else "not_in_old_264"] += 1
        if previous:
            old_status["in_old_210_labeled" if previous["DMSA_APN"] else "in_old_54_unlabeled"] += 1
            if previous["DMSA_APN"] and previous["DMSA_APN"] != code:
                code_disagreement[f"{previous['DMSA_APN']}->{code}"] += 1
    for folder in FOLDERS:
        unmatched_directory_ids.update(set(files[folder]) - set(patients))

    summary = {
        "source_workbook": SOURCE.name,
        "source_sha256": sha256(SOURCE.read_bytes()).hexdigest(),
        "sheet": ws.title,
        "patients": len(patients),
        "DMSA_APN_counts": dict(sorted(clinical.items())),
        "image_patient_status": dict(sorted(image_status.items())),
        "image_folder_link_status": dict(sorted(folder_status.items())),
        "filename_view_parse": dict(sorted(image_parse.items())),
        "patient_side_coverage_from_filename": dict(sorted(side_coverage.items())),
        "kidneys_with_filename_side_and_DMSA_label": dict(sorted(imaged_kidney_labels.items())),
        "matched_raw_image_files_in_listed_folders": matched_images,
        "all_raw_image_extensions_in_five_folders": dict(sorted(suffix_counts.items())),
        "old_cohort_overlap": dict(sorted(old_status.items())),
        "old_vs_new_code_disagreements": dict(sorted(code_disagreement.items())),
        "directory_patient_ids_not_in_revised_roster": len(unmatched_directory_ids),
        "probe_type": "not established by workbook or filename; image review or acquisition metadata required",
        "scope": "Raw files only; no segmentation QC, probe filtering or model evaluation performed",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
