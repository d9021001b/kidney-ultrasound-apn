"""Compare revised probe workbook to locked cohort without exporting MRNs."""

from collections import Counter
import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from prepare_nckuh_383_contour_cohort_20261111 import digest


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "2004-2019-NCKUH-亞大.xlsx"
NEW = ROOT / "2004-2019-NCKUH-亞大(1).xlsx"
BASE = ROOT / "03_data_models/ki-data"
AUDIT = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
OUT = ROOT / "02_experiments/NCKUH_revised_383_probe_router_20261112"


def roster(path):
    sheet = load_workbook(path, read_only=True, data_only=True)["完整資料彙整"]
    values = sheet.iter_rows(values_only=True)
    header = list(next(values))
    rows = {str(row[0]).strip().lstrip("0"): row for row in values if row[0] is not None}
    assert len(rows) == 383
    return header, rows


def main():
    old_header, old = roster(OLD)
    new_header, new = roster(NEW)
    assert set(old) == set(new)
    assert old_header[:7] == new_header[:7]
    assert new_header[7] == "Probe"
    clinical_changes = Counter()
    for key in old:
        for col in range(6):
            if str(old[key][col]).strip() != str(new[key][col]).strip():
                clinical_changes[col] += 1
    assert not clinical_changes, "Source ID or clinical fields changed"
    changed_folders = [key for key in old if str(old[key][6]) != str(new[key][6])]
    assert len(changed_folders) == 139
    linkage = pd.read_csv(AUDIT / "source_linkage_private.csv", dtype=str)
    images = pd.read_csv(AUDIT / "contour_image_manifest.csv")
    contour_ids = set(linkage[linkage.image_id.isin(images.image_id)].patient_id.str.lstrip("0"))
    no_contour = set(new) - contour_ids
    assert len(no_contour) == 36
    existing = []
    for folder in ("1", "2", "3", "NCKUH3", "test_predict"):
        existing.extend(path for path in (BASE / folder).iterdir()
                        if path.is_file() and path.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
                        and path.name.split("_", 1)[0].lstrip("0") in no_contour)
    patients_with_raw = {path.name.split("_", 1)[0].lstrip("0") for path in existing}
    assert patients_with_raw == no_contour
    listed = sorted({part.strip() for row in new.values() for part in str(row[6]).split(",")})
    result = {
        "old_workbook_sha256": digest(OLD), "new_workbook_sha256": digest(NEW),
        "same_383_ids": True, "changes_in_original_columns_0_through_5": dict(clinical_changes),
        "changed_image_folder_entries": len(changed_folders),
        "new_probe_patient_counts": dict(Counter(str(row[7]).strip().lower() for row in new.values())),
        "patients_without_current_contour_source": len(no_contour),
        "probe_of_patients_without_contour_source": dict(Counter(str(new[key][7]).strip().lower() for key in no_contour)),
        "patients_without_contour_with_existing_raw_images": len(patients_with_raw),
        "raw_image_files_for_those_patients": len(existing),
        "new_listed_folders_absent_on_disk": [folder for folder in listed if not (BASE / folder).is_dir()],
        "caveat": "Existing raw images do not imply acceptable kidney contours. No medical record numbers or raw filenames are exported.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "workbook_reconciliation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
