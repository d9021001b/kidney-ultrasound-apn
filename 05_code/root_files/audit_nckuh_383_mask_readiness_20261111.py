"""Reconcile revised NCKUH images to contour sources without changing labels."""

from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
from openpyxl import load_workbook

from nckuh_dmsa_fullcohort_fullframe_20261103 import (
    normalize_patient_id, parse_source_key, polygon_index, split_panel,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "2004-2019-NCKUH-亞大.xlsx"
IMAGE_ROOT = ROOT / "03_data_models/ki-data"
OUT = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
FOLDERS = ("1", "2", "3", "NCKUH3", "test_predict")
EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
PANEL_FOLDERS = {"NCKUH3", "test_predict"}


def main():
    ws = load_workbook(SOURCE, read_only=True, data_only=True)["完整資料彙整"]
    patients = {normalize_patient_id(str(row[0]).strip()): (str(row[5]).strip().upper(),
                                                           {x.strip() for x in str(row[6]).split(",")})
                for row in list(ws.iter_rows(values_only=True))[1:] if row[0] is not None}
    assert len(patients) == 383
    raw = defaultdict(list)
    unparsed = Counter()
    for folder in FOLDERS:
        for path in (IMAGE_ROOT / folder).iterdir():
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            patient_id = normalize_patient_id(path.name.split("_", 1)[0])
            if patient_id not in patients or folder not in patients[patient_id][1]:
                continue
            key = parse_source_key(path)
            if key is None or key[1] is None:
                unparsed[folder] += 1
            else:
                raw[key].append((folder, path))
    polygons = polygon_index()
    source_counts = Counter()
    patient_ready = set()
    kidney_ready = set()
    unique_keys_by_folder = Counter()
    for key, candidates in raw.items():
        patient_id, _, view = key
        side = "left" if view.startswith("L") else "right"
        for folder, _ in candidates:
            unique_keys_by_folder[folder] += 1
        # The same examination/view exported to multiple folders is one instance.
        panels = sorted((path for folder, path in candidates if folder in PANEL_FOLDERS),
                        key=lambda path: ("test_predict" not in path.parts, str(path)))
        valid_panel = False
        for path in panels:
            try:
                split_panel(path)
                valid_panel = True
                break
            except (ValueError, cv2.error):
                continue
        if valid_panel:
            source_type = "panel_with_mask"
        elif key in polygons:
            source_type = "labelme_polygon"
        else:
            source_type = "no_contour_source"
        source_counts[source_type] += 1
        if source_type != "no_contour_source":
            patient_ready.add(patient_id)
            kidney_ready.add((patient_id, side))
    labels = Counter()
    for patient_id, side in kidney_ready:
        code = patients[patient_id][0]
        positive = code == "B" or side[0].upper() in code
        labels["APN" if positive else "non_APN"] += 1
    summary = {
        "patients_in_revised_roster": len(patients),
        "unique_patient_date_view_keys": len(raw),
        "source_type_by_unique_key": dict(sorted(source_counts.items())),
        "source_folder_key_occurrences_before_deduplication": dict(sorted(unique_keys_by_folder.items())),
        "unparsed_or_dateless_raw_files_by_folder": dict(sorted(unparsed.items())),
        "patients_with_at_least_one_contour_source": len(patient_ready),
        "kidneys_with_at_least_one_contour_source": len(kidney_ready),
        "DMSA_labels_of_contour_source_kidneys": dict(sorted(labels.items())),
        "limitations": "Panel mask presence or LabelMe file existence only. Contour accuracy, anatomical laterality and episode timing need separate QA. No clinical label was used to choose a source.",
    }
    target = OUT / "mask_source_readiness.json"
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
