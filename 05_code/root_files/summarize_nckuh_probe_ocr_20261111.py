"""Summarize OCR probe labels at the patient level without exporting identifiers."""

from collections import Counter, defaultdict
import csv
import json

import cv2
from openpyxl import load_workbook

from nckuh_probe_overlay_ocr_20261111 import ROOT, SOURCE, OUT, source_images


def main():
    result = json.loads((OUT / "probe_overlay_ocr_full_test_predict.json").read_text())
    paths = source_images()["test_predict"]
    assert result["total_examined"] == len(paths) == 826
    ws = load_workbook(SOURCE, read_only=True, data_only=True)["完整資料彙整"]
    clinical = {str(row[0]).strip(): str(row[5]).strip().upper()
                for row in list(ws.iter_rows(values_only=True))[1:] if row[0] is not None}
    with (ROOT / "02_experiments/NCKUH_DMSA_label_audit_20261102/patient_label_linkage_private.csv").open(
            encoding="utf-8-sig", newline="") as handle:
        old_labeled = {row["patient_id"] for row in csv.DictReader(handle) if row["DMSA_APN"]}
    patient_probes = defaultdict(set)
    patients_per_probe = defaultdict(set)
    patient_code_by_probe = defaultdict(dict)
    code_by_probe = defaultdict(Counter)
    old_by_probe = defaultdict(Counter)
    confidence = defaultdict(list)
    panel_qc = defaultdict(Counter)
    patients_with_valid_panel = defaultdict(set)
    for record in result["deidentified_records"]:
        path = paths[record["file_ordinal"]]
        patient_id = path.name.split("_", 1)[0]
        probe = record["probe"]
        patient_probes[patient_id].add(probe)
        patients_per_probe[probe].add(patient_id)
        patient_code_by_probe[probe][patient_id] = clinical[patient_id]
        code_by_probe[probe][clinical[patient_id]] += 1
        old_by_probe[probe]["old_210" if patient_id in old_labeled else "outside_old_210"] += 1
        if probe in ("convex", "linear"):
            confidence[probe].append(record["confidence"])
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape[1] % 2:
            status = "unreadable_or_odd_width"
        else:
            half = image.shape[1] // 2
            binary = (image[:, half:] > 127).astype("uint8")
            count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
            fraction = largest / (image.shape[0] * half)
            status = "panel_mask_present" if largest >= 256 and .002 <= fraction <= .5 else "panel_mask_unusable"
        panel_qc[probe][status] += 1
        if status == "panel_mask_present":
            patients_with_valid_panel[probe].add(patient_id)
    summary = {
        "source": "test_predict only; OCR candidate labels, not clinical probe verification",
        "images": len(paths),
        "patients": len(patient_probes),
        "patients_with_each_probe_candidate": {k: len(v) for k, v in sorted(patients_per_probe.items())},
        "patient_DMSA_code_by_probe_candidate": {
            k: dict(sorted(Counter(v.values()).items())) for k, v in sorted(patient_code_by_probe.items())},
        "patients_with_both_explicit_convex_and_linear": sum(
            {"convex", "linear"} <= probes for probes in patient_probes.values()),
        "patients_with_only_unknown": sum(probes == {"unknown"} for probes in patient_probes.values()),
        "image_DMSA_code_by_probe_candidate": {k: dict(sorted(v.items())) for k, v in sorted(code_by_probe.items())},
        "image_old_cohort_overlap_by_probe_candidate": {k: dict(sorted(v.items())) for k, v in sorted(old_by_probe.items())},
        "minimum_OCR_confidence_of_explicit_probe": {k: min(v) for k, v in sorted(confidence.items())},
        "panel_mask_presence_by_probe_candidate": {k: dict(sorted(v.items())) for k, v in sorted(panel_qc.items())},
        "patients_with_panel_mask_by_probe_candidate": {k: len(v) for k, v in sorted(patients_with_valid_panel.items())},
        "mask_caveat": "Binary component presence is not contour accuracy or anatomical-side verification",
        "interpretation": "Counts describe machine-overlay OCR candidates. Probe-dependent performance remains untested.",
    }
    target = OUT / "probe_overlay_patient_summary_test_predict.json"
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
