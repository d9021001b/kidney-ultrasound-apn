"""Verify the CYCH full longitudinal annotation packets and public archives."""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "CYCH_random_validation_20260913"
OUT = ROOT / "CYCH_full_longitudinal_annotation_packets_v1_20260914"
PRIMARY = OUT / "primary_public"
QC = OUT / "second_rater_qc_public"
PRIMARY_ZIP = ROOT / "CYCH_full_longitudinal_primary_v1_20260914_public.zip"
QC_ZIP = ROOT / "CYCH_full_longitudinal_second_rater_qc_v1_20260914_public.zip"
BUILDER = ROOT / "build_full_longitudinal_annotation_packets.py"
VALIDATOR = ROOT / "CYCH_anatomy_artifact_annotation_packet_v2_20260914" / "validate_internal_region_annotations.py"
CLASSES = ("cec", "medulla", "cortex", "artifact", "uncertain")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_packet(packet: Path, expected_tasks: int) -> dict[str, object]:
    tasks = pd.read_csv(packet / "annotation_tasks_blinded.csv", dtype={"task_id": str})
    status = pd.read_csv(packet / "ANNOTATION_STATUS_TEMPLATE.csv", dtype={"task_id": str})
    require(list(tasks.columns) == ["task_id", "display_order", "view"], "Public task columns changed")
    require(len(tasks) == expected_tasks and tasks.task_id.nunique() == expected_tasks, "Task count mismatch")
    require(len(status) == expected_tasks and set(status.task_id) == set(tasks.task_id), "Status task mismatch")
    for name in ("cec", "medulla", "cortex", "artifact"):
        require(status[f"{name}_status"].eq("not_annotated").all(), f"Unexpected initial {name} status")
    expected_template_count = expected_tasks * len(CLASSES)
    templates = list((packet / "annotation_templates").glob("*.png"))
    require(len(templates) == expected_template_count, "Template count mismatch")
    for path in templates:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        require(image is not None and int(np.count_nonzero(image)) == 0, f"Template is not blank: {path.name}")
    qc_summary = json.loads((OUT / f"{packet.name.replace('_public', '')}_template_qc" / "annotation_qc_summary.json").read_text(encoding="utf-8"))
    require(qc_summary["rows_with_errors"] == 0, f"Template validator errors in {packet.name}")
    with Image.open(packet / "figures_600dpi" / "Figure_FL1_blinded_longitudinal_packet.tif") as figure:
        dpi = tuple(round(float(value)) for value in figure.info.get("dpi", (0, 0)))
        require(dpi == (600, 600), f"TIFF dpi mismatch: {dpi}")
    return {
        "tasks": expected_tasks,
        "views": tasks.view.value_counts().sort_index().to_dict(),
        "blank_templates": expected_template_count,
        "structural_qc_error_rows": qc_summary["rows_with_errors"],
        "tiff_embedded_dpi": list(dpi),
    }


def verify_linkage_and_pixels() -> dict[str, object]:
    private = pd.read_csv(OUT / "private_task_key.csv", dtype={"patient_id": str, "task_id": str})
    qc_key = pd.read_csv(OUT / "private_second_rater_qc_key.csv", dtype={"patient_id": str, "task_id": str})
    require(len(private) == 190 and private.image_id.nunique() == 190, "Private primary linkage mismatch")
    require(private.patient_id.nunique() == 98, "Longitudinal packet does not cover 98 patients")
    require(private.view.value_counts().to_dict() == {"RL": 96, "LL": 94}, "Primary view counts changed")
    require(len(qc_key) == 38 and qc_key.patient_id.nunique() == 38, "QC subset is not 38 distinct patients")
    require(qc_key.view.value_counts().to_dict() == {"LL": 19, "RL": 19}, "QC subset is not view-balanced")
    require(set(qc_key.task_id).issubset(set(private.task_id)), "QC tasks are not a primary subset")
    private_index = private.set_index("task_id")
    for task in private.itertuples(index=False):
        task_id, image_id = str(task.task_id), str(task.image_id)
        original_image = cv2.imread(str(BASE / "native" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        public_image = cv2.imread(str(PRIMARY / "images" / f"{task_id}.png"), cv2.IMREAD_GRAYSCALE)
        original_mask = cv2.imread(str(BASE / "masks" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        public_mask = cv2.imread(str(PRIMARY / "outer_roi_masks" / f"{task_id}.png"), cv2.IMREAD_GRAYSCALE)
        require(original_image is not None and np.array_equal(original_image, public_image), f"Image pixel mismatch: {task_id}")
        require(original_mask is not None, f"Original mask missing: {image_id}")
        require(np.array_equal(original_mask > 0, public_mask > 0), f"Mask pixel mismatch: {task_id}")
    for task_id in qc_key.task_id:
        require(task_id in private_index.index, f"QC task missing from primary: {task_id}")
        require(
            np.array_equal(
                cv2.imread(str(PRIMARY / "images" / f"{task_id}.png"), cv2.IMREAD_GRAYSCALE),
                cv2.imread(str(QC / "images" / f"{task_id}.png"), cv2.IMREAD_GRAYSCALE),
            ),
            f"QC image differs from primary: {task_id}",
        )
    return {
        "primary_images": len(private),
        "covered_patients": int(private.patient_id.nunique()),
        "primary_views": private.view.value_counts().sort_index().to_dict(),
        "second_rater_qc_images": len(qc_key),
        "second_rater_qc_distinct_patients": int(qc_key.patient_id.nunique()),
        "second_rater_qc_views": qc_key.view.value_counts().sort_index().to_dict(),
        "source_pixel_checks": len(private),
    }


def verify_archive(path: Path, expected_entries: int) -> dict[str, object]:
    forbidden = re.compile(r"private|patient|source_lock|filename|label|_red_", re.IGNORECASE)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == expected_entries, f"Archive entry count mismatch for {path.name}")
        require(archive.testzip() is None, f"CRC failure in {path.name}")
        prohibited = [name for name in names if forbidden.search(name)]
        require(not prohibited, f"Prohibited public archive names: {prohibited[:5]}")
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "entries": expected_entries,
        "crc_errors": 0,
        "prohibited_entry_names": 0,
    }


def main() -> None:
    source_lock = json.loads((OUT / "source_lock.json").read_text(encoding="utf-8"))
    require(source_lock["diagnosis_label_read"] is False, "Source lock says diagnosis labels were read")
    require(source_lock["builder_sha256"] == sha256(BUILDER), "Builder hash mismatch")
    require(source_lock["validator_sha256"] == sha256(VALIDATOR), "Validator hash mismatch")
    result = {
        "status": "verified",
        "source_lock": source_lock,
        "linkage_and_pixels": verify_linkage_and_pixels(),
        "primary_packet": verify_packet(PRIMARY, 190),
        "second_rater_qc_packet": verify_packet(QC, 38),
        "primary_archive": verify_archive(PRIMARY_ZIP, 1529),
        "second_rater_qc_archive": verify_archive(QC_ZIP, 313),
    }
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
