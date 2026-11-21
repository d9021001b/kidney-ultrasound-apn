"""Generate review-only masks for the 36 routed linear patients without panels."""

from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from ultralytics import YOLO

from finetune_nckuh_383_linear_segmentation_20261112 import (
    AUDIT, INITIAL, OUT as FINETUNE, ROOT, digest, predict_mask, require_protocol,
)
from nckuh_dmsa_fullcohort_fullframe_20261103 import normalize_patient_id, parse_source_key


BASE = ROOT / "03_data_models/ki-data"
WORKBOOK = ROOT / "2004-2019-NCKUH-亞大(1).xlsx"
OUT = FINETUNE / "missing_linear_candidates"
WEIGHT = FINETUNE / "finetune/weights/best.pt"


def sources():
    require_protocol()
    assert WEIGHT.is_file()
    aliases = pd.read_csv(AUDIT / "patient_linkage_private.csv", dtype=str)
    roles = pd.read_csv(AUDIT / "patient_assignments.csv", dtype=str)
    covered = set(pd.read_csv(AUDIT / "contour_image_manifest.csv").patient_alias)
    missing = roles[~roles.patient_alias.isin(covered)].merge(
        aliases, on="patient_alias", validate="one_to_one"
    )
    assert len(missing) == 36
    sheet = load_workbook(WORKBOOK, read_only=True, data_only=True)["完整資料彙整"]
    rows = sheet.iter_rows(values_only=True)
    headers = list(next(rows))
    probes = {
        normalize_patient_id(str(row[headers.index("id")]).strip()):
        str(row[headers.index("Probe")]).strip().lower()
        for row in rows if row[headers.index("id")] is not None
    }
    assert all(probes[normalize_patient_id(value)] == "linear" for value in missing.patient_id)
    keyed = {normalize_patient_id(row.patient_id): row for row in missing.itertuples(index=False)}
    files = []
    for folder in ("1", "2", "3"):
        for path in sorted((BASE / folder).iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
                continue
            key = parse_source_key(path)
            if key is None or key[0] not in keyed:
                continue
            assert key[1] is not None
            patient = keyed[key[0]]
            files.append({"patient_alias": patient.patient_alias, "role": patient.role,
                          "side": "L" if key[2].startswith("L") else "R",
                          "source_path": str(path), "source_key": key})
    files.sort(key=lambda row: (row["patient_alias"], row["source_key"]))
    assert len(files) == 138 and len({tuple(row["source_key"]) for row in files}) == 138
    assert len({row["patient_alias"] for row in files}) == 36
    assert Counter({row["patient_alias"]: row["role"] for row in files}.values()) == {
        "train": 26, "development": 6, "exploratory_test": 4,
    }
    return files


def main():
    if (OUT / "audit.json").exists():
        raise FileExistsError("Preserve existing mask-candidate audit")
    files = sources()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "masks").mkdir(exist_ok=True)
    model = YOLO(str(WEIGHT))
    public, private = [], []
    for index, source in enumerate(files, 1):
        image_id = f"U{index:04d}"
        frame = cv2.imread(source["source_path"], cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError(f"Cannot read source for {image_id}")
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        assert frame.ndim == 2 and frame.dtype == np.uint8
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        result = model.predict(rgb, imgsz=512, conf=.25, iou=.7, device=0, verbose=False)[0]
        mask = predict_mask(result, frame.shape)
        assert cv2.imwrite(str(OUT / "masks" / f"{image_id}.png"), mask * 255)
        coordinates = cv2.findNonZero(mask)
        area = int(mask.sum())
        bbox = [int(value) for value in cv2.boundingRect(coordinates)] if coordinates is not None else None
        confidence = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
        public.append({"candidate_id": image_id, "patient_alias": source["patient_alias"],
                       "role": source["role"], "side": source["side"],
                       "height": frame.shape[0], "width": frame.shape[1],
                       "instances": 0 if result.masks is None else len(result.masks.xy),
                       "max_confidence": float(confidence.max()) if len(confidence) else None,
                       "mask_pixels": area, "mask_fraction": area / mask.size,
                       "bbox_xywh": json.dumps(bbox), "review_status": "pending_manual_review"})
        private.append({"candidate_id": image_id, "source_path": source["source_path"],
                        "source_sha256": digest(Path(source["source_path"]))})
    report = pd.DataFrame(public)
    report.to_csv(OUT / "candidate_qc_deidentified.csv", index=False)
    pd.DataFrame(private).to_csv(OUT / "source_linkage_private.csv", index=False)
    summary = {"checkpoint_sha256": digest(WEIGHT), "workbook_sha256": digest(WORKBOOK),
               "candidate_images": len(report), "candidate_patients": report.patient_alias.nunique(),
               "patients_by_locked_role": report.groupby("role").patient_alias.nunique().to_dict(),
               "images_by_locked_role": report.role.value_counts().to_dict(),
               "zero_mask_images": int(report.mask_pixels.eq(0).sum()),
               "images_with_multiple_instances": int(report.instances.gt(1).sum()),
               "all_pending_manual_review": bool(report.review_status.eq("pending_manual_review").all()),
               "apn_labels_used": False, "apn_test_scored": False,
               "policy": "Predicted masks are candidates only; do not merge with APN features or call them ground truth without image-level manual QC.",
               "candidate_qc_sha256": digest(OUT / "candidate_qc_deidentified.csv"),
               "source_linkage_sha256": digest(OUT / "source_linkage_private.csv")}
    (OUT / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
