"""Audit supplied red kidney outlines in previously contour-missing TIFFs."""

from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
import pandas as pd
from openpyxl import load_workbook

from nckuh_dmsa_fullcohort_fullframe_20261103 import normalize_patient_id, parse_source_key
from validate_nckuh_tiff_red_contour_20261111 import red_contour_mask


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "2004-2019-NCKUH-亞大.xlsx"
IMAGE_ROOT = ROOT / "03_data_models/ki-data"
OUT = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"


def main():
    ws = load_workbook(SOURCE, read_only=True, data_only=True)["完整資料彙整"]
    allowed = {normalize_patient_id(str(row[0]).strip()):
               {value.strip() for value in str(row[6]).split(",")}
               for row in list(ws.iter_rows(values_only=True))[1:] if row[0] is not None}
    ready = pd.read_csv(OUT / "source_linkage_private.csv", dtype=str)
    ready_keys = set(zip(ready.patient_id, ready.exam_date_raw, pd.read_csv(
        OUT / "contour_image_manifest.csv").view))
    raw = defaultdict(list)
    for folder in ("1", "2", "3"):
        for path in (IMAGE_ROOT / folder).glob("*.tif"):
            patient_id = normalize_patient_id(path.name.split("_", 1)[0])
            if folder not in allowed.get(patient_id, ()):
                continue
            key = parse_source_key(path)
            if key is not None and key[1] is not None and key not in ready_keys:
                raw[key].append(path)
    rows, private = [], []
    for key in sorted(raw):
        path = sorted(raw[key], key=str)[0]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape[1] % 2:
            status, pixels, fraction, candidates = "unreadable", 0, 0.0, 0
        else:
            frame = image[:, image.shape[1] // 2:]
            mask, candidates = red_contour_mask(frame)
            pixels = int(mask.sum())
            fraction = float(mask.mean())
            status = "candidate" if .02 <= fraction <= .4 and candidates == 1 else "review_needed"
        alias = f"R{len(rows) + 1:04d}"
        rows.append({"candidate_id": alias, "view": key[2], "status": status,
                     "contour_candidates": candidates, "mask_pixels": pixels,
                     "mask_fraction": fraction})
        private.append({"candidate_id": alias, "patient_id": key[0],
                        "exam_date_raw": key[1], "view": key[2],
                        "source_path": str(path)})
    public = pd.DataFrame(rows)
    assert len(public) >= 130
    public.to_csv(OUT / "red_contour_recovery_candidates.csv", index=False)
    pd.DataFrame(private).to_csv(OUT / "red_contour_recovery_source_private.csv", index=False)
    summary = {"missing_unique_keys_with_TIFF": len(public),
               "candidate_status": public.status.value_counts().to_dict(),
               "candidate_patients": len({row["patient_id"] for row, item in zip(private, rows)
                                           if item["status"] == "candidate"}),
               "mask_fraction_min_median_max_for_candidates": [
                   float(public.loc[public.status.eq("candidate"), "mask_fraction"].min()),
                   float(public.loc[public.status.eq("candidate"), "mask_fraction"].median()),
                   float(public.loc[public.status.eq("candidate"), "mask_fraction"].max())],
               "rule": "one closed red outline in right pane; filled mask fraction 0.02-0.40",
               "caveat": "Candidate transcription only. Visual/anatomical review and artifact removal before classification; no APN label used."}
    (OUT / "red_contour_recovery_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
