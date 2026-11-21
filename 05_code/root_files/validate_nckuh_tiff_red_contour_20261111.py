"""Compare supplied red-outline masks with same-pane LabelMe kidney polygons."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nckuh_dmsa_fullcohort_fullframe_20261103 import load_polygon


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "03_data_models/ki-data/before-labeled"
OUT = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"


def red_contour_mask(frame):
    b, g, r = cv2.split(frame[:, :, :3])
    red = ((r > 140) & (r > g.astype(np.float32) * 1.5) &
           (r > b.astype(np.float32) * 1.5)).astype(np.uint8)
    red[:max(35, frame.shape[0] // 12)] = 0
    red[frame.shape[0] - max(55, frame.shape[0] // 8):] = 0
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [contour for contour in contours if cv2.contourArea(contour) >= 1000]
    if not candidates:
        return np.zeros(red.shape, np.uint8), 0
    selected = max(candidates, key=cv2.contourArea)
    mask = np.zeros(red.shape, np.uint8)
    cv2.drawContours(mask, [selected], -1, 1, -1)
    return mask, len(candidates)


def main():
    if (OUT / "red_contour_labelme_validation.csv").exists():
        raise FileExistsError("Preserve red-contour validation")
    rows = []
    for json_path in sorted(SOURCE.glob("*.json")):
        image_path = json_path.with_suffix(".tif")
        if not image_path.exists():
            continue
        image, manual = load_polygon(image_path, json_path)
        width = image.shape[1]
        if width % 2:
            continue
        half = width // 2
        if manual[:, half:].sum() <= manual[:, :half].sum():
            continue
        predicted, count = red_contour_mask(image[:, half:])
        reference = manual[:, half:].astype(np.uint8)
        overlap = np.count_nonzero(predicted & reference)
        union = np.count_nonzero(predicted | reference)
        iou = overlap / union if union else 0.0
        rows.append({"case_ordinal": len(rows) + 1, "resolution": f"{width}x{image.shape[0]}",
                     "contour_candidates": count, "predicted_pixels": int(predicted.sum()),
                     "manual_pixels": int(reference.sum()), "iou": float(iou)})
    frame = pd.DataFrame(rows)
    assert len(frame) >= 40
    frame.to_csv(OUT / "red_contour_labelme_validation.csv", index=False)
    by_resolution = frame.groupby("resolution").agg(
        cases=("iou", "size"), median_iou=("iou", "median"),
        iou_ge_0_75=("iou", lambda x: int((x >= .75).sum())),
        no_contour=("predicted_pixels", lambda x: int((x == 0).sum())))
    summary = {"cases": len(frame), "median_iou": float(frame.iou.median()),
               "iou_ge_0_75": int(frame.iou.ge(.75).sum()),
               "by_resolution": by_resolution.to_dict("index"),
               "method": "Largest closed red line in the same right-hand TIFF pane, filled; no APN label used",
               "scope": "Source-contour transcription versus LabelMe, not external model segmentation validation",
               "caveat": "The two annotations may derive from the same human tracing; IoU measures transcription fidelity, not independent anatomical accuracy."}
    (OUT / "red_contour_labelme_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
