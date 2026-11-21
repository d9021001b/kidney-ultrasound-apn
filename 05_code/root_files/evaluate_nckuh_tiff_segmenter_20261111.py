"""Exploratory mask IoU on NCKUH TIFFs with left-pane manual polygons."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from nckuh_dmsa_fullcohort_fullframe_20261103 import load_polygon
from prepare_nckuh_383_contour_cohort_20261111 import digest


ROOT = Path(__file__).resolve().parents[2]
POLYGONS = ROOT / "03_data_models/ki-data/before-labeled"
WEIGHTS = Path(r"C:\Users\udoo_w2\Desktop\work-kidney-tumor\kidney-tumor\best.pt")
OUT = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"


def binary_polygon(points, shape):
    mask = np.zeros(shape, np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(mask, [np.asarray(points, np.int32)], 1)
    return mask


def iou(a, b):
    return float(np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=WEIGHTS)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    prefix = "segmenter_left_polygon_validation" + (f"_{args.tag}" if args.tag else "")
    if (OUT / f"{prefix}.csv").exists():
        raise FileExistsError("Preserve completed mask comparison")
    model = YOLO(str(args.checkpoint))
    records = []
    for json_path in sorted(POLYGONS.glob("*.json")):
        image_path = json_path.with_suffix(".tif")
        if not image_path.exists():
            continue
        source, manual = load_polygon(image_path, json_path)
        height, width = source.shape[:2]
        if (height, width) not in {(600, 800), (720, 960)}:
            continue
        half = width // 2
        if manual[:, :half].sum() <= manual[:, half:].sum():
            continue
        image = source[55:height - 65, :half]
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        reference = manual[55:height - 65, :half].astype(np.uint8)
        if reference.sum() < 256:
            continue
        result = model.predict(image, imgsz=640, conf=.25, device=0, verbose=False)[0]
        polygons = [] if result.masks is None else result.masks.xy
        confidences = [] if result.boxes is None else result.boxes.conf.cpu().numpy().tolist()
        masks = [binary_polygon(points, reference.shape) for points in polygons]
        comparisons = [iou(mask, reference) for mask in masks]
        areas = [int(mask.sum()) for mask in masks]
        top_conf = int(np.argmax(confidences)) if confidences else None
        largest = int(np.argmax(areas)) if areas else None
        records.append({"case_ordinal": len(records) + 1,
                        "resolution": f"{width}x{height}",
                        "prediction_count": len(masks), "manual_pixels": int(reference.sum()),
                        "highest_confidence": max(confidences) if confidences else None,
                        "top_conf_iou": comparisons[top_conf] if top_conf is not None else 0.0,
                        "largest_area_iou": comparisons[largest] if largest is not None else 0.0,
                        "oracle_best_iou": max(comparisons, default=0.0)})
    frame = pd.DataFrame(records)
    assert len(frame) >= 25
    frame.to_csv(OUT / f"{prefix}.csv", index=False)
    summary = {"checkpoint_sha256": digest(args.checkpoint), "cases": len(frame),
               "input": "800x600 or 960x720 TIFF left pane, rows 55:height-65, converted explicitly to BGR3",
               "cases_by_resolution": frame.resolution.value_counts().to_dict(),
               "manual_polygon": "left-pane kidney polygon only; right-pane annotations excluded",
               "no_prediction": int(frame.prediction_count.eq(0).sum()),
               "median_top_conf_iou": float(frame.top_conf_iou.median()),
               "top_conf_iou_ge_0_75": int(frame.top_conf_iou.ge(.75).sum()),
               "median_largest_area_iou": float(frame.largest_area_iou.median()),
               "largest_area_iou_ge_0_75": int(frame.largest_area_iou.ge(.75).sum()),
               "median_oracle_iou": float(frame.oracle_best_iou.median()),
               "caveat": "Exploratory local comparison; checkpoint training overlap with these validation patients is not excluded. Oracle IoU uses ground truth and cannot be used as a deployed selection rule."}
    (OUT / f"{prefix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
