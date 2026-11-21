"""Measure segmentation checkpoint transfer separately by probe on supplied masks."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from nckuh_dmsa_fullcohort_fullframe_20261103 import split_panel
from nckuh_probe_blind_sector import blind_sector
from prepare_nckuh_383_contour_cohort_20261111 import digest


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
ROUTER = EXP / "NCKUH_revised_383_probe_router_20261112"
AUDIT = EXP / "NCKUH_revised_383_audit_20261111"
WEIGHTS = {
    "external_pretrained": Path(r"C:\Users\udoo_w2\Desktop\work-kidney-tumor\kidney-tumor\best.pt"),
    "prior_finetuned": EXP / "CYCH_NCKUH_segmentation_finetune_20261004/finetune/weights/best.pt",
}
OUT = ROUTER / "segmentation_transfer"


def prediction_mask(result, shape):
    mask = np.zeros(shape, np.uint8)
    if result.masks is not None:
        for polygon in result.masks.xy:
            if len(polygon) >= 3:
                cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 1)
    return mask


def main():
    routes = pd.read_csv(ROUTER / "image_routes.csv")
    public = pd.read_csv(AUDIT / "contour_image_manifest.csv")
    private = pd.read_csv(AUDIT / "source_linkage_private.csv", dtype=str)
    selected = public[["image_id", "patient_alias", "source_type"]].merge(
        routes[["image_id", "route", "role"]], on="image_id", validate="one_to_one"
    ).merge(private[["image_id", "source_path"]], on="image_id", validate="one_to_one")
    selected = selected[selected.role.eq("development") & selected.route.isin(("convex", "linear"))]
    assert len(selected) >= 200 and selected.source_type.eq("panel_with_mask").all()
    assert all(path.is_file() for path in WEIGHTS.values())
    models = {name: YOLO(str(path)) for name, path in WEIGHTS.items()}
    records = []
    for row in selected.itertuples(index=False):
        frame, supplied = split_panel(Path(row.source_path))
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        assert supplied.shape == frame.shape[:2]
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        sector, fallback = blind_sector(frame)
        yy, xx = np.nonzero(sector)
        margin_y, margin_x = round(frame.shape[0] * .03), round(frame.shape[1] * .03)
        y0, y1 = max(0, int(yy.min()) - margin_y), min(frame.shape[0], int(yy.max()) + 1 + margin_y)
        x0, x1 = max(0, int(xx.min()) - margin_x), min(frame.shape[1], int(xx.max()) + 1 + margin_x)
        for name in ("external_pretrained", "prior_finetuned", "prior_finetuned_sector_crop"):
            cropped = name.endswith("sector_crop")
            model = models["prior_finetuned"] if cropped else models[name]
            input_image = frame[y0:y1, x0:x1] if cropped else frame
            result = model.predict(input_image, imgsz=640, conf=.25, iou=.7, device=0, verbose=False)[0]
            predicted = np.zeros(supplied.shape, np.uint8)
            if cropped:
                predicted[y0:y1, x0:x1] = prediction_mask(result, input_image.shape[:2])
            else:
                predicted = prediction_mask(result, supplied.shape)
            gt = supplied.astype(bool)
            pred = predicted.astype(bool)
            intersection = np.count_nonzero(gt & pred)
            union = np.count_nonzero(gt | pred)
            records.append({"image_id": row.image_id, "patient_alias": row.patient_alias,
                            "probe": row.route, "checkpoint": name,
                            "iou": intersection / union if union else 0.0,
                            "sector_fallback": bool(fallback),
                            "predicted_instances": 0 if result.masks is None else len(result.masks.xy),
                            "gt_pixels": int(gt.sum()), "pred_pixels": int(pred.sum())})
    frame = pd.DataFrame(records)
    assert len(frame) == len(selected) * (len(models) + 1)
    summary = {"source_probe_routes_sha256": digest(ROUTER / "image_routes.csv"),
               "checkpoint_sha256": {name: digest(path) for name, path in WEIGHTS.items()},
               "protocol": "Development-role panel clean grayscale frame; supplied right-pane mask as reference; all predicted polygons unioned; full-frame and mask-blind sector crop with 3% frame margin; 640px, conf .25, IoU NMS .7; no APN labels.",
               "groups": {},
               "caveat": "Supplied masks may be imperfect; prior fine-tuning patient overlap is not ruled out. This is segmentation transfer QC, not independent APN classification validation."}
    for (probe, checkpoint), group in frame.groupby(["probe", "checkpoint"]):
        summary["groups"][f"{probe}_{checkpoint}"] = {
            "patients": group.patient_alias.nunique(), "images": len(group),
            "mean_iou": float(group.iou.mean()), "median_iou": float(group.iou.median()),
            "iou_ge_0_75": int(group.iou.ge(.75).sum()),
            "zero_predictions": int(group.predicted_instances.eq(0).sum()),
            "sector_fallback_images": int(group.sector_fallback.sum()),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "development_per_image_deidentified.csv", index=False)
    (OUT / "development_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
