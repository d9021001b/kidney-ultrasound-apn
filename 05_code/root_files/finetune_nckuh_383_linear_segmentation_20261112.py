"""Fine-tune the external kidney segmenter on routed linear training patients."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

from nckuh_dmsa_fullcohort_fullframe_20261103 import split_panel
from prepare_nckuh_383_contour_cohort_20261111 import digest


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
AUDIT = EXP / "NCKUH_revised_383_audit_20261111"
ROUTER = EXP / "NCKUH_revised_383_probe_router_20261112"
OUT = ROUTER / "linear_segmentation_finetune"
INITIAL = Path(r"C:\Users\udoo_w2\Desktop\work-kidney-tumor\kidney-tumor\best.pt")
SEED = 20261112


def polygon_text(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
    assert len(approx) >= 3
    h, w = mask.shape
    points = np.clip(approx / np.asarray([w, h], dtype=np.float32), 0, 1).ravel()
    return "0 " + " ".join(f"{value:.6f}" for value in points) + "\n"


def source_rows():
    route = pd.read_csv(ROUTER / "image_routes.csv")
    public = pd.read_csv(AUDIT / "contour_image_manifest.csv")
    private = pd.read_csv(AUDIT / "source_linkage_private.csv", dtype=str)
    rows = public[["image_id", "patient_alias", "source_type"]].merge(
        route[["image_id", "route", "role"]], on="image_id", validate="one_to_one"
    ).merge(private[["image_id", "source_path"]], on="image_id", validate="one_to_one")
    rows = rows[rows.route.eq("linear") & rows.role.isin(("train", "development"))].reset_index(drop=True)
    assert rows.source_type.eq("panel_with_mask").all()
    assert rows.groupby("patient_alias").role.nunique().eq(1).all()
    assert rows.role.value_counts().to_dict() == {"train": 227, "development": 80}
    return rows


def prepare():
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve existing segmentation split")
    assert INITIAL.is_file() and torch.cuda.is_available()
    rows = source_rows()
    patient_train = sorted(rows.loc[rows.role.eq("train"), "patient_alias"].unique())
    train, val = train_test_split(patient_train, test_size=.2, random_state=SEED)
    assert set(train).isdisjoint(val)
    rows["seg_role"] = np.where(rows.role.eq("development"), "heldout_development",
                                np.where(rows.patient_alias.isin(val), "val", "train"))
    assert rows.groupby("patient_alias").seg_role.nunique().eq(1).all()
    for row in rows.itertuples(index=False):
        if row.seg_role == "heldout_development":
            continue
        frame, mask = split_panel(Path(row.source_path))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        base = OUT / "dataset"
        image_path = base / "images" / row.seg_role / f"{row.image_id}.png"
        label_path = base / "labels" / row.seg_role / f"{row.image_id}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image_path), gray)
        label_path.write_text(polygon_text(mask), encoding="ascii")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = rows[["image_id", "patient_alias", "role", "seg_role"]]
    manifest.to_csv(OUT / "manifest_deidentified.csv", index=False)
    config = {"path": str(OUT / "dataset"), "train": "images/train", "val": "images/val",
              "names": {0: "kidney"}}
    (OUT / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    params = {"epochs": 30, "imgsz": 512, "batch": 12, "freeze": 10,
              "optimizer": "AdamW", "lr0": .0001, "lrf": .1, "patience": 8,
              "seed": SEED, "degrees": 7, "translate": .05, "scale": .1,
              "fliplr": .5, "flipud": 0, "mosaic": 0, "mixup": 0,
              "hsv_h": 0, "hsv_s": 0, "hsv_v": .1}
    protocol = {"source_router_sha256": digest(ROUTER / "image_routes.csv"),
                "source_manifest_sha256": digest(AUDIT / "contour_image_manifest.csv"),
                "initial_weight_sha256": digest(INITIAL),
                "initialization": "Externally pretrained kidney segmenter; separate original patient cohort by author confirmation.",
                "data": "Only routed linear training-role paired-panel clean left frames and supplied right-pane contours; patient-level 80/20 internal train/val. Original APN test and development patients never fit or select checkpoint.",
                "training": params,
                "counts": manifest.seg_role.value_counts().to_dict(),
                "patients": manifest.groupby("seg_role").patient_alias.nunique().to_dict(),
                "manifest_sha256": digest(OUT / "manifest_deidentified.csv"),
                "test_APN_scored": False}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2))


def require_protocol():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    assert digest(ROUTER / "image_routes.csv") == protocol["source_router_sha256"]
    assert digest(AUDIT / "contour_image_manifest.csv") == protocol["source_manifest_sha256"]
    assert digest(INITIAL) == protocol["initial_weight_sha256"]
    assert digest(OUT / "manifest_deidentified.csv") == protocol["manifest_sha256"]
    return protocol


def train():
    protocol = require_protocol()
    if (OUT / "finetune/weights/best.pt").exists():
        raise FileExistsError("Preserve completed checkpoint")
    torch.set_num_threads(4)
    model = YOLO(str(INITIAL))
    model.train(data=str(OUT / "data.yaml"), project=str(OUT), name="finetune",
                exist_ok=False, **protocol["training"], device=0, workers=0,
                cache=True, amp=True, plots=False, deterministic=True,
                warmup_epochs=1, weight_decay=.0001, close_mosaic=0,
                save=True, verbose=False)
    best = OUT / "finetune/weights/best.pt"
    assert best.is_file()
    (OUT / "training_complete.json").write_text(json.dumps(
        {"checkpoint_sha256": digest(best), "device": torch.cuda.get_device_name(0)}, indent=2),
        encoding="utf-8")


def predict_mask(result, shape):
    mask = np.zeros(shape, np.uint8)
    if result.masks is not None:
        for polygon in result.masks.xy:
            if len(polygon) >= 3:
                cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 1)
    return mask


def evaluate(baseline=False):
    require_protocol()
    best = INITIAL if baseline else OUT / "finetune/weights/best.pt"
    prefix = "baseline_512" if baseline else "development"
    summary_path = OUT / f"{prefix}_segmentation_summary.json"
    if not best.is_file() or summary_path.exists():
        raise FileExistsError("Checkpoint missing or development already evaluated")
    rows = source_rows()
    rows = rows[rows.role.eq("development")]
    model = YOLO(str(best))
    records = []
    for row in rows.itertuples(index=False):
        frame, gt = split_panel(Path(row.source_path))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = model.predict(rgb, imgsz=512, conf=.25, iou=.7, device=0, verbose=False)[0]
        predicted = predict_mask(result, gt.shape)
        intersection = int(np.count_nonzero((gt > 0) & (predicted > 0)))
        union = int(np.count_nonzero((gt > 0) | (predicted > 0)))
        records.append({"patient_alias": row.patient_alias, "image_id": row.image_id,
                        "iou": intersection / union if union else 0,
                        "instances": 0 if result.masks is None else len(result.masks.xy),
                        "gt_pixels": int(gt.sum()), "pred_pixels": int(predicted.sum())})
    frame = pd.DataFrame(records)
    assert len(frame) == 80 and frame.patient_alias.nunique() == 20
    summary = {"checkpoint_sha256": digest(best), "input": "full grayscale frame, 512px, conf 0.25, NMS IoU 0.7", "development_images": len(frame),
               "development_patients": frame.patient_alias.nunique(),
               "image_mean_iou": float(frame.iou.mean()),
               "image_median_iou": float(frame.iou.median()),
               "patient_mean_iou": float(frame.groupby("patient_alias").iou.mean().mean()),
               "images_iou_ge_0_75": int(frame.iou.ge(.75).sum()),
               "images_no_prediction": int(frame.instances.eq(0).sum()),
               "test_APN_scored": False,
               "caveat": "Supplied contours are the reference; their clinical authorship and accuracy need independent review. APN model inputs must be rebuilt under the locked patient role after any contour change."}
    frame.to_csv(OUT / f"{prefix}_per_image_deidentified.csv", index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train", "evaluate", "baseline"))
    stage = parser.parse_args().stage
    {"prepare": prepare, "train": train, "evaluate": evaluate,
     "baseline": lambda: evaluate(baseline=True)}[stage]()
