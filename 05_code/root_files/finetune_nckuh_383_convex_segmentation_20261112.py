"""Fine-tune external kidney segmenter on routed convex training patients."""

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

from finetune_nckuh_383_linear_segmentation_20261112 import INITIAL, polygon_text, predict_mask
from nckuh_dmsa_fullcohort_fullframe_20261103 import split_panel
from prepare_nckuh_383_contour_cohort_20261111 import OUT as REGISTRY, ROOT, digest


EXP = ROOT / "02_experiments"
ROUTER = EXP / "NCKUH_revised_383_probe_router_20261112"
OUT = ROUTER / "convex_segmentation_finetune"
SEED = 20261112


def source_rows():
    routes = pd.read_csv(ROUTER / "image_routes.csv")
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    private = pd.read_csv(REGISTRY / "source_linkage_private.csv", dtype=str)
    joined = images[["image_id", "patient_alias", "source_type"]].merge(
        routes[["image_id", "route", "role"]], on="image_id", validate="one_to_one"
    ).merge(private[["image_id", "source_path"]], on="image_id", validate="one_to_one")
    joined = joined[joined.route.eq("convex") & joined.role.isin(("train", "development"))].reset_index(drop=True)
    assert joined.source_type.eq("panel_with_mask").all()
    assert joined.groupby("patient_alias").role.nunique().eq(1).all()
    assert joined.role.value_counts().to_dict() == {"train": 566, "development": 195}
    return joined


def prepare():
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve convex segmentation split")
    assert INITIAL.is_file() and torch.cuda.is_available()
    rows = source_rows()
    train_people = sorted(rows.loc[rows.role.eq("train"), "patient_alias"].unique())
    train, val = train_test_split(train_people, test_size=.2, random_state=SEED)
    assert set(train).isdisjoint(val)
    rows["seg_role"] = np.where(rows.role.eq("development"), "heldout_development",
                                np.where(rows.patient_alias.isin(val), "val", "train"))
    for row in rows.itertuples(index=False):
        if row.seg_role == "heldout_development":
            continue
        image, mask = split_panel(Path(row.source_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_path = OUT / "dataset/images" / row.seg_role / f"{row.image_id}.png"
        label_path = OUT / "dataset/labels" / row.seg_role / f"{row.image_id}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image_path), gray)
        label_path.write_text(polygon_text(mask), encoding="ascii")
    manifest = rows[["image_id", "patient_alias", "role", "seg_role"]]
    manifest.to_csv(OUT / "manifest_deidentified.csv", index=False)
    config = {"path": str(OUT / "dataset"), "train": "images/train", "val": "images/val",
              "names": {0: "kidney"}}
    (OUT / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    parameters = {"epochs": 30, "imgsz": 512, "batch": 12, "freeze": 10,
                  "optimizer": "AdamW", "lr0": .0001, "lrf": .1, "patience": 8,
                  "seed": SEED, "degrees": 7, "translate": .05, "scale": .1,
                  "fliplr": .5, "flipud": 0, "mosaic": 0, "mixup": 0,
                  "hsv_h": 0, "hsv_s": 0, "hsv_v": .1}
    report = {"source_router_sha256": digest(ROUTER / "image_routes.csv"),
              "source_manifest_sha256": digest(REGISTRY / "contour_image_manifest.csv"),
              "initial_weight_sha256": digest(INITIAL),
              "manifest_sha256": digest(OUT / "manifest_deidentified.csv"),
              "counts": manifest.seg_role.value_counts().to_dict(),
              "patients": manifest.groupby("seg_role").patient_alias.nunique().to_dict(),
              "training": parameters, "test_APN_scored": False,
              "data": "Only routed convex training-role paired-panel left frames and supplied right-pane masks; patient-disjoint internal train/val. No development or test patients used to fit or select checkpoint."}
    (OUT / "protocol.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def protocol():
    report = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    assert digest(ROUTER / "image_routes.csv") == report["source_router_sha256"]
    assert digest(REGISTRY / "contour_image_manifest.csv") == report["source_manifest_sha256"]
    assert digest(INITIAL) == report["initial_weight_sha256"]
    assert digest(OUT / "manifest_deidentified.csv") == report["manifest_sha256"]
    return report


def train():
    report = protocol()
    if (OUT / "finetune/weights/best.pt").exists():
        raise FileExistsError("Preserve convex checkpoint")
    torch.set_num_threads(4)
    YOLO(str(INITIAL)).train(data=str(OUT / "data.yaml"), project=str(OUT), name="finetune",
                             exist_ok=False, **report["training"], device=0, workers=0,
                             cache=True, amp=True, plots=False, deterministic=True,
                             warmup_epochs=1, weight_decay=.0001, close_mosaic=0,
                             save=True, verbose=False)
    best = OUT / "finetune/weights/best.pt"
    assert best.is_file()
    (OUT / "training_complete.json").write_text(json.dumps(
        {"checkpoint_sha256": digest(best), "device": torch.cuda.get_device_name(0)}, indent=2),
        encoding="utf-8")


def evaluate(baseline=False):
    protocol()
    weight = INITIAL if baseline else OUT / "finetune/weights/best.pt"
    name = "baseline_512" if baseline else "development"
    target = OUT / f"{name}_segmentation_summary.json"
    if not weight.is_file() or target.exists():
        raise FileExistsError("Checkpoint missing or evaluation already exists")
    model = YOLO(str(weight))
    rows = source_rows()
    rows = rows[rows.role.eq("development")]
    records = []
    for row in rows.itertuples(index=False):
        image, truth = split_panel(Path(row.source_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = model.predict(rgb, imgsz=512, conf=.25, iou=.7, device=0, verbose=False)[0]
        predicted = predict_mask(result, truth.shape)
        intersection = int(np.count_nonzero((truth > 0) & (predicted > 0)))
        union = int(np.count_nonzero((truth > 0) | (predicted > 0)))
        records.append({"patient_alias": row.patient_alias, "image_id": row.image_id,
                        "iou": intersection / union if union else 0,
                        "instances": 0 if result.masks is None else len(result.masks.xy)})
    frame = pd.DataFrame(records)
    assert len(frame) == 195 and frame.patient_alias.nunique() == 50
    summary = {"checkpoint_sha256": digest(weight), "input": "full grayscale frame, 512px, conf 0.25, NMS IoU 0.7",
               "development_images": len(frame), "development_patients": frame.patient_alias.nunique(),
               "image_mean_iou": float(frame.iou.mean()), "image_median_iou": float(frame.iou.median()),
               "patient_mean_iou": float(frame.groupby("patient_alias").iou.mean().mean()),
               "images_iou_ge_0_75": int(frame.iou.ge(.75).sum()),
               "images_no_prediction": int(frame.instances.eq(0).sum()),
               "test_APN_scored": False,
               "caveat": "Supplied masks are the reference, not independently adjudicated expert contours. This is development segmentation QC, not APN classification."}
    frame.to_csv(OUT / f"{name}_per_image_deidentified.csv", index=False)
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train", "evaluate", "baseline"))
    stage = parser.parse_args().stage
    {"prepare": prepare, "train": train, "evaluate": evaluate,
     "baseline": lambda: evaluate(True)}[stage]()
