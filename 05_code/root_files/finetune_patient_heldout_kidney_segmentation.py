"""Fine-tune the existing segmenter on clean paired frames, with patient-held-out QA."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/CYCH_NCKUH_segmentation_finetune_20261004"
PRETRAINED = Path("C:/Users/udoo_w2/Desktop/work-kidney-tumor/kidney-tumor/best.pt")
SEED = 20261004


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_polygon(path):
    lines = Path(path).read_text().strip().splitlines()
    if len(lines) != 1:
        raise ValueError("Expected one supplied kidney polygon")
    v = np.array(lines[0].split(), dtype=float)
    if v[0] != 0 or len(v) < 7 or len(v) % 2 != 1:
        raise ValueError("Invalid YOLO polygon")
    pts = v[1:].reshape(-1, 2)
    if not np.isfinite(pts).all() or pts.min() < 0 or pts.max() > 1:
        raise ValueError("Invalid normalized coordinates")
    return pts


def rasterize(points, shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [(points * [w, h]).astype(np.int32)], 1)
    return mask


def prediction_mask(polygons, shape):
    mask = np.zeros(shape[:2], np.uint8)
    if len(polygons):
        j = int(np.argmax([cv2.contourArea(p.astype(np.float32)) for p in polygons]))
        cv2.fillPoly(mask, [polygons[j].astype(np.int32)], 1)
    return mask


def overlap_metrics(mask, gt):
    inter = int((mask & gt).sum())
    union = int((mask | gt).sum())
    total = int(mask.sum() + gt.sum())
    return inter / max(1, union), 2 * inter / max(1, total)


def prepare():
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Split already locked; use train/evaluate, not prepare")
    OUT.mkdir(parents=True, exist_ok=True)
    data = ROOT / "03_data_models"
    src = pd.read_csv(data / "yolo_crops/manifest.csv", dtype={"patient_id": str})
    cy = pd.read_csv(ROOT / "02_experiments/CYCH_random_validation_20260913/image_manifest.csv", dtype={"patient_id": str})
    assignments = pd.read_csv(ROOT / "02_experiments/CYCH_random_validation_20260913/random_patient_assignments.csv", dtype={"patient_id": str})
    locked_cy = assignments[assignments.seed.eq(20260913)].set_index("patient_id").role.to_dict()
    cy_names = set(cy.filename)
    src = src[src.site.eq("NCKUH") | src.filename.isin(cy_names)].copy()
    assert len(src[src.site.eq("CYCH")]) == 349
    site_ids = {s: set(g.patient_id) for s, g in src.groupby("site")}
    assert not (site_ids["CYCH"] & site_ids["NCKUH"]), "Cross-site patient IDs require manual adjudication"
    patients = src.groupby(["site", "patient_id"]).agg(label=("label", "first")).reset_index()
    roles = {( "CYCH", p): ("val" if role == "validation" else role) for p, role in locked_cy.items()}
    nck = patients[patients.site.eq("NCKUH")].reset_index(drop=True)
    pool, test = next(StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(nck, nck.label))
    a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=.25, random_state=SEED + 1).split(nck.iloc[pool], nck.label.iloc[pool]))
    for role, idxs in (("train", pool[a]), ("val", pool[b]), ("test", test)):
        for i in idxs:
            roles[("NCKUH", nck.iloc[i].patient_id)] = role
    alias = {(r.site, r.patient_id): f"{r.site}_P{i+1:04d}" for i, r in patients.iterrows()}
    rows = []
    seen = {}
    for i, row in src.sort_values(["site", "patient_id", "filename"]).reset_index(drop=True).iterrows():
        site = row.site
        clean_dir = data / ("original_png" if site == "CYCH" else "original_png_nckuh")
        label_dir = data / ("cych_image_label" if site == "CYCH" else "nckuh_image_label")
        clean = clean_dir / row.filename.replace("_red_", "_", 1)
        label = label_dir / Path(row.filename).with_suffix(".txt")
        frame = cv2.imread(str(clean))
        if frame is None:
            raise FileNotFoundError(clean)
        polygon = read_polygon(label)
        manual = rasterize(polygon, frame.shape)
        if manual.sum() < 64:
            raise ValueError("Invalid supplied contour area")
        # Use the clean frame, never the red-annotated twin or a ground-truth crop.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_hash = hashlib.sha256(str(gray.shape).encode() + gray.tobytes()).hexdigest()
        key = (site, row.patient_id)
        if gray_hash in seen and seen[gray_hash] != key:
            raise ValueError("Duplicate decoded image across patient identities")
        seen[gray_hash] = key
        role = roles[key]
        iid = f"{site}_I{i+1:05d}"
        image_out = OUT / "dataset/images" / role / f"{iid}.png"
        label_out = OUT / "dataset/labels" / role / f"{iid}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image_out), gray)
        label_out.write_bytes(label.read_bytes())
        rows.append(dict(site=site, patient_alias=alias[key], image_id=iid, role=role,
                         image_path=str(image_out), label_path=str(label_out),
                         width=gray.shape[1], height=gray.shape[0], mask_pixels=int(manual.sum()),
                         source_sha256=digest(clean), label_sha256=digest(label),
                         decoded_grayscale_sha256=gray_hash))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT / "manifest_deidentified.csv", index=False)
    assert manifest.groupby("patient_alias").role.nunique().max() == 1
    assert manifest.groupby("decoded_grayscale_sha256").role.nunique().max() == 1
    config = dict(path=str(OUT / "dataset"), train="images/train", val="images/val", test="images/test", names={0: "kidney"})
    (OUT / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    counts = [dict(site=s, role=role, patients=int(g.patient_alias.nunique()), images=len(g)) for (s, role), g in manifest.groupby(["site", "role"])]
    protocol = dict(seed=SEED, status="locked_before_training_and_test_prediction", counts=counts,
        source_checkpoint=str(PRETRAINED), source_checkpoint_sha256=digest(PRETRAINED),
        manifest_sha256=digest(OUT / "manifest_deidentified.csv"),
        original_pretraining_patient_overlap="Unresolved: original 2025 training manifest unavailable. Test is held out from this fine-tuning, not certified unseen by original pretraining.",
        labels="Existing single kidney contour, derived from supplied red-outline twin. Expert authorship and diagnostic blinding not independently verified.",
        image_input="Paired clean full available frame, grayscale, no red outline, no supplied-mask cropping or masking.",
        split="CYCH uses seed-20260913 classification train/dev/test patients (58/20/20). NCKUH patient-stratified 60/20/20. This model must not replace masks across the previous ten classification draws without nested split-specific retraining.",
        training=dict(epochs=20, imgsz=512, batch=12, freeze=10, optimizer="AdamW", lr0=.0001, lrf=.1, patience=6, seed=SEED,
                      degrees=7, translate=.05, scale=.1, fliplr=.5, flipud=0, mosaic=0, mixup=0, hsv_h=0, hsv_s=0, hsv_v=.1),
        selection="Ultralytics validation fitness from mask/box mAP only; no test metrics used to select checkpoint, epochs, threshold or input resolution.",
        evaluation=dict(imgsz=640, conf=.25, nms_iou=.7, selection="Largest predicted polygon by area, no ground-truth selection", fallback=False,
                        primary="Mean per-patient mean raw prediction IoU over all held-out test patients. Empty prediction = IoU 0.",
                        target="Primary mean IoU > 0.75; report CI, each site, all-image mean, median, threshold pass fraction, missing predictions, Dice and paired change.",
                        bootstrap="10000 within-site patient-stratified paired resamples of patient mean IoU"),
        references=["https://docs.ultralytics.com/tasks/segment/", "https://docs.ultralytics.com/modes/train/", "https://arxiv.org/abs/2502.12524"])
    dump(OUT / "protocol.json", protocol)
    dump(OUT / "protocol_lock.json", dict(protocol_sha256=digest(OUT / "protocol.json"), manifest_sha256=digest(OUT / "manifest_deidentified.csv")))
    print(json.dumps(counts, indent=2), flush=True)


def verify():
    lock = json.loads((OUT / "protocol_lock.json").read_text())
    for name in ("protocol", "manifest"):
        filename = "protocol.json" if name == "protocol" else "manifest_deidentified.csv"
        assert digest(OUT / filename) == lock[f"{name}_sha256"]
    assert digest(PRETRAINED) == json.loads((OUT / "protocol.json").read_text())["source_checkpoint_sha256"]


def train():
    import torch
    from ultralytics import YOLO
    verify()
    torch.set_num_threads(4)
    model = YOLO(str(PRETRAINED))
    protocol = json.loads((OUT / "protocol.json").read_text())
    params = protocol["training"]
    print(f"Fine-tuning existing weights on {torch.cuda.get_device_name(0)}", flush=True)
    model.train(data=str(OUT / "data.yaml"), project=str(OUT), name="finetune", exist_ok=False,
                **params, device=0, workers=0, cache=True, amp=True, plots=False,
                deterministic=True, warmup_epochs=1, weight_decay=.0001, close_mosaic=0,
                save=True, verbose=False)
    checkpoint = OUT / "finetune/weights/best.pt"
    dump(OUT / "training_complete.json", dict(best_checkpoint=str(checkpoint), sha256=digest(checkpoint),
                                              torch_version=torch.__version__, device=torch.cuda.get_device_name(0)))


def summarize(frame):
    patients = frame.groupby(["site", "patient_alias"]).agg(iou=("iou", "mean"), dice=("dice", "mean"))
    return dict(images=len(frame), patients=len(patients), image_mean_iou=float(frame.iou.mean()),
                image_median_iou=float(frame.iou.median()), patient_mean_iou=float(patients.iou.mean()),
                patient_mean_dice=float(patients.dice.mean()), missing_predictions=int(frame.instances.eq(0).sum()),
                fraction_images_iou_above_075=float(frame.iou.gt(.75).mean()))


def evaluate():
    from ultralytics import YOLO
    verify()
    if (OUT / "test_results.json").exists():
        raise FileExistsError("Test results already locked; do not evaluate for model selection")
    manifest = pd.read_csv(OUT / "manifest_deidentified.csv")
    test = manifest[manifest.role.eq("test")].copy()
    checkpoints = {"original_pretrained": PRETRAINED, "fine_tuned": OUT / "finetune/weights/best.pt"}
    rows = []
    for version, checkpoint in checkpoints.items():
        model = YOLO(str(checkpoint))
        target = OUT / "test_predictions" / version
        target.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(test.itertuples()):
            img = cv2.imread(row.image_path)
            gt = rasterize(read_polygon(row.label_path), img.shape)
            pred = model.predict(img, imgsz=640, conf=.25, iou=.7, device=0, verbose=False)[0]
            n = 0 if pred.masks is None else len(pred.masks.xy)
            mask = prediction_mask([] if n == 0 else pred.masks.xy, gt.shape)
            iou, dice = overlap_metrics(mask, gt)
            assert cv2.imwrite(str(target / f"{row.image_id}.png"), mask * 255)
            rows.append(dict(model=version, site=row.site, patient_alias=row.patient_alias, image_id=row.image_id,
                             iou=iou, dice=dice, instances=n, gt_pixels=int(gt.sum()), predicted_pixels=int(mask.sum())))
            if i % 40 == 0:
                print(f"Test {version}: {i+1}/{len(test)}", flush=True)
        del model
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "test_per_image_deidentified.csv", index=False)
    p = result.groupby(["model", "site", "patient_alias"]).iou.mean().unstack("model")
    rng = np.random.default_rng(SEED)
    groups = [g for _, g in p.groupby(level="site")]
    estimates = []
    for _ in range(10000):
        sampled = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups])
        base = sampled.original_pretrained.mean()
        ft = sampled.fine_tuned.mean()
        estimates.append([base, ft, ft-base])
    intervals = np.quantile(estimates, [.025, .975], axis=0)
    report = dict(protocol_sha256=digest(OUT / "protocol.json"), checkpoint_hashes={k: digest(v) for k, v in checkpoints.items()},
                  overall={k: summarize(g) for k, g in result.groupby("model")},
                  by_site={s: {k: summarize(g) for k, g in f.groupby("model")} for s, f in result.groupby("site")},
                  patient_mean_iou_ci95={"original_pretrained": intervals[:, 0].tolist(), "fine_tuned": intervals[:, 1].tolist(),
                                         "paired_change": intervals[:, 2].tolist()},
                  paired_patient_mean_iou_change=float((p.fine_tuned - p.original_pretrained).mean()),
                  target_passed=bool(p.fine_tuned.mean() > .75),
                  qualification="Patient-held-out from new fine-tuning only. Original pretraining overlap and independent annotation provenance remain unresolved. No manual fallback or IoU-based acceptance filtering.")
    dump(OUT / "test_results.json", report)
    result.groupby(["model", "site", "patient_alias"]).agg(mean_iou=("iou", "mean"), mean_dice=("dice", "mean"), images=("iou", "size")).to_csv(OUT / "test_per_patient_deidentified.csv")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "evaluate"])
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "evaluate": evaluate}[args.stage]()
