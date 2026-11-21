"""Probe kidneyUS Task002 using kidney-centred full-frame context.

The earlier Task002 probe pasted a tight kidney crop on a black canvas. The
released nnU-Net was trained on complete ultrasound frames, so this label-blind
probe instead zooms the clean source frame around the established kidney mask.
Inflammation labels are never loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import torch

from kidneyus_internal_region_probe import (
    CANVAS_SHAPE,
    CLASS_COLORS,
    CLASS_NAMES,
    SOURCE_RUN,
    array_to_nifti,
    build_trainer,
    load_checkpoint,
    model_folder_for,
    selected_packet_image_ids,
    validate_model_family,
)
from kidneyus_segmentation_probe import read_manifest_without_diagnosis
from recrop_with_yolo import bbox_of, polygon_mask


ROOT = Path(__file__).resolve().parent
OUT_PREFIX = "CYCH_kidneyUS_context_regions"
CHECKPOINT_NAME = "model_final_checkpoint.model"
FOLDS = tuple(range(5))


def output_folder(variant: str) -> Path:
    return ROOT / f"{OUT_PREFIX}_{variant}_20260914"


def context_zoom_canvas(
    gray: np.ndarray,
    outer_mask: np.ndarray,
    scale: float,
    canvas_shape: tuple[int, int] = CANVAS_SHAPE,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Crop around the kidney, retaining context, then resize to the model canvas."""
    if gray.ndim != 2 or gray.shape != outer_mask.shape:
        raise ValueError("gray and outer_mask must be aligned 2D arrays")
    if scale <= 0 or not outer_mask.any():
        raise ValueError("scale must be positive and outer_mask nonempty")

    canvas_h, canvas_w = canvas_shape
    window_h = max(1, int(round(canvas_h / scale)))
    window_w = max(1, int(round(canvas_w / scale)))
    ys, xs = np.nonzero(outer_mask)
    center_y = float(np.mean(ys))
    center_x = float(np.mean(xs))
    y0 = int(round(center_y - window_h / 2))
    x0 = int(round(center_x - window_w / 2))
    y1, x1 = y0 + window_h, x0 + window_w

    context = np.zeros((window_h, window_w), dtype=gray.dtype)
    context_outer = np.zeros((window_h, window_w), dtype=np.uint8)
    src_y0, src_x0 = max(0, y0), max(0, x0)
    src_y1, src_x1 = min(gray.shape[0], y1), min(gray.shape[1], x1)
    dst_y0, dst_x0 = src_y0 - y0, src_x0 - x0
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    if src_y1 > src_y0 and src_x1 > src_x0:
        context[dst_y0:dst_y1, dst_x0:dst_x1] = gray[src_y0:src_y1, src_x0:src_x1]
        context_outer[dst_y0:dst_y1, dst_x0:dst_x1] = outer_mask[
            src_y0:src_y1, src_x0:src_x1
        ]

    canvas = cv2.resize(context, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
    canvas_outer = cv2.resize(
        context_outer, (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST
    )
    transform = {
        "original_height": int(gray.shape[0]),
        "original_width": int(gray.shape[1]),
        "canvas_height": canvas_h,
        "canvas_width": canvas_w,
        "scale": float(scale),
        "window_height": window_h,
        "window_width": window_w,
        "y0": y0,
        "x0": x0,
        "src_y0": src_y0,
        "src_x0": src_x0,
        "src_y1": src_y1,
        "src_x1": src_x1,
        "dst_y0": dst_y0,
        "dst_x0": dst_x0,
        "dst_y1": dst_y1,
        "dst_x1": dst_x1,
    }
    return canvas, canvas_outer, transform


def invert_context_labels(labels: np.ndarray, transform: dict) -> np.ndarray:
    window = cv2.resize(
        labels.astype(np.uint8),
        (int(transform["window_width"]), int(transform["window_height"])),
        interpolation=cv2.INTER_NEAREST,
    )
    restored = np.zeros(
        (int(transform["original_height"]), int(transform["original_width"])),
        dtype=np.uint8,
    )
    restored[
        int(transform["src_y0"]):int(transform["src_y1"]),
        int(transform["src_x0"]):int(transform["src_x1"]),
    ] = window[
        int(transform["dst_y0"]):int(transform["dst_y1"]),
        int(transform["dst_x0"]):int(transform["dst_x1"]),
    ]
    return restored


def full_outer_mask(row: dict, gray: np.ndarray) -> np.ndarray:
    if row["mask_source"] == "existing_yolo":
        mask = cv2.imread(
            str(SOURCE_RUN / "predicted_masks" / f"{row['image_id']}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None or mask.shape != gray.shape:
            raise RuntimeError(f"Invalid stored YOLO mask for {row['image_id']}")
        return (mask > 0).astype(np.uint8)
    mask = polygon_mask(
        ROOT / "cych_image_label" / Path(row["filename"]).with_suffix(".txt"),
        *gray.shape,
    )
    if not mask.any():
        raise RuntimeError(f"Empty supplied polygon for {row['image_id']}")
    return mask.astype(np.uint8)


def native_bounds(outer: np.ndarray) -> tuple[int, int, int, int]:
    box = bbox_of(outer)
    if box is None:
        raise RuntimeError("Empty outer mask")
    x0, y0, x1, y1 = box
    return (
        max(0, x0 - 4),
        max(0, y0 - 4),
        min(outer.shape[1], x1 + 4),
        min(outer.shape[0], y1 + 4),
    )


def write_overlay(
    gray: np.ndarray, outer: np.ndarray, labels: np.ndarray, path: Path, title: str
) -> None:
    x0, y0, x1, y1 = native_bounds(outer)
    panel = cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    local_outer = outer[y0:y1, x0:x1]
    local_labels = labels[y0:y1, x0:x1]
    for class_id, color in CLASS_COLORS.items():
        region = local_labels == class_id
        if region.any():
            tint = np.zeros_like(panel)
            tint[:] = color
            panel[region] = cv2.addWeighted(panel[region], 0.45, tint[region], 0.55, 0)
    contours = cv2.findContours(
        local_outer.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    cv2.drawContours(panel, contours, -1, (255, 255, 255), 1)
    factor = max(2, int(np.ceil(720 / max(1, panel.shape[1]))))
    panel = cv2.resize(panel, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 58), (0, 0, 0), -1)
    cv2.putText(panel, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
    cv2.putText(
        panel,
        "CEC=yellow  Medulla=blue  Cortex=green  Outer=white",
        (10, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (230, 230, 230),
        1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)


def run_probe(variant: str, scale: float, limit: int | None = None) -> list[dict]:
    from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax

    run_out = output_folder(variant)
    run_out.mkdir(parents=True, exist_ok=True)
    checkpoint_records = validate_model_family(variant)
    selected_ids = selected_packet_image_ids(limit)
    rows = {row["image_id"]: row for row in read_manifest_without_diagnosis()}
    trainer = build_trainer(variant, run_out)
    items = {}

    for image_id in selected_ids:
        row = rows[image_id]
        gray = cv2.imread(row["clean_path"], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(row["clean_path"])
        outer = full_outer_mask(row, gray)
        canvas, canvas_outer, transform = context_zoom_canvas(gray, outer, scale)
        nifti = run_out / "input_nifti" / f"{image_id}_0000.nii.gz"
        array_to_nifti(canvas, nifti)
        data, _, properties = trainer.preprocess_patient([str(nifti)])
        items[image_id] = {
            "row": row,
            "gray": gray,
            "outer": outer,
            "canvas_outer": canvas_outer,
            "transform": transform,
            "data": data,
            "properties": properties,
            "softmax_sum": None,
            "input_pixel_sha256": hashlib.sha256(gray.tobytes()).hexdigest(),
        }

    for checkpoint_record in checkpoint_records:
        checkpoint = load_checkpoint(Path(checkpoint_record["path"]))
        trainer.load_checkpoint_ram(checkpoint, train=False)
        trainer.network.eval()
        with torch.inference_mode():
            for item in items.values():
                softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
                    item["data"],
                    do_mirroring=False,
                    mirror_axes=(),
                    use_sliding_window=True,
                    step_size=0.5,
                    use_gaussian=True,
                    all_in_gpu=False,
                    verbose=False,
                    mixed_precision=True,
                )[1].astype(np.float32)
                item["softmax_sum"] = (
                    softmax
                    if item["softmax_sum"] is None
                    else item["softmax_sum"] + softmax
                )
        del checkpoint
        torch.cuda.empty_cache()

    metrics = []
    for image_id, item in items.items():
        softmax = item["softmax_sum"] / len(FOLDS)
        softmax = softmax.transpose([0] + [axis + 1 for axis in trainer.transpose_backward])
        output_nifti = run_out / "output_nifti" / f"{image_id}.nii.gz"
        output_nifti.parent.mkdir(parents=True, exist_ok=True)
        export = trainer.plans.get("segmentation_export_params", {})
        save_segmentation_nifti_from_softmax(
            softmax,
            str(output_nifti),
            item["properties"],
            export.get("interpolation_order", 1),
            trainer.regions_class_order,
            None,
            None,
            None,
            None,
            force_separate_z=export.get("force_separate_z"),
            interpolation_order_z=export.get("interpolation_order_z", 0),
        )
        canvas_labels = sitk.GetArrayFromImage(sitk.ReadImage(str(output_nifti)))
        if canvas_labels.shape != (1, *CANVAS_SHAPE):
            raise RuntimeError(f"Unexpected output shape for {image_id}: {canvas_labels.shape}")
        raw = invert_context_labels(canvas_labels[0], item["transform"])
        outer = item["outer"].astype(bool)
        raw_union = raw > 0
        raw_pixels = int(raw_union.sum())
        outside_pixels = int(np.logical_and(raw_union, ~outer).sum())
        restricted = raw.copy()
        restricted[~outer] = 0
        x0, y0, x1, y1 = native_bounds(item["outer"])
        native_labels = restricted[y0:y1, x0:x1]
        native_shape = cv2.imread(
            str(SOURCE_RUN / "native" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE
        ).shape
        if native_labels.shape != native_shape:
            raise RuntimeError(
                f"Native crop mismatch for {image_id}: {native_labels.shape} != {native_shape}"
            )
        mask_dir = run_out / "region_masks"
        mask_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(mask_dir / f"{image_id}_labels.png"), native_labels)
        for class_id, class_name in CLASS_NAMES.items():
            cv2.imwrite(
                str(mask_dir / f"{image_id}_{class_name}.png"),
                (native_labels == class_id).astype(np.uint8) * 255,
            )
        write_overlay(
            item["gray"],
            item["outer"],
            restricted,
            run_out / "overlays" / f"{image_id}.png",
            f"{image_id} | {variant} | context zoom {scale:.2f}",
        )
        outer_pixels = int(outer.sum())
        metric = {
            "image_id": image_id,
            "patient_alias": item["row"]["patient_alias"],
            "view": item["row"]["view"],
            "variant": variant,
            "scale": float(scale),
            "outer_pixels": outer_pixels,
            "raw_region_pixels": raw_pixels,
            "raw_outside_outer_fraction": outside_pixels / max(raw_pixels, 1),
            "region_pixels_in_outer": int((restricted > 0).sum()),
            "region_coverage_of_outer": int((restricted > 0).sum()) / outer_pixels,
            "present_class_count": int(sum((restricted == c).any() for c in CLASS_NAMES)),
            "input_pixel_sha256": item["input_pixel_sha256"],
        }
        for class_id, class_name in CLASS_NAMES.items():
            metric[f"{class_name}_pixels"] = int((restricted == class_id).sum())
        metrics.append(metric)

    with (run_out / "region_plausibility_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    protocol = {
        "scope": "Outcome-blind kidneyUS Task002 technical transfer probe.",
        "diagnostic_labels_read": False,
        "input": "Clean full ultrasound frame, kidney-centred context window, isotropic zoom, resize to 768x1024.",
        "outer_roi": "Existing accepted YOLO mask or supplied polygon fallback, matching the fixed CYCH image manifest.",
        "variant": variant,
        "scale": float(scale),
        "images": len(metrics),
        "image_ids": selected_ids,
        "folds": list(FOLDS),
        "ensemble": "Arithmetic mean of five released fold softmax outputs; no TTA.",
        "checkpoint_records": checkpoint_records,
    }
    (run_out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    summary = {
        "variant": variant,
        "images": len(metrics),
        "nonempty": int(sum(row["region_pixels_in_outer"] > 0 for row in metrics)),
        "three_class": int(sum(row["present_class_count"] == 3 for row in metrics)),
        "mean_coverage": float(np.mean([row["region_coverage_of_outer"] for row in metrics])),
    }
    (run_out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return metrics


def self_test() -> None:
    gray = np.arange(240 * 320, dtype=np.uint8).reshape(240, 320)
    outer = np.zeros_like(gray, dtype=np.uint8)
    cv2.ellipse(outer, (160, 120), (60, 35), 0, 0, 360, 1, -1)
    _, canvas_outer, transform = context_zoom_canvas(gray, outer, 2.0)
    restored = invert_context_labels(canvas_outer, transform) > 0
    reference = outer > 0
    intersection = np.logical_and(restored, reference).sum()
    dice = 2 * intersection / (restored.sum() + reference.sum())
    if dice < 0.98:
        raise AssertionError(f"Context transform round-trip Dice too low: {dice}")
    print(json.dumps({"self_test": "passed", "round_trip_dice": float(dice)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("annotator_1", "annotator_2"))
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.variant:
        raise SystemExit("--variant is required unless --self-test is used")
    run_probe(args.variant, args.scale, args.limit)


if __name__ == "__main__":
    main()
