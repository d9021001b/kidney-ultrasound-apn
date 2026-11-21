"""Probe kidneyUS internal-region checkpoints on CYCH kidney crops.

The established CYCH mask remains the outer-kidney boundary. The downloaded
kidneyUS Task002 model is used only as a candidate partition into Central Echo
Complex (CEC), medulla, and cortex. Diagnostic labels are never read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import torch

from kidneyus_segmentation_probe import (
    ALLOWED_TORCH_GLOBALS,
    read_manifest_without_diagnosis,
    restricted_pickle_load,
    safe_torch_load,
    sha256,
    torch_pickle_globals,
)


ROOT = Path(__file__).resolve().parent
SOURCE_RUN = ROOT / "CYCH_random_validation_20260913"
WEIGHT_ROOT = ROOT / "external_models" / "kidneyUS_official_20260914" / "weights"
ANNOTATION_PACKET = ROOT / "CYCH_internal_region_annotation_packet_20260914"
OUT = ROOT / "CYCH_kidneyUS_internal_regions_20260914"
CHECKPOINT_NAME = "model_final_checkpoint.model"
FOLDS = tuple(range(5))
CANVAS_SHAPE = (768, 1024)
CLASS_NAMES = {1: "cec", 2: "medulla", 3: "cortex"}
CLASS_COLORS = {
    1: (0, 215, 255),
    2: (255, 90, 40),
    3: (70, 210, 80),
}


def model_folder_for(variant: str) -> Path:
    if variant not in {"annotator_1", "annotator_2", "mixed"}:
        raise ValueError(f"Unsupported kidneyUS variant: {variant}")
    return (
        WEIGHT_ROOT
        / variant
        / "Task002_KidneyRegions"
        / "nnUNetTrainerV2__nnUNetPlansv2.1"
    )


def output_folder_for(variant: str) -> Path:
    return OUT if variant == "annotator_1" else ROOT / f"CYCH_kidneyUS_internal_regions_{variant}_20260914"


def validate_model_family(variant: str = "annotator_1") -> list[dict]:
    model_folder = model_folder_for(variant)
    records = []
    for fold in FOLDS:
        checkpoint = model_folder / f"fold_{fold}" / CHECKPOINT_NAME
        sidecar = Path(str(checkpoint) + ".pkl")
        if not checkpoint.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"Incomplete Task002 fold {fold}: {checkpoint}")
        denied = sorted(torch_pickle_globals(checkpoint) - ALLOWED_TORCH_GLOBALS)
        if denied:
            raise RuntimeError(f"Unexpected pickle globals in {checkpoint}: {denied}")
        metadata = restricted_pickle_load(sidecar)
        plans = metadata["plans"]
        if metadata.get("name") != "nnUNetTrainerV2":
            raise RuntimeError(f"Unexpected trainer in {sidecar}")
        if int(plans["num_classes"]) != 3 or list(plans["all_classes"]) != [1, 2, 3]:
            raise RuntimeError(f"Unexpected Task002 classes in {sidecar}")
        records.append(
            {
                "fold": fold,
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
            }
        )
    return records


def build_trainer(variant: str, run_out: Path):
    model_folder = model_folder_for(variant)
    path_root = run_out / "nnunet_paths"
    raw_base = path_root / "raw"
    preprocessed_base = path_root / "preprocessed"
    results_base = path_root / "results"
    dataset_placeholder = preprocessed_base / "Task002_KidneyRegions"
    for path in (raw_base, preprocessed_base, results_base, dataset_placeholder):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["nnUNet_raw_data_base"] = str(raw_base)
    os.environ["nnUNet_preprocessed"] = str(preprocessed_base)
    os.environ["RESULTS_FOLDER"] = str(results_base)

    from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2

    sidecar = model_folder / "fold_0" / f"{CHECKPOINT_NAME}.pkl"
    info = restricted_pickle_load(sidecar)
    init_args = list(info["init"])
    init_args[0] = str(model_folder / "plans.pkl")
    init_args[1] = 0
    init_args[2] = str(run_out / "nnunet_runtime")
    init_args[3] = str(dataset_placeholder)
    init_args[5] = 0
    init_args[6] = False
    init_args[7] = True
    init_args[8] = False
    trainer = nnUNetTrainerV2(*init_args)
    trainer.process_plans(info["plans"])
    trainer.initialize(training=False)
    trainer.network.eval()
    if int(trainer.num_classes) != 4:  # background plus three regions
        raise RuntimeError(f"Unexpected network output count: {trainer.num_classes}")
    return trainer


def load_checkpoint(path: Path) -> dict:
    payload = safe_torch_load(path)
    state = payload.get("state_dict")
    if not isinstance(state, dict) or len(state) != 134:
        raise RuntimeError(f"Unexpected state dict in {path}")
    return {
        "state_dict": state,
        "epoch": int(payload.get("epoch", 0)),
        "plot_stuff": payload.get("plot_stuff", ([], [], [], [])),
        "best_stuff": payload.get("best_stuff", (None, None, None)),
    }


def embed_on_training_canvas(
    gray: np.ndarray,
    outer_mask: np.ndarray,
    requested_scale: float,
    canvas_shape: tuple[int, int] = CANVAS_SHAPE,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Scale the kidney crop and center it on a source-sized black canvas."""
    if gray.ndim != 2 or outer_mask.shape != gray.shape:
        raise ValueError("gray and outer_mask must be aligned 2D arrays")
    if requested_scale <= 0:
        raise ValueError("requested_scale must be positive")
    height, width = gray.shape
    canvas_h, canvas_w = canvas_shape
    fit_scale = min(canvas_h / height, canvas_w / width)
    applied_scale = min(float(requested_scale), float(fit_scale))
    scaled_h = max(1, int(round(height * applied_scale)))
    scaled_w = max(1, int(round(width * applied_scale)))
    scaled_gray = cv2.resize(gray, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    scaled_mask = cv2.resize(
        outer_mask.astype(np.uint8),
        (scaled_w, scaled_h),
        interpolation=cv2.INTER_NEAREST,
    )
    y0 = (canvas_h - scaled_h) // 2
    x0 = (canvas_w - scaled_w) // 2
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    canvas_mask = np.zeros(canvas_shape, dtype=np.uint8)
    canvas[y0 : y0 + scaled_h, x0 : x0 + scaled_w] = scaled_gray
    canvas_mask[y0 : y0 + scaled_h, x0 : x0 + scaled_w] = scaled_mask
    transform = {
        "original_height": height,
        "original_width": width,
        "canvas_height": canvas_h,
        "canvas_width": canvas_w,
        "requested_scale": float(requested_scale),
        "applied_scale": applied_scale,
        "scaled_height": scaled_h,
        "scaled_width": scaled_w,
        "y0": y0,
        "x0": x0,
    }
    return canvas, canvas_mask, transform


def invert_canvas_labels(labels: np.ndarray, transform: dict) -> np.ndarray:
    y0, x0 = int(transform["y0"]), int(transform["x0"])
    scaled_h, scaled_w = int(transform["scaled_height"]), int(transform["scaled_width"])
    crop = labels[y0 : y0 + scaled_h, x0 : x0 + scaled_w]
    return cv2.resize(
        crop.astype(np.uint8),
        (int(transform["original_width"]), int(transform["original_height"])),
        interpolation=cv2.INTER_NEAREST,
    )


def array_to_nifti(array: np.ndarray, path: Path) -> None:
    image = sitk.GetImageFromArray(array[None].astype(np.float32))
    image.SetSpacing((1.0, 1.0, 999.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), True)


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    denominator = int(a.sum()) + int(b.sum())
    return 2.0 * int(np.logical_and(a, b).sum()) / denominator if denominator else 1.0


def connected_components(mask: np.ndarray) -> int:
    count = cv2.connectedComponents(mask.astype(np.uint8), 8)[0]
    return max(0, int(count) - 1)


def write_overlay(gray: np.ndarray, outer: np.ndarray, labels: np.ndarray, path: Path, title: str) -> None:
    panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for class_id, color in CLASS_COLORS.items():
        region = labels == class_id
        if region.any():
            tint = np.zeros_like(panel)
            tint[:] = color
            panel[region] = cv2.addWeighted(panel[region], 0.45, tint[region], 0.55, 0)
    contours = cv2.findContours(outer.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(panel, contours, -1, (255, 255, 255), 1)
    scale = max(2, int(np.ceil(720 / max(1, panel.shape[1]))))
    panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(panel, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    legend = "CEC=yellow  Medulla=blue  Cortex=green  Outer=white"
    cv2.putText(panel, legend, (10, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)


def selected_packet_image_ids(limit: int | None = None) -> list[str]:
    path = ANNOTATION_PACKET / "private_task_key.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    image_ids = []
    seen = set()
    for row in rows:
        image_id = row["image_id"]
        if row["task_type"] != "core" or image_id in seen:
            continue
        seen.add(image_id)
        image_ids.append(image_id)
        if limit is not None and len(image_ids) >= limit:
            break
    return image_ids


def run_probe(
    image_ids: list[str], scales: list[float], tta: bool, variant: str = "annotator_1"
) -> list[dict]:
    from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax

    model_folder = model_folder_for(variant)
    run_out = output_folder_for(variant)
    checkpoint_records = validate_model_family(variant)
    manifest = {row["image_id"]: row for row in read_manifest_without_diagnosis()}
    missing = sorted(set(image_ids) - set(manifest))
    if missing:
        raise KeyError(f"Unknown image ids: {missing}")

    trainer = build_trainer(variant, run_out)
    items: dict[tuple[str, float], dict] = {}
    for image_id in image_ids:
        row = manifest[image_id]
        # Use the immutable, dimension-locked image paired with the final mask.
        # Some upstream clean_path files retain one or two extra border pixels.
        gray_path = SOURCE_RUN / "native" / f"{image_id}.png"
        gray = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
        outer_path = SOURCE_RUN / "masks" / f"{image_id}.png"
        outer = cv2.imread(str(outer_path), cv2.IMREAD_GRAYSCALE)
        if gray is None or outer is None or gray.shape != outer.shape:
            raise RuntimeError(f"Invalid aligned input for {image_id}")
        outer = (outer > 0).astype(np.uint8)
        if not outer.any():
            raise RuntimeError(f"Empty established outer ROI for {image_id}")
        for requested_scale in scales:
            canvas, canvas_outer, transform = embed_on_training_canvas(
                gray, outer, requested_scale
            )
            scale_key = f"s{int(round(requested_scale * 100)):03d}"
            nifti = run_out / "input_nifti" / scale_key / f"{image_id}_0000.nii.gz"
            array_to_nifti(canvas, nifti)
            data, _, properties = trainer.preprocess_patient([str(nifti)])
            items[(image_id, requested_scale)] = {
                "row": row,
                "gray": gray,
                "outer": outer,
                "canvas_outer": canvas_outer,
                "transform": transform,
                "data": data,
                "properties": properties,
                "softmax_sum": None,
                "scale_key": scale_key,
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
                    do_mirroring=tta,
                    mirror_axes=trainer.data_aug_params["mirror_axes"] if tta else (),
                    use_sliding_window=True,
                    step_size=0.5,
                    use_gaussian=True,
                    all_in_gpu=False,
                    verbose=False,
                    mixed_precision=True,
                )[1].astype(np.float32)
                item["softmax_sum"] = (
                    softmax if item["softmax_sum"] is None else item["softmax_sum"] + softmax
                )
        del checkpoint
        torch.cuda.empty_cache()

    metric_rows = []
    restricted_by_key: dict[tuple[str, float], np.ndarray] = {}
    for (image_id, requested_scale), item in items.items():
        softmax = item["softmax_sum"] / len(FOLDS)
        softmax = softmax.transpose([0] + [axis + 1 for axis in trainer.transpose_backward])
        output_nifti = run_out / "output_nifti" / item["scale_key"] / f"{image_id}.nii.gz"
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
        raw = invert_canvas_labels(canvas_labels[0], item["transform"])
        outer = item["outer"].astype(bool)
        raw_union = raw > 0
        raw_pixels = int(raw_union.sum())
        outside_pixels = int(np.logical_and(raw_union, ~outer).sum())
        restricted = raw.copy()
        restricted[~outer] = 0
        restricted_by_key[(image_id, requested_scale)] = restricted

        mask_dir = run_out / "region_masks" / item["scale_key"]
        mask_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_dir / f"{image_id}_labels.png"), restricted)
        for class_id, class_name in CLASS_NAMES.items():
            cv2.imwrite(
                str(mask_dir / f"{image_id}_{class_name}.png"),
                (restricted == class_id).astype(np.uint8) * 255,
            )
        write_overlay(
            item["gray"],
            item["outer"],
            restricted,
            run_out / "overlays" / item["scale_key"] / f"{image_id}.png",
            f"{image_id} kidneyUS Task002, scale={item['transform']['applied_scale']:.3f}",
        )

        outer_pixels = int(outer.sum())
        class_pixels = {name: int((restricted == class_id).sum()) for class_id, name in CLASS_NAMES.items()}
        metric_rows.append(
            {
                "image_id": image_id,
                "view": item["row"]["view"],
                "outer_mask_source": item["row"]["mask_source"],
                "requested_scale": requested_scale,
                "applied_scale": item["transform"]["applied_scale"],
                "outer_pixels": outer_pixels,
                "raw_region_pixels": raw_pixels,
                "raw_outside_outer_pixels": outside_pixels,
                "raw_outside_fraction": outside_pixels / raw_pixels if raw_pixels else 0.0,
                "restricted_region_pixels": int((restricted > 0).sum()),
                "region_coverage_of_outer": int((restricted > 0).sum()) / outer_pixels,
                "cec_pixels": class_pixels["cec"],
                "cec_fraction_of_outer": class_pixels["cec"] / outer_pixels,
                "cec_components": connected_components(restricted == 1),
                "medulla_pixels": class_pixels["medulla"],
                "medulla_fraction_of_outer": class_pixels["medulla"] / outer_pixels,
                "medulla_components": connected_components(restricted == 2),
                "cortex_pixels": class_pixels["cortex"],
                "cortex_fraction_of_outer": class_pixels["cortex"] / outer_pixels,
                "cortex_components": connected_components(restricted == 3),
                "present_class_count": sum(value > 0 for value in class_pixels.values()),
                "input_pixel_sha256": item["input_pixel_sha256"],
            }
        )

    run_out.mkdir(parents=True, exist_ok=True)
    with (run_out / "region_plausibility_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)

    reference_scale = min(scales, key=lambda value: abs(value - 2.0))
    stability_rows = []
    for image_id in image_ids:
        reference = restricted_by_key[(image_id, reference_scale)]
        for scale in scales:
            candidate = restricted_by_key[(image_id, scale)]
            for class_id, class_name in {0: "any_region", **CLASS_NAMES}.items():
                if class_id == 0:
                    reference_mask = reference > 0
                    candidate_mask = candidate > 0
                else:
                    reference_mask = reference == class_id
                    candidate_mask = candidate == class_id
                stability_rows.append(
                    {
                        "image_id": image_id,
                        "reference_scale": reference_scale,
                        "candidate_scale": scale,
                        "class": class_name,
                        "dice_to_reference_scale": dice(reference_mask, candidate_mask),
                    }
                )
    with (run_out / "scale_stability.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=stability_rows[0].keys())
        writer.writeheader()
        writer.writerows(stability_rows)

    protocol = {
        "scope": "Segmentation feasibility only; inflammation labels were not read.",
        "outer_boundary": "Established CYCH final mask (YOLO or prespecified manual fallback); never replaced.",
        "model_role": "Candidate internal anatomical partition only; not disease localization.",
        "model_folder": str(model_folder),
        "variant": variant,
        "official_weight_source": "https://github.com/rsingla92/kidneyUS",
        "class_mapping": {"1": "Central Echo Complex", "2": "Medulla", "3": "Cortex"},
        "class_mapping_evidence": str(ROOT / "third_party" / "kidneyUS" / "src" / "annotation_analysis" / "create_masks.py"),
        "checkpoint_loading": "Static pickle-global audit plus torch.load(weights_only=True)",
        "checkpoints": checkpoint_records,
        "fold_ensemble": "Arithmetic mean of five-fold softmax outputs",
        "tta": tta,
        "canvas_shape": list(CANVAS_SHAPE),
        "scale_basis": {
            "official_median_foreground_bbox_hw": [282, 444],
            "cych_median_outer_bbox_hw": [142, 221],
            "median_height_ratio": 1.9859154929577465,
            "median_width_ratio": 2.009049773755656,
            "interpretation": "CYCH files are kidney-centered crops; scaling by approximately 2 and padding restores source-like kidney size and frame occupancy.",
        },
        "requested_scales": scales,
        "reference_scale_for_stability": reference_scale,
        "postprocessing": "Map to original resolution, then restrict all classes to the fixed CYCH outer ROI.",
        "clinical_gate": "No anatomical validity claim until blinded clinician annotations are returned and evaluated.",
        "image_ids": image_ids,
    }
    (run_out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return metric_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "smoke", "systematic", "packet"))
    parser.add_argument("--image-id", action="append", default=[])
    parser.add_argument("--scales", type=float, nargs="+", default=[1.6, 2.0, 2.4])
    parser.add_argument(
        "--variant",
        choices=("annotator_1", "annotator_2", "mixed"),
        default="annotator_1",
    )
    parser.add_argument("--tta", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = validate_model_family(args.variant)
    if args.mode == "audit":
        print(
            json.dumps(
                {"variant": args.variant, "folds": len(records), "classes": CLASS_NAMES},
                indent=2,
            )
        )
        return
    if args.image_id:
        image_ids = args.image_id
    elif args.mode == "smoke":
        image_ids = selected_packet_image_ids(limit=1)
    elif args.mode == "systematic":
        image_ids = selected_packet_image_ids(limit=8)
    else:
        image_ids = selected_packet_image_ids()
    metrics = run_probe(
        image_ids,
        sorted(set(args.scales)),
        tta=args.tta,
        variant=args.variant,
    )
    print(
        json.dumps(
            {
                "images": len(image_ids),
                "variant": args.variant,
                "scales": sorted(set(args.scales)),
                "rows": len(metrics),
                "nonempty_rows": sum(row["restricted_region_pixels"] > 0 for row in metrics),
                "mean_region_coverage_of_outer": float(
                    np.mean([row["region_coverage_of_outer"] for row in metrics])
                ),
                "mean_raw_outside_fraction": float(
                    np.mean([row["raw_outside_fraction"] for row in metrics])
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
