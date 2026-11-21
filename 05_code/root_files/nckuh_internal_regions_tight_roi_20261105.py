"""Outcome-blind tight-ROI compatibility test for kidneyUS compartments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

import kidneyus_internal_region_probe as K
import nckuh_internal_regions_20261102 as prior


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_internal_regions_tight_roi_20261105"
SOURCE_SELECTION = ROOT / "02_experiments/NCKUH_internal_regions_20261102/selected_images_private.csv"
K.WEIGHT_ROOT = ROOT / "03_data_models/external_models/kidneyUS_official_20260914/weights"
VARIANTS = ("annotator_1", "annotator_2")
REQUESTED_SCALE = 2.0
MODE = "tight_roi_scale2"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_selection() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE_SELECTION, dtype=str)
    required = {"image_id", "patient_alias", "filename", "view"}
    if not required.issubset(frame.columns) or len(frame) != 32:
        raise ValueError("Unexpected outcome-blind source selection")
    forbidden = {"label", "kidney_apn", "dmsa_apn", "state"}
    if forbidden.intersection(column.lower() for column in frame.columns):
        raise ValueError("Diagnostic columns are prohibited")
    return frame


def native_roi(row) -> tuple[np.ndarray, np.ndarray, dict]:
    gray, outer, location = prior.locate_outer(row)
    x, y, width, height = [location[key] for key in ("x", "y", "width", "height")]
    native_gray = gray[y : y + height, x : x + width]
    native_outer = outer[y : y + height, x : x + width]
    if native_gray.shape != native_outer.shape or not native_outer.any():
        raise ValueError(f"Invalid native ROI for {row.image_id}")
    return native_gray, native_outer, location


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve the registered tight-ROI protocol")
    selected = load_selection()
    selected.to_csv(OUT / "selected_images_private.csv", index=False)

    bbox_sizes = []
    for row in selected.itertuples(index=False):
        _, mask, _ = native_roi(row)
        yy, xx = np.nonzero(mask)
        bbox_sizes.append((int(yy.max() - yy.min() + 1), int(xx.max() - xx.min() + 1)))
    median_bbox = np.median(np.asarray(bbox_sizes), axis=0).tolist()
    checkpoints = {variant: K.validate_model_family(variant) for variant in VARIANTS}

    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_labels_read": False,
        "source": "https://github.com/rsingla92/kidneyUS",
        "task": "Task002_KidneyRegions",
        "class_map": {"1": "central_echo_complex", "2": "medulla", "3": "cortex"},
        "sample": "The unchanged 32-patient outcome-blind NCKUH compatibility sample",
        "geometry": (
            "Use the existing kidney-centred native ROI, scale it by 2.0, and centre it on the "
            "official 768x1024 source canvas; 2.0 was fixed from the official-to-local median "
            "foreground-size ratio reported before this experiment"
        ),
        "requested_scale": REQUESTED_SCALE,
        "canvas_shape": list(K.CANVAS_SHAPE),
        "official_median_foreground_bbox_hw": [282, 444],
        "nckuh_sample_median_foreground_bbox_hw": median_bbox,
        "normalization": "Released nnU-Net nonCT preprocessing; no APN-derived transform or percentile selection",
        "inference": "Arithmetic mean of five released source checkpoints per annotator; no TTA",
        "postprocessing": "Invert to native ROI resolution, then restrict all internal classes to the supplied outer kidney contour",
        "necessary_gates": {
            "each_model_nonempty_rate_at_least": 0.8,
            "each_model_median_outer_coverage_at_least": 0.5,
            "each_model_two_region_rate_at_least": 0.8,
            "median_union_dice_when_either_nonempty_at_least": 0.6,
            "median_class_dice_when_either_nonempty_at_least": 0.6,
        },
        "prohibited": "No APN, DMSA, filename diagnosis suffix, model score, development score, or test role enters this compatibility decision",
        "selection_sha256": K.sha256(OUT / "selected_images_private.csv"),
        "source_script_sha256": K.sha256(Path(__file__)),
        "checkpoints": checkpoints,
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(
        OUT / "run_lock.json",
        {
            "protocol_sha256": K.sha256(OUT / "protocol.json"),
            "selection_sha256": K.sha256(OUT / "selected_images_private.csv"),
            "source_sha256": K.sha256(Path(__file__)),
        },
    )
    print(json.dumps({"patients": 32, "median_bbox": median_bbox, "scale": REQUESTED_SCALE}))


def require_lock() -> dict:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert K.sha256(OUT / "protocol.json") == lock["protocol_sha256"]
    assert K.sha256(OUT / "selected_images_private.csv") == lock["selection_sha256"]
    assert K.sha256(Path(__file__)) == lock["source_sha256"]
    return json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))


def run(variant: str) -> None:
    protocol = require_lock()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    nnunet_source = ROOT / "03_data_models/third_party/nnUNet_v1"
    if str(nnunet_source) not in sys.path:
        sys.path.insert(0, str(nnunet_source))
    selected = pd.read_csv(OUT / "selected_images_private.csv", dtype=str)
    run_out = OUT / variant
    trainer = K.build_trainer(variant, run_out)
    checkpoints = K.validate_model_family(variant)
    assert [item["sha256"] for item in checkpoints] == [
        item["sha256"] for item in protocol["checkpoints"][variant]
    ]

    items = []
    for row in selected.itertuples(index=False):
        gray, outer, location = native_roi(row)
        canvas, _, transform = K.embed_on_training_canvas(
            gray, outer, REQUESTED_SCALE, K.CANVAS_SHAPE
        )
        nifti = run_out / "inputs_nifti" / f"{row.image_id}_{MODE}_0000.nii.gz"
        K.array_to_nifti(canvas, nifti)
        data, _, properties = trainer.preprocess_patient([str(nifti)])
        items.append(
            {
                "image_id": row.image_id,
                "view": row.view,
                "gray": gray,
                "outer": outer.astype(bool),
                "location": location,
                "transform": transform,
                "data": data,
                "properties": properties,
                "softmax": None,
            }
        )

    for record in checkpoints:
        checkpoint = K.load_checkpoint(Path(record["path"]))
        trainer.load_checkpoint_ram(checkpoint, train=False)
        trainer.network.eval()
        with torch.inference_mode():
            for item in items:
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
                item["softmax"] = softmax if item["softmax"] is None else item["softmax"] + softmax
        print(f"{variant} source checkpoint {record['fold']} completed", flush=True)
        del checkpoint
        torch.cuda.empty_cache()

    from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax

    metrics = []
    for item in items:
        softmax = (item["softmax"] / len(checkpoints)).transpose(
            [0] + [axis + 1 for axis in trainer.transpose_backward]
        )
        exported = run_out / "output_nifti" / f"{item['image_id']}_{MODE}.nii.gz"
        exported.parent.mkdir(parents=True, exist_ok=True)
        export = trainer.plans.get("segmentation_export_params", {})
        save_segmentation_nifti_from_softmax(
            softmax,
            str(exported),
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
        canvas_labels = sitk.GetArrayFromImage(sitk.ReadImage(str(exported)))
        assert canvas_labels.shape == (1, *K.CANVAS_SHAPE)
        raw = K.invert_canvas_labels(canvas_labels[0], item["transform"])
        native = raw.copy()
        native[~item["outer"]] = 0
        mask_folder = run_out / "region_masks"
        mask_folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_folder / f"{item['image_id']}.png"), native)
        K.write_overlay(
            item["gray"],
            item["outer"],
            native,
            run_out / "overlays" / f"{item['image_id']}.png",
            f"{item['image_id']} {variant} {MODE}",
        )
        outer_pixels = int(item["outer"].sum())
        metrics.append(
            {
                "image_id": item["image_id"],
                "view": item["view"],
                "variant": variant,
                "mode": MODE,
                "outer_pixels": outer_pixels,
                "inside_region_pixels": int((native > 0).sum()),
                "outer_coverage": float((native > 0).sum() / outer_pixels),
                "raw_outside_fraction": float(
                    ((raw > 0) & ~item["outer"]).sum() / max(1, (raw > 0).sum())
                ),
                "classes_present": int(
                    sum(np.count_nonzero(native == class_id) >= 16 for class_id in (1, 2, 3))
                ),
                "applied_scale": float(item["transform"]["applied_scale"]),
            }
        )
    pd.DataFrame(metrics).to_csv(run_out / "technical_metrics.csv", index=False)
    print(f"{variant} tight-ROI technical outputs saved; no APN diagnosis loaded", flush=True)


def summarize() -> None:
    protocol = require_lock()
    summaries = []
    frames = {}
    for variant in VARIANTS:
        frame = pd.read_csv(OUT / variant / "technical_metrics.csv")
        assert len(frame) == 32
        summary = {
            "variant": variant,
            "images": len(frame),
            "nonempty_rate": float(frame.inside_region_pixels.ge(64).mean()),
            "median_outer_coverage": float(frame.outer_coverage.median()),
            "two_region_rate": float(frame.classes_present.ge(2).mean()),
        }
        summaries.append(summary)
        frames[variant] = frame

    union_dice, class_dice = [], []
    for image_id in frames[VARIANTS[0]].image_id:
        masks = [
            cv2.imread(str(OUT / variant / "region_masks" / f"{image_id}.png"), 0)
            for variant in VARIANTS
        ]
        assert masks[0] is not None and masks[0].shape == masks[1].shape
        if np.any(masks[0]) or np.any(masks[1]):
            union_dice.append(K.dice(masks[0] > 0, masks[1] > 0))
            per_class = [
                K.dice(masks[0] == class_id, masks[1] == class_id)
                for class_id in (1, 2, 3)
                if (masks[0] == class_id).any() or (masks[1] == class_id).any()
            ]
            class_dice.append(float(np.mean(per_class)))

    median_union = float(np.median(union_dice)) if union_dice else 0.0
    median_class = float(np.median(class_dice)) if class_dice else 0.0
    gates = protocol["necessary_gates"]
    passed = bool(
        all(
            summary["nonempty_rate"] >= gates["each_model_nonempty_rate_at_least"]
            and summary["median_outer_coverage"]
            >= gates["each_model_median_outer_coverage_at_least"]
            and summary["two_region_rate"] >= gates["each_model_two_region_rate_at_least"]
            for summary in summaries
        )
        and median_union >= gates["median_union_dice_when_either_nonempty_at_least"]
        and median_class >= gates["median_class_dice_when_either_nonempty_at_least"]
    )
    result = {
        "diagnostic_labels_read": False,
        "source_models": summaries,
        "intermodel_agreement": {
            "median_union_dice": median_union,
            "median_present_class_dice": median_class,
            "either_nonempty_images": len(union_dice),
        },
        "necessary_technical_gates_passed": passed,
        "classification_performed": False,
        "next_action": (
            "Run full-cohort compartment extraction, then evaluate training-only compartment features"
            if passed
            else "Exclude released compartment model on NCKUH; await clinician anatomy annotation"
        ),
    }
    write_json(OUT / "decision.json", result)
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize"))
    parser.add_argument("--variant", choices=VARIANTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.stage == "run":
        if args.variant is None:
            raise SystemExit("run requires --variant")
        run(args.variant)
    elif args.stage == "prepare":
        prepare()
    else:
        summarize()
