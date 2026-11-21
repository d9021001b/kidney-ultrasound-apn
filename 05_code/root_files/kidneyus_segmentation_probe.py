"""Evaluate official kidneyUS nnU-Net v1 checkpoints on CYCH images.

This is a segmentation-only probe. It never reads the inflammation label column,
does not overwrite the established masks, and loads downloaded checkpoints with
PyTorch's restricted weights-only unpickler.
"""
from __future__ import annotations

import argparse
import _codecs
from collections import OrderedDict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import pickletools
import zipfile

import cv2
import numpy as np
import SimpleITK as sitk
import torch


ROOT = Path(__file__).resolve().parent
WEIGHT_ROOT = (
    ROOT
    / "external_models"
    / "kidneyUS_official_20260914"
    / "weights"
)
MODEL_FOLDER = (
    WEIGHT_ROOT
    / "annotator_1"
    / "Task001_KidneyCapsule"
    / "nnUNetTrainerV2__nnUNetPlansv2.1"
)
SOURCE_RUN = ROOT / "CYCH_random_validation_20260913"
OUT = ROOT / "CYCH_kidneyUS_segmentation_20260914"
CHECKPOINT_NAME = "model_final_checkpoint.model"
FOLDS = tuple(range(5))

ALLOWED_TORCH_GLOBALS = {
    "_codecs encode",
    "collections OrderedDict",
    "numpy dtype",
    "numpy.core.multiarray scalar",
    "torch FloatStorage",
    "torch._utils _rebuild_tensor_v2",
}


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """Load nnU-Net metadata while denying arbitrary imported globals."""

    _allowed = {
        ("collections", "OrderedDict"): OrderedDict,
        ("numpy", "dtype"): np.dtype,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
        ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
    }

    def find_class(self, module: str, name: str):
        try:
            return self._allowed[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(
                f"Denied pickle global: {module}.{name}"
            ) from exc


def restricted_pickle_load(path: Path):
    with path.open("rb") as handle:
        return RestrictedNumpyUnpickler(handle).load()


def _legacy_numpy_scalar(*args):
    return np._core.multiarray.scalar(*args)


_legacy_numpy_scalar.__module__ = "numpy.core.multiarray"
_legacy_numpy_scalar.__name__ = "scalar"


def safe_torch_load(path: Path):
    dtype_classes = {
        type(np.dtype(name))
        for name in ("bool", "uint8", "int16", "int32", "int64", "float16", "float32", "float64")
    }
    with torch.serialization.safe_globals(
        [_legacy_numpy_scalar, np.dtype, _codecs.encode, *dtype_classes]
    ):
        return torch.load(path, map_location="cpu", weights_only=True)


def torch_pickle_globals(path: Path) -> set[str]:
    """Inspect the small pickle instruction stream without loading tensor data."""
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        payload = archive.read(member)
    return {
        str(arg)
        for op, arg, _ in pickletools.genops(io.BytesIO(payload))
        if op.name == "GLOBAL"
    }


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit_checkpoints() -> list[dict]:
    rows = []
    for checkpoint in sorted(WEIGHT_ROOT.rglob("*.model")):
        globals_used = torch_pickle_globals(checkpoint)
        denied = sorted(globals_used - ALLOWED_TORCH_GLOBALS)
        if denied:
            raise RuntimeError(f"Unexpected pickle globals in {checkpoint}: {denied}")
        sidecar = Path(str(checkpoint) + ".pkl")
        info = restricted_pickle_load(sidecar)
        if info.get("name") != "nnUNetTrainerV2":
            raise RuntimeError(f"Unexpected trainer in {sidecar}: {info.get('name')}")
        plans = info["plans"]
        if int(plans["num_modalities"]) != 1:
            raise RuntimeError(f"Unexpected modality count in {sidecar}")
        relative = checkpoint.relative_to(WEIGHT_ROOT)
        rows.append(
            {
                "relative_path": str(relative),
                "bytes": checkpoint.stat().st_size,
                "sidecar_bytes": sidecar.stat().st_size,
                "trainer": info["name"],
                "num_modalities": int(plans["num_modalities"]),
                "num_classes_excluding_background": int(plans["num_classes"]),
                "pickle_globals": ";".join(sorted(globals_used)),
                "status": "restricted_metadata_and_static_checkpoint_audit_passed",
            }
        )
    if len(rows) != 41:
        raise RuntimeError(f"Expected 41 checkpoints, found {len(rows)}")
    OUT.mkdir(exist_ok=True)
    with (OUT / "safe_checkpoint_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "safe_checkpoint_audit.json").write_text(
        json.dumps(
            {
                "checkpoint_count": len(rows),
                "allowed_torch_pickle_globals": sorted(ALLOWED_TORCH_GLOBALS),
                "sidecar_loader": "RestrictedNumpyUnpickler",
                "checkpoint_loader": "torch.load(weights_only=True)",
                "passed": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def build_trainer():
    path_root = OUT / "nnunet_paths"
    raw_base = path_root / "raw"
    preprocessed_base = path_root / "preprocessed"
    results_base = path_root / "results"
    dataset_placeholder = preprocessed_base / "Task001_KidneyCapsule"
    for path in (raw_base, preprocessed_base, results_base, dataset_placeholder):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("nnUNet_raw_data_base", str(raw_base))
    os.environ.setdefault("nnUNet_preprocessed", str(preprocessed_base))
    os.environ.setdefault("RESULTS_FOLDER", str(results_base))

    from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2

    sidecar = MODEL_FOLDER / "fold_0" / f"{CHECKPOINT_NAME}.pkl"
    info = restricted_pickle_load(sidecar)
    if info.get("name") != "nnUNetTrainerV2":
        raise RuntimeError(f"Unexpected trainer: {info.get('name')}")
    init_args = list(info["init"])
    init_args[0] = str(MODEL_FOLDER / "plans.pkl")
    init_args[1] = 0
    init_args[2] = str(OUT / "nnunet_runtime")
    init_args[3] = str(dataset_placeholder)
    init_args[5] = 0
    init_args[6] = False
    init_args[7] = True
    init_args[8] = False
    trainer = nnUNetTrainerV2(*init_args)
    trainer.process_plans(info["plans"])
    trainer.initialize(training=False)
    trainer.network.eval()
    return trainer


def load_slim_checkpoint(path: Path) -> dict:
    payload = safe_torch_load(path)
    state = payload.get("state_dict")
    if not isinstance(state, (dict, OrderedDict)) or len(state) != 134:
        raise RuntimeError(f"Unexpected state dict in {path}")
    return {
        "state_dict": state,
        "epoch": int(payload.get("epoch", 0)),
        "plot_stuff": payload.get("plot_stuff", ([], [], [], [])),
        "best_stuff": payload.get("best_stuff", (None, None, None)),
    }


def png_to_nifti(png_path: Path, nifti_path: Path) -> tuple[np.ndarray, str]:
    gray = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.ndim != 2:
        raise ValueError(f"Could not read grayscale image: {png_path}")
    image = sitk.GetImageFromArray(gray[None].astype(np.float32))
    image.SetSpacing((1.0, 1.0, 999.0))
    nifti_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(nifti_path), True)
    return gray, hashlib.sha256(gray.tobytes()).hexdigest()


def largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == keep).astype(np.uint8)


def overlap_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    pred = prediction.astype(bool)
    ref = reference.astype(bool)
    intersection = int(np.logical_and(pred, ref).sum())
    union = int(np.logical_or(pred, ref).sum())
    return {
        "iou": intersection / union if union else 1.0,
        "dice": (2 * intersection) / (int(pred.sum()) + int(ref.sum()))
        if pred.any() or ref.any()
        else 1.0,
    }


def border_fraction(mask: np.ndarray) -> float:
    binary = mask.astype(bool)
    total = int(binary.sum())
    if not total:
        return 0.0
    border = np.zeros_like(binary)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    return int(np.logical_and(binary, border).sum()) / total


def write_overlay(
    gray: np.ndarray,
    manual: np.ndarray,
    existing: np.ndarray,
    official: np.ndarray,
    output: Path,
    image_id: str,
) -> None:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    panels = []
    definitions = (
        ("Manual reference", manual, (0, 220, 220)),
        ("Existing YOLO", existing, (220, 180, 0)),
        ("kidneyUS nnU-Net", official, (220, 0, 220)),
    )
    for title, mask, color in definitions:
        panel = base.copy()
        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(panel, contours, -1, color, 2)
        scale = max(2, int(np.ceil(640 / max(1, panel.shape[1]))))
        panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        panels.append(panel)
    height = max(panel.shape[0] for panel in panels)
    padded = [cv2.copyMakeBorder(p, 0, height - p.shape[0], 0, 0, cv2.BORDER_CONSTANT) for p in panels]
    canvas = np.hstack(padded)
    cv2.putText(canvas, image_id, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def read_manifest_without_diagnosis() -> list[dict]:
    with (SOURCE_RUN / "image_manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    allowed = {
        "image_id",
        "patient_alias",
        "view",
        "clean_path",
        "filename",
        "predicted_iou",
        "mask_source",
    }
    return [{key: row[key] for key in allowed} for row in rows]


def run_probe(image_ids: list[str], tta: bool) -> list[dict]:
    from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
    from recrop_with_yolo import polygon_mask

    records = read_manifest_without_diagnosis()
    by_id = {row["image_id"]: row for row in records}
    missing = sorted(set(image_ids) - set(by_id))
    if missing:
        raise KeyError(f"Unknown image ids: {missing}")
    selected = [by_id[image_id] for image_id in image_ids]

    for subdir in ("input_nifti", "output_nifti", "predicted_masks", "overlays", "nnunet_runtime"):
        (OUT / subdir).mkdir(parents=True, exist_ok=True)

    trainer = build_trainer()
    preprocessed = {}
    for row in selected:
        image_id = row["image_id"]
        nifti = OUT / "input_nifti" / f"{image_id}_0000.nii.gz"
        gray, pixel_sha = png_to_nifti(Path(row["clean_path"]), nifti)
        data, _, properties = trainer.preprocess_patient([str(nifti)])
        preprocessed[image_id] = {
            "gray": gray,
            "pixel_sha256": pixel_sha,
            "data": data,
            "properties": properties,
            "softmax_sum": None,
        }

    checkpoint_records = []
    for fold in FOLDS:
        checkpoint_path = MODEL_FOLDER / f"fold_{fold}" / CHECKPOINT_NAME
        checkpoint_records.append(
            {
                "fold": fold,
                "path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": sha256(checkpoint_path),
            }
        )
        checkpoint = load_slim_checkpoint(checkpoint_path)
        trainer.load_checkpoint_ram(checkpoint, train=False)
        trainer.network.eval()
        with torch.inference_mode():
            for row in selected:
                item = preprocessed[row["image_id"]]
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
                if item["softmax_sum"] is None:
                    item["softmax_sum"] = softmax
                else:
                    item["softmax_sum"] += softmax
        del checkpoint
        torch.cuda.empty_cache()

    metrics = []
    for row in selected:
        image_id = row["image_id"]
        item = preprocessed[image_id]
        softmax = item["softmax_sum"] / len(FOLDS)
        softmax = softmax.transpose([0] + [axis + 1 for axis in trainer.transpose_backward])
        output_nifti = OUT / "output_nifti" / f"{image_id}.nii.gz"
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
        predicted = sitk.GetArrayFromImage(sitk.ReadImage(str(output_nifti)))
        if predicted.shape[0] != 1:
            raise RuntimeError(f"Unexpected output shape for {image_id}: {predicted.shape}")
        official = largest_component(predicted[0])
        gray = item["gray"]
        if official.shape != gray.shape:
            raise RuntimeError(f"Shape mismatch for {image_id}: {official.shape} vs {gray.shape}")
        manual = polygon_mask(
            ROOT / "cych_image_label" / Path(row["filename"]).with_suffix(".txt"),
            *gray.shape,
        ).astype(np.uint8)
        existing_path = SOURCE_RUN / "predicted_masks" / f"{image_id}.png"
        existing = (cv2.imread(str(existing_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        official_metrics = overlap_metrics(official, manual)
        existing_metrics = overlap_metrics(existing, manual)
        cv2.imwrite(str(OUT / "predicted_masks" / f"{image_id}.png"), official * 255)
        write_overlay(
            gray,
            manual,
            existing,
            official,
            OUT / "overlays" / f"{image_id}.png",
            image_id,
        )
        metrics.append(
            {
                "image_id": image_id,
                "patient_alias": row["patient_alias"],
                "view": row["view"],
                "tta": tta,
                "fold_count": len(FOLDS),
                "official_pixels": int(official.sum()),
                "manual_pixels": int(manual.sum()),
                "existing_yolo_pixels": int(existing.sum()),
                "official_iou_to_manual": official_metrics["iou"],
                "official_dice_to_manual": official_metrics["dice"],
                "existing_iou_to_manual": existing_metrics["iou"],
                "existing_dice_to_manual": existing_metrics["dice"],
                "official_border_fraction": border_fraction(official),
                "input_pixel_sha256": item["pixel_sha256"],
            }
        )
    with (OUT / "segmentation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    protocol = {
        "scope": "Segmentation only; inflammation labels were not read.",
        "source_manifest": str(SOURCE_RUN / "image_manifest.csv"),
        "official_weight_source": "https://github.com/rsingla92/kidneyUS",
        "model_folder": str(MODEL_FOLDER),
        "variant": "annotator_1",
        "task": "Task001_KidneyCapsule",
        "checkpoint_name": CHECKPOINT_NAME,
        "folds": list(FOLDS),
        "ensemble": "Arithmetic mean of five foreground/background softmax outputs",
        "tta": tta,
        "input_conversion": "Grayscale float32 NIfTI, array shape 1xHxW, spacing 1x1x999",
        "postprocessing": "Largest connected foreground component",
        "checkpoint_loading": "Static pickle-global audit plus torch.load(weights_only=True)",
        "checkpoint_records": checkpoint_records,
        "image_ids": image_ids,
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "smoke", "systematic", "full"))
    parser.add_argument("--image-id", action="append", default=[])
    parser.add_argument("--tta", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_checkpoints()
    if args.mode == "audit":
        print("Safe checkpoint audit passed for 41 model files.")
        return
    rows = read_manifest_without_diagnosis()
    if args.image_id:
        image_ids = args.image_id
    elif args.mode == "smoke":
        image_ids = ["I0050"]
    elif args.mode == "systematic":
        indexes = np.linspace(0, len(rows) - 1, 12, dtype=int)
        image_ids = [rows[index]["image_id"] for index in indexes]
    else:
        image_ids = [row["image_id"] for row in rows]
    metrics = run_probe(image_ids, tta=args.tta)
    mean_official = float(np.mean([row["official_iou_to_manual"] for row in metrics]))
    mean_existing = float(np.mean([row["existing_iou_to_manual"] for row in metrics]))
    print(
        json.dumps(
            {
                "images": len(metrics),
                "official_mean_iou_to_manual": mean_official,
                "existing_yolo_mean_iou_to_manual": mean_existing,
                "tta": args.tta,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
