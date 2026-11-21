"""Encode contour-first NCKUH kidneys with the official OpenUS-S teacher."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "03_data_models/third_party/OpenUS_official_20261105"
WEIGHT = MODEL_ROOT / "openus_cpt0150.pth"
SOURCE = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_fullframe_20261103"
OUT = ROOT / "02_experiments/NCKUH_DMSA_openus_frozen_20261105"
WEIGHT_SHA256 = "a82365cc2994b9826c8c307a6ee19acf7dc7c42468eb76b95285de58aea91720"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def tight_canvas(image: np.ndarray, mask: np.ndarray, size: int = 224) -> tuple[np.ndarray, np.ndarray]:
    """Exact contour-crop geometry used by the registered full-cohort path."""
    binary = mask.astype(bool)
    yy, xx = np.nonzero(binary)
    region = image[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    selected = binary[yy.min():yy.max() + 1, xx.min():xx.max() + 1].astype(np.uint8)
    scale = size / max(region.shape[:2])
    height = max(1, round(region.shape[0] * scale))
    width = max(1, round(region.shape[1] * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    output = np.zeros((size, size, 3), np.uint8)
    output_mask = np.zeros((size, size), np.uint8)
    top, left = (size - height) // 2, (size - width) // 2
    output[top:top + height, left:left + width] = cv2.resize(
        region, (width, height), interpolation=interpolation
    )
    output_mask[top:top + height, left:left + width] = cv2.resize(
        selected, (width, height), interpolation=cv2.INTER_NEAREST
    )
    output *= output_mask[..., None]
    return output, output_mask


def load_model(device: torch.device):
    if digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError("OpenUS checkpoint SHA-256 mismatch")
    scan_path = MODEL_ROOT / "vmamba_models/selective_scan"
    sys.path[:0] = [str(scan_path), str(MODEL_ROOT)]

    # The repository imports the compiled core but leaves its availability flag false.
    import selective_scan_cuda_core  # noqa: F401
    from vmamba_models import csms6s
    from vmamba_models.dino_vmamba import Backbone_DINOv2_VSSM_2

    csms6s.WITH_SELECTIVESCAN_CORE = True
    checkpoint = torch.load(WEIGHT, map_location="cpu", weights_only=False)
    state = {
        key[9:]: value
        for key, value in checkpoint["teacher"].items()
        if key.startswith("backbone.")
    }
    model = Backbone_DINOv2_VSSM_2(masked_im_modeling=True, training=False)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys or len(state) != 412:
        raise ValueError("OpenUS strict teacher loading failed")
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint


def selected_mean(
    tokens: np.ndarray, selected: np.ndarray, fallback: np.ndarray
) -> np.ndarray:
    if not selected.any():
        selected = fallback
    return tokens.reshape(-1, tokens.shape[-1])[selected.reshape(-1)].mean(0)


def summarize(tokens: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, ...]:
    if tokens.ndim != 3 or tokens.shape[1:] != (50, 768):
        raise ValueError(f"Unexpected OpenUS token shape: {tokens.shape}")
    coverage = masks.reshape(-1, 7, 32, 7, 32).mean((2, 4))
    patch = tokens[:, 1:].reshape(-1, 7, 7, 768)
    roi_mean, roi_std, contrasts = [], [], []
    yy, xx = np.mgrid[:7, :7]
    for current, current_coverage in zip(patch, coverage):
        valid = current_coverage >= 0.25
        if not valid.any():
            valid.flat[np.argmax(current_coverage)] = True
        selected = current[valid]
        roi_mean.append(selected.mean(0))
        roi_std.append(selected.std(0))

        top = selected_mean(current, valid & (yy <= 3), valid)
        bottom = selected_mean(current, valid & (yy >= 3), valid)
        left = selected_mean(current, valid & (xx <= 3), valid)
        right = selected_mean(current, valid & (xx >= 3), valid)
        center_region = (yy >= 2) & (yy <= 4) & (xx >= 2) & (xx <= 4)
        center = selected_mean(current, valid & center_region, valid)
        periphery = selected_mean(current, valid & ~center_region, valid)
        contrasts.append(np.concatenate([top - bottom, left - right, center - periphery]))
    return (
        tokens[:, 0].astype(np.float32),
        np.asarray(roi_mean, np.float32),
        np.asarray(roi_std, np.float32),
        np.asarray(contrasts, np.float32),
    )


def run() -> None:
    manifest_path = OUT / "encoding_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("Run the Windows preparation step first")
    output = OUT / "openus_embeddings.npz"
    if output.exists():
        raise FileExistsError(f"Preserve existing embeddings: {output}")
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 820 or manifest.image_id.duplicated().any():
        raise ValueError("Unexpected encoding manifest")

    device = torch.device("cuda")
    model, checkpoint = load_model(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    global_rows, roi_rows, deviation_rows, contrast_rows = [], [], [], []
    batch_size = 16
    with torch.inference_mode():
        for start in range(0, len(manifest), batch_size):
            canvases, masks = [], []
            for identifier in manifest.image_id.iloc[start:start + batch_size]:
                bgr = cv2.imread(str(SOURCE / "clean_rgb" / f"{identifier}.png"), cv2.IMREAD_COLOR)
                mask_image = cv2.imread(str(SOURCE / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE)
                if bgr is None or mask_image is None:
                    raise FileNotFoundError(identifier)
                mask = mask_image > 0
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                canvas, canvas_mask = tight_canvas(rgb * mask[..., None], mask)
                canvases.append(canvas)
                masks.append(canvas_mask.astype(np.float32))
            tensor = torch.from_numpy(
                np.stack(canvases).transpose(0, 3, 1, 2).copy()
            ).to(device).float() / 255.0
            tokens = model((tensor - mean) / std).cpu().numpy()
            values = summarize(tokens, np.stack(masks))
            for target, value in zip(
                (global_rows, roi_rows, deviation_rows, contrast_rows), values
            ):
                target.append(value)
            print(f"OpenUS encoded {min(start + batch_size, len(manifest))}/{len(manifest)}", flush=True)

    arrays = {
        "global_mean": np.concatenate(global_rows),
        "roi_mean": np.concatenate(roi_rows),
        "roi_std": np.concatenate(deviation_rows),
        "regional_contrasts": np.concatenate(contrast_rows),
    }
    expected = {
        "global_mean": (820, 768),
        "roi_mean": (820, 768),
        "roi_std": (820, 768),
        "regional_contrasts": (820, 2304),
    }
    if {name: value.shape for name, value in arrays.items()} != expected:
        raise ValueError("Unexpected OpenUS embedding shapes")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite OpenUS embeddings")
    np.savez_compressed(output, **arrays)
    audit = {
        "images": 820,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "teacher_state_keys": 412,
        "strict_weight_load": True,
        "cuda_core": True,
        "labels_used_in_encoding": [],
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
        "weight_sha256": digest(WEIGHT),
        "embedding_sha256": digest(output),
    }
    (OUT / "encoder_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    run()
