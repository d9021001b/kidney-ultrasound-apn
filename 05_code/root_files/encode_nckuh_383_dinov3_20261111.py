"""Frozen official DINOv3 encoding of registered NCKUH contour-first images."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from nckuh_dmsa_fullcohort_fullframe_20261103 import tight_canvas


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
CACHE = ROOT / "02_experiments/NCKUH_revised_383_contour_cache_20261111"
WEIGHTS = ROOT / "03_data_models/dinov3_vith16plus_weights_20260913"
OUT = ROOT / "02_experiments/NCKUH_revised_383_dinov3_20261111"


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    import pandas as pd
    import torch
    from transformers import AutoImageProcessor, DINOv3ViTConfig, DINOv3ViTModel

    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "dinov3_class.npy").exists() or (OUT / "dinov3_patch.npy").exists():
        raise FileExistsError("Preserve existing DINOv3 encoding")
    audit = json.loads((CACHE / "extraction_audit.json").read_text(encoding="utf-8"))
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    assert len(images) == 1362 and audit["images"] == len(images)
    manifest = json.loads((WEIGHTS / "WEIGHT_MANIFEST.json").read_text(encoding="utf-8"))
    assert digest(WEIGHTS / "model.safetensors") == manifest["weight_sha256"]
    torch.set_num_threads(4)
    torch.manual_seed(20261111)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = DINOv3ViTConfig.from_pretrained(WEIGHTS, local_files_only=True)
    model, info = DINOv3ViTModel.from_pretrained(
        WEIGHTS, config=config, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, output_loading_info=True, attn_implementation="sdpa")
    assert not any(info.get(k, []) for k in
                   ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    model = model.cuda().eval().requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(
        WEIGHTS, local_files_only=True, trust_remote_code=False)
    classes, patches, counts = [], [], []
    with torch.inference_mode():
        for start in range(0, len(images), 4):
            canvases, masks = [], []
            for image_id in images.image_id.iloc[start:start + 4]:
                gray = cv2.imread(str(CACHE / "percentile" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(str(CACHE / "masks" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
                assert gray is not None and mask is not None and gray.shape == mask.shape
                canvas, selected = tight_canvas(gray, mask > 0)
                canvases.append(canvas)
                masks.append(selected)
            batch = processor(images=canvases, return_tensors="pt", do_resize=False)
            tokens = model(pixel_values=batch["pixel_values"].cuda()).last_hidden_state.cpu().numpy()
            classes.extend(tokens[:, 0])
            for local, mask in enumerate(masks):
                coverage = mask.reshape(14, 16, 14, 16).mean((1, 3)).reshape(-1)
                selected = coverage >= .8
                if not selected.any():
                    selected[np.argmax(coverage)] = True
                features = tokens[local, 5:][selected]
                patches.append(np.r_[features.mean(0), features.std(0)])
                counts.append(int(selected.sum()))
            if start % 80 == 0:
                print(f"DINOv3 {min(start + 4, len(images))}/{len(images)}", flush=True)
    classes = np.asarray(classes, dtype=np.float32)
    patches = np.asarray(patches, dtype=np.float32)
    assert classes.shape == (1362, 1280) and patches.shape == (1362, 2560)
    assert np.isfinite(classes).all() and np.isfinite(patches).all()
    np.save(OUT / "dinov3_class.npy", classes)
    np.save(OUT / "dinov3_patch.npy", patches)
    summary = {"images": len(images), "weight_sha256": manifest["weight_sha256"],
               "image_manifest_sha256": digest(REGISTRY / "contour_image_manifest.csv"),
               "class_sha256": digest(OUT / "dinov3_class.npy"),
               "patch_sha256": digest(OUT / "dinov3_patch.npy"),
               "selected_patch_min_median_max": [min(counts), float(np.median(counts)), max(counts)],
               "frozen": True, "label_used_in_encoding": False,
               "input": "registered masked percentile kidney ROI; 224px tight canvas"}
    (OUT / "encoding_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
