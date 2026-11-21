"""Compare native Transformers inference to the existing custom-loader cache.

No labels are read and no classifier is trained. Original bundles/results are
left intact. Run this before interpreting the previous DINOv3 screening result.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import transformers
from safetensors import safe_open
from transformers import AutoImageProcessor, DINOv3ViTConfig, DINOv3ViTModel
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTRopePositionEmbedding

from renal_random_validation import canvas_image


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "dinov3_vith16plus_weights_20260913"
PREVIOUS = ROOT / "CYCH_dinov3_single_encoder_20260913"
OUT = ROOT / "CYCH_next_steps_20260913"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(exist_ok=True)
    config = DINOv3ViTConfig.from_pretrained(BUNDLE, local_files_only=True)
    native_rope = DINOv3ViTRopePositionEmbedding(config).eval()
    native_cos, native_sin = native_rope(torch.zeros(1, 3, 224, 224))

    # Execute only the self-contained rotary class already inspected in source.
    tree = ast.parse((BUNDLE / "modeling_dinov3_vit.py").read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "Dinov3RotaryEmbedding")
    namespace = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "inspected_rotary_class", "exec"), namespace)
    custom_rope = namespace["Dinov3RotaryEmbedding"](64, theta=config.rope_theta).eval()
    custom_cos, custom_sin = custom_rope._build_freqs(14, 14, torch.device("cpu"), torch.float32)
    custom_cos, custom_sin = custom_cos.tile(1, 2), custom_sin.tile(1, 2)
    rotary = {
        "shape": list(native_cos.shape),
        "cos_max_absolute_difference": float((custom_cos - native_cos).abs().max()),
        "sin_max_absolute_difference": float((custom_sin - native_sin).abs().max()),
        "equal_at_tolerance_1e_6": bool(torch.allclose(custom_cos, native_cos, atol=1e-6, rtol=1e-6) and torch.allclose(custom_sin, native_sin, atol=1e-6, rtol=1e-6)),
        "custom_coordinates": "integer patch coordinates starting at zero; no 2*pi factor",
        "native_coordinates": "normalized patch-center coordinates in [-1,1]; angles include 2*pi",
    }
    print(json.dumps({"rotary_comparison": rotary}), flush=True)
    model, info = DINOv3ViTModel.from_pretrained(
        BUNDLE, config=config, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, output_loading_info=True, attn_implementation="eager",
    )
    unmatched = {key: list(info.get(key, [])) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
    if any(unmatched.values()):
        raise RuntimeError(f"Native model load mismatch: {unmatched}")
    model.cuda().eval().requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(BUNDLE, local_files_only=True, trust_remote_code=False)
    images = []
    for alias in ("I0001", "I0002"):
        gray = cv2.imread(str(PREVIOUS / "images/percentile" / f"{alias}.png"), 0)
        mask = cv2.imread(str(PREVIOUS / "masks" / f"{alias}.png"), 0)
        if gray is None or mask is None:
            raise RuntimeError("Missing fixed diagnostic image")
        images.append(canvas_image(gray, (mask > 0).astype(np.uint8), channels="gray", size=224))
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].cuda()
    with torch.inference_mode():
        result = model(pixel_values=pixel_values).last_hidden_state.cpu().numpy()
    old = np.load(PREVIOUS / "features/dinov3_vith16plus_image.npy")[:2]
    cls = result[:, 0, :]
    cosine = np.sum(cls * old, axis=1) / (np.linalg.norm(cls, axis=1) * np.linalg.norm(old, axis=1))
    weight = BUNDLE / "model.safetensors"
    with safe_open(weight, framework="pt", device="cpu") as handle:
        tensor_count = len(handle.keys())
    record = {
        "purpose": "Label-free implementation audit; not a new disease-performance experiment",
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "loader_class": f"{type(model).__module__}.{type(model).__name__}",
        "trust_remote_code": False, "local_files_only": True,
        "dtype": "float32", "attention_implementation": "eager",
        "weight_sha256": digest(weight), "weight_tensors": tensor_count,
        "loading_info": unmatched, "native_implementation_sha256": digest(inspect.getfile(type(model))),
        "custom_implementation_sha256": digest(BUNDLE / "modeling_dinov3_vit.py"),
        "rotary_comparison": rotary,
        "diagnostic_aliases": ["I0001", "I0002"], "native_output_shape": list(result.shape),
        "native_finite": bool(np.isfinite(result).all()),
        "cls_cosine_native_vs_previous_custom_cache": cosine.astype(float).tolist(),
        "cls_max_absolute_difference": float(np.max(np.abs(cls - old))),
        "interpretation": "Previous custom-loader scores cannot be used to reject official DINOv3. Correct native inference must be assessed with a new locked experiment.",
    }
    (OUT / "dinov3_loader_audit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
