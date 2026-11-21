"""Label-blind re-encoding of the old tile and USFM branches on revised ROIs."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_native_tile_probe as old_tile
import nckuh_dmsa_usfm_frozen as old_usfm
import nckuh_dmsa_fullcohort_parenchyma_mil_20261103 as old_patch
import nckuh_dmsa_fullcohort_directional as old_directional
from prepare_nckuh_383_contour_cohort_20261111 import OUT as REGISTRY, digest


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "02_experiments/NCKUH_revised_383_contour_cache_20261111"
OUT = ROOT / "02_experiments/NCKUH_revised_383_legacy_features_20261112"


def manifest():
    lock = json.loads((REGISTRY / "cohort_lock.json").read_text(encoding="utf-8"))
    source = REGISTRY / "contour_image_manifest.csv"
    assert digest(source) == lock["files_sha256"][source.name]
    frame = pd.read_csv(source)
    assert len(frame) == 1362 and frame.image_id.is_unique
    return frame


def encode_tiles(ids):
    target = OUT / "tile_mean.npy"
    if target.exists():
        raise FileExistsError(target)
    sys.path.insert(0, str(base.RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    model = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    rows, tile_counts, fallbacks = [], [], 0
    with torch.inference_mode():
        for start in range(0, len(ids), 16):
            canvases, lengths = [], []
            for image_id in ids[start:start + 16]:
                rgb = cv2.cvtColor(
                    cv2.imread(str(CACHE / "clean_rgb" / f"{image_id}.png")),
                    cv2.COLOR_BGR2RGB,
                )
                mask = cv2.imread(str(CACHE / "masks" / f"{image_id}.png"), 0) > 0
                tiles, fallback = old_tile.tiles(rgb, mask)
                canvases.extend(tiles)
                lengths.append(len(tiles))
                fallbacks += int(fallback)
            tensor = torch.from_numpy(
                np.stack(canvases).transpose(0, 3, 1, 2).copy()
            ).cuda().float() / 127.5 - 1
            features = torch.cat([model(part) for part in tensor.split(32)]).cpu().numpy()
            rows.extend(part.mean(0) for part in np.split(features, np.cumsum(lengths)[:-1]))
            tile_counts.extend(lengths)
            print(f"Tile features {min(start + 16, len(ids))}/{len(ids)}", flush=True)
    values = np.asarray(rows, dtype=np.float32)
    assert values.shape == (len(ids), 1024) and np.isfinite(values).all()
    np.save(target, values)
    return {"file": target.name, "sha256": digest(target), "images": len(ids),
            "tiles": int(sum(tile_counts)), "fallbacks": fallbacks}


def encode_usfm(ids):
    target = OUT / "usfm_gap.npy"
    if target.exists():
        raise FileExistsError(target)
    assert old_usfm.digest(old_usfm.CHECKPOINT) == old_usfm.CHECKPOINT_SHA256
    model, _, _ = old_usfm.construct_model()
    model = model.cuda().eval().requires_grad_(False)
    mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1)
    rows = []
    with torch.inference_mode():
        for start in range(0, len(ids), 24):
            canvases = []
            for image_id in ids[start:start + 24]:
                bgr = cv2.imread(str(CACHE / "clean_rgb" / f"{image_id}.png"))
                mask = cv2.imread(str(CACHE / "masks" / f"{image_id}.png"), 0) > 0
                canvas, _ = base.tight_canvas(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask)
                canvases.append(canvas)
            tensor = torch.from_numpy(
                np.stack(canvases).transpose(0, 3, 1, 2).copy()
            ).cuda().float() / 255
            tokens = old_usfm.all_tokens(model, (tensor - mean) / std)
            rows.append(tokens[:, 1:].mean(1).cpu().numpy())
            print(f"USFM features {min(start + 24, len(ids))}/{len(ids)}", flush=True)
    values = np.concatenate(rows).astype(np.float32)
    assert values.shape == (len(ids), 768) and np.isfinite(values).all()
    np.save(target, values)
    return {"file": target.name, "sha256": digest(target), "images": len(ids),
            "checkpoint_sha256": old_usfm.CHECKPOINT_SHA256}


def encode_reflection(ids):
    original_path = OUT / "rad_color.npy"
    mirror_path = OUT / "rad_color_horizontal_reflection.npy"
    if original_path.exists() or mirror_path.exists():
        raise FileExistsError("Preserve completed reflection cache")
    sys.path.insert(0, str(base.RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    model = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    original, mirrored = [], []
    with torch.inference_mode():
        for start in range(0, len(ids), 32):
            canvases = []
            for image_id in ids[start:start + 32]:
                bgr = cv2.imread(str(CACHE / "clean_rgb" / f"{image_id}.png"))
                mask = cv2.imread(str(CACHE / "masks" / f"{image_id}.png"), 0) > 0
                canvas, _ = base.tight_canvas(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask)
                canvases.append(canvas)
            tensor = torch.from_numpy(
                np.stack(canvases).transpose(0, 3, 1, 2).copy()
            ).cuda().float() / 127.5 - 1
            original.append(model(tensor).cpu().numpy())
            mirrored.append(model(tensor.flip((-1,))).cpu().numpy())
            print(f"Reflection features {min(start + 32, len(ids))}/{len(ids)}", flush=True)
    original = np.concatenate(original).astype(np.float32)
    mirrored = np.concatenate(mirrored).astype(np.float32)
    assert original.shape == mirrored.shape == (len(ids), 1024)
    np.save(original_path, original)
    np.save(mirror_path, mirrored)
    return {"files_sha256": {path.name: digest(path) for path in (original_path, mirror_path)},
            "images": len(ids)}


def encode_patches(ids):
    features_path = OUT / "radimagenet_local_float16.npy"
    offsets_path = OUT / "patch_offsets.npy"
    if features_path.exists() or offsets_path.exists():
        raise FileExistsError("Preserve completed local patch cache")
    sys.path.insert(0, str(base.RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    model = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    flat, offsets, excluded = [], [0], 0
    with torch.inference_mode():
        for start in range(0, len(ids), 32):
            canvases, kidneys, artifacts = [], [], []
            for image_id in ids[start:start + 32]:
                bgr = cv2.imread(str(CACHE / "clean_rgb" / f"{image_id}.png"))
                mask = cv2.imread(str(CACHE / "masks" / f"{image_id}.png"), 0) > 0
                canvas, canvas_mask = base.tight_canvas(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask)
                canvases.append(canvas)
                kidneys.append(canvas_mask.astype(bool))
                artifacts.append(old_patch.cyan_overlay_mask(canvas) & canvas_mask.astype(bool))
            tensor = torch.from_numpy(
                np.stack(canvases).transpose(0, 3, 1, 2).copy()
            ).cuda().float()
            maps = torch.relu(model.features(tensor / 127.5 - 1)).permute(0, 2, 3, 1).cpu().numpy()
            assert maps.shape[1:] == (7, 7, 1024)
            for index in range(len(canvases)):
                kidney_fraction = old_patch.grid_coverage(kidneys[index])
                artifact_fraction = old_patch.grid_coverage(artifacts[index])
                selected = (kidney_fraction >= .30) & (artifact_fraction <= .05)
                if not selected.any():
                    selected.flat[np.argmax(kidney_fraction - artifact_fraction)] = True
                values = maps[index][selected].astype(np.float16)
                flat.append(values)
                offsets.append(offsets[-1] + len(values))
                excluded += int(((kidney_fraction >= .30) & (artifact_fraction > .05)).sum())
            print(f"Local patches {min(start + 32, len(ids))}/{len(ids)}", flush=True)
    features = np.concatenate(flat)
    assert len(offsets) == len(ids) + 1 and features.shape[1] == 1024
    np.save(features_path, features)
    np.save(offsets_path, np.asarray(offsets, np.int64))
    return {"files_sha256": {path.name: digest(path) for path in (features_path, offsets_path)},
            "images": len(ids), "patches": len(features), "excluded_cyan_cells": excluded}


def encode_texture(ids):
    target = OUT / "directional_features.csv"
    if target.exists():
        raise FileExistsError(target)
    sys.path.insert(0, str(base.ROOT / "02_experiments/CYCH_honest_reevaluation_20261003"))
    from extract_site import run_pool

    jobs = [
        (image_id, CACHE / "percentile" / f"{image_id}.png", CACHE / "masks" / f"{image_id}.png")
        for image_id in ids
    ]
    frame = run_pool(old_directional.worker, jobs, processes=8)
    assert len(frame) == len(ids) and frame.index.tolist() == ids
    assert frame.shape[1] == 51
    frame.to_csv(target)
    return {"file": target.name, "sha256": digest(target), "images": len(ids),
            "feature_columns": frame.shape[1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=("tile", "usfm", "reflection", "patches", "texture"))
    args = parser.parse_args()
    frame = manifest()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoders = {"tile": encode_tiles, "usfm": encode_usfm,
                "reflection": encode_reflection, "patches": encode_patches,
                "texture": encode_texture}
    result = encoders[args.branch](frame.image_id.tolist())
    result.update({"branch": args.branch, "manifest_sha256": digest(REGISTRY / "contour_image_manifest.csv"),
                   "labels_used": [], "test_scored": False})
    (OUT / f"{args.branch}_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
