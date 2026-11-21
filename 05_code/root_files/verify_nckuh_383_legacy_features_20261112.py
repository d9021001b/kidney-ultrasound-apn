"""Validate old and revised frozen features on identical source ROIs."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
OLD = EXP / "NCKUH_DMSA_fullcohort_fullframe_20261103"
OLD_REFLECTION = EXP / "NCKUH_DMSA_reflection_probe_20261103"
OLD_PATCH = EXP / "NCKUH_DMSA_fullcohort_parenchyma_mil_20261103"
OLD_TEXTURE = EXP / "NCKUH_DMSA_fullcohort_directional_20261103"
NEW = EXP / "NCKUH_revised_383_audit_20261111"
CACHE = EXP / "NCKUH_revised_383_contour_cache_20261111"
FEATURES = EXP / "NCKUH_revised_383_legacy_features_20261112"
OUT = EXP / "NCKUH_revised_383_root_cause_20261112"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.digest()


def main():
    old_ids = [row["image_id"] for row in read_csv(OLD / "deidentified_source_manifest.csv")]
    new_images = read_csv(NEW / "contour_image_manifest.csv")
    new_ids = [row["image_id"] for row in new_images]
    old_pos = {key: index for index, key in enumerate(old_ids)}
    new_pos = {key: index for index, key in enumerate(new_ids)}
    new_by_id = {row["image_id"]: row for row in new_images}
    lookup = defaultdict(list)
    for row in read_csv(NEW / "source_linkage_private.csv"):
        image = new_by_id.get(row["image_id"])
        if image is not None:
            lookup[row["patient_id"], row["exam_date_raw"], image["view"]].append(row["image_id"])
    paired = []
    for row in read_csv(EXP / "NCKUH_DMSA_label_audit_20261102/image_labels_dmsa_private.csv"):
        old_id = row["image_id"]
        if old_id not in old_pos:
            continue
        keys = lookup[row["patient_id"], row["exam_date_raw"], row["view"]]
        if len(keys) != 1:
            continue
        new_id = keys[0]
        if digest(OLD / "clean_rgb" / f"{old_id}.png") != digest(CACHE / "clean_rgb" / f"{new_id}.png"):
            continue
        paired.append((old_id, new_id, old_pos[old_id], new_pos[new_id]))
    assert len(paired) == 786

    vectors = {
        "color": (np.load(OLD / "radimagenet_color.npy", mmap_mode="r"),
                  np.load(FEATURES / "rad_color.npy", mmap_mode="r")),
        "horizontal_reflection": (
            np.load(OLD_REFLECTION / "radimagenet_color_horizontal_reflection.npy", mmap_mode="r"),
            np.load(FEATURES / "rad_color_horizontal_reflection.npy", mmap_mode="r"),
        ),
    }
    comparison = {}
    for name, (old, new) in vectors.items():
        differences, similarities = [], []
        for _, _, i, j in paired:
            a, b = old[i].astype(float), new[j].astype(float)
            differences.append(float(np.mean(np.abs(a - b))))
            similarities.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        comparison[name] = {
            "mean_abs_difference_q50_q99_max": np.quantile(differences, [0.5, 0.99, 1]).tolist(),
            "cosine_min_q01_q50": np.quantile(similarities, [0, 0.01, 0.5]).tolist(),
        }

    old_offsets = np.load(OLD_PATCH / "patch_offsets.npy")
    new_offsets = np.load(FEATURES / "patch_offsets.npy")
    old_patch = np.load(OLD_PATCH / "radimagenet_local_float16.npy", mmap_mode="r")
    new_patch = np.load(FEATURES / "radimagenet_local_float16.npy", mmap_mode="r")
    patch_count = Counter()
    patch_abs = []
    for _, _, i, j in paired:
        a = old_patch[old_offsets[i]:old_offsets[i + 1]]
        b = new_patch[new_offsets[j]:new_offsets[j + 1]]
        if a.shape != b.shape:
            patch_count["different_selected_patch_count"] += 1
        else:
            patch_count["same_selected_patch_count"] += 1
            patch_abs.append(float(np.mean(np.abs(a.astype(float) - b.astype(float)))))

    old_texture = pd.read_csv(OLD_TEXTURE / "directional_features.csv").set_index("image_id")
    new_texture = pd.read_csv(FEATURES / "directional_features.csv").set_index("image_id")
    assert old_texture.columns.tolist() == new_texture.columns.tolist()
    texture_abs, texture_rel = [], []
    for old_id, new_id, _, _ in paired:
        a = old_texture.loc[old_id].to_numpy(float)
        b = new_texture.loc[new_id].to_numpy(float)
        texture_abs.append(float(np.max(np.abs(a - b))))
        texture_rel.append(float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1))))
    result = {
        "identical_roi_pairs": len(paired),
        "reflected_embeddings": comparison,
        "local_patch": dict(patch_count) | {
            "mean_absolute_difference_q50_q99_max": np.quantile(patch_abs, [0.5, 0.99, 1]).tolist()
        },
        "directional_texture": {
            "max_absolute_difference_q50_q99_max": np.quantile(texture_abs, [0.5, 0.99, 1]).tolist(),
            "max_relative_difference_q50_q99_max": np.quantile(texture_rel, [0.5, 0.99, 1]).tolist(),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "legacy_feature_reproduction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
