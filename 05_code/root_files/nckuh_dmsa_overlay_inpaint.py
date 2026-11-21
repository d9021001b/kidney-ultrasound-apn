"""Test conservative UI-overlay inpainting before the strongest tile pipeline."""

import sys

import cv2
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nckuh_dmsa_tile_replacement as replacement
from nckuh_dmsa_cluster_ci import cluster_intervals


BASE = replacement.BASE
TILE = replacement.tile
OUT = BASE.ROOT / "02_experiments/NCKUH_DMSA_overlay_inpaint_20261104"


def overlay_masks(bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return conservative cyan UI and boundary-red contour masks.

    Interior red/blue pixels are retained because they may be Doppler signal.
    The detector is label-free and only targets conspicuous supplied UI colours.
    """
    binary = np.asarray(mask, dtype=bool)
    hue, saturation, value = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    chromatic = (saturation >= 80) & (value >= 100)
    cyan = binary & chromatic & (hue >= 70) & (hue <= 100)
    eroded = cv2.erode(binary.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0
    boundary = binary & ~eroded
    red = binary & boundary & chromatic & ((hue <= 12) | (hue >= 165))
    artifact = cv2.dilate((cyan | red).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    return artifact & binary, cyan, red


def clean_image(bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    artifact, cyan, red = overlay_masks(bgr, mask)
    cleaned = cv2.inpaint(bgr, artifact.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    denominator = max(int(np.asarray(mask, bool).sum()), 1)
    return cleaned, {
        "ui_pixels": int(artifact.sum()),
        "ui_fraction": float(artifact.sum() / denominator),
        "cyan_pixels": int(cyan.sum()),
        "boundary_red_pixels": int(red.sum()),
    }


def encode(ids: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    sys.path.insert(0, str(BASE.RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    encoder = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    means, audits = [], []
    with torch.inference_mode():
        for start in range(0, len(ids), 16):
            canvases, lengths = [], []
            for identifier in ids[start:start + 16]:
                bgr = cv2.imread(str(BASE.OUT / "clean_rgb" / f"{identifier}.png"), cv2.IMREAD_COLOR)
                mask = cv2.imread(str(BASE.OUT / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE) > 0
                cleaned, audit = clean_image(bgr, mask)
                current, fallback = TILE.tiles(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB), mask)
                audits.append({"image_id": identifier, "tiles": len(current), "fallback": fallback, **audit})
                canvases.extend(current)
                lengths.append(len(current))
            x = torch.from_numpy(np.stack(canvases).transpose(0, 3, 1, 2).copy()).cuda().float() / 127.5 - 1
            values = torch.cat([encoder(part) for part in x.split(32)]).cpu().numpy()
            means.extend(image.mean(0) for image in np.split(values, np.cumsum(lengths)[:-1]))
            print(f"Encoded inpainted tiles {min(start + 16, len(ids))}/{len(ids)}", flush=True)
    return np.asarray(means, np.float32), pd.DataFrame(audits)


def score_features(features, patients, kidneys, images, assignments, sources, seed):
    y_kidney = kidneys.kidney_APN.to_numpy(int)
    y_image = images.kidney_APN.to_numpy(int)
    mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
    kidney_roles = kidneys.patient_alias.map(mapping)
    image_roles = images.patient_alias.map(mapping)
    train_kidney = np.flatnonzero(kidney_roles.eq("train"))
    development_kidney = np.flatnonzero(kidney_roles.eq("development"))
    train_image = np.flatnonzero(image_roles.eq("train"))
    development_image = np.flatnonzero(image_roles.eq("development"))
    target = kidneys.iloc[development_kidney]
    assert target.patient_alias.nunique() == 42 and kidneys.iloc[train_kidney].patient_alias.nunique() == 126

    oriented = BASE.orient_kidneys(BASE.patient_pair_features(features, images, patients), patients, kidneys)
    direct, _ = BASE.fit_deep(oriented, train_kidney, development_kidney, y_kidney, seed, 32)
    model = BASE.fit_image_head(features, train_image, y_image, BASE.image_weights(images, train_image, y_image), seed)
    image_score = BASE.aggregate_image_logits(
        model.predict_proba(features[development_image])[:, 1],
        images.iloc[development_image], target.kidney_alias)

    index = [(seed, alias) for alias in target.kidney_alias]
    selected = {name: frame.loc[index] for name, frame in sources.items()}
    for frame in selected.values():
        np.testing.assert_array_equal(frame.label, y_kidney[development_kidney])
        np.testing.assert_array_equal(frame.patient_alias, target.patient_alias)
    q, k, t = (selected[name].score.to_numpy() for name in ("cross_attention_q4", "knn", "aligned"))
    return target, replacement.fixed_chain((image_score + direct) / 2, q, k, t)


def run() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    patients, kidneys, images = TILE.load_reference()
    ids = images.image_id.tolist()
    original = np.load(TILE.OUT / "tile_features.npz")["mean"]
    cleaned, audit = encode(ids)
    assert cleaned.shape == original.shape == (820, 1024)
    assert np.isfinite(cleaned).all()
    candidates = {
        "original_tile_two_heads_recomputed": original,
        "inpainted_tile_two_heads": cleaned,
        "original_inpainted_embedding_average": (original + cleaned) / 2,
    }
    assignments = pd.read_csv(BASE.OUT / "patient_assignments.csv")
    paths = {
        "cross_attention_q4": replacement.audit.MIL,
        "knn": replacement.graph.OUT,
        "aligned": replacement.directional.OUT,
    }
    sources = {}
    for name, folder in paths.items():
        frame = pd.read_csv(folder / "development_predictions.csv")
        sources[name] = frame[frame.group.eq(name)].set_index(["seed", "kidney_alias"], verify_integrity=True)
    saved = pd.read_csv(replacement.OUT / "development_predictions.csv")
    saved = saved[saved.group.eq("replace_with_tile_two_heads")].set_index(["seed", "kidney_alias"], verify_integrity=True)

    records = []
    for seed in BASE.SEEDS:
        for group, features in candidates.items():
            target, score = score_features(features, patients, kidneys, images, assignments, sources, seed)
            if group == "original_tile_two_heads_recomputed":
                expected = saved.loc[[(seed, alias) for alias in target.kidney_alias]]
                np.testing.assert_allclose(score, expected.score, atol=1e-10, rtol=0)
            records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=group, label=target.kidney_APN.astype(int), score=score))
    prediction = pd.concat(records, ignore_index=True)
    groups = list(candidates)
    ci, delta, _ = cluster_intervals(prediction, groups, BASE.SEEDS, groups[0])
    metrics = pd.DataFrame([
        dict(seed=seed, group=group, **BASE.metric(part.label, part.score))
        for (seed, group), part in prediction.groupby(["seed", "group"])
    ])
    OUT.mkdir()
    np.save(OUT / "radimagenet_inpainted_tile_mean.npy", cleaned)
    audit.to_csv(OUT / "overlay_audit.csv", index=False)
    prediction.to_csv(OUT / "development_predictions.csv", index=False)
    metrics.to_csv(OUT / "development_metrics.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    print(audit[["ui_pixels", "ui_fraction", "cyan_pixels", "boundary_red_pixels"]].describe().to_string())
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
