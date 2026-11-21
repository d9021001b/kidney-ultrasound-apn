"""Hold masking and resizing constant in the kidney tile comparison."""
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nckuh_dmsa_native_tile_probe as tile
from nckuh_dmsa_cluster_ci import cluster_intervals

BASE = tile.base
OUT = BASE.ROOT / "02_experiments/NCKUH_DMSA_matched_tile_control_20261103"


def encode(ids):
    sys.path.insert(0, str(BASE.RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    encoder = radimagenet_densenet121().cuda().eval().requires_grad_(False)

    def embed(canvases):
        batch = torch.from_numpy(np.stack(canvases).transpose(0, 3, 1, 2).copy()).cuda().float()
        return torch.cat([encoder(x / 127.5 - 1) for x in batch.split(32)]).cpu().numpy()

    whole, local, offsets = [], [], [0]
    with torch.inference_mode():
        for start in range(0, len(ids), 16):
            whole_batch, tile_batch = [], []
            for ident in ids[start:start + 16]:
                rgb = cv2.cvtColor(cv2.imread(str(BASE.OUT / "clean_rgb" / f"{ident}.png")), cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(BASE.OUT / "masks" / f"{ident}.png"), 0) > 0
                whole_batch.append(BASE.tight_canvas(rgb * mask[..., None], mask)[0])
                local_tiles, _ = tile.tiles(rgb, mask)
                tile_batch.extend(local_tiles)
                offsets.append(offsets[-1] + len(local_tiles))
            whole.append(embed(whole_batch))
            local.append(embed(tile_batch))
    return np.concatenate(whole), np.concatenate(local), np.asarray(offsets, dtype=np.int64)


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, kidneys, images = tile.load_reference()
    coverage = pd.read_csv(tile.OUT / "encoding_coverage.csv")
    np.testing.assert_array_equal(coverage.image_id, images.image_id)
    whole, local, offsets = encode(images.image_id.tolist())
    assert whole.shape == (820, 1024) and local.shape == (3277, 1024)
    assert np.isfinite(whole).all() and np.isfinite(local).all()
    np.testing.assert_array_equal(np.diff(offsets), coverage.tiles)
    means = np.stack([local[a:b].mean(0) for a, b in zip(offsets[:-1], offsets[1:])])
    with np.load(tile.OUT / "tile_features.npz") as saved:
        np.testing.assert_allclose(means, saved["mean"], atol=1e-4, rtol=1e-4)

    old = pd.read_csv(tile.OUT / "development_predictions.csv")
    old = old[old.group.isin(["aligned_fusion", "tile_mean", "tile_mean_fusion"])]
    assignments = pd.read_csv(BASE.OUT / "patient_assignments.csv")
    records = [old]
    y = images.kidney_APN.to_numpy(int)
    for seed in BASE.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train = np.flatnonzero(images.patient_alias.map(mapping).eq("train"))
        dev = np.flatnonzero(images.patient_alias.map(mapping).eq("development"))
        target = kidneys[kidneys.patient_alias.map(mapping).eq("development")]
        assert target.patient_alias.nunique() == 42 and images.iloc[train].patient_alias.nunique() == 126
        ref = old[old.seed.eq(seed) & old.group.eq("aligned_fusion")].set_index("kidney_alias").loc[target.kidney_alias]
        np.testing.assert_array_equal(ref.patient_alias, target.patient_alias)
        np.testing.assert_array_equal(ref.label, target.kidney_APN)
        model = BASE.fit_image_head(whole, train, y, BASE.image_weights(images, train, y), seed)
        score = BASE.aggregate_image_logits(model.predict_proba(whole[dev])[:, 1],
                                           images.iloc[dev], target.kidney_alias)
        for group, s in (("matched_whole", score),
                         ("matched_whole_fusion", .75 * ref.score.to_numpy() + .25 * score)):
            records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=group, label=target.kidney_APN.astype(int), score=s))
    pred = pd.concat(records, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), BASE.SEEDS, "matched_whole")
    _, fusion, _ = cluster_intervals(pred, ["matched_whole_fusion", "tile_mean_fusion"],
                                     BASE.SEEDS, "matched_whole_fusion")
    OUT.mkdir()
    np.savez_compressed(OUT / "matched_embeddings.npz", whole=whole, tiles=local,
                        offsets=offsets, image_ids=images.image_id.to_numpy(dtype=str))
    pred.to_csv(OUT / "development_predictions.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    pd.concat([delta, fusion]).to_csv(OUT / "paired_delta_ci.csv", index=False)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
