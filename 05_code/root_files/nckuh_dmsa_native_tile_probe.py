"""Compare zoomed overlapping kidney crops against whole-frame embeddings."""
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_directional as directional
import nckuh_dmsa_fullcohort_fullframe_20261103 as base
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_fullcohort_radimagenet_finetune_20261103 import load_reference

OUT = base.ROOT / "02_experiments/NCKUH_DMSA_native_tile_probe_20261103"


def tiles(rgb, mask):
    y, x = np.nonzero(mask)
    rgb = rgb[y.min():y.max() + 1, x.min():x.max() + 1]
    mask = mask[y.min():y.max() + 1, x.min():x.max() + 1]
    h, w = mask.shape
    dh, dw = max(1, round(h * 2 / 3)), max(1, round(w * 2 / 3))
    selected = []
    for i, j in sorted({(a, b) for a in (0, h - dh) for b in (0, w - dw)}):
        part = mask[i:i + dh, j:j + dw]
        if part.sum() >= 128 and part.mean() >= .25:
            color = rgb[i:i + dh, j:j + dw]
            selected.append(base.tight_canvas(color * part[..., None], part)[0])
    if not selected:
        return [base.tight_canvas(rgb * mask[..., None], mask)[0]], True
    return selected, False


def encode(ids):
    sys.path.insert(0, str(base.RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    encoder = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    means, deviations, coverage = [], [], []
    with torch.inference_mode():
        for start in range(0, len(ids), 16):
            canvases, lengths = [], []
            for ident in ids[start:start + 16]:
                rgb = cv2.cvtColor(cv2.imread(str(base.OUT / "clean_rgb" / f"{ident}.png")), cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(base.OUT / "masks" / f"{ident}.png"), 0) > 0
                current, fallback = tiles(rgb, mask)
                coverage.append(dict(image_id=ident, tiles=len(current), fallback=fallback))
                canvases.extend(current)
                lengths.append(len(current))
            x = torch.from_numpy(np.stack(canvases).transpose(0, 3, 1, 2).copy()).cuda().float() / 127.5 - 1
            values = torch.cat([encoder(part) for part in x.split(32)]).cpu().numpy()
            for image in np.split(values, np.cumsum(lengths)[:-1]):
                means.append(image.mean(0))
                deviations.append(image.std(0))
    return np.asarray(means), np.asarray(deviations), pd.DataFrame(coverage)


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, kidneys, images = load_reference()
    ids = images.image_id.tolist()
    manifest = pd.read_csv(base.OUT / "deidentified_source_manifest.csv")
    assert ids == manifest.image_id.tolist()
    mean, sd, coverage = encode(ids)
    assert mean.shape == sd.shape == (820, 1024)
    assert np.isfinite(mean).all() and np.isfinite(sd).all()
    inputs = {"whole": np.load(base.OUT / "radimagenet_color.npy"), "tile_mean": mean,
              "tile_mean_sd": np.c_[mean, sd]}
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    reference = pd.read_csv(directional.OUT / "development_predictions.csv")
    reference = reference[reference.group.eq("aligned_fusion")]
    records = [reference]
    y = images.kidney_APN.to_numpy(int)
    for seed in base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train = np.flatnonzero(images.patient_alias.map(mapping).eq("train"))
        dev = np.flatnonzero(images.patient_alias.map(mapping).eq("development"))
        target = kidneys[kidneys.patient_alias.map(mapping).eq("development")]
        assert target.patient_alias.nunique() == 42 and images.iloc[train].patient_alias.nunique() == 126
        ref = reference[reference.seed.eq(seed)].set_index("kidney_alias").loc[target.kidney_alias]
        np.testing.assert_array_equal(ref.label, target.kidney_APN)
        np.testing.assert_array_equal(ref.patient_alias, target.patient_alias)
        weights = base.image_weights(images, train, y)
        for arm, x in inputs.items():
            model = base.fit_image_head(x, train, y, weights, seed)
            score = base.aggregate_image_logits(model.predict_proba(x[dev])[:, 1],
                                               images.iloc[dev], target.kidney_alias)
            for group, s in ((arm, score), (arm + "_fusion", .75 * ref.score.to_numpy() + .25 * score)):
                records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                    seed=seed, group=group, label=target.kidney_APN.astype(int), score=s))
    pred = pd.concat(records, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), base.SEEDS, "aligned_fusion")
    _, control, _ = cluster_intervals(pred, list(inputs), base.SEEDS, "whole")
    OUT.mkdir()
    coverage.to_csv(OUT / "encoding_coverage.csv", index=False)
    np.savez_compressed(OUT / "tile_features.npz", mean=mean, std=sd)
    pred.to_csv(OUT / "development_predictions.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    pd.concat([delta, control]).to_csv(OUT / "paired_delta_ci.csv", index=False)
    print(f"Tiles: {coverage.tiles.sum()}; whole fallback: {coverage.fallback.sum()}")
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
