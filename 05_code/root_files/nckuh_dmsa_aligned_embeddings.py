"""Test whether contour-axis normalization transfers to frozen RadImageNet."""
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
from nckuh_dmsa_reflection_probe import OUT as REFLECTION

OUT = base.ROOT / "02_experiments/NCKUH_DMSA_aligned_embeddings_20261103"


def aligned_canvas(color, mask, angle):
    size = int(np.ceil(np.hypot(*mask.shape))) + 8
    y, x = (size - mask.shape[0]) // 2, (size - mask.shape[1]) // 2
    pad = ((y, size - mask.shape[0] - y), (x, size - mask.shape[1] - x))
    masked = np.pad(np.where(mask[..., None], color, 0), (*pad, (0, 0)))
    roi = np.pad(mask.astype(np.uint8), pad)
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.)
    rgb = cv2.warpAffine(masked, matrix, (size, size), flags=cv2.INTER_LINEAR)
    binary = cv2.warpAffine(roi, matrix, (size, size), flags=cv2.INTER_NEAREST)
    rgb[binary == 0] = 0
    return base.tight_canvas(rgb, binary)[0]


def encode(ids, angles):
    sys.path.insert(0, str(base.RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    model = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    output = []
    with torch.inference_mode():
        for start in range(0, len(ids), 32):
            canvases = []
            for ident, angle in zip(ids[start:start + 32], angles[start:start + 32]):
                color = cv2.cvtColor(cv2.imread(str(base.OUT / "clean_rgb" / f"{ident}.png")), cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(base.OUT / "masks" / f"{ident}.png"), cv2.IMREAD_GRAYSCALE) > 0
                canvases.append(aligned_canvas(color, mask, angle))
            x = torch.from_numpy(np.stack(canvases).transpose(0, 3, 1, 2).copy()).cuda().float() / 127.5 - 1
            output.append(((model(x) + model(x.flip((-1,)))) / 2).cpu().numpy())
    return np.concatenate(output)


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    patients, kidneys, images = load_reference()
    angles = pd.read_csv(directional.OUT / "directional_features.csv").set_index("image_id")
    np.testing.assert_array_equal(angles.index, images.image_id)
    aligned = encode(images.image_id.tolist(), angles.alignment_degrees.to_numpy())
    assert aligned.shape == (820, 1024) and np.isfinite(aligned).all()
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    existing = pd.read_csv(REFLECTION / "development_predictions.csv")
    existing = existing[existing.group.eq("reflection_average")]
    reference = pd.read_csv(directional.OUT / "development_predictions.csv")
    reference = reference[reference.group.eq("aligned_fusion")]
    rows = [existing, reference]
    oriented = base.orient_kidneys(base.patient_pair_features(aligned, images, patients), patients, kidneys)
    yi, yk = images.kidney_APN.to_numpy(int), kidneys.kidney_APN.to_numpy(int)
    for seed in base.SEEDS:
        role = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        ti = np.flatnonzero(images.patient_alias.map(role).eq("train"))
        di = np.flatnonzero(images.patient_alias.map(role).eq("development"))
        tk = np.flatnonzero(kidneys.patient_alias.map(role).eq("train"))
        dk = np.flatnonzero(kidneys.patient_alias.map(role).eq("development"))
        target = kidneys.iloc[dk]
        assert target.patient_alias.nunique() == 42 and images.iloc[ti].patient_alias.nunique() == 126
        head = base.fit_image_head(aligned, ti, yi, base.image_weights(images, ti, yi), seed)
        img = base.aggregate_image_logits(head.predict_proba(aligned[di])[:, 1], images.iloc[di], target.kidney_alias)
        direct, _ = base.fit_deep(oriented, tk, dk, yk, seed, 32)
        score = (img + direct) / 2
        ref = reference[reference.seed.eq(seed)].set_index("kidney_alias").loc[target.kidney_alias]
        np.testing.assert_array_equal(ref.label, yk[dk])
        np.testing.assert_array_equal(ref.patient_alias, target.patient_alias)
        for name, s in (("aligned_reflection", score),
                        ("aligned_reflection_fusion", .75 * ref.score.to_numpy() + .25 * score)):
            rows.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=name, label=yk[dk], score=s))
    pred = pd.concat(rows, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), base.SEEDS, "reflection_average")
    _, fusion, _ = cluster_intervals(pred, ["aligned_fusion", "aligned_reflection_fusion"],
                                     base.SEEDS, "aligned_fusion")
    OUT.mkdir()
    np.save(OUT / "aligned_reflection_embeddings.npy", aligned)
    pred.to_csv(OUT / "development_predictions.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    pd.concat([delta, fusion]).to_csv(OUT / "paired_delta_ci.csv", index=False)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
