"""Compare original and reflection-averaged frozen kidney image embeddings."""
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_fullcohort_radimagenet_finetune_20261103 import load_reference

OUT = base.ROOT / "02_experiments/NCKUH_DMSA_reflection_probe_20261103"
GROUPS = ("original", "reflection_average")


def encode(ids, expected):
    sys.path.insert(0, str(base.RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    model = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    output = []
    with torch.inference_mode():
        for start in range(0, len(ids), 32):
            canvases = []
            for ident in ids[start:start + 32]:
                rgb = cv2.cvtColor(cv2.imread(str(base.OUT / "clean_rgb" / f"{ident}.png")), cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(base.OUT / "masks" / f"{ident}.png"), cv2.IMREAD_GRAYSCALE) > 0
                canvas, _ = base.tight_canvas(rgb, mask)
                canvases.append(canvas)
            x = torch.from_numpy(np.stack(canvases).transpose(0, 3, 1, 2).copy()).cuda().float() / 127.5 - 1
            if start == 0:
                np.testing.assert_allclose(model(x).cpu().numpy(), expected[:len(x)], atol=1e-4, rtol=1e-4)
            output.append(model(x.flip((-1,))).cpu().numpy())
    return np.concatenate(output)


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    patients, kidneys, images = load_reference()
    manifest = pd.read_csv(base.OUT / "deidentified_source_manifest.csv")
    assert manifest.image_id.tolist() == images.image_id.tolist()
    roles = pd.read_csv(base.OUT / "patient_assignments.csv")
    saved = pd.read_csv(base.OUT / "instance_development_predictions.csv")
    saved = saved[saved.group.eq(base.INSTANCE_GROUPS[4])].set_index(["seed", "kidney_alias"])
    original = np.load(base.OUT / "radimagenet_color.npy")
    mirrored = encode(images.image_id.tolist(), original)
    assert mirrored.shape == original.shape and np.isfinite(mirrored).all()
    predictions, measurements = [], []
    y_img, y_kid = images.kidney_APN.to_numpy(int), kidneys.kidney_APN.to_numpy(int)
    for group, feats in zip(GROUPS, (original, (original + mirrored) / 2)):
        oriented = base.orient_kidneys(base.patient_pair_features(feats, images, patients), patients, kidneys)
        for seed in base.SEEDS:
            role = roles[roles.seed.eq(seed)].set_index("patient_alias").role
            train_img = np.flatnonzero(images.patient_alias.map(role).eq("train"))
            train = np.flatnonzero(kidneys.patient_alias.map(role).eq("train"))
            dev = np.flatnonzero(kidneys.patient_alias.map(role).eq("development"))
            dev_img = np.flatnonzero(images.patient_alias.map(role).eq("development"))
            assert kidneys.iloc[dev].patient_alias.nunique() == 42
            weights = base.image_weights(images, train_img, y_img)
            model = base.fit_image_head(feats, train_img, y_img, weights, seed)
            alias = kidneys.iloc[dev].kidney_alias
            image_score = base.aggregate_image_logits(model.predict_proba(feats[dev_img])[:, 1],
                                                     images.iloc[dev_img], alias)
            direct, _ = base.fit_deep(oriented, train, dev, y_kid, seed, 32)
            score = (image_score + direct) / 2
            if group == "original":
                np.testing.assert_allclose(score, saved.loc[[(seed, key) for key in alias]].score,
                                           atol=5e-5, rtol=0)
            frame = kidneys.iloc[dev][["patient_alias", "kidney_alias", "DMSA_APN", "side"]].assign(
                seed=seed, group=group, label=y_kid[dev], score=score)
            predictions.append(frame)
            measurements.append(dict(group=group, seed=seed, **base.metric(y_kid[dev], score)))
    frame = pd.concat(predictions, ignore_index=True)
    ci, delta, _ = cluster_intervals(frame, list(GROUPS), base.SEEDS, GROUPS[0])
    OUT.mkdir()
    for name, data in (("development_predictions", frame), ("development_metrics", pd.DataFrame(measurements)),
                       ("development_ci", ci), ("paired_delta_ci", delta)):
        data.to_csv(OUT / f"{name}.csv", index=False)
    np.save(OUT / "radimagenet_color_horizontal_reflection.npy", mirrored)
    print(ci.to_string(index=False))
    print(delta.to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
