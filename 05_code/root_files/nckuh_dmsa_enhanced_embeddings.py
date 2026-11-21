"""Development-only comparison of contrast and speckle processing before encoding."""
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_fullcohort_radimagenet_finetune_20261103 import load_reference
from nckuh_dmsa_mask_enhancement_20261102 import enhancement_variants

OUT = base.ROOT / "02_experiments/NCKUH_DMSA_enhanced_embeddings_20261103"
ARMS = ("base", "clahe", "bilateral")


def encode(ids):
    sys.path.insert(0, str(base.RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    model = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    arrays = {arm: [] for arm in ARMS}
    with torch.inference_mode():
        for start in range(0, len(ids), 32):
            batch = {arm: [] for arm in ARMS}
            for ident in ids[start:start+32]:
                rgb = cv2.imread(str(base.OUT / "clean_rgb" / f"{ident}.png"))
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                mask = cv2.imread(str(base.OUT / "masks" / f"{ident}.png"), cv2.IMREAD_GRAYSCALE) > 0
                variants = enhancement_variants(gray, mask)
                for arm in ARMS:
                    canvas, _ = base.tight_canvas(variants[arm], mask)
                    batch[arm].append(canvas)
            for arm in ARMS:
                x = torch.from_numpy(np.stack(batch[arm]).transpose(0, 3, 1, 2).copy()).cuda().float()
                x = x / 127.5 - 1
                arrays[arm].append(model(x).cpu().numpy())
    return {arm: np.concatenate(chunks) for arm, chunks in arrays.items()}


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, kidneys, images = load_reference()
    manifest = pd.read_csv(base.OUT / "deidentified_source_manifest.csv")
    assert images.image_id.tolist() == manifest.image_id.tolist()
    arrays = encode(images.image_id.tolist())
    np.testing.assert_allclose(arrays["base"], np.load(base.OUT / "radimagenet_gray.npy"), atol=1e-4, rtol=1e-4)
    y = images.kidney_APN.to_numpy(int)
    roles = pd.read_csv(base.OUT / "patient_assignments.csv")
    reference = pd.read_csv(base.OUT / "instance_development_predictions.csv")
    reference = reference[reference.group.eq(base.INSTANCE_GROUPS[4])].set_index(["seed", "kidney_alias"])
    records = []
    for seed in base.SEEDS:
        mapping = roles[roles.seed.eq(seed)].set_index("patient_alias").role
        train = np.flatnonzero(images.patient_alias.map(mapping).eq("train"))
        dev = np.flatnonzero(images.patient_alias.map(mapping).eq("development"))
        target = kidneys[kidneys.patient_alias.map(mapping).eq("development")]
        assert target.patient_alias.nunique() == 42
        assert images.iloc[train].patient_alias.nunique() == 126
        ref = reference.loc[[(seed, alias) for alias in target.kidney_alias]]
        np.testing.assert_array_equal(ref.label, target.kidney_APN)
        groups = {"fixed_color_reference": ref.score.to_numpy()}
        for arm, features in arrays.items():
            model = base.fit_image_head(features, train, y, base.image_weights(images, train, y), seed)
            groups[arm] = base.aggregate_image_logits(model.predict_proba(features[dev])[:, 1],
                                                      images.iloc[dev], target.kidney_alias)
            groups[arm + "_fusion"] = .25 * groups[arm] + .75 * groups["fixed_color_reference"]
        for group, score in groups.items():
            records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=group, label=target.kidney_APN.astype(int), score=score))
    pred = pd.concat(records, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), base.SEEDS, "fixed_color_reference")
    _, head_delta, _ = cluster_intervals(pred, list(ARMS), base.SEEDS, "base")
    OUT.mkdir()
    for name, frame in (("development_predictions", pred), ("development_ci", ci),
                        ("paired_reference_delta_ci", delta), ("paired_gray_delta_ci", head_delta)):
        frame.to_csv(OUT / f"{name}.csv", index=False)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
