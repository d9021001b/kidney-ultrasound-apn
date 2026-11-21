"""Frozen USF-MAE and enhancement ablation on all source-labeled kidneys."""
import json

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_fullcohort_radimagenet_finetune_20261103 import load_reference
from nckuh_dmsa_inductive_graph import OUT as GRAPH
from nckuh_dmsa_mask_enhancement_20261102 import nearest_extension
from nckuh_ultrasound_probe_20261102 import WEIGHT

OUT = base.ROOT / "02_experiments/NCKUH_DMSA_fullcohort_usf_20261103"
ARMS = ("usf_gray", "usf_bilateral")


def encode(ids):
    state = torch.load(WEIGHT, map_location="cpu", weights_only=True)
    encoder = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    keys = set(encoder.state_dict())
    assert len(keys) == 150 and keys <= state.keys()
    encoder.load_state_dict({k: state[k] for k in keys}, strict=True)
    encoder.cuda().eval().requires_grad_(False)
    mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1)
    sd = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1)
    features = {arm: [] for arm in ARMS}
    fallback = 0
    with torch.inference_mode():
        for start in range(0, len(ids), 32):
            batches, masks = [[], []], []
            for ident in ids[start:start + 32]:
                gray = cv2.imread(str(base.OUT / "percentile" / f"{ident}.png"), cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(str(base.OUT / "masks" / f"{ident}.png"), cv2.IMREAD_GRAYSCALE) > 0
                smooth = cv2.bilateralFilter(nearest_extension(gray, mask), 5, 25, 3)
                smooth[~mask] = 0
                for batch, image in zip(batches, (gray, smooth)):
                    canvas, resized = base.tight_canvas(image, mask)
                    batch.append(canvas)
                masks.append(resized)
            coverage = np.stack(masks).reshape(-1, 14, 16, 14, 16).mean((2, 4)).reshape(-1, 196)
            chosen = coverage >= .8
            empty = np.flatnonzero(~chosen.any(1))
            chosen[empty, coverage[empty].argmax(1)] = True
            fallback += len(empty)
            selected = torch.from_numpy(chosen).cuda()
            for arm, batch in zip(ARMS, batches):
                x = torch.from_numpy(np.stack(batch).transpose(0, 3, 1, 2).copy()).cuda().float() / 255
                tokens = encoder.forward_features((x - mean) / sd)
                assert tokens.shape[1:] == (197, 768)
                local = (tokens[:, 1:] * selected[..., None]).sum(1) / selected.sum(1)[:, None]
                features[arm].append(torch.cat([tokens[:, 0], local], dim=1).cpu().numpy())
    return {arm: np.concatenate(a) for arm, a in features.items()}, fallback


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, kidneys, images = load_reference()
    manifest = pd.read_csv(base.OUT / "deidentified_source_manifest.csv")
    np.testing.assert_array_equal(manifest.image_id, images.image_id)
    arrays, fallback = encode(images.image_id)
    assert all(a.shape == (820, 1536) and np.isfinite(a).all() for a in arrays.values())
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    ref = pd.read_csv(GRAPH / "development_predictions.csv")
    ref = ref[ref.group.eq("knn_fusion")].set_index(["seed", "kidney_alias"])
    y = images.kidney_APN.to_numpy(int)
    records = []
    for seed in base.SEEDS:
        role = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train = np.flatnonzero(images.patient_alias.map(role).eq("train"))
        dev = np.flatnonzero(images.patient_alias.map(role).eq("development"))
        target = kidneys[kidneys.patient_alias.map(role).eq("development")]
        assert target.patient_alias.nunique() == 42 and images.iloc[train].patient_alias.nunique() == 126
        fixed = ref.loc[[(seed, a) for a in target.kidney_alias]]
        np.testing.assert_array_equal(fixed.label, target.kidney_APN)
        np.testing.assert_array_equal(fixed.patient_alias, target.patient_alias)
        scores = {"fixed_reference": fixed.score.to_numpy()}
        weights = base.image_weights(images, train, y)
        for arm, x in arrays.items():
            model = base.fit_image_head(x, train, y, weights, seed)
            scores[arm] = base.aggregate_image_logits(model.predict_proba(x[dev])[:, 1],
                                                     images.iloc[dev], target.kidney_alias)
            scores[arm + "_fusion"] = .75 * scores["fixed_reference"] + .25 * scores[arm]
        for group, score in scores.items():
            records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=group, label=target.kidney_APN.astype(int), score=score))
    pred = pd.concat(records, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), base.SEEDS, "fixed_reference")
    _, filter_delta, _ = cluster_intervals(pred, list(ARMS), base.SEEDS, "usf_gray")
    OUT.mkdir()
    pred.to_csv(OUT / "development_predictions.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    pd.concat([delta, filter_delta]).to_csv(OUT / "paired_delta_ci.csv", index=False)
    for arm, values in arrays.items():
        np.save(OUT / (arm + ".npy"), values)
    (OUT / "encoder_audit.json").write_text(json.dumps(
        dict(images=len(images), weight_sha256=base.digest(WEIGHT), patch_fallbacks=fallback,
             labels_used_in_encoding=[], test_predictions=0), indent=2), encoding="utf-8")
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
