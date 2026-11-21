"""Frozen EchoCare hierarchical ultrasound representations on NCKUH DMSA."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

import nckuh_dmsa_tile_replacement as replacement
from nckuh_dmsa_cluster_ci import cluster_intervals


BASE = replacement.BASE
OUT = BASE.ROOT / "02_experiments/NCKUH_DMSA_echocare_frozen_20261104"
MODEL_ROOT = BASE.ROOT / "03_data_models/external_models/EchoCare_official_20261104"
VENDOR = MODEL_ROOT / "vendor"
CHECKPOINT = MODEL_ROOT / "echocare_encoder.pth"
CHECKPOINT_BYTES = 481_812_900
CHECKPOINT_SHA256 = "f9039c62fcab62cfbedcdd9fe849b9acd7edaa740c0acac903573931680c5883"
SOURCE_COMMIT = "aad463832ccf51b8d55ece302832fa8b7abb19c7"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def model_digest(model: torch.nn.Module) -> str:
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def construct_model():
    sys.path.insert(0, str(VENDOR))
    from monai.networks.nets.swin_unetr import SwinTransformer

    model = SwinTransformer(
        in_chans=3,
        embed_dim=128,
        window_size=[8, 8],
        patch_size=[2, 2],
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        spatial_dims=2,
        use_v2=True,
    )
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    mask_token = state.pop("mask_token")
    model.load_state_dict(state, strict=True)
    return model, len(state), tuple(mask_token.shape)


def summarize(features: list[torch.Tensor], masks: torch.Tensor):
    expected_channels = (128, 256, 512, 1024, 2048)
    if len(features) != len(expected_channels) or masks.shape[1:] != (256, 256):
        raise ValueError("Unexpected EchoCare features or masks")
    final_gap = features[-1].mean((2, 3))
    moments, weight_sums = [], []
    for feature, channels in zip(features, expected_channels):
        if feature.shape[1] != channels:
            raise ValueError("Unexpected EchoCare channel count")
        weight = F.interpolate(masks[:, None], size=feature.shape[-2:], mode="area")
        denominator = weight.sum((2, 3)).clamp_min(1e-6)
        mean = (feature * weight).sum((2, 3)) / denominator
        variance = ((feature - mean[:, :, None, None]).square() * weight).sum((2, 3)) / denominator
        moments.extend((mean, variance.clamp_min(0).sqrt()))
        weight_sums.extend(denominator[:, 0].detach().cpu().tolist())
    multiscale = torch.cat(moments, dim=1)
    return (
        final_gap.detach().cpu().numpy().astype(np.float32),
        multiscale.detach().cpu().numpy().astype(np.float32),
        np.asarray(weight_sums, dtype=np.float32),
    )


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if CHECKPOINT.stat().st_size != CHECKPOINT_BYTES or digest(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("EchoCare checkpoint integrity mismatch")
    OUT.mkdir()
    write_json(OUT / "protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Official EchoCare hierarchical SwinTransformer encoder with five 2D feature scales",
        "source": "https://github.com/CAIR-HKISI/EchoCare",
        "source_commit": SOURCE_COMMIT,
        "paper": "https://arxiv.org/abs/2509.11752",
        "license": "No repository-level license was found; research evaluation only pending author clarification",
        "checkpoint_source": "Official public SharePoint link in the upstream README",
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "declared_pretraining": "Self-supervised learning on 4.5 million ultrasound images from 23 clinical centers and 38 devices",
        "cohort": "Unchanged 210 patients, 414 kidneys and 820 contour-first images",
        "input": "256x256 tight masked RGB kidney canvas scaled to 0-1, matching the official ToTensor preprocessing without ImageNet normalization",
        "representations": {
            "echocare_final_gap": "Global average of the deepest 2048-channel feature map, matching the official hierarchical classification head input",
            "echocare_multiscale_roi": "Soft kidney-mask-weighted mean and standard deviation from all five normalized feature scales (7936 dimensions)",
        },
        "head": "Frozen encoder; unchanged PCA32 logistic image and ordered-patient-context heads, equal probability average",
        "fusion": "Replace only the deep branch of the unchanged fixed chain; equal fixed-reference/EchoCare average is a prespecified control",
        "roles": "Three fixed 126-patient training and 42-patient development roles; exploratory-test role unscored",
        "selection": "No development-selected layer, mask threshold, head, component count or fusion weight",
        "target": "Development AUROC and AP both above 0.8 before locked evaluation",
    })
    write_json(OUT / "run_lock.json", {
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "checkpoint_sha256": digest(CHECKPOINT),
    })
    print("Registered frozen EchoCare experiment", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(Path(__file__)) == lock["script_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    assert digest(CHECKPOINT) == lock["checkpoint_sha256"] == CHECKPOINT_SHA256


def encode() -> None:
    require_lock()
    if (OUT / "echocare_final_gap.npy").exists():
        raise FileExistsError("Preserve completed embeddings")
    torch.manual_seed(20261104)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, loaded_tensors, ignored_mask_shape = construct_model()
    model = model.cuda().eval().requires_grad_(False)
    before = model_digest(model)
    _, _, images = replacement.tile.load_reference()
    gap_rows, multiscale_rows, weight_sums = [], [], []
    repeat_maxabs = batch_maxabs = None
    with torch.inference_mode():
        for start in range(0, len(images), 8):
            canvases, masks = [], []
            for identifier in images.image_id.iloc[start:start + 8]:
                bgr = cv2.imread(str(BASE.OUT / "clean_rgb" / f"{identifier}.png"), cv2.IMREAD_COLOR)
                mask = cv2.imread(str(BASE.OUT / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE) > 0
                canvas, canvas_mask = BASE.tight_canvas(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask, size=256)
                canvases.append(canvas)
                masks.append(canvas_mask.astype(np.float32))
            tensor = torch.from_numpy(np.stack(canvases).transpose(0, 3, 1, 2).copy()).cuda().float() / 255
            mask_tensor = torch.from_numpy(np.stack(masks)).cuda().float()
            features = model(tensor)
            if start == 0:
                repeated = model(tensor)
                single = model(tensor[:1])
                repeat_maxabs = max(float((a - b).abs().max()) for a, b in zip(features, repeated))
                batch_maxabs = max(float((a[:1] - b).abs().max()) for a, b in zip(features, single))
                assert max(repeat_maxabs, batch_maxabs) <= 1e-4
            gap, multiscale, sums = summarize(features, mask_tensor)
            gap_rows.append(gap)
            multiscale_rows.append(multiscale)
            weight_sums.extend(sums.tolist())
            print(f"EchoCare encoded {min(start + 8, len(images))}/{len(images)}", flush=True)
    gap = np.concatenate(gap_rows)
    multiscale = np.concatenate(multiscale_rows)
    assert gap.shape == (820, 2048) and multiscale.shape == (820, 7936)
    assert np.isfinite(gap).all() and np.isfinite(multiscale).all()
    after = model_digest(model)
    assert before == after
    np.save(OUT / "echocare_final_gap.npy", gap)
    np.save(OUT / "echocare_multiscale_roi.npy", multiscale)
    write_json(OUT / "encoder_audit.json", {
        "images": 820,
        "loaded_tensors": loaded_tensors,
        "ignored_pretraining_mask_token_shape": list(ignored_mask_shape),
        "other_missing_or_unexpected_tensors": [],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "model_state_before": before,
        "model_state_after": after,
        "repeat_maxabs": repeat_maxabs,
        "batch_maxabs": batch_maxabs,
        "mask_weight_sum_min_median_max_across_all_scales": [
            min(weight_sums), float(np.median(weight_sums)), max(weight_sums)],
        "labels_used_in_encoding": [],
        "frozen": True,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "monai": "1.5.1",
    })
    print("Saved frozen EchoCare embeddings", flush=True)


def evaluate() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed predictions")
    patients, kidneys, images = replacement.tile.load_reference()
    assignments = pd.read_csv(BASE.OUT / "patient_assignments.csv")
    y_kidney = kidneys.kidney_APN.to_numpy(int)
    y_image = images.kidney_APN.to_numpy(int)
    arrays = {
        "echocare_final_gap": np.load(OUT / "echocare_final_gap.npy"),
        "echocare_multiscale_roi": np.load(OUT / "echocare_multiscale_roi.npy"),
    }
    oriented = {
        name: BASE.orient_kidneys(BASE.patient_pair_features(values, images, patients), patients, kidneys)
        for name, values in arrays.items()
    }
    sources = {}
    for name, folder in {
        "cross_attention_q4": replacement.audit.MIL,
        "knn": replacement.graph.OUT,
        "aligned": replacement.directional.OUT,
    }.items():
        frame = pd.read_csv(folder / "development_predictions.csv")
        sources[name] = frame[frame.group.eq(name)].set_index(["seed", "kidney_alias"], verify_integrity=True)
    fixed = pd.read_csv(replacement.OUT / "development_predictions.csv")
    fixed = fixed[fixed.group.eq("replace_with_tile_two_heads")].set_index(
        ["seed", "kidney_alias"], verify_integrity=True)
    records = []
    for seed in BASE.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        kidney_train = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("train"))
        kidney_dev = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        image_train = np.flatnonzero(images.patient_alias.map(mapping).eq("train"))
        image_dev = np.flatnonzero(images.patient_alias.map(mapping).eq("development"))
        target = kidneys.iloc[kidney_dev]
        index = [(seed, alias) for alias in target.kidney_alias]
        q, k, t = (sources[name].loc[index].score.to_numpy()
                   for name in ("cross_attention_q4", "knn", "aligned"))
        fixed_score = fixed.loc[index].score.to_numpy()
        scores = {"fixed_chain_reference": fixed_score}
        for name, values in arrays.items():
            image_model = BASE.fit_image_head(
                values, image_train, y_image, BASE.image_weights(images, image_train, y_image), seed)
            image_score = BASE.aggregate_image_logits(
                image_model.predict_proba(values[image_dev])[:, 1], images.iloc[image_dev], target.kidney_alias)
            context_score, _ = BASE.fit_deep(
                oriented[name], kidney_train, kidney_dev, y_kidney, seed, 32)
            two_head = (image_score + context_score) / 2
            scores[name + "_two_heads"] = two_head
            scores["replace_deep_" + name] = replacement.fixed_chain(two_head, q, k, t)
            scores["fixed_plus_" + name + "_equal"] = (fixed_score + two_head) / 2
        for group, score in scores.items():
            records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=group, label=y_kidney[kidney_dev], score=score))
        print(f"Completed EchoCare development draw {seed}", flush=True)
    prediction = pd.concat(records, ignore_index=True)
    groups = list(prediction.group.unique())
    ci, delta, _ = cluster_intervals(prediction, groups, BASE.SEEDS, "fixed_chain_reference")
    metrics = pd.DataFrame([
        dict(seed=seed, group=group, **BASE.metric(part.label, part.score))
        for (seed, group), part in prediction.groupby(["seed", "group"])
    ])
    prediction.to_csv(OUT / "development_predictions.csv", index=False)
    metrics.to_csv(OUT / "development_metrics.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    summary = ci.pivot(index="group", columns="metric", values="mean")
    write_json(OUT / "development_decision.json", {
        "groups_above_both_targets": summary[(summary.auroc > .8) & (summary.ap > .8)].index.tolist(),
        "exploratory_test_scored": False,
    })
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "encode", "evaluate"))
    with threadpool_limits(limits=4):
        globals()[parser.parse_args().stage]()
