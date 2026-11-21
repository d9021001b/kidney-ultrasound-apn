"""Frozen official full USFM representations on the NCKUH DMSA cohort."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nckuh_dmsa_tile_replacement as replacement
from nckuh_dmsa_cluster_ci import cluster_intervals


BASE = replacement.BASE
OUT = BASE.ROOT / "02_experiments/NCKUH_DMSA_usfm_frozen_20261104"
MODEL_ROOT = BASE.ROOT / "03_data_models/external_models/USFM_official_20261104"
SOURCE = MODEL_ROOT / "source"
CHECKPOINT = MODEL_ROOT / "USFM_latest.pth"
CHECKPOINT_SHA256 = "d5fdab3edd140e4ca61471bb4087f91cd7ff2ce270db71b9cab30feda881bd17"
SOURCE_COMMIT = "960a1e1d30b9490e53e6e7cf1b4dd24425fffc67"


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
    sys.path.insert(0, str(SOURCE))
    from usdsgen.modules.backbone.vision_transformer import VisionTransformer
    import torch.nn as nn

    model = VisionTransformer(
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0,
        qkv_bias=True, init_values=.1, use_abs_pos_emb=False,
        use_rel_pos_bias=False, use_shared_rel_pos_bias=True,
        use_mean_pooling=False)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert set(state).issuperset({"mask_token", "cls_token", "rel_pos_bias.relative_position_bias_table"})
    mask_token = state.pop("mask_token")
    model.load_state_dict(state, strict=True)
    return model, len(state), tuple(mask_token.shape)


def all_tokens(model, images):
    x = model.patch_embed(images)
    x = torch.cat((model.cls_token.expand(len(x), -1, -1), x), dim=1)
    if model.pos_embed is not None:
        x = x + model.pos_embed
    x = model.pos_drop(x)
    relative = model.rel_pos_bias() if model.rel_pos_bias is not None else None
    for block in model.blocks:
        x = block(x, rel_pos_bias=relative)
    return model.norm(x)


def summarize(tokens: np.ndarray, masks: np.ndarray):
    if tokens.shape[1] != 197 or tokens.shape[2] != 768 or masks.shape[1:] != (224, 224):
        raise ValueError("Unexpected USFM token or mask shape")
    coverage = masks.reshape(-1, 14, 16, 14, 16).mean((2, 4)).reshape(-1, 196)
    gap = tokens[:, 1:].mean(1)
    roi, counts = [], []
    for token, local in zip(tokens[:, 1:], coverage):
        selected = local >= .30
        if not selected.any():
            selected[np.argmax(local)] = True
        values = token[selected]
        roi.append(np.concatenate([values.mean(0), values.std(0)]))
        counts.append(int(selected.sum()))
    return gap.astype(np.float32), np.asarray(roi, np.float32), np.asarray(counts)


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("USFM checkpoint hash mismatch")
    OUT.mkdir()
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Official full USFM ViT-B/16, 12 layers, 768-dimensional tokens",
        "source": "https://github.com/openmedlab/USFM",
        "source_commit": SOURCE_COMMIT,
        "paper": "https://doi.org/10.1016/j.media.2024.103202",
        "license": "Repository declares CC-BY-NC 4.0; research evaluation only",
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "declared_pretraining": "Unsupervised spatial/frequency masked image modeling on about two million multi-organ, multi-center ultrasound images",
        "cohort": "Unchanged 210 patients, 414 kidneys and 820 contour-first images",
        "input": "224x224 tight masked RGB kidney canvas with official ImageNet normalization",
        "representations": {
            "usfm_gap": "Mean of 196 final normalized patch tokens",
            "usfm_roi_heterogeneity": "Mean and standard deviation of final tokens with kidney-mask patch coverage at least 0.30",
        },
        "head": "Frozen encoder; unchanged PCA32 logistic image and ordered-patient-context heads, equal probability average",
        "fusion": "Replace only the deep branch of the unchanged fixed chain; equal fixed-reference/USFM average is a prespecified control",
        "roles": "Three fixed 126-patient training and 42-patient development roles; exploratory-test role unscored",
        "selection": "No development-selected layer, threshold, head, component count or weight",
        "target": "Development AUROC and AP both above 0.8 before locked evaluation",
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "checkpoint_sha256": digest(CHECKPOINT),
    })
    print("Registered frozen full USFM experiment", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(Path(__file__)) == lock["script_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    assert digest(CHECKPOINT) == lock["checkpoint_sha256"] == CHECKPOINT_SHA256


def encode() -> None:
    require_lock()
    if (OUT / "usfm_gap.npy").exists():
        raise FileExistsError("Preserve completed embeddings")
    torch.manual_seed(20261104)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, loaded_tensors, ignored_mask_shape = construct_model()
    model = model.cuda().eval().requires_grad_(False)
    before = model_digest(model)
    _, _, images = replacement.tile.load_reference()
    mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1)
    gap_rows, roi_rows, counts = [], [], []
    repeat_maxabs = batch_maxabs = None
    with torch.inference_mode():
        for start in range(0, len(images), 24):
            canvases, masks = [], []
            for identifier in images.image_id.iloc[start:start + 24]:
                bgr = cv2.imread(str(BASE.OUT / "clean_rgb" / f"{identifier}.png"), cv2.IMREAD_COLOR)
                mask = cv2.imread(str(BASE.OUT / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE) > 0
                canvas, canvas_mask = BASE.tight_canvas(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask)
                canvases.append(canvas); masks.append(canvas_mask.astype(np.float32))
            tensor = torch.from_numpy(np.stack(canvases).transpose(0, 3, 1, 2).copy()).cuda().float() / 255
            tensor = (tensor - mean) / std
            tokens = all_tokens(model, tensor)
            if start == 0:
                repeated = all_tokens(model, tensor)
                single = all_tokens(model, tensor[:1])
                repeat_maxabs = float((tokens - repeated).abs().max())
                batch_maxabs = float((tokens[:1] - single).abs().max())
                assert max(repeat_maxabs, batch_maxabs) <= 1e-4
            gap, roi, selected = summarize(tokens.cpu().numpy(), np.stack(masks))
            gap_rows.append(gap); roi_rows.append(roi); counts.extend(selected.tolist())
            print(f"USFM encoded {min(start + 24, len(images))}/{len(images)}", flush=True)
    gap = np.concatenate(gap_rows); roi = np.concatenate(roi_rows)
    assert gap.shape == (820, 768) and roi.shape == (820, 1536)
    assert np.isfinite(gap).all() and np.isfinite(roi).all()
    after = model_digest(model)
    assert before == after
    np.save(OUT / "usfm_gap.npy", gap)
    np.save(OUT / "usfm_roi_heterogeneity.npy", roi)
    write_json(OUT / "encoder_audit.json", {
        "images": 820, "loaded_tensors": loaded_tensors,
        "ignored_pretraining_mask_token_shape": list(ignored_mask_shape),
        "other_missing_or_unexpected_tensors": [],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "model_state_before": before, "model_state_after": after,
        "repeat_maxabs": repeat_maxabs, "batch_maxabs": batch_maxabs,
        "selected_patch_min_median_max": [min(counts), float(np.median(counts)), max(counts)],
        "labels_used_in_encoding": [], "frozen": True,
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
    })
    print("Saved frozen full USFM embeddings", flush=True)


def evaluate() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed predictions")
    patients, kidneys, images = replacement.tile.load_reference()
    assignments = pd.read_csv(BASE.OUT / "patient_assignments.csv")
    y_kidney = kidneys.kidney_APN.to_numpy(int)
    y_image = images.kidney_APN.to_numpy(int)
    arrays = {
        "usfm_gap": np.load(OUT / "usfm_gap.npy"),
        "usfm_roi_heterogeneity": np.load(OUT / "usfm_roi_heterogeneity.npy"),
    }
    oriented = {name: BASE.orient_kidneys(
        BASE.patient_pair_features(values, images, patients), patients, kidneys)
        for name, values in arrays.items()}
    sources = {}
    for name, folder in {
        "cross_attention_q4": replacement.audit.MIL,
        "knn": replacement.graph.OUT,
        "aligned": replacement.directional.OUT,
    }.items():
        frame = pd.read_csv(folder / "development_predictions.csv")
        sources[name] = frame[frame.group.eq(name)].set_index(["seed", "kidney_alias"], verify_integrity=True)
    fixed = pd.read_csv(replacement.OUT / "development_predictions.csv")
    fixed = fixed[fixed.group.eq("replace_with_tile_two_heads")].set_index(["seed", "kidney_alias"], verify_integrity=True)
    records = []
    for seed in BASE.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        kt = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("train"))
        kd = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        it = np.flatnonzero(images.patient_alias.map(mapping).eq("train"))
        iv = np.flatnonzero(images.patient_alias.map(mapping).eq("development"))
        target = kidneys.iloc[kd]
        index = [(seed, alias) for alias in target.kidney_alias]
        q, k, t = (sources[name].loc[index].score.to_numpy()
                   for name in ("cross_attention_q4", "knn", "aligned"))
        fixed_score = fixed.loc[index].score.to_numpy()
        scores = {"fixed_chain_reference": fixed_score}
        for name, values in arrays.items():
            model = BASE.fit_image_head(values, it, y_image, BASE.image_weights(images, it, y_image), seed)
            image_score = BASE.aggregate_image_logits(
                model.predict_proba(values[iv])[:, 1], images.iloc[iv], target.kidney_alias)
            context_score, _ = BASE.fit_deep(oriented[name], kt, kd, y_kidney, seed, 32)
            two_head = (image_score + context_score) / 2
            replaced = replacement.fixed_chain(two_head, q, k, t)
            scores[name + "_two_heads"] = two_head
            scores["replace_deep_" + name] = replaced
            scores["fixed_plus_" + name + "_equal"] = (fixed_score + two_head) / 2
        for group, score in scores.items():
            records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=group, label=y_kidney[kd], score=score))
        print(f"Completed full USFM development draw {seed}", flush=True)
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
