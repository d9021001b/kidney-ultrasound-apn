"""Frozen UltraFedFM representations on the NCKUH DMSA cohort."""

import argparse
import argparse as argparse_module
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nckuh_dmsa_tile_replacement as replacement
from nckuh_dmsa_cluster_ci import cluster_intervals


BASE = replacement.BASE
OUT = BASE.ROOT / "02_experiments/NCKUH_DMSA_ultrafedfm_frozen_20261104"
MODEL_ROOT = BASE.ROOT / "03_data_models/external_models/UltraFedFM_official_20261104"
CHECKPOINT = MODEL_ROOT / "checkpoint.pth"
CHECKPOINT_BYTES = 1_341_123_093
CHECKPOINT_SHA256 = "837fe48d140cf62fa2a9efbf056939d80d1714311e3c1b5fa26da7d65018e549"
SOURCE_COMMIT = "19bbbcb10264144bc115691c0b27f912ff3c0059"
MIRROR_COMMIT = "d65af8ade5127485702b0e6015947884f4bc6426"


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
    sys.path.insert(0, str(MODEL_ROOT))
    import models_vit

    model = models_vit.vit_base_patch16(num_classes=0, global_pool=False)
    torch.serialization.add_safe_globals([argparse_module.Namespace])
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert set(payload) == {"model", "optimizer", "epoch", "scaler", "args"}
    checkpoint_state = payload["model"]
    model_state = model.state_dict()
    encoder_state = {name: value for name, value in checkpoint_state.items() if name in model_state}
    assert set(encoder_state) == set(model_state)
    assert all(encoder_state[name].shape == model_state[name].shape for name in model_state)
    model.load_state_dict(encoder_state, strict=True)
    ignored = sorted(set(checkpoint_state) - set(encoder_state))
    assert ignored and all(name.startswith(("mask_token", "decoder_")) for name in ignored)
    return model, len(encoder_state), ignored, int(payload["epoch"])


def all_tokens(model, images):
    tokens = model.patch_embed(images)
    tokens = torch.cat((model.cls_token.expand(len(tokens), -1, -1), tokens), dim=1)
    tokens = model.pos_drop(tokens + model.pos_embed)
    for block in model.blocks:
        tokens = block(tokens)
    return model.norm(tokens)


def summarize(tokens: np.ndarray, masks: np.ndarray):
    if tokens.shape[1:] != (197, 768) or masks.shape[1:] != (224, 224):
        raise ValueError("Unexpected UltraFedFM token or mask shape")
    coverage = masks.reshape(-1, 14, 16, 14, 16).mean((2, 4)).reshape(-1, 196)
    cls = tokens[:, 0].astype(np.float32)
    roi, counts = [], []
    for local_tokens, local_coverage in zip(tokens[:, 1:], coverage):
        selected = local_coverage >= .30
        if not selected.any():
            selected[np.argmax(local_coverage)] = True
        values = local_tokens[selected]
        roi.append(np.concatenate([values.mean(0), values.std(0)]))
        counts.append(int(selected.sum()))
    return cls, np.asarray(roi, np.float32), np.asarray(counts)


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if CHECKPOINT.stat().st_size != CHECKPOINT_BYTES or digest(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("UltraFedFM checkpoint integrity mismatch")
    OUT.mkdir()
    write_json(OUT / "protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "UltraFedFM MAE-pretrained ViT-B/16 encoder, 12 layers, 768-dimensional tokens",
        "source": "https://github.com/yuncheng97/UltraFedFM",
        "source_commit": SOURCE_COMMIT,
        "checkpoint_mirror": "https://huggingface.co/fmassimo/UltraFedFM",
        "mirror_commit": MIRROR_COMMIT,
        "mirror_claim": "Model card declares an unmodified mirror of the upstream log_2024-07-16_13_53_08/checkpoint.pth; no independent upstream checksum was published",
        "paper": "https://doi.org/10.1038/s41746-025-02165-3",
        "license": "Apache-2.0 in the upstream repository",
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "declared_pretraining": "Federated self-supervised masked-autoencoder pretraining on 1,015,754 ultrasound images from 16 institutions",
        "cohort": "Unchanged 210 patients, 414 kidneys and 820 contour-first images",
        "input": "224x224 tight masked RGB kidney canvas with the official ImageNet normalization",
        "representations": {
            "ultrafed_cls": "Final normalized CLS token used by the official diagnosis implementation",
            "ultrafed_roi_heterogeneity": "Mean and standard deviation of final patch tokens with kidney-mask coverage at least 0.30",
        },
        "head": "Frozen encoder; unchanged PCA32 logistic image and ordered-patient-context heads, equal probability average",
        "fusion": "Replace only the deep branch of the unchanged fixed chain; equal fixed-reference/UltraFedFM average is a prespecified control",
        "roles": "Three fixed 126-patient training and 42-patient development roles; exploratory-test role unscored",
        "selection": "No development-selected layer, mask threshold, head, component count or fusion weight",
        "target": "Development AUROC and AP both above 0.8 before locked evaluation",
    })
    write_json(OUT / "run_lock.json", {
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "checkpoint_sha256": digest(CHECKPOINT),
    })
    print("Registered frozen UltraFedFM experiment", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(Path(__file__)) == lock["script_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    assert digest(CHECKPOINT) == lock["checkpoint_sha256"] == CHECKPOINT_SHA256


def encode() -> None:
    require_lock()
    if (OUT / "ultrafed_cls.npy").exists():
        raise FileExistsError("Preserve completed embeddings")
    torch.manual_seed(20261104)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, loaded_tensors, ignored, epoch = construct_model()
    model = model.cuda().eval().requires_grad_(False)
    before = model_digest(model)
    _, _, images = replacement.tile.load_reference()
    mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1)
    cls_rows, roi_rows, counts = [], [], []
    repeat_maxabs = batch_maxabs = None
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            canvases, masks = [], []
            for identifier in images.image_id.iloc[start:start + 32]:
                bgr = cv2.imread(str(BASE.OUT / "clean_rgb" / f"{identifier}.png"), cv2.IMREAD_COLOR)
                mask = cv2.imread(str(BASE.OUT / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE) > 0
                canvas, canvas_mask = BASE.tight_canvas(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask)
                canvases.append(canvas)
                masks.append(canvas_mask.astype(np.float32))
            tensor = torch.from_numpy(np.stack(canvases).transpose(0, 3, 1, 2).copy()).cuda().float() / 255
            tensor = (tensor - mean) / std
            tokens = all_tokens(model, tensor)
            if start == 0:
                repeated = all_tokens(model, tensor)
                single = all_tokens(model, tensor[:1])
                repeat_maxabs = float((tokens - repeated).abs().max())
                batch_maxabs = float((tokens[:1] - single).abs().max())
                assert max(repeat_maxabs, batch_maxabs) <= 1e-4
            cls, roi, selected = summarize(tokens.cpu().numpy(), np.stack(masks))
            cls_rows.append(cls)
            roi_rows.append(roi)
            counts.extend(selected.tolist())
            print(f"UltraFedFM encoded {min(start + 32, len(images))}/{len(images)}", flush=True)
    cls = np.concatenate(cls_rows)
    roi = np.concatenate(roi_rows)
    assert cls.shape == (820, 768) and roi.shape == (820, 1536)
    assert np.isfinite(cls).all() and np.isfinite(roi).all()
    after = model_digest(model)
    assert before == after
    np.save(OUT / "ultrafed_cls.npy", cls)
    np.save(OUT / "ultrafed_roi_heterogeneity.npy", roi)
    write_json(OUT / "encoder_audit.json", {
        "images": 820,
        "checkpoint_epoch": epoch,
        "loaded_encoder_tensors": loaded_tensors,
        "ignored_pretraining_tensors": len(ignored),
        "ignored_prefixes": sorted(set(name.split(".")[0] for name in ignored)),
        "other_missing_or_unexpected_tensors": [],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "model_state_before": before,
        "model_state_after": after,
        "repeat_maxabs": repeat_maxabs,
        "batch_maxabs": batch_maxabs,
        "selected_patch_min_median_max": [min(counts), float(np.median(counts)), max(counts)],
        "labels_used_in_encoding": [],
        "frozen": True,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
    })
    print("Saved frozen UltraFedFM embeddings", flush=True)


def evaluate() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed predictions")
    patients, kidneys, images = replacement.tile.load_reference()
    assignments = pd.read_csv(BASE.OUT / "patient_assignments.csv")
    y_kidney = kidneys.kidney_APN.to_numpy(int)
    y_image = images.kidney_APN.to_numpy(int)
    arrays = {
        "ultrafed_cls": np.load(OUT / "ultrafed_cls.npy"),
        "ultrafed_roi_heterogeneity": np.load(OUT / "ultrafed_roi_heterogeneity.npy"),
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
        print(f"Completed UltraFedFM development draw {seed}", flush=True)
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
