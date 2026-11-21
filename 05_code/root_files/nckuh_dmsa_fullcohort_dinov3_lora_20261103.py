"""Mask-constrained DINOv3 last-four-layer LoRA for full-cohort NCKUH APN."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

import nckuh_dmsa_fullcohort_clinical_features_20261103 as C
import nckuh_dmsa_fullcohort_fullframe_20261103 as F
from nckuh_only_study_20261102 import digest, write_json


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_dinov3_lora_20261103"
DINO = ROOT / "03_data_models/dinov3_vith16plus_weights_20260913"
REFERENCE = "fullcohort_rad_color_image_plus_patient_direct"
EPOCHS = (2, 5, 10)
GROUPS = ("dinov3_lora_last4_softmax2", "dinov3_lora_last4_plus_instance_reference")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0, dropout: float = .05):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=np.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        base = self.base(values)
        delta = nn.functional.linear(
            nn.functional.linear(self.dropout(values.float()), self.a), self.b)
        return base + (delta * self.scale).to(base.dtype)


class KidneyHead(nn.Module):
    def __init__(self, width: int = 1280):
        super().__init__()
        self.model = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, 128),
                                   nn.GELU(), nn.Dropout(.20), nn.Linear(128, 1))

    def forward(self, hidden: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        cls = hidden[:, 0]
        patches = hidden[:, 5:]
        selected = patch_mask.to(patches.dtype)
        pooled = (patches * selected[..., None]).sum(1) / selected.sum(1).clamp_min(1)[:, None]
        return self.model(torch.cat([cls, pooled], dim=1).float()).squeeze(1)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered DINOv3 LoRA protocol")
    F.require_lock()
    inputs = [F.OUT / "patient_assignments.csv", F.OUT / "deidentified_source_manifest.csv",
              F.OUT / "instance_development_predictions.csv", DINO / "model.safetensors", Path(__file__)]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same 210 patients, 414 kidneys and 820 contour-first images with unchanged patient roles",
        "encoder": "Official local DINOv3 ViT-H+/16; first 28 of 32 transformer layers frozen and token-cached; rank4 alpha8 LoRA on q_proj and v_proj of layers28-31 only",
        "input": "Kidney contour tight canvas with all non-kidney pixels zero; 224 by 224; official local image processor",
        "pooling": "Concatenate class token and mean of patch tokens with kidney-mask coverage at least0.30; fixed softmax-logit pooling across images of one kidney",
        "selection": "Within each 126-patient training role, fixed stratified100/26 patient split selects earliest epoch among2,5,10 by mean kidney AUROC and AP; reset then refit all126",
        "loss": "Patient-balanced and class-balanced image BCE; no DMSA field enters encoding",
        "fusion": "Only fixed equal-probability mean with unchanged full-cohort RadImageNet image-instance reference",
        "prohibited": "No development-selected weight, test scoring, identifier, filename diagnosis suffix or full-model fine-tuning",
        "target": "Development AUROC>0.8 and AP>0.8; locked confirmation remains required",
        "groups": list(GROUPS), "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {"source_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs}})
    print("Registered full-cohort DINOv3 LoRA experiment", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(Path(__file__)) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(Path(name)) == expected, name


def load_model():
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    config = DINOv3ViTConfig.from_pretrained(DINO, local_files_only=True)
    model = DINOv3ViTModel.from_pretrained(
        DINO, config=config, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16).cuda().eval().requires_grad_(False)
    return model


def canvas_batch(images: pd.DataFrame, start: int, stop: int):
    canvases, masks = [], []
    for identifier in images.image_id.iloc[start:stop]:
        bgr = cv2.imread(str(F.OUT / "clean_rgb" / f"{identifier}.png"), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(F.OUT / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE) > 0
        canvas, canvas_mask = F.tight_canvas(rgb, mask)
        coverage = cv2.resize(canvas_mask.astype(np.float32), (14, 14), interpolation=cv2.INTER_AREA)
        patch = coverage >= .30
        if not patch.any():
            patch.flat[int(np.argmax(coverage))] = True
        canvases.append(canvas); masks.append(patch.reshape(-1))
    return canvases, np.asarray(masks, dtype=bool)


def cache() -> None:
    require_lock()
    if (OUT / "layer28_tokens_float32.npy").exists():
        raise FileExistsError("Preserve completed DINOv3 token cache")
    from transformers import AutoImageProcessor

    _, _, images = C.load_reference()
    processor = AutoImageProcessor.from_pretrained(DINO, local_files_only=True, trust_remote_code=False)
    model = load_model()
    tokens = np.lib.format.open_memmap(OUT / "layer28_tokens_float32.npy", mode="w+",
                                       dtype=np.float32, shape=(820, 201, 1280))
    masks = np.zeros((820, 196), dtype=bool)
    max_error = 0.0
    with torch.inference_mode():
        for start in range(0, 820, 8):
            canvases, current_masks = canvas_batch(images, start, min(start + 8, 820))
            pixel = processor(images=canvases, return_tensors="pt").pixel_values.cuda()
            pixel = pixel.to(model.embeddings.patch_embeddings.weight.dtype)
            hidden = model.embeddings(pixel)
            position = model.rope_embeddings(pixel)
            for layer in model.model.layer[:28]:
                hidden = layer(hidden, position_embeddings=position)
            tokens[start:start + len(canvases)] = hidden.float().cpu().numpy()
            masks[start:start + len(canvases)] = current_masks
            if start == 0:
                replay = hidden
                for layer in model.model.layer[28:]:
                    replay = layer(replay, position_embeddings=position)
                replay = model.norm(replay)[:, 0]
                full = model(pixel).pooler_output
                max_error = float((replay - full).abs().max())
                assert max_error < .02
            if start % 80 == 0:
                print(f"Cached DINOv3 layer28 tokens {min(start + 8, 820)}/820", flush=True)
    tokens.flush()
    np.save(OUT / "kidney_patch_mask.npy", masks)
    write_json(OUT / "cache_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "images": 820, "token_shape": [820, 201, 1280],
        "patch_mask_shape": [820, 196], "selected_patch_min_median_max": [
            int(masks.sum(1).min()), float(np.median(masks.sum(1))), int(masks.sum(1).max())],
        "cached_replay_maxabs_vs_full": max_error,
        "labels_used_in_encoding": [], "weight_sha256": digest(DINO / "model.safetensors"),
        "token_sha256": digest(OUT / "layer28_tokens_float32.npy")})
    print("Completed DINOv3 layer28 token cache", flush=True)


def patch_masks() -> None:
    require_lock()
    if not (OUT / "layer28_tokens_float32.npy").exists():
        raise FileNotFoundError("Token cache must exist before patch-mask repair")
    _, _, images = C.load_reference()
    masks = []
    for start in range(0, 820, 32):
        _, current = canvas_batch(images, start, min(start + 32, 820))
        masks.append(current)
    masks = np.concatenate(masks)
    assert masks.shape == (820, 196) and masks.sum(1).min() >= 1
    np.save(OUT / "kidney_patch_mask.npy", masks)
    audit = json.loads((OUT / "cache_audit.json").read_text(encoding="utf-8"))
    audit["selected_patch_min_median_max"] = [
        int(masks.sum(1).min()), float(np.median(masks.sum(1))), int(masks.sum(1).max())]
    audit["patch_mask_sha256"] = digest(OUT / "kidney_patch_mask.npy")
    write_json(OUT / "cache_audit.json", audit)
    print("Rebuilt nonempty DINOv3 kidney patch masks", flush=True)


def add_lora(model) -> list[nn.Parameter]:
    parameters = []
    for layer in model.model.layer[28:]:
        for name in ("q_proj", "v_proj"):
            wrapped = LoRALinear(getattr(layer.attention, name)).cuda()
            setattr(layer.attention, name, wrapped)
            parameters.extend([wrapped.a, wrapped.b])
    return parameters


def image_weights(images: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    subset = images.iloc[indices]
    weights = 1.0 / subset.patient_alias.map(subset.patient_alias.value_counts()).to_numpy(float)
    labels = subset.kidney_APN.to_numpy(int)
    for value in (0, 1):
        selected = labels == value
        weights[selected] *= .5 / weights[selected].sum()
    return weights / weights.mean()


def aggregate_softmax2(probabilities: np.ndarray, images: pd.DataFrame,
                       kidney_aliases: pd.Series) -> np.ndarray:
    output = []
    for alias in kidney_aliases:
        probability = np.clip(probabilities[images.kidney_alias.eq(alias)], 1e-6, 1 - 1e-6)
        logits = np.log(probability / (1 - probability))
        weights = np.exp(2 * logits - np.max(2 * logits))
        value = float(np.sum(weights * logits) / np.sum(weights))
        output.append(1 / (1 + np.exp(-np.clip(value, -30, 30))))
    return np.asarray(output)


def runtime(seed: int):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = load_model()
    lora = add_lora(model)
    torch.manual_seed(seed + 31); torch.cuda.manual_seed_all(seed + 31)
    head = KidneyHead().cuda()
    optimizer = torch.optim.AdamW([{"params": lora, "lr": 1e-4},
                                   {"params": head.parameters(), "lr": 1e-3}], weight_decay=1e-3)
    return model, head, optimizer, lora


def position_embeddings(model):
    pixel = torch.zeros((1, 3, 224, 224), device="cuda", dtype=torch.bfloat16)
    return model.rope_embeddings(pixel)


def train_and_score(seed: int, fit_images: np.ndarray, score_images: np.ndarray,
                    epochs: int, checkpoints: set[int], tokens, patch_masks,
                    images: pd.DataFrame, score_kidneys: pd.DataFrame):
    model, head, optimizer, lora = runtime(seed)
    position = position_embeddings(model)
    weights = image_weights(images, fit_images)
    labels = images.kidney_APN.to_numpy(int)
    rng = np.random.default_rng(seed + 701)
    records = {}

    def score():
        head.eval(); probabilities = np.zeros(len(images), dtype=float)
        with torch.inference_mode():
            for start in range(0, len(score_images), 16):
                index = score_images[start:start + 16]
                hidden = torch.from_numpy(np.asarray(tokens[index])).cuda().to(torch.bfloat16)
                mask = torch.from_numpy(np.asarray(patch_masks[index])).cuda()
                for layer in model.model.layer[28:]:
                    hidden = layer(hidden, position_embeddings=position)
                hidden = model.norm(hidden)
                probabilities[index] = torch.sigmoid(head(hidden, mask)).cpu().numpy()
        return aggregate_softmax2(probabilities, images, score_kidneys.kidney_alias)

    for epoch in range(1, epochs + 1):
        head.train()
        order = rng.permutation(fit_images)
        for start in range(0, len(order), 16):
            index = order[start:start + 16]
            hidden = torch.from_numpy(np.asarray(tokens[index])).cuda().to(torch.bfloat16)
            mask = torch.from_numpy(np.asarray(patch_masks[index])).cuda()
            for layer in model.model.layer[28:]:
                hidden = layer(hidden, position_embeddings=position)
            hidden = model.norm(hidden)
            logits = head(hidden, mask)
            target = torch.from_numpy(labels[index]).cuda().float()
            sample_weight = torch.from_numpy(weights[np.searchsorted(fit_images, index)]).cuda().float()
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * sample_weight).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(head.parameters()) + lora, 1.0)
            optimizer.step()
        if epoch in checkpoints:
            records[epoch] = score()
    trainable = int(sum(parameter.numel() for parameter in list(head.parameters()) + lora))
    del model, head, optimizer
    torch.cuda.empty_cache()
    return records, trainable


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed DINOv3 LoRA predictions")
    from sklearn.model_selection import train_test_split

    patients, kidneys, images = C.load_reference()
    tokens = np.load(OUT / "layer28_tokens_float32.npy", mmap_mode="r")
    patch_masks = np.load(OUT / "kidney_patch_mask.npy", mmap_mode="r")
    assignments = pd.read_csv(F.OUT / "patient_assignments.csv")
    reference_frame = pd.read_csv(F.OUT / "instance_development_predictions.csv")
    reference = reference_frame[reference_frame.group.eq(REFERENCE)].set_index(["seed", "kidney_alias"]).score
    metrics, predictions, selections = [], [], []
    for seed in F.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train_patients = patients[patients.patient_alias.map(mapping).eq("train")].copy()
        inner_fit, inner_select = train_test_split(
            np.arange(len(train_patients)), train_size=100, random_state=seed + 4000,
            stratify=train_patients.DMSA_APN)
        fit_aliases = set(train_patients.patient_alias.iloc[inner_fit])
        select_aliases = set(train_patients.patient_alias.iloc[inner_select])
        fit_images = np.flatnonzero(images.patient_alias.isin(fit_aliases))
        select_images = np.flatnonzero(images.patient_alias.isin(select_aliases))
        select_kidneys = kidneys[kidneys.patient_alias.isin(select_aliases)].copy().reset_index(drop=True)
        inner, trainable = train_and_score(seed, fit_images, select_images, max(EPOCHS), set(EPOCHS),
                                           tokens, patch_masks, images, select_kidneys)
        inner_rows = []
        for epoch, score in inner.items():
            result = F.metric(select_kidneys.kidney_APN.to_numpy(int), score)
            inner_rows.append((np.mean([result["auroc"], result["ap"]]), epoch, result))
        best_value = max(row[0] for row in inner_rows)
        selected_epoch = min(row[1] for row in inner_rows if row[0] == best_value)
        selections.extend({"seed": seed, "epoch": epoch, **result, "selected": epoch == selected_epoch}
                          for _, epoch, result in inner_rows)

        all_train_aliases = set(train_patients.patient_alias)
        train_images = np.flatnonzero(images.patient_alias.isin(all_train_aliases))
        development_aliases = set(patients.patient_alias[patients.patient_alias.map(mapping).eq("development")])
        development_images = np.flatnonzero(images.patient_alias.isin(development_aliases))
        development_kidneys = kidneys[kidneys.patient_alias.isin(development_aliases)].copy().reset_index(drop=True)
        score = train_and_score(seed, train_images, development_images, selected_epoch, {selected_epoch},
                                tokens, patch_masks, images, development_kidneys)[0][selected_epoch]
        reference_score = np.asarray([reference.loc[(seed, alias)] for alias in development_kidneys.kidney_alias])
        scores = {GROUPS[0]: score, GROUPS[1]: (score + reference_score) / 2}
        for group, values in scores.items():
            result = F.metric(development_kidneys.kidney_APN.to_numpy(int), values)
            metrics.append({"seed": seed, "role": "development", "group": group,
                            "patients": 42, "kidneys": len(development_kidneys), **result})
            for local, row in enumerate(development_kidneys.itertuples(index=False)):
                predictions.append({"seed": seed, "group": group, "patient_alias": row.patient_alias,
                    "kidney_alias": row.kidney_alias, "DMSA_APN": row.DMSA_APN, "side": row.side,
                    "label": int(row.kidney_APN), "score": float(values[local])})
        print(f"Completed DINOv3 LoRA draw {seed}; epoch {selected_epoch}", flush=True)

    frame = pd.DataFrame(metrics); frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "inner_epoch_selections.csv", index=False)
    summary = frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    write_json(OUT / "development_decision.json", {
        "groups_above_both_development_targets": summary[(summary.auroc_mean > .8) & (summary.ap_mean > .8)].index.tolist(),
        "exploratory_test_scored": False, "trainable_parameters": trainable})
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "cache", "patch_masks", "run"))
    args = parser.parse_args()
    {"prepare": prepare, "cache": cache, "patch_masks": patch_masks, "run": run}[args.stage]()
