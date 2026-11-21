"""Matched frozen-head versus last-four-layer DINOv3 LoRA with mild augmentation."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

import nckuh_dmsa_fullcohort_dinov3_lora_20261103 as old
import nckuh_dmsa_fullcohort_fullframe_20261103 as full
from prepare_nckuh_383_contour_cohort_20261111 import digest


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
CACHE = ROOT / "02_experiments/NCKUH_revised_383_contour_cache_20261111"
DINO = ROOT / "03_data_models/dinov3_vith16plus_weights_20260913"
OUT = ROOT / "02_experiments/NCKUH_revised_383_dinov3_lora_aug_20261112"
VARIANTS = ((0.0, 1.0), (-3.0, .9), (3.0, 1.1))
EPOCHS = (2, 5, 10)
SEED = 20261112


def canvas(image_id, angle, gamma):
    bgr = cv2.imread(str(CACHE / "clean_rgb" / f"{image_id}.png"), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(CACHE / "masks" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE) > 0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result, selected = full.tight_canvas(rgb, mask)
    if angle:
        matrix = cv2.getRotationMatrix2D((111.5, 111.5), angle, 1.0)
        result = cv2.warpAffine(result, matrix, (224, 224), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        selected = cv2.warpAffine(selected, matrix, (224, 224), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if gamma != 1:
        result = np.rint(255 * np.power(result.astype(np.float32) / 255, gamma)).clip(0, 255).astype(np.uint8)
        result *= selected[..., None]
    coverage = cv2.resize(selected.astype(np.float32), (14, 14), interpolation=cv2.INTER_AREA)
    patch = coverage >= .30
    if not patch.any():
        patch.flat[np.argmax(coverage)] = True
    return result, patch.reshape(-1)


def cache_tokens():
    from transformers import AutoImageProcessor

    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "layer28_tokens_float32.npy").exists():
        raise FileExistsError("Preserve augmented token cache")
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    assert len(images) == 1362
    processor = AutoImageProcessor.from_pretrained(DINO, local_files_only=True, trust_remote_code=False)
    model = old.load_model()
    tokens = np.lib.format.open_memmap(OUT / "layer28_tokens_float32.npy", mode="w+",
                                       dtype=np.float32, shape=(3, len(images), 201, 1280))
    masks = np.zeros((3, len(images), 196), dtype=bool)
    with torch.inference_mode():
        for variant, (angle, gamma) in enumerate(VARIANTS):
            for start in range(0, len(images), 8):
                batch, current_masks = zip(*(canvas(identifier, angle, gamma)
                    for identifier in images.image_id.iloc[start:start + 8]))
                pixel = processor(images=list(batch), return_tensors="pt", do_resize=False).pixel_values.cuda()
                pixel = pixel.to(model.embeddings.patch_embeddings.weight.dtype)
                hidden = model.embeddings(pixel); position = model.rope_embeddings(pixel)
                for layer in model.model.layer[:28]:
                    hidden = layer(hidden, position_embeddings=position)
                values = hidden.float().cpu().numpy()
                assert np.isfinite(values).all()
                tokens[variant, start:start + len(batch)] = values
                masks[variant, start:start + len(batch)] = np.asarray(current_masks)
                if start % 160 == 0:
                    print(f"variant {variant} tokens {min(start + 8, len(images))}/{len(images)}", flush=True)
    tokens.flush(); np.save(OUT / "patch_masks.npy", masks)
    audit = {"images": len(images), "variants": [{"rotation_degrees": a, "gamma": g} for a, g in VARIANTS],
             "token_shape": list(tokens.shape), "mask_shape": list(masks.shape),
             "selected_patch_min_median_max": [int(masks.sum(2).min()), float(np.median(masks.sum(2))),
                                                int(masks.sum(2).max())],
             "labels_used_in_cache": [], "weight_sha256": digest(DINO / "model.safetensors"),
             "image_manifest_sha256": digest(REGISTRY / "contour_image_manifest.csv"),
             "token_sha256": digest(OUT / "layer28_tokens_float32.npy")}
    (OUT / "cache_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    del model; torch.cuda.empty_cache()
    print(json.dumps(audit, indent=2), flush=True)


def runtime(seed, lora):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = old.load_model()
    parameters = old.add_lora(model) if lora else []
    torch.manual_seed(seed + 31); torch.cuda.manual_seed_all(seed + 31)
    head = old.KidneyHead().cuda()
    groups = [{"params": head.parameters(), "lr": 1e-3}]
    if parameters:
        groups.insert(0, {"params": parameters, "lr": 1e-4})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-3)
    return model, head, optimizer, parameters


def train_score(arm, fit_images, score_images, epochs, checkpoints, tokens, masks, images, score_kidneys):
    model, head, optimizer, lora = runtime(SEED, arm == "lora_aug")
    position = old.position_embeddings(model)
    weights = old.image_weights(images, fit_images)
    labels = images.kidney_APN.to_numpy(int)
    patient_code = pd.factorize(images.patient_alias)[0]
    rng = np.random.default_rng(SEED + (17 if arm == "lora_aug" else 0))
    records = {}

    def evaluate():
        head.eval(); probabilities = np.zeros(len(images), dtype=float)
        with torch.inference_mode():
            for start in range(0, len(score_images), 16):
                index = score_images[start:start + 16]
                hidden = torch.from_numpy(np.asarray(tokens[0, index]).astype(np.float32)).cuda().to(torch.bfloat16)
                patch = torch.from_numpy(np.asarray(masks[0, index])).cuda()
                for layer in model.model.layer[28:]:
                    hidden = layer(hidden, position_embeddings=position)
                probabilities[index] = torch.sigmoid(head(model.norm(hidden), patch)).cpu().numpy()
        return old.aggregate_softmax2(probabilities, images, score_kidneys.kidney_alias)

    for epoch in range(1, epochs + 1):
        head.train(); order = rng.permutation(fit_images)
        for start in range(0, len(order), 16):
            index = order[start:start + 16]
            variant = (patient_code[index] + epoch) % len(VARIANTS)
            hidden_np = np.stack([tokens[v, i] for v, i in zip(variant, index)]).astype(np.float32)
            patch_np = np.stack([masks[v, i] for v, i in zip(variant, index)])
            hidden = torch.from_numpy(hidden_np).cuda().to(torch.bfloat16)
            patch = torch.from_numpy(patch_np).cuda()
            for layer in model.model.layer[28:]:
                hidden = layer(hidden, position_embeddings=position)
            logits = head(model.norm(hidden), patch)
            target = torch.from_numpy(labels[index]).cuda().float()
            sample_weight = torch.from_numpy(weights[np.searchsorted(fit_images, index)]).cuda().float()
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * sample_weight).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(head.parameters()) + lora, 1.0); optimizer.step()
        if epoch in checkpoints:
            records[epoch] = evaluate()
            print(f"{arm} epoch {epoch}", flush=True)
    trainable = sum(parameter.numel() for parameter in list(head.parameters()) + lora)
    del model, head, optimizer; torch.cuda.empty_cache()
    return records, trainable


def run():
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve LoRA experiment")
    audit = json.loads((OUT / "cache_audit.json").read_text())
    assert audit["image_manifest_sha256"] == digest(REGISTRY / "contour_image_manifest.csv")
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    assignments = pd.read_csv(REGISTRY / "patient_assignments.csv")
    tokens = np.load(OUT / "layer28_tokens_float32.npy", mmap_mode="r")
    masks = np.load(OUT / "patch_masks.npy", mmap_mode="r")
    train_patients = assignments[assignments.role.eq("train") & assignments.patient_alias.isin(images.patient_alias)].reset_index(drop=True)
    inner_fit, inner_select = train_test_split(np.arange(len(train_patients)), train_size=160,
                                               stratify=train_patients.DMSA_APN, random_state=SEED)
    fit_aliases = set(train_patients.patient_alias.iloc[inner_fit]); select_aliases = set(train_patients.patient_alias.iloc[inner_select])
    fit_images = np.flatnonzero(images.patient_alias.isin(fit_aliases)); select_images = np.flatnonzero(images.patient_alias.isin(select_aliases))
    select_kidneys = kidneys[kidneys.patient_alias.isin(select_aliases)].reset_index(drop=True)
    development_aliases = set(assignments.loc[assignments.role.eq("development"), "patient_alias"]) & set(images.patient_alias)
    development_images = np.flatnonzero(images.patient_alias.isin(development_aliases))
    development_kidneys = kidneys[kidneys.patient_alias.isin(development_aliases)].reset_index(drop=True)
    all_train_images = np.flatnonzero(images.patient_alias.isin(set(train_patients.patient_alias)))
    results, predictions, selections = [], [], []
    for arm in ("frozen_head_aug", "lora_aug"):
        inner, trainable = train_score(arm, fit_images, select_images, max(EPOCHS), set(EPOCHS),
                                       tokens, masks, images, select_kidneys)
        rows = []
        y_select = select_kidneys.kidney_APN.to_numpy(int)
        for epoch, values in inner.items():
            auc = roc_auc_score(y_select, values); ap = average_precision_score(y_select, values)
            rows.append((min(auc, ap), epoch, auc, ap))
        best = max(row[0] for row in rows); selected_epoch = min(row[1] for row in rows if row[0] == best)
        selections.extend({"arm": arm, "epoch": e, "auroc": a, "ap": p, "selected": e == selected_epoch}
                          for _, e, a, p in rows)
        values = train_score(arm, all_train_images, development_images, selected_epoch, {selected_epoch},
                             tokens, masks, images, development_kidneys)[0][selected_epoch]
        y = development_kidneys.kidney_APN.to_numpy(int)
        clusters = [np.flatnonzero(development_kidneys.patient_alias.eq(pid))
                    for pid in sorted(development_kidneys.patient_alias.unique())]
        rng = np.random.default_rng(SEED)
        draws = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
                 for _ in range(2000)]
        draws = [idx for idx in draws if np.unique(y[idx]).size == 2]
        auc_boot = np.asarray([roc_auc_score(y[idx], values[idx]) for idx in draws])
        ap_boot = np.asarray([average_precision_score(y[idx], values[idx]) for idx in draws])
        results.append({"group": arm, "selected_epoch": selected_epoch, "trainable_parameters": trainable,
                        "patients": development_kidneys.patient_alias.nunique(), "kidneys": len(y),
                        "auroc": roc_auc_score(y, values), "ap": average_precision_score(y, values),
                        "auroc_ci_low": np.percentile(auc_boot, 2.5), "auroc_ci_high": np.percentile(auc_boot, 97.5),
                        "ap_ci_low": np.percentile(ap_boot, 2.5), "ap_ci_high": np.percentile(ap_boot, 97.5)})
        predictions.extend({"group": arm, "patient_alias": row.patient_alias, "kidney_alias": row.kidney_alias,
                            "side": row.side, "label": int(row.kidney_APN), "score": float(value)}
                           for row, value in zip(development_kidneys.itertuples(index=False), values))
    pd.DataFrame(results).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "inner_epoch_selections.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "cohort": "revised 383 roster; 204 contour-ready train and 70 development patients",
        "arms": ["matched frozen last-four blocks plus trainable head", "rank4 alpha8 LoRA q/v in last four blocks plus same head"],
        "augmentation": [{"rotation_degrees": a, "gamma": g} for a, g in VARIANTS],
        "augmentation_assignment": "same deterministic variant for every image of a patient in an epoch",
        "selection": "160/44 training-patient inner split stratified by N/L/R/B; earliest epoch maximizing min(AUROC, AP)",
        "test_scored": False}, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["cache", "run"])
    {"cache": cache_tokens, "run": run}[parser.parse_args().stage]()
