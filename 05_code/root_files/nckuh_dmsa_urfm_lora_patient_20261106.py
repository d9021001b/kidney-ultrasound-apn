"""Patient-bag supervised adaptation of official URFM for kidney-level APN."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_urfm_frozen_20261106 as urfm
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_urfm_lora_patient_20261106"
CACHE = OUT / "cache"
REFERENCE = uri.OUT / "development_predictions.csv"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"
SEEDS = tuple(base.SEEDS)
ARMS = ("head_only", "lora_rank")
MAX_EPOCHS = 36
MIN_EPOCHS = 6
PATIENCE = 7
BATCH_PATIENTS = 8
EMBED_DIM = urfm.EMBED_DIM
CACHE_BLOCKS = 8
MODEL_BLOCKS = 12


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def augmented_canvas(image: np.ndarray, mask: np.ndarray, variant: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply a small fixed rotation/gamma transform to image and mask together."""
    if variant == 0:
        return image.copy(), mask.copy()
    if variant not in (1, 2):
        raise ValueError(f"Unknown augmentation variant: {variant}")
    angle = -3.0 if variant == 1 else 3.0
    gamma = 0.90 if variant == 1 else 1.10
    height, width = mask.shape
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    rotated_mask = cv2.warpAffine(
        mask.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ) > 0
    adjusted = np.clip((rotated.astype(np.float32) / 255.0) ** gamma * 255.0, 0, 255)
    adjusted[~rotated_mask] = 0
    return adjusted.astype(np.uint8), rotated_mask


def patient_design(
    patients: pd.DataFrame, kidneys: pd.DataFrame, images: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map each patient to padded image rows, sides, kidney labels, and presence."""
    lookup = {alias: index for index, alias in enumerate(patients.patient_alias)}
    width = int(images.groupby("patient_alias").size().max())
    image_index = np.full((len(patients), width), -1, dtype=np.int64)
    image_side = np.full((len(patients), width), -1, dtype=np.int64)
    for alias, group in images.groupby("patient_alias", sort=False):
        patient_index = lookup[alias]
        rows = group.index.to_numpy(np.int64)
        image_index[patient_index, : len(rows)] = rows
        image_side[patient_index, : len(rows)] = group.side.map({"left": 0, "right": 1}).to_numpy()

    labels = np.full((len(patients), 2), -1, dtype=np.int64)
    present = np.zeros((len(patients), 2), dtype=bool)
    for row in kidneys.itertuples(index=False):
        patient_index = lookup[row.patient_alias]
        side = 0 if row.side == "left" else 1
        labels[patient_index, side] = int(row.kidney_APN)
        present[patient_index, side] = True
    if not np.array_equal(labels[present], labels[present].astype(bool).astype(int)):
        raise ValueError("Kidney labels must be binary")
    return image_index, image_side, labels, present


def pairwise_rank_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth all-positive/all-negative ranking loss within a patient batch."""
    positive = logits[target == 1]
    negative = logits[target == 0]
    if len(positive) == 0 or len(negative) == 0:
        return logits.sum() * 0.0
    return torch.nn.functional.softplus(-(positive[:, None] - negative[None, :])).mean()


def cache_tokens() -> None:
    if CACHE.exists() and any(CACHE.iterdir()):
        raise FileExistsError(f"Preserve existing token cache: {CACHE}")
    CACHE.mkdir(parents=True, exist_ok=True)
    patients, kidneys, images = cohort.load_reference()
    colors, masks = urfm.contour_canvases(images)
    model, load_audit = urfm.load_encoder()
    model = model.cuda().eval()
    lower = np.lib.format.open_memmap(
        CACHE / "lower8_float16.npy", mode="w+", dtype=np.float16,
        shape=(3, len(images), 197, EMBED_DIM),
    )
    selections = np.lib.format.open_memmap(
        CACHE / "patch_selection.npy", mode="w+", dtype=np.bool_,
        shape=(3, len(images), 196),
    )
    mean = torch.from_numpy(urfm.MEAN).cuda()[None, :, None, None]
    std = torch.from_numpy(urfm.STD).cuda()[None, :, None, None]
    fallback_count = 0
    with torch.inference_mode():
        for variant in range(3):
            variant_colors, variant_masks = [], []
            for color, mask in zip(colors, masks):
                current_color, current_mask = augmented_canvas(color, mask, variant)
                variant_colors.append(current_color)
                variant_masks.append(current_mask)
            variant_colors = np.stack(variant_colors)
            variant_masks = np.stack(variant_masks)
            for start in range(0, len(images), 24):
                stop = min(start + 24, len(images))
                tensor = torch.from_numpy(
                    variant_colors[start:stop].transpose(0, 3, 1, 2).copy()
                ).cuda().float() / 255.0
                tensor = (tensor - mean) / std
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    token = model.patch_embed(tensor)
                    cls = model.cls_token.expand(len(tensor), -1, -1)
                    token = model.pos_drop(torch.cat((cls, token), dim=1) + model.pos_embed)
                    for block in model.blocks[:CACHE_BLOCKS]:
                        token = block(token)
                current_mask = torch.from_numpy(variant_masks[start:stop]).cuda().float()[:, None]
                coverage = torch.nn.functional.avg_pool2d(current_mask, 16, 16)[:, 0]
                selected = coverage >= 0.25
                empty = ~selected.any((1, 2))
                if empty.any():
                    flat = selected.reshape(len(selected), -1)
                    coverage_flat = coverage.reshape(len(selected), -1)
                    flat[empty, coverage_flat[empty].argmax(1)] = True
                    selected = flat.reshape_as(selected)
                    fallback_count += int(empty.sum())
                lower[variant, start:stop] = token.float().cpu().numpy().astype(np.float16)
                selections[variant, start:stop] = selected.cpu().numpy().reshape(-1, 196)
                print(f"URFM cache variant {variant} {stop}/{len(images)}", flush=True)
    lower.flush()
    selections.flush()
    write_json(CACHE / "audit.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "patients": len(patients), "kidneys": len(kidneys), "images": len(images),
        "variants": ["original", "rotation_minus3_gamma090", "rotation_plus3_gamma110"],
        "lower_shape": [3, len(images), 197, EMBED_DIM],
        "cached_encoder_blocks": CACHE_BLOCKS,
        "patch_selection_shape": [3, len(images), 196],
        "coverage_fallbacks": fallback_count,
        "checkpoint_sha256": urfm.WEIGHT_SHA256,
        "load_audit": load_audit,
        "labels_used": [],
    })
    del model
    torch.cuda.empty_cache()


class PatientBagHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.project = torch.nn.Sequential(
            torch.nn.LayerNorm(EMBED_DIM), torch.nn.Linear(EMBED_DIM, 128), torch.nn.GELU(),
        )
        self.attention = torch.nn.Linear(128, 1)
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(513), torch.nn.Linear(513, 64), torch.nn.GELU(),
            torch.nn.Dropout(0.20), torch.nn.Linear(64, 1),
        )

    def forward(self, image_values: torch.Tensor, sides: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = torch.nn.functional.normalize(self.project(image_values.float()), dim=-1)
        kidneys, present = [], []
        for side in (0, 1):
            mask = sides.eq(side)
            attention = self.attention(projected).squeeze(-1).masked_fill(~mask, -1e4)
            weights = torch.softmax(attention, dim=1) * mask.float()
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
            pooled = (projected * weights[..., None]).sum(1)
            exists = mask.any(1)
            kidneys.append(torch.nn.functional.normalize(pooled, dim=-1) * exists[:, None])
            present.append(exists)
        kidneys = torch.stack(kidneys, dim=1)
        present = torch.stack(present, dim=1)
        logits = []
        for side in (0, 1):
            own, opposite = kidneys[:, side], kidneys[:, 1 - side]
            context = torch.cat(
                [own, opposite, own - opposite, torch.abs(own - opposite), present[:, 1 - side, None].float()],
                dim=1,
            )
            logits.append(self.classifier(context).squeeze(1))
        return torch.stack(logits, dim=1), present


def build_runtime(seed: int, arm: str):
    from peft import LoraConfig, inject_adapter_in_model

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    encoder, _ = urfm.load_encoder()
    encoder.requires_grad_(False)
    if arm == "lora_rank":
        targets = [
            f"blocks.{index}.attn.qkv" for index in range(CACHE_BLOCKS, MODEL_BLOCKS)
        ]
        encoder = inject_adapter_in_model(LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules=targets,
        ), encoder)
    elif arm != "head_only":
        raise ValueError(arm)
    encoder.fc_norm.requires_grad_(True)
    encoder = encoder.cuda()
    torch.manual_seed(seed + 17)
    head = PatientBagHead().cuda()
    parameters = [
        {"params": head.parameters(), "lr": 5e-4},
        {"params": encoder.fc_norm.parameters(), "lr": 1e-4},
    ]
    if arm == "lora_rank":
        parameters.append({
            "params": [parameter for name, parameter in encoder.named_parameters() if "lora_" in name],
            "lr": 1e-4,
        })
    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
    trainable = [name for name, parameter in encoder.named_parameters() if parameter.requires_grad]
    return encoder, head, optimizer, trainable


def metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return base.metric(labels, scores)


def train_once(
    seed: int,
    arm: str,
    fit_patients: np.ndarray,
    score_patients: np.ndarray,
    epochs: int,
    select_epoch: bool,
    design: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    lower: np.ndarray,
    selections: np.ndarray,
):
    image_index, image_side, labels, present = design
    encoder, head, optimizer, trainable = build_runtime(seed, arm)
    fit_labels = labels[fit_patients][present[fit_patients]]
    positive = int(fit_labels.sum())
    negative = int((fit_labels == 0).sum())
    class_weight = torch.tensor(
        [len(fit_labels) / (2 * negative), len(fit_labels) / (2 * positive)],
        device="cuda", dtype=torch.float32,
    )
    rng = np.random.default_rng(seed + 503)
    best, stale, records = None, 0, []

    def forward(indices: np.ndarray, variants: np.ndarray, training: bool):
        rows = image_index[indices]
        valid = rows >= 0
        flat_rows = rows[valid]
        flat_variants = np.broadcast_to(variants[:, None], rows.shape)[valid]
        token = torch.from_numpy(np.array(lower[flat_variants, flat_rows], dtype=np.float32)).cuda()
        selected = torch.from_numpy(np.array(selections[flat_variants, flat_rows])).cuda()
        encoder.train(training and arm == "lora_rank")
        encoder.fc_norm.train(training)
        head.train(training)
        with torch.set_grad_enabled(training):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for block in encoder.blocks[CACHE_BLOCKS:]:
                    token = block(token)
                patches = token[:, 1:]
                pooled = (patches * selected[..., None]).sum(1) / selected.sum(1, keepdim=True)
                embedding = encoder.fc_norm(pooled)
                padded = torch.zeros(
                    len(indices) * image_index.shape[1], EMBED_DIM,
                    device="cuda", dtype=embedding.dtype,
                )
                locations = torch.from_numpy(np.flatnonzero(valid.ravel())).cuda()
                padded = padded.index_copy(0, locations, embedding)
                sides = torch.from_numpy(image_side[indices]).cuda()
                logits, exists = head(
                    padded.reshape(len(indices), image_index.shape[1], EMBED_DIM), sides
                )
        return logits, exists

    def score(indices: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        encoder.eval()
        head.eval()
        all_scores = []
        with torch.inference_mode():
            for start in range(0, len(indices), BATCH_PATIENTS):
                batch = indices[start:start + BATCH_PATIENTS]
                variant_scores = []
                for variant in range(3):
                    logits, _ = forward(batch, np.full(len(batch), variant, dtype=np.int64), False)
                    variant_scores.append(torch.sigmoid(logits).float())
                all_scores.append(torch.stack(variant_scores).mean(0).cpu().numpy())
        values = np.concatenate(all_scores)
        return metric(labels[indices][present[indices]], values[present[indices]]), values

    start_time = time.monotonic()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(fit_patients)
        total_loss = total_kidneys = 0.0
        for start in range(0, len(order), BATCH_PATIENTS):
            indices = order[start:start + BATCH_PATIENTS]
            variants = rng.integers(0, 3, size=len(indices), dtype=np.int64)
            optimizer.zero_grad(set_to_none=True)
            logits, exists = forward(indices, variants, True)
            target = torch.from_numpy(labels[indices]).cuda()
            valid = torch.from_numpy(present[indices]).cuda() & exists
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[valid], target[valid].float(), reduction="none",
            )
            bce = (bce * class_weight[target[valid]]).mean()
            rank = pairwise_rank_loss(logits[valid].float(), target[valid])
            loss = bce + (0.20 * rank if arm == "lora_rank" else 0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]], 5.0,
            )
            optimizer.step()
            total_loss += float(loss.detach()) * int(valid.sum())
            total_kidneys += int(valid.sum())
        row = {"epoch": epoch, "loss": total_loss / total_kidneys}
        if select_epoch:
            inner, _ = score(score_patients)
            objective = (inner["auroc"] + inner["ap"]) / 2.0
            row.update({f"inner_{name}": value for name, value in inner.items()})
            row["objective"] = objective
            if epoch >= MIN_EPOCHS and (best is None or objective > best["objective"] + 1e-12):
                best = {"epoch": epoch, "objective": objective, **inner}
                stale = 0
            elif epoch >= MIN_EPOCHS:
                stale += 1
            if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                records.append(row)
                break
        records.append(row)
        if epoch == 1 or epoch % 6 == 0:
            print(
                f"{arm} seed={seed} {'inner' if select_epoch else 'refit'} "
                f"epoch={epoch} loss={row['loss']:.4f} elapsed={time.monotonic()-start_time:.0f}s",
                flush=True,
            )
    if select_epoch:
        if best is None:
            raise RuntimeError("No inner epoch selected")
        result = {"selected_epoch": best["epoch"], "inner": best, "records": records}
        del encoder, head, optimizer
        torch.cuda.empty_cache()
        return result
    development_metric, values = score(score_patients)
    state = {
        "head": {name: value.detach().cpu() for name, value in head.state_dict().items()},
        "fc_norm": {name: value.detach().cpu() for name, value in encoder.fc_norm.state_dict().items()},
        "adapter": {
            name: value.detach().cpu() for name, value in encoder.state_dict().items() if "lora_" in name
        },
    }
    result = {
        "metrics": development_metric, "scores": values, "records": records,
        "trainable_encoder_parameters": int(sum(
            parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
        )),
        "trainable_encoder_names": trainable,
    }
    del encoder, head, optimizer
    torch.cuda.empty_cache()
    return result, state


def evaluate() -> None:
    if not (CACHE / "lower8_float16.npy").is_file():
        raise FileNotFoundError("Run cache stage first")
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed experiment")
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    patients, kidneys, images = cohort.load_reference()
    design = patient_design(patients, kidneys, images)
    image_index, image_side, labels, present = design
    lower = np.load(CACHE / "lower8_float16.npy", mmap_mode="r")
    selections = np.load(CACHE / "patch_selection.npy", mmap_mode="r")
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    reference = pd.read_csv(REFERENCE)
    reference = reference[reference.group.eq(URI_GROUP)].set_index(["seed", "kidney_alias"])
    predictions, metrics_rows, training_rows = [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        roles = patients.patient_alias.map(mapping)
        train = np.flatnonzero(roles.eq("train"))
        development_patients = np.flatnonzero(roles.eq("development"))
        inner_train, inner_development = train_test_split(
            train, train_size=100, random_state=seed + 4000,
            stratify=patients.DMSA_APN.to_numpy()[train],
        )
        development_kidneys = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        development_frame = kidneys.iloc[development_kidneys]
        fixed_uri = reference.loc[
            [(seed, alias) for alias in development_frame.kidney_alias], "score"
        ].to_numpy(float)
        draw_scores = {"official_uri_fixed_reference": fixed_uri}
        for arm in ARMS:
            selection = train_once(
                seed, arm, inner_train, inner_development, MAX_EPOCHS, True,
                design, lower, selections,
            )
            selected_epoch = int(selection["selected_epoch"])
            final, state = train_once(
                seed, arm, train, development_patients, selected_epoch, False,
                design, lower, selections,
            )
            torch.save(state, OUT / "models" / f"{seed}_{arm}.pth")
            patient_scores = final.pop("scores")
            score_lookup = {
                (patients.patient_alias.iloc[index], side): patient_scores[local, side]
                for local, index in enumerate(development_patients) for side in (0, 1)
            }
            score = np.asarray([
                score_lookup[(row.patient_alias, 0 if row.side == "left" else 1)]
                for row in development_frame.itertuples(index=False)
            ])
            group = f"urfm_{arm}_tta"
            draw_scores[group] = score
            draw_scores[f"{group}_20pct_plus_uri"] = 0.20 * score + 0.80 * fixed_uri
            draw_scores[f"{group}_25pct_plus_uri"] = 0.25 * score + 0.75 * fixed_uri
            training_rows.append({
                "seed": seed, "arm": arm, "selected_epoch": selected_epoch,
                "inner_auroc": selection["inner"]["auroc"],
                "inner_ap": selection["inner"]["ap"],
                "inner_objective": selection["inner"]["objective"],
                "trainable_encoder_parameters": final["trainable_encoder_parameters"],
                "trainable_encoder_names": "|".join(final["trainable_encoder_names"]),
            })
        for group, score in draw_scores.items():
            current_metric = metric(development_frame.kidney_APN.to_numpy(int), score)
            metrics_rows.append({
                "seed": seed, "role": "development", "group": group,
                "patients": development_frame.patient_alias.nunique(),
                "kidneys": len(development_frame), **current_metric,
            })
            for row, current_score in zip(development_frame.itertuples(index=False), score):
                predictions.append({
                    "seed": seed, "group": group, "patient_alias": row.patient_alias,
                    "kidney_alias": row.kidney_alias, "DMSA_APN": row.DMSA_APN,
                    "side": row.side, "label": int(row.kidney_APN), "score": float(current_score),
                })
        print(f"URFM supervised draw {seed} complete", flush=True)
    prediction_frame = pd.DataFrame(predictions)
    metric_frame = pd.DataFrame(metrics_rows)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(training_rows).to_csv(OUT / "training_records.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv")
    groups = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups, SEEDS, "official_uri_fixed_reference",
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(OUT / "decision.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "groups_above_both_development_targets": achieved,
        "promotion": bool(achieved),
        "exploratory_test_rows_scored": 0,
        "split_unit": "patient",
        "inner_selection_patients": 26,
        "development_patients_per_draw": 42,
        "motivation": "Official URFM prescribes supervised downstream fine-tuning; LoRA limits capacity for the small NCKUH cohort.",
    })
    print(summary[["auroc_mean", "ap_mean", "brier_mean"]].to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cache", "evaluate"))
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    {"cache": cache_tokens, "evaluate": evaluate}[arguments.stage]()
