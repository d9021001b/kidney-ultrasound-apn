"""Frozen official URFM ViT-B representations for contour-first NCKUH APN."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import torch.nn as nn

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "third_party/URFM"
WEIGHT = MODEL_DIR / "weights/mae_vit_base_patch16_dec768d8b_all_biomedclip_1199.pth"
OUT = ROOT / "02_experiments/NCKUH_DMSA_urfm_frozen_20261106"
REFERENCE = uri.OUT / "development_predictions.csv"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"
MODEL_REVISION = "5fd405798e6ff604d4addeca5559ce0cb7608e4b"
EXPECTED_WEIGHT_BYTES = 2_299_362_921
WEIGHT_SHA256 = "0799de60ecfbbdd0bb15ae4c35b35e784d96b45f63ffb84490499ef4f00ed59d"
MODEL_FACTORY_NAME = "vit_base_patch16"
ARCHITECTURE_NAME = "URFM ViT-B/16 with official global pooling"
PATCH_SIZE = 16
PATCH_GRID = 14
EMBED_DIM = 768
TRANSFER_NORM_TO_FC_NORM = False
DROP_PATH_RATE = 0.0
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def contour_canvases(images: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    colors: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for row in images.itertuples(index=False):
        color = cv2.imread(str(base.OUT / "clean_rgb" / f"{row.image_id}.png"))
        mask = cv2.imread(
            str(base.OUT / "masks" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE
        )
        if color is None or mask is None or color.shape[:2] != mask.shape:
            raise ValueError(f"Unreadable contour-first input: {row.image_id}")
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        canvas, canvas_mask = base.tight_canvas(color, mask > 0, size=224)
        colors.append(canvas)
        masks.append(canvas_mask)
    return np.stack(colors), np.stack(masks)


def summarize_tokens(
    tokens: torch.Tensor, coverage: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mask-covered mean/SD/P90/max and selected patch counts."""
    if tokens.ndim != 4 or coverage.shape != tokens.shape[:3]:
        raise ValueError("Expected BxHxWxD tokens and matching BxHxW coverage")
    rows: list[torch.Tensor] = []
    counts: list[int] = []
    for current, current_coverage in zip(tokens, coverage):
        selected = current[current_coverage >= 0.25]
        if len(selected) == 0:
            selected = current.reshape(-1, current.shape[-1])[
                current_coverage.reshape(-1).argmax().reshape(1)
            ]
        counts.append(len(selected))
        rows.append(
            torch.cat(
                [
                    selected.mean(0),
                    selected.std(0, unbiased=False),
                    torch.quantile(selected, 0.90, dim=0),
                    selected.max(0).values,
                ]
            )
        )
    return torch.stack(rows), torch.tensor(counts, device=tokens.device)


def urfm_features(
    model: torch.nn.Module, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce official global pooling and retain final mask-aligned patches."""
    tokens = model.patch_embed(images)
    cls_tokens = model.cls_token.expand(len(images), -1, -1)
    tokens = torch.cat((cls_tokens, tokens), dim=1)
    tokens = model.pos_drop(tokens + model.pos_embed)
    for block in model.blocks:
        tokens = block(tokens)
    patches = tokens[:, 1:]
    global_embedding = model.fc_norm(patches.mean(dim=1))
    patch_embedding = model.fc_norm(patches)
    if patch_embedding.ndim != 3 or patch_embedding.shape[1] != PATCH_GRID**2:
        raise ValueError(f"Unexpected URFM patch shape: {tuple(patch_embedding.shape)}")
    return global_embedding, patch_embedding.reshape(
        len(images), PATCH_GRID, PATCH_GRID, patch_embedding.shape[-1]
    )


def checkpoint_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    """Extract the state dict by the same precedence as official fine-tuning."""
    key = "state_dict" if "state_dict" in checkpoint else "model"
    if key not in checkpoint or not isinstance(checkpoint[key], dict):
        raise ValueError("URFM checkpoint has no model state dictionary")
    return {name.removeprefix("module."): value for name, value in checkpoint[key].items()}


def transfer_global_pool_norm(
    state: dict[str, torch.Tensor], current: dict[str, torch.Tensor]
) -> dict[str, str]:
    """Optionally initialize global-pool LayerNorm from pretrained token norm."""
    transferred: dict[str, str] = {}
    if not TRANSFER_NORM_TO_FC_NORM:
        return transferred
    for suffix in ("weight", "bias"):
        source = f"norm.{suffix}"
        target = f"fc_norm.{suffix}"
        if (
            source in state
            and target in current
            and state[source].shape == current[target].shape
        ):
            state[target] = state[source].clone()
            transferred[target] = source
    return transferred


def build_encoder(models_vit) -> torch.nn.Module:
    """Build released URFM encoders, including the undocumented ViT-H/16 file."""
    factory = getattr(models_vit, MODEL_FACTORY_NAME, None)
    if factory is not None:
        return factory(num_classes=2, drop_path_rate=DROP_PATH_RATE, global_pool=True)
    if MODEL_FACTORY_NAME != "vit_huge_patch16":
        raise ValueError(f"Unknown URFM encoder factory: {MODEL_FACTORY_NAME}")
    return models_vit.VisionTransformer(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=2,
        drop_path_rate=DROP_PATH_RATE,
        global_pool=True,
    )


def load_encoder() -> tuple[torch.nn.Module, dict]:
    if WEIGHT.stat().st_size != EXPECTED_WEIGHT_BYTES:
        raise ValueError("URFM official weight byte count mismatch")
    if digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError("URFM official weight hash mismatch")
    sys.path.insert(0, str(MODEL_DIR))
    import models_vit
    from util.pos_embed import interpolate_pos_embed

    checkpoint = torch.load(WEIGHT, map_location="cpu", weights_only=True)
    state = checkpoint_state(checkpoint)
    model = build_encoder(models_vit)
    current = model.state_dict()
    transferred_norm = transfer_global_pool_norm(state, current)
    removed: dict[str, list[int]] = {}
    for key in ("head.weight", "head.bias"):
        if key in state and state[key].shape != current[key].shape:
            removed[key] = list(state[key].shape)
            del state[key]
    interpolate_pos_embed(model, state)
    message = model.load_state_dict(state, strict=False)
    audit = {
        "checkpoint_top_level_keys": sorted(checkpoint),
        "checkpoint_model_tensors": len(state),
        "removed_incompatible_heads": removed,
        "transferred_global_pool_norm": transferred_norm,
        "missing_keys": list(message.missing_keys),
        "unexpected_keys": list(message.unexpected_keys),
    }
    return model.eval().requires_grad_(False), audit


def encode() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Preserve completed or partial URFM experiment: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    patients, kidneys, images = cohort.load_reference()
    colors, masks = contour_canvases(images)
    model, load_audit = load_encoder()
    model = model.cuda()
    global_rows: list[np.ndarray] = []
    raw_rows: list[np.ndarray] = []
    normalized_rows: list[np.ndarray] = []
    selected_counts: list[int] = []
    coverage_fallbacks = 0
    mean = torch.from_numpy(MEAN).cuda()[None, :, None, None]
    std = torch.from_numpy(STD).cuda()[None, :, None, None]
    with torch.inference_mode():
        for start in range(0, len(colors), 32):
            current = torch.from_numpy(
                colors[start : start + 32].transpose(0, 3, 1, 2).copy()
            ).cuda().float() / 255.0
            current = (current - mean) / std
            global_embedding, raw_tokens = urfm_features(model, current)
            global_embedding = functional.normalize(global_embedding, dim=-1)
            normalized_tokens = functional.normalize(raw_tokens, dim=-1)
            current_mask = torch.from_numpy(masks[start : start + 32]).cuda().float()[:, None]
            coverage = functional.avg_pool2d(current_mask, PATCH_SIZE, PATCH_SIZE)[:, 0]
            raw_summary, counts = summarize_tokens(raw_tokens, coverage)
            normalized_summary, normalized_counts = summarize_tokens(
                normalized_tokens, coverage
            )
            if not torch.equal(counts, normalized_counts):
                raise AssertionError("Raw and normalized patch selections diverged")
            available = (coverage >= 0.25).sum((1, 2))
            coverage_fallbacks += int((available == 0).sum().item())
            selected_counts.extend(counts.cpu().tolist())
            global_rows.append(global_embedding.cpu().numpy())
            raw_rows.append(raw_summary.cpu().numpy())
            normalized_rows.append(normalized_summary.cpu().numpy())
            print(f"URFM encoded {min(start + 32, len(colors))}/820", flush=True)

    global_array = np.concatenate(global_rows).astype(np.float32)
    raw_array = np.concatenate(raw_rows).astype(np.float32)
    normalized_array = np.concatenate(normalized_rows).astype(np.float32)
    if global_array.shape != (820, EMBED_DIM):
        raise ValueError(f"Unexpected global shape: {global_array.shape}")
    expected_summary_shape = (820, 4 * EMBED_DIM)
    if raw_array.shape != expected_summary_shape or normalized_array.shape != expected_summary_shape:
        raise ValueError("Unexpected URFM patch-summary shape")
    if not all(np.isfinite(value).all() for value in (global_array, raw_array, normalized_array)):
        raise ValueError("Nonfinite URFM representation")
    np.save(OUT / "urfm_global.npy", global_array)
    np.save(OUT / "urfm_patch_raw.npy", raw_array)
    np.save(OUT / "urfm_patch_normalized.npy", normalized_array)
    write_json(
        OUT / "encoder_audit.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "source": "QingboKang/URFM",
            "repository_revision": MODEL_REVISION,
            "license": "MIT",
            "official_weight_sha256": digest(WEIGHT),
            "official_weight_bytes": WEIGHT.stat().st_size,
            "architecture": ARCHITECTURE_NAME,
            "pretraining": "Multi-organ ultrasound representation MIM with BioMedCLIP semantic targets; includes kidney ultrasound",
            "motivation": "Ultrasound-domain semantic pretraining may recover kidney texture signals absent from generic or radiograph encoders",
            "load_audit": load_audit,
            "images": len(images),
            "global_shape": list(global_array.shape),
            "patch_raw_shape": list(raw_array.shape),
            "patch_normalized_shape": list(normalized_array.shape),
            "selected_patches_min_median_max": [
                min(selected_counts),
                float(np.median(selected_counts)),
                max(selected_counts),
            ],
            "max_coverage_patch_fallbacks": coverage_fallbacks,
            "preprocessing": "Contour-first 224x224 RGB tight canvas; official ImageNet mean/SD; original orientation only",
            "patch_rule": f"{PATCH_GRID}x{PATCH_GRID} final {EMBED_DIM}-D tokens; mask coverage >=0.25; maximum-coverage fallback",
            "labels_used_in_encoding": [],
        },
    )
    del model
    torch.cuda.empty_cache()


def evaluate() -> None:
    if not (OUT / "urfm_global.npy").is_file():
        raise FileNotFoundError("Run the encode stage first")
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed URFM evaluation")
    (OUT / "models").mkdir(exist_ok=True)
    patients, kidneys, images = cohort.load_reference()
    global_array = np.load(OUT / "urfm_global.npy")
    raw_array = np.load(OUT / "urfm_patch_raw.npy")
    normalized_array = np.load(OUT / "urfm_patch_normalized.npy")
    representations = {
        "urfm_global": global_array,
        "urfm_patch_raw": raw_array,
        "urfm_patch_normalized": normalized_array,
        "urfm_global_patch_raw": np.c_[global_array, raw_array],
        "urfm_global_patch_normalized": np.c_[global_array, normalized_array],
    }
    structured = {
        name: hierarchy.structured_views(base.patient_pair_features(values, images, patients))
        for name, values in representations.items()
    }
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    references = pd.read_csv(REFERENCE)
    references = references[references.group.eq(URI_GROUP)].set_index(
        ["seed", "kidney_alias"], verify_integrity=True
    )
    metrics: list[dict] = []
    predictions: list[dict] = []
    selections: list[dict] = []
    for seed in base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        patient_roles = patients.patient_alias.map(mapping)
        kidney_roles = kidneys.patient_alias.map(mapping)
        train = np.flatnonzero(patient_roles.eq("train"))
        development = np.flatnonzero(kidney_roles.eq("development"))
        target = kidneys.iloc[development]
        fixed_uri = references.loc[
            [(seed, alias) for alias in target.kidney_alias], "score"
        ].to_numpy(float)
        draw_scores = {"official_uri_fixed_reference": fixed_uri}
        for name, (symmetric, antisymmetric) in structured.items():
            probability, models = hierarchy.state_probabilities(
                symmetric, antisymmetric, states, train, seed
            )
            score = hierarchy.kidney_scores(probability, patients, kidneys)[development]
            draw_scores[name] = score
            draw_scores[f"{name}_20pct_plus_uri"] = 0.20 * score + 0.80 * fixed_uri
            draw_scores[f"{name}_25pct_plus_uri"] = 0.25 * score + 0.75 * fixed_uri
            joblib.dump(models, OUT / "models" / f"{seed}_{name}.joblib", compress=3)
            for head, model in models.items():
                selections.append(
                    {
                        "seed": seed,
                        "source": name,
                        "head": head,
                        "inner_cv_auroc": float(model.best_score_),
                        **model.best_params_,
                    }
                )
        for group, score in draw_scores.items():
            result = base.metric(labels[development], score)
            metrics.append(
                {
                    "seed": seed,
                    "role": "development",
                    "group": group,
                    "patients": target.patient_alias.nunique(),
                    "kidneys": len(target),
                    **result,
                }
            )
            for local, index in enumerate(development):
                predictions.append(
                    {
                        "seed": seed,
                        "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[index],
                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                        "side": kidneys.side.iloc[index],
                        "label": int(labels[index]),
                        "score": float(score[local]),
                    }
                )
        print(f"URFM development draw {seed} complete", flush=True)
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv")
    pd.DataFrame(selections).to_csv(OUT / "training_inner_cv_selections.csv", index=False)
    groups = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups, base.SEEDS, "official_uri_fixed_reference"
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(
        OUT / "decision.json",
        {
            "groups_above_both_development_targets": achieved,
            "promotion": bool(achieved),
            "reference": "official_uri_fixed_reference",
            "exploratory_test_rows_scored": 0,
        },
    )
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    import argparse
    from threadpoolctl import threadpool_limits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("encode", "evaluate"))
    arguments = parser.parse_args()
    with threadpool_limits(limits=4):
        {"encode": encode, "evaluate": evaluate}[arguments.stage]()
