"""Frozen BioMedCLIP teacher representations for contour-first NCKUH APN."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "third_party/BiomedCLIP"
WEIGHT = MODEL_DIR / "open_clip_pytorch_model.bin"
CACHE_DIR = MODEL_DIR / "cache"
OUT = ROOT / "02_experiments/NCKUH_DMSA_biomedclip_frozen_20261106"
REFERENCE = uri.OUT / "development_predictions.csv"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"
MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
MODEL_REVISION = "9f341de24bfb00180f1b847274256e9b65a3a32e"
WEIGHT_SHA256 = "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be"
MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


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


def projected_patch_tokens(visual: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Apply the official BioMedCLIP visual projection to all non-CLS tokens."""
    features = visual.trunk.forward_features(images)
    if features.ndim != 3 or features.shape[1:] != (197, 768):
        raise ValueError(f"Unexpected BioMedCLIP token shape: {tuple(features.shape)}")
    projected = visual.head(features[:, 1:])
    if projected.shape[1:] != (196, 512):
        raise ValueError(f"Unexpected projected token shape: {tuple(projected.shape)}")
    return projected.reshape(len(images), 14, 14, 512)


def encode() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Preserve completed or partial BioMedCLIP experiment: {OUT}")
    if digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError("BioMedCLIP official weight hash mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    patients, kidneys, images = cohort.load_reference()
    colors, masks = contour_canvases(images)

    import open_clip

    model, _ = open_clip.create_model_from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
    visual = model.visual.cuda().eval().requires_grad_(False)
    del model
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
            global_embedding = functional.normalize(visual(current), dim=-1)
            raw_tokens = projected_patch_tokens(visual, current)
            normalized_tokens = functional.normalize(raw_tokens, dim=-1)
            current_mask = torch.from_numpy(masks[start : start + 32]).cuda().float()[:, None]
            coverage = functional.avg_pool2d(current_mask, 16, 16)[:, 0]
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
            print(f"BioMedCLIP encoded {min(start + 32, len(colors))}/820", flush=True)

    global_array = np.concatenate(global_rows).astype(np.float32)
    raw_array = np.concatenate(raw_rows).astype(np.float32)
    normalized_array = np.concatenate(normalized_rows).astype(np.float32)
    if global_array.shape != (820, 512):
        raise ValueError(f"Unexpected global shape: {global_array.shape}")
    if raw_array.shape != (820, 2048) or normalized_array.shape != (820, 2048):
        raise ValueError("Unexpected BioMedCLIP patch-summary shape")
    if not all(np.isfinite(value).all() for value in (global_array, raw_array, normalized_array)):
        raise ValueError("Nonfinite BioMedCLIP representation")
    np.save(OUT / "biomedclip_global.npy", global_array)
    np.save(OUT / "biomedclip_teacher_raw.npy", raw_array)
    np.save(OUT / "biomedclip_teacher_normalized.npy", normalized_array)
    write_json(
        OUT / "encoder_audit.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "source": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            "repository_revision": MODEL_REVISION,
            "license": "MIT",
            "official_weight_sha256": digest(WEIGHT),
            "official_weight_bytes": WEIGHT.stat().st_size,
            "pretraining": "15 million biomedical image-text pairs; not ultrasound-specific",
            "motivation": "BioMedCLIP patch tokens are the semantic reconstruction target used by the public URFM pretraining method",
            "images": len(images),
            "global_shape": list(global_array.shape),
            "teacher_raw_shape": list(raw_array.shape),
            "teacher_normalized_shape": list(normalized_array.shape),
            "selected_patches_min_median_max": [
                min(selected_counts),
                float(np.median(selected_counts)),
                max(selected_counts),
            ],
            "max_coverage_patch_fallbacks": coverage_fallbacks,
            "preprocessing": "Contour-first 224x224 RGB tight canvas; official BioMedCLIP mean/SD; original orientation only",
            "patch_rule": "14x14 projected 512-D tokens; mask coverage >=0.25; maximum-coverage fallback",
            "labels_used_in_encoding": [],
        },
    )
    del visual
    torch.cuda.empty_cache()


def evaluate() -> None:
    if not (OUT / "biomedclip_global.npy").is_file():
        raise FileNotFoundError("Run the encode stage first")
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed BioMedCLIP evaluation")
    (OUT / "models").mkdir(exist_ok=True)
    patients, kidneys, images = cohort.load_reference()
    global_array = np.load(OUT / "biomedclip_global.npy")
    raw_array = np.load(OUT / "biomedclip_teacher_raw.npy")
    normalized_array = np.load(OUT / "biomedclip_teacher_normalized.npy")
    representations = {
        "biomedclip_global": global_array,
        "biomedclip_teacher_raw": raw_array,
        "biomedclip_teacher_normalized": normalized_array,
        "biomedclip_global_teacher_raw": np.c_[global_array, raw_array],
        "biomedclip_global_teacher_normalized": np.c_[global_array, normalized_array],
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
        print(f"BioMedCLIP development draw {seed} complete", flush=True)
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
