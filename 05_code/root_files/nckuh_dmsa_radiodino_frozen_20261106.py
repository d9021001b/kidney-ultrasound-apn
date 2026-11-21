"""Frozen RadioDINO representations for contour-first NCKUH APN."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

import nckuh_dmsa_biomedclip_frozen_20261106 as common
import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "03_data_models/external_models/RadioDino-s16_official_20260914"
WEIGHT = MODEL_DIR / "model.safetensors"
OUT = ROOT / "02_experiments/NCKUH_DMSA_radiodino_frozen_20261106"
REFERENCE = uri.OUT / "development_predictions.csv"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"
WEIGHT_SHA256 = "6f88c66e9155a5d223d8d08a78aebc294e76deede181c4eb8b1895a70af49a01"
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


def patch_tokens(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    features = model.forward_features(images)
    if features.ndim != 3 or features.shape[1:] != (197, 384):
        raise ValueError(f"Unexpected RadioDINO token shape: {tuple(features.shape)}")
    return features[:, 1:].reshape(len(images), 14, 14, 384)


def encode() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Preserve completed or partial RadioDINO experiment: {OUT}")
    if digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError("RadioDINO official weight hash mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    patients, kidneys, images = cohort.load_reference()
    colors, masks = common.contour_canvases(images)

    import timm
    from safetensors.torch import load_file

    model = timm.create_model(
        "vit_small_patch16_224", pretrained=False, num_classes=0, global_pool="token"
    )
    result = model.load_state_dict(load_file(str(WEIGHT)), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("RadioDINO strict load failed")
    model = model.cuda().eval().requires_grad_(False)
    global_rows: list[np.ndarray] = []
    masked_rows: list[np.ndarray] = []
    selected_counts: list[int] = []
    coverage_fallbacks = 0
    mean = torch.from_numpy(MEAN).cuda()[None, :, None, None]
    std = torch.from_numpy(STD).cuda()[None, :, None, None]
    with torch.inference_mode():
        for start in range(0, len(colors), 64):
            current = torch.from_numpy(
                colors[start : start + 64].transpose(0, 3, 1, 2).copy()
            ).cuda().float() / 255.0
            current = (current - mean) / std
            tokens = patch_tokens(model, current)
            global_embedding = functional.normalize(model(current), dim=-1)
            current_mask = torch.from_numpy(masks[start : start + 64]).cuda().float()[:, None]
            coverage = functional.avg_pool2d(current_mask, 16, 16)[:, 0]
            masked_summary, counts = common.summarize_tokens(tokens, coverage)
            available = (coverage >= 0.25).sum((1, 2))
            coverage_fallbacks += int((available == 0).sum().item())
            selected_counts.extend(counts.cpu().tolist())
            global_rows.append(global_embedding.cpu().numpy())
            masked_rows.append(masked_summary.cpu().numpy())
            print(f"RadioDINO encoded {min(start + 64, len(colors))}/820", flush=True)
    global_array = np.concatenate(global_rows).astype(np.float32)
    masked_array = np.concatenate(masked_rows).astype(np.float32)
    if global_array.shape != (820, 384) or masked_array.shape != (820, 1536):
        raise ValueError("Unexpected RadioDINO representation shape")
    if not np.isfinite(global_array).all() or not np.isfinite(masked_array).all():
        raise ValueError("Nonfinite RadioDINO representation")
    np.save(OUT / "radiodino_global.npy", global_array)
    np.save(OUT / "radiodino_masked.npy", masked_array)
    write_json(
        OUT / "encoder_audit.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "source": "Snarcy/RadioDino-s16",
            "license": "CC-BY-4.0",
            "official_weight_sha256": digest(WEIGHT),
            "official_weight_bytes": WEIGHT.stat().st_size,
            "strict_loaded_tensors": 150,
            "pretraining": "DINO self-supervision on 1.35 million RadImageNet CT, MRI, and ultrasound images",
            "images": len(images),
            "global_shape": list(global_array.shape),
            "masked_shape": list(masked_array.shape),
            "selected_patches_min_median_max": [
                min(selected_counts),
                float(np.median(selected_counts)),
                max(selected_counts),
            ],
            "max_coverage_patch_fallbacks": coverage_fallbacks,
            "preprocessing": "Contour-first 224x224 RGB tight canvas; official ImageNet mean/SD; original orientation only",
            "patch_rule": "14x14 post-norm 384-D tokens; mask coverage >=0.25; maximum-coverage fallback",
            "labels_used_in_encoding": [],
        },
    )
    del model
    torch.cuda.empty_cache()


def evaluate() -> None:
    if not (OUT / "radiodino_global.npy").is_file():
        raise FileNotFoundError("Run the encode stage first")
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed RadioDINO evaluation")
    (OUT / "models").mkdir(exist_ok=True)
    patients, kidneys, images = cohort.load_reference()
    global_array = np.load(OUT / "radiodino_global.npy")
    masked_array = np.load(OUT / "radiodino_masked.npy")
    representations = {
        "radiodino_global": global_array,
        "radiodino_masked": masked_array,
        "radiodino_global_masked": np.c_[global_array, masked_array],
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
        print(f"RadioDINO development draw {seed} complete", flush=True)
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
