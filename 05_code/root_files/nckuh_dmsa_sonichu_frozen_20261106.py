"""Frozen SONICHU I-JEPA ultrasound representations for NCKUH APN."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "third_party/SONICHU"
WEIGHT = MODEL_DIR / "model.safetensors"
OUT = ROOT / "02_experiments/NCKUH_DMSA_sonichu_frozen_20261106"
REFERENCE = uri.OUT / "development_predictions.csv"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"
WEIGHT_SHA256 = "61c3a1fb017235727f90c8797ef554fc0586223aa309bc4bea31547cdf1b217b"
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
        canvas, canvas_mask = base.tight_canvas(color, mask > 0, size=256)
        colors.append(canvas)
        masks.append(canvas_mask)
    return np.stack(colors), np.stack(masks)


def summarize_tokens(tokens: torch.Tensor, coverage: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global mean and mask-covered mean/SD/P90/max descriptors."""
    if tokens.ndim != 4 or coverage.shape != tokens.shape[:3]:
        raise ValueError("Expected BxHxWxD tokens and matching BxHxW coverage")
    global_rows: list[torch.Tensor] = []
    masked_rows: list[torch.Tensor] = []
    for current, current_coverage in zip(tokens, coverage):
        selected = current[current_coverage >= 0.25]
        if len(selected) == 0:
            selected = current.reshape(-1, current.shape[-1])[
                current_coverage.reshape(-1).argmax().reshape(1)
            ]
        global_rows.append(current.mean((0, 1)))
        masked_rows.append(
            torch.cat(
                [
                    selected.mean(0),
                    selected.std(0, unbiased=False),
                    torch.quantile(selected, 0.90, dim=0),
                    selected.max(0).values,
                ]
            )
        )
    return torch.stack(global_rows), torch.stack(masked_rows)


def encode() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Preserve completed or partial SONICHU experiment: {OUT}")
    if digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError("SONICHU official weight hash mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    patients, kidneys, images = cohort.load_reference()
    colors, masks = contour_canvases(images)
    sys.path.insert(0, str(MODEL_DIR))
    from modeling_sonichu import SonichuModel

    model = SonichuModel.from_pretrained(str(MODEL_DIR)).cuda().eval().requires_grad_(False)
    global_rows: list[np.ndarray] = []
    masked_rows: list[np.ndarray] = []
    selected_counts: list[int] = []
    coverage_fallbacks = 0
    mean = torch.from_numpy(MEAN).cuda()[None, :, None, None]
    std = torch.from_numpy(STD).cuda()[None, :, None, None]
    with torch.inference_mode():
        for start in range(0, len(colors), 24):
            current = torch.from_numpy(
                colors[start : start + 24].transpose(0, 3, 1, 2).copy()
            ).cuda().float() / 255.0
            current = (current - mean) / std
            original = model.backbone(current).reshape(len(current), 16, 16, 768)
            flipped = model.backbone(torch.flip(current, dims=[3])).reshape(
                len(current), 16, 16, 768
            )
            tokens = (original + torch.flip(flipped, dims=[2])) / 2.0
            current_mask = torch.from_numpy(masks[start : start + 24]).cuda().float()[:, None]
            coverage = torch.nn.functional.avg_pool2d(current_mask, 16, 16)[:, 0]
            global_feature, masked_feature = summarize_tokens(tokens, coverage)
            global_rows.append(global_feature.cpu().numpy())
            masked_rows.append(masked_feature.cpu().numpy())
            counts = (coverage >= 0.25).sum((1, 2))
            coverage_fallbacks += int((counts == 0).sum().item())
            selected_counts.extend(torch.clamp(counts, min=1).cpu().tolist())
            print(f"SONICHU encoded {min(start + 24, len(colors))}/820", flush=True)
    global_array = np.concatenate(global_rows).astype(np.float32)
    masked_array = np.concatenate(masked_rows).astype(np.float32)
    if global_array.shape != (820, 768) or masked_array.shape != (820, 3072):
        raise ValueError("Unexpected SONICHU representation shapes")
    if not np.isfinite(global_array).all() or not np.isfinite(masked_array).all():
        raise ValueError("Nonfinite SONICHU representation")
    np.save(OUT / "sonichu_global_tta.npy", global_array)
    np.save(OUT / "sonichu_masked_tta.npy", masked_array)
    write_json(
        OUT / "encoder_audit.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "source": "chrisvoncsefalvay/sonichu",
            "repository_revision": "28dfddc6b7df82f82928526318ffcecb46d8787e",
            "license": "Apache-2.0; upstream training datasets retain their stated terms",
            "official_weight_sha256": digest(WEIGHT),
            "official_weight_bytes": WEIGHT.stat().st_size,
            "pretraining": "I-JEPA on 160,486 fetal/cardiac ultrasound frames; not kidney-specific",
            "images": len(images),
            "global_shape": list(global_array.shape),
            "masked_shape": list(masked_array.shape),
            "selected_patches_min_median_max": [
                min(selected_counts),
                float(np.median(selected_counts)),
                max(selected_counts),
            ],
            "max_coverage_patch_fallbacks": coverage_fallbacks,
            "preprocessing": "Contour-first 256x256 RGB tight canvas; official ImageNet mean/SD; aligned horizontal-flip token average",
            "labels_used_in_encoding": [],
        },
    )
    del model
    torch.cuda.empty_cache()


def evaluate() -> None:
    if not (OUT / "sonichu_global_tta.npy").is_file():
        raise FileNotFoundError("Run the encode stage first")
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed SONICHU evaluation")
    (OUT / "models").mkdir(exist_ok=True)
    patients, kidneys, images = cohort.load_reference()
    global_array = np.load(OUT / "sonichu_global_tta.npy")
    masked_array = np.load(OUT / "sonichu_masked_tta.npy")
    representations = {
        "sonichu_global_tta": global_array,
        "sonichu_masked_tta": masked_array,
        "sonichu_global_masked_tta": np.c_[global_array, masked_array],
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
        print(f"SONICHU development draw {seed} complete", flush=True)
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
