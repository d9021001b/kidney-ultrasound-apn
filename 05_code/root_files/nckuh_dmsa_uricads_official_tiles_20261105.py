"""Probe local kidney tiles with the official five-fold URI-CADS encoders."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_multiquery_mil_20261103 as cohort
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_uricads_official_frozen_20261105 import (
    MODEL_PATHS,
    OUT as FULL_OUT,
    OfficialGlobalEncoder,
    encode_batches,
    public_normalization,
)


ROOT = Path(__file__).resolve().parents[2]
CURRENT = ROOT / "02_experiments/NCKUH_DMSA_hierarchical_tile_usfm_20261104/development_predictions.csv"
OUT = ROOT / "02_experiments/NCKUH_DMSA_uricads_official_tiles_20261105"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def tile_canvases(
    rgb: np.ndarray, mask: np.ndarray, size: int = 800
) -> tuple[list[np.ndarray], bool]:
    y, x = np.nonzero(mask)
    rgb = rgb[y.min() : y.max() + 1, x.min() : x.max() + 1]
    mask = mask[y.min() : y.max() + 1, x.min() : x.max() + 1]
    height, width = mask.shape
    tile_height = max(1, round(height * 2 / 3))
    tile_width = max(1, round(width * 2 / 3))
    selected = []
    corners = {(a, b) for a in (0, height - tile_height) for b in (0, width - tile_width)}
    for top, left in sorted(corners):
        part = mask[top : top + tile_height, left : left + tile_width]
        if part.sum() >= 128 and part.mean() >= 0.25:
            color = rgb[top : top + tile_height, left : left + tile_width]
            canvas, _ = base.tight_canvas(color * part[..., None], part, size=size)
            selected.append(canvas)
    if selected:
        return selected, False
    canvas, _ = base.tight_canvas(rgb * mask[..., None], mask, size=size)
    return [canvas], True


def encode() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve completed tile encoding: {OUT}")
    OUT.mkdir(parents=True)
    mean, std = public_normalization()
    _, _, images = cohort.load_reference()
    fold_embeddings, fold_scores = [], []
    coverage_reference = None
    for fold, checkpoint in enumerate(MODEL_PATHS, start=1):
        model = OfficialGlobalEncoder(checkpoint).cuda().eval().requires_grad_(False)
        image_embeddings, image_scores, coverage = [], [], []
        for start in range(0, len(images), 8):
            canvases, lengths = [], []
            for identifier in images.image_id.iloc[start : start + 8]:
                bgr = cv2.imread(str(base.OUT / "clean_rgb" / f"{identifier}.png"))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(base.OUT / "masks" / f"{identifier}.png"), 0) > 0
                current, fallback = tile_canvases(rgb, mask)
                canvases.extend(current)
                lengths.append(len(current))
                coverage.append(
                    {"image_id": identifier, "tiles": len(current), "fallback": fallback}
                )
            embedding, scores = encode_batches(model, np.stack(canvases), mean, std)
            for current_embedding, current_scores in zip(
                np.split(embedding, np.cumsum(lengths)[:-1]),
                np.split(scores, np.cumsum(lengths)[:-1]),
            ):
                image_embeddings.append(current_embedding.mean(0))
                image_scores.append(current_scores.mean(0))
        fold_embeddings.append(np.asarray(image_embeddings))
        fold_scores.append(np.asarray(image_scores))
        coverage_frame = pd.DataFrame(coverage)
        if coverage_reference is None:
            coverage_reference = coverage_frame
        else:
            pd.testing.assert_frame_equal(coverage_reference, coverage_frame)
        print(f"Official URI-CADS tile fold {fold}/5 encoded", flush=True)
        del model
        torch.cuda.empty_cache()
    embeddings = np.stack(fold_embeddings, axis=1).astype(np.float32)
    scores = np.stack(fold_scores, axis=1).astype(np.float32)
    np.save(OUT / "official_tile_fold_embeddings.npy", embeddings)
    np.save(OUT / "official_tile_fold_pathology_scores.npy", scores)
    coverage_reference.to_csv(OUT / "tile_coverage.csv", index=False)
    write_json(
        OUT / "encoder_audit.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "nckuh_images": 820,
            "input_size": 800,
            "folds": 5,
            "embedding_shape": list(embeddings.shape),
            "score_shape": list(scores.shape),
            "total_tiles": int(coverage_reference.tiles.sum()),
            "fallback_images": int(coverage_reference.fallback.sum()),
            "labels_used_in_encoding": [],
        },
    )


def representations() -> dict[str, np.ndarray]:
    tile_embedding = np.load(OUT / "official_tile_fold_embeddings.npy")
    tile_scores = np.load(OUT / "official_tile_fold_pathology_scores.npy")
    full_embedding = np.load(FULL_OUT / "official_fold_embeddings.npy")
    full_scores = np.load(FULL_OUT / "official_fold_pathology_scores.npy")
    return {
        "official_uri_tile_score_mean": tile_scores.mean(1),
        "official_uri_tile_score_folds": tile_scores.reshape(len(tile_scores), -1),
        "official_uri_tile_embedding_mean": tile_embedding.mean(1),
        "official_uri_tile_embedding_folds": tile_embedding.reshape(len(tile_embedding), -1),
        "official_uri_full_tile_scores": np.c_[
            full_scores.reshape(len(full_scores), -1), tile_scores.reshape(len(tile_scores), -1)
        ],
        "official_uri_full_tile_embeddings": np.c_[
            full_embedding.reshape(len(full_embedding), -1),
            tile_embedding.reshape(len(tile_embedding), -1),
        ],
    }


def evaluate() -> None:
    patients, kidneys, images = cohort.load_reference()
    values = representations()
    structured = {
        name: hierarchy.structured_views(base.patient_pair_features(array, images, patients))
        for name, array in values.items()
    }
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    current = pd.read_csv(CURRENT)
    current = current[current.group.eq("hierarchical_tile_usfm_quarter_plus_fixed")].set_index(
        ["seed", "kidney_alias"], verify_integrity=True
    )
    metrics, predictions, selections = [], [], []
    model_dir = OUT / "development_models"
    model_dir.mkdir(exist_ok=True)
    for seed in base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        patient_roles = patients.patient_alias.map(mapping)
        kidney_roles = kidneys.patient_alias.map(mapping)
        train = np.flatnonzero(patient_roles.eq("train"))
        development = np.flatnonzero(kidney_roles.eq("development"))
        target = kidneys.iloc[development]
        current_rows = current.loc[[(seed, alias) for alias in target.kidney_alias]]
        np.testing.assert_array_equal(current_rows.label, labels[development])
        reference = current_rows.score.to_numpy()
        draw_scores = {"current_tile_usfm_reference": reference}
        for name, (symmetric, antisymmetric) in structured.items():
            probability, models = hierarchy.state_probabilities(
                symmetric, antisymmetric, states, train, seed
            )
            score = hierarchy.kidney_scores(probability, patients, kidneys)[development]
            draw_scores[name] = score
            draw_scores[f"{name}_quarter_plus_current"] = 0.25 * score + 0.75 * reference
            draw_scores[f"{name}_equal_plus_current"] = 0.50 * score + 0.50 * reference
            joblib.dump(models, model_dir / f"{seed}_{name}.joblib", compress=3)
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
            result = hierarchy.metric(labels[development], score)
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
        print(f"Official URI-CADS tile development draw {seed} complete", flush=True)
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
        prediction_frame, groups, base.SEEDS, "current_tile_usfm_reference"
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(
        OUT / "decision.json",
        {
            "groups_above_both_development_targets": achieved,
            "exploratory_test_rows_scored": 0,
            "promotion": bool(achieved),
            "reference": "hierarchical_tile_usfm_quarter_plus_fixed",
        },
    )
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("encode", "evaluate"))
    arguments = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with threadpool_limits(limits=4):
        {"encode": encode, "evaluate": evaluate}[arguments.stage]()
