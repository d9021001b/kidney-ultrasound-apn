"""Evaluate the official five-fold URI-CADS global renal pathology encoders."""

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
import torchvision.models as tvm
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from torch import nn

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_multiquery_mil_20261103 as cohort
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_uricads_transfer_20261104 import LABEL_NAMES, parse_label


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "third_party/URI-CADS/bbdd"
MODEL_DIR = ROOT / "third_party/URI-CADS/models"
MODEL_ZIP = ROOT / "third_party/URI-CADS/downloads/models.zip"
CURRENT = ROOT / "02_experiments/NCKUH_DMSA_hierarchical_tile_usfm_20261104/development_predictions.csv"
OUT = ROOT / "02_experiments/NCKUH_DMSA_uricads_official_frozen_20261105"
MODEL_PATHS = tuple(MODEL_DIR / f"URICADS_max_fold{fold}-epoch95.pth" for fold in range(1, 6))


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


class OfficialGlobalEncoder(nn.Module):
    def __init__(self, checkpoint: Path):
        super().__init__()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        backbone_state = {
            key.removeprefix("backbone.body."): value
            for key, value in state.items()
            if key.startswith("backbone.body.")
        }
        backbone = tvm.resnet50(weights=None)
        result = backbone.load_state_dict(backbone_state, strict=False)
        if set(result.missing_keys) != {"fc.weight", "fc.bias"} or result.unexpected_keys:
            raise RuntimeError(
                f"Unexpected URI-CADS backbone mapping: {result.missing_keys}, {result.unexpected_keys}"
            )
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, 7)
        self.classifier.weight.data.copy_(state["classifier.1.weight"])
        self.classifier.bias.data.copy_(state["classifier.1.bias"])

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.pool(self.features(images)).flatten(1)
        return embedding, torch.sigmoid(self.classifier(embedding))


def public_normalization() -> tuple[np.ndarray, np.ndarray]:
    saved = OUT / "public_rgb_normalization.json"
    if saved.exists():
        values = json.loads(saved.read_text(encoding="utf-8"))
        return np.asarray(values["mean"], np.float32), np.asarray(values["std"], np.float32)
    channel_sum = np.zeros(3, np.float64)
    channel_square = np.zeros(3, np.float64)
    pixels = 0
    for path in sorted((PUBLIC / "images").glob("*.jpg"), key=lambda item: int(item.stem)):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255
        values = rgb.reshape(-1, 3)
        channel_sum += values.sum(0)
        channel_square += np.square(values).sum(0)
        pixels += len(values)
    mean = channel_sum / pixels
    std = np.sqrt(np.maximum(channel_square / pixels - np.square(mean), 1e-12))
    write_json(
        saved,
        {"mean": mean.tolist(), "std": std.tolist(), "pixels": pixels, "images": 1985},
    )
    return mean.astype(np.float32), std.astype(np.float32)


def nckuh_canvases(images: pd.DataFrame, size: int = 800) -> np.ndarray:
    rows = []
    for identifier in images.image_id:
        bgr = cv2.imread(str(base.OUT / "clean_rgb" / f"{identifier}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(base.OUT / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE) > 0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        canvas, _ = base.tight_canvas(rgb, mask, size=size)
        rows.append(canvas)
    return np.stack(rows)


def encode_batches(
    model: OfficialGlobalEncoder,
    canvases: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings, scores = [], []
    mean_tensor = torch.from_numpy(mean).cuda().view(1, 3, 1, 1)
    std_tensor = torch.from_numpy(std).cuda().view(1, 3, 1, 1)
    model = model.cuda().eval().requires_grad_(False)
    with torch.inference_mode():
        for start in range(0, len(canvases), batch_size):
            tensor = torch.from_numpy(
                canvases[start : start + batch_size].transpose(0, 3, 1, 2).copy()
            ).cuda().float() / 255
            embedding, score = model((tensor - mean_tensor) / std_tensor)
            embeddings.append(embedding.cpu().numpy())
            scores.append(score.cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(scores)


def encode() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve completed encoder output: {OUT}")
    OUT.mkdir(parents=True)
    mean, std = public_normalization()
    _, _, images = cohort.load_reference()
    canvases = nckuh_canvases(images)
    fold_embeddings, fold_scores = [], []
    model_audit = []
    for fold, checkpoint in enumerate(MODEL_PATHS, start=1):
        model = OfficialGlobalEncoder(checkpoint)
        embedding, scores = encode_batches(model, canvases, mean, std)
        if embedding.shape != (820, 2048) or scores.shape != (820, 7):
            raise ValueError(f"Unexpected fold geometry: {fold}")
        fold_embeddings.append(embedding)
        fold_scores.append(scores)
        model_audit.append(
            {
                "fold": fold,
                "checkpoint": checkpoint.name,
                "sha256": digest(checkpoint),
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
            }
        )
        print(f"Official URI-CADS fold {fold}/5 encoded", flush=True)
        del model
        torch.cuda.empty_cache()
    embeddings = np.stack(fold_embeddings, axis=1).astype(np.float32)
    scores = np.stack(fold_scores, axis=1).astype(np.float32)
    np.save(OUT / "official_fold_embeddings.npy", embeddings)
    np.save(OUT / "official_fold_pathology_scores.npy", scores)
    write_json(
        OUT / "encoder_audit.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "official_model_zip_md5": digest(MODEL_ZIP, "md5"),
            "models": model_audit,
            "nckuh_images": 820,
            "input_size": 800,
            "normalization": {"mean": mean.tolist(), "std": std.tolist()},
            "labels_used_in_encoding": [],
            "embedding_shape": list(embeddings.shape),
            "score_shape": list(scores.shape),
            "class_order": list(LABEL_NAMES),
        },
    )


def representations() -> dict[str, np.ndarray]:
    embedding = np.load(OUT / "official_fold_embeddings.npy")
    scores = np.load(OUT / "official_fold_pathology_scores.npy")
    values = {
        "official_uri_score_mean": scores.mean(1),
        "official_uri_score_folds": scores.reshape(len(scores), -1),
        "official_uri_embedding_mean": embedding.mean(1),
        "official_uri_embedding_folds": embedding.reshape(len(embedding), -1),
    }
    for fold in range(5):
        values[f"official_uri_embedding_fold{fold + 1}"] = embedding[:, fold]
    return values


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
        print(f"Official URI-CADS development draw {seed} complete", flush=True)
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
