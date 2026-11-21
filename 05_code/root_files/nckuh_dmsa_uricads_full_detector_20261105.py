"""Port and evaluate the complete official URI-CADS local-lesion detector."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_uri_contralateral_local_20261105 as local_study
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_uricads_full_detector_20261105_r2"
REFERENCE = (
    ROOT
    / "02_experiments/NCKUH_DMSA_uri_contralateral_local_20261105/development_predictions.csv"
)
CURRENT_GROUP = "current_tile_usfm_reference"
URI_GROUP = "official_uri_fixed_quarter_reference"
SEEDS = (20261102, 20261103, 20261104)
LOCAL_CLASSES = ("healthy", "cyst", "prominent_pyramid", "hydronephrosis", "other", "class5", "class6")
DETECTION_FEATURES = (
    "max_score",
    "mean_score",
    "count_scaled",
    "summed_area_fraction",
    "max_area_fraction",
    "score_weighted_center_x",
    "score_weighted_center_y",
    "score_weighted_width",
    "score_weighted_height",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def map_official_detector_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped = {}
    for key, value in state.items():
        target = None
        if key.startswith("backbone.body."):
            target = key
        elif key.startswith("backbone.fpn2."):
            target = key.replace("backbone.fpn2.", "backbone.fpn.", 1)
            parts = target.split(".")
            if parts[2] in {"inner_blocks", "layer_blocks"}:
                parts.insert(4, "0")
                target = ".".join(parts)
        elif key.startswith("rpn2."):
            target = key.replace("rpn2.", "rpn.", 1)
            if target == "rpn.head.conv.weight":
                target = "rpn.head.conv.0.0.weight"
            elif target == "rpn.head.conv.bias":
                target = "rpn.head.conv.0.0.bias"
        elif key.startswith("roi_heads2."):
            target = key.replace("roi_heads2.", "roi_heads.", 1)
        if target is not None:
            mapped[target] = value
    return mapped


class OfficialLocalDetector(torch.nn.Module):
    def __init__(self, checkpoint: Path):
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        self.detector = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=7,
            min_size=800,
            max_size=800,
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
            box_score_thresh=0.05,
            box_nms_thresh=0.5,
            box_detections_per_img=100,
        )
        mapped = map_official_detector_state(payload)
        expected = {
            key
            for key in payload
            if key.startswith(("backbone.body.", "backbone.fpn2.", "rpn2.", "roi_heads2."))
        }
        if len(mapped) != len(expected):
            raise RuntimeError("Not every official local-detector tensor was mapped")
        result = self.detector.load_state_dict(mapped, strict=False)
        missing = [key for key in result.missing_keys if not key.endswith("num_batches_tracked")]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"URI-CADS detector mapping mismatch: {missing}, {result.unexpected_keys}")
        self.mapped_tensors = len(mapped)
        self.mapped_parameters = int(sum(value.numel() for value in mapped.values()))
        self.register_buffer("alpha_weight", payload["alpha.0.weight"].clone())
        self.register_buffer("alpha_bias", payload["alpha.0.bias"].clone())
        self.register_buffer("beta_weight", payload["beta.0.weight"].clone())
        self.register_buffer("beta_bias", payload["beta.0.bias"].clone())

    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return self.detector(images)

    def official_fusion(
        self, global_scores: torch.Tensor, local_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        combined = torch.cat((global_scores, local_scores[:, :5]), dim=1)
        alpha = torch.relu(torch.nn.functional.linear(combined, self.alpha_weight, self.alpha_bias))
        beta = torch.relu(torch.nn.functional.linear(combined, self.beta_weight, self.beta_bias))
        epsilon = torch.finfo(global_scores.dtype).eps
        alpha_weight = (alpha + epsilon) / (alpha + beta + 2 * epsilon)
        beta_weight = (beta + epsilon) / (alpha + beta + 2 * epsilon)
        final = alpha_weight * global_scores + beta_weight * local_scores
        return final, alpha_weight, beta_weight


def aggregate_detection(
    detection: dict[str, torch.Tensor], height: int = 800, width: int = 800
) -> tuple[np.ndarray, np.ndarray]:
    labels = detection["labels"].detach().cpu().numpy().astype(int)
    scores = detection["scores"].detach().cpu().numpy().astype(np.float32)
    boxes = detection["boxes"].detach().cpu().numpy().astype(np.float32)
    local = np.zeros(7, np.float32)
    features = []
    for label in range(1, 7):
        selected = labels == label
        if not selected.any():
            features.extend([0.0] * len(DETECTION_FEATURES))
            continue
        current_scores = scores[selected]
        current_boxes = boxes[selected]
        box_width = np.clip(current_boxes[:, 2] - current_boxes[:, 0], 0, width) / width
        box_height = np.clip(current_boxes[:, 3] - current_boxes[:, 1], 0, height) / height
        areas = box_width * box_height
        total = float(np.maximum(current_scores.sum(), 1e-8))
        center_x = (current_boxes[:, 0] + current_boxes[:, 2]) / (2 * width)
        center_y = (current_boxes[:, 1] + current_boxes[:, 3]) / (2 * height)
        local[label] = float(current_scores.max())
        features.extend(
            [
                float(current_scores.max()),
                float(current_scores.mean()),
                float(min(len(current_scores), 20) / 20),
                float(areas.sum()),
                float(areas.max()),
                float(np.sum(current_scores * center_x) / total),
                float(np.sum(current_scores * center_y) / total),
                float(np.sum(current_scores * box_width) / total),
                float(np.sum(current_scores * box_height) / total),
            ]
        )
    local[0] = 1 - float(local[1:5].mean())
    return local, np.asarray(features, np.float32)


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve registered experiment: {OUT}")
    OUT.mkdir(parents=True)
    _, _, images = local_study.load_reference()
    images[["image_id", "patient_alias", "view"]].to_csv(
        OUT / "encoding_manifest.csv", index=False
    )
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "Recover complementary local-lesion evidence from the previously unused official URI-CADS Faster R-CNN branch",
        "cohort": "210 patients, 414 imaged kidneys, 820 contour-first images",
        "official_source": "https://github.com/rsingla92/URI-CADS",
        "official_commit": "5f5ef8ee4271be758dd7978964bff37f7dbbc6f9",
        "port": "Map official backbone.body, second FPN, rpn2 and roi_heads2 tensors into torchvision 0.20 Faster R-CNN; preserve 800-pixel input, public normalization, 0.05 box threshold, 0.5 NMS and 100 detections",
        "features": "Per-fold local logits, official alpha/beta global-local fusion, fusion weights, and class-specific score/count/area/location summaries",
        "evaluation": "Training-only hierarchical state heads on flattened five-fold outputs; fixed 20%/25% fusions with current and official URI references",
        "roles": "Unchanged three 126/42/42 patient random assignments; exploratory test unopened",
        "labels_used_in_encoding": [],
        "checkpoints": {path.name: digest(path) for path in uri.MODEL_PATHS},
        "source_sha256": digest(Path(__file__)),
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(
        OUT / "run_lock.json",
        {
            "protocol_sha256": digest(OUT / "protocol.json"),
            "manifest_sha256": digest(OUT / "encoding_manifest.csv"),
            "source_sha256": digest(Path(__file__)),
        },
    )
    print("Registered complete URI-CADS local-detector experiment", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    for path, key in (
        (OUT / "protocol.json", "protocol_sha256"),
        (OUT / "encoding_manifest.csv", "manifest_sha256"),
        (Path(__file__), "source_sha256"),
    ):
        if digest(path) != lock[key]:
            raise ValueError(f"Registered input changed: {path}")


def feature_names() -> list[str]:
    names = []
    for fold in range(1, 6):
        names.extend([f"fold{fold}_local_{name}" for name in LOCAL_CLASSES])
        names.extend([f"fold{fold}_final_{name}" for name in LOCAL_CLASSES])
        names.extend([f"fold{fold}_alpha_{name}" for name in LOCAL_CLASSES])
        names.extend([f"fold{fold}_beta_{name}" for name in LOCAL_CLASSES])
        for class_name in LOCAL_CLASSES[1:]:
            names.extend(
                [f"fold{fold}_detect_{class_name}_{metric}" for metric in DETECTION_FEATURES]
            )
    return names


def encode() -> None:
    require_lock()
    output = OUT / "official_full_detector_features.npy"
    if output.exists():
        raise FileExistsError(f"Preserve completed detector output: {output}")
    _, _, images = local_study.load_reference()
    if pd.read_csv(OUT / "encoding_manifest.csv").image_id.tolist() != images.image_id.tolist():
        raise ValueError("Image order changed")
    canvases = uri.nckuh_canvases(images)
    mean, std = uri.public_normalization()
    global_scores = np.load(uri.OUT / "official_fold_pathology_scores.npy")
    if global_scores.shape != (820, 5, 7):
        raise ValueError("Unexpected official global score geometry")
    fold_features, model_audit = [], []
    for fold, checkpoint in enumerate(uri.MODEL_PATHS):
        model = OfficialLocalDetector(checkpoint).cuda().eval().requires_grad_(False)
        rows = []
        with torch.inference_mode():
            for start in range(0, len(canvases), 2):
                batch = (
                    torch.from_numpy(
                        canvases[start : start + 2].transpose(0, 3, 1, 2).copy()
                    ).cuda().float()
                    / 255.0
                )
                batch = (batch - torch.from_numpy(mean).cuda()[None, :, None, None]) / torch.from_numpy(std).cuda()[None, :, None, None]
                detections = model([image for image in batch])
                local_rows, geometry_rows = [], []
                for detection in detections:
                    local_values, geometry = aggregate_detection(detection)
                    local_rows.append(local_values)
                    geometry_rows.append(geometry)
                local_tensor = torch.from_numpy(np.stack(local_rows)).cuda()
                global_tensor = torch.from_numpy(
                    global_scores[start : start + len(detections), fold]
                ).cuda()
                final, alpha, beta = model.official_fusion(global_tensor, local_tensor)
                rows.append(
                    np.c_[
                        local_tensor.cpu().numpy(),
                        final.cpu().numpy(),
                        alpha.cpu().numpy(),
                        beta.cpu().numpy(),
                        np.stack(geometry_rows),
                    ]
                )
                if start % 80 == 0:
                    print(
                        f"URI-CADS full detector fold {fold + 1}/5 encoded {min(start + 2, len(canvases))}/820",
                        flush=True,
                    )
        current = np.concatenate(rows).astype(np.float32)
        fold_features.append(current)
        model_audit.append(
            {
                "fold": fold + 1,
                "checkpoint": checkpoint.name,
                "sha256": digest(checkpoint),
                "mapped_tensors": model.mapped_tensors,
                "mapped_parameters": model.mapped_parameters,
                "detections_nonzero_fraction": float((current[:, 1:7] > 0).any(1).mean()),
            }
        )
        del model
        torch.cuda.empty_cache()
    features = np.concatenate(fold_features, axis=1)
    names = feature_names()
    if features.shape != (820, len(names)) or not np.isfinite(features).all():
        raise ValueError(f"Unexpected detector feature geometry: {features.shape}, {len(names)}")
    np.save(output, features)
    pd.DataFrame({"column": np.arange(len(names)), "feature": names}).to_csv(
        OUT / "feature_dictionary.csv", index=False
    )
    write_json(
        OUT / "encoder_audit.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "images": len(images),
            "shape": list(features.shape),
            "finite": True,
            "labels_used_in_encoding": [],
            "feature_sha256": digest(output),
            "models": model_audit,
        },
    )
    print(f"Encoded complete URI-CADS detector features {features.shape}", flush=True)


def evaluate() -> None:
    require_lock()
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve completed development evaluation")
    feature_path = OUT / "official_full_detector_features.npy"
    encoder = json.loads((OUT / "encoder_audit.json").read_text(encoding="utf-8"))
    if digest(feature_path) != encoder["feature_sha256"]:
        raise ValueError("Detector feature hash mismatch")
    features = np.load(feature_path)
    patients, kidneys, images = local_study.load_reference()
    views = hierarchy.structured_views(base.patient_pair_features(features, images, patients))
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    references = pd.read_csv(REFERENCE)
    references = references[
        references.group.isin([CURRENT_GROUP, URI_GROUP])
    ].set_index(["seed", "group", "kidney_alias"], verify_integrity=True)
    (OUT / "models").mkdir()
    metrics, predictions, selections = [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train = np.flatnonzero(patients.patient_alias.map(mapping).eq("train"))
        development = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        target = kidneys.iloc[development]
        current = references.loc[
            [(seed, CURRENT_GROUP, alias) for alias in target.kidney_alias]
        ].score.to_numpy(float)
        fixed_uri = references.loc[
            [(seed, URI_GROUP, alias) for alias in target.kidney_alias]
        ].score.to_numpy(float)
        probability, models = local_study.probability_from_heads(
            views[0], views[1], states, train, seed
        )
        candidate = hierarchy.kidney_scores(probability, patients, kidneys)[development]
        joblib.dump(models, OUT / "models" / f"{seed}_uricads_full_detector.joblib", compress=3)
        for head, model in models.items():
            selections.append(
                {
                    "seed": seed,
                    "head": head,
                    "inner_cv_auroc": float(model.best_score_),
                    **model.best_params_,
                }
            )
        scores = {CURRENT_GROUP: current, URI_GROUP: fixed_uri, "uricads_full_detector_state": candidate}
        for fraction, suffix in ((0.20, "fifth"), (0.25, "quarter")):
            scores[f"uricads_full_detector_state_{suffix}_plus_current"] = fraction * candidate + (1 - fraction) * current
            scores[f"uricads_full_detector_state_{suffix}_plus_uri"] = fraction * candidate + (1 - fraction) * fixed_uri
        for group, score in scores.items():
            metrics.append(
                {
                    "seed": seed,
                    "role": "development",
                    "group": group,
                    "patients": target.patient_alias.nunique(),
                    "kidneys": len(target),
                    **hierarchy.metric(labels[development], score),
                }
            )
            for local_index, index in enumerate(development):
                predictions.append(
                    {
                        "seed": seed,
                        "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[index],
                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                        "side": kidneys.side.iloc[index],
                        "label": int(labels[index]),
                        "score": float(score[local_index]),
                    }
                )
        print(f"Completed full URI-CADS detector draw {seed}", flush=True)
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "training_inner_cv_selections.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    groups = prediction_frame.group.drop_duplicates().tolist()
    intervals, differences, samples = cluster_intervals(
        prediction_frame, groups, SEEDS, URI_GROUP
    )
    intervals.to_csv(OUT / "development_ci.csv", index=False)
    differences.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(
        OUT / "decision.json",
        {
            "groups_above_both_development_targets": achieved,
            "promotion": bool(achieved),
            "reference": URI_GROUP,
            "exploratory_test_rows_scored": 0,
        },
    )
    write_json(
        OUT / "integrity_audit.json",
        {
            "status": "pass",
            "patients": len(patients),
            "kidneys": len(kidneys),
            "images": len(images),
            "features": int(features.shape[1]),
            "development_groups": prediction_frame.group.nunique(),
            "finite_scores": bool(np.isfinite(prediction_frame.score).all()),
            "unit_interval_scores": bool(prediction_frame.score.between(0, 1).all()),
            "exploratory_test_rows_scored": 0,
            "labels_used_in_encoding": [],
        },
    )
    print(summary.to_string(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "encode", "evaluate"))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        globals()[args.stage]()


if __name__ == "__main__":
    main()
