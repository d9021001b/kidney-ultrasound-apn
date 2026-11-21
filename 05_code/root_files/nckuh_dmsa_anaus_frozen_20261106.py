"""Frozen ANAUS global and mask-constrained multiscale features for NCKUH APN."""

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
import torch.nn.functional as F
import torchvision.models as tvm
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_uri_contralateral_local_20261105 as local_study
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_anaus_frozen_20261106"
REPOSITORY = ROOT / "third_party/ANAUS"
WEIGHT = REPOSITORY / "checkpoint/pre-trained-anaus.pth"
REFERENCE = uri.OUT / "development_predictions.csv"
CURRENT_GROUP = "current_tile_usfm_reference"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"
SEEDS = (20261102, 20261103, 20261104)
GROUPS = {
    "global": "anaus_global_hierarchical",
    "pyramid": "anaus_masked_pyramid_hierarchical",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve registered experiment: {OUT}")
    if not WEIGHT.exists():
        raise FileNotFoundError(WEIGHT)
    OUT.mkdir(parents=True)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "An anatomy-anchored ultrasound encoder and mask-constrained multiscale activation summaries may retain focal parenchymal APN evidence that global natural-image and generic ultrasound representations miss.",
        "source": "Official ANAUS MICCAI 2026 repository checkpoint",
        "repository_url": "https://github.com/zhcz328/ANAUS",
        "repository_commit": "92dab0d6233fd642d2d3987cd3071fe38532f1aa",
        "weight_sha256": digest(WEIGHT),
        "preprocessing": "Contour-first 224x224 RGB tight canvas; official validation normalization mean 0.5 and SD 0.25 per channel",
        "representations": {
            "global": "Frozen ANAUS ResNet-18 global average-pooled layer4 descriptor",
            "pyramid": "Global descriptor plus within-kidney layer3 and layer4 mean, SD, 90th percentile and maximum summaries",
        },
        "classifier": "Training-only hierarchical any-APN, bilateral-given-APN and left-given-unilateral logistic heads",
        "outer_design": "Three fixed patient-level draws with 126 train, 42 development and 42 untouched exploratory-test patients",
        "fixed_fusions": "Standalone and fixed 25% candidate plus 75% unchanged current reference",
        "labels_used_in_encoding": [],
        "source_sha256": digest(Path(__file__)),
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(
        OUT / "run_lock.json",
        {"protocol_sha256": digest(OUT / "protocol.json"), "source_sha256": digest(Path(__file__))},
    )
    print("Registered frozen ANAUS experiment", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    if digest(OUT / "protocol.json") != lock["protocol_sha256"]:
        raise ValueError("Protocol changed")
    if digest(Path(__file__)) != lock["source_sha256"]:
        raise ValueError("Source changed")


def load_model() -> torch.nn.Module:
    model = tvm.resnet18(weights=None)
    state = torch.load(WEIGHT, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(result)
    model.fc = torch.nn.Identity()
    return model.eval().requires_grad_(False)


def forward_features(model: torch.nn.Module, tensor: torch.Tensor):
    x = model.conv1(tensor)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    layer3 = model.layer3(x)
    layer4 = model.layer4(layer3)
    global_feature = model.avgpool(layer4).flatten(1)
    return global_feature, layer3, layer4


def masked_summary(feature: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    resized = F.interpolate(masks.float(), size=feature.shape[-2:], mode="nearest")[:, 0] > 0
    rows = []
    for index in range(len(feature)):
        values = feature[index, :, resized[index]]
        if values.shape[1] == 0:
            raise ValueError("Empty feature mask")
        rows.append(
            torch.cat(
                [
                    values.mean(1),
                    values.std(1, unbiased=False),
                    torch.quantile(values, 0.9, dim=1),
                    values.max(1).values,
                ]
            )
        )
    return torch.stack(rows)


def canvases(images: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    color_rows, mask_rows = [], []
    for identifier in images.image_id:
        bgr = cv2.imread(str(base.OUT / "clean_rgb" / f"{identifier}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(base.OUT / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE) > 0
        if bgr is None or mask is None:
            raise ValueError(f"Unreadable contour-first input: {identifier}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        color, canvas_mask = base.tight_canvas(rgb, mask, size=224)
        color_rows.append(color)
        mask_rows.append(canvas_mask)
    return np.stack(color_rows), np.stack(mask_rows)


def encode() -> None:
    require_lock()
    if (OUT / "anaus_global.npy").exists():
        raise FileExistsError("Preserve completed ANAUS encoding")
    _, _, images = local_study.load_reference()
    colors, masks = canvases(images)
    model = load_model().cuda()
    global_rows, pyramid_rows = [], []
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            batch = torch.from_numpy(colors[start : start + 32].transpose(0, 3, 1, 2).copy()).cuda().float() / 255
            batch = (batch - 0.5) / 0.25
            batch_masks = torch.from_numpy(masks[start : start + 32, None].copy()).cuda()
            global_feature, layer3, layer4 = forward_features(model, batch)
            local = torch.cat(
                [global_feature, masked_summary(layer3, batch_masks), masked_summary(layer4, batch_masks)],
                dim=1,
            )
            global_rows.append(global_feature.cpu().numpy())
            pyramid_rows.append(local.cpu().numpy())
            print(f"Encoded ANAUS {min(start + 32, len(images))}/{len(images)}", flush=True)
    global_array = np.concatenate(global_rows).astype(np.float32)
    pyramid_array = np.concatenate(pyramid_rows).astype(np.float32)
    if global_array.shape != (820, 512) or pyramid_array.shape != (820, 3584):
        raise ValueError((global_array.shape, pyramid_array.shape))
    if not np.isfinite(global_array).all() or not np.isfinite(pyramid_array).all():
        raise ValueError("Non-finite ANAUS feature")
    np.save(OUT / "anaus_global.npy", global_array)
    np.save(OUT / "anaus_masked_pyramid.npy", pyramid_array)
    write_json(
        OUT / "encoding_audit.json",
        {
            "images": len(images),
            "global_shape": list(global_array.shape),
            "pyramid_shape": list(pyramid_array.shape),
            "finite": True,
            "weight_sha256": digest(WEIGHT),
            "labels_used_in_encoding": [],
        },
    )
    print("Completed frozen ANAUS encoding", flush=True)


def evaluate() -> None:
    require_lock()
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve completed ANAUS evaluation")
    (OUT / "models").mkdir()
    patients, kidneys, images = local_study.load_reference()
    representations = {
        "global": np.load(OUT / "anaus_global.npy"),
        "pyramid": np.load(OUT / "anaus_masked_pyramid.npy"),
    }
    structured = {
        name: hierarchy.structured_views(base.patient_pair_features(values, images, patients))
        for name, values in representations.items()
    }
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    references = pd.read_csv(REFERENCE)
    references = references[references.group.isin([CURRENT_GROUP, URI_GROUP])].set_index(
        ["seed", "group", "kidney_alias"], verify_integrity=True
    )
    metrics, predictions, selections, splits = [], [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        patient_roles = patients.patient_alias.map(mapping)
        kidney_roles = kidneys.patient_alias.map(mapping)
        train = np.flatnonzero(patient_roles.eq("train"))
        development = np.flatnonzero(kidney_roles.eq("development"))
        target = kidneys.iloc[development]
        if len(train) != 126 or target.patient_alias.nunique() != 42:
            raise ValueError("Unexpected split")
        current = references.loc[
            [(seed, CURRENT_GROUP, alias) for alias in target.kidney_alias]
        ].score.to_numpy(float)
        fixed_uri = references.loc[
            [(seed, URI_GROUP, alias) for alias in target.kidney_alias]
        ].score.to_numpy(float)
        score_sets = {CURRENT_GROUP: current, URI_GROUP: fixed_uri}
        for name, (symmetric, antisymmetric) in structured.items():
            probability, models = hierarchy.state_probabilities(
                symmetric, antisymmetric, states, train, seed
            )
            candidate = hierarchy.kidney_scores(probability, patients, kidneys)[development]
            group = GROUPS[name]
            score_sets[group] = candidate
            score_sets[f"{group}_quarter_plus_current"] = 0.25 * candidate + 0.75 * current
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
        for group, score in score_sets.items():
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
        splits.append(
            {
                "seed": seed,
                "training_patients": len(train),
                "development_patients": target.patient_alias.nunique(),
                "patient_overlap": 0,
                "exploratory_test_scored": 0,
            }
        )
        print(f"Completed ANAUS draw {seed}", flush=True)

    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv")
    pd.DataFrame(selections).to_csv(OUT / "training_inner_cv_selections.csv", index=False)
    pd.DataFrame(splits).to_csv(OUT / "split_audit.csv", index=False)
    groups = prediction_frame.group.drop_duplicates().tolist()
    intervals, differences, samples = cluster_intervals(
        prediction_frame, groups, SEEDS, URI_GROUP
    )
    intervals.to_csv(OUT / "development_ci.csv", index=False)
    differences.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(
        OUT / "integrity_and_decision.json",
        {
            "status": "pass",
            "patients": len(patients),
            "kidneys": len(kidneys),
            "images": len(images),
            "patient_overlap": 0,
            "exploratory_test_rows_scored": 0,
            "groups_above_both_development_targets": achieved,
            "labels_used_in_encoding": [],
        },
    )
    print(summary.to_string(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "encode", "evaluate"))
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with threadpool_limits(limits=4):
        globals()[args.stage]()


if __name__ == "__main__":
    main()
