"""Frozen official TinyUSFM representations on contour-first NCKUH kidneys."""

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
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "03_data_models/third_party/TinyUSFM_official_20261105"
WEIGHT = MODEL_ROOT / "TinyUSFM.pth"
SOURCE = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_fullframe_20261103"
REFERENCE = ROOT / "02_experiments/NCKUH_DMSA_uricads_official_frozen_20261105/development_predictions.csv"
OUT = ROOT / "02_experiments/NCKUH_DMSA_tinyusfm_frozen_20261105"
SEEDS = (20261102, 20261103, 20261104)
CURRENT_GROUP = "current_tile_usfm_reference"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"
WEIGHT_SHA256 = "776a50eeff402802c8cd06e00fe9de73df4161504fa733284bc240904b48252c"
COMMIT = "f3096e97da3814011c5f59176b5cc19b5477ac64"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_model(device: torch.device):
    sys.path.insert(0, str(MODEL_ROOT))
    from model.tinyusfm import TinyUSFM

    if digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError("TinyUSFM checkpoint SHA-256 mismatch")
    model = TinyUSFM(num_classes=0, global_pool=False)
    state = torch.load(WEIGHT, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("TinyUSFM strict loading failed")
    model.to(device).eval().requires_grad_(False)
    return model


def forward_tokens(model, tensor: torch.Tensor) -> torch.Tensor:
    """Run the official encoder while bypassing a timm>=1 forward signature change."""
    encoder = model.model
    values = encoder.patch_embed(tensor)
    cls = encoder.cls_token.expand(values.shape[0], -1, -1)
    values = torch.cat((cls, values), dim=1)
    values = encoder.pos_drop(values + encoder.pos_embed)
    for block in encoder.blocks:
        values = block(values)
    return encoder.norm(values)


def summarize(tokens: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tokens.ndim != 3 or tokens.shape[1:] != (197, 192) or masks.shape[1:] != (224, 224):
        raise ValueError("Unexpected TinyUSFM tokens or masks")
    coverage = masks.reshape(-1, 14, 16, 14, 16).mean((2, 4)).reshape(-1, 196)
    cls = tokens[:, 0]
    patch = tokens[:, 1:]
    means, deviations = [], []
    for current, current_coverage in zip(patch, coverage):
        selected = current_coverage >= 0.25
        if not selected.any():
            selected[np.argmax(current_coverage)] = True
        means.append(current[selected].mean(0))
        deviations.append(current[selected].std(0))
    return cls.astype(np.float32), np.asarray(means, np.float32), np.asarray(deviations, np.float32)


def load_reference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference as load

    patients, kidneys, images = load()
    images = images[images.kidney_APN.notna()].copy().reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].copy().reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].copy().reset_index(drop=True)
    if (len(patients), len(kidneys), len(images)) != (210, 414, 820):
        raise ValueError("Unexpected cohort")
    return patients, kidneys, images


def encode(images: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    cls_rows, roi_rows, sd_rows = [], [], []
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            canvases, masks = [], []
            for identifier in images.image_id.iloc[start:start + 32]:
                bgr = cv2.imread(str(SOURCE / "clean_rgb" / f"{identifier}.png"), cv2.IMREAD_COLOR)
                mask = cv2.imread(str(SOURCE / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE) > 0
                if bgr is None or mask is None:
                    raise FileNotFoundError(identifier)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                canvas, canvas_mask = base.tight_canvas(rgb * mask[..., None], mask)
                canvases.append(canvas)
                masks.append(canvas_mask.astype(np.float32))
            tensor = torch.from_numpy(np.stack(canvases).transpose(0, 3, 1, 2).copy()).to(device).float() / 255.0
            tokens = forward_tokens(model, (tensor - mean) / std).cpu().numpy()
            cls, roi, deviation = summarize(tokens, np.stack(masks))
            cls_rows.append(cls)
            roi_rows.append(roi)
            sd_rows.append(deviation)
            print(f"TinyUSFM encoded {min(start + 32, len(images))}/{len(images)}", flush=True)
    return np.concatenate(cls_rows), np.concatenate(roi_rows), np.concatenate(sd_rows)


def run() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve completed or partial experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "A compact ultrasound-specific encoder preserving spatial and frequency representations may transfer differently from previously screened USFM and general medical encoders",
        "source": "https://github.com/MacDunno/TinyUSFM",
        "paper": "TinyUSFM: Towards Compact and Efficient Ultrasound Foundation Models, IEEE JBHI 2026, DOI 10.1109/JBHI.2026.3678309",
        "license": "Apache-2.0",
        "official_commit": COMMIT,
        "official_weight_url": "https://drive.google.com/file/d/1P_wDuU-StALLJEBaymioaYu52CScW-l-/view",
        "weight_sha256": WEIGHT_SHA256,
        "parameters": 5524416,
        "strict_weight_load": True,
        "compatibility": "Official VisionTransformer.forward_features logic is called explicitly because timm 1.0 passes new attn_mask arguments to the repository's older override; third-party source is unmodified",
        "input": "Contour-masked 224x224 RGB, official ImageNet normalization",
        "representations": "Pretrained CLS token; mask-covered final patch-token mean and standard deviation",
        "models": "Frozen encoder; training-only hierarchical four-state LR heads",
        "fusion": "Candidate alone and fixed 25% contribution to the unchanged official URI-CADS quarter-fusion reference",
        "cohort": "210 patients, 414 kidneys, 820 images; unchanged three 126/42/42 assignments",
        "overlap": "No known NCKUH pretraining overlap; NCKUH is not a public TinyUSFM benchmark source",
        "exploratory_test": "Unopened",
        "inputs": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (Path(__file__), WEIGHT, SOURCE / "patient_assignments.csv", REFERENCE)
        },
    }
    write_json(OUT / "protocol.json", protocol)

    patients, kidneys, images = load_reference()
    cls, roi, deviation = encode(images)
    if cls.shape != roi.shape or cls.shape != deviation.shape or cls.shape != (820, 192):
        raise ValueError("Unexpected TinyUSFM representation shape")
    if not all(np.isfinite(value).all() for value in (cls, roi, deviation)):
        raise ValueError("Nonfinite TinyUSFM representation")
    np.savez_compressed(OUT / "tinyusfm_embeddings.npz", cls=cls, roi_mean=roi, roi_std=deviation)
    write_json(OUT / "encoder_audit.json", {
        "images": 820,
        "shapes": {"cls": list(cls.shape), "roi_mean": list(roi.shape), "roi_std": list(deviation.shape)},
        "frozen": True,
        "labels_used_in_encoding": [],
        "weight_sha256": digest(WEIGHT),
        "embedding_sha256": digest(OUT / "tinyusfm_embeddings.npz"),
    })

    representations = {
        "cls": cls,
        "roi": np.c_[roi, deviation],
        "cls_roi": np.c_[cls, roi, deviation],
    }
    structured = {
        name: hierarchy.structured_views(base.patient_pair_features(values, images, patients))
        for name, values in representations.items()
    }
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(SOURCE / "patient_assignments.csv")
    references = pd.read_csv(REFERENCE)
    references = references[references.group.isin([CURRENT_GROUP, URI_GROUP])]
    references = references.set_index(["seed", "group", "kidney_alias"], verify_integrity=True)

    metrics, predictions, selections = [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train = np.flatnonzero(patients.patient_alias.map(mapping).eq("train"))
        development = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        target = kidneys.iloc[development]
        if len(train) != 126 or target.patient_alias.nunique() != 42:
            raise ValueError("Unexpected split")
        current = references.loc[[(seed, CURRENT_GROUP, alias) for alias in target.kidney_alias]].score.to_numpy(float)
        uri = references.loc[[(seed, URI_GROUP, alias) for alias in target.kidney_alias]].score.to_numpy(float)
        scores = {"current_tile_usfm_reference": current, "official_uri_fixed_quarter_reference": uri}
        for source, (symmetric, antisymmetric) in structured.items():
            probability, models = hierarchy.state_probabilities(symmetric, antisymmetric, states, train, seed)
            candidate = hierarchy.kidney_scores(probability, patients, kidneys)[development]
            stem = f"tinyusfm_{source}"
            scores[stem] = candidate
            scores[f"{stem}_quarter_plus_uri"] = 0.25 * candidate + 0.75 * uri
            scores[f"{stem}_fifth_plus_uri"] = 0.20 * candidate + 0.80 * uri
            joblib.dump(models, OUT / "models" / f"{seed}_{source}.joblib", compress=3)
            for head, model in models.items():
                selections.append({"seed": seed, "source": source, "head": head, "inner_cv_auroc": float(model.best_score_), **model.best_params_})

        for group, score in scores.items():
            result = hierarchy.metric(labels[development], score)
            metrics.append({"seed": seed, "role": "development", "group": group, "patients": 42, "kidneys": len(development), **result})
            for local, index in enumerate(development):
                predictions.append({
                    "seed": seed,
                    "group": group,
                    "patient_alias": kidneys.patient_alias.iloc[index],
                    "kidney_alias": kidneys.kidney_alias.iloc[index],
                    "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                    "side": kidneys.side.iloc[index],
                    "label": int(labels[index]),
                    "score": float(score[local]),
                })
        print(f"Completed TinyUSFM draw {seed}", flush=True)

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
    ci, delta, samples = cluster_intervals(prediction_frame, groups, SEEDS, "official_uri_fixed_quarter_reference")
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(OUT / "decision.json", {
        "groups_above_both_development_targets": achieved,
        "promotion": bool(achieved),
        "reference": "official_uri_fixed_quarter_reference",
        "exploratory_test_rows_scored": 0,
    })
    write_json(OUT / "integrity_audit.json", {
        "patients": len(patients),
        "kidneys": len(kidneys),
        "images": len(images),
        "prediction_rows": len(prediction_frame),
        "development_groups": len(groups),
        "finite_scores": bool(np.isfinite(prediction_frame.score).all()),
        "unit_interval_scores": bool(prediction_frame.score.between(0.0, 1.0).all()),
        "exploratory_test_rows_scored": 0,
    })
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    run()
