"""Frozen medical/ultrasound encoders on clean archived NCKUH full-frame ROIs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullframe_clean_probe_20261102 as F
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json
from nckuh_ultrasound_probe_20261102 import WEIGHT as USF_WEIGHT

ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullframe_deep_probe_20261102"
RAD_SOURCE = ROOT / "02_experiments/CYCH_feature_research_results_20260913/source"
RAD_WEIGHT = Path(r"C:/Users/udoo_w2/Desktop/lung_cancerimage/codex_experiments/pretrained/RadImageNet_pytorch/DenseNet121.pt")
GROUPS = (
    "radimagenet_gray_pca32_lr",
    "radimagenet_color_pca32_lr",
    "usf_gray_pca32_lr",
    "deep_concat_pca64_lr",
    "radimagenet_gray_plus_clean_radiomics",
    "radimagenet_color_plus_clean_radiomics",
    "usf_gray_plus_clean_radiomics",
    "deep_concat_plus_clean_radiomics",
)
REFERENCE = "clean_fullframe_radiomics_anova64_lr"


def tight_canvas(image: np.ndarray, mask: np.ndarray, size: int = 224) -> tuple[np.ndarray, np.ndarray]:
    binary = mask.astype(bool)
    yy, xx = np.nonzero(binary)
    region = image[yy.min():yy.max()+1, xx.min():xx.max()+1]
    selected = binary[yy.min():yy.max()+1, xx.min():xx.max()+1].astype(np.uint8)
    scale = size / max(region.shape[:2])
    height, width = max(1, round(region.shape[0] * scale)), max(1, round(region.shape[1] * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    if image.ndim == 2:
        output = np.zeros((size, size), np.uint8)
    else:
        output = np.zeros((size, size, image.shape[2]), np.uint8)
    output_mask = np.zeros((size, size), np.uint8)
    top, left = (size-height)//2, (size-width)//2
    output[top:top+height, left:left+width] = cv2.resize(region, (width, height), interpolation=interpolation)
    output_mask[top:top+height, left:left+width] = cv2.resize(selected, (width, height), interpolation=cv2.INTER_NEAREST)
    output *= output_mask[..., None] if output.ndim == 3 else output_mask
    if output.ndim == 2:
        output = np.repeat(output[..., None], 3, axis=2)
    return output, output_mask


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered full-frame deep protocol")
    F.require_lock()
    required = [
        F.OUT / "archive_extraction_audit.json",
        F.OUT / "deidentified_image_manifest.csv",
        F.OUT / "patient_assignments.csv",
        F.OUT / "development_predictions.csv",
        USF_WEIGHT,
        RAD_WEIGHT,
        Path(__file__),
    ]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same availability-defined 111 patients, 212 kidneys and 418 clean full-frame-derived views",
        "roles": "Reuse the exact three fixed 70/20/21 patient assignments; exploratory-test roles remain unscored",
        "encoders": {
            "RadImageNet DenseNet121": "Frozen official local medical-image checkpoint; grayscale and original-color ROI controls; x/127.5-1 normalization",
            "USF-MAE ViT-B16": "Frozen official local ultrasound checkpoint; grayscale P1-P99 ROI; ImageNet normalization; class token plus mean of patches with at least80% mask coverage",
        },
        "aggregation": "Preserve longitudinal/transverse views and ordered own/opposite-kidney context, then train-only standardized PCA32 or PCA64 and balanced LR C.1",
        "fusion": "Four preregistered equal-probability averages with the completed clean-full-frame radiomics ANOVA64 LR branch",
        "prohibited": "No diagnosis suffix, DMSA field, MRN, dates, target-fitted encoder, test-derived component/weight/threshold, or exploratory-test scoring",
        "target": "Development AUROC>0.8 and AP>0.8; full objective still requires a later locked confirmation",
        "groups": list(GROUPS), "seeds": list(F.SEEDS),
        "inputs": {str(p): digest(p) for p in required},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "source_sha256": digest(__file__),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "assignments_sha256": digest(F.OUT / "patient_assignments.csv"),
        "manifest_sha256": digest(F.OUT / "deidentified_image_manifest.csv"),
        "usf_weight_sha256": digest(USF_WEIGHT),
        "radimagenet_weight_sha256": digest(RAD_WEIGHT),
    })
    print("Registered clean full-frame frozen-encoder probe", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    assert digest(F.OUT / "patient_assignments.csv") == lock["assignments_sha256"]
    assert digest(F.OUT / "deidentified_image_manifest.csv") == lock["manifest_sha256"]
    assert digest(USF_WEIGHT) == lock["usf_weight_sha256"]
    assert digest(RAD_WEIGHT) == lock["radimagenet_weight_sha256"]


def encode() -> None:
    require_lock()
    if (OUT / "encoder_audit.json").exists():
        raise FileExistsError("Preserve completed clean full-frame embeddings")
    import torch
    import timm
    sys.path.insert(0, str(RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    torch.set_num_threads(4)
    torch.manual_seed(20261102)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, _, images = F.matched_reference()
    gray_canvases, color_canvases, masks = [], [], []
    for ident in images.image_id:
        color = cv2.imread(str(F.OUT / "clean_rgb" / f"{ident}.png"), cv2.IMREAD_COLOR)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        gray = cv2.imread(str(F.OUT / "percentile" / f"{ident}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(F.OUT / "masks" / f"{ident}.png"), cv2.IMREAD_GRAYSCALE) > 0
        gray_canvas, canvas_mask = tight_canvas(gray, mask)
        color_canvas, color_mask = tight_canvas(color, mask)
        assert np.array_equal(canvas_mask, color_mask)
        gray_canvases.append(gray_canvas); color_canvases.append(color_canvas); masks.append(canvas_mask)

    rad = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    rad_gray, rad_color = [], []
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            for canvases, output in ((gray_canvases, rad_gray), (color_canvases, rad_color)):
                tensor = torch.from_numpy(np.stack(canvases[start:start+32]).transpose(0, 3, 1, 2).copy()).cuda().float()
                tensor = tensor / 127.5 - 1
                output.append(rad(tensor).cpu().numpy())
    rad_gray = np.concatenate(rad_gray).astype(np.float32)
    rad_color = np.concatenate(rad_color).astype(np.float32)
    del rad

    checkpoint = torch.load(USF_WEIGHT, map_location="cpu", weights_only=True)
    usf = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    keys = set(usf.state_dict())
    encoder = {key: value for key, value in checkpoint.items() if key in keys}
    assert set(encoder) == keys and len(keys) == 150
    usf.load_state_dict(encoder, strict=True)
    usf = usf.cuda().eval().requires_grad_(False)
    mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1)
    usf_features, selected_counts = [], []
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            batch_masks = np.stack(masks[start:start+32])
            coverage = batch_masks.reshape(len(batch_masks), 14, 16, 14, 16).mean((2, 4)).reshape(len(batch_masks), 196)
            selected = coverage >= .8
            for local, row in enumerate(selected):
                if not row.any(): row[np.argmax(coverage[local])] = True
                selected_counts.append(int(row.sum()))
            tensor = torch.from_numpy(np.stack(gray_canvases[start:start+32]).transpose(0, 3, 1, 2).copy()).cuda().float() / 255
            features = usf.forward_features((tensor-mean)/std)
            chosen = torch.from_numpy(selected).cuda()
            patch_mean = (features[:, 1:] * chosen[..., None]).sum(1) / chosen.sum(1)[:, None]
            usf_features.append(torch.cat([features[:, 0], patch_mean], dim=1).cpu().numpy())
    usf_features = np.concatenate(usf_features).astype(np.float32)
    assert rad_gray.shape == rad_color.shape == (418, 1024)
    assert usf_features.shape == (418, 1536)
    np.save(OUT / "radimagenet_gray.npy", rad_gray)
    np.save(OUT / "radimagenet_color.npy", rad_color)
    np.save(OUT / "usf_gray.npy", usf_features)
    write_json(OUT / "encoder_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "images": 418,
        "shapes": {"radimagenet_gray": list(rad_gray.shape), "radimagenet_color": list(rad_color.shape),
                   "usf_gray": list(usf_features.shape)},
        "selected_patch_count_min_median_max": [int(np.min(selected_counts)), float(np.median(selected_counts)), int(np.max(selected_counts))],
        "weights_unchanged": True,
        "labels_used_in_encoding": [],
        "hashes": {name: digest(OUT / name) for name in ("radimagenet_gray.npy", "radimagenet_color.npy", "usf_gray.npy")},
    })
    print("Encoded 418 clean ROIs with RadImageNet and USF-MAE", flush=True)


def fit_predict(values, train, dev, labels, seed, components):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          PCA(components, whiten=True, random_state=seed),
                          LogisticRegression(C=.1, class_weight="balanced", solver="liblinear",
                                             max_iter=3000, random_state=seed))
    model.fit(values[train], labels[train])
    return model.predict_proba(values[dev])[:, 1], model


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed full-frame deep predictions")
    patients, kidneys, images = F.matched_reference()
    labels = kidneys.kidney_APN.to_numpy(int)
    image_arrays = {
        "rad_gray": np.load(OUT / "radimagenet_gray.npy"),
        "rad_color": np.load(OUT / "radimagenet_color.npy"),
        "usf": np.load(OUT / "usf_gray.npy"),
    }
    oriented = {name: F.orient_kidneys(F.patient_pair_features(values, images, patients), patients, kidneys)
                for name, values in image_arrays.items()}
    oriented["concat"] = np.concatenate(list(oriented.values()), axis=1)
    assignments = pd.read_csv(F.OUT / "patient_assignments.csv")
    reference = pd.read_csv(F.OUT / "development_predictions.csv")
    reference = reference[reference.group.eq(REFERENCE)].set_index(["seed", "kidney_alias"]).score
    predictions, metrics = [], []
    (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in F.SEEDS:
            roles = kidneys.patient_alias.map(assignments[assignments.seed.eq(seed)].set_index("patient_alias").role)
            train = np.flatnonzero(roles.eq("train")); dev = np.flatnonzero(roles.eq("development"))
            specs = {GROUPS[0]: (oriented["rad_gray"], 32), GROUPS[1]: (oriented["rad_color"], 32),
                     GROUPS[2]: (oriented["usf"], 32), GROUPS[3]: (oriented["concat"], 64)}
            scores = {}
            for group, (values, components) in specs.items():
                scores[group], model = fit_predict(values, train, dev, labels, seed, components)
                joblib.dump(model, OUT / "models" / f"{seed}_{group}.joblib", compress=3)
            ref = np.asarray([reference.loc[(seed, alias)] for alias in kidneys.kidney_alias.iloc[dev]])
            for source, target in zip(GROUPS[:4], GROUPS[4:]):
                scores[target] = (scores[source] + ref) / 2
            for group in GROUPS:
                result = metric(labels[dev], scores[group])
                metrics.append({"seed": seed, "role": "development", "group": group,
                                "patients": 20, "kidneys": len(dev), **result})
                for local, index in enumerate(dev):
                    predictions.append({"seed": seed, "group": group,
                                        "patient_alias": kidneys.patient_alias.iloc[index],
                                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                                        "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                                        "side": kidneys.side.iloc[index], "label": int(labels[index]),
                                        "score": float(scores[group][local])})
            print(f"Completed clean full-frame deep draw {seed}", flush=True)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap"]].mean().sort_values(["auroc", "ap"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc > .8) & (summary.ap > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {
        "groups_above_both_development_targets": achieved,
        "test_scored": False, "goal_achieved": False,
        "reason": "Development-only availability subgroup cannot complete the full objective",
    })
    print(summary.to_string(), flush=True)


def bootstrap() -> None:
    require_lock()
    frame = pd.read_csv(OUT / "development_predictions.csv")
    reference = pd.read_csv(F.OUT / "development_predictions.csv")
    reference = reference[reference.group.eq(REFERENCE)]
    ci, paired, samples = cluster_intervals(pd.concat([frame, reference], ignore_index=True),
                                            list(GROUPS) + [REFERENCE], F.SEEDS, REFERENCE)
    ci.to_csv(OUT / "development_confidence_intervals.csv", index=False)
    paired.to_csv(OUT / "development_paired_intervals.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    write_json(OUT / "completion_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "patients": 111, "kidneys": 212, "images": 418,
        "test_scored": False, "scored_unit": "kidney", "bootstrap_cluster": "patient",
    })
    print("Clean full-frame deep intervals completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "encode", "run", "bootstrap"])
    args = parser.parse_args()
    {"prepare": prepare, "encode": encode, "run": run, "bootstrap": bootstrap}[args.stage]()
