"""Native frozen DINOv3 on clean full-frame-derived NCKUH renal ROIs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullframe_clean_probe_20261102 as F
import nckuh_dmsa_fullframe_deep_probe_20261102 as D
import nckuh_dmsa_fullframe_extreme_specialist_20261102 as E
import nckuh_dmsa_fullframe_hierarchical_probe_20261102 as H
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullframe_dinov3_probe_20261103"
BUNDLE = ROOT / "03_data_models/dinov3_vith16plus_weights_20260913"
REFERENCE = "deep_concat_plus_clean_radiomics"
GROUPS = (
    "dinov3_class_pca32_lr",
    "dinov3_patch_summary_pca32_lr",
    "dinov3_class_patch_pca64_lr",
    "dinov3_class_plus_deep_radiomics",
    "dinov3_patch_plus_deep_radiomics",
    "dinov3_class_patch_plus_deep_radiomics",
    "dinov3_early_concat_clean_radiomics_anova128_lr",
    "dinov3_extreme_replace_plus_deep_radiomics",
    "dinov3_extreme_halfblend_plus_deep_radiomics",
)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered clean DINOv3 protocol")
    F.require_lock(); D.require_lock(); H.require_lock()
    inputs = [F.OUT / "patient_assignments.csv", F.OUT / "deidentified_image_manifest.csv",
              F.OUT / "archive_extraction_audit.json", F.OUT / "radiomics_clean_fullframe.csv",
              D.OUT / "development_predictions.csv", H.OUT / "development_predictions.csv",
              *(H.OUT / "models" / f"{seed}_hierarchical.joblib" for seed in F.SEEDS),
              BUNDLE / "WEIGHT_MANIFEST.json", BUNDLE / "config.json",
              BUNDLE / "preprocessor_config.json", BUNDLE / "model.safetensors", Path(__file__)]
    manifest = json.loads((BUNDLE / "WEIGHT_MANIFEST.json").read_text(encoding="utf-8"))
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same clean-full-frame 111 patients, 212 kidneys, 418 views and exact 70/20/21 patient roles",
        "encoder": "Official local facebook/dinov3-vith16plus-pretrain-lvd1689m ViT-H+/16 bundle through installed transformers DINOv3ViTModel; strict local-only load, remote code disabled, frozen float32 inference",
        "checkpoint_sha256": manifest["weight_sha256"],
        "input": "P1-P99 grayscale renal ROI, mask outside zero, tight aspect-preserving 224 square canvas, replicated to RGB and normalized by the official local processor",
        "representations": "Final class token; and mean plus population SD of final patch tokens whose corresponding 16x16 input patch has at least 80% renal-mask coverage. Register tokens are excluded",
        "heads": "View-preserving own/opposite-kidney aggregation; training-only median imputation, scaling, whitened PCA32/64 and balanced LR C=.1. One fixed ANOVA128 early-concat control with clean radiomics",
        "extreme_specialist": "A DINOv3 patient-symmetric PCA16 B/N specialist fitted only to training B/N patients may only reallocate completed powerset N+B mass",
        "fusion": "Only fixed equal-probability averages with the unchanged deep-radiomics reference",
        "prohibited": "No encoder fine-tuning, label in encoding, development-selected token/weight/threshold, identifier/date/DMSA input or exploratory-test scoring",
        "target": "Development AUROC>0.8 and AP>0.8; later locked confirmation remains necessary",
        "groups": list(GROUPS), "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {"source_sha256": digest(__file__),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs}})
    print("Registered native frozen DINOv3 clean-ROI probe", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(Path(name)) == expected, name


def encode() -> None:
    require_lock()
    if (OUT / "dinov3_class.npy").exists():
        raise FileExistsError("Preserve completed clean DINOv3 embeddings")
    import torch
    from transformers import AutoImageProcessor, DINOv3ViTConfig, DINOv3ViTModel

    torch.set_num_threads(4); torch.manual_seed(20261103)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    config = DINOv3ViTConfig.from_pretrained(BUNDLE, local_files_only=True)
    model, info = DINOv3ViTModel.from_pretrained(
        BUNDLE, config=config, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, output_loading_info=True, attn_implementation="sdpa")
    assert not any(info.get(key, []) for key in
                   ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    model = model.cuda().eval().requires_grad_(False)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    processor = AutoImageProcessor.from_pretrained(BUNDLE, local_files_only=True, trust_remote_code=False)
    _, _, images = F.matched_reference()
    class_rows, patch_rows, counts = [], [], []
    with torch.inference_mode():
        for start in range(0, len(images), 4):
            canvases, masks = [], []
            for identifier in images.image_id.iloc[start:start + 4]:
                gray = cv2.imread(str(F.OUT / "percentile" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(str(F.OUT / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE) > 0
                canvas, binary = D.tight_canvas(gray, mask)
                canvases.append(canvas); masks.append(binary)
            batch = processor(images=canvases, return_tensors="pt", do_resize=False)
            tokens = model(pixel_values=batch["pixel_values"].cuda()).last_hidden_state.cpu().numpy()
            assert tokens.shape == (len(canvases), 201, 1280) and np.isfinite(tokens).all()
            class_rows.extend(tokens[:, 0])
            for local, binary in enumerate(masks):
                coverage = binary.reshape(14, 16, 14, 16).mean((1, 3)).reshape(-1)
                selected = coverage >= .8
                if not selected.any():
                    selected[np.argmax(coverage)] = True
                patches = tokens[local, 5:][selected]
                patch_rows.append(np.r_[patches.mean(0), patches.std(0)])
                counts.append(int(selected.sum()))
            if start % 64 == 0:
                print(f"Frozen clean DINOv3 {min(start + 4, len(images))}/{len(images)}", flush=True)
    class_rows = np.asarray(class_rows, dtype=np.float32)
    patch_rows = np.asarray(patch_rows, dtype=np.float32)
    assert class_rows.shape == (418, 1280) and patch_rows.shape == (418, 2560)
    np.save(OUT / "dinov3_class.npy", class_rows); np.save(OUT / "dinov3_patch_summary.npy", patch_rows)
    write_json(OUT / "encoder_audit.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "images": 418, "class_shape": list(class_rows.shape), "patch_shape": list(patch_rows.shape),
        "selected_patch_min_median_max": [min(counts), float(np.median(counts)), max(counts)],
        "strict_load": True, "frozen": True, "remote_code": False, "labels_used_in_encoding": [],
        "weight_sha256_after": digest(BUNDLE / "model.safetensors"),
        "class_sha256": digest(OUT / "dinov3_class.npy"),
        "patch_sha256": digest(OUT / "dinov3_patch_summary.npy")})
    print("Encoded 418 clean renal ROIs with native frozen DINOv3", flush=True)


def fit_extreme(values: np.ndarray, train: np.ndarray, states: np.ndarray, seed: int):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    half = values.shape[1] // 2
    left, right = values[:, :half], values[:, half:]
    symmetric = np.c_[left + right, np.abs(left - right)]
    extreme = train[np.isin(states[train], [0, 3])]
    target = (states[extreme] == 3).astype(int)
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          PCA(16, whiten=True, random_state=seed),
                          LogisticRegression(C=.1, class_weight="balanced", solver="liblinear",
                                             max_iter=3000, random_state=seed))
    model.fit(symmetric[extreme], target)
    return model.predict_proba(symmetric)[:, 1], model


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed clean DINOv3 predictions")
    patients, kidneys, images = F.matched_reference(); labels = kidneys.kidney_APN.to_numpy(int)
    states = patients.DMSA_APN.map(H.CODE).to_numpy(int)
    class_image = np.load(OUT / "dinov3_class.npy")
    patch_image = np.load(OUT / "dinov3_patch_summary.npy")
    concat_image = np.c_[class_image, patch_image]
    class_pairs = F.patient_pair_features(class_image, images, patients)
    patch_pairs = F.patient_pair_features(patch_image, images, patients)
    concat_pairs = F.patient_pair_features(concat_image, images, patients)
    oriented_class = F.orient_kidneys(class_pairs, patients, kidneys)
    oriented_patch = F.orient_kidneys(patch_pairs, patients, kidneys)
    oriented_concat = F.orient_kidneys(concat_pairs, patients, kidneys)
    clean_frame = pd.read_csv(F.OUT / "radiomics_clean_fullframe.csv").set_index("image_id").reindex(images.image_id)
    clean = clean_frame[[column for column in clean_frame if not column.endswith(("_Energy", "_TotalEnergy"))]].to_numpy(float)
    oriented_clean = F.orient_kidneys(F.patient_pair_features(clean, images, patients), patients, kidneys)
    early = np.c_[oriented_concat, oriented_clean]
    assignments = pd.read_csv(F.OUT / "patient_assignments.csv")
    deep_predictions = pd.read_csv(D.OUT / "development_predictions.csv")
    deep_reference = deep_predictions[deep_predictions.group.eq(REFERENCE)].set_index(["seed", "kidney_alias"]).score
    patient_lookup = {alias: index for index, alias in enumerate(patients.patient_alias)}
    records, predictions = [], []
    (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in F.SEEDS:
            mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
            train_patients = np.flatnonzero(patients.patient_alias.map(mapping).eq("train"))
            roles = kidneys.patient_alias.map(mapping)
            train, development = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
            aliases = kidneys.kidney_alias.iloc[development]
            deep_score = np.asarray([deep_reference.loc[(seed, alias)] for alias in aliases])
            scores, models = {}, {}
            for group, values, components in ((GROUPS[0], oriented_class, 32),
                                               (GROUPS[1], oriented_patch, 32),
                                               (GROUPS[2], oriented_concat, 64)):
                scores[group], model = D.fit_predict(values, train, development, labels, seed, components)
                models[group] = model
            scores[GROUPS[3]] = (scores[GROUPS[0]] + deep_score) / 2
            scores[GROUPS[4]] = (scores[GROUPS[1]] + deep_score) / 2
            scores[GROUPS[5]] = (scores[GROUPS[2]] + deep_score) / 2
            scores[GROUPS[6]], transform, model = F.fit_predict(early, train, development, labels, seed, 128)
            models[GROUPS[6]] = (transform, model)
            extreme_probability, extreme_model = fit_extreme(concat_pairs, train_patients, states, seed)
            saved = joblib.load(H.OUT / "models" / f"{seed}_hierarchical.joblib")
            deep_images = np.concatenate([np.load(D.OUT / name) for name in
                                          ("radimagenet_gray.npy", "radimagenet_color.npy", "usf_gray.npy")], axis=1)
            power_pairs = F.patient_pair_features(deep_images, images, patients)
            power_states = E.powerset_state_probabilities(saved["powerset_model"], saved["powerset_pre"], power_pairs)
            replaced = E.replace_extreme_probability(power_states, extreme_probability)
            halfblend = E.replace_extreme_probability(power_states, extreme_probability, True)

            def kidney_values(state_values):
                left, right = E.kidney_marginals(state_values)
                return np.asarray([left[patient_lookup[kidneys.patient_alias.iloc[index]]]
                                   if kidneys.side.iloc[index] == "left"
                                   else right[patient_lookup[kidneys.patient_alias.iloc[index]]]
                                   for index in development])

            scores[GROUPS[7]] = (kidney_values(replaced) + deep_score) / 2
            scores[GROUPS[8]] = (kidney_values(halfblend) + deep_score) / 2
            models["extreme"] = extreme_model
            joblib.dump(models, OUT / "models" / f"{seed}_dinov3.joblib", compress=3)
            for group in GROUPS:
                result = metric(labels[development], scores[group])
                records.append({"seed": seed, "role": "development", "group": group,
                                "patients": 20, "kidneys": len(development), **result})
                for local, row_index in enumerate(development):
                    predictions.append({"seed": seed, "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[row_index],
                        "kidney_alias": kidneys.kidney_alias.iloc[row_index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[row_index], "side": kidneys.side.iloc[row_index],
                        "label": int(labels[row_index]), "score": float(scores[group][local])})
            print(f"Completed clean DINOv3 draw {seed}", flush=True)
    frame = pd.DataFrame(records); frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    summary = frame.groupby("group")[["auroc", "ap"]].mean().sort_values(["auroc", "ap"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc > .8) & (summary.ap > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {"groups_above_both_development_targets": achieved,
        "test_scored": False, "goal_achieved": False,
        "reason": "Development-only subgroup cannot complete the full objective"})
    print(summary.to_string(), flush=True)


def bootstrap() -> None:
    require_lock(); frame = pd.read_csv(OUT / "development_predictions.csv")
    previous = pd.read_csv(D.OUT / "development_predictions.csv"); reference = previous[previous.group.eq(REFERENCE)]
    ci, paired, samples = cluster_intervals(pd.concat([frame, reference], ignore_index=True),
        list(GROUPS) + [REFERENCE], F.SEEDS, REFERENCE)
    ci.to_csv(OUT / "development_confidence_intervals.csv", index=False)
    paired.to_csv(OUT / "development_paired_intervals.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    write_json(OUT / "completion_audit.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "patients": 111, "kidneys": 212, "images": 418, "test_scored": False,
        "scored_unit": "kidney", "bootstrap_cluster": "patient"})
    print("Clean DINOv3 intervals completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "encode", "run", "bootstrap"])
    arguments = parser.parse_args(); globals()[arguments.stage]()
