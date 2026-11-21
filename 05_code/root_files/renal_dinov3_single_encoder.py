"""Locked DINOv3 frozen-feature gate before any DINOv3 voting experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import cv2
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import renal_local_features as L
import renal_random_validation as R
import renal_voting_validation as V


ROOT = R.ROOT
OUT = Path(os.environ.get("CYCH_DINOV3_SINGLE_DIR", str(ROOT / "CYCH_dinov3_single_encoder_20260913")))
DINO3 = ROOT / "dinov3_vith16plus_weights_20260913"
BASELINE = "radimagenet_percentile"
CANDIDATE = "dinov3_vith16plus"
GROUPS = (BASELINE, CANDIDATE)
LABELS = {
    BASELINE: "Frozen RadImageNet DenseNet-121",
    CANDIDATE: "Frozen DINOv3 ViT-H+/16",
}
REFERENCES = [
    {
        "id": "dinov3",
        "citation": "Simeoni O, et al. DINOv3. arXiv:2508.10104. 2025.",
        "url": "https://github.com/facebookresearch/dinov3",
        "scope": "Exact local model ID facebook/dinov3-vith16plus-pretrain-lvd1689m. Frozen ViT-H+/16, LVD-1689M web-pretrained/distilled backbone. It is a general visual encoder and is not renal- or ultrasound-specific.",
    },
    {
        "id": "dinov3_license",
        "citation": "Meta. DINOv3 License. 2025.",
        "url": "https://ai.meta.com/resources/models-and-libraries/dinov3-license/",
        "scope": "The local bundle includes the DINOv3 license. Publication using DINO materials requires acknowledgement under the license terms; this experiment does not redistribute the checkpoint.",
    },
    {
        "id": "radimagenet",
        "citation": "Mei X, et al. RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning. Radiology: Artificial Intelligence. 2022;4:e210315. doi:10.1148/ryai.210315.",
        "url": "https://github.com/BMEII-AI/RadImageNet",
        "scope": "Existing frozen DenseNet-121 comparator and its established local x / 127.5 - 1 preprocessing.",
    },
    {
        "id": "logistic_regression",
        "citation": "scikit-learn LogisticRegression documentation, version 1.8.",
        "url": "https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html",
        "scope": "Fixed train-only median imputation, standardization, ANOVA top-32 selection, and L2 logistic-regression head.",
    },
]


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name, dtype={"patient_id": str})


def require_lock() -> None:
    lock = json.loads((OUT / "protocol_locked.json").read_text(encoding="utf-8"))
    if lock["sha256"] != R.sha256(OUT / "protocol.json"):
        raise RuntimeError("Protocol hash mismatch")


def sha256(path: Path) -> str:
    return R.sha256(path)


def verify_bundle() -> dict:
    """Validate the supplied safetensors bundle without running its helper scripts."""
    required = ("config.json", "preprocessor_config.json", "model.safetensors", "LICENSE.md", "README.md", "WEIGHT_MANIFEST.json")
    missing = [name for name in required if not (DINO3 / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete DINOv3 bundle: {missing}")
    manifest = json.loads((DINO3 / "WEIGHT_MANIFEST.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = DINO3 / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"DINOv3 bundle manifest mismatch: {item['path']}")
    config = json.loads((DINO3 / "config.json").read_text(encoding="utf-8"))
    expected = {"hidden_size": 1280, "num_hidden_layers": 32, "num_attention_heads": 20, "num_register_tokens": 4, "patch_size": 16, "image_size": 224}
    if {key: config.get(key) for key in expected} != expected:
        raise RuntimeError("DINOv3 configuration does not match the expected ViT-H+/16 geometry")
    from safetensors import safe_open
    with safe_open(DINO3 / "model.safetensors", framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        expected_shapes = {
            "embeddings.cls_token": (1, 1, 1280),
            "embeddings.register_tokens": (1, 4, 1280),
            "embeddings.patch_embeddings.weight": (1280, 3, 16, 16),
            "layer.31.norm1.weight": (1280,),
            "norm.weight": (1280,),
        }
        actual_shapes = {key: tuple(handle.get_tensor(key).shape) for key in expected_shapes}
        if len(keys) != manifest["tensor_count"] or actual_shapes != expected_shapes:
            raise RuntimeError("DINOv3 safetensors structure does not match the manifest/configuration")
        metadata = handle.metadata()
    return {"bundle": str(DINO3), "model_id": manifest["model_id"], "variant": manifest["variant"], "weight_bytes": manifest["weight_bytes"], "weight_sha256": manifest["weight_sha256"], "tensor_count": len(keys), "safetensors_metadata": metadata, "configuration": expected, "files_verified": len(manifest["files"])}


def prepare() -> None:
    OUT.mkdir(exist_ok=True)
    for sub in ("features", "models", "images/percentile", "masks", "figures_600dpi", "qa", "source"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv"):
        source, target = L.OUT / name, OUT / name
        if target.exists() and sha256(target) != sha256(source):
            raise RuntimeError(f"Cohort drift in {name}")
        shutil.copy2(source, target)
    df, patients, split = read("image_manifest.csv"), read("patients.csv"), read("random_patient_assignments.csv")
    if len(df) != 349 or len(patients) != 98 or df.patient_id.nunique() != 98 or set(split.role) != {"train", "validation", "test"}:
        raise RuntimeError("Unexpected CYCH image-label cohort or patient split")
    for row in df.itertuples():
        for directory in ("images/percentile", "masks"):
            src, dst = L.OUT / directory / f"{row.image_id}.png", OUT / directory / f"{row.image_id}.png"
            shutil.copy2(src, dst)
        if sha256(OUT / "masks" / f"{row.image_id}.png") != row.mask_sha256:
            raise RuntimeError(f"Mask hash mismatch: {row.image_id}")
    src, dst = L.OUT / "features" / f"{BASELINE}.npy", OUT / "features" / f"{BASELINE}.npy"
    shutil.copy2(src, dst)
    baseline = np.load(dst)
    if baseline.shape[0] != 98 or not np.isfinite(baseline).all():
        raise RuntimeError("Invalid inherited DenseNet patient features")
    bundle = verify_bundle()
    protocol = {
        "design": "DINOv3 single-encoder gate before any DINOv3 voting experiment; exploratory analysis of the same previously examined CYCH cohort.",
        "cohort": {"patients": 98, "first_exam_images": 349, "positive_patients": int(patients.label.sum()), "negative_patients": int((1 - patients.label).sum())},
        "fixed_roles": {"train": 58, "development_validation": 20, "exploratory_test": 20},
        "seeds": R.SEEDS,
        "segmentation_and_input": "Existing saved kidney masks and ROI P1-P99 images copied hash-for-hash. Centered 224x224 masked grayscale canvas replicated to RGB. No augmentation, GAN, synthetic image, caliper removal, inpainting, pseudo-lesion label, clinical covariate, or test-time augmentation.",
        "candidate": {"name": "DINOv3 ViT-H+/16", "group": CANDIDATE, "feature": "class token only from last_hidden_state[:, 0, :]", "image_dimension": 1280, "patient_aggregation": "mean and standard deviation across each patient's images", "normalization": "ImageNet mean/std from the local official preprocessor configuration", "frozen": True, "bundle_verification": bundle},
        "comparator": {"name": "RadImageNet DenseNet-121", "group": BASELINE, "existing_patient_feature_sha256": sha256(dst), "feature_dimension": int(baseline.shape[1])},
        "head": "Identical fixed L2 logistic regression for both groups: train-only median imputation, standardization, ANOVA top-32, C=0.1, liblinear, class_weight=balanced, max_iter=2000.",
        "single_encoder_gate": {"selection_role": "development_validation only", "advance_if": "DINOv3 mean AUROC and mean AP are each no more than 0.02 lower than the DenseNet comparator across the ten fixed development draws. The test role cannot affect this decision.", "next_stage": "Only an advance decision permits a separately locked fixed equal-probability comparison of radiomics + DenseNet versus radiomics + DenseNet + DINOv3."},
        "limitations": "The complete 98-patient cohort has already been explored. Test-role values are exploratory, not untouched confirmation; the local bundle is internally integrity-verified but source download provenance cannot be re-established from a local folder alone.",
        "cohort_hashes": {name: sha256(OUT / name) for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv")},
    }
    path = OUT / "protocol.json"
    protocol = json.loads(json.dumps(protocol))
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError("Protocol already differs; choose a new experiment directory")
    R.write_json(path, protocol)
    if not (OUT / "protocol_locked.json").exists():
        R.write_json(OUT / "protocol_locked.json", {"utc": datetime.now(timezone.utc).isoformat(), "sha256": sha256(path)})
    R.write_json(OUT / "references.json", REFERENCES)
    R.write_json(OUT / "features" / "inherited_source.json", {"source": str(src), "source_sha256": sha256(src), "destination_sha256": sha256(dst), "shape": list(baseline.shape)})
    R.write_json(OUT / "original_manuscript_preservation.json", {"path": str(ROOT / "yolo_aug_triEMR_PCA_SVM_renal_2022_v2.docx"), "sha256": sha256(ROOT / "yolo_aug_triEMR_PCA_SVM_renal_2022_v2.docx")})
    print("Locked DINOv3 single-encoder gate: 98 patients and 349 unchanged kidney ROI images", flush=True)


def encode() -> None:
    require_lock()
    import torch
    from transformers import AutoImageProcessor, AutoModel

    torch.set_num_threads(4)
    df, patients = read("image_manifest.csv"), read("patients.csv")
    images = []
    for row in df.itertuples():
        gray = cv2.imread(str(OUT / "images" / "percentile" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(OUT / "masks" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None or np.any(gray[mask == 0]):
            raise RuntimeError(f"Invalid masked ROI input for {row.image_id}")
        images.append(R.canvas_image(gray, (mask > 0).astype(np.uint8), channels="gray", size=224))
    pre_hash = sha256(DINO3 / "model.safetensors")
    processor = AutoImageProcessor.from_pretrained(DINO3, local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(DINO3, local_files_only=True, trust_remote_code=True, dtype=torch.float32).cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("DINOv3 model is not fully frozen")
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(images), 8):
            array = np.stack(images[start:start + 8])
            batch = processor(images=list(array), return_tensors="pt")
            output = model(pixel_values=batch["pixel_values"].cuda()).last_hidden_state
            if output.shape[1:] != (201, 1280):
                raise RuntimeError(f"Unexpected DINOv3 token geometry: {tuple(output.shape)}")
            chunks.append(output[:, 0, :].detach().cpu().numpy())
    image_features = np.concatenate(chunks).astype(np.float32)
    if image_features.shape != (349, 1280) or not np.isfinite(image_features).all():
        raise RuntimeError(f"Invalid DINOv3 feature matrix: {image_features.shape}")
    post_hash = sha256(DINO3 / "model.safetensors")
    if pre_hash != post_hash:
        raise RuntimeError("DINOv3 checkpoint changed during feature extraction")
    image_path, patient_path = OUT / "features" / f"{CANDIDATE}_image.npy", OUT / "features" / f"{CANDIDATE}.npy"
    np.save(image_path, image_features)
    np.save(patient_path, R.aggregate(image_features, df, patients))
    R.write_json(OUT / "features" / "dinov3_provenance.json", {**verify_bundle(), "input_count": len(images), "image_feature_shape": list(image_features.shape), "patient_feature_shape": list(np.load(patient_path).shape), "image_feature_sha256": sha256(image_path), "patient_feature_sha256": sha256(patient_path), "checkpoint_sha256_before": pre_hash, "checkpoint_sha256_after": post_hash, "trainable_parameters": 0, "transformers_version": __import__("transformers").__version__})
    print(f"Frozen DINOv3 class-token features: {image_features.shape}; checkpoint hash unchanged", flush=True)
    del model
    torch.cuda.empty_cache()


def advancement_decision(development: pd.DataFrame) -> dict:
    dense = development[development.group.eq(BASELINE)].set_index("seed")
    dino = development[development.group.eq(CANDIDATE)].set_index("seed").loc[dense.index]
    delta_auroc = dino.auroc - dense.auroc
    delta_ap = dino.average_precision - dense.average_precision
    threshold = -0.02
    within_margin = bool(delta_auroc.mean() >= threshold - 1e-12 and delta_ap.mean() >= threshold - 1e-12)
    return {
        "candidate": CANDIDATE,
        "comparator": BASELINE,
        "data_role": "development_validation",
        "draws": int(len(dense)),
        "delta_auroc_mean": float(delta_auroc.mean()),
        "delta_average_precision_mean": float(delta_ap.mean()),
        "thresholds": {"minimum_mean_delta_auroc": threshold, "minimum_mean_delta_average_precision": threshold, "numeric_tolerance": 1e-12},
        "advance_to_fixed_vote": within_margin,
        "reason": "Both development mean metrics are within 0.02 of DenseNet." if within_margin else "At least one development mean metric is more than 0.02 below DenseNet; DINOv3 voting is not permitted.",
    }


def evaluate() -> None:
    require_lock()
    patients, split = read("patients.csv"), read("random_patient_assignments.csv")
    y = patients.label.to_numpy(dtype=int)
    arrays = {group: np.load(OUT / "features" / f"{group}.npy") for group in GROUPS}
    records, predictions, heads = [], [], []
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            per = split[split.seed.eq(seed)]
            indices = {role: np.flatnonzero(patients.patient_id.isin(per.loc[per.role.eq(role), "patient_id"])) for role in ("train", "validation", "test")}
            if any(set(indices[a]) & set(indices[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
                raise RuntimeError(f"Patient role overlap at seed {seed}")
            for group, values in arrays.items():
                head = V.make_head("lr", seed, values.shape[1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    warnings.simplefilter("ignore", RuntimeWarning)
                    head.fit(values[indices["train"]], y[indices["train"]])
                score = head.predict_proba(values)[:, list(head.classes_).index(1)]
                path = OUT / "models" / f"{group}_lr_{seed}.joblib"
                joblib.dump(head, path, compress=3)
                heads.append({"seed": seed, "group": group, "model": str(path.relative_to(OUT)), "sha256": sha256(path), "train_patients": patients.patient_id.iloc[indices["train"]].tolist(), "selected_indices": head[2].get_support(indices=True).tolist(), "scaler_n": int(head[1].n_samples_seen_)})
                for role in ("validation", "test"):
                    role_name = "development_validation" if role == "validation" else "exploratory_test"
                    records.append({"seed": seed, "role": role_name, "group": group, **V.score_metrics(y[indices[role]], score[indices[role]])})
                    for index, value in zip(indices[role], score[indices[role]]):
                        predictions.append({"seed": seed, "role": role_name, "group": group, "patient_id": patients.patient_id.iloc[index], "label": int(y[index]), "score": float(value), "prediction": int(value > 0.5)})
            item = next(row for row in reversed(records) if row["seed"] == seed and row["role"] == "development_validation" and row["group"] == CANDIDATE)
            print(f"{seed}: DINOv3 development AUROC={item['auroc']:.4f}, AP={item['average_precision']:.4f}", flush=True)
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT / "metrics_by_role_per_draw.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "patient_predictions_by_role.csv", index=False)
    R.write_json(OUT / "head_provenance.json", heads)
    R.write_json(OUT / "dinov3_single_encoder_decision.json", advancement_decision(metrics[metrics.role.eq("development_validation")]))
    summarize()


def summarize() -> None:
    data = pd.read_csv(OUT / "metrics_by_role_per_draw.csv")
    rows = []
    for (role, group), values in data.groupby(["role", "group"], sort=False):
        row = {"role": role, "group": group, "label": LABELS[group], "draws": int(len(values)), "joint_target_draws": int(((values.auroc > 0.8) & (values.average_precision > 0.8)).sum())}
        for metric in ("auroc", "average_precision", "auprc_trapezoid", "brier", "log_loss", "sensitivity", "specificity", "accuracy"):
            for statistic in ("mean", "std", "min", "max"):
                row[f"{metric}_{statistic}"] = float(getattr(values[metric], statistic)())
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "summary_by_role.csv", index=False)


def export_figure(figure, name: str) -> None:
    from PIL import Image
    png = OUT / "figures_600dpi" / f"{name}.png"
    figure.savefig(png, dpi=600, facecolor="white")
    with Image.open(png) as image:
        image.save(png.with_suffix(".tiff"), compression="tiff_lzw", dpi=(600, 600))
    figure.savefig(OUT / "qa" / f"{name}_preview.png", dpi=150, facecolor="white")


def report() -> None:
    require_lock()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, summary = pd.read_csv(OUT / "metrics_by_role_per_draw.csv"), pd.read_csv(OUT / "summary_by_role.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.9))
    colors = {"development_validation": "#247a72", "exploratory_test": "#b66048"}
    for axis, metric, title in zip(axes, ("auroc", "average_precision"), ("AUROC", "Average precision")):
        for y_position, group in enumerate(GROUPS):
            for role, marker in (("development_validation", "o"), ("exploratory_test", "s")):
                values = data.loc[(data.group.eq(group)) & (data.role.eq(role)), metric].to_numpy()
                axis.scatter(values, y_position + np.linspace(-0.10, 0.10, len(values)), color=colors[role], marker=marker, s=27, alpha=0.75, label=role.replace("_", " ") if y_position == 0 else None)
                axis.scatter(values.mean(), y_position, color="#202020", marker="D", s=38)
        axis.axvline(0.8, color="#9f3150", lw=1, ls="--")
        axis.set_xlim(0, 1)
        axis.set_yticks(range(len(GROUPS)), [LABELS[group] for group in GROUPS] if axis is axes[0] else ["", ""])
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8, frameon=False)
    figure.suptitle("DINOv3 single-encoder gate: voting is prohibited until the development rule passes", fontsize=12)
    figure.subplots_adjust(left=0.27, right=0.99, bottom=0.20, top=0.84, wspace=0.30)
    export_figure(figure, "Figure_D3_1_single_encoder_gate")
    plt.close(figure)
    decision = json.loads((OUT / "dinov3_single_encoder_decision.json").read_text(encoding="utf-8"))
    lines = [
        "# DINOv3 single-encoder gate",
        "",
        "This locked experiment evaluates the local DINOv3 ViT-H+/16 representation alone against the existing frozen RadImageNet DenseNet-121 representation. Both use the same kidney masks, patient splits, fixed logistic-regression head, and patient mean/standard-deviation aggregation.",
        "",
        "The DINOv3 image feature is the final class token only. No DINOv3 fine-tuning, augmentation, GAN, voting, artifact removal, clinical covariate, lesion annotation, or test-time augmentation is performed.",
        "",
        "## Development-only gate",
        "",
        f"- Decision: {'ADVANCE to a separately locked fixed-vote experiment' if decision['advance_to_fixed_vote'] else 'DO NOT ADVANCE to DINOv3 voting'}.",
        f"- Mean development delta versus DenseNet: AUROC {decision['delta_auroc_mean']:+.4f}; AP {decision['delta_average_precision_mean']:+.4f}.",
        f"- Rule: DINOv3 must be no more than 0.02 lower on both development mean AUROC and AP. {decision['reason']}",
        "",
        "## Mean metrics",
        "",
        "| Role | Representation | AUROC mean | AP mean | AUROC/AP >0.8 draws |",
        "|---|---|---:|---:|---:|",
    ]
    for item in summary.itertuples():
        lines.append(f"| {item.role} | {item.label} | {item.auroc_mean:.4f} | {item.average_precision_mean:.4f} | {item.joint_target_draws} |")
    lines.extend([
        "",
        "## Limits",
        "",
        "- The test role is exploratory and cannot affect advancement because the same 98-patient cohort has already been explored.",
        "- Passing this gate would demonstrate only a candidate representation suitable for a next comparison, not a clinical performance claim or evidence that AUROC/AP >0.8 has been confirmed.",
        "- The supplied local DINOv3 bundle is internally hash- and structure-verified. Its original remote-download chain is not independently recoverable from the local folder.",
    ])
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package() -> None:
    require_lock()
    report()
    archive = ROOT / f"{OUT.name}.zip"
    if archive.exists():
        archive.unlink()
    shutil.make_archive(str(archive.with_suffix("")), "zip", ROOT, OUT.name)
    R.write_json(OUT / "package_manifest.json", {"archive": str(archive), "archive_sha256": sha256(archive), "archive_bytes": archive.stat().st_size, "created_utc": datetime.now(timezone.utc).isoformat()})
    print(f"Package: {archive}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "encode", "evaluate", "report", "package", "all"))
    args = parser.parse_args()
    for stage in (("prepare", "encode", "evaluate", "report", "package") if args.stage == "all" else (args.stage,)):
        globals()[stage]()
