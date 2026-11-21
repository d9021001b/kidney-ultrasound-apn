"""Locked, patient-level screening of locally available frozen visual encoders.

This is a follow-up exploratory experiment on the previously examined CYCH cohort.
It does not create lesion labels, alter kidney masks, or tune on the 20-patient test
partition.  Candidate advancement is decided solely from the fixed validation role.
"""
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

import renal_random_validation as R
import renal_enhancement_validation as E
import renal_local_features as L
import renal_voting_validation as V


ROOT = R.ROOT
OUT = Path(os.environ.get("CYCH_FOUNDATION_DIR", str(ROOT / "CYCH_foundation_screening_20260913")))
LOCAL_DINO_REPOSITORY = Path(r"C:\Users\udoo_w2\.cache\torch\hub\facebookresearch_dinov2_main")
LOCAL_DINO_CHECKPOINT = Path(r"C:\Users\udoo_w2\.cache\torch\hub\checkpoints\dinov2_vitl14_reg4_pretrain.pth")

GROUPS = ("radiomics_percentile", "radimagenet_percentile", "dinov2_vitl14_reg4", "radimagenet_inception_v3")
VOTES = {
    "fixed_radiomics_dense": ("radiomics_percentile", "radimagenet_percentile"),
    "fixed_radiomics_dense_dinov2": ("radiomics_percentile", "radimagenet_percentile", "dinov2_vitl14_reg4"),
    "fixed_radiomics_dense_inception": ("radiomics_percentile", "radimagenet_percentile", "radimagenet_inception_v3"),
    "fixed_all_four": ("radiomics_percentile", "radimagenet_percentile", "dinov2_vitl14_reg4", "radimagenet_inception_v3"),
}
BASELINE = "fixed_radiomics_dense"
CANDIDATE_VOTES = {
    "dinov2_vitl14_reg4": "fixed_radiomics_dense_dinov2",
    "radimagenet_inception_v3": "fixed_radiomics_dense_inception",
}
LABELS = {
    "radiomics_percentile": "ROI-normalized PyRadiomics + L2 logistic regression",
    "radimagenet_percentile": "Frozen RadImageNet DenseNet-121 + L2 logistic regression",
    "dinov2_vitl14_reg4": "Frozen DINOv2 ViT-L/14-reg4 + L2 logistic regression",
    "radimagenet_inception_v3": "Frozen RadImageNet InceptionV3 + L2 logistic regression",
    "fixed_radiomics_dense": "Fixed equal mean: radiomics + DenseNet-121",
    "fixed_radiomics_dense_dinov2": "Fixed equal mean: radiomics + DenseNet-121 + DINOv2",
    "fixed_radiomics_dense_inception": "Fixed equal mean: radiomics + DenseNet-121 + InceptionV3",
    "fixed_all_four": "Fixed equal mean: all four heads",
}
REFERENCES = [
    {
        "id": "dinov2",
        "citation": "Oquab M, et al. DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research. 2024.",
        "url": "https://github.com/facebookresearch/dinov2",
        "scope": "Frozen DINOv2 ViT-L/14 model with four registers. The local official model card specifies 1024-dimensional output, LVD-142M pretraining, and Apache-2.0 licensing. It is a general visual model, not a renal or ultrasound-specific model.",
    },
    {
        "id": "dinov3_pending_access",
        "citation": "Simeoni O, et al. DINOv3. arXiv:2508.10104. 2025.",
        "url": "https://github.com/facebookresearch/dinov3",
        "scope": "Registry-only primary next candidate: DINOv3 ViT-B/16 pretrained on LVD-1689M. The official Hugging Face repository facebook/dinov3-vitb16-pretrain-lvd1689m is manually gated and was unavailable without an authenticated account authorized by Meta. It is not substituted by DINOv2 and is not reported as evaluated.",
    },
    {
        "id": "registers",
        "citation": "Darcet T, et al. Vision Transformers Need Registers. ICLR. 2024.",
        "url": "https://arxiv.org/abs/2309.16588",
        "scope": "Architecture source for the four register tokens in the exact DINOv2 checkpoint name; no new register mechanism is introduced here.",
    },
    {
        "id": "radimagenet",
        "citation": "Mei X, et al. RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning. Radiology: Artificial Intelligence. 2022;4:e210315. doi:10.1148/ryai.210315.",
        "url": "https://github.com/BMEII-AI/RadImageNet",
        "scope": "Frozen local DenseNet-121 and newly screened InceptionV3 weights. Their documented preprocessing is reused as x / 127.5 - 1; InceptionV3 is a pretrained architecture, not a new proposed model.",
    },
    {
        "id": "inception",
        "citation": "Szegedy C, et al. Rethinking the Inception Architecture for Computer Vision. CVPR. 2016.",
        "url": "https://arxiv.org/abs/1512.00567",
        "scope": "Architecture citation for the existing RadImageNet InceptionV3 backbone.",
    },
    {
        "id": "logistic_regression",
        "citation": "scikit-learn LogisticRegression documentation, version 1.8.",
        "url": "https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html",
        "scope": "Same train-only median imputation, standardization, ANOVA top-32 selection, and L2 logistic-regression head as the previous local experiment.",
    },
    {
        "id": "usf_mae_pending",
        "citation": "Yusuf et al. USF-MAE: Ultrasound Foundation Model with Masked Autoencoders.",
        "url": "https://github.com/Yusufii9/USF-MAE",
        "scope": "Registry only; not downloaded or run in this experiment. Checkpoint provenance, license, and preprocessing compatibility must be verified before a separate locked experiment.",
    },
    {
        "id": "openus_pending",
        "citation": "Zheng et al. OpenUS: Ultrasound Foundation Model for Universal Ultrasound Image Analysis.",
        "url": "https://github.com/XZheng0427/OpenUS",
        "scope": "Registry only; not downloaded or run. The official implementation has platform-specific dependencies and requires a separately reproducible environment review.",
    },
]


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name, dtype={"patient_id": str})


def require_lock() -> None:
    lock = json.loads((OUT / "protocol_locked.json").read_text(encoding="utf-8"))
    if lock["sha256"] != R.sha256(OUT / "protocol.json"):
        raise RuntimeError("Protocol hash mismatch")


def model_digest(model) -> str:
    """Hash frozen parameters before/after feature extraction without changing them."""
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def canvas_arrays() -> tuple[list[np.ndarray], pd.DataFrame, pd.DataFrame]:
    df, patients = read("image_manifest.csv"), read("patients.csv")
    images = []
    for row in df.itertuples():
        gray = cv2.imread(str(OUT / "images" / "percentile" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(OUT / "masks" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None:
            raise RuntimeError(f"Missing ROI input for {row.image_id}")
        if np.any(gray[mask == 0]):
            raise RuntimeError(f"Nonzero pixels outside kidney mask for {row.image_id}")
        images.append(R.canvas_image(gray, (mask > 0).astype(np.uint8), channels="gray", size=224))
    return images, df, patients


def checkpoint_record(path: Path) -> dict:
    return {"path": str(path), "sha256": R.sha256(path), "bytes": path.stat().st_size}


def prepare() -> None:
    """Create an immutable derivative experiment and copy only checked existing inputs."""
    OUT.mkdir(exist_ok=True)
    for sub in ("features", "models", "images/percentile", "masks", "figures_600dpi", "qa", "source"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv"):
        source, target = L.OUT / name, OUT / name
        if target.exists() and R.sha256(source) != R.sha256(target):
            raise RuntimeError(f"Cohort drift in {name}")
        shutil.copy2(source, target)
    df, patients, splits = read("image_manifest.csv"), read("patients.csv"), read("random_patient_assignments.csv")
    if len(df) != 349 or df.patient_id.nunique() != 98 or len(patients) != 98:
        raise RuntimeError("Expected the established 98-patient / 349-image cohort")
    if set(splits.role) != {"train", "validation", "test"}:
        raise RuntimeError("Unexpected split roles")
    if not LOCAL_DINO_REPOSITORY.is_dir() or not LOCAL_DINO_CHECKPOINT.is_file():
        raise RuntimeError("Local DINOv2 repository or exact checkpoint is unavailable")
    from radimagenet_backbones import RADIMAGENET_DIR
    inception = RADIMAGENET_DIR / "InceptionV3.pt"
    if not inception.is_file():
        raise RuntimeError("Local RadImageNet InceptionV3 checkpoint is unavailable")
    for row in df.itertuples():
        for directory in ("images/percentile", "masks"):
            src = L.OUT / directory / f"{row.image_id}.png"
            dst = OUT / directory / src.name
            shutil.copy2(src, dst)
        if R.sha256(OUT / "masks" / f"{row.image_id}.png") != row.mask_sha256:
            raise RuntimeError(f"Mask hash mismatch for {row.image_id}")
    inherited = []
    for group in ("radiomics_percentile", "radimagenet_percentile"):
        src, dst = L.OUT / "features" / f"{group}.npy", OUT / "features" / f"{group}.npy"
        shutil.copy2(src, dst)
        a = np.load(dst)
        if a.shape[0] != 98 or not np.isfinite(a).all():
            raise RuntimeError(f"Invalid inherited group {group}")
        inherited.append({"group": group, "source": str(src), "source_sha256": R.sha256(src), "destination_sha256": R.sha256(dst), "shape": list(a.shape)})
    protocol = {
        "design": "Exploratory frozen-encoder screening on the previously examined CYCH cohort; not external validation and not an independent test-set claim.",
        "cohort": {"patients": 98, "first_exam_images": 349, "positive_patients": int(patients.label.sum()), "negative_patients": int((1 - patients.label).sum())},
        "seeds": R.SEEDS,
        "roles": {"train": 58, "development_validation": 20, "exploratory_test": 20},
        "segmentation": "Existing saved kidney masks and ROI P1-P99 images are copied hash-for-hash; no segmentation retraining or mask modification.",
        "inputs": "Masked grayscale renal ROI converted to the existing 224x224 centered canvas, replicated to RGB. No augmentation, GAN, synthetic image, caliper removal, inpainting, pseudo-lesion label, clinical covariate, or test-time augmentation.",
        "heads": "For every group and seed: train-only median imputation, standardization, ANOVA top-32 features, then fixed L2 logistic regression (C=0.1, liblinear, balanced, max_iter=2000).",
        "candidates": {
            "dinov2_vitl14_reg4": {"exact_name": "DINOv2 ViT-L/14-reg4", "training": "Frozen official LVD-142M self-supervised encoder", "input_normalization": "ImageNet mean/std", "expected_image_dimension": 1024, "checkpoint": checkpoint_record(LOCAL_DINO_CHECKPOINT), "repository": str(LOCAL_DINO_REPOSITORY), "repository_license": "Apache-2.0"},
            "radimagenet_inception_v3": {"exact_name": "RadImageNet InceptionV3", "training": "Frozen local RadImageNet encoder", "input_normalization": "x / 127.5 - 1", "expected_image_dimension": 2048, "checkpoint": checkpoint_record(inception)},
        },
        "fixed_workflows": VOTES,
        "selection_rule": {"baseline": BASELINE, "candidate_vote": CANDIDATE_VOTES, "data_role": "development_validation only", "advance_if": "Across ten draws, mean delta AUROC >= 0.03 and mean delta AP >= 0.02 versus baseline, with both metrics strictly higher in at least six draws.", "not_used": "The exploratory_test scores are never used to select, weight, calibrate, train, or advance a candidate."},
        "reporting": "Report all fixed single heads and fixed equal-weight probability means. Candidate status is determined before inspecting exploratory test comparison summaries.",
        "pending_models": ["DINOv3 ViT-B/16 LVD-1689M: official manually gated checkpoint requires an authorized authenticated account; not evaluated", "USF-MAE: checkpoint/license/preprocessing review required", "OpenUS: separate environment compatibility, checkpoint/license/preprocessing review required"],
        "cohort_hashes": {name: R.sha256(OUT / name) for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv")},
    }
    protocol = json.loads(json.dumps(protocol))
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError("Protocol already differs; choose a new experiment directory")
    R.write_json(path, protocol)
    if not (OUT / "protocol_locked.json").exists():
        R.write_json(OUT / "protocol_locked.json", {"utc": datetime.now(timezone.utc).isoformat(), "sha256": R.sha256(path)})
    R.write_json(OUT / "references.json", REFERENCES)
    R.write_json(OUT / "features" / "inherited_sources.json", inherited)
    R.write_json(OUT / "original_manuscript_preservation.json", {"path": str(ROOT / "yolo_aug_triEMR_PCA_SVM_renal_2022_v2.docx"), "sha256": R.sha256(ROOT / "yolo_aug_triEMR_PCA_SVM_renal_2022_v2.docx")})
    print("Locked foundation-model screening protocol: 98 patients, 349 unchanged kidney ROI images", flush=True)


def encoders() -> None:
    require_lock()
    import torch
    from radimagenet_backbones import RADIMAGENET_DIR, radimagenet_inception_v3

    torch.set_num_threads(4)
    images, df, patients = canvas_arrays()
    values = []

    def extract(name, model, dimension, normalization, checkpoint, batch_size=12):
        model = model.cuda().eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        before = model_digest(model)
        result = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                x = torch.from_numpy(np.stack(images[start:start + batch_size]).transpose(0, 3, 1, 2).copy()).cuda().float() / 255.0
                if normalization == "ImageNet mean/std":
                    mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
                    std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
                    x = (x - mean) / std
                elif normalization == "x / 127.5 - 1":
                    x = (x - 0.5) / 0.5
                else:
                    raise ValueError(normalization)
                result.append(model(x).detach().cpu().numpy())
        image_features = np.concatenate(result).astype(np.float32)
        if image_features.shape != (349, dimension) or not np.isfinite(image_features).all():
            raise RuntimeError(f"Unexpected {name} feature shape: {image_features.shape}")
        after = model_digest(model)
        if before != after:
            raise RuntimeError(f"Frozen {name} parameters changed during extraction")
        image_path = OUT / "features" / f"{name}_image.npy"
        patient_path = OUT / "features" / f"{name}.npy"
        np.save(image_path, image_features)
        np.save(patient_path, R.aggregate(image_features, df, patients))
        values.append({"group": name, "dimension_image": dimension, "dimension_patient": int(np.load(patient_path).shape[1]), "normalization": normalization, "checkpoint": str(checkpoint), "checkpoint_sha256": R.sha256(checkpoint), "trainable_parameters": 0, "state_before": before, "state_after": after, "image_features_sha256": R.sha256(image_path), "patient_features_sha256": R.sha256(patient_path)})
        print(f"Frozen {name}: {image_features.shape}; state hash unchanged", flush=True)
        del model
        torch.cuda.empty_cache()

    dino = torch.hub.load(str(LOCAL_DINO_REPOSITORY), "dinov2_vitl14_reg", source="local", pretrained=True)
    extract("dinov2_vitl14_reg4", dino, 1024, "ImageNet mean/std", LOCAL_DINO_CHECKPOINT, batch_size=8)
    extract("radimagenet_inception_v3", radimagenet_inception_v3(), 2048, "x / 127.5 - 1", RADIMAGENET_DIR / "InceptionV3.pt", batch_size=16)
    R.write_json(OUT / "features" / "encoder_provenance.json", values)


def load_features() -> dict[str, np.ndarray]:
    result = {}
    for group in GROUPS:
        value = np.load(OUT / "features" / f"{group}.npy")
        if value.shape[0] != 98 or not np.isfinite(value).all():
            raise RuntimeError(f"Invalid patient features for {group}")
        result[group] = value
    return result


def fixed_votes(probabilities: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = dict(probabilities)
    for workflow, members in VOTES.items():
        p = np.column_stack([probabilities[member] for member in members])
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f"Invalid native probabilities for {workflow}")
        result[workflow] = p.mean(axis=1)
    return result


def stage_decisions(development: pd.DataFrame) -> list[dict]:
    """Apply pre-registered development-only advancement criteria."""
    baseline = development[development.workflow.eq(BASELINE)].set_index("seed")
    decisions = []
    for candidate, workflow in CANDIDATE_VOTES.items():
        trial = development[development.workflow.eq(workflow)].set_index("seed").loc[baseline.index]
        auroc_delta = trial.auroc - baseline.auroc
        ap_delta = trial.average_precision - baseline.average_precision
        decisions.append({
            "candidate": candidate,
            "workflow": workflow,
            "baseline": BASELINE,
            "data_role": "development_validation",
            "draws": int(len(trial)),
            "delta_auroc_mean": float(auroc_delta.mean()),
            "delta_average_precision_mean": float(ap_delta.mean()),
            "both_improved_draws": int(((auroc_delta > 0) & (ap_delta > 0)).sum()),
            "thresholds": {"mean_delta_auroc": 0.03, "mean_delta_average_precision": 0.02, "both_improved_draws": 6},
            "advance": bool(auroc_delta.mean() >= 0.03 and ap_delta.mean() >= 0.02 and ((auroc_delta > 0) & (ap_delta > 0)).sum() >= 6),
        })
    return decisions


def evaluate() -> None:
    require_lock()
    patients, splits = read("patients.csv"), read("random_patient_assignments.csv")
    y = patients.label.to_numpy(dtype=int)
    arrays = load_features()
    metrics, predictions, heads = [], [], []
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            per = splits[splits.seed.eq(seed)]
            ix = {role: np.flatnonzero(patients.patient_id.isin(per.loc[per.role.eq(role), "patient_id"])) for role in ("train", "validation", "test")}
            if set(ix["train"]) & set(ix["validation"]) or set(ix["train"]) & set(ix["test"]) or set(ix["validation"]) & set(ix["test"]):
                raise RuntimeError(f"Patient role overlap at seed {seed}")
            probabilities = {}
            for group, x in arrays.items():
                head = V.make_head("lr", seed, x.shape[1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    warnings.simplefilter("ignore", RuntimeWarning)
                    head.fit(x[ix["train"]], y[ix["train"]])
                probabilities[group] = head.predict_proba(x)[:, list(head.classes_).index(1)]
                model_path = OUT / "models" / f"{group}_lr_{seed}.joblib"
                joblib.dump(head, model_path, compress=3)
                heads.append({"seed": seed, "group": group, "model": str(model_path.relative_to(OUT)), "sha256": R.sha256(model_path), "train_patients": patients.patient_id.iloc[ix["train"]].tolist(), "selected_indices": head[2].get_support(indices=True).tolist(), "scaler_n": int(head[1].n_samples_seen_)})
            scores = fixed_votes(probabilities)
            for role in ("validation", "test"):
                for workflow, score in scores.items():
                    item = V.score_metrics(y[ix[role]], score[ix[role]])
                    metrics.append({"seed": seed, "role": "development_validation" if role == "validation" else "exploratory_test", "workflow": workflow, "kind": "single_head" if workflow in GROUPS else "fixed_equal_probability_mean", "members": ";".join(VOTES[workflow]) if workflow in VOTES else workflow, **item})
                    for patient_index, value in zip(ix[role], score[ix[role]]):
                        predictions.append({"seed": seed, "role": "development_validation" if role == "validation" else "exploratory_test", "workflow": workflow, "patient_id": patients.patient_id.iloc[patient_index], "label": int(y[patient_index]), "score": float(value), "prediction": int(value > 0.5)})
            baseline_item = next(row for row in reversed(metrics) if row["seed"] == seed and row["role"] == "development_validation" and row["workflow"] == BASELINE)
            print(f"{seed}: development fixed radiomics + DenseNet AUROC={baseline_item['auroc']:.4f}, AP={baseline_item['average_precision']:.4f}", flush=True)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / "metrics_by_role_per_draw.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "patient_predictions_by_role.csv", index=False)
    R.write_json(OUT / "head_provenance.json", heads)
    development = metric_frame[metric_frame.role.eq("development_validation")]
    R.write_json(OUT / "candidate_advancement_decisions.json", stage_decisions(development))
    summarize()


def summarize() -> None:
    data = pd.read_csv(OUT / "metrics_by_role_per_draw.csv")
    rows = []
    for (role, workflow), group in data.groupby(["role", "workflow"], sort=False):
        row = {"role": role, "workflow": workflow, "label": LABELS[workflow], "kind": group.kind.iloc[0], "members": group.members.iloc[0], "draws": int(len(group)), "joint_target_draws": int(((group.auroc > 0.8) & (group.average_precision > 0.8)).sum())}
        for field in ("auroc", "average_precision", "auprc_trapezoid", "brier", "log_loss", "sensitivity", "specificity", "accuracy"):
            for statistic in ("mean", "std", "min", "max"):
                row[f"{field}_{statistic}"] = float(getattr(group[field], statistic)())
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "summary_by_role.csv", index=False)
    development = data[data.role.eq("development_validation")]
    baseline = development[development.workflow.eq(BASELINE)].set_index("seed")
    comparisons = []
    for workflow in VOTES:
        trial = development[development.workflow.eq(workflow)].set_index("seed").loc[baseline.index]
        comparisons.append({"role": "development_validation", "candidate": workflow, "baseline": BASELINE, "delta_auroc_mean": float((trial.auroc - baseline.auroc).mean()), "delta_average_precision_mean": float((trial.average_precision - baseline.average_precision).mean()), "both_improved_draws": int(((trial.auroc > baseline.auroc) & (trial.average_precision > baseline.average_precision)).sum()), "selection_eligible": workflow in CANDIDATE_VOTES.values()})
    pd.DataFrame(comparisons).to_csv(OUT / "development_fixed_vote_comparisons.csv", index=False)


def export_figure(fig, name: str) -> None:
    from PIL import Image
    png = OUT / "figures_600dpi" / f"{name}.png"
    fig.savefig(png, dpi=600, facecolor="white")
    with Image.open(png) as image:
        image.save(png.with_suffix(".tiff"), compression="tiff_lzw", dpi=(600, 600))
    fig.savefig(OUT / "qa" / f"{name}_preview.png", dpi=150, facecolor="white")


def report() -> None:
    require_lock()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = pd.read_csv(OUT / "metrics_by_role_per_draw.csv")
    workflows = [BASELINE, "fixed_radiomics_dense_dinov2", "fixed_radiomics_dense_inception", "fixed_all_four"]
    plot_labels = ["Radiomics + DenseNet-121", "+ DINOv2", "+ InceptionV3", "All four heads"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = {"development_validation": "#247a72", "exploratory_test": "#b66048"}
    for axis, field, title in zip(axes, ("auroc", "average_precision"), ("AUROC", "Average precision")):
        for row_index, workflow in enumerate(workflows):
            for role, marker in (("development_validation", "o"), ("exploratory_test", "s")):
                values = data.loc[(data.workflow.eq(workflow)) & (data.role.eq(role)), field].to_numpy()
                axis.scatter(values, row_index + np.linspace(-0.12, 0.12, len(values)), marker=marker, s=24, alpha=0.72, color=colors[role], label=role.replace("_", " ") if row_index == 0 else None)
                axis.scatter(values.mean(), row_index, marker="D", s=36, color="#202020")
        axis.axvline(0.8, color="#9f3150", lw=1, ls="--")
        axis.set_xlim(0, 1)
        axis.set_yticks(range(len(workflows)), plot_labels if axis is axes[0] else [""] * len(workflows), fontsize=9)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8, frameon=False)
    figure.suptitle("Frozen-encoder screening: development role decides advancement; test role is exploratory", fontsize=12)
    figure.subplots_adjust(left=0.19, right=0.99, bottom=0.20, top=0.86, wspace=0.30)
    export_figure(figure, "Figure_F1_foundation_screening")
    plt.close(figure)
    decisions = json.loads((OUT / "candidate_advancement_decisions.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(OUT / "summary_by_role.csv")
    lines = [
        "# CYCH frozen-encoder screening",
        "",
        "This is an exploratory continuation on the same 98 patients and 349 first-exam kidney ROI images. Existing segmentation masks and fixed patient assignments were copied hash-for-hash. No disease, scar, swelling, or pseudo-label was created from the images.",
        "",
        "## Locked procedure",
        "",
        "- Each draw fits every head on 58 patients only. The 20-patient development-validation role alone decides candidate advancement. The 20-patient test role is explicitly exploratory because prior analyses have already used these cohort partitions.",
        "- All probability combinations are fixed arithmetic means. There is no learned voting weight, model selection, temperature calibration, GAN, augmentation, or test-time augmentation in this screening stage.",
        "- Advancement threshold: mean development delta AUROC >= 0.03 and AP >= 0.02 versus the fixed radiomics + DenseNet-121 mean, with both metrics higher in at least six of ten draws.",
        "",
        "## Candidate status",
        "",
    ]
    for item in decisions:
        lines.append(f"- {LABELS[item['candidate']]}: {'ADVANCE' if item['advance'] else 'do not advance'}; development delta AUROC {item['delta_auroc_mean']:+.4f}, delta AP {item['delta_average_precision_mean']:+.4f}, both improved {item['both_improved_draws']}/10.")
    lines.extend(["", "## Mean metrics", "", "| Role | Workflow | AUROC mean | AP mean | AUROC/AP >0.8 draws |", "|---|---|---:|---:|---:|"])
    for row in summary.itertuples():
        lines.append(f"| {row.role} | {row.workflow} | {row.auroc_mean:.4f} | {row.average_precision_mean:.4f} | {row.joint_target_draws} |")
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- These results do not establish a generalizable accuracy claim, clinical decision threshold, renal-inflammation imaging biomarker, or ROC/AP target attainment. Repeated random partitions are not independent cohorts.",
        "- DINOv2 is general-domain pretraining. RadImageNet is medical-image pretraining but not a renal-inflammation-specific checkpoint. A passed candidate would justify a separately locked, training-patient-only MIL or fine-tuning study, not immediate manuscript efficacy language.",
        "- DINOv3 ViT-B/16 is the primary next foundation-model candidate, but its official repository is manually gated and unavailable to the current unauthenticated environment. USF-MAE and OpenUS also remain registry candidates only; none of these was evaluated in this artifact.",
        "",
        "## Artifacts",
        "",
        "- `protocol.json` and `protocol_locked.json`: pre-specified data, models, and advancement rule.",
        "- `references.json`: sources for every named model and method.",
        "- `metrics_by_role_per_draw.csv`, `summary_by_role.csv`, and `candidate_advancement_decisions.json`: auditable result tables.",
        "- `figures_600dpi/Figure_F1_foundation_screening.png` and `.tiff`: 600 dpi result figure.",
    ])
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package() -> None:
    require_lock()
    report()
    archive = ROOT / f"{OUT.name}.zip"
    if archive.exists():
        archive.unlink()
    shutil.make_archive(str(archive.with_suffix("")), "zip", ROOT, OUT.name)
    R.write_json(OUT / "package_manifest.json", {"archive": str(archive), "archive_sha256": R.sha256(archive), "archive_bytes": archive.stat().st_size, "created_utc": datetime.now(timezone.utc).isoformat()})
    print(f"Package: {archive}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "encoders", "evaluate", "report", "package", "all"))
    args = parser.parse_args()
    for stage in (("prepare", "encoders", "evaluate", "report", "package") if args.stage == "all" else (args.stage,)):
        globals()[stage]()
