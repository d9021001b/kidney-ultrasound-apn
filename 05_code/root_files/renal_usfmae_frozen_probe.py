"""Locked USF-MAE frozen-encoder probe on the existing CYCH ROI cohort.

This is deliberately a one-representation screen.  It does not fine-tune the
encoder, construct a new ensemble, or use the exploratory-test role to choose
a model.  The compatibility adapter is kept here rather than modifying either
official third-party repository.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import warnings
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import renal_local_features as L
import renal_random_validation as R
import renal_voting_validation as V


OUT = R.ROOT / "CYCH_usfmae_frozen_probe_20260913"
USF_REPOSITORY = R.ROOT / "third_party" / "USF-MAE"
MAE_REPOSITORY = R.ROOT / "third_party" / "facebook-mae"
CHECKPOINT = R.ROOT / "external_models" / "USF-MAE" / "USF-MAE_full_pretrain_43dataset_100epochs.pt"
DENSE_REFERENCE = R.ROOT / "CYCH_dinov3_native_probe_20260913"
GROUP = "USF-MAE-CLS"
COMPARATOR = "DN-B"
BATCH_SIZE = 8
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

REFERENCES = [
    {
        "citation": "Megahed Y, et al. USF-MAE: An ultrasound self-supervised foundation model. Official repository and 100-epoch checkpoint.",
        "url": "https://github.com/Yusufii9/USF-MAE",
        "scope": "Official source for the 100-epoch 43-dataset checkpoint, its stated ultrasound pretraining collection and input normalization example. This probe uses only its frozen encoder and does not claim an independent renal validation because kidneyUS is listed in the pretraining collection.",
    },
    {
        "citation": "He K, Chen X, Xie S, Li Y, Dollar P, Girshick R. Masked Autoencoders Are Scalable Vision Learners. CVPR. 2022.",
        "url": "https://arxiv.org/abs/2111.06377",
        "scope": "Official MAE ViT-B/16 implementation used to instantiate the checkpoint. The local source checkout is CC BY-NC 4.0; this experiment is a non-commercial research probe and is not a deployment license.",
    },
    {
        "citation": "scikit-learn LogisticRegression documentation.",
        "url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        "scope": "Fixed, train-only regularized logistic-regression head used for the frozen patient representations.",
    },
]


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name, dtype={"patient_id": str})


def repository_commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def array_digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def full_head(seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        VarianceThreshold(),
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        ),
    )


def indices_for(patients: pd.DataFrame, split: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    per_seed = split.loc[split.seed.eq(seed)]
    if len(per_seed) != 98 or per_seed.patient_id.nunique() != 98:
        raise RuntimeError(f"Invalid assignment for seed {seed}")
    if set(per_seed.patient_id) != set(patients.patient_id):
        raise RuntimeError(f"Patient coverage drift for seed {seed}")
    index = {
        role: np.flatnonzero(patients.patient_id.isin(per_seed.loc[per_seed.role.eq(role), "patient_id"]))
        for role in ("train", "validation", "test")
    }
    if [len(index[role]) for role in ("train", "validation", "test")] != [58, 20, 20]:
        raise RuntimeError(f"Invalid role sizes for seed {seed}")
    if len(set(np.concatenate(list(index.values())))) != 98:
        raise RuntimeError(f"Overlapping roles for seed {seed}")
    return index


def normalize_canvas(arrays: np.ndarray, device):
    """Official ImageNet normalization after the project's masked grayscale canvas."""
    import torch

    arrays = np.asarray(arrays)
    if arrays.ndim != 4 or arrays.shape[1:] != (224, 224, 3) or arrays.dtype != np.uint8:
        raise ValueError("Expected N x 224 x 224 x 3 uint8 canvas images")
    x = torch.from_numpy(arrays.transpose(0, 3, 1, 2).copy()).to(device=device, dtype=torch.float32)
    x = x / 255.0
    mean = torch.as_tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def encoder_tokens_without_mask(model, x):
    """Exact encoder path for mask_ratio=0 with identity patch order.

    The original MAE method randomly permutes every patch even at zero masking.
    Keeping the identity order produces the same unmasked computation while
    making frozen feature extraction deterministic across repeated calls.
    """
    tokens = model.patch_embed(x)
    tokens = tokens + model.pos_embed[:, 1:, :]
    cls_token = model.cls_token + model.pos_embed[:, :1, :]
    tokens = __import__("torch").cat((cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)
    for block in model.blocks:
        tokens = block(tokens)
    return model.norm(tokens)


def forced_identity_forward_encoder(model, x):
    """Reference the official forward_encoder after replacing only its zero-mask shuffle."""
    import torch

    original = model.random_masking

    def identity_masking(tokens, mask_ratio):
        if mask_ratio != 0.0:
            raise ValueError("This check is only defined for mask_ratio=0")
        batch, length, _ = tokens.shape
        ids = torch.arange(length, device=tokens.device).unsqueeze(0).expand(batch, -1)
        return tokens, torch.zeros((batch, length), device=tokens.device), ids

    model.random_masking = identity_masking
    try:
        return model.forward_encoder(x, mask_ratio=0.0)[0]
    finally:
        model.random_masking = original


def load_usf_mae():
    """Load the public USF-MAE weights strictly through an isolated adapter."""
    import torch
    import timm

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    if not MAE_REPOSITORY.exists():
        raise FileNotFoundError(MAE_REPOSITORY)
    numpy_float_alias_installed = not hasattr(np, "float")
    if numpy_float_alias_installed:
        # The unmodified official MAE initialization calls the removed NumPy alias.
        np.float = float
    source = str(MAE_REPOSITORY.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    import models_mae
    from timm.models.vision_transformer import Block as CurrentBlock

    class LegacyMAEBlock(CurrentBlock):
        """Accept the official old qk_scale=None call without changing attention."""

        def __init__(self, *args, qk_scale=None, **kwargs):
            if qk_scale is not None:
                raise ValueError("Only the official qk_scale=None setting is permitted")
            super().__init__(*args, **kwargs)

    models_mae.Block = LegacyMAEBlock
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("Expected a raw state dictionary from the official checkpoint")
    model = models_mae.mae_vit_base_patch16_dec512d8b()
    loaded = model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen encoder contains trainable parameters")
    return model, {
        "checkpoint_keys": len(state),
        "strict_load": str(loaded),
        "architecture": "official mae_vit_base_patch16_dec512d8b",
        "model_module": type(model).__module__,
        "model_class": type(model).__name__,
        "timm": timm.__version__,
        "numpy_float_alias_installed": numpy_float_alias_installed,
        "qk_scale_adapter": "The official code passes qk_scale=None. The local wrapper accepts only None and delegates all current timm defaults.",
        "models_mae_sha256": R.sha256(MAE_REPOSITORY / "models_mae.py"),
        "pos_embed_sha256": R.sha256(MAE_REPOSITORY / "util" / "pos_embed.py"),
        "legacy_block_source": inspect.getsource(LegacyMAEBlock),
    }


def source_paths() -> list[tuple[Path, Path]]:
    return [
        (Path(__file__), Path("source") / Path(__file__).name),
        (USF_REPOSITORY / "README.md", Path("source") / "third_party" / "USF-MAE" / "README.md"),
        (USF_REPOSITORY / "LICENSE", Path("source") / "third_party" / "USF-MAE" / "LICENSE"),
        (MAE_REPOSITORY / "models_mae.py", Path("source") / "third_party" / "facebook-mae" / "models_mae.py"),
        (MAE_REPOSITORY / "util" / "pos_embed.py", Path("source") / "third_party" / "facebook-mae" / "util" / "pos_embed.py"),
        (MAE_REPOSITORY / "LICENSE", Path("source") / "third_party" / "facebook-mae" / "LICENSE"),
    ]


def require_lock() -> None:
    lock = json.loads((OUT / "protocol_locked.json").read_text(encoding="utf-8"))
    if lock["sha256"] != R.sha256(OUT / "protocol.json"):
        raise RuntimeError("Protocol changed after lock")
    if lock["source_sha256"] != R.sha256(Path(__file__)):
        raise RuntimeError("Probe source changed after lock; choose a new output directory")
    for source, relative in source_paths()[1:]:
        expected = lock["third_party_source_sha256"][str(relative).replace("\\", "/")]
        if expected != R.sha256(source):
            raise RuntimeError(f"Third-party source changed after lock: {source}")


def prepare() -> None:
    OUT.mkdir(exist_ok=True)
    for directory in ("features", "models", "source", "qa", "figures_600dpi"):
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.exists() or not USF_REPOSITORY.exists() or not MAE_REPOSITORY.exists():
        raise RuntimeError("USF-MAE checkpoint or official source repository is unavailable")
    for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv"):
        source, target = L.OUT / name, OUT / name
        if target.exists() and R.sha256(target) != R.sha256(source):
            raise RuntimeError(f"Cohort drift: {name}")
        shutil.copy2(source, target)
    df, patients, split = read("image_manifest.csv"), read("patients.csv"), read("random_patient_assignments.csv")
    if len(df) != 349 or len(patients) != 98 or int(patients.label.sum()) != 52:
        raise RuntimeError("Unexpected CYCH cohort dimensions")
    if df.image_id.nunique() != 349 or df.groupby("patient_id").label.nunique().gt(1).any():
        raise RuntimeError("Image/patient grain is invalid")
    if df.groupby("source_sha256").patient_id.nunique().max() != 1:
        raise RuntimeError("Exact source-image duplicate across patients")
    for seed in R.SEEDS:
        indices_for(patients, split, seed)
    reference_metrics = pd.read_csv(DENSE_REFERENCE / "metrics_by_role_per_draw.csv")
    reference_predictions = pd.read_csv(DENSE_REFERENCE / "patient_predictions_by_role.csv")
    reference_metrics = reference_metrics.loc[reference_metrics.group.eq(COMPARATOR)].copy()
    reference_predictions = reference_predictions.loc[reference_predictions.group.eq(COMPARATOR)].copy()
    if len(reference_metrics) != 20 or len(reference_predictions) != 400:
        raise RuntimeError("DN-B comparator output is incomplete")
    for source, relative in source_paths():
        if not source.exists():
            raise FileNotFoundError(source)
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    protocol = {
        "design": "Further retrospective exploration on the same previously explored 98-patient/349-image CYCH cohort. Ten fixed patient-level 58/20/20 random assignments. No external or untouched-cohort claim.",
        "roles": {"train": 58, "development_validation": 20, "exploratory_test": 20},
        "seeds": R.SEEDS,
        "cohort": {"patients": 98, "positive": 52, "negative": 46, "images": 349},
        "assignment_sha256": {name: R.sha256(OUT / name) for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv")},
        "input": "Existing P1-P99 grayscale masked-kidney image, transformed by the established aspect-preserving R.canvas_image 224x224 canvas, replicated to RGB. Official USF-MAE ImageNet mean/std normalization follows. The use of a masked canvas rather than the official full-image resize is a declared project-specific adaptation, not an exact USF-MAE reproduction.",
        "encoder": "Official USF-MAE_full_pretrain_43dataset_100epochs.pt, loaded strictly as mae_vit_base_patch16_dec512d8b. The public README lists kidneyUS in the pretraining collection, so no independent renal-external-validation or leakage-free-pretraining claim is made.",
        "representation": "One pre-specified frozen representation: unmasked encoder CLS token (768 values) per image, then existing patient-level image mean and standard deviation (1536 values). No decoder feature, patch feature, pooling alternative, augmentation, fine-tuning, GAN or learned aggregation is screened.",
        "compatibility": "Official MAE source calls qk_scale=None and np.float. The local adapter accepts only qk_scale=None under current timm, and restores np.float=float only inside the process for source initialization. Official code and checkpoint files are never edited.",
        "head": "Fixed train-only median imputer, variance removal, standardization and balanced liblinear L2 logistic regression (C=0.1, max_iter=2000), matching the existing DN-B full-head convention.",
        "comparator": "Existing frozen RadImageNet DenseNet-121 DN-B output using the identical cohort assignments. Its output files are hash-recorded and copied only as a fixed reference; DN-B is not re-fit in this batch.",
        "development_gate": "USF-MAE-CLS is eligible only for a future separately locked equal-weight vote comparison if its mean development AUROC and AP are each no worse than DN-B by 0.02. This batch does not fit any vote. The exploratory test does not affect the gate.",
        "leakage_controls": "Only the outer 58 training patients fit each LR pipeline. Model weights remain frozen; state digests, strict load, deterministic no-mask checks, input hashes and per-patient predictions are retained.",
        "limitations": "All 98 patients have already been explored. Ten overlapping draws are not independent cohorts or k-fold validation. Existing masks include unresolved acquisition annotations; no lesion, parenchyma, cortex or medulla claim is made.",
        "exclusions": "No clinical covariates, target labels in encoder fitting, segmentation retraining, artifact removal/inpainting, image synthesis, test-time augmentation, hyperparameter search or model-vote fitting.",
    }
    protocol = json.loads(json.dumps(protocol))
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError("Protocol changed; use a new output directory")
    R.write_json(path, protocol)
    third_party = {
        "usf_mae": {
            "repository": "https://github.com/Yusufii9/USF-MAE",
            "local_commit": repository_commit(USF_REPOSITORY),
            "license": "MIT (repository LICENSE copied under source/third_party/USF-MAE/)",
        },
        "facebook_mae": {
            "repository": "https://github.com/facebookresearch/mae",
            "local_commit": repository_commit(MAE_REPOSITORY),
            "license": "CC BY-NC 4.0 (repository LICENSE copied under source/third_party/facebook-mae/)",
        },
        "checkpoint": {"path": str(CHECKPOINT), "bytes": CHECKPOINT.stat().st_size, "sha256": R.sha256(CHECKPOINT)},
        "copied_sources": [
            {"path": str(relative).replace("\\", "/"), "sha256": R.sha256(source)}
            for source, relative in source_paths()[1:]
        ],
        "fixed_comparator": {
            "metrics_path": str(DENSE_REFERENCE / "metrics_by_role_per_draw.csv"),
            "metrics_sha256": R.sha256(DENSE_REFERENCE / "metrics_by_role_per_draw.csv"),
            "predictions_path": str(DENSE_REFERENCE / "patient_predictions_by_role.csv"),
            "predictions_sha256": R.sha256(DENSE_REFERENCE / "patient_predictions_by_role.csv"),
        },
    }
    R.write_json(OUT / "third_party_provenance.json", third_party)
    R.write_json(OUT / "references.json", REFERENCES)
    quality = {
        "image_rows": len(df),
        "patients": len(patients),
        "positive_patients": int(patients.label.sum()),
        "negative_patients": int((1 - patients.label).sum()),
        "views": df.view.value_counts().sort_index().to_dict(),
        "mask_sources": df.mask_source.value_counts().sort_index().to_dict(),
        "exact_duplicate_source_hashes": int(df.source_sha256.duplicated().sum()),
        "labels_constant_within_patient": bool(not df.groupby("patient_id").label.nunique().gt(1).any()),
    }
    R.write_json(OUT / "data_quality_summary.json", quality)
    R.write_json(OUT / "original_manuscript_preservation.json", {
        "path": str(R.ROOT / "yolo_aug_triEMR_PCA_SVM_renal_2022_v2.docx"),
        "sha256": R.sha256(R.ROOT / "yolo_aug_triEMR_PCA_SVM_renal_2022_v2.docx"),
    })
    if not (OUT / "protocol_locked.json").exists():
        R.write_json(
            OUT / "protocol_locked.json",
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "sha256": R.sha256(path),
                "source_sha256": R.sha256(Path(__file__)),
                "third_party_source_sha256": {
                    str(relative).replace("\\", "/"): R.sha256(source) for source, relative in source_paths()[1:]
                },
            },
        )
    print("USF-MAE frozen probe protocol locked; cohort, comparator, sources and checkpoint recorded", flush=True)


def encode() -> None:
    require_lock()
    import torch
    import timm

    torch.set_num_threads(4)
    torch.manual_seed(20260913)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df, patients = read("image_manifest.csv"), read("patients.csv")
    arrays, audit = [], []
    for row in df.itertuples():
        image_path = L.OUT / "images" / "percentile" / f"{row.image_id}.png"
        mask_path = L.OUT / "masks" / f"{row.image_id}.png"
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None or gray.shape != mask.shape:
            raise RuntimeError(f"Invalid locked input: {row.image_id}")
        if R.sha256(mask_path) != row.mask_sha256:
            raise RuntimeError(f"Mask hash mismatch: {row.image_id}")
        mask = (mask > 0).astype(np.uint8)
        if np.any(gray[mask == 0]):
            raise RuntimeError(f"Nonzero background after ROI masking: {row.image_id}")
        canvas = R.canvas_image(gray, mask, angle=0, channels="gray", size=224)
        if canvas.shape != (224, 224, 3) or canvas.dtype != np.uint8:
            raise RuntimeError(f"Invalid canvas: {row.image_id}")
        if not (np.array_equal(canvas[..., 0], canvas[..., 1]) and np.array_equal(canvas[..., 1], canvas[..., 2])):
            raise RuntimeError(f"Canvas channels drifted: {row.image_id}")
        arrays.append(canvas)
        audit.append(
            {
                "image_id": row.image_id,
                "patient_alias": row.patient_alias,
                "view": row.view,
                "source_image_sha256": R.sha256(image_path),
                "mask_sha256": R.sha256(mask_path),
                "source_height": int(gray.shape[0]),
                "source_width": int(gray.shape[1]),
                "mask_pixels": int(mask.sum()),
                "canvas_sha256": array_digest(canvas),
                "canvas_nonzero_pixels": int(np.count_nonzero(canvas[..., 0])),
            }
        )
    arrays = np.stack(arrays)
    if arrays.shape != (349, 224, 224, 3):
        raise RuntimeError("Unexpected canvas array shape")
    pd.DataFrame(audit).to_csv(OUT / "input_audit.csv", index=False)
    model, compatibility = load_usf_mae()
    model = model.to(device).eval().requires_grad_(False)
    state_before = L.model_digest(model)
    with torch.inference_mode():
        x = normalize_canvas(arrays[:2], device)
        identity = encoder_tokens_without_mask(model, x)
        reference = forced_identity_forward_encoder(model, x)
        repeated = encoder_tokens_without_mask(model, x)
        single = encoder_tokens_without_mask(model, x[:1])
        checks = {
            "identity_reference_maxabs": float((identity - reference).abs().max().item()),
            "repeat_maxabs": float((identity - repeated).abs().max().item()),
            "batch_single_maxabs": float((identity[:1] - single).abs().max().item()),
        }
        if max(checks.values()) > 1e-5 or identity.shape != (2, 197, 768) or not bool(torch.isfinite(identity).all()):
            raise RuntimeError(f"Unmasked encoder preflight failed: {checks}")
        chunks = []
        for start in range(0, len(arrays), BATCH_SIZE):
            x = normalize_canvas(arrays[start:start + BATCH_SIZE], device)
            tokens = encoder_tokens_without_mask(model, x)
            if tokens.shape[1:] != (197, 768) or not bool(torch.isfinite(tokens).all()):
                raise RuntimeError("Invalid frozen encoder output")
            chunks.append(tokens[:, 0].cpu().numpy())
            if start % 64 == 0:
                print(f"USF-MAE frozen CLS: {min(start + BATCH_SIZE, len(arrays))}/{len(arrays)}", flush=True)
    image_features = np.concatenate(chunks).astype(np.float32)
    patient_features = R.aggregate(image_features, df, patients)
    if image_features.shape != (349, 768) or patient_features.shape != (98, 1536):
        raise RuntimeError("Unexpected feature dimensions")
    if not np.isfinite(image_features).all() or not np.isfinite(patient_features).all():
        raise RuntimeError("Non-finite frozen features")
    np.save(OUT / "features" / "usfmae_canvas_cls_image.npy", image_features)
    np.save(OUT / "features" / "usfmae_canvas_cls.npy", patient_features)
    state_after = L.model_digest(model)
    if state_before != state_after or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen checkpoint changed during inference")
    provenance = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch": torch.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": R.sha256(CHECKPOINT),
        "state_before": state_before,
        "state_after": state_after,
        "all_parameters_frozen": True,
        "normalization": {"mean": IMAGENET_MEAN.tolist(), "std": IMAGENET_STD.tolist()},
        "input_array_sha256": array_digest(arrays),
        "image_feature_shape": list(image_features.shape),
        "patient_feature_shape": list(patient_features.shape),
        "inference_checks": checks,
        **compatibility,
    }
    R.write_json(OUT / "encoder_provenance.json", provenance)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("USF-MAE frozen CLS cache, state audit and input audit complete", flush=True)


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (role, group), frame in metrics.groupby(["role", "group"], sort=False):
        row = {
            "role": role,
            "group": group,
            "draws": len(frame),
            "joint_target_draws": int(((frame.auroc > 0.8) & (frame.average_precision > 0.8)).sum()),
        }
        for metric in ("auroc", "average_precision", "auprc_trapezoid", "sensitivity", "specificity", "brier", "log_loss"):
            for aggregation in ("mean", "std", "min", "max"):
                row[f"{metric}_{aggregation}"] = float(getattr(frame[metric], aggregation)())
        rows.append(row)
    return pd.DataFrame(rows)


def verify_metrics(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    for row in metrics.itertuples():
        records = predictions.loc[
            predictions.seed.eq(row.seed)
            & predictions.group.eq(row.group)
            & predictions.role.eq(row.role)
        ]
        if len(records) != 20:
            raise RuntimeError("Metric/prediction grain mismatch")
        auroc = roc_auc_score(records.label, records.score)
        ap = average_precision_score(records.label, records.score)
        if not np.isclose(auroc, row.auroc, atol=1e-12) or not np.isclose(ap, row.average_precision, atol=1e-12):
            raise RuntimeError("Independent AUROC/AP recomputation failed")


def evaluate() -> None:
    require_lock()
    patients, split = read("patients.csv"), read("random_patient_assignments.csv")
    features = np.load(OUT / "features" / "usfmae_canvas_cls.npy")
    if features.shape != (98, 1536) or not np.isfinite(features).all():
        raise RuntimeError("Frozen patient feature cache is unavailable or invalid")
    labels = patients.label.to_numpy(dtype=int)
    metrics, predictions, heads = [], [], []
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            index = indices_for(patients, split, seed)
            head = full_head(seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                head.fit(features[index["train"]], labels[index["train"]])
            scores = head.predict_proba(features)[:, list(head.classes_).index(1)]
            if not np.isfinite(scores).all():
                raise RuntimeError("Non-finite logistic-regression score")
            model_path = OUT / "models" / f"{GROUP}_{seed}.joblib"
            joblib.dump(head, model_path, compress=3)
            reloaded = joblib.load(model_path).predict_proba(features)[:, 1]
            if not np.allclose(scores, reloaded, atol=1e-12):
                raise RuntimeError("Saved LR head reload mismatch")
            heads.append(
                {
                    "seed": seed,
                    "group": GROUP,
                    "sha256": R.sha256(model_path),
                    "dimension": int(features.shape[1]),
                    "train_patient_aliases": patients.patient_alias.iloc[index["train"]].tolist(),
                }
            )
            for role, report_role in (("validation", "development_validation"), ("test", "exploratory_test")):
                values = index[role]
                metrics.append(
                    {"seed": seed, "group": GROUP, "role": report_role, **V.score_metrics(labels[values], scores[values])}
                )
                predictions.extend(
                    {
                        "seed": seed,
                        "group": GROUP,
                        "role": report_role,
                        "patient_alias": patients.patient_alias.iloc[value],
                        "label": int(labels[value]),
                        "score": float(scores[value]),
                    }
                    for value in values
                )
            print(f"Fixed USF-MAE LR head completed: {seed}", flush=True)
    current_metrics = pd.DataFrame(metrics)
    current_predictions = pd.DataFrame(predictions)
    reference_metrics = pd.read_csv(DENSE_REFERENCE / "metrics_by_role_per_draw.csv")
    reference_predictions = pd.read_csv(DENSE_REFERENCE / "patient_predictions_by_role.csv")
    reference_metrics = reference_metrics.loc[reference_metrics.group.eq(COMPARATOR)].copy()
    reference_predictions = reference_predictions.loc[reference_predictions.group.eq(COMPARATOR)].copy()
    if len(reference_metrics) != 20 or len(reference_predictions) != 400:
        raise RuntimeError("Fixed DN-B reference changed or is incomplete")
    for seed in R.SEEDS:
        index = indices_for(patients, split, seed)
        for role, report_role in (("validation", "development_validation"), ("test", "exploratory_test")):
            expected = set(patients.patient_alias.iloc[index[role]])
            observed = set(reference_predictions.loc[
                reference_predictions.seed.eq(seed) & reference_predictions.role.eq(report_role), "patient_alias"
            ])
            if expected != observed:
                raise RuntimeError(f"DN-B reference assignment drift at seed {seed} / {role}")
    all_metrics = pd.concat([reference_metrics, current_metrics], ignore_index=True)
    all_predictions = pd.concat([reference_predictions, current_predictions], ignore_index=True)
    verify_metrics(all_metrics, all_predictions)
    all_metrics.to_csv(OUT / "metrics_by_role_per_draw.csv", index=False)
    all_predictions.to_csv(OUT / "patient_predictions_by_role.csv", index=False)
    pd.DataFrame(heads).to_json(OUT / "lr_fit_audit.json", orient="records", indent=2, force_ascii=False)
    summary = summary_table(all_metrics)
    summary.to_csv(OUT / "summary_by_role.csv", index=False)
    development = all_metrics.loc[all_metrics.role.eq("development_validation")].pivot(index="seed", columns="group", values=["auroc", "average_precision"])
    delta_auroc = development["auroc", GROUP] - development["auroc", COMPARATOR]
    delta_ap = development["average_precision", GROUP] - development["average_precision", COMPARATOR]
    decision = {
        "selection_role": "development_validation only",
        "candidate": GROUP,
        "fixed_comparator": COMPARATOR,
        "mean_delta_auroc": float(delta_auroc.mean()),
        "mean_delta_average_precision": float(delta_ap.mean()),
        "joint_improvement_draws": int(((delta_auroc > 0) & (delta_ap > 0)).sum()),
        "minimum_comparator_retention": -0.02,
        "eligible_for_future_separately_locked_vote_comparison": bool(delta_auroc.mean() >= -0.02 - 1e-12 and delta_ap.mean() >= -0.02 - 1e-12),
        "no_vote_fitted_in_this_batch": True,
        "exploratory_test_not_used_for_decision": True,
    }
    R.write_json(OUT / "development_decision.json", decision)
    print(summary[["role", "group", "auroc_mean", "average_precision_mean"]].to_string(index=False), flush=True)
    print(json.dumps(decision, ensure_ascii=False), flush=True)


def save_figure(fig, stem: str) -> dict:
    png = OUT / "figures_600dpi" / f"{stem}.png"
    tiff = OUT / "figures_600dpi" / f"{stem}.tiff"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(tiff, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    image = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("Unable to validate rendered figure")
    return {
        "stem": stem,
        "png": png.name,
        "tiff": tiff.name,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "dpi": 600,
        "png_sha256": R.sha256(png),
        "tiff_sha256": R.sha256(tiff),
    }


def render_figure() -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = pd.read_csv(OUT / "metrics_by_role_per_draw.csv")
    summary = pd.read_csv(OUT / "summary_by_role.csv")
    dev = metrics.loc[metrics.role.eq("development_validation")].pivot(index="seed", columns="group", values=["auroc", "average_precision"])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    groups = [COMPARATOR, GROUP]
    labels = ["Frozen\nRadImageNet DenseNet (DN-B)", "Frozen\nUSF-MAE CLS"]
    positions = np.arange(2)
    for metric, offset, marker in (("auroc_mean", -0.16, "o"), ("average_precision_mean", 0.16, "s")):
        development = summary.loc[summary.role.eq("development_validation")].set_index("group").reindex(groups)[metric]
        exploratory = summary.loc[summary.role.eq("exploratory_test")].set_index("group").reindex(groups)[metric]
        axes[0].scatter(positions + offset, development, marker=marker, s=50, color="#176b87", zorder=3)
        axes[0].scatter(positions + offset, exploratory, marker=marker, s=50, facecolors="none", edgecolors="#176b87", zorder=3)
        for position, a, b in zip(positions + offset, development, exploratory):
            axes[0].plot([position, position], [a, b], color="#a8b7bd", linewidth=0.8, zorder=1)
    axes[0].axhline(0.8, color="#b45309", linewidth=0.9, linestyle="--")
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(0.35, 1.02)
    axes[0].set_ylabel("Mean metric across 10 fixed patient draws")
    axes[0].set_title("Frozen representation screen")
    axes[0].scatter([], [], marker="o", color="#176b87", label="Development AUROC")
    axes[0].scatter([], [], marker="o", facecolors="none", edgecolors="#176b87", label="Exploratory test AUROC")
    axes[0].scatter([], [], marker="s", color="#176b87", label="Development AP")
    axes[0].scatter([], [], marker="s", facecolors="none", edgecolors="#176b87", label="Exploratory test AP")
    axes[0].legend(loc="lower left", fontsize=7)
    for metric, color, marker, label in (("auroc", "#1d4e89", "o", "AUROC"), ("average_precision", "#b45309", "s", "AP")):
        delta = dev[metric, GROUP] - dev[metric, COMPARATOR]
        axes[1].scatter(np.arange(len(delta)), delta, color=color, marker=marker, s=38, label=label)
    axes[1].axhline(0, color="#374151", linewidth=0.8)
    axes[1].axhline(-0.02, color="#b45309", linewidth=0.8, linestyle="--", label="Future-screen floor")
    axes[1].set_xticks(np.arange(len(R.SEEDS)), [str(seed)[-2:] for seed in R.SEEDS])
    axes[1].set_xlabel("Fixed draw suffix")
    axes[1].set_ylabel("USF-MAE minus DN-B development metric")
    axes[1].set_title("Development-only paired difference")
    axes[1].legend(fontsize=7, loc="lower left")
    fig.suptitle("USF-MAE frozen CLS probe; no voting model fitted", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    manifest = save_figure(fig, "Figure_UM1_usfmae_frozen_probe")
    plt.close(fig)
    return manifest


def report() -> None:
    require_lock()
    summary = pd.read_csv(OUT / "summary_by_role.csv")
    decision = json.loads((OUT / "development_decision.json").read_text(encoding="utf-8"))
    provenance = json.loads((OUT / "encoder_provenance.json").read_text(encoding="utf-8"))
    figure = render_figure()
    R.write_json(OUT / "figure_manifest.json", [figure])
    values = summary.pivot(index="group", columns="role", values=["auroc_mean", "average_precision_mean"])
    eligibility = "PASSED" if decision["eligible_for_future_separately_locked_vote_comparison"] else "NOT PASSED"
    lines = [
        "# CYCH USF-MAE Frozen Feature Probe",
        "",
        "Completed: 2026-09-13. This is a single frozen-representation screen on the same already explored CYCH cohort. It does not modify the manuscript, existing votes, masks, checkpoint files or prior experiment packages.",
        "",
        "## Design",
        "",
        "- Same 98 patients (52 positive / 46 negative), 349 first-exam images and ten fixed 58/20/20 patient assignments.",
        "- One fixed USF-MAE representation only: official 100-epoch ViT-B/16 MAE encoder, no-mask CLS token (768 values) per image, then existing patient image mean/std aggregation (1536 values).",
        "- Existing P1-P99 masked kidney canvas is retained and replicated to RGB; official ImageNet mean/std is applied. This masked-canvas geometry is a declared project-specific adaptation, not an exact reproduction of the repository full-image resize example.",
        "- The encoder is strictly loaded and frozen. A local compatibility adapter accepts only the official old `qk_scale=None` call under current `timm`; it does not alter weights, layers or attention settings. The official MAE source requires a CC BY-NC 4.0 license, so this is documented as a non-commercial research probe, not a deployment permission.",
        "- Each fixed LR pipeline is fitted on only the 58 training patients. Development is compared with the existing fixed frozen RadImageNet DenseNet DN-B reference. No vote, fine-tuning, augmentation, GAN, patch variant or test-selected configuration is fitted in this batch.",
        "",
        "## Results",
        "",
        "| Representation | Development AUROC / AP | Exploratory test AUROC / AP |",
        "|---|---:|---:|",
    ]
    for group, label in ((COMPARATOR, "Frozen RadImageNet DenseNet (DN-B reference)"), (GROUP, "Frozen USF-MAE CLS")):
        lines.append(
            f"| {label} | {values.loc[group, ('auroc_mean', 'development_validation')]:.4f} / {values.loc[group, ('average_precision_mean', 'development_validation')]:.4f} | {values.loc[group, ('auroc_mean', 'exploratory_test')]:.4f} / {values.loc[group, ('average_precision_mean', 'exploratory_test')]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Development-Only Gate",
            "",
            f"- USF-MAE CLS minus DN-B: mean development AUROC {decision['mean_delta_auroc']:+.4f}; AP {decision['mean_delta_average_precision']:+.4f}; joint strict improvements {decision['joint_improvement_draws']}/10.",
            f"- Future separately locked vote-comparison eligibility (both development mean deltas at least -0.0200): {eligibility}. No voting model was fitted here, and exploratory test values did not influence this decision.",
            "",
            "## Reproducibility and Limits",
            "",
            f"- Checkpoint: `{CHECKPOINT.name}`; SHA-256 `{provenance['checkpoint_sha256']}`; strict load `{provenance['strict_load']}`; output shapes {provenance['image_feature_shape']} image and {provenance['patient_feature_shape']} patient.",
            f"- Inference checks: identity/reference max abs {provenance['inference_checks']['identity_reference_maxabs']:.3g}; repeat max abs {provenance['inference_checks']['repeat_maxabs']:.3g}; batch/single max abs {provenance['inference_checks']['batch_single_maxabs']:.3g}. Frozen checkpoint digest was unchanged.",
            "- The USF-MAE README lists kidneyUS among pretraining data. Pretraining overlap with this local CYCH cohort is not demonstrated, but this is not an independent renal external-validation claim.",
            "- All 98 patients were already explored, the ten draws overlap, existing image annotations remain in some ROIs, and no output identifies a lesion, inflammation scar, cortex, medulla or other compartment.",
            "",
            "## Files",
            "",
            "- `encoder_provenance.json`, `third_party_provenance.json`, `input_audit.csv`, `data_quality_summary.json`: checkpoint, source, input and cohort audits.",
            "- `features/usfmae_canvas_cls_image.npy` and `features/usfmae_canvas_cls.npy`: frozen image- and patient-level feature caches.",
            "- `metrics_by_role_per_draw.csv`, `summary_by_role.csv`, `patient_predictions_by_role.csv`, `development_decision.json`: all results and the development-only gate.",
            "- `figures_600dpi/`: verified 600 dpi PNG/TIFF figure; dimensions and SHA-256 are in `figure_manifest.json`.",
            "",
            "## References",
            "",
        ]
    )
    lines.extend(f"- {entry['citation']} [{entry['url']}]({entry['url']}) - {entry['scope']}" for entry in REFERENCES)
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("USF-MAE README and 600 dpi figure rendered", flush=True)


def package() -> None:
    require_lock()
    archive = R.ROOT / f"{OUT.name}.zip"
    files = [path for path in sorted(OUT.rglob("*")) if path.is_file()]
    pd.DataFrame(
        [
            {"path": str(path.relative_to(OUT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": R.sha256(path)}
            for path in files
        ]
    ).to_csv(OUT / "archive_manifest.csv", index=False)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package_file:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                package_file.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(archive) as package_file:
        if package_file.testzip() is not None:
            raise RuntimeError("ZIP CRC verification failed")
        entries = len(package_file.infolist())
    R.write_json(
        OUT / "archive_verification.json",
        {"archive": str(archive), "bytes": archive.stat().st_size, "sha256": R.sha256(archive), "entries": entries, "crc_verified": True},
    )
    print(f"ZIP verified: {archive.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "encode", "evaluate", "report", "package", "all"))
    args = parser.parse_args()
    stages = ("prepare", "encode", "evaluate", "report", "package") if args.stage == "all" else (args.stage,)
    for stage in stages:
        globals()[stage]()
