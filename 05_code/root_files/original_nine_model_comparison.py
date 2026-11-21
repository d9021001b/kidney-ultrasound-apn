"""Compare the nine CNN families listed in the 2022 CYCH draft.

The comparison is deliberately patient-disjoint and image-only. It reuses the
locked 98-patient, 349-image whole-kidney ROI cohort and the ten established
58/20/20 random assignments. The original draft's clinical fields are not used
because their provenance is unresolved.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    auc,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

import renal_random_validation as R
import renal_voting_validation as V


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "CYCH_random_validation_20260913"
OUT = ROOT / "CYCH_original_nine_model_comparison_20260914"
WEIGHTS = ROOT / "external_models" / "original_paper_nine_cnn_20260914" / "weights"
ORIGINAL_DOCX = ROOT / "yolo_aug_triEMR_PCA_SVM_renal_2022_0307a-chou.docx"
TRIPLET_EPOCHS = 200
TRIPLET_MARGIN = 0.2
EMBEDDING_DIMENSION = 128


MODEL_SPECS = OrderedDict(
    [
        (
            "vgg16",
            dict(
                paper_name="VGG16",
                framework="torchvision",
                implementation="torchvision.models.vgg16",
                weight_file="vgg16-397923af.pth",
                weight_source="https://download.pytorch.org/models/vgg16-397923af.pth",
                weight_license="TorchVision BSD-3-Clause; ImageNet terms also apply",
                architecture_reference="https://arxiv.org/abs/1409.1556",
                input_size=224,
                normalization="ImageNet mean/std",
                feature_dimension=4096,
                retrieval="copied from verified local TorchVision cache",
            ),
        ),
        (
            "vgg19",
            dict(
                paper_name="VGG19",
                framework="torchvision",
                implementation="torchvision.models.vgg19",
                weight_file="vgg19-dcbb9e9d.pth",
                weight_source="https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
                weight_license="TorchVision BSD-3-Clause; ImageNet terms also apply",
                architecture_reference="https://arxiv.org/abs/1409.1556",
                input_size=224,
                normalization="ImageNet mean/std",
                feature_dimension=4096,
                retrieval="copied from verified local TorchVision cache",
            ),
        ),
        (
            "resnet50",
            dict(
                paper_name="ResNet50",
                framework="torchvision",
                implementation="torchvision.models.resnet50",
                weight_file="resnet50-0676ba61.pth",
                weight_source="https://download.pytorch.org/models/resnet50-0676ba61.pth",
                weight_license="TorchVision BSD-3-Clause; ImageNet terms also apply",
                architecture_reference="https://arxiv.org/abs/1512.03385",
                input_size=224,
                normalization="ImageNet mean/std",
                feature_dimension=2048,
                retrieval="copied from verified local TorchVision cache",
            ),
        ),
        (
            "resnet101",
            dict(
                paper_name="ResNet101",
                framework="torchvision",
                implementation="torchvision.models.resnet101",
                weight_file="resnet101-63fe2227.pth",
                weight_source="https://download.pytorch.org/models/resnet101-63fe2227.pth",
                weight_license="TorchVision BSD-3-Clause; ImageNet terms also apply",
                architecture_reference="https://arxiv.org/abs/1512.03385",
                input_size=224,
                normalization="ImageNet mean/std",
                feature_dimension=2048,
                retrieval="copied from verified local TorchVision cache",
            ),
        ),
        (
            "densenet121",
            dict(
                paper_name="DenseNet121",
                framework="torchvision",
                implementation="torchvision.models.densenet121",
                weight_file="densenet121-a639ec97.pth",
                weight_source="https://download.pytorch.org/models/densenet121-a639ec97.pth",
                weight_license="TorchVision BSD-3-Clause; ImageNet terms also apply",
                architecture_reference="https://arxiv.org/abs/1608.06993",
                input_size=224,
                normalization="ImageNet mean/std",
                feature_dimension=1024,
                retrieval="copied from verified local TorchVision cache",
            ),
        ),
        (
            "mobilenet",
            dict(
                paper_name="MobileNet",
                framework="timm",
                implementation="timm mobilenetv1_100.ra4_e3600_r224_in1k",
                weight_file="mobilenetv1_100.ra4_e3600_r224_in1k.safetensors",
                weight_source="https://huggingface.co/timm/mobilenetv1_100.ra4_e3600_r224_in1k/resolve/main/model.safetensors",
                weight_license="Apache-2.0 model card; ImageNet terms also apply",
                architecture_reference="https://arxiv.org/abs/1704.04861",
                input_size=224,
                normalization="x / 127.5 - 1",
                feature_dimension=1024,
                retrieval="downloaded from official timm Hugging Face model card",
            ),
        ),
        (
            "xception",
            dict(
                paper_name="Xception",
                framework="keras",
                implementation="tf.keras.applications.Xception(include_top=False, pooling=avg)",
                weight_file="xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
                weight_source="https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
                weight_license="Keras Apache-2.0; ImageNet terms also apply",
                architecture_reference="https://arxiv.org/abs/1610.02357",
                input_size=299,
                normalization="x / 127.5 - 1",
                feature_dimension=2048,
                retrieval="downloaded from official TensorFlow/Keras storage",
            ),
        ),
        (
            "inception_v3",
            dict(
                paper_name="InceptionV3",
                framework="torchvision",
                implementation="torchvision.models.inception_v3",
                weight_file="inception_v3_google-0cc3c7bd.pth",
                weight_source="https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth",
                weight_license="TorchVision BSD-3-Clause; ImageNet terms also apply",
                architecture_reference="https://arxiv.org/abs/1512.00567",
                input_size=299,
                normalization="ImageNet mean/std plus TorchVision transform_input",
                feature_dimension=2048,
                retrieval="copied from verified local TorchVision cache",
            ),
        ),
        (
            "inception_resnet_v2",
            dict(
                paper_name="InceptionResNetV2",
                framework="timm",
                implementation="timm inception_resnet_v2.tf_in1k",
                weight_file="inception_resnet_v2.tf_in1k.safetensors",
                weight_source="https://huggingface.co/timm/inception_resnet_v2.tf_in1k/resolve/main/model.safetensors",
                weight_license="Apache-2.0 model card; ImageNet terms also apply",
                architecture_reference="https://arxiv.org/abs/1602.07261",
                input_size=299,
                normalization="x / 127.5 - 1",
                feature_dimension=1536,
                retrieval="downloaded from official timm Hugging Face model card",
            ),
        ),
    ]
)


PAPER_TABLE = [
    ("VGG16", 171, 50, 146, 42, 8, 25, 0.854, 0.84, 0.16, 0.80),
    ("VGG19", 171, 50, 113, 40, 10, 58, 0.661, 0.80, 0.20, 0.63),
    ("ResNet50", 171, 50, 98, 23, 27, 73, 0.573, 0.46, 0.54, 0.45),
    ("ResNet101", 171, 50, 124, 41, 9, 47, 0.725, 0.82, 0.18, 0.62),
    ("DenseNet121", 171, 50, 96, 9, 41, 75, 0.561, 0.18, 0.82, 0.33),
    ("MobileNet", 171, 50, 124, 24, 26, 47, 0.725, 0.48, 0.52, 0.42),
    ("Xception", 171, 50, 161, 30, 20, 10, 0.942, 0.60, 0.40, 0.74),
    ("InceptionV3", 171, 50, 152, 27, 23, 19, 0.889, 0.54, 0.46, 0.68),
    ("InceptionResNetV2", 171, 50, 153, 17, 33, 18, 0.895, 0.34, 0.66, 0.85),
]


WORKFLOWS = OrderedDict(
    frozen_lr="Frozen encoder + current fixed LR head",
    frozen_pca2_svm="Frozen encoder + PCA(2) + RBF SVM",
    triplet128_lr="Training-only patient triplet(128) + current fixed LR head",
    triplet128_pca2_svm="Training-only patient triplet(128) + PCA(2) + RBF SVM",
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name, dtype={"patient_id": str})


def weight_path(model_key: str) -> Path:
    return WEIGHTS / MODEL_SPECS[model_key]["weight_file"]


def cohort_indices(patients: pd.DataFrame, assignments: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    per_seed = assignments.loc[assignments.seed.eq(seed)]
    result = {
        role: np.flatnonzero(patients.patient_id.isin(per_seed.loc[per_seed.role.eq(role), "patient_id"]))
        for role in ("train", "validation", "test")
    }
    assert {key: len(value) for key, value in result.items()} == {"train": 58, "validation": 20, "test": 20}
    assert not (set(result["train"]) & set(result["validation"]))
    assert not (set(result["train"]) & set(result["test"]))
    assert not (set(result["validation"]) & set(result["test"]))
    return result


def prepare() -> None:
    for directory in ("features/image", "features/patient", "models", "figures_600dpi", "qa", "source"):
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv"):
        shutil.copy2(SOURCE / name, OUT / name)

    manifest = read("image_manifest.csv")
    patients = read("patients.csv")
    assignments = read("random_patient_assignments.csv")
    assert len(manifest) == 349 and manifest.image_id.nunique() == 349
    assert len(patients) == 98 and patients.patient_id.nunique() == 98
    assert int(patients.label.sum()) == 52
    assert len(assignments) == 980
    for seed in R.SEEDS:
        cohort_indices(patients, assignments, seed)

    registry = []
    for key, spec in MODEL_SPECS.items():
        path = weight_path(key)
        if not path.exists():
            raise FileNotFoundError(f"Missing required weight: {path}")
        registry.append(
            {
                "model_key": key,
                **spec,
                "weight_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": R.sha256(path),
                "load_validation": "pending extraction",
                "comparison_status": "eligible after strict load and finite-feature validation",
            }
        )
    pd.DataFrame(registry).to_csv(OUT / "model_registry.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        PAPER_TABLE,
        columns=["CNN_model", "P", "N", "TP", "TN", "FP", "FN", "TPR", "TNR", "FPR", "AUC"],
    ).to_csv(OUT / "original_paper_table2.csv", index=False, encoding="utf-8-sig")
    local_model_audit()

    protocol = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Patient-safe comparison of the exact nine CNN families named in Table 2 of the 2022 CYCH draft",
        "cohort": {"patients": 98, "images": 349, "positive_patients": 52, "negative_patients": 46},
        "roles": {"train": 58, "development_validation": 20, "exploratory_test": 20},
        "seeds": R.SEEDS,
        "segmentation": "Existing whole-kidney masks are mandatory and hash-checked before every feature extraction.",
        "input": "Masked grayscale whole-kidney ROI on a centered square canvas, replicated to RGB; no clinical variables, augmentation, GAN, pseudo-lesion labels, or test-time augmentation.",
        "models": list(MODEL_SPECS),
        "pretraining": "Frozen ImageNet-1K weights; no outcome-based encoder fine-tuning.",
        "patient_representation": "Per-feature mean and standard deviation across each patient's first-exam kidney images.",
        "workflows": WORKFLOWS,
        "triplet": {
            "scope": "A training-only linear projection of patient representations; validation and test rows never enter scaling, mining, optimization, PCA, or classifier fitting.",
            "embedding_dimension": EMBEDDING_DIMENSION,
            "epochs": TRIPLET_EPOCHS,
            "margin": TRIPLET_MARGIN,
            "distance": "Euclidean on L2-normalized embeddings",
            "mining": "One different-patient positive and one semi-hard opposite-class negative per training patient, with deterministic random fallback.",
        },
        "selection": "No model is selected or tuned from the exploratory test role. Development ranking is descriptive and predeclared across all nine models.",
        "reporting_limit": "All 98 patients were previously explored, so neither role is an independent confirmation cohort and no k-fold claim is made.",
        "excluded_clinical_data": "Age, gender, renal axes, and other EMR fields are omitted because the user directed unresolved chart contradictions to be set aside.",
        "inputs_sha256": {
            name: R.sha256(OUT / name)
            for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv", "model_registry.csv")
        },
        "source_docx": {"path": str(ORIGINAL_DOCX), "sha256": R.sha256(ORIGINAL_DOCX)},
    }
    write_json(OUT / "protocol.json", protocol)
    lock = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": R.sha256(OUT / "protocol.json"),
        "source_sha256": R.sha256(Path(__file__)),
        "weights": {key: R.sha256(weight_path(key)) for key in MODEL_SPECS},
    }
    write_json(OUT / "protocol_locked.json", lock)
    shutil.copy2(Path(__file__), OUT / "source" / Path(__file__).name)
    print("Prepared locked nine-model protocol for 98 patients and 349 masked images", flush=True)


def local_model_audit() -> None:
    entries = []
    candidates = [
        *sorted(ROOT.glob("best_*.pth")),
        ROOT / "Densenet201_checkpoint_ntuh.h5",
        ROOT / "cv_fold1_best_densenet121.pth",
        ROOT / "augmentation_20260912/cych/resnet50_none_best.pth",
        ROOT / "augmentation_20260912/cych/resnet50_realistic_best.pth",
        ROOT / "external_models/radimagenet/DenseNet121.pt",
        ROOT / "external_models/radimagenet/InceptionV3.pt",
    ]
    seen = set()
    for path in candidates:
        if not path.exists() or path.resolve() in seen:
            continue
        seen.add(path.resolve())
        relative = str(path.relative_to(ROOT))
        if path.name == "best_mobilenetv2_triplet.pth":
            provenance = "finetune_mobilenetv2_triplet_optuna.py: NCKUH image-level folds"
            reason = "Different site and image-level split; architecture is MobileNetV2, not the draft's MobileNet v1 comparator"
        elif path.name.startswith("best_") or path.name == "cv_fold1_best_densenet121.pth":
            provenance = "advanced_cnn_comparison.py or related legacy mixed-site image-level workflow"
            reason = "Legacy image-level k-fold and/or mixed-site training; not comparable with fixed patient assignments"
        elif "augmentation_20260912" in relative:
            provenance = "Earlier CYCH augmentation experiment"
            reason = "Different training and validation protocol; retained as historical evidence, not reused as a nine-model head"
        elif "radimagenet" in relative.lower():
            provenance = "Existing RadImageNet medical-image encoder"
            reason = "Valuable domain comparator already evaluated elsewhere, but not the ImageNet implementation used for the draft-family replication"
        else:
            provenance = "Legacy project checkpoint with incomplete split provenance"
            reason = "Insufficient evidence of the fixed 58/20/20 patient-disjoint protocol"
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": R.sha256(path),
                "provenance": provenance,
                "eligible_for_new_nine_model_comparison": False,
                "reason": reason,
            }
        )
    pd.DataFrame(entries).to_csv(OUT / "legacy_local_model_audit.csv", index=False, encoding="utf-8-sig")


def require_lock() -> None:
    lock = json.loads((OUT / "protocol_locked.json").read_text(encoding="utf-8"))
    assert lock["protocol_sha256"] == R.sha256(OUT / "protocol.json")
    assert lock["source_sha256"] == R.sha256(Path(__file__))
    for key, digest in lock["weights"].items():
        assert digest == R.sha256(weight_path(key)), f"Weight changed: {key}"


class VGGFeatures(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(*list(model.classifier.children())[:4])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def repair_densenet_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )
    fixed = dict(state)
    for key in list(fixed):
        match = pattern.match(key)
        if match:
            fixed[match.group(1) + match.group(2)] = fixed.pop(key)
    return fixed


def load_torch_model(model_key: str) -> nn.Module:
    from safetensors.torch import load_file
    from torchvision import models
    import timm

    path = weight_path(model_key)
    if model_key in {"mobilenet", "inception_resnet_v2"}:
        identifier = {
            "mobilenet": "mobilenetv1_100.ra4_e3600_r224_in1k",
            "inception_resnet_v2": "inception_resnet_v2.tf_in1k",
        }[model_key]
        model = timm.create_model(identifier, pretrained=False)
        model.load_state_dict(load_file(str(path)), strict=True)
        model.reset_classifier(0)
        return model

    constructors = {
        "vgg16": lambda: models.vgg16(weights=None),
        "vgg19": lambda: models.vgg19(weights=None),
        "resnet50": lambda: models.resnet50(weights=None),
        "resnet101": lambda: models.resnet101(weights=None),
        "densenet121": lambda: models.densenet121(weights=None),
        "inception_v3": lambda: models.inception_v3(
            weights=None, aux_logits=True, transform_input=True, init_weights=False
        ),
    }
    model = constructors[model_key]()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if model_key == "densenet121":
        state = repair_densenet_keys(state)
    model.load_state_dict(state, strict=True)
    if model_key in {"vgg16", "vgg19"}:
        model = VGGFeatures(model)
    elif model_key in {"resnet50", "resnet101"}:
        model.fc = nn.Identity()
    elif model_key == "densenet121":
        model.classifier = nn.Identity()
    elif model_key == "inception_v3":
        model.fc = nn.Identity()
        model.AuxLogits = None
        model.aux_logits = False
    return model


def state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def keras_state_digest(model: object) -> str:
    digest = hashlib.sha256()
    for variable in model.weights:
        digest.update(variable.name.encode("utf-8"))
        digest.update(np.asarray(variable.numpy()).tobytes())
    return digest.hexdigest()


def roi_canvases(size: int) -> tuple[np.ndarray, list[dict[str, str]]]:
    manifest = read("image_manifest.csv")
    arrays = []
    audit = []
    for row in manifest.itertuples():
        image_path = SOURCE / "native" / f"{row.image_id}.png"
        mask_path = SOURCE / "masks" / f"{row.image_id}.png"
        if R.sha256(mask_path) != row.mask_sha256:
            raise RuntimeError(f"Mask hash mismatch: {row.image_id}")
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None or gray.shape != mask.shape:
            raise RuntimeError(f"Invalid image/mask pair: {row.image_id}")
        canvas = R.canvas_image(gray, (mask > 0).astype(np.uint8), size=size)
        if canvas.shape != (size, size, 3) or not np.any(canvas):
            raise RuntimeError(f"Unexpected canvas content: {row.image_id}")
        if not np.array_equal(canvas[..., 0], canvas[..., 1]) or not np.array_equal(canvas[..., 1], canvas[..., 2]):
            raise RuntimeError(f"Grayscale channel replication failed: {row.image_id}")
        arrays.append(canvas)
        audit.append({"image_id": row.image_id, "native_sha256": R.sha256(image_path), "mask_sha256": row.mask_sha256})
    result = np.stack(arrays)
    assert result.shape == (349, size, size, 3)
    return result, audit


def normalize_torch(images: np.ndarray, mode: str) -> torch.Tensor:
    x = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).float() / 255.0
    if mode.startswith("ImageNet mean/std"):
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return (x - mean) / std
    if mode == "x / 127.5 - 1":
        return (x - 0.5) / 0.5
    raise ValueError(mode)


def extract_torch_features(model_key: str, images: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_torch_model(model_key).eval().requires_grad_(False)
    before = state_digest(model)
    model = model.to(device)
    batch_size = 12 if MODEL_SPECS[model_key]["input_size"] == 299 else 24
    values = []
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            x = normalize_torch(images[start : start + batch_size], MODEL_SPECS[model_key]["normalization"])
            values.append(model(x.to(device)).detach().cpu().numpy())
    model = model.cpu()
    after = state_digest(model)
    result = np.concatenate(values).astype(np.float32)
    expected = MODEL_SPECS[model_key]["feature_dimension"]
    if result.shape != (349, expected) or not np.isfinite(result).all():
        raise RuntimeError(f"Unexpected {model_key} features: {result.shape}")
    if before != after:
        raise RuntimeError(f"Frozen model changed during extraction: {model_key}")
    return result, {
        "device": str(device),
        "seconds": time.monotonic() - started,
        "strict_weight_load": True,
        "state_before": before,
        "state_after": after,
        "trainable_parameters": 0,
    }


def extract_xception_features(images: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    import tensorflow as tf

    tf.random.set_seed(20260914)
    model = tf.keras.applications.Xception(
        include_top=False,
        weights=str(weight_path("xception")),
        pooling="avg",
        input_shape=(299, 299, 3),
    )
    model.trainable = False
    before = keras_state_digest(model)
    values = []
    started = time.monotonic()
    with tf.device("/CPU:0"):
        for start in range(0, len(images), 12):
            x = tf.keras.applications.xception.preprocess_input(images[start : start + 12].astype(np.float32))
            values.append(model(x, training=False).numpy())
    after = keras_state_digest(model)
    result = np.concatenate(values).astype(np.float32)
    if result.shape != (349, 2048) or not np.isfinite(result).all():
        raise RuntimeError(f"Unexpected Xception features: {result.shape}")
    if before != after:
        raise RuntimeError("Frozen Xception changed during extraction")
    return result, {
        "device": "/CPU:0",
        "seconds": time.monotonic() - started,
        "strict_weight_load": True,
        "state_before": before,
        "state_after": after,
        "trainable_parameters": 0,
    }


def extract() -> None:
    require_lock()
    manifest = read("image_manifest.csv")
    patients = read("patients.csv")
    registry = pd.read_csv(OUT / "model_registry.csv")
    provenance = []
    canvas_cache: dict[int, tuple[np.ndarray, list[dict[str, str]]]] = {}
    for model_key, spec in MODEL_SPECS.items():
        image_path = OUT / "features" / "image" / f"{model_key}.npy"
        patient_path = OUT / "features" / "patient" / f"{model_key}.npy"
        if image_path.exists() and patient_path.exists():
            image_features = np.load(image_path)
            patient_features = np.load(patient_path)
            if image_features.shape == (349, spec["feature_dimension"]) and patient_features.shape == (
                98,
                spec["feature_dimension"] * 2,
            ):
                registry.loc[registry.model_key.eq(model_key), "load_validation"] = (
                    "strict load; finite frozen features; state hash unchanged"
                )
                registry.to_csv(OUT / "model_registry.csv", index=False, encoding="utf-8-sig")
                print(f"Verified existing features: {model_key}", flush=True)
                continue
        size = spec["input_size"]
        if size not in canvas_cache:
            canvas_cache[size] = roi_canvases(size)
        images, input_audit = canvas_cache[size]
        if model_key == "xception":
            image_features, audit = extract_xception_features(images)
        else:
            image_features, audit = extract_torch_features(model_key, images)
        patient_features = R.aggregate(image_features, manifest, patients)
        assert patient_features.shape == (98, spec["feature_dimension"] * 2)
        np.save(image_path, image_features)
        np.save(patient_path, patient_features)
        audit.update(
            model_key=model_key,
            paper_name=spec["paper_name"],
            image_feature_shape=list(image_features.shape),
            patient_feature_shape=list(patient_features.shape),
            image_feature_sha256=R.sha256(image_path),
            patient_feature_sha256=R.sha256(patient_path),
            input_audit=input_audit,
        )
        provenance.append(audit)
        registry.loc[registry.model_key.eq(model_key), "load_validation"] = "strict load; finite frozen features; state hash unchanged"
        registry.to_csv(OUT / "model_registry.csv", index=False, encoding="utf-8-sig")
        write_json(OUT / "features" / "extraction_provenance.json", provenance)
        print(f"Extracted {model_key}: image {image_features.shape}, patient {patient_features.shape}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def mine_triplets(embeddings: torch.Tensor, labels: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, float]:
    distances = torch.cdist(embeddings.detach().cpu(), embeddings.detach().cpu())
    y = labels.detach().cpu()
    positives = []
    negatives = []
    semi_hard = []
    for anchor in range(len(y)):
        positive_pool = torch.where((y == y[anchor]) & (torch.arange(len(y)) != anchor))[0]
        negative_pool = torch.where(y != y[anchor])[0]
        if not len(positive_pool) or not len(negative_pool):
            raise ValueError("Triplet mining needs two patients in each class")
        positive = positive_pool[torch.randint(len(positive_pool), (1,), generator=generator).item()].item()
        distance = distances[anchor]
        eligible = negative_pool[
            (distance[negative_pool] > distance[positive])
            & (distance[negative_pool] < distance[positive] + TRIPLET_MARGIN)
        ]
        pool = eligible if len(eligible) else negative_pool
        negative = pool[torch.randint(len(pool), (1,), generator=generator).item()].item()
        positives.append(positive)
        negatives.append(negative)
        semi_hard.append(bool(len(eligible)))
    return (
        torch.tensor(positives, device=embeddings.device),
        torch.tensor(negatives, device=embeddings.device),
        float(np.mean(semi_hard)),
    )


def train_triplet_projection(
    train_x: np.ndarray, train_y: np.ndarray, all_x: np.ndarray, seed: int
) -> tuple[np.ndarray, nn.Module, dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(imputer.fit_transform(train_x)).astype(np.float32)
    scaled_all = scaler.transform(imputer.transform(all_x)).astype(np.float32)
    scaled_train /= np.maximum(np.linalg.norm(scaled_train, axis=1, keepdims=True), 1e-8)
    scaled_all /= np.maximum(np.linalg.norm(scaled_all, axis=1, keepdims=True), 1e-8)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    projection = nn.Linear(train_x.shape[1], EMBEDDING_DIMENSION, bias=False).to(device)
    optimizer = torch.optim.Adam(projection.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2, eps=1e-6, reduction="none")
    generator = torch.Generator().manual_seed(seed + 97)
    x = torch.from_numpy(scaled_train).to(device)
    y = torch.from_numpy(train_y.astype(np.int64)).to(device)
    class_weights = torch.tensor(
        [len(y) / (2 * int((y == label).sum())) for label in y], dtype=torch.float32, device=device
    )
    history = []
    initial = state_digest(projection.cpu())
    projection = projection.to(device)
    for epoch in range(1, TRIPLET_EPOCHS + 1):
        z = F.normalize(projection(x), dim=1)
        positive, negative, semi_fraction = mine_triplets(z, y, generator)
        losses = criterion(z, z[positive], z[negative])
        loss = (losses * class_weights).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite triplet loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(projection.parameters(), 5.0)
        optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach()),
                "active_fraction": float((losses > 0).float().mean()),
                "semi_hard_fraction": semi_fraction,
            }
        )
    final = state_digest(projection.cpu())
    if initial == final:
        raise RuntimeError("Triplet projection did not update")
    projection = projection.to(device).eval()
    with torch.inference_mode():
        embedded = F.normalize(projection(torch.from_numpy(scaled_all).to(device)), dim=1).cpu().numpy()
    assert embedded.shape == (98, EMBEDDING_DIMENSION) and np.isfinite(embedded).all()
    preprocessing = {"imputer_statistics": imputer.statistics_.tolist(), "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()}
    audit = {
        "seed": seed,
        "train_rows": len(train_x),
        "input_dimension": train_x.shape[1],
        "embedding_dimension": EMBEDDING_DIMENSION,
        "epochs": TRIPLET_EPOCHS,
        "margin": TRIPLET_MARGIN,
        "initial_state": initial,
        "final_state": final,
        "loss_first": history[0]["loss"],
        "loss_last": history[-1]["loss"],
        "active_fraction_last": history[-1]["active_fraction"],
        "semi_hard_fraction_last": history[-1]["semi_hard_fraction"],
        "preprocessing": preprocessing,
        "history": history,
    }
    return embedded.astype(np.float32), projection.cpu(), audit


def pca_svm(seed: int) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        PCA(n_components=2, random_state=seed),
        SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=False),
    )


def score_row(
    model_key: str,
    workflow: str,
    seed: int,
    role: str,
    y: np.ndarray,
    score: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y, score)
    return {
        "model_key": model_key,
        "model": MODEL_SPECS[model_key]["paper_name"],
        "workflow": workflow,
        "workflow_label": WORKFLOWS[workflow],
        "seed": seed,
        "role": "development_validation" if role == "validation" else "exploratory_test",
        "n": len(y),
        "auroc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "auprc_trapezoid": float(auc(recall[::-1], precision[::-1])),
        "accuracy": float(accuracy_score(y, prediction)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def evaluate() -> None:
    require_lock()
    patients = read("patients.csv")
    assignments = read("random_patient_assignments.csv")
    y = patients.label.to_numpy(dtype=np.int64)
    metrics = []
    predictions = []
    fit_audit = []
    with threadpool_limits(limits=4):
        for model_key in MODEL_SPECS:
            frozen = np.load(OUT / "features" / "patient" / f"{model_key}.npy")
            assert frozen.shape[0] == 98 and np.isfinite(frozen).all()
            for seed in R.SEEDS:
                indices = cohort_indices(patients, assignments, seed)
                train = indices["train"]
                heads = {}
                frozen_lr = V.make_head("lr", seed, frozen.shape[1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    frozen_lr.fit(frozen[train], y[train])
                heads["frozen_lr"] = (frozen_lr, V.raw_score(frozen_lr, frozen), frozen_lr.predict(frozen))

                frozen_svm = pca_svm(seed)
                frozen_svm.fit(frozen[train], y[train])
                heads["frozen_pca2_svm"] = (
                    frozen_svm,
                    frozen_svm.decision_function(frozen),
                    frozen_svm.predict(frozen),
                )

                embedded, projection, triplet_audit = train_triplet_projection(
                    frozen[train], y[train], frozen, seed
                )
                projection_path = OUT / "models" / f"{model_key}_triplet128_{seed}.pt"
                torch.save(
                    {
                        "model_key": model_key,
                        "seed": seed,
                        "state_dict": projection.state_dict(),
                        "input_dimension": frozen.shape[1],
                        "embedding_dimension": EMBEDDING_DIMENSION,
                        "margin": TRIPLET_MARGIN,
                    },
                    projection_path,
                )
                write_json(OUT / "models" / f"{model_key}_triplet128_{seed}_audit.json", triplet_audit)

                triplet_lr = V.make_head("lr", seed, embedded.shape[1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    triplet_lr.fit(embedded[train], y[train])
                heads["triplet128_lr"] = (
                    triplet_lr,
                    V.raw_score(triplet_lr, embedded),
                    triplet_lr.predict(embedded),
                )
                triplet_svm = pca_svm(seed)
                triplet_svm.fit(embedded[train], y[train])
                heads["triplet128_pca2_svm"] = (
                    triplet_svm,
                    triplet_svm.decision_function(embedded),
                    triplet_svm.predict(embedded),
                )

                for workflow, (head, scores, predicted) in heads.items():
                    model_path = OUT / "models" / f"{model_key}_{workflow}_{seed}.joblib"
                    joblib.dump(head, model_path, compress=3)
                    fit_audit.append(
                        {
                            "model_key": model_key,
                            "workflow": workflow,
                            "seed": seed,
                            "training_patient_count": len(train),
                            "training_patient_ids": patients.patient_id.iloc[train].tolist(),
                            "model_path": str(model_path.relative_to(OUT)),
                            "model_sha256": R.sha256(model_path),
                        }
                    )
                    for role in ("validation", "test"):
                        idx = indices[role]
                        metrics.append(score_row(model_key, workflow, seed, role, y[idx], scores[idx], predicted[idx]))
                        for patient_index in idx:
                            predictions.append(
                                {
                                    "model_key": model_key,
                                    "model": MODEL_SPECS[model_key]["paper_name"],
                                    "workflow": workflow,
                                    "seed": seed,
                                    "role": "development_validation" if role == "validation" else "exploratory_test",
                                    "patient_id": patients.patient_id.iloc[patient_index],
                                    "label": int(y[patient_index]),
                                    "score": float(scores[patient_index]),
                                    "prediction": int(predicted[patient_index]),
                                }
                            )
                print(f"{model_key} seed={seed}: four predeclared workflows complete", flush=True)
    pd.DataFrame(metrics).to_csv(OUT / "metrics_by_role_per_draw.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "patient_predictions_by_role.csv", index=False)
    write_json(OUT / "fit_audit.json", fit_audit)
    summarize()


def summarize() -> None:
    metrics = pd.read_csv(OUT / "metrics_by_role_per_draw.csv")
    rows = []
    for (role, model_key, model, workflow), group in metrics.groupby(
        ["role", "model_key", "model", "workflow"], sort=False
    ):
        row = {
            "role": role,
            "model_key": model_key,
            "model": model,
            "workflow": workflow,
            "workflow_label": WORKFLOWS[workflow],
            "draws": len(group),
            "joint_target_draws": int(((group.auroc > 0.8) & (group.average_precision > 0.8)).sum()),
        }
        for metric in ("auroc", "average_precision", "auprc_trapezoid", "accuracy", "sensitivity", "specificity"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std())
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary_by_role.csv", index=False)

    paired = []
    comparisons = [
        ("triplet128_lr", "frozen_lr"),
        ("triplet128_pca2_svm", "frozen_pca2_svm"),
    ]
    for role in ("development_validation", "exploratory_test"):
        role_data = metrics.loc[metrics.role.eq(role)]
        for model_key in MODEL_SPECS:
            for triplet_workflow, baseline_workflow in comparisons:
                baseline = role_data.loc[
                    role_data.model_key.eq(model_key) & role_data.workflow.eq(baseline_workflow)
                ].set_index("seed")
                triplet = role_data.loc[
                    role_data.model_key.eq(model_key) & role_data.workflow.eq(triplet_workflow)
                ].set_index("seed")
                for seed in baseline.index:
                    paired.append(
                        {
                            "role": role,
                            "model_key": model_key,
                            "model": MODEL_SPECS[model_key]["paper_name"],
                            "triplet_workflow": triplet_workflow,
                            "baseline_workflow": baseline_workflow,
                            "seed": seed,
                            "delta_auroc": float(triplet.loc[seed, "auroc"] - baseline.loc[seed, "auroc"]),
                            "delta_average_precision": float(
                                triplet.loc[seed, "average_precision"] - baseline.loc[seed, "average_precision"]
                            ),
                        }
                    )
    pd.DataFrame(paired).to_csv(OUT / "paired_triplet_effect.csv", index=False)

    development = summary.loc[summary.role.eq("development_validation")].copy()
    development["mean_rank_score"] = (development.auroc_mean + development.average_precision_mean) / 2
    development.sort_values(["workflow", "mean_rank_score"], ascending=[True, False]).to_csv(
        OUT / "development_ranking.csv", index=False
    )


def export_figure(figure: object, stem: str) -> None:
    from PIL import Image

    png = OUT / "figures_600dpi" / f"{stem}.png"
    tiff = OUT / "figures_600dpi" / f"{stem}.tiff"
    preview = OUT / "qa" / f"{stem}_preview.png"
    figure.savefig(png, dpi=600, facecolor="white")
    with Image.open(png) as image:
        image.save(tiff, compression="tiff_lzw", dpi=(600, 600))
    figure.savefig(preview, dpi=150, facecolor="white")
    with Image.open(png) as image:
        extrema = image.convert("L").getextrema()
        if extrema[0] == extrema[1]:
            raise RuntimeError(f"Blank figure: {png}")
    write_json(
        OUT / "qa" / f"{stem}_qa.json",
        {
            "png": str(png),
            "png_sha256": R.sha256(png),
            "tiff": str(tiff),
            "tiff_sha256": R.sha256(tiff),
            "dpi": 600,
            "nonblank": True,
        },
    )


def report() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = pd.read_csv(OUT / "summary_by_role.csv")
    plot_workflows = ["frozen_lr", "triplet128_pca2_svm"]
    models = [spec["paper_name"] for spec in MODEL_SPECS.values()]
    colors = {"development_validation": "#187c73", "exploratory_test": "#bd4f3f"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 11), sharex=True)
    for row, workflow in enumerate(plot_workflows):
        for column, metric in enumerate(("auroc", "average_precision")):
            axis = axes[row, column]
            for role, offset, marker in (
                ("development_validation", -0.10, "o"),
                ("exploratory_test", 0.10, "s"),
            ):
                values = summary.loc[
                    summary.role.eq(role) & summary.workflow.eq(workflow)
                ].set_index("model").loc[models]
                y = np.arange(len(models)) + offset
                axis.errorbar(
                    values[f"{metric}_mean"],
                    y,
                    xerr=values[f"{metric}_std"],
                    fmt=marker,
                    color=colors[role],
                    capsize=2,
                    markersize=4,
                    linewidth=1,
                    label=role.replace("_", " ") if row == 0 and column == 0 else None,
                )
            axis.axvline(0.8, color="#8f2d56", linestyle="--", linewidth=1)
            axis.set_xlim(0, 1)
            axis.set_yticks(range(len(models)), models if column == 0 else [""] * len(models))
            workflow_title = "Frozen LR" if workflow == "frozen_lr" else "Triplet(128) + PCA(2) + RBF SVM"
            metric_title = "AUROC" if metric == "auroc" else "Average precision"
            axis.set_title(f"{workflow_title}: {metric_title}", fontsize=10)
            axis.grid(axis="x", alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
    figure.legend(loc="lower center", ncol=2, frameon=False)
    figure.suptitle("Nine original CNN families on segmented CYCH kidney ROIs", fontsize=13)
    figure.subplots_adjust(left=0.20, right=0.99, top=0.93, bottom=0.08, hspace=0.22, wspace=0.14)
    export_figure(figure, "Figure_N1_nine_model_comparison")
    plt.close(figure)

    development = summary.loc[
        summary.role.eq("development_validation") & summary.workflow.eq("triplet128_pca2_svm")
    ].sort_values(["auroc_mean", "average_precision_mean"], ascending=False)
    test = summary.loc[
        summary.role.eq("exploratory_test") & summary.workflow.eq("triplet128_pca2_svm")
    ].set_index("model_key")
    paired = pd.read_csv(OUT / "paired_triplet_effect.csv")
    paired_dev = paired.loc[
        paired.role.eq("development_validation")
        & paired.triplet_workflow.eq("triplet128_pca2_svm")
    ]
    registry = pd.read_csv(OUT / "model_registry.csv")
    legacy = pd.read_csv(OUT / "legacy_local_model_audit.csv")
    lines = [
        "# CYCH original nine-model comparison",
        "",
        "The 2022 draft compares nine CNN feature backbones followed by triplet features, PCA, and an SVM. This run evaluates those nine named architecture families using the current verified image/label layer and mandatory whole-kidney masks.",
        "",
        "## What was found",
        "",
        f"- The project contained {len(legacy)} relevant historical checkpoints, but none could be reused as a fair nine-model comparator because their site, split, architecture version, or patient-disjoint provenance differs.",
        "- Six official TorchVision ImageNet checkpoints were already present in the machine cache and were copied into the project after SHA-256 verification.",
        "- MobileNet v1 and InceptionResNetV2 safetensors were downloaded from official timm Hugging Face model cards. Xception no-top HDF5 weights were downloaded from official TensorFlow/Keras storage.",
        "- Every eligible weight was strictly loaded; extracted features were finite; each frozen model state hash was unchanged after inference.",
        "",
        "## Locked comparison",
        "",
        "- Cohort: 98 patients, 349 first-exam images, 52 positive and 46 negative patients.",
        "- Each of ten repeated stratified draws uses 58 training, 20 development-validation, and 20 exploratory-test patients with no overlap.",
        "- Inputs are masked whole-kidney grayscale ROIs. No EMR variables, augmentation, GAN, or test-time augmentation are used.",
        "- Encoders are frozen. Patient features are the mean and standard deviation of image embeddings.",
        "- The original-inspired branch uses a training-only 128-dimensional triplet projection, PCA(2), and RBF SVM. It is not an exact reproduction because the original 209-patient split and verified EMR mapping are unavailable.",
        "- Development results describe model ordering. Exploratory test results are reported for all predeclared models and were not used for selection.",
        "",
        "## Original-inspired triplet + PCA(2) + SVM",
        "",
        "| Model | Development AUROC | Development AP | Exploratory AUROC | Exploratory AP | Triplet dev delta AUROC | Triplet dev delta AP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in development.itertuples():
        test_row = test.loc[row.model_key]
        delta = paired_dev.loc[paired_dev.model_key.eq(row.model_key)]
        lines.append(
            f"| {row.model} | {row.auroc_mean:.3f} | {row.average_precision_mean:.3f} | {test_row.auroc_mean:.3f} | {test_row.average_precision_mean:.3f} | {delta.delta_auroc.mean():+.3f} | {delta.delta_average_precision.mean():+.3f} |"
        )
    best = development.iloc[0]
    best_test = test.loc[best.model_key]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The highest mean development result in the original-inspired branch is {best.model} (AUROC {best.auroc_mean:.3f}, AP {best.average_precision_mean:.3f}); its exploratory-test means are AUROC {best_test.auroc_mean:.3f} and AP {best_test.average_precision_mean:.3f}.",
            "",
            "These are retrospective exploratory estimates, not independent validation and not evidence that the 2022 Table 2 values were reproduced. The comparison answers which of the nine named ImageNet representation families transfers best under the current patient-safe protocol. A model should enter a vote only after development-only advancement criteria and diversity analysis; a high single draw is not sufficient.",
            "",
            "## Files",
            "",
            "- `model_registry.csv`: exact implementation, weight source, license note, size, and SHA-256 for all nine models.",
            "- `legacy_local_model_audit.csv`: relevant project checkpoints and why they are or are not comparable.",
            "- `original_paper_table2.csv`: values transcribed from page 11 of the original draft.",
            "- `metrics_by_role_per_draw.csv` and `summary_by_role.csv`: complete repeated-draw results.",
            "- `paired_triplet_effect.csv`: within-seed effect of adding the 128-dimensional triplet projection.",
            "- `figures_600dpi/Figure_N1_nine_model_comparison.png` and `.tiff`: journal-resolution comparison figure.",
            "",
            "## Weight inventory",
            "",
            "| Model | Framework | Weight file | MiB | SHA-256 |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in registry.itertuples():
        lines.append(
            f"| {row.paper_name} | {row.framework} | {row.weight_file} | {row.bytes / 2**20:.1f} | `{row.sha256}` |"
        )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package() -> None:
    require_lock()
    report()
    archive = ROOT / f"{OUT.name}.zip"
    if archive.exists():
        archive.unlink()
    shutil.make_archive(str(archive.with_suffix("")), "zip", ROOT, OUT.name)
    write_json(
        OUT / "package_manifest.json",
        {
            "archive": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": R.sha256(archive),
            "weights_directory_not_duplicated_in_zip": str(WEIGHTS),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Package: {archive}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "extract", "evaluate", "report", "package", "all"))
    args = parser.parse_args()
    stages = ("prepare", "extract", "evaluate", "report", "package") if args.stage == "all" else (args.stage,)
    for stage in stages:
        globals()[stage]()


if __name__ == "__main__":
    main()
