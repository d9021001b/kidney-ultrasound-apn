"""Run an auditable first reproduction baseline for the 2022 kidney paper.

The exact 2022 cohort manifest and diagnosis mapping are not present in the
workspace, so this runner deliberately labels its output as a baseline. It
uses the current labelled CYCH/NCKUH image layer, patient-grouped splitting,
VGG16 image features, a 128-D triplet projection, available metadata, PCA(2),
and an SVM classifier. It never mixes the original and `_red_` image variants
in one baseline sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
PID_RE = re.compile(r"^(\d{8})_")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def logical_name(filename: str) -> str:
    return filename.replace("_red_", "_")


def patient_id(filename: str) -> str:
    match = PID_RE.match(filename)
    return match.group(1) if match else ""


def load_records(site: str) -> list[dict[str, Any]]:
    csv_path = ROOT / site / f"{site}.csv"
    original_dir = ROOT / ("original_png" if site == "CYCH" else "original_png_nckuh")
    data_dir = ROOT / site / "Data"
    metadata_path = ROOT / f"{site.lower()}_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_by_name = {item["filename"]: item for item in metadata}
    all_rows = list(csv.DictReader(csv_path.open("r", newline="", encoding="utf-8-sig")))
    original_rows = [row for row in all_rows if "_red_" not in row["filename"]]
    red_rows = {logical_name(row["filename"]): row for row in all_rows if "_red_" in row["filename"]}
    records: list[dict[str, Any]] = []
    for row in original_rows:
        filename = row["filename"]
        image_path = original_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Missing original image: {image_path}")
        item = metadata_by_name.get(filename)
        if item is None:
            raise KeyError(f"Missing metadata for {filename}")
        label = int(float(row["label"]))
        red = red_rows.get(logical_name(filename))
        if red is None:
            raise ValueError(f"Missing red-variant pair for {filename}")
        if int(float(red["label"])) != label:
            raise ValueError(f"Original/red label mismatch for {filename}")
        gender_value = str(item.get("gender", "")).strip().lower()
        gender = 1.0 if gender_value in {"male", "m", "1"} else 0.0
        records.append(
            {
                "site": site,
                "filename": filename,
                "image_path": str(image_path),
                "patient_id": patient_id(filename),
                "exam_date": filename.split("_")[1] if len(filename.split("_")) > 1 else "",
                "label": label,
                "gender": gender,
                "long_axis": float(item.get("long_axis", 0.0)),
                "short_axis": float(item.get("short_axis", 0.0)),
                "red_variant_exists": (data_dir / red["filename"]).exists(),
            }
        )
    if len({record["filename"] for record in records}) != len(records):
        raise ValueError("Duplicate original filenames detected")
    return records


class ImageDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], transform: transforms.Compose):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")
            return self.transform(image), index


class VGG16FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(*list(model.classifier.children())[:4])
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.features(images)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)


def extract_features(
    records: list[dict[str, Any]], batch_size: int, device: torch.device
) -> np.ndarray:
    weights = models.VGG16_Weights.DEFAULT
    transform = weights.transforms()
    loader = DataLoader(
        ImageDataset(records, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    extractor = VGG16FeatureExtractor().to(device).eval()
    output: list[np.ndarray] = []
    for images, _ in loader:
        output.append(extractor(images.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def patient_group_split(
    records: list[dict[str, Any]], test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    patient_labels: dict[str, int] = {}
    patient_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        pid = record["patient_id"]
        patient_indices[pid].append(index)
        patient_labels.setdefault(pid, record["label"])
        if patient_labels[pid] != record["label"]:
            raise ValueError(f"Patient has inconsistent labels: {pid}")
    patients = np.array(sorted(patient_indices), dtype=object)
    labels = np.array([patient_labels[pid] for pid in patients], dtype=np.int64)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_patient_idx, test_patient_idx = next(splitter.split(patients, labels))
    train_patients = set(patients[train_patient_idx].tolist())
    test_patients = set(patients[test_patient_idx].tolist())
    train_indices = np.array([i for pid in train_patients for i in patient_indices[pid]], dtype=np.int64)
    test_indices = np.array([i for pid in test_patients for i in patient_indices[pid]], dtype=np.int64)
    return np.sort(train_indices), np.sort(test_indices)


def train_triplet_projection(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    margin: float,
) -> tuple[nn.Module, list[float]]:
    set_seed(seed)
    feature_tensor = torch.from_numpy(train_features).to(device)
    labels = train_labels.astype(np.int64)
    by_class = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    if any(len(indices) < 2 for indices in by_class.values()):
        raise ValueError("Triplet training requires at least two samples in each class")

    projection = nn.Sequential(
        nn.Linear(train_features.shape[1], 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    ).to(device)
    optimizer = torch.optim.Adam(projection.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    steps = max(4, int(np.ceil(len(labels) / batch_size)))
    for _ in range(epochs):
        projection.train()
        epoch_losses: list[float] = []
        for _ in range(steps):
            anchor_indices = rng.integers(0, len(labels), size=min(batch_size, len(labels)))
            positive_indices = []
            negative_indices = []
            for anchor_index in anchor_indices:
                label = labels[anchor_index]
                candidates = by_class[label]
                positive = int(rng.choice(candidates[candidates != anchor_index]))
                negative = int(rng.choice(by_class[1 - label]))
                positive_indices.append(positive)
                negative_indices.append(negative)
            anchor = F.normalize(projection(feature_tensor[anchor_indices]), dim=1)
            positive = F.normalize(projection(feature_tensor[positive_indices]), dim=1)
            negative = F.normalize(projection(feature_tensor[negative_indices]), dim=1)
            loss = criterion(anchor, positive, negative)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    projection.eval()
    return projection, losses


def project_features(projection: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.from_numpy(features).to(device)
        return F.normalize(projection(tensor), dim=1).cpu().numpy()


def metric_row(
    variant: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    row: dict[str, Any] = {
        "variant": variant,
        "n_test": int(len(y_true)),
        "positive_test": int(np.sum(y_true == 1)),
        "negative_test": int(np.sum(y_true == 0)),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "TPR": float(tp / (tp + fn)) if tp + fn else 0.0,
        "TNR": float(tn / (tn + fp)) if tn + fp else 0.0,
        "FPR": float(fp / (fp + tn)) if fp + tn else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "AUROC": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
    }
    return row


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["CYCH", "NCKUH"], default="CYCH")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=221 / 828)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--margin", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    output = ROOT / "reproduction_20260912" / args.site.lower()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_records(args.site)
    train_idx, test_idx = patient_group_split(records, args.test_size, args.seed)
    labels = np.array([record["label"] for record in records], dtype=np.int64)
    raw_features = extract_features(records, args.feature_batch_size, device)
    np.save(output / "vgg16_imagenet_features.npy", raw_features)

    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    projection, triplet_losses = train_triplet_projection(
        raw_features[train_idx],
        train_labels,
        device,
        args.seed,
        args.epochs,
        args.batch_size,
        args.margin,
    )
    torch.save(projection.state_dict(), output / "triplet_projection_128.pth")
    train_embedding = project_features(projection, raw_features[train_idx], device)
    test_embedding = project_features(projection, raw_features[test_idx], device)

    meta = np.array([[record["gender"], record["long_axis"], record["short_axis"]] for record in records], dtype=np.float32)
    variants = {
        "image_triplet_128": (train_embedding, test_embedding),
        "metadata_gender_long_short_axis": (meta[train_idx], meta[test_idx]),
        "fusion_triplet_128_plus_metadata": (
            np.concatenate([train_embedding, meta[train_idx]], axis=1),
            np.concatenate([test_embedding, meta[test_idx]], axis=1),
        ),
    }
    result_rows: list[dict[str, Any]] = []
    for variant, (x_train, x_test) in variants.items():
        classifier = Pipeline(
            [
                ("standardize", StandardScaler()),
                ("pca_2d", PCA(n_components=2, random_state=args.seed)),
                ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=args.seed)),
            ]
        )
        classifier.fit(x_train, train_labels)
        predictions = classifier.predict(x_test)
        scores = classifier.predict_proba(x_test)[:, 1]
        result_rows.append(metric_row(variant, test_labels, predictions, scores))

    manifest_rows = []
    train_set = set(train_idx.tolist())
    test_set = set(test_idx.tolist())
    for index, record in enumerate(records):
        manifest_rows.append(
            {
                "site": args.site,
                "filename": record["filename"],
                "patient_id": record["patient_id"],
                "exam_date": record["exam_date"],
                "label": record["label"],
                "image_variant": "original",
                "split": "train" if index in train_set else "test" if index in test_set else "error",
                "image_path": record["image_path"],
                "red_variant_exists": record["red_variant_exists"],
            }
        )
    save_csv(output / "split_manifest.csv", manifest_rows)
    save_csv(output / "baseline_results.csv", result_rows)

    config = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "site": args.site,
        "device": str(device),
        "seed": args.seed,
        "records": len(records),
        "unique_patients": len({record["patient_id"] for record in records}),
        "train_records": int(len(train_idx)),
        "test_records": int(len(test_idx)),
        "train_patients": len({records[i]["patient_id"] for i in train_idx}),
        "test_patients": len({records[i]["patient_id"] for i in test_idx}),
        "train_label_counts": dict(sorted(Counter(map(int, train_labels)).items())),
        "test_label_counts": dict(sorted(Counter(map(int, test_labels)).items())),
        "paper_reference_target_ratio": "221/828",
        "image_source": "original_png or original_png_nckuh; non-_red_ logical images only",
        "image_feature_model": "torchvision VGG16 ImageNet DEFAULT weights",
        "triplet_embedding_dimension": 128,
        "triplet_epochs": args.epochs,
        "triplet_batch_size": args.batch_size,
        "triplet_margin": args.margin,
        "metadata_fields_used": ["gender", "long_axis", "short_axis"],
        "classifier": "StandardScaler -> PCA(n_components=2) -> SVC(RBF, C=1.0, gamma=scale)",
        "exact_2022_replication": False,
        "reason_exact_is_false": [
            "The original 209-patient diagnosis mapping is unavailable in the workspace.",
            "The exact 828-image and 607/221 split manifest is unavailable.",
            "The available current metadata has gender and kidney axes, but no verified age field matching the paper's EMR description.",
            "The original preprocessing, CNN training, and SVM hyperparameters are not fully specified by the available files.",
        ],
        "triplet_loss_last_5_mean": float(np.mean(triplet_losses[-5:])),
        "triplet_loss_first": float(triplet_losses[0]),
        "triplet_loss_last": float(triplet_losses[-1]),
    }
    (output / "baseline_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary_lines = [
        "# Original-paper reproduction baseline",
        "",
        f"Generated: {config['generated']}",
        f"Site: `{args.site}`; device: `{device}`",
        "",
        "This is a reproducible baseline on the current labelled image layer. It is not the exact 2022 result because the original diagnosis mapping and 828-image split are not available.",
        "",
        f"- Logical original images: {len(records)}",
        f"- Patients: {config['unique_patients']}",
        f"- Train: {config['train_records']} images / {config['train_patients']} patients",
        f"- Test: {config['test_records']} images / {config['test_patients']} patients",
        f"- Test ratio target: {config['paper_reference_target_ratio']}",
        "- Image feature: torchvision VGG16 ImageNet features",
        "- Triplet projection: 128 dimensions",
        "- Fusion metadata: gender, long_axis, short_axis",
        "- Evaluation: StandardScaler -> PCA(2) -> RBF SVM; test split is patient-grouped",
        "",
        "## Results",
        "",
        "| Variant | n | TP | TN | FP | FN | Accuracy | TPR | TNR | FPR | Precision | F1 | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        summary_lines.append(
            "| {variant} | {n_test} | {TP} | {TN} | {FP} | {FN} | {accuracy:.4f} | {TPR:.4f} | {TNR:.4f} | {FPR:.4f} | {precision:.4f} | {F1:.4f} | {AUROC:.4f} |".format(
                **{**row, "AUROC": row["AUROC"] if row["AUROC"] is not None else 0.0}
            )
        )
    summary_lines += [
        "",
        "## Files",
        "",
        "- `baseline_config.json`: complete run configuration and exact-replication limitations.",
        "- `split_manifest.csv`: patient-grouped image split used by this run.",
        "- `baseline_results.csv`: machine-readable confusion counts and metrics.",
        "- `vgg16_imagenet_features.npy`: extracted pre-trained image features.",
        "- `triplet_projection_128.pth`: trained 128-D projection checkpoint.",
    ]
    (output / "baseline_report.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "config": config, "results": result_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
