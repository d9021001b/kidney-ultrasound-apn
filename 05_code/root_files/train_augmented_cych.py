"""Evaluate realistic image augmentation on segmented CYCH kidney crops.

The independent test set is fixed before training and is not used for early
stopping or augmentation selection. Kidney contour segmentation is completed
first by the existing YOLO segmentation model; this runner consumes only the
resulting `yolo_crops/CYCH` images and manifest. This is an improvement
experiment, not an exact recreation of the 2022 paper.
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
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
PID_RE = re.compile(r"^(\d{8})_")
OUTPUT = ROOT / "augmentation_20260912" / "cych"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patient_id(filename: str) -> str:
    match = PID_RE.match(filename)
    return match.group(1) if match else ""


def load_records() -> list[dict[str, Any]]:
    site = "CYCH"
    csv_path = ROOT / site / f"{site}.csv"
    segmentation_manifest = ROOT / "yolo_crops" / "manifest.csv"
    metadata = json.loads((ROOT / "cych_metadata.json").read_text(encoding="utf-8"))
    metadata_by_name = {item["filename"]: item for item in metadata}
    label_rows = {row["filename"]: row for row in csv.DictReader(csv_path.open("r", newline="", encoding="utf-8-sig"))}
    rows = [row for row in csv.DictReader(segmentation_manifest.open("r", newline="", encoding="utf-8-sig")) if row["site"] == site]
    records: list[dict[str, Any]] = []
    for row in rows:
        filename = row["filename"]
        path = ROOT / Path(row["relative_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        item = metadata_by_name.get(filename)
        if item is None:
            item = metadata_by_name.get(filename.replace("_red_", "_"))
        if item is None:
            raise KeyError(f"Missing metadata for segmented image {filename}")
        logical_filename = filename.replace("_red_", "_")
        if logical_filename not in label_rows:
            raise KeyError(f"Missing diagnosis label for segmented image {filename}")
        if int(float(label_rows[logical_filename]["label"])) != int(row["label"]):
            raise ValueError(f"Segmentation manifest label mismatch for {filename}")
        records.append(
            {
                "filename": filename,
                "path": str(path),
                "patient_id": patient_id(filename),
                "label": int(float(row["label"])),
                "gender": 1.0 if str(item.get("gender", "")).lower() in {"male", "m", "1"} else 0.0,
                "long_axis": float(item.get("long_axis", 0.0)),
                "short_axis": float(item.get("short_axis", 0.0)),
                "roi_source": row.get("roi_source", ""),
                "roi_w": int(float(row.get("roi_w", 0))),
                "roi_h": int(float(row.get("roi_h", 0))),
                "roi_aspect": float(row.get("roi_aspect", 0.0)),
            }
        )
    if len(records) != 353:
        raise ValueError(f"Expected 353 segmented CYCH images, found {len(records)}")
    if Counter(record["roi_source"] for record in records) != Counter({"yolo": 309, "physician_polygon": 44}):
        raise ValueError("Unexpected segmentation source counts; inspect yolo_crops/manifest.csv before training")
    return records


def grouped_split(records: list[dict[str, Any]], test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    by_patient: dict[str, list[int]] = defaultdict(list)
    patient_label: dict[str, int] = {}
    for index, record in enumerate(records):
        by_patient[record["patient_id"]].append(index)
        patient_label.setdefault(record["patient_id"], record["label"])
        if patient_label[record["patient_id"]] != record["label"]:
            raise ValueError(f"Inconsistent labels for patient {record['patient_id']}")
    patients = np.array(sorted(by_patient), dtype=object)
    labels = np.array([patient_label[pid] for pid in patients])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_patient_idx, test_patient_idx = next(splitter.split(patients, labels))
    train_patients = set(patients[train_patient_idx].tolist())
    test_patients = set(patients[test_patient_idx].tolist())
    train = np.array([i for pid in train_patients for i in by_patient[pid]], dtype=np.int64)
    test = np.array([i for pid in test_patients for i in by_patient[pid]], dtype=np.int64)
    return np.sort(train), np.sort(test)


def split_train_validation(records: list[dict[str, Any]], train_idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    by_patient: dict[str, list[int]] = defaultdict(list)
    patient_label: dict[str, int] = {}
    for index in train_idx:
        record = records[int(index)]
        by_patient[record["patient_id"]].append(int(index))
        patient_label.setdefault(record["patient_id"], record["label"])
    patients = np.array(sorted(by_patient), dtype=object)
    labels = np.array([patient_label[pid] for pid in patients])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 1)
    inner_train_patient_idx, val_patient_idx = next(splitter.split(patients, labels))
    inner_train_patients = set(patients[inner_train_patient_idx].tolist())
    val_patients = set(patients[val_patient_idx].tolist())
    inner_train = np.array([i for pid in inner_train_patients for i in by_patient[pid]], dtype=np.int64)
    val = np.array([i for pid in val_patients for i in by_patient[pid]], dtype=np.int64)
    return np.sort(inner_train), np.sort(val)


def build_transform(policy: str, train: bool) -> transforms.Compose:
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if not train or policy == "none":
        return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    if policy == "realistic":
        augmentations = [
            transforms.RandomResizedCrop(224, scale=(0.90, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5, fill=0),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.97, 1.03), fill=0),
            transforms.ColorJitter(brightness=0.12, contrast=0.15),
        ]
    elif policy == "strong":
        augmentations = [
            transforms.RandomResizedCrop(224, scale=(0.80, 1.0), ratio=(0.90, 1.10)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12, fill=0),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.90, 1.10), shear=5, fill=0),
            transforms.ColorJitter(brightness=0.20, contrast=0.25),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
        ]
    else:
        raise ValueError(f"Unknown augmentation policy: {policy}")
    return transforms.Compose(augmentations + [transforms.ToTensor(), normalize])


class ImageDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], indices: np.ndarray, transform: transforms.Compose):
        self.records = records
        self.indices = [int(index) for index in indices]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int, int]:
        index = self.indices[item]
        record = self.records[index]
        with Image.open(record["path"]) as image:
            image = image.convert("RGB")
            return self.transform(image), record["label"], index


def make_model() -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.30), nn.Linear(in_features, 2))
    return model


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    indices: list[int] = []
    with torch.no_grad():
        for images, batch_labels, batch_indices in loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            scores.extend(probabilities.cpu().numpy().tolist())
            labels.extend(batch_labels.numpy().tolist())
            indices.extend(batch_indices.numpy().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64), np.asarray(indices, dtype=np.int64)


def metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    y_pred = (scores >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positive": int(np.sum(y_true == 1)),
        "negative": int(np.sum(y_true == 0)),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "AUROC": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "AUPRC": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "TPR": float(tp / (tp + fn)) if tp + fn else 0.0,
        "TNR": float(tn / (tn + fp)) if tn + fp else 0.0,
    }


def patient_aggregate(records: list[dict[str, Any]], indices: np.ndarray, y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, int] = {}
    for index, label, score in zip(indices.tolist(), y_true.tolist(), scores.tolist()):
        pid = records[index]["patient_id"]
        grouped[pid].append(score)
        labels[pid] = label
    patient_ids = sorted(grouped)
    return np.asarray([labels[pid] for pid in patient_ids]), np.asarray([np.mean(grouped[pid]) for pid in patient_ids])


def train_policy(
    policy: str,
    records: list[dict[str, Any]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_loader = DataLoader(ImageDataset(records, train_idx, build_transform(policy, True)), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(ImageDataset(records, val_idx, build_transform(policy, False)), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(ImageDataset(records, test_idx, build_transform(policy, False)), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = make_model().to(device)
    train_labels = np.asarray([records[int(index)]["label"] for index in train_idx])
    counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    class_weights = torch.tensor([len(train_labels) / (2 * max(count, 1.0)) for count in counts], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_state = None
    best_score = -np.inf
    best_epoch = 0
    history: list[dict[str, Any]] = []
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device, non_blocking=True))
            loss = criterion(logits, labels.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_true, val_scores, _ = predict(model, val_loader, device)
        val_metrics = metrics(val_true, val_scores)
        selection_score = float((val_metrics["AUROC"] + val_metrics["AUPRC"]) / 2)
        history.append({"policy": policy, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_AUROC": val_metrics["AUROC"], "val_AUPRC": val_metrics["AUPRC"], "selection_score": selection_score})
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is None:
        raise RuntimeError(f"No valid checkpoint for policy {policy}")
    model.load_state_dict(best_state)
    test_true, test_scores, test_indices = predict(model, test_loader, device)
    image_metrics = metrics(test_true, test_scores)
    patient_true, patient_scores = patient_aggregate(records, test_indices, test_true, test_scores)
    patient_metrics = metrics(patient_true, patient_scores)
    checkpoint = OUTPUT / f"resnet50_{policy}_best.pth"
    torch.save({"model_state": best_state, "policy": policy, "best_epoch": best_epoch, "seed": args.seed}, checkpoint)
    prediction_path = OUTPUT / f"test_predictions_{policy}.csv"
    with prediction_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "patient_id", "label", "score", "predicted_label"])
        for index, label, score in zip(test_indices, test_true, test_scores):
            writer.writerow([records[int(index)]["filename"], records[int(index)]["patient_id"], int(label), float(score), int(score >= 0.5)])
    result = {
        "policy": policy,
        "best_epoch": best_epoch,
        "train_images": len(train_idx),
        "validation_images": len(val_idx),
        "test_images": len(test_idx),
        "test_patients": len({records[int(index)]["patient_id"] for index in test_idx}),
        "image_metrics": image_metrics,
        "patient_metrics": patient_metrics,
        "checkpoint": str(checkpoint),
        "predictions": str(prediction_path),
    }
    return result, history


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    set_seed(args.seed)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_records()
    train_idx, test_idx = grouped_split(records, 221 / 828, args.seed)
    inner_train_idx, val_idx = split_train_validation(records, train_idx, args.seed)
    results = []
    histories = []
    for policy in ("none", "realistic", "strong"):
        set_seed(args.seed)
        print(f"Training policy={policy} on {len(inner_train_idx)} images; validation={len(val_idx)}; test={len(test_idx)}")
        result, history = train_policy(policy, records, inner_train_idx, val_idx, test_idx, args, device)
        results.append(result)
        histories.extend(history)
        print(json.dumps(result, indent=2))
    (OUTPUT / "augmentation_results.json").write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "site": "CYCH",
        "device": str(device),
        "seed": args.seed,
        "segmentation": {
            "model": "C:/Users/udoo_w2/Desktop/work-kidney-tumor/kidney-tumor/best.pt",
            "task": "segment",
            "input_manifest": "yolo_crops/manifest.csv",
            "roi_source_counts": dict(Counter(record["roi_source"] for record in records)),
            "segmented_crops": len(records),
        },
        "results": results,
    }, indent=2), encoding="utf-8")
    with (OUTPUT / "augmentation_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = ["policy", "level", "AUROC", "AUPRC", "accuracy", "precision", "F1", "TPR", "TNR", "n", "positive", "negative", "TP", "TN", "FP", "FN", "best_epoch"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for level, metric_set in (("image", result["image_metrics"]), ("patient", result["patient_metrics"])):
                row = {"policy": result["policy"], "level": level, "best_epoch": result["best_epoch"]}
                row.update(metric_set)
                writer.writerow(row)
    with (OUTPUT / "augmentation_training_history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = sorted({key for row in histories for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(histories)
    report_lines = [
        "# CYCH augmentation experiment",
        "",
        "The test set is fixed by patient before training and is used only once for the final comparison. Results are baselines on the current labelled CYCH layer, not the exact 2022 paper result.",
        "",
        f"- Segmented CYCH kidney crops: {len(records)}",
        f"- Train/validation/test images: {len(inner_train_idx)}/{len(val_idx)}/{len(test_idx)}",
        f"- Test patients: {len({records[int(index)]['patient_id'] for index in test_idx})}",
        "- Segmentation: existing YOLO kidney segmentation model; 309 YOLO ROIs and 44 physician-polygon fallbacks.",
        "- Image source: `yolo_crops/CYCH`; no unsegmented `original_png` image is passed to training.",
        "- Backbone: pretrained torchvision ResNet50; layer4 and classifier fine-tuned.",
        "- Selection: mean(validation AUROC, validation AUPRC); test untouched.",
        "",
        "| Policy | Level | AUROC | AUPRC | Accuracy | TPR | TNR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for level, metric_set in (("image", result["image_metrics"]), ("patient", result["patient_metrics"])):
            report_lines.append(f"| {result['policy']} | {level} | {metric_set['AUROC']:.4f} | {metric_set['AUPRC']:.4f} | {metric_set['accuracy']:.4f} | {metric_set['TPR']:.4f} | {metric_set['TNR']:.4f} |")
    (OUTPUT / "augmentation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
