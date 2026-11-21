"""End-to-end RadImageNet multi-view MIL on clean segmented NCKUH kidneys."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

import nckuh_dmsa_fullframe_clean_probe_20261102 as F
import nckuh_dmsa_fullframe_deep_probe_20261102 as D
import nckuh_dmsa_fullframe_image_instance_probe_20261102 as I
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullframe_radimagenet_mil_20261102"
REFERENCE = "deep_concat_plus_clean_radiomics"
GROUPS = (
    "radimagenet_lastblock_probability_mean_mil",
    "radimagenet_lastblock_logit_mean_mil",
    "radimagenet_probability_mean_mil_plus_reference",
    "radimagenet_logit_mean_mil_plus_reference",
)
MAX_EPOCHS = 30
MIN_EPOCHS = 8
PATIENCE = 6


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered RadImageNet MIL protocol")
    F.require_lock()
    D.require_lock()
    inputs = [
        F.OUT / "patient_assignments.csv",
        F.OUT / "deidentified_image_manifest.csv",
        F.OUT / "archive_extraction_audit.json",
        D.OUT / "development_predictions.csv",
        D.RAD_WEIGHT,
        Path(__file__),
    ]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same clean-full-frame 111 patients, 212 kidneys, 418 images and exact 70/20/21 patient roles",
        "method_source": "Yin et al., Urology 2020, doi:10.1016/j.urology.2020.05.019; multi-view instance-level CNN fine-tuning followed by mean bag pooling",
        "source_code": "https://github.com/YS181818/CAKUT_diagnosis",
        "adaptation": "One imaged kidney is one bag and its longitudinal/transverse clean segmented ROI images are instances; DMSA side truth supplies the bag label",
        "encoder": "Official local RadImageNet DenseNet121; freeze through transition3 and fine-tune denseblock4, norm5 and a new binary head",
        "augmentation": "Training-only random rotation up to5 degrees, translation2 percent, scale0.95-1.05, brightness/contrast0.10; no flips, GAN or synthetic lesions",
        "loss": "Image-instance BCE with every kidney assigned total weight one before binary class balancing",
        "selection": "Within each fixed70-patient training role, stratified55/15 patient inner split; earliest best mean inner kidney AUROC/AP, minimum8, maximum30, patience6; refit all70 patients for selected epochs",
        "pooling": "Prespecified arithmetic mean of image probabilities and sigmoid of arithmetic image logits",
        "fusion": "Prespecified equal-probability averages with the completed kidney-level deep-radiomics reference",
        "optimization": "AdamW, denseblock4/norm5 lr1e-5, binary head lr3e-4, weight decay1e-4, batch32, bfloat16 autocast",
        "prohibited": "No development labels for epoch selection, no diagnosis filename suffix, no DMSA code as predictor, no exploratory-test scoring or weight selection",
        "target": "Development AUROC>0.8 and AP>0.8; later locked confirmation remains necessary",
        "groups": list(GROUPS),
        "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "source_sha256": digest(__file__),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs},
    })
    print("Registered end-to-end RadImageNet multi-view MIL", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(Path(name)) == expected, name


def load_canvases(images: pd.DataFrame) -> np.ndarray:
    canvases = []
    for ident in images.image_id:
        color = cv2.imread(str(F.OUT / "clean_rgb" / f"{ident}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(F.OUT / "masks" / f"{ident}.png"), cv2.IMREAD_GRAYSCALE) > 0
        if color is None or mask is None or not mask.any():
            raise ValueError(f"Invalid clean ROI: {ident}")
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        canvas, _ = D.tight_canvas(color, mask)
        canvases.append(canvas)
    output = np.stack(canvases).astype(np.uint8)
    assert output.shape == (418, 224, 224, 3)
    return output


def build_model():
    import torch
    sys.path.insert(0, str(D.RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    model = radimagenet_densenet121()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name in ("denseblock4", "norm5"):
        for parameter in getattr(model.features, name).parameters():
            parameter.requires_grad_(True)
    model.classifier = torch.nn.Linear(1024, 1)
    return model


def trainable_parameter_names(model) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def set_training_mode(model) -> None:
    model.train()
    for name, module in model.features.named_children():
        if name not in {"denseblock4", "norm5"}:
            module.eval()


def transforms(training: bool):
    from torchvision import transforms as T
    items = [T.ToPILImage()]
    if training:
        items.extend([
            T.RandomAffine(degrees=5, translate=(.02, .02), scale=(.95, 1.05), fill=0),
            T.ColorJitter(brightness=.10, contrast=.10),
        ])
    items.extend([T.ToTensor(), T.Normalize([.5, .5, .5], [.5, .5, .5])])
    return T.Compose(items)


class ImageDataset:
    def __init__(self, canvases, indices, labels, weights, training):
        self.canvases = canvases
        self.indices = np.asarray(indices)
        self.labels = labels
        self.weights = weights
        self.transform = transforms(training)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local):
        index = self.indices[local]
        return self.transform(self.canvases[index]), np.float32(self.labels[index]), np.float32(self.weights[local])


def predict_images(model, canvases: np.ndarray, batch_size: int = 64) -> np.ndarray:
    import torch
    transform = transforms(False)
    scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(canvases), batch_size):
            tensor = torch.stack([transform(image) for image in canvases[start:start + batch_size]]).cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(tensor).flatten()
            scores.extend(torch.sigmoid(logits.float()).cpu().numpy())
    return np.asarray(scores)


def train_epoch(model, loader, optimizer) -> float:
    import torch
    import torch.nn.functional as functional
    set_training_mode(model)
    total_loss, total_weight = 0.0, 0.0
    for tensor, target, weight in loader:
        tensor, target, weight = tensor.cuda(), target.cuda(), weight.cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(tensor).flatten()
            losses = functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
            loss = (losses * weight).sum() / weight.sum()
        loss.backward()
        optimizer.step()
        total_loss += float((losses.detach() * weight).sum().cpu())
        total_weight += float(weight.sum().cpu())
    return total_loss / total_weight


def fit_epochs(canvases, images, train_images, image_labels, epochs, seed):
    import torch
    from torch.utils.data import DataLoader
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model().cuda()
    weights = I.balanced_image_weights(images.kidney_alias.to_numpy()[train_images], image_labels[train_images])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(ImageDataset(canvases, train_images, image_labels, weights, True), batch_size=32,
                        shuffle=True, num_workers=0, generator=generator)
    tail = [parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("classifier")]
    optimizer = torch.optim.AdamW([
        {"params": tail, "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 3e-4},
    ], weight_decay=1e-4)
    losses = []
    for _ in range(epochs):
        losses.append(train_epoch(model, loader, optimizer))
    return model, losses


def select_epoch(canvases, patients, kidneys, images, outer_train, image_labels, kidney_labels, seed):
    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    inner_train, inner_validation = train_test_split(
        outer_train, train_size=55, random_state=seed + 7000,
        stratify=patients.DMSA_APN.to_numpy()[outer_train])
    inner_train_aliases = set(patients.patient_alias.iloc[inner_train])
    inner_validation_aliases = set(patients.patient_alias.iloc[inner_validation])
    train_images = np.flatnonzero(images.patient_alias.isin(inner_train_aliases))
    validation_kidneys = np.flatnonzero(kidneys.patient_alias.isin(inner_validation_aliases))
    weights = I.balanced_image_weights(images.kidney_alias.to_numpy()[train_images], image_labels[train_images])
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model().cuda()
    loader = DataLoader(ImageDataset(canvases, train_images, image_labels, weights, True), batch_size=32,
                        shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
    tail = [parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("classifier")]
    optimizer = torch.optim.AdamW([
        {"params": tail, "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 3e-4},
    ], weight_decay=1e-4)
    records, best, stale = [], None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_epoch(model, loader, optimizer)
        image_score = predict_images(model, canvases)
        probability = I.pool_to_kidneys(image_score, images, kidneys, "probability_mean")
        result = metric(kidney_labels[validation_kidneys], probability[validation_kidneys])
        selection = (result["auroc"] + result["ap"]) / 2
        records.append({"epoch": epoch, "loss": loss, "inner_auroc": result["auroc"],
                        "inner_ap": result["ap"], "selection": selection})
        if best is None or selection > best["selection"] + 1e-12:
            best = records[-1].copy()
            stale = 0
        else:
            stale += 1
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    del model
    torch.cuda.empty_cache()
    return int(best["epoch"]), records


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed RadImageNet MIL predictions")
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    patients, kidneys, images = F.matched_reference()
    canvases = load_canvases(images)
    image_labels = images.kidney_APN.to_numpy(int)
    kidney_labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(F.OUT / "patient_assignments.csv")
    reference = pd.read_csv(D.OUT / "development_predictions.csv")
    reference = reference[reference.group.eq(REFERENCE)].set_index(["seed", "kidney_alias"]).score
    metrics, predictions, epoch_records = [], [], []
    (OUT / "models").mkdir(exist_ok=True)
    for seed in F.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        outer_train = np.flatnonzero(patients.patient_alias.map(mapping).eq("train"))
        selected_epoch, inner_records = select_epoch(
            canvases, patients, kidneys, images, outer_train, image_labels, kidney_labels, seed)
        epoch_records.extend({"seed": seed, **row, "selected_epoch": selected_epoch} for row in inner_records)
        train_aliases = set(patients.patient_alias.iloc[outer_train])
        train_images = np.flatnonzero(images.patient_alias.isin(train_aliases))
        model, losses = fit_epochs(canvases, images, train_images, image_labels, selected_epoch, seed + 10000)
        torch.save({"state_dict": model.state_dict(), "selected_epoch": selected_epoch,
                    "trainable": trainable_parameter_names(model)}, OUT / "models" / f"{seed}_radimagenet_mil.pth")
        image_score = predict_images(model, canvases)
        probability = I.pool_to_kidneys(image_score, images, kidneys, "probability_mean")
        logit = I.pool_to_kidneys(image_score, images, kidneys, "logit_mean")
        dev = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        ref = np.asarray([reference.loc[(seed, alias)] for alias in kidneys.kidney_alias.iloc[dev]])
        scores = {
            GROUPS[0]: probability[dev],
            GROUPS[1]: logit[dev],
            GROUPS[2]: (probability[dev] + ref) / 2,
            GROUPS[3]: (logit[dev] + ref) / 2,
        }
        for group, values in scores.items():
            result = metric(kidney_labels[dev], values)
            metrics.append({"seed": seed, "role": "development", "group": group,
                            "patients": 20, "kidneys": len(dev), **result})
            for local, index in enumerate(dev):
                predictions.append({"seed": seed, "group": group,
                                    "patient_alias": kidneys.patient_alias.iloc[index],
                                    "kidney_alias": kidneys.kidney_alias.iloc[index],
                                    "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                                    "side": kidneys.side.iloc[index], "label": int(kidney_labels[index]),
                                    "score": float(values[local])})
        print(f"Completed RadImageNet MIL draw {seed}; selected epoch {selected_epoch}", flush=True)
        del model
        torch.cuda.empty_cache()
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(epoch_records).to_csv(OUT / "inner_epoch_records.csv", index=False)
    summary = frame.groupby("group")[["auroc", "ap"]].mean().sort_values(["auroc", "ap"], ascending=False)
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
    previous = pd.read_csv(D.OUT / "development_predictions.csv")
    reference = previous[previous.group.eq(REFERENCE)]
    ci, paired, samples = cluster_intervals(
        pd.concat([frame, reference], ignore_index=True), list(GROUPS) + [REFERENCE], F.SEEDS, REFERENCE)
    ci.to_csv(OUT / "development_confidence_intervals.csv", index=False)
    paired.to_csv(OUT / "development_paired_intervals.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    write_json(OUT / "completion_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "patients": 111, "kidneys": 212, "images": 418,
        "test_scored": False, "scored_unit": "kidney", "bootstrap_cluster": "patient",
    })
    print("RadImageNet MIL intervals completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "bootstrap"])
    arguments = parser.parse_args()
    globals()[arguments.stage]()
