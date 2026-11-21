"""Controlled RadImageNet tail fine-tuning for full-cohort kidney-level APN."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

import nckuh_dmsa_fullcohort_fullframe_20261103 as F


ROOT = F.ROOT
BASE = F.OUT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_radimagenet_finetune_20261103"
RAD_SOURCE = ROOT / "02_experiments/CYCH_feature_research_results_20260913/source"
SEEDS = F.SEEDS
EPOCHS = (2, 5, 10, 20)
ARMS = {
    "tail_ce_noaug": {"augment": False, "l2sp": False},
    "tail_ce_aug": {"augment": True, "l2sp": False},
    "tail_ce_aug_l2sp": {"augment": True, "l2sp": True},
}
REFERENCE = "fullcohort_rad_color_image_plus_patient_direct"
GROUPS = tuple(ARMS) + tuple(f"{name}_plus_instance_reference" for name in ARMS)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_reference():
    sys.path.insert(0, str(Path(__file__).parent))
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference as load

    patients, kidneys, images = load()
    images = images[images.kidney_APN.notna()].copy().reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].copy().reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].copy().reset_index(drop=True)
    assert (len(patients), len(kidneys), len(images)) == (210, 414, 820)
    return patients, kidneys, images


def split_encoder(model):
    prefix = nn.Sequential(OrderedDict((name, module) for name, module in model.features.named_children()
                                       if name not in ("denseblock4", "norm5")))
    tail = nn.Sequential(OrderedDict(
        denseblock4=model.features.denseblock4,
        norm5=model.features.norm5,
        relu=nn.ReLU(),
        pool=nn.AdaptiveAvgPool2d(1),
        flatten=nn.Flatten(),
    ))
    return prefix, tail


class TailClassifier(nn.Module):
    def __init__(self, tail: nn.Module):
        super().__init__()
        self.tail = tail
        self.classifier = nn.Linear(1024, 1)

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.tail(maps)).squeeze(-1)


def feature_augment(maps: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    output = maps.clone()
    flips = torch.rand(len(output), generator=generator, device="cpu") < .5
    if flips.any():
        output[flips.to(output.device)] = torch.flip(output[flips.to(output.device)], dims=(-1,))
    keep = (torch.rand((len(output), 1, 7, 7), generator=generator, device="cpu") >= .10).to(output.device)
    output = output * keep
    noise = torch.randn(output.shape, generator=generator, device="cpu", dtype=output.dtype).to(output.device)
    return output + .02 * noise


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered fine-tuning protocol")
    inputs = [
        Path(__file__),
        BASE / "patient_assignments.csv",
        BASE / "deidentified_extraction_qc.csv",
        BASE / "instance_development_predictions.csv",
        ROOT / "02_experiments/NCKUH_DMSA_label_audit_20261102/image_labels_dmsa_private.csv",
    ]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "motivation": "Frozen RadImageNet was the strongest transferable full-cohort encoder; test restricted medical-pretrained tail adaptation rather than a new backbone",
        "cohort": "210 patients,414 kidneys,820 contour-first full-frame images; exact first three126/42/42 patient roles",
        "initialization": "Existing verified RadImageNet DenseNet-121; prefix through transition3 frozen and cached; denseblock4,norm5 affine and new binary head trainable",
        "labels": "Source DMSA kidney label inherited only by images of that same kidney; patient-balanced and class-balanced image loss",
        "arms": ARMS,
        "augmentation": "Training feature maps only: horizontal flip probability0.5, independent10% spatial-cell dropout, Gaussian noise SD0.02; no synthetic diagnosis images or GAN",
        "optimization": "AdamW head lr0.001, tail lr0.00001, gradient norm5; optional L2-SP weight0.0001 to original pretrained tail; BN running buffers fixed",
        "selection": "Fixed100/26 R/L/B/N-stratified split inside each126-patient training role; choose2/5/10/20 epochs by kidney-level mean(AUROC,AP), tie earliest; reset/refit all126",
        "evaluation": "Aggregate image logits within kidney; outer42 development patients only; exploratory test unscored",
        "fusion": "Fixed equal average with completed full-cohort image-instance plus patient-direct reference",
        "target": "Development AUROC>0.8 and AP>0.8 followed by locked confirmation",
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs},
    })
    print("Registered controlled full-cohort RadImageNet tail adaptation", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for path, expected in lock["inputs"].items():
        assert digest(Path(path)) == expected, path


def cache() -> None:
    require_lock()
    if (OUT / "prefix_maps_float16.npy").exists():
        raise FileExistsError("Preserve completed prefix cache")
    sys.path.insert(0, str(RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121, RADIMAGENET_DIR

    _, _, images = load_reference()
    model = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    prefix, tail = split_encoder(model)
    maps, embeddings = [], []
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            canvases = []
            for identifier in images.image_id.iloc[start:start+32]:
                bgr = cv2.imread(str(BASE / "clean_rgb" / f"{identifier}.png"), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(BASE / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE) > 0
                canvas, _ = F.tight_canvas(rgb, mask)
                canvases.append(canvas)
            tensor = torch.from_numpy(np.stack(canvases).transpose(0, 3, 1, 2).copy()).cuda().float()/127.5-1
            current = prefix(tensor)
            maps.append(current.cpu().numpy().astype(np.float16))
            embeddings.append(tail(current).cpu().numpy())
    maps = np.concatenate(maps)
    embeddings = np.concatenate(embeddings)
    expected = np.load(BASE / "radimagenet_color.npy")
    assert maps.shape == (820, 512, 7, 7)
    error = float(np.max(np.abs(embeddings - expected)))
    assert np.allclose(embeddings, expected, rtol=1e-4, atol=1e-4)
    np.save(OUT / "prefix_maps_float16.npy", maps)
    write_json(OUT / "cache_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "images": 820,
        "shape": list(maps.shape),
        "prefix_tail_embedding_maxabs_vs_frozen_cache": error,
        "pretrained_weight_sha256": digest(RADIMAGENET_DIR / "DenseNet121.pt"),
        "labels_used_in_encoding": [],
        "prefix_sha256": digest(OUT / "prefix_maps_float16.npy"),
    })
    print("Cached 820 frozen RadImageNet prefix maps", flush=True)


def image_weights(images: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    subset = images.iloc[indices]
    counts = subset.patient_alias.value_counts()
    weights = 1.0 / subset.patient_alias.map(counts).to_numpy(float)
    labels = subset.kidney_APN.to_numpy(int)
    for value in (0, 1):
        chosen = labels == value
        weights[chosen] *= .5 / weights[chosen].sum()
    return weights / weights.mean()


def aggregate(scores: np.ndarray, images: pd.DataFrame, kidney_aliases: pd.Series) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1-1e-6)
    logits = np.log(clipped/(1-clipped))
    means = pd.Series(logits, index=images.kidney_alias).groupby(level=0).mean()
    values = means.reindex(kidney_aliases).to_numpy(float)
    return 1/(1+np.exp(-np.clip(values, -30, 30)))


def build_model(seed: int):
    sys.path.insert(0, str(RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    torch.manual_seed(seed + 37)
    original = radimagenet_densenet121().cuda().eval()
    _, original_tail = split_encoder(original)
    model = TailClassifier(copy.deepcopy(original_tail)).cuda()
    anchor = {name: parameter.detach().clone() for name, parameter in original_tail.named_parameters()}
    optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": .001, "weight_decay": .0001},
        {"params": model.tail.parameters(), "lr": .00001, "weight_decay": 0.0},
    ])
    return model, anchor, optimizer


def predict_images(model, maps, indices, batch_size=64):
    output = np.full(len(maps), np.nan, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start+batch_size]
            tensor = torch.from_numpy(np.asarray(maps[selected], dtype=np.float32)).cuda()
            output[selected] = torch.sigmoid(model(tensor)).cpu().numpy()
    return output


def train(arm, seed, maps, images, train_indices, epochs, evaluation_indices=None):
    settings = ARMS[arm]
    model, anchor, optimizer = build_model(seed)
    labels = torch.from_numpy(images.kidney_APN.to_numpy(np.float32)).cuda()
    weights = torch.from_numpy(image_weights(images, train_indices).astype(np.float32)).cuda()
    generator = torch.Generator(device="cpu").manual_seed(seed + 9000)
    rng = np.random.default_rng(seed + 9100)
    history = {}
    for epoch in range(1, max(epochs)+1):
        order = rng.permutation(len(train_indices))
        model.train()
        model.tail.eval()  # keep all pretrained BatchNorm running buffers fixed
        for start in range(0, len(order), 32):
            local = order[start:start+32]
            selected = train_indices[local]
            tensor = torch.from_numpy(np.asarray(maps[selected], dtype=np.float32)).cuda()
            if settings["augment"]:
                tensor = feature_augment(tensor, generator)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tensor)
            loss = (nn.functional.binary_cross_entropy_with_logits(
                logits, labels[selected], reduction="none") * weights[local]).mean()
            if settings["l2sp"]:
                penalty = sum((parameter - anchor[name]).pow(2).sum()
                              for name, parameter in model.tail.named_parameters())
                loss = loss + .0001 * penalty
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        if epoch in epochs and evaluation_indices is not None:
            score = predict_images(model, maps, evaluation_indices)
            history[epoch] = score
    return model, history


def run() -> None:
    from sklearn.model_selection import train_test_split

    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed fine-tuning predictions")
    patients, kidneys, images = load_reference()
    maps = np.load(OUT / "prefix_maps_float16.npy", mmap_mode="r")
    assignments = pd.read_csv(BASE / "patient_assignments.csv")
    reference = pd.read_csv(BASE / "instance_development_predictions.csv")
    reference = reference[reference.group.eq(REFERENCE)].set_index(["seed", "kidney_alias"]).score
    records, predictions, selections = [], [], []
    (OUT / "models").mkdir(exist_ok=True)
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train_patients = patients[patients.patient_alias.isin(mapping[mapping.eq("train")].index)].reset_index(drop=True)
        fit_local, select_local = train_test_split(
            np.arange(len(train_patients)), train_size=100, random_state=seed+4000,
            stratify=train_patients.DMSA_APN)
        fit_alias = train_patients.patient_alias.iloc[fit_local]
        select_alias = train_patients.patient_alias.iloc[select_local]
        fit_images = np.flatnonzero(images.patient_alias.isin(fit_alias))
        select_images = np.flatnonzero(images.patient_alias.isin(select_alias))
        select_kidneys = np.flatnonzero(kidneys.patient_alias.isin(select_alias))
        chosen_epochs = {}
        for arm in ARMS:
            _, history = train(arm, seed, maps, images, fit_images, EPOCHS, select_images)
            choices = []
            for epoch, image_score in history.items():
                kidney_score = aggregate(image_score, images, kidneys.kidney_alias.iloc[select_kidneys])
                result = F.metric(kidneys.kidney_APN.to_numpy(int)[select_kidneys], kidney_score)
                choices.append((-(result["auroc"]+result["ap"])/2, epoch, result))
            _, chosen, result = min(choices, key=lambda item: (item[0], item[1]))
            chosen_epochs[arm] = chosen
            selections.append({"seed": seed, "arm": arm, "selected_epoch": chosen,
                               "inner_auroc": result["auroc"], "inner_ap": result["ap"]})
            torch.cuda.empty_cache()
        train_images = np.flatnonzero(images.patient_alias.map(mapping).eq("train"))
        development_kidneys = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        all_indices = np.arange(len(images))
        scores = {}
        for arm in ARMS:
            model, _ = train(arm, seed, maps, images, train_images, (chosen_epochs[arm],), None)
            image_score = predict_images(model, maps, all_indices)
            scores[arm] = aggregate(image_score, images, kidneys.kidney_alias.iloc[development_kidneys])
            torch.save({"state_dict": model.state_dict(), "arm": arm,
                        "selected_epoch": chosen_epochs[arm], "seed": seed}, OUT / "models" / f"{seed}_{arm}.pth")
            del model
            torch.cuda.empty_cache()
        ref = np.asarray([reference.loc[(seed, alias)] for alias in kidneys.kidney_alias.iloc[development_kidneys]])
        for arm in ARMS:
            scores[f"{arm}_plus_instance_reference"] = (scores[arm] + ref) / 2
        labels = kidneys.kidney_APN.to_numpy(int)
        for group in GROUPS:
            result = F.metric(labels[development_kidneys], scores[group])
            records.append({"seed": seed, "role": "development", "group": group,
                            "patients": 42, "kidneys": len(development_kidneys), **result})
            for local, index in enumerate(development_kidneys):
                predictions.append({
                    "seed": seed, "group": group,
                    "patient_alias": kidneys.patient_alias.iloc[index],
                    "kidney_alias": kidneys.kidney_alias.iloc[index],
                    "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                    "side": kidneys.side.iloc[index],
                    "label": int(labels[index]),
                    "score": float(scores[group][local]),
                })
        print(f"Completed RadImageNet fine-tuning draw {seed}", flush=True)
    pd.DataFrame(records).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "inner_epoch_selections.csv", index=False)
    summary = pd.DataFrame(records).groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    write_json(OUT / "development_decision.json", {
        "groups_above_both_development_targets": summary[(summary.auroc_mean > .8) & (summary.ap_mean > .8)].index.tolist(),
        "exploratory_test_scored": False,
    })
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "cache", "run"))
    args = parser.parse_args()
    {"prepare": prepare, "cache": cache, "run": run}[args.stage]()
