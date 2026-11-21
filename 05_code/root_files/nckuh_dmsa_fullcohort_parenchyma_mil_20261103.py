"""Mask-constrained RadImageNet local-patch MIL on the full NCKUH cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

import nckuh_dmsa_fullcohort_fullframe_20261103 as F


ROOT = F.ROOT
BASE = F.OUT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_parenchyma_mil_20261103"
RAD_SOURCE = ROOT / "02_experiments/CYCH_feature_research_results_20260913/source"
SEEDS = F.SEEDS
EPOCHS = (10, 20, 40, 80)
ARMS = ("mean_pool", "gated_attention", "gated_bn_auxiliary")
GROUPS = ARMS + tuple(f"{name}_plus_fullcohort_instance_reference" for name in ARMS)
REFERENCE = "fullcohort_rad_color_image_plus_patient_direct"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def cyan_overlay_mask(rgb: np.ndarray) -> np.ndarray:
    """Conservative cyan UI-color detector; it does not modify image intensities."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = (hsv[..., index] for index in range(3))
    return (hue >= 78) & (hue <= 102) & (saturation >= 100) & (value >= 140)


def grid_coverage(binary: np.ndarray, grid: int = 7) -> np.ndarray:
    size = binary.shape[0]
    if binary.shape != (size, size) or size % grid:
        raise ValueError("Expected square mask divisible by grid")
    step = size // grid
    return binary.reshape(grid, step, grid, step).mean((1, 3))


class ParenchymaMIL(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        if mode not in ("mean", "gated"):
            raise ValueError(mode)
        self.mode = mode
        self.projection = nn.Linear(32, 32)
        self.dropout = nn.Dropout(.25)
        if mode == "mean":
            self.second = nn.Linear(32, 32)
        else:
            self.v = nn.Linear(32, 16)
            self.u = nn.Linear(32, 16)
            self.w = nn.Linear(16, 1, bias=False)
        self.classifier = nn.Linear(32, 1)
        self.extreme_classifier = nn.Linear(32, 1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor):
        hidden = self.dropout(torch.relu(self.projection(values)))
        if self.mode == "mean":
            hidden = torch.relu(self.second(hidden))
            attention = mask.float() / mask.sum(1, keepdim=True)
        else:
            raw = self.w(torch.tanh(self.v(hidden)) * torch.sigmoid(self.u(hidden))).squeeze(-1)
            attention = torch.softmax(raw.masked_fill(~mask, -torch.inf), dim=1)
        pooled = (attention.unsqueeze(-1) * hidden).sum(1)
        return self.classifier(pooled).squeeze(-1), attention, pooled


def load_reference():
    sys.path.insert(0, str(Path(__file__).parent))
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference as load

    patients, kidneys, images = load()
    images = images[images.kidney_APN.notna()].copy().reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].copy().reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].copy().reset_index(drop=True)
    assert (len(patients), len(kidneys), len(images)) == (210, 414, 820)
    return patients, kidneys, images


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered parenchyma-MIL protocol")
    inputs = [
        Path(__file__),
        BASE / "patient_assignments.csv",
        BASE / "deidentified_extraction_qc.csv",
        BASE / "deep_encoder_audit.json",
        BASE / "instance_development_predictions.csv",
        ROOT / "02_experiments/NCKUH_DMSA_label_audit_20261102/image_labels_dmsa_private.csv",
    ]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Whole-kidney pooling dilutes focal parenchymal APN signal; local medical-domain feature maps with kidney-bag attention can preserve it",
        "cohort": "210 patients,414 imaged kidneys,820 contour-first images; exact first three126/42/42 patient roles",
        "encoder": "Frozen RadImageNet DenseNet-121 final convolutional map,7x7x1024, input scaled to[-1,1]",
        "instances": "Feature-map cells with kidney-mask coverage>=0.30 and conservative cyan-overlay coverage<=0.05; image intensities are not inpainted or recolored",
        "cyan_rule": "OpenCV HSV hue78-102,saturation>=100,value>=140; intended only as a conservative UI-color exclusion sensitivity",
        "preprocessing": "L2 normalize local vectors; sample64 patches per fitting patient; fit whitened randomized PCA32 only on fitting patients; cap each kidney bag at128 deterministic patches",
        "models": {
            "mean_pool": "Capacity control with uniform pooling",
            "gated_attention": "Ilse-style gated attention",
            "gated_bn_auxiliary": "Same attention model plus weight0.30 patient-level B-versus-N BCE on mean left/right kidney embeddings",
        },
        "selection": "Within each126-patient training role, fixed100/26 R/L/B/N-stratified inner split chooses10/20/40/80 epochs by mean AUROC and AP; tie earliest; reset and refit all126",
        "evaluation": "Outer42 development patients only; kidney scoring; exploratory-test roles remain unscored",
        "fusion": "Fixed equal average with the completed full-cohort RadImageNet image-instance plus patient-direct reference",
        "interpretation": "Attention is contribution weighting, not lesion or anatomical-compartment truth",
        "target": "Development AUROC>0.8 and AP>0.8 followed by locked confirmation",
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs},
    })
    print("Registered full-cohort parenchyma MIL before feature extraction", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for path, expected in lock["inputs"].items():
        assert digest(Path(path)) == expected, path


def cache() -> None:
    require_lock()
    if (OUT / "radimagenet_local_float16.npy").exists():
        raise FileExistsError("Preserve completed local feature cache")
    sys.path.insert(0, str(RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    _, _, images = load_reference()
    model = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    flat, offsets, records = [], [0], []
    torch.set_num_threads(4)
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            canvases, kidneys, artifacts = [], [], []
            for identifier in images.image_id.iloc[start:start+32]:
                bgr = cv2.imread(str(BASE / "clean_rgb" / f"{identifier}.png"), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(BASE / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE) > 0
                canvas, canvas_mask = F.tight_canvas(rgb, mask)
                artifact = cyan_overlay_mask(canvas) & canvas_mask.astype(bool)
                canvases.append(canvas)
                kidneys.append(canvas_mask.astype(bool))
                artifacts.append(artifact)
            tensor = torch.from_numpy(np.stack(canvases).transpose(0, 3, 1, 2).copy()).cuda().float()
            fmap = torch.relu(model.features(tensor / 127.5 - 1)).permute(0, 2, 3, 1).cpu().numpy()
            assert fmap.shape[1:] == (7, 7, 1024)
            for local, identifier in enumerate(images.image_id.iloc[start:start+32]):
                kidney_fraction = grid_coverage(kidneys[local])
                artifact_fraction = grid_coverage(artifacts[local])
                selected = (kidney_fraction >= .30) & (artifact_fraction <= .05)
                if not selected.any():
                    selected.flat[np.argmax(kidney_fraction - artifact_fraction)] = True
                values = fmap[local][selected].astype(np.float16)
                flat.append(values)
                offsets.append(offsets[-1] + len(values))
                records.append({
                    "image_id": identifier,
                    "selected_patches": len(values),
                    "kidney_cells": int((kidney_fraction >= .30).sum()),
                    "excluded_cyan_cells": int(((kidney_fraction >= .30) & (artifact_fraction > .05)).sum()),
                    "cyan_pixel_fraction_inside_kidney": float(artifacts[local].sum() / max(kidneys[local].sum(), 1)),
                })
            if start % 160 == 0:
                print(f"RadImageNet local maps {min(start+32, len(images))}/{len(images)}", flush=True)
    features = np.concatenate(flat)
    np.save(OUT / "radimagenet_local_float16.npy", features)
    np.save(OUT / "patch_offsets.npy", np.asarray(offsets, np.int64))
    audit = pd.DataFrame(records)
    audit.to_csv(OUT / "deidentified_patch_qc.csv", index=False)
    write_json(OUT / "patch_cache_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "images": len(images),
        "patches": len(features),
        "shape": list(features.shape),
        "selected_patch_min_median_max": [int(audit.selected_patches.min()), float(audit.selected_patches.median()), int(audit.selected_patches.max())],
        "images_with_excluded_cyan_cells": int((audit.excluded_cyan_cells > 0).sum()),
        "excluded_cyan_cells": int(audit.excluded_cyan_cells.sum()),
        "labels_used_in_encoding": [],
        "feature_sha256": digest(OUT / "radimagenet_local_float16.npy"),
        "offset_sha256": digest(OUT / "patch_offsets.npy"),
    })
    print(f"Cached {len(features)} mask-constrained RadImageNet local patches", flush=True)


def patch_data(images: pd.DataFrame, kidneys: pd.DataFrame):
    raw = np.load(OUT / "radimagenet_local_float16.npy", mmap_mode="r")
    offsets = np.load(OUT / "patch_offsets.npy")
    assert len(offsets) == len(images) + 1 and offsets[-1] == len(raw)
    values = np.asarray(raw, dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True).clip(1e-8)
    image_index = np.repeat(np.arange(len(images)), np.diff(offsets))
    patch_patients = images.patient_alias.to_numpy()[image_index]
    patch_kidneys = images.kidney_alias.to_numpy()[image_index]
    groups = [np.flatnonzero(patch_kidneys == alias) for alias in kidneys.kidney_alias]
    assert all(len(group) for group in groups)
    return values, patch_patients, groups


def prepare_bags(values, patch_patients, groups, train_aliases, seed):
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed + 7000)
    sample = []
    for alias in sorted(train_aliases):
        available = np.flatnonzero(patch_patients == alias)
        if not len(available):
            raise ValueError(f"No training patch for {alias}")
        sample.extend(rng.choice(available, 64, replace=len(available) < 64))
    sample = np.asarray(sample, np.int64)
    pca = PCA(32, whiten=True, random_state=seed, svd_solver="randomized").fit(values[sample])
    transformed = pca.transform(values).astype(np.float32)
    compact = []
    for group in groups:
        if len(group) > 128:
            group = group[np.linspace(0, len(group)-1, 128).round().astype(int)]
        compact.append(group)
    width = max(map(len, compact))
    bags = np.zeros((len(compact), width, 32), np.float32)
    mask = np.zeros((len(compact), width), bool)
    for index, group in enumerate(compact):
        bags[index, :len(group)] = transformed[group]
        mask[index, :len(group)] = True
    return torch.from_numpy(bags).cuda(), torch.from_numpy(mask).cuda(), {"pca": pca, "sample": sample}


def extreme_groups(fit: np.ndarray, kidneys: pd.DataFrame):
    groups, targets = [], []
    fit_set = set(fit.tolist())
    for _, rows in kidneys.groupby("patient_alias", sort=False):
        indices = [index for index in rows.index if index in fit_set]
        state = rows.DMSA_APN.iloc[0]
        if indices and state in ("B", "N"):
            groups.append(indices)
            targets.append(float(state == "B"))
    return groups, torch.tensor(targets, dtype=torch.float32, device="cuda")


def train_model(arm, seed, bags, mask, labels, fit, kidneys, epochs, evaluation):
    torch.manual_seed(seed + 37)
    mode = "mean" if arm == "mean_pool" else "gated"
    model = ParenchymaMIL(mode).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    target = labels[fit]
    weights = torch.where(target > 0, len(target)/(2*target.sum()), len(target)/(2*(len(target)-target.sum())))
    groups, extreme_target = extreme_groups(fit, kidneys)
    history = {}
    for epoch in range(1, max(epochs)+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _, pooled = model(bags[fit], mask[fit])
        loss = (nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).mean()
        if arm == "gated_bn_auxiliary" and groups:
            local = {global_index: local_index for local_index, global_index in enumerate(fit)}
            patient_embedding = torch.stack([pooled[[local[index] for index in group]].mean(0) for group in groups])
            extreme_logits = model.extreme_classifier(patient_embedding).squeeze(-1)
            extreme_weights = torch.where(extreme_target > 0, len(extreme_target)/(2*extreme_target.sum()),
                                          len(extreme_target)/(2*(len(extreme_target)-extreme_target.sum())))
            extreme_loss = (nn.functional.binary_cross_entropy_with_logits(
                extreme_logits, extreme_target, reduction="none") * extreme_weights).mean()
            loss = loss + .30 * extreme_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if epoch in epochs and evaluation is not None:
            model.eval()
            with torch.inference_mode():
                history[epoch] = torch.sigmoid(model(bags[evaluation], mask[evaluation])[0]).cpu().numpy()
    return model, history


def metric(labels, scores):
    return F.metric(np.asarray(labels), np.asarray(scores))


def run() -> None:
    from sklearn.model_selection import train_test_split

    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed parenchyma-MIL predictions")
    patients, kidneys, images = load_reference()
    values, patch_patients, groups = patch_data(images, kidneys)
    labels = torch.from_numpy(kidneys.kidney_APN.to_numpy(np.float32)).cuda()
    assignments = pd.read_csv(BASE / "patient_assignments.csv")
    reference = pd.read_csv(BASE / "instance_development_predictions.csv")
    reference = reference[reference.group.eq(REFERENCE)].set_index(["seed", "kidney_alias"]).score
    records, predictions, selections = [], [], []
    (OUT / "models").mkdir(exist_ok=True)
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        outer_train = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("train"))
        development = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        train_patients = patients[patients.patient_alias.isin(mapping[mapping.eq("train")].index)].reset_index(drop=True)
        fit_patient, select_patient = train_test_split(
            np.arange(len(train_patients)), train_size=100, random_state=seed+4000,
            stratify=train_patients.DMSA_APN)
        fit_alias = train_patients.patient_alias.iloc[fit_patient].to_numpy()
        select_alias = train_patients.patient_alias.iloc[select_patient].to_numpy()
        fit = np.flatnonzero(kidneys.patient_alias.isin(fit_alias))
        select = np.flatnonzero(kidneys.patient_alias.isin(select_alias))
        inner_bags, inner_mask, inner_proof = prepare_bags(values, patch_patients, groups, fit_alias, seed)
        chosen_epochs = {}
        for arm in ARMS:
            _, history = train_model(arm, seed, inner_bags, inner_mask, labels, fit, kidneys, EPOCHS, select)
            choices = []
            for epoch, score in history.items():
                result = metric(kidneys.kidney_APN.to_numpy(int)[select], score)
                choices.append((-(result["auroc"]+result["ap"])/2, epoch, result))
            _, chosen, result = min(choices, key=lambda item: (item[0], item[1]))
            chosen_epochs[arm] = chosen
            selections.append({"seed": seed, "arm": arm, "selected_epoch": chosen,
                               "inner_auroc": result["auroc"], "inner_ap": result["ap"]})
        del inner_bags, inner_mask
        torch.cuda.empty_cache()

        train_alias = mapping[mapping.eq("train")].index.to_numpy()
        bags, mask, full_proof = prepare_bags(values, patch_patients, groups, train_alias, seed)
        joblib.dump({"inner": inner_proof, "full": full_proof}, OUT / "models" / f"{seed}_preprocess.joblib", compress=3)
        scores = {}
        for arm in ARMS:
            model, _ = train_model(arm, seed, bags, mask, labels, outer_train, kidneys, (chosen_epochs[arm],), None)
            model.eval()
            with torch.inference_mode():
                logits, attention, _ = model(bags[development], mask[development])
                scores[arm] = torch.sigmoid(logits).cpu().numpy()
            torch.save({"state_dict": model.state_dict(), "arm": arm, "epoch": chosen_epochs[arm], "seed": seed},
                       OUT / "models" / f"{seed}_{arm}.pth")
            np.savez_compressed(OUT / f"{seed}_{arm}_development_attention.npz",
                                attention=attention.cpu().numpy(), mask=mask[development].cpu().numpy())
        ref = np.asarray([reference.loc[(seed, alias)] for alias in kidneys.kidney_alias.iloc[development]])
        for arm in ARMS:
            scores[f"{arm}_plus_fullcohort_instance_reference"] = (scores[arm] + ref) / 2
        y = kidneys.kidney_APN.to_numpy(int)
        for group in GROUPS:
            result = metric(y[development], scores[group])
            records.append({"seed": seed, "role": "development", "group": group,
                            "patients": 42, "kidneys": len(development), **result})
            for local, index in enumerate(development):
                predictions.append({
                    "seed": seed,
                    "group": group,
                    "patient_alias": kidneys.patient_alias.iloc[index],
                    "kidney_alias": kidneys.kidney_alias.iloc[index],
                    "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                    "side": kidneys.side.iloc[index],
                    "label": int(y[index]),
                    "score": float(scores[group][local]),
                })
        print(f"Completed parenchyma-MIL draw {seed}", flush=True)
        del bags, mask
        torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "inner_epoch_selections.csv", index=False)
    summary = pd.DataFrame(records).groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc_mean > .8) & (summary.ap_mean > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {
        "groups_above_both_development_targets": achieved,
        "exploratory_test_scored": False,
        "selection_unit": "kidney",
        "split_unit": "patient",
    })
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "cache", "run"))
    args = parser.parse_args()
    {"prepare": prepare, "cache": cache, "run": run}[args.stage]()
