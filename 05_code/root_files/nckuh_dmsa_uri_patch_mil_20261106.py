"""Weakly supervised URI-CADS local-patch MIL for kidney-level NCKUH APN."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_uri_patch_mil_20261106"
REFERENCE = uri.OUT / "development_predictions.csv"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"
SEEDS = (20261102, 20261103, 20261104)
ARMS = {
    "uri_patch_mean": ("mean", 0.0),
    "uri_patch_attention": ("gated", 0.0),
    "uri_patch_attention_reliable_negative": ("gated", 0.20),
}
EPOCHS = (10, 20, 40, 80)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def tight_mask(mask: np.ndarray, size: int = 800) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    yy, xx = np.nonzero(binary)
    if len(xx) == 0:
        raise ValueError("Empty kidney mask")
    region = binary[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1].astype(np.uint8)
    scale = size / max(region.shape)
    height = max(1, round(region.shape[0] * scale))
    width = max(1, round(region.shape[1] * scale))
    output = np.zeros((size, size), np.uint8)
    top, left = (size - height) // 2, (size - width) // 2
    output[top : top + height, left : left + width] = cv2.resize(
        region, (width, height), interpolation=cv2.INTER_NEAREST
    )
    return output


def patch_geometry(canvas_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selected grid positions, coverage, and reflection-invariant geometry."""
    coverage = cv2.resize(
        canvas_mask.astype(np.float32), (25, 25), interpolation=cv2.INTER_AREA
    )
    selected = coverage >= 0.25
    if not selected.any():
        selected.flat[np.argmax(coverage)] = True
    positions = np.column_stack(np.nonzero(selected)).astype(np.float32)
    weights = coverage[selected].astype(np.float64)
    center = np.average(positions, axis=0, weights=weights)
    centered = positions - center
    covariance = (centered * weights[:, None]).T @ centered / max(weights.sum(), 1e-8)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    projected = centered @ axes
    projected /= np.maximum(np.abs(projected).max(axis=0, keepdims=True), 1.0)
    depth_map = distance_transform_edt(selected)
    depth = depth_map[selected]
    depth /= max(float(depth.max()), 1.0)
    geometry = np.c_[
        np.abs(projected[:, 0]),
        np.abs(projected[:, 1]),
        depth,
        coverage[selected],
    ].astype(np.float32)
    return np.flatnonzero(selected.reshape(-1)), coverage, geometry


class PatchMIL(nn.Module):
    def __init__(self, input_dim: int, mode: str):
        super().__init__()
        if mode not in ("mean", "gated"):
            raise ValueError(mode)
        self.mode = mode
        self.projection = nn.Linear(input_dim, 64)
        self.dropout = nn.Dropout(0.25)
        if mode == "gated":
            self.v = nn.Linear(64, 32)
            self.u = nn.Linear(64, 32)
            self.w = nn.Linear(32, 1, bias=False)
        self.bag_classifier = nn.Linear(64, 1)
        self.patch_classifier = nn.Linear(64, 1)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.dropout(torch.relu(self.projection(values)))
        if self.mode == "mean":
            attention = mask.float() / mask.sum(1, keepdim=True).clamp_min(1)
        else:
            raw = self.w(torch.tanh(self.v(hidden)) * torch.sigmoid(self.u(hidden))).squeeze(-1)
            attention = torch.softmax(raw.masked_fill(~mask, -torch.inf), dim=1)
        pooled = (attention.unsqueeze(-1) * hidden).sum(1)
        bag_logit = self.bag_classifier(pooled).squeeze(-1)
        patch_logit = self.patch_classifier(hidden).squeeze(-1)
        return bag_logit, patch_logit, attention


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve completed or partial experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    _, _, images = cohort.load_reference()
    images[["image_id", "patient_alias", "kidney_alias", "side", "view"]].to_csv(
        OUT / "encoding_manifest.csv", index=False
    )
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Whole-kidney pooling suppresses focal APN-compatible cortical abnormalities; kidney-bag MIL can rank kidneys using localized renal-domain evidence",
        "clinical_features": "Patch geometry adds absolute long-axis position, absolute short-axis position, contour depth, and kidney coverage without assigning an anatomical compartment label",
        "encoder": "Five official frozen URI-CADS ResNet-50 folds",
        "instances": "Final 25x25 feature cells with kidney-mask coverage >=0.25; each instance contains 35 URI-CADS pathology logits plus four contour-relative geometry values",
        "bags": "All longitudinal and transverse patches from one kidney; deterministic maximum 512 instances",
        "arms": {
            "uri_patch_mean": "uniform-pooling capacity control",
            "uri_patch_attention": "Ilse-style gated attention MIL",
            "uri_patch_attention_reliable_negative": "gated MIL plus weight-0.20 patch BCE for APN-negative training kidneys only",
        },
        "selection": "Within each 126-patient outer-training role, a fixed stratified 100/26 patient split selects 10/20/40/80 epochs by mean inner AUROC and AP; refit on all 126",
        "preprocessing": "StandardScaler fitted on a patient-balanced patch sample from training-role patients only",
        "evaluation": "Kidney scoring on the 42 development patients; exploratory-test roles remain unscored",
        "fusion": "Standalone score and fixed 20% or 25% contribution to the unchanged official URI-CADS reference",
        "inputs": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (
                Path(__file__),
                base.OUT / "patient_assignments.csv",
                REFERENCE,
                *uri.MODEL_PATHS,
            )
        },
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(
        OUT / "run_lock.json",
        {
            "protocol_sha256": digest(OUT / "protocol.json"),
            "manifest_sha256": digest(OUT / "encoding_manifest.csv"),
        },
    )


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    if digest(OUT / "protocol.json") != lock["protocol_sha256"]:
        raise ValueError("Protocol changed after registration")
    if digest(OUT / "encoding_manifest.csv") != lock["manifest_sha256"]:
        raise ValueError("Manifest changed after registration")


def cache() -> None:
    require_lock()
    output = OUT / "uri_patch_features_float16.npy"
    if output.exists():
        raise FileExistsError(f"Preserve existing patch cache: {output}")
    _, _, images = cohort.load_reference()
    selections: list[np.ndarray] = []
    geometries: list[np.ndarray] = []
    offsets = [0]
    coverage_counts = []
    for identifier in images.image_id:
        mask = cv2.imread(
            str(base.OUT / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE
        )
        if mask is None:
            raise FileNotFoundError(identifier)
        indices, coverage, geometry = patch_geometry(tight_mask(mask > 0))
        selections.append(indices)
        geometries.append(geometry)
        offsets.append(offsets[-1] + len(indices))
        coverage_counts.append(len(indices))
    features = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float16, shape=(offsets[-1], 39)
    )
    features[:, :4] = np.concatenate(geometries).astype(np.float16)
    mean, std = uri.public_normalization()
    mean_tensor = torch.from_numpy(mean).cuda().view(1, 3, 1, 1)
    std_tensor = torch.from_numpy(std).cuda().view(1, 3, 1, 1)
    fold_audit = []
    for fold, checkpoint in enumerate(uri.MODEL_PATHS, start=1):
        model = uri.OfficialGlobalEncoder(checkpoint).cuda().eval().requires_grad_(False)
        weight = model.classifier.weight.detach()
        bias = model.classifier.bias.detach()
        with torch.inference_mode():
            for start in range(0, len(images), 8):
                canvases = []
                for identifier in images.image_id.iloc[start : start + 8]:
                    bgr = cv2.imread(
                        str(base.OUT / "clean_rgb" / f"{identifier}.png"), cv2.IMREAD_COLOR
                    )
                    mask = cv2.imread(
                        str(base.OUT / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE
                    ) > 0
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    canvas, _ = base.tight_canvas(rgb * mask[..., None], mask, size=800)
                    canvases.append(canvas)
                tensor = torch.from_numpy(
                    np.stack(canvases).transpose(0, 3, 1, 2).copy()
                ).cuda().float() / 255.0
                fmap = model.features((tensor - mean_tensor) / std_tensor)
                logits = torch.einsum("bchw,kc->bhwk", fmap, weight) + bias
                logits = logits.cpu().numpy().reshape(len(canvases), 625, 7)
                for local in range(len(canvases)):
                    index = start + local
                    features[
                        offsets[index] : offsets[index + 1],
                        4 + (fold - 1) * 7 : 4 + fold * 7,
                    ] = logits[local, selections[index]].astype(np.float16)
                if start % 160 == 0:
                    print(
                        f"URI patch logits fold {fold}/5: {min(start + 8, len(images))}/820",
                        flush=True,
                    )
        fold_audit.append(
            {
                "fold": fold,
                "checkpoint": checkpoint.name,
                "sha256": digest(checkpoint),
            }
        )
        del model
        torch.cuda.empty_cache()
    features.flush()
    np.save(OUT / "patch_offsets.npy", np.asarray(offsets, np.int64))
    write_json(
        OUT / "encoder_audit.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "images": len(images),
            "patches": offsets[-1],
            "feature_shape": [offsets[-1], 39],
            "patches_per_image_min_median_max": [
                min(coverage_counts),
                float(np.median(coverage_counts)),
                max(coverage_counts),
            ],
            "folds": fold_audit,
            "labels_used_in_encoding": [],
            "feature_sha256": digest(output),
            "offset_sha256": digest(OUT / "patch_offsets.npy"),
        },
    )


def patch_groups(images: pd.DataFrame, kidneys: pd.DataFrame):
    values = np.load(OUT / "uri_patch_features_float16.npy", mmap_mode="r")
    offsets = np.load(OUT / "patch_offsets.npy")
    if len(offsets) != len(images) + 1 or offsets[-1] != len(values):
        raise ValueError("Patch offset mismatch")
    image_index = np.repeat(np.arange(len(images)), np.diff(offsets))
    patch_patients = images.patient_alias.to_numpy()[image_index]
    patch_kidneys = images.kidney_alias.to_numpy()[image_index]
    groups = [np.flatnonzero(patch_kidneys == alias) for alias in kidneys.kidney_alias]
    if not all(len(group) for group in groups):
        raise ValueError("Empty kidney bag")
    return values, patch_patients, groups


def prepare_bags(values, patch_patients, groups, train_aliases, seed):
    rng = np.random.default_rng(seed + 7000)
    sample = []
    for alias in sorted(train_aliases):
        available = np.flatnonzero(patch_patients == alias)
        count = min(256, len(available))
        sample.extend(rng.choice(available, count, replace=False))
    sample = np.asarray(sample, np.int64)
    scaler = StandardScaler().fit(np.asarray(values[sample], np.float32))
    compact = []
    for group in groups:
        if len(group) > 512:
            group = group[np.linspace(0, len(group) - 1, 512).round().astype(int)]
        compact.append(group)
    width = max(map(len, compact))
    bags = np.zeros((len(compact), width, values.shape[1]), np.float32)
    mask = np.zeros((len(compact), width), bool)
    for index, group in enumerate(compact):
        bags[index, : len(group)] = scaler.transform(np.asarray(values[group], np.float32))
        mask[index, : len(group)] = True
    proof = {"scaler": scaler, "sample": sample, "compact_groups": compact}
    return torch.from_numpy(bags).cuda(), torch.from_numpy(mask).cuda(), proof


def train_model(
    mode,
    negative_weight,
    seed,
    bags,
    mask,
    labels,
    fit,
    epochs,
    evaluation,
):
    torch.manual_seed(seed + (0 if mode == "mean" else 101) + int(negative_weight * 1000))
    model = PatchMIL(bags.shape[-1], mode).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    target = labels[fit]
    positive = target.sum().clamp_min(1)
    negative = (1 - target).sum().clamp_min(1)
    weights = torch.where(target > 0, len(target) / (2 * positive), len(target) / (2 * negative))
    history = {}
    for epoch in range(1, max(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        bag_logit, patch_logit, _ = model(bags[fit], mask[fit])
        loss = (
            nn.functional.binary_cross_entropy_with_logits(
                bag_logit, target, reduction="none"
            )
            * weights
        ).mean()
        if negative_weight:
            negative_rows = target == 0
            selected_patch_logits = patch_logit[negative_rows][mask[fit][negative_rows]]
            patch_loss = nn.functional.binary_cross_entropy_with_logits(
                selected_patch_logits, torch.zeros_like(selected_patch_logits)
            )
            loss = loss + negative_weight * patch_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if evaluation is not None and epoch in epochs:
            model.eval()
            with torch.inference_mode():
                score = torch.sigmoid(model(bags[evaluation], mask[evaluation])[0])
            history[epoch] = score.cpu().numpy()
    return model, history


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed URI patch-MIL evaluation")
    audit = json.loads((OUT / "encoder_audit.json").read_text(encoding="utf-8"))
    feature_path = OUT / "uri_patch_features_float16.npy"
    if audit["feature_sha256"] != digest(feature_path):
        raise ValueError("Patch feature hash mismatch")
    patients, kidneys, images = cohort.load_reference()
    values, patch_patients, groups = patch_groups(images, kidneys)
    labels_np = kidneys.kidney_APN.to_numpy(int)
    labels = torch.from_numpy(labels_np.astype(np.float32)).cuda()
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    references = pd.read_csv(REFERENCE)
    references = references[references.group.eq(URI_GROUP)].set_index(
        ["seed", "kidney_alias"], verify_integrity=True
    )
    metrics = []
    predictions = []
    selections = []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        training_patients = patients[patients.patient_alias.map(mapping).eq("train")].reset_index(drop=True)
        fit_patient, select_patient = train_test_split(
            np.arange(len(training_patients)),
            train_size=100,
            random_state=seed + 4000,
            stratify=training_patients.DMSA_APN,
        )
        fit_aliases = training_patients.patient_alias.iloc[fit_patient].to_numpy()
        select_aliases = training_patients.patient_alias.iloc[select_patient].to_numpy()
        fit = np.flatnonzero(kidneys.patient_alias.isin(fit_aliases))
        select = np.flatnonzero(kidneys.patient_alias.isin(select_aliases))
        inner_bags, inner_mask, inner_proof = prepare_bags(
            values, patch_patients, groups, fit_aliases, seed
        )
        chosen_epochs = {}
        for arm, (mode, negative_weight) in ARMS.items():
            _, history = train_model(
                mode,
                negative_weight,
                seed,
                inner_bags,
                inner_mask,
                labels,
                fit,
                EPOCHS,
                select,
            )
            choices = []
            for epoch, score in history.items():
                result = base.metric(labels_np[select], score)
                choices.append((-(result["auroc"] + result["ap"]) / 2, epoch, result))
            _, chosen, result = min(choices, key=lambda item: (item[0], item[1]))
            chosen_epochs[arm] = chosen
            selections.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "selected_epoch": chosen,
                    "inner_auroc": result["auroc"],
                    "inner_ap": result["ap"],
                }
            )
        del inner_bags, inner_mask
        torch.cuda.empty_cache()

        train_aliases = mapping[mapping.eq("train")].index.to_numpy()
        outer_train = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("train"))
        development = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        bags, mask, full_proof = prepare_bags(
            values, patch_patients, groups, train_aliases, seed
        )
        joblib.dump(
            {"inner": inner_proof, "full": full_proof},
            OUT / "models" / f"{seed}_preprocessing.joblib",
            compress=3,
        )
        target = kidneys.iloc[development]
        fixed_uri = references.loc[
            [(seed, alias) for alias in target.kidney_alias], "score"
        ].to_numpy(float)
        draw_scores = {"official_uri_fixed_reference": fixed_uri}
        for arm, (mode, negative_weight) in ARMS.items():
            model, _ = train_model(
                mode,
                negative_weight,
                seed,
                bags,
                mask,
                labels,
                outer_train,
                (chosen_epochs[arm],),
                None,
            )
            model.eval()
            with torch.inference_mode():
                logit, _, attention = model(bags[development], mask[development])
            score = torch.sigmoid(logit).cpu().numpy()
            draw_scores[arm] = score
            draw_scores[f"{arm}_20pct_plus_uri"] = 0.20 * score + 0.80 * fixed_uri
            draw_scores[f"{arm}_25pct_plus_uri"] = 0.25 * score + 0.75 * fixed_uri
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arm": arm,
                    "epoch": chosen_epochs[arm],
                    "seed": seed,
                },
                OUT / "models" / f"{seed}_{arm}.pth",
            )
            np.savez_compressed(
                OUT / f"{seed}_{arm}_development_attention.npz",
                attention=attention.cpu().numpy(),
                mask=mask[development].cpu().numpy(),
            )
        for group, score in draw_scores.items():
            result = base.metric(labels_np[development], score)
            metrics.append(
                {
                    "seed": seed,
                    "role": "development",
                    "group": group,
                    "patients": target.patient_alias.nunique(),
                    "kidneys": len(target),
                    **result,
                }
            )
            for local, index in enumerate(development):
                predictions.append(
                    {
                        "seed": seed,
                        "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[index],
                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                        "side": kidneys.side.iloc[index],
                        "label": int(labels_np[index]),
                        "score": float(score[local]),
                    }
                )
        del bags, mask
        torch.cuda.empty_cache()
        print(f"URI patch-MIL development draw {seed} complete", flush=True)
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "training_epoch_selections.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv")
    groups_to_score = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups_to_score, SEEDS, "official_uri_fixed_reference"
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(
        OUT / "decision.json",
        {
            "groups_above_both_development_targets": achieved,
            "promotion": bool(achieved),
            "reference": "official_uri_fixed_reference",
            "exploratory_test_rows_scored": 0,
        },
    )
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    import argparse
    from threadpoolctl import threadpool_limits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "cache", "run"))
    arguments = parser.parse_args()
    with threadpool_limits(limits=4):
        {"prepare": prepare, "cache": cache, "run": run}[arguments.stage]()
