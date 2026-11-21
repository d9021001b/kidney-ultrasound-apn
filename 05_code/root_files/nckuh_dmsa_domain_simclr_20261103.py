"""Patient-separated in-domain self-supervised ultrasound representation probe.

The raw NCKUH panel archive includes unlabeled same-center examinations.  For
each outer draw this script pretrains only on outer-training patients plus panels
whose outcome is unavailable, explicitly excluding all labelled development and
exploratory-test patients before any augmentation or encoder update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

import nckuh_dmsa_fullcohort_fullframe_20261103 as F
from nckuh_only_study_20261102 import metric


ROOT = F.ROOT
BASE = F.OUT
OUT = ROOT / "02_experiments/NCKUH_DMSA_domain_simclr_20261103"
RAD_SOURCE = ROOT / "02_experiments/CYCH_feature_research_results_20260913/source"
REFERENCE_GROUP = "fullcohort_rad_color_image_plus_patient_direct"
SEEDS = F.SEEDS
EPOCHS = 20
BATCH = 64
INNER_C = (0.01, 0.1, 1.0)
INNER_K = (32, 64)
GROUPS = ("domain_simclr_pca_lr", "domain_simclr_quarter_plus_instance_reference")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_reference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference as load

    patients, kidneys, images = load()
    images = images[images.kidney_APN.notna()].reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].reset_index(drop=True)
    assert (len(patients), len(kidneys), len(images)) == (210, 414, 820)
    return patients, kidneys, images


def build_cache() -> tuple[np.ndarray, pd.DataFrame]:
    """Cache all parsable aligned panels without reading disease outcomes."""
    cache = OUT / "all_panel_canvas_128.npy"
    manifest_path = OUT / "panel_manifest_private.csv"
    if cache.exists() and manifest_path.exists():
        values = np.load(cache, mmap_mode="r")
        manifest = pd.read_csv(manifest_path, keep_default_na=False)
        assert len(values) == len(manifest)
        return values, manifest
    _, _, labelled = load_reference()
    known = labelled[["patient_id", "patient_alias"]].drop_duplicates().copy()
    known["normalized_patient_id"] = known.patient_id.astype(str).map(F.normalize_patient_id)
    alias_by_source = known.set_index("normalized_patient_id").patient_alias.to_dict()
    exact, _ = F.panel_index()
    records, canvases = [], []
    for (patient_id, _, _), paths in sorted(exact.items()):
        for path in paths:
            try:
                bgr, mask = F.split_panel(path)
            except ValueError:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            canvas, _ = F.tight_canvas(rgb, mask, size=128)
            canvases.append(canvas.transpose(2, 0, 1))
            records.append({"known_patient_alias": alias_by_source.get(patient_id, ""),
                            "source_is_labelled": int(patient_id in alias_by_source)})
    values = np.stack(canvases).astype(np.uint8)
    np.save(cache, values)
    manifest = pd.DataFrame(records)
    manifest.to_csv(manifest_path, index=False)
    write_json(OUT / "panel_cache_audit.json", {
        "panels": len(values), "shape": list(values.shape),
        "labelled_patient_panels": int(manifest.source_is_labelled.sum()),
        "outcome_unavailable_panels": int((~manifest.source_is_labelled.astype(bool)).sum()),
        "labels_used_to_create_pixels": [],
    })
    return values, manifest


def color_augment(x: torch.Tensor) -> torch.Tensor:
    """Small ultrasound-safe photometric/geometric SimCLR view generator."""
    n = len(x)
    flipped = torch.rand(n, device=x.device) < .5
    x = x.clone()
    x[flipped] = x[flipped].flip(-1)
    gain = 1.0 + .20 * (torch.rand(n, 1, 1, 1, device=x.device) - .5) * 2
    offset = .10 * (torch.rand(n, 1, 1, 1, device=x.device) - .5) * 2
    noise = .025 * torch.randn_like(x)
    return (x * gain + offset + noise).clamp(-1, 1)


class SimCLR(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 128))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.adaptive_avg_pool2d(self.encoder.features(x), 1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.projector(self.encode(x)), dim=1)


def nt_xent(first: torch.Tensor, second: torch.Tensor, temperature: float = .2) -> torch.Tensor:
    z = torch.cat([first, second])
    similarity = z @ z.T / temperature
    similarity.fill_diagonal_(-torch.inf)
    n = len(first)
    target = torch.cat([torch.arange(n, 2 * n, device=z.device), torch.arange(0, n, device=z.device)])
    return nn.functional.cross_entropy(similarity, target)


def radimagenet_encoder() -> nn.Module:
    sys.path.insert(0, str(RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121
    return radimagenet_densenet121()


def pretrain(values: np.ndarray, indices: np.ndarray, seed: int) -> SimCLR:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SimCLR(radimagenet_encoder()).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(indices)
        for start in range(0, len(order), BATCH):
            batch = values[order[start:start + BATCH]]
            x = torch.from_numpy(np.asarray(batch).copy()).cuda().float() / 127.5 - 1
            optimizer.zero_grad(set_to_none=True)
            loss = nt_xent(model(color_augment(x)), model(color_augment(x)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
    return model.eval()


def encode_images(model: SimCLR, images: pd.DataFrame) -> np.ndarray:
    output = []
    with torch.inference_mode():
        for start in range(0, len(images), BATCH):
            arrays = []
            for image_id in images.image_id.iloc[start:start + BATCH]:
                bgr = cv2.imread(str(BASE / "clean_rgb" / f"{image_id}.png"), cv2.IMREAD_COLOR)
                mask = cv2.imread(str(BASE / "masks" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE) > 0
                canvas, _ = F.tight_canvas(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask, size=128)
                arrays.append(canvas.transpose(2, 0, 1))
            x = torch.from_numpy(np.stack(arrays)).cuda().float() / 127.5 - 1
            output.append(model.encode(x).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def kidney_means(images: pd.DataFrame, kidneys: pd.DataFrame, features: np.ndarray) -> np.ndarray:
    result = []
    for alias in kidneys.kidney_alias:
        rows = np.flatnonzero(images.kidney_alias.eq(alias))
        result.append(features[rows].mean(0))
    return np.stack(result)


def make_head(k: int, c: float, n_features: int):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=min(k, n_features), random_state=0),
                         LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=4000))


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered domain-SimCLR protocol")
    inputs = [Path(__file__), BASE / "run_lock.json", BASE / "patient_assignments.csv",
              BASE / "instance_development_predictions.csv", ROOT / "02_experiments/NCKUH_DMSA_label_audit_20261102/image_labels_dmsa_private.csv"]
    write_json(OUT / "protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Same-center unlabeled ultrasound panels can adapt a medical image encoder to NCKUH acquisition/appearance without consuming APN outcomes.",
        "corpus": "All parsable aligned panels; for each outer draw, use outer-training patients plus panels whose outcomes are unavailable. Exclude every labelled development and exploratory-test patient before self-supervised pretraining.",
        "encoder": "RadImageNet DenseNet-121 adapted by 20 fixed epochs of two-view NT-Xent SimCLR (128px, horizontal flip, mild gain/offset/noise), then average embeddings across same-kidney images.",
        "supervised_head": "Only a patient-grouped 100/26 split inside outer training selects PCA components [32,64] and LR C [.01,.1,1]; reset/refit on all outer training patients.",
        "evaluation": "Fixed 42-patient development roles only; no exploratory test scoring or development-selected training duration/fusion weight.",
        "fusion": "Predefined 25% domain-SimCLR plus 75% unchanged image-instance reference.",
        "target": "Development AUROC>0.8 and AP>0.8 followed by locked confirmation.",
    })
    write_json(OUT / "run_lock.json", {"protocol_sha256": digest(OUT / "protocol.json"),
                                         "inputs": {str(path): digest(path) for path in inputs}})


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for path, expected in lock["inputs"].items():
        assert digest(Path(path)) == expected, path


def run() -> None:
    from sklearn.model_selection import train_test_split
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed domain-SimCLR predictions")
    torch.set_num_threads(4)
    patients, kidneys, images = load_reference()
    all_panels, panel_manifest = build_cache()
    assignments = pd.read_csv(BASE / "patient_assignments.csv")
    reference = pd.read_csv(BASE / "instance_development_predictions.csv")
    reference = reference[reference.group.eq(REFERENCE_GROUP)].set_index(["seed", "kidney_alias"]).score
    labels = kidneys.kidney_APN.to_numpy(int)
    metrics, predictions, selections = [], [], []
    (OUT / "models").mkdir(exist_ok=True)
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train_alias = set(mapping[mapping.eq("train")].index)
        excluded_alias = set(mapping[~mapping.eq("train")].index)
        allowed = panel_manifest.known_patient_alias.eq("") | panel_manifest.known_patient_alias.isin(train_alias)
        assert not panel_manifest.loc[allowed, "known_patient_alias"].isin(excluded_alias).any()
        model = pretrain(all_panels, np.flatnonzero(allowed), seed)
        features = kidney_means(images, kidneys, encode_images(model, images))
        roles = kidneys.patient_alias.map(mapping)
        train, development = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
        train_patients = patients[patients.patient_alias.isin(train_alias)].reset_index(drop=True)
        fit_patient, inner_patient = train_test_split(np.arange(len(train_patients)), train_size=100,
            random_state=seed + 71103, stratify=train_patients.DMSA_APN)
        fit_alias = set(train_patients.patient_alias.iloc[fit_patient]); inner_alias = set(train_patients.patient_alias.iloc[inner_patient])
        fit, inner = np.flatnonzero(kidneys.patient_alias.isin(fit_alias)), np.flatnonzero(kidneys.patient_alias.isin(inner_alias))
        options = []
        for c in INNER_C:
            for k in INNER_K:
                head = make_head(k, c, features.shape[1]).fit(features[fit], labels[fit])
                result = metric(labels[inner], head.predict_proba(features[inner])[:, 1])
                options.append((-(result["auroc"] + result["ap"]) / 2, k, c, result))
        _, chosen_k, chosen_c, inner_result = min(options, key=lambda x: (x[0], x[1], x[2]))
        head = make_head(chosen_k, chosen_c, features.shape[1]).fit(features[train], labels[train])
        score = head.predict_proba(features[development])[:, 1]
        ref = np.asarray([reference.loc[(seed, alias)] for alias in kidneys.kidney_alias.iloc[development]])
        scores = {GROUPS[0]: score, GROUPS[1]: .25 * score + .75 * ref}
        torch.save({"encoder": model.encoder.state_dict(), "projector": model.projector.state_dict(), "seed": seed}, OUT / "models" / f"{seed}.pth")
        selections.append({"seed": seed, "pretrain_panels": int(allowed.sum()), "excluded_panels": int((~allowed).sum()),
                           "chosen_c": chosen_c, "chosen_pca": chosen_k, "inner_auroc": inner_result["auroc"], "inner_ap": inner_result["ap"]})
        for group, candidate in scores.items():
            metrics.append({"seed": seed, "role": "development", "group": group, "patients": 42, "kidneys": len(development), **metric(labels[development], candidate)})
            predictions.extend({"seed": seed, "group": group, "patient_alias": kidneys.patient_alias.iloc[index], "kidney_alias": kidneys.kidney_alias.iloc[index], "DMSA_APN": kidneys.DMSA_APN.iloc[index], "side": kidneys.side.iloc[index], "label": int(labels[index]), "score": float(candidate[local])} for local, index in enumerate(development))
        print(f"Completed domain SimCLR draw {seed}: {int(allowed.sum())} panels", flush=True)
        del model; torch.cuda.empty_cache()
    pd.DataFrame(metrics).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "inner_selection.csv", index=False)
    summary = pd.DataFrame(metrics).groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(x) for x in summary.columns]
    summary.to_csv(OUT / "development_summary.csv")
    write_json(OUT / "development_decision.json", {"groups_above_both_development_targets": summary[(summary.auroc_mean > .8) & (summary.ap_mean > .8)].index.tolist(), "exploratory_test_scored": False})
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    globals()[parser.parse_args().stage]()
