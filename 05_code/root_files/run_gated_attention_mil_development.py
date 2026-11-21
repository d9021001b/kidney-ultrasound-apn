"""Development-only gated-attention MIL on frozen kidney image embeddings."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import renal_random_validation as R
import renal_voting_validation as V
from renal_dinov3_native_probe import indices_for


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CYCH_gated_attention_mil_development_20260914"
FEATURE_ROOT = ROOT / "CYCH_dinov3_native_probe_20260913"
FEATURE_PATH = FEATURE_ROOT / "features" / "dense_tight_image.npy"
PAIR_PREDICTIONS = FEATURE_ROOT / "patient_predictions_by_role.csv"
I3_PREDICTIONS = ROOT / "CYCH_directional_robustness_selection_20260914" / "development_predictions.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"patient_id": str})


def require_lock() -> None:
    lock = json.loads((OUT / "protocol_locked.json").read_text(encoding="utf-8"))
    assert lock["protocol_sha256"] == sha256(OUT / "protocol.json")
    assert lock["source_sha256"] == sha256(Path(__file__))


def prepare() -> None:
    for name in ("models", "source"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv"):
        source = FEATURE_ROOT / name
        destination = OUT / name
        if destination.exists():
            assert sha256(destination) == sha256(source)
        else:
            shutil.copy2(source, destination)
    patients = read_csv(OUT / "patients.csv")
    manifest = read_csv(OUT / "image_manifest.csv")
    assignments = read_csv(OUT / "random_patient_assignments.csv")
    assert len(patients) == 98 and int(patients.label.sum()) == 52
    assert len(manifest) == 349 and manifest.image_id.nunique() == 349
    for seed in R.SEEDS:
        indices_for(patients, assignments, seed)
    protocol = {
        "scope": "development-only gated-attention MIL screen; no new exploratory-test evaluation",
        "cohort": {"patients": 98, "images": 349, "positive": 52, "negative": 46},
        "draws": {"seeds": R.SEEDS, "train": 58, "development": 20},
        "method_sources": [
            {
                "name": "Attention-based Deep Multiple Instance Learning",
                "url": "https://proceedings.mlr.press/v80/ilse18a.html",
            },
            {
                "name": "Multi-instance Deep Learning with Graph Convolutional Neural Networks for Diagnosis of Kidney Diseases Using Ultrasound Imaging",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6938161/",
            },
        ],
        "feature": {
            "description": "Frozen RadImageNet DenseNet-121 tight segmented-kidney ROI image embedding",
            "path": str(FEATURE_PATH),
            "sha256": sha256(FEATURE_PATH),
        },
        "training_only_preprocessing": "StandardScaler then PCA(32) fitted only to images from the 58 training patients in each draw",
        "model": "Gated attention: Linear(32,32)-ReLU-Dropout(0.25); tanh and sigmoid attention streams of width 16; softmax over patient images; linear bag classifier",
        "optimization": "CPU deterministic AdamW, learning rate 0.001, weight decay 0.001, 300 fixed epochs, class-balanced bag BCE, gradient norm 5",
        "attention_scope": "Attention ranks images within a patient bag and is not a lesion annotation or causal explanation",
        "pair_adaptation": "Average the gated-attention DenseNet bag probability with the unchanged patient-level PyRadiomics probability",
        "candidate_gate": "Adapted pair mean development AUROC and AP are each no more than 0.02 below the existing fixed pair",
        "ensemble": "Only an eligible adapted pair may form fixed 2/3 adapted pair + 1/3 directional I3",
        "ensemble_gate": "Compared with existing 2/3 fixed pair + 1/3 I3: both mean metrics improve and at least 6/10 draws improve jointly",
        "reference_hashes": {
            "pair": sha256(PAIR_PREDICTIONS),
            "i3": sha256(I3_PREDICTIONS),
        },
    }
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding="utf-8")) == protocol
    else:
        write_json(protocol_path, protocol)
    lock_path = OUT / "protocol_locked.json"
    if not lock_path.exists():
        write_json(
            lock_path,
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": sha256(protocol_path),
                "source_sha256": sha256(Path(__file__)),
            },
        )
    shutil.copy2(__file__, OUT / "source" / Path(__file__).name)
    print("Gated-attention MIL protocol locked", flush=True)


def load_reference(path: Path, group: str, aliases: list[str]) -> dict[int, np.ndarray]:
    frame = pd.read_csv(path)
    frame = frame.loc[
        frame.role.eq("development_validation") & frame.group.eq(group),
        ["seed", "patient_alias", "label", "score"],
    ]
    result = {}
    for seed in R.SEEDS:
        subset = frame.loc[frame.seed.eq(seed)].set_index("patient_alias")
        assert len(subset) == 20 and subset.index.is_unique
        result[seed] = subset.reindex(aliases).score.to_numpy(float)
    return result


def build_bags(values: np.ndarray, manifest: pd.DataFrame, patients: pd.DataFrame):
    import torch

    groups = [np.flatnonzero(manifest.patient_id.eq(patient_id).to_numpy()) for patient_id in patients.patient_id]
    width = max(map(len, groups))
    bags = np.zeros((len(groups), width, values.shape[1]), dtype=np.float32)
    masks = np.zeros((len(groups), width), dtype=bool)
    image_ids = []
    for row, indices in enumerate(groups):
        bags[row, : len(indices)] = values[indices]
        masks[row, : len(indices)] = True
        image_ids.append(manifest.image_id.iloc[indices].tolist())
    return torch.from_numpy(bags), torch.from_numpy(masks), image_ids


def metric_row(seed: int, group: str, labels: np.ndarray, scores: np.ndarray) -> dict:
    return {"seed": seed, "group": group, **V.score_metrics(labels, scores)}


def train_one(seed: int, train_indices: np.ndarray, labels: np.ndarray, bags, masks):
    import torch
    from torch import nn

    class GatedAttentionMIL(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.25))
            self.attention_v = nn.Linear(32, 16)
            self.attention_u = nn.Linear(32, 16)
            self.attention_w = nn.Linear(16, 1, bias=False)
            self.classifier = nn.Linear(32, 1)

        def forward(self, values, valid):
            hidden = self.project(values)
            raw = self.attention_w(torch.tanh(self.attention_v(hidden)) * torch.sigmoid(self.attention_u(hidden))).squeeze(-1)
            raw = raw.masked_fill(~valid, -1e9)
            attention = torch.softmax(raw, dim=1)
            pooled = torch.sum(attention.unsqueeze(-1) * hidden, dim=1)
            return self.classifier(pooled).squeeze(-1), attention

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = GatedAttentionMIL()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    train_tensor = torch.as_tensor(train_indices, dtype=torch.long)
    target = torch.as_tensor(labels[train_indices], dtype=torch.float32)
    positives = float(target.sum())
    negatives = float(len(target) - positives)
    weights = torch.where(target > 0, len(target) / (2.0 * positives), len(target) / (2.0 * negatives))
    losses = []
    for _ in range(300):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(bags[train_tensor], masks[train_tensor])
        loss = (nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.inference_mode():
        logits, attention = model(bags, masks)
        scores = torch.sigmoid(logits).numpy()
        attention = attention.numpy()
    assert np.isfinite(scores).all() and np.isfinite(attention).all()
    assert np.allclose((attention * masks.numpy()).sum(axis=1), 1.0, atol=1e-6)
    return model, scores, attention, {"first_loss": losses[0], "last_loss": losses[-1]}


def evaluate() -> None:
    require_lock()
    import torch

    torch.set_num_threads(4)
    patients = read_csv(OUT / "patients.csv")
    manifest = read_csv(OUT / "image_manifest.csv")
    assignments = read_csv(OUT / "random_patient_assignments.csv")
    features = np.load(FEATURE_PATH)
    labels = patients.label.to_numpy(int)
    aliases = patients.patient_alias.tolist()
    assert features.shape[0] == len(manifest) == 349 and np.isfinite(features).all()
    pair_reference = load_reference(PAIR_PREDICTIONS, "PAIR", aliases)
    rad_reference = load_reference(PAIR_PREDICTIONS, "RAD", aliases)
    i3_reference = load_reference(I3_PREDICTIONS, "I3-ALL", aliases)

    metrics = []
    predictions = []
    attention_rows = []
    adapted_scores = {}
    training_records = []
    for seed in R.SEEDS:
        indices = indices_for(patients, assignments, seed)
        train_ids = set(patients.patient_id.iloc[indices["train"]])
        train_images = manifest.patient_id.isin(train_ids).to_numpy()
        scaler = StandardScaler().fit(features[train_images])
        standardized = scaler.transform(features)
        pca = PCA(n_components=32, random_state=seed).fit(standardized[train_images])
        reduced = pca.transform(standardized).astype(np.float32)
        bags, masks, image_ids = build_bags(reduced, manifest, patients)
        model, dense_scores, attention, loss_record = train_one(seed, indices["train"], labels, bags, masks)
        adapted = (rad_reference[seed] + dense_scores) / 2.0
        adapted_scores[seed] = adapted
        development = indices["validation"]
        metrics.append(metric_row(seed, "PAIR", labels[development], pair_reference[seed][development]))
        metrics.append(metric_row(seed, "PAIR-GATED-MIL", labels[development], adapted[development]))
        metrics.append(
            metric_row(
                seed,
                "PAIR+I3-REFERENCE",
                labels[development],
                ((2.0 * pair_reference[seed] + i3_reference[seed]) / 3.0)[development],
            )
        )
        for index in development:
            predictions.append(
                {
                    "seed": seed,
                    "group": "PAIR-GATED-MIL",
                    "role": "development_validation",
                    "patient_alias": aliases[index],
                    "label": int(labels[index]),
                    "score": float(adapted[index]),
                }
            )
            for position, image_id in enumerate(image_ids[index]):
                attention_rows.append(
                    {
                        "seed": seed,
                        "role": "development_validation",
                        "patient_alias": aliases[index],
                        "image_id": image_id,
                        "attention_weight": float(attention[index, position]),
                    }
                )
        state_path = OUT / "models" / f"gated_attention_{seed}.pt"
        torch.save(model.state_dict(), state_path)
        preprocessing_path = OUT / "models" / f"preprocessing_{seed}.joblib"
        joblib.dump({"scaler": scaler, "pca": pca}, preprocessing_path, compress=3)
        training_records.append(
            {
                "seed": seed,
                "model_sha256": sha256(state_path),
                "preprocessing_sha256": sha256(preprocessing_path),
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                **loss_record,
            }
        )
        print(f"Gated-attention MIL complete: {seed}", flush=True)

    frame = pd.DataFrame(metrics)
    means = frame.groupby("group")[["auroc", "average_precision"]].mean()
    delta = means.loc["PAIR-GATED-MIL"] - means.loc["PAIR"]
    eligible = bool((delta >= -0.02 - 1e-12).all())
    decision = {
        "selection_role": "development_validation",
        "delta_auroc_vs_pair": float(delta.auroc),
        "delta_ap_vs_pair": float(delta.average_precision),
        "eligible": eligible,
    }
    if eligible:
        for seed in R.SEEDS:
            development = indices_for(patients, assignments, seed)["validation"]
            score = (2.0 * adapted_scores[seed] + i3_reference[seed]) / 3.0
            metrics.append(metric_row(seed, "GATED-MIL+I3", labels[development], score[development]))
            for index in development:
                predictions.append(
                    {
                        "seed": seed,
                        "group": "GATED-MIL+I3",
                        "role": "development_validation",
                        "patient_alias": aliases[index],
                        "label": int(labels[index]),
                        "score": float(score[index]),
                    }
                )
        frame = pd.DataFrame(metrics)
        compare = frame.loc[frame.group.isin(["PAIR+I3-REFERENCE", "GATED-MIL+I3"])]
        wide = compare.pivot(index="seed", columns="group", values=["auroc", "average_precision"])
        delta_auc = wide["auroc"]["GATED-MIL+I3"] - wide["auroc"]["PAIR+I3-REFERENCE"]
        delta_ap = wide["average_precision"]["GATED-MIL+I3"] - wide["average_precision"]["PAIR+I3-REFERENCE"]
        joint_wins = int(((delta_auc > 0) & (delta_ap > 0)).sum())
        decision["ensemble"] = {
            "delta_auroc": float(delta_auc.mean()),
            "delta_ap": float(delta_ap.mean()),
            "joint_wins": joint_wins,
            "promoted": bool(delta_auc.mean() > 0 and delta_ap.mean() > 0 and joint_wins >= 6),
        }

    frame = pd.DataFrame(metrics)
    summary = frame.groupby("group", as_index=False).agg(
        mean_auroc=("auroc", "mean"),
        std_auroc=("auroc", "std"),
        mean_average_precision=("average_precision", "mean"),
        std_average_precision=("average_precision", "std"),
    )
    summary["joint_target_draws"] = [
        int(((frame.loc[frame.group.eq(group), "auroc"] > 0.8) &
             (frame.loc[frame.group.eq(group), "average_precision"] > 0.8)).sum())
        for group in summary.group
    ]
    frame.to_csv(OUT / "development_metrics_per_draw.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(attention_rows).to_csv(OUT / "development_image_attention.csv", index=False)
    write_json(OUT / "training_provenance.json", training_records)
    write_json(OUT / "development_decision.json", decision)

    lines = [
        "# Gated-attention MIL development screen",
        "",
        "This experiment learns patient-bag image weights from frozen RadImageNet DenseNet-121 ROI embeddings.",
        "Attention values rank images within a bag; they are not lesion truth. No new exploratory-test result was computed.",
        "",
        "| Group | Mean AUROC | Mean AP | Both >0.8 draws |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(f"| {row.group} | {row.mean_auroc:.3f} | {row.mean_average_precision:.3f} | {row.joint_target_draws}/10 |")
    lines.extend([
        "",
        "## Decision",
        "",
        f"Decision record: `{json.dumps(decision, separators=(',', ':'))}`",
        "",
        "Gated-attention reference: https://proceedings.mlr.press/v80/ilse18a.html",
        "Kidney-ultrasound MIL reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC6938161/",
    ])
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(decision), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "evaluate", "all"))
    args = parser.parse_args()
    stages = ("prepare", "evaluate") if args.stage == "all" else (args.stage,)
    for stage in stages:
        globals()[stage]()


if __name__ == "__main__":
    main()
