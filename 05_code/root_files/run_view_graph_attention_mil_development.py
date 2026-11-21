"""Development-only view-informed graph-attention MIL extension."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import renal_random_validation as R
from renal_dinov3_native_probe import indices_for
from run_gated_attention_mil_development import build_bags, load_reference, metric_row, read_csv, sha256, write_json
from run_graph_attention_mil_development import normalized_rbf_adjacency


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CYCH_view_graph_attention_mil_development_20260914"
FEATURE_ROOT = ROOT / "CYCH_dinov3_native_probe_20260913"
FEATURE_PATH = FEATURE_ROOT / "features" / "dense_tight_image.npy"
PAIR_PREDICTIONS = FEATURE_ROOT / "patient_predictions_by_role.csv"
I3_PREDICTIONS = ROOT / "CYCH_directional_robustness_selection_20260914" / "development_predictions.csv"
GRAPH_PREDICTIONS = ROOT / "CYCH_graph_attention_mil_development_20260914" / "development_predictions.csv"
VIEWS = ("LL", "LT", "RL", "RT")


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
    assert set(manifest.view) == set(VIEWS)
    assert int(manifest.groupby(["patient_alias", "view"]).size().max()) == 1
    for seed in R.SEEDS:
        indices_for(patients, assignments, seed)
    protocol = {
        "scope": "development-only view-informed graph-attention MIL extension; no exploratory-test evaluation",
        "cohort": {"patients": 98, "images": 349, "positive": 52, "negative": 46},
        "draws": {"seeds": R.SEEDS, "train": 58, "development": 20},
        "method_sources": [
            {
                "name": "Multi-instance Deep Learning with Graph Convolutional Neural Networks for Diagnosis of Kidney Diseases Using Ultrasound Imaging",
                "url": "https://doi.org/10.1007/978-3-030-32689-0_15",
            },
            {
                "name": "Multi-instance deep learning of ultrasound imaging data for pattern classification of congenital abnormalities of the kidney and urinary tract in children",
                "url": "https://doi.org/10.1016/j.urology.2020.05.019",
            },
        ],
        "feature": {
            "description": "Frozen RadImageNet DenseNet-121 tight segmented-kidney ROI image embedding",
            "sha256": sha256(FEATURE_PATH),
        },
        "training_only_preprocessing": "StandardScaler then PCA(32), fitted only to images from the 58 training patients in each draw",
        "view_encoding": "Append a fixed LL/LT/RL/RT one-hot vector to each 32-dimensional node; no learned view classifier",
        "graph": "Similarity adjacency uses only the 32 image components, with the locked RBF and symmetric degree normalization from the preceding graph-attention experiment",
        "model": "Two GCN layers 36->64->32 with ReLU and dropout 0.25; gated attention width 16; linear bag classifier",
        "optimization": "CPU deterministic AdamW, learning rate 0.001, weight decay 0.001, 300 fixed epochs, class-balanced bag BCE, gradient norm 5",
        "supervision_limit": "Only patient bag labels are used; no instance is assigned a disease label and attention is not lesion truth",
        "pair_adaptation": "Average graph-attention DenseNet probability with unchanged patient-level PyRadiomics probability",
        "ensemble": "Fixed 2/3 adapted pair + 1/3 directional I3",
        "promotion_gate": "Compared with the preceding GRAPH-MIL+I3: both mean metrics improve and at least 6/10 draws improve jointly",
        "reference_hashes": {
            "pair": sha256(PAIR_PREDICTIONS),
            "i3": sha256(I3_PREDICTIONS),
            "graph": sha256(GRAPH_PREDICTIONS),
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
    print("View-informed graph-attention protocol locked", flush=True)


def append_view_one_hot(values: np.ndarray, manifest: pd.DataFrame) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(manifest), 32):
        raise ValueError("Expected one 32-dimensional vector per image")
    mapping = {view: index for index, view in enumerate(VIEWS)}
    one_hot = np.zeros((len(manifest), len(VIEWS)), dtype=np.float32)
    for row, view in enumerate(manifest.view):
        one_hot[row, mapping[view]] = 1.0
    return np.concatenate([values, one_hot], axis=1)


def train_one(seed: int, train_indices: np.ndarray, labels: np.ndarray, bags, masks):
    import torch
    from torch import nn

    class ViewGraphAttentionMIL(nn.Module):
        def __init__(self):
            super().__init__()
            self.gcn1 = nn.Linear(36, 64, bias=False)
            self.gcn2 = nn.Linear(64, 32, bias=False)
            self.dropout = nn.Dropout(0.25)
            self.attention_v = nn.Linear(32, 16)
            self.attention_u = nn.Linear(32, 16)
            self.attention_w = nn.Linear(16, 1, bias=False)
            self.classifier = nn.Linear(32, 1)

        def forward(self, values, valid):
            adjacency = normalized_rbf_adjacency(values[:, :, :32], valid)
            hidden = torch.relu(self.gcn1(torch.bmm(adjacency, values)))
            hidden = self.dropout(hidden)
            hidden = torch.relu(self.gcn2(torch.bmm(adjacency, hidden)))
            raw = self.attention_w(
                torch.tanh(self.attention_v(hidden)) * torch.sigmoid(self.attention_u(hidden))
            ).squeeze(-1)
            raw = raw.masked_fill(~valid, -1e9)
            attention = torch.softmax(raw, dim=1)
            pooled = torch.sum(attention.unsqueeze(-1) * hidden, dim=1)
            return self.classifier(pooled).squeeze(-1), attention

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = ViewGraphAttentionMIL()
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
    assert features.shape == (349, 1024) and np.isfinite(features).all()
    rad_reference = load_reference(PAIR_PREDICTIONS, "RAD", aliases)
    i3_reference = load_reference(I3_PREDICTIONS, "I3-ALL", aliases)
    graph_reference = load_reference(GRAPH_PREDICTIONS, "GRAPH-MIL+I3", aliases)

    metrics = []
    predictions = []
    attention_rows = []
    training_records = []
    for seed in R.SEEDS:
        indices = indices_for(patients, assignments, seed)
        train_ids = set(patients.patient_id.iloc[indices["train"]])
        train_images = manifest.patient_id.isin(train_ids).to_numpy()
        scaler = StandardScaler().fit(features[train_images])
        standardized = scaler.transform(features)
        pca = PCA(n_components=32, random_state=seed).fit(standardized[train_images])
        reduced = pca.transform(standardized).astype(np.float32)
        view_reduced = append_view_one_hot(reduced, manifest)
        bags, masks, image_ids = build_bags(view_reduced, manifest, patients)
        model, graph_scores, attention, loss_record = train_one(seed, indices["train"], labels, bags, masks)
        adapted = (rad_reference[seed] + graph_scores) / 2.0
        score = (2.0 * adapted + i3_reference[seed]) / 3.0
        development = indices["validation"]
        metrics.append(metric_row(seed, "GRAPH-MIL+I3-REFERENCE", labels[development], graph_reference[seed][development]))
        metrics.append(metric_row(seed, "VIEW-GRAPH-MIL+I3", labels[development], score[development]))
        for patient_index in development:
            predictions.append(
                {
                    "seed": seed,
                    "group": "VIEW-GRAPH-MIL+I3",
                    "role": "development_validation",
                    "patient_alias": aliases[patient_index],
                    "label": int(labels[patient_index]),
                    "score": float(score[patient_index]),
                }
            )
            for position, image_id in enumerate(image_ids[patient_index]):
                attention_rows.append(
                    {
                        "seed": seed,
                        "patient_alias": aliases[patient_index],
                        "image_id": image_id,
                        "view": manifest.set_index("image_id").loc[image_id, "view"],
                        "attention": float(attention[patient_index, position]),
                    }
                )
        joblib.dump({"scaler": scaler, "pca": pca, "views": VIEWS}, OUT / "models" / f"preprocess_{seed}.joblib", compress=3)
        torch.save(model.state_dict(), OUT / "models" / f"view_graph_attention_{seed}.pth")
        training_records.append(
            {
                "seed": seed,
                **loss_record,
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            }
        )
        print(f"View-informed graph-attention development: {seed}", flush=True)

    frame = pd.DataFrame(metrics)
    summary = frame.groupby("group", as_index=False).agg(
        mean_auroc=("auroc", "mean"),
        std_auroc=("auroc", "std"),
        mean_average_precision=("average_precision", "mean"),
        std_average_precision=("average_precision", "std"),
    )
    summary["joint_target_draws"] = [
        int(
            (
                (frame.loc[frame.group.eq(group), "auroc"] > 0.8)
                & (frame.loc[frame.group.eq(group), "average_precision"] > 0.8)
            ).sum()
        )
        for group in summary.group
    ]
    wide = frame.pivot(index="seed", columns="group", values=["auroc", "average_precision"])
    delta_auc = wide["auroc"]["VIEW-GRAPH-MIL+I3"] - wide["auroc"]["GRAPH-MIL+I3-REFERENCE"]
    delta_ap = (
        wide["average_precision"]["VIEW-GRAPH-MIL+I3"]
        - wide["average_precision"]["GRAPH-MIL+I3-REFERENCE"]
    )
    wins = int(((delta_auc > 0) & (delta_ap > 0)).sum())
    decision = {
        "selection_role": "development_validation",
        "delta_auroc": float(delta_auc.mean()),
        "delta_ap": float(delta_ap.mean()),
        "joint_wins": wins,
        "promoted": bool(delta_auc.mean() > 0 and delta_ap.mean() > 0 and wins >= 6),
        "exploratory_test_access": "prohibited",
    }
    frame.to_csv(OUT / "development_metrics_per_draw.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(attention_rows).to_csv(OUT / "development_image_attention.csv", index=False)
    write_json(OUT / "development_decision.json", decision)
    write_json(OUT / "training_provenance.json", training_records)
    lines = [
        "# View-informed graph-attention MIL development screen",
        "",
        "No exploratory-test result was computed.",
        "",
        "The fixed LL/LT/RL/RT code is node metadata. Only patient-level disease labels were used.",
        "",
        "| Group | Mean AUROC | Mean AP | Both >0.8 draws |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.group} | {row.mean_auroc:.3f} | {row.mean_average_precision:.3f} | {row.joint_target_draws}/10 |"
        )
    lines.extend(["", "## Decision", "", f"Decision record: `{json.dumps(decision, separators=(',', ':'))}`"])
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
