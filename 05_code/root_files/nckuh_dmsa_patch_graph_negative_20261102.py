"""USF-MAE patch graph MIL with reliable-negative instance supervision."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

import nckuh_dmsa_patch_mil_20261102 as B
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json


ROOT = B.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_patch_graph_negative_20261102"
ENCODER = "usf_mae"
SEEDS = B.PILOT_SEEDS
MODES = ["gated_bag", "gated_negative", "graph_bag", "graph_negative"]
GROUPS = [f"usf_patch_{mode}" for mode in MODES] + ["usf_patch_graph_negative_plus_BCE_Adam"]
REFERENCE = "BCE_Adam"
MAX_NODES = 64


def prepare():
    B.require_lock()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered protocol")
    proof = json.loads((B.CACHE / "encoder_provenance.json").read_text(encoding="utf-8"))
    assert proof[ENCODER]["frozen"]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Whole-kidney/image pooling may dilute focal APN evidence. Test within-kidney graphs of frozen in-mask local USF-MAE patches and one-sided reliable-negative supervision",
        "references": [
            "https://doi.org/10.1007/978-3-030-32689-0_15",
            "https://proceedings.mlr.press/v80/ilse18a.html",
            "https://arxiv.org/abs/2510.22990",
        ],
        "cohort": "Same 210 source-code-available patients, 414 imaged kidneys and 820 images; 54 missing-reference patients excluded",
        "patches": "Existing frozen USF-MAE ROI tokens with >=80% kidney-mask coverage. Training-only PCA32 whitening; deterministic evenly spaced compaction to at most64 tokens per kidney after PCA so every bag remains nonempty",
        "models": "Matched gated-attention versus two-layer RBF graph propagation; each with bag BCE only or bag BCE plus reliable-negative patch BCE",
        "reliable_negative": "Only valid patches from DMSA-negative kidneys receive instance target0. No patch from a positive kidney receives a positive or negative instance target. Attention/instance scores are not lesion annotations",
        "loss": "Class-balanced kidney bag BCE plus coefficient1.0 reliable-negative BCE, matching the prior CYCH mechanism; no coefficient search",
        "selection": "Separate R/L/B/N-stratified 100/26 patient inner split within each126 training role; select earliest epoch maximizing mean kidney AUROC/AP, max80, minimum10, patience15; reset/refit all126",
        "evaluation": "Three fixed42-patient development roles only; exploratory test remains unscored. Patient keeps both kidneys, views and patches in one role",
        "fusion": "One prespecified equal-probability average of graph-negative output and locked paired-handcrafted BCE-Adam development output; no fitted voting weight",
        "gate": "Advance if mean AUROC>=.70 and AP>=.78, gain vs BCE-Adam >=.03/.02 and joint improvement >=2/3 draws, or absolute means both>.8 with both>.8 in >=2/3 draws",
        "uncertainty": "10000 R/L/B/N-stratified patient-cluster fixed-prediction resamples across three development draws",
        "limitations": "Exploratory reused cohort; patch graph is feature similarity, not spatial histology; exact DMSA timing caveat remains",
    }
    write_json(OUT / "protocol.json", protocol)
    paths = [
        Path(__file__), Path(B.__file__), B.CACHE / "encoder_provenance.json",
        B.CACHE / "patch_offsets.npy", B.CACHE / f"{ENCODER}_roi_patches.npy",
        B.BASE / "patient_assignments.csv", B.AUDIT / "image_labels_dmsa_private.csv",
        ROOT / "02_experiments/NCKUH_DMSA_metric_optimization_20261102/pilot_predictions.csv",
    ]
    write_json(OUT / "run_lock.json", {
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in paths},
    })
    print("Registered USF patch graph/reliable-negative pilot", flush=True)


def require_lock():
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(ROOT / name) == expected, name


def compact_bags(bags, valid, max_nodes=MAX_NODES):
    output = torch.zeros((len(bags), max_nodes, bags.shape[-1]), dtype=bags.dtype, device=bags.device)
    mask = torch.zeros((len(bags), max_nodes), dtype=torch.bool, device=bags.device)
    selected = []
    for i in range(len(bags)):
        count = int(valid[i].sum())
        if count <= 0:
            raise ValueError("Empty kidney patch bag")
        index = np.arange(count) if count <= max_nodes else np.linspace(0, count-1, max_nodes).round().astype(int)
        index = np.unique(index)
        source = torch.as_tensor(index, dtype=torch.long, device=bags.device)
        output[i, :len(index)] = bags[i, source]
        mask[i, :len(index)] = True
        selected.append(index.tolist())
    return output, mask, selected


def normalized_rbf_adjacency(values, valid):
    difference = values.unsqueeze(2) - values.unsqueeze(1)
    distance2 = torch.sum(difference * difference, dim=-1)
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    eye = torch.eye(values.shape[1], dtype=torch.bool, device=values.device).unsqueeze(0)
    off_diagonal = pair_valid & ~eye
    positive = torch.where(off_diagonal & (distance2 > 0), distance2, torch.nan)
    # CUDA nanmedian-with-indices has no deterministic implementation in the
    # pinned torch build. Bandwidth is a bag statistic, so compute it from a
    # detached CPU copy while retaining gradients through distance2 itself.
    bandwidth = torch.nanmedian(positive.detach().cpu().flatten(1), dim=1).values.to(values.device)
    bandwidth = torch.where(torch.isfinite(bandwidth), bandwidth, torch.ones_like(bandwidth)).clamp_min(1e-6)
    adjacency = torch.exp(-distance2 / bandwidth[:, None, None]) * pair_valid
    adjacency = torch.where(eye & pair_valid, torch.ones_like(adjacency), adjacency)
    degree = adjacency.sum(dim=-1).clamp_min(1e-8)
    normalized = adjacency / torch.sqrt(degree.unsqueeze(-1) * degree.unsqueeze(-2))
    return normalized * pair_valid


class PatchGraphModel(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.graph = mode.startswith("graph_")
        self.negative = mode.endswith("_negative")
        if self.graph:
            self.layer1 = nn.Linear(32, 64, bias=False)
            self.layer2 = nn.Linear(64, 32, bias=False)
        else:
            self.layer1 = nn.Linear(32, 32)
            self.layer2 = None
        self.dropout = nn.Dropout(.25)
        self.v, self.u, self.w = nn.Linear(32, 16), nn.Linear(32, 16), nn.Linear(16, 1, bias=False)
        self.bag_classifier = nn.Linear(32, 1)
        self.instance_classifier = nn.Linear(32, 1)

    def forward(self, x, valid):
        if self.graph:
            adjacency = normalized_rbf_adjacency(x, valid)
            hidden = self.dropout(torch.relu(self.layer1(torch.bmm(adjacency, x))))
            hidden = torch.relu(self.layer2(torch.bmm(adjacency, hidden)))
        else:
            hidden = self.dropout(torch.relu(self.layer1(x)))
        raw = self.w(torch.tanh(self.v(hidden)) * torch.sigmoid(self.u(hidden))).squeeze(-1)
        attention = torch.softmax(raw.masked_fill(~valid, -1e9), dim=1)
        pooled = (attention[..., None] * hidden).sum(1)
        return self.bag_classifier(pooled).squeeze(-1), self.instance_classifier(hidden).squeeze(-1), attention


def reliable_negative_loss(instance_logits, valid, labels):
    selected = valid & labels[:, None].eq(0)
    if not bool(selected.any()):
        return instance_logits.sum() * 0
    return nn.functional.binary_cross_entropy_with_logits(instance_logits[selected],
                                                           torch.zeros_like(instance_logits[selected]))


def fit_model(bags, valid, y, fit, inner, mode, seed, epochs=None):
    torch.manual_seed(seed + 73)
    torch.cuda.manual_seed_all(seed + 73)
    model = PatchGraphModel(mode).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    target = y[fit]
    weights = torch.where(target > 0, len(target)/(2*target.sum()), len(target)/(2*(len(target)-target.sum())))
    best, selected_epoch, stale = -float("inf"), 1, 0
    history = []
    for epoch in range(1, (epochs if epochs is not None else 80)+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits, instances, _ = model(bags[fit], valid[fit])
        bag_loss = (nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).mean()
        instance_loss = reliable_negative_loss(instances, valid[fit], target)
        loss = bag_loss + (instance_loss if model.negative else 0)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite patch graph objective")
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        row = {"phase": "inner" if epochs is None else "refit", "epoch": epoch,
               "bag_bce": float(bag_loss.detach()), "reliable_negative_bce": float(instance_loss.detach())}
        if inner is not None:
            model.eval()
            with torch.inference_mode():
                current = torch.sigmoid(model(bags[inner], valid[inner])[0]).cpu().numpy()
            m = metric(y[inner].cpu().numpy().astype(int), current)
            criterion = (m["auroc"] + m["ap"])/2
            row.update({f"inner_{k}": v for k, v in m.items()})
            if criterion > best + 1e-12:
                best, selected_epoch, stale = criterion, epoch, 0
            else:
                stale += 1
        history.append(row)
        if inner is not None and epoch >= 10 and stale >= 15:
            break
    return model.eval(), selected_epoch if inner is not None else epochs, history


def run():
    from sklearn.model_selection import train_test_split
    from threadpoolctl import threadpool_limits
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed pilot")
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    patients, kidneys, images = B.load_reference()
    values, patch_aliases, patch_groups, global_rows = B.local_patch_data(ENCODER, images, kidneys)
    assignments = pd.read_csv(B.BASE / "patient_assignments.csv")
    y = torch.tensor(kidneys.kidney_APN.to_numpy(int), dtype=torch.float32, device="cuda")
    outputs, metrics_rows, audits = [], [], []
    (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in SEEDS:
            mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
            outer = patients[patients.patient_alias.map(mapping).eq("train")]
            fit_alias, inner_alias = train_test_split(outer.patient_alias.to_numpy(), train_size=100,
                random_state=seed+4000, stratify=outer.DMSA_APN)
            fit = np.flatnonzero(kidneys.patient_alias.isin(fit_alias)); inner = np.flatnonzero(kidneys.patient_alias.isin(inner_alias))
            train = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("train")); dev = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
            inner_bags, inner_valid, inner_proof = B.prepare_bags(values, patch_aliases, patch_groups, fit_alias, seed)
            inner_bags, inner_valid, inner_selected = compact_bags(inner_bags, inner_valid)
            chosen, histories = {}, {}
            for mode in MODES:
                model, epoch, history = fit_model(inner_bags, inner_valid, y, fit, inner, mode, seed)
                chosen[mode], histories[mode] = epoch, history
                del model
            del inner_bags, inner_valid; torch.cuda.empty_cache()
            bags, valid, full_proof = B.prepare_bags(values, patch_aliases, patch_groups,
                                                     outer.patient_alias.to_numpy(), seed)
            bags, valid, full_selected = compact_bags(bags, valid)
            joblib.dump({"inner": inner_proof, "full": full_proof, "source_patch_rows": global_rows,
                         "inner_compaction": inner_selected, "full_compaction": full_selected},
                        OUT / "models" / f"{seed}_preprocess.joblib", compress=3)
            seed_predictions = []
            for mode in MODES:
                model, epoch, refit_history = fit_model(bags, valid, y, train, None, mode, seed, chosen[mode])
                with torch.inference_mode():
                    logits, instances, attention = model(bags[dev], valid[dev])
                    scores = torch.sigmoid(logits).cpu().numpy()
                group = f"usf_patch_{mode}"
                m = metric(y[dev].cpu().numpy().astype(int), scores)
                metrics_rows.append({"seed": seed, "group": group, "patients": 42, "kidneys": len(dev),
                                     "selected_epoch": epoch, **m})
                current = []
                for i, score in zip(dev, scores):
                    row = kidneys.iloc[i]
                    current.append({"seed": seed, "group": group, "patient_alias": row.patient_alias,
                                    "kidney_alias": row.kidney_alias, "DMSA_APN": row.DMSA_APN,
                                    "side": row.side, "label": int(row.kidney_APN), "score": float(score)})
                outputs.extend(current); seed_predictions.extend(current)
                torch.save({"model": model.state_dict(), "mode": mode, "epoch": epoch},
                           OUT / "models" / f"{seed}_{group}.pth")
                pd.DataFrame(histories[mode] + refit_history).to_csv(OUT / f"history_{seed}_{group}.csv", index=False)
                np.savez_compressed(OUT / f"attention_{seed}_{group}.npz", kidney_indices=dev,
                                    attention=attention.cpu().numpy(), valid=valid[dev].cpu().numpy(),
                                    instance_score=torch.sigmoid(instances).cpu().numpy())
                del model
            audits.append({"seed": seed, "fit_patients": len(fit_alias), "inner_patients": len(inner_alias),
                           "full_train_patients": len(outer), "selected_epochs": chosen,
                           "inner_pca_patches": inner_proof["patch_count"], "full_pca_patches": full_proof["patch_count"],
                           "max_nodes": MAX_NODES, "test_scored": False})
            print(f"Completed USF patch graph draw {seed}: {chosen}", flush=True)
            del bags, valid; torch.cuda.empty_cache()
    predictions = pd.DataFrame(outputs)
    locked = pd.read_csv(ROOT / "02_experiments/NCKUH_DMSA_metric_optimization_20261102/pilot_predictions.csv")
    locked = locked[locked.group.eq(REFERENCE) & locked.seed.isin(SEEDS)].copy()
    graph = predictions[predictions.group.eq("usf_patch_graph_negative")]
    merged = graph.merge(locked[["seed", "kidney_alias", "score"]], on=["seed", "kidney_alias"],
                         suffixes=("", "_reference"), validate="one_to_one")
    merged["group"] = "usf_patch_graph_negative_plus_BCE_Adam"
    merged["score"] = (merged.score + merged.score_reference)/2
    merged = merged[predictions.columns]
    predictions = pd.concat([predictions, merged], ignore_index=True)
    for (seed, group), frame in merged.groupby(["seed", "group"]):
        metrics_rows.append({"seed": seed, "group": group, "patients": frame.patient_alias.nunique(),
                             "kidneys": len(frame), **metric(frame.label, frame.score)})
    predictions.to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(OUT / "development_metrics.csv", index=False)
    write_json(OUT / "training_audits.json", audits)
    summary = pd.DataFrame(metrics_rows).groupby("group")[["auroc", "ap", "auprc", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]; summary.to_csv(OUT / "development_summary.csv")
    all_predictions = pd.concat([predictions, locked[predictions.columns]], ignore_index=True)
    ci_groups = [REFERENCE] + GROUPS
    ci, differences, _ = cluster_intervals(all_predictions, ci_groups, SEEDS, REFERENCE, 10000)
    ci.to_csv(OUT / "development_cluster_ci.csv", index=False)
    differences.to_csv(OUT / "development_cluster_differences.csv", index=False)
    print(summary[["auroc_mean", "ap_mean"]].to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.command]()
