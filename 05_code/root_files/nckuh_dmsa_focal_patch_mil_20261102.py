"""Focal top-k patch MIL and unilateral kidney ranking under DMSA side labels."""
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
OUT = ROOT / "02_experiments/NCKUH_DMSA_focal_patch_mil_20261102"
ENCODER = "usf_mae"
SEEDS = B.PILOT_SEEDS
MODES = ["top1_bce", "top4_bce", "logmeanexp_bce", "top4_bce_rank"]
GROUPS = [f"usf_focal_{mode}" for mode in MODES] + ["usf_focal_top4_bce_rank_plus_BCE_Adam"]
REFERENCE = "BCE_Adam"
MAX_NODES = 64
PCA_WIDTH = 64
RANK_WEIGHT = 0.2


def prepare():
    B.require_lock()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered protocol")
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "APN may be focal and visible in only a small intrarenal region. Standard MIL top-k pooling should preserve a focal high-risk patch that whole-kidney averaging or diffuse attention can dilute",
        "references": ["https://proceedings.mlr.press/v80/ilse18a.html", "https://arxiv.org/abs/2510.22990"],
        "cohort": "Same210 patients,414 imaged kidneys,820 images; side-specific DMSA_APN endpoint;54 unknown references excluded",
        "patches": "Frozen USF-MAE in-mask tokens (>=80% mask coverage), L2 normalized. Training-patient-balanced sampling fits PCA64 whitening. Every kidney is deterministically compacted to at most64 evenly spaced tokens",
        "pooling": "Compare maximum single-instance, mean of top4 instance logits, and log-mean-exp instance aggregation. No positive patch target is assigned",
        "ranking": "One prespecified top4 arm adds weight0.2 softplus pair loss for training patients whose two imaged kidneys have opposite source-supported labels; positive kidney score should exceed negative kidney score",
        "optimization": "Class-balanced kidney BCE; AdamW lr.001, weight_decay.001, dropout.25, gradient norm5",
        "selection": "R/L/B/N-stratified100/26 inner patient split inside each126 training role; earliest max mean(inner AUROC,AP), max100, min10, patience20; reset/refit all126",
        "evaluation": "Three fixed42-patient development roles only; exploratory test unscored; no k-fold",
        "fusion": "One prespecified equal average of top4+ranking probability and locked paired-handcrafted BCE-Adam probability",
        "gate": "Advance if mean AUROC>=.70/AP>=.78, gain>=.03/.02 vs BCE-Adam and joint improvement>=2/3, or absolute means both>.8 with both>.8 in>=2/3 draws",
        "uncertainty": "10000 R/L/B/N-stratified patient-cluster fixed-prediction resamples",
        "interpretation": "Selected patches are computational instances, not APN lesions, scars, cortex or clinical annotations",
    }
    write_json(OUT / "protocol.json", protocol)
    paths = [Path(__file__), Path(B.__file__), B.CACHE / "encoder_provenance.json",
             B.CACHE / "patch_offsets.npy", B.CACHE / f"{ENCODER}_roi_patches.npy",
             B.BASE / "patient_assignments.csv", B.AUDIT / "image_labels_dmsa_private.csv",
             ROOT / "02_experiments/NCKUH_DMSA_metric_optimization_20261102/pilot_predictions.csv"]
    write_json(OUT / "run_lock.json", {"protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in paths}})
    print("Registered focal top-k patch MIL pilot", flush=True)


def require_lock():
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items(): assert digest(ROOT / name) == expected, name


def prepare_compact_bags(values, patch_aliases, groups, train_aliases, seed):
    from sklearn.decomposition import PCA
    sample = B.balanced_patch_sample(patch_aliases, train_aliases, seed)
    pca = PCA(PCA_WIDTH, whiten=True, random_state=seed, svd_solver="randomized").fit(values[sample])
    transformed = pca.transform(values).astype(np.float32)
    bags = np.zeros((len(groups), MAX_NODES, PCA_WIDTH), dtype=np.float32)
    valid = np.zeros((len(groups), MAX_NODES), dtype=bool)
    selected = []
    for i, group in enumerate(groups):
        if not len(group): raise ValueError("Empty patch bag")
        local = np.arange(len(group)) if len(group) <= MAX_NODES else np.unique(
            np.linspace(0, len(group)-1, MAX_NODES).round().astype(int))
        rows = group[local]
        bags[i, :len(rows)] = transformed[rows]; valid[i, :len(rows)] = True
        selected.append(rows.tolist())
    proof = {"train_aliases": sorted(train_aliases), "pca_fit_patch_indices": sample,
             "selected_transformed_rows": selected, "pca": pca}
    return torch.from_numpy(bags).cuda(), torch.from_numpy(valid).cuda(), proof


class FocalMIL(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in MODES: raise ValueError(mode)
        self.mode = mode
        self.project = nn.Linear(PCA_WIDTH, 32)
        self.dropout = nn.Dropout(.25)
        self.instance = nn.Linear(32, 1)

    def forward(self, values, valid):
        hidden = self.dropout(torch.relu(self.project(values)))
        instance = self.instance(hidden).squeeze(-1).masked_fill(~valid, -torch.inf)
        if self.mode == "top1_bce":
            bag = instance.max(1).values
        elif self.mode == "logmeanexp_bce":
            bag = torch.logsumexp(instance, dim=1) - valid.sum(1).float().log()
        else:
            top = instance.topk(min(4, instance.shape[1]), dim=1).values
            finite = torch.isfinite(top)
            bag = torch.where(finite, top, torch.zeros_like(top)).sum(1) / finite.sum(1).clamp_min(1)
        return bag, instance


def unilateral_pairs(indices, kidneys):
    location = {int(global_index): local for local, global_index in enumerate(indices)}
    pairs = []
    subset = kidneys.iloc[indices]
    for _, group in subset.groupby("patient_alias"):
        if len(group) != 2 or group.kidney_APN.nunique() != 2: continue
        positive = int(group.index[group.kidney_APN.eq(1)][0]); negative = int(group.index[group.kidney_APN.eq(0)][0])
        pairs.append((location[positive], location[negative]))
    return pairs


def fit_model(bags, valid, y, kidneys, fit, inner, mode, seed, epochs=None):
    torch.manual_seed(seed + 107); torch.cuda.manual_seed_all(seed + 107)
    model = FocalMIL(mode).cuda(); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    target = y[fit]
    weights = torch.where(target > 0, len(target)/(2*target.sum()), len(target)/(2*(len(target)-target.sum())))
    pairs = unilateral_pairs(fit, kidneys)
    best, chosen, stale = -float("inf"), 1, 0; history = []
    for epoch in range(1, (epochs if epochs is not None else 100)+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits, _ = model(bags[fit], valid[fit])
        bce = (nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")*weights).mean()
        rank = logits.sum()*0
        if mode == "top4_bce_rank" and pairs:
            pos = torch.tensor([p for p, _ in pairs], device="cuda")
            neg = torch.tensor([n for _, n in pairs], device="cuda")
            rank = nn.functional.softplus(-(logits[pos]-logits[neg])).mean()
        loss = bce + RANK_WEIGHT*rank
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        row = {"phase": "inner" if epochs is None else "refit", "epoch": epoch,
               "bce": float(bce.detach()), "rank": float(rank.detach()), "unilateral_pairs": len(pairs)}
        if inner is not None:
            model.eval()
            with torch.inference_mode(): score = torch.sigmoid(model(bags[inner], valid[inner])[0]).cpu().numpy()
            m = metric(y[inner].cpu().numpy().astype(int), score); criterion = (m["auroc"]+m["ap"])/2
            row.update({f"inner_{k}": v for k, v in m.items()})
            if criterion > best + 1e-12: best, chosen, stale = criterion, epoch, 0
            else: stale += 1
        history.append(row)
        if inner is not None and epoch >= 10 and stale >= 20: break
    return model.eval(), chosen if inner is not None else epochs, history


def run():
    from sklearn.model_selection import train_test_split
    from threadpoolctl import threadpool_limits
    require_lock()
    if (OUT / "development_predictions.csv").exists(): raise FileExistsError("Preserve completed pilot")
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    patients, kidneys, images = B.load_reference()
    values, patch_aliases, groups, global_rows = B.local_patch_data(ENCODER, images, kidneys)
    assignments = pd.read_csv(B.BASE / "patient_assignments.csv")
    y = torch.tensor(kidneys.kidney_APN.to_numpy(int), dtype=torch.float32, device="cuda")
    outputs, metric_rows, audits = [], [], []
    (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in SEEDS:
            mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
            outer = patients[patients.patient_alias.map(mapping).eq("train")]
            fit_alias, inner_alias = train_test_split(outer.patient_alias.to_numpy(), train_size=100,
                random_state=seed+4000, stratify=outer.DMSA_APN)
            fit = np.flatnonzero(kidneys.patient_alias.isin(fit_alias)); inner = np.flatnonzero(kidneys.patient_alias.isin(inner_alias))
            train = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("train")); dev = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
            inner_bags, inner_valid, inner_proof = prepare_compact_bags(values, patch_aliases, groups, fit_alias, seed)
            selected, histories = {}, {}
            for mode in MODES:
                model, epoch, history = fit_model(inner_bags, inner_valid, y, kidneys, fit, inner, mode, seed)
                selected[mode], histories[mode] = epoch, history; del model
            del inner_bags, inner_valid; torch.cuda.empty_cache()
            bags, valid, full_proof = prepare_compact_bags(values, patch_aliases, groups, outer.patient_alias.to_numpy(), seed)
            joblib.dump({"inner": inner_proof, "full": full_proof, "source_patch_rows": global_rows},
                        OUT / "models" / f"{seed}_preprocess.joblib", compress=3)
            for mode in MODES:
                model, epoch, refit = fit_model(bags, valid, y, kidneys, train, None, mode, seed, selected[mode])
                with torch.inference_mode():
                    logits, instances = model(bags[dev], valid[dev]); score = torch.sigmoid(logits).cpu().numpy()
                group = f"usf_focal_{mode}"; current = metric(y[dev].cpu().numpy().astype(int), score)
                metric_rows.append({"seed": seed, "group": group, "patients": 42, "kidneys": len(dev),
                                    "selected_epoch": epoch, **current})
                for i, value in zip(dev, score):
                    row = kidneys.iloc[i]
                    outputs.append({"seed": seed, "group": group, "patient_alias": row.patient_alias,
                                    "kidney_alias": row.kidney_alias, "DMSA_APN": row.DMSA_APN,
                                    "side": row.side, "label": int(row.kidney_APN), "score": float(value)})
                torch.save({"model": model.state_dict(), "mode": mode, "epoch": epoch},
                           OUT / "models" / f"{seed}_{group}.pth")
                pd.DataFrame(histories[mode]+refit).to_csv(OUT / f"history_{seed}_{group}.csv", index=False)
                np.savez_compressed(OUT / f"instances_{seed}_{group}.npz", kidney_indices=dev,
                                    valid=valid[dev].cpu().numpy(), instance_logits=instances.cpu().numpy())
                del model
            audits.append({"seed": seed, "selected_epochs": selected, "fit_patients": len(fit_alias),
                           "inner_patients": len(inner_alias), "train_patients": len(outer),
                           "pca_width": PCA_WIDTH, "max_nodes": MAX_NODES, "test_scored": False})
            print(f"Completed focal patch draw {seed}: {selected}", flush=True)
            del bags, valid; torch.cuda.empty_cache()
    predictions = pd.DataFrame(outputs)
    locked = pd.read_csv(ROOT / "02_experiments/NCKUH_DMSA_metric_optimization_20261102/pilot_predictions.csv")
    locked = locked[locked.group.eq(REFERENCE) & locked.seed.isin(SEEDS)].copy()
    focal = predictions[predictions.group.eq("usf_focal_top4_bce_rank")]
    merged = focal.merge(locked[["seed", "kidney_alias", "score"]], on=["seed", "kidney_alias"],
                         suffixes=("", "_reference"), validate="one_to_one")
    merged["group"] = "usf_focal_top4_bce_rank_plus_BCE_Adam"; merged["score"] = (merged.score+merged.score_reference)/2
    merged = merged[predictions.columns]; predictions = pd.concat([predictions, merged], ignore_index=True)
    for (seed, group), frame in merged.groupby(["seed", "group"]):
        metric_rows.append({"seed": seed, "group": group, "patients": frame.patient_alias.nunique(),
                            "kidneys": len(frame), **metric(frame.label, frame.score)})
    predictions.to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUT / "development_metrics.csv", index=False)
    write_json(OUT / "training_audits.json", audits)
    summary = pd.DataFrame(metric_rows).groupby("group")[["auroc", "ap", "auprc", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]; summary.to_csv(OUT / "development_summary.csv")
    all_predictions = pd.concat([predictions, locked[predictions.columns]], ignore_index=True)
    ci, differences, _ = cluster_intervals(all_predictions, [REFERENCE]+GROUPS, SEEDS, REFERENCE, 10000)
    ci.to_csv(OUT / "development_cluster_ci.csv", index=False); differences.to_csv(OUT / "development_cluster_differences.csv", index=False)
    print(summary[["auroc_mean", "ap_mean"]].to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=["prepare", "run"]); args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.command]()
