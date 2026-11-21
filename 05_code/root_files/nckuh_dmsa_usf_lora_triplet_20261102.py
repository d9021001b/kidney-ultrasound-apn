"""Kidney-level USF-MAE LoRA/triplet pilot under corrected DMSA side labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_kidney_baseline_20261102 import load_reference
from nckuh_lora_bilateral_pilot_20261102 import original_encoder
from nckuh_only_study_20261102 import digest, metric, write_json


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_usf_lora_triplet_20261102"
BASE = ROOT / "02_experiments/NCKUH_DMSA_kidney_baseline_20261102"
CACHE = ROOT / "02_experiments/NCKUH_lora_bilateral_pilot_20261102"
AUDIT = ROOT / "02_experiments/NCKUH_DMSA_label_audit_20261102"
SEEDS = [20261102, 20261103, 20261104]
ARMS = [
    "frozen_bce_aug", "frozen_triplet_aug",
    "lora_bce_aug", "lora_triplet_aug",
    "frozen_bce_noaug", "lora_bce_noaug",
]
MAX_EPOCHS = 50
MIN_EPOCHS = 10
PATIENCE = 10
TRIPLET_WEIGHT = 0.2


def arm_flags(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return {
        "adapt": arm.startswith("lora_"),
        "augmented": arm.endswith("_aug") and not arm.endswith("_noaug"),
        "triplet": "_triplet_" in arm,
    }


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered protocol")
    cache_provenance = json.loads((CACHE / "cache_provenance.json").read_text(encoding="utf-8"))
    assert cache_provenance["base_unchanged"] and cache_provenance["original_forward_maxabs"] <= 1e-6
    assignments = pd.read_csv(BASE / "patient_assignments.csv")
    assert set(assignments[assignments.seed.isin(SEEDS)].role) == {"train", "development", "exploratory_test"}
    assert [a for a in ARMS if arm_flags(a)["augmented"]] == ARMS[:4]
    assert [a for a in ARMS if arm_flags(a)["triplet"]] == ["frozen_triplet_aug", "lora_triplet_aug"]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Test corrected-label kidney-specific USF-MAE representation adaptation after fixed-feature heads plateaued",
        "endpoint": "One probability per imaged kidney using DMSA_APN R/L/B/N side truth; 414 kidneys from 210 patients",
        "seeds": SEEDS,
        "arms": ARMS,
        "encoder": "Official USF-MAE ViT-B/16. First eight blocks use the existing label-independent float32 token cache; last four blocks execute in bfloat16 for every arm",
        "adapter": "PEFT LoRA rank8 alpha16 dropout.05 on blocks8-11 qkv only. All pretrained tensors remain frozen",
        "head": "Shared image projection1536-to128; within-kidney view mean; ordered own, opposite, own-minus-opposite, opposite-present context; shared side-exchange-equivariant classifier. Matched arms reset to the same head initialization seed after any adapter construction",
        "augmentation": "For augmented arms, one of original, rotation-2deg/gamma.9, rotation+2deg/gamma1.1 is selected per patient per training visit and applied coherently to both kidneys/all views. Evaluation uses original only",
        "triplet": "Batch-hard kidney embeddings: positives share label but must be from a different patient; a same-patient opposite-label kidney is the preferred hard negative; margin.2 TripletMarginLoss weighted .2",
        "selection": "Within each fixed 126-patient training role, stratified 100/26 patient inner split. Select earliest epoch maximizing mean(inner kidney AUROC, AP), minimum10, maximum50, patience10; reset and refit all126 for that epoch",
        "evaluation": "Development-only 42-patient role. Exploratory-test roles remain unscored. Same patient never crosses a role",
        "promotion": "Relative gate: mean LoRA-minus-matched-frozen >=.03 AUROC and >=.02 AP with joint improvement in >=2/3 draws. Absolute signal gate: both mean AUROC and AP >.8 and both >.8 in >=2/3 draws. Only a passing arm may expand to ten draws",
        "uncertainty": "10000 R/L/B/N-stratified patient-cluster fixed-prediction resamples across all three development draws",
        "limitations": "Exploratory development experiment on previously examined patients; no lesion-patch labels, no test-derived tuning, and no claim that triplet embeddings localize APN",
    }
    write_json(OUT / "protocol.json", protocol)
    locked = [
        Path(__file__), BASE / "patient_assignments.csv", CACHE / "cache_provenance.json",
        CACHE / "lower8_float32.npy", CACHE / "patch_selection.npy", CACHE / "frozen_embedding.npy",
        AUDIT / "patient_label_linkage_private.csv", AUDIT / "kidney_label_linkage_private.csv",
        AUDIT / "image_labels_dmsa_private.csv",
    ]
    write_json(OUT / "pre_run_lock.json", {
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in locked},
    })
    print("Registered corrected-label USF-MAE LoRA/triplet pilot", flush=True)


def verify_lock():
    lock = json.loads((OUT / "pre_run_lock.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(ROOT / name) == expected, name


def load_design():
    patients, kidneys, images = load_reference()
    patient_lookup = {a: i for i, a in enumerate(patients.patient_alias)}
    width = int(images[images.patient_alias.isin(patient_lookup)].groupby("patient_alias").size().max())
    image_index = np.full((len(patients), width), -1, dtype=np.int64)
    image_side = np.full((len(patients), width), -1, dtype=np.int64)
    for alias, group in images[images.patient_alias.isin(patient_lookup)].groupby("patient_alias", sort=False):
        i = patient_lookup[alias]
        rows = group.index.to_numpy(np.int64)
        image_index[i, :len(rows)] = rows
        image_side[i, :len(rows)] = group.side.map({"left": 0, "right": 1}).to_numpy(np.int64)
    labels = np.full((len(patients), 2), -1, dtype=np.int64)
    present = np.zeros((len(patients), 2), dtype=bool)
    for row in kidneys.itertuples(index=False):
        i, side = patient_lookup[row.patient_alias], 0 if row.side == "left" else 1
        labels[i, side] = int(row.kidney_APN)
        present[i, side] = True
    assert present.sum() == 414 and (labels[present] == kidneys.kidney_APN.to_numpy(int)).all()
    assert set(images.image_id.iloc[image_index[image_index >= 0]]) == set(images[images.kidney_APN.notna()].image_id)
    return patients, kidneys, images, image_index, image_side, labels, present


def make_head(torch):
    from torch import nn

    class OrderedKidneyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 128), nn.ReLU())
            self.classifier = nn.Sequential(nn.Linear(385, 16), nn.ReLU(), nn.Dropout(.25), nn.Linear(16, 1))

        def forward(self, values, sides):
            import torch.nn.functional as F
            node = F.normalize(self.project(values.float()), dim=-1)
            kidney = []
            exists = []
            for side in (0, 1):
                mask = sides.eq(side)
                pooled = (node * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
                pooled = F.normalize(pooled, dim=-1) * mask.any(1)[:, None]
                kidney.append(pooled)
                exists.append(mask.any(1))
            kidney = torch.stack(kidney, dim=1)
            exists = torch.stack(exists, dim=1)
            logits = []
            for side in (0, 1):
                own, opposite = kidney[:, side], kidney[:, 1-side]
                context = torch.cat([own, opposite, own-opposite, exists[:, 1-side, None].float()], dim=1)
                logits.append(self.classifier(context).squeeze(1))
            return torch.stack(logits, dim=1), kidney, exists

    return OrderedKidneyHead().cuda()


def add_lora(encoder):
    from peft import LoraConfig, inject_adapter_in_model
    targets = [f"blocks.{i}.attn.qkv" for i in range(8, 12)]
    encoder = inject_adapter_in_model(LoraConfig(
        r=8, lora_alpha=16, lora_dropout=.05, bias="none", target_modules=targets), encoder)
    trainable = [name for name, p in encoder.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name and any(f"blocks.{i}." in name for i in range(8, 12)) for name in trainable)
    return encoder


def batch_triplet_loss(torch, embedding, labels, present, patient_indices):
    import torch.nn.functional as F
    valid = present.reshape(-1)
    z = embedding.reshape(-1, embedding.shape[-1])[valid]
    y = labels.reshape(-1)[valid]
    pid = patient_indices[:, None].expand(-1, 2).reshape(-1)[valid]
    if len(z) < 3:
        return z.sum() * 0, 0
    distance = torch.cdist(z, z)
    anchors, positives, negatives = [], [], []
    for i in range(len(z)):
        pos = (y == y[i]) & (pid != pid[i])
        neg = y != y[i]
        if not pos.any() or not neg.any():
            continue
        pos_index = torch.where(pos)[0][distance[i, pos].argmax()]
        same_patient_neg = neg & (pid == pid[i])
        pool = same_patient_neg if same_patient_neg.any() else neg
        neg_index = torch.where(pool)[0][distance[i, pool].argmin()]
        anchors.append(z[i]); positives.append(z[pos_index]); negatives.append(z[neg_index])
    if not anchors:
        return z.sum() * 0, 0
    loss = F.triplet_margin_loss(torch.stack(anchors), torch.stack(positives), torch.stack(negatives),
                                 margin=.2, p=2, eps=1e-6, swap=False)
    return loss, len(anchors)


def build_runtime(seed, arm):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    encoder = original_encoder()
    adapt = arm_flags(arm)["adapt"]
    if adapt:
        torch.manual_seed(seed + 17)
        torch.cuda.manual_seed_all(seed + 17)
        encoder = add_lora(encoder)
    # Adapter construction consumes random numbers. Reset before the head so
    # matched frozen/LoRA arms start from identical head parameters.
    torch.manual_seed(seed + 23)
    torch.cuda.manual_seed_all(seed + 23)
    head = make_head(torch)
    parameters = [{"params": list(head.parameters()), "lr": 1e-3}]
    if adapt:
        parameters.append({"params": [p for p in encoder.parameters() if p.requires_grad], "lr": 1e-4})
    optimizer = torch.optim.AdamW(parameters, weight_decay=1e-3)
    base_snapshot = {name: p.detach().clone() for name, p in encoder.named_parameters() if "lora_" not in name}
    return encoder, head, optimizer, base_snapshot


def assert_base_unchanged(torch, encoder, snapshot):
    current = dict(encoder.named_parameters())
    assert set(snapshot).issubset(current)
    assert all(torch.equal(value, current[name].detach()) for name, value in snapshot.items())


def run_training(seed, arm, fit_patients, score_patients, epochs, select_epoch,
                 lower, selections, design):
    import torch
    from torch import nn
    patients, kidneys, images, image_index, image_side, labels, present = design
    flags = arm_flags(arm)
    adapt, augmented, use_triplet = flags["adapt"], flags["augmented"], flags["triplet"]
    encoder, head, optimizer, base_snapshot = build_runtime(seed, arm)
    train_labels = labels[fit_patients][present[fit_patients]]
    positives, negatives = int(train_labels.sum()), int((train_labels == 0).sum())
    class_weight = torch.tensor([len(train_labels)/(2*negatives), len(train_labels)/(2*positives)],
                                dtype=torch.float32, device="cuda")
    rng = np.random.default_rng(seed + 931)
    records, best, stale = [], None, 0

    def forward(index, variants, training):
        idx = image_index[index]
        valid = idx >= 0
        flattened = idx[valid]
        per_image_variant = np.broadcast_to(variants[:, None], idx.shape)[valid]
        token = torch.from_numpy(np.array(lower[per_image_variant, flattened])).cuda(non_blocking=True)
        chosen = torch.from_numpy(np.array(selections[per_image_variant, flattened])).cuda(non_blocking=True)
        encoder.train(training and adapt)
        context = torch.enable_grad() if adapt and training else torch.no_grad()
        with context:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for block in encoder.blocks[8:]:
                    token = block(token)
                token = encoder.norm(token)
                patch_mean = (token[:, 1:] * chosen[..., None]).sum(1) / chosen.sum(1)[:, None]
                embedding = torch.cat([token[:, 0], patch_mean], dim=1)
        padded = torch.zeros(len(index) * image_index.shape[1], 1536, device="cuda", dtype=embedding.dtype)
        locations = torch.from_numpy(np.flatnonzero(valid.ravel())).cuda()
        padded = padded.index_copy(0, locations, embedding)
        sides = torch.from_numpy(image_side[index]).cuda()
        return head(padded.reshape(len(index), image_index.shape[1], 1536), sides)

    def score(index):
        head.eval(); encoder.eval()
        logits_all, rows = [], []
        with torch.inference_mode():
            for start in range(0, len(index), 16):
                batch = index[start:start+16]
                logits, _, _ = forward(batch, np.zeros(len(batch), dtype=np.int64), False)
                logits_all.append(torch.sigmoid(logits).cpu().numpy())
        values = np.concatenate(logits_all)
        y, s = labels[index][present[index]], values[present[index]]
        return metric(y, s), values

    start_time = time.monotonic()
    for epoch in range(1, epochs + 1):
        head.train(); encoder.train(adapt)
        total_bce = total_triplet = 0.0; total_anchors = total_kidneys = 0
        for offset in range(0, len(fit_patients), 16):
            batch = rng.permutation(fit_patients) if offset == 0 else order
            order = batch
            index = order[offset:offset+16]
            variants = rng.integers(0, 3, len(index)) if augmented else np.zeros(len(index), dtype=np.int64)
            optimizer.zero_grad(set_to_none=True)
            logits, embedding, exists = forward(index, variants, True)
            target = torch.from_numpy(labels[index]).cuda()
            valid = torch.from_numpy(present[index]).cuda()
            bce = nn.functional.binary_cross_entropy_with_logits(logits[valid], target[valid].float(), reduction="none")
            bce = (bce * class_weight[target[valid]]).mean()
            triplet, anchors = batch_triplet_loss(torch, embedding, target, valid,
                                                  torch.from_numpy(index).cuda())
            loss = bce + (TRIPLET_WEIGHT * triplet if use_triplet else 0)
            loss.backward()
            nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]], 5.)
            optimizer.step()
            n = int(valid.sum()); total_kidneys += n
            total_bce += float(bce.detach()) * n
            total_triplet += float(triplet.detach()) * max(anchors, 1)
            total_anchors += anchors
        row = {"epoch": epoch, "bce": total_bce/total_kidneys,
               "triplet": total_triplet/max(total_anchors, 1), "triplet_anchors": total_anchors}
        if select_epoch:
            inner_metric, _ = score(score_patients)
            objective = (inner_metric["auroc"] + inner_metric["ap"]) / 2
            row.update({f"inner_{k}": v for k, v in inner_metric.items()})
            row["selection_objective"] = objective
            if epoch >= MIN_EPOCHS and (best is None or objective > best["objective"] + 1e-12):
                best = {"epoch": epoch, "objective": objective, **inner_metric}; stale = 0
            elif epoch >= MIN_EPOCHS:
                stale += 1
            if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                records.append(row)
                break
        records.append(row)
        if epoch % 10 == 0:
            print(f"{arm} seed{seed} {'inner' if select_epoch else 'refit'} epoch{epoch} "
                  f"bce={row['bce']:.4f} elapsed={time.monotonic()-start_time:.0f}s", flush=True)
    assert_base_unchanged(torch, encoder, base_snapshot)
    if select_epoch:
        assert best is not None
        result = {"selected_epoch": best["epoch"], "inner": best, "records": records,
                  "fit_seconds": time.monotonic()-start_time, "base_unchanged": True}
        del encoder, head, optimizer, base_snapshot
        torch.cuda.empty_cache()
        return result
    dev_metric, values = score(score_patients)
    model = {"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
             "adapter": {k: v.detach().cpu() for k, v in encoder.state_dict().items() if "lora_" in k},
             "seed": seed, "arm": arm, "epochs": epochs, "base_unchanged": True}
    result = {"metrics": dev_metric, "scores": values, "records": records,
              "fit_seconds": time.monotonic()-start_time, "base_unchanged": True,
              "trainable_parameters": sum(p.numel() for g in optimizer.param_groups for p in g["params"])}
    del encoder, head, optimizer, base_snapshot
    torch.cuda.empty_cache()
    return result, model


def preflight(lower, selections):
    import torch
    torch.set_num_threads(4)
    frozen = np.load(CACHE / "frozen_embedding.npy", mmap_mode="r")
    encoder = original_encoder()
    # Reproduce the original cache writer's contiguous batch shape. Sparse row
    # gathering changes the CUDA GEMM reduction order by about 2e-6 even though
    # every token is independent.
    rows = np.arange(32)
    token = torch.from_numpy(np.array(lower[0, rows])).cuda()
    selected = torch.from_numpy(np.array(selections[0, rows])).cuda()
    with torch.inference_mode():
        tail = token
        for block in encoder.blocks[8:]: tail = block(tail)
        tail = encoder.norm(tail)
        mean = (tail[:, 1:] * selected[..., None]).sum(1) / selected.sum(1)[:, None]
        embedding = torch.cat([tail[:, 0], mean], 1).cpu().numpy()
    original_error = float(np.max(np.abs(embedding - frozen[0, rows])))
    lora = add_lora(original_encoder()).eval()
    with torch.inference_mode():
        adapted = token
        for block in lora.blocks[8:]: adapted = block(adapted)
        adapted = lora.norm(adapted)
    zero_lora_error = float((adapted-tail).abs().max().cpu())
    report = {"rows": rows.tolist(), "float32_tail_vs_frozen_maxabs": original_error,
              "zero_init_lora_vs_original_maxabs": zero_lora_error,
              "lower_shape": list(lower.shape), "selection_shape": list(selections.shape),
              "bf16_supported": bool(torch.cuda.is_bf16_supported()),
              "gpu": torch.cuda.get_device_name(0)}
    assert original_error <= 1e-6 and zero_lora_error <= 1e-6 and report["bf16_supported"]
    write_json(OUT / "preflight.json", report)
    del encoder, lora, token, tail, adapted
    torch.cuda.empty_cache()
    print(json.dumps(report), flush=True)


def run():
    import torch
    from sklearn.model_selection import train_test_split
    verify_lock()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    lower = np.load(CACHE / "lower8_float32.npy", mmap_mode="r")
    selections = np.load(CACHE / "patch_selection.npy", mmap_mode="r")
    assert lower.shape == (3, 1027, 197, 768) and selections.shape == (3, 1027, 196)
    if not (OUT / "preflight.json").exists(): preflight(lower, selections)
    design = load_design()
    patients, kidneys, images, image_index, image_side, labels, present = design
    assignments = pd.read_csv(BASE / "patient_assignments.csv")
    (OUT / "models").mkdir(exist_ok=True)
    predictions, metrics_rows, training_rows = [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        roles = patients.patient_alias.map(mapping)
        train = np.flatnonzero(roles.eq("train")); dev = np.flatnonzero(roles.eq("development"))
        inner_train, inner_dev = train_test_split(train, train_size=100, random_state=seed+4000,
                                                  stratify=patients.DMSA_APN.to_numpy()[train])
        assert len(inner_train) == 100 and len(inner_dev) == 26 and not set(inner_train) & set(inner_dev)
        for arm in ARMS:
            result_path = OUT / f"result_{seed}_{arm}.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                metrics_rows.append(saved["metrics"]); predictions.extend(saved["predictions"])
                continue
            selection = run_training(seed, arm, inner_train, inner_dev, MAX_EPOCHS, True,
                                     lower, selections, design)
            selected_epoch = int(selection["selected_epoch"])
            final, model = run_training(seed, arm, train, dev, selected_epoch, False,
                                        lower, selections, design)
            import torch
            torch.save(model, OUT / "models" / f"{seed}_{arm}.pth")
            scores = final.pop("scores")
            dev_rows = []
            for local, patient_index in enumerate(dev):
                for side in (0, 1):
                    if not present[patient_index, side]: continue
                    match = kidneys[(kidneys.patient_alias == patients.patient_alias.iloc[patient_index]) &
                                    (kidneys.side == ("left" if side == 0 else "right"))].iloc[0]
                    dev_rows.append({"seed": seed, "group": arm, "patient_alias": match.patient_alias,
                                     "kidney_alias": match.kidney_alias, "DMSA_APN": match.DMSA_APN,
                                     "side": match.side, "label": int(labels[patient_index, side]),
                                     "score": float(scores[local, side])})
            metric_row = {"seed": seed, "group": arm, "patients": len(dev), "kidneys": len(dev_rows),
                          "selected_epoch": selected_epoch, **final["metrics"]}
            payload = {"metrics": metric_row, "predictions": dev_rows, "selection": selection,
                       "refit": {k: v for k, v in final.items() if k != "records"},
                       "training_records": final["records"]}
            write_json(result_path, payload)
            metrics_rows.append(metric_row); predictions.extend(dev_rows)
            training_rows.append({"seed": seed, "group": arm, "selected_epoch": selected_epoch,
                                  "inner_objective": selection["inner"]["objective"],
                                  "inner_auroc": selection["inner"]["auroc"], "inner_ap": selection["inner"]["ap"],
                                  "fit_seconds": selection["fit_seconds"] + final["fit_seconds"],
                                  "trainable_parameters": final["trainable_parameters"], "base_unchanged": True})
            print(json.dumps(metric_row), flush=True)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(training_rows).to_csv(OUT / "training_records.csv", index=False)
    summary = pd.DataFrame(metrics_rows).groupby("group")[["auroc", "ap", "auprc", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(OUT / "development_summary.csv")
    ci, differences, _ = cluster_intervals(pd.DataFrame(predictions), ARMS, SEEDS,
                                           reference="frozen_bce_aug", replicates=10000)
    ci.to_csv(OUT / "development_cluster_ci.csv", index=False)
    differences.to_csv(OUT / "development_cluster_differences.csv", index=False)
    print(summary[["auroc_mean", "ap_mean"]].to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.command]()
