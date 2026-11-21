"""Training-only early stopping and L2-SP comparison for the prior triplet probe.

This separate batch tests a constrained explanation for the prior fine-tuned
triplet overfit: regularize pretrained DenseNet tail parameters toward their
RadImageNet initialization (L2-SP) rather than toward zero.  It is still an
exploratory follow-up on the previously explored 98-patient cohort.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import warnings
import zipfile

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

import renal_dinov3_native_probe as N
import renal_interpretable_probe as I
import renal_local_features as L
import renal_random_validation as R
import renal_triplet_probe as T
import renal_voting_validation as V
from radimagenet_backbones import RADIMAGENET_DIR, radimagenet_densenet121

ROOT = R.ROOT
TRIPLET = T.OUT
OUT = ROOT / "CYCH_l2sp_earlystop_probe_20260913"
CONFIGS = {
    "F-CE-ZERO": dict(objective="CE", l2sp=False),
    "F-CE-L2SP": dict(objective="CE", l2sp=True),
    "F-CE-T-ZERO": dict(objective="CE-T", l2sp=False),
    "F-CE-T-L2SP": dict(objective="CE-T", l2sp=True),
}
MAX_EPOCHS = 200
MIN_EPOCHS = 20
PATIENCE = 20
INNER_VALIDATION_PATIENTS = 12
HEAD_LR = 1e-3
TAIL_LR = 1e-5
WEIGHT_DECAY = 1e-4
L2SP_LAMBDA = 1e-4
MARGIN = .2

REFERENCES = [
    {
        "id": "li2018",
        "citation": "Li X, Grandvalet Y, Davoine F. Explicit Inductive Bias for Transfer Learning with Convolutional Networks. ICML. 2018.",
        "url": "https://proceedings.mlr.press/v80/li18a.html",
        "scope": "L2-SP regularizes pretrained parameters toward their initial pretrained values. This is a controlled adaptation, not proof that the prior result reflected catastrophic forgetting.",
    },
    {
        "id": "radimagenet",
        "citation": "Mei X, et al. RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning. Radiol Artif Intell. 2022. doi:10.1148/ryai.210315.",
        "url": "https://github.com/BMEII-AI/RadImageNet",
        "scope": "Existing frozen-prefix and initialized DenseNet-121 source.",
    },
    {
        "id": "triplet",
        "citation": "Schroff F, Kalenichenko D, Philbin J. FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR. 2015.",
        "url": "https://arxiv.org/abs/1503.03832",
        "scope": "Triplet loss source; the present patient-centroid implementation is not an exact FaceNet replication.",
    },
]


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name, dtype={"patient_id": str})


def source_files() -> list[Path]:
    names = (
        "renal_l2sp_earlystop_probe.py",
        "renal_triplet_probe.py",
        "renal_random_validation.py",
        "renal_dinov3_native_probe.py",
        "renal_interpretable_probe.py",
        "renal_local_features.py",
        "renal_voting_validation.py",
        "radimagenet_backbones.py",
    )
    return [ROOT / name for name in names]


def require_lock() -> None:
    lock = json.loads((OUT / "protocol_locked.json").read_text(encoding="utf-8"))
    if lock["protocol_sha256"] != R.sha256(OUT / "protocol.json"):
        raise RuntimeError("Protocol hash mismatch")
    for path, digest in lock["sources"].items():
        if R.sha256(Path(path)) != digest:
            raise RuntimeError(f"Locked source changed: {path}")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    for path, digest in protocol["inputs"].items():
        if R.sha256(Path(path)) != digest:
            raise RuntimeError(f"Locked input changed: {path}")


def outer_inner_split(patients: pd.DataFrame, outer_train: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One stratified 46/12 split drawn only from an outer training role."""
    outer_train = np.asarray(outer_train, dtype=int)
    y = patients.label.to_numpy(int)[outer_train]
    train_local, inner_local = next(
        StratifiedShuffleSplit(n_splits=1, test_size=INNER_VALIDATION_PATIENTS, random_state=seed + 4000).split(outer_train, y)
    )
    fit, inner = outer_train[train_local], outer_train[inner_local]
    if len(fit) != 46 or len(inner) != INNER_VALIDATION_PATIENTS:
        raise RuntimeError("Unexpected 46/12 inner split size")
    if set(fit) & set(inner) or set(np.concatenate((fit, inner))) != set(outer_train):
        raise RuntimeError("Inner split is not a partition of outer training patients")
    return fit, inner


def l2sp_penalty(tail: nn.Module, anchor: dict[str, torch.Tensor], enabled: bool) -> torch.Tensor:
    if not enabled:
        return next(tail.parameters()).new_zeros(())
    penalty = next(tail.parameters()).new_zeros(())
    for name, parameter in tail.named_parameters():
        penalty = penalty + torch.sum((parameter - anchor[name]) ** 2)
    return penalty


def patient_rows(image_manifest: pd.DataFrame, aliases: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(image_manifest.patient_alias.isin(aliases))
    if len(rows) == 0:
        raise RuntimeError("No image rows for a selected patient subset")
    return rows


def make_bags(image_manifest: pd.DataFrame, patients: pd.DataFrame, patient_indices: np.ndarray, device: torch.device) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    aliases = patients.patient_alias.iloc[patient_indices].to_numpy()
    rows = patient_rows(image_manifest, aliases)
    bags = T.bag_matrix(image_manifest.patient_alias.iloc[rows], aliases).to(device)
    labels = torch.from_numpy(patients.label.iloc[patient_indices].to_numpy(int)).to(device)
    return rows, bags, labels


def build_network(original_tail: nn.Module, seed: int, l2sp: bool, device: torch.device) -> tuple[nn.Module, nn.Module, dict[str, torch.Tensor], torch.optim.Optimizer, dict]:
    torch.manual_seed(seed + 37)
    head = T.Projection(1024).to(device)
    tail = copy.deepcopy(original_tail).to(device).eval().requires_grad_(True)
    anchor = {name: value.detach().clone() for name, value in tail.named_parameters()}
    parameters = [
        dict(params=head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY),
        dict(params=tail.parameters(), lr=TAIL_LR, weight_decay=0.0 if l2sp else WEIGHT_DECAY),
    ]
    optimizer = torch.optim.Adam(parameters)
    initial = dict(head=L.model_digest(head), tail=L.model_digest(tail))
    return head, tail, anchor, optimizer, initial


def training_loss(
    head: nn.Module,
    tail: nn.Module,
    anchor: dict[str, torch.Tensor],
    x: torch.Tensor,
    bags: torch.Tensor,
    labels: torch.Tensor,
    objective: str,
    l2sp: bool,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict]:
    image_embeddings = head(tail(x))
    patient_embeddings = F.normalize(bags @ image_embeddings, dim=1)
    weights = torch.tensor([len(labels) / (2 * int((labels == label).sum())) for label in labels], device=x.device)
    ce = (F.binary_cross_entropy_with_logits(head.classifier(patient_embeddings).squeeze(1), labels.float(), reduction="none") * weights).mean()
    triplet = patient_embeddings.new_zeros(())
    active, semi_fraction = 0.0, 0.0
    if objective == "CE-T":
        criterion = nn.TripletMarginLoss(margin=MARGIN, p=2, eps=1e-6, swap=False, reduction="none")
        positive, negative, semi = T.mine_triplets(patient_embeddings, labels, generator, margin=MARGIN)
        values = criterion(patient_embeddings, patient_embeddings[positive], patient_embeddings[negative])
        triplet = (values * weights).mean()
        active, semi_fraction = float((values > 0).float().mean()), float(np.mean(semi))
    elif objective != "CE":
        raise ValueError(objective)
    anchor_penalty = l2sp_penalty(tail, anchor, l2sp)
    total = ce + triplet + L2SP_LAMBDA * anchor_penalty
    if not torch.isfinite(total):
        raise RuntimeError("Non-finite training loss")
    return total, dict(ce=float(ce.detach()), triplet=float(triplet.detach()), l2sp_penalty=float(anchor_penalty.detach()), active_fraction=active, semi_hard_fraction=semi_fraction)


def patient_validation_bce(head: nn.Module, tail: nn.Module, x: torch.Tensor, bags: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.inference_mode():
        embeddings = F.normalize(bags @ head(tail(x)), dim=1)
        value = F.binary_cross_entropy_with_logits(head.classifier(embeddings).squeeze(1), labels.float())
    return float(value.cpu())


def run_inner_early_stopping(
    maps: torch.Tensor,
    image_manifest: pd.DataFrame,
    patients: pd.DataFrame,
    fit_indices: np.ndarray,
    inner_indices: np.ndarray,
    original_tail: nn.Module,
    seed: int,
    objective: str,
    l2sp: bool,
) -> tuple[int, list[dict], dict]:
    device = maps.device
    fit_rows, fit_bags, fit_y = make_bags(image_manifest, patients, fit_indices, device)
    inner_rows, inner_bags, inner_y = make_bags(image_manifest, patients, inner_indices, device)
    head, tail, anchor, optimizer, initial = build_network(original_tail, seed, l2sp, device)
    buffers = {name: value.clone() for name, value in tail.named_buffers()}
    generator = torch.Generator().manual_seed(seed + 97)
    history, best_epoch, best_value, bad_epochs = [], 1, float("inf"), 0
    started = time.monotonic()
    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, details = training_loss(head, tail, anchor, maps[fit_rows], fit_bags, fit_y, objective, l2sp, generator)
        loss.backward()
        nn.utils.clip_grad_norm_([parameter for group in optimizer.param_groups for parameter in group["params"]], 5)
        optimizer.step()
        inner_bce = patient_validation_bce(head.eval(), tail.eval(), maps[inner_rows], inner_bags, inner_y)
        improved = inner_bce < best_value - 1e-12
        if improved:
            best_value, best_epoch, bad_epochs = inner_bce, epoch, 0
        else:
            bad_epochs += 1
        history.append(dict(phase="inner", epoch=epoch, train_loss=float(loss.detach()), inner_validation_bce=inner_bce, best_epoch=best_epoch, **details))
        if epoch >= MIN_EPOCHS and bad_epochs >= PATIENCE:
            break
    if not all(torch.equal(value, buffers[name]) for name, value in tail.named_buffers()):
        raise RuntimeError("BatchNorm running buffers changed during inner training")
    audit = dict(
        initial=initial,
        final_head=L.model_digest(head),
        final_tail=L.model_digest(tail),
        inner_epochs=len(history),
        selected_epoch=best_epoch,
        selected_inner_validation_bce=best_value,
        stopped_early=bool(len(history) < MAX_EPOCHS),
        elapsed_seconds=time.monotonic() - started,
        batchnorm_buffers_unchanged=True,
    )
    del head, tail, optimizer
    torch.cuda.empty_cache()
    return best_epoch, history, audit


def refit_full_outer_train(
    maps: torch.Tensor,
    image_manifest: pd.DataFrame,
    patients: pd.DataFrame,
    outer_train: np.ndarray,
    original_tail: nn.Module,
    seed: int,
    objective: str,
    l2sp: bool,
    epochs: int,
) -> tuple[nn.Module, nn.Module, list[dict], dict]:
    device = maps.device
    rows, bags, labels = make_bags(image_manifest, patients, outer_train, device)
    head, tail, anchor, optimizer, initial = build_network(original_tail, seed, l2sp, device)
    buffers = {name: value.clone() for name, value in tail.named_buffers()}
    generator = torch.Generator().manual_seed(seed + 97)
    history, started = [], time.monotonic()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, details = training_loss(head, tail, anchor, maps[rows], bags, labels, objective, l2sp, generator)
        loss.backward()
        nn.utils.clip_grad_norm_([parameter for group in optimizer.param_groups for parameter in group["params"]], 5)
        optimizer.step()
        history.append(dict(phase="refit", epoch=epoch, train_loss=float(loss.detach()), **details))
    if not all(torch.equal(value, buffers[name]) for name, value in tail.named_buffers()):
        raise RuntimeError("BatchNorm running buffers changed during outer-train refit")
    final_tail = L.model_digest(tail)
    if final_tail == initial["tail"]:
        raise RuntimeError("Fine-tuned tail did not change")
    drift = float(sum(torch.sum((parameter - anchor[name]) ** 2).detach().cpu() for name, parameter in tail.named_parameters()))
    audit = dict(
        initial=initial,
        final_head=L.model_digest(head),
        final_tail=final_tail,
        outer_train_epochs=epochs,
        final_tail_squared_distance_from_pretrained=drift,
        l2sp_enabled=l2sp,
        l2sp_lambda=L2SP_LAMBDA if l2sp else 0.0,
        batchnorm_buffers_unchanged=True,
        elapsed_seconds=time.monotonic() - started,
    )
    return head.eval(), tail.eval(), history, audit


def prepare() -> None:
    if (OUT / "protocol_locked.json").exists():
        require_lock()
        print("Existing L2-SP protocol verified", flush=True)
        return
    for directory in ("source", "features", "models", "runs", "qa", "figures_600dpi"):
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv"):
        shutil.copy2(TRIPLET / name, OUT / name)
    image_manifest, patients, split = read("image_manifest.csv"), read("patients.csv"), read("random_patient_assignments.csv")
    if len(image_manifest) != 349 or len(patients) != 98 or int(patients.label.sum()) != 52:
        raise RuntimeError("Unexpected inherited cohort")
    if image_manifest.image_id.nunique() != 349 or image_manifest.source_sha256.duplicated().any():
        raise RuntimeError("Unexpected duplicated inherited image")
    inner_records = []
    for seed in R.SEEDS:
        outer = N.indices_for(patients, split, seed)["train"]
        fit, inner = outer_inner_split(patients, outer, seed)
        inner_records.extend(
            dict(seed=seed, outer_role="train", inner_role=role, patient_alias=patients.patient_alias.iloc[index], label=int(patients.label.iloc[index]))
            for role, indices in (("fit", fit), ("inner_validation", inner))
            for index in indices
        )
    pd.DataFrame(inner_records).to_csv(OUT / "inner_training_assignments.csv", index=False)
    inputs = [
        OUT / "image_manifest.csv",
        OUT / "patients.csv",
        OUT / "random_patient_assignments.csv",
        OUT / "inner_training_assignments.csv",
        TRIPLET / "cache" / "prefix_maps.npy",
        TRIPLET / "cache_audit.json",
        RADIMAGENET_DIR / "DenseNet121.pt",
    ]
    protocol = dict(
        design="Further retrospective exploratory patient-level random-sampling experiment on the same previously explored CYCH cohort. No k-fold, external, or untouched test claim.",
        cohort=dict(patients=98, images=349, positive=52, negative=46),
        seeds=R.SEEDS,
        roles=dict(outer_train=58, development=20, exploratory_test=20, inner_fit=46, inner_validation=12),
        configs=CONFIGS,
        cached_input="Reuse hash-locked frozen DenseNet prefix maps from the completed triplet ablation. Original P1-P99 masked images and masks are unchanged. No new image augmentation, segmentation, feature generation, clinical covariate, GAN, or external data.",
        fine_tuning="Denseblock4 and norm5 affine parameters are fine-tuned. Prefix through transition3 and all BatchNorm running buffers remain frozen/eval.",
        representation="Linear 1024-to-128 projection with L2-normalized image and patient-centroid embeddings. Downstream patient vector is image embedding mean plus population standard deviation (256 features).",
        objectives=dict(
            CE="Patient-balanced binary cross entropy on patient centroids.",
            CE_T="Unweighted sum of the same CE and patient-balanced triplet loss; random same-class distinct-patient positive, semi-hard opposite-class negative otherwise random fallback; margin 0.2, p=2, eps=1e-6.",
        ),
        regularization=dict(
            zero="Adam weight decay 1e-4 on the new projection and pretrained tail.",
            l2sp="Adam weight decay 1e-4 on the new projection; no tail zero-decay. Add 1e-4 times the sum of squared difference between every fine-tuned tail parameter and its original RadImageNet value.",
        ),
        early_stopping=dict(
            inner_split="One stratified 46/12 patient partition of the 58 outer-training patients, fixed seed+4000.",
            selection_metric="Unweighted patient-level binary cross entropy on the inner 12 patients.",
            max_epochs=MAX_EPOCHS,
            minimum_epochs_before_stop=MIN_EPOCHS,
            patience=PATIENCE,
            refit="After selecting one epoch only from the inner validation patients, reset the network and refit on all 58 outer-training patients exactly for that epoch count. Development/test labels and scores never choose the epoch.",
        ),
        head="N.full_head: train-only median imputer, variance filter, standardizer, balanced liblinear logistic regression C=0.1/max_iter=2000.",
        decision="Compare L2-SP only against the matched zero-centered configuration using development data. This batch fits no new voting model. Any later voter requires a separately locked candidate test.",
        caveats=[
            "All 98 patients have already been explored and random draws overlap across roles.",
            "L2-SP is a causal-method test, not proof of the cause of prior generalization failure.",
            "The 12-patient inner validation selection can itself be unstable; its roles and loss histories are retained.",
        ],
        inputs={str(path): R.sha256(path) for path in inputs},
        references=REFERENCES,
    )
    R.write_json(OUT / "protocol.json", protocol)
    sources = {str(path): R.sha256(path) for path in source_files()}
    R.write_json(OUT / "protocol_locked.json", dict(utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=R.sha256(OUT / "protocol.json"), sources=sources))
    for path in source_files():
        shutil.copy2(path, OUT / "source" / path.name)
    R.write_json(OUT / "references.json", REFERENCES)
    print("Locked four-config L2-SP / early-stopping protocol before outcome evaluation", flush=True)


def train() -> None:
    require_lock()
    T.configure_torch()
    device = torch.device("cuda")
    patients, image_manifest, split = read("patients.csv"), read("image_manifest.csv"), read("random_patient_assignments.csv")
    maps = torch.from_numpy(np.load(TRIPLET / "cache" / "prefix_maps.npy")).to(device)
    if maps.shape != (349, 512, 7, 7):
        raise RuntimeError("Unexpected frozen-prefix cache")
    base_model = radimagenet_densenet121().eval().requires_grad_(False)
    _, original_tail = T.split_encoder(base_model)
    original_digest = L.model_digest(original_tail)
    outputs = []
    for seed in R.SEEDS:
        outer = N.indices_for(patients, split, seed)["train"]
        fit, inner = outer_inner_split(patients, outer, seed)
        for group, config in CONFIGS.items():
            run = OUT / "runs" / f"{group}_{seed}"
            complete = run / "complete.json"
            if complete.exists():
                saved = json.loads(complete.read_text(encoding="utf-8"))
                for relative, digest in saved["outputs"].items():
                    if R.sha256(OUT / relative) != digest:
                        raise RuntimeError(f"Corrupt completed output: {relative}")
                print(f"Verified completed {group} seed={seed}", flush=True)
                continue
            run.mkdir(parents=True, exist_ok=True)
            selected_epoch, inner_history, inner_audit = run_inner_early_stopping(
                maps, image_manifest, patients, fit, inner, original_tail, seed, config["objective"], config["l2sp"]
            )
            head, tail, refit_history, refit_audit = refit_full_outer_train(
                maps, image_manifest, patients, outer, original_tail, seed, config["objective"], config["l2sp"], selected_epoch
            )
            image_embeddings = []
            with torch.inference_mode():
                for start in range(0, len(image_manifest), 32):
                    image_embeddings.append(head(tail(maps[start : start + 32])).cpu().numpy())
            image_embeddings = np.concatenate(image_embeddings)
            if not np.isfinite(image_embeddings).all() or not np.allclose(np.linalg.norm(image_embeddings, axis=1), 1, atol=1e-5):
                raise RuntimeError("Invalid normalized image embeddings")
            patient_features = R.aggregate(image_embeddings, image_manifest, patients)
            feature_path = OUT / "features" / f"{group}_{seed}.npy"
            image_path = OUT / "features" / f"{group}_{seed}_image.npy"
            checkpoint = OUT / "models" / f"{group}_{seed}.pt"
            np.save(feature_path, patient_features)
            np.save(image_path, image_embeddings)
            torch.save(
                dict(
                    head={name: value.cpu() for name, value in head.state_dict().items()},
                    tail={name: value.cpu() for name, value in tail.state_dict().items()},
                    seed=seed,
                    group=group,
                    objective=config["objective"],
                    l2sp=config["l2sp"],
                    selected_epoch=selected_epoch,
                    pretrained_sha256=R.sha256(RADIMAGENET_DIR / "DenseNet121.pt"),
                ),
                checkpoint,
            )
            history_path = run / "loss_history.csv"
            audit_path = run / "fit_audit.json"
            pd.DataFrame([*inner_history, *refit_history]).to_csv(history_path, index=False)
            fit_audit = dict(
                seed=seed,
                group=group,
                objective=config["objective"],
                l2sp=config["l2sp"],
                selected_epoch=selected_epoch,
                outer_train_patient_aliases=patients.patient_alias.iloc[outer].tolist(),
                inner_fit_patient_aliases=patients.patient_alias.iloc[fit].tolist(),
                inner_validation_patient_aliases=patients.patient_alias.iloc[inner].tolist(),
                inner=inner_audit,
                refit=refit_audit,
                original_tail_digest=original_digest,
                original_tail_unchanged=bool(L.model_digest(original_tail) == original_digest),
                checkpoint_sha256=R.sha256(checkpoint),
            )
            R.write_json(audit_path, fit_audit)
            saved_paths = [feature_path, image_path, checkpoint, history_path, audit_path]
            R.write_json(complete, dict(outputs={str(path.relative_to(OUT)): R.sha256(path) for path in saved_paths}))
            outputs.append(dict(seed=seed, group=group, selected_epoch=selected_epoch, tail_drift=refit_audit["final_tail_squared_distance_from_pretrained"]))
            print(f"Completed {group} seed={seed}; selected epoch {selected_epoch}", flush=True)
            del head, tail
            torch.cuda.empty_cache()
    if L.model_digest(original_tail) != original_digest:
        raise RuntimeError("Pretrained tail was mutated")
    pd.DataFrame(outputs).to_csv(OUT / "training_summary.csv", index=False)
    print("All 40 L2-SP / early-stopping neural fits completed", flush=True)


def paired_gate(metrics: pd.DataFrame, anchored: str, zero: str) -> dict:
    left = metrics.loc[metrics.group.eq(anchored) & metrics.role.eq("development_validation"), ["seed", "auroc", "average_precision"]].set_index("seed")
    right = metrics.loc[metrics.group.eq(zero) & metrics.role.eq("development_validation"), ["seed", "auroc", "average_precision"]].set_index("seed")
    delta = left - right
    return dict(
        l2sp=anchored,
        zero_centered=zero,
        delta_auroc_mean=float(delta.auroc.mean()),
        delta_average_precision_mean=float(delta.average_precision.mean()),
        joint_improvement_draws=int(delta.gt(0).all(axis=1).sum()),
        note="Development-only paired comparison. It selects no voter and does not certify a test result.",
    )


def evaluate() -> None:
    require_lock()
    patients, split = read("patients.csv"), read("random_patient_assignments.csv")
    y = patients.label.to_numpy(int)
    metrics, predictions, heads = [], [], []
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            index = N.indices_for(patients, split, seed)
            for group in CONFIGS:
                feature_path = OUT / "features" / f"{group}_{seed}.npy"
                if not feature_path.exists():
                    raise RuntimeError(f"Missing trained feature matrix: {feature_path}")
                features = np.load(feature_path)
                if features.shape != (98, 256) or not np.isfinite(features).all():
                    raise RuntimeError("Invalid patient embedding aggregate")
                model = N.full_head(seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(features[index["train"]], y[index["train"]])
                scores = model.predict_proba(features)[:, 1]
                path = OUT / "models" / f"{group}_{seed}_lr.joblib"
                joblib.dump(model, path, compress=3)
                if not np.allclose(joblib.load(path).predict_proba(features)[:, 1], scores, atol=1e-12):
                    raise RuntimeError("Logistic-regression reload mismatch")
                heads.append(dict(seed=seed, group=group, sha256=R.sha256(path), train_patient_aliases=patients.patient_alias.iloc[index["train"]].tolist()))
                for role, label in (("validation", "development_validation"), ("test", "exploratory_test")):
                    values = index[role]
                    metrics.append(dict(seed=seed, group=group, role=label, **V.score_metrics(y[values], scores[values])))
                    predictions.extend(
                        dict(seed=seed, group=group, role=label, patient_alias=patients.patient_alias.iloc[value], label=int(y[value]), score=float(scores[value]))
                        for value in values
                    )
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / "metrics_by_role_per_draw.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "patient_predictions_by_role.csv", index=False)
    pd.DataFrame(heads).to_json(OUT / "lr_fit_audit.json", orient="records", indent=2, force_ascii=False)
    summary = N.summarize(metric_frame)
    summary.to_csv(OUT / "summary_by_role.csv", index=False)
    gate = dict(
        CE=paired_gate(metric_frame, "F-CE-L2SP", "F-CE-ZERO"),
        CE_T=paired_gate(metric_frame, "F-CE-T-L2SP", "F-CE-T-ZERO"),
    )
    R.write_json(OUT / "development_paired_comparisons.json", gate)
    print(summary[["role", "group", "auroc_mean", "average_precision_mean"]].to_string(index=False), flush=True)
    print("Development-only L2-SP comparisons written; no vote fitted", flush=True)


def save_figure(fig, stem: str) -> dict:
    import cv2

    png = OUT / "figures_600dpi" / f"{stem}.png"
    tiff = OUT / "figures_600dpi" / f"{stem}.tiff"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(tiff, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    image = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("Unable to validate figure")
    return dict(stem=stem, png=png.name, tiff=tiff.name, width=int(image.shape[1]), height=int(image.shape[0]), dpi=600,
                png_sha256=R.sha256(png), tiff_sha256=R.sha256(tiff))


def render_figure() -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = pd.read_csv(OUT / "summary_by_role.csv")
    training = pd.read_csv(OUT / "training_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    groups = list(CONFIGS)
    labels = ["CE\nzero", "CE\nL2-SP", "CE+T\nzero", "CE+T\nL2-SP"]
    positions = np.arange(len(groups))
    for metric, offset, marker in (("auroc_mean", -.16, "o"), ("average_precision_mean", .16, "s")):
        development = summary.loc[summary.role.eq("development_validation")].set_index("group").reindex(groups)[metric]
        test = summary.loc[summary.role.eq("exploratory_test")].set_index("group").reindex(groups)[metric]
        axes[0].scatter(positions + offset, development, marker=marker, s=45, color="#1f2937")
        axes[0].scatter(positions + offset, test, marker=marker, s=45, facecolors="none", edgecolors="#1f2937")
        for pos, dev, exploratory in zip(positions + offset, development, test):
            axes[0].plot([pos, pos], [dev, exploratory], color="#9ca3af", linewidth=.7, zorder=0)
    axes[0].axhline(.8, color="#d97706", linestyle="--", linewidth=.8)
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(.35, 1.02)
    axes[0].set_ylabel("Mean metric across 10 fixed patient draws")
    axes[0].set_title("Development and exploratory test")
    axes[0].scatter([], [], marker="o", color="#1f2937", label="Development AUROC")
    axes[0].scatter([], [], marker="o", facecolors="none", edgecolors="#1f2937", label="Exploratory test AUROC")
    axes[0].scatter([], [], marker="s", color="#1f2937", label="Development AP")
    axes[0].scatter([], [], marker="s", facecolors="none", edgecolors="#1f2937", label="Exploratory test AP")
    axes[0].legend(fontsize=7, loc="lower left")
    palette = {"F-CE-ZERO": "#427aa1", "F-CE-L2SP": "#2a9d8f", "F-CE-T-ZERO": "#e76f51", "F-CE-T-L2SP": "#9c6644"}
    for group in groups:
        values = training.loc[training.group.eq(group), "selected_epoch"]
        axes[1].scatter(np.full(len(values), groups.index(group)), values, color=palette[group], s=34, label=group)
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylim(0, MAX_EPOCHS + 5)
    axes[1].set_ylabel("Epoch selected inside outer training role")
    axes[1].set_title("46/12 training-only early stopping")
    axes[1].legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle("L2-SP anchored tail fine-tuning; no new voting model", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    manifest = save_figure(fig, "Figure_LS1_l2sp_earlystop_metrics")
    plt.close(fig)
    return manifest


def report() -> None:
    require_lock()
    summary = pd.read_csv(OUT / "summary_by_role.csv")
    training = pd.read_csv(OUT / "training_summary.csv")
    paired = json.loads((OUT / "development_paired_comparisons.json").read_text(encoding="utf-8"))
    figure = render_figure()
    R.write_json(OUT / "figure_manifest.json", [figure])
    values = summary.pivot(index="group", columns="role", values=["auroc_mean", "average_precision_mean"])
    lines = [
        "# CYCH L2-SP and Training-Only Early-Stopping Probe",
        "",
        "Completed: 2026-09-13. This separate batch follows the completed triplet ablation. It tests a limited transfer-learning hypothesis and does not change earlier model, vote, manuscript or cohort claims.",
        "",
        "## Design",
        "",
        "- Same 98 patients / 349 first-exam images and ten fixed 58/20/20 patient assignments.",
        "- For each outer 58-patient training role, one fixed stratified 46/12 inner split selected an epoch using patient-level binary cross entropy. The network then reset and refit on all 58 outer-training patients for exactly that selected epoch count.",
        "- Four paired configurations: CE and CE+triplet, each with conventional zero-centered tail weight decay or L2-SP anchoring of the pretrained DenseNet tail. The new projection receives ordinary weight decay in both conditions.",
        "- No development/test role chose an epoch, hyperparameter, feature subset or voting weight. No new vote is fitted.",
        "",
        "## Results",
        "",
        "| Configuration | Development AUROC / AP | Exploratory test AUROC / AP |",
        "|---|---:|---:|",
    ]
    labels = {"F-CE-ZERO": "CE, zero-centered", "F-CE-L2SP": "CE, L2-SP", "F-CE-T-ZERO": "CE+triplet, zero-centered", "F-CE-T-L2SP": "CE+triplet, L2-SP"}
    for group in CONFIGS:
        lines.append(
            f"| {labels[group]} | {values.loc[group, ('auroc_mean', 'development_validation')]:.4f} / {values.loc[group, ('average_precision_mean', 'development_validation')]:.4f} | {values.loc[group, ('auroc_mean', 'exploratory_test')]:.4f} / {values.loc[group, ('average_precision_mean', 'exploratory_test')]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Development-Only Paired Comparison",
            "",
            f"- CE L2-SP minus CE zero: AUROC {paired['CE']['delta_auroc_mean']:+.4f}; AP {paired['CE']['delta_average_precision_mean']:+.4f}; joint improvement {paired['CE']['joint_improvement_draws']}/10.",
            f"- CE+triplet L2-SP minus CE+triplet zero: AUROC {paired['CE_T']['delta_auroc_mean']:+.4f}; AP {paired['CE_T']['delta_average_precision_mean']:+.4f}; joint improvement {paired['CE_T']['joint_improvement_draws']}/10.",
            "",
            "## Training Selection Audit",
            "",
            "| Configuration | Selected epoch: min / median / max |",
            "|---|---:|",
        ]
    )
    for group in CONFIGS:
        epochs = training.loc[training.group.eq(group), "selected_epoch"]
        lines.append(f"| {labels[group]} | {int(epochs.min())} / {float(epochs.median()):.1f} / {int(epochs.max())} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "L2-SP adds an explicit preference for the pretrained tail, but it cannot by itself prove why a previous configuration generalized poorly. The 12-patient inner selection is deliberately retained for audit and may be unstable. All development/test estimates remain exploratory because the same 98 patients have already been explored; no result here is external confirmation or a basis for a new clinical claim.",
            "",
            "## Files",
            "",
            "- `inner_training_assignments.csv` and `runs/*/loss_history.csv`: inner roles, selected epochs and losses.",
            "- `metrics_by_role_per_draw.csv`, `summary_by_role.csv`, and `patient_predictions_by_role.csv`: complete patient-level results.",
            "- `development_paired_comparisons.json`: L2-SP versus matched zero-centered development comparisons.",
            "- `figures_600dpi/`: 600 dpi PNG/TIFF figure with checksums in `figure_manifest.json`.",
            "",
            "## References",
            "",
        ]
    )
    lines.extend(f"- {reference['citation']} [{reference['url']}]({reference['url']}) — {reference['scope']}" for reference in REFERENCES)
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("README and 600 dpi L2-SP figure rendered", flush=True)


def package() -> None:
    require_lock()
    archive = ROOT / f"{OUT.name}.zip"
    files = [path for path in sorted(OUT.rglob("*")) if path.is_file()]
    pd.DataFrame(
        [dict(path=str(path.relative_to(OUT)).replace("\\", "/"), bytes=path.stat().st_size, sha256=R.sha256(path)) for path in files]
    ).to_csv(OUT / "archive_manifest.csv", index=False)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise RuntimeError("ZIP CRC verification failed")
        entry_count = len(zf.infolist())
    R.write_json(OUT / "archive_verification.json", dict(archive=str(archive), bytes=archive.stat().st_size, sha256=R.sha256(archive), entries=entry_count, crc_verified=True))
    print(f"ZIP verified: {archive.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train", "evaluate", "report", "package", "all"))
    args = parser.parse_args()
    stages = ("prepare", "train", "evaluate", "report", "package") if args.stage == "all" else (args.stage,)
    for stage in stages:
        globals()[stage]()
