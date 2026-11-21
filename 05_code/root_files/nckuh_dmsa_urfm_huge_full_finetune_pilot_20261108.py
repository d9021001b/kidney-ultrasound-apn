"""One-draw full URFM ViT-H/16 patient-held-out APN fine-tuning pilot."""
from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_urfm_huge_frozen_20261108 as huge
import nckuh_dmsa_urfm_lora_patient_20261106 as supervised
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_urfm_huge_full_finetune_pilot_20261108"
REFERENCE = uri.OUT / "development_predictions.csv"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"
SEED = 20261102
MAX_EPOCHS = 12
MIN_EPOCHS = 4
PATIENCE = 4
BATCH_SIZE = 16
BACKBONE_LR = 3e-5
HEAD_LR = 3e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 2


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


@contextmanager
def encoder_configuration():
    with huge.configured():
        original_transfer = huge.study.TRANSFER_NORM_TO_FC_NORM
        original_drop = huge.study.DROP_PATH_RATE
        try:
            huge.study.TRANSFER_NORM_TO_FC_NORM = True
            huge.study.DROP_PATH_RATE = 0.10
            yield
        finally:
            huge.study.TRANSFER_NORM_TO_FC_NORM = original_transfer
            huge.study.DROP_PATH_RATE = original_drop


def patient_partitions(patients: pd.DataFrame) -> dict[str, np.ndarray]:
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    current = assignments[assignments.seed.eq(SEED)].set_index("patient_alias").role
    roles = patients.patient_alias.map(current)
    train = np.flatnonzero(roles.eq("train"))
    outer = np.flatnonzero(roles.eq("development"))
    inner_train, inner_development = train_test_split(
        train,
        train_size=100,
        random_state=SEED + 4000,
        stratify=patients.DMSA_APN.to_numpy()[train],
    )
    if set(train) & set(outer):
        raise ValueError("Patient leakage between train and development")
    return {
        "train": train,
        "outer_development": outer,
        "inner_train": inner_train,
        "inner_development": inner_development,
    }


def kidney_rows(images: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        alias: group.index.to_numpy(np.int64)
        for alias, group in images.groupby("kidney_alias", sort=False)
    }


def sample_one_image_per_kidney(
    patients: np.ndarray,
    patient_frame: pd.DataFrame,
    kidney_frame: pd.DataFrame,
    rows_by_kidney: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aliases = set(patient_frame.patient_alias.iloc[patients])
    current = kidney_frame[kidney_frame.patient_alias.isin(aliases)]
    rows, labels, kidney_aliases = [], [], []
    for kidney in current.itertuples(index=False):
        candidates = rows_by_kidney[kidney.kidney_alias]
        rows.append(int(rng.choice(candidates)))
        labels.append(int(kidney.kidney_APN))
        kidney_aliases.append(kidney.kidney_alias)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(kidney_aliases),
    )


def prepare_variants(images: pd.DataFrame) -> np.ndarray:
    colors, masks = huge.study.contour_canvases(images)
    variants = []
    for variant in range(3):
        current = [
            supervised.augmented_canvas(color, mask, variant)[0]
            for color, mask in zip(colors, masks)
        ]
        variants.append(np.stack(current))
    return np.stack(variants)


def tensor_batch(colors: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(colors.transpose(0, 3, 1, 2).copy()).cuda().float() / 255.0
    mean = torch.as_tensor(huge.study.MEAN, device="cuda")[None, :, None, None]
    std = torch.as_tensor(huge.study.STD, device="cuda")[None, :, None, None]
    return (tensor - mean) / std


def aggregate_kidneys(
    image_scores: np.ndarray, images: pd.DataFrame, kidneys: pd.DataFrame
) -> np.ndarray:
    score_by_alias = pd.Series(image_scores, index=images.kidney_alias).groupby(level=0).mean()
    return kidneys.kidney_alias.map(score_by_alias).to_numpy(float)


def build_model(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with encoder_configuration():
        model, audit = huge.study.load_encoder()
    model.head = torch.nn.Linear(1280, 2)
    torch.nn.init.trunc_normal_(model.head.weight, std=2e-5)
    torch.nn.init.zeros_(model.head.bias)
    model = model.cuda()
    backbone = [parameter for name, parameter in model.named_parameters() if not name.startswith("head.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": BACKBONE_LR, "initial_lr": BACKBONE_LR},
            {"params": model.head.parameters(), "lr": HEAD_LR, "initial_lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    return model, optimizer, audit


def set_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, total: int) -> None:
    if epoch <= WARMUP_EPOCHS:
        scale = epoch / WARMUP_EPOCHS
    else:
        progress = (epoch - WARMUP_EPOCHS) / max(1, total - WARMUP_EPOCHS)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * scale


def score_patients(
    model: torch.nn.Module,
    patient_indices: np.ndarray,
    patients: pd.DataFrame,
    kidneys: pd.DataFrame,
    images: pd.DataFrame,
    variants: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    aliases = set(patients.patient_alias.iloc[patient_indices])
    current_images = images[images.patient_alias.isin(aliases)]
    current_kidneys = kidneys[kidneys.patient_alias.isin(aliases)].copy()
    variant_predictions = []
    model.eval()
    with torch.inference_mode():
        for variant in range(3):
            values = []
            rows = current_images.index.to_numpy(np.int64)
            for start in range(0, len(rows), BATCH_SIZE):
                batch = tensor_batch(variants[variant, rows[start : start + BATCH_SIZE]])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch)
                values.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
            variant_predictions.append(np.concatenate(values))
    image_scores = np.stack(variant_predictions).mean(0)
    current_kidneys["score"] = aggregate_kidneys(
        image_scores, current_images, current_kidneys
    )
    metrics = base.metric(
        current_kidneys.kidney_APN.to_numpy(int), current_kidneys.score.to_numpy(float)
    )
    return metrics, current_kidneys


def train_once(
    seed: int,
    fit_patients: np.ndarray,
    score_patient_indices: np.ndarray,
    total_epochs: int,
    select_epoch: bool,
    patients: pd.DataFrame,
    kidneys: pd.DataFrame,
    images: pd.DataFrame,
    variants: np.ndarray,
    rows_by_kidney: dict[str, np.ndarray],
):
    model, optimizer, audit = build_model(seed)
    fit_aliases = set(patients.patient_alias.iloc[fit_patients])
    fit_kidneys = kidneys[kidneys.patient_alias.isin(fit_aliases)]
    counts = fit_kidneys.kidney_APN.value_counts().to_dict()
    weights = torch.tensor(
        [len(fit_kidneys) / (2 * counts[label]) for label in (0, 1)],
        device="cuda",
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    rng = np.random.default_rng(seed + 503)
    best, stale, records = None, 0, []
    start_time = time.monotonic()
    for epoch in range(1, total_epochs + 1):
        set_learning_rate(optimizer, epoch, total_epochs)
        rows, labels, _ = sample_one_image_per_kidney(
            fit_patients, patients, kidneys, rows_by_kidney, rng
        )
        order = rng.permutation(len(rows))
        model.train()
        loss_sum = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            selected = order[start : start + BATCH_SIZE]
            variant = rng.integers(0, 3, size=len(selected), dtype=np.int64)
            batch = tensor_batch(variants[variant, rows[selected]])
            target = torch.from_numpy(labels[selected]).cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch)
                loss = criterion(logits.float(), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(selected)
        row = {"epoch": epoch, "loss": loss_sum / len(rows)}
        if select_epoch:
            inner, _ = score_patients(
                model, score_patient_indices, patients, kidneys, images, variants
            )
            objective = (inner["auroc"] + inner["ap"]) / 2.0
            row.update({f"inner_{name}": value for name, value in inner.items()})
            row["objective"] = objective
            if epoch >= MIN_EPOCHS and (best is None or objective > best["objective"] + 1e-12):
                best = {"epoch": epoch, "objective": objective, **inner}
                stale = 0
            elif epoch >= MIN_EPOCHS:
                stale += 1
        records.append(row)
        print(
            f"full_h seed={seed} {'inner' if select_epoch else 'refit'} "
            f"epoch={epoch} loss={row['loss']:.4f} elapsed={time.monotonic()-start_time:.0f}s",
            flush=True,
        )
        if select_epoch and epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    if select_epoch:
        if best is None:
            raise RuntimeError("No inner epoch selected")
        del model, optimizer
        torch.cuda.empty_cache()
        return {"selected_epoch": best["epoch"], "inner": best, "records": records, "load_audit": audit}
    metrics, scored = score_patients(
        model, score_patient_indices, patients, kidneys, images, variants
    )
    del model, optimizer
    torch.cuda.empty_cache()
    return metrics, scored, records, audit


def evaluate() -> None:
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed experiment")
    OUT.mkdir(parents=True, exist_ok=True)
    patients, kidneys, images = cohort.load_reference()
    partitions = patient_partitions(patients)
    variants = prepare_variants(images)
    rows_by_kidney = kidney_rows(images)
    selection = train_once(
        SEED,
        partitions["inner_train"],
        partitions["inner_development"],
        MAX_EPOCHS,
        True,
        patients,
        kidneys,
        images,
        variants,
        rows_by_kidney,
    )
    selected_epoch = int(selection["selected_epoch"])
    metrics, scored, refit_records, refit_audit = train_once(
        SEED,
        partitions["train"],
        partitions["outer_development"],
        selected_epoch,
        False,
        patients,
        kidneys,
        images,
        variants,
        rows_by_kidney,
    )
    reference = pd.read_csv(REFERENCE)
    reference = reference[
        reference.seed.eq(SEED) & reference.group.eq(URI_GROUP)
    ].set_index("kidney_alias")
    fixed_uri = scored.kidney_alias.map(reference.score).to_numpy(float)
    model_score = scored.score.to_numpy(float)
    groups = {
        "official_uri_fixed_reference": fixed_uri,
        "urfm_huge_full_tta": model_score,
        "urfm_huge_full_tta_20pct_plus_uri": 0.20 * model_score + 0.80 * fixed_uri,
        "urfm_huge_full_tta_25pct_plus_uri": 0.25 * model_score + 0.75 * fixed_uri,
    }
    predictions, metric_rows = [], []
    for group, score in groups.items():
        current = base.metric(scored.kidney_APN.to_numpy(int), score)
        metric_rows.append({"seed": SEED, "role": "development", "group": group, **current})
        for row, value in zip(scored.itertuples(index=False), score):
            predictions.append(
                {
                    "seed": SEED,
                    "group": group,
                    "patient_alias": row.patient_alias,
                    "kidney_alias": row.kidney_alias,
                    "DMSA_APN": row.DMSA_APN,
                    "side": row.side,
                    "label": int(row.kidney_APN),
                    "score": float(value),
                }
            )
    prediction_frame = pd.DataFrame(predictions)
    metric_frame = pd.DataFrame(metric_rows)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(selection["records"]).to_csv(OUT / "inner_training_records.csv", index=False)
    pd.DataFrame(refit_records).to_csv(OUT / "refit_training_records.csv", index=False)
    ci, delta, samples = cluster_intervals(
        prediction_frame, list(groups), (SEED,), "official_uri_fixed_reference"
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    candidates = metric_frame[
        metric_frame.auroc.gt(0.8) & metric_frame.ap.gt(0.8)
    ].group.tolist()
    write_json(
        OUT / "decision.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selected_epoch": selected_epoch,
            "inner_selection": selection["inner"],
            "groups_above_both_pilot_targets": candidates,
            "expand_to_three_development_draws": bool(candidates),
            "promotion": False,
            "exploratory_test_rows_scored": 0,
            "split_unit": "patient",
            "training_sampling_unit": "one randomly selected image per kidney per epoch",
            "model": "official URFM ViT-H/16, all encoder parameters trainable",
            "augmentation": ["original", "rotation_minus3_gamma090", "rotation_plus3_gamma110"],
            "selection_load_audit": selection["load_audit"],
            "refit_load_audit": refit_audit,
        },
    )
    print(metric_frame[["group", "auroc", "ap", "brier"]].to_string(index=False), flush=True)


def finalize_existing() -> None:
    """Complete uncertainty files after a post-training packaging interruption."""
    prediction_path = OUT / "development_predictions.csv"
    metric_path = OUT / "development_metrics.csv"
    if not prediction_path.is_file() or not metric_path.is_file():
        raise FileNotFoundError("No completed fixed predictions to finalize")
    prediction_frame = pd.read_csv(prediction_path)
    metric_frame = pd.read_csv(metric_path)
    _, kidneys, _ = cohort.load_reference()
    if "DMSA_APN" not in prediction_frame:
        reference = kidneys.set_index("kidney_alias").DMSA_APN
        prediction_frame["DMSA_APN"] = prediction_frame.kidney_alias.map(reference)
    if prediction_frame.DMSA_APN.isna().any():
        raise ValueError("Could not restore all DMSA strata")
    prediction_frame.to_csv(prediction_path, index=False)
    groups = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups, (SEED,), "official_uri_fixed_reference"
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    inner = pd.read_csv(OUT / "inner_training_records.csv")
    eligible = inner[inner.epoch.ge(MIN_EPOCHS)].sort_values(
        ["objective", "epoch"], ascending=[False, True]
    )
    if eligible.empty:
        raise ValueError("No eligible selected epoch in fixed training record")
    selected = eligible.iloc[0]
    candidates = metric_frame[
        metric_frame.auroc.gt(0.8) & metric_frame.ap.gt(0.8)
    ].group.tolist()
    write_json(
        OUT / "decision.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selected_epoch": int(selected.epoch),
            "inner_selection": {
                "objective": float(selected.objective),
                "auroc": float(selected.inner_auroc),
                "ap": float(selected.inner_ap),
                "brier": float(selected.inner_brier),
            },
            "groups_above_both_pilot_targets": candidates,
            "expand_to_three_development_draws": bool(candidates),
            "promotion": False,
            "exploratory_test_rows_scored": 0,
            "split_unit": "patient",
            "training_sampling_unit": "one randomly selected image per kidney per epoch",
            "model": "official URFM ViT-H/16, all encoder parameters trainable",
            "augmentation": ["original", "rotation_minus3_gamma090", "rotation_plus3_gamma110"],
            "finalized_from_fixed_predictions_after_missing_DMSA_APN_packaging_error": True,
        },
    )
    print(metric_frame[["group", "auroc", "ap", "brier"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="evaluate", choices=("evaluate", "finalize-existing"))
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    {"evaluate": evaluate, "finalize-existing": finalize_existing}[arguments.stage]()
