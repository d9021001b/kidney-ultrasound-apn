"""Multi-query cross-attention MIL for heterogeneous kidney ultrasound evidence."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_fullframe_20261103"
PATCH_SOURCE = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_parenchyma_mil_20261103"
OUT = ROOT / "02_experiments/NCKUH_DMSA_multiquery_mil_20261103"
SEEDS = (20261102, 20261103, 20261104)
EPOCHS = (10, 20, 40, 80)
ARMS = {
    "cross_attention_q1": {"queries": 1, "diversity_weight": 0.0},
    "cross_attention_q4": {"queries": 4, "diversity_weight": 0.05},
}
REFERENCE_GROUP = "fullcohort_rad_color_image_plus_patient_direct"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


class MultiQueryMIL(nn.Module):
    def __init__(self, queries: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(queries, 32) * 0.02)
        self.view_embedding = nn.Embedding(2, 32)
        self.input = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.LayerNorm(32))
        self.attention = nn.MultiheadAttention(32, 4, dropout=0.1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(queries * 32),
            nn.Dropout(0.25),
            nn.Linear(queries * 32, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor, views: torch.Tensor):
        view_index = views.clamp_min(0)
        hidden = self.input(values) + self.view_embedding(view_index)
        query = self.query.unsqueeze(0).expand(len(values), -1, -1)
        evidence, weights = self.attention(
            query, hidden, hidden, key_padding_mask=~mask, need_weights=True,
        )
        logits = self.classifier(evidence.flatten(1)).squeeze(1)
        return logits, weights, evidence


def attention_diversity(weights: torch.Tensor) -> torch.Tensor:
    if weights.shape[1] == 1:
        return weights.new_zeros(())
    normalized = nn.functional.normalize(weights, dim=2)
    gram = normalized @ normalized.transpose(1, 2)
    identity = torch.eye(gram.shape[1], device=gram.device).unsqueeze(0)
    return ((gram - identity) ** 2).sum((1, 2)).div(gram.shape[1] * (gram.shape[1] - 1)).mean()


def load_reference():
    sys.path.insert(0, str(Path(__file__).parent))
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference as load

    patients, kidneys, images = load()
    images = images[images.kidney_APN.notna()].copy().reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].copy().reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].copy().reset_index(drop=True)
    assert (len(patients), len(kidneys), len(images)) == (210, 414, 820)
    return patients, kidneys, images


def patch_data(images: pd.DataFrame, kidneys: pd.DataFrame):
    raw = np.load(PATCH_SOURCE / "radimagenet_local_float16.npy", mmap_mode="r")
    offsets = np.load(PATCH_SOURCE / "patch_offsets.npy")
    values = np.asarray(raw, dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True).clip(1e-8)
    image_index = np.repeat(np.arange(len(images)), np.diff(offsets))
    patch_patients = images.patient_alias.to_numpy()[image_index]
    patch_kidneys = images.kidney_alias.to_numpy()[image_index]
    patch_views = np.asarray([
        0 if str(view).endswith("L") else 1 for view in images.view.to_numpy()[image_index]
    ], dtype=np.int64)
    groups = [np.flatnonzero(patch_kidneys == alias) for alias in kidneys.kidney_alias]
    assert len(offsets) == len(images) + 1 and offsets[-1] == len(values)
    assert all(len(group) for group in groups)
    return values, patch_patients, patch_views, groups


def prepare_bags(values, patch_patients, patch_views, groups, train_aliases, seed):
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed + 7000)
    sample = []
    for alias in sorted(train_aliases):
        available = np.flatnonzero(patch_patients == alias)
        sample.extend(rng.choice(available, 64, replace=len(available) < 64))
    sample = np.asarray(sample, np.int64)
    pca = PCA(32, whiten=True, random_state=seed, svd_solver="randomized").fit(values[sample])
    transformed = pca.transform(values).astype(np.float32)
    compact = []
    for group in groups:
        if len(group) > 128:
            group = group[np.linspace(0, len(group) - 1, 128).round().astype(int)]
        compact.append(group)
    width = max(map(len, compact))
    bags = np.zeros((len(compact), width, 32), np.float32)
    mask = np.zeros((len(compact), width), bool)
    views = np.full((len(compact), width), -1, np.int64)
    for index, group in enumerate(compact):
        bags[index, :len(group)] = transformed[group]
        mask[index, :len(group)] = True
        views[index, :len(group)] = patch_views[group]
    return (
        torch.from_numpy(bags).cuda(),
        torch.from_numpy(mask).cuda(),
        torch.from_numpy(views).cuda(),
        pca,
    )


def train_model(arm, seed, bags, mask, views, labels, fit, epochs, evaluation):
    settings = ARMS[arm]
    torch.manual_seed(seed + 37)
    model = MultiQueryMIL(settings["queries"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    target = labels[fit]
    weights = torch.where(
        target > 0,
        len(target) / (2 * target.sum()),
        len(target) / (2 * (len(target) - target.sum())),
    )
    history = {}
    for epoch in range(1, max(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, attention, _ = model(bags[fit], mask[fit], views[fit])
        bce = (nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none") * weights).mean()
        loss = bce + settings["diversity_weight"] * attention_diversity(attention)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch in epochs and evaluation is not None:
            model.eval()
            with torch.inference_mode():
                evaluation_logits = model(bags[evaluation], mask[evaluation], views[evaluation])[0]
                history[epoch] = torch.sigmoid(evaluation_logits).cpu().numpy()
    return model, history


def metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
    }


def run() -> None:
    from sklearn.model_selection import train_test_split

    if OUT.exists():
        raise FileExistsError(f"Preserve completed experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    patients, kidneys, images = load_reference()
    values, patch_patients, patch_views, groups = patch_data(images, kidneys)
    labels_np = kidneys.kidney_APN.to_numpy(int)
    labels = torch.from_numpy(labels_np.astype(np.float32)).cuda()
    assignments = pd.read_csv(BASE / "patient_assignments.csv")
    reference = pd.read_csv(BASE / "instance_development_predictions.csv")
    reference = reference[reference.group.eq(REFERENCE_GROUP)].set_index(["seed", "kidney_alias"]).score

    records, predictions, selections = [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train_patients = patients[patients.patient_alias.isin(mapping[mapping.eq("train")].index)].reset_index(drop=True)
        fit_local, select_local = train_test_split(
            np.arange(len(train_patients)), train_size=100, random_state=seed + 6000,
            stratify=train_patients.DMSA_APN,
        )
        fit_alias = train_patients.patient_alias.iloc[fit_local]
        select_alias = train_patients.patient_alias.iloc[select_local]
        fit = np.flatnonzero(kidneys.patient_alias.isin(fit_alias))
        selection = np.flatnonzero(kidneys.patient_alias.isin(select_alias))
        bags, mask, views, _ = prepare_bags(
            values, patch_patients, patch_views, groups, set(fit_alias), seed,
        )
        chosen_epochs = {}
        for arm in ARMS:
            _, history = train_model(arm, seed, bags, mask, views, labels, fit, EPOCHS, selection)
            choices = []
            for epoch, score in history.items():
                result = metric(labels_np[selection], score)
                choices.append((-(result["auroc"] + result["ap"]) / 2.0, epoch, result))
            _, chosen, result = min(choices, key=lambda item: (item[0], item[1]))
            chosen_epochs[arm] = chosen
            selections.append({
                "seed": seed,
                "arm": arm,
                "selected_epoch": chosen,
                "inner_auroc": result["auroc"],
                "inner_ap": result["ap"],
            })
            torch.cuda.empty_cache()

        train = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("train"))
        development = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        train_aliases = set(mapping[mapping.eq("train")].index)
        bags, mask, views, pca = prepare_bags(
            values, patch_patients, patch_views, groups, train_aliases, seed + 100,
        )
        ref = np.asarray([reference.loc[(seed, alias)] for alias in kidneys.kidney_alias.iloc[development]])
        for arm in ARMS:
            model, _ = train_model(
                arm, seed + 10000, bags, mask, views, labels, train,
                (chosen_epochs[arm],), None,
            )
            model.eval()
            with torch.inference_mode():
                score = torch.sigmoid(model(bags[development], mask[development], views[development])[0]).cpu().numpy()
            candidate_scores = {
                arm: score,
                f"{arm}_plus_instance_equal": (score + ref) / 2.0,
                f"{arm}_quarter_plus_instance": score * 0.25 + ref * 0.75,
            }
            torch.save({
                "state_dict": model.state_dict(),
                "selected_epoch": chosen_epochs[arm],
                "seed": seed,
                "arm": arm,
            }, OUT / "models" / f"{seed}_{arm}.pth")
            joblib.dump(pca, OUT / "models" / f"{seed}_{arm}_pca.joblib", compress=3)
            for group, candidate_score in candidate_scores.items():
                result = metric(labels_np[development], candidate_score)
                records.append({
                    "seed": seed,
                    "role": "development",
                    "group": group,
                    "patients": 42,
                    "kidneys": len(development),
                    **result,
                })
                for local, index in enumerate(development):
                    predictions.append({
                        "seed": seed,
                        "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[index],
                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                        "side": kidneys.side.iloc[index],
                        "label": int(labels_np[index]),
                        "score": float(candidate_score[local]),
                    })
            del model
            torch.cuda.empty_cache()
        print(f"Completed multi-query MIL draw {seed}", flush=True)

    metric_frame = pd.DataFrame(records)
    prediction_frame = pd.DataFrame(predictions)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "training_inner_epoch_selections.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(OUT / "protocol_and_decision.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "motivation": "Test whether multiple latent queries preserve heterogeneous focal ultrasound evidence that single pooling dilutes",
        "literature_basis": "C2LUE arXiv:2610.14143; four-query cross-attention ablation",
        "instances": "Kidney-mask-constrained RadImageNet local patches with longitudinal/transverse view embeddings",
        "arms": ARMS,
        "selection": "Fixed 100/26 patient split inside each 126-patient training role; epoch selected from 10/20/40/80",
        "groups_above_both_development_targets": achieved,
        "exploratory_test_scored": False,
    })
    write_json(OUT / "integrity_audit.json", {
        "patients": len(patients),
        "kidneys": len(kidneys),
        "images": len(images),
        "prediction_rows": len(prediction_frame),
        "prediction_roles": ["development"],
        "exploratory_test_rows_scored": 0,
        "scores_finite": bool(np.isfinite(prediction_frame.score).all()),
        "scores_in_unit_interval": bool(prediction_frame.score.between(0, 1).all()),
    })
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    run()
