"""Training-only triplet and contralateral ranking on RadImageNet kidney features."""

from __future__ import annotations

import copy
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
SOURCE = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_fullframe_20261103"
OUT = ROOT / "02_experiments/NCKUH_DMSA_triplet_rank_20261103"
SEEDS = (20261102, 20261103, 20261104)
EPOCHS = (25, 50, 100, 200)
REFERENCE_GROUP = "fullcohort_rad_color_image_plus_patient_direct"
ARMS = {
    "rad_mlp_pair_rank": {"triplet_weight": 0.0},
    "rad_mlp_pair_rank_triplet": {"triplet_weight": 0.1},
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def kidney_summary_features(
    image_values: np.ndarray,
    images: pd.DataFrame,
    kidneys: pd.DataFrame,
) -> np.ndarray:
    width = image_values.shape[1]
    rows = []
    for kidney in kidneys.itertuples(index=False):
        subset = images[images.kidney_alias.eq(kidney.kidney_alias)]
        indices = subset.index.to_numpy()
        mean = image_values[indices].mean(0)
        std = image_values[indices].std(0)
        views = []
        flags = []
        for suffix in ("L", "T"):
            selected = subset.index[subset.view.str.endswith(suffix)].to_numpy()
            views.append(image_values[selected].mean(0) if len(selected) else np.full(width, np.nan))
            flags.append(float(len(selected) > 0))
        rows.append(np.concatenate([mean, std, views[0], views[1], views[0] - views[1], flags]))
    return np.asarray(rows, dtype=np.float32)


def fit_transformer(values: np.ndarray, train: np.ndarray, seed: int):
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    transformer = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold()),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=32, whiten=True, random_state=seed)),
    ])
    transformer.fit(values[train])
    return transformer, transformer.transform(values).astype(np.float32)


class MetricHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(width, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
        )
        self.classifier = nn.Linear(16, 1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = nn.functional.normalize(self.encoder(values), dim=1)
        return self.classifier(latent).squeeze(1), latent


def patient_pairs(kidneys: pd.DataFrame, selected: np.ndarray) -> list[tuple[int, int, int]]:
    selected_set = set(map(int, selected))
    pairs = []
    for _, group in kidneys.iloc[selected].groupby("patient_alias"):
        lookup = {row.side: int(index) for index, row in group.iterrows() if int(index) in selected_set}
        if "left" not in lookup or "right" not in lookup:
            continue
        state = str(group.DMSA_APN.iloc[0])
        if state == "L":
            pairs.append((lookup["left"], lookup["right"], 1))
        elif state == "R":
            pairs.append((lookup["left"], lookup["right"], -1))
    return pairs


def batch_hard_triplet(latent: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    distance = torch.cdist(latent, latent, p=2)
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    different = ~labels[:, None].eq(labels[None, :])
    hardest_positive = distance.masked_fill(~same, float("-inf")).max(dim=1).values
    hardest_negative = distance.masked_fill(~different, float("inf")).min(dim=1).values
    valid = torch.isfinite(hardest_positive) & torch.isfinite(hardest_negative)
    return nn.functional.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()


def fit_model(
    values: np.ndarray,
    kidneys: pd.DataFrame,
    selected: np.ndarray,
    epochs: int,
    seed: int,
    triplet_weight: float,
) -> MetricHead:
    torch.manual_seed(seed)
    model = MetricHead(values.shape[1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    x = torch.from_numpy(values).cuda()
    labels_np = kidneys.kidney_APN.to_numpy(np.float32)
    labels = torch.from_numpy(labels_np).cuda()
    chosen = torch.from_numpy(selected).long().cuda()
    selected_labels = labels[chosen]
    positives = float((selected_labels == 1).sum())
    negatives = float((selected_labels == 0).sum())
    class_weights = torch.where(selected_labels == 1, 0.5 / positives, 0.5 / negatives)
    class_weights *= len(selected)
    pairs = patient_pairs(kidneys, selected)
    left_indices = torch.tensor([item[0] for item in pairs], device="cuda", dtype=torch.long)
    right_indices = torch.tensor([item[1] for item in pairs], device="cuda", dtype=torch.long)
    directions = torch.tensor([item[2] for item in pairs], device="cuda", dtype=torch.float32)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, latent = model(x)
        bce = (nn.functional.binary_cross_entropy_with_logits(
            logits[chosen], selected_labels, reduction="none") * class_weights).mean()
        rank = nn.functional.softplus(0.5 - directions * (logits[left_indices] - logits[right_indices])).mean()
        triplet = batch_hard_triplet(latent[chosen], selected_labels.long())
        loss = bce + 0.5 * rank + triplet_weight * triplet
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model


def predict(model: MetricHead, values: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        logits, _ = model(torch.from_numpy(values).cuda())
        return torch.sigmoid(logits).cpu().numpy()


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

    sys.path.insert(0, str(Path(__file__).parent))
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference

    patients, kidneys, images = load_reference()
    images = images[images.kidney_APN.notna()].copy().reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].copy().reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].copy().reset_index(drop=True)
    assert (len(patients), len(kidneys), len(images)) == (210, 414, 820)
    raw = kidney_summary_features(np.load(SOURCE / "radimagenet_color.npy"), images, kidneys)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(SOURCE / "patient_assignments.csv")
    reference = pd.read_csv(SOURCE / "instance_development_predictions.csv")
    reference = reference[reference.group.eq(REFERENCE_GROUP)].set_index(["seed", "kidney_alias"]).score

    records, predictions, selections = [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train_patients = patients[patients.patient_alias.isin(mapping[mapping.eq("train")].index)].reset_index(drop=True)
        fit_local, select_local = train_test_split(
            np.arange(len(train_patients)), train_size=100, random_state=seed + 5000,
            stratify=train_patients.DMSA_APN,
        )
        fit_alias = train_patients.patient_alias.iloc[fit_local]
        select_alias = train_patients.patient_alias.iloc[select_local]
        fit_kidneys = np.flatnonzero(kidneys.patient_alias.isin(fit_alias))
        select_kidneys = np.flatnonzero(kidneys.patient_alias.isin(select_alias))
        inner_transformer, inner_values = fit_transformer(raw, fit_kidneys, seed)
        chosen_epochs = {}
        for arm, settings in ARMS.items():
            choices = []
            for epoch in EPOCHS:
                model = fit_model(
                    inner_values, kidneys, fit_kidneys, epoch, seed + epoch,
                    settings["triplet_weight"],
                )
                score = predict(model, inner_values)[select_kidneys]
                result = metric(labels[select_kidneys], score)
                choices.append((-(result["auroc"] + result["ap"]) / 2.0, epoch, result))
                del model
                torch.cuda.empty_cache()
            _, chosen_epoch, inner_result = min(choices, key=lambda item: (item[0], item[1]))
            chosen_epochs[arm] = chosen_epoch
            selections.append({
                "seed": seed,
                "arm": arm,
                "selected_epoch": chosen_epoch,
                "inner_auroc": inner_result["auroc"],
                "inner_ap": inner_result["ap"],
            })

        train_kidneys = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("train"))
        development = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        transformer, values = fit_transformer(raw, train_kidneys, seed)
        ref = np.asarray([reference.loc[(seed, alias)] for alias in kidneys.kidney_alias.iloc[development]])
        for arm, settings in ARMS.items():
            model = fit_model(
                values, kidneys, train_kidneys, chosen_epochs[arm], seed + 10000,
                settings["triplet_weight"],
            )
            score = predict(model, values)[development]
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
            joblib.dump(transformer, OUT / "models" / f"{seed}_{arm}_transformer.joblib", compress=3)
            for group, candidate_score in candidate_scores.items():
                result = metric(labels[development], candidate_score)
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
                        "label": int(labels[index]),
                        "score": float(candidate_score[local]),
                    })
            del model
            torch.cuda.empty_cache()
        print(f"Completed triplet/ranking draw {seed}", flush=True)

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
        "motivation": "Use unilateral patients as within-patient affected-versus-contralateral supervision and test the original-paper triplet rationale without touching exploratory test patients",
        "features": "Frozen RadImageNet DenseNet-121 kidney summaries; training-only imputation, scaling, and PCA32",
        "loss": "Class-balanced kidney BCE + 0.5 contralateral ranking + optional 0.1 batch-hard triplet margin0.3",
        "selection": "Fixed 100/26 patient split inside each 126-patient training role; epoch selected from 25/50/100/200",
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
