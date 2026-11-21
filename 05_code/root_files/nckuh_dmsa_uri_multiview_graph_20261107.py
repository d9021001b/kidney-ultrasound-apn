"""Patient-local multiview graph over frozen URI-CADS image embeddings."""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_uri_multiview_graph_20261107"
WORK = OUT.with_name(OUT.name + "_working")
REFERENCE_GROUP = "official_uri_embedding_folds_quarter_plus_current"
ARMS = ("no_edges", "relation_graph")
SLOTS = ("LL", "LT", "RL", "RT")
CROSS_VIEW = (1, 0, 3, 2)
CONTRALATERAL_SAME_VIEW = (2, 3, 0, 1)
CONTRALATERAL_OTHER_VIEW = (3, 2, 1, 0)
MAX_EPOCHS = 100
MIN_EPOCHS = 15
PATIENCE = 20


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_preprocessor(
    values: np.ndarray,
    images: pd.DataFrame,
    patient_aliases: set[str],
    seed: int,
) -> Pipeline:
    selected = images.patient_alias.isin(patient_aliases).to_numpy()
    if selected.sum() < 32:
        raise ValueError("Too few training images for PCA32")
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(32, whiten=True, random_state=seed)),
        ]
    )
    pipeline.fit(values[selected])
    return pipeline


def make_patient_slots(
    image_values: np.ndarray,
    images: pd.DataFrame,
    patients: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if len(image_values) != len(images):
        raise ValueError("Image features and manifest have different lengths")
    if images.duplicated(["patient_alias", "view"]).any():
        raise ValueError("Expected at most one image in each patient/view slot")
    patient_lookup = {alias: index for index, alias in enumerate(patients.patient_alias)}
    slot_lookup = {name: index for index, name in enumerate(SLOTS)}
    slots = np.zeros((len(patients), len(SLOTS), image_values.shape[1]), np.float32)
    mask = np.zeros((len(patients), len(SLOTS)), bool)
    for row_index, row in images.reset_index(drop=True).iterrows():
        patient_index = patient_lookup[str(row.patient_alias)]
        slot_index = slot_lookup[str(row["view"])]
        slots[patient_index, slot_index] = image_values[row_index]
        mask[patient_index, slot_index] = True
    if not mask.any(1).all():
        raise ValueError("Every scored patient must have at least one image")
    return slots, mask


def make_patient_targets(
    patients: pd.DataFrame,
    kidneys: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    patient_lookup = {alias: index for index, alias in enumerate(patients.patient_alias)}
    labels = np.zeros((len(patients), 2), np.float32)
    mask = np.zeros((len(patients), 2), bool)
    for row in kidneys.itertuples(index=False):
        patient_index = patient_lookup[str(row.patient_alias)]
        side_index = 0 if row.side == "left" else 1
        labels[patient_index, side_index] = float(row.kidney_APN)
        mask[patient_index, side_index] = True
    return labels, mask


class MultiviewRelationGraph(nn.Module):
    """Four-slot patient graph with a parameter-matched disconnected control."""

    def __init__(self, relation_graph: bool, input_dim: int = 32, hidden_dim: int = 32):
        super().__init__()
        self.relation_graph = relation_graph
        self.input = nn.Linear(input_dim, hidden_dim)
        self.self_message = nn.Linear(hidden_dim, hidden_dim)
        self.cross_view_message = nn.Linear(hidden_dim, hidden_dim)
        self.contralateral_same_message = nn.Linear(hidden_dim, hidden_dim)
        self.contralateral_other_message = nn.Linear(hidden_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.Tanh(), nn.Linear(16, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def message_nodes(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.input(values))
        if self.relation_graph:
            cross = hidden[:, CROSS_VIEW]
            same = hidden[:, CONTRALATERAL_SAME_VIEW]
            other = hidden[:, CONTRALATERAL_OTHER_VIEW]
            cross_mask = mask[:, CROSS_VIEW].unsqueeze(-1)
            same_mask = mask[:, CONTRALATERAL_SAME_VIEW].unsqueeze(-1)
            other_mask = mask[:, CONTRALATERAL_OTHER_VIEW].unsqueeze(-1)
            cross = cross * cross_mask
            same = same * same_mask
            other = other * other_mask
        else:
            # This preserves the exact parameter count while removing every edge.
            cross = same = other = hidden
        message = (
            self.self_message(hidden)
            + self.cross_view_message(cross)
            + self.contralateral_same_message(same)
            + self.contralateral_other_message(other)
        )
        message = torch.nn.functional.gelu(self.normalization(message))
        return message * mask.unsqueeze(-1)

    def pool(self, nodes: torch.Tensor, mask: torch.Tensor, indices: tuple[int, int]) -> torch.Tensor:
        current = nodes[:, indices]
        current_mask = mask[:, indices]
        logits = self.attention(current).squeeze(-1).masked_fill(~current_mask, -1e4)
        weights = torch.softmax(logits, dim=1) * current_mask
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
        return (weights.unsqueeze(-1) * current).sum(1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        nodes = self.message_nodes(values, mask)
        left = self.pool(nodes, mask, (0, 1))
        right = self.pool(nodes, mask, (2, 3))
        left_logit = self.head(torch.cat([left, right, torch.abs(left - right)], dim=1))
        right_logit = self.head(torch.cat([right, left, torch.abs(left - right)], dim=1))
        return torch.cat([left_logit, right_logit], dim=1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def masked_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    kidney_mask: torch.Tensor,
    patient_indices: np.ndarray,
) -> torch.Tensor:
    selected = torch.as_tensor(patient_indices, dtype=torch.long)
    valid = kidney_mask[selected]
    target = labels[selected][valid]
    prediction = logits[selected][valid]
    positive = target.sum().clamp_min(1.0)
    negative = (1.0 - target).sum().clamp_min(1.0)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        prediction, target, pos_weight=negative / positive
    )


def probabilities_to_kidneys(
    probabilities: np.ndarray,
    patients: pd.DataFrame,
    kidneys: pd.DataFrame,
) -> np.ndarray:
    lookup = {
        (str(alias), side): float(probabilities[index, side_index])
        for index, alias in enumerate(patients.patient_alias)
        for side_index, side in enumerate(("left", "right"))
    }
    return np.asarray(
        [lookup[(str(row.patient_alias), str(row.side))] for row in kidneys.itertuples(index=False)],
        dtype=float,
    )


def evaluate_patients(
    model: nn.Module,
    slots: torch.Tensor,
    slot_mask: torch.Tensor,
    patients: pd.DataFrame,
    kidneys: pd.DataFrame,
    patient_indices: np.ndarray,
) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(model(slots, slot_mask)).cpu().numpy()
    patient_aliases = set(patients.patient_alias.iloc[patient_indices])
    kidney_indices = np.flatnonzero(kidneys.patient_alias.isin(patient_aliases))
    kidney_probability = probabilities_to_kidneys(probability, patients, kidneys)
    return base.metric(
        kidneys.kidney_APN.to_numpy(int)[kidney_indices], kidney_probability[kidney_indices]
    )


def train_model(
    slots: np.ndarray,
    slot_mask: np.ndarray,
    labels: np.ndarray,
    kidney_mask: np.ndarray,
    patients: pd.DataFrame,
    kidneys: pd.DataFrame,
    train_patients: np.ndarray,
    arm: str,
    seed: int,
    epochs: int,
    validation_patients: np.ndarray | None = None,
) -> tuple[MultiviewRelationGraph, int, list[dict[str, float]]]:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    seed_everything(seed)
    model = MultiviewRelationGraph(arm == "relation_graph")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    slot_tensor = torch.from_numpy(slots)
    slot_mask_tensor = torch.from_numpy(slot_mask)
    label_tensor = torch.from_numpy(labels)
    kidney_mask_tensor = torch.from_numpy(kidney_mask)
    best_epoch = 1
    best_score = -np.inf
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(slot_tensor, slot_mask_tensor)
        loss = masked_loss(logits, label_tensor, kidney_mask_tensor, train_patients)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        record: dict[str, float] = {"epoch": epoch, "loss": float(loss.detach())}
        if validation_patients is not None:
            result = evaluate_patients(
                model,
                slot_tensor,
                slot_mask_tensor,
                patients,
                kidneys,
                validation_patients,
            )
            score = (result["auroc"] + result["ap"]) / 2.0
            record.update({"validation_auroc": result["auroc"], "validation_ap": result["ap"]})
            if score > best_score + 1e-12:
                best_score = score
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
            if epoch >= MIN_EPOCHS and wait >= PATIENCE:
                history.append(record)
                break
        history.append(record)
    return model, best_epoch, history


def transform_for_fit(
    raw_values: np.ndarray,
    images: pd.DataFrame,
    patients: pd.DataFrame,
    fit_patients: np.ndarray,
    seed: int,
) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    aliases = set(patients.patient_alias.iloc[fit_patients])
    preprocessor = fit_preprocessor(raw_values, images, aliases, seed)
    transformed = preprocessor.transform(raw_values).astype(np.float32)
    slots, slot_mask = make_patient_slots(transformed, images, patients)
    return preprocessor, slots, slot_mask


def protocol() -> dict[str, object]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Explicit same-kidney cross-view and contralateral same-view message passing may recover renal APN evidence that fixed kidney pooling obscures",
        "literature": [
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC6938161/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC7387180/",
        ],
        "cohort": "210 patients, 414 imaged kidneys, 820 contour-first images with corrected side-specific DMSA labels",
        "representation": "Mean of five official frozen URI-CADS 2048-dimensional image embeddings plus mean and SD of seven pathology outputs; training-only standardized PCA32",
        "slots": list(SLOTS),
        "relations": ["same kidney cross-view", "contralateral same-view", "contralateral other-view"],
        "control": "Identical parameter count and patient pooling, but every relation layer receives only its own node",
        "selection": "Within each 126-patient outer training role, fixed stratified 100/26 patient split selects earliest epoch separately for each arm; PCA is refitted inside the 100-patient fit set",
        "refit": "Fresh PCA and model initialization on all 126 outer training patients for the selected epoch",
        "evaluation": "Three fixed 42-patient development roles only; scoring unit kidney; both kidneys and all views remain in one role",
        "fusion": "Predeclared 20% and 25% candidate probability with unchanged official URI-CADS reference",
        "prohibited": "No development-selected architecture, graph edge, fusion weight, threshold, identifier, DMSA code in representation, or exploratory-test scoring",
        "target": "Development mean AUROC>0.8 and AP>0.8 followed by locked confirmation",
        "hyperparameters": {
            "hidden": 32,
            "lr": 0.001,
            "weight_decay": 0.001,
            "dropout": 0.20,
            "max_epochs": MAX_EPOCHS,
            "minimum_epochs": MIN_EPOCHS,
            "patience": PATIENCE,
        },
    }


def run() -> None:
    if OUT.exists() or WORK.exists():
        raise FileExistsError("Preserve existing multiview-graph experiment")
    WORK.mkdir(parents=True)
    (WORK / "models").mkdir()
    write_json(WORK / "protocol.json", protocol())
    patients, kidneys, images = cohort.load_reference()
    embeddings = np.load(uri.OUT / "official_fold_embeddings.npy")
    pathology = np.load(uri.OUT / "official_fold_pathology_scores.npy")
    if embeddings.shape != (820, 5, 2048) or pathology.shape != (820, 5, 7):
        raise ValueError("Unexpected URI-CADS cache geometry")
    raw_values = np.c_[embeddings.mean(1), pathology.mean(1), pathology.std(1)].astype(np.float32)
    labels, kidney_mask = make_patient_targets(patients, kidneys)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    reference = pd.read_csv(uri.OUT / "development_predictions.csv")
    reference = reference[reference.group.eq(REFERENCE_GROUP)].set_index(
        ["seed", "kidney_alias"], verify_integrity=True
    )
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    for seed in base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        patient_roles = patients.patient_alias.map(mapping)
        kidney_roles = kidneys.patient_alias.map(mapping)
        outer_train = np.flatnonzero(patient_roles.eq("train"))
        development_kidneys = np.flatnonzero(kidney_roles.eq("development"))
        inner_fit, inner_validation = train_test_split(
            outer_train,
            train_size=100,
            random_state=seed + 71107,
            stratify=patients.DMSA_APN.iloc[outer_train],
        )
        inner_preprocessor, inner_slots, inner_slot_mask = transform_for_fit(
            raw_values, images, patients, inner_fit, seed + 101
        )
        del inner_preprocessor
        outer_preprocessor, outer_slots, outer_slot_mask = transform_for_fit(
            raw_values, images, patients, outer_train, seed + 201
        )
        target = kidneys.iloc[development_kidneys]
        fixed_reference = reference.loc[
            [(seed, alias) for alias in target.kidney_alias], "score"
        ].to_numpy(float)
        draw_scores: dict[str, np.ndarray] = {"official_uri_fixed_reference": fixed_reference}
        for arm_index, arm in enumerate(ARMS):
            _, selected_epoch, history = train_model(
                inner_slots,
                inner_slot_mask,
                labels,
                kidney_mask,
                patients,
                kidneys,
                inner_fit,
                arm,
                seed + 1000 + arm_index,
                MAX_EPOCHS,
                inner_validation,
            )
            for row in history:
                history_rows.append({"seed": seed, "arm": arm, **row})
            model, _, _ = train_model(
                outer_slots,
                outer_slot_mask,
                labels,
                kidney_mask,
                patients,
                kidneys,
                outer_train,
                arm,
                seed + 2000 + arm_index,
                selected_epoch,
            )
            with torch.inference_mode():
                patient_probability = torch.sigmoid(
                    model(torch.from_numpy(outer_slots), torch.from_numpy(outer_slot_mask))
                ).numpy()
            candidate = probabilities_to_kidneys(patient_probability, patients, kidneys)[
                development_kidneys
            ]
            prefix = f"uri_multiview_{arm}"
            draw_scores[prefix] = candidate
            draw_scores[f"{prefix}_20pct_plus_reference"] = 0.20 * candidate + 0.80 * fixed_reference
            draw_scores[f"{prefix}_25pct_plus_reference"] = 0.25 * candidate + 0.75 * fixed_reference
            selections.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "inner_fit_patients": len(inner_fit),
                    "inner_validation_patients": len(inner_validation),
                    "selected_epoch": selected_epoch,
                    "parameters": parameter_count(model),
                }
            )
            torch.save(
                {
                    "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "arm": arm,
                    "selected_epoch": selected_epoch,
                },
                WORK / "models" / f"{seed}_{arm}.pth",
            )
        joblib.dump(outer_preprocessor, WORK / "models" / f"{seed}_pca32.joblib", compress=3)
        for group, score in draw_scores.items():
            result = base.metric(kidneys.kidney_APN.to_numpy(int)[development_kidneys], score)
            metrics.append(
                {
                    "seed": seed,
                    "role": "development",
                    "group": group,
                    "patients": target.patient_alias.nunique(),
                    "kidneys": len(target),
                    **result,
                }
            )
            for local, kidney_index in enumerate(development_kidneys):
                predictions.append(
                    {
                        "seed": seed,
                        "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[kidney_index],
                        "kidney_alias": kidneys.kidney_alias.iloc[kidney_index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[kidney_index],
                        "side": kidneys.side.iloc[kidney_index],
                        "label": int(kidneys.kidney_APN.iloc[kidney_index]),
                        "score": float(score[local]),
                    }
                )
        print(f"URI multiview graph draw {seed} complete", flush=True)
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    metric_frame.to_csv(WORK / "development_metrics.csv", index=False)
    prediction_frame.to_csv(WORK / "development_predictions.csv", index=False)
    summary.to_csv(WORK / "development_summary.csv")
    pd.DataFrame(selections).to_csv(WORK / "training_inner_selections.csv", index=False)
    pd.DataFrame(history_rows).to_csv(WORK / "training_inner_history.csv", index=False)
    groups = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups, base.SEEDS, "official_uri_fixed_reference"
    )
    ci.to_csv(WORK / "development_ci.csv", index=False)
    delta.to_csv(WORK / "paired_delta_ci.csv", index=False)
    np.save(WORK / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(
        WORK / "decision.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "groups_above_both_development_targets": achieved,
            "promotion": bool(achieved),
            "exploratory_test_rows_scored": 0,
            "parameter_counts_equal": len({row["parameters"] for row in selections}) == 1,
            "inputs": {
                str((uri.OUT / "official_fold_embeddings.npy").relative_to(ROOT)): digest(
                    uri.OUT / "official_fold_embeddings.npy"
                ),
                str((uri.OUT / "official_fold_pathology_scores.npy").relative_to(ROOT)): digest(
                    uri.OUT / "official_fold_pathology_scores.npy"
                ),
                str((base.OUT / "patient_assignments.csv").relative_to(ROOT)): digest(
                    base.OUT / "patient_assignments.csv"
                ),
            },
        },
    )
    best_group = summary.index[0]
    best = summary.loc[best_group]
    reference_row = summary.loc["official_uri_fixed_reference"]
    readme = f"""# NCKUH URI-CADS patient-local multiview graph

## Motivation and method

Kidney-ultrasound multi-instance literature reports that explicit relationships among
multiple views can outperform independent image pooling. This development-only control
therefore represented each patient as four fixed slots (LL, LT, RL, RT) using the
official frozen URI-CADS embeddings. A small relation graph passed messages across the
same kidney's orthogonal view and the contralateral kidney's matched and unmatched
views. The disconnected control had exactly the same trainable parameter count but
fed every relation layer only its own node.

PCA and model fitting used training patients only. A stratified 100/26 split inside
each 126-patient training role selected the earliest epoch separately for each arm,
after which preprocessing and the model were reset and refitted on all 126 patients.
Only the 42-patient development role was scored; no exploratory-test row was opened.

## Result

The leading registered configuration was `{best_group}` with mean AUROC
{best.auroc_mean:.4f} and AP {best.ap_mean:.4f}. The unchanged official URI-CADS
reference remained AUROC {reference_row.auroc_mean:.4f} and AP
{reference_row.ap_mean:.4f}. See `development_ci.csv` and `paired_delta_ci.csv` for
patient-cluster intervals and paired changes.

The relation graph is promoted only if both development means exceed 0.8. The formal
decision and integrity fields are in `decision.json`.
"""
    (WORK / "README.md").write_text(readme, encoding="utf-8")
    WORK.rename(OUT)
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    run()
