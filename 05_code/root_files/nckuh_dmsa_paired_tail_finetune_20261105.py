"""Patient-paired hierarchical RadImageNet tail adaptation for NCKUH APN."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from torch import nn

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_fullcohort_radimagenet_finetune_20261103 as radft
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_multiquery_mil_20261103 as cohort
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_uricads_transfer_20261104 import PathologyTail


ROOT = Path(__file__).resolve().parents[2]
RAD_SOURCE = ROOT / "02_experiments/CYCH_feature_research_results_20260913/source"
PREFIX_CACHE = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_radimagenet_finetune_20261103/prefix_maps_float16.npy"
URI_MODEL = ROOT / "02_experiments/NCKUH_DMSA_uricads_transfer_20261104/models/uricads_tail_20261104.pt"
CURRENT = ROOT / "02_experiments/NCKUH_DMSA_hierarchical_tile_usfm_20261104/development_predictions.csv"
OUT = ROOT / "02_experiments/NCKUH_DMSA_paired_tail_finetune_20261105"
CHECKPOINT_EPOCHS = (5, 10, 20, 40)
ARMS = ("radimagenet_pair", "uricads_pair")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def probability_from_logits(
    any_logit: torch.Tensor,
    bilateral_logit: torch.Tensor,
    left_logit: torch.Tensor,
) -> torch.Tensor:
    p_any = torch.sigmoid(any_logit)
    p_bilateral = torch.sigmoid(bilateral_logit)
    p_left = torch.sigmoid(left_logit)
    p_n = 1 - p_any
    p_b = p_any * p_bilateral
    p_unilateral = p_any * (1 - p_bilateral)
    p_l = p_unilateral * p_left
    p_r = p_unilateral * (1 - p_left)
    return torch.stack((p_n, p_l, p_r, p_b), dim=1)


class PairedTail(nn.Module):
    def __init__(self, tail: nn.Module):
        super().__init__()
        self.tail = tail
        self.any_head = nn.Sequential(nn.LayerNorm(2050), nn.Dropout(0.2), nn.Linear(2050, 1))
        self.bilateral_head = nn.Sequential(
            nn.LayerNorm(2050), nn.Dropout(0.2), nn.Linear(2050, 1)
        )
        self.left_head = nn.Sequential(nn.LayerNorm(1025), nn.Dropout(0.2), nn.Linear(1025, 1))

    def forward(
        self, maps: torch.Tensor, availability: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        batch = len(maps)
        embedding = self.tail(maps.reshape(batch * 2, 512, 7, 7)).reshape(batch, 2, 1024)
        embedding = embedding * availability.unsqueeze(-1)
        left, right = embedding[:, 0], embedding[:, 1]
        symmetric = torch.cat(
            ((left + right) / 2, torch.abs(left - right), availability), dim=1
        )
        antisymmetric = torch.cat((left - right, availability[:, :1] - availability[:, 1:]), dim=1)
        logits = (
            self.any_head(symmetric).squeeze(1),
            self.bilateral_head(symmetric).squeeze(1),
            self.left_head(antisymmetric).squeeze(1),
        )
        return probability_from_logits(*logits), logits


def construct(arm: str, seed: int) -> PairedTail:
    sys.path.insert(0, str(RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    torch.manual_seed(seed)
    original = radimagenet_densenet121().cuda().eval()
    _, tail = radft.split_encoder(original)
    if arm == "uricads_pair":
        pathology = PathologyTail(copy.deepcopy(tail)).cuda()
        blob = torch.load(URI_MODEL, map_location="cpu", weights_only=True)
        pathology.load_state_dict(blob["state_dict"])
        tail = pathology.tail
    elif arm != "radimagenet_pair":
        raise ValueError(arm)
    return PairedTail(copy.deepcopy(tail)).cuda()


def keep_bn_fixed(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def patient_side_maps(
    maps: np.ndarray, images: pd.DataFrame, patients: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(patients), 2, 512, 7, 7), dtype=np.float32)
    availability = np.zeros((len(patients), 2), dtype=np.float32)
    for patient_index, alias in enumerate(patients.patient_alias):
        for side_index, side in enumerate(("left", "right")):
            rows = images.index[(images.patient_alias == alias) & (images.side == side)].to_numpy()
            if len(rows):
                values[patient_index, side_index] = np.asarray(maps[rows], dtype=np.float32).mean(0)
                availability[patient_index, side_index] = 1
    return values, availability


def augment(maps: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    output = maps.clone()
    flip = torch.rand((len(output), 2), generator=generator) < 0.5
    for side in range(2):
        selected = flip[:, side].to(output.device)
        if selected.any():
            output[selected, side] = torch.flip(output[selected, side], dims=(-1,))
    keep = torch.rand((len(output), 2, 1, 7, 7), generator=generator) >= 0.05
    noise = torch.randn(output.shape, generator=generator, dtype=output.dtype)
    return output * keep.to(output.device) + 0.01 * noise.to(output.device)


def positive_weight(labels: np.ndarray) -> torch.Tensor:
    positive = max(float(labels.sum()), 1.0)
    negative = max(float(len(labels) - labels.sum()), 1.0)
    return torch.tensor(min(negative / positive, 10.0), device="cuda")


def hierarchical_loss(
    logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    states: torch.Tensor,
    state_values: np.ndarray,
) -> torch.Tensor:
    any_logit, bilateral_logit, left_logit = logits
    any_label = (states != hierarchy.STATE_CODE["N"]).float()
    any_loss = nn.functional.binary_cross_entropy_with_logits(
        any_logit, any_label, pos_weight=positive_weight(state_values != hierarchy.STATE_CODE["N"])
    )
    losses = [any_loss]
    affected = states != hierarchy.STATE_CODE["N"]
    if affected.any():
        bilateral_label = (states[affected] == hierarchy.STATE_CODE["B"]).float()
        losses.append(
            nn.functional.binary_cross_entropy_with_logits(
                bilateral_logit[affected],
                bilateral_label,
                pos_weight=positive_weight(
                    state_values[state_values != 0] == hierarchy.STATE_CODE["B"]
                ),
            )
        )
    unilateral = (states == hierarchy.STATE_CODE["L"]) | (states == hierarchy.STATE_CODE["R"])
    if unilateral.any():
        left_label = (states[unilateral] == hierarchy.STATE_CODE["L"]).float()
        unilateral_values = state_values[np.isin(state_values, [1, 2])]
        losses.append(
            nn.functional.binary_cross_entropy_with_logits(
                left_logit[unilateral],
                left_label,
                pos_weight=positive_weight(unilateral_values == hierarchy.STATE_CODE["L"]),
            )
        )
    return sum(losses)


def predict(
    model: PairedTail, maps: np.ndarray, availability: np.ndarray, selected: np.ndarray
) -> np.ndarray:
    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(selected), 64):
            index = selected[start : start + 64]
            tensor = torch.from_numpy(maps[index]).cuda()
            present = torch.from_numpy(availability[index]).cuda()
            probability, _ = model(tensor, present)
            rows.append(probability.cpu().numpy())
    return np.concatenate(rows)


def train_model(
    arm: str,
    seed: int,
    maps: np.ndarray,
    availability: np.ndarray,
    states: np.ndarray,
    selected: np.ndarray,
    epochs: int,
    patients: pd.DataFrame,
    kidneys: pd.DataFrame,
    validation: np.ndarray | None = None,
) -> tuple[PairedTail, list[dict]]:
    model = construct(arm, seed)
    anchor = {name: parameter.detach().clone() for name, parameter in model.tail.named_parameters()}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.any_head.parameters())
                + list(model.bilateral_head.parameters())
                + list(model.left_head.parameters()),
                "lr": 1e-3,
                "weight_decay": 1e-4,
            },
            {"params": model.tail.parameters(), "lr": 1e-5, "weight_decay": 0.0},
        ]
    )
    rng = np.random.default_rng(seed + 700)
    generator = torch.Generator(device="cpu").manual_seed(seed + 800)
    state_tensor = torch.from_numpy(states.astype(np.int64)).cuda()
    records: list[dict] = []
    for epoch in range(1, epochs + 1):
        order = selected[rng.permutation(len(selected))]
        model.train()
        keep_bn_fixed(model)
        for start in range(0, len(order), 16):
            index = order[start : start + 16]
            tensor = augment(torch.from_numpy(maps[index]).cuda(), generator)
            present = torch.from_numpy(availability[index]).cuda()
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(tensor, present)
            loss = hierarchical_loss(logits, state_tensor[index], states[index])
            penalty = sum(
                (parameter - anchor[name]).pow(2).sum()
                for name, parameter in model.tail.named_parameters()
            )
            loss = loss + 1e-4 * penalty
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        if validation is not None and epoch in CHECKPOINT_EPOCHS:
            probability = predict(model, maps, availability, validation)
            patient_lookup = {alias: index for index, alias in enumerate(patients.patient_alias)}
            validation_aliases = set(patients.patient_alias.iloc[validation])
            kidney_index = np.flatnonzero(kidneys.patient_alias.isin(validation_aliases))
            scores = []
            for row in kidneys.iloc[kidney_index].itertuples(index=False):
                patient_probability = probability[
                    np.flatnonzero(validation == patient_lookup[row.patient_alias])[0]
                ]
                scores.append(
                    patient_probability[hierarchy.STATE_CODE["L"]]
                    + patient_probability[hierarchy.STATE_CODE["B"]]
                    if row.side == "left"
                    else patient_probability[hierarchy.STATE_CODE["R"]]
                    + patient_probability[hierarchy.STATE_CODE["B"]]
                )
            result = hierarchy.metric(
                kidneys.kidney_APN.to_numpy(int)[kidney_index], np.asarray(scores)
            )
            records.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "epoch": epoch,
                    "inner_auroc": result["auroc"],
                    "inner_ap": result["ap"],
                    "selection_score": (result["auroc"] + result["ap"]) / 2,
                }
            )
    return model, records


def run() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve completed experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    patients, kidneys, images = cohort.load_reference()
    prefix = np.load(PREFIX_CACHE, mmap_mode="r")
    maps, availability = patient_side_maps(prefix, images, patients)
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    current = pd.read_csv(CURRENT)
    current = current[current.group.eq("hierarchical_tile_usfm_quarter_plus_fixed")].set_index(
        ["seed", "kidney_alias"], verify_integrity=True
    )
    metrics, predictions, selections = [], [], []
    for seed in base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        patient_roles = patients.patient_alias.map(mapping)
        kidney_roles = kidneys.patient_alias.map(mapping)
        outer_train = np.flatnonzero(patient_roles.eq("train"))
        development = np.flatnonzero(kidney_roles.eq("development"))
        fit, select = train_test_split(
            outer_train,
            train_size=100,
            random_state=seed + 5000,
            stratify=states[outer_train],
        )
        target = kidneys.iloc[development]
        current_rows = current.loc[[(seed, alias) for alias in target.kidney_alias]]
        np.testing.assert_array_equal(current_rows.label, labels[development])
        reference = current_rows.score.to_numpy()
        draw_scores = {"current_tile_usfm_reference": reference}
        for arm in ARMS:
            _, records = train_model(
                arm,
                seed,
                maps,
                availability,
                states,
                fit,
                max(CHECKPOINT_EPOCHS),
                patients,
                kidneys,
                select,
            )
            selections.extend(records)
            chosen = max(records, key=lambda row: (row["selection_score"], -row["epoch"]))["epoch"]
            model, _ = train_model(
                arm,
                seed + 1000,
                maps,
                availability,
                states,
                outer_train,
                int(chosen),
                patients,
                kidneys,
            )
            probability = predict(model, maps, availability, np.arange(len(patients)))
            score = hierarchy.kidney_scores(probability, patients, kidneys)[development]
            draw_scores[arm] = score
            draw_scores[f"{arm}_quarter_plus_current"] = 0.25 * score + 0.75 * reference
            draw_scores[f"{arm}_equal_plus_current"] = 0.50 * score + 0.50 * reference
            torch.save(
                {"state_dict": model.state_dict(), "selected_epoch": int(chosen)},
                OUT / "models" / f"{seed}_{arm}.pt",
            )
            del model
            torch.cuda.empty_cache()
        for group, score in draw_scores.items():
            result = hierarchy.metric(labels[development], score)
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
            for local, index in enumerate(development):
                predictions.append(
                    {
                        "seed": seed,
                        "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[index],
                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                        "side": kidneys.side.iloc[index],
                        "label": int(labels[index]),
                        "score": float(score[local]),
                    }
                )
        print(f"Paired tail draw {seed} complete", flush=True)
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv")
    pd.DataFrame(selections).to_csv(OUT / "training_inner_selections.csv", index=False)
    groups = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups, base.SEEDS, "current_tile_usfm_reference"
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(
        OUT / "protocol_and_decision.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "motivation": "Directly optimize patient-paired any/bilateral/laterality heads instead of assigning one kidney label to every image",
            "arms": list(ARMS),
            "selection": "100 fit and 26 selection patients within each 126-patient outer training role",
            "checkpoints": list(CHECKPOINT_EPOCHS),
            "prefix_cache_sha256": digest(PREFIX_CACHE),
            "uri_initialization_sha256": digest(URI_MODEL),
            "groups_above_both_development_targets": achieved,
            "exploratory_test_rows_scored": 0,
            "promotion": bool(achieved),
        },
    )
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
