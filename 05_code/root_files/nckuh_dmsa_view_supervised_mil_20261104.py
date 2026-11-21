"""View-supervised multi-query MIL on contour-constrained kidney patches."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

import nckuh_dmsa_multiquery_mil_20261103 as prior


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_fullframe_20261103"
FIXED = ROOT / "02_experiments/NCKUH_DMSA_tile_replacement_20261103"
OUT = ROOT / "02_experiments/NCKUH_DMSA_view_supervised_mil_20261104"
SEEDS = prior.SEEDS
EPOCHS = prior.EPOCHS
ARMS = {
    "view_supervised_q2": {"roles": (0, 1), "diversity_weight": 0.0},
    "view_supervised_q4": {"roles": (0, 0, 1, 1), "diversity_weight": 0.05},
    "view_hybrid_q4": {"roles": (0, 1, -1, -1), "diversity_weight": 0.05},
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


class ViewSupervisedMIL(nn.Module):
    """Assign queries to longitudinal, transverse, or unrestricted evidence."""

    def __init__(self, roles: tuple[int, ...]):
        super().__init__()
        self.query = nn.Parameter(torch.randn(len(roles), 32) * 0.02)
        self.view_embedding = nn.Embedding(2, 32)
        self.input = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.LayerNorm(32))
        self.classifier = nn.Sequential(
            nn.LayerNorm(len(roles) * 32),
            nn.Dropout(0.25),
            nn.Linear(len(roles) * 32, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.register_buffer("query_roles", torch.tensor(roles, dtype=torch.long))

    def forward(self, values: torch.Tensor, mask: torch.Tensor, views: torch.Tensor):
        hidden = self.input(values) + self.view_embedding(views.clamp_min(0))
        scores = torch.einsum("qd,bnd->bqn", self.query, hidden) / (hidden.shape[-1] ** 0.5)
        roles = self.query_roles.view(1, -1, 1)
        allowed = mask.unsqueeze(1) & ((roles < 0) | (views.unsqueeze(1) == roles))
        missing = ~allowed.any(dim=2, keepdim=True)
        allowed = torch.where(missing, mask.unsqueeze(1), allowed)
        weights = torch.softmax(scores.masked_fill(~allowed, -1e4), dim=2)
        weights = weights * allowed
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        evidence = torch.einsum("bqn,bnd->bqd", weights, hidden)
        logits = self.classifier(evidence.flatten(1)).squeeze(1)
        return logits, weights, evidence


def train_model(arm, seed, bags, mask, views, labels, fit, epochs, evaluation):
    settings = ARMS[arm]
    torch.manual_seed(seed + 137)
    model = ViewSupervisedMIL(settings["roles"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    target = labels[fit]
    sample_weight = torch.where(
        target > 0,
        len(target) / (2 * target.sum()),
        len(target) / (2 * (len(target) - target.sum())),
    )
    history = {}
    for epoch in range(1, max(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, attention, _ = model(bags[fit], mask[fit], views[fit])
        bce = (
            nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
            * sample_weight
        ).mean()
        loss = bce + settings["diversity_weight"] * prior.attention_diversity(attention)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch in epochs and evaluation is not None:
            model.eval()
            with torch.inference_mode():
                evaluation_logits = model(bags[evaluation], mask[evaluation], views[evaluation])[0]
            history[epoch] = torch.sigmoid(evaluation_logits).cpu().numpy()
    return model, history


def run() -> None:
    from sklearn.model_selection import train_test_split

    if OUT.exists():
        raise FileExistsError(f"Preserve completed experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    patients, kidneys, images = prior.load_reference()
    values, patch_patients, patch_views, groups = prior.patch_data(images, kidneys)
    labels_np = kidneys.kidney_APN.to_numpy(int)
    labels = torch.from_numpy(labels_np.astype(np.float32)).cuda()
    assignments = pd.read_csv(BASE / "patient_assignments.csv")
    fixed = pd.read_csv(FIXED / "development_predictions.csv")
    fixed = fixed[fixed.group.eq("replace_with_tile_two_heads")].set_index(
        ["seed", "kidney_alias"], verify_integrity=True
    )

    records, predictions, selections = [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train_patients = patients[
            patients.patient_alias.isin(mapping[mapping.eq("train")].index)
        ].reset_index(drop=True)
        fit_local, select_local = train_test_split(
            np.arange(len(train_patients)), train_size=100, random_state=seed + 6000,
            stratify=train_patients.DMSA_APN,
        )
        fit_alias = train_patients.patient_alias.iloc[fit_local]
        select_alias = train_patients.patient_alias.iloc[select_local]
        fit = np.flatnonzero(kidneys.patient_alias.isin(fit_alias))
        selection = np.flatnonzero(kidneys.patient_alias.isin(select_alias))
        bags, mask, views, _ = prior.prepare_bags(
            values, patch_patients, patch_views, groups, set(fit_alias), seed,
        )
        chosen_epochs = {}
        for arm in ARMS:
            _, history = train_model(arm, seed, bags, mask, views, labels, fit, EPOCHS, selection)
            choices = []
            for epoch, score in history.items():
                result = prior.metric(labels_np[selection], score)
                choices.append((-(result["auroc"] + result["ap"]) / 2.0, epoch, result))
            _, chosen, result = min(choices, key=lambda item: (item[0], item[1]))
            chosen_epochs[arm] = chosen
            selections.append({
                "seed": seed, "arm": arm, "selected_epoch": chosen,
                "inner_auroc": result["auroc"], "inner_ap": result["ap"],
            })
            torch.cuda.empty_cache()

        train = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("train"))
        development = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        train_aliases = set(mapping[mapping.eq("train")].index)
        bags, mask, views, pca = prior.prepare_bags(
            values, patch_patients, patch_views, groups, train_aliases, seed + 100,
        )
        target = kidneys.iloc[development]
        fixed_rows = fixed.loc[[(seed, alias) for alias in target.kidney_alias]]
        np.testing.assert_array_equal(fixed_rows.label, labels_np[development])
        reference = fixed_rows.score.to_numpy()
        reference_result = prior.metric(labels_np[development], reference)
        records.append({
            "seed": seed, "role": "development", "group": "fixed_chain_reference",
            "patients": 42, "kidneys": len(development), **reference_result,
        })
        for local, index in enumerate(development):
            predictions.append({
                "seed": seed, "group": "fixed_chain_reference",
                "patient_alias": kidneys.patient_alias.iloc[index],
                "kidney_alias": kidneys.kidney_alias.iloc[index],
                "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                "side": kidneys.side.iloc[index], "label": int(labels_np[index]),
                "score": float(reference[local]),
            })

        for arm in ARMS:
            model, _ = train_model(
                arm, seed + 10000, bags, mask, views, labels, train,
                (chosen_epochs[arm],), None,
            )
            model.eval()
            with torch.inference_mode():
                score = torch.sigmoid(
                    model(bags[development], mask[development], views[development])[0]
                ).cpu().numpy()
            candidate_scores = {
                arm: score,
                f"{arm}_quarter_plus_fixed": 0.25 * score + 0.75 * reference,
                f"{arm}_equal_plus_fixed": 0.50 * score + 0.50 * reference,
            }
            torch.save({
                "state_dict": model.state_dict(), "selected_epoch": chosen_epochs[arm],
                "seed": seed, "arm": arm,
            }, OUT / "models" / f"{seed}_{arm}.pth")
            joblib.dump(pca, OUT / "models" / f"{seed}_{arm}_pca.joblib", compress=3)
            for group, candidate_score in candidate_scores.items():
                result = prior.metric(labels_np[development], candidate_score)
                records.append({
                    "seed": seed, "role": "development", "group": group,
                    "patients": 42, "kidneys": len(development), **result,
                })
                for local, index in enumerate(development):
                    predictions.append({
                        "seed": seed, "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[index],
                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                        "side": kidneys.side.iloc[index], "label": int(labels_np[index]),
                        "score": float(candidate_score[local]),
                    })
            del model
            torch.cuda.empty_cache()
        print(f"Completed view-supervised MIL draw {seed}", flush=True)

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
        "motivation": "Use known acquisition-plane labels to prevent latent-query collapse across longitudinal and transverse kidney evidence",
        "literature_basis": [
            "Huang et al., Supervised Attention in Multiple Instance Learning, PMLR 219 (2023)",
            "C2LUE arXiv:2610.14143 multi-query ultrasound evidence aggregation",
        ],
        "instances": "Kidney-mask-constrained RadImageNet local patches; view roles are acquisition metadata, not APN annotations",
        "arms": {key: {**value, "roles": list(value["roles"])} for key, value in ARMS.items()},
        "selection": "Fixed 100/26 patient split inside each 126-patient training role; epoch selected from 10/20/40/80",
        "groups_above_both_development_targets": achieved,
        "exploratory_test_scored": False,
    })
    write_json(OUT / "integrity_audit.json", {
        "patients": len(patients), "kidneys": len(kidneys), "images": len(images),
        "prediction_rows": len(prediction_frame), "prediction_roles": ["development"],
        "exploratory_test_rows_scored": 0,
        "scores_finite": bool(np.isfinite(prediction_frame.score).all()),
        "scores_in_unit_interval": bool(prediction_frame.score.between(0, 1).all()),
    })
    print(summary.to_string(), flush=True)


def bootstrap() -> None:
    from nckuh_dmsa_cluster_ci import cluster_intervals

    prediction = pd.read_csv(OUT / "development_predictions.csv")
    if "DMSA_APN" not in prediction:
        _, kidneys, _ = prior.load_reference()
        state = kidneys.set_index("kidney_alias").DMSA_APN
        prediction.insert(
            prediction.columns.get_loc("side"),
            "DMSA_APN",
            prediction.kidney_alias.map(state),
        )
        if prediction.DMSA_APN.isna().any():
            raise ValueError("Could not restore a DMSA state for every scored kidney")
        prediction.to_csv(OUT / "development_predictions.csv", index=False)
    groups = prediction.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(
        prediction, groups, SEEDS, "fixed_chain_reference"
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "bootstrap"), nargs="?", default="run")
    args = parser.parse_args()
    {"run": run, "bootstrap": bootstrap}[args.stage]()

