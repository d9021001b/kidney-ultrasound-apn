"""Rank-aware frozen-feature heads on clean NCKUH full-frame kidney ROIs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import nckuh_dmsa_fullframe_clean_probe_20261102 as F
import nckuh_dmsa_fullframe_deep_probe_20261102 as D
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json

ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullframe_rank_probe_20261102"
ARMS = ("bce_mlp", "bce_global_rank", "bce_global_and_unilateral_rank")
GROUPS = ARMS + tuple(f"{arm}_plus_clean_radiomics" for arm in ARMS)
REFERENCE = "deep_concat_plus_clean_radiomics"
EPOCHS = (10, 20, 40, 80)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered full-frame rank protocol")
    F.require_lock(); D.require_lock()
    inputs = [
        F.OUT / "patient_assignments.csv", F.OUT / "deidentified_image_manifest.csv",
        D.OUT / "radimagenet_gray.npy", D.OUT / "radimagenet_color.npy", D.OUT / "usf_gray.npy",
        D.OUT / "development_predictions.csv", Path(__file__),
    ]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same clean-full-frame111 patients,212 kidneys,418 images and exact70/20/21 patient roles",
        "input": "Frozen RadImageNet grayscale/color and USF-MAE image embeddings; longitudinal/transverse and ordered own/opposite context; train-only standardized PCA64",
        "head": "64-32-ReLU-dropout0.2-linear MLP, balanced full-batch BCE, AdamW lr0.001 weight_decay0.001",
        "rank_losses": {
            "global": "RankNet softplus(negative logit-positive logit) over all training positive/negative kidneys, weight0.2",
            "unilateral": "Same-patient discordant positive-over-negative kidney RankNet loss, additional weight0.5; bilateral-positive and bilateral-negative patients receive no ordering",
        },
        "selection": "Within each70-patient outer training role, fixed55/15 R/L/B/N-stratified inner split selects10/20/40/80 epochs by mean(AUROC,AP), tie lower epoch; reset and refit all70",
        "fusion": "Each registered head has one fixed equal-probability average with the completed clean-full-frame radiomics ANOVA64 LR branch",
        "prohibited": "No DMSA or opposite-kidney label as inference input, no positive patch label, no development-selected loss weight/fusion weight/threshold, no exploratory-test scoring",
        "target": "Development AUROC>0.8 and AP>0.8; full objective requires later locked confirmation",
        "arms": list(ARMS), "groups": list(GROUPS), "epochs": list(EPOCHS),
        "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "source_sha256": digest(__file__), "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs},
    })
    print("Registered clean full-frame rank-aware probe", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(Path(name)) == expected, name


def unilateral_pairs(indices: np.ndarray, kidneys: pd.DataFrame, labels: np.ndarray) -> list[tuple[int, int]]:
    position = {int(index): local for local, index in enumerate(indices)}
    pairs = []
    subset = kidneys.iloc[indices].copy()
    subset["global_index"] = indices
    for _, frame in subset.groupby("patient_alias"):
        if len(frame) != 2:
            continue
        global_indices = frame.global_index.to_numpy(int)
        outcomes = labels[global_indices]
        if sorted(outcomes.tolist()) != [0, 1]:
            continue
        positive = global_indices[int(np.argmax(outcomes))]
        negative = global_indices[int(np.argmin(outcomes))]
        pairs.append((position[int(positive)], position[int(negative)]))
    return pairs


def rank_loss(logits, labels):
    import torch
    positive = logits[labels > .5]
    negative = logits[labels <= .5]
    if not len(positive) or not len(negative):
        return logits.sum() * 0
    return torch.nn.functional.softplus(negative[:, None] - positive[None, :]).mean()


def unilateral_rank_loss(logits, pairs):
    import torch
    if not pairs:
        return logits.sum() * 0
    positive = torch.stack([logits[p] for p, _ in pairs])
    negative = torch.stack([logits[n] for _, n in pairs])
    return torch.nn.functional.softplus(negative - positive).mean()


def make_preprocessor(seed: int):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         PCA(64, whiten=True, random_state=seed))


def make_head(seed: int):
    import torch
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(),
                               torch.nn.Dropout(.2), torch.nn.Linear(32, 1))


def train_head(arm, seed, x_train, y_train, pairs, epochs, evaluation=None):
    import torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = make_head(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    x = torch.from_numpy(np.asarray(x_train, np.float32))
    y = torch.from_numpy(np.asarray(y_train, np.float32))
    weights = torch.where(y > .5, len(y)/(2*y.sum()), len(y)/(2*(len(y)-y.sum())))
    history = {}
    for epoch in range(1, max(epochs)+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, weight=weights)
        if arm in (ARMS[1], ARMS[2]):
            loss = loss + .2 * rank_loss(logits, y)
        if arm == ARMS[2]:
            loss = loss + .5 * unilateral_rank_loss(logits, pairs)
        loss.backward(); optimizer.step()
        if evaluation is not None and epoch in epochs:
            model.eval()
            with torch.inference_mode():
                history[epoch] = torch.sigmoid(model(torch.from_numpy(np.asarray(evaluation, np.float32))).squeeze(1)).numpy()
    return model, history


def run() -> None:
    from sklearn.model_selection import train_test_split
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed rank-aware predictions")
    patients, kidneys, images = F.matched_reference()
    labels = kidneys.kidney_APN.to_numpy(int)
    image_arrays = [np.load(D.OUT / name) for name in ("radimagenet_gray.npy", "radimagenet_color.npy", "usf_gray.npy")]
    oriented = [F.orient_kidneys(F.patient_pair_features(values, images, patients), patients, kidneys)
                for values in image_arrays]
    values = np.concatenate(oriented, axis=1)
    assignments = pd.read_csv(F.OUT / "patient_assignments.csv")
    reference = pd.read_csv(F.OUT / "development_predictions.csv")
    reference = reference[reference.group.eq("clean_fullframe_radiomics_anova64_lr")].set_index(["seed", "kidney_alias"]).score
    predictions, metrics, selections = [], [], []
    (OUT / "models").mkdir(exist_ok=True)
    for seed in F.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        roles = kidneys.patient_alias.map(mapping)
        outer_train = np.flatnonzero(roles.eq("train")); outer_dev = np.flatnonzero(roles.eq("development"))
        train_patients = patients[patients.patient_alias.isin(mapping[mapping.eq("train")].index)].reset_index(drop=True)
        inner_fit_patient, inner_select_patient = train_test_split(
            np.arange(len(train_patients)), train_size=55, random_state=seed+2000,
            stratify=train_patients.DMSA_APN)
        fit_alias = set(train_patients.patient_alias.iloc[inner_fit_patient])
        select_alias = set(train_patients.patient_alias.iloc[inner_select_patient])
        inner_fit = np.asarray([i for i in outer_train if kidneys.patient_alias.iloc[i] in fit_alias])
        inner_select = np.asarray([i for i in outer_train if kidneys.patient_alias.iloc[i] in select_alias])
        inner_pre = make_preprocessor(seed).fit(values[inner_fit])
        inner_x = inner_pre.transform(values)
        inner_pairs = unilateral_pairs(inner_fit, kidneys, labels)
        selected_epochs = {}
        for arm in ARMS:
            _, history = train_head(arm, seed, inner_x[inner_fit], labels[inner_fit], inner_pairs,
                                    EPOCHS, inner_x[inner_select])
            choices = []
            for epoch, scores in history.items():
                result = metric(labels[inner_select], scores)
                choices.append((-(result["auroc"] + result["ap"])/2, epoch, result))
            _, selected, selected_result = min(choices, key=lambda item: (item[0], item[1]))
            selected_epochs[arm] = selected
            selections.append({"seed": seed, "arm": arm, "selected_epoch": selected,
                               "inner_patients": 15, "inner_kidneys": len(inner_select),
                               "inner_auroc": selected_result["auroc"], "inner_ap": selected_result["ap"]})
        pre = make_preprocessor(seed).fit(values[outer_train])
        transformed = pre.transform(values)
        pairs = unilateral_pairs(outer_train, kidneys, labels)
        joblib.dump(pre, OUT / "models" / f"{seed}_preprocess.joblib", compress=3)
        scores = {}
        for arm in ARMS:
            model, _ = train_head(arm, seed, transformed[outer_train], labels[outer_train], pairs,
                                  (selected_epochs[arm],))
            import torch
            model.eval()
            with torch.inference_mode():
                scores[arm] = torch.sigmoid(model(torch.from_numpy(transformed[outer_dev].astype(np.float32))).squeeze(1)).numpy()
            torch.save({"state_dict": model.state_dict(), "epoch": selected_epochs[arm], "seed": seed,
                        "arm": arm}, OUT / "models" / f"{seed}_{arm}.pth")
        ref = np.asarray([reference.loc[(seed, alias)] for alias in kidneys.kidney_alias.iloc[outer_dev]])
        for arm in ARMS:
            scores[f"{arm}_plus_clean_radiomics"] = (scores[arm] + ref) / 2
        for group in GROUPS:
            result = metric(labels[outer_dev], scores[group])
            metrics.append({"seed": seed, "role": "development", "group": group,
                            "patients": 20, "kidneys": len(outer_dev), **result})
            for local, index in enumerate(outer_dev):
                predictions.append({"seed": seed, "group": group,
                                    "patient_alias": kidneys.patient_alias.iloc[index],
                                    "kidney_alias": kidneys.kidney_alias.iloc[index],
                                    "DMSA_APN": kidneys.DMSA_APN.iloc[index], "side": kidneys.side.iloc[index],
                                    "label": int(labels[index]), "score": float(scores[group][local])})
        print(f"Completed rank-aware draw {seed}", flush=True)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    metric_frame = pd.DataFrame(metrics); metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "inner_epoch_selections.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap"]].mean().sort_values(["auroc", "ap"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc > .8) & (summary.ap > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {
        "groups_above_both_development_targets": achieved, "test_scored": False,
        "goal_achieved": False, "reason": "Development-only subgroup cannot complete the full objective",
    })
    print(summary.to_string(), flush=True)


def bootstrap() -> None:
    require_lock()
    frame = pd.read_csv(OUT / "development_predictions.csv")
    previous = pd.read_csv(D.OUT / "development_predictions.csv")
    reference = previous[previous.group.eq(REFERENCE)]
    ci, paired, samples = cluster_intervals(pd.concat([frame, reference], ignore_index=True),
                                            list(GROUPS) + [REFERENCE], F.SEEDS, REFERENCE)
    ci.to_csv(OUT / "development_confidence_intervals.csv", index=False)
    paired.to_csv(OUT / "development_paired_intervals.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    write_json(OUT / "completion_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "patients": 111,
        "kidneys": 212, "images": 418, "test_scored": False,
        "scored_unit": "kidney", "bootstrap_cluster": "patient",
    })
    print("Rank-aware intervals completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "bootstrap"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "bootstrap": bootstrap}[args.stage]()
