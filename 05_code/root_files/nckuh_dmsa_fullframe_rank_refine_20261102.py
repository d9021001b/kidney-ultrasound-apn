"""Logistic-initialized rank refinement on clean full-frame frozen features."""
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
import nckuh_dmsa_fullframe_rank_probe_20261102 as R
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json

ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullframe_rank_refine_20261102"
ARMS = ("logistic_epoch0", "bce_adam", "global_rank", "global_and_unilateral_rank")
GROUPS = ARMS + tuple(f"{arm}_plus_clean_radiomics" for arm in ARMS)
REFERENCE = "deep_concat_plus_clean_radiomics"
EPOCHS = (0, 5, 10, 20, 40)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered logistic-initialized rank protocol")
    F.require_lock(); D.require_lock()
    inputs = [F.OUT / "patient_assignments.csv", D.OUT / "radimagenet_gray.npy",
              D.OUT / "radimagenet_color.npy", D.OUT / "usf_gray.npy",
              F.OUT / "development_predictions.csv", Path(__file__)]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same clean-full-frame111 patients,212 kidneys,418 images and exact70/20/21 roles",
        "representation": "Frozen three-encoder ordered kidney/view context; train-only imputation, standardization and whitened PCA64",
        "initialization": "Balanced liblinear logistic regression C0.1 fitted only on the current fitting patients; exact coefficients copied to a torch Linear head",
        "refinement": "AdamW lr0.001 weight_decay1e-5; balanced BCE alone or with global RankNet0.2 and same-patient unilateral RankNet0.5",
        "selection": "Within each70-patient training role, fixed55/15 stratified inner split selects epoch0/5/10/20/40 by mean(AUROC,AP), tie lower epoch; reset/refit all70",
        "fallback": "Epoch0 is always eligible, so a refinement arm may retain the original logistic head",
        "fusion": "Fixed equal-probability average with clean-full-frame radiomics ANOVA64 LR",
        "prohibited": "No diagnosis input, no development-selected weights or loss coefficients, no exploratory-test scoring",
        "target": "Development AUROC>0.8 and AP>0.8; later locked confirmation still required",
        "arms": list(ARMS), "groups": list(GROUPS), "epochs": list(EPOCHS),
        "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {"source_sha256": digest(__file__),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs}})
    print("Registered logistic-initialized rank refinement", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(Path(name)) == expected, name


def fit_logistic(x, y, seed):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=.1, class_weight="balanced", solver="liblinear",
                               max_iter=3000, random_state=seed)
    model.fit(x, y)
    return model


def torch_linear(model):
    import torch
    head = torch.nn.Linear(model.coef_.shape[1], 1)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(model.coef_.astype(np.float32)))
        head.bias.copy_(torch.from_numpy(model.intercept_.astype(np.float32)))
    return head


def refine(arm, initial, x_train, y_train, pairs, epochs, evaluation):
    import torch
    head = torch_linear(initial)
    x = torch.from_numpy(np.asarray(x_train, np.float32)); y = torch.from_numpy(np.asarray(y_train, np.float32))
    evaluation_tensor = torch.from_numpy(np.asarray(evaluation, np.float32))
    weights = torch.where(y > .5, len(y)/(2*y.sum()), len(y)/(2*(len(y)-y.sum())))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=1e-5)
    history = {}
    with torch.inference_mode():
        history[0] = torch.sigmoid(head(evaluation_tensor).squeeze(1)).numpy()
    if arm == ARMS[0]:
        return head, history
    for epoch in range(1, max(epochs)+1):
        optimizer.zero_grad(set_to_none=True)
        logits = head(x).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, weight=weights)
        if arm in (ARMS[2], ARMS[3]):
            loss = loss + .2 * R.rank_loss(logits, y)
        if arm == ARMS[3]:
            loss = loss + .5 * R.unilateral_rank_loss(logits, pairs)
        loss.backward(); optimizer.step()
        if epoch in epochs:
            with torch.inference_mode():
                history[epoch] = torch.sigmoid(head(evaluation_tensor).squeeze(1)).numpy()
    return head, history


def run() -> None:
    from sklearn.model_selection import train_test_split
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed logistic-initialized predictions")
    patients, kidneys, images = F.matched_reference()
    labels = kidneys.kidney_APN.to_numpy(int)
    image_arrays = [np.load(D.OUT / name) for name in ("radimagenet_gray.npy", "radimagenet_color.npy", "usf_gray.npy")]
    values = np.concatenate([F.orient_kidneys(F.patient_pair_features(array, images, patients), patients, kidneys)
                             for array in image_arrays], axis=1)
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
        fit_patient, select_patient = train_test_split(np.arange(len(train_patients)), train_size=55,
            random_state=seed+3000, stratify=train_patients.DMSA_APN)
        fit_alias = set(train_patients.patient_alias.iloc[fit_patient]); select_alias = set(train_patients.patient_alias.iloc[select_patient])
        inner_fit = np.asarray([i for i in outer_train if kidneys.patient_alias.iloc[i] in fit_alias])
        inner_select = np.asarray([i for i in outer_train if kidneys.patient_alias.iloc[i] in select_alias])
        inner_pre = R.make_preprocessor(seed).fit(values[inner_fit]); inner_values = inner_pre.transform(values)
        initial = fit_logistic(inner_values[inner_fit], labels[inner_fit], seed)
        inner_pairs = R.unilateral_pairs(inner_fit, kidneys, labels)
        selected_epochs = {}
        for arm in ARMS:
            _, history = refine(arm, initial, inner_values[inner_fit], labels[inner_fit], inner_pairs,
                                EPOCHS, inner_values[inner_select])
            choices = []
            for epoch, score in history.items():
                result = metric(labels[inner_select], score)
                choices.append((-(result["auroc"]+result["ap"])/2, epoch, result))
            _, selected, selected_result = min(choices, key=lambda item: (item[0], item[1]))
            selected_epochs[arm] = selected
            selections.append({"seed": seed, "arm": arm, "selected_epoch": selected,
                               "inner_auroc": selected_result["auroc"], "inner_ap": selected_result["ap"]})
        pre = R.make_preprocessor(seed).fit(values[outer_train]); transformed = pre.transform(values)
        initial = fit_logistic(transformed[outer_train], labels[outer_train], seed)
        pairs = R.unilateral_pairs(outer_train, kidneys, labels)
        joblib.dump({"preprocessor": pre, "initial_logistic": initial}, OUT / "models" / f"{seed}_preprocess_logistic.joblib", compress=3)
        scores = {}
        import torch
        for arm in ARMS:
            head, _ = refine(arm, initial, transformed[outer_train], labels[outer_train], pairs,
                             tuple(range(selected_epochs[arm]+1)), transformed[outer_dev])
            with torch.inference_mode():
                scores[arm] = torch.sigmoid(head(torch.from_numpy(transformed[outer_dev].astype(np.float32))).squeeze(1)).numpy()
            torch.save({"state_dict": head.state_dict(), "epoch": selected_epochs[arm], "arm": arm, "seed": seed},
                       OUT / "models" / f"{seed}_{arm}.pth")
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
        print(f"Completed logistic-initialized rank draw {seed}", flush=True)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    metric_frame = pd.DataFrame(metrics); metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "inner_epoch_selections.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap"]].mean().sort_values(["auroc", "ap"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc > .8) & (summary.ap > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {"groups_above_both_development_targets": achieved,
        "test_scored": False, "goal_achieved": False,
        "reason": "Development-only subgroup cannot complete the full objective"})
    print(summary.to_string(), flush=True)


def bootstrap() -> None:
    require_lock()
    frame = pd.read_csv(OUT / "development_predictions.csv")
    previous = pd.read_csv(D.OUT / "development_predictions.csv")
    reference = previous[previous.group.eq(REFERENCE)]
    ci, paired, samples = cluster_intervals(pd.concat([frame, reference], ignore_index=True),
                                            list(GROUPS)+[REFERENCE], F.SEEDS, REFERENCE)
    ci.to_csv(OUT / "development_confidence_intervals.csv", index=False)
    paired.to_csv(OUT / "development_paired_intervals.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    write_json(OUT / "completion_audit.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "patients":111, "kidneys":212, "images":418, "test_scored":False,
        "scored_unit":"kidney", "bootstrap_cluster":"patient"})
    print("Logistic-initialized rank intervals completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "bootstrap"])
    args = parser.parse_args()
    {"prepare":prepare, "run":run, "bootstrap":bootstrap}[args.stage]()
