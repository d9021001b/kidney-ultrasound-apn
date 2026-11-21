"""Expand training with labeled NCKUH contour ROIs while preserving clean roles."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullframe_clean_probe_20261102 as F
import nckuh_dmsa_fullframe_deep_probe_20261102 as D
import nckuh_dmsa_fullframe_extreme_specialist_20261102 as E
import nckuh_dmsa_fullframe_hierarchical_probe_20261102 as H
import nckuh_dmsa_kidney_baseline_20261102 as B
import nckuh_dmsa_label_powerset_20261102 as P
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullframe_expanded_training_20261103"
REFERENCE = "deep_concat_plus_clean_radiomics"
CLEAN_POWERSET_REFERENCE = "four_state_powerset_plus_deep"
GROUPS = (
    "expanded_handcrafted_powerset",
    "expanded_usf_powerset",
    "expanded_handcrafted_extreme_replace",
    "expanded_handcrafted_usf_extreme_replace",
    "expanded_handcrafted_powerset_plus_clean_reference",
    "expanded_extreme_replace_plus_clean_reference",
)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered expanded-training protocol")
    F.require_lock(); D.require_lock(); H.require_lock(); P.require_lock(); B.verify_lock()
    inputs = [
        F.OUT / "patient_assignments.csv", F.OUT / "deidentified_image_manifest.csv",
        D.OUT / "development_predictions.csv", H.OUT / "development_predictions.csv",
        P.OUT / "handcrafted_pairs.npy", P.OUT / "usf_class_pairs.npy",
        B.AUDIT / "patient_label_linkage_private.csv", B.AUDIT / "kidney_label_linkage_private.csv",
        B.AUDIT / "image_labels_dmsa_private.csv", Path(__file__),
    ]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Full corrected source-code-available NCKUH cohort has 210 patients; evaluation remains the same clean availability subset of 111 patients, 212 kidneys and 418 views",
        "training_expansion": "For each fixed clean draw, exclude all 20 development and 21 exploratory-test patients, then train on the remaining 169 labeled patients: the original 70 clean training patients plus 99 labeled patients without clean full-frame availability",
        "training_distribution": "Each expanded role contains B75, N30, L39 and R25 patients; the clean development role remains B8, N3, L6 and R3",
        "representations": "Previously extracted segmentation-first contour-ROI handcrafted pair (PyRadiomics plus directional descriptors) and frozen USF-MAE class-token pair. Expanded models and their clean-development predictions use this same legacy ROI domain; clean full-frame scores enter only named equal-weight fusions",
        "joint_model": "Four-state N/L/R/B balanced LR C=.01 with training-only imputation, scaling and variance filtering; left-right swap augmentation and symmetrized inference",
        "extreme_specialist": "B/N-only patient-level symmetric sum/absolute-difference features, training-only ANOVA64 and balanced LR C=.01. It only reallocates the powerset model's fixed N+B probability mass",
        "fusion": "Only fixed equal-probability averages with the completed clean powerset-deep reference; no development-selected coefficient",
        "segmentation": "All legacy images first use the established whole-kidney contour-assisted ROI pipeline. Legacy caliper overlays remain a limitation and are never called disease features",
        "prohibited": "No clean development or exploratory-test patient enters fitting; no test score, DMSA field as predictor, identifier, date, diagnosis suffix, development-selected weight or threshold",
        "target": "Development AUROC>0.8 and AP>0.8; later locked confirmation remains required",
        "groups": list(GROUPS), "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {"source_sha256": digest(__file__),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs}})
    print("Registered expanded contour-ROI training with clean roles preserved", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(Path(name)) == expected, name


def expanded_train_indices(full_patients: pd.DataFrame, assignments: pd.DataFrame, seed: int) -> np.ndarray:
    mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
    held_out = set(mapping[mapping.isin(["development", "exploratory_test"])].index)
    selected = np.flatnonzero(~full_patients.patient_alias.isin(held_out))
    assert len(selected) == 169
    assert not set(full_patients.patient_alias.iloc[selected]).intersection(held_out)
    return selected


def fit_powerset(values: np.ndarray, train: np.ndarray, states: np.ndarray, seed: int):
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(), VarianceThreshold(),
                          LogisticRegression(C=.01, class_weight="balanced", solver="lbfgs",
                                             max_iter=3000, random_state=seed))
    augmented = np.r_[values[train], H.swap_pairs(values[train])]
    target = np.r_[states[train], P.SWAP[states[train]]]
    model.fit(augmented, target)
    order = [list(model.classes_).index(index) for index in range(4)]
    direct = model.predict_proba(values)[:, order]
    swapped = model.predict_proba(H.swap_pairs(values))[:, order][:, P.SWAP]
    probability = (direct + swapped) / 2
    np.testing.assert_allclose(probability.sum(1), 1, rtol=0, atol=1e-12)
    return probability, model


def fit_extreme(values: np.ndarray, train: np.ndarray, states: np.ndarray, seed: int):
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    half = values.shape[1] // 2
    left, right = values[:, :half], values[:, half:]
    symmetric = np.c_[left + right, np.abs(left - right)]
    extreme = train[np.isin(states[train], [0, 3])]
    target = (states[extreme] == 3).astype(int)
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(), VarianceThreshold(), SelectKBest(f_classif, k=64),
                          LogisticRegression(C=.01, class_weight="balanced", solver="liblinear",
                                             max_iter=3000, random_state=seed))
    model.fit(symmetric[extreme], target)
    probability = model.predict_proba(symmetric)[:, list(model.classes_).index(1)]
    return probability, model


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed expanded-training predictions")
    full_patients, _, _ = B.load_reference()
    clean_patients, clean_kidneys, _ = F.matched_reference()
    labels = clean_kidneys.kidney_APN.to_numpy(int)
    states = full_patients.DMSA_APN.map(H.CODE).to_numpy(int)
    handcrafted = np.load(P.OUT / "handcrafted_pairs.npy")
    usf = np.load(P.OUT / "usf_class_pairs.npy")
    assert handcrafted.shape == (210, 1930) and usf.shape == (210, 3074)
    assignments = pd.read_csv(F.OUT / "patient_assignments.csv")
    deep_predictions = pd.read_csv(D.OUT / "development_predictions.csv")
    deep_reference = deep_predictions[deep_predictions.group.eq(REFERENCE)].set_index(["seed", "kidney_alias"]).score
    clean_predictions = pd.read_csv(H.OUT / "development_predictions.csv")
    clean_reference = clean_predictions[clean_predictions.group.eq(CLEAN_POWERSET_REFERENCE)].set_index(["seed", "kidney_alias"]).score
    full_lookup = {alias: index for index, alias in enumerate(full_patients.patient_alias)}
    clean_patient_full_index = np.asarray([full_lookup[alias] for alias in clean_patients.patient_alias])
    clean_patient_lookup = {alias: index for index, alias in enumerate(clean_patients.patient_alias)}

    records, predictions, audits = [], [], []
    (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in F.SEEDS:
            train = expanded_train_indices(full_patients, assignments, seed)
            distribution = full_patients.DMSA_APN.iloc[train].value_counts().sort_index().to_dict()
            assert distribution == {"B": 75, "L": 39, "N": 30, "R": 25}
            hand_states, hand_model = fit_powerset(handcrafted, train, states, seed)
            usf_states, usf_model = fit_powerset(usf, train, states, seed)
            hand_extreme, hand_extreme_model = fit_extreme(handcrafted, train, states, seed)
            usf_extreme, usf_extreme_model = fit_extreme(usf, train, states, seed)
            fused_extreme = (hand_extreme + usf_extreme) / 2
            hand_revised = E.replace_extreme_probability(hand_states, hand_extreme)
            fused_revised = E.replace_extreme_probability(hand_states, fused_extreme)
            marginal = {
                "hand": E.kidney_marginals(hand_states), "usf": E.kidney_marginals(usf_states),
                "hand_revised": E.kidney_marginals(hand_revised),
                "fused_revised": E.kidney_marginals(fused_revised),
            }
            role_mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
            development = np.flatnonzero(clean_kidneys.patient_alias.map(role_mapping).eq("development"))
            aliases = clean_kidneys.kidney_alias.iloc[development]
            clean_score = np.asarray([clean_reference.loc[(seed, alias)] for alias in aliases])
            deep_score = np.asarray([deep_reference.loc[(seed, alias)] for alias in aliases])

            def kidney_values(pair):
                left, right = pair
                output = []
                for index in development:
                    clean_index = clean_patient_lookup[clean_kidneys.patient_alias.iloc[index]]
                    full_index = clean_patient_full_index[clean_index]
                    output.append(left[full_index] if clean_kidneys.side.iloc[index] == "left" else right[full_index])
                return np.asarray(output)

            hand_score = kidney_values(marginal["hand"])
            usf_score = kidney_values(marginal["usf"])
            hand_revised_score = kidney_values(marginal["hand_revised"])
            fused_revised_score = kidney_values(marginal["fused_revised"])
            scores = {
                GROUPS[0]: hand_score,
                GROUPS[1]: usf_score,
                GROUPS[2]: hand_revised_score,
                GROUPS[3]: fused_revised_score,
                GROUPS[4]: (hand_score + clean_score) / 2,
                GROUPS[5]: (fused_revised_score + clean_score) / 2,
            }
            joblib.dump({"hand_powerset": hand_model, "usf_powerset": usf_model,
                         "hand_extreme": hand_extreme_model, "usf_extreme": usf_extreme_model},
                        OUT / "models" / f"{seed}_expanded_training.joblib", compress=3)
            audits.append({"seed": seed, "expanded_training_patients": len(train),
                           "distribution": distribution, "clean_development_patients": 20,
                           "clean_exploratory_test_scored": False,
                           "unused_deep_reference_mean": float(deep_score.mean())})
            for group in GROUPS:
                result = metric(labels[development], scores[group])
                records.append({"seed": seed, "role": "development", "group": group,
                                "patients": 20, "kidneys": len(development), **result})
                for local, row_index in enumerate(development):
                    predictions.append({"seed": seed, "group": group,
                        "patient_alias": clean_kidneys.patient_alias.iloc[row_index],
                        "kidney_alias": clean_kidneys.kidney_alias.iloc[row_index],
                        "DMSA_APN": clean_kidneys.DMSA_APN.iloc[row_index],
                        "side": clean_kidneys.side.iloc[row_index], "label": int(labels[row_index]),
                        "score": float(scores[group][local])})
            print(f"Completed expanded-training draw {seed}", flush=True)

    metric_frame = pd.DataFrame(records)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    write_json(OUT / "training_audit.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "draws": audits, "test_scored": False})
    summary = metric_frame.groupby("group")[["auroc", "ap"]].mean().sort_values(["auroc", "ap"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc > .8) & (summary.ap > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {"groups_above_both_development_targets": achieved,
        "test_scored": False, "goal_achieved": False,
        "reason": "Development-only clean subset cannot complete the full objective"})
    print(summary.to_string(), flush=True)


def bootstrap() -> None:
    require_lock()
    frame = pd.read_csv(OUT / "development_predictions.csv")
    previous = pd.read_csv(D.OUT / "development_predictions.csv")
    reference = previous[previous.group.eq(REFERENCE)]
    ci, paired, samples = cluster_intervals(
        pd.concat([frame, reference], ignore_index=True), list(GROUPS) + [REFERENCE], F.SEEDS, REFERENCE)
    ci.to_csv(OUT / "development_confidence_intervals.csv", index=False)
    paired.to_csv(OUT / "development_paired_intervals.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    write_json(OUT / "completion_audit.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "full_labeled_patients": 210, "expanded_training_patients_per_draw": 169,
        "clean_development_patients_per_draw": 20, "clean_subset_kidneys": 212,
        "test_scored": False, "scored_unit": "kidney", "bootstrap_cluster": "patient"})
    print("Expanded-training intervals completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "bootstrap"])
    arguments = parser.parse_args()
    globals()[arguments.stage]()
