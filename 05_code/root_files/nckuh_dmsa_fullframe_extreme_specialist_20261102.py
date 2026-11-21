"""Bilateral-APN versus bilateral-normal specialist for clean full-frame features."""
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
import nckuh_dmsa_fullframe_geometric_texture_20261102 as T
import nckuh_dmsa_fullframe_hierarchical_probe_20261102 as H
import nckuh_dmsa_fullframe_relative_geometry_probe_20261102 as G
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullframe_extreme_specialist_20261102"
REFERENCE = "deep_concat_plus_clean_radiomics"
POWERSET_REFERENCE = "four_state_powerset_plus_deep"
GROUPS = (
    "recomputed_powerset_plus_deep_reference",
    "powerset_extreme_image_handcrafted_replace_plus_deep",
    "powerset_extreme_image_deep_replace_plus_deep",
    "powerset_extreme_image_fused_replace_plus_deep",
    "powerset_extreme_image_fused_replace",
    "powerset_extreme_image_fused_halfblend_plus_deep",
    "powerset_extreme_patient_deep_replace_plus_deep",
)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered B/N-specialist protocol")
    F.require_lock(); D.require_lock(); G.require_lock(); H.require_lock(); T.require_lock()
    inputs = [
        F.OUT / "patient_assignments.csv", F.OUT / "deidentified_image_manifest.csv",
        F.OUT / "radiomics_clean_fullframe.csv",
        D.OUT / "radimagenet_gray.npy", D.OUT / "radimagenet_color.npy", D.OUT / "usf_gray.npy",
        D.OUT / "development_predictions.csv", G.OUT / "relative_geometry.csv",
        T.OUT / "image_geometric_texture.npy", H.OUT / "development_predictions.csv",
        *(H.OUT / "models" / f"{seed}_hierarchical.joblib" for seed in F.SEEDS),
        Path(__file__),
    ]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same clean-full-frame 111 patients, 212 kidneys, 418 images and exact 70/20/21 roles",
        "motivation": "Completed subgroup decomposition showed unilateral affected-side ranking was substantially stronger than bilateral-APN versus bilateral-normal ranking. A dedicated absolute-parenchymal specialist targets only the identified B/N deficit",
        "specialist_training": "Only training-role B and N patients fit the B/N specialist. Image instances retain patient grouping and receive weights that equalize total contribution per patient and per B/N class",
        "specialist_features": "One handcrafted arm uses clean PyRadiomics, stable geometric-region texture and relative full-frame geometry with fixed ANOVA64 LR. One frozen-encoder arm uses RadImageNet gray/color and USF-MAE embeddings with fixed PCA32 LR. Their fusion is an equal probability average. A patient-symmetric deep PCA16 control is also fixed",
        "integration": "The completed four-state powerset probabilities retain their total N+B mass and unchanged L/R probabilities. The specialist only reallocates the N/B conditional probability. Replacement and one fixed half-blend with the original N/B conditional are preregistered",
        "deep_control": "Where named, the revised powerset marginal is averaged equally with the unchanged completed deep-radiomics score",
        "prohibited": "No development-fit fusion weight, threshold, model selection, DMSA field as input, identifier, filename diagnosis suffix, or exploratory-test scoring",
        "target": "Development AUROC>0.8 and AP>0.8; independent locked confirmation remains necessary",
        "groups": list(GROUPS),
        "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "source_sha256": digest(__file__), "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs},
    })
    print("Registered bilateral-APN versus bilateral-normal specialist", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(Path(name)) == expected, name


def powerset_state_probabilities(model, preprocessor, raw_values: np.ndarray) -> np.ndarray:
    order = [list(model.classes_).index(index) for index in range(4)]
    direct = model.predict_proba(preprocessor.transform(raw_values))[:, order]
    swapped = model.predict_proba(preprocessor.transform(H.swap_pairs(raw_values)))[:, order][:, [0, 2, 1, 3]]
    output = (direct + swapped) / 2
    np.testing.assert_allclose(output.sum(1), 1, rtol=0, atol=1e-12)
    return output


def replace_extreme_probability(states: np.ndarray, bilateral_given_extreme: np.ndarray,
                                blend_original: bool = False) -> np.ndarray:
    output = np.asarray(states, dtype=float).copy()
    extreme = output[:, 0] + output[:, 3]
    original = np.divide(output[:, 3], extreme, out=np.full(len(output), .5), where=extreme > 0)
    revised = np.clip(bilateral_given_extreme, 0, 1)
    if blend_original:
        revised = (original + revised) / 2
    output[:, 0] = extreme * (1 - revised)
    output[:, 3] = extreme * revised
    np.testing.assert_allclose(output.sum(1), 1, rtol=0, atol=1e-12)
    return output


def kidney_marginals(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return states[:, 1] + states[:, 3], states[:, 2] + states[:, 3]


def image_training_weights(image_patient_indices: np.ndarray, patient_codes: np.ndarray) -> np.ndarray:
    unique = np.unique(image_patient_indices)
    class_by_patient = np.asarray([int(patient_codes[index] == 3) for index in unique])
    class_counts = np.bincount(class_by_patient, minlength=2)
    weights = np.zeros(len(image_patient_indices), dtype=float)
    for patient in unique:
        rows = np.flatnonzero(image_patient_indices == patient)
        target = int(patient_codes[patient] == 3)
        weights[rows] = len(unique) / (2 * class_counts[target] * len(rows))
    return weights


def image_specialist(values: np.ndarray, patients: pd.DataFrame, images: pd.DataFrame,
                     train: np.ndarray, codes: np.ndarray, seed: int, deep: bool):
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    patient_lookup = {alias: index for index, alias in enumerate(patients.patient_alias)}
    image_patients = images.patient_alias.map(patient_lookup).to_numpy(int)
    extreme_patients = train[np.isin(codes[train], [0, 3])]
    selected = np.flatnonzero(np.isin(image_patients, extreme_patients))
    target = (codes[image_patients[selected]] == 3).astype(int)
    weights = image_training_weights(image_patients[selected], codes)
    if deep:
        transform = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                  StandardScaler(), PCA(32, whiten=True, random_state=seed))
        regularization = .1
    else:
        transform = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                  StandardScaler(), VarianceThreshold(), SelectKBest(f_classif, k=64))
        regularization = .01
    training = transform.fit_transform(values[selected], target)
    model = LogisticRegression(C=regularization, class_weight=None, solver="liblinear",
                               max_iter=3000, random_state=seed)
    model.fit(training, target, sample_weight=weights)
    image_logits = model.decision_function(transform.transform(values))
    patient_logits = np.asarray([image_logits[images.patient_alias.eq(alias)].mean()
                                 for alias in patients.patient_alias])
    probability = 1 / (1 + np.exp(-np.clip(patient_logits, -30, 30)))
    audit = {"extreme_training_patients": int(len(extreme_patients)),
             "normal_training_patients": int((codes[extreme_patients] == 0).sum()),
             "bilateral_training_patients": int((codes[extreme_patients] == 3).sum()),
             "training_images": int(len(selected)),
             "patient_weight_min": float(weights.min()), "patient_weight_max": float(weights.max())}
    return probability, transform, model, audit


def patient_deep_specialist(deep_pairs: np.ndarray, train: np.ndarray, codes: np.ndarray, seed: int):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    half = deep_pairs.shape[1] // 2
    left, right = deep_pairs[:, :half], deep_pairs[:, half:]
    symmetric = np.c_[left + right, np.abs(left - right)]
    extreme = train[np.isin(codes[train], [0, 3])]
    target = (codes[extreme] == 3).astype(int)
    transform = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              StandardScaler(), PCA(16, whiten=True, random_state=seed))
    model = LogisticRegression(C=.1, class_weight="balanced", solver="liblinear",
                               max_iter=3000, random_state=seed)
    model.fit(transform.fit_transform(symmetric[extreme]), target)
    probability = model.predict_proba(transform.transform(symmetric))[:, 1]
    return probability, transform, model


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed B/N specialist predictions")
    patients, kidneys, images = F.matched_reference()
    labels = kidneys.kidney_APN.to_numpy(int)
    codes = patients.DMSA_APN.map(H.CODE).to_numpy(int)
    deep_images = np.concatenate([np.load(D.OUT / name) for name in
                                  ("radimagenet_gray.npy", "radimagenet_color.npy", "usf_gray.npy")], axis=1)
    radiomics_frame = pd.read_csv(F.OUT / "radiomics_clean_fullframe.csv").set_index("image_id").reindex(images.image_id)
    radiomics = radiomics_frame[[column for column in radiomics_frame
                                 if not column.endswith(("_Energy", "_TotalEnergy"))]].to_numpy(float)
    texture = np.load(T.OUT / "image_geometric_texture.npy")
    geometry_frame = pd.read_csv(G.OUT / "relative_geometry.csv").set_index("image_id").reindex(images.image_id)
    geometry = geometry_frame[[column for column in geometry_frame if column.startswith("geometry_")]].to_numpy(float)
    handcrafted_images = np.c_[radiomics, texture, geometry]
    deep_pairs = F.patient_pair_features(deep_images, images, patients)
    assignments = pd.read_csv(F.OUT / "patient_assignments.csv")
    deep_predictions = pd.read_csv(D.OUT / "development_predictions.csv")
    deep_reference = deep_predictions[deep_predictions.group.eq(REFERENCE)].set_index(["seed", "kidney_alias"]).score
    completed_powerset = pd.read_csv(H.OUT / "development_predictions.csv")
    completed_powerset = completed_powerset[completed_powerset.group.eq(POWERSET_REFERENCE)].set_index(["seed", "kidney_alias"]).score
    patient_lookup = {alias: index for index, alias in enumerate(patients.patient_alias)}

    records, predictions, specialist_records, audits = [], [], [], []
    (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in F.SEEDS:
            mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
            train = np.flatnonzero(patients.patient_alias.map(mapping).eq("train"))
            development_patients = np.flatnonzero(patients.patient_alias.map(mapping).eq("development"))
            hand_q, hand_transform, hand_model, hand_audit = image_specialist(
                handcrafted_images, patients, images, train, codes, seed, False)
            deep_q, deep_transform, deep_model, deep_audit = image_specialist(
                deep_images, patients, images, train, codes, seed, True)
            fused_q = (hand_q + deep_q) / 2
            patient_q, patient_transform, patient_model = patient_deep_specialist(deep_pairs, train, codes, seed)
            saved = joblib.load(H.OUT / "models" / f"{seed}_hierarchical.joblib")
            states = powerset_state_probabilities(saved["powerset_model"], saved["powerset_pre"], deep_pairs)
            revised = {
                "hand": replace_extreme_probability(states, hand_q),
                "deep": replace_extreme_probability(states, deep_q),
                "fused": replace_extreme_probability(states, fused_q),
                "fused_blend": replace_extreme_probability(states, fused_q, True),
                "patient": replace_extreme_probability(states, patient_q),
            }
            marginal = {name: kidney_marginals(value) for name, value in revised.items()}
            baseline_marginal = kidney_marginals(states)
            development = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
            aliases = kidneys.kidney_alias.iloc[development]
            deep_score = np.asarray([deep_reference.loc[(seed, alias)] for alias in aliases])
            completed_score = np.asarray([completed_powerset.loc[(seed, alias)] for alias in aliases])

            def kidney_values(pair):
                left, right = pair
                return np.asarray([left[patient_lookup[kidneys.patient_alias.iloc[index]]]
                                   if kidneys.side.iloc[index] == "left"
                                   else right[patient_lookup[kidneys.patient_alias.iloc[index]]]
                                   for index in development])

            baseline = (kidney_values(baseline_marginal) + deep_score) / 2
            # Existing predictions are round-tripped through CSV, so allow only
            # the observed float32-scale serialization error.
            np.testing.assert_allclose(baseline, completed_score, rtol=0, atol=1e-7)
            hand_score = kidney_values(marginal["hand"])
            specialist_deep_score = kidney_values(marginal["deep"])
            fused_score = kidney_values(marginal["fused"])
            patient_score = kidney_values(marginal["patient"])
            scores = {
                GROUPS[0]: baseline,
                GROUPS[1]: (hand_score + deep_score) / 2,
                GROUPS[2]: (specialist_deep_score + deep_score) / 2,
                GROUPS[3]: (fused_score + deep_score) / 2,
                GROUPS[4]: fused_score,
                GROUPS[5]: (kidney_values(marginal["fused_blend"]) + deep_score) / 2,
                GROUPS[6]: (patient_score + deep_score) / 2,
            }
            joblib.dump({"hand_transform": hand_transform, "hand_model": hand_model,
                         "deep_transform": deep_transform, "deep_model": deep_model,
                         "patient_transform": patient_transform, "patient_model": patient_model},
                        OUT / "models" / f"{seed}_extreme_specialist.joblib", compress=3)
            audits.append({"seed": seed, "hand": hand_audit, "deep": deep_audit,
                           "baseline_reconstruction_max_abs_difference": float(np.max(np.abs(baseline - completed_score)))})
            extreme_development = development_patients[np.isin(codes[development_patients], [0, 3])]
            target = (codes[extreme_development] == 3).astype(int)
            for name, probability in (("image_handcrafted", hand_q), ("image_deep", deep_q),
                                      ("image_fused", fused_q), ("patient_deep", patient_q)):
                result = metric(target, probability[extreme_development])
                specialist_records.append({"seed": seed, "group": name,
                    "patients": len(extreme_development), **result})
            for group in GROUPS:
                result = metric(labels[development], scores[group])
                records.append({"seed": seed, "role": "development", "group": group,
                                "patients": 20, "kidneys": len(development), **result})
                for local, row_index in enumerate(development):
                    predictions.append({"seed": seed, "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[row_index],
                        "kidney_alias": kidneys.kidney_alias.iloc[row_index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[row_index],
                        "side": kidneys.side.iloc[row_index], "label": int(labels[row_index]),
                        "score": float(scores[group][local])})
            print(f"Completed B/N specialist draw {seed}", flush=True)

    metric_frame = pd.DataFrame(records)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(specialist_records).to_csv(OUT / "specialist_bn_metrics.csv", index=False)
    write_json(OUT / "training_audit.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "draws": audits, "test_scored": False})
    summary = metric_frame.groupby("group")[["auroc", "ap"]].mean().sort_values(["auroc", "ap"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    specialist_summary = pd.DataFrame(specialist_records).groupby("group")[["auroc", "ap"]].mean()
    specialist_summary.to_csv(OUT / "specialist_bn_summary.csv")
    achieved = summary[(summary.auroc > .8) & (summary.ap > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {"groups_above_both_development_targets": achieved,
        "test_scored": False, "goal_achieved": False,
        "reason": "Development-only availability subgroup cannot complete the full objective"})
    print("Kidney-level summary\n" + summary.to_string(), flush=True)
    print("B/N specialist summary\n" + specialist_summary.to_string(), flush=True)


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
        "patients": 111, "kidneys": 212, "images": 418, "test_scored": False,
        "scored_unit": "kidney", "bootstrap_cluster": "patient"})
    print("B/N specialist intervals completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "bootstrap"])
    arguments = parser.parse_args()
    globals()[arguments.stage]()
