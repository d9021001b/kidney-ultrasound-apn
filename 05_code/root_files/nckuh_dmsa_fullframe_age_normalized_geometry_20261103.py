"""Training-only age-normalized renal geometry for pediatric NCKUH ultrasound."""
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
import nckuh_dmsa_fullframe_relative_echo_probe_20261103 as R
import nckuh_dmsa_fullframe_relative_geometry_probe_20261102 as G
import nckuh_dmsa_fullframe_tick_geometry_probe_20261102 as K
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullframe_age_normalized_geometry_20261103"
DEMOGRAPHICS = ROOT / "02_experiments/NCKUH_DMSA_fullframe_demographic_hierarchy_20261102/deidentified_demographics.csv"
REFERENCE = "deep_concat_plus_clean_radiomics"
POWERSET_REFERENCE = "four_state_powerset_plus_deep"
GEOMETRY_REFERENCE = "relative_frame_geometry_plus_deep_radiomics"
ECHO_REFERENCE = "relative_context_echo_extreme_replace_plus_deep"
GROUPS = (
    "age_normalized_geometry_anova32_lr",
    "raw_plus_age_normalized_geometry_anova64_lr",
    "age_normalized_geometry_plus_deep_radiomics",
    "age_normalized_geometry_extreme_replace_plus_deep",
    "age_normalized_geometry_extreme_halfblend_plus_deep",
    "age_normalized_extreme_plus_relative_echo",
    "age_normalized_extreme_echo_geometry_three_source_fusion",
)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered age-normalized geometry protocol")
    F.require_lock(); D.require_lock(); E.require_lock(); G.require_lock(); H.require_lock(); K.require_lock(); R.require_lock()
    inputs = [F.OUT / "patient_assignments.csv", F.OUT / "deidentified_image_manifest.csv",
              DEMOGRAPHICS, K.OUT / "tick_geometry.csv", G.OUT / "relative_geometry.csv",
              D.OUT / "development_predictions.csv", G.OUT / "development_predictions.csv",
              H.OUT / "development_predictions.csv", R.OUT / "development_predictions.csv",
              *(H.OUT / "models" / f"{seed}_hierarchical.joblib" for seed in F.SEEDS), Path(__file__)]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same clean-full-frame 111 patients, 212 kidneys, 418 images and exact 70/20/21 roles; age available for 110 patients and sex for 111",
        "motivation": "Renal dimensions in children vary with age, body size, side and view, while APN can enlarge renal volume. Raw size can therefore hide disease signal behind normal growth variation",
        "references": [
            "https://pubmed.ncbi.nlm.nih.gov/21459422/",
            "https://pubmed.ncbi.nlm.nih.gov/12394745/",
            "https://pubmed.ncbi.nlm.nih.gov/20093617/",
            "https://pubmed.ncbi.nlm.nih.gov/3513490/",
        ],
        "normalizer": "For each draw, fit a multi-output ridge model only on training-role images. Predict 28 tick-normalized/relative geometry values from log-age, age, age squared, sex, kidney side and longitudinal/transverse view; use robustly scaled residuals as label-free age/context-normalized geometry",
        "missing": "Training median imputes one missing age and four missing tick detections. Missingness is not converted into an APN label",
        "scope": "Height and weight are unavailable, so this is an internal training-only residualization control, not a complete pediatric renal nomogram or physical-centimeter claim",
        "models": "View-preserving own/opposite kidney residual geometry; fixed ANOVA32/64 balanced LR C=.01. A B/N-only residual-geometry specialist may only reallocate powerset N+B probability mass",
        "fusion": "Only preregistered equal-probability averages with unchanged deep-radiomics, relative-echo and relative-geometry completed branches",
        "prohibited": "No APN label in geometry normalization, no development/test fit, no direct date/identifier/DMSA input, no development-selected weight/threshold, no exploratory-test scoring",
        "target": "Development AUROC>0.8 and AP>0.8; later locked confirmation remains necessary",
        "groups": list(GROUPS), "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {"source_sha256": digest(__file__),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs}})
    print("Registered training-only age-normalized renal geometry", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(Path(name)) == expected, name


def image_covariates(images: pd.DataFrame, demographics: pd.DataFrame) -> np.ndarray:
    table = images[["patient_alias", "side", "view"]].merge(demographics, on="patient_alias", how="left")
    age = table.age_years.to_numpy(float)
    return np.c_[np.log1p(age), age, age * age, table.female.to_numpy(float),
                 table.side.eq("right").to_numpy(float), table["view"].str.endswith("T").to_numpy(float)]


def residualize_geometry(values: np.ndarray, covariates: np.ndarray,
                         training_images: np.ndarray, seed: int):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    geometry_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    geometry_imputer.fit(values[training_images])
    completed = geometry_imputer.transform(values)
    normalizer = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                               StandardScaler(), Ridge(alpha=.01, random_state=seed))
    normalizer.fit(covariates[training_images], completed[training_images])
    residual = completed - normalizer.predict(covariates)
    center = np.median(residual[training_images], axis=0)
    mad = np.median(np.abs(residual[training_images] - center), axis=0) / .6745
    mad[mad < 1e-6] = 1.0
    normalized = (residual - center) / mad
    return normalized.astype(np.float32), {
        "geometry_imputer": geometry_imputer, "normalizer": normalizer,
        "residual_center": center, "residual_scale": mad,
    }


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
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          VarianceThreshold(), SelectKBest(f_classif, k=32),
                          LogisticRegression(C=.01, class_weight="balanced", solver="liblinear",
                                             max_iter=3000, random_state=seed))
    model.fit(symmetric[extreme], target)
    return model.predict_proba(symmetric)[:, 1], model


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed age-normalized predictions")
    patients, kidneys, images = F.matched_reference(); labels = kidneys.kidney_APN.to_numpy(int)
    states = patients.DMSA_APN.map(H.CODE).to_numpy(int)
    demographics = pd.read_csv(DEMOGRAPHICS)
    covariates = image_covariates(images, demographics)
    tick_frame = pd.read_csv(K.OUT / "tick_geometry.csv").set_index("image_id").reindex(images.image_id)
    tick = tick_frame[[f"tick_{index:02d}" for index in range(7)]].to_numpy(float)
    relative_frame = pd.read_csv(G.OUT / "relative_geometry.csv").set_index("image_id").reindex(images.image_id)
    relative = relative_frame[[column for column in relative_frame if column.startswith("geometry_")]].to_numpy(float)
    raw_geometry = np.c_[tick, relative]
    assignments = pd.read_csv(F.OUT / "patient_assignments.csv")
    deep_predictions = pd.read_csv(D.OUT / "development_predictions.csv")
    deep_reference = deep_predictions[deep_predictions.group.eq(REFERENCE)].set_index(["seed", "kidney_alias"]).score
    power_predictions = pd.read_csv(H.OUT / "development_predictions.csv")
    power_reference = power_predictions[power_predictions.group.eq(POWERSET_REFERENCE)].set_index(["seed", "kidney_alias"]).score
    geometry_predictions = pd.read_csv(G.OUT / "development_predictions.csv")
    geometry_reference = geometry_predictions[geometry_predictions.group.eq(GEOMETRY_REFERENCE)].set_index(["seed", "kidney_alias"]).score
    echo_predictions = pd.read_csv(R.OUT / "development_predictions.csv")
    echo_reference = echo_predictions[echo_predictions.group.eq(ECHO_REFERENCE)].set_index(["seed", "kidney_alias"]).score
    deep_images = np.concatenate([np.load(D.OUT / name) for name in
                                  ("radimagenet_gray.npy", "radimagenet_color.npy", "usf_gray.npy")], axis=1)
    deep_pairs = F.patient_pair_features(deep_images, images, patients)
    patient_lookup = {alias: index for index, alias in enumerate(patients.patient_alias)}

    records, predictions, audits = [], [], []
    (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in F.SEEDS:
            mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
            train_patients = np.flatnonzero(patients.patient_alias.map(mapping).eq("train"))
            train_images = np.flatnonzero(images.patient_alias.map(mapping).eq("train"))
            residual_images, normalizer = residualize_geometry(raw_geometry, covariates, train_images, seed)
            residual_pairs = F.patient_pair_features(residual_images, images, patients)
            raw_pairs = F.patient_pair_features(raw_geometry, images, patients)
            oriented_residual = F.orient_kidneys(residual_pairs, patients, kidneys)
            oriented_raw = F.orient_kidneys(raw_pairs, patients, kidneys)
            raw_residual = np.c_[oriented_raw, oriented_residual]
            roles = kidneys.patient_alias.map(mapping)
            train, development = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
            aliases = kidneys.kidney_alias.iloc[development]
            deep_score = np.asarray([deep_reference.loc[(seed, alias)] for alias in aliases])
            power_score = np.asarray([power_reference.loc[(seed, alias)] for alias in aliases])
            geometry_score = np.asarray([geometry_reference.loc[(seed, alias)] for alias in aliases])
            echo_score = np.asarray([echo_reference.loc[(seed, alias)] for alias in aliases])
            residual_score, residual_transform, residual_model = F.fit_predict(
                oriented_residual, train, development, labels, seed, 32)
            combined_score, combined_transform, combined_model = F.fit_predict(
                raw_residual, train, development, labels, seed, 64)
            extreme_probability, extreme_model = fit_extreme(residual_pairs, train_patients, states, seed)
            saved = joblib.load(H.OUT / "models" / f"{seed}_hierarchical.joblib")
            power_states = E.powerset_state_probabilities(saved["powerset_model"], saved["powerset_pre"], deep_pairs)
            replaced = E.replace_extreme_probability(power_states, extreme_probability)
            halfblend = E.replace_extreme_probability(power_states, extreme_probability, True)

            def kidney_values(state_values):
                left, right = E.kidney_marginals(state_values)
                return np.asarray([left[patient_lookup[kidneys.patient_alias.iloc[index]]]
                                   if kidneys.side.iloc[index] == "left"
                                   else right[patient_lookup[kidneys.patient_alias.iloc[index]]]
                                   for index in development])

            replaced_score = kidney_values(replaced); halfblend_score = kidney_values(halfblend)
            age_extreme_score = (replaced_score + deep_score) / 2
            scores = {GROUPS[0]: residual_score, GROUPS[1]: combined_score,
                      GROUPS[2]: (residual_score + deep_score) / 2,
                      GROUPS[3]: age_extreme_score,
                      GROUPS[4]: (halfblend_score + deep_score) / 2,
                      GROUPS[5]: (age_extreme_score + echo_score) / 2,
                      GROUPS[6]: (age_extreme_score + echo_score + geometry_score) / 3}
            joblib.dump({"normalizer": normalizer, "residual": (residual_transform, residual_model),
                         "raw_residual": (combined_transform, combined_model), "extreme": extreme_model},
                        OUT / "models" / f"{seed}_age_normalized_geometry.joblib", compress=3)
            audits.append({"seed": seed, "normalizer_training_patients": 70,
                           "normalizer_training_images": len(train_images),
                           "labels_used_in_normalizer": [], "test_scored": False,
                           "unused_power_reference_mean": float(power_score.mean())})
            for group in GROUPS:
                result = metric(labels[development], scores[group])
                records.append({"seed": seed, "role": "development", "group": group,
                                "patients": 20, "kidneys": len(development), **result})
                for local, row_index in enumerate(development):
                    predictions.append({"seed": seed, "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[row_index],
                        "kidney_alias": kidneys.kidney_alias.iloc[row_index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[row_index], "side": kidneys.side.iloc[row_index],
                        "label": int(labels[row_index]), "score": float(scores[group][local])})
            print(f"Completed age-normalized geometry draw {seed}", flush=True)
    frame = pd.DataFrame(records); frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    write_json(OUT / "normalizer_audit.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "draws": audits, "test_scored": False})
    summary = frame.groupby("group")[["auroc", "ap"]].mean().sort_values(["auroc", "ap"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc > .8) & (summary.ap > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {"groups_above_both_development_targets": achieved,
        "test_scored": False, "goal_achieved": False,
        "reason": "Development-only subgroup cannot complete the full objective"})
    print(summary.to_string(), flush=True)


def bootstrap() -> None:
    require_lock(); frame = pd.read_csv(OUT / "development_predictions.csv")
    previous = pd.read_csv(D.OUT / "development_predictions.csv"); reference = previous[previous.group.eq(REFERENCE)]
    ci, paired, samples = cluster_intervals(pd.concat([frame, reference], ignore_index=True),
        list(GROUPS) + [REFERENCE], F.SEEDS, REFERENCE)
    ci.to_csv(OUT / "development_confidence_intervals.csv", index=False)
    paired.to_csv(OUT / "development_paired_intervals.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    write_json(OUT / "completion_audit.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "patients": 111, "kidneys": 212, "images": 418, "test_scored": False,
        "scored_unit": "kidney", "bootstrap_cluster": "patient"})
    print("Age-normalized geometry intervals completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "bootstrap"])
    arguments = parser.parse_args(); globals()[arguments.stage]()
