"""Two-stage bilateral/normal hierarchy with imaging geometry and demographics."""
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
import nckuh_dmsa_fullframe_hierarchical_probe_20261102 as H
import nckuh_dmsa_fullframe_relative_geometry_probe_20261102 as G
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullframe_demographic_hierarchy_20261102"
PRIVATE = ROOT / "02_experiments/NCKUH_DMSA_label_audit_20261102/patient_label_linkage_private.csv"
REFERENCE = "deep_concat_plus_clean_radiomics"
POWERSET = "four_state_powerset_plus_deep"
GROUPS = (
    "two_stage_geometry_count_deep_laterality",
    "two_stage_geometry_demographic_count_deep_laterality",
    "two_stage_demographic_hierarchy_plus_deep_reference",
    "two_stage_demographic_hierarchy_plus_powerset",
)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists(): raise FileExistsError("Preserve registered demographic hierarchy")
    F.require_lock(); D.require_lock(); H.require_lock(); G.require_lock()
    inputs = [PRIVATE, F.OUT / "patient_assignments.csv", F.OUT / "deidentified_image_manifest.csv",
              G.OUT / "relative_geometry.csv", D.OUT / "radimagenet_gray.npy",
              D.OUT / "radimagenet_color.npy", D.OUT / "usf_gray.npy",
              D.OUT / "development_predictions.csv", H.OUT / "development_predictions.csv", Path(__file__)]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same clean-full-frame 111 patients, 212 kidneys, 418 images and exact 70/20/21 patient roles",
        "motivation": "Error decomposition showed good unilateral affected-side ranking but weak bilateral-APN versus bilateral-normal discrimination",
        "hierarchy": "Stage1 predicts unilateral versus non-unilateral from symmetric relative-geometry features. Stage2, fitted only to N/B training patients, predicts bilateral APN versus normal; the demographic arm adds age at image exam and sex only to Stage2",
        "laterality": "For predicted unilateral mass, a separate L-versus-R head uses signed frozen deep-feature differences and is fitted only to unilateral training patients",
        "marginals": "P(left APN)=P(B)+P(unilateral)*P(left|unilateral), with the corresponding right formula",
        "demographics": "Age in years from audited birth and image-exam dates; sex mapping1/M and2/F follows the audited cross-sheet concordance. One missing age is train-only median imputed",
        "excluded": "DMSA_APN is target only. DMSA_RF, DMSA_LF, DMSA_YN, admission timing, identifiers and every other clinical field are excluded",
        "models": "Train-only median imputation, scaling, ANOVA32 and balanced LR C0.1 for count stages; PCA16 LR for laterality; two fixed equal-probability fusion controls",
        "prohibited": "No development-selected covariate, threshold or weight and no exploratory-test scoring",
        "target": "Development AUROC>0.8 and AP>0.8; later locked confirmation remains necessary",
        "groups": list(GROUPS), "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {"source_sha256": digest(__file__),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs}})
    print("Registered two-stage demographic/geometry hierarchy", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items(): assert digest(Path(name)) == expected, name


def extract() -> None:
    require_lock()
    if (OUT / "deidentified_demographics.csv").exists(): raise FileExistsError("Preserve demographics")
    patients, _, _ = F.matched_reference()
    private = pd.read_csv(PRIVATE, dtype={"patient_id": str})
    linked = patients[["patient_id", "patient_alias"]].merge(
        private[["patient_id", "birth_date", "image_exam_date", "gender_source_values"]], on="patient_id", how="left")
    birth = pd.to_datetime(linked.birth_date, errors="coerce")
    exam = pd.to_datetime(linked.image_exam_date, errors="coerce")
    age = (exam - birth).dt.days / 365.2425
    female = linked.gender_source_values.astype(str).str.contains(r"2|F", regex=True).astype(float)
    output = pd.DataFrame({"patient_alias": linked.patient_alias, "age_years": age, "female": female})
    assert len(output) == 111 and output.female.notna().all() and output.age_years.notna().sum() == 110
    output.to_csv(OUT / "deidentified_demographics.csv", index=False)
    write_json(OUT / "demographic_audit.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "patients": 111, "age_available": 110, "sex_available": 111,
        "age_min_median_max": [float(age.min()), float(age.median()), float(age.max())],
        "included_fields": ["age_at_image_exam", "sex"],
        "excluded_fields": ["DMSA_RF", "DMSA_LF", "DMSA_YN", "admission_date", "identifiers"],
        "contains_identifier": False, "sha256": digest(OUT / "deidentified_demographics.csv")})
    print("Deidentified age/sex covariates prepared for 111 patients", flush=True)


def binary_model(seed: int):
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         StandardScaler(), VarianceThreshold(), SelectKBest(f_classif, k=32),
                         LogisticRegression(C=.1, class_weight="balanced", solver="liblinear",
                                            max_iter=3000, random_state=seed))


def compose_count_marginals(unilateral_probability: np.ndarray, bilateral_given_extreme: np.ndarray,
                            left_given_unilateral: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p1 = np.clip(unilateral_probability, 0, 1)
    p2 = (1 - p1) * np.clip(bilateral_given_extreme, 0, 1)
    left = p2 + p1 * np.clip(left_given_unilateral, 0, 1)
    right = p2 + p1 * (1 - np.clip(left_given_unilateral, 0, 1))
    return left, right


def fit_hierarchy(count_values: np.ndarray, demographics: np.ndarray, side_values: np.ndarray,
                  count: np.ndarray, side: np.ndarray, train: np.ndarray, seed: int,
                  include_demographics: bool):
    stage1 = binary_model(seed).fit(count_values[train], (count[train] == 1).astype(int))
    extreme = train[count[train] != 1]
    stage2_values = np.c_[count_values, demographics] if include_demographics else count_values
    stage2 = binary_model(seed + 1).fit(stage2_values[extreme], (count[extreme] == 2).astype(int))
    unilateral = train[side[train] >= 0]
    side_pre = H.preprocessing(16, seed).fit(side_values[unilateral])
    side_model = H.logistic(seed).fit(side_pre.transform(side_values[unilateral]), side[unilateral])
    p1 = stage1.predict_proba(count_values)[:, list(stage1.classes_).index(1)]
    p2_given = stage2.predict_proba(stage2_values)[:, list(stage2.classes_).index(1)]
    left_index = list(side_model.classes_).index(0)
    left_given = side_model.predict_proba(side_pre.transform(side_values))[:, left_index]
    left, right = compose_count_marginals(p1, p2_given, left_given)
    return left, right, {"stage1": stage1, "stage2": stage2,
                         "side_pre": side_pre, "side_model": side_model}


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists(): raise FileExistsError("Preserve completed hierarchy")
    patients, kidneys, images = F.matched_reference(); labels = kidneys.kidney_APN.to_numpy(int)
    demographics = pd.read_csv(OUT / "deidentified_demographics.csv").set_index("patient_alias").reindex(patients.patient_alias)
    demo = np.c_[demographics.age_years.to_numpy(float), np.log1p(demographics.age_years.to_numpy(float)),
                 demographics.female.to_numpy(float)]
    relative = pd.read_csv(G.OUT / "relative_geometry.csv").set_index("image_id").reindex(images.image_id)
    relative = relative[[column for column in relative if column.startswith("geometry_")]].to_numpy(float)
    geometry_pairs = F.patient_pair_features(relative, images, patients); half = geometry_pairs.shape[1] // 2
    geometry_left, geometry_right = geometry_pairs[:, :half], geometry_pairs[:, half:]
    count_values = np.c_[geometry_left + geometry_right, np.abs(geometry_left - geometry_right)]
    deep = np.concatenate([np.load(D.OUT / name) for name in
                           ("radimagenet_gray.npy", "radimagenet_color.npy", "usf_gray.npy")], axis=1)
    deep_pairs = F.patient_pair_features(deep, images, patients); deep_half = deep_pairs.shape[1] // 2
    deep_left, deep_right = deep_pairs[:, :deep_half], deep_pairs[:, deep_half:]
    side_values = np.c_[deep_left - deep_right, deep_left + deep_right]
    code = patients.DMSA_APN.map(H.CODE).to_numpy(int)
    count = np.asarray([0 if value == 0 else 2 if value == 3 else 1 for value in code])
    side = np.asarray([0 if value == 1 else 1 if value == 2 else -1 for value in code])
    assignments = pd.read_csv(F.OUT / "patient_assignments.csv")
    reference = pd.read_csv(D.OUT / "development_predictions.csv")
    reference = reference[reference.group.eq(REFERENCE)].set_index(["seed", "kidney_alias"]).score
    powerset = pd.read_csv(H.OUT / "development_predictions.csv")
    powerset = powerset[powerset.group.eq(POWERSET)].set_index(["seed", "kidney_alias"]).score
    patient_lookup = {alias: index for index, alias in enumerate(patients.patient_alias)}
    records, predictions = [], []; (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in F.SEEDS:
            mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
            train = np.flatnonzero(patients.patient_alias.map(mapping).eq("train"))
            left_g, right_g, model_g = fit_hierarchy(count_values, demo, side_values, count, side, train, seed, False)
            left_d, right_d, model_d = fit_hierarchy(count_values, demo, side_values, count, side, train, seed, True)
            dev = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
            geometry_score, demographic_score = [], []
            for index in dev:
                patient = patient_lookup[kidneys.patient_alias.iloc[index]]; is_left = kidneys.side.iloc[index] == "left"
                geometry_score.append(left_g[patient] if is_left else right_g[patient])
                demographic_score.append(left_d[patient] if is_left else right_d[patient])
            geometry_score, demographic_score = np.asarray(geometry_score), np.asarray(demographic_score)
            aliases = kidneys.kidney_alias.iloc[dev]
            ref = np.asarray([reference.loc[(seed, alias)] for alias in aliases])
            power = np.asarray([powerset.loc[(seed, alias)] for alias in aliases])
            scores = {GROUPS[0]: geometry_score, GROUPS[1]: demographic_score,
                      GROUPS[2]: (demographic_score + ref) / 2,
                      GROUPS[3]: (demographic_score + power) / 2}
            joblib.dump({"geometry": model_g, "demographic": model_d},
                        OUT / "models" / f"{seed}_two_stage_hierarchy.joblib", compress=3)
            for group in GROUPS:
                result = metric(labels[dev], scores[group]); records.append({"seed": seed, "role": "development",
                    "group": group, "patients": 20, "kidneys": len(dev), **result})
                for local, index in enumerate(dev): predictions.append({"seed": seed, "group": group,
                    "patient_alias": kidneys.patient_alias.iloc[index], "kidney_alias": kidneys.kidney_alias.iloc[index],
                    "DMSA_APN": kidneys.DMSA_APN.iloc[index], "side": kidneys.side.iloc[index],
                    "label": int(labels[index]), "score": float(scores[group][local])})
            print(f"Completed demographic hierarchy draw {seed}", flush=True)
    frame = pd.DataFrame(records); frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    summary = frame.groupby("group")[["auroc", "ap"]].mean().sort_values(["auroc", "ap"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc > .8) & (summary.ap > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {"groups_above_both_development_targets": achieved,
        "test_scored": False, "goal_achieved": False,
        "reason": "Development-only availability subgroup cannot complete the full objective"})
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
    print("Demographic hierarchy intervals completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "extract", "run", "bootstrap"])
    arguments = parser.parse_args(); globals()[arguments.stage]()
