"""Kidney-balanced image-instance learning on clean full-frame NCKUH ROIs."""
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
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullframe_image_instance_probe_20261102"
REFERENCE = "deep_concat_plus_clean_radiomics"
GROUPS = (
    "deep_image_probability_mean",
    "deep_image_logit_mean",
    "deep_radiomics_image_probability_mean",
    "deep_radiomics_image_logit_mean",
    "view_specific_deep_radiomics_probability_mean",
    "view_specific_deep_radiomics_logit_mean",
    "deep_radiomics_image_logit_mean_plus_kidney_reference",
    "view_specific_logit_mean_plus_kidney_reference",
)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered image-instance protocol")
    F.require_lock()
    D.require_lock()
    inputs = [
        F.OUT / "patient_assignments.csv",
        F.OUT / "deidentified_image_manifest.csv",
        F.OUT / "radiomics_clean_fullframe.csv",
        D.OUT / "radimagenet_gray.npy",
        D.OUT / "radimagenet_color.npy",
        D.OUT / "usf_gray.npy",
        D.OUT / "development_predictions.csv",
        Path(__file__),
    ]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same clean-full-frame 111 patients, 212 kidneys, 418 images and exact 70/20/21 patient roles",
        "motivation": "Kidney-level feature averaging can erase view-local APN appearance; fit on clean individual ROI views and aggregate predictions only after scoring",
        "representations": "Frozen RadImageNet gray/color plus USF-MAE features, alone or early-concatenated with clean-mask PyRadiomics",
        "training": "Each image inherits only its own kidney DMSA side label; each training kidney has total weight one before binary class balancing",
        "heads": "Train-only median imputation, standardization, whitened PCA64 and balanced-by-explicit-weight LR C0.1; pooled and separate longitudinal/transverse controls",
        "pooling": "Fixed arithmetic probability mean or sigmoid of arithmetic image-logit mean per kidney",
        "fusion": "Two prespecified equal-probability averages with the completed kidney-level deep-radiomics reference",
        "prohibited": "No filename diagnosis suffix, DMSA code as predictor, image-count weighting, development-selected pooling, threshold or fusion weight, and no exploratory-test scoring",
        "target": "Development AUROC>0.8 and AP>0.8; later locked confirmation remains necessary",
        "groups": list(GROUPS),
        "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "source_sha256": digest(__file__),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs},
    })
    print("Registered kidney-balanced clean-ROI image-instance probe", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(Path(name)) == expected, name


def balanced_image_weights(kidney_aliases: np.ndarray, labels: np.ndarray) -> np.ndarray:
    aliases = np.asarray(kidney_aliases)
    target = np.asarray(labels, dtype=int)
    unique, first, counts = np.unique(aliases, return_index=True, return_counts=True)
    kidney_labels = target[first]
    if any(np.unique(target[aliases == alias]).size != 1 for alias in unique):
        raise ValueError("An imaged kidney has inconsistent labels")
    class_counts = np.bincount(kidney_labels, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError("Both classes are required")
    class_weight = len(unique) / (2 * class_counts)
    count_lookup = dict(zip(unique, counts))
    return np.asarray([class_weight[label] / count_lookup[alias] for alias, label in zip(aliases, target)])


def fit_image_model(values: np.ndarray, train: np.ndarray, labels: np.ndarray,
                    kidney_aliases: np.ndarray, seed: int, components: int = 64):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    transform = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        PCA(components, whiten=True, random_state=seed),
    )
    transformed = transform.fit_transform(values[train])
    model = LogisticRegression(C=.1, solver="liblinear", max_iter=3000, random_state=seed)
    weights = balanced_image_weights(kidney_aliases[train], labels[train])
    model.fit(transformed, labels[train], sample_weight=weights)
    score = model.predict_proba(transform.transform(values))[:, list(model.classes_).index(1)]
    return score, transform, model


def pool_to_kidneys(image_scores: np.ndarray, images: pd.DataFrame,
                    kidneys: pd.DataFrame, method: str) -> np.ndarray:
    score = np.asarray(image_scores, dtype=float)
    output = []
    for alias in kidneys.kidney_alias:
        selected = score[images.kidney_alias.eq(alias).to_numpy()]
        if not len(selected):
            raise ValueError(f"Kidney without images: {alias}")
        if method == "probability_mean":
            value = selected.mean()
        elif method == "logit_mean":
            clipped = np.clip(selected, 1e-6, 1 - 1e-6)
            mean_logit = np.log(clipped / (1 - clipped)).mean()
            value = 1 / (1 + np.exp(-mean_logit))
        else:
            raise ValueError(method)
        output.append(value)
    return np.asarray(output)


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed image-instance predictions")
    patients, kidneys, images = F.matched_reference()
    image_labels = images.kidney_APN.to_numpy(int)
    kidney_labels = kidneys.kidney_APN.to_numpy(int)
    deep = np.concatenate([
        np.load(D.OUT / "radimagenet_gray.npy"),
        np.load(D.OUT / "radimagenet_color.npy"),
        np.load(D.OUT / "usf_gray.npy"),
    ], axis=1)
    radiomics = pd.read_csv(F.OUT / "radiomics_clean_fullframe.csv").set_index("image_id").reindex(images.image_id)
    radiomics = radiomics[[column for column in radiomics if not column.endswith(("_Energy", "_TotalEnergy"))]].to_numpy(float)
    combined = np.concatenate([deep, radiomics], axis=1)
    assignments = pd.read_csv(F.OUT / "patient_assignments.csv")
    reference = pd.read_csv(D.OUT / "development_predictions.csv")
    reference = reference[reference.group.eq(REFERENCE)].set_index(["seed", "kidney_alias"]).score
    records, predictions = [], []
    (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in F.SEEDS:
            mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
            image_train = np.flatnonzero(images.patient_alias.map(mapping).eq("train"))
            kidney_dev = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
            deep_score, deep_transform, deep_model = fit_image_model(
                deep, image_train, image_labels, images.kidney_alias.to_numpy(), seed)
            combined_score, combined_transform, combined_model = fit_image_model(
                combined, image_train, image_labels, images.kidney_alias.to_numpy(), seed)
            view_score = np.full(len(images), np.nan)
            view_models = {}
            for suffix in ("L", "T"):
                subset = images.view.str.endswith(suffix).to_numpy()
                selected_train = np.flatnonzero(subset & images.patient_alias.map(mapping).eq("train").to_numpy())
                selected_score, transform, model = fit_image_model(
                    combined, selected_train, image_labels, images.kidney_alias.to_numpy(), seed, components=48)
                view_score[subset] = selected_score[subset]
                view_models[suffix] = {"transform": transform, "model": model}
            assert np.isfinite(view_score).all()
            pooled = {
                GROUPS[0]: pool_to_kidneys(deep_score, images, kidneys, "probability_mean"),
                GROUPS[1]: pool_to_kidneys(deep_score, images, kidneys, "logit_mean"),
                GROUPS[2]: pool_to_kidneys(combined_score, images, kidneys, "probability_mean"),
                GROUPS[3]: pool_to_kidneys(combined_score, images, kidneys, "logit_mean"),
                GROUPS[4]: pool_to_kidneys(view_score, images, kidneys, "probability_mean"),
                GROUPS[5]: pool_to_kidneys(view_score, images, kidneys, "logit_mean"),
            }
            ref = np.asarray([reference.loc[(seed, alias)] for alias in kidneys.kidney_alias.iloc[kidney_dev]])
            pooled[GROUPS[6]] = pooled[GROUPS[3]].copy()
            pooled[GROUPS[7]] = pooled[GROUPS[5]].copy()
            pooled[GROUPS[6]][kidney_dev] = (pooled[GROUPS[3]][kidney_dev] + ref) / 2
            pooled[GROUPS[7]][kidney_dev] = (pooled[GROUPS[5]][kidney_dev] + ref) / 2
            joblib.dump({"transform": deep_transform, "model": deep_model}, OUT / "models" / f"{seed}_deep.joblib", compress=3)
            joblib.dump({"transform": combined_transform, "model": combined_model}, OUT / "models" / f"{seed}_deep_radiomics.joblib", compress=3)
            joblib.dump(view_models, OUT / "models" / f"{seed}_view_specific.joblib", compress=3)
            for group in GROUPS:
                result = metric(kidney_labels[kidney_dev], pooled[group][kidney_dev])
                records.append({"seed": seed, "role": "development", "group": group,
                                "patients": 20, "kidneys": len(kidney_dev), **result})
                for index in kidney_dev:
                    predictions.append({
                        "seed": seed, "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[index],
                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                        "side": kidneys.side.iloc[index],
                        "label": int(kidney_labels[index]),
                        "score": float(pooled[group][index]),
                    })
            print(f"Completed image-instance draw {seed}", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    summary = frame.groupby("group")[["auroc", "ap"]].mean().sort_values(["auroc", "ap"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc > .8) & (summary.ap > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {
        "groups_above_both_development_targets": achieved,
        "test_scored": False,
        "goal_achieved": False,
        "reason": "Development-only availability subgroup cannot complete the full objective",
    })
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
    write_json(OUT / "completion_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "patients": 111, "kidneys": 212, "images": 418,
        "test_scored": False, "scored_unit": "kidney", "bootstrap_cluster": "patient",
    })
    print("Image-instance intervals completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "bootstrap"])
    arguments = parser.parse_args()
    globals()[arguments.stage]()
