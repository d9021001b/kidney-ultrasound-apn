"""Patient-grouped OOF stacking of DINOv3, renal geometry, and demographics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_nckuh_383_clinical_hierarchy_20261111 import clinical_table
from evaluate_nckuh_383_view_paired_20261112 import paired_features


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
DINO = ROOT / "02_experiments/NCKUH_revised_383_dinov3_20261111"
GEOMETRY = ROOT / "02_experiments/NCKUH_revised_383_relative_geometry_20261112"
OUT = ROOT / "02_experiments/NCKUH_revised_383_oof_stacking_20261112"
SEED = 20261112


def dino_model():
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scale", StandardScaler()), ("pca", PCA(32, random_state=SEED)),
                     ("lr", LogisticRegression(C=.001, class_weight="balanced", solver="liblinear",
                                               max_iter=4000, random_state=SEED))])


def geometry_model():
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scale", StandardScaler()), ("select", SelectKBest(f_classif, k=32)),
                     ("lr", LogisticRegression(C=.001, class_weight="balanced", solver="liblinear",
                                               max_iter=4000, random_state=SEED))])


def demographic_model():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=1, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=SEED))


def make_geometry(image_geometry, images, kidneys):
    own = paired_features(image_geometry, images, kidneys)
    lookup = {kidney: i for i, kidney in enumerate(kidneys.kidney_alias)}
    opposite = []
    for row in kidneys.itertuples(index=False):
        suffix = "R" if row.side == "left" else "L"
        index = lookup.get(f"{row.patient_alias}_{suffix}")
        opposite.append(own[index] if index is not None else np.full(own.shape[1], np.nan))
    opposite = np.asarray(opposite, dtype=np.float32)
    return np.c_[own, opposite, own - opposite, np.abs(own - opposite)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve OOF stacking experiment")
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv")
    kidneys = kidneys.merge(roles[["patient_alias", "role"]], on="patient_alias", validate="many_to_one")
    labels = kidneys.kidney_APN.to_numpy(int)
    groups = kidneys.patient_alias.to_numpy()
    train = np.flatnonzero(kidneys.role.eq("train"))
    dev = np.flatnonzero(kidneys.role.eq("development"))
    dino_values = paired_features(np.load(DINO / "dinov3_class.npy"), images, kidneys)
    geometry_values = make_geometry(np.load(GEOMETRY / "relative_geometry_image.npy"), images, kidneys)
    active_patients = roles[roles.patient_alias.isin(kidneys.patient_alias)].reset_index(drop=True)
    demo = clinical_table(active_patients.patient_alias)
    demo_lookup = {alias: demo[i] for i, alias in enumerate(active_patients.patient_alias)}
    demo_values = np.asarray([demo_lookup[alias] for alias in kidneys.patient_alias], dtype=np.float32)
    feature_sets = {"dino": dino_values, "geometry": geometry_values, "demographics": demo_values}
    factories = {"dino": dino_model, "geometry": geometry_model, "demographics": demographic_model}
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    oof = np.full((len(train), len(feature_sets)), np.nan, np.float32)
    dev_base = np.empty((len(dev), len(feature_sets)), np.float32)
    fold_rows = []
    for column, name in enumerate(feature_sets):
        values = feature_sets[name]
        for fold, (fit_local, hold_local) in enumerate(splitter.split(
                values[train], labels[train], groups[train])):
            model = factories[name]()
            model.fit(values[train[fit_local]], labels[train[fit_local]])
            oof[hold_local, column] = model.predict_proba(values[train[hold_local]])[:, 1]
            fold_rows.append({"source": name, "fold": fold,
                              "patients": len(set(groups[train[hold_local]])),
                              "kidneys": len(hold_local),
                              "auroc": roc_auc_score(labels[train[hold_local]], oof[hold_local, column]),
                              "ap": average_precision_score(labels[train[hold_local]], oof[hold_local, column])})
        final = factories[name]()
        final.fit(values[train], labels[train])
        dev_base[:, column] = final.predict_proba(values[dev])[:, 1]
    assert np.isfinite(oof).all() and np.isfinite(dev_base).all()
    meta = make_pipeline(StandardScaler(), LogisticRegression(
        C=.1, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=SEED))
    meta.fit(oof, labels[train])
    score = meta.predict_proba(dev_base)[:, 1]
    target = kidneys.iloc[dev].reset_index(drop=True)
    clusters = [np.flatnonzero(target.patient_alias.eq(pid)) for pid in sorted(target.patient_alias.unique())]
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
             for _ in range(2000)]
    draws = [idx for idx in draws if np.unique(labels[dev][idx]).size == 2]
    auc = np.asarray([roc_auc_score(labels[dev][idx], score[idx]) for idx in draws])
    ap = np.asarray([average_precision_score(labels[dev][idx], score[idx]) for idx in draws])
    result = {"group": "oof_dino_geometry_demographics_stack", "patients": target.patient_alias.nunique(),
              "kidneys": len(dev), "prevalence": labels[dev].mean(),
              "auroc": roc_auc_score(labels[dev], score), "ap": average_precision_score(labels[dev], score),
              "auroc_ci_low": np.percentile(auc, 2.5), "auroc_ci_high": np.percentile(auc, 97.5),
              "ap_ci_low": np.percentile(ap, 2.5), "ap_ci_high": np.percentile(ap, 97.5)}
    pd.DataFrame([result]).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame({"patient_alias": target.patient_alias, "kidney_alias": target.kidney_alias,
                  "side": target.side, "label": labels[dev], "score": score,
                  **{f"base_{name}": dev_base[:, i] for i, name in enumerate(feature_sets)}}).to_csv(
                      OUT / "development_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "training_oof_metrics.csv", index=False)
    pd.DataFrame({"source": list(feature_sets),
                  "meta_coefficient": meta.named_steps["logisticregression"].coef_[0]}).to_csv(
                      OUT / "meta_coefficients.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "base_sources": list(feature_sets),
        "oof": "5-fold stratified patient-grouped training-only predictions",
        "base_hyperparameters": "fixed from prior training-only CV probes",
        "meta": "standardized balanced logistic regression C=.1 fit only on training OOF scores",
        "test_scored": False,
        "development_uncertainty": "2000 patient-cluster bootstrap draws"}, indent=2), encoding="utf-8")
    print(pd.DataFrame([result]).to_string(index=False))
    print(pd.read_csv(OUT / "meta_coefficients.csv").to_string(index=False))


if __name__ == "__main__":
    main()
