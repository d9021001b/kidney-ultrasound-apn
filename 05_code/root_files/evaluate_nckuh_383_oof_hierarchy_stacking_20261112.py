"""OOF stack with a patient-state DINOv3/clinical base learner."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
from evaluate_nckuh_383_clinical_hierarchy_20261111 import clinical_table
from evaluate_nckuh_383_domain_residual_dinov3_20261112 import nuisance_table
from evaluate_nckuh_383_oof_stacking_20261112 import demographic_model, dino_model, geometry_model, make_geometry
from evaluate_nckuh_383_view_paired_20261112 import paired_features


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
DINO = ROOT / "02_experiments/NCKUH_revised_383_dinov3_20261111"
GEOMETRY = ROOT / "02_experiments/NCKUH_revised_383_relative_geometry_20261112"
OUT = ROOT / "02_experiments/NCKUH_revised_383_oof_hierarchy_stacking_20261112"
SEED = 20261112


def fixed_head(values, labels, selected):
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(), LogisticRegression(C=1, class_weight="balanced",
                          solver="liblinear", max_iter=4000, random_state=SEED))
    return model.fit(values[selected], labels[selected])


def hierarchy_score(disease, antisymmetric, states, fit_patients, patients, kidneys):
    any_label = states != hierarchy.STATE_CODE["N"]
    any_model = fixed_head(disease, any_label.astype(int), fit_patients)
    affected = fit_patients[any_label[fit_patients]]
    bilateral = states == hierarchy.STATE_CODE["B"]
    bilateral_model = fixed_head(disease, bilateral.astype(int), affected)
    unilateral = fit_patients[np.isin(states[fit_patients], [hierarchy.STATE_CODE["L"], hierarchy.STATE_CODE["R"]])]
    left = states == hierarchy.STATE_CODE["L"]
    left_model = fixed_head(antisymmetric, left.astype(int), unilateral)
    p_any = any_model.predict_proba(disease)[:, 1]
    p_bg = bilateral_model.predict_proba(disease)[:, 1]
    p_left = left_model.predict_proba(antisymmetric)[:, 1]
    p_b = p_any * p_bg; p_u = p_any * (1 - p_bg)
    probability = np.c_[1 - p_any, p_u * p_left, p_u * (1 - p_left), p_b]
    return hierarchy.kidney_scores(probability, patients, kidneys)


def residual_score(values, nuisance, labels, fit, hold):
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(values[fit])
    filled = imp.transform(values)
    scale = StandardScaler().fit(nuisance[fit]); z = scale.transform(nuisance)
    ridge = Ridge(alpha=10).fit(z[fit], filled[fit])
    residual = filled - ridge.predict(z) + filled[fit].mean(0)
    return dino_model().fit(residual[fit], labels[fit]).predict_proba(residual[hold])[:, 1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve hierarchy OOF stack")
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv")
    kidneys = kidneys.merge(roles[["patient_alias", "role"]], on="patient_alias", validate="many_to_one")
    labels = kidneys.kidney_APN.to_numpy(int); groups = kidneys.patient_alias.to_numpy()
    train = np.flatnonzero(kidneys.role.eq("train")); dev = np.flatnonzero(kidneys.role.eq("development"))
    dino_image = np.load(DINO / "dinov3_class.npy")
    dino_values = paired_features(dino_image, images, kidneys)
    geometry_values = make_geometry(np.load(GEOMETRY / "relative_geometry_image.npy"), images, kidneys)
    nuisance = nuisance_table(images, kidneys, roles); demo_values = nuisance[:, -2:]
    names = ["dino", "domain_residual_dino", "geometry", "demographics", "clinical_four_state"]
    oof = np.full((len(train), len(names)), np.nan, np.float32); dev_base = np.full((len(dev), len(names)), np.nan, np.float32)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(dino_values[train], labels[train], groups[train]))
    ordinary = {"dino": (dino_values, dino_model), "geometry": (geometry_values, geometry_model),
                "demographics": (demo_values, demographic_model)}
    for col, name in enumerate(names[:-1]):
        if name == "domain_residual_dino":
            for fit_local, hold_local in folds:
                oof[hold_local, col] = residual_score(dino_values, nuisance, labels, train[fit_local], train[hold_local])
            dev_base[:, col] = residual_score(dino_values, nuisance, labels, train, dev)
        else:
            values, factory = ordinary[name]
            for fit_local, hold_local in folds:
                fitted = factory().fit(values[train[fit_local]], labels[train[fit_local]])
                oof[hold_local, col] = fitted.predict_proba(values[train[hold_local]])[:, 1]
            fitted = factory().fit(values[train], labels[train]); dev_base[:, col] = fitted.predict_proba(values[dev])[:, 1]
    active = roles[roles.role.isin(["train", "development"]) & roles.patient_alias.isin(images.patient_alias)].reset_index(drop=True)
    active_images = images[images.patient_alias.isin(active.patient_alias)].reset_index(drop=True)
    active_kidneys = kidneys[kidneys.patient_alias.isin(active.patient_alias)].reset_index(drop=True)
    image_rows = images.index[images.patient_alias.isin(active.patient_alias)].to_numpy()
    pairs = base.patient_pair_features(dino_image[image_rows], active_images, active)
    symmetric, antisymmetric = hierarchy.structured_views(pairs)
    disease = np.c_[clinical_table(active.patient_alias), symmetric]
    states = active.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    patient_train = np.flatnonzero(active.role.eq("train")); patient_dev = np.flatnonzero(active.role.eq("development"))
    kidney_score_lookup = {kidney: i for i, kidney in enumerate(active_kidneys.kidney_alias)}
    train_local = {global_i: local for local, global_i in enumerate(train)}
    for fit_p_local, hold_p_local in StratifiedKFold(5, shuffle=True, random_state=SEED).split(patient_train, states[patient_train]):
        fit_p = patient_train[fit_p_local]; hold_p = patient_train[hold_p_local]
        scores = hierarchy_score(disease, antisymmetric, states, fit_p, active, active_kidneys)
        aliases = set(active.patient_alias.iloc[hold_p])
        for global_i in train:
            if kidneys.patient_alias.iloc[global_i] in aliases:
                oof[train_local[global_i], 4] = scores[kidney_score_lookup[kidneys.kidney_alias.iloc[global_i]]]
    final_hierarchy = hierarchy_score(disease, antisymmetric, states, patient_train, active, active_kidneys)
    dev_base[:, 4] = [final_hierarchy[kidney_score_lookup[kidneys.kidney_alias.iloc[i]]] for i in dev]
    assert np.isfinite(oof).all() and np.isfinite(dev_base).all()
    meta = make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight="balanced", solver="liblinear",
                                                               max_iter=2000, random_state=SEED))
    train_target = kidneys.iloc[train].reset_index(drop=True)
    pd.DataFrame({"patient_alias": train_target.patient_alias,
                  "kidney_alias": train_target.kidney_alias,
                  "label": labels[train],
                  **{f"base_{n}": oof[:, i] for i, n in enumerate(names)}}).to_csv(
                      OUT / "training_oof_predictions.csv", index=False)
    meta.fit(oof, labels[train]); score = meta.predict_proba(dev_base)[:, 1]
    target = kidneys.iloc[dev].reset_index(drop=True)
    clusters = [np.flatnonzero(target.patient_alias.eq(pid)) for pid in sorted(target.patient_alias.unique())]
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))]) for _ in range(2000)]
    draws = [idx for idx in draws if np.unique(labels[dev][idx]).size == 2]
    auc = np.asarray([roc_auc_score(labels[dev][idx], score[idx]) for idx in draws])
    ap = np.asarray([average_precision_score(labels[dev][idx], score[idx]) for idx in draws])
    result = {"group": "oof_domain_geometry_clinical_hierarchy_stack", "patients": target.patient_alias.nunique(),
              "kidneys": len(dev), "prevalence": labels[dev].mean(), "auroc": roc_auc_score(labels[dev], score),
              "ap": average_precision_score(labels[dev], score), "auroc_ci_low": np.percentile(auc, 2.5),
              "auroc_ci_high": np.percentile(auc, 97.5), "ap_ci_low": np.percentile(ap, 2.5),
              "ap_ci_high": np.percentile(ap, 97.5)}
    pd.DataFrame([result]).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame({"patient_alias": target.patient_alias, "kidney_alias": target.kidney_alias,
                  "side": target.side, "label": labels[dev], "score": score,
                  **{f"base_{n}": dev_base[:, i] for i, n in enumerate(names)}}).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame({"source": names, "meta_coefficient": meta.named_steps["logisticregression"].coef_[0]}).to_csv(
        OUT / "meta_coefficients.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({"sources": names,
        "oof": "Every base score is out-of-fold by patient; four-state folds stratified by N/L/R/B",
        "hierarchy_hyperparameters": "fixed C=1 from prior training-only selections",
        "meta": "training OOF balanced LR C=.1", "test_scored": False}, indent=2), encoding="utf-8")
    print(pd.DataFrame([result]).to_string(index=False)); print(pd.read_csv(OUT / "meta_coefficients.csv").to_string(index=False))


if __name__ == "__main__":
    main()
