"""OOF stack adding fold-local domain-residual DINOv3 predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_nckuh_383_domain_residual_dinov3_20261112 import nuisance_table
from evaluate_nckuh_383_oof_stacking_20261112 import (
    demographic_model, dino_model, geometry_model, make_geometry,
)
from evaluate_nckuh_383_view_paired_20261112 import paired_features


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
DINO = ROOT / "02_experiments/NCKUH_revised_383_dinov3_20261111"
GEOMETRY = ROOT / "02_experiments/NCKUH_revised_383_relative_geometry_20261112"
OUT = ROOT / "02_experiments/NCKUH_revised_383_oof_domain_stacking_20261112"
SEED = 20261112


def residual_scores(values, nuisance, labels, fit, hold):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(values[fit])
    filled = imputer.transform(values)
    scaler = StandardScaler().fit(nuisance[fit])
    z = scaler.transform(nuisance)
    residualizer = Ridge(alpha=10).fit(z[fit], filled[fit])
    residual = filled - residualizer.predict(z) + filled[fit].mean(axis=0)
    classifier = dino_model().fit(residual[fit], labels[fit])
    return classifier.predict_proba(residual[hold])[:, 1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve OOF domain stack")
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
    nuisance = nuisance_table(images, kidneys, roles)
    demo_values = nuisance[:, -2:]
    names = ["dino", "domain_residual_dino", "geometry", "demographics"]
    oof = np.full((len(train), len(names)), np.nan, np.float32)
    dev_base = np.full((len(dev), len(names)), np.nan, np.float32)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(
        dino_values[train], labels[train], groups[train]))
    ordinary = {"dino": (dino_values, dino_model), "geometry": (geometry_values, geometry_model),
                "demographics": (demo_values, demographic_model)}
    for column, name in enumerate(names):
        if name == "domain_residual_dino":
            for fit_local, hold_local in folds:
                fit, hold = train[fit_local], train[hold_local]
                oof[hold_local, column] = residual_scores(dino_values, nuisance, labels, fit, hold)
            dev_base[:, column] = residual_scores(dino_values, nuisance, labels, train, dev)
        else:
            values, factory = ordinary[name]
            for fit_local, hold_local in folds:
                model = factory().fit(values[train[fit_local]], labels[train[fit_local]])
                oof[hold_local, column] = model.predict_proba(values[train[hold_local]])[:, 1]
            model = factory().fit(values[train], labels[train])
            dev_base[:, column] = model.predict_proba(values[dev])[:, 1]
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
    result = {"group": "oof_domain_dino_geometry_demographics", "patients": target.patient_alias.nunique(),
              "kidneys": len(dev), "prevalence": labels[dev].mean(),
              "auroc": roc_auc_score(labels[dev], score), "ap": average_precision_score(labels[dev], score),
              "auroc_ci_low": np.percentile(auc, 2.5), "auroc_ci_high": np.percentile(auc, 97.5),
              "ap_ci_low": np.percentile(ap, 2.5), "ap_ci_high": np.percentile(ap, 97.5)}
    pd.DataFrame([result]).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame({"patient_alias": target.patient_alias, "kidney_alias": target.kidney_alias,
                  "side": target.side, "label": labels[dev], "score": score,
                  **{f"base_{name}": dev_base[:, i] for i, name in enumerate(names)}}).to_csv(
                      OUT / "development_predictions.csv", index=False)
    pd.DataFrame({"source": names, "meta_coefficient": meta.named_steps["logisticregression"].coef_[0]}).to_csv(
        OUT / "meta_coefficients.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "oof": "5-fold patient-grouped; nuisance residualizer, each base learner, and predictions fit within fold",
        "sources": names, "meta": "training OOF standardized balanced LR C=.1",
        "test_scored": False}, indent=2), encoding="utf-8")
    print(pd.DataFrame([result]).to_string(index=False))
    print(pd.read_csv(OUT / "meta_coefficients.csv").to_string(index=False))


if __name__ == "__main__":
    main()
