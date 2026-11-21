"""Relative renal geometry and contralateral asymmetry probe on revised NCKUH."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_nckuh_383_clinical_hierarchy_20261111 import clinical_table
from evaluate_nckuh_383_view_paired_20261112 import paired_features
from nckuh_dmsa_fullcohort_fullframe_20261103 import split_panel
from nckuh_dmsa_fullframe_relative_geometry_probe_20261102 import relative_geometry


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
OUT = ROOT / "02_experiments/NCKUH_revised_383_relative_geometry_20261112"
SEED = 20261112


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve relative geometry experiment")
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    sources = pd.read_csv(REGISTRY / "source_linkage_private.csv", dtype=str).set_index("image_id")
    sources = sources.reindex(images.image_id)
    rows, fallbacks = [], 0
    for path in sources.source_path:
        frame, mask = split_panel(Path(path))
        values, fallback = relative_geometry(frame, mask)
        rows.append(values)
        fallbacks += int(fallback)
    image_geometry = np.asarray(rows, dtype=np.float32)
    assert image_geometry.shape == (1362, 21) and np.isfinite(image_geometry).all()
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv")
    kidneys = kidneys.merge(roles[["patient_alias", "role"]], on="patient_alias", validate="many_to_one")
    own = paired_features(image_geometry, images, kidneys)
    lookup = {kidney: i for i, kidney in enumerate(kidneys.kidney_alias)}
    opposite = []
    for row in kidneys.itertuples(index=False):
        suffix = "R" if row.side == "left" else "L"
        index = lookup.get(f"{row.patient_alias}_{suffix}")
        opposite.append(own[index] if index is not None else np.full(own.shape[1], np.nan))
    opposite = np.asarray(opposite, dtype=np.float32)
    active_patients = roles[roles.patient_alias.isin(kidneys.patient_alias)].reset_index(drop=True)
    demo = clinical_table(active_patients.patient_alias)
    demo_lookup = {alias: demo[i] for i, alias in enumerate(active_patients.patient_alias)}
    covariates = np.asarray([demo_lookup[alias] for alias in kidneys.patient_alias], dtype=np.float32)
    values = np.c_[own, opposite, own - opposite, np.abs(own - opposite), covariates]
    labels = kidneys.kidney_APN.to_numpy(int)
    groups = kidneys.patient_alias.to_numpy()
    train = np.flatnonzero(kidneys.role.eq("train"))
    dev = np.flatnonzero(kidneys.role.eq("development"))
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif)),
        ("lr", LogisticRegression(class_weight="balanced", solver="liblinear",
                                  max_iter=4000, random_state=SEED)),
    ])
    search = GridSearchCV(
        pipeline, {"select__k": [16, 32, 64, 128], "lr__C": [0.001, 0.01, 0.1, 1.0]},
        scoring="roc_auc", cv=StratifiedGroupKFold(4, shuffle=True, random_state=SEED),
        n_jobs=1, refit=True)
    search.fit(values[train], labels[train], groups=groups[train])
    score = search.predict_proba(values[dev])[:, 1]
    target = kidneys.iloc[dev].reset_index(drop=True)
    clusters = [np.flatnonzero(target.patient_alias.eq(pid)) for pid in sorted(target.patient_alias.unique())]
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
             for _ in range(2000)]
    draws = [idx for idx in draws if np.unique(labels[dev][idx]).size == 2]
    auc = np.asarray([roc_auc_score(labels[dev][idx], score[idx]) for idx in draws])
    ap = np.asarray([average_precision_score(labels[dev][idx], score[idx]) for idx in draws])
    result = {"group": "relative_geometry_contralateral", "patients": target.patient_alias.nunique(),
              "kidneys": len(dev), "prevalence": labels[dev].mean(),
              "auroc": roc_auc_score(labels[dev], score), "ap": average_precision_score(labels[dev], score),
              "auroc_ci_low": np.percentile(auc, 2.5), "auroc_ci_high": np.percentile(auc, 97.5),
              "ap_ci_low": np.percentile(ap, 2.5), "ap_ci_high": np.percentile(ap, 97.5)}
    pd.DataFrame([result]).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame({"patient_alias": target.patient_alias, "kidney_alias": target.kidney_alias,
                  "side": target.side, "label": labels[dev], "score": score}).to_csv(
                      OUT / "development_predictions.csv", index=False)
    pd.DataFrame([{"inner_cv_auroc": search.best_score_, **search.best_params_}]).to_csv(
        OUT / "training_inner_cv_selection.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "features": "21 label-blind contour/sector-relative geometry measures per image; paired longitudinal/transverse; own/opposite/difference/absolute difference; age and sex covariates",
        "sector_fallbacks": fallbacks,
        "selection": "4-fold patient-grouped training-only CV",
        "test_scored": False,
        "warning": "Relative image geometry, not calibrated physical renal length or volume"}, indent=2), encoding="utf-8")
    np.save(OUT / "relative_geometry_image.npy", image_geometry)
    print(search.best_score_, search.best_params_)
    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
