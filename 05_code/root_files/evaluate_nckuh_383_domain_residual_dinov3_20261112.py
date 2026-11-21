"""Train-only nuisance residualization of paired-view DINOv3 representations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_nckuh_383_clinical_hierarchy_20261111 import clinical_table
from evaluate_nckuh_383_view_paired_20261112 import paired_features


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
DINO = ROOT / "02_experiments/NCKUH_revised_383_dinov3_20261111"
PROBE = ROOT / "02_experiments/NCKUH_revised_383_probe_geometry_20261112"
OUT = ROOT / "02_experiments/NCKUH_revised_383_domain_residual_dinov3_20261112"
SEED = 20261112


def nuisance_table(images, kidneys, roles):
    probe = pd.read_csv(PROBE / "registered_image_probe_labels.csv")
    source = pd.read_csv(REGISTRY / "source_linkage_private.csv", dtype=str)
    image_meta = images[["image_id", "kidney_alias"]].merge(probe[["image_id", "probe"]],
                                                              on="image_id", how="left")
    image_meta = image_meta.merge(source[["image_id", "source_path"]], on="image_id", validate="one_to_one")
    image_meta["probe"] = image_meta.probe.fillna("unclassified")
    image_meta["folder"] = image_meta.source_path.map(lambda x: Path(x).parent.name)
    columns = []
    for field, categories in (("probe", ["convex", "linear", "unknown", "unclassified"]),
                              ("folder", ["1", "2", "3", "NCKUH3", "test_predict"])):
        table = pd.crosstab(image_meta.kidney_alias, image_meta[field], normalize="index")
        table = table.reindex(index=kidneys.kidney_alias, columns=categories, fill_value=0)
        columns.append(table.to_numpy(np.float32))
    count = image_meta.groupby("kidney_alias").size().reindex(kidneys.kidney_alias).to_numpy(np.float32)[:, None]
    active = roles[roles.patient_alias.isin(kidneys.patient_alias)].reset_index(drop=True)
    demo = clinical_table(active.patient_alias)
    lookup = {alias: demo[i] for i, alias in enumerate(active.patient_alias)}
    demographics = np.asarray([lookup[alias] for alias in kidneys.patient_alias], dtype=np.float32)
    return np.c_[*columns, count, demographics]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve domain-residual experiment")
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv")
    kidneys = kidneys.merge(roles[["patient_alias", "role"]], on="patient_alias", validate="many_to_one")
    values = paired_features(np.load(DINO / "dinov3_class.npy"), images, kidneys)
    nuisance = nuisance_table(images, kidneys, roles)
    labels = kidneys.kidney_APN.to_numpy(int)
    groups = kidneys.patient_alias.to_numpy()
    train = np.flatnonzero(kidneys.role.eq("train"))
    dev = np.flatnonzero(kidneys.role.eq("development"))
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(values[train])
    filled = imputer.transform(values)
    nuisance_scaler = StandardScaler().fit(nuisance[train])
    z = nuisance_scaler.transform(nuisance)
    residualizer = Ridge(alpha=10).fit(z[train], filled[train])
    residual = filled - residualizer.predict(z)
    residual += filled[train].mean(axis=0)
    pipeline = Pipeline([
        ("scale", StandardScaler()), ("pca", PCA(random_state=SEED)),
        ("lr", LogisticRegression(class_weight="balanced", solver="liblinear",
                                  max_iter=4000, random_state=SEED)),
    ])
    search = GridSearchCV(
        pipeline, {"pca__n_components": [8, 16, 32, 64], "lr__C": [.001, .01, .1, 1]},
        scoring="roc_auc", cv=StratifiedGroupKFold(4, shuffle=True, random_state=SEED),
        n_jobs=1, refit=True)
    search.fit(residual[train], labels[train], groups=groups[train])
    score = search.predict_proba(residual[dev])[:, 1]
    target = kidneys.iloc[dev].reset_index(drop=True)
    clusters = [np.flatnonzero(target.patient_alias.eq(pid)) for pid in sorted(target.patient_alias.unique())]
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
             for _ in range(2000)]
    draws = [idx for idx in draws if np.unique(labels[dev][idx]).size == 2]
    auc = np.asarray([roc_auc_score(labels[dev][idx], score[idx]) for idx in draws])
    ap = np.asarray([average_precision_score(labels[dev][idx], score[idx]) for idx in draws])
    result = {"group": "domain_residual_dinov3_view_paired", "patients": target.patient_alias.nunique(),
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
        "residualization": "train-only ridge alpha10 removes representation directions explained by probe fractions, source-folder fractions, image count, age and sex",
        "probe_source": "OCR plus gated sector-geometry inference; unclassified retained as a category",
        "classifier_selection": "4-fold patient-grouped training-only CV",
        "test_scored": False,
        "warning": "Exploratory nuisance adjustment; no causal interpretation"}, indent=2), encoding="utf-8")
    print(search.best_score_, search.best_params_)
    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
