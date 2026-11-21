"""Training-selected longitudinal/transverse paired representation probes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
DINO = ROOT / "02_experiments/NCKUH_revised_383_dinov3_20261111"
URI = ROOT / "02_experiments/NCKUH_revised_383_contour_cache_20261111"
OUT = ROOT / "02_experiments/NCKUH_revised_383_view_paired_20261112"
SEED = 20261112


def paired_features(image_values, images, kidneys):
    width = image_values.shape[1]
    output = []
    for kidney in kidneys.itertuples(index=False):
        subset = images[images.kidney_alias.eq(kidney.kidney_alias)]
        views = []
        flags = []
        for suffix in ("L", "T"):
            rows = subset.index[subset.view.str.endswith(suffix)].to_numpy()
            views.append(image_values[rows].mean(axis=0) if len(rows) else np.full(width, np.nan))
            flags.append(float(len(rows) > 0))
        long, trans = views
        output.append(np.r_[long, trans, (long + trans) / 2, np.abs(long - trans), flags])
    return np.asarray(output, dtype=np.float32)


def fit_probe(values, labels, train, groups, name):
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("pca", PCA(random_state=SEED)),
        ("lr", LogisticRegression(class_weight="balanced", solver="liblinear",
                                  max_iter=4000, random_state=SEED)),
    ])
    search = GridSearchCV(
        pipeline,
        {"pca__n_components": [8, 16, 32], "lr__C": [0.001, 0.01, 0.1, 1.0]},
        scoring="roc_auc",
        cv=StratifiedGroupKFold(4, shuffle=True, random_state=SEED),
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    search.fit(values[train], labels[train], groups=groups[train])
    return search


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve paired-view experiment")
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv")
    kidneys = kidneys.merge(roles[["patient_alias", "role"]], on="patient_alias", validate="many_to_one")
    train = np.flatnonzero(kidneys.role.eq("train"))
    dev = np.flatnonzero(kidneys.role.eq("development"))
    labels = kidneys.kidney_APN.to_numpy(int)
    groups = kidneys.patient_alias.to_numpy()
    sources = {
        "dinov3_class_view_paired": np.load(DINO / "dinov3_class.npy"),
        "uri_embedding_mean_view_paired": np.load(URI / "uricads_fold_embeddings.npy").mean(axis=1),
    }
    target = kidneys.iloc[dev].reset_index(drop=True)
    clusters = [np.flatnonzero(target.patient_alias.eq(pid)) for pid in sorted(target.patient_alias.unique())]
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
             for _ in range(2000)]
    draws = [idx for idx in draws if np.unique(labels[dev][idx]).size == 2]
    results, predictions, selections = [], [], []
    for name, image_values in sources.items():
        assert image_values.shape[0] == len(images) and np.isfinite(image_values).all()
        values = paired_features(image_values, images, kidneys)
        model = fit_probe(values, labels, train, groups, name)
        score = model.predict_proba(values[dev])[:, 1]
        auc = np.asarray([roc_auc_score(labels[dev][idx], score[idx]) for idx in draws])
        ap = np.asarray([average_precision_score(labels[dev][idx], score[idx]) for idx in draws])
        results.append({"group": name, "patients": target.patient_alias.nunique(),
                        "kidneys": len(dev), "prevalence": labels[dev].mean(),
                        "auroc": roc_auc_score(labels[dev], score),
                        "ap": average_precision_score(labels[dev], score),
                        "auroc_ci_low": np.percentile(auc, 2.5),
                        "auroc_ci_high": np.percentile(auc, 97.5),
                        "ap_ci_low": np.percentile(ap, 2.5),
                        "ap_ci_high": np.percentile(ap, 97.5)})
        selections.append({"group": name, "inner_cv_auroc": model.best_score_, **model.best_params_})
        predictions.extend({"group": name, "patient_alias": row.patient_alias,
                            "kidney_alias": row.kidney_alias, "side": row.side,
                            "label": int(row.kidney_APN), "score": float(value)}
                           for row, value in zip(target.itertuples(index=False), score))
        print(name, model.best_score_, model.best_params_, results[-1], flush=True)
    pd.DataFrame(results).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "training_inner_cv_selections.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "motivation": "Preserve longitudinal/transverse identity and their difference because focal APN findings may be visible in only one plane.",
        "representation": "[longitudinal, transverse, mean, absolute difference, presence flags]",
        "selection": "4-fold stratified grouped CV on training patients only",
        "grid": {"PCA": [8, 16, 32], "C": [0.001, 0.01, 0.1, 1.0]},
        "test_scored": False,
        "development_uncertainty": "2000 patient-cluster bootstrap draws"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
