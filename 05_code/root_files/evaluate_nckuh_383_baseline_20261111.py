"""Fixed, patient-separated development baseline on revised NCKUH labels."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prepare_nckuh_383_contour_cohort_20261111 import OUT as REGISTRY, digest


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "02_experiments/NCKUH_revised_383_contour_cache_20261111"
OUT = ROOT / "02_experiments/NCKUH_revised_383_baseline_20261111"
SEED = 20261111


def aggregate(matrix, images, kidney_ids):
    grouped = images.groupby("kidney_alias", sort=False).indices
    values = []
    for kidney_id in kidney_ids:
        rows = grouped[kidney_id]
        with np.errstate(invalid="ignore"):
            values.append(np.r_[np.nanmean(matrix[rows], axis=0), np.nanstd(matrix[rows], axis=0)])
    return np.asarray(values, dtype=np.float64)


def metric(y, score):
    return {"auroc": float(roc_auc_score(y, score)),
            "ap": float(average_precision_score(y, score)),
            "brier": float(brier_score_loss(y, score))}


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve development experiment")
    lock = json.loads((REGISTRY / "cohort_lock.json").read_text())
    assert digest(REGISTRY / "contour_image_manifest.csv") == lock["files_sha256"]["contour_image_manifest.csv"]
    assert digest(REGISTRY / "contour_kidney_manifest.csv") == lock["files_sha256"]["contour_kidney_manifest.csv"]
    for name, target in (("radiomics", "radiomics.csv"), ("dense", "radimagenet_dense_tight.npy")):
        audit = json.loads((CACHE / f"{name}_audit.json").read_text())
        assert digest(CACHE / target) == audit["sha256"] and audit["images"] == 1362
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    assignments = pd.read_csv(REGISTRY / "patient_assignments.csv")
    kidneys = kidneys.merge(assignments[["patient_alias", "role"]], on="patient_alias", validate="many_to_one")
    assert len(images) == 1362 and len(kidneys) == 686
    assert images.groupby("patient_alias").kidney_APN.nunique().max() <= 2
    assert kidneys.groupby("patient_alias").role.nunique().max() == 1
    train = np.flatnonzero(kidneys.role.eq("train"))
    dev = np.flatnonzero(kidneys.role.eq("development"))
    assert (len(train), len(dev)) == (403, 139) or (len(train) > 0 and len(dev) > 0)
    y = kidneys.kidney_APN.to_numpy(int)
    train_patients = set(kidneys.patient_alias.iloc[train])
    dev_patients = set(kidneys.patient_alias.iloc[dev])
    assert not train_patients & dev_patients
    train_image_rows = np.flatnonzero(images.patient_alias.isin(train_patients))
    assert len(train_image_rows) == 796

    radiomic = pd.read_csv(CACHE / "radiomics.csv").set_index("image_id").reindex(images.image_id)
    assert radiomic.index.tolist() == images.image_id.tolist()
    columns = [c for c in radiomic if not c.endswith(("_Energy", "_TotalEnergy"))]
    radio_image = radiomic[columns].to_numpy(np.float64)
    radio_image[~np.isfinite(radio_image)] = np.nan
    radio_kidney = aggregate(radio_image, images, kidneys.kidney_alias)
    radio_model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                StandardScaler(), SelectKBest(f_classif, k=64),
                                LogisticRegression(C=.01, class_weight="balanced", solver="liblinear",
                                                   max_iter=3000, random_state=SEED))
    radio_model.fit(radio_kidney[train], y[train])
    radio_score = radio_model.predict_proba(radio_kidney[dev])[:, 1]

    dense_image = np.load(CACHE / "radimagenet_dense_tight.npy").astype(np.float32)
    assert dense_image.shape == (1362, 1024) and np.isfinite(dense_image).all()
    image_scaler = StandardScaler().fit(dense_image[train_image_rows])
    dense_standardized = image_scaler.transform(dense_image)
    pca = PCA(32, random_state=SEED).fit(dense_standardized[train_image_rows])
    dense_kidney = aggregate(pca.transform(dense_standardized), images, kidneys.kidney_alias)
    dense_model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                StandardScaler(), LogisticRegression(C=.1, class_weight="balanced",
                                                                       solver="liblinear", max_iter=3000,
                                                                       random_state=SEED))
    dense_model.fit(dense_kidney[train], y[train])
    dense_score = dense_model.predict_proba(dense_kidney[dev])[:, 1]
    scores = {"radiomics": radio_score, "radimagenet_dense": dense_score,
              "fixed_mean": .5 * (radio_score + dense_score)}
    rows = []
    for group, score in scores.items():
        for i, value in zip(dev, score):
            rows.append({"group": group, "patient_alias": kidneys.patient_alias.iloc[i],
                         "kidney_alias": kidneys.kidney_alias.iloc[i], "side": kidneys.side.iloc[i],
                         "label": int(y[i]), "score": float(value)})
    pd.DataFrame(rows).to_csv(OUT / "development_predictions.csv", index=False)
    metrics = []
    rng = np.random.default_rng(SEED)
    cluster_indices = [np.flatnonzero(kidneys.patient_alias.iloc[dev].eq(alias))
                       for alias in sorted(dev_patients)]
    for group, score in scores.items():
        boot = []
        for _ in range(2000):
            sampled = rng.integers(0, len(cluster_indices), len(cluster_indices))
            index = np.concatenate([cluster_indices[j] for j in sampled])
            if len(np.unique(y[dev][index])) < 2:
                continue
            boot.append((roc_auc_score(y[dev][index], score[index]),
                         average_precision_score(y[dev][index], score[index])))
        interval = np.percentile(np.asarray(boot), [2.5, 97.5], axis=0)
        metrics.append({"group": group, "patients": len(dev_patients), "kidneys": len(dev),
                        "APN_prevalence": float(y[dev].mean()), **metric(y[dev], score),
                        "auroc_ci_low": float(interval[0, 0]), "auroc_ci_high": float(interval[1, 0]),
                        "ap_ci_low": float(interval[0, 1]), "ap_ci_high": float(interval[1, 1]),
                        "bootstrap_valid": len(boot)})
    pd.DataFrame(metrics).to_csv(OUT / "development_metrics.csv", index=False)
    protocol = {"source_cohort_lock_sha256": digest(REGISTRY / "cohort_lock.json"),
                "model_role": "fixed development baseline, not selected on exploratory test",
                "train": {"patients": len(train_patients), "kidneys": len(train),
                          "images": len(train_image_rows)},
                "development": {"patients": len(dev_patients), "kidneys": len(dev)},
                "test_scored": False, "patient_grouped": True,
                "radiomics": "kidney image mean/std, train-only imputer/scaler/ANOVA64, balanced LR C=.01",
                "deep": "frozen RadImageNet DenseNet-121, train-image-only scaler/PCA32, kidney mean/std, balanced LR C=.1",
                "fusion": "fixed 0.5 radiomics + 0.5 deep score",
                "uncertainty": "2000 development patient-cluster bootstrap resamples; conditional within-cohort intervals"}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
