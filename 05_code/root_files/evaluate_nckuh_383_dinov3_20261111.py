"""Frozen DINOv3 class/patch probes on the fixed revised NCKUH split."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_nckuh_383_baseline_20261111 import aggregate
from prepare_nckuh_383_contour_cohort_20261111 import digest


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
ENCODED = ROOT / "02_experiments/NCKUH_revised_383_dinov3_20261111"


def main():
    if (ENCODED / "development_metrics.csv").exists():
        raise FileExistsError("Preserve DINOv3 development experiment")
    audit = json.loads((ENCODED / "encoding_audit.json").read_text(encoding="utf-8"))
    assert digest(REGISTRY / "contour_image_manifest.csv") == audit["image_manifest_sha256"]
    for name, key in (("dinov3_class.npy", "class_sha256"), ("dinov3_patch.npy", "patch_sha256")):
        assert digest(ENCODED / name) == audit[key]
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv")
    kidneys = kidneys.merge(roles[["patient_alias", "role"]], on="patient_alias", validate="many_to_one")
    train = np.flatnonzero(kidneys.role.eq("train"))
    dev = np.flatnonzero(kidneys.role.eq("development"))
    train_patients = set(kidneys.patient_alias.iloc[train])
    dev_patients = set(kidneys.patient_alias.iloc[dev])
    assert not train_patients & dev_patients
    train_images = np.flatnonzero(images.patient_alias.isin(train_patients))
    assert (len(images), len(train_images), len(train), len(dev)) == (1362, 796, 402, 140)
    labels = kidneys.kidney_APN.to_numpy(int)
    results, predictions = [], []
    clusters = [np.flatnonzero(kidneys.patient_alias.iloc[dev].eq(pid))
                for pid in sorted(dev_patients)]
    rng = np.random.default_rng(20261111)
    draws = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
             for _ in range(2000)]
    draws = [idx for idx in draws if np.unique(labels[dev][idx]).size == 2]
    for group, filename in (("dinov3_class", "dinov3_class.npy"),
                            ("dinov3_mask_patch", "dinov3_patch.npy")):
        matrix = np.load(ENCODED / filename).astype(np.float32)
        assert matrix.shape[0] == len(images) and np.isfinite(matrix).all()
        image_scaler = StandardScaler().fit(matrix[train_images])
        transformed = image_scaler.transform(matrix)
        pca = PCA(32, random_state=20261111).fit(transformed[train_images])
        kidney_features = aggregate(pca.transform(transformed), images, kidneys.kidney_alias)
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              StandardScaler(), LogisticRegression(C=.1, class_weight="balanced",
                              solver="liblinear", max_iter=3000, random_state=20261111))
        model.fit(kidney_features[train], labels[train])
        score = model.predict_proba(kidney_features[dev])[:, 1]
        auc = np.asarray([roc_auc_score(labels[dev][idx], score[idx]) for idx in draws])
        ap = np.asarray([average_precision_score(labels[dev][idx], score[idx]) for idx in draws])
        results.append({"group": group, "patients": len(dev_patients), "kidneys": len(dev),
                        "prevalence": float(labels[dev].mean()),
                        "auroc": roc_auc_score(labels[dev], score),
                        "ap": average_precision_score(labels[dev], score),
                        "auroc_ci_low": np.percentile(auc, 2.5),
                        "auroc_ci_high": np.percentile(auc, 97.5),
                        "ap_ci_low": np.percentile(ap, 2.5),
                        "ap_ci_high": np.percentile(ap, 97.5)})
        predictions.extend({"group": group, "patient_alias": kidneys.patient_alias.iloc[i],
                            "kidney_alias": kidneys.kidney_alias.iloc[i], "side": kidneys.side.iloc[i],
                            "label": int(labels[i]), "score": float(value)}
                           for i, value in zip(dev, score))
        print(f"{group}: AUROC {results[-1]['auroc']:.3f}, AP {results[-1]['ap']:.3f}", flush=True)
    pd.DataFrame(results).to_csv(ENCODED / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(ENCODED / "development_predictions.csv", index=False)
    (ENCODED / "development_protocol.json").write_text(json.dumps({
        "train_patients": len(train_patients), "development_patients": len(dev_patients),
        "model": "frozen DINOv3 features; training-image scaler/PCA32; kidney image mean/std; balanced LR C=.1",
        "test_scored": False, "uncertainty": "2000 patient-cluster bootstrap draws",
        "comparison": "same head as revised 383 RadImageNet DenseNet baseline"}, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
