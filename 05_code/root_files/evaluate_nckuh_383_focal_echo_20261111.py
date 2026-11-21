"""Fixed development probe of literature-motivated focal renal echo features."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_nckuh_383_baseline_20261111 import aggregate
from prepare_nckuh_383_contour_cohort_20261111 import digest


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
OUT = ROOT / "02_experiments/NCKUH_revised_383_focal_echo_20261111"


def main():
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve focal echo development experiment")
    audit = json.loads((OUT / "extraction_audit.json").read_text(encoding="utf-8"))
    assert digest(REGISTRY / "contour_image_manifest.csv") == audit["image_manifest_sha256"]
    assert digest(OUT / "focal_echo.npy") == audit["features_sha256"]
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv")
    kidneys = kidneys.merge(roles[["patient_alias", "role"]], on="patient_alias", validate="many_to_one")
    features = np.load(OUT / "focal_echo.npy")
    assert features.shape == (1362, 176) and np.isfinite(features).all()
    x = aggregate(features, images, kidneys.kidney_alias)
    y = kidneys.kidney_APN.to_numpy(int)
    train = np.flatnonzero(kidneys.role.eq("train"))
    dev = np.flatnonzero(kidneys.role.eq("development"))
    assert len(train) == 402 and len(dev) == 140
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(), SelectKBest(f_classif, k=64),
                          LogisticRegression(C=.01, class_weight="balanced", solver="liblinear",
                                             max_iter=3000, random_state=20261111))
    model.fit(x[train], y[train])
    score = model.predict_proba(x[dev])[:, 1]
    target = kidneys.iloc[dev].reset_index(drop=True)
    groups = [np.flatnonzero(target.patient_alias.eq(pid)) for pid in sorted(target.patient_alias.unique())]
    rng = np.random.default_rng(20261111)
    draws = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
             for _ in range(2000)]
    draws = [idx for idx in draws if np.unique(y[dev][idx]).size == 2]
    auc = np.asarray([roc_auc_score(y[dev][idx], score[idx]) for idx in draws])
    ap = np.asarray([average_precision_score(y[dev][idx], score[idx]) for idx in draws])
    metrics = {"group": "focal_echo_anova64_lr", "patients": target.patient_alias.nunique(),
               "kidneys": len(dev), "prevalence": float(y[dev].mean()),
               "auroc": roc_auc_score(y[dev], score), "ap": average_precision_score(y[dev], score),
               "auroc_ci_low": np.percentile(auc, 2.5), "auroc_ci_high": np.percentile(auc, 97.5),
               "ap_ci_low": np.percentile(ap, 2.5), "ap_ci_high": np.percentile(ap, 97.5)}
    pd.DataFrame([metrics]).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame({"patient_alias": target.patient_alias, "kidney_alias": target.kidney_alias,
                  "side": target.side, "label": y[dev], "score": score}).to_csv(
                      OUT / "development_predictions.csv", index=False)
    (OUT / "development_protocol.json").write_text(json.dumps({
        "train_patients": kidneys.patient_alias.iloc[train].nunique(),
        "development_patients": target.patient_alias.nunique(),
        "method": "kidney image mean/std; train-only imputer/scaler/ANOVA64; balanced LR C=.01",
        "test_scored": False, "uncertainty": "2000 patient-cluster bootstrap draws",
        "references": ["https://pubmed.ncbi.nlm.nih.gov/1651644/",
                       "https://pubmed.ncbi.nlm.nih.gov/10353409/"]}, indent=2), encoding="utf-8")
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
