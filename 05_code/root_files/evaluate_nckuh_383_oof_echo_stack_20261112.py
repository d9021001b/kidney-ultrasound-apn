"""Add sparse ultrasound-report evidence to the patient-OOF five-source stack."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_nckuh_383_echo_report_20261112 import patient_key, report_mapping, side_features


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
SOURCE = ROOT / "02_experiments/NCKUH_revised_383_oof_hierarchy_stack_base_export_20261112"
OUT = ROOT / "02_experiments/NCKUH_revised_383_oof_echo_stack_20261112"
SEED = 20261112


def report_model():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=.01, class_weight="balanced", solver="liblinear",
                                            max_iter=3000, random_state=SEED))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve OOF echo stack")
    train = pd.read_csv(SOURCE / "training_oof_predictions.csv")
    dev = pd.read_csv(SOURCE / "development_predictions.csv")
    base_columns = [column for column in train if column.startswith("base_")]
    linkage = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype=str)
    key_to_alias = {patient_key(row.patient_id): row.patient_alias for row in linkage.itertuples(index=False)}
    reports = {key_to_alias[key]: value[1] for key, value in report_mapping().items() if key in key_to_alias}

    def features(frame):
        matrix, available = [], []
        for row in frame.itertuples(index=False):
            if row.patient_alias in reports:
                side = "left" if row.kidney_alias.endswith("_L") else "right"
                matrix.append(side_features(reports[row.patient_alias], side)); available.append(True)
            else:
                matrix.append(np.full(9, np.nan, np.float32)); available.append(False)
        return np.asarray(matrix), np.asarray(available)

    x_train, has_train = features(train); x_dev, has_dev = features(dev)
    y = train.label.to_numpy(int); y_dev = dev.label.to_numpy(int)
    report_oof = np.full(len(train), .5, np.float32)
    report_train_idx = np.flatnonzero(has_train)
    splitter = StratifiedGroupKFold(4, shuffle=True, random_state=SEED)
    for fit_local, hold_local in splitter.split(x_train[report_train_idx], y[report_train_idx],
                                                train.patient_alias.to_numpy()[report_train_idx]):
        fit = report_train_idx[fit_local]; hold = report_train_idx[hold_local]
        fitted = report_model().fit(x_train[fit], y[fit])
        report_oof[hold] = fitted.predict_proba(x_train[hold])[:, 1]
    report_dev = np.full(len(dev), .5, np.float32)
    final_report = report_model().fit(x_train[report_train_idx], y[report_train_idx])
    report_dev[has_dev] = final_report.predict_proba(x_dev[has_dev])[:, 1]
    composite_train = np.where(has_train, np.nansum(x_train[:, :3], axis=1) / 3, .5)
    composite_dev = np.where(has_dev, np.nansum(x_dev[:, :3], axis=1) / 3, .5)
    meta_train = np.c_[train[base_columns].to_numpy(float), report_oof,
                       has_train.astype(float), composite_train]
    meta_dev = np.c_[dev[base_columns].to_numpy(float), report_dev,
                     has_dev.astype(float), composite_dev]
    meta_pipeline = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced",
                                  solver="liblinear", max_iter=3000, random_state=SEED))
    meta = GridSearchCV(meta_pipeline, {"logisticregression__C": [.001, .01, .1, 1, 10]},
                        scoring="roc_auc", cv=StratifiedGroupKFold(5, shuffle=True, random_state=SEED),
                        n_jobs=1, refit=True)
    meta.fit(meta_train, y, groups=train.patient_alias); score = meta.predict_proba(meta_dev)[:, 1]
    clusters = [np.flatnonzero(dev.patient_alias.eq(pid)) for pid in sorted(dev.patient_alias.unique())]
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))]) for _ in range(3000)]
    draws = [idx for idx in draws if np.unique(y_dev[idx]).size == 2]
    auc = np.asarray([roc_auc_score(y_dev[idx], score[idx]) for idx in draws])
    ap = np.asarray([average_precision_score(y_dev[idx], score[idx]) for idx in draws])
    result = {"group": "oof_five_source_plus_sparse_echo", "patients": dev.patient_alias.nunique(),
              "kidneys": len(dev), "report_train_patients": train.loc[has_train, "patient_alias"].nunique(),
              "report_development_patients": dev.loc[has_dev, "patient_alias"].nunique(),
              "auroc": roc_auc_score(y_dev, score), "ap": average_precision_score(y_dev, score),
              "auroc_ci_low": np.percentile(auc, 2.5), "auroc_ci_high": np.percentile(auc, 97.5),
              "ap_ci_low": np.percentile(ap, 2.5), "ap_ci_high": np.percentile(ap, 97.5)}
    names = [column.removeprefix("base_") for column in base_columns] + ["echo_oof", "echo_available", "echo_composite"]
    pd.DataFrame([result]).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame({"patient_alias": dev.patient_alias, "kidney_alias": dev.kidney_alias,
                  "side": dev.side, "label": y_dev, "score": score}).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame({"source": names, "meta_coefficient": meta.best_estimator_.named_steps["logisticregression"].coef_[0]}).to_csv(
        OUT / "meta_coefficients.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "echo_features": "side-specific report parser; report model predictions are patient-grouped OOF for training rows",
        "missing_reports": "score and composite fixed at neutral 0.5 plus explicit availability indicator",
        "meta": "balanced LR C selected from [.001,.01,.1,1,10] by 5-fold patient-grouped training-OOF AUROC",
        "meta_selection": {"inner_cv_auroc": meta.best_score_, "parameters": meta.best_params_},
        "test_scored": False,
        "limitation": "Only 33 training and 11 development patients have echo reports; nonrandom availability."}, indent=2), encoding="utf-8")
    print(pd.DataFrame([result]).to_string(index=False)); print(pd.read_csv(OUT / "meta_coefficients.csv").to_string(index=False))


if __name__ == "__main__":
    main()
