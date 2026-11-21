"""Training-OOF-only selection of linear, RBF, or shallow forest meta learners."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "02_experiments/NCKUH_revised_383_oof_hierarchy_stack_base_export_20261112"
OUT = ROOT / "02_experiments/NCKUH_revised_383_nonlinear_meta_20261112"
SEED = 20261112


def score(model, values):
    if hasattr(model, "decision_function"):
        return model.decision_function(values)
    return model.predict_proba(values)[:, 1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve nonlinear meta experiment")
    train = pd.read_csv(SOURCE / "training_oof_predictions.csv")
    dev = pd.read_csv(SOURCE / "development_predictions.csv")
    columns = [column for column in train if column.startswith("base_")]
    assert columns == [column for column in dev if column.startswith("base_")]
    x = train[columns].to_numpy(float); y = train.label.to_numpy(int)
    x_dev = dev[columns].to_numpy(float); y_dev = dev.label.to_numpy(int)
    groups = train.patient_alias.to_numpy()
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    candidates = {
        "linear": GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", solver="liblinear", max_iter=3000, random_state=SEED)),
            {"logisticregression__C": [.01, .1, 1, 10]}, scoring="roc_auc", cv=cv, n_jobs=1, refit=True),
        "rbf_svm": GridSearchCV(make_pipeline(StandardScaler(), SVC(
            kernel="rbf", class_weight="balanced", random_state=SEED)),
            {"svc__C": [.1, 1, 10, 100], "svc__gamma": [.01, .1, 1, "scale"]},
            scoring="roc_auc", cv=cv, n_jobs=1, refit=True),
        "shallow_forest": GridSearchCV(RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=SEED, n_jobs=4),
            {"max_depth": [2, 3, 5], "min_samples_leaf": [3, 7, 15], "max_features": [2, 3, 5]},
            scoring="roc_auc", cv=cv, n_jobs=1, refit=True),
    }
    selection = []
    for name, search in candidates.items():
        search.fit(x, y, groups=groups)
        selection.append({"family": name, "training_grouped_cv_auroc": search.best_score_,
                          "parameters": json.dumps(search.best_params_, sort_keys=True)})
        print(selection[-1], flush=True)
    selected = max(selection, key=lambda row: row["training_grouped_cv_auroc"])["family"]
    fitted = candidates[selected].best_estimator_
    dev_score = score(fitted, x_dev)
    clusters = [np.flatnonzero(dev.patient_alias.eq(pid)) for pid in sorted(dev.patient_alias.unique())]
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
             for _ in range(2000)]
    draws = [idx for idx in draws if np.unique(y_dev[idx]).size == 2]
    auc = np.asarray([roc_auc_score(y_dev[idx], dev_score[idx]) for idx in draws])
    ap = np.asarray([average_precision_score(y_dev[idx], dev_score[idx]) for idx in draws])
    result = {"group": f"training_selected_{selected}_meta", "patients": dev.patient_alias.nunique(),
              "kidneys": len(dev), "prevalence": y_dev.mean(), "auroc": roc_auc_score(y_dev, dev_score),
              "ap": average_precision_score(y_dev, dev_score), "auroc_ci_low": np.percentile(auc, 2.5),
              "auroc_ci_high": np.percentile(auc, 97.5), "ap_ci_low": np.percentile(ap, 2.5),
              "ap_ci_high": np.percentile(ap, 97.5)}
    pd.DataFrame(selection).to_csv(OUT / "training_meta_selection.csv", index=False)
    pd.DataFrame([result]).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame({"patient_alias": dev.patient_alias, "kidney_alias": dev.kidney_alias,
                  "side": dev.side, "label": y_dev, "score": dev_score}).to_csv(
                      OUT / "development_predictions.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "inputs": columns, "selection": "5-fold patient-grouped ROC-AUC on training OOF base predictions only",
        "candidate_families": list(candidates), "selected": selected,
        "development_used_for_selection": False, "test_scored": False}, indent=2), encoding="utf-8")
    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
