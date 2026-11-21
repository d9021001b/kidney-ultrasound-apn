"""Side-normalized deidentified TF-IDF probe of templated renal ultrasound reports."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from evaluate_nckuh_383_echo_report_20261112 import patient_key, report_mapping


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
STACK = ROOT / "02_experiments/NCKUH_revised_383_oof_hierarchy_stacking_20261112"
OUT = ROOT / "02_experiments/NCKUH_revised_383_echo_tfidf_20261112"
SEED = 20261112


def normalize_report(text, side):
    text = str(text).lower()
    text = re.split(r"\bexam(?:er|iner)\s*:", text)[0]
    own, other = ("right", "left") if side == "right" else ("left", "right")
    text = re.sub(rf"\b{own}\b", " ownside ", text)
    text = re.sub(rf"\b{other}\b", " otherside ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " num ", text)
    text = re.sub(r"[^a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve echo TF-IDF experiment")
    linkage = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype=str)
    aliases = {patient_key(row.patient_id): row.patient_alias for row in linkage.itertuples(index=False)}
    reports = {aliases[key]: value[1] for key, value in report_mapping().items() if key in aliases}
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv")
    kidneys = kidneys.merge(roles[["patient_alias", "role"]], on="patient_alias", validate="many_to_one")
    available = kidneys[kidneys.patient_alias.isin(reports)].copy().reset_index(drop=True)
    text = np.asarray([normalize_report(reports[row.patient_alias], row.side)
                       for row in available.itertuples(index=False)], dtype=object)
    labels = available.kidney_APN.to_numpy(int); groups = available.patient_alias.to_numpy()
    train = np.flatnonzero(available.role.eq("train")); dev = np.flatnonzero(available.role.eq("development"))
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=2500,
                                 sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 max_features=3500, sublinear_tf=True)),
    ])
    pipeline = Pipeline([("features", features), ("lr", LogisticRegression(
        class_weight="balanced", solver="liblinear", max_iter=4000, random_state=SEED))])
    search = GridSearchCV(pipeline, {"lr__C": [.01, .1, 1, 10]}, scoring="roc_auc",
                          cv=StratifiedGroupKFold(4, shuffle=True, random_state=SEED),
                          n_jobs=1, refit=True)
    search.fit(text[train], labels[train], groups=groups[train])
    report_score = search.predict_proba(text[dev])[:, 1]
    target = available.iloc[dev].reset_index(drop=True)
    stack = pd.read_csv(STACK / "development_predictions.csv").set_index("kidney_alias")
    hybrid = stack.score.to_numpy(float).copy(); full_labels = stack.label.to_numpy(int)
    lookup = {kidney: i for i, kidney in enumerate(stack.index)}
    for row, value in zip(target.itertuples(index=False), report_score):
        i = lookup[row.kidney_alias]; hybrid[i] = (hybrid[i] + value) / 2
    results = []
    for name, y, score, patients in (
        ("side_normalized_echo_tfidf_subset", labels[dev], report_score, target.patient_alias),
        ("fixed_stack_plus_tfidf_when_available", full_labels, hybrid, stack.patient_alias),
    ):
        clusters = [np.flatnonzero(np.asarray(patients) == pid) for pid in sorted(set(patients))]
        rng = np.random.default_rng(SEED)
        draws = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))]) for _ in range(3000)]
        draws = [idx for idx in draws if np.unique(y[idx]).size == 2]
        auc = np.asarray([roc_auc_score(y[idx], score[idx]) for idx in draws])
        ap = np.asarray([average_precision_score(y[idx], score[idx]) for idx in draws])
        results.append({"group": name, "patients": len(set(patients)), "kidneys": len(y),
                        "auroc": roc_auc_score(y, score), "ap": average_precision_score(y, score),
                        "auroc_ci_low": np.percentile(auc, 2.5), "auroc_ci_high": np.percentile(auc, 97.5),
                        "ap_ci_low": np.percentile(ap, 2.5), "ap_ci_high": np.percentile(ap, 97.5)})
    pd.DataFrame(results).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame({"patient_alias": target.patient_alias, "kidney_alias": target.kidney_alias,
                  "side": target.side, "label": labels[dev], "score": report_score}).to_csv(
                      OUT / "report_subset_predictions.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "deidentification": "examiner tail removed; target/opposite side replaced with ownside/otherside; all numbers replaced by num",
        "features": "word 1-2 grams plus character 3-5 grams, training-only vocabulary",
        "selection": "C selected by 4-fold patient-grouped training-report AUROC",
        "hybrid": "fixed equal mean with five-source OOF image stack where report exists",
        "test_scored": False,
        "limitation": "33 train and 11 development patients with reports; timing relative to DMSA and nonrandom availability remain unresolved."}, indent=2), encoding="utf-8")
    print(search.best_score_, search.best_params_); print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
