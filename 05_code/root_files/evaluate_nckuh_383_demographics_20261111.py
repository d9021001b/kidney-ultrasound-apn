"""Fixed development probe of age/sex added to the contour-first image baseline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_nckuh_383_clinical_availability_20261111 import patient_key, sex_code


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
BASE = ROOT / "02_experiments/NCKUH_revised_383_baseline_20261111"
OUT = ROOT / "02_experiments/NCKUH_revised_383_clinical_audit_20261111"


def main():
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve development result")
    book = load_workbook(ROOT / "2004-2019-NCKUH-亞大.xlsx", read_only=True, data_only=True)
    ages = {patient_key(r[0]): float(r[4]) for r in list(book["完整資料彙整"].values)[1:]}
    sexes = {}
    for name in ("2007-2012", "2014-2019", "工作表2"):
        for row in list(book[name].values)[1:]:
            if row[0] is not None and row[3] is not None:
                pid = patient_key(row[0])
                sex = sex_code(row[3])
                assert pid not in sexes or sexes[pid] == sex
                sexes[pid] = sex
    assignments = pd.read_csv(REGISTRY / "patient_assignments.csv")
    linkage = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype=str)
    patients = assignments.merge(linkage, on="patient_alias", validate="one_to_one")
    patients["key"] = patients.patient_id.map(patient_key)
    patients["age_years"] = patients.key.map(ages)
    patients["male"] = patients.key.map(lambda key: int(sexes[key] == "M"))
    assert patients.age_years.notna().all() and len(patients) == 383
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv").merge(
        patients[["patient_alias", "role", "age_years", "male"]], on="patient_alias", validate="many_to_one")
    train = kidneys.role.eq("train").to_numpy()
    dev = kidneys.role.eq("development").to_numpy()
    x = kidneys[["age_years", "male"]].to_numpy(float)
    y = kidneys.kidney_APN.to_numpy(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced",
                          solver="liblinear", random_state=20261111))
    model.fit(x[train], y[train])
    demographics = model.predict_proba(x[dev])[:, 1]
    baseline = pd.read_csv(BASE / "development_predictions.csv")
    baseline = baseline.loc[baseline.group.eq("fixed_mean")].set_index("kidney_alias")
    dev_kidneys = kidneys.loc[dev].reset_index(drop=True)
    baseline = baseline.reindex(dev_kidneys.kidney_alias)
    assert baseline.score.notna().all()
    assert np.array_equal(baseline.label.to_numpy(int), y[dev])
    image = baseline.score.to_numpy(float)
    scores = {"image_fixed_mean": image, "age_sex": demographics,
              "image_plus_age_sex_equal_mean": (image + demographics) / 2}
    rng = np.random.default_rng(20261111)
    groups = [np.flatnonzero(dev_kidneys.patient_alias.eq(pid))
              for pid in sorted(dev_kidneys.patient_alias.unique())]
    results = []
    predictions = []
    for group, score in scores.items():
        auc = roc_auc_score(y[dev], score)
        ap = average_precision_score(y[dev], score)
        intervals = []
        for _ in range(2000):
            idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            if np.unique(y[dev][idx]).size == 2:
                intervals.append((roc_auc_score(y[dev][idx], score[idx]),
                                  average_precision_score(y[dev][idx], score[idx])))
        bounds = np.percentile(intervals, [2.5, 97.5], axis=0)
        results.append({"group": group, "patients": len(groups), "kidneys": len(score),
                        "auroc": auc, "ap": ap, "auroc_ci_low": bounds[0, 0],
                        "auroc_ci_high": bounds[1, 0], "ap_ci_low": bounds[0, 1],
                        "ap_ci_high": bounds[1, 1]})
        predictions.extend({"group": group, "patient_alias": row.patient_alias,
                            "kidney_alias": row.kidney_alias, "label": int(label),
                            "score": float(value)}
                           for row, label, value in zip(dev_kidneys.itertuples(), y[dev], score))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "purpose": "fixed age/sex availability and incremental development probe",
        "unit": "kidney, patient-cluster bootstrap",
        "inputs": "age years and sex from source sheets; DMSA fields excluded",
        "model": "train-only standardized balanced logistic regression C=1",
        "fusion": "fixed 0.5 image baseline + 0.5 demographic probability",
        "test_scored": False,
        "limitation": "Age and sex are patient-level and cannot localize unilateral APN; development is exploratory."}, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
