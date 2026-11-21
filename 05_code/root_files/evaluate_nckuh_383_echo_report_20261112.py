"""Exploratory side-specific structured features from available renal ultrasound reports."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_nckuh_383_clinical_availability_20261111 import patient_key


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
STACK = ROOT / "02_experiments/NCKUH_revised_383_oof_hierarchy_stacking_20261112"
OUT = ROOT / "02_experiments/NCKUH_revised_383_echo_report_20261112"
SEED = 20261112


def date(value):
    return pd.to_datetime(value, errors="coerce")


def fragments(text):
    cleaned = re.sub(r"\s+", " ", str(text).lower()).strip()
    return [part.strip() for part in re.split(r"(?=\b\d+\s*[.)])|[;\n\r]+", cleaned) if part.strip()]


def applies(fragment, side):
    return side in fragment or "bilateral" in fragment or "both renal" in fragment or "both kidney" in fragment


def side_features(text, side):
    parts = fragments(text)
    windows = [part if i == 0 else parts[i - 1] + " " + part for i, part in enumerate(parts)]
    relevant = [part for part in parts if applies(part, side)]
    explicit = [window for window in windows if applies(window, side) and
                ("apn" in window or "pyelonephritis" in window)]
    abnormal_density = [part for part in relevant if ("density" in part or "echogenic" in part) and
                        any(term in part for term in ("increased", "decreased", "abnormal", "hetero"))]
    swelling = [part for part in relevant if any(term in part for term in
                ("swelling", "enlarged", "enlargement", "increasement"))]
    pelvic = [part for part in relevant if any(term in part for term in
              ("hydroneph", "dilat", "pelvictasia", "pelviectasia"))]
    focal = [part for part in relevant if any(term in part for term in
             ("upper pole", "lower pole", "upper moiety", "lower moiety", "focal"))]
    cortical_abnormal = [part for part in relevant if "cortical thickness" in part and
                         "within normal" not in part and "normal limit" not in part]
    pattern = rf"{side}\s+renal\s+size\s*[:is]*\s*([0-9]+(?:\.[0-9]+)?)\s*cm\s*[x×]\s*([0-9]+(?:\.[0-9]+)?)"
    match = re.search(pattern, re.sub(r"\s+", " ", str(text).lower()))
    length, width = (float(match.group(1)), float(match.group(2))) if match else (np.nan, np.nan)
    return np.asarray([bool(explicit), bool(abnormal_density), bool(swelling), bool(pelvic),
                       bool(focal), bool(cortical_abnormal), length, width,
                       length * width if np.isfinite(length * width) else np.nan], dtype=np.float32)


def report_mapping():
    book = load_workbook(ROOT / "2004-2019-NCKUH-亞大.xlsx", read_only=True, data_only=True)
    roster = {patient_key(row[0]): date(row[3]) for row in list(book["完整資料彙整"].values)[1:] if row[0] is not None}
    candidates = {}
    for row in list(book["成大51位DMSA報告"].values)[1:]:
        if row[0] is None or row[3] is None:
            continue
        key = patient_key(row[0])
        if key not in roster:
            continue
        gap = abs((date(row[2]) - roster[key]).days) if not pd.isna(date(row[2])) and not pd.isna(roster[key]) else 999999
        if key not in candidates or gap < candidates[key][0]:
            candidates[key] = (gap, str(row[3]))
    return candidates


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve report experiment")
    reports = report_mapping()
    linkage = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype=str)
    key_to_alias = {patient_key(row.patient_id): row.patient_alias for row in linkage.itertuples(index=False)}
    by_alias = {key_to_alias[key]: payload for key, (_, payload) in reports.items() if key in key_to_alias}
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv")
    kidneys = kidneys.merge(roles[["patient_alias", "role"]], on="patient_alias", validate="many_to_one")
    available = kidneys[kidneys.patient_alias.isin(by_alias)].copy().reset_index(drop=True)
    features = np.stack([side_features(by_alias[row.patient_alias], row.side) for row in available.itertuples(index=False)])
    labels = available.kidney_APN.to_numpy(int); groups = available.patient_alias.to_numpy()
    train = np.flatnonzero(available.role.eq("train")); dev = np.flatnonzero(available.role.eq("development"))
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                          LogisticRegression(class_weight="balanced", solver="liblinear",
                                             max_iter=3000, random_state=SEED))
    search = GridSearchCV(model, {"logisticregression__C": [.01, .1, 1, 10]}, scoring="roc_auc",
                          cv=StratifiedGroupKFold(4, shuffle=True, random_state=SEED), n_jobs=1, refit=True)
    search.fit(features[train], labels[train], groups=groups[train])
    report_score = search.predict_proba(features[dev])[:, 1]
    report_target = available.iloc[dev].reset_index(drop=True)
    stack = pd.read_csv(STACK / "development_predictions.csv").set_index("kidney_alias")
    stack_score = stack.score.to_numpy(float); full_labels = stack.label.to_numpy(int)
    hybrid = stack_score.copy()
    report_index = {kidney: i for i, kidney in enumerate(stack.index)}
    for row, value in zip(report_target.itertuples(index=False), report_score):
        index = report_index[row.kidney_alias]
        hybrid[index] = (hybrid[index] + value) / 2
    metrics = []
    for name, y, score, patients in (
        ("echo_report_available_subset", labels[dev], report_score, report_target.patient_alias),
        ("fixed_stack_plus_echo_when_available", full_labels, hybrid, stack.patient_alias),
    ):
        clusters = [np.flatnonzero(np.asarray(patients) == pid) for pid in sorted(set(patients))]
        rng = np.random.default_rng(SEED)
        draws = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))]) for _ in range(3000)]
        draws = [idx for idx in draws if np.unique(y[idx]).size == 2]
        auc = np.asarray([roc_auc_score(y[idx], score[idx]) for idx in draws])
        ap = np.asarray([average_precision_score(y[idx], score[idx]) for idx in draws])
        metrics.append({"group": name, "patients": len(set(patients)), "kidneys": len(y),
                        "prevalence": y.mean(), "auroc": roc_auc_score(y, score),
                        "ap": average_precision_score(y, score),
                        "auroc_ci_low": np.percentile(auc, 2.5), "auroc_ci_high": np.percentile(auc, 97.5),
                        "ap_ci_low": np.percentile(ap, 2.5), "ap_ci_high": np.percentile(ap, 97.5)})
    pd.DataFrame(metrics).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame({"patient_alias": report_target.patient_alias, "kidney_alias": report_target.kidney_alias,
                  "side": report_target.side, "label": labels[dev], "score": report_score}).to_csv(
                      OUT / "report_subset_predictions.csv", index=False)
    pd.DataFrame({"patient_alias": stack.patient_alias, "kidney_alias": stack.index,
                  "label": full_labels, "score": hybrid}).to_csv(OUT / "hybrid_predictions.csv", index=False)
    feature_counts = {"reports_mapped": len(by_alias), "contour_ready_report_patients": available.patient_alias.nunique(),
                      "train_report_patients": available.loc[train, "patient_alias"].nunique(),
                      "development_report_patients": report_target.patient_alias.nunique(),
                      "explicit_apn_positive_kidneys": int(features[:, 0].sum()),
                      "abnormal_density_kidneys": int(features[:, 1].sum()),
                      "swelling_kidneys": int(features[:, 2].sum()),
                      "size_measurement_kidneys": int(np.isfinite(features[:, 6]).sum())}
    (OUT / "audit.json").write_text(json.dumps(feature_counts, indent=2), encoding="utf-8")
    (OUT / "protocol.json").write_text(json.dumps({
        "input": "renal ultrasound echo report only; side-specific rule-based features; no DMSA columns used in parsing",
        "features": ["explicit APN wording", "parenchymal density abnormality", "swelling/enlargement",
                     "pelvic dilation", "focal/polar wording", "cortical abnormality", "reported length/width/area"],
        "selection": "training report patients only, 4-fold patient-grouped CV",
        "hybrid": "fixed equal mean with OOF hierarchy stack only where a report is available",
        "test_scored": False,
        "limitation": "Report timing relative to DMSA is not established; only a small nonrandom subset has reports. Not eligible as a main-result model until timing and cohort selection are verified."}, indent=2), encoding="utf-8")
    print(feature_counts); print(search.best_score_, search.best_params_); print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
