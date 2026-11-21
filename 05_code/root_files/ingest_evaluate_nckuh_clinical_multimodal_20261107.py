"""Validate blinded NCKUH clinical extraction and evaluate development-only fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
REQUEST = ROOT / "02_experiments/NCKUH_clinical_data_request_20261107"
DEFAULT_WORKBOOK = REQUEST / "NCKUH_210_clinical_data_request_blinded_20261107.xlsx"
STATUS = REQUEST / "clinical_return_status.json"
PRIVATE_OUTPUT = REQUEST / "returned_private"
MODEL_OUT = ROOT / "02_experiments/NCKUH_DMSA_clinical_multimodal_20261107"
REFERENCE = uri.OUT / "development_predictions.csv"
REFERENCE_GROUP = "official_uri_embedding_folds_quarter_plus_current"
SEEDS = tuple(base.SEEDS)
STATE_CODE = hierarchy.STATE_CODE

COLUMNS = [
    "病歷號", "研究代碼", "腎臟超音波日期", "回填狀態", "最高體溫_C", "發燒天數",
    "CRP_mg_L", "WBC_10e9_L", "嗜中性球_%", "尿液WBC_HPF", "白血球酯酶",
    "亞硝酸鹽", "尿液培養陽性", "尿液病原菌", "血液培養陽性", "超音波前抗生素_小時",
    "檢驗距超音波_小時", "左腎_報告疑似APN", "右腎_報告疑似APN",
    "左腎_局部實質異常", "右腎_局部實質異常", "左腎_腎盂擴張", "右腎_腎盂擴張",
    "左腎_Doppler灌流缺損", "右腎_Doppler灌流缺損", "左側VUR分級", "右側VUR分級", "備註",
]
NUMERIC_SOURCE = [
    "最高體溫_C", "發燒天數", "CRP_mg_L", "WBC_10e9_L", "嗜中性球_%",
    "尿液WBC_HPF", "超音波前抗生素_小時", "檢驗距超音波_小時",
]
PATIENT_CATEGORICAL_SOURCE = ["白血球酯酶", "亞硝酸鹽", "尿液培養陽性", "血液培養陽性"]
SIDE_FEATURES = ["report_apn", "parenchymal_abnormality", "pelvic_dilation", "doppler_defect", "vur_grade"]
FEATURE_SETS = {
    "infection_labs": NUMERIC_SOURCE + PATIENT_CATEGORICAL_SOURCE,
    "side_report": SIDE_FEATURES,
    "combined": NUMERIC_SOURCE + PATIENT_CATEGORICAL_SOURCE + SIDE_FEATURES,
}
STATUS_VALUES = {"完成", "無病歷", "時間窗不符"}
ALLOWED = {
    "白血球酯酶": {"陰性", "trace", "1+", "2+", "3+"},
    "亞硝酸鹽": {"陰性", "陽性"},
    "尿液培養陽性": {"否", "是", "污染", "未做"},
    "血液培養陽性": {"否", "是", "污染", "未做"},
    "report_apn": {"否", "是", "不確定", "未描述"},
    "parenchymal_abnormality": {"否", "是", "不確定", "未描述"},
    "pelvic_dilation": {"否", "是", "不確定", "未描述"},
    "doppler_defect": {"否", "是", "不確定", "未做"},
    "vur_grade": {"0", "1", "2", "3", "4", "5", "未知", "未做"},
}
RANGES = {
    "最高體溫_C": (30.0, 45.0), "發燒天數": (0.0, 60.0), "CRP_mg_L": (0.0, 1000.0),
    "WBC_10e9_L": (0.0, 100.0), "嗜中性球_%": (0.0, 100.0), "尿液WBC_HPF": (0.0, 100000.0),
    "超音波前抗生素_小時": (-720.0, 720.0), "檢驗距超音波_小時": (-168.0, 168.0),
}
REGULARIZATION = (0.01, 0.1, 1.0, 10.0)
PENALTIES = ("l1", "l2")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def normalize(value) -> str:
    return "" if value is None else str(value).strip()


def read_request(path: Path) -> pd.DataFrame:
    workbook = load_workbook(path, data_only=True, read_only=False)
    if workbook.sheetnames[:2] != ["資料回填", "欄位定義"]:
        raise ValueError("Unexpected clinical workbook sheets")
    sheet = workbook["資料回填"]
    headers = [normalize(sheet.cell(3, column).value) for column in range(1, 29)]
    if headers != COLUMNS:
        raise ValueError("Unexpected clinical workbook columns")
    rows = [[sheet.cell(row, column).value for column in range(1, 29)] for row in range(4, 214)]
    frame = pd.DataFrame(rows, columns=headers)
    if len(frame) != 210 or frame.研究代碼.duplicated().any() or frame.病歷號.duplicated().any():
        raise ValueError("Expected 210 unique patient rows")
    if not frame.研究代碼.map(normalize).str.fullmatch(r"P\d{3}").all():
        raise ValueError("Unexpected deidentified patient aliases")
    return frame


def completion(frame: pd.DataFrame) -> dict:
    status = frame.回填狀態.map(normalize)
    complete = status.isin(STATUS_VALUES)
    return {
        "patients": len(frame),
        "completed_status_rows": int(complete.sum()),
        "remaining_status_rows": int((~complete).sum()),
        "status_counts": status.value_counts(dropna=False).to_dict(),
    }


def validate_completed(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["回填狀態"] = normalized["回填狀態"].map(normalize)
    invalid_status = ~normalized.回填狀態.isin(STATUS_VALUES)
    if invalid_status.any():
        raise ValueError(f"Incomplete clinical rows: {normalized.loc[invalid_status, '研究代碼'].tolist()}")
    for column, (low, high) in RANGES.items():
        values = pd.to_numeric(normalized[column], errors="coerce")
        supplied = normalized[column].map(normalize).ne("")
        invalid = supplied & (~values.between(low, high, inclusive="both"))
        if invalid.any():
            raise ValueError(f"Out-of-range {column}: {normalized.loc[invalid, '研究代碼'].tolist()}")
        normalized[column] = values
    for column in PATIENT_CATEGORICAL_SOURCE:
        normalized[column] = normalized[column].map(normalize)
        invalid = normalized[column].ne("") & ~normalized[column].isin(ALLOWED[column])
        if invalid.any():
            raise ValueError(f"Invalid {column}: {normalized.loc[invalid, '研究代碼'].tolist()}")
    return normalized


def finalize(frame: pd.DataFrame, source: Path) -> dict:
    validated = validate_completed(frame)
    PRIVATE_OUTPUT.mkdir(parents=True, exist_ok=True)
    output = PRIVATE_OUTPUT / "NCKUH_clinical_return_normalized_PRIVATE.csv"
    validated.to_csv(output, index=False)
    audit = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_development_only_multimodal_modeling",
        "source_workbook_sha256": digest(source),
        "patients": len(validated),
        "DMSA_columns_present": False,
        "unknown_values_preserved_as_missing": True,
        "normalized_output_sha256": digest(output),
    }
    (PRIVATE_OUTPUT / "clinical_ingestion_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


def kidney_feature_frame(clinical: pd.DataFrame, kidneys: pd.DataFrame) -> pd.DataFrame:
    merged = kidneys[["patient_alias", "side", "kidney_alias", "kidney_APN", "DMSA_APN"]].merge(
        clinical,
        left_on="patient_alias",
        right_on="研究代碼",
        how="left",
        validate="many_to_one",
    )
    if merged.研究代碼.isna().any():
        raise ValueError("Clinical return does not cover all labeled patients")
    left = merged.side.eq("left")
    pairs = {
        "report_apn": ("左腎_報告疑似APN", "右腎_報告疑似APN"),
        "parenchymal_abnormality": ("左腎_局部實質異常", "右腎_局部實質異常"),
        "pelvic_dilation": ("左腎_腎盂擴張", "右腎_腎盂擴張"),
        "doppler_defect": ("左腎_Doppler灌流缺損", "右腎_Doppler灌流缺損"),
        "vur_grade": ("左側VUR分級", "右側VUR分級"),
    }
    for target, (left_column, right_column) in pairs.items():
        merged[target] = np.where(left, merged[left_column], merged[right_column])
        merged[target] = pd.Series(merged[target], index=merged.index).map(normalize)
    for column in NUMERIC_SOURCE:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    for column in PATIENT_CATEGORICAL_SOURCE + SIDE_FEATURES:
        merged[column] = merged[column].map(normalize).replace("", "missing")
    return merged


def validate_model_categories(frame: pd.DataFrame) -> None:
    for column in PATIENT_CATEGORICAL_SOURCE + SIDE_FEATURES:
        invalid = ~frame[column].isin(ALLOWED[column] | {"missing"})
        if invalid.any():
            raise ValueError(f"Invalid model category in {column}: {sorted(frame.loc[invalid, column].unique())}")


def make_model(feature_set: str, c_value: float, penalty: str, seed: int) -> Pipeline:
    columns = FEATURE_SETS[feature_set]
    numeric = [column for column in columns if column in NUMERIC_SOURCE]
    categorical = [column for column in columns if column not in NUMERIC_SOURCE]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers)),
        ("model", LogisticRegression(
            C=c_value, l1_ratio=1.0 if penalty == "l1" else 0.0,
            solver="liblinear", class_weight="balanced",
            max_iter=2000, random_state=seed,
        )),
    ])


def direct_report_score(frame: pd.DataFrame) -> np.ndarray:
    mapping = {"是": 1.0, "否": 0.0, "不確定": 0.5, "未描述": 0.5, "未做": 0.5, "missing": 0.5}
    report = frame.report_apn.map(mapping).to_numpy(float)
    parenchymal = frame.parenchymal_abnormality.map(mapping).to_numpy(float)
    doppler = frame.doppler_defect.map(mapping).to_numpy(float)
    return 0.50 * report + 0.30 * parenchymal + 0.20 * doppler


def select_model(
    frame: pd.DataFrame,
    patients: pd.DataFrame,
    train: np.ndarray,
    states: np.ndarray,
    seed: int,
) -> tuple[Pipeline, dict, list[dict]]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 1107)
    records = []
    for feature_set in FEATURE_SETS:
        for penalty in PENALTIES:
            for c_value in REGULARIZATION:
                metrics = []
                for fold, (local_fit, local_validation) in enumerate(splitter.split(train, states[train])):
                    fit_aliases = set(patients.patient_alias.iloc[train[local_fit]])
                    validation_aliases = set(patients.patient_alias.iloc[train[local_validation]])
                    fit = frame.patient_alias.isin(fit_aliases)
                    validation = frame.patient_alias.isin(validation_aliases)
                    model = make_model(feature_set, c_value, penalty, seed + fold)
                    model.fit(frame.loc[fit], frame.loc[fit, "kidney_APN"])
                    score = model.predict_proba(frame.loc[validation])[:, 1]
                    metrics.append(hierarchy.metric(frame.loc[validation, "kidney_APN"], score))
                auroc = float(np.mean([row["auroc"] for row in metrics]))
                ap = float(np.mean([row["ap"] for row in metrics]))
                records.append({
                    "feature_set": feature_set, "penalty": penalty, "C": c_value,
                    "inner_auroc": auroc, "inner_ap": ap, "objective": (auroc + ap) / 2,
                })
    selection = pd.DataFrame(records).sort_values(
        ["objective", "inner_auroc", "feature_set", "penalty", "C"],
        ascending=[False, False, True, True, True],
    ).iloc[0].to_dict()
    aliases = set(patients.patient_alias.iloc[train])
    selected = frame.patient_alias.isin(aliases)
    model = make_model(selection["feature_set"], float(selection["C"]), selection["penalty"], seed)
    model.fit(frame.loc[selected], frame.loc[selected, "kidney_APN"])
    return model, selection, records


def evaluate() -> None:
    source = PRIVATE_OUTPUT / "NCKUH_clinical_return_normalized_PRIVATE.csv"
    if not source.exists():
        raise FileNotFoundError("Finalize a complete blinded clinical return before evaluation")
    if MODEL_OUT.exists():
        raise FileExistsError(f"Preserve completed or partial experiment: {MODEL_OUT}")
    MODEL_OUT.mkdir(parents=True)
    (MODEL_OUT / "models").mkdir()
    clinical = pd.read_csv(source).replace({np.nan: ""})
    patients, kidneys, _ = cohort.load_reference()
    frame = kidney_feature_frame(clinical, kidneys)
    validate_model_categories(frame)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    states = patients.DMSA_APN.map(STATE_CODE).to_numpy(int)
    reference = pd.read_csv(REFERENCE)
    reference = reference[reference.group.eq(REFERENCE_GROUP)].set_index(["seed", "kidney_alias"], verify_integrity=True)
    predictions, metric_rows, selections, grids = [], [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        patient_roles = patients.patient_alias.map(mapping)
        kidney_roles = frame.patient_alias.map(mapping)
        train = np.flatnonzero(patient_roles.eq("train"))
        development = kidney_roles.eq("development").to_numpy()
        target = frame.loc[development]
        model, selection, records = select_model(frame, patients, train, states, seed)
        learned = model.predict_proba(target)[:, 1]
        direct = direct_report_score(target)
        fixed = reference.loc[[(seed, alias) for alias in target.kidney_alias], "score"].to_numpy(float)
        scores = {
            "official_uri_fixed_reference": fixed,
            "clinical_direct_report": direct,
            "clinical_selected": learned,
            "clinical_10pct_plus_uri": 0.10 * learned + 0.90 * fixed,
            "clinical_20pct_plus_uri": 0.20 * learned + 0.80 * fixed,
            "clinical_25pct_plus_uri": 0.25 * learned + 0.75 * fixed,
            "clinical_50pct_plus_uri": 0.50 * learned + 0.50 * fixed,
        }
        joblib.dump(model, MODEL_OUT / "models" / f"{seed}_clinical.joblib", compress=3)
        selections.append({"seed": seed, **selection})
        grids.extend({"seed": seed, **record} for record in records)
        for group, score in scores.items():
            metric = hierarchy.metric(target.kidney_APN.to_numpy(int), score)
            metric_rows.append({"seed": seed, "role": "development", "group": group, "patients": target.patient_alias.nunique(), "kidneys": len(target), **metric})
            predictions.extend({
                "seed": seed, "group": group, "patient_alias": row.patient_alias,
                "kidney_alias": row.kidney_alias, "DMSA_APN": row.DMSA_APN,
                "side": row.side, "label": int(row.kidney_APN), "score": float(value),
            } for row, value in zip(target.itertuples(index=False), score))
    metrics = pd.DataFrame(metric_rows)
    prediction_frame = pd.DataFrame(predictions)
    summary = metrics.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    metrics.to_csv(MODEL_OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(MODEL_OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(MODEL_OUT / "training_inner_selections.csv", index=False)
    pd.DataFrame(grids).to_csv(MODEL_OUT / "training_inner_grid.csv", index=False)
    summary.to_csv(MODEL_OUT / "development_summary.csv")
    groups = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(prediction_frame, groups, SEEDS, "official_uri_fixed_reference")
    ci.to_csv(MODEL_OUT / "development_ci.csv", index=False)
    delta.to_csv(MODEL_OUT / "paired_delta_ci.csv", index=False)
    np.save(MODEL_OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    (MODEL_OUT / "decision.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "groups_above_both_development_targets": achieved,
        "promotion": bool(achieved), "exploratory_test_rows_scored": 0,
        "selection": "Feature family, L1/L2 and C selected by four-fold patient-level CV inside each training role",
        "missingness": "Unknown values preserved; numeric missing indicators and categorical missing level",
    }, indent=2), encoding="utf-8")
    print(summary[["auroc_mean", "ap_mean", "brier_mean"]].to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        evaluate()
        return
    frame = read_request(args.workbook)
    state = completion(frame)
    if args.finalize:
        if state["remaining_status_rows"]:
            raise SystemExit(f"Clinical extraction incomplete: {state['remaining_status_rows']} rows remain")
        state = finalize(frame, args.workbook)
    else:
        state.update({
            "checked_utc": datetime.now(timezone.utc).isoformat(),
            "status": "ready_to_finalize" if state["remaining_status_rows"] == 0 else "awaiting_clinical_extraction",
            "source_workbook_sha256": digest(args.workbook),
            "DMSA_values_read": False,
        })
        STATUS.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
