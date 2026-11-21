"""Validate, unlock, and development-evaluate blinded full-cohort visibility review."""

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
EXPERIMENT = ROOT / "02_experiments/NCKUH_fullcohort_visibility_annotation_20261103"
DEFAULT_WORKBOOK = EXPERIMENT / "blinded_packet/annotation_form_clinician_fullcohort_20261107.xlsx"
PRIVATE_KEY = EXPERIMENT / "private_key/case_to_DMSA_key_PRIVATE.csv"
PRIVATE_OUTPUT = EXPERIMENT / "returned_private"
STATUS = EXPERIMENT / "clinician_annotation_return_status_20261107.json"
MODEL_OUT = ROOT / "02_experiments/NCKUH_DMSA_visibility_clinician_features_20261107"
REFERENCE = uri.OUT / "development_predictions.csv"
REFERENCE_GROUP = "official_uri_embedding_folds_quarter_plus_current"
SEEDS = tuple(base.SEEDS)

EXPECTED_COLUMNS = [
    "case_id", "kidney_side", "available_views", "image_readable",
    "laterality_confirmed", "apn_compatible_finding", "primary_echo_pattern",
    "abnormal_region", "wedge_shaped_focus", "renal_enlargement",
    "poor_corticomedullary_diff", "pelvic_dilation", "perfusion_defect",
    "artifact_obscuration", "confidence_1_to_5", "reviewer_notes",
]
CATEGORICAL_FEATURES = [
    "available_views", "image_readable", "laterality_confirmed",
    "apn_compatible_finding", "primary_echo_pattern", "abnormal_region",
    "wedge_shaped_focus", "renal_enlargement", "poor_corticomedullary_diff",
    "pelvic_dilation", "perfusion_defect", "artifact_obscuration",
]
NUMERIC_FEATURES = ["confidence_1_to_5"]
ALLOWED = {
    "image_readable": {"yes", "no", "uncertain", "not imaged"},
    "laterality_confirmed": {"yes", "no", "uncertain", "not imaged"},
    "apn_compatible_finding": {"yes", "no", "uncertain", "not assessable"},
    "primary_echo_pattern": {
        "none", "hyperechoic", "hypoechoic", "mixed", "heterogeneous",
        "uncertain", "not assessable",
    },
    "abnormal_region": {
        "upper pole", "middle", "lower pole", "multifocal", "diffuse",
        "none", "uncertain", "not assessable",
    },
    "wedge_shaped_focus": {"yes", "no", "uncertain", "not assessable"},
    "renal_enlargement": {"yes", "no", "uncertain", "not assessable"},
    "poor_corticomedullary_diff": {"yes", "no", "uncertain", "not assessable"},
    "pelvic_dilation": {"yes", "no", "uncertain", "not assessable"},
    "perfusion_defect": {"yes", "no", "uncertain", "not shown", "not assessable"},
    "artifact_obscuration": {"yes", "no", "uncertain", "not assessable"},
}
STATE_CODE = hierarchy.STATE_CODE
REGULARIZATION = (0.01, 0.1, 1.0, 10.0)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def normalize(value) -> str:
    return "" if value is None else str(value).strip().lower()


def read_review(path: Path) -> tuple[pd.DataFrame, str, str]:
    workbook = load_workbook(path, data_only=True, read_only=False)
    if workbook.sheetnames[:2] != ["腎臟標註", "填寫說明"]:
        raise ValueError("Unexpected clinician workbook sheets")
    sheet = workbook["腎臟標註"]
    headers = [str(sheet.cell(4, column).value).split("\n", 1)[0] for column in range(1, 17)]
    if headers != EXPECTED_COLUMNS:
        raise ValueError("Unexpected clinician workbook columns")
    rows = [[sheet.cell(row, column).value for column in range(1, 17)] for row in range(5, 425)]
    frame = pd.DataFrame(rows, columns=headers)
    expected = [(f"V{case:03d}", side) for case in range(1, 211) for side in ("left", "right")]
    if list(frame[["case_id", "kidney_side"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("The 420 blinded kidney identifiers changed")
    return frame, normalize(sheet["B2"].value), normalize(sheet["D2"].value)


def completion(frame: pd.DataFrame) -> dict:
    required = [column for column in EXPECTED_COLUMNS[3:14]]
    complete = frame[required].apply(lambda column: column.map(normalize).ne("")).all(axis=1)
    confidence_present = frame.confidence_1_to_5.map(normalize).ne("")
    confidence_required = frame.available_views.map(normalize).ne("none")
    complete &= confidence_present | ~confidence_required
    return {
        "kidney_rows": len(frame),
        "complete_required_rows": int(complete.sum()),
        "remaining_required_rows": int((~complete).sum()),
    }


def validate_completed(frame: pd.DataFrame, reviewer: str, review_date: str) -> pd.DataFrame:
    if not reviewer or not review_date:
        raise ValueError("Reviewer code and review date are required before finalization")
    normalized = frame.copy()
    for column in EXPECTED_COLUMNS[3:14] + ["reviewer_notes"]:
        normalized[column] = normalized[column].map(normalize)
    for column, values in ALLOWED.items():
        invalid = ~normalized[column].isin(values)
        if invalid.any():
            bad = normalized.loc[invalid, ["case_id", "kidney_side"]].to_dict("records")
            raise ValueError(f"Invalid or missing {column}: {bad}")
    confidence = pd.to_numeric(normalized.confidence_1_to_5, errors="coerce")
    imaged = normalized.available_views.ne("none")
    invalid_confidence = imaged & ~confidence.between(1, 5, inclusive="both")
    if invalid_confidence.any():
        bad = normalized.loc[invalid_confidence, ["case_id", "kidney_side"]].to_dict("records")
        raise ValueError(f"Invalid or missing confidence: {bad}")
    normalized["confidence_1_to_5"] = confidence

    unavailable = ~imaged
    unavailable_expected = (
        normalized.image_readable.eq("not imaged")
        & normalized.laterality_confirmed.eq("not imaged")
        & normalized.apn_compatible_finding.eq("not assessable")
    )
    if (unavailable & ~unavailable_expected).any():
        raise ValueError("Rows with no available view must be marked not imaged/not assessable")
    if (imaged & normalized.image_readable.eq("not imaged")).any():
        raise ValueError("Imaged rows cannot be marked not imaged")

    visible = normalized.apn_compatible_finding.eq("yes")
    contradictory = visible & (
        normalized.primary_echo_pattern.eq("none") | normalized.abnormal_region.eq("none")
    )
    if contradictory.any():
        bad = normalized.loc[contradictory, ["case_id", "kidney_side"]].to_dict("records")
        raise ValueError(f"Visible APN-compatible finding cannot use none: {bad}")
    return normalized


def finalize(frame: pd.DataFrame, reviewer: str, review_date: str, source: Path) -> dict:
    validated = validate_completed(frame, reviewer, review_date)
    key = pd.read_csv(PRIVATE_KEY, dtype=str)
    if set(key.columns) != {"case_id", "patient_alias", "DMSA_APN"}:
        raise ValueError("Unexpected private linkage key")
    if len(key) != 210 or key.case_id.nunique() != 210:
        raise ValueError("Private linkage key is incomplete")
    linked = validated.merge(key, on="case_id", how="left", validate="many_to_one")
    if linked.patient_alias.isna().any():
        raise ValueError("Unlinked blinded cases")
    PRIVATE_OUTPUT.mkdir(parents=True, exist_ok=True)
    output = PRIVATE_OUTPUT / "fullcohort_visibility_annotations_PRIVATE.csv"
    linked.to_csv(output, index=False)
    audit = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_development_only_modeling",
        "source_workbook_sha256": digest(source),
        "reviewer_code_recorded": True,
        "review_date_recorded": True,
        "cases": int(linked.case_id.nunique()),
        "kidney_rows": len(linked),
        "DMSA_key_opened_only_after_completion": True,
        "annotation_output_sha256": digest(output),
    }
    (PRIVATE_OUTPUT / "fullcohort_ingestion_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


def make_model(c_value: float, seed: int) -> Pipeline:
    preprocess = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=seed,
        )),
    ])


def direct_visibility_score(frame: pd.DataFrame) -> np.ndarray:
    visible = frame.apn_compatible_finding.map({
        "yes": 1.0, "no": 0.0, "uncertain": 0.5, "not assessable": 0.5,
    }).to_numpy(float)
    confidence = frame.confidence_1_to_5.fillna(1).to_numpy(float)
    return 0.5 + (visible - 0.5) * (confidence / 5.0)


def select_regularization(
    frame: pd.DataFrame,
    patients: pd.DataFrame,
    train: np.ndarray,
    states: np.ndarray,
    seed: int,
) -> tuple[Pipeline, dict, list[dict]]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 1107)
    records = []
    for c_value in REGULARIZATION:
        metrics = []
        for fold, (local_fit, local_validation) in enumerate(splitter.split(train, states[train])):
            fit_aliases = set(patients.patient_alias.iloc[train[local_fit]])
            validation_aliases = set(patients.patient_alias.iloc[train[local_validation]])
            fit = frame.patient_alias.isin(fit_aliases)
            validation = frame.patient_alias.isin(validation_aliases)
            model = make_model(c_value, seed + fold)
            model.fit(frame.loc[fit], frame.loc[fit, "kidney_APN"])
            score = model.predict_proba(frame.loc[validation])[:, 1]
            metrics.append(hierarchy.metric(frame.loc[validation, "kidney_APN"], score))
        auroc = float(np.mean([row["auroc"] for row in metrics]))
        ap = float(np.mean([row["ap"] for row in metrics]))
        records.append({"C": c_value, "inner_auroc": auroc, "inner_ap": ap, "objective": (auroc + ap) / 2})
    selection = pd.DataFrame(records).sort_values(
        ["objective", "inner_auroc", "C"], ascending=[False, False, True]
    ).iloc[0].to_dict()
    aliases = set(patients.patient_alias.iloc[train])
    selected = frame.patient_alias.isin(aliases)
    model = make_model(float(selection["C"]), seed)
    model.fit(frame.loc[selected], frame.loc[selected, "kidney_APN"])
    return model, selection, records


def evaluate() -> None:
    source = PRIVATE_OUTPUT / "fullcohort_visibility_annotations_PRIVATE.csv"
    if not source.exists():
        raise FileNotFoundError("Finalize a complete blinded review before evaluation")
    if MODEL_OUT.exists():
        raise FileExistsError(f"Preserve completed or partial experiment: {MODEL_OUT}")
    MODEL_OUT.mkdir(parents=True)
    (MODEL_OUT / "models").mkdir()
    annotations = pd.read_csv(source).fillna("")
    patients, kidneys, _ = cohort.load_reference()
    frame = kidneys[["patient_alias", "side", "kidney_alias", "kidney_APN", "DMSA_APN"]].merge(
        annotations,
        left_on=["patient_alias", "side"],
        right_on=["patient_alias", "kidney_side"],
        how="left",
        validate="one_to_one",
    )
    if frame.case_id.isna().any() or len(frame) != 414:
        raise ValueError("Clinician annotations did not link to all 414 imaged kidneys")
    frame["confidence_1_to_5"] = pd.to_numeric(frame.confidence_1_to_5, errors="coerce")
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    states = patients.DMSA_APN.map(STATE_CODE).to_numpy(int)
    reference = pd.read_csv(REFERENCE)
    reference = reference[reference.group.eq(REFERENCE_GROUP)].set_index(
        ["seed", "kidney_alias"], verify_integrity=True,
    )
    predictions, metrics_rows, selections, grids = [], [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        patient_roles = patients.patient_alias.map(mapping)
        kidney_roles = frame.patient_alias.map(mapping)
        train = np.flatnonzero(patient_roles.eq("train"))
        development = kidney_roles.eq("development").to_numpy()
        target = frame.loc[development]
        model, selection, records = select_regularization(frame, patients, train, states, seed)
        learned = model.predict_proba(target)[:, 1]
        direct = direct_visibility_score(target)
        fixed = reference.loc[
            [(seed, alias) for alias in target.kidney_alias], "score"
        ].to_numpy(float)
        draw_scores = {
            "official_uri_fixed_reference": fixed,
            "clinician_direct_visibility": direct,
            "clinician_learned_features": learned,
            "clinician_learned_10pct_plus_uri": 0.10 * learned + 0.90 * fixed,
            "clinician_learned_20pct_plus_uri": 0.20 * learned + 0.80 * fixed,
            "clinician_learned_25pct_plus_uri": 0.25 * learned + 0.75 * fixed,
            "clinician_learned_50pct_plus_uri": 0.50 * learned + 0.50 * fixed,
        }
        joblib.dump(model, MODEL_OUT / "models" / f"{seed}_visibility.joblib", compress=3)
        selections.append({"seed": seed, **selection})
        grids.extend({"seed": seed, **record} for record in records)
        for group, score in draw_scores.items():
            metric = hierarchy.metric(target.kidney_APN.to_numpy(int), score)
            metrics_rows.append({
                "seed": seed, "role": "development", "group": group,
                "patients": target.patient_alias.nunique(), "kidneys": len(target), **metric,
            })
            for row, value in zip(target.itertuples(index=False), score):
                predictions.append({
                    "seed": seed, "group": group, "patient_alias": row.patient_alias,
                    "kidney_alias": row.kidney_alias, "DMSA_APN": row.DMSA_APN_x,
                    "side": row.side, "label": int(row.kidney_APN), "score": float(value),
                })
    metrics = pd.DataFrame(metrics_rows)
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
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups, SEEDS, "official_uri_fixed_reference",
    )
    ci.to_csv(MODEL_OUT / "development_ci.csv", index=False)
    delta.to_csv(MODEL_OUT / "paired_delta_ci.csv", index=False)
    np.save(MODEL_OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    (MODEL_OUT / "decision.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "groups_above_both_development_targets": achieved,
        "promotion": bool(achieved),
        "exploratory_test_rows_scored": 0,
        "selection": "Four-fold patient-level inner CV within each 126-patient training role",
        "annotation_blinding": "Clinician workbook completed before DMSA linkage",
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
    frame, reviewer, review_date = read_review(args.workbook)
    state = completion(frame)
    if args.finalize:
        if state["remaining_required_rows"]:
            raise SystemExit(
                f"Clinician review incomplete: {state['remaining_required_rows']} kidney rows remain"
            )
        state = finalize(frame, reviewer, review_date, args.workbook)
    else:
        state.update({
            "checked_utc": datetime.now(timezone.utc).isoformat(),
            "status": "ready_to_finalize" if state["remaining_required_rows"] == 0 else "awaiting_clinician_review",
            "source_workbook_sha256": digest(args.workbook),
            "DMSA_values_read": False,
            "private_linkage_key_read": False,
        })
        STATUS.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
