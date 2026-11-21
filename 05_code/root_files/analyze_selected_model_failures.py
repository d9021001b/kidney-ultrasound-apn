from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "CYCH_selected_pipeline_ablation_20260914" / "development_predictions_deidentified.csv"
OUTPUT = ROOT / "CYCH_selected_failure_analysis_20260915"

FINAL = "FINAL-EQUAL-VOTE"
COMPONENTS = {
    "graph_score": "RELIABLE-NEGATIVE-GCN-MIL",
    "radiomics_score": "PYRADIOMICS",
    "directional_i3_score": "DIRECTIONAL-WAVELET-GLRLM-I3",
}
THRESHOLD = 0.50


def classify(label: pd.Series, score: pd.Series) -> pd.Series:
    prediction = (score >= THRESHOLD).astype(int)
    return pd.Series(
        np.select(
            [
                (label == 0) & (prediction == 0),
                (label == 0) & (prediction == 1),
                (label == 1) & (prediction == 0),
            ],
            ["true_negative", "false_positive", "false_negative"],
            default="true_positive",
        ),
        index=label.index,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(INPUT)
    source = source[source["role"] == "development_validation"].copy()

    final = source[source["group"] == FINAL][
        ["seed", "patient_alias", "label", "score"]
    ].rename(columns={"score": "final_score"})

    for column, group in COMPONENTS.items():
        component = source[source["group"] == group][
            ["seed", "patient_alias", "label", "score"]
        ].rename(columns={"score": column})
        final = final.merge(
            component,
            on=["seed", "patient_alias", "label"],
            how="inner",
            validate="one_to_one",
        )

    expected = final[list(COMPONENTS)].mean(axis=1)
    max_reconstruction_error = float(np.max(np.abs(final["final_score"] - expected)))
    if max_reconstruction_error > 1e-12:
        raise AssertionError(f"Final-vote reconstruction error: {max_reconstruction_error}")

    final["outcome"] = classify(final["label"], final["final_score"])
    final["final_error"] = final["outcome"].isin(["false_positive", "false_negative"]).astype(int)
    final["high_confidence_error"] = (
        ((final["label"] == 0) & (final["final_score"] >= 0.75))
        | ((final["label"] == 1) & (final["final_score"] <= 0.25))
    ).astype(int)

    component_error_columns: list[str] = []
    for column in COMPONENTS:
        error_column = column.replace("_score", "_error")
        component_error_columns.append(error_column)
        final[error_column] = ((final[column] >= THRESHOLD).astype(int) != final["label"]).astype(int)

    final["component_error_count"] = final[component_error_columns].sum(axis=1)
    final["unanimous_component_error"] = (final["component_error_count"] == 3).astype(int)
    final["component_score_range"] = final[list(COMPONENTS)].max(axis=1) - final[list(COMPONENTS)].min(axis=1)
    final["fusion_rescue"] = (
        (final["final_error"] == 0) & (final["component_error_count"] >= 2)
    ).astype(int)
    final["final_error_with_any_correct_component"] = (
        (final["final_error"] == 1) & (final["component_error_count"] < 3)
    ).astype(int)

    row_columns = [
        "seed",
        "patient_alias",
        "label",
        "final_score",
        "outcome",
        "graph_score",
        "radiomics_score",
        "directional_i3_score",
        "component_error_count",
        "component_score_range",
        "unanimous_component_error",
        "fusion_rescue",
        "high_confidence_error",
    ]
    final[row_columns].sort_values(["seed", "patient_alias"]).to_csv(
        OUTPUT / "failure_analysis_by_draw.csv", index=False
    )

    patient = final.groupby(["patient_alias", "label"], as_index=False).agg(
        appearance_count=("seed", "size"),
        error_count=("final_error", "sum"),
        mean_final_score=("final_score", "mean"),
        min_final_score=("final_score", "min"),
        max_final_score=("final_score", "max"),
        mean_graph_score=("graph_score", "mean"),
        mean_radiomics_score=("radiomics_score", "mean"),
        mean_directional_i3_score=("directional_i3_score", "mean"),
        unanimous_component_error_count=("unanimous_component_error", "sum"),
        fusion_rescue_count=("fusion_rescue", "sum"),
        high_confidence_error_count=("high_confidence_error", "sum"),
        mean_component_score_range=("component_score_range", "mean"),
    )
    patient["error_rate"] = patient["error_count"] / patient["appearance_count"]
    patient["persistent_error"] = (
        (patient["appearance_count"] >= 2) & (patient["error_rate"] >= 0.75)
    ).astype(int)
    patient = patient.sort_values(
        ["persistent_error", "error_rate", "appearance_count", "patient_alias"],
        ascending=[False, False, False, True],
    )
    patient.to_csv(OUTPUT / "failure_analysis_by_patient.csv", index=False)

    draw = final.groupby("seed", as_index=False).agg(
        patients=("patient_alias", "size"),
        false_positive=("outcome", lambda values: int((values == "false_positive").sum())),
        false_negative=("outcome", lambda values: int((values == "false_negative").sum())),
        errors=("final_error", "sum"),
        unanimous_component_errors=("unanimous_component_error", "sum"),
        fusion_rescues=("fusion_rescue", "sum"),
        mean_component_score_range=("component_score_range", "mean"),
    )
    draw["accuracy"] = 1.0 - draw["errors"] / draw["patients"]
    draw.to_csv(OUTPUT / "failure_analysis_by_seed.csv", index=False)

    error_rows = final[final["final_error"] == 1]
    correct_rows = final[final["final_error"] == 0]
    multi_appearance = patient[patient["appearance_count"] >= 2]
    summary = {
        "audit_date": "2026-09-15",
        "scope": "Deidentified post hoc engineering failure characterization on the ten fixed development draws; no exploratory-test predictions were accessed.",
        "decision_threshold": THRESHOLD,
        "development_rows": int(len(final)),
        "unique_patients_represented": int(final["patient_alias"].nunique()),
        "patients_with_two_or_more_appearances": int(len(multi_appearance)),
        "false_positive_rows": int((final["outcome"] == "false_positive").sum()),
        "false_negative_rows": int((final["outcome"] == "false_negative").sum()),
        "total_error_rows": int(final["final_error"].sum()),
        "row_accuracy": float(1.0 - final["final_error"].mean()),
        "high_confidence_error_rows": int(final["high_confidence_error"].sum()),
        "unanimous_component_error_rows": int(final["unanimous_component_error"].sum()),
        "error_rows_with_any_correct_component": int(
            final["final_error_with_any_correct_component"].sum()
        ),
        "fusion_rescue_rows": int(final["fusion_rescue"].sum()),
        "persistent_error_patients": int(patient["persistent_error"].sum()),
        "mean_component_score_range_error_rows": float(error_rows["component_score_range"].mean()),
        "mean_component_score_range_correct_rows": float(correct_rows["component_score_range"].mean()),
        "max_final_vote_reconstruction_error": max_reconstruction_error,
        "interpretation": (
            "Counts describe repeated development appearances, not independent patients or clinical error rates. "
            "Persistent-error flags are review priorities only and do not establish label error, lesion localization, or subgroup bias."
        ),
    }
    (OUTPUT / "failure_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    persistent = patient[patient["persistent_error"] == 1]
    readme = f"""# Selected-model failure analysis

## Motivation

Area-under-curve summaries do not show whether development errors arise from consensus failure across all three branches, branch disagreement, or fixed-probability fusion. This post hoc engineering audit characterizes those mechanisms without opening the selected candidate's exploratory-test predictions.

## Fixed analysis

- Source: deidentified predictions from the ten prespecified development draws.
- Decision threshold: {THRESHOLD:.2f}, identical to the manuscript's operating-point summary.
- Final score reconstruction: arithmetic mean of reliable-negative graph MIL, PyRadiomics, and directional I3 probabilities; maximum reconstruction error {max_reconstruction_error:.3g}.
- Persistent review flag: at least two development appearances and an error rate of at least 0.75.

## Results

- Development appearances: {len(final)} from {final['patient_alias'].nunique()} unique patients.
- Errors: {summary['total_error_rows']} ({summary['false_positive_rows']} false-positive and {summary['false_negative_rows']} false-negative appearances); row-level accuracy {summary['row_accuracy']:.3f}.
- High-confidence errors: {summary['high_confidence_error_rows']}.
- Errors shared by all three components: {summary['unanimous_component_error_rows']}.
- Final errors despite at least one correct component: {summary['error_rows_with_any_correct_component']}.
- Final-vote rescues despite at least two incorrect component classifications: {summary['fusion_rescue_rows']}.
- Patients meeting the persistent review flag: {len(persistent)} of {len(multi_appearance)} patients observed in at least two development draws.
- Mean component probability range was {summary['mean_component_score_range_error_rows']:.3f} on error rows and {summary['mean_component_score_range_correct_rows']:.3f} on correct rows.

## Interpretation boundary

Each row is a development appearance, and the same patient may recur across draws. These counts therefore characterize split sensitivity and component disagreement; they are not independent error-rate estimates. Alias-level review priorities do not establish label error, inflammatory lesion location, clinical subgroup bias, or diagnostic failure. Clinical adjudication requires source-verified labels, demographics, acquisition details, and blinded reviewer assessment.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
