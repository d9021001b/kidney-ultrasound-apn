"""Equal-patient sensitivity analysis for locked exploratory-test predictions."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CYCH_locked_selected_exploratory_test_20260915"
INPUT = OUT / "exploratory_test_comparison_predictions_deidentified.csv"
FINAL = "FINAL-EQUAL-VOTE"
PAIR = "FIXED-RADIOMICS-RADIMAGENET-PAIR"
GROUPS = [
    "RADIMAGENET-DENSENET-SUMMARY",
    "PYRADIOMICS",
    PAIR,
    "DIRECTIONAL-WAVELET-GLRLM-I3",
    "RELIABLE-NEGATIVE-GCN-MIL",
    FINAL,
]
SEEDS = list(range(20260913, 20260923))
BOOTSTRAP_SEED = 20260915
N_BOOTSTRAP = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def load_and_validate() -> pd.DataFrame:
    predictions = pd.read_csv(INPUT)
    required = {"seed", "role", "group", "patient_alias", "label", "score"}
    if not required.issubset(predictions.columns):
        raise RuntimeError(f"Missing columns: {sorted(required - set(predictions.columns))}")
    predictions = predictions.loc[predictions.group.isin(GROUPS)].copy()
    if len(predictions) != 1200:
        raise RuntimeError(f"Expected 1200 fixed prediction rows, found {len(predictions)}")
    if predictions.duplicated(["seed", "group", "patient_alias"]).any():
        raise RuntimeError("Duplicate seed/group/patient rows")
    if set(predictions.seed) != set(SEEDS):
        raise RuntimeError("Unexpected fixed draw seeds")
    label_check = predictions.groupby("patient_alias").label.nunique()
    if not label_check.eq(1).all():
        raise RuntimeError("Patient aliases have inconsistent labels")

    expected_aliases: dict[int, set[str]] = {}
    for seed in SEEDS:
        current = predictions.loc[predictions.seed.eq(seed)]
        for group in GROUPS:
            subset = current.loc[current.group.eq(group)]
            if len(subset) != 20 or subset.patient_alias.nunique() != 20:
                raise RuntimeError(f"Invalid rows for {seed}/{group}")
            aliases = set(subset.patient_alias)
            if seed not in expected_aliases:
                expected_aliases[seed] = aliases
            elif aliases != expected_aliases[seed]:
                raise RuntimeError(f"Fixed groups are not patient-paired for seed {seed}")
    return predictions


def collapse_to_equal_patients(predictions: pd.DataFrame) -> pd.DataFrame:
    collapsed = (
        predictions.groupby(["group", "patient_alias", "label"], as_index=False)
        .agg(score=("score", "mean"), test_draw_appearances=("seed", "nunique"))
        .sort_values(["group", "patient_alias"])
        .reset_index(drop=True)
    )
    for group in GROUPS:
        subset = collapsed.loc[collapsed.group.eq(group)]
        if len(subset) != 87 or subset.label.value_counts().to_dict() != {1: 46, 0: 41}:
            raise RuntimeError(f"Unexpected unique-patient coverage for {group}")
        if (
            int(subset.test_draw_appearances.min()) != 1
            or float(subset.test_draw_appearances.median()) != 2.0
            or int(subset.test_draw_appearances.max()) != 5
        ):
            raise RuntimeError(f"Unexpected appearance-count distribution for {group}")
    return collapsed


def weighted_metrics(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, float]:
    if weights is None:
        weights = np.ones(len(labels), dtype=float)
    keep = weights > 0
    y = labels[keep]
    s = scores[keep]
    w = weights[keep]
    if len(np.unique(y)) != 2:
        raise ValueError("Resample lacks one outcome class")
    precision, recall, _ = precision_recall_curve(y, s, sample_weight=w)
    return {
        "auroc": float(roc_auc_score(y, s, sample_weight=w)),
        "average_precision": float(average_precision_score(y, s, sample_weight=w)),
        "auprc_trapezoid": float(abs(np.trapezoid(precision, recall))),
        "brier": float(np.average((y - s) ** 2, weights=w)),
    }


def summarize_groups(collapsed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        subset = collapsed.loc[collapsed.group.eq(group)]
        rows.append(
            {
                "group": group,
                "unique_patients": len(subset),
                "positive_patients": int(subset.label.sum()),
                "negative_patients": int((1 - subset.label).sum()),
                "test_draw_appearances_min": int(subset.test_draw_appearances.min()),
                "test_draw_appearances_median": float(subset.test_draw_appearances.median()),
                "test_draw_appearances_max": int(subset.test_draw_appearances.max()),
                **weighted_metrics(
                    subset.label.to_numpy(int), subset.score.to_numpy(float)
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_table(collapsed: pd.DataFrame) -> pd.DataFrame:
    paired = (
        collapsed.loc[collapsed.group.isin([FINAL, PAIR])]
        .pivot(
            index=["patient_alias", "label", "test_draw_appearances"],
            columns="group",
            values="score",
        )
        .reset_index()
        .sort_values("patient_alias")
        .reset_index(drop=True)
    )
    if len(paired) != 87 or paired[[FINAL, PAIR]].isna().any().any():
        raise RuntimeError("Final and fixed-pair patient summaries are not aligned")
    return paired


def paired_estimands(paired: pd.DataFrame, weights: np.ndarray) -> dict[str, float]:
    labels = paired.label.to_numpy(int)
    final = weighted_metrics(labels, paired[FINAL].to_numpy(float), weights)
    pair = weighted_metrics(labels, paired[PAIR].to_numpy(float), weights)
    result: dict[str, float] = {}
    for prefix, values in [("final", final), ("pair", pair)]:
        for metric_name, value in values.items():
            result[f"{prefix}_{metric_name}"] = value
    for metric_name in final:
        result[f"delta_{metric_name}"] = final[metric_name] - pair[metric_name]
    return result


def bootstrap(paired: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    observed = paired_estimands(paired, np.ones(len(paired), dtype=int))
    labels = paired.label.to_numpy(int)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for bootstrap_id in range(1, N_BOOTSTRAP + 1):
        sampled = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        weights = np.bincount(sampled, minlength=len(paired))
        rows.append({"bootstrap_id": bootstrap_id, **paired_estimands(paired, weights)})
    return observed, pd.DataFrame(rows)


def summarize_bootstrap(observed: dict[str, float], replicates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, estimate in observed.items():
        low, high = np.quantile(replicates[name], [0.025, 0.975])
        rows.append(
            {
                "estimand": name,
                "estimate": estimate,
                "ci_95_percentile_low": float(low),
                "ci_95_percentile_high": float(high),
                "bootstrap_replicates": len(replicates),
                "resampling_unit": "unique_patient_alias",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    predictions = load_and_validate()
    collapsed = collapse_to_equal_patients(predictions)
    group_summary = summarize_groups(collapsed)
    paired = paired_table(collapsed)
    observed, replicates = bootstrap(paired)
    bootstrap_summary = summarize_bootstrap(observed, replicates)

    collapsed_path = OUT / "unique_patient_sensitivity_predictions_deidentified.csv"
    group_summary_path = OUT / "unique_patient_sensitivity_summary.csv"
    bootstrap_summary_path = OUT / "unique_patient_sensitivity_bootstrap_summary.csv"
    bootstrap_replicates_path = OUT / "unique_patient_sensitivity_bootstrap_replicates.csv"
    protocol_path = OUT / "unique_patient_sensitivity_protocol.json"
    provenance_path = OUT / "unique_patient_sensitivity_provenance.json"
    readme_path = OUT / "UNIQUE_PATIENT_SENSITIVITY.md"

    collapsed.to_csv(collapsed_path, index=False)
    group_summary.to_csv(group_summary_path, index=False)
    bootstrap_summary.to_csv(bootstrap_summary_path, index=False)
    replicates.to_csv(bootstrap_replicates_path, index=False)

    protocol = {
        "analysis_role": "post_hoc_equal_patient_sensitivity_of_locked_exploratory_test",
        "input": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
        "fixed_draw_seeds": SEEDS,
        "fixed_groups": GROUPS,
        "patient_score_rule": (
            "For each fixed group, average only the locked exploratory-test probabilities "
            "from draws in which that patient was assigned to the test role"
        ),
        "estimand": (
            "Equal-patient discrimination and probability error among the 87 patients "
            "represented in at least one fixed exploratory-test draw"
        ),
        "coverage_limit": "Eleven of 98 patients never entered a test role and are not represented",
        "paired_bootstrap_groups": [FINAL, PAIR],
        "bootstrap_stratification": "binary_patient_label",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": N_BOOTSTRAP,
        "test_dependent_model_adaptation": "none",
        "interpretation_limit": (
            "Sensitivity analysis of locked within-cohort transport; not independent or "
            "external validation and not used for model selection"
        ),
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    provenance = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": sha256(INPUT),
        "script_sha256": sha256(Path(__file__)),
        "input_rows": len(predictions),
        "collapsed_rows": len(collapsed),
        "unique_patients": len(paired),
        "positive_patients": int(paired.label.sum()),
        "negative_patients": int((1 - paired.label).sum()),
        "test_draw_appearances_min": int(paired.test_draw_appearances.min()),
        "test_draw_appearances_median": float(paired.test_draw_appearances.median()),
        "test_draw_appearances_max": int(paired.test_draw_appearances.max()),
        "post_test_model_changes": "none",
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    indexed = bootstrap_summary.set_index("estimand")
    final = group_summary.set_index("group").loc[FINAL]
    pair = group_summary.set_index("group").loc[PAIR]
    lines = [
        "# Equal-patient locked-test sensitivity analysis",
        "",
        "This post hoc analysis addresses unequal reuse of patients across the ten fixed exploratory-test draws. It averages each fixed model's locked test probabilities within patient and then gives every covered patient one equal-weight row.",
        "",
        "Coverage: 87 unique patient aliases (46 positive and 41 negative), each represented in 1-5 test draws (median 2). Eleven of the 98 analytic patients never entered a test role and are not represented.",
        "",
        "The analysis did not refit a model, change a feature, alter a fusion weight, or select a threshold. It is a sensitivity analysis of within-cohort transport, not an independent or external validation.",
        "",
        "## Results",
        "",
        f"- Final fixed vote: AUROC {final.auroc:.3f}, AP {final.average_precision:.3f}, trapezoidal AUPRC {final.auprc_trapezoid:.3f}, Brier score {final.brier:.3f}.",
        f"- Fixed PyRadiomics-RadImageNet pair: AUROC {pair.auroc:.3f}, AP {pair.average_precision:.3f}, trapezoidal AUPRC {pair.auprc_trapezoid:.3f}, Brier score {pair.brier:.3f}.",
        f"- Paired final-minus-pair AUROC difference: {indexed.loc['delta_auroc'].estimate:.3f} (95% patient-bootstrap interval {indexed.loc['delta_auroc'].ci_95_percentile_low:.3f} to {indexed.loc['delta_auroc'].ci_95_percentile_high:.3f}).",
        f"- Paired final-minus-pair AP difference: {indexed.loc['delta_average_precision'].estimate:.3f} (95% patient-bootstrap interval {indexed.loc['delta_average_precision'].ci_95_percentile_low:.3f} to {indexed.loc['delta_average_precision'].ci_95_percentile_high:.3f}).",
    ]
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(group_summary.to_string(index=False))
    print(bootstrap_summary.to_string(index=False))


if __name__ == "__main__":
    main()
