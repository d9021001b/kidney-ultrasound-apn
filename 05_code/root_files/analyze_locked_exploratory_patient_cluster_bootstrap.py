"""Patient-clustered uncertainty analysis for the locked exploratory test."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CYCH_locked_selected_exploratory_test_20260915"
INPUT = OUT / "exploratory_test_comparison_predictions_deidentified.csv"
FINAL = "FINAL-EQUAL-VOTE"
PAIR = "FIXED-RADIOMICS-RADIMAGENET-PAIR"
GROUPS = [FINAL, PAIR]
SEEDS = list(range(20260913, 20260923))
BOOTSTRAP_SEED = 20260915
N_BOOTSTRAP = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(INPUT)
    required = {"seed", "role", "group", "patient_alias", "label", "score"}
    if not required.issubset(predictions.columns):
        raise RuntimeError(f"Missing columns: {sorted(required - set(predictions.columns))}")
    predictions = predictions.loc[predictions.group.isin(GROUPS)].copy()
    if len(predictions) != 400:
        raise RuntimeError(f"Expected 400 selected prediction rows, found {len(predictions)}")
    if predictions.duplicated(["seed", "group", "patient_alias"]).any():
        raise RuntimeError("Duplicate seed/group/patient rows")
    if set(predictions.seed) != set(SEEDS):
        raise RuntimeError("Unexpected fixed draw seeds")
    expected_roles = {
        FINAL: "exploratory_internal_test",
        PAIR: "exploratory_test",
    }
    for group, role in expected_roles.items():
        if not predictions.loc[predictions.group.eq(group), "role"].eq(role).all():
            raise RuntimeError(f"Unexpected prediction role for {group}")

    label_check = predictions.groupby("patient_alias").label.nunique()
    if not label_check.eq(1).all():
        raise RuntimeError("Patient aliases have inconsistent labels")
    patients = (
        predictions.groupby("patient_alias", as_index=False)
        .agg(label=("label", "first"), test_draw_appearances=("seed", "nunique"))
        .sort_values("patient_alias")
        .reset_index(drop=True)
    )
    if len(patients) != 87 or patients.label.value_counts().to_dict() != {1: 46, 0: 41}:
        raise RuntimeError("Unexpected unique-patient coverage")

    for seed in SEEDS:
        current = predictions.loc[predictions.seed.eq(seed)]
        expected_aliases = None
        for group in GROUPS:
            subset = current.loc[current.group.eq(group)]
            if len(subset) != 20 or subset.patient_alias.nunique() != 20:
                raise RuntimeError(f"Invalid rows for {seed}/{group}")
            aliases = set(subset.patient_alias)
            expected_aliases = aliases if expected_aliases is None else expected_aliases
            if aliases != expected_aliases:
                raise RuntimeError(f"Paired groups are not aligned for seed {seed}")
    return predictions, patients


def prepare_draws(predictions: pd.DataFrame, patients: pd.DataFrame) -> list[dict[str, np.ndarray]]:
    patient_index = {alias: index for index, alias in enumerate(patients.patient_alias)}
    draws: list[dict[str, np.ndarray]] = []
    for seed in SEEDS:
        current = predictions.loc[predictions.seed.eq(seed)]
        wide = current.pivot(index=["patient_alias", "label"], columns="group", values="score").reset_index()
        wide = wide.sort_values("patient_alias").reset_index(drop=True)
        draws.append(
            {
                "seed": np.asarray([seed]),
                "patient_index": wide.patient_alias.map(patient_index).to_numpy(int),
                "label": wide.label.to_numpy(int),
                FINAL: wide[FINAL].to_numpy(float),
                PAIR: wide[PAIR].to_numpy(float),
            }
        )
    return draws


def weighted_metrics(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    keep = weights > 0
    y = labels[keep]
    s = scores[keep]
    w = weights[keep]
    if len(np.unique(y)) != 2:
        raise ValueError("Bootstrap draw lacks one outcome class")
    return (
        float(roc_auc_score(y, s, sample_weight=w)),
        float(average_precision_score(y, s, sample_weight=w)),
    )


def mean_draw_metrics(draws: list[dict[str, np.ndarray]], patient_weights: np.ndarray) -> dict[str, float]:
    values: dict[str, list[float]] = {
        "final_mean_auroc": [],
        "final_mean_average_precision": [],
        "pair_mean_auroc": [],
        "pair_mean_average_precision": [],
    }
    for draw in draws:
        weights = patient_weights[draw["patient_index"]]
        final_auc, final_ap = weighted_metrics(draw["label"], draw[FINAL], weights)
        pair_auc, pair_ap = weighted_metrics(draw["label"], draw[PAIR], weights)
        values["final_mean_auroc"].append(final_auc)
        values["final_mean_average_precision"].append(final_ap)
        values["pair_mean_auroc"].append(pair_auc)
        values["pair_mean_average_precision"].append(pair_ap)
    result = {name: float(np.mean(metric_values)) for name, metric_values in values.items()}
    result["delta_auroc"] = result["final_mean_auroc"] - result["pair_mean_auroc"]
    result["delta_average_precision"] = (
        result["final_mean_average_precision"] - result["pair_mean_average_precision"]
    )
    return result


def bootstrap(draws: list[dict[str, np.ndarray]], patients: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame, int]:
    observed = mean_draw_metrics(draws, np.ones(len(patients), dtype=int))
    positive = np.flatnonzero(patients.label.to_numpy(int) == 1)
    negative = np.flatnonzero(patients.label.to_numpy(int) == 0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, float | int]] = []
    attempts = 0
    while len(rows) < N_BOOTSTRAP:
        attempts += 1
        sampled = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        weights = np.bincount(sampled, minlength=len(patients))
        try:
            metrics = mean_draw_metrics(draws, weights)
        except ValueError:
            continue
        rows.append({"bootstrap_id": len(rows) + 1, **metrics})
    return observed, pd.DataFrame(rows), attempts


def summarize(observed: dict[str, float], replicates: pd.DataFrame) -> pd.DataFrame:
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
                "resampling_unit": "patient_alias",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    predictions, patients = load_and_validate()
    draws = prepare_draws(predictions, patients)
    observed, replicates, attempts = bootstrap(draws, patients)
    summary = summarize(observed, replicates)

    summary_path = OUT / "patient_cluster_bootstrap_summary.csv"
    replicates_path = OUT / "patient_cluster_bootstrap_replicates.csv"
    protocol_path = OUT / "patient_cluster_bootstrap_protocol.json"
    provenance_path = OUT / "patient_cluster_bootstrap_provenance.json"
    readme_path = OUT / "PATIENT_CLUSTER_BOOTSTRAP.md"
    summary.to_csv(summary_path, index=False)
    replicates.to_csv(replicates_path, index=False)

    protocol = {
        "analysis_role": "post_hoc_uncertainty_analysis_of_locked_exploratory_internal_test",
        "input": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
        "fixed_draw_seeds": SEEDS,
        "groups": GROUPS,
        "resampling_unit": "patient_alias",
        "stratified_by": "binary_patient_label",
        "cluster_rule": "A sampled patient retains all appearances and paired group predictions across fixed draws",
        "point_estimate": "Arithmetic mean of the ten fixed draw-level metrics",
        "interval": "2.5th and 97.5th percentiles of valid bootstrap estimates",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": N_BOOTSTRAP,
        "test_dependent_model_adaptation": "none",
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    provenance = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": sha256(INPUT),
        "script_sha256": sha256(Path(__file__)),
        "unique_patients": len(patients),
        "positive_patients": int(patients.label.sum()),
        "negative_patients": int((1 - patients.label).sum()),
        "test_draw_appearances_min": int(patients.test_draw_appearances.min()),
        "test_draw_appearances_median": float(patients.test_draw_appearances.median()),
        "test_draw_appearances_max": int(patients.test_draw_appearances.max()),
        "valid_bootstrap_replicates": len(replicates),
        "sampling_attempts": attempts,
        "post_test_model_changes": "none",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    indexed = summary.set_index("estimand")
    lines = [
        "# Patient-clustered bootstrap uncertainty analysis",
        "",
        "This post hoc analysis uses only predictions produced under the immutable exploratory-test lock. It resamples patient aliases within outcome class and preserves each sampled patient's linked appearances across the ten fixed draws and both paired model groups.",
        "",
        f"Coverage: {len(patients)} unique patient aliases ({int(patients.label.sum())} positive, {int((1 - patients.label).sum())} negative); each appeared in {int(patients.test_draw_appearances.min())}-{int(patients.test_draw_appearances.max())} test draws.",
        "",
        "The percentile intervals describe uncertainty under patient-cluster resampling within this explored cohort. They are not independent-cohort confidence intervals and do not convert the exploratory test into external validation.",
        "",
        "## Results",
        "",
    ]
    for name in ["final_mean_auroc", "final_mean_average_precision", "delta_auroc", "delta_average_precision"]:
        row = indexed.loc[name]
        lines.append(
            f"- `{name}`: {row.estimate:.3f} (95% percentile interval {row.ci_95_percentile_low:.3f} to {row.ci_95_percentile_high:.3f})"
        )
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"valid_replicates={len(replicates)} attempts={attempts}")


if __name__ == "__main__":
    main()
