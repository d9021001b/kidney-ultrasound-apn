"""Conditional patient-cluster bootstrap intervals for archived fixed scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
DEV = EXP / "CYCH_selected_pipeline_ablation_20260914"
TEST = EXP / "CYCH_locked_selected_exploratory_test_20260915"
OUT = DEV / "ci_20261005"
SEEDS = list(range(20260913, 20260923))
METRICS = ["auroc", "average_precision", "auprc_trapezoid"]
REPLICATES = 10000
RNG_SEED = 20260915


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_metrics(labels, scores, weights):
    """Evaluate sorted threshold curves in parallel, including exact ties."""
    order = np.argsort(-scores, kind="stable")
    y, s, w = labels[order], scores[order], weights[:, order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(w * y, axis=1)[:, ends]
    fp = np.cumsum(w * (1 - y), axis=1)[:, ends]
    assert np.all(tp[:, -1] > 0) and np.all(fp[:, -1] > 0)
    recall = np.c_[np.zeros(len(w)), tp / tp[:, -1, None]]
    fpr = np.c_[np.zeros(len(w)), fp / fp[:, -1, None]]
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp, dtype=float), where=(tp + fp) > 0)
    precision = np.c_[np.ones(len(w)), precision]
    auc = np.sum(np.diff(fpr) * (recall[:, 1:] + recall[:, :-1]) / 2, axis=1)
    ap = np.sum(np.diff(recall) * precision[:, 1:], axis=1)
    pr_auc = np.sum(np.diff(recall) * (precision[:, 1:] + precision[:, :-1]) / 2, axis=1)
    return np.stack([auc, ap, pr_auc], axis=1)


def prepare(path):
    data = pd.read_csv(path)
    assert not data.duplicated(["seed", "group", "patient_alias"]).any()
    assert set(data.seed) == set(SEEDS)
    assert data.groupby("patient_alias").label.nunique().eq(1).all()
    assert np.isfinite(data.score).all() and data.score.between(0, 1).all()
    patients = data.groupby("patient_alias").label.first().sort_index()
    index = {p: i for i, p in enumerate(patients.index)}
    groups = list(data.group.unique())
    draws = []
    for seed in SEEDS:
        draw = data.loc[data.seed.eq(seed)].pivot(index=["patient_alias", "label"], columns="group", values="score")
        draw = draw.reset_index().sort_values("patient_alias")
        assert len(draw) == 20 and not draw[groups].isna().any().any()
        draws.append((draw.patient_alias.map(index).to_numpy(int), draw.label.to_numpy(int),
                      {g: draw[g].to_numpy(float) for g in groups}))
    return groups, patients, draws


def sample_weights(patients, draws):
    labels = patients.to_numpy(int)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    rng = np.random.default_rng(RNG_SEED)
    weights, attempts = [], 0
    while len(weights) < REPLICATES:
        attempts += 1
        sampled = np.r_[rng.choice(positive, len(positive), replace=True),
                        rng.choice(negative, len(negative), replace=True)]
        w = np.bincount(sampled, minlength=len(labels))
        if all(w[idx][y == 1].sum() > 0 and w[idx][y == 0].sum() > 0 for idx, y, _ in draws):
            weights.append(w)
    return np.asarray(weights), attempts


def analyze(role, prediction_path, summary_path):
    groups, patients, draws = prepare(prediction_path)
    weights, attempts = sample_weights(patients, draws)
    all_weights = np.vstack([np.ones(len(patients), dtype=int), weights])
    result = np.zeros((len(all_weights), len(groups), len(METRICS)))
    for idx, labels, scores in draws:
        for j, group in enumerate(groups):
            result[:, j] += weighted_metrics(labels, scores[group], all_weights[:, idx]) / len(draws)
    archived = pd.read_csv(summary_path).set_index("group")
    rows = []
    for j, group in enumerate(groups):
        for k, metric in enumerate(METRICS):
            assert np.isclose(result[0, j, k], archived.loc[group, f"mean_{metric}"], atol=1e-12, rtol=0)
            low, high = np.quantile(result[1:, j, k], [0.025, 0.975])
            rows.append(dict(role=role, group=group, metric=metric, estimate=result[0, j, k],
                             ci95_low=low, ci95_high=high, bootstrap_replicates=REPLICATES,
                             covered_patients=len(patients)))
    baseline = "RELIABLE-NEGATIVE-GCN-MIL" if role == "development" else "FIXED-RADIOMICS-RADIMAGENET-PAIR"
    base = groups.index(baseline)
    differences = []
    for j, group in enumerate(groups):
        if group == baseline:
            continue
        for k, metric in enumerate(METRICS):
            delta = result[:, j, k] - result[:, base, k]
            low, high = np.quantile(delta[1:], [0.025, 0.975])
            differences.append(dict(role=role, group=group, baseline=baseline, metric=metric,
                                    estimate=delta[0], ci95_low=low, ci95_high=high))
    np.savez_compressed(OUT / f"{role}_bootstrap_replicates.npz", metrics=result[1:], groups=groups,
                        metric_names=METRICS)
    return rows, differences, dict(role=role, covered_patients=len(patients),
                                  positive_patients=int(patients.sum()), negative_patients=int((1 - patients).sum()),
                                  sampling_attempts=attempts, groups=groups)


def reference_test():
    # The research runtime supplies scikit-learn, absent from the bundled runtime.
    from sklearn import __version__ as sklearn_version
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, auc
    rng = np.random.default_rng(27)
    for _ in range(250):
        y = np.r_[np.zeros(9, dtype=int), np.ones(11, dtype=int)]
        s = rng.integers(0, 8, len(y)) / 7
        w = rng.integers(0, 4, len(y))
        if not w[y == 0].sum() or not w[y == 1].sum():
            continue
        observed = weighted_metrics(y, s, w[None, :])[0]
        precision, recall, _ = precision_recall_curve(y, s, sample_weight=w)
        expected = [roc_auc_score(y, s, sample_weight=w),
                    average_precision_score(y, s, sample_weight=w), auc(recall, precision)]
        np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=0)
    (OUT / "metric_reference_validation.json").write_text(json.dumps(
        dict(reference="scikit-learn", version=sklearn_version, random_tie_and_zero_weight_tests=250,
             absolute_tolerance=1e-12, result="PASS"), indent=2), encoding="utf-8")
    print("Scikit-learn reference validation: PASS")


def main():
    assert not OUT.exists(), "Preserve any previous analysis; inspect before replacing"
    inputs = [DEV / "development_predictions_deidentified.csv", DEV / "development_summary.csv",
              TEST / "exploratory_test_comparison_predictions_deidentified.csv", TEST / "exploratory_test_comparison_summary.csv",
              TEST / "patient_cluster_bootstrap_summary.csv"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    OUT.mkdir()
    rows, differences, coverage = [], [], []
    for role, prediction, summary in [("development", inputs[0], inputs[1]), ("exploratory_internal_test", inputs[2], inputs[3])]:
        values, changes, audit = analyze(role, prediction, summary)
        rows.extend(values)
        differences.extend(changes)
        coverage.append(audit)
    table = pd.DataFrame(rows)
    existing = pd.read_csv(inputs[4]).set_index("estimand")
    for metric, old_name in [("auroc", "final_mean_auroc"), ("average_precision", "final_mean_average_precision")]:
        row = table.loc[(table.role == "exploratory_internal_test") & (table.group == "FINAL-EQUAL-VOTE") & (table.metric == metric)].iloc[0]
        np.testing.assert_allclose([row.estimate, row.ci95_low, row.ci95_high],
                                   existing.loc[old_name, ["estimate", "ci_95_percentile_low", "ci_95_percentile_high"]].to_numpy(float),
                                   atol=1e-12, rtol=0)
    table.to_csv(OUT / "performance_ci95.csv", index=False)
    pd.DataFrame(differences).to_csv(OUT / "paired_changes_ci95.csv", index=False)
    assert hashes == {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    protocol = dict(analysis_date="2026-10-05", source_sha256=hashes, script_sha256=digest(Path(__file__)),
                    bootstrap_seed=RNG_SEED, valid_replicates=REPLICATES, resampling_unit="patient_alias",
                    stratification="patient diagnosis label", cluster_rule="Retain all linked appearances across ten fixed draws and all paired groups",
                    estimand="Arithmetic mean of ten fixed draw-level metrics", interval="2.5th and 97.5th percentiles",
                    coverage=coverage, existing_test_auroc_ap_intervals_reproduced=True, inputs_unchanged=True,
                    model_refitting=False, model_or_feature_selection=False,
                    limitations="Post hoc, conditional on fixed scores and an already explored cohort. Does not account for model selection optimism, training uncertainty or independent-cohort transport. Development intervals are descriptive, not confirmatory.")
    (OUT / "protocol_and_provenance.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(table.loc[table.group.isin(["RELIABLE-NEGATIVE-GCN-MIL", "RELIABLE-NEGATIVE-GCN-MIL+PYRADIOMICS", "RELIABLE-NEGATIVE-GCN-MIL+I3", "FINAL-EQUAL-VOTE"])].to_string(index=False))
    print("Archived point estimates and existing test AUROC/AP intervals reproduced; all input hashes unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-test", action="store_true")
    args = parser.parse_args()
    reference_test() if args.reference_test else main()
