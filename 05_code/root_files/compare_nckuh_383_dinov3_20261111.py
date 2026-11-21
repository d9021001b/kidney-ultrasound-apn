"""Paired patient-cluster comparison of frozen DINOv3 and DenseNet probes."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
DINO = ROOT / "02_experiments/NCKUH_revised_383_dinov3_20261111"
BASE = ROOT / "02_experiments/NCKUH_revised_383_baseline_20261111"


def main():
    baseline = pd.read_csv(BASE / "development_predictions.csv")
    baseline = baseline[baseline.group.eq("radimagenet_dense")].set_index("kidney_alias")
    candidates = pd.read_csv(DINO / "development_predictions.csv")
    groups = [np.flatnonzero(baseline.patient_alias.eq(pid))
              for pid in sorted(baseline.patient_alias.unique())]
    labels = baseline.label.to_numpy(int)
    base_score = baseline.score.to_numpy(float)
    rng = np.random.default_rng(20261111)
    draws = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
             for _ in range(2000)]
    draws = [idx for idx in draws if np.unique(labels[idx]).size == 2]
    base_auc = np.asarray([roc_auc_score(labels[idx], base_score[idx]) for idx in draws])
    base_ap = np.asarray([average_precision_score(labels[idx], base_score[idx]) for idx in draws])
    results = []
    for name, frame in candidates.groupby("group"):
        aligned = frame.set_index("kidney_alias").reindex(baseline.index)
        assert aligned.score.notna().all() and np.array_equal(aligned.label.to_numpy(int), labels)
        score = aligned.score.to_numpy(float)
        auc = np.asarray([roc_auc_score(labels[idx], score[idx]) for idx in draws]) - base_auc
        ap = np.asarray([average_precision_score(labels[idx], score[idx]) for idx in draws]) - base_ap
        results.append({"group": name, "reference": "radimagenet_dense",
                        "delta_auroc": roc_auc_score(labels, score) - roc_auc_score(labels, base_score),
                        "delta_auroc_ci_low": np.percentile(auc, 2.5),
                        "delta_auroc_ci_high": np.percentile(auc, 97.5),
                        "delta_ap": average_precision_score(labels, score) - average_precision_score(labels, base_score),
                        "delta_ap_ci_low": np.percentile(ap, 2.5),
                        "delta_ap_ci_high": np.percentile(ap, 97.5),
                        "bootstrap_valid": len(draws)})
    result = pd.DataFrame(results)
    result.to_csv(DINO / "paired_dense_comparison.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
