"""Patient-cluster confidence intervals and paired deltas for fixed development heads."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_revised_383_clinical_hierarchy_20261111"
URI = ROOT / "02_experiments/NCKUH_revised_383_uricads_20261111"


def main():
    proposed = pd.read_csv(OUT / "development_predictions.csv")
    reference = pd.read_csv(URI / "development_predictions.csv")
    reference = reference[reference.group.eq("uri_embedding_folds")]
    scores = reference.set_index("kidney_alias")[["patient_alias", "label", "score"]].rename(
        columns={"score": "uri_reference"})
    for group, frame in proposed.groupby("group"):
        scores[group] = frame.set_index("kidney_alias").reindex(scores.index).score
    assert scores.notna().all().all()
    assert len(scores) == 140 and scores.patient_alias.nunique() == 70
    labels = scores.label.to_numpy(int)
    names = ["uri_reference", *proposed.group.unique()]
    values = {name: scores[name].to_numpy(float) for name in names}
    clusters = [np.flatnonzero(scores.patient_alias.eq(pid))
                for pid in sorted(scores.patient_alias.unique())]
    draws = []
    rng = np.random.default_rng(20261111)
    for _ in range(2000):
        idx = np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
        if np.unique(labels[idx]).size == 2:
            draws.append(idx)
    rows = []
    for name in names:
        auc = np.asarray([roc_auc_score(labels[idx], values[name][idx]) for idx in draws])
        ap = np.asarray([average_precision_score(labels[idx], values[name][idx]) for idx in draws])
        if name == "uri_reference":
            ref_auc, ref_ap = auc, ap
        rows.append({"group": name, "bootstrap_valid": len(draws),
                     "auroc_low": np.percentile(auc, 2.5), "auroc_high": np.percentile(auc, 97.5),
                     "ap_low": np.percentile(ap, 2.5), "ap_high": np.percentile(ap, 97.5),
                     "paired_delta_auroc_low": np.percentile(auc - ref_auc, 2.5),
                     "paired_delta_auroc_high": np.percentile(auc - ref_auc, 97.5),
                     "paired_delta_ap_low": np.percentile(ap - ref_ap, 2.5),
                     "paired_delta_ap_high": np.percentile(ap - ref_ap, 97.5)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "development_ci.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
