"""Exploratory fixed equal fusion of independently motivated development probes."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_revised_383_fixed_complementary_fusion_20261112"


def load(path, group=None):
    frame = pd.read_csv(ROOT / path)
    if group is not None:
        frame = frame[frame.group.eq(group)]
    return frame.set_index("kidney_alias")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve fixed fusion result")
    dino = load("02_experiments/NCKUH_revised_383_view_paired_20261112/development_predictions.csv",
                "dinov3_class_view_paired")
    geometry = load("02_experiments/NCKUH_revised_383_relative_geometry_20261112/development_predictions.csv").reindex(dino.index)
    hierarchy = load("02_experiments/NCKUH_revised_383_dinov3_clinical_hierarchy_20261112/development_predictions.csv").reindex(dino.index)
    assert geometry.score.notna().all() and hierarchy.score.notna().all()
    labels = dino.label.to_numpy(int)
    scores = {
        "fixed_dino_geometry_mean": (dino.score.to_numpy() + geometry.score.to_numpy()) / 2,
        "fixed_dino_geometry_hierarchy_mean": (dino.score.to_numpy() + geometry.score.to_numpy() + hierarchy.score.to_numpy()) / 3,
    }
    clusters = [np.flatnonzero(dino.patient_alias.eq(pid)) for pid in sorted(dino.patient_alias.unique())]
    rng = np.random.default_rng(20261112)
    draws = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
             for _ in range(2000)]
    draws = [idx for idx in draws if np.unique(labels[idx]).size == 2]
    rows, predictions = [], []
    for name, score in scores.items():
        auc = np.asarray([roc_auc_score(labels[idx], score[idx]) for idx in draws])
        ap = np.asarray([average_precision_score(labels[idx], score[idx]) for idx in draws])
        rows.append({"group": name, "patients": dino.patient_alias.nunique(), "kidneys": len(dino),
                     "auroc": roc_auc_score(labels, score), "ap": average_precision_score(labels, score),
                     "auroc_ci_low": np.percentile(auc, 2.5), "auroc_ci_high": np.percentile(auc, 97.5),
                     "ap_ci_low": np.percentile(ap, 2.5), "ap_ci_high": np.percentile(ap, 97.5)})
        predictions.extend({"group": name, "patient_alias": row.patient_alias,
                            "kidney_alias": kidney, "label": int(label), "score": float(value)}
                           for kidney, row, label, value in zip(dino.index, dino.itertuples(), labels, score))
    pd.DataFrame(rows).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame([{"pair": "dino_geometry", "pearson": np.corrcoef(dino.score, geometry.score)[0, 1]},
                  {"pair": "dino_hierarchy", "pearson": np.corrcoef(dino.score, hierarchy.score)[0, 1]},
                  {"pair": "geometry_hierarchy", "pearson": np.corrcoef(geometry.score, hierarchy.score)[0, 1]}]).to_csv(
                      OUT / "score_correlations.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
