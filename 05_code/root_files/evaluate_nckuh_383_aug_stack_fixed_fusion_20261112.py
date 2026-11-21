"""Fixed equal fusion of independently selected OOF stack and augmented frozen head."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_revised_383_aug_stack_fixed_fusion_20261112"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stack = pd.read_csv(ROOT / "02_experiments/NCKUH_revised_383_oof_hierarchy_stacking_20261112/development_predictions.csv").set_index("kidney_alias")
    aug = pd.read_csv(ROOT / "02_experiments/NCKUH_revised_383_dinov3_lora_aug_20261112/development_predictions.csv")
    aug = aug[aug.group.eq("frozen_head_aug")].set_index("kidney_alias").reindex(stack.index)
    assert aug.score.notna().all() and np.array_equal(stack.label.to_numpy(int), aug.label.to_numpy(int))
    labels = stack.label.to_numpy(int); score = (stack.score.to_numpy() + aug.score.to_numpy()) / 2
    clusters = [np.flatnonzero(stack.patient_alias.eq(pid)) for pid in sorted(stack.patient_alias.unique())]
    rng = np.random.default_rng(20261112)
    draws = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))]) for _ in range(2000)]
    draws = [idx for idx in draws if np.unique(labels[idx]).size == 2]
    auc = np.asarray([roc_auc_score(labels[idx], score[idx]) for idx in draws])
    ap = np.asarray([average_precision_score(labels[idx], score[idx]) for idx in draws])
    result = {"group": "fixed_equal_oof_stack_plus_frozen_aug", "patients": stack.patient_alias.nunique(),
              "kidneys": len(stack), "auroc": roc_auc_score(labels, score),
              "ap": average_precision_score(labels, score), "auroc_ci_low": np.percentile(auc, 2.5),
              "auroc_ci_high": np.percentile(auc, 97.5), "ap_ci_low": np.percentile(ap, 2.5),
              "ap_ci_high": np.percentile(ap, 97.5),
              "score_correlation": np.corrcoef(stack.score, aug.score)[0, 1]}
    pd.DataFrame([result]).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame({"patient_alias": stack.patient_alias, "kidney_alias": stack.index,
                  "label": labels, "score": score}).to_csv(OUT / "development_predictions.csv", index=False)
    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
