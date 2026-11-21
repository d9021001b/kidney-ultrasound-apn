"""Optimistic ranking ceiling if only patients with extra views can change."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
from audit_nckuh_additional_same_exam_views import OUT as CANDIDATES

OUT = base.ROOT / "02_experiments/NCKUH_candidate_view_headroom_20261103"
REFERENCE = base.ROOT / "02_experiments/NCKUH_DMSA_fullcohort_directional_20261103"


def summarize(draw, affected):
    changed = draw.patient_alias.isin(affected).to_numpy()
    y = draw.label.to_numpy(int)
    score = draw.score.to_numpy(float)
    if not np.isfinite(score).all() or set(y) != {0, 1}:
        raise ValueError("Invalid scores or missing reference class")
    extreme = np.where(changed, np.where(y == 1, score.max() + 1, score.min() - 1), score)
    auc, ap = roc_auc_score(y, score), average_precision_score(y, score)
    max_auc = roc_auc_score(y, extreme)
    p, n = y == 1, y == 0
    delta = score[p, None] - score[None, n]
    fixable = changed[p, None] | changed[None, n]
    lost = (delta < 0).astype(float) + .5 * (delta == 0)
    recoverable = lost[fixable].sum()
    assert np.isclose(max_auc - auc, recoverable / delta.size)
    return dict(patients=draw.patient_alias.nunique(), kidneys=len(draw),
                affected_patients=draw.loc[changed, "patient_alias"].nunique(),
                affected_kidneys=int(changed.sum()), auroc=auc, ap=ap,
                optimistic_auroc=max_auc, optimistic_ap=average_precision_score(y, extreme),
                cross_class_pairs=delta.size, recoverable_pair_loss=recoverable)


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    candidate = pd.read_csv(CANDIDATES / "candidate_qc_internal.csv")
    affected = set(candidate.patient_alias)
    assert len(affected) == 6
    saved = pd.read_csv(REFERENCE / "development_predictions.csv")
    data = saved[saved.group.eq("aligned_fusion")]
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv", usecols=["seed", "patient_alias", "role"])
    data = data.merge(assignments, on=["seed", "patient_alias"], how="left", validate="many_to_one")
    assert data.role.eq("development").all() and not data.duplicated(["seed", "kidney_alias"]).any()
    table = pd.DataFrame([dict(seed=seed, **summarize(draw, affected)) for seed, draw in data.groupby("seed")])
    assert len(table) == 3 and table.patients.eq(42).all()
    OUT.mkdir()
    table.to_csv(OUT / "development_sensitivity.csv", index=False)
    print(table.to_string(index=False))
    print(table[["auroc", "ap", "optimistic_auroc", "optimistic_ap"]].mean().to_string())


if __name__ == "__main__":
    run()
