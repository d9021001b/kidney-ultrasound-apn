"""Audit paired development predictions without refitting or scoring test rows."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nckuh_dmsa_cluster_ci import cluster_intervals

ROOT = Path(__file__).resolve().parents[2] / "02_experiments"
BASE = ROOT / "NCKUH_DMSA_fullcohort_fullframe_20261103"
MIL = ROOT / "NCKUH_DMSA_multiquery_mil_20261103"
OUT = ROOT / "NCKUH_DMSA_fullcohort_ranking_audit_20261103"
REFERENCE = "fullcohort_rad_color_image_plus_patient_direct"
CANDIDATE = "cross_attention_q4_quarter_plus_instance"
KEYS = ["seed", "patient_alias", "kidney_alias", "side", "label"]


def rank_breakdown(draw):
    positive = draw.loc[draw.label.eq(1)]
    negative = draw.loc[draw.label.eq(0)]
    gap = positive.score.to_numpy()[:, None] - negative.score.to_numpy()
    wins = (gap > 0).astype(float) + 0.5 * (gap == 0)
    within = positive.patient_alias.to_numpy()[:, None] == negative.patient_alias.to_numpy()
    result = []
    for kind, mask in (("same_patient", within), ("different_patient", ~within)):
        pairs = int(mask.sum())
        correct = float(wins[mask].sum())
        result.append(dict(comparison=kind, pairs=pairs, concordant=correct,
                           concordance=correct / pairs if pairs else np.nan,
                           contribution=correct / wins.size))
    if not np.isclose(sum(row["contribution"] for row in result),
                      roc_auc_score(draw.label, draw.score)):
        raise ValueError("Pair counts do not reconstruct AUROC")
    return result


def state_breakdown(draw):
    rows = []
    for positive_state, positive in draw[draw.label.eq(1)].groupby("DMSA_APN"):
        for negative_state, negative in draw[draw.label.eq(0)].groupby("DMSA_APN"):
            different = positive.patient_alias.to_numpy()[:, None] != negative.patient_alias.to_numpy()
            gap = (positive.score.to_numpy()[:, None] - negative.score.to_numpy())[different]
            rows.append(dict(positive_state=positive_state, negative_state=negative_state,
                             pairs=len(gap), concordant=float((gap > 0).sum() + 0.5 * (gap == 0).sum())))
    return rows


def run():
    sources = [(BASE / "instance_development_predictions.csv", REFERENCE),
               (MIL / "development_predictions.csv", CANDIDATE)]
    assignments = pd.read_csv(BASE / "patient_assignments.csv")
    frames = []
    for path, group in sources:
        data = pd.read_csv(path)
        rows = data.loc[data.group.eq(group), KEYS + ["group", "score"]]
        if rows.empty or rows.duplicated(["seed", "kidney_alias"]).any():
            raise ValueError(f"Missing or duplicate rows: {group}")
        joined = rows.merge(assignments, on=["seed", "patient_alias"], how="left", validate="many_to_one")
        if not joined.role.eq("development").all():
            raise ValueError(f"Non-development prediction: {group}")
        affected = joined.DMSA_APN.eq("B") | (
            joined.DMSA_APN.eq(joined.side.map({"left": "L", "right": "R"})))
        if not joined.side.isin(["left", "right"]).all() or not affected.eq(joined.label).all():
            raise ValueError(f"Laterality or label mismatch: {group}")
        frames.append(joined.sort_values(KEYS).reset_index(drop=True))
    pd.testing.assert_frame_equal(frames[0][KEYS], frames[1][KEYS])
    data = pd.concat(frames, ignore_index=True)
    seeds = sorted(assignments.seed.unique())
    ci, delta, _ = cluster_intervals(data, [REFERENCE, CANDIDATE], seeds, REFERENCE)
    rows, states = [], []
    for (seed, group), draw in data.groupby(["seed", "group"]):
        if draw.patient_alias.nunique() != 42:
            raise ValueError("Unexpected development cohort size")
        for row in rank_breakdown(draw):
            rows.append(dict(seed=seed, group=group, **row))
        for row in state_breakdown(draw):
            states.append(dict(seed=seed, group=group, **row))
    OUT.mkdir(exist_ok=True)
    ci.to_csv(OUT / "development_patient_cluster_ci.csv", index=False)
    delta.to_csv(OUT / "development_paired_delta_ci.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "development_rank_breakdown.csv", index=False)
    state_summary = pd.DataFrame(states).groupby(
        ["group", "positive_state", "negative_state"], as_index=False)[["pairs", "concordant"]].sum()
    state_summary["discordant_equivalent"] = state_summary.pairs - state_summary.concordant
    state_summary["concordance"] = state_summary.concordant / state_summary.pairs
    state_summary.to_csv(OUT / "development_cross_patient_by_state.csv", index=False)
    print(ci.to_string(index=False))
    print(delta.to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    run()
