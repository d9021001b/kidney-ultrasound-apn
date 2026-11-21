"""Audit a fixed fusion of reflection-averaged and local MIL development scores."""
import numpy as np
import pandas as pd

import audit_nckuh_fullcohort_ranking_20261103 as audit
import nckuh_dmsa_fullcohort_fullframe_20261103 as base
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_reflection_probe import OUT as REFLECTION

OUT = audit.ROOT / "NCKUH_DMSA_reflection_mil_fusion_20261103"
NEW = "reflection_average_plus_q4"


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    reflection = pd.read_csv(REFLECTION / "development_predictions.csv")
    mil = pd.read_csv(audit.MIL / "development_predictions.csv")
    keys = audit.KEYS
    assignments = pd.read_csv(audit.BASE / "patient_assignments.csv")
    scores = []
    for frame, group in ((reflection, "original"), (reflection, "reflection_average"),
                         (mil, "cross_attention_q4"), (mil, audit.CANDIDATE)):
        values = frame.loc[frame.group.eq(group), keys + ["score"]].set_index(keys).sort_index().score
        if not values.index.is_unique or values.empty or not values.between(0, 1).all():
            raise ValueError(f"Invalid prediction rows for {group}")
        if scores:
            pd.testing.assert_index_equal(scores[0].index, values.index)
        scores.append(values.rename(group))
    original, mirrored, local, published = scores
    # The original branch is independently refitted in the reflection experiment.
    np.testing.assert_allclose(.75 * original + .25 * local, published, rtol=0, atol=5e-5)
    fused = (.75 * mirrored + .25 * local).rename(NEW)
    data = pd.concat([s.rename("score").reset_index().assign(group=s.name)
                      for s in [*scores, fused]], ignore_index=True)
    data = data.merge(assignments, on=["seed", "patient_alias"], how="left", validate="many_to_one")
    affected = data.DMSA_APN.eq("B") | data.DMSA_APN.eq(data.side.map({"left": "L", "right": "R"}))
    if not data.role.eq("development").all() or not affected.eq(data.label).all():
        raise ValueError("Unexpected split role or DMSA-derived outcome")
    groups = list(data.group.unique())
    ci, paired, _ = cluster_intervals(data, groups, base.SEEDS, audit.CANDIDATE)
    _, paired_reflection, _ = cluster_intervals(data, ["reflection_average", NEW],
                                                 base.SEEDS, "reflection_average")
    metrics = []
    for (seed, group), draw in data.groupby(["seed", "group"]):
        if draw.patient_alias.nunique() != 42:
            raise ValueError("Unexpected development patient count")
        metrics.append(dict(seed=seed, group=group, **base.metric(draw.label, draw.score)))
    OUT.mkdir()
    for name, frame in (("development_predictions", data), ("development_ci", ci),
                        ("paired_delta_ci", pd.concat([paired, paired_reflection], ignore_index=True)),
                        ("development_metrics", pd.DataFrame(metrics))):
        frame.to_csv(OUT / f"{name}.csv", index=False)
    print(ci[ci.group.eq(NEW)].to_string(index=False))
    print(paired[paired.group.eq(NEW)].to_string(index=False))
    print(paired_reflection.to_string(index=False))


if __name__ == "__main__":
    run()
