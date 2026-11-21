"""Compare frozen development predictions across historical archive availability."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
from nckuh_dmsa_inductive_graph import OUT as GRAPH

OUT = base.ROOT / "02_experiments/NCKUH_DMSA_archive_coverage_audit_20261103"
HISTORICAL = base.ROOT / "02_experiments/NCKUH_DMSA_fullframe_clean_probe_20261102"


def summary(draw):
    y, s = draw.label.to_numpy(), draw.score.to_numpy()
    prevalence = y.mean()
    if not 0 < prevalence < 1:
        raise ValueError("An availability stratum lacks outcome variation")
    equal_prevalence = np.where(y == 1, .5 / prevalence, .5 / (1 - prevalence))
    return dict(kidneys=len(y), patients=draw.patient_alias.nunique(),
                positives=int(y.sum()), prevalence=prevalence, auroc=roc_auc_score(y, s),
                ap=average_precision_score(y, s),
                ap_standardized_to_50pct=average_precision_score(y, s, sample_weight=equal_prevalence))


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    old = pd.read_csv(HISTORICAL / "deidentified_image_manifest.csv")
    assert old.patient_alias.nunique() == 111 and old.kidney_alias.nunique() == 212
    data = pd.read_csv(GRAPH / "development_predictions.csv")
    data = data[data.group.eq("knn_fusion")].copy()
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv", usecols=["seed", "patient_alias", "role"])
    data = data.merge(assignments, on=["seed", "patient_alias"], validate="many_to_one", how="left")
    assert data.role.eq("development").all() and not data.duplicated(["seed", "kidney_alias"]).any()
    matching = old[["kidney_alias", "kidney_APN", "patient_alias"]].drop_duplicates()
    matched = data.merge(matching, on=["kidney_alias", "patient_alias"], validate="many_to_one")
    np.testing.assert_array_equal(matched.label, matched.kidney_APN)
    data["archive_status"] = np.select(
        [data.kidney_alias.isin(old.kidney_alias), data.patient_alias.isin(old.patient_alias)],
        ["original_archive", "additional_kidney_same_patient"], default="additional_patient")
    results, counts, mix = [], [], []
    for seed, draw in data.groupby("seed"):
        assert draw.patient_alias.nunique() == 42
        archived = draw.archive_status.eq("original_archive")
        for name, part in (("all", draw), ("original_archive", draw[archived]), ("not_in_archive", draw[~archived])):
            results.append(dict(seed=seed, stratum=name, **summary(part)))
            for (state, label), subset in part.groupby(["DMSA_APN", "label"]):
                mix.append(dict(seed=seed, stratum=name, state=state, kidney_label=label, kidneys=len(subset)))
        p, n = draw[draw.label.eq(1)], draw[draw.label.eq(0)]
        difference = p.score.to_numpy()[:, None] - n.score.to_numpy()
        wins = (difference > 0).astype(float) + .5 * (difference == 0)
        for p_name, pm in p.groupby("archive_status").indices.items():
            for n_name, nm in n.groupby("archive_status").indices.items():
                block = wins[np.ix_(pm, nm)]
                counts.append(dict(seed=seed, positive_archive_status=p_name, negative_archive_status=n_name,
                                   pairs=block.size, concordant=block.sum()))
        assert np.isclose(wins.mean(), roc_auc_score(draw.label, draw.score))
    results = pd.DataFrame(results)
    OUT.mkdir()
    results.to_csv(OUT / "development_availability_metrics.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / "development_availability_pairs.csv", index=False)
    pd.DataFrame(mix).to_csv(OUT / "development_availability_state_mix.csv", index=False)
    print(results.to_string(index=False))
    print(results.groupby("stratum")[["prevalence", "auroc", "ap", "ap_standardized_to_50pct"]].mean())


if __name__ == "__main__":
    run()
