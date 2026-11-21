"""Paper 2, step 14: development performance with paper 1's bootstrap interval.

Same procedure as `nckuh_dmsa_kidney_baseline_20261102.py::bootstrap` (paper 1): 10,000 outcome-stratified
patient-cluster bootstrap replicates of the fixed development predictions. Patients are resampled with
replacement within the DMSA strata R/L/B/N (R right-only, L left-only, B bilateral, N no APN), each sampled patient
keeps both kidneys and every appearance in the three fixed splits, metrics are recalculated within each split and then
averaged over the splits, and the 2.5th and 97.5th percentiles give the interval. Replicates in which a split
lacks one outcome class are discarded. Run once per probe group and model.
Usage: python p2_14_bootstrap_summary.py [draws_folder]
"""

import sys
import warnings

import numpy as np
import pandas as pd

from nckuh_only_study_20261102 import weighted_metrics_batch
from p2_common import DRAW_SEEDS, P2

BASE = P2 / (sys.argv[1] if len(sys.argv) > 1 else "draws_paper1_split")
REPLICATES, SEED = 10000, 20261102

# (label in the manuscript tables, file, column or view)
MODELS = [
    ("First-study chain, refitted per probe group (final)", "all", "nckuh1_final"),
    ("Graph multiple-instance chain, refitted per probe group (final)", "all", "cych_final"),
    ("Both chains, equal mean", "all", "combined_equal_mean"),
    ("Nested frozen-feature selection", "all", "frozen_nested"),
    ("End-to-end ConvNeXt, context crop", "all", "specific_context"),
    ("End-to-end ConvNeXt, masked kidney", "all", "specific_masked"),
    ("Pre-training on both probes, then probe fine-tuning", "all", "pooled_init"),
    ("Fixed fusion: frozen + end-to-end", "all", "fusion_frozen_e2e"),
    ("Fixed fusion: four models", "all", "fusion_all"),
    ("Saab (Green Learning), own view", "green", "own"),
    ("Saab (Green Learning), paired view", "green", "paired"),
    ("Fixed rank mean: Saab + nested frozen", "green", "fusion_saab_frozen"),
    ("Nested selection with the Saab source added", "nested_saab", "score"),
]


def load(probe, draw, seed):
    folder = BASE / f"draw{draw}_seed{seed}"
    return {"all": pd.read_csv(folder / f"all_models_{probe}_kidney_oof_scores.csv"),
            "green": pd.read_csv(folder / "green_saab" / "kidney_oof_scores.csv"),
            "nested_saab": pd.read_csv(folder / "nested_saab" / "kidney_oof_scores.csv")}


def scores(tables, probe, source, column):
    if source == "all":
        t = tables["all"]
        return t[["kidney_alias", "patient_alias", "label"]].assign(score=t[column].to_numpy())
    t = tables[source]
    t = t[t.route == probe]
    if source == "green":
        t = t[t.view == column]
    return t[["kidney_alias", "patient_alias", "label", "score"]]


def main():
    patients = pd.read_csv(P2 / "cohort/patients.csv").set_index("patient_alias")
    rows, rng = [], np.random.default_rng(SEED)
    for probe in ("convex", "linear"):
        tables = [load(probe, d, s) for d, s in enumerate(DRAW_SEEDS)]
        aliases = sorted(set().union(*[set(t["all"].patient_alias) for t in tables]))
        codes = patients.DMSA_APN.reindex(aliases).to_numpy()
        strata = [np.flatnonzero(codes == c) for c in ("R", "L", "B", "N")]
        assert all(len(s) for s in strata), (probe, [len(s) for s in strata])
        index = {a: i for i, a in enumerate(aliases)}
        # one draw of patient weights shared by every model, as in paper 1 (paired replicates)
        size = REPLICATES
        sample = np.concatenate([rng.choice(s, (size, len(s)), replace=True) for s in strata], axis=1)
        counts = np.zeros((size, len(aliases)), dtype=np.float32)
        np.add.at(counts, (np.arange(size)[:, None], sample), 1)
        for label, source, column in MODELS:
            per_split, values, keep = [], np.zeros((size, 2)), np.ones(size, dtype=bool)
            for t in tables:
                s = scores(t, probe, source, column).reset_index(drop=True)
                clusters = np.array([index[a] for a in s.patient_alias])
                y, x = s.label.to_numpy(int), s.score.to_numpy(float)
                weights = counts[:, clusters]
                keep &= (weights[:, y == 0].sum(1) > 0) & (weights[:, y == 1].sum(1) > 0)
                values += weighted_metrics_batch(y, x, weights)[:, :2]
                point = weighted_metrics_batch(y, x, np.ones((1, len(y))))[0, :2]
                per_split.append(point)
            values = values[keep] / len(tables)
            point = np.mean(per_split, axis=0)
            sd = np.std(per_split, axis=0, ddof=1)
            lo, hi = np.quantile(values, [0.025, 0.975], axis=0)
            rows.append(dict(probe=probe, model=label, auroc=point[0], auroc_sd=sd[0], auroc_low=lo[0], auroc_high=hi[0],
                             ap=point[1], ap_sd=sd[1], ap_low=lo[1], ap_high=hi[1], valid_replicates=int(keep.sum())))
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "paper1_design_summary_ci.csv", index=False)
    for r in out.itertuples():
        print(f"{r.probe:6s} {r.model[:48]:48s} AUROC {r.auroc:.3f} ({r.auroc_low:.3f}-{r.auroc_high:.3f}) "
              f"AP {r.ap:.3f} ({r.ap_low:.3f}-{r.ap_high:.3f}) n={r.valid_replicates}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
