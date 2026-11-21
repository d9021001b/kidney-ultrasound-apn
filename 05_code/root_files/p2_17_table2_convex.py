"""Table 2 of NCKUH paper 1 on the convex-probe development patients of the unified 255-patient convex cohort.
Reads the saved scores of draws_conv255 (first-study chain refitted in the convex group, 171 training patients per split);
adds the label-free semantic score (0.2 semantic + 0.8 fusion) and the paired change. 10,000 patient-cluster bootstrap replicates."""
import numpy as np, pandas as pd
import nckuh_dmsa_uricads_semantic_score_20261108 as sem
from nckuh_dmsa_cluster_ci import cluster_intervals
from p2_common import CACHE, P2
from p2_features import routed_images

SEEDS = (20261102, 20261103, 20261104)
_, _, routed = routed_images()
raw = np.load(CACHE / "uricads_fold_pathology_scores.npy")  # one row per cache row, all 388 patients
convex = routed[routed.route == "convex"]
semantic_image = sem.semantic_image_scores(raw[convex.cache_row.to_numpy()])["semantic_mean"]
key = pd.Series(semantic_image, index=convex.kidney_alias).groupby(level=0).mean()
patients = pd.read_csv(P2 / "cohort/patients.csv").set_index("patient_alias")
rows = []
for seed in SEEDS:
    d = pd.read_csv(P2 / f"draws_conv255/draw{SEEDS.index(seed)}_seed{seed}/all_models_convex_kidney_oof_scores.csv")
    d["sem"] = d.kidney_alias.map(key)
    d["DMSA_APN"] = d.patient_alias.map(patients.DMSA_APN)
    ref = .75 * d.nckuh1_fixed_chain + .25 * d.nckuh1_hierarchy
    sc = {"Tile + USFM reference": ref, "URI-CADS embedding alone": d.nckuh1_uricads, "Fixed URI-CADS fusion": d.nckuh1_final,
          "Semantic-score fusion": .2 * d["sem"] + .8 * d.nckuh1_final}
    for g, s in sc.items():
        rows.append(pd.DataFrame({"seed": seed, "group": g, "patient_alias": d.patient_alias, "kidney_alias": d.kidney_alias,
                                  "DMSA_APN": d.DMSA_APN, "label": d.label, "score": s}))
    print(seed, len(d), d["sem"].isna().sum(), d.patient_alias.nunique())
pred = pd.concat(rows)
groups = list(pred.group.unique())
ci, delta, _ = cluster_intervals(pred, groups, SEEDS, "Fixed URI-CADS fusion")
out = P2 / "training_size" / "conv255"
out.mkdir(exist_ok=True)
ci.to_csv(out / "table2_convex_ci.csv", index=False); delta.to_csv(out / "table2_convex_paired_delta.csv", index=False)
print(ci.round(4).to_string()); print(delta.round(4).to_string())
