"""Same-ruler check: score the same development patients with the first-study chain trained on different patient sets.

Rulers (convex group of the paper-2 pool; each is three splits):
  R1 = the development patients of the first study's three original splits (seeds 20261102-4)
  R2 = the development patients of paper 2's three 188/42/42 splits (same seeds)
Training sets for every ruler (ruler development patients always excluded):
  small = patients of the first study's 210-patient cohort in the convex group
  full  = all convex-group patients of the paper-2 pool
Chain: nckuh_chain of p2_07 (original code, fixed weights), refitted on the training patients.
"""
import json, os, sys, warnings
os.environ["P2_SPLIT"] = "paper1"  # paper-2 188/42/42 split for ruler R2
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_multiquery_mil_20261103 as old_mil
from p2_07_paper_methods import nckuh_chain
from p2_common import CACHE, P2, REGISTRY, metric, assign_folds
import p2_common
from p2_features import LEGACY, routed_images

OUT = P2 / "same_ruler"
SEEDS = (20261102, 20261103, 20261104)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_patients, _, all_images = routed_images()
    pool = all_images[all_images.role.isin(("train", "development")) & all_images.route.eq("convex")]
    images = pool.reset_index(drop=True)
    patients = all_patients[all_patients.patient_alias.isin(images.patient_alias)].reset_index(drop=True)
    kidneys = images[["kidney_alias", "patient_alias", "side", "kidney_APN"]].drop_duplicates().reset_index(drop=True)
    uri = np.load(CACHE / "uricads_fold_embeddings.npy", mmap_mode="r")
    rows = images.cache_row.to_numpy()
    arrays = {"tile": np.load(LEGACY / "tile_mean.npy")[rows], "usfm": np.load(LEGACY / "usfm_gap.npy")[rows],
              "rad_color": np.load(LEGACY / "rad_color.npy")[rows],
              "rad_mirror": np.load(LEGACY / "rad_color_horizontal_reflection.npy")[rows],
              "uri_folds": np.asarray(uri[rows]).reshape(len(rows), -1)}
    texture = pd.read_csv(LEGACY / "directional_features.csv").set_index("image_id").loc[images.image_id].reset_index(drop=True)
    # the first study's aliases are a different numbering: map old alias -> patient_id -> paper-2 alias
    link = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype=str).set_index("patient_id").patient_alias
    old_ref = old_mil.load_reference()[0]
    to_new = dict(zip(old_ref.patient_alias, old_ref.patient_id.astype(str).map(link)))
    old_alias = {a for a in to_new.values() if isinstance(a, str)}
    assign = pd.read_csv(base.OUT / "patient_assignments.csv")
    assign["patient_alias"] = assign.patient_alias.map(to_new)
    assign = assign[assign.patient_alias.notna()]
    print("first-study patients mapped:", len(old_alias), flush=True)
    y = kidneys.kidney_APN.to_numpy(int)
    # paper-2 splits over the 272-patient pool
    p2_pool = all_patients[all_patients.patient_alias.isin(
        all_images[all_images.role.isin(("train", "development")) & all_images.route.isin(("convex", "linear"))].patient_alias)].reset_index(drop=True)
    result = []
    rulers = sys.argv[1].split(",") if len(sys.argv) > 1 else ("R1", "R2")
    for ruler in rulers:
        for k, seed in enumerate(SEEDS):
            if ruler == "R1":
                dev_alias = set(assign[(assign.seed == seed) & assign.role.eq("development")].patient_alias)
            else:
                fold = assign_folds(p2_pool, seed)
                dev_alias = {a for a, f in fold.items() if f == 0}
            dev_alias &= set(patients.patient_alias)
            dev = np.flatnonzero(kidneys.patient_alias.isin(dev_alias))
            for name, eligible in (("small", old_alias), ("full", set(patients.patient_alias))):
                train_alias = (eligible & set(patients.patient_alias)) - dev_alias
                train_patients = np.flatnonzero(patients.patient_alias.isin(train_alias))
                parts = nckuh_chain(patients, kidneys, images, arrays, texture, train_patients, dev, seed)
                m = metric(y[dev], parts["nckuh1_final"])
                row = {"ruler": ruler, "seed": seed, "training": name, "train_patients": len(train_patients),
                       "dev_patients": len(dev_alias), "dev_kidneys": len(dev), **m}
                result.append(row)
                print(row, flush=True)
                pd.DataFrame(result).to_csv(OUT / ("same_ruler_per_split_" + "_".join(rulers) + ".csv"), index=False)
    table = pd.DataFrame(result)
    summary = table.groupby(["ruler", "training"])[["train_patients", "dev_patients", "auroc", "ap"]].mean().round(3)
    summary.to_csv(OUT / "same_ruler_summary.csv")
    print(summary)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    with threadpool_limits(limits=6):
        main()
