"""Training-size curve of the first-study chain on the first study's convex development patients.

Evaluation patients: the development patients of the first study's three splits (seeds 20261102-4) that belong to the
convex group of the paper-2 pool (31, 30, 32 patients); the same patients at every training size.
Training patients: all other convex patients of the pool (164, 165, 164) are put in a random order (REPEATS orders per
split); the first n patients of each order train the chain (nested subsets, n = SIZES; the points at 122 and at the full set come from same_ruler/).
Chain: nckuh_chain of p2_07 (original code, fixed weights), refitted on the training patients each time.
Usage: python p2_16_training_size_curve.py [repeats]
"""
import os, sys, warnings
os.environ["P2_SPLIT"] = "paper1"
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_multiquery_mil_20261103 as old_mil
from p2_07_paper_methods import nckuh_chain
from p2_common import CACHE, P2, REGISTRY, metric
from p2_features import LEGACY, routed_images

OUT = P2 / "training_size"
SEEDS = (20261102, 20261103, 20261104)
SIZES = (40, 60, 80, 100)  # 122 and the full set (164-165) are in same_ruler/
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_patients, _, all_images = routed_images()
    images = all_images[all_images.role.isin(("train", "development")) & all_images.route.eq("convex")].reset_index(drop=True)
    patients = all_patients[all_patients.patient_alias.isin(images.patient_alias)].reset_index(drop=True)
    kidneys = images[["kidney_alias", "patient_alias", "side", "kidney_APN"]].drop_duplicates().reset_index(drop=True)
    uri = np.load(CACHE / "uricads_fold_embeddings.npy", mmap_mode="r")
    rows = images.cache_row.to_numpy()
    arrays = {"tile": np.load(LEGACY / "tile_mean.npy")[rows], "usfm": np.load(LEGACY / "usfm_gap.npy")[rows],
              "rad_color": np.load(LEGACY / "rad_color.npy")[rows],
              "rad_mirror": np.load(LEGACY / "rad_color_horizontal_reflection.npy")[rows],
              "uri_folds": np.asarray(uri[rows]).reshape(len(rows), -1)}
    texture = pd.read_csv(LEGACY / "directional_features.csv").set_index("image_id").loc[images.image_id].reset_index(drop=True)
    link = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype=str).set_index("patient_id").patient_alias
    old_ref = old_mil.load_reference()[0]
    to_new = dict(zip(old_ref.patient_alias, old_ref.patient_id.astype(str).map(link)))
    assign = pd.read_csv(base.OUT / "patient_assignments.csv")
    assign["patient_alias"] = assign.patient_alias.map(to_new)
    assign = assign[assign.patient_alias.notna()]
    y = kidneys.kidney_APN.to_numpy(int)
    result, path = [], OUT / "training_size_per_fit.csv"
    for seed in SEEDS:
        dev_alias = set(assign[(assign.seed == seed) & assign.role.eq("development")].patient_alias) & set(patients.patient_alias)
        dev = np.flatnonzero(kidneys.patient_alias.isin(dev_alias))
        train_all = np.flatnonzero(~patients.patient_alias.isin(dev_alias))
        plan = [(size, r) for size in SIZES for r in range(REPEATS)]
        for size, r in plan:
            order = np.random.default_rng(seed * 100 + r).permutation(train_all)
            train_patients = np.sort(order[:size])
            parts = nckuh_chain(patients, kidneys, images, arrays, texture, train_patients, dev, seed)
            m = metric(y[dev], parts["nckuh1_final"])
            row = {"seed": seed, "train_patients": int(size), "repeat": r, "dev_patients": len(dev_alias), "dev_kidneys": len(dev), **m}
            result.append(row)
            print(row, flush=True)
            pd.DataFrame(result).to_csv(path, index=False)
    table = pd.DataFrame(result)
    table["size"] = table.train_patients.astype(str)
    summary = table.groupby("size").agg(train_patients=("train_patients", "mean"), fits=("auroc", "size"),
                                        auroc=("auroc", "mean"), ap=("ap", "mean")).sort_values("train_patients")
    summary.round(3).to_csv(OUT / "training_size_summary.csv")
    print(summary.round(3))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    with threadpool_limits(limits=6):
        main()
