"""Re-fit the old tile+USFM four-state head on the revised cohort roles."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_multiquery_mil_20261103 as old_cohort
from prepare_nckuh_383_contour_cohort_20261111 import OUT as REGISTRY, digest


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
FEATURES = EXP / "NCKUH_revised_383_legacy_features_20261112"
OUT = EXP / "NCKUH_revised_383_root_cause_20261112"


def score_head(patients, kidneys, images, image_values, train, development, seed):
    symmetric, antisymmetric = hierarchy.structured_views(
        base.patient_pair_features(image_values, images, patients)
    )
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    probability, models = hierarchy.state_probabilities(
        symmetric, antisymmetric, states, train, seed
    )
    score = hierarchy.kidney_scores(probability, patients, kidneys)[development]
    return score, {name: {
        "pca_components": int(model.best_params_["pca__n_components"]),
        "C": float(model.best_params_["lr__C"]),
        "inner_cv_auroc": float(model.best_score_),
    } for name, model in models.items()}


def main():
    old_patients, old_kidneys, old_images = old_cohort.load_reference()
    with np.load(EXP / "NCKUH_DMSA_native_tile_probe_20261103/tile_features.npz") as old_tile:
        old_values = np.c_[
            old_tile["mean"], np.load(EXP / "NCKUH_DMSA_usfm_frozen_20261104/usfm_gap.npy")
        ]
    old_roles = pd.read_csv(base.OUT / "patient_assignments.csv")
    seed = 20261102
    role = old_roles[old_roles.seed.eq(seed)].set_index("patient_alias").role
    old_train = np.flatnonzero(old_patients.patient_alias.map(role).eq("train"))
    old_dev = np.flatnonzero(old_kidneys.patient_alias.map(role).eq("development"))
    old_score, _ = score_head(
        old_patients, old_kidneys, old_images, old_values, old_train, old_dev, seed
    )
    saved = pd.read_csv(
        EXP / "NCKUH_DMSA_hierarchical_tile_usfm_20261104/development_predictions.csv"
    )
    saved = saved[
        saved.seed.eq(seed) & saved.group.eq("hierarchical_tile_usfm")
    ].set_index("kidney_alias")
    old_target = old_kidneys.iloc[old_dev]
    assert np.array_equal(saved.loc[old_target.kidney_alias].label.to_numpy(),
                          old_target.kidney_APN.to_numpy(int))
    max_reproduction_error = float(np.max(
        np.abs(old_score - saved.loc[old_target.kidney_alias].score.to_numpy())
    ))
    print(f"Old draw reproduction max absolute difference: {max_reproduction_error}", flush=True)
    assert max_reproduction_error < 1e-6

    lock = json.loads((REGISTRY / "cohort_lock.json").read_text(encoding="utf-8"))
    assert digest(REGISTRY / "contour_image_manifest.csv") == lock["files_sha256"]["contour_image_manifest.csv"]
    image_manifest = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidney_manifest = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    assignments = pd.read_csv(REGISTRY / "patient_assignments.csv")
    active = assignments[
        assignments.role.isin(("train", "development"))
        & assignments.patient_alias.isin(image_manifest.patient_alias)
    ].reset_index(drop=True)
    assert active.role.value_counts().to_dict() == {"train": 204, "development": 70}
    selected = np.flatnonzero(image_manifest.patient_alias.isin(active.patient_alias))
    images = image_manifest.iloc[selected].reset_index(drop=True)
    kidneys = kidney_manifest[kidney_manifest.patient_alias.isin(active.patient_alias)].reset_index(drop=True)
    roles = active.set_index("patient_alias").role
    train = np.flatnonzero(active.role.eq("train"))
    development = np.flatnonzero(kidneys.patient_alias.map(roles).eq("development"))
    tile = np.load(FEATURES / "tile_mean.npy")[selected]
    usfm = np.load(FEATURES / "usfm_gap.npy")[selected]
    score, selection = score_head(
        active, kidneys, images, np.c_[tile, usfm], train, development, 20261111
    )
    old_ids = set(old_patients.patient_id.astype(str))
    new_link = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype={"patient_id": str})
    old_aliases = set(new_link.loc[new_link.patient_id.isin(old_ids), "patient_alias"])
    old_train = np.flatnonzero(active.role.eq("train") & active.patient_alias.isin(old_aliases))
    old_train_score, old_train_selection = score_head(
        active, kidneys, images, np.c_[tile, usfm], old_train, development, 20261111
    )
    added_train = np.flatnonzero(active.role.eq("train") & ~active.patient_alias.isin(old_aliases))
    added_train_score, added_train_selection = score_head(
        active, kidneys, images, np.c_[tile, usfm], added_train, development, 20261111
    )
    target = kidneys.iloc[development].reset_index(drop=True)
    labels = target.kidney_APN.to_numpy(int)

    def subgroup_metrics(values, selected):
        return {"patients": int(target.patient_alias.iloc[selected].nunique()),
                "kidneys": int(selected.sum()),
                "auroc": float(roc_auc_score(labels[selected], values[selected])),
                "ap": float(average_precision_score(labels[selected], values[selected]))}

    overlap = target.patient_alias.isin(old_aliases).to_numpy()
    result = {
        "old_draw_reproduction_max_abs": max_reproduction_error,
        "train_patients": len(train),
        "development_patients": target.patient_alias.nunique(),
        "development_kidneys": len(target),
        "auroc": float(roc_auc_score(labels, score)),
        "ap": float(average_precision_score(labels, score)),
        "head_selection": selection,
        "training_ablation": {
            "old_labeled_train_patients": len(old_train),
            "all_train": {
                "all_dev": subgroup_metrics(score, np.ones(len(target), dtype=bool)),
                "old_overlap_dev": subgroup_metrics(score, overlap),
                "other_dev": subgroup_metrics(score, ~overlap),
            },
            "old_labeled_train_only": {
                "all_dev": subgroup_metrics(old_train_score, np.ones(len(target), dtype=bool)),
                "old_overlap_dev": subgroup_metrics(old_train_score, overlap),
                "other_dev": subgroup_metrics(old_train_score, ~overlap),
                "head_selection": old_train_selection,
            },
            "added_train_only": {
                "train_patients": len(added_train),
                "all_dev": subgroup_metrics(added_train_score, np.ones(len(target), dtype=bool)),
                "old_overlap_dev": subgroup_metrics(added_train_score, overlap),
                "other_dev": subgroup_metrics(added_train_score, ~overlap),
                "head_selection": added_train_selection,
            },
        },
        "test_scored": False,
        "scope": "Old tile+USFM hierarchy alone; not the full old fixed-chain fusion.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "legacy_hierarchy_new_roles.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    target[["patient_alias", "kidney_alias", "side", "kidney_APN"]].assign(
        score=score, old_labeled_train_only_score=old_train_score,
        added_train_only_score=added_train_score
    ).to_csv(OUT / "legacy_hierarchy_development_predictions.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
