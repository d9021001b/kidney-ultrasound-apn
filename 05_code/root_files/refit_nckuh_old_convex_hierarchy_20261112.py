"""Old locked draws: route patients before fitting the tile+USFM hierarchy."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_hierarchical_tile_usfm_20261104 as old_head
import nckuh_dmsa_multiquery_mil_20261103 as cohort
from audit_nckuh_old_draw_probe_routing_20261112 import (
    LABELS, OLD, OUT as ROUTE_AUDIT, SEEDS, WORKBOOK, interval, score,
)
from openpyxl import load_workbook
from prepare_nckuh_383_contour_cohort_20261111 import digest


ROOT = Path(__file__).resolve().parents[2]
OUT = ROUTE_AUDIT / "convex_hierarchy_refit"
SOURCE = ROOT / "02_experiments/NCKUH_DMSA_hierarchical_tile_usfm_20261104/development_predictions.csv"
FIXED = ROOT / "02_experiments/NCKUH_DMSA_tile_replacement_20261103/development_predictions.csv"
URI = ROOT / "02_experiments/NCKUH_DMSA_uricads_official_frozen_20261105/development_predictions.csv"
FINAL = ROOT / "02_experiments/NCKUH_DMSA_uricads_full_detector_20261105_r2/development_predictions.csv"
GROUP = "hierarchical_tile_usfm"


def aligned(path, group, seed, target):
    data = pd.read_csv(path)
    selected = data[data.seed.eq(seed) & data.group.eq(group)].set_index("kidney_alias", verify_integrity=True)
    selected = selected.loc[target.kidney_alias]
    assert np.array_equal(selected.label.to_numpy(int), target.kidney_APN.to_numpy(int))
    return selected.score.to_numpy(float)


def main():
    sheet = load_workbook(WORKBOOK, read_only=True, data_only=True)["完整資料彙整"]
    rows = sheet.iter_rows(values_only=True)
    header = list(next(rows))
    assert header[0] == "id" and header[7] == "Probe"
    probe = {str(row[0]).strip().lstrip("0"): str(row[7]).strip().lower()
             for row in rows if row[0] is not None}
    linkage = pd.read_csv(LABELS / "patient_label_linkage_private.csv", dtype={"patient_id": str})
    alias_probe = linkage.set_index("patient_alias").patient_id.str.lstrip("0").map(probe)
    patients, kidneys, images = cohort.load_reference()
    assert len(patients) == 210 and len(kidneys) == 414 and len(images) == 820
    values = old_head.image_representations(images)["tile_usfm"]
    symmetric, antisymmetric = hierarchy.structured_views(base.patient_pair_features(values, images, patients))
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(OLD / "patient_assignments.csv")
    saved = pd.read_csv(SOURCE)
    saved = saved[saved.group.eq(GROUP)].set_index(["seed", "kidney_alias"], verify_integrity=True)
    report = {"source_sha256": {"workbook": digest(WORKBOOK),
                                 "old_predictions": digest(SOURCE),
                                 "old_fixed_predictions": digest(FIXED),
                                 "old_uri_predictions": digest(URI),
                                 "old_final_predictions": digest(FINAL),
                                 "old_assignments": digest(OLD / "patient_assignments.csv"),
                                 "tile_features": digest(old_head.TILE / "tile_features.npz"),
                                 "usfm_gap": digest(old_head.USFM / "usfm_gap.npy")},
              "experiment": "Re-fit original tile+USFM hierarchical four-state branch after convex-only patient routing; old 126/42/42 split membership unchanged before exclusion; other original fusion branches are not re-fit. Reconstruct final fusion and replace this branch only as a sensitivity control.",
              "test_APN_scored": False, "draws": []}
    outputs = []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train_all = np.flatnonzero(patients.patient_alias.map(mapping).eq("train"))
        train_convex = np.asarray([idx for idx in train_all if alias_probe.loc[patients.patient_alias.iloc[idx]] == "convex"])
        dev_all = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        dev_convex = np.asarray([idx for idx in dev_all if alias_probe.loc[kidneys.patient_alias.iloc[idx]] == "convex"])
        assert len(train_all) == 126 and 110 <= len(train_convex) <= 125
        assert 70 <= len(dev_convex) <= 82
        assert set(patients.patient_alias.iloc[train_convex]).isdisjoint(kidneys.patient_alias.iloc[dev_convex])
        original_probability, _ = hierarchy.state_probabilities(
            symmetric, antisymmetric, states, train_all, seed)
        original_score = hierarchy.kidney_scores(original_probability, patients, kidneys)[dev_all]
        old_target = kidneys.iloc[dev_all]
        old_saved = saved.loc[[(seed, alias) for alias in old_target.kidney_alias]].score.to_numpy(float)
        max_error = float(np.max(np.abs(original_score - old_saved)))
        assert max_error < 1e-4, (seed, max_error)
        routed_probability, models = hierarchy.state_probabilities(
            symmetric, antisymmetric, states, train_convex, seed)
        routed_scores = hierarchy.kidney_scores(routed_probability, patients, kidneys)[dev_convex]
        target = kidneys.iloc[dev_convex].reset_index(drop=True)
        frame = target[["patient_alias", "kidney_alias", "side", "kidney_APN"]].rename(
            columns={"kidney_APN": "label"}).copy()
        frame["score"] = routed_scores
        frame["seed"] = seed
        frame["training_route"] = "convex_only"
        outputs.append(frame)
        result = score(frame) | interval(frame, seed)
        fixed = aligned(FIXED, "replace_with_tile_two_heads", seed, target)
        uri = aligned(URI, "official_uri_embedding_folds", seed, target)
        old_hierarchy = original_score[np.isin(dev_all, dev_convex)]
        reconstructed = .75 * (.75 * fixed + .25 * old_hierarchy) + .25 * uri
        expected_final = aligned(FINAL, "official_uri_fixed_quarter_reference", seed, target)
        final_error = float(np.max(np.abs(reconstructed - expected_final)))
        assert final_error < 1e-5, (seed, final_error)
        partial = .75 * (.75 * fixed + .25 * routed_scores) + .25 * uri
        partial_frame = frame.copy()
        partial_frame["score"] = partial
        partial_result = score(partial_frame) | interval(partial_frame, seed)
        frame["old_final_score"] = expected_final
        frame["partial_routed_fusion_score"] = partial
        report["draws"].append({"seed": seed, "full_old_branch_reproduction_max_abs_error": max_error,
                                "full_old_final_reconstruction_max_abs_error": final_error,
                                "convex_train_patients": len(train_convex),
                                "convex_development": result,
                                "partial_routed_fusion_convex_development": partial_result,
                                "head_inner_cv": {name: {"best_score": float(fit.best_score_),
                                                         "best_params": fit.best_params_}
                                                  for name, fit in models.items()}})
        print(seed, len(train_convex), result, partial_result, flush=True)
    report["three_draw_mean"] = {name: float(np.mean([draw["convex_development"][name]
                                                      for draw in report["draws"]]))
                                 for name in ("auroc", "ap")}
    report["partial_routed_fusion_three_draw_mean"] = {
        name: float(np.mean([draw["partial_routed_fusion_convex_development"][name]
                             for draw in report["draws"]]))
        for name in ("auroc", "ap")}
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(outputs, ignore_index=True).to_csv(OUT / "development_predictions.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("three_draw_mean", report["three_draw_mean"])


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
