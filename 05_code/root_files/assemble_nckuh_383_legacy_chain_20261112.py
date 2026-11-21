"""Assemble the original fixed weights on revised development predictions."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
OUT = EXP / "NCKUH_revised_383_root_cause_20261112"


def fixed_chain(deep, mil, neighbors, texture):
    return .75 * (.75 * (.75 * deep + .25 * mil) + .25 * neighbors) + .25 * texture


def align(path, group, target, seed=None, label_name="label"):
    frame = pd.read_csv(path)
    if seed is not None:
        frame = frame[frame.seed.eq(seed)]
    if group is not None:
        frame = frame[frame.group.eq(group)]
    frame = frame.set_index("kidney_alias", verify_integrity=True).loc[target.kidney_alias]
    assert np.array_equal(frame[label_name].to_numpy(int), target.kidney_APN.to_numpy(int))
    assert np.array_equal(frame.patient_alias.to_numpy(), target.patient_alias.to_numpy())
    return frame.score.to_numpy(float)


def old_reproduction():
    seed = 20261102
    source = EXP / "NCKUH_DMSA_tile_replacement_20261103/development_predictions.csv"
    frame = pd.read_csv(source)
    target = frame[frame.seed.eq(seed) & frame.group.eq("replace_with_tile_two_heads")][
        ["kidney_alias", "patient_alias", "label"]
    ].rename(columns={"label": "kidney_APN"}).reset_index(drop=True)
    fixed = align(source, "replace_with_tile_two_heads", target, seed)
    hierarchy = align(
        EXP / "NCKUH_DMSA_hierarchical_tile_usfm_20261104/development_predictions.csv",
        "hierarchical_tile_usfm", target, seed,
    )
    uri = align(
        EXP / "NCKUH_DMSA_uricads_official_frozen_20261105/development_predictions.csv",
        "official_uri_embedding_folds", target, seed,
    )
    hierarchical = .75 * fixed + .25 * hierarchy
    expected_hierarchical = align(
        EXP / "NCKUH_DMSA_hierarchical_tile_usfm_20261104/development_predictions.csv",
        "hierarchical_tile_usfm_quarter_plus_fixed", target, seed,
    )
    final = .75 * hierarchical + .25 * uri
    expected_final = align(
        EXP / "NCKUH_DMSA_uri_lightgbm_factorized_20261107/development_predictions.csv",
        "official_uri_fixed_reference", target, seed,
    )
    errors = {
        "hierarchical_weighted_score": float(np.max(np.abs(hierarchical - expected_hierarchical))),
        "full_final_score": float(np.max(np.abs(final - expected_final))),
    }
    assert max(errors.values()) < 1e-12
    return errors


def metric(labels, scores):
    return {"auroc": float(roc_auc_score(labels, scores)),
            "ap": float(average_precision_score(labels, scores))}


def paired_bootstrap(target, old_score, new_score):
    groups = defaultdict(list)
    for row, a, b in zip(target.itertuples(index=False), old_score, new_score):
        groups[row.patient_alias].append((int(row.kidney_APN), float(a), float(b)))
    bags = list(groups.values())
    rng = np.random.default_rng(20261112)
    deltas = []
    for _ in range(3000):
        draw = np.asarray([
            value for index in rng.integers(0, len(bags), len(bags))
            for value in bags[index]
        ])
        if np.unique(draw[:, 0]).size < 2:
            continue
        old, new = metric(draw[:, 0], draw[:, 1]), metric(draw[:, 0], draw[:, 2])
        deltas.append((new["auroc"] - old["auroc"], new["ap"] - old["ap"]))
    interval = np.quantile(np.asarray(deltas), [0.025, 0.975], axis=0)
    return {"valid_resamples": len(deltas),
            "new_minus_legacy_auroc_95pct": interval[:, 0].tolist(),
            "new_minus_legacy_ap_95pct": interval[:, 1].tolist()}


def main():
    reproduction = old_reproduction()
    assignments = pd.read_csv(
        EXP / "NCKUH_revised_383_audit_20261111/patient_assignments.csv"
    ).set_index("patient_alias")
    kidneys = pd.read_csv(
        EXP / "NCKUH_revised_383_audit_20261111/contour_kidney_manifest.csv"
    )
    target = kidneys[kidneys.patient_alias.map(assignments.role).eq("development")].reset_index(drop=True)
    assert target.patient_alias.nunique() == 70 and len(target) == 140
    components = EXP / "NCKUH_revised_383_root_cause_20261112/legacy_fixed_component_predictions.csv"
    scores = {
        name: align(components, name, target, label_name="kidney_APN")
        for name in ("tile_two_heads", "reflection_average", "knn", "aligned")
    }
    scores["cross_attention_q4"] = align(
        OUT / "legacy_mil_development_predictions.csv", None, target, label_name="kidney_APN"
    )
    scores["hierarchical_tile_usfm"] = align(
        OUT / "legacy_hierarchy_development_predictions.csv", None, target,
        label_name="kidney_APN",
    )
    scores["official_uri_embedding_folds"] = align(
        EXP / "NCKUH_revised_383_uricads_20261111/development_predictions.csv",
        "uri_embedding_folds", target,
    )
    scores["old_fixed_chain"] = fixed_chain(
        scores["tile_two_heads"], scores["cross_attention_q4"],
        scores["knn"], scores["aligned"],
    )
    scores["old_fixed_plus_hierarchy"] = (
        .75 * scores["old_fixed_chain"] + .25 * scores["hierarchical_tile_usfm"]
    )
    scores["old_full_final"] = (
        .75 * scores["old_fixed_plus_hierarchy"] + .25 * scores["official_uri_embedding_folds"]
    )
    scores["new_revised_stack"] = align(
        EXP / "NCKUH_revised_383_oof_hierarchy_stacking_20261112/development_predictions.csv",
        None, target,
    )
    labels = target.kidney_APN.to_numpy(int)
    metrics = {name: metric(labels, value) for name, value in scores.items()}
    old_registry = pd.read_csv(
        EXP / "NCKUH_DMSA_label_audit_20261102/patient_label_linkage_private.csv",
        dtype={"patient_id": str},
    )
    old_labels = old_registry.set_index("patient_id").DMSA_APN
    linkage = pd.read_csv(
        EXP / "NCKUH_revised_383_audit_20261111/patient_linkage_private.csv",
        dtype={"patient_id": str},
    ).set_index("patient_alias")
    source = []
    for alias in target.patient_alias:
        patient_id = linkage.loc[alias, "patient_id"]
        if patient_id not in old_labels.index:
            source.append("new_registry")
        elif pd.isna(old_labels.loc[patient_id]):
            source.append("old_unlabeled")
        else:
            source.append("old_labeled")
    source = np.asarray(source)
    subgroup_metrics = {}
    for name, selected in (
        ("old_labeled", source == "old_labeled"),
        ("old_unlabeled", source == "old_unlabeled"),
        ("new_registry", source == "new_registry"),
        ("outside_old_labeled", source != "old_labeled"),
    ):
        subset = target.loc[selected].reset_index(drop=True)
        subgroup_metrics[name] = {
            "patients": subset.patient_alias.nunique(),
            "kidneys": len(subset),
            "old_full_final": metric(labels[selected], scores["old_full_final"][selected]),
            "new_revised_stack": metric(labels[selected], scores["new_revised_stack"][selected]),
            "paired_new_minus_old": paired_bootstrap(
                subset, scores["old_full_final"][selected],
                scores["new_revised_stack"][selected],
            ),
        }
    report = {
        "old_draw_formula_reproduction_max_abs": reproduction,
        "train_patients": 204,
        "development_patients": 70,
        "development_kidneys": 140,
        "metrics": metrics,
        "subgroups": subgroup_metrics,
        "new_minus_old_final": {
            "auroc": metrics["new_revised_stack"]["auroc"] - metrics["old_full_final"]["auroc"],
            "ap": metrics["new_revised_stack"]["ap"] - metrics["old_full_final"]["ap"],
            **paired_bootstrap(target, scores["old_full_final"], scores["new_revised_stack"]),
        },
        "test_scored": False,
        "adaptations": [
            "Four-query MIL inner patient split scaled from 100/26 to 162/42 for 204 training patients; all epoch candidates and final weights unchanged.",
            "All other fixed branches, four-state head, URI head, and original nested weights were refit on revised training roles only.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "legacy_full_chain.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    output = target[["patient_alias", "kidney_alias", "side", "kidney_APN"]].copy()
    for name, score in scores.items():
        output[name] = score
    output.to_csv(OUT / "legacy_full_chain_development_predictions.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
