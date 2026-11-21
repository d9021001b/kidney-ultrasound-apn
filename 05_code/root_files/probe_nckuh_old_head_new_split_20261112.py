"""Re-fit the unchanged old RadImageNet color image head on the revised split."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import nckuh_dmsa_fullcohort_fullframe_20261103 as old
from nckuh_dmsa_kidney_baseline_20261102 import load_reference


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
NEW = EXP / "NCKUH_revised_383_audit_20261111"
OUT = EXP / "NCKUH_revised_383_root_cause_20261112"


def evaluate(images, kidneys, values, roles, seed):
    train_images = np.flatnonzero(images.patient_alias.map(roles).eq("train"))
    dev_kidneys = np.flatnonzero(kidneys.patient_alias.map(roles).eq("development"))
    assert len(train_images) > 0 and len(dev_kidneys) > 0
    image_labels = images.kidney_APN.to_numpy(int)
    weights = old.image_weights(images, train_images, image_labels)
    model = old.fit_image_head(values, train_images, image_labels, weights, seed)
    image_scores = model.predict_proba(values)[:, 1]
    score = old.aggregate_image_logits(
        image_scores, images, kidneys.kidney_alias.iloc[dev_kidneys]
    )
    result = old.metric(kidneys.kidney_APN.iloc[dev_kidneys].to_numpy(int), score)
    return {
        "train_patients": int(images.patient_alias.iloc[train_images].nunique()),
        "train_images": len(train_images),
        "dev_patients": int(kidneys.patient_alias.iloc[dev_kidneys].nunique()),
        "dev_kidneys": len(dev_kidneys),
        **result,
    }


def main():
    _, kidneys, images = load_reference()
    images = images[images.kidney_APN.notna()].copy().reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].copy().reset_index(drop=True)
    old_manifest = pd.read_csv(old.OUT / "deidentified_source_manifest.csv")
    assert images.image_id.tolist() == old_manifest.image_id.tolist()
    values = np.load(old.OUT / "radimagenet_color.npy")
    assert values.shape == (820, 1024)

    old_assign = pd.read_csv(old.OUT / "patient_assignments.csv")
    historical = {}
    saved = pd.read_csv(old.OUT / "instance_development_metrics.csv")
    for seed in old.SEEDS:
        roles = old_assign[old_assign.seed.eq(seed)].set_index("patient_alias").role
        historical[str(seed)] = evaluate(images, kidneys, values, roles, seed)
        reference = saved[
            saved.seed.eq(seed)
            & saved.group.eq("fullcohort_rad_color_image_logit_mean")
        ].iloc[0]
        for metric_name in ("auroc", "ap", "brier"):
            assert abs(historical[str(seed)][metric_name] - reference[metric_name]) < 1e-12

    old_patients = pd.read_csv(
        EXP / "NCKUH_DMSA_label_audit_20261102/patient_label_linkage_private.csv",
        dtype={"patient_id": str},
    )
    new_patients = pd.read_csv(NEW / "patient_linkage_private.csv", dtype={"patient_id": str})
    new_roles = pd.read_csv(NEW / "patient_assignments.csv")
    ready = set(pd.read_csv(NEW / "contour_kidney_manifest.csv").patient_alias)
    role_by_new_alias = new_roles.set_index("patient_alias").role
    role_by_id = {
        row.patient_id: role_by_new_alias[row.patient_alias]
        for row in new_patients.itertuples(index=False)
        if row.patient_alias in ready
    }
    roles = old_patients.set_index("patient_alias").patient_id.map(role_by_id)
    roles.index = old_patients.patient_alias
    selected_images = images[images.patient_alias.map(roles).notna()].copy()
    selected_values = values[selected_images.index.to_numpy()]
    selected_images = selected_images.reset_index(drop=True)
    selected_kidneys = kidneys[kidneys.patient_alias.map(roles).notna()].reset_index(drop=True)
    revised = evaluate(selected_images, selected_kidneys, selected_values, roles, 20261111)
    result = {
        "model": "unchanged old RadImageNet color image-instance PCA32/LR; logit-mean kidney aggregation",
        "old_three_draws": historical,
        "new_split_old_cohort_only": revised,
        "scope": "Role-sensitivity probe only. It does not reproduce the full old fixed ensemble or score the new exploratory test.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "old_head_new_split.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
