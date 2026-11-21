"""DINOv3 paired-view four-state model with age/sex disease heads."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
from evaluate_nckuh_383_clinical_hierarchy_20261111 import clinical_table, fit_head


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
DINO = ROOT / "02_experiments/NCKUH_revised_383_dinov3_20261111"
OUT = ROOT / "02_experiments/NCKUH_revised_383_dinov3_clinical_hierarchy_20261112"
SEED = 20261112


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve DINOv3 clinical hierarchy experiment")
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    assignments = pd.read_csv(REGISTRY / "patient_assignments.csv")
    active = assignments[assignments.role.isin(["train", "development"]) &
                         assignments.patient_alias.isin(images.patient_alias)].reset_index(drop=True)
    active_images = images[images.patient_alias.isin(active.patient_alias)].reset_index(drop=True)
    active_kidneys = kidneys[kidneys.patient_alias.isin(active.patient_alias)].reset_index(drop=True)
    active_kidneys = active_kidneys.merge(active[["patient_alias", "role"]],
                                          on="patient_alias", validate="many_to_one")
    assert active.role.value_counts().to_dict() == {"train": 204, "development": 70}
    image_rows = images.index[images.patient_alias.isin(active.patient_alias)].to_numpy()
    dino = np.load(DINO / "dinov3_class.npy")[image_rows]
    pairs = base.patient_pair_features(dino, active_images, active)
    symmetric, antisymmetric = hierarchy.structured_views(pairs)
    clinical = clinical_table(active.patient_alias)
    disease_features = np.c_[clinical, symmetric]
    states = active.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    train = np.flatnonzero(active.role.eq("train"))
    any_label = states != hierarchy.STATE_CODE["N"]
    any_model = fit_head(disease_features[train], any_label[train].astype(int), SEED)
    affected = train[any_label[train]]
    bilateral = states == hierarchy.STATE_CODE["B"]
    bilateral_model = fit_head(disease_features[affected], bilateral[affected].astype(int), SEED + 101)
    unilateral = train[np.isin(states[train], [hierarchy.STATE_CODE["L"], hierarchy.STATE_CODE["R"]])]
    left = states == hierarchy.STATE_CODE["L"]
    laterality_model = fit_head(antisymmetric[unilateral], left[unilateral].astype(int), SEED + 202)
    p_any = any_model.predict_proba(disease_features)[:, 1]
    p_b_given = bilateral_model.predict_proba(disease_features)[:, 1]
    p_left = laterality_model.predict_proba(antisymmetric)[:, 1]
    p_b = p_any * p_b_given
    p_u = p_any * (1 - p_b_given)
    probability = np.c_[1 - p_any, p_u * p_left, p_u * (1 - p_left), p_b]
    assert np.allclose(probability.sum(axis=1), 1)
    dev = np.flatnonzero(active_kidneys.role.eq("development"))
    target = active_kidneys.iloc[dev].reset_index(drop=True)
    labels = target.kidney_APN.to_numpy(int)
    score = hierarchy.kidney_scores(probability, active, active_kidneys)[dev]
    clusters = [np.flatnonzero(target.patient_alias.eq(pid)) for pid in sorted(target.patient_alias.unique())]
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
             for _ in range(2000)]
    draws = [idx for idx in draws if np.unique(labels[idx]).size == 2]
    auc = np.asarray([roc_auc_score(labels[idx], score[idx]) for idx in draws])
    ap = np.asarray([average_precision_score(labels[idx], score[idx]) for idx in draws])
    metrics = {"group": "dinov3_view_clinical_four_state", "patients": target.patient_alias.nunique(),
               "kidneys": len(target), "prevalence": labels.mean(),
               "auroc": roc_auc_score(labels, score), "ap": average_precision_score(labels, score),
               "auroc_ci_low": np.percentile(auc, 2.5), "auroc_ci_high": np.percentile(auc, 97.5),
               "ap_ci_low": np.percentile(ap, 2.5), "ap_ci_high": np.percentile(ap, 97.5)}
    pd.DataFrame([metrics]).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame({"patient_alias": target.patient_alias, "kidney_alias": target.kidney_alias,
                  "side": target.side, "label": labels, "score": score}).to_csv(
                      OUT / "development_predictions.csv", index=False)
    pd.DataFrame([
        {"head": "any_apn", "inner_cv_auroc": any_model.best_score_, **any_model.best_params_},
        {"head": "bilateral_given_any", "inner_cv_auroc": bilateral_model.best_score_, **bilateral_model.best_params_},
        {"head": "left_given_unilateral", "inner_cv_auroc": laterality_model.best_score_, **laterality_model.best_params_},
    ]).to_csv(OUT / "training_inner_cv_selections.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "representation": "DINOv3 per-image class features retaining longitudinal/transverse summaries per side",
        "disease_heads": "age/sex plus side-symmetric DINOv3 features",
        "laterality_head": "signed left-minus-right DINOv3 features only",
        "selection": "4-fold training-only stratified CV for each head",
        "test_scored": False,
        "uncertainty": "2000 development patient-cluster bootstrap draws"}, indent=2), encoding="utf-8")
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
