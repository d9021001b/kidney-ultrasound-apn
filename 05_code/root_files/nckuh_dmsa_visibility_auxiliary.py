"""Train a matched auxiliary visibility ablation after blinded review is returned."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from torch import nn

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_fullcohort_parenchyma_mil_20261103 as mil
from build_nckuh_visibility_training_pilot import OUT as PILOT
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_inductive_graph import OUT as GRAPH

OUT = base.ROOT / "02_experiments/NCKUH_DMSA_visibility_auxiliary_20261103"
KEYS = ["case_id", "side", "view_type", "page_file", "image_slot"]


def validate_review(review, template):
    if list(review.columns) != list(template.columns):
        raise ValueError("Review columns do not match the blinded template")
    review = review.fillna("").copy()
    if review.duplicated(KEYS).any() or set(map(tuple, review[KEYS].to_numpy())) != set(
            map(tuple, template[KEYS].to_numpy())):
        raise ValueError("Review rows do not match the assigned image slots")
    for col, choices in (("readability", ["readable", "limited", "unreadable"]),
                         ("visible_finding", ["present", "absent", "uncertain"])):
        review[col] = review[col].astype(str).str.strip().str.lower()
        if not review[col].isin(choices).all():
            raise ValueError(f"Incomplete or invalid {col}")
    if (review.readability.eq("unreadable") & ~review.visible_finding.eq("uncertain")).any():
        raise ValueError("Unreadable views must be marked uncertain, not negative")
    confidence = pd.to_numeric(review.confidence_1_to_5, errors="coerce")
    if not confidence.isin(range(1, 6)).all():
        raise ValueError("Every reviewed view requires a confidence from 1 to 5")
    present = review.visible_finding.eq("present")
    for col in ("finding_type", "region_on_display"):
        if review.loc[present, col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Visible findings require {col}")
    return review


def balanced_bce(logits, target):
    count = target.sum()
    weights = torch.where(target > 0, len(target) / (2 * count),
                          len(target) / (2 * (len(target) - count)))
    return (nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none") * weights).mean()


def fit_head(bags, masks, fit, kidney_y, aux, aux_y, seed, strength, epochs=40):
    torch.manual_seed(seed)
    model = mil.ParenchymaMIL("gated").to(bags.device)
    visibility = nn.Linear(32, 1).to(bags.device)
    optimizer = torch.optim.AdamW([*model.parameters(), *visibility.parameters()],
                                  lr=.001, weight_decay=.001)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(bags[fit], masks[fit])[0]
        loss = balanced_bce(logits, kidney_y)
        if strength:
            pooled = model(bags[aux], masks[aux])[2]
            loss = loss + strength * balanced_bce(visibility(pooled).squeeze(-1), aux_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    return model.eval()


def run(review_path):
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.backends.cuda.matmul.allow_tf32 = False
    review = validate_review(pd.read_csv(review_path), pd.read_csv(PILOT / "blinded_packet/view_review.csv"))
    key = pd.read_csv(PILOT / "private_key/view_to_image_PRIVATE.csv")
    review = review.merge(key, on=KEYS, validate="one_to_one")
    if len(review) != len(key):
        raise ValueError("Missing image linkage")
    usable = review[~review.readability.eq("unreadable") & ~review.visible_finding.eq("uncertain")]
    support = usable.groupby("visible_finding").patient_alias.nunique()
    if any(support.get(label, 0) < 5 for label in ("present", "absent")):
        raise ValueError("Need at least five distinct reviewed patients per visibility class")
    _, kidneys, images = mil.load_reference()
    qc = pd.read_csv(mil.OUT / "deidentified_patch_qc.csv")
    np.testing.assert_array_equal(qc.image_id, images.image_id)
    values, patch_patients, kidney_groups = mil.patch_data(images, kidneys)
    offsets = np.load(mil.OUT / "patch_offsets.npy")
    loc = pd.Series(np.arange(len(images)), index=images.image_id).loc[usable.image_id].to_numpy()
    np.testing.assert_array_equal(images.patient_alias.iloc[loc], usable.patient_alias)
    groups = kidney_groups + [np.arange(offsets[i], offsets[i + 1]) for i in loc]
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    reference = pd.read_csv(GRAPH / "development_predictions.csv")
    reference = reference[reference.group.eq("knn_fusion")].set_index(["seed", "kidney_alias"])
    records, models = [], {}
    for seed in base.SEEDS:
        roles = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        assert review.patient_alias.map(roles).eq("train").all()
        fit = np.flatnonzero(kidneys.patient_alias.map(roles).eq("train"))
        dev = np.flatnonzero(kidneys.patient_alias.map(roles).eq("development"))
        target = kidneys.iloc[dev]
        assert target.patient_alias.nunique() == 42 and kidneys.iloc[fit].patient_alias.nunique() == 126
        bags, mask, transform = mil.prepare_bags(values, patch_patients, groups,
                                                 set(kidneys.iloc[fit].patient_alias), seed)
        aux = np.arange(len(kidneys), len(groups))
        aux_y = torch.tensor(usable.visible_finding.eq("present").to_numpy(float), device="cuda", dtype=torch.float32)
        kidney_y = torch.tensor(kidneys.iloc[fit].kidney_APN.to_numpy(float), device="cuda", dtype=torch.float32)
        ref = reference.loc[[(seed, alias) for alias in target.kidney_alias]]
        np.testing.assert_array_equal(ref.label, target.kidney_APN)
        scores = {"fixed_reference": ref.score.to_numpy()}
        for name, strength in (("apn_only", 0.0), ("apn_plus_visibility", 0.3)):
            model = fit_head(bags, mask, fit, kidney_y, aux, aux_y, seed, strength)
            with torch.inference_mode():
                scores[name] = model(bags[dev], mask[dev])[0].sigmoid().cpu().numpy()
            scores[name + "_fusion"] = .75 * scores["fixed_reference"] + .25 * scores[name]
            models[(seed, name)] = {k: v.cpu() for k, v in model.state_dict().items()}
        models[(seed, "pca")] = transform
        for group, score in scores.items():
            records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=group, label=target.kidney_APN.astype(int), score=score))
    pred = pd.concat(records, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), base.SEEDS, "fixed_reference")
    _, controlled, _ = cluster_intervals(pred, ["apn_only", "apn_plus_visibility"],
                                         base.SEEDS, "apn_only")
    _, fused, _ = cluster_intervals(pred, ["apn_only_fusion", "apn_plus_visibility_fusion"],
                                     base.SEEDS, "apn_only_fusion")
    OUT.mkdir()
    pred.to_csv(OUT / "development_predictions.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    pd.concat([delta, controlled, fused]).to_csv(OUT / "paired_delta_ci.csv", index=False)
    import joblib
    joblib.dump(models, OUT / "models.joblib", compress=3)
    print(ci.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("completed_review", type=Path)
    with threadpool_limits(limits=4):
        run(parser.parse_args().completed_review)
