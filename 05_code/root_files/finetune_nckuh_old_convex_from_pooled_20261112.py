"""Design A continuation: convex-only fine-tuning that starts from the pooled initial model.

For each historical draw the pooled tile+USFM hierarchical branch is fitted on all
126 training patients exactly as before (reproduction is asserted). Its fitted
preprocessing (imputer, variance filter, scaler, PCA) is frozen as the pre-trained
representation. Each logistic head is then fine-tuned on convex-only training
patients, starting from the pooled coefficients and penalising departure from
them (L2 toward the starting point):

    minimise  C * sum_i w_i * logloss_i  +  0.5 * ||theta - theta_pooled||^2

C is chosen from the original grid by four-fold CV inside the convex-only training
patients only. Development patients are never used for fitting or selection, and
no exploratory-test APN label is read. The fixed-chain and URI-CADS branches are
archived scores from the pooled fit and are not updated.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_hierarchical_tile_usfm_20261104 as old_head
import nckuh_dmsa_multiquery_mil_20261103 as cohort
from audit_nckuh_old_draw_probe_routing_20261112 import (
    LABELS, OLD, OUT as ROUTE_AUDIT, SEEDS, WORKBOOK, interval, score,
)
from prepare_nckuh_383_contour_cohort_20261111 import digest
from refit_nckuh_old_convex_hierarchy_20261112 import FINAL, FIXED, GROUP, SOURCE, URI, aligned


OUT = ROUTE_AUDIT / "convex_finetune_from_pooled"
C_GRID = (0.01, 0.1, 1.0)
STATE = hierarchy.STATE_CODE


def finetune(z, y, start, c):
    """Logistic fit pulled toward `start` = [coef..., intercept]; balanced class weights."""
    weight = np.where(y == 1, len(y) / (2.0 * y.sum()), len(y) / (2.0 * (len(y) - y.sum())))
    design = np.c_[z, np.ones(len(z))]
    sign = 2.0 * y - 1.0

    def objective(theta):
        margin = sign * (design @ theta)
        loss = c * np.sum(weight * np.logaddexp(0.0, -margin)) + 0.5 * np.sum((theta - start) ** 2)
        gradient = design.T @ (-c * weight * sign * expit(-margin)) + (theta - start)
        return loss, gradient

    fitted = minimize(objective, start.copy(), jac=True, method="L-BFGS-B",
                      options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8})
    assert fitted.success, fitted.message
    return fitted.x


def finetune_head(search, values, labels, selected, seed):
    """Freeze the pooled head's preprocessing; fine-tune its logistic layer on `selected`."""
    pipeline = search.best_estimator_
    z_all = pipeline[:-1].transform(values)
    lr = pipeline.named_steps["lr"]
    start = np.r_[lr.coef_.ravel(), lr.intercept_]
    assert np.allclose(expit(np.c_[z_all, np.ones(len(z_all))] @ start),
                       pipeline.predict_proba(values)[:, 1], atol=1e-6)
    z, y = z_all[selected], labels[selected]
    folds = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=seed).split(z, y))
    cv = {}
    for c in C_GRID:
        aucs = []
        for fit_idx, held_idx in folds:
            theta = finetune(z[fit_idx], y[fit_idx], start, c)
            aucs.append(roc_auc_score(y[held_idx], z[held_idx] @ theta[:-1] + theta[-1]))
        cv[c] = float(np.mean(aucs))
    best = max(C_GRID, key=lambda c: (cv[c], -c))  # ties: stay closer to the pooled model
    theta = finetune(z, y, start, best)
    info = {"selected_C": best, "inner_cv_auroc": cv, "fine_tune_patients": int(len(selected)),
            "pooled_C": float(search.best_params_["lr__C"]),
            "pooled_pca_components": int(search.best_params_["pca__n_components"]),
            "relative_coefficient_change": float(np.linalg.norm(theta - start) / np.linalg.norm(start))}
    return expit(z_all @ theta[:-1] + theta[-1]), info


def paired_interval(frame, first, second, draws=10000):
    """Patient-cluster bootstrap of three-draw mean metrics for two score columns and their difference."""
    aliases = sorted(frame.patient_alias.unique())
    bags = {key: group[["label", first, second]].to_numpy(float)
            for key, group in frame.groupby(["seed", "patient_alias"])}
    rng = np.random.default_rng(20261112)
    kept = []
    for _ in range(draws):
        sample = [aliases[k] for k in rng.integers(0, len(aliases), len(aliases))]
        per_draw = []
        for seed in SEEDS:
            rows = [bags[(seed, alias)] for alias in sample if (seed, alias) in bags]
            table = np.concatenate(rows) if rows else np.empty((0, 3))
            if np.unique(table[:, 0]).size != 2:
                break
            per_draw.append([roc_auc_score(table[:, 0], table[:, 1]), average_precision_score(table[:, 0], table[:, 1]),
                             roc_auc_score(table[:, 0], table[:, 2]), average_precision_score(table[:, 0], table[:, 2])])
        if len(per_draw) == len(SEEDS):
            kept.append(np.mean(per_draw, axis=0))
    kept = np.asarray(kept)
    q = lambda v: np.quantile(v, [.025, .975]).tolist()
    return {"valid_bootstraps": len(kept),
            f"{first}_auroc_95pct": q(kept[:, 0]), f"{first}_ap_95pct": q(kept[:, 1]),
            f"{second}_auroc_95pct": q(kept[:, 2]), f"{second}_ap_95pct": q(kept[:, 3]),
            "difference_auroc_95pct": q(kept[:, 0] - kept[:, 2]),
            "difference_ap_95pct": q(kept[:, 1] - kept[:, 3])}


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve completed experiment: {OUT}")
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
    states = patients.DMSA_APN.map(STATE).to_numpy(int)
    assignments = pd.read_csv(OLD / "patient_assignments.csv")
    saved = pd.read_csv(SOURCE)
    saved = saved[saved.group.eq(GROUP)].set_index(["seed", "kidney_alias"], verify_integrity=True)
    is_convex = patients.patient_alias.map(alias_probe).eq("convex").to_numpy()

    report = {"source_sha256": {"workbook": digest(WORKBOOK), "old_predictions": digest(SOURCE),
                                 "old_fixed_predictions": digest(FIXED), "old_uri_predictions": digest(URI),
                                 "old_final_predictions": digest(FINAL),
                                 "old_assignments": digest(OLD / "patient_assignments.csv")},
              "experiment": "Design A continuation. Pooled tile+USFM hierarchical branch (126 training patients, no probe exclusion) is the pre-trained initial model; its preprocessing and PCA are frozen and each logistic head is fine-tuned on convex-only training patients with an L2 penalty toward the pooled coefficients. C from {0.01, 0.1, 1.0} by 4-fold CV within convex-only training patients. Fixed-chain and URI-CADS branches are unchanged archived pooled scores.",
              "test_APN_scored": False, "development_used_for_selection": False, "draws": []}
    outputs = []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train_all = np.flatnonzero(patients.patient_alias.map(mapping).eq("train"))
        train_convex = train_all[is_convex[train_all]]
        dev_all = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        dev_convex = dev_all[kidneys.patient_alias.iloc[dev_all].map(alias_probe).eq("convex").to_numpy()]
        assert len(train_all) == 126 and 110 <= len(train_convex) <= 125 and 70 <= len(dev_convex) <= 82
        assert set(patients.patient_alias.iloc[train_convex]).isdisjoint(kidneys.patient_alias.iloc[dev_convex])

        pooled_probability, models = hierarchy.state_probabilities(symmetric, antisymmetric, states, train_all, seed)
        pooled_all = hierarchy.kidney_scores(pooled_probability, patients, kidneys)
        target_all = kidneys.iloc[dev_all]
        old_saved = saved.loc[[(seed, alias) for alias in target_all.kidney_alias]].score.to_numpy(float)
        reproduction = float(np.max(np.abs(pooled_all[dev_all] - old_saved)))
        assert reproduction < 1e-4, (seed, reproduction)

        affected = train_convex[states[train_convex] != STATE["N"]]
        unilateral = train_convex[np.isin(states[train_convex], [STATE["L"], STATE["R"]])]
        p_any, info_any = finetune_head(models["any"], symmetric, (states != STATE["N"]).astype(int), train_convex, seed)
        p_bil, info_bil = finetune_head(models["bilateral_given_any"], symmetric,
                                        (states == STATE["B"]).astype(int), affected, seed + 101)
        p_left, info_left = finetune_head(models["left_given_unilateral"], antisymmetric,
                                          (states == STATE["L"]).astype(int), unilateral, seed + 202)
        p_uni = p_any * (1.0 - p_bil)
        probability = np.c_[1.0 - p_any, p_uni * p_left, p_uni * (1.0 - p_left), p_any * p_bil]
        probability /= probability.sum(axis=1, keepdims=True)
        tuned_all = hierarchy.kidney_scores(probability, patients, kidneys)

        target = kidneys.iloc[dev_convex].reset_index(drop=True)
        fixed = aligned(FIXED, "replace_with_tile_two_heads", seed, target)
        uri = aligned(URI, "official_uri_embedding_folds", seed, target)
        pooled_final = aligned(FINAL, "official_uri_fixed_quarter_reference", seed, target)
        rebuilt = .75 * (.75 * fixed + .25 * pooled_all[dev_convex]) + .25 * uri
        final_error = float(np.max(np.abs(rebuilt - pooled_final)))
        assert final_error < 1e-5, (seed, final_error)

        frame = target[["patient_alias", "kidney_alias", "side", "kidney_APN"]].rename(columns={"kidney_APN": "label"})
        frame["seed"] = seed
        frame["pooled_branch"] = pooled_all[dev_convex]
        frame["finetuned_branch"] = tuned_all[dev_convex]
        frame["pooled_fusion"] = pooled_final
        frame["finetuned_fusion"] = .75 * (.75 * fixed + .25 * tuned_all[dev_convex]) + .25 * uri
        outputs.append(frame)
        draw = {"seed": seed, "pooled_branch_reproduction_max_abs_error": reproduction,
                "pooled_fusion_reconstruction_max_abs_error": final_error,
                "convex_train_patients": int(len(train_convex)),
                "heads": {"any": info_any, "bilateral_given_any": info_bil, "left_given_unilateral": info_left}}
        for column in ("pooled_branch", "finetuned_branch", "pooled_fusion", "finetuned_fusion"):
            scored = frame.rename(columns={column: "score"})
            draw[column] = score(scored) | interval(scored, seed)
        report["draws"].append(draw)
        print(seed, {c: (round(draw[c]["auroc"], 3), round(draw[c]["ap"], 3))
                     for c in ("pooled_branch", "finetuned_branch", "pooled_fusion", "finetuned_fusion")},
              {h: (v["selected_C"], round(v["relative_coefficient_change"], 3)) for h, v in draw["heads"].items()},
              flush=True)

    combined = pd.concat(outputs, ignore_index=True)
    report["three_draw_mean"] = {
        column: {name: float(np.mean([d[column][name] for d in report["draws"]])) for name in ("auroc", "ap")}
        for column in ("pooled_branch", "finetuned_branch", "pooled_fusion", "finetuned_fusion")}
    report["three_draw_cluster_intervals"] = {
        "fusion": paired_interval(combined, "finetuned_fusion", "pooled_fusion"),
        "branch": paired_interval(combined, "finetuned_branch", "pooled_branch")}
    OUT.mkdir(parents=True)
    combined.to_csv(OUT / "development_predictions.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["three_draw_mean"], indent=1))
    print(json.dumps(report["three_draw_cluster_intervals"], indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
