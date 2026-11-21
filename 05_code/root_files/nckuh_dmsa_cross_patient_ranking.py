"""Matched cross-patient pairwise logistic ranking on fixed development draws."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
from audit_nckuh_fullcohort_ranking_20261103 import REFERENCE, state_breakdown
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_fullcohort_radimagenet_finetune_20261103 import load_reference
from nckuh_dmsa_triplet_rank_20261103 import fit_transformer, kidney_summary_features

OUT = base.ROOT / "02_experiments/NCKUH_DMSA_cross_patient_ranking_20261103"
ARMS = ("bce", "all_cross_patient", "matched_uniform", "matched_weighted")


def pair_indices(rows, arm):
    y = rows.kidney_APN.to_numpy()
    patient = rows.patient_alias.to_numpy()
    p, n = np.where((y[:, None] == 1) & (y[None, :] == 0)
                    & (patient[:, None] != patient[None, :]))
    if arm.startswith("matched_"):
        match = rows.match_key.to_numpy()
        keep = match[p] == match[n]
        p, n = p[keep], n[keep]
    hard = rows.DMSA_APN.isin(["L", "R"]).to_numpy()[n]
    weights = 1.0 + 2.0 * hard if arm == "matched_weighted" else np.ones(len(p))
    return p, n, weights, hard


def fit_head(x, rows, arm):
    y = rows.kidney_APN.to_numpy(float)
    x = np.c_[x.astype(float), np.ones(len(x))]
    obs = np.where(y == 1, 0.5 / y.sum(), 0.5 / (len(y) - y.sum()))
    p, n, w, hard = pair_indices(rows, arm)
    strength = 0.0 if arm == "bce" else 0.5
    if strength and not len(p):
        raise ValueError("No eligible cross-patient pairs")
    if strength:
        w *= strength / w.sum()

    def objective(beta):
        logits = x @ beta
        loss = obs @ (np.logaddexp(0, logits) - y * logits)
        grad = obs * (expit(logits) - y)
        if strength:
            delta = logits[p] - logits[n]
            loss += w @ np.logaddexp(0, -delta)
            d = w * expit(-delta)
            grad += np.bincount(n, d, len(x)) - np.bincount(p, d, len(x))
        penalty = np.r_[beta[:-1], 0.0]
        return loss + 0.05 * (penalty @ penalty), x.T @ grad + 0.1 * penalty

    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                      options={"gtol": 1e-8})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x, int(len(p) if strength else 0), int(hard.sum() if strength else 0)


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    _, kidneys, images = load_reference()
    manifest = pd.read_csv(base.OUT / "deidentified_source_manifest.csv")
    qc = pd.read_csv(base.OUT / "deidentified_extraction_qc.csv")
    aligned = images[["image_id", "kidney_alias", "view"]].merge(
        manifest[["image_id", "source_type"]], on="image_id", validate="one_to_one").merge(
        qc[["image_id", "frame_width", "frame_height"]], on="image_id", validate="one_to_one")
    assert len(aligned) == len(images)
    aligned["source_view"] = (aligned.view.str[-1] + ":" + aligned.source_type + ":"
                              + aligned.frame_width.astype(str) + "x" + aligned.frame_height.astype(str))
    signature = aligned.groupby("kidney_alias").source_view.agg(lambda a: "|".join(sorted(a)))
    kidneys["match_key"] = kidneys.side + ":" + kidneys.kidney_alias.map(signature)
    assert kidneys.match_key.notna().all()
    embedding = np.load(base.OUT / "radimagenet_color.npy")
    assert images.image_id.tolist() == manifest.image_id.tolist()
    raw = kidney_summary_features(embedding, images, kidneys)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    reference = pd.read_csv(base.OUT / "instance_development_predictions.csv").query(
        "group == @REFERENCE").set_index(["seed", "kidney_alias"])
    predictions, metrics, pairs, states, models = [], [], [], [], {}
    for seed in base.SEEDS:
        roles = kidneys.patient_alias.map(assignments[assignments.seed.eq(seed)].set_index("patient_alias").role)
        train = np.flatnonzero(roles.eq("train"))
        dev = np.flatnonzero(roles.eq("development"))
        assert kidneys.iloc[train].patient_alias.nunique() == 126
        assert kidneys.iloc[dev].patient_alias.nunique() == 42
        transform, values = fit_transformer(raw, train, seed)
        ref = reference.loc[[(seed, key) for key in kidneys.iloc[dev].kidney_alias]]
        assert np.array_equal(ref.label, kidneys.iloc[dev].kidney_APN)
        scores = {REFERENCE: ref.score.to_numpy()}
        heads = {}
        for arm in ARMS:
            beta, count, hard_count = fit_head(values[train], kidneys.iloc[train], arm)
            heads[arm] = beta
            pairs.append(dict(seed=seed, group=arm, training_pairs=count, unilateral_negative_pairs=hard_count))
            scores[arm] = expit(values[dev] @ beta[:-1] + beta[-1])
            scores[arm + "_fusion"] = 0.25 * scores[arm] + 0.75 * scores[REFERENCE]
        models[seed] = (transform, heads)
        for group, score in scores.items():
            frame = kidneys.iloc[dev][["patient_alias", "kidney_alias", "side", "DMSA_APN", "kidney_APN"]].rename(
                columns={"kidney_APN": "label"}).assign(seed=seed, group=group, score=score)
            predictions.append(frame)
            metrics.append(dict(seed=seed, group=group, **base.metric(frame.label, score)))
            states.extend(dict(seed=seed, group=group, **row) for row in state_breakdown(frame))
    pred = pd.concat(predictions, ignore_index=True)
    groups = list(dict.fromkeys(pred.group))
    ci, delta, _ = cluster_intervals(pred, groups, base.SEEDS, REFERENCE)
    _, matched_delta, _ = cluster_intervals(pred, ["matched_uniform", "matched_weighted"],
                                            base.SEEDS, "matched_uniform")
    _, fusion_delta, _ = cluster_intervals(pred, ["matched_uniform_fusion", "matched_weighted_fusion"],
                                           base.SEEDS, "matched_uniform_fusion")
    OUT.mkdir()
    for name, frame in (("development_predictions", pred), ("development_metrics", pd.DataFrame(metrics)),
                        ("development_ci", ci), ("paired_reference_delta_ci", delta),
                        ("paired_uniform_head_delta_ci", matched_delta),
                        ("paired_uniform_fusion_delta_ci", fusion_delta),
                        ("training_pairs", pd.DataFrame(pairs)), ("development_state_pairs", pd.DataFrame(states))):
        frame.to_csv(OUT / f"{name}.csv", index=False)
    joblib.dump(models, OUT / "models.joblib", compress=3)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())
    print(pd.DataFrame(pairs).to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
