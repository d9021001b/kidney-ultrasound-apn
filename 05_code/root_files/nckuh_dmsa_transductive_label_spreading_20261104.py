"""Training-selected transductive label spreading on tile plus USFM evidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_tile_usfm_20261104 as representation
import nckuh_dmsa_multiquery_mil_20261103 as cohort
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_triplet_rank_20261103 import fit_transformer, kidney_summary_features


ROOT = Path(__file__).resolve().parents[2]
FIXED = ROOT / "02_experiments/NCKUH_DMSA_tile_replacement_20261103"
OUT = ROOT / "02_experiments/NCKUH_DMSA_transductive_label_spreading_20261104"
K_VALUES = (5, 8, 12, 20)
ALPHAS = (0.2, 0.5, 0.8)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def transductive_predict(
    fit_values: np.ndarray,
    fit_labels: np.ndarray,
    query_values: np.ndarray,
    k: int,
    alpha: float,
    max_iter: int = 200,
) -> tuple[np.ndarray, object]:
    from sklearn.semi_supervised import LabelSpreading

    values = np.vstack([fit_values, query_values])
    labels = np.r_[fit_labels.astype(int), np.full(len(query_values), -1, dtype=int)]
    model = LabelSpreading(
        kernel="knn", n_neighbors=k, alpha=alpha, max_iter=max_iter, tol=1e-5,
    )
    model.fit(values, labels)
    score = model.label_distributions_[len(fit_values):, 1]
    if score.shape != (len(query_values),) or not np.isfinite(score).all():
        raise ValueError("Invalid transductive scores")
    return score, model


def run() -> None:
    from sklearn.model_selection import train_test_split

    if OUT.exists():
        raise FileExistsError(f"Preserve completed experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()

    patients, kidneys, images = cohort.load_reference()
    image_values = representation.image_representations(images)["tile_usfm"]
    raw = kidney_summary_features(image_values, images, kidneys)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    fixed = pd.read_csv(FIXED / "development_predictions.csv")
    fixed = fixed[fixed.group.eq("replace_with_tile_two_heads")].set_index(
        ["seed", "kidney_alias"], verify_integrity=True
    )

    metrics, predictions, selections = [], [], []
    for seed in base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train_patients = patients[
            patients.patient_alias.isin(mapping[mapping.eq("train")].index)
        ].reset_index(drop=True)
        fit_local, select_local = train_test_split(
            np.arange(len(train_patients)), train_size=100,
            random_state=seed + 81104, stratify=train_patients.DMSA_APN,
        )
        fit_alias = train_patients.patient_alias.iloc[fit_local]
        select_alias = train_patients.patient_alias.iloc[select_local]
        fit = np.flatnonzero(kidneys.patient_alias.isin(fit_alias))
        selection = np.flatnonzero(kidneys.patient_alias.isin(select_alias))
        inner_transformer, inner_values = fit_transformer(raw, fit, seed)
        candidates = []
        for k in K_VALUES:
            for alpha in ALPHAS:
                score, _ = transductive_predict(
                    inner_values[fit], labels[fit], inner_values[selection], k, alpha
                )
                result = base.metric(labels[selection], score)
                candidates.append((-(result["auroc"] + result["ap"]) / 2, k, alpha, result))
        _, chosen_k, chosen_alpha, inner_result = min(
            candidates, key=lambda item: (item[0], item[1], item[2])
        )
        selections.append({
            "seed": seed, "chosen_k": chosen_k, "chosen_alpha": chosen_alpha,
            "inner_auroc": inner_result["auroc"], "inner_ap": inner_result["ap"],
            "fit_patients": 100, "selection_patients": 26,
        })

        train = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("train"))
        development = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        target = kidneys.iloc[development]
        transformer, values = fit_transformer(raw, train, seed + 100)
        score, model = transductive_predict(
            values[train], labels[train], values[development], chosen_k, chosen_alpha
        )
        joblib.dump(
            {"transformer": transformer, "label_spreading": model,
             "chosen_k": chosen_k, "chosen_alpha": chosen_alpha},
            OUT / "models" / f"{seed}_label_spreading.joblib", compress=3,
        )
        fixed_rows = fixed.loc[[(seed, alias) for alias in target.kidney_alias]]
        np.testing.assert_array_equal(fixed_rows.label, labels[development])
        reference = fixed_rows.score.to_numpy()
        scores = {
            "fixed_chain_reference": reference,
            "tile_usfm_label_spreading": score,
            "tile_usfm_label_spreading_quarter_plus_fixed": 0.25 * score + 0.75 * reference,
            "tile_usfm_label_spreading_equal_plus_fixed": 0.50 * score + 0.50 * reference,
        }
        for group, candidate in scores.items():
            metrics.append({
                "seed": seed, "role": "development", "group": group,
                "patients": 42, "kidneys": len(development),
                **base.metric(labels[development], candidate),
            })
            for local, index in enumerate(development):
                predictions.append({
                    "seed": seed, "group": group,
                    "patient_alias": kidneys.patient_alias.iloc[index],
                    "kidney_alias": kidneys.kidney_alias.iloc[index],
                    "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                    "side": kidneys.side.iloc[index], "label": int(labels[index]),
                    "score": float(candidate[local]),
                })
        print(
            f"Completed label spreading draw {seed}: k={chosen_k}, alpha={chosen_alpha}",
            flush=True,
        )

    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "training_inner_selection.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    groups = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups, base.SEEDS, "fixed_chain_reference"
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(OUT / "protocol_and_decision.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "motivation": "Test whether unlabeled development kidneys improve manifold geometry beyond independent inductive KNN queries",
        "representation": "Training-fitted PCA32 of contour-constrained RadImageNet tile and frozen full-USFM kidney summaries",
        "method": "sklearn LabelSpreading with KNN kernel; development nodes use label -1",
        "selection": "100 labeled and 26 simulated-unlabeled patients inside each outer training role select k and alpha",
        "transductive_boundary": "Development features jointly define the graph, but development outcomes are never passed to fit or selection",
        "groups_above_both_development_targets": achieved,
        "exploratory_test_scored": False,
    })
    write_json(OUT / "integrity_audit.json", {
        "patients": len(patients), "kidneys": len(kidneys), "images": len(images),
        "prediction_rows": len(prediction_frame), "exploratory_test_rows_scored": 0,
        "scores_finite": bool(np.isfinite(prediction_frame.score).all()),
        "scores_in_unit_interval": bool(prediction_frame.score.between(0, 1).all()),
    })
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()

