"""Development-only four-state URI-CADS heads on the revised NCKUH cohort."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
from prepare_nckuh_383_contour_cohort_20261111 import OUT as REGISTRY, digest


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "02_experiments/NCKUH_revised_383_contour_cache_20261111"
OUT = ROOT / "02_experiments/NCKUH_revised_383_uricads_20261111"
SEED = 20261111


def metric(labels, score):
    return {"auroc": float(roc_auc_score(labels, score)),
            "ap": float(average_precision_score(labels, score)),
            "brier": float(brier_score_loss(labels, score))}


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve URI-CADS development experiment")
    lock = json.loads((REGISTRY / "cohort_lock.json").read_text())
    audit = json.loads((CACHE / "uricads_encoder_audit.json").read_text())
    assert digest(REGISTRY / "contour_image_manifest.csv") == lock["files_sha256"]["contour_image_manifest.csv"]
    assert digest(CACHE / "uricads_fold_embeddings.npy") == audit["embedding_sha256"]
    assert digest(CACHE / "uricads_fold_pathology_scores.npy") == audit["scores_sha256"]
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    assignments = pd.read_csv(REGISTRY / "patient_assignments.csv")
    active = assignments[assignments.role.isin(["train", "development"]) &
                         assignments.patient_alias.isin(images.patient_alias)].reset_index(drop=True)
    assert active.role.value_counts().to_dict() == {"train": 204, "development": 70}
    image_rows = np.flatnonzero(images.patient_alias.isin(active.patient_alias))
    active_images = images.iloc[image_rows].reset_index(drop=True)
    active_kidneys = kidneys[kidneys.patient_alias.isin(active.patient_alias)].reset_index(drop=True)
    active_kidneys = active_kidneys.merge(active[["patient_alias", "role"]],
                                          on="patient_alias", validate="many_to_one")
    train = np.flatnonzero(active.role.eq("train"))
    dev = np.flatnonzero(active_kidneys.role.eq("development"))
    assert active_kidneys.patient_alias.iloc[dev].nunique() == 70
    states = active.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = active_kidneys.kidney_APN.to_numpy(int)
    embeddings = np.load(CACHE / "uricads_fold_embeddings.npy")[image_rows]
    semantic = np.load(CACHE / "uricads_fold_pathology_scores.npy")[image_rows]
    representations = {
        "uri_embedding_folds": embeddings.reshape(len(embeddings), -1),
        "uri_embedding_mean": embeddings.mean(axis=1),
        "uri_semantic_mean": semantic.mean(axis=1),
    }
    predictions, metrics = [], []
    (OUT / "models").mkdir(exist_ok=True)
    for group, image_values in representations.items():
        patient_pairs = base.patient_pair_features(image_values, active_images, active)
        symmetric, antisymmetric = hierarchy.structured_views(patient_pairs)
        probabilities, models = hierarchy.state_probabilities(
            symmetric, antisymmetric, states, train, SEED)
        score = hierarchy.kidney_scores(probabilities, active, active_kidneys)[dev]
        y_dev = labels[dev]
        target = active_kidneys.iloc[dev].reset_index(drop=True)
        rng = np.random.default_rng(SEED)
        clusters = [np.flatnonzero(target.patient_alias.eq(alias))
                    for alias in sorted(target.patient_alias.unique())]
        samples = []
        for _ in range(2000):
            draw = rng.integers(0, len(clusters), len(clusters))
            selected = np.concatenate([clusters[j] for j in draw])
            if np.unique(y_dev[selected]).size < 2:
                continue
            samples.append((roc_auc_score(y_dev[selected], score[selected]),
                            average_precision_score(y_dev[selected], score[selected])))
        ci = np.percentile(np.asarray(samples), [2.5, 97.5], axis=0)
        metrics.append({"group": group, "patients": 70, "kidneys": len(dev),
                        "prevalence": float(y_dev.mean()), **metric(y_dev, score),
                        "auroc_ci_low": float(ci[0, 0]), "auroc_ci_high": float(ci[1, 0]),
                        "ap_ci_low": float(ci[0, 1]), "ap_ci_high": float(ci[1, 1]),
                        "bootstrap_valid": len(samples)})
        for row, value in zip(target.itertuples(index=False), score):
            predictions.append({"group": group, "patient_alias": row.patient_alias,
                                "kidney_alias": row.kidney_alias, "side": row.side,
                                "label": int(row.kidney_APN), "score": float(value)})
        joblib.dump(models, OUT / "models" / f"{group}.joblib", compress=3)
        print(f"{group}: AUROC={metrics[-1]['auroc']:.3f}, AP={metrics[-1]['ap']:.3f}", flush=True)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "development_metrics.csv", index=False)
    protocol = {"source_cohort_lock_sha256": digest(REGISTRY / "cohort_lock.json"),
                "encoder_audit_sha256": digest(CACHE / "uricads_encoder_audit.json"),
                "train_patients": 204, "development_patients": 70,
                "test_scored": False, "patient_grouped": True,
                "heads": "Train-only four-state factorization: any APN; bilateral conditional on APN; laterality conditional on unilateral; symmetric and antisymmetric left/right features, inner 4-fold patient CV",
                "representations": list(representations),
                "uncertainty": "2000 conditional development patient-cluster bootstrap resamples",
                "grid_n_jobs": int(os.environ.get("NCKUH_GRID_N_JOBS", "1"))}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
