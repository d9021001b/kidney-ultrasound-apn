"""Repeated 80/20 patient holdout with the strongest completed NCKUH features."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_hierarchical_tile_usfm_20261104 as tile_usfm
import nckuh_dmsa_uri_contralateral_local_20261105 as local_study
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
import nckuh_dmsa_uricads_full_detector_20261105 as detector
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_expanded_80_20_20261105"
REFERENCE = (
    ROOT
    / "02_experiments/NCKUH_DMSA_uri_contralateral_local_20261105/development_predictions.csv"
)
CURRENT_GROUP = "current_tile_usfm_reference"
URI_GROUP = "official_uri_fixed_quarter_reference"
SEEDS = (20261102, 20261103, 20261104)
GROUPS = {
    "tile_usfm": "expanded80_tile_usfm",
    "uri_detector": "expanded80_uri_detector",
    "all": "expanded80_tile_usfm_uri_detector",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def image_representations(images: pd.DataFrame) -> dict[str, np.ndarray]:
    tile = tile_usfm.image_representations(images)["tile_usfm"]
    uri_embedding = np.load(uri.OUT / "official_fold_embeddings.npy").mean(axis=1)
    uri_scores = np.load(uri.OUT / "official_fold_pathology_scores.npy").reshape(len(images), -1)
    detector_features = np.load(detector.OUT / "official_full_detector_features.npy")
    uri_detector = np.c_[uri_embedding, uri_scores, detector_features].astype(np.float32)
    outputs = {
        "tile_usfm": tile.astype(np.float32),
        "uri_detector": uri_detector,
        "all": np.c_[tile, uri_detector].astype(np.float32),
    }
    if not all(len(values) == len(images) and np.isfinite(values).all() for values in outputs.values()):
        raise ValueError("Invalid expanded-training representation")
    return outputs


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve registered experiment: {OUT}")
    OUT.mkdir(parents=True)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "Test whether the AUROC deficit is primarily limited by 60% rather than 80% outer-training coverage",
        "design": "Three repeated patient-level 80/20 random holdouts; historical train and unscored exploratory-test roles are merged into 168 training patients while each 42-patient development role remains excluded",
        "interpretation": "Exploratory learning-curve analysis; no independent internal-test claim remains after repurposing the historical test roles for training",
        "representations": {
            "tile_usfm": "Current strongest frozen RadImageNet tile plus official USFM image representation",
            "uri_detector": "Mean official URI-CADS embedding, five-fold pathology scores, and complete official local-detector features",
            "all": "Concatenation of tile+USFM and URI-CADS detector evidence",
        },
        "classifier": "Hierarchical any-APN, bilateral-given-APN and left-given-unilateral balanced logistic heads after training-only imputation, scaling and PCA selected by four-fold inner CV",
        "fixed_comparison": "Standalone candidates and fixed 25% contribution to the unchanged 126-patient current reference",
        "development_patients_per_draw": 42,
        "training_patients_per_draw": 168,
        "labels_used_in_frozen_feature_encoding": [],
        "source_sha256": digest(Path(__file__)),
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(
        OUT / "run_lock.json",
        {
            "protocol_sha256": digest(OUT / "protocol.json"),
            "source_sha256": digest(Path(__file__)),
        },
    )
    print("Registered repeated 80/20 NCKUH learning-curve experiment", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    if digest(OUT / "protocol.json") != lock["protocol_sha256"]:
        raise ValueError("Protocol changed")
    if digest(Path(__file__)) != lock["source_sha256"]:
        raise ValueError("Source changed")


def run() -> None:
    require_lock()
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve completed 80/20 evaluation")
    (OUT / "models").mkdir()
    patients, kidneys, images = local_study.load_reference()
    representations = image_representations(images)
    structured = {
        name: hierarchy.structured_views(base.patient_pair_features(values, images, patients))
        for name, values in representations.items()
    }
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    references = pd.read_csv(REFERENCE)
    references = references[
        references.group.isin([CURRENT_GROUP, URI_GROUP])
    ].set_index(["seed", "group", "kidney_alias"], verify_integrity=True)

    metrics, predictions, selections, split_audit = [], [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        patient_roles = patients.patient_alias.map(mapping)
        kidney_roles = kidneys.patient_alias.map(mapping)
        train = np.flatnonzero(~patient_roles.eq("development"))
        development = np.flatnonzero(kidney_roles.eq("development"))
        target = kidneys.iloc[development]
        if len(train) != 168 or target.patient_alias.nunique() != 42:
            raise ValueError("Unexpected 80/20 split")
        if set(patients.patient_alias.iloc[train]).intersection(target.patient_alias):
            raise ValueError("Patient overlap between expanded training and development")
        split_audit.append(
            {
                "seed": seed,
                "training_patients": len(train),
                "development_patients": target.patient_alias.nunique(),
                "patient_overlap": 0,
            }
        )
        current = references.loc[
            [(seed, CURRENT_GROUP, alias) for alias in target.kidney_alias]
        ].score.to_numpy(float)
        fixed_uri = references.loc[
            [(seed, URI_GROUP, alias) for alias in target.kidney_alias]
        ].score.to_numpy(float)
        scores = {CURRENT_GROUP: current, URI_GROUP: fixed_uri}
        for source, (symmetric, antisymmetric) in structured.items():
            probability, models = hierarchy.state_probabilities(
                symmetric, antisymmetric, states, train, seed
            )
            candidate = hierarchy.kidney_scores(probability, patients, kidneys)[development]
            group = GROUPS[source]
            scores[group] = candidate
            scores[f"{group}_quarter_plus_current"] = 0.25 * candidate + 0.75 * current
            joblib.dump(models, OUT / "models" / f"{seed}_{source}.joblib", compress=3)
            for head, model in models.items():
                selections.append(
                    {
                        "seed": seed,
                        "source": source,
                        "head": head,
                        "inner_cv_auroc": float(model.best_score_),
                        **model.best_params_,
                    }
                )
        for group, score in scores.items():
            metrics.append(
                {
                    "seed": seed,
                    "role": "development",
                    "group": group,
                    "training_patients": len(train),
                    "patients": target.patient_alias.nunique(),
                    "kidneys": len(target),
                    **hierarchy.metric(labels[development], score),
                }
            )
            for local_index, index in enumerate(development):
                predictions.append(
                    {
                        "seed": seed,
                        "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[index],
                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                        "side": kidneys.side.iloc[index],
                        "label": int(labels[index]),
                        "score": float(score[local_index]),
                    }
                )
        print(f"Completed expanded 80/20 draw {seed}", flush=True)

    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "training_inner_cv_selections.csv", index=False)
    pd.DataFrame(split_audit).to_csv(OUT / "split_audit.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    groups = prediction_frame.group.drop_duplicates().tolist()
    intervals, differences, samples = cluster_intervals(
        prediction_frame, groups, SEEDS, URI_GROUP
    )
    intervals.to_csv(OUT / "development_ci.csv", index=False)
    differences.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(
        OUT / "integrity_and_decision.json",
        {
            "status": "pass",
            "patients": len(patients),
            "kidneys": len(kidneys),
            "images": len(images),
            "training_patients_per_draw": 168,
            "development_patients_per_draw": 42,
            "patient_overlap": 0,
            "groups_above_both_development_targets": achieved,
            "development_only": True,
            "independent_internal_test_available_after_this_experiment": False,
            "labels_used_in_frozen_feature_encoding": [],
        },
    )
    print(summary.to_string(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        globals()[args.stage]()


if __name__ == "__main__":
    main()
