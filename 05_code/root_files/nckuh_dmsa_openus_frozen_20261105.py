"""Evaluate frozen official OpenUS-S representations on contour-first NCKUH kidneys."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "03_data_models/third_party/OpenUS_official_20261105"
WEIGHT = MODEL_ROOT / "openus_cpt0150.pth"
SOURCE = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_fullframe_20261103"
REFERENCE = ROOT / "02_experiments/NCKUH_DMSA_uricads_official_frozen_20261105/development_predictions.csv"
OUT = ROOT / "02_experiments/NCKUH_DMSA_openus_frozen_20261105"
SEEDS = (20261102, 20261103, 20261104)
CURRENT_GROUP = "current_tile_usfm_reference"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"
WEIGHT_SHA256 = "a82365cc2994b9826c8c307a6ee19acf7dc7c42468eb76b95285de58aea91720"
COMMIT = "b6508c8a833b38fc69f71de6e9f7685bed298358"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_reference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference as load

    patients, kidneys, images = load()
    images = images[images.kidney_APN.notna()].copy().reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].copy().reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].copy().reset_index(drop=True)
    if (len(patients), len(kidneys), len(images)) != (210, 414, 820):
        raise ValueError("Unexpected cohort")
    return patients, kidneys, images


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve completed or partial experiment: {OUT}")
    if digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError("OpenUS checkpoint SHA-256 mismatch")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    _, _, images = load_reference()
    images[["image_id", "patient_alias", "kidney_alias", "side", "view"]].to_csv(
        OUT / "encoding_manifest.csv", index=False
    )
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "An ultrasound-domain VMamba teacher may encode subtle renal parenchymal heterogeneity and directional scar-like abnormalities not represented by the current radiology classifier",
        "source": "https://github.com/XZheng0427/OpenUS",
        "paper": "OpenUS: A Large-Scale Foundation Model for Ultrasound Imaging",
        "official_commit": COMMIT,
        "official_weight": "OpenUS-S checkpoint openus_cpt0150.pth, teacher at epoch 151",
        "weight_sha256": WEIGHT_SHA256,
        "strict_weight_load": "Backbone_DINOv2_VSSM_2 teacher, 412/412 keys",
        "compatibility": "Project-supplied selective_scan CUDA core compiled under WSL2; repository availability flag enabled after explicit successful import; numerical comparison to the torch reference had maximum absolute difference 1.91e-6",
        "input": "Registered contour-masked tight 224x224 RGB kidney canvas; outside-mask pixels zero; official ImageNet normalization",
        "representations": "Global token; mask-covered final-token mean and standard deviation; canvas upper-lower, left-right, and center-periphery token contrasts without anatomical-orientation claims",
        "models": "Frozen encoder; training-only hierarchical four-state LR heads with unchanged inner grid",
        "fusion": "Candidate alone and preregistered fixed 20% or 25% contribution to the unchanged official URI-CADS quarter-fusion reference",
        "cohort": "210 patients, 414 kidneys, 820 images; unchanged three 126/42/42 assignments",
        "overlap": "No known NCKUH pretraining overlap; public kidneyUS is listed by OpenUS but is not this private NCKUH cohort",
        "exploratory_test": "Unopened",
        "inputs": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("nckuh_dmsa_openus_encode_wsl_20261105.py"),
                WEIGHT,
                SOURCE / "patient_assignments.csv",
                REFERENCE,
            )
        },
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "protocol_sha256": digest(OUT / "protocol.json"),
        "manifest_sha256": digest(OUT / "encoding_manifest.csv"),
    })
    print(f"Prepared OpenUS experiment: {OUT}", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    if digest(OUT / "protocol.json") != lock["protocol_sha256"]:
        raise ValueError("OpenUS protocol changed after registration")
    if digest(OUT / "encoding_manifest.csv") != lock["manifest_sha256"]:
        raise ValueError("OpenUS manifest changed after registration")


def evaluate() -> None:
    require_lock()
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve completed OpenUS evaluation")
    embedding_path = OUT / "openus_embeddings.npz"
    audit = json.loads((OUT / "encoder_audit.json").read_text(encoding="utf-8"))
    if audit["embedding_sha256"] != digest(embedding_path):
        raise ValueError("OpenUS embedding hash mismatch")
    encoded = np.load(embedding_path)
    global_mean = encoded["global_mean"]
    roi_mean = encoded["roi_mean"]
    roi_std = encoded["roi_std"]
    regional = encoded["regional_contrasts"]
    arrays = (global_mean, roi_mean, roi_std, regional)
    if [value.shape for value in arrays] != [(820, 768), (820, 768), (820, 768), (820, 2304)]:
        raise ValueError("Unexpected OpenUS representation shape")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Nonfinite OpenUS representation")

    patients, kidneys, images = load_reference()
    saved_manifest = pd.read_csv(OUT / "encoding_manifest.csv")
    if saved_manifest.image_id.tolist() != images.image_id.tolist():
        raise ValueError("OpenUS embedding order mismatch")
    representations = {
        "global": global_mean,
        "roi": np.c_[roi_mean, roi_std],
        "roi_directional": np.c_[roi_mean, roi_std, regional],
    }
    structured = {
        name: hierarchy.structured_views(base.patient_pair_features(values, images, patients))
        for name, values in representations.items()
    }
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(SOURCE / "patient_assignments.csv")
    references = pd.read_csv(REFERENCE)
    references = references[references.group.isin([CURRENT_GROUP, URI_GROUP])]
    references = references.set_index(["seed", "group", "kidney_alias"], verify_integrity=True)

    metrics, predictions, selections = [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train = np.flatnonzero(patients.patient_alias.map(mapping).eq("train"))
        development = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        target = kidneys.iloc[development]
        if len(train) != 126 or target.patient_alias.nunique() != 42:
            raise ValueError("Unexpected split")
        current = references.loc[
            [(seed, CURRENT_GROUP, alias) for alias in target.kidney_alias]
        ].score.to_numpy(float)
        uri = references.loc[
            [(seed, URI_GROUP, alias) for alias in target.kidney_alias]
        ].score.to_numpy(float)
        scores = {
            "current_tile_usfm_reference": current,
            "official_uri_fixed_quarter_reference": uri,
        }
        for source, (symmetric, antisymmetric) in structured.items():
            probability, models = hierarchy.state_probabilities(
                symmetric, antisymmetric, states, train, seed
            )
            candidate = hierarchy.kidney_scores(probability, patients, kidneys)[development]
            stem = f"openus_{source}"
            scores[stem] = candidate
            scores[f"{stem}_quarter_plus_uri"] = 0.25 * candidate + 0.75 * uri
            scores[f"{stem}_fifth_plus_uri"] = 0.20 * candidate + 0.80 * uri
            joblib.dump(models, OUT / "models" / f"{seed}_{source}.joblib", compress=3)
            for head, model in models.items():
                selections.append({
                    "seed": seed,
                    "source": source,
                    "head": head,
                    "inner_cv_auroc": float(model.best_score_),
                    **model.best_params_,
                })

        for group, score in scores.items():
            result = hierarchy.metric(labels[development], score)
            metrics.append({
                "seed": seed,
                "role": "development",
                "group": group,
                "patients": 42,
                "kidneys": len(development),
                **result,
            })
            for local, index in enumerate(development):
                predictions.append({
                    "seed": seed,
                    "group": group,
                    "patient_alias": kidneys.patient_alias.iloc[index],
                    "kidney_alias": kidneys.kidney_alias.iloc[index],
                    "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                    "side": kidneys.side.iloc[index],
                    "label": int(labels[index]),
                    "score": float(score[local]),
                })
        print(f"Completed OpenUS draw {seed}", flush=True)

    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "training_inner_cv_selections.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    groups = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups, SEEDS, "official_uri_fixed_quarter_reference"
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(OUT / "decision.json", {
        "groups_above_both_development_targets": achieved,
        "promotion": bool(achieved),
        "reference": "official_uri_fixed_quarter_reference",
        "exploratory_test_rows_scored": 0,
    })
    write_json(OUT / "integrity_audit.json", {
        "patients": len(patients),
        "kidneys": len(kidneys),
        "images": len(images),
        "prediction_rows": len(prediction_frame),
        "development_groups": len(groups),
        "finite_scores": bool(np.isfinite(prediction_frame.score).all()),
        "unit_interval_scores": bool(prediction_frame.score.between(0.0, 1.0).all()),
        "exploratory_test_rows_scored": 0,
    })
    print(summary.to_string(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "evaluate"))
    arguments = parser.parse_args()
    prepare() if arguments.action == "prepare" else evaluate()


if __name__ == "__main__":
    main()
