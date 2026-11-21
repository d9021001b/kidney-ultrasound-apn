"""Age-normalized display-tick renal geometry for the full NCKUH cohort."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_clinical_features_20261103 as C
import nckuh_dmsa_fullcohort_fullframe_20261103 as F
import nckuh_dmsa_fullframe_tick_geometry_probe_20261102 as T
from nckuh_only_study_20261102 import digest, write_json


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_tick_geometry_20261103"
REFERENCE = "fullcohort_rad_color_image_plus_patient_direct"
GROUPS = (
    "tick_age_residual_anova32_lr",
    "tick_age_residual_anova64_lr",
    "tick_age_residual_plus_radimagenet_anova64_lr",
    "tick_age_residual_state_pca16_lr",
    "tick_anova32_plus_instance_reference",
    "tick_early_plus_instance_reference",
    "tick_state_plus_instance_reference",
)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered full-cohort tick-geometry protocol")
    F.require_lock(); C.require_lock()
    inputs = [
        F.OUT / "patient_assignments.csv",
        F.OUT / "radimagenet_color.npy",
        F.OUT / "instance_development_predictions.csv",
        C.OUT / "clinical_image_features.csv",
        C.OUT / "deidentified_demographics.csv",
        Path(__file__),
    ]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same 210 patients, 414 kidneys and 820 contour-first images with unchanged patient roles",
        "motivation": "Pediatric APN can enlarge renal length or volume, but pixel and frame-relative size are confounded by age and scanner zoom",
        "references": [
            "https://pubmed.ncbi.nlm.nih.gov/3513490/",
            "https://pubmed.ncbi.nlm.nih.gov/21459422/",
            "https://pubmed.ncbi.nlm.nih.gov/34657197/",
        ],
        "measurement": "Detect a periodic vertical ultrasound display scale, express contour dimensions in adjacent tick intervals, and retain an explicit missing-scale flag; tick intervals are not called centimeters",
        "normalization": "Residualize tick and sector-relative geometry only on training images against age, sex, side, view and deidentified source domain",
        "models": "Fixed ANOVA logistic heads, patient four-state model converted to kidney marginals, early RadImageNet concatenation, and equal-probability reference fusions",
        "prohibited": "No filename measurements, identifiers, DMSA code, development score or test role enters measurement or normalization",
        "target": "Development AUROC>0.8 and AP>0.8; locked confirmation remains required",
        "groups": list(GROUPS),
        "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "source_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs},
    })
    print("Registered full-cohort tick-geometry experiment", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(Path(__file__)) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(Path(name)) == expected, name


def _extract_one(payload: tuple[str, dict]) -> dict:
    image_id, source_dict = payload
    source = SimpleNamespace(**source_dict)
    frame, mask = C._load_source(source)
    values = T.tick_geometry(frame, mask)
    return {"image_id": image_id,
            **{f"tick_{i:02d}": float(value) for i, value in enumerate(values)}}


def extract() -> None:
    require_lock()
    if (OUT / "tick_geometry.csv").exists():
        raise FileExistsError("Preserve completed full-cohort tick geometry")
    _, _, images = C.load_reference()
    plan = F.source_plan().set_index("image_id").reindex(images.image_id)
    payloads = [(image.image_id, source._asdict()) for image, source in
                zip(images.itertuples(index=False), plan.itertuples(index=False))]
    context = mp.get_context("spawn")
    records = []
    with context.Pool(processes=6) as pool:
        for index, record in enumerate(pool.imap(_extract_one, payloads, chunksize=3), start=1):
            records.append(record)
            if index % 100 == 0:
                print(f"Extracted tick geometry {index}/820", flush=True)
    frame = pd.DataFrame(records).set_index("image_id").reindex(images.image_id).reset_index()
    frame.to_csv(OUT / "tick_geometry.csv", index=False)
    parsed = frame.tick_08.eq(1)
    write_json(OUT / "feature_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "images": len(frame), "features_per_image": 9,
        "parsed_tick_spacing": int(parsed.sum()),
        "parsed_patients": int(images.loc[parsed, "patient_alias"].nunique()),
        "spacing_min_median_max": [float(frame.loc[parsed, "tick_07"].min()),
                                    float(frame.loc[parsed, "tick_07"].median()),
                                    float(frame.loc[parsed, "tick_07"].max())],
        "labels_used_in_extraction": [],
        "sha256": digest(OUT / "tick_geometry.csv"),
    })
    print(f"Depth-tick geometry extracted: {parsed.sum()}/820 images", flush=True)


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed full-cohort tick predictions")
    patients, kidneys, images = C.load_reference()
    labels = kidneys.kidney_APN.to_numpy(int)
    states = patients.DMSA_APN.map(F.STATE_CODE).to_numpy(int)
    tick_frame = pd.read_csv(OUT / "tick_geometry.csv").set_index("image_id").reindex(images.image_id)
    tick = tick_frame[[f"tick_{i:02d}" for i in range(9)]].to_numpy(float)
    clinical = pd.read_csv(C.OUT / "clinical_image_features.csv").set_index("image_id").reindex(images.image_id)
    relative = clinical[[c for c in clinical if c.startswith("geometry_")]].to_numpy(float)
    geometry = np.c_[tick, relative]
    demo = pd.read_csv(C.OUT / "deidentified_demographics.csv")
    context = C.covariates(images, clinical, demo)
    rad_image = np.load(F.OUT / "radimagenet_color.npy")
    assignments = pd.read_csv(F.OUT / "patient_assignments.csv")
    reference_frame = pd.read_csv(F.OUT / "instance_development_predictions.csv")
    reference = reference_frame[reference_frame.group.eq(REFERENCE)].set_index(["seed", "kidney_alias"]).score
    patient_lookup = {alias: index for index, alias in enumerate(patients.patient_alias)}
    predictions, metrics, audits = [], [], []
    (OUT / "models").mkdir(exist_ok=True)

    with threadpool_limits(limits=4):
        for seed in F.SEEDS:
            mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
            image_roles = images.patient_alias.map(mapping)
            kidney_roles = kidneys.patient_alias.map(mapping)
            patient_roles = patients.patient_alias.map(mapping)
            train_images = np.flatnonzero(image_roles.eq("train"))
            train = np.flatnonzero(kidney_roles.eq("train"))
            development = np.flatnonzero(kidney_roles.eq("development"))
            train_patients = np.flatnonzero(patient_roles.eq("train"))
            residual, normalizer = C.residualize(geometry, context, train_images, seed)
            pairs = F.patient_pair_features(residual, images, patients)
            oriented = F.orient_kidneys(pairs, patients, kidneys)
            rad_oriented = F.orient_kidneys(F.patient_pair_features(rad_image, images, patients), patients, kidneys)
            early = np.c_[oriented, rad_oriented]

            score32, transform32, model32 = C.fit_anova(oriented, train, development, labels, seed, 32)
            score64, transform64, model64 = C.fit_anova(oriented, train, development, labels, seed, 64)
            early_score, early_transform, early_model = C.fit_anova(early, train, development, labels, seed, 64)
            state_probability, state_model = F.fit_state_model(pairs, train_patients, states, seed, 16)
            left, right = F.kidney_marginals(state_probability)
            state_score = np.asarray([
                left[patient_lookup[kidneys.patient_alias.iloc[index]]]
                if kidneys.side.iloc[index] == "left"
                else right[patient_lookup[kidneys.patient_alias.iloc[index]]]
                for index in development
            ])
            aliases = kidneys.kidney_alias.iloc[development]
            reference_score = np.asarray([reference.loc[(seed, alias)] for alias in aliases])
            scores = {
                GROUPS[0]: score32, GROUPS[1]: score64, GROUPS[2]: early_score,
                GROUPS[3]: state_score, GROUPS[4]: (score32 + reference_score) / 2,
                GROUPS[5]: (early_score + reference_score) / 2,
                GROUPS[6]: (state_score + reference_score) / 2,
            }
            joblib.dump({"normalizer": normalizer, "anova32": (transform32, model32),
                         "anova64": (transform64, model64), "early": (early_transform, early_model),
                         "state": state_model},
                        OUT / "models" / f"{seed}_tick_geometry.joblib", compress=3)
            audits.append({"seed": seed, "normalizer_training_patients": 126,
                           "normalizer_training_images": len(train_images),
                           "labels_used_in_normalizer": [], "exploratory_test_scored": False})
            for group, score in scores.items():
                result = F.metric(labels[development], score)
                metrics.append({"seed": seed, "role": "development", "group": group,
                                "patients": 42, "kidneys": len(development), **result})
                for local, index in enumerate(development):
                    predictions.append({"seed": seed, "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[index],
                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[index], "side": kidneys.side.iloc[index],
                        "label": int(labels[index]), "score": float(score[local])})
            print(f"Completed tick-geometry draw {seed}", flush=True)

    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc_mean > .8) & (summary.ap_mean > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {
        "groups_above_both_development_targets": achieved, "exploratory_test_scored": False})
    write_json(OUT / "normalizer_audit.json", {"draws": audits})
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "extract", "run"))
    args = parser.parse_args()
    {"prepare": prepare, "extract": extract, "run": run}[args.stage]()
