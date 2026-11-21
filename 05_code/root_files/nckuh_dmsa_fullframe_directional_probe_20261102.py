"""Canonical-axis directional Coiflet/GLRLM features on clean NCKUH full-frame ROIs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullframe_clean_probe_20261102 as F
import nckuh_dmsa_fullframe_deep_probe_20261102 as D
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullframe_directional_probe_20261102"
REFERENCE = "deep_concat_plus_clean_radiomics"
GROUPS = (
    "clean_axis_i3_anova32_lr",
    "clean_axis_i3_anova64_lr",
    "clean_axis_i3_plus_deep_radiomics",
    "early_concat_deep_radiomics_i3_anova64_lr",
    "early_concat_deep_radiomics_i3_anova128_lr",
)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered directional protocol")
    F.require_lock()
    D.require_lock()
    extractor = ROOT / "02_experiments/CYCH_honest_reevaluation_20261003/extract_site.py"
    inputs = [
        F.OUT / "patient_assignments.csv", F.OUT / "deidentified_image_manifest.csv",
        F.OUT / "radiomics_clean_fullframe.csv", D.OUT / "radimagenet_gray.npy",
        D.OUT / "radimagenet_color.npy", D.OUT / "usf_gray.npy",
        D.OUT / "development_predictions.csv", extractor, Path(__file__),
    ]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same clean-full-frame 111 patients, 212 kidneys, 418 images and exact 70/20/21 patient roles",
        "motivation": "Canonical renal-axis rotation tests directional texture that ordinary image-axis radiomics can blur across acquisition angle",
        "features": "Exactly25 preregistered original/Coiflet-1 directional first-order and GLRLM features after mask-derived native-axis rotation; no label enters extraction",
        "models": "Ordered own/opposite kidney and longitudinal/transverse context; train-only imputation, scaling, variance filter, ANOVA32/64/128 and balanced LR C0.01",
        "fusion": "One fixed equal-probability average with completed deep-radiomics reference, plus two prespecified early-concatenation controls",
        "prohibited": "No diagnosis suffix, DMSA field/code as predictor, development-selected feature list/weight/threshold, or exploratory-test scoring",
        "target": "Development AUROC>0.8 and AP>0.8; later locked confirmation remains necessary",
        "groups": list(GROUPS),
        "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "source_sha256": digest(__file__), "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs},
    })
    print("Registered clean full-frame canonical-axis directional probe", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(Path(name)) == expected, name


def extract() -> None:
    require_lock()
    if (OUT / "i3_clean_fullframe.csv").exists():
        raise FileExistsError("Preserve completed directional features")
    _, _, images = F.matched_reference()
    sys.path.insert(0, str(ROOT / "02_experiments/CYCH_honest_reevaluation_20261003"))
    from extract_site import _i3_worker, run_pool
    jobs = [(ident, F.OUT / "percentile" / f"{ident}.png", F.OUT / "masks" / f"{ident}.png")
            for ident in images.image_id]
    frame = run_pool(_i3_worker, jobs, processes=8)
    assert frame.shape == (418, 25) and np.isfinite(frame.to_numpy()).all()
    frame.to_csv(OUT / "i3_clean_fullframe.csv")
    write_json(OUT / "feature_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "images": 418, "features_per_image": 25,
        "labels_used_in_extraction": [], "sha256": digest(OUT / "i3_clean_fullframe.csv"),
    })
    print("Extracted 25 canonical-axis directional features from 418 clean ROIs", flush=True)


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed directional predictions")
    patients, kidneys, images = F.matched_reference()
    labels = kidneys.kidney_APN.to_numpy(int)
    i3 = pd.read_csv(OUT / "i3_clean_fullframe.csv").set_index("image_id").reindex(images.image_id).to_numpy(float)
    clean = pd.read_csv(F.OUT / "radiomics_clean_fullframe.csv").set_index("image_id").reindex(images.image_id)
    clean = clean[[column for column in clean if not column.endswith(("_Energy", "_TotalEnergy"))]].to_numpy(float)
    deep = np.concatenate([np.load(D.OUT / name) for name in
                           ("radimagenet_gray.npy", "radimagenet_color.npy", "usf_gray.npy")], axis=1)
    oriented_i3 = F.orient_kidneys(F.patient_pair_features(i3, images, patients), patients, kidneys)
    oriented_clean = F.orient_kidneys(F.patient_pair_features(clean, images, patients), patients, kidneys)
    oriented_deep = F.orient_kidneys(F.patient_pair_features(deep, images, patients), patients, kidneys)
    early = np.concatenate([oriented_deep, oriented_clean, oriented_i3], axis=1)
    assignments = pd.read_csv(F.OUT / "patient_assignments.csv")
    reference = pd.read_csv(D.OUT / "development_predictions.csv")
    reference = reference[reference.group.eq(REFERENCE)].set_index(["seed", "kidney_alias"]).score
    records, predictions = [], []
    (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in F.SEEDS:
            roles = kidneys.patient_alias.map(assignments[assignments.seed.eq(seed)].set_index("patient_alias").role)
            train, dev = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
            scores, objects = {}, {}
            scores[GROUPS[0]], *objects[GROUPS[0]] = F.fit_predict(oriented_i3, train, dev, labels, seed, 32)
            scores[GROUPS[1]], *objects[GROUPS[1]] = F.fit_predict(oriented_i3, train, dev, labels, seed, 64)
            ref = np.asarray([reference.loc[(seed, alias)] for alias in kidneys.kidney_alias.iloc[dev]])
            scores[GROUPS[2]] = (scores[GROUPS[1]] + ref) / 2
            scores[GROUPS[3]], *objects[GROUPS[3]] = F.fit_predict(early, train, dev, labels, seed, 64)
            scores[GROUPS[4]], *objects[GROUPS[4]] = F.fit_predict(early, train, dev, labels, seed, 128)
            joblib.dump(objects, OUT / "models" / f"{seed}_directional.joblib", compress=3)
            for group in GROUPS:
                result = metric(labels[dev], scores[group])
                records.append({"seed": seed, "role": "development", "group": group,
                                "patients": 20, "kidneys": len(dev), **result})
                for local, index in enumerate(dev):
                    predictions.append({"seed": seed, "group": group,
                                        "patient_alias": kidneys.patient_alias.iloc[index],
                                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                                        "DMSA_APN": kidneys.DMSA_APN.iloc[index], "side": kidneys.side.iloc[index],
                                        "label": int(labels[index]), "score": float(scores[group][local])})
            print(f"Completed clean directional draw {seed}", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    summary = frame.groupby("group")[["auroc", "ap"]].mean().sort_values(["auroc", "ap"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc > .8) & (summary.ap > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {
        "groups_above_both_development_targets": achieved, "test_scored": False,
        "goal_achieved": False, "reason": "Development-only availability subgroup cannot complete the full objective",
    })
    print(summary.to_string(), flush=True)


def bootstrap() -> None:
    require_lock()
    frame = pd.read_csv(OUT / "development_predictions.csv")
    previous = pd.read_csv(D.OUT / "development_predictions.csv")
    reference = previous[previous.group.eq(REFERENCE)]
    ci, paired, samples = cluster_intervals(
        pd.concat([frame, reference], ignore_index=True), list(GROUPS) + [REFERENCE], F.SEEDS, REFERENCE)
    ci.to_csv(OUT / "development_confidence_intervals.csv", index=False)
    paired.to_csv(OUT / "development_paired_intervals.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    write_json(OUT / "completion_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "patients": 111,
        "kidneys": 212, "images": 418, "test_scored": False,
        "scored_unit": "kidney", "bootstrap_cluster": "patient",
    })
    print("Directional intervals completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "extract", "run", "bootstrap"])
    arguments = parser.parse_args()
    globals()[arguments.stage]()
