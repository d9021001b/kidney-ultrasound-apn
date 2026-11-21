"""Depth-tick-normalized kidney geometry from archived NCKUH ultrasound frames."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import cv2
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullframe_clean_probe_20261102 as F
import nckuh_dmsa_fullframe_deep_probe_20261102 as D
import nckuh_dmsa_fullframe_relative_geometry_probe_20261102 as G
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullframe_tick_geometry_probe_20261102"
REFERENCE = "deep_concat_plus_clean_radiomics"
GROUPS = (
    "depth_tick_geometry_anova32_lr",
    "depth_tick_geometry_plus_deep_radiomics",
    "early_concat_deep_radiomics_tick_geometry_anova64_lr",
    "early_concat_deep_radiomics_tick_geometry_anova128_lr",
)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists(): raise FileExistsError("Preserve registered tick-geometry protocol")
    F.require_lock(); D.require_lock(); G.require_lock()
    inputs = [F.ARCHIVE, F.OUT / "patient_assignments.csv", F.OUT / "deidentified_image_manifest.csv",
              F.OUT / "radiomics_clean_fullframe.csv", G.OUT / "relative_geometry.csv",
              D.OUT / "radimagenet_gray.npy", D.OUT / "radimagenet_color.npy", D.OUT / "usf_gray.npy",
              D.OUT / "development_predictions.csv", Path(__file__)]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same clean-full-frame 111 patients, 212 kidneys, 418 images and exact 70/20/21 patient roles",
        "motivation": "Relative geometry improved the absolute bilateral signal; depth-tick normalization further reduces scanner zoom dependence without using clinical metadata",
        "tick_detection": "Search the left quarter below the header for a bright periodic vertical depth scale; cluster contiguous marks and require at least five consistent15-60-pixel intervals",
        "units": "Lengths are reported in adjacent display-tick intervals and areas in squared tick intervals. They are not called centimeters or millimeters because DICOM pixel spacing is unavailable",
        "features": "Mask PCA major/minor axes, bounding box, area, perimeter and equivalent diameter normalized by detected tick spacing, availability, plus completed relative-frame geometry",
        "models": "Ordered own/opposite kidney and longitudinal/transverse context; fixed ANOVA32/64/128 balanced LR and equal-probability fusion with completed deep-radiomics",
        "prohibited": "No age, sex, DMSA code, filename suffix, development-selected rule/weight/threshold, or exploratory-test scoring",
        "target": "Development AUROC>0.8 and AP>0.8; later locked confirmation remains necessary",
        "groups": list(GROUPS), "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {"source_sha256": digest(__file__),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs}})
    print("Registered depth-tick-normalized renal geometry probe", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items(): assert digest(Path(name)) == expected, name


def depth_tick_spacing(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    height, width = gray.shape; best = None
    for x in range(5, int(width * .25)):
        ys = np.flatnonzero(gray[int(height * .08):int(height * .88), x] > 180) + int(height * .08)
        if len(ys) < 5: continue
        groups = np.split(ys, np.where(np.diff(ys) > 2)[0] + 1)
        centers = np.asarray([group.mean() for group in groups if 1 <= len(group) <= 8])
        if len(centers) < 6: continue
        differences = np.diff(centers)
        valid = differences[(differences >= 15) & (differences <= 60)]
        if len(valid) < 5: continue
        core = valid[valid <= np.quantile(valid, .70)]
        median = float(np.median(core))
        near = valid[(valid > .75 * median) & (valid < 1.25 * median)]
        if len(near) < 5: continue
        spacing = float(np.median(near)); relative_mad = float(np.median(np.abs(near - spacing)) / spacing)
        score = (len(near), -relative_mad, -x)
        if best is None or score > best[0]: best = (score, spacing)
    return np.nan if best is None else best[1]


def tick_geometry(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    spacing = depth_tick_spacing(frame); present = float(np.isfinite(spacing))
    if not present: spacing = np.nan
    binary = mask.astype(np.uint8); yy, xx = np.nonzero(binary)
    covariance = np.cov(np.c_[xx, yy], rowvar=False); eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1e-8)
    minor, major = 4 * np.sqrt(eigenvalues[0]), 4 * np.sqrt(eigenvalues[1])
    bbox_w, bbox_h = float(xx.max() - xx.min() + 1), float(yy.max() - yy.min() + 1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = float(cv2.arcLength(max(contours, key=cv2.contourArea), True))
    area = float(binary.sum()); equivalent = np.sqrt(4 * area / np.pi)
    return np.asarray([major / spacing, minor / spacing, bbox_w / spacing, bbox_h / spacing,
                       perimeter / spacing, area / (spacing * spacing), equivalent / spacing,
                       spacing, present], dtype=np.float32)


def extract() -> None:
    require_lock()
    if (OUT / "tick_geometry.csv").exists(): raise FileExistsError("Preserve completed tick geometry")
    _, _, images = F.matched_reference(); index = F.archive_index(); records = []
    with zipfile.ZipFile(F.ARCHIVE) as handle:
        for row in images.itertuples(index=False):
            members = index[(row.patient_id, row.date_key, row.view)]
            if len(members) != 1: raise ValueError(f"Nonunique archive key: {row.image_id}")
            frame, mask = F.split_archive_panel(handle.read(members[0])); values = tick_geometry(frame, mask)
            records.append({"image_id": row.image_id, **{f"tick_{i:02d}": float(value) for i, value in enumerate(values)}})
    result = pd.DataFrame(records); result.to_csv(OUT / "tick_geometry.csv", index=False)
    parsed = result.tick_08.eq(1)
    write_json(OUT / "feature_audit.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "images": 418, "parsed_tick_spacing": int(parsed.sum()), "features_per_image": 9,
        "spacing_min_median_max": [float(result.loc[parsed, "tick_07"].min()),
            float(result.loc[parsed, "tick_07"].median()), float(result.loc[parsed, "tick_07"].max())],
        "labels_used_in_extraction": [], "sha256": digest(OUT / "tick_geometry.csv")})
    print(f"Depth-tick geometry extracted: {parsed.sum()}/418 parsed", flush=True)


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists(): raise FileExistsError("Preserve completed tick predictions")
    patients, kidneys, images = F.matched_reference(); labels = kidneys.kidney_APN.to_numpy(int)
    tick = pd.read_csv(OUT / "tick_geometry.csv").set_index("image_id").reindex(images.image_id).to_numpy(float)
    relative = pd.read_csv(G.OUT / "relative_geometry.csv").set_index("image_id").reindex(images.image_id)
    relative = relative[[column for column in relative if column.startswith("geometry_")]].to_numpy(float)
    geometry = np.concatenate([tick, relative], axis=1)
    clean = pd.read_csv(F.OUT / "radiomics_clean_fullframe.csv").set_index("image_id").reindex(images.image_id)
    clean = clean[[column for column in clean if not column.endswith(("_Energy", "_TotalEnergy"))]].to_numpy(float)
    deep = np.concatenate([np.load(D.OUT / name) for name in
                           ("radimagenet_gray.npy", "radimagenet_color.npy", "usf_gray.npy")], axis=1)
    oriented_geometry = F.orient_kidneys(F.patient_pair_features(geometry, images, patients), patients, kidneys)
    oriented_clean = F.orient_kidneys(F.patient_pair_features(clean, images, patients), patients, kidneys)
    oriented_deep = F.orient_kidneys(F.patient_pair_features(deep, images, patients), patients, kidneys)
    early = np.concatenate([oriented_deep, oriented_clean, oriented_geometry], axis=1)
    assignments = pd.read_csv(F.OUT / "patient_assignments.csv")
    reference = pd.read_csv(D.OUT / "development_predictions.csv")
    reference = reference[reference.group.eq(REFERENCE)].set_index(["seed", "kidney_alias"]).score
    records, predictions = [], []; (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in F.SEEDS:
            roles = kidneys.patient_alias.map(assignments[assignments.seed.eq(seed)].set_index("patient_alias").role)
            train, dev = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
            scores, models = {}, {}
            score, transform, model = F.fit_predict(oriented_geometry, train, dev, labels, seed, 32)
            scores[GROUPS[0]], models[GROUPS[0]] = score, (transform, model)
            ref = np.asarray([reference.loc[(seed, alias)] for alias in kidneys.kidney_alias.iloc[dev]])
            scores[GROUPS[1]] = (score + ref) / 2
            for group, k in ((GROUPS[2], 64), (GROUPS[3], 128)):
                score, transform, model = F.fit_predict(early, train, dev, labels, seed, k)
                scores[group], models[group] = score, (transform, model)
            joblib.dump(models, OUT / "models" / f"{seed}_tick_geometry.joblib", compress=3)
            for group in GROUPS:
                result = metric(labels[dev], scores[group]); records.append({"seed": seed, "role": "development",
                    "group": group, "patients": 20, "kidneys": len(dev), **result})
                for local, index in enumerate(dev): predictions.append({"seed": seed, "group": group,
                    "patient_alias": kidneys.patient_alias.iloc[index], "kidney_alias": kidneys.kidney_alias.iloc[index],
                    "DMSA_APN": kidneys.DMSA_APN.iloc[index], "side": kidneys.side.iloc[index],
                    "label": int(labels[index]), "score": float(scores[group][local])})
            print(f"Completed depth-tick geometry draw {seed}", flush=True)
    frame = pd.DataFrame(records); frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    summary = frame.groupby("group")[["auroc", "ap"]].mean().sort_values(["auroc", "ap"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc > .8) & (summary.ap > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {"groups_above_both_development_targets": achieved,
        "test_scored": False, "goal_achieved": False,
        "reason": "Development-only availability subgroup cannot complete the full objective"})
    print(summary.to_string(), flush=True)


def bootstrap() -> None:
    require_lock(); frame = pd.read_csv(OUT / "development_predictions.csv")
    previous = pd.read_csv(D.OUT / "development_predictions.csv"); reference = previous[previous.group.eq(REFERENCE)]
    ci, paired, samples = cluster_intervals(pd.concat([frame, reference], ignore_index=True),
        list(GROUPS) + [REFERENCE], F.SEEDS, REFERENCE)
    ci.to_csv(OUT / "development_confidence_intervals.csv", index=False)
    paired.to_csv(OUT / "development_paired_intervals.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    write_json(OUT / "completion_audit.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "patients": 111, "kidneys": 212, "images": 418, "test_scored": False,
        "scored_unit": "kidney", "bootstrap_cluster": "patient"})
    print("Depth-tick geometry intervals completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "extract", "run", "bootstrap"])
    arguments = parser.parse_args(); globals()[arguments.stage]()
