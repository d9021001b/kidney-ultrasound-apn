"""Field-of-view-normalized renal geometry from clean NCKUH full-frame masks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile

import cv2
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullframe_clean_probe_20261102 as F
import nckuh_dmsa_fullframe_deep_probe_20261102 as D
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullframe_relative_geometry_probe_20261102"
REFERENCE = "deep_concat_plus_clean_radiomics"
GROUPS = (
    "relative_frame_geometry_anova32_lr",
    "relative_frame_geometry_plus_deep_radiomics",
    "early_concat_deep_radiomics_geometry_anova64_lr",
    "early_concat_deep_radiomics_geometry_anova128_lr",
)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered relative-geometry protocol")
    F.require_lock(); D.require_lock()
    inputs = [F.ARCHIVE, F.OUT / "patient_assignments.csv", F.OUT / "deidentified_image_manifest.csv",
              F.OUT / "radiomics_clean_fullframe.csv", D.OUT / "radimagenet_gray.npy",
              D.OUT / "radimagenet_color.npy", D.OUT / "usf_gray.npy",
              D.OUT / "development_predictions.csv", Path(__file__)]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same clean-full-frame 111 patients, 212 kidneys, 418 images and exact 70/20/21 patient roles",
        "motivation": "Error decomposition localized the remaining ranking deficit to bilateral-APN versus bilateral-normal patients, where contralateral asymmetry cannot help",
        "features": "Label-independent contour shape and renal size normalized by full frame and detected ultrasound-sector dimensions/area; no uncalibrated pixel length is called a physical measurement",
        "sector": "Threshold nonzero scan content below the header, morphological closing, then select the connected component with maximal overlap with the dilated renal mask; frame fallback is explicit",
        "models": "Ordered own/opposite kidney and longitudinal/transverse context; fixed ANOVA32/64/128 balanced LR and one equal-probability fusion with completed deep-radiomics",
        "prohibited": "No age, sex, DMSA code, filename suffix, OCR diagnosis, development-selected feature/weight/threshold, or exploratory-test scoring",
        "target": "Development AUROC>0.8 and AP>0.8; later locked confirmation remains necessary",
        "groups": list(GROUPS), "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {"source_sha256": digest(__file__),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs}})
    print("Registered full-frame relative renal-geometry probe", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items(): assert digest(Path(name)) == expected, name


def ultrasound_sector(frame: np.ndarray, kidney_mask: np.ndarray) -> tuple[np.ndarray, bool]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    binary = (gray > 4).astype(np.uint8)
    binary[:max(2, gray.shape[0] // 20)] = 0
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    dilated = cv2.dilate(kidney_mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    best, best_score = None, -1.0
    for label in range(1, count):
        component = labels == label
        overlap = np.logical_and(component, dilated).sum()
        score = overlap + 1e-4 * stats[label, cv2.CC_STAT_AREA]
        if overlap and score > best_score:
            best, best_score = component, score
    if best is None:
        return np.ones_like(kidney_mask, dtype=bool), True
    return best, False


def relative_geometry(frame: np.ndarray, kidney_mask: np.ndarray) -> tuple[np.ndarray, bool]:
    mask = kidney_mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    area = float(mask.sum()); perimeter = float(cv2.arcLength(contour, True))
    hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
    yy, xx = np.nonzero(mask)
    covariance = np.cov(np.c_[xx, yy], rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1e-8)
    minor, major = 4 * np.sqrt(eigenvalues[0]), 4 * np.sqrt(eigenvalues[1])
    x0, x1, y0, y1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
    bbox_w, bbox_h = float(x1 - x0), float(y1 - y0)
    cx, cy = float(xx.mean()), float(yy.mean())
    sector, fallback = ultrasound_sector(frame, mask)
    sy, sx = np.nonzero(sector)
    sector_area = max(float(sector.sum()), 1.0)
    sx0, sx1, sy0, sy1 = sx.min(), sx.max() + 1, sy.min(), sy.max() + 1
    sector_w, sector_h = float(sx1 - sx0), float(sy1 - sy0)
    height, width = mask.shape
    values = np.asarray([
        area / (height * width), area / sector_area,
        major / height, minor / width, major / np.sqrt(sector_area), minor / np.sqrt(sector_area),
        bbox_w / width, bbox_h / height, bbox_w / sector_w, bbox_h / sector_h,
        major / max(minor, 1e-6), area / max(bbox_w * bbox_h, 1.0), area / hull_area,
        4 * np.pi * area / max(perimeter * perimeter, 1e-6),
        cx / width, cy / height, (cx - sx0) / sector_w, (cy - sy0) / sector_h,
        sector_area / (height * width), sector_w / width, sector_h / height,
    ], dtype=np.float32)
    assert values.shape == (21,) and np.isfinite(values).all()
    return values, fallback


def extract() -> None:
    require_lock()
    if (OUT / "relative_geometry.csv").exists():
        raise FileExistsError("Preserve completed relative geometry")
    _, _, images = F.matched_reference(); index = F.archive_index()
    records, fallbacks = [], 0
    with zipfile.ZipFile(F.ARCHIVE) as handle:
        for row in images.itertuples(index=False):
            key = (row.patient_id, row.date_key, row.view); members = index[key]
            if len(members) != 1: raise ValueError(f"Nonunique archive key: {row.image_id}")
            frame, mask = F.split_archive_panel(handle.read(members[0]))
            values, fallback = relative_geometry(frame, mask); fallbacks += int(fallback)
            records.append({"image_id": row.image_id, **{f"geometry_{i:02d}": float(v) for i, v in enumerate(values)},
                            "sector_fallback": int(fallback)})
    result = pd.DataFrame(records); result.to_csv(OUT / "relative_geometry.csv", index=False)
    write_json(OUT / "feature_audit.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "images": 418, "features_per_image": 21, "sector_fallbacks": fallbacks,
        "labels_used_in_extraction": [], "sha256": digest(OUT / "relative_geometry.csv")})
    print(f"Extracted relative geometry from 418 full frames; sector fallbacks {fallbacks}", flush=True)


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists(): raise FileExistsError("Preserve completed geometry predictions")
    patients, kidneys, images = F.matched_reference(); labels = kidneys.kidney_APN.to_numpy(int)
    geometry = pd.read_csv(OUT / "relative_geometry.csv").set_index("image_id").reindex(images.image_id)
    geometry = geometry[[column for column in geometry if column.startswith("geometry_")]].to_numpy(float)
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
            joblib.dump(models, OUT / "models" / f"{seed}_relative_geometry.joblib", compress=3)
            for group in GROUPS:
                result = metric(labels[dev], scores[group]); records.append({"seed": seed, "role": "development",
                    "group": group, "patients": 20, "kidneys": len(dev), **result})
                for local, index in enumerate(dev): predictions.append({"seed": seed, "group": group,
                    "patient_alias": kidneys.patient_alias.iloc[index], "kidney_alias": kidneys.kidney_alias.iloc[index],
                    "DMSA_APN": kidneys.DMSA_APN.iloc[index], "side": kidneys.side.iloc[index],
                    "label": int(labels[index]), "score": float(scores[group][local])})
            print(f"Completed relative-geometry draw {seed}", flush=True)
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
    print("Relative-geometry intervals completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "extract", "run", "bootstrap"])
    arguments = parser.parse_args(); globals()[arguments.stage]()
