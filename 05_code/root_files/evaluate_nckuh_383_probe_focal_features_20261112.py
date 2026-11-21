"""Probe-specific APN screen of label-free renal echo and shape proxies."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_nckuh_383_acquisition_strata_20261112 import bootstrap, metric, threshold_for_sensitivity
from prepare_nckuh_383_contour_cohort_20261111 import digest


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
AUDIT = EXP / "NCKUH_revised_383_audit_20261111"
CACHE = EXP / "NCKUH_revised_383_contour_cache_20261111"
ROUTER = EXP / "NCKUH_revised_383_probe_router_20261112"
OUT = ROUTER / "focal_echo_feature_screen"
SEED = 20261112


def image_features(image_id):
    gray = cv2.imread(str(CACHE / "clean_rgb" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(CACHE / "masks" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    if gray is None or mask is None or gray.shape != mask.shape:
        raise ValueError(f"Missing or mismatched ROI: {image_id}")
    inside = mask > 127
    values = gray[inside].astype(np.float32)
    if len(values) < 64:
        raise ValueError(f"Insufficient masked pixels: {image_id}")
    median = float(np.median(values))
    scale = max(float(np.percentile(values, 75) - np.percentile(values, 25)), 5.0)
    normalized = np.clip((gray.astype(np.float32) - median) / scale, -5, 5)
    z = normalized[inside]
    echo = [float(np.mean(z < -1.5)), float(np.mean(z < -.75)),
            float(np.mean(z > .75)), float(np.mean(z > 1.5)),
            float(np.percentile(z, 10)), float(np.percentile(z, 90)),
            float(np.std(z))]
    texture = []
    for kernel in (9, 21):
        local_mean = cv2.GaussianBlur(normalized, (kernel, kernel), 0)
        residual = np.abs(normalized - local_mean)[inside]
        texture.extend((float(np.median(residual)), float(np.percentile(residual, 90)),
                        float(np.mean(residual > 1.0))))
    gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)[inside]
    texture.extend((float(np.median(gradient)), float(np.percentile(gradient, 90))))
    contours, _ = cv2.findContours(inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    area = float(inside.sum())
    perimeter = float(cv2.arcLength(contour, True))
    yy, xx = np.nonzero(inside)
    shape = [area / gray.size, len(xx) / max(1, (xx.max() - xx.min() + 1) * (yy.max() - yy.min() + 1)),
             (xx.max() - xx.min() + 1) / max(1, (yy.max() - yy.min() + 1)),
             4 * np.pi * area / max(perimeter ** 2, 1.0)]
    return np.asarray(echo + texture + shape, np.float32)


def kidney_features(image_features_array, image_table, kidney_table, columns):
    grouped = image_table.groupby("kidney_alias", sort=False).indices
    selected = image_features_array[:, columns]
    return np.asarray([np.r_[selected[grouped[alias]].mean(0), selected[grouped[alias]].std(0)]
                       for alias in kidney_table.kidney_alias], dtype=np.float64)


def model(c):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         LogisticRegression(C=c, class_weight="balanced", solver="liblinear",
                                            max_iter=3000, random_state=SEED))


def main():
    images = pd.read_csv(AUDIT / "contour_image_manifest.csv")
    kidneys = pd.read_csv(AUDIT / "contour_kidney_manifest.csv")
    people = pd.read_csv(AUDIT / "patient_assignments.csv")
    routes = pd.read_csv(ROUTER / "image_routes.csv")
    assert routes.image_id.tolist() == images.image_id.tolist()
    features = np.stack([image_features(image_id) for image_id in images.image_id])
    assert features.shape == (1362, 19) and np.isfinite(features).all()
    report = {"router_sha256": digest(ROUTER / "router_validation.json"),
              "features": "Seven within-kidney relative echo statistics; eight two-scale local texture/gradient statistics; four mask shape statistics. No anatomical compartment, lesion, probe frequency, or APN label is inferred at extraction.",
              "candidate_sets": {"echo": list(range(7)), "echo_texture": list(range(15)),
                                 "echo_texture_shape": list(range(19))},
              "C_values": [.01, .1, 1.0],
              "selection": "Per-probe 4-fold patient-level training OOF, maximize min(AUROC, AP), AUROC tie-break. Development labels not used for selection.",
              "test_APN_scored": False, "groups": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": images.image_id, **{f"proxy_{i:02d}": features[:, i]
                  for i in range(features.shape[1])}}).to_csv(OUT / "image_features.csv", index=False)
    for probe in ("convex", "linear"):
        row_ids = np.flatnonzero(routes.route.eq(probe))
        sub_image = images.iloc[row_ids].reset_index(drop=True)
        active = set(people.loc[people.role.isin(("train", "development")), "patient_alias"])
        retain = sub_image.patient_alias.isin(active).to_numpy()
        sub_image = sub_image.loc[retain].reset_index(drop=True)
        sub_features = features[row_ids[retain]]
        sub_kidney = kidneys[kidneys.kidney_alias.isin(sub_image.kidney_alias)].reset_index(drop=True)
        train_people = people[people.role.eq("train") & people.patient_alias.isin(sub_image.patient_alias)].reset_index(drop=True)
        train_idx = np.flatnonzero(sub_kidney.patient_alias.isin(train_people.patient_alias))
        dev_idx = np.flatnonzero(~sub_kidney.patient_alias.isin(train_people.patient_alias))
        assert set(sub_kidney.patient_alias.iloc[train_idx]).isdisjoint(sub_kidney.patient_alias.iloc[dev_idx])
        split = list(StratifiedKFold(4, shuffle=True, random_state=SEED).split(train_people, train_people.DMSA_APN))
        candidates = {}
        for name, columns in report["candidate_sets"].items():
            x = kidney_features(sub_features, sub_image, sub_kidney, columns)
            for c in report["C_values"]:
                oof = np.full(len(sub_kidney), np.nan)
                for fit, hold in split:
                    fit_idx = np.flatnonzero(sub_kidney.patient_alias.isin(train_people.patient_alias.iloc[fit]))
                    hold_idx = np.flatnonzero(sub_kidney.patient_alias.isin(train_people.patient_alias.iloc[hold]))
                    oof[hold_idx] = model(c).fit(x[fit_idx], sub_kidney.kidney_APN.iloc[fit_idx]).predict_proba(x[hold_idx])[:, 1]
                assert np.isfinite(oof[train_idx]).all()
                key = f"{name}_C{c}"
                candidates[key] = metric(sub_kidney.kidney_APN.iloc[train_idx], oof[train_idx])
                candidates[key]["scores"] = oof[train_idx]
        winner = max(candidates, key=lambda key: (min(candidates[key]["auroc"], candidates[key]["ap"]), candidates[key]["auroc"]))
        name, c_string = winner.split("_C")
        threshold, _ = threshold_for_sensitivity(sub_kidney.kidney_APN.iloc[train_idx].to_numpy(int), candidates[winner]["scores"])
        for item in candidates.values():
            item.pop("scores")
        x = kidney_features(sub_features, sub_image, sub_kidney, report["candidate_sets"][name])
        score = model(float(c_string)).fit(x[train_idx], sub_kidney.kidney_APN.iloc[train_idx]).predict_proba(x[dev_idx])[:, 1]
        target = sub_kidney.iloc[dev_idx].reset_index(drop=True)
        labels = target.kidney_APN.to_numpy(int)
        report["groups"][probe] = {"train_patients": len(train_people), "development_patients": target.patient_alias.nunique(),
                                   "development_kidneys": len(target), "selected": winner, "training_oof": candidates,
                                   "training_oof_threshold": threshold,
                                   "development": metric(labels, score) | bootstrap(target, score)}
        target[["patient_alias", "kidney_alias", "side", "kidney_APN"]].assign(
            score=score, selected=winner).to_csv(OUT / f"{probe}_development_predictions.csv", index=False)
        print(probe, winner, report["groups"][probe]["development"], flush=True)
    (OUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
