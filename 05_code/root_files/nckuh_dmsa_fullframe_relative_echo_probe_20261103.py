"""Within-frame relative renal echogenicity on clean NCKUH full frames."""
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
from scipy import ndimage
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullframe_clean_probe_20261102 as F
import nckuh_dmsa_fullframe_deep_probe_20261102 as D
import nckuh_dmsa_fullframe_extreme_specialist_20261102 as E
import nckuh_dmsa_fullframe_hierarchical_probe_20261102 as H
import nckuh_dmsa_fullframe_relative_geometry_probe_20261102 as G
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullframe_relative_echo_probe_20261103"
REFERENCE = "deep_concat_plus_clean_radiomics"
POWERSET_REFERENCE = "four_state_powerset_plus_deep"
GEOMETRY_REFERENCE = "relative_frame_geometry_plus_deep_radiomics"
GROUPS = (
    "relative_context_echo_anova32_lr",
    "relative_context_echo_geometry_anova64_lr",
    "relative_context_echo_plus_deep_radiomics",
    "relative_context_echo_extreme_replace_plus_deep",
    "relative_context_echo_extreme_halfblend_plus_deep",
    "fixed_powerset_geometry_relative_echo_three_source_fusion",
    "early_concat_deep_radiomics_relative_echo_geometry_anova128_lr",
)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered relative-echo protocol")
    F.require_lock(); D.require_lock(); G.require_lock(); H.require_lock()
    inputs = [F.ARCHIVE, F.OUT / "patient_assignments.csv", F.OUT / "deidentified_image_manifest.csv",
              F.OUT / "radiomics_clean_fullframe.csv", D.OUT / "radimagenet_gray.npy",
              D.OUT / "radimagenet_color.npy", D.OUT / "usf_gray.npy",
              D.OUT / "development_predictions.csv", G.OUT / "relative_geometry.csv",
              G.OUT / "development_predictions.csv", H.OUT / "development_predictions.csv",
              *(H.OUT / "models" / f"{seed}_hierarchical.joblib" for seed in F.SEEDS), Path(__file__)]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same clean-full-frame 111 patients, 212 kidneys, 418 views and exact 70/20/21 patient roles",
        "motivation": "APN ultrasound findings can include renal enlargement, altered parenchymal echogenicity and impaired corticomedullary differentiation. Per-ROI P1-P99 scaling removes absolute echo shifts, so this experiment uses same-frame local tissue as a robust acquisition reference",
        "references": [
            "https://pubmed.ncbi.nlm.nih.gov/28525662/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC6177687/",
            "https://pubmed.ncbi.nlm.nih.gov/3513490/",
            "https://pubmed.ncbi.nlm.nih.gov/12069464/",
        ],
        "reference_region": "A label-independent perirenal annulus from 0.12 to 0.80 equivalent renal radii outside the supplied contour, intersected with the detected ultrasound sector. Near-black and saturated pixels are excluded; sector-wide fallback is explicit",
        "features": "Renal intensities, local gradients and local dispersion expressed relative to the annulus median and robust IQR; three normalized distance-to-boundary bands and their contrasts. These geometric bands are not named cortex, medulla or lesions",
        "artifact_robustness": "Robust quantiles and exclusion of saturated reference pixels reduce, but do not claim to remove, burned-in calipers and dotted measurement lines",
        "models": "View-preserving own/opposite-kidney representation; fixed training-only ANOVA32/64/128 balanced LR C=.01; a B/N-only symmetric specialist may only reallocate powerset N+B mass",
        "fusion": "Only preregistered equal-probability fusion with completed deep-radiomics, powerset-deep and relative-geometry-deep branches",
        "prohibited": "No labels in extraction, no DMSA field as predictor, no identifier/date/diagnosis suffix, no development-selected feature/weight/threshold, no exploratory-test score",
        "target": "Development AUROC>0.8 and AP>0.8; later locked confirmation remains necessary",
        "groups": list(GROUPS), "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {"source_sha256": digest(__file__),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs}})
    print("Registered within-frame relative renal echogenicity probe", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(Path(name)) == expected, name


def _summary(values: np.ndarray) -> np.ndarray:
    return np.asarray([values.mean(), values.std(), *np.percentile(values, [10, 25, 50, 75, 90])],
                      dtype=np.float32)


def relative_echo_features(gray: np.ndarray, mask: np.ndarray,
                           sector: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    image = np.asarray(gray, dtype=np.float32)
    binary = np.asarray(mask, dtype=bool)
    if image.ndim != 2 or image.shape != binary.shape or binary.sum() < 1024:
        raise ValueError("Invalid full-frame kidney image/mask")
    if sector is None:
        sector = np.ones_like(binary)
    sector = np.asarray(sector, dtype=bool)
    equivalent_radius = np.sqrt(binary.sum() / np.pi)
    outside_distance = ndimage.distance_transform_edt(~binary)
    annulus = (~binary) & sector & (outside_distance >= .12 * equivalent_radius) & (
        outside_distance <= .80 * equivalent_radius)
    valid_reference = annulus & (image > 4) & (image < 251)
    fallback = False
    if valid_reference.sum() < 1024:
        valid_reference = (~binary) & sector & (image > 4) & (image < 251)
        fallback = True
    reference = image[valid_reference]
    if len(reference) < 1024:
        raise ValueError("Insufficient same-frame reference tissue")
    low, high = np.percentile(reference, [2, 98])
    reference = reference[(reference >= low) & (reference <= high)]
    median = float(np.median(reference))
    q25, q75 = np.percentile(reference, [25, 75])
    scale = float(q75 - q25)
    if scale < 2:
        scale = max(float(np.median(np.abs(reference - median))) / .6745, 2.0)
    z_image = (image - median) / scale
    kidney = z_image[binary]

    nearest = ndimage.distance_transform_edt(~binary, return_distances=False, return_indices=True)
    extended = image[tuple(nearest)]
    gradient = np.hypot(ndimage.sobel(extended, axis=1), ndimage.sobel(extended, axis=0)) / scale
    local_mean = ndimage.uniform_filter(extended, size=5)
    local_square = ndimage.uniform_filter(extended * extended, size=5)
    local_sd = np.sqrt(np.maximum(local_square - local_mean * local_mean, 0)) / scale

    inside_distance = ndimage.distance_transform_edt(binary)
    normalized = inside_distance / max(float(inside_distance.max()), 1.0)
    regions = [binary & (normalized <= 1 / 3),
               binary & (normalized > 1 / 3) & (normalized <= 2 / 3),
               binary & (normalized > 2 / 3)]
    region_summary = [_summary(z_image[region]) for region in regions]
    features = [_summary(kidney),
                np.asarray([np.median(np.abs(kidney)), np.percentile(np.abs(kidney), 90)], dtype=np.float32),
                np.asarray([(kidney < -1).mean(), (kidney < -.5).mean(),
                            (kidney > .5).mean(), (kidney > 1).mean()], dtype=np.float32),
                *region_summary,
                np.asarray([region_summary[2][0] - region_summary[0][0],
                            region_summary[2][4] - region_summary[0][4],
                            region_summary[1][0] - region_summary[0][0],
                            region_summary[2][0] - region_summary[1][0]], dtype=np.float32)]
    for field in (gradient, local_sd):
        features.append(np.asarray([value for region in regions
                                    for value in (field[region].mean(), np.percentile(field[region], 90))],
                                   dtype=np.float32))
    output = np.concatenate(features).astype(np.float32)
    assert output.shape == (50,) and np.isfinite(output).all()
    audit = {"annulus_pixels": int(annulus.sum()), "reference_pixels": int(len(reference)),
             "reference_median": median, "reference_iqr": scale, "reference_fallback": bool(fallback)}
    return output, audit


def extract() -> None:
    require_lock()
    if (OUT / "relative_echo.csv").exists():
        raise FileExistsError("Preserve completed relative-echo extraction")
    _, _, images = F.matched_reference(); archive_index = F.archive_index()
    rows, fallbacks = [], 0
    with zipfile.ZipFile(F.ARCHIVE) as handle:
        for index, row in enumerate(images.itertuples(index=False)):
            members = archive_index[(row.patient_id, row.date_key, row.view)]
            if len(members) != 1:
                raise ValueError(f"Nonunique archive key: {row.image_id}")
            frame, mask = F.split_archive_panel(handle.read(members[0]))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sector, _ = G.ultrasound_sector(frame, mask)
            values, audit = relative_echo_features(gray, mask, sector)
            fallbacks += int(audit["reference_fallback"])
            rows.append({"image_id": row.image_id,
                         **{f"relative_echo_{feature:02d}": float(value) for feature, value in enumerate(values)},
                         **audit})
            if (index + 1) % 100 == 0:
                print(f"Extracted relative echo {index + 1}/{len(images)}", flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(OUT / "relative_echo.csv", index=False)
    write_json(OUT / "feature_audit.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "images": 418, "features_per_image": 50, "reference_fallbacks": fallbacks,
        "labels_used_in_extraction": [], "sha256": digest(OUT / "relative_echo.csv")})
    print(f"Extracted relative echo from 418 full frames; reference fallbacks {fallbacks}", flush=True)


def fit_extreme(values: np.ndarray, train: np.ndarray, states: np.ndarray, seed: int):
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    half = values.shape[1] // 2
    left, right = values[:, :half], values[:, half:]
    symmetric = np.c_[left + right, np.abs(left - right)]
    extreme = train[np.isin(states[train], [0, 3])]
    target = (states[extreme] == 3).astype(int)
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          VarianceThreshold(), SelectKBest(f_classif, k=32),
                          LogisticRegression(C=.01, class_weight="balanced", solver="liblinear",
                                             max_iter=3000, random_state=seed))
    model.fit(symmetric[extreme], target)
    return model.predict_proba(symmetric)[:, 1], model


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed relative-echo predictions")
    patients, kidneys, images = F.matched_reference(); labels = kidneys.kidney_APN.to_numpy(int)
    states = patients.DMSA_APN.map(H.CODE).to_numpy(int)
    echo_frame = pd.read_csv(OUT / "relative_echo.csv").set_index("image_id").reindex(images.image_id)
    echo = echo_frame[[column for column in echo_frame if column.startswith("relative_echo_")]].to_numpy(float)
    geometry_frame = pd.read_csv(G.OUT / "relative_geometry.csv").set_index("image_id").reindex(images.image_id)
    geometry = geometry_frame[[column for column in geometry_frame if column.startswith("geometry_")]].to_numpy(float)
    clean_frame = pd.read_csv(F.OUT / "radiomics_clean_fullframe.csv").set_index("image_id").reindex(images.image_id)
    clean = clean_frame[[column for column in clean_frame if not column.endswith(("_Energy", "_TotalEnergy"))]].to_numpy(float)
    deep = np.concatenate([np.load(D.OUT / name) for name in
                           ("radimagenet_gray.npy", "radimagenet_color.npy", "usf_gray.npy")], axis=1)
    echo_pairs = F.patient_pair_features(echo, images, patients)
    oriented_echo = F.orient_kidneys(echo_pairs, patients, kidneys)
    oriented_geometry = F.orient_kidneys(F.patient_pair_features(geometry, images, patients), patients, kidneys)
    oriented_clean = F.orient_kidneys(F.patient_pair_features(clean, images, patients), patients, kidneys)
    oriented_deep = F.orient_kidneys(F.patient_pair_features(deep, images, patients), patients, kidneys)
    echo_geometry = np.c_[oriented_echo, oriented_geometry]
    early = np.c_[oriented_deep, oriented_clean, oriented_echo, oriented_geometry]
    assignments = pd.read_csv(F.OUT / "patient_assignments.csv")
    deep_predictions = pd.read_csv(D.OUT / "development_predictions.csv")
    deep_reference = deep_predictions[deep_predictions.group.eq(REFERENCE)].set_index(["seed", "kidney_alias"]).score
    power_predictions = pd.read_csv(H.OUT / "development_predictions.csv")
    powerset_reference = power_predictions[power_predictions.group.eq(POWERSET_REFERENCE)].set_index(["seed", "kidney_alias"]).score
    geometry_predictions = pd.read_csv(G.OUT / "development_predictions.csv")
    geometry_reference = geometry_predictions[geometry_predictions.group.eq(GEOMETRY_REFERENCE)].set_index(["seed", "kidney_alias"]).score
    patient_lookup = {alias: index for index, alias in enumerate(patients.patient_alias)}

    records, predictions = [], []
    (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in F.SEEDS:
            mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
            train_patients = np.flatnonzero(patients.patient_alias.map(mapping).eq("train"))
            roles = kidneys.patient_alias.map(mapping)
            train, development = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
            aliases = kidneys.kidney_alias.iloc[development]
            deep_score = np.asarray([deep_reference.loc[(seed, alias)] for alias in aliases])
            power_score = np.asarray([powerset_reference.loc[(seed, alias)] for alias in aliases])
            geometry_score = np.asarray([geometry_reference.loc[(seed, alias)] for alias in aliases])
            echo_score, echo_transform, echo_model = F.fit_predict(oriented_echo, train, development, labels, seed, 32)
            combined_score, combined_transform, combined_model = F.fit_predict(
                echo_geometry, train, development, labels, seed, 64)
            early_score, early_transform, early_model = F.fit_predict(early, train, development, labels, seed, 128)
            extreme_probability, extreme_model = fit_extreme(echo_pairs, train_patients, states, seed)
            saved = joblib.load(H.OUT / "models" / f"{seed}_hierarchical.joblib")
            powerset_states = E.powerset_state_probabilities(
                saved["powerset_model"], saved["powerset_pre"], F.patient_pair_features(deep, images, patients))
            replaced = E.replace_extreme_probability(powerset_states, extreme_probability)
            halfblend = E.replace_extreme_probability(powerset_states, extreme_probability, True)

            def kidney_values(state_values):
                left, right = E.kidney_marginals(state_values)
                return np.asarray([left[patient_lookup[kidneys.patient_alias.iloc[index]]]
                                   if kidneys.side.iloc[index] == "left"
                                   else right[patient_lookup[kidneys.patient_alias.iloc[index]]]
                                   for index in development])

            replaced_score = kidney_values(replaced); halfblend_score = kidney_values(halfblend)
            scores = {GROUPS[0]: echo_score, GROUPS[1]: combined_score,
                      GROUPS[2]: (echo_score + deep_score) / 2,
                      GROUPS[3]: (replaced_score + deep_score) / 2,
                      GROUPS[4]: (halfblend_score + deep_score) / 2,
                      GROUPS[5]: (power_score + geometry_score + echo_score) / 3,
                      GROUPS[6]: early_score}
            joblib.dump({"echo": (echo_transform, echo_model),
                         "echo_geometry": (combined_transform, combined_model),
                         "early": (early_transform, early_model), "extreme": extreme_model},
                        OUT / "models" / f"{seed}_relative_echo.joblib", compress=3)
            for group in GROUPS:
                result = metric(labels[development], scores[group])
                records.append({"seed": seed, "role": "development", "group": group,
                                "patients": 20, "kidneys": len(development), **result})
                for local, row_index in enumerate(development):
                    predictions.append({"seed": seed, "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[row_index],
                        "kidney_alias": kidneys.kidney_alias.iloc[row_index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[row_index], "side": kidneys.side.iloc[row_index],
                        "label": int(labels[row_index]), "score": float(scores[group][local])})
            print(f"Completed relative-echo draw {seed}", flush=True)
    metric_frame = pd.DataFrame(records); metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap"]].mean().sort_values(["auroc", "ap"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc > .8) & (summary.ap > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {"groups_above_both_development_targets": achieved,
        "test_scored": False, "goal_achieved": False,
        "reason": "Development-only subgroup cannot complete the full objective"})
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
    print("Relative-echo intervals completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "extract", "run", "bootstrap"])
    arguments = parser.parse_args(); globals()[arguments.stage]()
