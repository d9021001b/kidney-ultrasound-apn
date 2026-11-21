"""Contour-first clinical ultrasound features for the full NCKUH DMSA cohort."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy import ndimage
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as F
import nckuh_dmsa_fullframe_relative_echo_probe_20261103 as E
import nckuh_dmsa_fullframe_relative_geometry_probe_20261102 as G
from nckuh_only_study_20261102 import digest, write_json


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_clinical_features_20261103"
WORKBOOK = ROOT / "2004-2019-NCKUH-\u4e9e\u5927-20210723.xlsx"
REFERENCE = "fullcohort_rad_color_image_plus_patient_direct"
GROUPS = (
    "clinical_age_residual_anova64_lr",
    "clinical_age_residual_anova128_lr",
    "clinical_plus_radimagenet_anova128_lr",
    "clinical_state_pca32_lr",
    "clinical_anova64_plus_instance_reference",
    "clinical_early_plus_instance_reference",
    "clinical_state_plus_instance_reference",
)


def load_reference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference as load

    patients, kidneys, images = load()
    images = images[images.kidney_APN.notna()].copy().reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].copy().reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].copy().reset_index(drop=True)
    assert (len(patients), len(kidneys), len(images)) == (210, 414, 820)
    return patients, kidneys, images


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered full-cohort clinical-feature protocol")
    F.require_lock()
    inputs = [
        WORKBOOK,
        F.OUT / "patient_assignments.csv",
        F.OUT / "deidentified_source_manifest.csv",
        F.OUT / "radimagenet_color.npy",
        F.OUT / "instance_development_predictions.csv",
        Path(__file__),
    ]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same 210 patients, 414 imaged kidneys and 820 contour-first images; patient split and kidney endpoint unchanged",
        "motivation": "Published pediatric APN ultrasound findings include age-dependent renal enlargement or asymmetry, altered parenchymal echogenicity, focal polar abnormalities and reduced corticomedullary differentiation",
        "references": [
            "https://pubmed.ncbi.nlm.nih.gov/3513490/",
            "https://pubmed.ncbi.nlm.nih.gov/1651644/",
            "https://pubmed.ncbi.nlm.nih.gov/17987762/",
            "https://pubmed.ncbi.nlm.nih.gov/21459422/",
        ],
        "features": "Full-frame sector-normalized contour geometry; same-frame annulus-normalized echo; mask-constrained canonical polar, radial, multiscale LoG, local-dispersion and color-flow summaries",
        "normalization": "For each draw, geometry is residualized using only training images against log-age, age, age squared, sex, side, view and deidentified acquisition domain; no APN outcome enters residualization",
        "models": "Training-only ANOVA logistic heads, a four-state patient model converted to kidney marginals, and fixed equal-probability fusion with the unchanged image-instance reference",
        "prohibited": "No identifier, diagnosis suffix, DMSA code, development score or exploratory-test row enters image extraction or feature normalization",
        "target": "Development AUROC>0.8 and AP>0.8; locked confirmation is still required after model selection",
        "groups": list(GROUPS),
        "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "source_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs},
    })
    print("Registered full-cohort clinical-feature experiment", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(Path(__file__)) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(Path(name)) == expected, name


def _load_source(row) -> tuple[np.ndarray, np.ndarray]:
    source = Path(row.source_path_private)
    if row.source_type in ("aligned_panel", "aligned_panel_dateless"):
        return F.split_panel(source)
    if row.source_type == "labelme_polygon":
        image, mask = F.load_polygon(source, Path(row.annotation_path_private))
        repaired = False
        if ndimage.distance_transform_edt(mask).max() <= 1.5:
            annotation = json.loads(Path(row.annotation_path_private).read_text(encoding="utf-8"))
            points = [np.rint(shape["points"]).astype(np.int32)
                      for shape in annotation["shapes"] if len(shape.get("points", [])) >= 3]
            outline = np.zeros(mask.shape, np.uint8)
            cv2.polylines(outline, points, True, 1, 2)
            outline = cv2.morphologyEx(outline, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            flood = outline.copy()
            cv2.floodFill(flood, None, (0, 0), 1)
            mask = outline | (1 - flood)
            repaired = True
        if hasattr(row, "__dict__"):
            row.outline_repaired = repaired
        return image, mask
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(row.annotation_path_private), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise FileNotFoundError(source)
    return image, (mask > 0).astype(np.uint8)


def relative_echo_with_small_mask_support(frame: np.ndarray, mask: np.ndarray):
    """Preserve the registered echo calculation while supporting one tiny polygon."""
    if int(mask.sum()) >= 1024:
        sector, _ = G.ultrasound_sector(frame, mask)
        values, audit = E.relative_echo_features(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask, sector)
        return values, audit, False
    scale = max(2, int(np.ceil(np.sqrt(1280 / max(int(mask.sum()), 1)))))
    resized_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask.astype(np.uint8), None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_NEAREST)
    sector, _ = G.ultrasound_sector(resized_frame, resized_mask)
    values, audit = E.relative_echo_features(
        cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY), resized_mask, sector)
    return values, audit, True


def _robust_summary(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if not len(values):
        return np.full(8, np.nan, np.float32)
    return np.asarray([
        values.mean(), values.std(), *np.percentile(values, [5, 10, 25, 50, 75, 90])
    ], dtype=np.float32)


def focal_features(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Label-free focal and radial echo summaries inside a supplied kidney mask."""
    binary = np.asarray(mask, dtype=bool)
    if binary.sum() < 256:
        raise ValueError("Kidney mask is too small")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kidney = gray[binary]
    center = float(np.median(kidney))
    q25, q75 = np.percentile(kidney, [25, 75])
    scale = max(float(q75 - q25), float(np.median(np.abs(kidney - center))) / .6745, 2.0)
    z = (gray - center) / scale

    yy, xx = np.nonzero(binary)
    points = np.c_[xx - xx.mean(), yy - yy.mean()]
    covariance = np.cov(points, rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    major = points @ vectors[:, -1]
    minor = points @ vectors[:, 0]
    major /= max(float(np.percentile(np.abs(major), 98)), 1.0)
    minor /= max(float(np.percentile(np.abs(minor), 98)), 1.0)
    major_map = np.zeros_like(gray); minor_map = np.zeros_like(gray)
    major_map[binary] = major; minor_map[binary] = minor

    nearest = ndimage.distance_transform_edt(~binary, return_distances=False, return_indices=True)
    extended = gray[tuple(nearest)]
    gradient = np.hypot(ndimage.sobel(extended, axis=0), ndimage.sobel(extended, axis=1)) / scale
    inside = ndimage.distance_transform_edt(binary)
    radial = inside / max(float(inside.max()), 1.0)

    features: list[np.ndarray] = []
    longitudinal = [binary & (major_map < -1 / 3),
                    binary & (np.abs(major_map) <= 1 / 3),
                    binary & (major_map > 1 / 3)]
    transverse = [binary & (minor_map < -1 / 3),
                  binary & (np.abs(minor_map) <= 1 / 3),
                  binary & (minor_map > 1 / 3)]
    radial_regions = [binary & (radial <= 1 / 3),
                      binary & (radial > 1 / 3) & (radial <= 2 / 3),
                      binary & (radial > 2 / 3)]
    for regions in (longitudinal, transverse, radial_regions):
        summaries = [_robust_summary(z[region]) for region in regions]
        features.extend(summaries)
        pole_low = np.minimum(summaries[0], summaries[2])
        pole_high = np.maximum(summaries[0], summaries[2])
        features.extend([pole_low, pole_high, np.abs(summaries[0] - summaries[2])])

    for sigma in (1.0, 2.0, 4.0, 8.0):
        response = np.abs(ndimage.gaussian_laplace(extended / scale, sigma=sigma))[binary]
        features.append(np.asarray([response.mean(), response.std(),
                                    np.percentile(response, 90), np.percentile(response, 99)], np.float32))
    for size in (3, 7, 15, 31):
        mean = ndimage.uniform_filter(extended, size=size)
        square = ndimage.uniform_filter(extended * extended, size=size)
        local_sd = np.sqrt(np.maximum(square - mean * mean, 0))[binary] / scale
        features.append(np.asarray([local_sd.mean(), local_sd.std(),
                                    np.percentile(local_sd, 90)], np.float32))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[..., 1]
    hue = hsv[..., 0]
    value = hsv[..., 2]
    color = binary & (saturation >= 80) & (value >= 50)
    red = color & ((hue <= 12) | (hue >= 165))
    blue = color & (hue >= 90) & (hue <= 135)
    denominator = max(int(binary.sum()), 1)
    features.append(np.asarray([color.sum() / denominator, red.sum() / denominator,
                                blue.sum() / denominator, abs(int(red.sum()) - int(blue.sum())) / denominator],
                               np.float32))
    output = np.concatenate(features).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Non-finite focal features")
    return output


def _normalize_id(value: object) -> str:
    return F.normalize_patient_id(str(value))


def demographics(images: pd.DataFrame) -> pd.DataFrame:
    patient_dates = images.groupby(["patient_alias", "patient_id"], as_index=False).exam_date_iso.first()
    patient_dates["exam_date"] = pd.to_datetime(patient_dates.exam_date_iso, errors="coerce")
    rows = []
    for sheet in ("2007-2012", "2014-2019"):
        table = pd.read_excel(WORKBOOK, sheet_name=sheet)
        table["patient_id"] = table.id.map(_normalize_id)
        if sheet == "2007-2012":
            table["female"] = table.gender.astype(str).str.upper().eq("F").astype(float)
            table["reference_date"] = pd.to_datetime(table["\u4f4f\u9662\u65e5"], errors="coerce")
        else:
            table["female"] = table.gender.astype(str).str.replace(r"\.0$", "", regex=True).eq("2").astype(float)
            table["reference_date"] = pd.to_datetime(table.admission_date, errors="coerce")
        table["source_sheet"] = sheet
        rows.append(table[["patient_id", "age", "female", "reference_date", "source_sheet"]])
    source = pd.concat(rows, ignore_index=True)
    merged = patient_dates.merge(source, on="patient_id", how="left")
    merged["date_distance"] = (merged.exam_date - merged.reference_date).dt.days.abs()
    merged = merged.sort_values(["patient_alias", "date_distance"], na_position="last").drop_duplicates("patient_alias")
    merged["age_years"] = pd.to_numeric(merged.age, errors="coerce")
    merged.loc[~merged.age_years.between(0, 18), "age_years"] = np.nan
    result = merged[["patient_alias", "age_years", "female", "source_sheet"]].copy()
    assert len(result) == 210 and result.female.notna().sum() == 210
    return result


def _extract_one(payload: tuple[str, dict]) -> tuple[dict, int, int, int]:
    image_id, source_dict = payload
    source = SimpleNamespace(**source_dict)
    frame, mask = _load_source(source)
    geometry, sector_fallback = G.relative_geometry(frame, mask)
    echo, audit, small_mask_resized = relative_echo_with_small_mask_support(frame, mask)
    focal = focal_features(frame, mask)
    domain = "NCKUH3" if "NCKUH3" in source.source_path_private else (
        "test_predict" if "test_predict" in source.source_path_private else "other")
    record = {
        "image_id": image_id,
        "source_domain": domain,
        "mask_outline_repaired": int(getattr(source, "outline_repaired", False)),
        "sector_fallback": int(sector_fallback),
        "echo_reference_fallback": int(audit["reference_fallback"]),
        "small_mask_resized": int(small_mask_resized),
        **{f"geometry_{i:02d}": float(v) for i, v in enumerate(geometry)},
        **{f"echo_{i:02d}": float(v) for i, v in enumerate(echo)},
        **{f"focal_{i:03d}": float(v) for i, v in enumerate(focal)},
    }
    return record, int(sector_fallback), int(audit["reference_fallback"]), int(small_mask_resized)


def extract() -> None:
    require_lock()
    if (OUT / "clinical_image_features.csv").exists():
        raise FileExistsError("Preserve completed clinical-feature extraction")
    _, _, images = load_reference()
    plan = F.source_plan().set_index("image_id").reindex(images.image_id)
    partial_path = OUT / "clinical_image_features.partial.csv"
    records = [] if not partial_path.exists() else pd.read_csv(partial_path).to_dict("records")
    completed = {row["image_id"] for row in records}
    payloads = []
    for image, source in zip(images.itertuples(index=False), plan.itertuples(index=False)):
        if image.image_id in completed:
            continue
        payloads.append((image.image_id, source._asdict()))
    sector_fallbacks = echo_fallbacks = small_masks_resized = 0
    context = mp.get_context("spawn")
    with context.Pool(processes=6) as pool:
        for index, (record, sector_fallback, echo_fallback, small_mask_resized) in enumerate(
                pool.imap(_extract_one, payloads, chunksize=2), start=len(records) + 1):
            records.append(record)
            sector_fallbacks += sector_fallback
            echo_fallbacks += echo_fallback
            small_masks_resized += small_mask_resized
            if index % 50 == 0:
                pd.DataFrame(records).to_csv(partial_path, index=False)
                print(f"Extracted clinical features {index}/820", flush=True)
    feature_frame = pd.DataFrame(records).set_index("image_id").reindex(images.image_id).reset_index()
    feature_frame.to_csv(OUT / "clinical_image_features.csv", index=False)
    partial_path.unlink(missing_ok=True)
    demo = demographics(images)
    demo.to_csv(OUT / "deidentified_demographics.csv", index=False)
    write_json(OUT / "feature_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "images": len(feature_frame),
        "features_per_image": len([c for c in feature_frame if c.startswith(("geometry_", "echo_", "focal_"))]),
        "sector_fallbacks": int(feature_frame.sector_fallback.sum()),
        "echo_reference_fallbacks": int(feature_frame.echo_reference_fallback.sum()),
        "small_masks_resized_for_echo_only": int(feature_frame.small_mask_resized.sum()),
        "outline_only_labelme_masks_repaired": int(feature_frame.mask_outline_repaired.fillna(0).sum()),
        "age_available_patients": int(demo.age_years.notna().sum()),
        "sex_available_patients": int(demo.female.notna().sum()),
        "labels_used_in_extraction": [],
        "feature_sha256": digest(OUT / "clinical_image_features.csv"),
    })
    print("Completed label-free full-cohort clinical-feature extraction", flush=True)


def covariates(images: pd.DataFrame, features: pd.DataFrame, demo: pd.DataFrame) -> np.ndarray:
    table = images[["patient_alias", "side", "view"]].merge(demo, on="patient_alias", how="left")
    age = table.age_years.to_numpy(float)
    domains = pd.get_dummies(features.source_domain).reindex(columns=["NCKUH3", "test_predict", "other"], fill_value=0)
    return np.c_[np.log1p(age), age, age * age, table.female.to_numpy(float),
                 table.side.eq("right").to_numpy(float), table["view"].str.endswith("T").to_numpy(float),
                 domains.to_numpy(float)]


def residualize(values: np.ndarray, contexts: np.ndarray, training: np.ndarray, seed: int):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    value_imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(values[training])
    completed = value_imputer.transform(values)
    normalizer = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                               StandardScaler(), Ridge(alpha=.1, random_state=seed))
    normalizer.fit(contexts[training], completed[training])
    result = completed - normalizer.predict(contexts)
    center = np.median(result[training], axis=0)
    scale = np.median(np.abs(result[training] - center), axis=0) / .6745
    scale[scale < 1e-6] = 1.0
    return ((result - center) / scale).astype(np.float32), {
        "value_imputer": value_imputer, "normalizer": normalizer,
        "center": center, "scale": scale,
    }


def fit_anova(values: np.ndarray, train: np.ndarray, development: np.ndarray,
              labels: np.ndarray, seed: int, selected: int):
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    transform = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                              VarianceThreshold(), SelectKBest(f_classif, k=selected))
    x_train = transform.fit_transform(values[train], labels[train])
    model = LogisticRegression(C=.01, class_weight="balanced", solver="liblinear",
                               max_iter=3000, random_state=seed)
    model.fit(x_train, labels[train])
    return model.predict_proba(transform.transform(values[development]))[:, 1], transform, model


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed clinical-feature predictions")
    patients, kidneys, images = load_reference()
    labels = kidneys.kidney_APN.to_numpy(int)
    states = patients.DMSA_APN.map(F.STATE_CODE).to_numpy(int)
    features = pd.read_csv(OUT / "clinical_image_features.csv").set_index("image_id").reindex(images.image_id)
    demo = pd.read_csv(OUT / "deidentified_demographics.csv")
    geometry_columns = [c for c in features if c.startswith("geometry_")]
    clinical_columns = [c for c in features if c.startswith(("geometry_", "echo_", "focal_"))]
    raw_clinical = features[clinical_columns].to_numpy(float)
    raw_geometry = features[geometry_columns].to_numpy(float)
    context = covariates(images, features, demo)
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
            assert patients.patient_alias.iloc[train_patients].nunique() == 126
            assert kidneys.patient_alias.iloc[development].nunique() == 42

            residual_geometry, normalizer = residualize(raw_geometry, context, train_images, seed)
            clinical_images = np.c_[residual_geometry, raw_clinical[:, len(geometry_columns):]]
            clinical_pairs = F.patient_pair_features(clinical_images, images, patients)
            clinical_oriented = F.orient_kidneys(clinical_pairs, patients, kidneys)
            rad_pairs = F.patient_pair_features(rad_image, images, patients)
            rad_oriented = F.orient_kidneys(rad_pairs, patients, kidneys)
            early = np.c_[clinical_oriented, rad_oriented]

            score64, transform64, model64 = fit_anova(
                clinical_oriented, train, development, labels, seed, 64)
            score128, transform128, model128 = fit_anova(
                clinical_oriented, train, development, labels, seed, 128)
            early_score, early_transform, early_model = fit_anova(
                early, train, development, labels, seed, 128)
            state_probability, state_model = F.fit_state_model(
                clinical_pairs, train_patients, states, seed, 32)
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
                GROUPS[0]: score64,
                GROUPS[1]: score128,
                GROUPS[2]: early_score,
                GROUPS[3]: state_score,
                GROUPS[4]: (score64 + reference_score) / 2,
                GROUPS[5]: (early_score + reference_score) / 2,
                GROUPS[6]: (state_score + reference_score) / 2,
            }
            joblib.dump({
                "normalizer": normalizer,
                "clinical64": (transform64, model64),
                "clinical128": (transform128, model128),
                "early": (early_transform, early_model),
                "state": state_model,
            }, OUT / "models" / f"{seed}_clinical_features.joblib", compress=3)
            audits.append({
                "seed": seed,
                "normalizer_training_patients": int(images.patient_alias.iloc[train_images].nunique()),
                "normalizer_training_images": len(train_images),
                "labels_used_in_normalizer": [],
                "exploratory_test_scored": False,
            })
            for group, score in scores.items():
                result = F.metric(labels[development], score)
                metrics.append({"seed": seed, "role": "development", "group": group,
                                "patients": 42, "kidneys": len(development), **result})
                for local, index in enumerate(development):
                    predictions.append({
                        "seed": seed, "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[index],
                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                        "side": kidneys.side.iloc[index],
                        "label": int(labels[index]), "score": float(score[local]),
                    })
            print(f"Completed clinical-feature draw {seed}", flush=True)

    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc_mean > .8) & (summary.ap_mean > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {
        "groups_above_both_development_targets": achieved,
        "exploratory_test_scored": False,
    })
    write_json(OUT / "normalizer_audit.json", {"draws": audits})
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "extract", "run"))
    args = parser.parse_args()
    {"prepare": prepare, "extract": extract, "run": run}[args.stage]()
