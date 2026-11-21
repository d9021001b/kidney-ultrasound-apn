"""OCR physical kidney measurements from archived NCKUH TIFF frames; development only."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import zipfile

import cv2
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullframe_clean_probe_20261102 as F
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, metric, write_json

ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_tiff_measurement_probe_20261102"
ARCHIVES = tuple(ROOT / "03_data_models/ki-data" / name for name in ("1.zip", "2.zip", "3.zip"))
EASYOCR_WEIGHTS = tuple(Path.home().joinpath(".EasyOCR/model", name) for name in ("craft_mlt_25k.pth", "english_g2.pth"))
GROUPS = (
    "current_radiomics_anova64_lr",
    "current_dense_pca32_lr",
    "physical_measurement_lr",
    "radiomics_measurement_anova64_lr",
    "radiomics_measurement_anova128_lr",
    "dense_plus_radiomics",
    "measurement_plus_dense_radiomics",
)
REFERENCE = GROUPS[0]


def tiff_index() -> dict[tuple[str, str, str], list[tuple[Path, str]]]:
    index = defaultdict(list)
    for archive in ARCHIVES:
        with zipfile.ZipFile(archive) as handle:
            for name in handle.namelist():
                if Path(name).suffix.lower() not in (".tif", ".tiff"):
                    continue
                key = F.archive_key(name)
                if key is not None:
                    index[key].append((archive, name))
    return dict(index)


def matched_tiff_reference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients, kidneys, images = F.load_reference()
    images = images[images.kidney_APN.notna()].copy()
    images["date_key"] = images.exam_date_raw.astype(str).str.replace(r"\D", "", regex=True).str[:8]
    index = tiff_index()
    images = images[[len(index.get((row.patient_id, row.date_key, row.view), [])) == 1
                     for row in images.itertuples(index=False)]].reset_index(drop=True)
    patient_aliases, kidney_aliases = set(images.patient_alias), set(images.kidney_alias)
    patients = patients[patients.patient_alias.isin(patient_aliases)].reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_alias.isin(kidney_aliases)].reset_index(drop=True)
    assert (len(patients), len(kidneys), len(images)) == (81, 159, 310)
    return patients, kidneys, images


def parse_measurement_mm(parts: list[tuple[str, float]]) -> tuple[float, float]:
    joined = " ".join(text for text, _ in parts)
    matches = re.findall(r"(?<!\d)(\d{1,3}(?:[\.,]\d+)?)\s*m{1,2}\b", joined, flags=re.I)
    valid = [float(value.replace(",", ".")) for value in matches]
    valid = [value for value in valid if 10 <= value <= 200]
    if len(valid) != 1:
        return np.nan, 0.0
    confidence = min((confidence for text, confidence in parts if re.search(r"\d", text)), default=0.0)
    return valid[0], float(confidence)


def prepare() -> None:
    from sklearn.model_selection import train_test_split
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered TIFF measurement protocol")
    patients, kidneys, images = matched_tiff_reference()
    for path in (*ARCHIVES, *EASYOCR_WEIGHTS):
        if not path.is_file():
            raise FileNotFoundError(path)
    assignments = []
    for seed in F.SEEDS:
        train, remainder = train_test_split(np.arange(len(patients)), train_size=51,
                                            random_state=seed, stratify=patients.DMSA_APN)
        development, test = train_test_split(remainder, train_size=15, random_state=seed + 1000,
                                              stratify=patients.DMSA_APN.to_numpy()[remainder])
        for role, indices in (("train", train), ("development", development), ("exploratory_test", test)):
            for index in indices:
                assignments.append({"seed": seed, "patient_alias": patients.patient_alias.iloc[index],
                                    "role": role, "DMSA_APN": patients.DMSA_APN.iloc[index]})
    pd.DataFrame(assignments).to_csv(OUT / "patient_assignments.csv", index=False)
    images[["image_id", "patient_alias", "kidney_alias", "side", "view", "kidney_APN", "DMSA_APN"]].to_csv(
        OUT / "deidentified_image_manifest.csv", index=False)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "TIFF-measurement availability cohort:81 patients,159 imaged kidneys,310 uniquely matched views",
        "split": "Three fixed patient-stratified draws,51 train/15 development/15 exploratory test; no k-fold",
        "measurement_source": "Dist A millimeter text rendered in the bottom measurement row of archived TIFF frames1.zip/2.zip/3.zip",
        "matching": "Internal patient/date/view key only; all suffixes after LL/LT/RL/RT ignored; duplicate TIFF keys treated as unavailable",
        "ocr": "EasyOCR local English detector/recognizer on the bottom20% and left half of each unique TIFF; accept exactly one10-200 mm value",
        "features": "Per-image millimeters, log1p millimeters, OCR confidence and availability; preserve view/side/contralateral context",
        "models": "Current masked-radiomics and frozen RadImageNet DenseNet controls; train-only imputation/scaling/ANOVA or PCA and balanced LR; fixed equal score fusion",
        "prohibited": "No diagnosis suffix, DMSA field, MRN, date, source row, test-derived OCR rule/fusion weight/threshold, or exploratory-test prediction",
        "target": "Development AUROC>0.8 and AP>0.8; no full-goal claim without later locked confirmation",
        "groups": list(GROUPS), "seeds": list(F.SEEDS),
        "source_hashes": {str(p): digest(p) for p in (*ARCHIVES, *EASYOCR_WEIGHTS)},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "source_sha256": digest(__file__), "protocol_sha256": digest(OUT / "protocol.json"),
        "assignments_sha256": digest(OUT / "patient_assignments.csv"),
        "manifest_sha256": digest(OUT / "deidentified_image_manifest.csv"),
        "radiomics_sha256": digest(F.FEATURES / "radiomics_percentile_image.csv"),
        "dense_sha256": digest(F.FEATURES / "dense_tight_image.npy"),
        "archive_hashes": {str(p): digest(p) for p in ARCHIVES},
        "ocr_weight_hashes": {str(p): digest(p) for p in EASYOCR_WEIGHTS},
    })
    print("Registered TIFF physical-measurement probe", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    assert digest(OUT / "patient_assignments.csv") == lock["assignments_sha256"]
    assert digest(OUT / "deidentified_image_manifest.csv") == lock["manifest_sha256"]
    assert digest(F.FEATURES / "radiomics_percentile_image.csv") == lock["radiomics_sha256"]
    assert digest(F.FEATURES / "dense_tight_image.npy") == lock["dense_sha256"]
    for path in ARCHIVES:
        assert digest(path) == lock["archive_hashes"][str(path)]
    for path in EASYOCR_WEIGHTS:
        assert digest(path) == lock["ocr_weight_hashes"][str(path)]


def ocr() -> None:
    require_lock()
    if (OUT / "ocr_audit.json").exists():
        raise FileExistsError("Preserve completed TIFF OCR")
    import easyocr
    _, _, images = matched_tiff_reference()
    index = tiff_index()
    readers = {archive: zipfile.ZipFile(archive) for archive in ARCHIVES}
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    records = []
    try:
        for number, row in enumerate(images.itertuples(index=False), 1):
            key = (row.patient_id, row.date_key, row.view)
            candidates = index.get(key, [])
            value, confidence, status = np.nan, 0.0, "no_unique_tiff"
            if len(candidates) == 1:
                archive, member = candidates[0]
                image = cv2.imdecode(np.frombuffer(readers[archive].read(member), np.uint8), cv2.IMREAD_COLOR)
                bottom = image[int(image.shape[0] * .80):, :image.shape[1] // 2]
                bottom = cv2.resize(bottom, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                result = reader.readtext(bottom, detail=1, paragraph=False)
                parts = [(str(item[1]), float(item[2])) for item in result]
                value, confidence = parse_measurement_mm(parts)
                status = "parsed" if np.isfinite(value) else "unique_tiff_unparsed"
            records.append({"image_id": row.image_id, "measurement_mm": value,
                            "ocr_confidence": confidence, "measurement_status": status})
            if number % 50 == 0:
                print(f"OCR {number}/{len(images)}", flush=True)
    finally:
        for handle in readers.values(): handle.close()
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "deidentified_measurements.csv", index=False)
    parsed = frame.measurement_mm.notna()
    write_json(OUT / "ocr_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "images": len(frame), "parsed_measurements": int(parsed.sum()),
        "unique_tiff_unparsed": int(frame.measurement_status.eq("unique_tiff_unparsed").sum()),
        "no_unique_tiff": int(frame.measurement_status.eq("no_unique_tiff").sum()),
        "measurement_mm_min_median_max": [float(frame.loc[parsed, "measurement_mm"].min()),
                                           float(frame.loc[parsed, "measurement_mm"].median()),
                                           float(frame.loc[parsed, "measurement_mm"].max())],
        "diagnosis_suffix_used": False, "labels_used_in_ocr": [],
        "output_sha256": digest(OUT / "deidentified_measurements.csv"),
    })
    print(f"Parsed {parsed.sum()} physical measurements from {len(frame)} study images", flush=True)


def measurement_features(frame: pd.DataFrame) -> np.ndarray:
    value = frame.measurement_mm.to_numpy(float)
    present = np.isfinite(value).astype(float)
    return np.c_[value, np.log1p(value), frame.ocr_confidence.to_numpy(float), present]


def fit_predict(values, train, dev, labels, seed):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                          StandardScaler(), LogisticRegression(C=.1, class_weight="balanced",
                                                               solver="liblinear", max_iter=3000,
                                                               random_state=seed))
    model.fit(values[train], labels[train])
    return model.predict_proba(values[dev])[:, 1], model


def fit_anova(values, train, dev, labels, seed, k):
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          VarianceThreshold(), SelectKBest(f_classif, k=k),
                          LogisticRegression(C=.01, class_weight="balanced", solver="liblinear",
                                             max_iter=3000, random_state=seed))
    model.fit(values[train], labels[train])
    return model.predict_proba(values[dev])[:, 1], model


def fit_pca(values, train, dev, labels, seed, components=32):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          PCA(components, whiten=True, random_state=seed),
                          LogisticRegression(C=.1, class_weight="balanced", solver="liblinear",
                                             max_iter=3000, random_state=seed))
    model.fit(values[train], labels[train])
    return model.predict_proba(values[dev])[:, 1], model


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed physical-measurement predictions")
    patients, kidneys, images = matched_tiff_reference()
    labels = kidneys.kidney_APN.to_numpy(int)
    measures = pd.read_csv(OUT / "deidentified_measurements.csv").set_index("image_id").reindex(images.image_id)
    measurement = measurement_features(measures)
    radiomics = pd.read_csv(F.FEATURES / "radiomics_percentile_image.csv").set_index("image_id").reindex(images.image_id)
    radiomics = radiomics[[c for c in radiomics if not c.endswith(("_Energy", "_TotalEnergy"))]].to_numpy(float)
    full_manifest = pd.read_csv(F.FEATURES / "image_manifest.csv")
    dense_all = np.load(F.FEATURES / "dense_tight_image.npy")
    dense_lookup = {ident: i for i, ident in enumerate(full_manifest.image_id)}
    dense = dense_all[[dense_lookup[ident] for ident in images.image_id]]
    oriented = {
        "measurement": F.orient_kidneys(F.patient_pair_features(measurement, images, patients), patients, kidneys),
        "radiomics": F.orient_kidneys(F.patient_pair_features(radiomics, images, patients), patients, kidneys),
        "dense": F.orient_kidneys(F.patient_pair_features(dense, images, patients), patients, kidneys),
    }
    oriented["radiomics_measurement"] = np.concatenate([oriented["radiomics"], oriented["measurement"]], axis=1)
    assignments = pd.read_csv(OUT / "patient_assignments.csv")
    predictions, metrics = [], []
    (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in F.SEEDS:
            roles = kidneys.patient_alias.map(assignments[assignments.seed.eq(seed)].set_index("patient_alias").role)
            train = np.flatnonzero(roles.eq("train")); dev = np.flatnonzero(roles.eq("development"))
            radiomics_score, radiomics_model = fit_anova(oriented["radiomics"], train, dev, labels, seed, 64)
            dense_score, dense_model = fit_pca(oriented["dense"], train, dev, labels, seed)
            measure_score, measure_model = fit_predict(oriented["measurement"], train, dev, labels, seed)
            combined64, combined64_model = fit_anova(oriented["radiomics_measurement"], train, dev, labels, seed, 64)
            combined128, combined128_model = fit_anova(oriented["radiomics_measurement"], train, dev, labels, seed, 128)
            for name, model in ((GROUPS[0], radiomics_model), (GROUPS[1], dense_model),
                                (GROUPS[2], measure_model), (GROUPS[3], combined64_model),
                                (GROUPS[4], combined128_model)):
                joblib.dump(model, OUT / "models" / f"{seed}_{name}.joblib", compress=3)
            scores = {GROUPS[0]: radiomics_score, GROUPS[1]: dense_score, GROUPS[2]: measure_score,
                      GROUPS[3]: combined64, GROUPS[4]: combined128,
                      GROUPS[5]: (dense_score + radiomics_score) / 2,
                      GROUPS[6]: (measure_score + dense_score + radiomics_score) / 3}
            for group in GROUPS:
                result = metric(labels[dev], scores[group])
                metrics.append({"seed": seed, "role": "development", "group": group,
                                "patients": 15, "kidneys": len(dev), **result})
                for local, index in enumerate(dev):
                    predictions.append({"seed": seed, "group": group,
                                        "patient_alias": kidneys.patient_alias.iloc[index],
                                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                                        "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                                        "side": kidneys.side.iloc[index], "label": int(labels[index]),
                                        "score": float(scores[group][local])})
            print(f"Completed physical-measurement draw {seed}", flush=True)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap"]].mean().sort_values(["auroc", "ap"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc > .8) & (summary.ap > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {
        "groups_above_both_development_targets": achieved, "test_scored": False,
        "goal_achieved": False, "reason": "Development-only subgroup cannot complete full objective",
    })
    print(summary.to_string(), flush=True)


def bootstrap() -> None:
    require_lock()
    frame = pd.read_csv(OUT / "development_predictions.csv")
    ci, paired, samples = cluster_intervals(frame, list(GROUPS), F.SEEDS, REFERENCE)
    ci.to_csv(OUT / "development_confidence_intervals.csv", index=False)
    paired.to_csv(OUT / "development_paired_intervals.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    write_json(OUT / "completion_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "patients": 81,
        "kidneys": 159, "images": 310, "test_scored": False,
        "scored_unit": "kidney", "bootstrap_cluster": "patient",
    })
    print("Physical-measurement intervals completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "ocr", "run", "bootstrap"])
    args = parser.parse_args()
    {"prepare": prepare, "ocr": ocr, "run": run, "bootstrap": bootstrap}[args.stage]()
