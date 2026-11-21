from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.special import gammainc, gammaln
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "CYCH_random_validation_20260913"
OUT = ROOT / "CYCH_literature_speckle_development_20260914"
MANIFEST_PATH = SOURCE / "image_manifest.csv"
ASSIGNMENTS_PATH = (
    ROOT / "CYCH_directional_robustness_selection_20260914" / "random_patient_assignments_deidentified.csv"
)
PAIR_PREDICTIONS_PATH = ROOT / "CYCH_dinov3_native_probe_20260913" / "patient_predictions_by_role.csv"
I3_PREDICTIONS_PATH = ROOT / "CYCH_directional_robustness_selection_20260914" / "development_predictions.csv"
PROTOCOL_PATH = OUT / "protocol.json"
REFERENCES_PATH = OUT / "references.json"
LOCK_PATH = OUT / "protocol_lock.json"
SCRIPT_COPY = OUT / "source" / Path(__file__).name

SEEDS = list(range(20260913, 20260923))
K_VALUES = [5, 10, 20, 30]
C_VALUES = [0.1, 1.0, 10.0]


REFERENCES = [
    {
        "method": "Regional kidney ultrasound speckle distributions",
        "citation": "Singla et al., Ultrasound in Medicine & Biology, 2022",
        "url": "https://arxiv.org/abs/2206.06654",
        "use_here": (
            "In 821 transplant-kidney B-mode images, Rayleigh and Nakagami parameters "
            "differed among cortex, medulla, and central echogenic complex. The current "
            "experiment adapts these distributions to geometry-defined zones and does not "
            "claim anatomical segmentation or transplant-disease equivalence."
        ),
    },
    {
        "finding": "B-mode echogenic changes in acute pyelonephritis",
        "citation": "Sim, Ultrasonography, 2018",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6177687/",
        "use_here": (
            "Supports testing quantitative intensity-distribution changes related to "
            "heterogeneous cortical echogenicity and focal abnormalities."
        ),
    },
    {
        "finding": "Focal echogenic lesions and cortical bulging in complicated APN",
        "citation": "An update on pyelonephritis: role of contrast-enhanced ultrasound",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10247636/",
        "use_here": "Supports regional rather than whole-image averaging of B-mode appearance.",
    },
]


PROTOCOL = {
    "title": "Development-only literature-guided B-mode speckle distribution experiment",
    "status": "locked before feature extraction and model fitting",
    "cohort": {
        "patients": 98,
        "images": 349,
        "unit": "patient",
        "fixed_seeds": SEEDS,
        "per_seed": {"train": 58, "development_validation": 20, "unused_test": 20},
    },
    "hard_constraints": [
        "Use the existing kidney segmentation mask for every image.",
        "Extract all speckle descriptors without outcome labels.",
        "Call geometry-derived zones boundary, middle, core, poles, and midlong, not anatomical compartments.",
        "Fit all preprocessing, feature selection, and classifiers on outer-training patients only.",
        "Do not load or evaluate any test-role prediction.",
    ],
    "preprocessing": (
        "Within-mask 1st/99th percentile clipping and [0,1] scaling. Values at or below "
        "1/255 are excluded from continuous positive-distribution fitting."
    ),
    "features": {
        "zones": "whole mask plus boundary-depth, middle-depth, core-depth, poles, and midlong zones",
        "Rayleigh": "maximum-likelihood scale sigma=sqrt(mean(x^2)/2)",
        "Nakagami": (
            "inverse normalized variance estimate: Omega=mean(x^2), "
            "m=Omega^2/variance(x^2), restricted to [0.5,50] for numerical stability"
        ),
        "fit_quality": "Kolmogorov-Smirnov distance and empirical-to-model histogram KL divergence",
        "local_tiles": "5x5 normalized bounding-box tiles with at least 64 kidney pixels",
        "boundary": (
            "These are descriptors of log-compressed B-mode pixel values, not raw RF envelope "
            "statistics and not an exact replication of Singla et al."
        ),
    },
    "patient_aggregation": "mean, SD, minimum, maximum, and LL/LT/RL/RT means per image feature",
    "model": {
        "pipeline": "median imputation, scaling, ANOVA F selection, balanced RBF SVC",
        "inner_selection": "four-fold stratified CV within each 58-patient outer-training set",
        "inner_objective": "mean of AUROC and average precision",
        "feature_counts": K_VALUES,
        "C": C_VALUES,
    },
    "registered_candidates": [
        "SPECKLE",
        "PAIR75+25%SPECKLE",
        "PAIR50+50%SPECKLE",
        "PAIR60+20%I3+20%SPECKLE",
        "PAIR50+25%I3+25%SPECKLE",
    ],
    "candidate_gate_vs_fixed_pair": {
        "mean_auroc_gain_at_least": 0.03,
        "mean_average_precision_gain_at_least": 0.02,
        "seeds_with_both_metrics_improved_at_least": 6,
        "action": "A separate pipeline lock is required before any test evaluation.",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def lock_protocol() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SCRIPT_COPY.parent.mkdir(parents=True, exist_ok=True)
    write_json(PROTOCOL_PATH, PROTOCOL)
    write_json(REFERENCES_PATH, REFERENCES)
    shutil.copy2(Path(__file__), SCRIPT_COPY)
    write_json(
        LOCK_PATH,
        {
            "protocol": sha256(PROTOCOL_PATH),
            "references": sha256(REFERENCES_PATH),
            "script": sha256(Path(__file__)),
            "manifest": sha256(MANIFEST_PATH),
            "assignments": sha256(ASSIGNMENTS_PATH),
            "pair_predictions": sha256(PAIR_PREDICTIONS_PATH),
            "i3_predictions": sha256(I3_PREDICTIONS_PATH),
        },
    )
    print(f"Locked protocol: {LOCK_PATH}", flush=True)


def verify_lock() -> None:
    expected = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    paths = {
        "protocol": PROTOCOL_PATH,
        "references": REFERENCES_PATH,
        "script": Path(__file__),
        "manifest": MANIFEST_PATH,
        "assignments": ASSIGNMENTS_PATH,
        "pair_predictions": PAIR_PREDICTIONS_PATH,
        "i3_predictions": I3_PREDICTIONS_PATH,
    }
    changed = [name for name, path in paths.items() if sha256(path) != expected[name]]
    if changed:
        raise RuntimeError(f"Locked inputs changed: {changed}")


def crop_to_mask(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("Empty kidney mask")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def robust_channel(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = image[mask].astype(np.float32)
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        high = low + 1.0
    channel = np.clip(image.astype(np.float32), low, high)
    channel = ((channel - low) / (high - low)).astype(np.float32)
    channel[~mask] = 0.0
    return channel


def geometry_zones(mask: np.ndarray) -> dict[str, np.ndarray]:
    ys, xs = np.where(mask)
    points = np.column_stack([xs, ys]).astype(float)
    center = points.mean(axis=0)
    centered = points - center
    values, vectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    major = vectors[:, int(np.argmax(values))]
    u = centered @ major
    u /= max(float(np.max(np.abs(u))), 1.0)
    u_map = np.zeros(mask.shape, dtype=np.float32)
    u_map[ys, xs] = u
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    distance /= max(float(distance[mask].max()), 1e-8)
    return {
        "all": mask,
        "boundary": mask & (distance <= 0.25),
        "middle": mask & (distance > 0.25) & (distance <= 0.55),
        "core": mask & (distance > 0.55),
        "poles": mask & (np.abs(u_map) > 0.55),
        "midlong": mask & (np.abs(u_map) <= 0.55),
    }


def model_probabilities(edges: np.ndarray, sigma: float, m: float, omega: float) -> tuple[np.ndarray, np.ndarray]:
    rayleigh_cdf = 1.0 - np.exp(-(edges**2) / (2.0 * sigma**2 + 1e-12))
    nakagami_cdf = gammainc(m, m * edges**2 / (omega + 1e-12))
    rayleigh = np.maximum(np.diff(rayleigh_cdf), 1e-12)
    nakagami = np.maximum(np.diff(nakagami_cdf), 1e-12)
    rayleigh /= rayleigh.sum()
    nakagami /= nakagami.sum()
    return rayleigh, nakagami


def speckle_fit(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 1.0 / 255.0)]
    if values.size < 32:
        return {
            "rayleigh_sigma": np.nan,
            "nakagami_m": np.nan,
            "nakagami_omega": np.nan,
            "rayleigh_ks": np.nan,
            "nakagami_ks": np.nan,
            "rayleigh_kl": np.nan,
            "nakagami_kl": np.nan,
            "mean_log_likelihood_delta_nakagami_minus_rayleigh": np.nan,
        }
    squared = values**2
    omega = max(float(squared.mean()), 1e-8)
    m = float(np.clip(omega**2 / max(float(squared.var()), 1e-12), 0.5, 50.0))
    sigma = float(np.sqrt(omega / 2.0))
    ordered = np.sort(values)
    empirical_cdf = np.arange(1, len(ordered) + 1, dtype=float) / len(ordered)
    rayleigh_cdf = 1.0 - np.exp(-(ordered**2) / (2.0 * sigma**2 + 1e-12))
    nakagami_cdf = gammainc(m, m * ordered**2 / (omega + 1e-12))
    rayleigh_ks = float(np.max(np.abs(empirical_cdf - rayleigh_cdf)))
    nakagami_ks = float(np.max(np.abs(empirical_cdf - nakagami_cdf)))
    edges = np.linspace(0.0, 1.0, 33)
    empirical = np.histogram(values, bins=edges)[0].astype(float)
    empirical = np.maximum(empirical / empirical.sum(), 1e-12)
    empirical /= empirical.sum()
    rayleigh_prob, nakagami_prob = model_probabilities(edges, sigma, m, omega)
    rayleigh_kl = float(np.sum(empirical * np.log(empirical / rayleigh_prob)))
    nakagami_kl = float(np.sum(empirical * np.log(empirical / nakagami_prob)))
    rayleigh_logpdf = np.log(values + 1e-12) - 2.0 * np.log(sigma) - values**2 / (2.0 * sigma**2)
    nakagami_logpdf = (
        np.log(2.0)
        + m * np.log(m)
        - gammaln(m)
        - m * np.log(omega)
        + (2.0 * m - 1.0) * np.log(values + 1e-12)
        - m * values**2 / omega
    )
    return {
        "rayleigh_sigma": sigma,
        "nakagami_m": m,
        "nakagami_omega": omega,
        "rayleigh_ks": rayleigh_ks,
        "nakagami_ks": nakagami_ks,
        "rayleigh_kl": rayleigh_kl,
        "nakagami_kl": nakagami_kl,
        "mean_log_likelihood_delta_nakagami_minus_rayleigh": float(
            np.mean(nakagami_logpdf - rayleigh_logpdf)
        ),
    }


def local_tile_features(channel: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    records: list[dict[str, float]] = []
    y_edges = np.linspace(0, channel.shape[0], 6).round().astype(int)
    x_edges = np.linspace(0, channel.shape[1], 6).round().astype(int)
    for yi in range(5):
        for xi in range(5):
            region = mask[y_edges[yi] : y_edges[yi + 1], x_edges[xi] : x_edges[xi + 1]]
            if int(region.sum()) < 64:
                continue
            tile = channel[y_edges[yi] : y_edges[yi + 1], x_edges[xi] : x_edges[xi + 1]]
            records.append(speckle_fit(tile[region]))
    output: dict[str, float] = {"local_valid_tiles": float(len(records))}
    for name in ["rayleigh_sigma", "nakagami_m", "nakagami_omega", "rayleigh_ks", "nakagami_ks"]:
        values = np.array([record[name] for record in records], dtype=float)
        values = values[np.isfinite(values)]
        for statistic in ("mean", "std", "min", "max"):
            output[f"local_{name}_{statistic}"] = float(getattr(values, statistic)()) if values.size else np.nan
    return output


def extract_image_features() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH, usecols=["image_id", "patient_alias", "view"], dtype=str)
    rows: list[dict[str, object]] = []
    for position, row in enumerate(manifest.itertuples(index=False), start=1):
        image = cv2.imread(str(SOURCE / "native" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE)
        mask_image = cv2.imread(str(SOURCE / "masks" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask_image is None:
            raise FileNotFoundError(row.image_id)
        image, mask = crop_to_mask(image, mask_image > 0)
        channel = robust_channel(image, mask)
        features: dict[str, object] = {
            "image_id": row.image_id,
            "patient_alias": row.patient_alias,
            "view": row.view,
        }
        zone_values: dict[str, dict[str, float]] = {}
        for zone_name, zone in geometry_zones(mask).items():
            fit = speckle_fit(channel[zone])
            zone_values[zone_name] = fit
            features.update({f"speckle_{zone_name}_{name}": value for name, value in fit.items()})
        for inner, outer in [("core", "boundary"), ("middle", "boundary"), ("poles", "midlong")]:
            for name in ["rayleigh_sigma", "nakagami_m", "nakagami_omega", "nakagami_ks", "nakagami_kl"]:
                first = zone_values[inner][name]
                second = zone_values[outer][name]
                features[f"speckle_{inner}_minus_{outer}_{name}"] = float(first - second)
                features[f"speckle_{inner}_vs_{outer}_{name}"] = float(
                    (first - second) / (abs(first) + abs(second) + 1e-8)
                )
        features.update({f"speckle_{name}": value for name, value in local_tile_features(channel, mask).items()})
        rows.append(features)
        if position % 50 == 0 or position == len(manifest):
            print(f"Extracted speckle features {position}/{len(manifest)}", flush=True)
    result = pd.DataFrame(rows)
    if result.shape[0] != 349 or result.patient_alias.nunique() != 98:
        raise RuntimeError(f"Unexpected image feature table: {result.shape}")
    return result


def aggregate_patients(images: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [column for column in images if column.startswith("speckle_")]
    rows: list[dict[str, object]] = []
    for patient_alias, group in images.groupby("patient_alias", sort=True):
        row: dict[str, object] = {"patient_alias": patient_alias}
        values = group[feature_columns]
        for statistic, series in [
            ("all_mean", values.mean()),
            ("all_std", values.std(ddof=0)),
            ("all_min", values.min()),
            ("all_max", values.max()),
        ]:
            row.update({f"{statistic}__{name}": float(value) for name, value in series.items()})
        for view in ("LL", "LT", "RL", "RT"):
            subset = group.loc[group.view.eq(view), feature_columns]
            row[f"view_present__{view}"] = int(not subset.empty)
            series = subset.mean() if not subset.empty else pd.Series(np.nan, index=feature_columns)
            row.update({f"{view}_mean__{name}": float(value) for name, value in series.items()})
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.shape[0] != 98:
        raise RuntimeError(f"Expected 98 patients, found {result.shape[0]}")
    return result


def load_reference(path: Path, group: str, score_name: str) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["role"] == "development_validation" and row["group"] == group:
                rows.append(
                    {
                        "seed": int(row["seed"]),
                        "patient_alias": row["patient_alias"],
                        "label": int(row["label"]),
                        score_name: float(row["score"]),
                    }
                )
    result = pd.DataFrame(rows)
    if result.shape[0] != 200:
        raise RuntimeError(f"Unexpected reference {group}: {result.shape}")
    return result


def build_model(k: int, c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            (
                "svc",
                SVC(
                    C=c_value,
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=seed,
                ),
            ),
        ]
    )


def select_model(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[int, float, float]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    records = []
    for k in K_VALUES:
        for c_value in C_VALUES:
            scores = []
            for train, validation in splitter.split(x, y):
                model = build_model(k, c_value, seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x[train], y[train])
                prediction = model.predict_proba(x[validation])[:, 1]
                scores.append(
                    (roc_auc_score(y[validation], prediction) + average_precision_score(y[validation], prediction)) / 2
                )
            records.append((float(np.mean(scores)), -k, -c_value, k, c_value))
    best = max(records)
    return int(best[3]), float(best[4]), float(best[0])


def metric(seed: int, group: str, y: np.ndarray, score: np.ndarray) -> dict[str, object]:
    return {
        "seed": seed,
        "role": "development_validation",
        "group": group,
        "n": len(y),
        "auroc": roc_auc_score(y, score),
        "average_precision": average_precision_score(y, score),
    }


def run_development(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments = pd.read_csv(ASSIGNMENTS_PATH, usecols=["seed", "role", "patient_alias", "label"], dtype={"patient_alias": str})
    assignments = assignments.loc[assignments.role.isin(["train", "validation"])]
    pair = load_reference(PAIR_PREDICTIONS_PATH, "PAIR", "pair_score")
    i3 = load_reference(I3_PREDICTIONS_PATH, "I3-ALL", "i3_score")
    feature_columns = [column for column in features if column != "patient_alias"]
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        split = assignments.loc[assignments.seed.eq(seed)]
        train = split.loc[split.role.eq("train"), ["patient_alias", "label"]]
        development = split.loc[split.role.eq("validation"), ["patient_alias", "label"]]
        train_frame = train.merge(features, on="patient_alias", validate="one_to_one")
        development_frame = development.merge(features, on="patient_alias", validate="one_to_one")
        reference = development.merge(
            pair.loc[pair.seed.eq(seed), ["patient_alias", "label", "pair_score"]],
            on=["patient_alias", "label"],
            validate="one_to_one",
        ).merge(
            i3.loc[i3.seed.eq(seed), ["patient_alias", "label", "i3_score"]],
            on=["patient_alias", "label"],
            validate="one_to_one",
        )
        if list(reference.patient_alias) != list(development_frame.patient_alias):
            raise RuntimeError(f"Patient order mismatch for {seed}")
        y_train = train_frame.label.to_numpy(int)
        y_development = development_frame.label.to_numpy(int)
        x_train = train_frame[feature_columns].to_numpy(float)
        x_development = development_frame[feature_columns].to_numpy(float)
        pair_score = reference.pair_score.to_numpy(float)
        i3_score = reference.i3_score.to_numpy(float)
        k, c_value, inner_objective = select_model(x_train, y_train, seed)
        model = build_model(k, c_value, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y_train)
        speckle_score = model.predict_proba(x_development)[:, 1]
        groups = {
            "PAIR": pair_score,
            "PAIR+I3-REFERENCE": (2 * pair_score + i3_score) / 3,
            "SPECKLE": speckle_score,
            "PAIR75+25%SPECKLE": 0.75 * pair_score + 0.25 * speckle_score,
            "PAIR50+50%SPECKLE": 0.50 * pair_score + 0.50 * speckle_score,
            "PAIR60+20%I3+20%SPECKLE": 0.60 * pair_score + 0.20 * i3_score + 0.20 * speckle_score,
            "PAIR50+25%I3+25%SPECKLE": 0.50 * pair_score + 0.25 * i3_score + 0.25 * speckle_score,
        }
        for group, score in groups.items():
            metric_rows.append(metric(seed, group, y_development, score))
            if group not in {"PAIR", "PAIR+I3-REFERENCE"}:
                for alias, label, value in zip(development_frame.patient_alias, y_development, score):
                    prediction_rows.append(
                        {
                            "seed": seed,
                            "role": "development_validation",
                            "group": group,
                            "patient_alias": alias,
                            "label": int(label),
                            "score": float(value),
                        }
                    )
        tuning_rows.append(
            {
                "seed": seed,
                "candidate_variables": len(feature_columns),
                "selected_k": k,
                "selected_C": c_value,
                "inner_cv_objective": inner_objective,
                "test_accessed": False,
            }
        )
        print(f"Completed speckle development seed {seed}", flush=True)
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), pd.DataFrame(tuning_rows)


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        metrics.groupby("group", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_auroc=("auroc", "mean"),
            sd_auroc=("auroc", "std"),
            min_auroc=("auroc", "min"),
            max_auroc=("auroc", "max"),
            mean_average_precision=("average_precision", "mean"),
            sd_average_precision=("average_precision", "std"),
            min_average_precision=("average_precision", "min"),
            max_average_precision=("average_precision", "max"),
        )
        .sort_values(["mean_auroc", "mean_average_precision"], ascending=False)
    )
    baseline = metrics.loc[metrics.group.eq("PAIR"), ["seed", "auroc", "average_precision"]].rename(
        columns={"auroc": "pair_auroc", "average_precision": "pair_average_precision"}
    )
    candidates = [
        "PAIR75+25%SPECKLE",
        "PAIR50+50%SPECKLE",
        "PAIR60+20%I3+20%SPECKLE",
        "PAIR50+25%I3+25%SPECKLE",
    ]
    rows = []
    for group in candidates:
        current = metrics.loc[metrics.group.eq(group)].merge(baseline, on="seed", validate="one_to_one")
        delta_auc = current.auroc - current.pair_auroc
        delta_ap = current.average_precision - current.pair_average_precision
        joint = int(((delta_auc > 0) & (delta_ap > 0)).sum())
        gain_auc = float(delta_auc.mean())
        gain_ap = float(delta_ap.mean())
        rows.append(
            {
                "group": group,
                "mean_auroc_gain": gain_auc,
                "mean_average_precision_gain": gain_ap,
                "seeds_both_improved": joint,
                "passes_registered_gate": bool(gain_auc >= 0.03 and gain_ap >= 0.02 and joint >= 6),
                "exploratory_test_authorized": False,
            }
        )
    return summary, pd.DataFrame(rows).sort_values("mean_auroc_gain", ascending=False)


def write_readme(summary: pd.DataFrame, gate: pd.DataFrame, features: pd.DataFrame) -> None:
    baseline = summary.loc[summary.group.eq("PAIR")].iloc[0]
    i3 = summary.loc[summary.group.eq("PAIR+I3-REFERENCE")].iloc[0]
    standalone = summary.loc[summary.group.eq("SPECKLE")].iloc[0]
    candidate = summary.loc[
        summary.group.isin([
            "PAIR75+25%SPECKLE",
            "PAIR50+50%SPECKLE",
            "PAIR60+20%I3+20%SPECKLE",
            "PAIR50+25%I3+25%SPECKLE",
        ])
    ].iloc[0]
    passed = gate.loc[gate.passes_registered_gate]
    decision = (
        f"Passing candidate(s): {', '.join(passed.group)}. A separate lock is required before test access."
        if not passed.empty
        else "No registered candidate passed; the exploratory internal test remains unopened."
    )
    text = f"""# Literature-guided B-mode speckle development experiment

## Scope

Rayleigh and Nakagami descriptors were extracted without labels from all 349 existing
kidney-masked images. The source study used transplant-kidney anatomical regions; this
adaptation uses explicitly non-anatomical geometry zones and log-compressed B-mode pixels.

Only fixed development-validation rows were scored. Each model was selected within its
58 outer-training patients; no test row or exploratory-test prediction was loaded.

## Development results

- Fixed pair: AUROC {baseline.mean_auroc:.3f}, AP {baseline.mean_average_precision:.3f}.
- Existing pair plus I3 reference: AUROC {i3.mean_auroc:.3f}, AP {i3.mean_average_precision:.3f}.
- Speckle alone: AUROC {standalone.mean_auroc:.3f}, AP {standalone.mean_average_precision:.3f}.
- Best registered speckle ensemble: {candidate.group}, AUROC {candidate.mean_auroc:.3f}, AP {candidate.mean_average_precision:.3f}.

Registered gate versus pair: mean AUROC gain >=0.03, mean AP gain >=0.02, and
both improve in at least 6/10 draws. {decision}

## Boundary

The {features.shape[1] - 1:,} patient variables are exploratory quantitative image
descriptors, not validated tissue or lesion labels. Results are repeated development
estimates on 98 previously explored patients, not k-fold reporting or independent confirmation.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    if args.lock_only:
        lock_protocol()
        return
    verify_lock()
    image_path = OUT / "image_features_deidentified.csv"
    patient_path = OUT / "patient_features_deidentified.csv"
    if args.reuse_features and image_path.exists() and patient_path.exists():
        images = pd.read_csv(image_path)
        patients = pd.read_csv(patient_path)
    else:
        images = extract_image_features()
        patients = aggregate_patients(images)
        images.to_csv(image_path, index=False)
        patients.to_csv(patient_path, index=False)
    metrics, predictions, tuning = run_development(patients)
    summary, gate = summarize(metrics)
    outputs = {
        "metrics_per_seed.csv": metrics,
        "development_predictions.csv": predictions,
        "selected_hyperparameters.csv": tuning,
        "summary_development.csv": summary,
        "candidate_gate.csv": gate,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False)
    write_json(
        OUT / "run_audit.json",
        {
            "images": int(len(images)),
            "patients": int(patients.patient_alias.nunique()),
            "roles_evaluated": ["train", "development_validation"],
            "test_role_evaluated": False,
            "exploratory_test_predictions_loaded": False,
            "passing_candidates": gate.loc[gate.passes_registered_gate, "group"].tolist(),
            "output_hashes": {
                image_path.name: sha256(image_path),
                patient_path.name: sha256(patient_path),
                **{name: sha256(OUT / name) for name in outputs},
            },
        },
    )
    write_readme(summary, gate, patients)
    print(summary.to_string(index=False))
    print("\nCandidate gate:")
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
