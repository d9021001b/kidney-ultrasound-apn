from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "CYCH_random_validation_20260913"
OUT = ROOT / "CYCH_literature_hos_lqp_development_20260914"
SCRIPT_COPY = OUT / "source" / Path(__file__).name
PROTOCOL_PATH = OUT / "protocol.json"
REFERENCES_PATH = OUT / "references.json"
LOCK_PATH = OUT / "protocol_lock.json"

MANIFEST_PATH = SOURCE / "image_manifest.csv"
ASSIGNMENTS_PATH = (
    ROOT
    / "CYCH_directional_robustness_selection_20260914"
    / "random_patient_assignments_deidentified.csv"
)
PAIR_PREDICTIONS_PATH = (
    ROOT / "CYCH_dinov3_native_probe_20260913" / "patient_predictions_by_role.csv"
)

SEEDS = list(range(20260913, 20260923))
FEATURE_SETS = {
    "HOS": "hos_",
    "LQP": "lqp_",
    "HOS+LQP": None,
}
K_VALUES = [5, 10, 20]
C_VALUES = [0.1, 1.0, 10.0]


REFERENCES = [
    {
        "method": "Higher-order features and elongated quinary patterns for CKD ultrasound",
        "citation": "Acharya et al., Neural Computing and Applications, 2020",
        "doi": "10.1007/s00521-019-04025-y",
        "url": "https://eprints.um.edu.my/36499/",
        "use_here": (
            "Motivates higher-order spectral/cumulant and quinary-pattern features. "
            "The present implementation is a literature-guided adaptation, not an exact "
            "replication, because the original source code and complete parameterization "
            "were not available."
        ),
    },
    {
        "method": "Local quinary pattern definition",
        "citation": "Open-access methodological description of LQP and GMAD thresholding",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9722294/",
        "use_here": (
            "Defines five states (-2, -1, 0, 1, 2), four binary pattern channels, "
            "and global median absolute deviation as a threshold scale."
        ),
    },
    {
        "method": "Higher-order spectrum definitions",
        "citation": "Cancer Informatics methodological application of higher-order spectra",
        "url": "https://journals.sagepub.com/doi/10.1177/1533034614547445",
        "use_here": (
            "Motivates third/fourth cumulants, bispectral magnitude, phase entropy, "
            "and normalized bispectral entropies."
        ),
    },
    {
        "method": "Directional renal ultrasound radiomics",
        "citation": "Academic Radiology CKD radiomics study",
        "doi": "10.1016/j.acra.2021.01.006",
        "url": "https://www.sciencedirect.com/science/article/abs/pii/S107663322100012X",
        "use_here": (
            "Supports directional texture analysis relative to renal image structure; "
            "the previously tested wavelet-LH GLRLM feature remains a separate baseline."
        ),
    },
    {
        "method": "Anatomical ultrasound signs of acute pyelonephritis",
        "citation": "Sim, Ultrasonography, 2018",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6177687/",
        "use_here": (
            "Supports sensitivity to heterogeneous echogenicity, corticomedullary-junction "
            "loss, enlargement, and focal parenchymal lesions."
        ),
    },
    {
        "method": "Ultrasound radiomics for lupus nephritis activity",
        "citation": "Qin et al., Journal of Inflammation Research, 2023",
        "doi": "10.2147/JIR.S398399",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9904229/",
        "use_here": (
            "Supports kidney-ROI PyRadiomics, reproducibility filtering, supervised feature "
            "selection, and comparison of conventional classifiers. Endpoint differs."
        ),
    },
    {
        "method": "Multi-view ultrasound multiple-instance learning",
        "citation": "SAMIL, arXiv:2306.00003",
        "url": "https://arxiv.org/abs/2306.00003",
        "use_here": (
            "Motivates preserving view identity during patient-level aggregation instead "
            "of collapsing all images to an unstructured mean."
        ),
    },
]


PROTOCOL = {
    "title": "Development-only literature-guided HOS and local quinary-pattern experiment",
    "cohort": {
        "patients": 98,
        "images": 349,
        "unit_of_split_and_evaluation": "patient",
        "fixed_seeds": SEEDS,
        "per_seed": {"train": 58, "development_validation": 20, "unused_test": 20},
    },
    "hard_constraints": [
        "Kidney segmentation masks are mandatory for every image.",
        "Feature extraction is label-free.",
        "Classifier fitting and hyperparameter selection use outer-training patients only.",
        "The fixed development sets are evaluated once per registered candidate.",
        "Rows assigned role=test and exploratory-test baseline predictions are not evaluated.",
        "No result from this script is an independent-test estimate.",
    ],
    "preprocessing": {
        "crop": "tight kidney-mask bounding box",
        "native_channel": "within-mask 1st/99th percentile clipping and [0,1] scaling",
        "enhanced_channel": (
            "CLAHE, clipLimit=2.0, 8x8 tiles, after filling outside-mask crop pixels "
            "with the within-mask median"
        ),
    },
    "features": {
        "HOS": (
            "Higher-order spectrum: third/fourth cumulants and averaged bispectral "
            "magnitude, phase entropy, normalized entropies, and bicoherence summaries "
            "from 0, 45, 90, and -45 degree renal line profiles."
        ),
        "LQP": (
            "Local quinary pattern: neighbor-center differences quantized into five "
            "states using T1=GMAD and T2=2*GMAD. Four nonzero states are encoded as "
            "rotation-invariant uniform binary histograms on circular, horizontally "
            "elongated, and vertically elongated eight-neighbor stencils."
        ),
        "terminology": (
            "This is a literature-guided adaptation. It is not labeled an exact "
            "replication of Acharya et al.'s elongated quinary pattern implementation."
        ),
    },
    "patient_aggregation": (
        "For every image feature: patient-level mean, standard deviation, minimum, and "
        "maximum, plus separate LL/LT/RL/RT means. Four view-presence indicators are added."
    ),
    "model": {
        "classifier": "RBF-kernel support vector machine with probability=True",
        "pipeline": "median imputation, standardization, univariate ANOVA F selection, SVC",
        "inner_selection": "four-fold stratified CV within each outer-training set",
        "inner_objective": "mean of AUROC and average precision",
        "feature_counts": K_VALUES,
        "C": C_VALUES,
        "gamma": "scale",
        "class_weight": "balanced",
    },
    "registered_candidates": [
        "HOS",
        "LQP",
        "HOS+LQP",
        "PAIR75+25%HOS",
        "PAIR75+25%LQP",
        "PAIR75+25%HOS+LQP",
        "PAIR50+50%HOS",
        "PAIR50+50%LQP",
        "PAIR50+50%HOS+LQP",
    ],
    "ensemble": (
        "Arithmetic mean of fixed-pair probability and candidate probability at "
        "predeclared candidate weights 0.25 and 0.50."
    ),
    "candidate_gate_vs_fixed_pair": {
        "mean_auroc_gain_at_least": 0.03,
        "mean_average_precision_gain_at_least": 0.02,
        "seeds_with_both_metrics_improved_at_least": 6,
        "action": (
            "Only a passing ensemble may be frozen for a single exploratory internal-test "
            "evaluation in a separate, later command."
        ),
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
    lock = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "references_sha256": sha256(REFERENCES_PATH),
        "script_sha256": sha256(Path(__file__)),
        "input_manifest_sha256": sha256(MANIFEST_PATH),
        "assignments_sha256": sha256(ASSIGNMENTS_PATH),
        "fixed_pair_predictions_sha256": sha256(PAIR_PREDICTIONS_PATH),
    }
    write_json(LOCK_PATH, lock)
    print(f"Locked protocol: {LOCK_PATH}")


def verify_lock() -> None:
    if not LOCK_PATH.exists():
        raise RuntimeError("Protocol is not locked. Run with --lock-only first.")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    checks = {
        "protocol_sha256": PROTOCOL_PATH,
        "references_sha256": REFERENCES_PATH,
        "script_sha256": Path(__file__),
        "input_manifest_sha256": MANIFEST_PATH,
        "assignments_sha256": ASSIGNMENTS_PATH,
        "fixed_pair_predictions_sha256": PAIR_PREDICTIONS_PATH,
    }
    mismatches = [key for key, path in checks.items() if sha256(path) != lock[key]]
    if mismatches:
        raise RuntimeError(f"Locked inputs changed: {mismatches}")


def robust_channel(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = image[mask]
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        high = low + 1.0
    out = np.clip(image.astype(np.float32), low, high)
    out = (out - low) / (high - low)
    out[~mask] = float(np.median(out[mask]))
    return out


def clahe_channel(native: np.ndarray, mask: np.ndarray) -> np.ndarray:
    work = np.round(np.clip(native, 0, 1) * 255).astype(np.uint8)
    work[~mask] = int(np.median(work[mask]))
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(work)
    out = enhanced.astype(np.float32) / 255.0
    out[~mask] = float(np.median(out[mask]))
    return out


def crop_to_mask(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Empty kidney mask")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size <= 1:
        return 0.0
    probabilities = values / values.sum()
    return float(-(probabilities * np.log(probabilities + 1e-12)).sum() / np.log(len(probabilities)))


def uniform_ri_bins(codes: np.ndarray) -> np.ndarray:
    bit_positions = np.arange(8, dtype=np.uint16)
    bits = ((codes[..., None] >> bit_positions) & 1).astype(np.uint8)
    transitions = np.sum(bits != np.roll(bits, 1, axis=-1), axis=-1)
    ones = bits.sum(axis=-1)
    return np.where(transitions <= 2, ones, 9).astype(np.uint8)


def lqp_features(channel: np.ndarray, mask: np.ndarray, channel_name: str) -> dict[str, float]:
    stencils = {
        "circular": [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)],
        "elongated_h": [(-1, -2), (-1, 0), (-1, 2), (0, 2), (1, 2), (1, 0), (1, -2), (0, -2)],
        "elongated_v": [(-2, -1), (0, -1), (2, -1), (2, 0), (2, 1), (0, 1), (-2, 1), (-2, 0)],
    }
    values = channel[mask]
    median = float(np.median(values))
    gmad = float(np.median(np.abs(values - median)))
    t1 = max(gmad, 1.0 / 255.0)
    t2 = 2.0 * t1
    features: dict[str, float] = {
        f"lqp_{channel_name}_gmad": gmad,
    }
    h, w = channel.shape
    for stencil_name, offsets in stencils.items():
        py = max(abs(dy) for dy, _ in offsets)
        px = max(abs(dx) for _, dx in offsets)
        if h <= 2 * py or w <= 2 * px:
            for state in (-2, -1, 0, 1, 2):
                features[f"lqp_{channel_name}_{stencil_name}_state_{state}"] = 0.0
            for state in (-2, -1, 1, 2):
                for bin_index in range(10):
                    features[f"lqp_{channel_name}_{stencil_name}_q{state}_u{bin_index}"] = 0.0
            continue
        center = channel[py : h - py, px : w - px]
        valid = mask[py : h - py, px : w - px].copy()
        neighbors = []
        for dy, dx in offsets:
            neighbor = channel[py + dy : h - py + dy, px + dx : w - px + dx]
            neighbor_mask = mask[py + dy : h - py + dy, px + dx : w - px + dx]
            neighbors.append(neighbor)
            valid &= neighbor_mask
        if not np.any(valid):
            states = np.empty((0, 8), dtype=np.int8)
        else:
            differences = np.stack([neighbor - center for neighbor in neighbors], axis=-1)[valid]
            states = np.zeros_like(differences, dtype=np.int8)
            states[differences >= t2] = 2
            states[(differences >= t1) & (differences < t2)] = 1
            states[(differences <= -t1) & (differences > -t2)] = -1
            states[differences <= -t2] = -2
        for state in (-2, -1, 0, 1, 2):
            features[f"lqp_{channel_name}_{stencil_name}_state_{state}"] = (
                float(np.mean(states == state)) if states.size else 0.0
            )
        for state in (-2, -1, 1, 2):
            if states.size:
                binary = (states == state).astype(np.uint16)
                codes = np.sum(binary * (1 << np.arange(8, dtype=np.uint16)), axis=1)
                bins = uniform_ri_bins(codes)
                histogram = np.bincount(bins, minlength=10).astype(np.float64)
                histogram /= max(histogram.sum(), 1.0)
            else:
                histogram = np.zeros(10, dtype=np.float64)
            for bin_index, value in enumerate(histogram):
                features[f"lqp_{channel_name}_{stencil_name}_q{state}_u{bin_index}"] = float(value)
    return features


def rotated_line_signals(
    channel: np.ndarray, mask: np.ndarray, angle: float, length: int = 64, max_lines: int = 16
) -> list[np.ndarray]:
    h, w = channel.shape
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    fill = float(np.median(channel[mask]))
    rotated = cv2.warpAffine(
        channel,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    rotated_mask = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    eligible = np.flatnonzero(rotated_mask.sum(axis=1) >= 16)
    if eligible.size == 0:
        return []
    chosen = np.unique(np.round(np.linspace(0, len(eligible) - 1, min(max_lines, len(eligible)))).astype(int))
    signals: list[np.ndarray] = []
    target = np.linspace(0.0, 1.0, length)
    for index in chosen:
        y = int(eligible[index])
        values = rotated[y, rotated_mask[y]].astype(np.float64)
        if values.size < 16:
            continue
        source = np.linspace(0.0, 1.0, values.size)
        signal = np.interp(target, source, values)
        signal -= signal.mean()
        scale = signal.std()
        if scale > 1e-8:
            signal /= scale
        signals.append(signal)
    return signals


def hos_summary(signals: list[np.ndarray]) -> dict[str, float]:
    names = [
        "cumulant3_mean",
        "cumulant3_std",
        "cumulant4_mean",
        "cumulant4_std",
        "log_bispectrum_mean",
        "log_bispectrum_std",
        "bispectrum_entropy_p1",
        "bispectrum_entropy_p2",
        "bispectrum_entropy_p3",
        "bispectrum_phase_entropy",
        "bicoherence_mean",
        "bicoherence_std",
        "bicoherence_p90",
        "bicoherence_max",
    ]
    if not signals:
        return {name: 0.0 for name in names}
    array = np.stack(signals)
    cumulant3 = np.mean(array**3, axis=1)
    variance = np.mean(array**2, axis=1)
    cumulant4 = np.mean(array**4, axis=1) - 3.0 * variance**2
    spectra = np.fft.rfft(array * np.hanning(array.shape[1])[None, :], axis=1)
    pairs = [(f1, f2) for f1 in range(1, 16) for f2 in range(1, 16) if f1 + f2 <= 16]
    bispectra = np.stack(
        [spectra[:, f1] * spectra[:, f2] * np.conj(spectra[:, f1 + f2]) for f1, f2 in pairs],
        axis=1,
    )
    mean_bispectrum = bispectra.mean(axis=0)
    numerator = np.abs(mean_bispectrum) ** 2
    denominator = []
    for f1, f2 in pairs:
        first = np.abs(spectra[:, f1] * spectra[:, f2]) ** 2
        second = np.abs(spectra[:, f1 + f2]) ** 2
        denominator.append(float(first.mean() * second.mean()))
    bicoherence = numerator / (np.asarray(denominator) + 1e-12)
    magnitude = np.abs(mean_bispectrum)
    phases = np.angle(mean_bispectrum[magnitude > 1e-12])
    phase_histogram = np.histogram(phases, bins=18, range=(-math.pi, math.pi))[0].astype(float)
    return {
        "cumulant3_mean": float(cumulant3.mean()),
        "cumulant3_std": float(cumulant3.std()),
        "cumulant4_mean": float(cumulant4.mean()),
        "cumulant4_std": float(cumulant4.std()),
        "log_bispectrum_mean": float(np.log1p(magnitude).mean()),
        "log_bispectrum_std": float(np.log1p(magnitude).std()),
        "bispectrum_entropy_p1": entropy(magnitude),
        "bispectrum_entropy_p2": entropy(magnitude**2),
        "bispectrum_entropy_p3": entropy(magnitude**3),
        "bispectrum_phase_entropy": entropy(phase_histogram),
        "bicoherence_mean": float(np.mean(bicoherence)),
        "bicoherence_std": float(np.std(bicoherence)),
        "bicoherence_p90": float(np.percentile(bicoherence, 90)),
        "bicoherence_max": float(np.max(bicoherence)),
    }


def hos_features(channel: np.ndarray, mask: np.ndarray, channel_name: str) -> dict[str, float]:
    features: dict[str, float] = {}
    for angle_name, angle in [("h", 0.0), ("d45", 45.0), ("v", 90.0), ("d135", -45.0)]:
        summary = hos_summary(rotated_line_signals(channel, mask, angle))
        for name, value in summary.items():
            features[f"hos_{channel_name}_{angle_name}_{name}"] = value
    return features


def extract_image_features() -> pd.DataFrame:
    manifest = pd.read_csv(
        MANIFEST_PATH,
        usecols=["image_id", "patient_alias", "view"],
        dtype={"image_id": str, "patient_alias": str, "view": str},
    )
    rows: list[dict[str, object]] = []
    for position, row in enumerate(manifest.itertuples(index=False), start=1):
        image_path = SOURCE / "native" / f"{row.image_id}.png"
        mask_path = SOURCE / "masks" / f"{row.image_id}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask_image is None:
            raise FileNotFoundError(f"Missing image or mask for {row.image_id}")
        mask = mask_image > 0
        image, mask = crop_to_mask(image, mask)
        native = robust_channel(image, mask)
        enhanced = clahe_channel(native, mask)
        features: dict[str, object] = {
            "image_id": row.image_id,
            "patient_alias": row.patient_alias,
            "view": row.view,
        }
        for channel_name, channel in [("native", native), ("clahe", enhanced)]:
            features.update(lqp_features(channel, mask, channel_name))
            features.update(hos_features(channel, mask, channel_name))
        rows.append(features)
        if position % 50 == 0 or position == len(manifest):
            print(f"Extracted {position}/{len(manifest)} images")
    frame = pd.DataFrame(rows)
    if frame.shape[0] != 349 or frame["patient_alias"].nunique() != 98:
        raise RuntimeError(f"Unexpected feature table shape: {frame.shape}")
    return frame


def aggregate_patients(image_features: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [column for column in image_features.columns if column.startswith(("hos_", "lqp_"))]
    rows: list[dict[str, object]] = []
    for patient_alias, group in image_features.groupby("patient_alias", sort=True):
        row: dict[str, object] = {"patient_alias": patient_alias}
        values = group[feature_columns]
        for statistic, series in [
            ("all_mean", values.mean()),
            ("all_std", values.std(ddof=0)),
            ("all_min", values.min()),
            ("all_max", values.max()),
        ]:
            row.update({f"{statistic}__{key}": float(value) for key, value in series.items()})
        for view in ("LL", "LT", "RL", "RT"):
            subset = group.loc[group["view"] == view, feature_columns]
            row[f"view_present__{view}"] = int(not subset.empty)
            series = subset.mean() if not subset.empty else pd.Series(np.nan, index=feature_columns)
            row.update({f"{view}_mean__{key}": float(value) for key, value in series.items()})
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.shape[0] != 98:
        raise RuntimeError(f"Expected 98 patient rows, found {result.shape[0]}")
    return result


def load_development_pair_predictions() -> pd.DataFrame:
    rows = []
    with PAIR_PREDICTIONS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["role"] != "development_validation" or row["group"] != "PAIR":
                continue
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "patient_alias": row["patient_alias"],
                    "label": int(row["label"]),
                    "pair_score": float(row["score"]),
                }
            )
    result = pd.DataFrame(rows)
    if result.shape[0] != 200 or sorted(result["seed"].unique()) != SEEDS:
        raise RuntimeError(f"Unexpected fixed-pair development table: {result.shape}")
    return result


def build_model(k: int, c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k)),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=c_value,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=seed,
                ),
            ),
        ]
    )


def select_inner_model(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[int, float, float]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    records = []
    for k in K_VALUES:
        for c_value in C_VALUES:
            fold_scores = []
            for train_index, validation_index in splitter.split(x, y):
                model = build_model(k, c_value, seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x[train_index], y[train_index])
                score = model.predict_proba(x[validation_index])[:, 1]
                auc = roc_auc_score(y[validation_index], score)
                ap = average_precision_score(y[validation_index], score)
                fold_scores.append((auc + ap) / 2.0)
            records.append((float(np.mean(fold_scores)), -k, -c_value, k, c_value))
    best = max(records)
    return int(best[3]), float(best[4]), float(best[0])


def metric_row(seed: int, group: str, y: np.ndarray, score: np.ndarray) -> dict[str, object]:
    return {
        "seed": seed,
        "group": group,
        "role": "development_validation",
        "n": len(y),
        "auroc": roc_auc_score(y, score),
        "average_precision": average_precision_score(y, score),
    }


def run_development(patient_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments = pd.read_csv(
        ASSIGNMENTS_PATH,
        usecols=["seed", "role", "patient_alias", "label"],
        dtype={"patient_alias": str},
    )
    assignments = assignments.loc[assignments["role"].isin(["train", "validation"])].copy()
    if set(assignments["role"]) != {"train", "validation"}:
        raise RuntimeError("Development runner received an unexpected role")
    pair = load_development_pair_predictions()
    feature_columns = [column for column in patient_features.columns if column != "patient_alias"]
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        split = assignments.loc[assignments["seed"] == seed]
        train = split.loc[split["role"] == "train", ["patient_alias", "label"]]
        development = split.loc[split["role"] == "validation", ["patient_alias", "label"]]
        if len(train) != 58 or len(development) != 20:
            raise RuntimeError(f"Unexpected split size for seed {seed}")
        train_frame = train.merge(patient_features, on="patient_alias", validate="one_to_one")
        development_frame = development.merge(patient_features, on="patient_alias", validate="one_to_one")
        pair_frame = development.merge(
            pair.loc[pair["seed"] == seed, ["patient_alias", "label", "pair_score"]],
            on=["patient_alias", "label"],
            validate="one_to_one",
        )
        if list(development_frame["patient_alias"]) != list(pair_frame["patient_alias"]):
            raise RuntimeError(f"Patient order mismatch for seed {seed}")
        y_train = train_frame["label"].to_numpy(dtype=int)
        y_development = development_frame["label"].to_numpy(dtype=int)
        pair_score = pair_frame["pair_score"].to_numpy(dtype=float)
        metric_rows.append(metric_row(seed, "PAIR", y_development, pair_score))
        candidate_scores: dict[str, np.ndarray] = {}
        for feature_set, prefix in FEATURE_SETS.items():
            if prefix is None:
                columns = [column for column in feature_columns if "__hos_" in column or "__lqp_" in column]
            else:
                columns = [column for column in feature_columns if f"__{prefix}" in column]
            columns += [column for column in feature_columns if column.startswith("view_present__")]
            x_train = train_frame[columns].to_numpy(dtype=float)
            x_development = development_frame[columns].to_numpy(dtype=float)
            k, c_value, inner_score = select_inner_model(x_train, y_train, seed)
            model = build_model(k, c_value, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_train, y_train)
            score = model.predict_proba(x_development)[:, 1]
            candidate_scores[feature_set] = score
            metric_rows.append(metric_row(seed, feature_set, y_development, score))
            tuning_rows.append(
                {
                    "seed": seed,
                    "feature_set": feature_set,
                    "selected_k": k,
                    "selected_C": c_value,
                    "inner_cv_objective": inner_score,
                    "outer_test_accessed": False,
                }
            )
            for patient_alias, label, value in zip(
                development_frame["patient_alias"], y_development, score
            ):
                prediction_rows.append(
                    {
                        "seed": seed,
                        "role": "development_validation",
                        "group": feature_set,
                        "patient_alias": patient_alias,
                        "label": int(label),
                        "score": float(value),
                    }
                )
        for feature_set, score in candidate_scores.items():
            for weight in (0.25, 0.50):
                group_name = f"PAIR{int((1-weight)*100)}+{int(weight*100)}%{feature_set}"
                ensemble_score = (1.0 - weight) * pair_score + weight * score
                metric_rows.append(metric_row(seed, group_name, y_development, ensemble_score))
                for patient_alias, label, value in zip(
                    development_frame["patient_alias"], y_development, ensemble_score
                ):
                    prediction_rows.append(
                        {
                            "seed": seed,
                            "role": "development_validation",
                            "group": group_name,
                            "patient_alias": patient_alias,
                            "label": int(label),
                            "score": float(value),
                        }
                    )
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
    baseline = metrics.loc[metrics["group"] == "PAIR", ["seed", "auroc", "average_precision"]].rename(
        columns={"auroc": "pair_auroc", "average_precision": "pair_average_precision"}
    )
    gate_rows = []
    for group, frame in metrics.loc[metrics["group"].str.startswith("PAIR") & (metrics["group"] != "PAIR")].groupby("group"):
        comparison = frame.merge(baseline, on="seed", validate="one_to_one")
        delta_auc = comparison["auroc"] - comparison["pair_auroc"]
        delta_ap = comparison["average_precision"] - comparison["pair_average_precision"]
        simultaneous = int(((delta_auc > 0) & (delta_ap > 0)).sum())
        mean_auc_gain = float(delta_auc.mean())
        mean_ap_gain = float(delta_ap.mean())
        gate_rows.append(
            {
                "group": group,
                "mean_auroc_gain": mean_auc_gain,
                "mean_average_precision_gain": mean_ap_gain,
                "seeds_both_improved": simultaneous,
                "passes_registered_gate": bool(
                    mean_auc_gain >= 0.03 and mean_ap_gain >= 0.02 and simultaneous >= 6
                ),
                "exploratory_test_authorized": False,
            }
        )
    gate = pd.DataFrame(gate_rows).sort_values(
        ["passes_registered_gate", "mean_auroc_gain", "mean_average_precision_gain"],
        ascending=False,
    )
    return summary, gate


def write_readme(summary: pd.DataFrame, gate: pd.DataFrame, patient_features: pd.DataFrame) -> None:
    baseline = summary.loc[summary["group"] == "PAIR"].iloc[0]
    best_candidate = summary.loc[~summary["group"].str.startswith("PAIR")].iloc[0]
    best_ensemble = summary.loc[summary["group"].str.startswith("PAIR") & (summary["group"] != "PAIR")].iloc[0]
    passed = gate.loc[gate["passes_registered_gate"]]
    decision = (
        f"{', '.join(passed['group'])} passed the registered development gate. A separate "
        "pipeline-lock step is required before any exploratory internal-test evaluation."
        if not passed.empty
        else "No ensemble passed the registered development gate; the exploratory internal test remains unopened."
    )
    text = f"""# Literature-guided HOS and local quinary-pattern development experiment

## Scope

This experiment evaluates a literature-guided adaptation of higher-order spectrum (HOS),
third/fourth cumulants, and local quinary patterns (LQP) inside the existing kidney masks.
It is not an exact replication of Acharya et al. The complete original implementation
was not available, so every adaptation is defined in `protocol.json` before evaluation.

Only the ten fixed development-validation sets were scored. Each outer training set
contained 58 patients and each development set contained 20 patients. The 20-patient
`test` role in every seed was excluded from modeling and evaluation.

## Data and features

- 349 kidney ultrasound images from 98 clinician-confirmed patients.
- {patient_features.shape[1] - 1:,} patient-level variables after deterministic aggregation.
- Native robust normalization and a fixed CLAHE enhancement channel.
- Four-direction line-profile bispectra and cumulants.
- Five-state quinary texture on circular and two elongated stencils.
- View-specific LL/LT/RL/RT means plus global mean, SD, minimum, and maximum.

## Development results

- Fixed pair: AUROC {baseline.mean_auroc:.3f}, AP {baseline.mean_average_precision:.3f}.
- Best standalone registered candidate: {best_candidate.group}, AUROC {best_candidate.mean_auroc:.3f}, AP {best_candidate.mean_average_precision:.3f}.
- Best registered ensemble by mean AUROC: {best_ensemble.group}, AUROC {best_ensemble.mean_auroc:.3f}, AP {best_ensemble.mean_average_precision:.3f}.

Registered gate: mean AUROC gain >= 0.03, mean AP gain >= 0.02, and both metrics improve
in at least 6 of 10 fixed draws. {decision}

## Interpretation boundary

These are repeated fixed development results, not cross-validation performance and not
independent confirmation. No claim of a visible scar, inflammatory focus, cortex, medulla,
or collecting-system compartment is made from weak patient labels. The extracted features
quantify intensity nonlinearity and local texture only.

See `references.json`, `protocol.json`, `protocol_lock.json`, `metrics_per_seed.csv`,
`summary_development.csv`, `candidate_gate.csv`, and `selected_hyperparameters.csv`.
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
    image_feature_path = OUT / "image_features_deidentified.csv"
    patient_feature_path = OUT / "patient_features_deidentified.csv"
    if args.reuse_features and image_feature_path.exists() and patient_feature_path.exists():
        image_features = pd.read_csv(image_feature_path)
        patient_features = pd.read_csv(patient_feature_path)
    else:
        image_features = extract_image_features()
        patient_features = aggregate_patients(image_features)
        image_features.to_csv(image_feature_path, index=False)
        patient_features.to_csv(patient_feature_path, index=False)
    metrics, predictions, tuning = run_development(patient_features)
    summary, gate = summarize(metrics)
    metrics.to_csv(OUT / "metrics_per_seed.csv", index=False)
    predictions.to_csv(OUT / "development_predictions.csv", index=False)
    tuning.to_csv(OUT / "selected_hyperparameters.csv", index=False)
    summary.to_csv(OUT / "summary_development.csv", index=False)
    gate.to_csv(OUT / "candidate_gate.csv", index=False)
    audit = {
        "images": int(len(image_features)),
        "patients": int(patient_features["patient_alias"].nunique()),
        "seeds": SEEDS,
        "roles_evaluated": ["train", "development_validation"],
        "test_role_evaluated": False,
        "exploratory_test_predictions_loaded": False,
        "passing_candidates": gate.loc[gate["passes_registered_gate"], "group"].tolist(),
        "output_hashes": {
            path.name: sha256(path)
            for path in [
                image_feature_path,
                patient_feature_path,
                OUT / "metrics_per_seed.csv",
                OUT / "development_predictions.csv",
                OUT / "selected_hyperparameters.csv",
                OUT / "summary_development.csv",
                OUT / "candidate_gate.csv",
            ]
        },
    }
    write_json(OUT / "run_audit.json", audit)
    write_readme(summary, gate, patient_features)
    print(summary.to_string(index=False))
    print("\nCandidate gate:")
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
