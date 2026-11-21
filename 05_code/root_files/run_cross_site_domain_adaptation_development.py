from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CYCH_cross_site_domain_adaptation_development_20260914"
TARGET_RUN = ROOT / "CYCH_dinov3_native_probe_20260913"
TARGET_MANIFEST_PATH = ROOT / "CYCH_random_validation_20260913" / "image_manifest.csv"
TARGET_IMAGE_FEATURES_PATH = TARGET_RUN / "features" / "dense_canvas_image.npy"
TARGET_PATIENT_FEATURES_REFERENCE = TARGET_RUN / "features" / "dense_canvas.npy"
SOURCE_MANIFEST_PATH = ROOT / "yolo_crops" / "manifest.csv"
ASSIGNMENTS_PATH = (
    ROOT
    / "CYCH_directional_robustness_selection_20260914"
    / "random_patient_assignments_deidentified.csv"
)
PAIR_PREDICTIONS_PATH = TARGET_RUN / "patient_predictions_by_role.csv"
WEIGHTS_PATH = (
    Path(r"C:\Users\udoo_w2\Desktop\lung_cancerimage\codex_experiments")
    / "pretrained"
    / "RadImageNet_pytorch"
    / "DenseNet121.pt"
)

PROTOCOL_PATH = OUT / "protocol.json"
REFERENCES_PATH = OUT / "references.json"
LOCK_PATH = OUT / "protocol_lock.json"
SCRIPT_COPY = OUT / "source" / Path(__file__).name
SEEDS = list(range(20260913, 20260923))
PCA_COMPONENTS = 32
C_VALUES = [0.01, 0.1, 1.0]
CANDIDATES = ["SOURCE_ONLY", "POOLED", "CORAL", "FEATURE_AUGMENTATION"]


REFERENCES = [
    {
        "method": "CORrelation ALignment (CORAL)",
        "citation": "Sun, Feng, and Saenko, AAAI 2016",
        "url": "https://ojs.aaai.org/index.php/AAAI/article/view/10306",
        "definition": (
            "Whiten source features and recolor them with target covariance so source "
            "and target second-order feature statistics align."
        ),
    },
    {
        "method": "Supervised feature augmentation for domain adaptation",
        "citation": "Daume III, ACL 2007",
        "url": "https://aclanthology.org/P07-1033/",
        "definition": (
            "Represent source as [common, source-specific, zero] and target as "
            "[common, zero, target-specific] before fitting a standard classifier."
        ),
    },
    {
        "method": "FedBN and feature-shift non-IID data",
        "citation": "Li et al., ICLR 2021",
        "url": "https://openreview.net/pdf?id=6YEQUn0QICG",
        "relevance": (
            "Motivates explicit treatment of scanner/site feature shift. FedBN itself is "
            "not implemented in this frozen-feature experiment."
        ),
    },
    {
        "method": "Ultrasound domain adaptation",
        "citation": "Pu et al., ICML 2024",
        "url": "https://proceedings.mlr.press/v235/pu24b.html",
        "relevance": (
            "Documents effectiveness loss when ultrasound models move between institutions "
            "and motivates target-aware source adaptation. The paper addresses detection, "
            "not renal inflammation classification."
        ),
    },
    {
        "method": "RadImageNet",
        "citation": "Mei et al., Radiology: Artificial Intelligence, 2022",
        "url": "https://github.com/BMEII-AI/RadImageNet",
        "relevance": "Supplies the frozen DenseNet-121 medical-image representation.",
    },
]


PROTOCOL = {
    "title": "Development-only cross-site frozen-feature domain-adaptation experiment",
    "cohorts": {
        "target_CYCH": {"patients": 98, "images": 349, "positive": 52, "negative": 46},
        "source_NCKUH": {"patients": 264, "images": 1031},
        "patient_id_overlap": 0,
    },
    "target_splits": {
        "fixed_seeds": SEEDS,
        "per_seed": {"train": 58, "development_validation": 20, "unused_test": 20},
    },
    "hard_constraints": [
        "The RadImageNet DenseNet-121 encoder is frozen and its BatchNorm layers remain in eval mode.",
        "Every source and target image enters after kidney localization/segmentation.",
        "No source or target patient identifier is written to experiment outputs.",
        "All target-dependent preprocessing is fitted on the 58 CYCH outer-training patients only.",
        "Inner model selection uses only target outer-training patients.",
        "Rows assigned target role=test are not evaluated.",
        "Exploratory-test baseline predictions are not loaded.",
    ],
    "source_preprocessing": (
        "Clean NCKUH frames, established physician kidney polygon, tight bounding crop, "
        "within-kidney P1-P99 normalization, zero exterior, diagonal square canvas, 224x224."
    ),
    "target_preprocessing": (
        "Reuse the previously verified CYCH P1-P99 kidney-mask canvas and frozen image features."
    ),
    "patient_aggregation": "Image-feature mean and standard deviation, yielding 2048 variables per patient.",
    "shared_reduction": (
        "StandardScaler and 32-component PCA fitted to source plus the applicable target-training patients. "
        "No development or test target is included in fitting."
    ),
    "candidate_definitions": {
        "SOURCE_ONLY": "Classifier fitted to source patients only after shared unlabeled reduction.",
        "POOLED": "Source and target-training patients share one reduced feature space.",
        "CORAL": "Reduced source features are covariance-aligned to target-training features before pooling.",
        "FEATURE_AUGMENTATION": (
            "Daume-style [common, domain-specific] augmented reduced features before pooling."
        ),
    },
    "classifier": {
        "type": "L2 logistic regression",
        "C_values": C_VALUES,
        "selection": "four-fold stratified inner CV on target outer-training patients",
        "objective": "mean of target-patient AUROC and average precision",
        "weighting": "equal total weight per site and equal total weight per class within site",
    },
    "ensembles": "Fixed pair plus each candidate at predeclared candidate weights 0.25 and 0.50.",
    "candidate_gate_vs_fixed_pair": {
        "mean_auroc_gain_at_least": 0.03,
        "mean_average_precision_gain_at_least": 0.02,
        "seeds_with_both_metrics_improved_at_least": 6,
        "test_action": "No exploratory internal-test evaluation unless all criteria pass.",
    },
    "interpretation_boundary": (
        "This is a centralized methodological source-transfer experiment, not a federated-learning run. "
        "Source/target label compatibility remains a clinical provenance assumption."
    ),
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
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "references_sha256": sha256(REFERENCES_PATH),
            "script_sha256": sha256(Path(__file__)),
            "target_manifest_sha256": sha256(TARGET_MANIFEST_PATH),
            "target_image_features_sha256": sha256(TARGET_IMAGE_FEATURES_PATH),
            "source_manifest_sha256": sha256(SOURCE_MANIFEST_PATH),
            "assignments_sha256": sha256(ASSIGNMENTS_PATH),
            "fixed_pair_predictions_sha256": sha256(PAIR_PREDICTIONS_PATH),
            "encoder_weights_sha256": sha256(WEIGHTS_PATH),
        },
    )
    print(f"Locked protocol: {LOCK_PATH}", flush=True)


def verify_lock() -> None:
    if not LOCK_PATH.exists():
        raise RuntimeError("Run --lock-only before feature extraction or model fitting")
    expected = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    paths = {
        "protocol_sha256": PROTOCOL_PATH,
        "references_sha256": REFERENCES_PATH,
        "script_sha256": Path(__file__),
        "target_manifest_sha256": TARGET_MANIFEST_PATH,
        "target_image_features_sha256": TARGET_IMAGE_FEATURES_PATH,
        "source_manifest_sha256": SOURCE_MANIFEST_PATH,
        "assignments_sha256": ASSIGNMENTS_PATH,
        "fixed_pair_predictions_sha256": PAIR_PREDICTIONS_PATH,
        "encoder_weights_sha256": WEIGHTS_PATH,
    }
    changed = [key for key, path in paths.items() if sha256(path) != expected[key]]
    if changed:
        raise RuntimeError(f"Locked inputs changed: {changed}")


def nearest_extension(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    if mask.all():
        return gray.copy()
    indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return gray[tuple(indices)]


def percentile_normalize(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    extended = nearest_extension(gray, mask)
    low, high = np.percentile(gray[mask], [1, 99])
    if high <= low:
        result = extended
    else:
        result = np.rint(np.clip((extended.astype(float) - low) / (high - low), 0, 1) * 255)
    return np.where(mask, result, 0).astype(np.uint8)


def canvas_image(gray: np.ndarray, mask: np.ndarray, size: int = 224) -> np.ndarray:
    side = int(np.ceil(np.hypot(*gray.shape))) + 8
    padded = np.zeros((side, side), np.uint8)
    padded_mask = np.zeros_like(padded)
    y0 = (side - gray.shape[0]) // 2
    x0 = (side - gray.shape[1]) // 2
    padded[y0 : y0 + gray.shape[0], x0 : x0 + gray.shape[1]] = gray
    padded_mask[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]] = mask.astype(np.uint8)
    padded *= padded_mask
    resized = cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(padded_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    resized *= resized_mask
    return np.repeat(resized[..., None], 3, axis=2)


def source_table_private() -> pd.DataFrame:
    frame = pd.read_csv(
        SOURCE_MANIFEST_PATH,
        usecols=["site", "label", "filename", "relative_path", "patient_id", "view"],
        dtype={"patient_id": str},
    )
    frame = frame.loc[frame["site"] == "NCKUH"].copy()
    frame = frame.sort_values(["patient_id", "filename"]).reset_index(drop=True)
    if len(frame) != 1031 or frame["patient_id"].nunique() != 264:
        raise RuntimeError(f"Unexpected NCKUH source table: {frame.shape}")
    if frame.groupby("patient_id")["label"].nunique().gt(1).any():
        raise RuntimeError("NCKUH patient-label conflict")
    aliases = {patient_id: f"N{index:03d}" for index, patient_id in enumerate(sorted(frame["patient_id"].unique()), 1)}
    frame["patient_alias"] = frame["patient_id"].map(aliases)
    frame["image_alias"] = [f"NI{index:04d}" for index in range(1, len(frame) + 1)]
    return frame


def prepare_source_canvases(source: pd.DataFrame) -> tuple[list[np.ndarray], pd.DataFrame]:
    canvases = []
    public_rows = []
    for position, row in enumerate(source.itertuples(index=False), 1):
        clean_name = row.filename.replace("_red_", "_")
        image_path = ROOT / "original_png_nckuh" / clean_name
        mask_path = ROOT / "nckuh_bw" / row.filename
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask_image is None or gray.shape != mask_image.shape:
            raise FileNotFoundError(f"Missing or mismatched source image/mask at row {position}")
        mask = mask_image > 0
        ys, xs = np.where(mask)
        if len(xs) < 64:
            raise RuntimeError(f"Source mask too small at row {position}")
        x0, x1 = max(0, int(xs.min()) - 4), min(gray.shape[1], int(xs.max()) + 5)
        y0, y1 = max(0, int(ys.min()) - 4), min(gray.shape[0], int(ys.max()) + 5)
        cropped_gray = gray[y0:y1, x0:x1]
        cropped_mask = mask[y0:y1, x0:x1]
        normalized = percentile_normalize(cropped_gray, cropped_mask)
        canvases.append(canvas_image(normalized, cropped_mask))
        public_rows.append(
            {
                "image_alias": row.image_alias,
                "patient_alias": row.patient_alias,
                "view": row.view,
                "label": int(row.label),
                "mask_pixels": int(cropped_mask.sum()),
            }
        )
        if position % 200 == 0 or position == len(source):
            print(f"Prepared source kidney canvases {position}/{len(source)}", flush=True)
    return canvases, pd.DataFrame(public_rows)


def model_digest(model: object) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def extract_source_features(source: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    import torch
    from radimagenet_backbones import radimagenet_densenet121

    torch.set_num_threads(4)
    torch.manual_seed(20260914)
    canvases, public_manifest = prepare_source_canvases(source)
    model = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    before = model_digest(model)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(canvases), 64):
            batch = np.stack(canvases[start : start + 64]).transpose(0, 3, 1, 2).copy()
            tensor = torch.from_numpy(batch).cuda().float() / 127.5 - 1.0
            chunks.append(model(tensor).cpu().numpy())
            if start % 256 == 0:
                print(f"Encoded source images {min(start + 64, len(canvases))}/{len(canvases)}", flush=True)
    features = np.concatenate(chunks).astype(np.float32)
    after = model_digest(model)
    if before != after or features.shape != (1031, 1024) or not np.isfinite(features).all():
        raise RuntimeError("Frozen source feature extraction verification failed")
    write_json(
        OUT / "encoder_provenance.json",
        {
            "model": "RadImageNet DenseNet-121",
            "weights_sha256": sha256(WEIGHTS_PATH),
            "state_before": before,
            "state_after": after,
            "normalization": "x/127.5-1",
            "feature_shape": list(features.shape),
            "trainable_parameters": 0,
        },
    )
    return features, public_manifest


def aggregate_features(features: np.ndarray, aliases: pd.Series) -> tuple[np.ndarray, list[str]]:
    patient_aliases = sorted(aliases.unique())
    rows = []
    values = aliases.to_numpy()
    for patient_alias in patient_aliases:
        patient_values = features[values == patient_alias]
        rows.append(np.concatenate([patient_values.mean(axis=0), patient_values.std(axis=0)]))
    return np.asarray(rows, dtype=np.float32), patient_aliases


def load_or_extract_features() -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    source_feature_path = OUT / "source_image_features.npy"
    source_manifest_public_path = OUT / "source_image_manifest_deidentified.csv"
    private_source = source_table_private()
    if source_feature_path.exists() and source_manifest_public_path.exists():
        source_image_features = np.load(source_feature_path)
        source_public = pd.read_csv(source_manifest_public_path)
    else:
        source_image_features, source_public = extract_source_features(private_source)
        np.save(source_feature_path, source_image_features)
        source_public.to_csv(source_manifest_public_path, index=False)
    source_patient_features, source_aliases = aggregate_features(
        source_image_features, source_public["patient_alias"]
    )
    source_labels = source_public.groupby("patient_alias")["label"].first().reindex(source_aliases).to_numpy(int)
    source_patients = pd.DataFrame({"patient_alias": source_aliases, "label": source_labels})

    target_manifest = pd.read_csv(
        TARGET_MANIFEST_PATH,
        usecols=["image_id", "patient_alias"],
        dtype=str,
    )
    target_image_features = np.load(TARGET_IMAGE_FEATURES_PATH)
    target_patient_features, target_aliases = aggregate_features(
        target_image_features, target_manifest["patient_alias"]
    )
    inherited = np.load(TARGET_PATIENT_FEATURES_REFERENCE)
    if not np.allclose(target_patient_features, inherited, rtol=1e-5, atol=1e-5):
        raise RuntimeError("Target patient aggregation does not reproduce the inherited cache")
    target_patients = pd.DataFrame({"patient_alias": target_aliases})
    np.save(OUT / "source_patient_features.npy", source_patient_features)
    np.save(OUT / "target_patient_features.npy", target_patient_features)
    source_patients.to_csv(OUT / "source_patients_deidentified.csv", index=False)
    target_patients.to_csv(OUT / "target_patients_deidentified.csv", index=False)
    return source_patient_features, source_patients, target_patient_features, target_patients


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
    if result.shape != (200, 4):
        raise RuntimeError(f"Unexpected pair-development table {result.shape}")
    return result


def matrix_sqrt(matrix: np.ndarray, inverse: bool = False) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    values = np.maximum(values, 1e-6)
    powers = values ** (-0.5 if inverse else 0.5)
    return (vectors * powers) @ vectors.T


def coral_source_to_target(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_covariance = np.cov(source_centered, rowvar=False) + np.eye(source.shape[1]) * 1e-3
    target_covariance = np.cov(target_centered, rowvar=False) + np.eye(target.shape[1]) * 1e-3
    transform = matrix_sqrt(source_covariance, inverse=True) @ matrix_sqrt(target_covariance)
    return source_centered @ transform + target_mean


def domain_class_weights(source_y: np.ndarray, target_y: np.ndarray) -> np.ndarray:
    weights = []
    for labels in (source_y, target_y):
        local = np.zeros(len(labels), dtype=float)
        for value in (0, 1):
            count = int(np.sum(labels == value))
            local[labels == value] = 0.25 / max(count, 1)
        weights.append(local)
    combined = np.concatenate(weights)
    return combined / combined.mean()


def fit_transform_reduction(source_x: np.ndarray, target_x: np.ndarray):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(np.vstack([source_x, target_x]))
    source_scaled = scaler.transform(source_x)
    target_scaled = scaler.transform(target_x)
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=20260914)
    pca.fit(np.vstack([source_scaled, target_scaled]))
    return scaler, pca, pca.transform(source_scaled), pca.transform(target_scaled)


def candidate_training_matrix(
    candidate: str,
    source_reduced: np.ndarray,
    target_reduced: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if candidate == "SOURCE_ONLY":
        return source_reduced, np.zeros((0, source_reduced.shape[1]))
    if candidate == "POOLED":
        return source_reduced, target_reduced
    if candidate == "CORAL":
        return coral_source_to_target(source_reduced, target_reduced), target_reduced
    if candidate == "FEATURE_AUGMENTATION":
        zeros_source = np.zeros_like(source_reduced)
        zeros_target = np.zeros_like(target_reduced)
        source_augmented = np.concatenate([source_reduced, source_reduced, zeros_source], axis=1)
        target_augmented = np.concatenate([target_reduced, zeros_target, target_reduced], axis=1)
        return source_augmented, target_augmented
    raise ValueError(candidate)


def target_evaluation_matrix(candidate: str, reduced: np.ndarray) -> np.ndarray:
    if candidate == "FEATURE_AUGMENTATION":
        return np.concatenate([reduced, np.zeros_like(reduced), reduced], axis=1)
    return reduced


def fit_candidate(
    candidate: str,
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    c_value: float,
    seed: int,
):
    from sklearn.linear_model import LogisticRegression

    scaler, pca, source_reduced, target_reduced = fit_transform_reduction(source_x, target_x)
    transformed_source, transformed_target = candidate_training_matrix(
        candidate, source_reduced, target_reduced
    )
    if candidate == "SOURCE_ONLY":
        train_x = transformed_source
        train_y = source_y
        weights = np.ones(len(source_y), dtype=float)
        for value in (0, 1):
            weights[source_y == value] = len(source_y) / (2 * max(np.sum(source_y == value), 1))
    else:
        train_x = np.vstack([transformed_source, transformed_target])
        train_y = np.concatenate([source_y, target_y])
        weights = domain_class_weights(source_y, target_y)
    model = LogisticRegression(
        C=c_value,
        solver="liblinear",
        max_iter=3000,
        random_state=seed,
    )
    model.fit(train_x, train_y, sample_weight=weights)
    return scaler, pca, model


def predict_candidate(candidate: str, fitted: tuple, target_x: np.ndarray) -> np.ndarray:
    scaler, pca, model = fitted
    reduced = pca.transform(scaler.transform(target_x))
    matrix = target_evaluation_matrix(candidate, reduced)
    return model.predict_proba(matrix)[:, 1]


def select_c_inside_target_training(
    candidate: str,
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    records = []
    for c_value in C_VALUES:
        scores = []
        for train_index, validation_index in splitter.split(target_x, target_y):
            fitted = fit_candidate(
                candidate,
                source_x,
                source_y,
                target_x[train_index],
                target_y[train_index],
                c_value,
                seed,
            )
            predictions = predict_candidate(candidate, fitted, target_x[validation_index])
            scores.append(
                (
                    roc_auc_score(target_y[validation_index], predictions)
                    + average_precision_score(target_y[validation_index], predictions)
                )
                / 2.0
            )
        records.append((float(np.mean(scores)), -c_value, c_value))
    best = max(records)
    return float(best[2]), float(best[0])


def metric(seed: int, group: str, y: np.ndarray, score: np.ndarray) -> dict[str, object]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "seed": seed,
        "group": group,
        "role": "development_validation",
        "n": len(y),
        "auroc": roc_auc_score(y, score),
        "average_precision": average_precision_score(y, score),
    }


def run_development(
    source_x: np.ndarray,
    source_patients: pd.DataFrame,
    target_x: np.ndarray,
    target_patients: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments = pd.read_csv(
        ASSIGNMENTS_PATH,
        usecols=["seed", "role", "patient_alias", "label"],
        dtype={"patient_alias": str},
    )
    assignments = assignments.loc[assignments["role"].isin(["train", "validation"])].copy()
    pair = load_development_pair_predictions()
    target_index = {alias: index for index, alias in enumerate(target_patients["patient_alias"])}
    source_y = source_patients["label"].to_numpy(int)
    metrics = []
    predictions = []
    tuning = []
    for seed in SEEDS:
        split = assignments.loc[assignments["seed"] == seed]
        train = split.loc[split["role"] == "train", ["patient_alias", "label"]]
        development = split.loc[split["role"] == "validation", ["patient_alias", "label"]]
        train_indices = np.asarray([target_index[value] for value in train["patient_alias"]])
        development_indices = np.asarray([target_index[value] for value in development["patient_alias"]])
        target_train_x = target_x[train_indices]
        target_train_y = train["label"].to_numpy(int)
        target_development_x = target_x[development_indices]
        target_development_y = development["label"].to_numpy(int)
        pair_frame = development.merge(
            pair.loc[pair["seed"] == seed, ["patient_alias", "label", "pair_score"]],
            on=["patient_alias", "label"],
            validate="one_to_one",
        )
        pair_score = pair_frame["pair_score"].to_numpy(float)
        metrics.append(metric(seed, "PAIR", target_development_y, pair_score))
        for candidate in CANDIDATES:
            c_value, inner_score = select_c_inside_target_training(
                candidate,
                source_x,
                source_y,
                target_train_x,
                target_train_y,
                seed,
            )
            fitted = fit_candidate(
                candidate,
                source_x,
                source_y,
                target_train_x,
                target_train_y,
                c_value,
                seed,
            )
            score = predict_candidate(candidate, fitted, target_development_x)
            metrics.append(metric(seed, candidate, target_development_y, score))
            tuning.append(
                {
                    "seed": seed,
                    "candidate": candidate,
                    "selected_C": c_value,
                    "inner_target_objective": inner_score,
                    "target_test_accessed": False,
                }
            )
            for patient_alias, label, value in zip(
                development["patient_alias"], target_development_y, score
            ):
                predictions.append(
                    {
                        "seed": seed,
                        "role": "development_validation",
                        "group": candidate,
                        "patient_alias": patient_alias,
                        "label": int(label),
                        "score": float(value),
                    }
                )
            for candidate_weight in (0.25, 0.50):
                group = f"PAIR{int((1-candidate_weight)*100)}+{int(candidate_weight*100)}%{candidate}"
                ensemble = (1.0 - candidate_weight) * pair_score + candidate_weight * score
                metrics.append(metric(seed, group, target_development_y, ensemble))
                for patient_alias, label, value in zip(
                    development["patient_alias"], target_development_y, ensemble
                ):
                    predictions.append(
                        {
                            "seed": seed,
                            "role": "development_validation",
                            "group": group,
                            "patient_alias": patient_alias,
                            "label": int(label),
                            "score": float(value),
                        }
                    )
        print(f"Completed target development seed {seed}", flush=True)
    return pd.DataFrame(metrics), pd.DataFrame(predictions), pd.DataFrame(tuning)


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
    rows = []
    ensembles = metrics.loc[metrics["group"].str.startswith("PAIR") & (metrics["group"] != "PAIR")]
    for group, frame in ensembles.groupby("group"):
        comparison = frame.merge(baseline, on="seed", validate="one_to_one")
        auc_gain = comparison["auroc"] - comparison["pair_auroc"]
        ap_gain = comparison["average_precision"] - comparison["pair_average_precision"]
        both = int(((auc_gain > 0) & (ap_gain > 0)).sum())
        rows.append(
            {
                "group": group,
                "mean_auroc_gain": float(auc_gain.mean()),
                "mean_average_precision_gain": float(ap_gain.mean()),
                "seeds_both_improved": both,
                "passes_registered_gate": bool(auc_gain.mean() >= 0.03 and ap_gain.mean() >= 0.02 and both >= 6),
                "exploratory_test_authorized": False,
            }
        )
    gate = pd.DataFrame(rows).sort_values(
        ["passes_registered_gate", "mean_auroc_gain", "mean_average_precision_gain"],
        ascending=False,
    )
    return summary, gate


def write_readme(summary: pd.DataFrame, gate: pd.DataFrame) -> None:
    baseline = summary.loc[summary["group"] == "PAIR"].iloc[0]
    standalone = summary.loc[summary["group"].isin(CANDIDATES)].iloc[0]
    ensemble = summary.loc[summary["group"].str.startswith("PAIR") & (summary["group"] != "PAIR")].iloc[0]
    passing = gate.loc[gate["passes_registered_gate"], "group"].tolist()
    decision = (
        f"Passing registered ensemble(s): {', '.join(passing)}. A separate pipeline lock is required before test evaluation."
        if passing
        else "No registered ensemble passed; the target exploratory internal test remains unopened."
    )
    text = f"""# Cross-site frozen-feature domain-adaptation development experiment

## Scope

This experiment uses 264 NCKUH source patients and the applicable 58 CYCH target-training
patients in each fixed draw. The two site patient-ID sets have zero overlap. Every image
was kidney-localized before frozen RadImageNet DenseNet-121 feature extraction.

CORAL, supervised feature augmentation, ordinary pooled training, and source-only transfer
are defined in `protocol.json`. Hyperparameters are selected only by four-fold splits of
the 58 target-training patients. The 20-patient development set is evaluated once per
registered candidate. Target `test` roles are not evaluated.

## Development results

- Fixed pair: AUROC {baseline.mean_auroc:.3f}, AP {baseline.mean_average_precision:.3f}.
- Best standalone source-transfer candidate: {standalone.group}, AUROC {standalone.mean_auroc:.3f}, AP {standalone.mean_average_precision:.3f}.
- Best registered ensemble: {ensemble.group}, AUROC {ensemble.mean_auroc:.3f}, AP {ensemble.mean_average_precision:.3f}.
- Decision: {decision}

## Interpretation boundary

This is centralized source-domain transfer, not federated learning. It does not establish
that sharing source images is permissible in deployment. NCKUH and CYCH label compatibility
comes from the current project manifests and still requires clinical provenance confirmation.
All target patients were previously explored; these are development results, not independent
confirmation and not k-fold performance claims.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    args = parser.parse_args()
    if args.lock_only:
        lock_protocol()
        return
    verify_lock()
    source_x, source_patients, target_x, target_patients = load_or_extract_features()
    if args.features_only:
        print("Source and target frozen features verified", flush=True)
        return
    metrics, predictions, tuning = run_development(
        source_x, source_patients, target_x, target_patients
    )
    summary, gate = summarize(metrics)
    metrics.to_csv(OUT / "metrics_per_seed.csv", index=False)
    predictions.to_csv(OUT / "development_predictions.csv", index=False)
    tuning.to_csv(OUT / "selected_hyperparameters.csv", index=False)
    summary.to_csv(OUT / "summary_development.csv", index=False)
    gate.to_csv(OUT / "candidate_gate.csv", index=False)
    write_json(
        OUT / "run_audit.json",
        {
            "source_patients": int(len(source_patients)),
            "source_images": 1031,
            "target_patients": int(len(target_patients)),
            "target_images": 349,
            "patient_id_overlap": 0,
            "target_roles_evaluated": ["train", "development_validation"],
            "target_test_role_evaluated": False,
            "exploratory_test_predictions_loaded": False,
            "passing_candidates": gate.loc[gate["passes_registered_gate"], "group"].tolist(),
        },
    )
    write_readme(summary, gate)
    print(summary.to_string(index=False))
    print("\nCandidate gate:")
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
