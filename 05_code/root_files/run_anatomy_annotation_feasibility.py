"""Label-free reliability analysis for CYCH anatomy and artifact annotations.

The 40-task packet contains 32 unique pilot images and eight hidden repeats.
This script deliberately does not read diagnosis labels and cannot run a
98-patient classifier. It validates two returned submissions, measures mask
agreement, extracts the preregistered compartment radiomics features, and
identifies features that pass the locked inter-rater reliability gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = (
    SCRIPT_DIR
    if (SCRIPT_DIR / "CYCH_anatomy_artifact_annotation_packet_v2_20260914").exists()
    else SCRIPT_DIR.parent
)
PACKET = ROOT / "CYCH_anatomy_artifact_annotation_packet_v2_20260914"
PROTOCOL = ROOT / "CYCH_anatomy_feature_protocol_20260914"
COHORT = ROOT / "CYCH_random_validation_20260913"
DEFAULT_OUTPUT = ROOT / "CYCH_anatomy_annotation_feasibility_20260914"
PRIVATE_KEY = PACKET / "private_task_key.csv"

ANATOMY_REGIONS = ("cortex", "cec", "medulla")
MINIMUM_REGION_PIXELS = 64
MINIMUM_PAIRED_IMAGES = 20
ICC_MINIMUM = 0.90
DISPERSION_MAXIMUM_PERCENT = 20.0
ARTIFACT_MARGIN_PIXELS = 2

BASE_FEATURES = {
    "firstorder": ("Entropy", "Variance", "Skewness", "Kurtosis"),
    "glcm": ("Correlation", "SumAverage"),
    "glrlm": (
        "GrayLevelVariance",
        "RunLengthNonUniformityNormalized",
        "ShortRunLowGrayLevelEmphasis",
        "RunVariance",
    ),
}
HELPER_FEATURES = ("Median", "InterquartileRange")
SIGNED_FEATURE_TOKENS = ("Skewness", "Correlation", "minus")


def load_validator():
    path = PACKET / "validate_internal_region_annotations.py"
    spec = importlib.util.spec_from_file_location("cych_annotation_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load annotation validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_validator()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_submission_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def icc_2_1(values: np.ndarray) -> float:
    """Two-way random, absolute-agreement, single-measure ICC(2,1)."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("ICC input must be a two-dimensional matrix")
    n, k = values.shape
    if n < 2 or k < 2 or not np.isfinite(values).all():
        return float("nan")
    grand = values.mean()
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ms_rows = k * np.square(row_means - grand).sum() / (n - 1)
    ms_columns = n * np.square(column_means - grand).sum() / (k - 1)
    residual = values - row_means[:, None] - column_means[None, :] + grand
    ms_error = np.square(residual).sum() / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")
    return float((ms_rows - ms_error) / denominator)


def percentile_normalize(image: np.ndarray, outer_mask: np.ndarray) -> np.ndarray:
    """Apply the existing fixed ROI percentile normalization without inpainting."""
    values = np.asarray(image, dtype=np.float32)[np.asarray(outer_mask, dtype=bool)]
    if values.size < MINIMUM_REGION_PIXELS:
        raise ValueError("Whole-kidney ROI is too small for normalization")
    low, high = np.percentile(values, (1.0, 99.0))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    normalized = (np.asarray(image, dtype=np.float32) - low) * (255.0 / (high - low))
    return np.clip(normalized, 0.0, 255.0).astype(np.float32)


def artifact_excluded(mask: np.ndarray, artifact: np.ndarray) -> tuple[np.ndarray, float]:
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    artifact = (np.asarray(artifact) > 0).astype(np.uint8)
    diameter = 2 * ARTIFACT_MARGIN_PIXELS + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    expanded = cv2.dilate(artifact, kernel, iterations=1)
    clean = np.logical_and(mask > 0, expanded == 0).astype(np.uint8)
    excluded_fraction = float((mask.sum() - clean.sum()) / max(1, mask.sum()))
    return clean, excluded_fraction


def make_extractor():
    import SimpleITK as sitk
    import radiomics
    from radiomics import featureextractor

    radiomics.setVerbosity(logging.ERROR)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=16,
        normalize=False,
        distances=[1, 2],
        force2D=False,
        minimumROIDimensions=2,
    )
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("Wavelet", customArgs={"wavelet": "coif1", "level": 1})
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(
        firstorder=list(BASE_FEATURES["firstorder"]) + list(HELPER_FEATURES),
        glcm=list(BASE_FEATURES["glcm"]),
        glrlm=list(BASE_FEATURES["glrlm"]),
    )
    return extractor


def registered_feature_names() -> tuple[str, ...]:
    suffixes = [f"{group}_{name}" for group, names in BASE_FEATURES.items() for name in names]
    names = [f"original_{suffix}" for suffix in suffixes]
    for subband in ("LL", "LH", "HL", "HH"):
        names.extend(f"wavelet-{subband}_{suffix}" for suffix in suffixes)
    return tuple(names)


REGISTERED_FEATURE_NAMES = registered_feature_names()


def extract_region_features(
    image: np.ndarray,
    mask: np.ndarray,
    extractor=None,
) -> tuple[dict[str, float], dict[str, float]]:
    import SimpleITK as sitk

    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if int(mask.sum()) < MINIMUM_REGION_PIXELS:
        raise ValueError(f"Region contains fewer than {MINIMUM_REGION_PIXELS} usable pixels")
    ys, xs = np.where(mask > 0)
    if np.ptp(xs) < 1 or np.ptp(ys) < 1:
        raise ValueError("Region is not two-dimensional")
    extractor = extractor or make_extractor()
    result = extractor.execute(
        sitk.GetImageFromArray(np.asarray(image, dtype=np.float32)),
        sitk.GetImageFromArray(mask),
    )
    available = {
        key: float(value)
        for key, value in result.items()
        if not key.startswith("diagnostics")
    }
    missing = sorted(set(REGISTERED_FEATURE_NAMES) - set(available))
    if missing:
        raise ValueError(f"PyRadiomics did not return registered features: {missing[:5]}")
    model = {name: available[name] for name in REGISTERED_FEATURE_NAMES}
    helpers = {
        "Median": available["original_firstorder_Median"],
        "InterquartileRange": available["original_firstorder_InterquartileRange"],
        "Entropy": available["original_firstorder_Entropy"],
    }
    values = np.asarray(list(model.values()) + list(helpers.values()), dtype=float)
    if len(model) != 50 or not np.isfinite(values).all():
        raise ValueError("Unexpected or non-finite registered radiomics result")
    return model, helpers


def read_status_table(annotation_root: Path, task_ids: set[str]) -> pd.DataFrame:
    path = annotation_root / "annotation_status.csv"
    if not path.exists():
        raise ValueError(f"Missing required status file: {path}")
    table = pd.read_csv(path, dtype={"task_id": str}).fillna("")
    required = {"task_id"} | {f"{name}_status" for name in VALIDATOR.STATUS_CLASSES}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Status file lacks columns: {sorted(missing)}")
    if table.task_id.duplicated().any():
        raise ValueError("Status file contains duplicate task ids")
    observed = set(table.task_id.astype(str))
    if observed != task_ids:
        raise ValueError(
            f"Status task ids differ from packet: missing={sorted(task_ids-observed)}, "
            f"extra={sorted(observed-task_ids)}"
        )
    for class_name in VALIDATOR.STATUS_CLASSES:
        column = f"{class_name}_status"
        values = table[column].astype(str).str.strip().str.lower()
        invalid = ~values.isin(VALIDATOR.VALID_STATUS)
        if invalid.any():
            bad = table.loc[invalid, ["task_id", column]].to_dict("records")
            raise ValueError(f"Every {column} must contain an allowed value: {bad[:5]}")
        if values.eq("not_annotated").any():
            unfinished = table.loc[values.eq("not_annotated"), "task_id"].astype(str).tolist()
            raise ValueError(f"Completed submission still contains not_annotated in {column}: {unfinished[:5]}")
        table[column] = values
    if "outer_roi_status" in table.columns:
        outer_status = table.outer_roi_status.astype(str).str.strip().str.lower()
        blocked = outer_status.isin({"requires_correction", "unusable"})
        if blocked.any():
            tasks = table.loc[blocked, ["task_id", "outer_roi_status"]].to_dict("records")
            raise ValueError(f"Whole-kidney ROI requires resolution before feature analysis: {tasks[:5]}")
    return table.set_index("task_id")


def read_required_mask(annotation_root: Path, task_id: str, class_name: str) -> np.ndarray:
    mask, errors = VALIDATOR.read_mask(annotation_root / f"{task_id}_{class_name}.png")
    if mask is None or errors:
        raise ValueError(f"Invalid mask {task_id}/{class_name}: {errors}")
    return mask


def extract_task_features(
    annotation_root: Path,
    rater: str,
    tasks: pd.DataFrame,
    statuses: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    extractor = make_extractor()
    feature_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for position, task in enumerate(tasks.itertuples(index=False), 1):
        task_id = str(task.task_id)
        image = cv2.imread(str(PACKET / "images" / f"{task_id}.png"), cv2.IMREAD_GRAYSCALE)
        outer = cv2.imread(str(PACKET / "outer_roi_masks" / f"{task_id}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or outer is None:
            raise RuntimeError(f"Packet assets unavailable for {task_id}")
        outer = (outer > 0).astype(np.uint8)
        normalized = percentile_normalize(image, outer)
        artifact = read_required_mask(annotation_root, task_id, "artifact")
        artifact_fraction = float(np.logical_and(artifact > 0, outer > 0).sum() / max(1, outer.sum()))
        row: dict[str, object] = {
            "rater": rater,
            "task_id": task_id,
            "view": str(task.view),
            "qc::artifact_area_fraction": artifact_fraction,
        }
        helpers: dict[str, dict[str, float]] = {}
        for region in ANATOMY_REGIONS:
            status = str(statuses.loc[task_id, f"{region}_status"])
            mask = read_required_mask(annotation_root, task_id, region)
            clean, excluded_fraction = artifact_excluded(mask, artifact)
            row[f"qc::{region}_artifact_exclusion_fraction"] = excluded_fraction
            row[f"{region}::area_fraction"] = float(mask.sum() / max(1, outer.sum()))
            issue = ""
            if status == "present":
                try:
                    values, region_helpers = extract_region_features(normalized, clean, extractor)
                    helpers[region] = region_helpers
                    row.update({f"{region}::{name}": value for name, value in values.items()})
                except ValueError as error:
                    issue = str(error)
            else:
                issue = f"status:{status}"
            audit_rows.append(
                {
                    "rater": rater,
                    "task_id": task_id,
                    "region": region,
                    "status": status,
                    "original_pixels": int(mask.sum()),
                    "artifact_excluded_pixels": int(clean.sum()),
                    "artifact_exclusion_fraction": excluded_fraction,
                    "feature_issue": issue,
                }
            )
        if "cortex" in helpers and "cec" in helpers:
            c, e = helpers["cortex"], helpers["cec"]
            row["ratio::cortex_to_cec_median_ratio"] = safe_ratio(c["Median"], e["Median"])
            row["ratio::cortex_minus_cec_entropy"] = c["Entropy"] - e["Entropy"]
            row["ratio::cortex_to_cec_interquartile_range_ratio"] = safe_ratio(
                c["InterquartileRange"], e["InterquartileRange"]
            )
        feature_rows.append(row)
        if position % 10 == 0 or position == len(tasks):
            print(f"{rater}: extracted {position}/{len(tasks)} blinded tasks", flush=True)
    return pd.DataFrame(feature_rows), pd.DataFrame(audit_rows)


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-8:
        return float("nan")
    return float(numerator / denominator)


def dispersion(values: np.ndarray, feature_name: str) -> tuple[str, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Dispersion input must have two rater columns")
    signed = any(token in feature_name for token in SIGNED_FEATURE_TOKENS) or bool((values < 0).any())
    if signed:
        denominator = np.abs(values[:, 0]) + np.abs(values[:, 1])
        differences = 200.0 * np.abs(values[:, 0] - values[:, 1]) / np.maximum(denominator, 1e-8)
        return "median_symmetric_percent_difference", float(np.median(differences))
    means = values.mean(axis=1)
    standard_deviations = values.std(axis=1, ddof=0)
    usable = np.abs(means) > 1e-8
    if not usable.any():
        return "median_within_image_cv_percent", float("nan")
    cv = 100.0 * standard_deviations[usable] / np.abs(means[usable])
    return "median_within_image_cv_percent", float(np.median(cv))


def reliability_table(features: pd.DataFrame, core_task_ids: set[str]) -> pd.DataFrame:
    core = features.loc[features.task_id.isin(core_task_ids)].copy()
    excluded = {"rater", "task_id", "view"}
    feature_names = sorted(name for name in core.columns if name not in excluded and not name.startswith("qc::"))
    rows: list[dict[str, object]] = []
    for name in feature_names:
        pair = core.pivot(index="task_id", columns="rater", values=name).dropna()
        if pair.shape[1] != 2:
            values = np.empty((0, 2), dtype=float)
        else:
            values = pair.to_numpy(dtype=float)
        icc = icc_2_1(values) if values.shape[0] >= 2 else float("nan")
        metric, amount = dispersion(values, name) if values.shape[0] else ("not_estimable", float("nan"))
        reliable = bool(
            values.shape[0] >= MINIMUM_PAIRED_IMAGES
            and np.isfinite(icc)
            and icc >= ICC_MINIMUM
            and np.isfinite(amount)
            and amount < DISPERSION_MAXIMUM_PERCENT
        )
        rows.append(
            {
                "feature": name,
                "paired_core_images": int(values.shape[0]),
                "icc_2_1": icc,
                "dispersion_metric": metric,
                "dispersion_percent": amount,
                "passes_locked_reliability_gate": reliable,
            }
        )
    return pd.DataFrame(rows)


def summarize_mask_agreement(table: pd.DataFrame) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for class_name, group in table.groupby("class_name"):
        values = group.dice.dropna().astype(float)
        summaries.append(
            {
                "class_name": str(class_name),
                "estimable_tasks": int(values.shape[0]),
                "mean_dice": None if values.empty else float(values.mean()),
                "median_dice": None if values.empty else float(values.median()),
            }
        )
    return summaries


def evaluate(annotation_a: Path, annotation_b: Path, output: Path) -> None:
    annotation_a = annotation_a.resolve()
    annotation_b = annotation_b.resolve()
    if annotation_a == annotation_b:
        raise ValueError("The two independent rater folders must be different")
    output.mkdir(parents=True, exist_ok=True)
    tasks = VALIDATOR.read_tasks(PACKET)
    task_ids = set(tasks.task_id.astype(str))
    statuses_a = read_status_table(annotation_a, task_ids)
    statuses_b = read_status_table(annotation_b, task_ids)

    qc_a = VALIDATOR.validate_submission(PACKET, annotation_a, output / "qc_rater_a")
    qc_b = VALIDATOR.validate_submission(PACKET, annotation_b, output / "qc_rater_b")
    errors_a = int(qc_a.errors.fillna("").str.len().gt(0).sum())
    errors_b = int(qc_b.errors.fillna("").str.len().gt(0).sum())
    if errors_a or errors_b:
        raise ValueError(f"Structural QC failed: rater_a_error_rows={errors_a}, rater_b_error_rows={errors_b}")

    interrater = VALIDATOR.compare_submissions(PACKET, annotation_a, annotation_b, output / "agreement")
    repeat_a = VALIDATOR.compare_hidden_repeats(PACKET, annotation_a, PRIVATE_KEY, output / "agreement_rater_a")
    repeat_b = VALIDATOR.compare_hidden_repeats(PACKET, annotation_b, PRIVATE_KEY, output / "agreement_rater_b")

    features_a, audit_a = extract_task_features(annotation_a, "rater_a", tasks, statuses_a)
    features_b, audit_b = extract_task_features(annotation_b, "rater_b", tasks, statuses_b)
    features = pd.concat([features_a, features_b], ignore_index=True, sort=False)
    audit = pd.concat([audit_a, audit_b], ignore_index=True)
    features.to_csv(output / "pilot_features_by_blinded_task.csv", index=False)
    audit.to_csv(output / "feature_extraction_audit.csv", index=False)

    key = pd.read_csv(PRIVATE_KEY, dtype=str).fillna("")
    core_task_ids = set(key.loc[key.task_type.eq("core"), "task_id"].astype(str))
    if len(core_task_ids) != 32:
        raise RuntimeError(f"Expected 32 unique core tasks, found {len(core_task_ids)}")
    reliability = reliability_table(features, core_task_ids)
    reliability.to_csv(output / "feature_reliability_core_images.csv", index=False)

    manifest = pd.read_csv(COHORT / "image_manifest.csv", dtype={"patient_id": str})
    longitudinal = manifest.loc[manifest.view.isin(["LL", "RL"]), ["patient_id", "image_id"]]
    reliable_count = int(reliability.passes_locked_reliability_gate.sum())
    readiness = {
        "analysis_scope": "Label-free 32-unique-image pilot with eight hidden repeats.",
        "diagnosis_labels_read": False,
        "pilot_core_images": len(core_task_ids),
        "pilot_hidden_repeats": int(key.task_type.eq("hidden_repeat").sum()),
        "interrater_mask_agreement": summarize_mask_agreement(interrater),
        "intrarater_repeat_rows": {
            "rater_a": int(repeat_a.shape[0]),
            "rater_b": int(repeat_b.shape[0]),
        },
        "registered_predictor_features_evaluated": int(reliability.shape[0]),
        "features_passing_locked_reliability_gate": reliable_count,
        "full_cohort_longitudinal_images": int(longitudinal.shape[0]),
        "full_cohort_patients_with_longitudinal_images": int(longitudinal.patient_id.nunique()),
        "full_cohort_classifier_ready": False,
        "blocking_reason": (
            "The pilot has only 32 unique images. A 98-patient classifier requires anatomy/artifact masks "
            "for the 190 longitudinal images covering all 98 patients, or predictions from a separately "
            "locked compartment segmenter that passes held-out segmentation validation."
        ),
        "prohibited_next_step": "Do not fit or report a disease classifier using only the 32-image pilot.",
        "submission_hashes": {
            "rater_a": aggregate_submission_hash(annotation_a),
            "rater_b": aggregate_submission_hash(annotation_b),
        },
        "protocol_hashes": {
            "protocol_locked_json": sha256(PROTOCOL / "protocol_locked.json"),
            "candidate_feature_registry_csv": sha256(PROTOCOL / "candidate_feature_registry.csv"),
        },
    }
    write_json(output / "readiness.json", readiness)
    print(json.dumps(readiness, ensure_ascii=False, indent=2), flush=True)


def self_test() -> None:
    height, width = 96, 128
    yy, xx = np.mgrid[:height, :width]
    outer = ((((xx - 64) / 48) ** 2 + ((yy - 48) / 31) ** 2) <= 1).astype(np.uint8)
    cec = ((((xx - 64) / 18) ** 2 + ((yy - 48) / 10) ** 2) <= 1).astype(np.uint8)
    medulla = np.zeros_like(outer)
    for center_x, center_y in ((45, 38), (45, 58), (83, 38), (83, 58)):
        medulla |= ((((xx - center_x) / 7) ** 2 + ((yy - center_y) / 5) ** 2) <= 1).astype(np.uint8)
    medulla &= outer
    medulla &= 1 - cec
    cortex = outer.copy()
    cortex[cec > 0] = 0
    cortex[medulla > 0] = 0
    artifact = np.zeros_like(outer)
    cv2.line(artifact, (30, 48), (98, 48), 1, 1)
    artifact &= outer
    image = np.clip(35 + 0.8 * xx + 0.5 * yy + 17 * np.sin(xx / 5), 0, 255).astype(np.uint8)
    normalized = percentile_normalize(image, outer)
    clean, excluded = artifact_excluded(cortex, artifact)
    values, helpers = extract_region_features(normalized, clean)
    assert len(values) == 50
    assert set(helpers) == {"Median", "InterquartileRange", "Entropy"}
    assert 0 < excluded < 1

    synthetic = np.column_stack([np.arange(1, 25), np.arange(1, 25) * 1.01])
    assert icc_2_1(synthetic) > 0.99
    metric, amount = dispersion(synthetic, "cortex::original_firstorder_Entropy")
    assert metric == "median_within_image_cv_percent" and amount < 1

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        packet = root / "packet"
        annotations = root / "annotations"
        output = root / "output"
        (packet / "images").mkdir(parents=True)
        (packet / "outer_roi_masks").mkdir()
        annotations.mkdir()
        pd.DataFrame([{"task_id": "T001", "display_order": 1, "view": "LL"}]).to_csv(
            packet / "annotation_tasks_blinded.csv", index=False
        )
        cv2.imwrite(str(packet / "images" / "T001.png"), image)
        cv2.imwrite(str(packet / "outer_roi_masks" / "T001.png"), outer * 255)
        masks = {"cec": cec, "medulla": medulla, "cortex": cortex, "artifact": artifact, "uncertain": np.zeros_like(outer)}
        for name, mask in masks.items():
            cv2.imwrite(str(annotations / f"T001_{name}.png"), mask * 255)
        with (annotations / "annotation_status.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["task_id", "cec_status", "medulla_status", "cortex_status", "artifact_status"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "task_id": "T001",
                    "cec_status": "present",
                    "medulla_status": "present",
                    "cortex_status": "present",
                    "artifact_status": "present",
                }
            )
        table = VALIDATOR.validate_submission(packet, annotations, output)
        assert not table.errors.fillna("").str.contains("class_overlap").any()
    print("Self-test passed: artifact overlap allowed; 50 registered radiomics features extracted; ICC/CV operational.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test", help="Run synthetic end-to-end checks without patient data.")
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Validate two returned blinded submissions and run label-free pilot reliability analysis.",
    )
    evaluate_parser.add_argument("--annotations-a", type=Path, required=True)
    evaluate_parser.add_argument("--annotations-b", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "evaluate":
        evaluate(args.annotations_a, args.annotations_b, args.output)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
