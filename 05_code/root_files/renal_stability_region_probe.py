"""Training-only stability screening and central-longitudinal ROI probe.

This is a separate, retrospective exploratory batch.  It retains the locked
98-patient CYCH cohort and only uses the existing whole-kidney masks.  The
central region is a geometric long-axis surrogate, not a cortex/parenchyma
segmentation and must never be reported as one.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import warnings
import zipfile

import cv2
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import renal_dinov3_native_probe as N
import renal_random_validation as R
import renal_voting_validation as V

ROOT = R.ROOT
BASE = R.OUT
PRIOR = ROOT / "CYCH_interpretable_features_20260913"
OUT = ROOT / "CYCH_stability_region_probe_20260913_r2"

REGIONS = ("whole_kidney", "central_longitudinal")
VARIANTS = ("baseline", "contrast_low", "contrast_high", "erode1", "dilate1")
CONTRAST_VARIANTS = ("baseline", "contrast_low", "contrast_high")
CONTOUR_VARIANTS = ("baseline", "erode1", "dilate1")
ICC_THRESHOLD = 0.90
EROSION_KERNEL = 13
SMALL_EROSION_KERNEL = 3
LONGITUDINAL_VIEWS = frozenset(("LL", "RL"))
RADIOMICS_FEATURES = {
    "firstorder": ["Entropy", "Variance"],
    "glrlm": [
        "RunLengthNonUniformityNormalized",
        "GrayLevelNonUniformityNormalized",
        "RunEntropy",
    ],
}
GROUPS = ("WHOLE-I3", "WHOLE-STABLE", "CENTRAL-GEOMETRIC", "CENTRAL-STABLE")

REFERENCES = [
    {
        "id": "soleymani2022",
        "citation": "Soleymani Y, et al. Reproducibility assessment of radiomics features in various ultrasound scan settings and different scanner vendors. J Med Imaging Radiat Sci. 2022;53:664-671. doi:10.1016/j.jmir.2022.09.018.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/36266173/",
        "scope": "Ultrasound phantom reproducibility across settings/vendors; motivates screening, not a CYCH outcome result.",
    },
    {
        "id": "zwanenburg2019",
        "citation": "Zwanenburg A, et al. Assessing robustness of radiomic features by image perturbation. Sci Rep. 2019;9:614.",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6345842/",
        "scope": "Image and contour perturbation framework from CT; adapted here to B-mode ultrasound as a predefined robustness screen, not an ultrasound validation study.",
    },
    {
        "id": "chen2025",
        "citation": "Chen Z, Wang Y, Wu C. Renal fibrosis assessment in chronic kidney disease: Exploring varied kidney regions with ultrasound-based radiomics analysis. Sci Prog. 2025;108:e399597. doi:10.1177/00368504251399597.",
        "url": "https://journals.sagepub.com/doi/10.1177/00368504251399597",
        "scope": "Uses a manually drawn central 50% of renal parenchyma in a fibrosis cohort. The present geometric central whole-kidney surrogate is not a replication and has no parenchymal truth mask.",
    },
    {
        "id": "bandara2022",
        "citation": "Bandara MS, et al. Ultrasound Based Radiomics Features of Chronic Kidney Disease. Acad Radiol. 2022;29:229-235. doi:10.1016/j.acra.2021.01.006.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/33589307/",
        "scope": "Motivates the inherited directional wavelet/GLRLM feature family; not an inflammation replication.",
    },
    {
        "id": "pyradiomics",
        "citation": "van Griethuysen JJM, et al. Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Res. 2017;77:e104-e107. doi:10.1158/0008-5472.CAN-17-0339.",
        "url": "https://pyradiomics.readthedocs.io/en/v3.0.1/features.html",
        "scope": "Feature implementation and definitions.",
    },
]


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name, dtype={"patient_id": str})


def is_longitudinal(view: str) -> bool:
    return str(view) in LONGITUDINAL_VIEWS


def require_lock() -> None:
    lock = json.loads((OUT / "protocol_locked.json").read_text(encoding="utf-8"))
    if lock["protocol_sha256"] != R.sha256(OUT / "protocol.json"):
        raise RuntimeError("Protocol hash mismatch")
    if lock["source_sha256"] != R.sha256(Path(__file__)):
        raise RuntimeError("Source changed after the protocol was locked; use a new batch directory")


def principal_axis(mask: np.ndarray) -> tuple[float, float, float, float]:
    """Return centroid and longest covariance axis for a nonempty binary ROI."""
    yy, xx = np.nonzero(mask)
    if len(xx) < 3:
        raise ValueError("A nonempty two-dimensional mask is required")
    cx, cy = float(xx.mean()), float(yy.mean())
    values, vectors = np.linalg.eigh(np.cov(np.stack((xx, yy))))
    vx, vy = vectors[:, int(np.argmax(values))]
    return cx, cy, float(vx), float(vy)


def central_half_mask(mask: np.ndarray) -> np.ndarray:
    """Middle half of a mask along its own principal long axis.

    This encodes the pole-to-pole 25%-75% rule as a reproducible geometric
    surrogate.  It does not distinguish parenchyma from collecting system.
    """
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    cx, cy, vx, vy = principal_axis(mask)
    yy, xx = np.nonzero(mask)
    projection = (xx - cx) * vx + (yy - cy) * vy
    lo, hi = float(projection.min()), float(projection.max())
    keep = (projection >= lo + .25 * (hi - lo)) & (projection <= lo + .75 * (hi - lo))
    result = np.zeros_like(mask)
    result[yy[keep], xx[keep]] = 1
    if int(result.sum()) < 64:
        raise ValueError("Central longitudinal surrogate has fewer than 64 pixels")
    return result


def percentile_normalized(raw: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    """Exact P1-P99 mapping used by the inherited ROI-normalized input."""
    raw = np.asarray(raw, dtype=np.uint8)
    mask = np.asarray(base_mask, dtype=bool)
    lo, hi = np.percentile(raw[mask], [1, 99])
    if hi <= lo:
        return raw.copy()
    return np.rint(np.clip((raw.astype(float) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def perturb_mask(base_mask: np.ndarray, variant: str) -> np.ndarray:
    mask = (np.asarray(base_mask) > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if variant in ("baseline", "contrast_low", "contrast_high"):
        result = mask
    elif variant == "erode1":
        result = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    elif variant == "dilate1":
        result = cv2.dilate(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        raise ValueError(variant)
    if int(result.sum()) < 64:
        raise ValueError(f"{variant} produced an unusably small mask")
    return result.astype(np.uint8)


def perturb_intensity(normalized: np.ndarray, variant: str) -> np.ndarray:
    image = np.asarray(normalized, dtype=np.uint8)
    if variant == "contrast_low":
        factor = .95
    elif variant == "contrast_high":
        factor = 1.05
    elif variant in ("baseline", "erode1", "dilate1"):
        return image.copy()
    else:
        raise ValueError(variant)
    return np.rint(np.clip(127.5 + factor * (image.astype(float) - 127.5), 0, 255)).astype(np.uint8)


def centered_canvas(image: np.ndarray, full_mask: np.ndarray, selected_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the inherited no-rotation I3 padding before wavelet extraction."""
    h, w = image.shape
    side = int(np.ceil(np.hypot(h, w))) + 8
    padded_image = np.zeros((side, side), np.uint8)
    padded_full = np.zeros((side, side), np.uint8)
    padded_selected = np.zeros((side, side), np.uint8)
    y, x = (side - h) // 2, (side - w) // 2
    padded_image[y : y + h, x : x + w] = image
    padded_full[y : y + h, x : x + w] = full_mask
    padded_selected[y : y + h, x : x + w] = selected_mask
    return padded_image * padded_full, padded_selected


def support_mask(full_mask: np.ndarray, region_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep the wavelet support inside the outer ROI, matching the I3 rule."""
    for kernel_size in (EROSION_KERNEL, SMALL_EROSION_KERNEL):
        support = cv2.erode(
            full_mask,
            np.ones((kernel_size, kernel_size), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        selected = (support & region_mask).astype(np.uint8)
        if int(selected.sum()) >= 64:
            return selected, kernel_size
    raise ValueError("Insufficient intrarenal support after the documented erosion fallbacks")


def radiomics_vector(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
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
    extractor.enableFeaturesByName(**RADIOMICS_FEATURES)
    result = extractor.execute(
        sitk.GetImageFromArray(np.asarray(image, dtype=np.float32)),
        sitk.GetImageFromArray(np.asarray(mask, dtype=np.uint8)),
    )
    values = {key: float(value) for key, value in result.items() if not key.startswith("diagnostics")}
    if len(values) != 25 or not np.isfinite(list(values.values())).all():
        raise ValueError("Expected exactly 25 finite inherited directional features")
    return values


def extract_one_image(item: tuple[str, str, str, str]) -> tuple[list[dict], list[dict]]:
    """Extract all predefined perturbations for one image in a worker process."""
    image_id, raw_path, mask_path, view = item
    raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    base_mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    if raw is None or base_mask is None or raw.shape != base_mask.shape:
        raise ValueError(f"Missing or incompatible source image/mask for {image_id}")
    normalized = percentile_normalized(raw, base_mask)
    records: list[dict] = []
    audit: list[dict] = []
    for variant in VARIANTS:
        full_mask = perturb_mask(base_mask, variant)
        pixel = perturb_intensity(normalized, variant)
        regions = {"whole_kidney": full_mask}
        if is_longitudinal(view):
            regions["central_longitudinal"] = central_half_mask(full_mask)
        for region, region_mask in regions.items():
            selected, erosion = support_mask(full_mask, region_mask)
            image, selected = centered_canvas(pixel, full_mask, selected)
            values = radiomics_vector(image, selected)
            records.append(dict(image_id=image_id, variant=variant, region=region, **values))
            audit.append(
                dict(
                    image_id=image_id,
                    variant=variant,
                    region=region,
                    base_mask_pixels=int(base_mask.sum()),
                    perturbed_mask_pixels=int(full_mask.sum()),
                    region_mask_pixels=int(region_mask.sum()),
                    selected_pixels=int(selected.sum()),
                    erosion_kernel=erosion,
                )
            )
    return records, audit


def aggregate_patient_features(
    image_features: pd.DataFrame, image_manifest: pd.DataFrame, patients: pd.DataFrame, region: str, variant: str
) -> tuple[np.ndarray, list[str]]:
    frame = image_features.loc[
        image_features.region.eq(region) & image_features.variant.eq(variant)
    ].copy()
    feature_names = [name for name in frame.columns if name not in ("image_id", "variant", "region")]
    expected = image_manifest.loc[
        image_manifest.view.map(is_longitudinal) if region == "central_longitudinal" else np.ones(len(image_manifest), dtype=bool),
        ["image_id", "patient_id"],
    ]
    joined = expected.merge(frame[["image_id", *feature_names]], on="image_id", how="left", validate="one_to_one")
    if joined[feature_names].isna().any().any():
        raise RuntimeError(f"Missing {region}/{variant} image features")
    vectors = []
    for patient_id in patients.patient_id:
        values = joined.loc[joined.patient_id.eq(patient_id), feature_names].to_numpy(np.float64)
        if len(values) == 0:
            raise RuntimeError(f"No eligible {region} images for patient {patient_id}")
        vectors.append(np.concatenate((values.mean(0), values.std(0))))
    names = [f"image_mean:{name}" for name in feature_names] + [f"image_std:{name}" for name in feature_names]
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (len(patients), len(names)) or not np.isfinite(matrix).all():
        raise RuntimeError("Invalid patient-level feature matrix")
    return matrix, names


def icc_2_1(values: np.ndarray) -> float:
    """Two-way random, absolute-agreement single-measure ICC(2,1)."""
    values = np.asarray(values, dtype=float)
    n, k = values.shape
    if n < 2 or k < 2 or not np.isfinite(values).all():
        return float("nan")
    grand = values.mean()
    row_means = values.mean(axis=1)
    col_means = values.mean(axis=0)
    ms_rows = k * np.square(row_means - grand).sum() / (n - 1)
    ms_cols = n * np.square(col_means - grand).sum() / (k - 1)
    residual = values - row_means[:, None] - col_means[None, :] + grand
    ms_error = np.square(residual).sum() / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")
    return float((ms_rows - ms_error) / denominator)


def stability_screen(
    arrays: dict[str, np.ndarray], feature_names: list[str], train_indices: np.ndarray, region: str, seed: int
) -> tuple[np.ndarray, pd.DataFrame]:
    rows = []
    for j, name in enumerate(feature_names):
        contrast = icc_2_1(np.column_stack([arrays[variant][train_indices, j] for variant in CONTRAST_VARIANTS]))
        contour = icc_2_1(np.column_stack([arrays[variant][train_indices, j] for variant in CONTOUR_VARIANTS]))
        stable = bool(np.isfinite(contrast) and np.isfinite(contour) and contrast >= ICC_THRESHOLD and contour >= ICC_THRESHOLD)
        rows.append(
            dict(
                seed=seed,
                region=region,
                feature_index=j,
                feature=name,
                contrast_icc_2_1=contrast,
                contour_icc_2_1=contour,
                minimum_icc_2_1=float(min(contrast, contour)) if np.isfinite(contrast) and np.isfinite(contour) else np.nan,
                stable=stable,
            )
        )
    frame = pd.DataFrame(rows)
    return frame.stable.to_numpy(bool), frame


def data_quality_audit(image_manifest: pd.DataFrame, patients: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    records = []
    for row in image_manifest.itertuples():
        raw_path = BASE / "native" / f"{row.image_id}.png"
        mask_path = BASE / "masks" / f"{row.image_id}.png"
        raw = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw is None or mask is None or raw.shape != mask.shape:
            raise RuntimeError(f"Missing or mismatched native/mask pair: {row.image_id}")
        mask_binary = (mask > 0).astype(np.uint8)
        if int(mask_binary.sum()) != int(row.mask_pixels):
            raise RuntimeError(f"Mask-pixel mismatch: {row.image_id}")
        if R.sha256(mask_path) != row.mask_sha256:
            raise RuntimeError(f"Mask hash mismatch: {row.image_id}")
        central_pixels = np.nan
        if is_longitudinal(row.view):
            central_pixels = int(central_half_mask(mask_binary).sum())
        records.append(
            dict(
                image_id=row.image_id,
                patient_alias=row.patient_alias,
                view=row.view,
                width=int(raw.shape[1]),
                height=int(raw.shape[0]),
                mask_pixels=int(mask_binary.sum()),
                central_longitudinal_pixels=central_pixels,
                mask_source=row.mask_source,
                raw_path=str(raw_path),
                mask_path=str(mask_path),
                raw_exists=bool(raw_path.exists()),
                mask_hash_matches=True,
            )
        )
    audit = pd.DataFrame(records)
    per_patient = image_manifest.groupby("patient_id").agg(
        label=("label", "nunique"),
        all_images=("image_id", "size"),
        long_images=("view", lambda values: int(sum(is_longitudinal(value) for value in values))),
    )
    if len(image_manifest) != 349 or len(patients) != 98 or image_manifest.image_id.nunique() != 349:
        raise RuntimeError("Unexpected cohort denominator")
    if image_manifest.groupby("patient_id").label.nunique().max() != 1 or patients.groupby("patient_id").label.nunique().max() != 1:
        raise RuntimeError("Inconsistent patient labels")
    if set(image_manifest.patient_id) != set(patients.patient_id):
        raise RuntimeError("Patient manifest mismatch")
    if int(per_patient.long_images.min()) < 1:
        raise RuntimeError("A patient lacks a longitudinal image required for the central-region comparison")
    if image_manifest.source_sha256.duplicated().any():
        raise RuntimeError("Exact duplicated source image requires reconciliation before modeling")
    summary = dict(
        intended_grain="one patient score; image-level features are aggregated by patient",
        images=int(len(image_manifest)),
        patients=int(len(patients)),
        positive=int(patients.label.sum()),
        negative=int((1 - patients.label).sum()),
        exact_duplicate_source_images=int(image_manifest.source_sha256.duplicated().sum()),
        views=image_manifest.view.value_counts().sort_index().to_dict(),
        images_by_mask_source=image_manifest.mask_source.value_counts().to_dict(),
        patient_longitudinal_image_minimum=int(per_patient.long_images.min()),
        patient_longitudinal_image_maximum=int(per_patient.long_images.max()),
        central_mask_retained_fraction=dict(
            min=float((audit.central_longitudinal_pixels / audit.mask_pixels).dropna().min()),
            median=float((audit.central_longitudinal_pixels / audit.mask_pixels).dropna().median()),
            max=float((audit.central_longitudinal_pixels / audit.mask_pixels).dropna().max()),
        ),
        checks_passed=[
            "349 unique image IDs and 98 patient IDs",
            "one label per patient",
            "all native image/mask shapes match",
            "all mask hashes and pixel counts match the locked manifest",
            "no exact source-image duplicate",
            "all patients have at least one longitudinal view",
        ],
        limitations=[
            "Whole-kidney masks are inherited; original segmentation-training overlap remains unknown.",
            "Central longitudinal masks are a geometry-only surrogate, not renal parenchyma/cortex/medulla labels.",
            "Calipers and other burned-in annotations have not been clinically annotated or removed.",
        ],
    )
    return audit, summary


def prepare() -> None:
    OUT.mkdir(exist_ok=True)
    for directory in ("features", "models", "masks/central_longitudinal", "qa", "source", "figures_600dpi"):
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv"):
        target = OUT / name
        source = BASE / name
        if target.exists() and R.sha256(target) != R.sha256(source):
            raise RuntimeError(f"Cohort drift in {name}")
        shutil.copy2(source, target)
    image_manifest, patients, split = read("image_manifest.csv"), read("patients.csv"), read("random_patient_assignments.csv")
    quality, quality_summary = data_quality_audit(image_manifest, patients)
    quality.to_csv(OUT / "image_geometry_audit.csv", index=False)
    R.write_json(OUT / "data_quality_summary.json", quality_summary)
    for row in image_manifest.itertuples():
        if not is_longitudinal(row.view):
            continue
        mask = (cv2.imread(str(BASE / "masks" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        target = OUT / "masks" / "central_longitudinal" / f"{row.image_id}.png"
        central = central_half_mask(mask)
        if not cv2.imwrite(str(target), central * 255):
            raise RuntimeError(f"Unable to write central mask: {row.image_id}")
    for seed in R.SEEDS:
        N.indices_for(patients, split, seed)
    protocol = dict(
        design="Further retrospective exploratory patient-level random-sampling experiment on the same previously explored CYCH cohort. No k-fold, external, untouched test, or clinical validation claim.",
        cohort=dict(patients=98, images=349, positive=52, negative=46),
        seeds=R.SEEDS,
        roles=dict(train=58, development=20, exploratory_test=20),
        inputs=dict(
            base_run=str(BASE),
            whole_kidney_masks="Existing locked mask input only; 228 YOLO and 121 manual-polygon fallback according to inherited manifest.",
            normalized_image="Within-original-mask P1-P99 grayscale mapping, identical to the inherited I3 baseline for baseline features.",
            central_region="Longitudinal images only. Principal-axis pole-to-pole 25%-75% geometric central half of the inherited whole-kidney mask. It is not a parenchyma, cortex, medulla, collecting-system, or lesion mask.",
        ),
        variants=dict(
            baseline="P1-P99 image and inherited mask",
            contrast_low="fixed 0.95 contrast around gray level 127.5, inherited mask",
            contrast_high="fixed 1.05 contrast around gray level 127.5, inherited mask",
            erode1="one-pixel elliptical erosion of inherited mask; normalized raw crop intensity unchanged",
            dilate1="one-pixel elliptical dilation of inherited mask; normalized raw crop intensity unchanged",
        ),
        radiomics=dict(
            image_types="Original plus coif1 level-1 wavelet",
            features=RADIOMICS_FEATURES,
            bin_width=16,
            distances=[1, 2],
            support=f"{EROSION_KERNEL}x{EROSION_KERNEL} erosion of perturbed outer mask; {SMALL_EROSION_KERNEL}x{SMALL_EROSION_KERNEL} fallback only if fewer than 64 selected pixels",
            image_features=25,
            patient_features=50,
        ),
        stability=dict(
            unit="patient-level mean and standard-deviation feature aggregate over eligible views",
            selection="For each seed and region, calculate ICC(2,1), two-way random absolute-agreement single measure, only over the 58 training patients. Retain a feature only when contrast ICC and contour ICC are both >= 0.90.",
            contrast_variants=list(CONTRAST_VARIANTS),
            contour_variants=list(CONTOUR_VARIANTS),
            threshold=ICC_THRESHOLD,
            labels="No labels enter the stability calculation. Labels enter only the subsequently fitted fixed logistic-regression head.",
        ),
        groups=dict(
            **{
                "WHOLE-I3": "All 50 inherited directional features on all image views; matched baseline.",
                "WHOLE-STABLE": "Training-only stability-filtered subset of WHOLE-I3.",
                "CENTRAL-GEOMETRIC": "All 50 features from longitudinal central geometry surrogate only; anatomical feasibility control, not parenchymal radiomics.",
                "CENTRAL-STABLE": "Training-only stability-filtered central-geometry subset.",
            }
        ),
        head="Fixed train-only median imputer, variance threshold, standardizer, L2 logistic regression C=0.1 / liblinear / balanced / max_iter=2000. No tuning, GAN, new encoder, learned vote weights, or clinical covariates.",
        decision="This batch does not add a candidate to the existing RAD+DenseNet+I3 vote. A stability candidate can only be considered for a later separately locked vote if its development metrics meet the documented candidate gate; test results never select features or a vote.",
        limitations=[
            "All 98 patients were previously explored; repeated random roles are not independent confirmation.",
            "The geometric central mask is not the manual renal-parenchyma ROI in Chen et al. and is not a clinical segmentation output.",
            "P1-P99 normalization, fixed perturbations and ICC screening cannot establish scanner-vendor invariance or pathology specificity.",
            "The result cannot establish scar, swelling, focal inflammation or clinical risk from B-mode single frames.",
        ],
        cohort_hashes={name: R.sha256(OUT / name) for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv")},
        prior_i3_patient_features=dict(path=str(PRIOR / "features" / "I3-direction-original.npy"), sha256=R.sha256(PRIOR / "features" / "I3-direction-original.npy")),
        references=REFERENCES,
    )
    protocol = json.loads(json.dumps(protocol))
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError("Protocol has changed; use a different batch directory")
    R.write_json(path, protocol)
    lock = OUT / "protocol_locked.json"
    if not lock.exists():
        R.write_json(
            lock,
            dict(
                utc=datetime.now(timezone.utc).isoformat(),
                protocol_sha256=R.sha256(path),
                source_sha256=R.sha256(Path(__file__)),
            ),
        )
    R.write_json(OUT / "references.json", REFERENCES)
    shutil.copy2(Path(__file__), OUT / "source" / Path(__file__).name)
    test_path = ROOT / "test_renal_stability_region_probe.py"
    if test_path.exists():
        shutil.copy2(test_path, OUT / "source" / test_path.name)
    print("Protocol locked; data-quality audit and 190 longitudinal central masks are ready", flush=True)


def extract() -> None:
    require_lock()
    image_manifest, patients = read("image_manifest.csv"), read("patients.csv")
    items = [
        (
            row.image_id,
            str(BASE / "native" / f"{row.image_id}.png"),
            str(BASE / "masks" / f"{row.image_id}.png"),
            row.view,
        )
        for row in image_manifest.itertuples()
    ]
    records: list[dict] = []
    audit: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, (feature_rows, audit_rows) in enumerate(pool.map(extract_one_image, items), 1):
            records.extend(feature_rows)
            audit.extend(audit_rows)
            if i % 25 == 0 or i == len(items):
                print(f"Extracted fixed perturbation features for {i}/{len(items)} images", flush=True)
    frame = pd.DataFrame(records)
    if frame.duplicated(["image_id", "variant", "region"]).any():
        raise RuntimeError("Duplicated extracted feature rows")
    feature_columns = [name for name in frame.columns if name not in ("image_id", "variant", "region")]
    if len(feature_columns) != 25 or not np.isfinite(frame[feature_columns].to_numpy()).all():
        raise RuntimeError("Unexpected directional feature matrix")
    frame.to_csv(OUT / "image_features.csv", index=False)
    pd.DataFrame(audit).to_csv(OUT / "feature_extraction_audit.csv", index=False)
    dictionaries = []
    for region in REGIONS:
        names: list[str] | None = None
        for variant in VARIANTS:
            matrix, names = aggregate_patient_features(frame, image_manifest, patients, region, variant)
            np.save(OUT / "features" / f"{region}_{variant}.npy", matrix)
            for feature_name in names:
                dictionaries.append(dict(region=region, variant=variant, patient_feature=feature_name))
        R.write_json(OUT / "features" / f"{region}_feature_names.json", names)
    pd.DataFrame(dictionaries).to_csv(OUT / "feature_dictionary.csv", index=False)
    base = np.load(OUT / "features" / "whole_kidney_baseline.npy")
    prior = np.load(PRIOR / "features" / "I3-direction-original.npy")
    if base.shape != prior.shape:
        raise RuntimeError("Inherited I3 baseline shape mismatch")
    difference = float(np.max(np.abs(base.astype(float) - prior.astype(float))))
    reproduced = bool(np.allclose(base, prior, rtol=1e-5, atol=1e-6))
    R.write_json(
        OUT / "baseline_reproduction.json",
        dict(
            compared_to=str(PRIOR / "features" / "I3-direction-original.npy"),
            shape=list(base.shape),
            maximum_absolute_difference=difference,
            relative_tolerance=1e-5,
            absolute_tolerance=1e-6,
            reproduced=reproduced,
        ),
    )
    if not reproduced:
        raise RuntimeError(f"Inherited I3 baseline failed to reproduce: max difference {difference}")
    print("All 25-feature perturbation matrices saved; inherited I3 baseline reproduced", flush=True)


def score_array(y: np.ndarray, score: np.ndarray) -> dict:
    return V.score_metrics(y, score)


def candidate_gate(metrics: pd.DataFrame) -> dict:
    """Development-only screen; no voting decision is made in this batch."""
    base = metrics.loc[
        metrics.group.eq("WHOLE-I3") & metrics.role.eq("development_validation"), ["seed", "auroc", "average_precision"]
    ].set_index("seed")
    candidate = metrics.loc[
        metrics.group.eq("WHOLE-STABLE") & metrics.role.eq("development_validation"), ["seed", "auroc", "average_precision"]
    ].set_index("seed")
    delta = candidate - base
    return dict(
        comparator="WHOLE-I3",
        candidate="WHOLE-STABLE",
        role="development_validation",
        delta_auroc_mean=float(delta.auroc.mean()),
        delta_average_precision_mean=float(delta.average_precision.mean()),
        joint_improvement_draws=int(delta.gt(0).all(axis=1).sum()),
        potential_later_vote_gate=dict(delta_auroc_at_least=.03, delta_average_precision_at_least=.02, joint_improvement_draws_at_least=6),
        passes_potential_later_vote_gate=bool(
            delta.auroc.mean() >= .03 - 1e-12
            and delta.average_precision.mean() >= .02 - 1e-12
            and int(delta.gt(0).all(axis=1).sum()) >= 6
        ),
        note="A pass is only eligibility for a later separately locked comparison; this batch performs no new vote.",
    )


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (role, group), frame in metrics.groupby(["role", "group"], sort=False):
        row = dict(
            role=role,
            group=group,
            draws=len(frame),
            joint_target_draws=int(((frame.auroc > .8) & (frame.average_precision > .8)).sum()),
        )
        for name in ("auroc", "average_precision", "auprc_trapezoid", "sensitivity", "specificity", "accuracy", "brier", "log_loss"):
            for aggregate in ("mean", "std", "min", "max"):
                row[f"{name}_{aggregate}"] = float(getattr(frame[name], aggregate)())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate() -> None:
    require_lock()
    patients, split = read("patients.csv"), read("random_patient_assignments.csv")
    y = patients.label.to_numpy(int)
    arrays = {
        region: {variant: np.load(OUT / "features" / f"{region}_{variant}.npy") for variant in VARIANTS}
        for region in REGIONS
    }
    names = {
        region: json.loads((OUT / "features" / f"{region}_feature_names.json").read_text(encoding="utf-8"))
        for region in REGIONS
    }
    for region in REGIONS:
        if any(array.shape != (98, 50) or not np.isfinite(array).all() for array in arrays[region].values()):
            raise RuntimeError(f"Invalid patient feature arrays for {region}")
    metrics: list[dict] = []
    predictions: list[dict] = []
    stability_rows: list[pd.DataFrame] = []
    provenance: list[dict] = []
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            index = N.indices_for(patients, split, seed)
            selected = {}
            for region in REGIONS:
                stable, frame = stability_screen(arrays[region], names[region], index["train"], region, seed)
                stability_rows.append(frame)
                if not stable.any():
                    raise RuntimeError(f"No stable {region} features under the fixed ICC threshold for seed {seed}")
                selected[region] = stable
            configurations = {
                "WHOLE-I3": ("whole_kidney", np.ones(50, dtype=bool), "all_inherited_directional_features"),
                "WHOLE-STABLE": ("whole_kidney", selected["whole_kidney"], "training_only_icc_filtered"),
                "CENTRAL-GEOMETRIC": ("central_longitudinal", np.ones(50, dtype=bool), "all_geometric_central_features"),
                "CENTRAL-STABLE": ("central_longitudinal", selected["central_longitudinal"], "training_only_icc_filtered"),
            }
            scores = {}
            for group, (region, included, selection_rule) in configurations.items():
                x = arrays[region]["baseline"][:, included]
                model = N.full_head(seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x[index["train"]], y[index["train"]])
                score = model.predict_proba(x)[:, 1]
                scores[group] = score
                path = OUT / "models" / f"{group}_{seed}.joblib"
                joblib.dump(model, path, compress=3)
                if not np.allclose(joblib.load(path).predict_proba(x)[:, 1], score, atol=1e-12):
                    raise RuntimeError(f"Model reload failed: {group}/{seed}")
                provenance.append(
                    dict(
                        seed=seed,
                        group=group,
                        region=region,
                        selection_rule=selection_rule,
                        included_feature_count=int(included.sum()),
                        included_features=[names[region][j] for j in np.flatnonzero(included)],
                        model_file=str(path.relative_to(OUT)),
                        model_sha256=R.sha256(path),
                        train_patient_aliases=patients.patient_alias.iloc[index["train"]].tolist(),
                    )
                )
                for role, label in (("validation", "development_validation"), ("test", "exploratory_test")):
                    current = index[role]
                    metrics.append(dict(seed=seed, group=group, role=label, **score_array(y[current], score[current])))
                    for patient_index in current:
                        predictions.append(
                            dict(
                                seed=seed,
                                group=group,
                                role=label,
                                patient_alias=patients.patient_alias.iloc[patient_index],
                                label=int(y[patient_index]),
                                score=float(score[patient_index]),
                            )
                        )
    metric_frame = pd.DataFrame(metrics)
    stability = pd.concat(stability_rows, ignore_index=True)
    metric_frame.to_csv(OUT / "metrics_by_role_per_draw.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "patient_predictions_by_role.csv", index=False)
    stability.to_csv(OUT / "stability_by_feature_seed.csv", index=False)
    pd.DataFrame(provenance).to_json(OUT / "model_provenance.json", orient="records", indent=2, force_ascii=False)
    summary = summarize(metric_frame)
    summary.to_csv(OUT / "summary_by_role.csv", index=False)
    R.write_json(OUT / "development_candidate_gate.json", candidate_gate(metric_frame))
    print(summary[["role", "group", "auroc_mean", "average_precision_mean"]].to_string(index=False), flush=True)
    print("Development-only candidate gate written; no new vote was fitted", flush=True)


def _figure_settings():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig, stem: str) -> dict:
    png = OUT / "figures_600dpi" / f"{stem}.png"
    tif = OUT / "figures_600dpi" / f"{stem}.tiff"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(tif, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    image = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Unable to read rendered figure {stem}")
    return dict(stem=stem, png=png.name, tiff=tif.name, width=int(image.shape[1]), height=int(image.shape[0]), dpi=600,
                png_sha256=R.sha256(png), tiff_sha256=R.sha256(tif))


def render_region_figure() -> dict:
    plt = _figure_settings()
    image_manifest = read("image_manifest.csv")
    candidates = image_manifest.loc[image_manifest.view.map(is_longitudinal)].groupby("patient_id", sort=True).head(1)
    selected = candidates.iloc[np.linspace(0, len(candidates) - 1, 6).astype(int)]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.2), squeeze=False)
    for axis, row in zip(axes.flat, selected.itertuples()):
        raw = cv2.imread(str(BASE / "native" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE)
        full = cv2.imread(str(BASE / "masks" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE) > 0
        central = cv2.imread(str(OUT / "masks" / "central_longitudinal" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE) > 0
        axis.imshow(raw, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        for mask, color, width in ((full, "#18a8a2", .75), (central, "#e85d75", 1.0)):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                axis.plot(contour[:, 0, 0], contour[:, 0, 1], color=color, linewidth=width)
        axis.set_title(f"{row.image_id} ({row.view})", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Inherited whole-kidney contour and geometric central-longitudinal surrogate", fontsize=12)
    fig.text(.5, .02, "Teal: inherited whole-kidney mask. Pink: principal-axis central 50%; not a parenchyma/cortex segmentation.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .94))
    manifest = save_figure(fig, "Figure_SR1_central_region_geometry")
    plt.close(fig)
    return manifest


def render_results_figure() -> dict:
    plt = _figure_settings()
    metrics = pd.read_csv(OUT / "metrics_by_role_per_draw.csv")
    stability = pd.read_csv(OUT / "stability_by_feature_seed.csv")
    groups = list(GROUPS)
    labels = ["Whole\nI3", "Whole\nstable", "Central\ngeometric", "Central\nstable"]
    colors = ["#427aa1", "#2a9d8f", "#e76f51", "#9c6644"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    positions = np.arange(len(groups))
    for metric, offset, marker in (("auroc", -.18, "o"), ("average_precision", .18, "s")):
        development = metrics.loc[metrics.role.eq("development_validation")].groupby("group")[metric].mean().reindex(groups)
        test = metrics.loc[metrics.role.eq("exploratory_test")].groupby("group")[metric].mean().reindex(groups)
        axes[0].scatter(positions + offset, development, marker=marker, s=45, color="#1f2937", label=f"Development {metric}" if offset < 0 else f"Development {metric}")
        axes[0].scatter(positions + offset, test, marker=marker, s=45, facecolors="none", edgecolors="#1f2937", label=f"Exploratory test {metric}" if offset < 0 else f"Exploratory test {metric}")
        for pos, a, b in zip(positions + offset, development, test):
            axes[0].plot([pos, pos], [a, b], color="#9ca3af", linewidth=.75, zorder=0)
    axes[0].axhline(.8, color="#d97706", linewidth=.8, linestyle="--")
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(.35, 1.02)
    axes[0].set_ylabel("Mean metric across 10 fixed patient draws")
    axes[0].set_title("Discrimination by fixed configuration")
    handles, labels_ = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[:4], ["Development AUROC", "Exploratory test AUROC", "Development AP", "Exploratory test AP"], fontsize=7, loc="lower left")
    counts = stability.groupby(["region", "seed"]).stable.sum().unstack("region").reindex(columns=list(REGIONS))
    x = np.arange(len(counts))
    axes[1].bar(x - .18, counts["whole_kidney"], .36, color=colors[1], label="Whole kidney")
    axes[1].bar(x + .18, counts["central_longitudinal"], .36, color=colors[2], label="Central-longitudinal")
    axes[1].set_xticks(x, [str(value)[-2:] for value in counts.index], rotation=0)
    axes[1].set_xlabel("Fixed random-draw seed suffix")
    axes[1].set_ylabel("Features retained by both ICC screens (of 50)")
    axes[1].set_title("Training-only ICC >= 0.90 selection")
    axes[1].legend(fontsize=8)
    fig.suptitle("Exploratory stability and region probe; no new voting model", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    manifest = save_figure(fig, "Figure_SR2_stability_and_metrics")
    plt.close(fig)
    return manifest


def report() -> None:
    require_lock()
    summary = pd.read_csv(OUT / "summary_by_role.csv")
    gate = json.loads((OUT / "development_candidate_gate.json").read_text(encoding="utf-8"))
    stability = pd.read_csv(OUT / "stability_by_feature_seed.csv")
    quality = json.loads((OUT / "data_quality_summary.json").read_text(encoding="utf-8"))
    figures = [render_region_figure(), render_results_figure()]
    R.write_json(OUT / "figure_manifest.json", figures)
    by_group = summary.pivot(index="group", columns="role", values=["auroc_mean", "average_precision_mean"])
    lines = [
        "# CYCH Stability and Central-Region Probe",
        "",
        "Completed: 2026-09-13. This is a separate, further exploratory batch on the same already explored 98-patient CYCH cohort. It does not modify the manuscript, prior feature packages, masks, classifiers or voting decisions.",
        "",
        "## Scope",
        "",
        f"- Locked cohort: {quality['patients']} patients ({quality['positive']} positive / {quality['negative']} negative), {quality['images']} first-exam images; ten fixed 58/20/20 patient assignments.",
        "- Stability screen: ICC(2,1) is calculated separately for fixed contrast and contour perturbations using only the 58 training patients in each draw. A patient feature is retained only when both ICC values are at least 0.90.",
        "- Central-region control: only longitudinal (LL/RL) views, using the central 50% along the whole-kidney principal axis. This is a geometry-only surrogate and is explicitly not a renal-parenchyma, cortex, medulla, collecting-system or lesion segmentation.",
        "- No candidate is added to the existing RAD + DenseNet + I3 vote in this batch. Development metrics determine only a possible future, separately locked candidate gate; exploratory test results do not select a configuration.",
        "",
        "## Results",
        "",
        "| Configuration | Development AUROC / AP | Exploratory test AUROC / AP |",
        "|---|---:|---:|",
    ]
    display = {
        "WHOLE-I3": "Whole kidney, inherited I3",
        "WHOLE-STABLE": "Whole kidney, ICC-stable I3",
        "CENTRAL-GEOMETRIC": "Central longitudinal geometry, all I3",
        "CENTRAL-STABLE": "Central longitudinal geometry, ICC-stable I3",
    }
    for group in GROUPS:
        d_auc = by_group.loc[group, ("auroc_mean", "development_validation")]
        d_ap = by_group.loc[group, ("average_precision_mean", "development_validation")]
        t_auc = by_group.loc[group, ("auroc_mean", "exploratory_test")]
        t_ap = by_group.loc[group, ("average_precision_mean", "exploratory_test")]
        lines.append(f"| {display[group]} | {d_auc:.4f} / {d_ap:.4f} | {t_auc:.4f} / {t_ap:.4f} |")
    counts = stability.groupby(["region", "seed"]).stable.sum().groupby("region").agg(["min", "median", "max"])
    lines.extend(
        [
            "",
            "## Stability Screen",
            "",
            f"- Whole-kidney retained-feature count per training draw: {int(counts.loc['whole_kidney', 'min'])}-{int(counts.loc['whole_kidney', 'max'])}, median {float(counts.loc['whole_kidney', 'median']):.1f} of 50.",
            f"- Central-longitudinal retained-feature count per training draw: {int(counts.loc['central_longitudinal', 'min'])}-{int(counts.loc['central_longitudinal', 'max'])}, median {float(counts.loc['central_longitudinal', 'median']):.1f} of 50.",
            f"- Development-only WHOLE-STABLE versus WHOLE-I3: delta AUROC {gate['delta_auroc_mean']:+.4f}, delta AP {gate['delta_average_precision_mean']:+.4f}, joint improvements {gate['joint_improvement_draws']}/10. Potential later-vote gate: {'PASS' if gate['passes_potential_later_vote_gate'] else 'NOT PASSED'}. This batch still fits no new vote.",
            "",
            "## Input QA",
            "",
            f"- All inherited native-image/mask shapes, mask-pixel counts and mask SHA-256 values matched the locked manifest. No exact source-image duplicate was found. Longitudinal views exist for every patient.",
            f"- Longitudinal central-mask retained fraction of whole mask: {quality['central_mask_retained_fraction']['min']:.3f}-{quality['central_mask_retained_fraction']['max']:.3f}; median {quality['central_mask_retained_fraction']['median']:.3f}.",
            "- Existing calipers, unresolved segmentation-training overlap and the eight previously flagged mask-quality images remain unresolved. No image was excluded after seeing results.",
            "",
            "## Interpretation",
            "",
            "These are not independent validation estimates. The central geometric result tests whether reducing polar portions of an inherited whole-kidney outline is promising enough to justify true compartment segmentation; it cannot support an anatomical or pathological claim. ICC screening identifies features less sensitive to these particular fixed perturbations, not features proven scanner-invariant or disease-specific.",
            "",
            "## Files",
            "",
            "- `data_quality_summary.json` and `image_geometry_audit.csv`: source/mask/grain checks and central-mask geometry.",
            "- `stability_by_feature_seed.csv`: training-only ICC values and feature inclusion for every seed.",
            "- `metrics_by_role_per_draw.csv`, `summary_by_role.csv`, and `patient_predictions_by_role.csv`: all scores and patient-level predictions.",
            "- `figures_600dpi/`: PNG and TIFF at 600 dpi; `figure_manifest.json` stores checksums and dimensions.",
            "",
            "## References",
            "",
        ]
    )
    lines.extend(f"- {reference['citation']} [{reference['url']}]({reference['url']}) — {reference['scope']}" for reference in REFERENCES)
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("README and two 600 dpi figures rendered", flush=True)


def package() -> None:
    require_lock()
    archive = ROOT / f"{OUT.name}.zip"
    manifest = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file():
            continue
        manifest.append(dict(path=str(path.relative_to(OUT)).replace("\\", "/"), bytes=path.stat().st_size, sha256=R.sha256(path)))
    pd.DataFrame(manifest).to_csv(OUT / "archive_manifest.csv", index=False)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise RuntimeError("ZIP CRC verification failed")
    R.write_json(
        OUT / "archive_verification.json",
        dict(archive=str(archive), bytes=archive.stat().st_size, sha256=R.sha256(archive), entries=len(zipfile.ZipFile(archive).infolist()), crc_verified=True),
    )
    print(f"ZIP verified: {archive.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "extract", "evaluate", "report", "package", "all"))
    args = parser.parse_args()
    stages = ("prepare", "extract", "evaluate", "report", "package") if args.stage == "all" else (args.stage,)
    for stage in stages:
        globals()[stage]()
