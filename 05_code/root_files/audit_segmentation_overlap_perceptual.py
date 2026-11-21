"""Deidentified structural and perceptual overlap audit for segmentation data."""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RF_PATTERN = re.compile(r"\.rf\.[^.]+$", flags=re.IGNORECASE)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_split(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized_source_stem(path: Path) -> str:
    return RF_PATTERN.sub("", path.stem).casefold()


def patient_token(path: Path) -> str:
    return normalized_source_stem(path).split("_", maxsplit=1)[0]


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return image


def phash64(image: np.ndarray) -> int:
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[:8, :8].copy()
    median = float(np.median(low.reshape(-1)[1:]))
    bits = (low >= median).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def variants(image: np.ndarray) -> list[np.ndarray]:
    rotated = [np.rot90(image, k) for k in range(4)]
    return rotated + [np.fliplr(item) for item in rotated]


def standardized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_r = cv2.resize(left, (192, 192), interpolation=cv2.INTER_AREA).astype(np.float32)
    right_r = cv2.resize(right, (192, 192), interpolation=cv2.INTER_AREA).astype(np.float32)
    left_r = (left_r - left_r.mean()) / max(float(left_r.std()), 1e-6)
    right_r = (right_r - right_r.mean()) / max(float(right_r.std()), 1e-6)
    return float(np.mean(left_r * right_r))


def fit_panel(image: np.ndarray, width: int, height: int) -> Image.Image:
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    rgb.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), "black")
    panel.paste(rgb, ((width - rgb.width) // 2, (height - rgb.height) // 2))
    return panel


def write_private_contact_sheets(
    output: Path,
    nearest_ranked: list[dict[str, object]],
    study_paths: list[Path],
    segmentation_paths: list[Path],
) -> list[Path]:
    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    font = ImageFont.truetype(str(font_path), 22) if font_path.is_file() else ImageFont.load_default()
    paths: list[Path] = []
    for sheet_index in range(2):
        rows = nearest_ranked[sheet_index * 5 : (sheet_index + 1) * 5]
        canvas = Image.new("RGB", (1280, 1320), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((35, 20), "Private QA of closest deidentified pHash pairs", fill="black", font=font)
        for row_index, row in enumerate(rows):
            study_index = int(str(row["study_image_alias"])[1:]) - 1
            segmentation_index = int(str(row["segmentation_file_alias"])[1:]) - 1
            variant_index = int(row["best_variant_index"])
            study_image = variants(read_gray(study_paths[study_index]))[variant_index]
            segmentation_image = read_gray(segmentation_paths[segmentation_index])
            top = 70 + row_index * 245
            canvas.paste(fit_panel(study_image, 560, 190), (35, top + 35))
            canvas.paste(fit_panel(segmentation_image, 560, 190), (685, top + 35))
            label = (
                f"{row['study_image_alias']} vs {row['segmentation_file_alias']}  "
                f"pHash={row['phash_hamming_distance']}  "
                f"corr={row['standardized_pixel_correlation']:.3f}"
            )
            draw.text((35, top), label, fill="black", font=font)
        path = output / f"qa_private_closest_pairs_{sheet_index + 1}.png"
        canvas.save(path, dpi=(150, 150))
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path(r"C:\Users\udoo_w2\Desktop\work-kidney-tumor\kidney-tumor\yolo_split"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "CYCH_random_validation_20260913" / "image_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "CYCH_segmentation_provenance_audit_20260915",
    )
    args = parser.parse_args()

    train = read_split(args.split_root / "train.txt")
    validation = read_split(args.split_root / "val.txt")
    segmentation_paths = train + validation
    missing = [path for path in segmentation_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} segmentation split files are missing")

    study_rows = read_csv(args.manifest)
    study_paths = [Path(row["clean_path"]) for row in study_rows]
    study_ids = {row["patient_id"] for row in study_rows}
    missing_study = [path for path in study_paths if not path.is_file()]
    if missing_study:
        raise FileNotFoundError(f"{len(missing_study)} study images are missing")

    train_stems = {normalized_source_stem(path) for path in train}
    validation_stems = {normalized_source_stem(path) for path in validation}
    train_tokens = {patient_token(path) for path in train}
    validation_tokens = {patient_token(path) for path in validation}
    all_tokens = train_tokens | validation_tokens
    suffix_six = {value[-6:] for value in study_ids if len(value) >= 6}

    segmentation_hashes: list[int] = []
    segmentation_images: list[np.ndarray] = []
    for path in segmentation_paths:
        image = read_gray(path)
        segmentation_images.append(image)
        segmentation_hashes.append(phash64(image))

    nearest: list[dict[str, object]] = []
    for study_index, path in enumerate(study_paths):
        image = read_gray(path)
        study_variants = variants(image)
        variant_hashes = [phash64(item) for item in study_variants]
        best_distance = 65
        best_segmentation_index = -1
        best_variant_index = -1
        for segmentation_index, segmentation_hash in enumerate(segmentation_hashes):
            for variant_index, variant_hash in enumerate(variant_hashes):
                distance = (segmentation_hash ^ variant_hash).bit_count()
                if distance < best_distance:
                    best_distance = distance
                    best_segmentation_index = segmentation_index
                    best_variant_index = variant_index
        correlation = standardized_correlation(
            study_variants[best_variant_index], segmentation_images[best_segmentation_index]
        )
        nearest.append(
            {
                "study_image_alias": f"I{study_index + 1:04d}",
                "segmentation_file_alias": f"S{best_segmentation_index + 1:04d}",
                "phash_hamming_distance": best_distance,
                "standardized_pixel_correlation": correlation,
                "best_variant_index": best_variant_index,
            }
        )

    distances = np.asarray([row["phash_hamming_distance"] for row in nearest], dtype=float)
    correlations = np.asarray([row["standardized_pixel_correlation"] for row in nearest], dtype=float)
    nearest_ranked = sorted(
        nearest,
        key=lambda row: (row["phash_hamming_distance"], -row["standardized_pixel_correlation"]),
    )
    threshold_counts = {
        f"distance_le_{threshold}": int(np.sum(distances <= threshold))
        for threshold in (0, 2, 4, 6, 8, 10, 12)
    }

    audit = {
        "audit_date": date.today().isoformat(),
        "scope": (
            "Deidentified comparison of a locally available segmentation split with the 349-image "
            "study layer. Original filenames and patient identifiers are not emitted."
        ),
        "method": {
            "normalized_stem": "Remove image extension and Roboflow .rf.<hash> suffix.",
            "patient_token": "Text before the first underscore in the normalized segmentation stem.",
            "perceptual_hash": (
                "64-bit DCT pHash on 32 x 32 grayscale images; each study image was compared in "
                "four rotations and their horizontal reflections against all local segmentation files."
            ),
            "near_duplicate_gate": (
                "pHash Hamming distance <= 6 is treated as a conservative review trigger, not proof "
                "of identity. Standardized pixel correlation is reported for each nearest pair."
            ),
        },
        "locally_available_segmentation_split": {
            "train_files": len(train),
            "validation_files": len(validation),
            "unique_normalized_source_stems": len(train_stems | validation_stems),
            "train_validation_normalized_stem_overlap": len(train_stems & validation_stems),
            "unique_patient_tokens": len(all_tokens),
            "train_validation_patient_token_overlap": len(train_tokens & validation_tokens),
            "roboflow_style_filename_fraction": sum(".rf." in path.stem.casefold() for path in segmentation_paths)
            / len(segmentation_paths),
        },
        "study_comparison": {
            "study_patients": len(study_ids),
            "study_images": len(study_paths),
            "exact_patient_token_overlap": len(all_tokens & study_ids),
            "study_suffix_six_token_overlap": len(all_tokens & suffix_six),
            "nearest_phash_distance_min": int(distances.min()),
            "nearest_phash_distance_median": float(np.median(distances)),
            "nearest_phash_distance_max": int(distances.max()),
            **threshold_counts,
            "nearest_pair_correlation_max": float(correlations.max()),
            "nearest_pair_correlation_median": float(np.median(correlations)),
            "ten_closest_deidentified_pairs": nearest_ranked[:10],
        },
        "interpretation": {
            "local_split_patient_token_disjoint": not bool(train_tokens & validation_tokens),
            "identifier_overlap_found": bool((all_tokens & study_ids) or (all_tokens & suffix_six)),
            "perceptual_review_trigger_found": bool(np.any(distances <= 6)),
            "local_split_proven_as_checkpoint_training_source": False,
            "conclusion": (
                "The locally available segmentation split is disjoint by normalized source stem and "
                "patient token, and no study identifier token overlap was found. Perceptual results "
                "further assess renamed or recompressed image overlap. These checks strengthen the "
                "local overlap audit but do not prove that this split is the archived checkpoint "
                "training source; the original training manifest remains the authoritative record."
            ),
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "segmentation_overlap_perceptual_audit.json"
    readme_path = args.output / "README.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    split = audit["locally_available_segmentation_split"]
    study = audit["study_comparison"]
    readme_path.write_text(
        "\n".join(
            [
                "# Segmentation overlap audit",
                "",
                "This deidentified audit extends the exact-file comparison to Roboflow-normalized names, case tokens, and DCT perceptual hashes.",
                "",
                f"- Local split: {split['train_files']} training and {split['validation_files']} validation files",
                f"- Normalized source stems: {split['unique_normalized_source_stems']}; train-validation overlap: {split['train_validation_normalized_stem_overlap']}",
                f"- Patient tokens: {split['unique_patient_tokens']}; train-validation overlap: {split['train_validation_patient_token_overlap']}",
                f"- Identifier token overlap with the 98-patient study: {study['exact_patient_token_overlap']}",
                f"- Six-digit suffix token overlap with the study: {study['study_suffix_six_token_overlap']}",
                f"- Nearest pHash distance across 349 study images: minimum {study['nearest_phash_distance_min']}, median {study['nearest_phash_distance_median']:.1f}, maximum {study['nearest_phash_distance_max']}",
                f"- Conservative pHash review triggers at distance <= 6: {study['distance_le_6']}",
                f"- Maximum standardized pixel correlation among nearest pHash pairs: {study['nearest_pair_correlation_max']:.3f}",
                "",
                "The available local split remains unproven as the checkpoint's archived training source. Consequently, the audit can identify evidence of overlap but cannot replace recovery of the original training manifest.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    qa_paths = write_private_contact_sheets(
        args.output,
        nearest_ranked,
        study_paths,
        segmentation_paths,
    )
    print(json_path)
    print(readme_path)
    for qa_path in qa_paths:
        print(qa_path)


if __name__ == "__main__":
    main()
