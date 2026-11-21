"""Summarize outcome-blind kidneyUS Task002 feasibility on the CYCH packet."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "external_models" / "kidneyUS_official_20260914" / "weights"
RUNS = {
    "annotator_1": ROOT / "CYCH_kidneyUS_internal_regions_20260914",
    "annotator_2": ROOT / "CYCH_kidneyUS_internal_regions_annotator_2_20260914",
}
OUT = ROOT / "CYCH_kidneyUS_internal_region_comparison_20260914"
CLASS_NAMES = {1: "cec", 2: "medulla", 3: "cortex"}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    denominator = int(a.sum()) + int(b.sum())
    return 2.0 * int(np.logical_and(a, b).sum()) / denominator if denominator else 1.0


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=float)
    return {
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def source_reference(variant: str) -> dict:
    base = WEIGHTS / variant
    capsule_dir = (
        base
        / "Task001_KidneyCapsule"
        / "nnUNetTrainerV2__nnUNetPlansv2.1"
        / "gt_niftis"
    )
    region_dir = (
        base
        / "Task002_KidneyRegions"
        / "nnUNetTrainerV2__nnUNetPlansv2.1"
        / "gt_niftis"
    )
    capsule_files = {path.name: path for path in capsule_dir.glob("*.nii.gz")}
    region_files = {path.name: path for path in region_dir.glob("*.nii.gz")}
    shared = sorted(set(capsule_files) & set(region_files))
    coverage = []
    class_fraction = {name: [] for name in CLASS_NAMES.values()}
    class_present = {name: 0 for name in CLASS_NAMES.values()}
    excluded_empty_capsules = 0
    outside_region_pixels = 0
    total_region_pixels = 0
    for filename in shared:
        capsule = sitk.GetArrayFromImage(sitk.ReadImage(str(capsule_files[filename]))) > 0
        regions = sitk.GetArrayFromImage(sitk.ReadImage(str(region_files[filename])))
        if capsule.shape != regions.shape:
            raise RuntimeError(f"Source mask shape mismatch: {filename}")
        if not capsule.any():
            excluded_empty_capsules += 1
            continue
        region_union = regions > 0
        total_region_pixels += int(region_union.sum())
        outside_region_pixels += int(np.logical_and(region_union, ~capsule).sum())
        clipped = np.where(capsule, regions, 0)
        denominator = int(capsule.sum())
        coverage.append(float((clipped > 0).sum() / denominator))
        for class_id, class_name in CLASS_NAMES.items():
            count = int((clipped == class_id).sum())
            class_fraction[class_name].append(float(count / denominator))
            class_present[class_name] += int(count > 0)
    usable = len(coverage)
    return {
        "matched_source_images": len(shared),
        "usable_nonempty_capsules": usable,
        "excluded_empty_capsules": excluded_empty_capsules,
        "union_fraction_of_capsule": quantiles(coverage),
        "raw_region_outside_capsule_fraction": (
            outside_region_pixels / total_region_pixels if total_region_pixels else 0.0
        ),
        "classes": {
            name: {
                "fraction_of_capsule": quantiles(class_fraction[name]),
                "present_images": class_present[name],
                "present_rate": class_present[name] / usable if usable else 0.0,
            }
            for name in CLASS_NAMES.values()
        },
    }


def source_validation_rows() -> list[dict]:
    rows = []
    for variant in ("annotator_1", "annotator_2", "mixed"):
        model = (
            WEIGHTS
            / variant
            / "Task002_KidneyRegions"
            / "nnUNetTrainerV2__nnUNetPlansv2.1"
        )
        per_class = {class_id: [] for class_id in CLASS_NAMES}
        available_folds = []
        for fold in range(5):
            summary = model / f"fold_{fold}" / "validation_raw_postprocessed" / "summary.json"
            if not summary.exists():
                continue
            mean = json.loads(summary.read_text(encoding="utf-8"))["results"]["mean"]
            available_folds.append(fold)
            for class_id in CLASS_NAMES:
                per_class[class_id].append(float(mean[str(class_id)]["Dice"]))
        for class_id, class_name in CLASS_NAMES.items():
            values = per_class[class_id]
            rows.append(
                {
                    "variant": variant,
                    "class": class_name,
                    "available_validation_folds": ";".join(map(str, available_folds)),
                    "available_fold_count": len(values),
                    "mean_of_reported_fold_mean_dice": float(np.mean(values)) if values else "",
                    "minimum_reported_fold_mean_dice": float(np.min(values)) if values else "",
                    "maximum_reported_fold_mean_dice": float(np.max(values)) if values else "",
                    "domain": "kidneyUS source data; not CYCH",
                }
            )
    return rows


def cych_and_agreement() -> tuple[list[dict], list[dict], dict]:
    metrics = {
        variant: read_csv(folder / "region_plausibility_metrics.csv")
        for variant, folder in RUNS.items()
    }
    indexed = {
        variant: {row["image_id"]: row for row in rows}
        for variant, rows in metrics.items()
    }
    image_ids = sorted(set(indexed["annotator_1"]) & set(indexed["annotator_2"]))
    if len(image_ids) != 32:
        raise RuntimeError(f"Expected 32 shared blinded core images, found {len(image_ids)}")

    variant_rows = []
    for variant, rows in metrics.items():
        if len(rows) != 32 or {float(row["requested_scale"]) for row in rows} != {2.0}:
            raise RuntimeError(f"Unexpected fixed packet run for {variant}")
        coverage = [float(row["region_coverage_of_outer"]) for row in rows]
        nonempty_coverage = [value for value in coverage if value > 0]
        variant_rows.append(
            {
                "variant": variant,
                "images": len(rows),
                "nonempty_images": len(nonempty_coverage),
                "nonempty_rate": len(nonempty_coverage) / len(rows),
                "mean_coverage_all_images": float(np.mean(coverage)),
                "median_coverage_all_images": float(np.median(coverage)),
                "median_coverage_nonempty_images": (
                    float(np.median(nonempty_coverage)) if nonempty_coverage else 0.0
                ),
                "three_class_images": sum(int(row["present_class_count"]) == 3 for row in rows),
                "cec_present_images": sum(int(row["cec_pixels"]) > 0 for row in rows),
                "medulla_present_images": sum(int(row["medulla_pixels"]) > 0 for row in rows),
                "cortex_present_images": sum(int(row["cortex_pixels"]) > 0 for row in rows),
            }
        )

    agreement_rows = []
    for image_id in image_ids:
        labels = {
            variant: cv2.imread(
                str(folder / "region_masks" / "s200" / f"{image_id}_labels.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            for variant, folder in RUNS.items()
        }
        if any(value is None for value in labels.values()):
            raise RuntimeError(f"Missing comparison mask for {image_id}")
        for class_id, class_name in {0: "any_region", **CLASS_NAMES}.items():
            a = labels["annotator_1"] > 0 if class_id == 0 else labels["annotator_1"] == class_id
            b = labels["annotator_2"] > 0 if class_id == 0 else labels["annotator_2"] == class_id
            agreement_rows.append(
                {
                    "image_id": image_id,
                    "class": class_name,
                    "annotator_1_pixels": int(a.sum()),
                    "annotator_2_pixels": int(b.sum()),
                    "either_nonempty": bool(a.any() or b.any()),
                    "both_nonempty": bool(a.any() and b.any()),
                    "dice": dice(a, b),
                }
            )

    any_region = [row for row in agreement_rows if row["class"] == "any_region"]
    either_nonempty = [row for row in any_region if row["either_nonempty"]]
    both_nonempty = [row for row in any_region if row["both_nonempty"]]
    agreement_summary = {
        "shared_images": len(image_ids),
        "either_variant_nonempty_images": len(either_nonempty),
        "both_variants_nonempty_images": len(both_nonempty),
        "median_any_region_dice_when_either_nonempty": (
            float(np.median([row["dice"] for row in either_nonempty])) if either_nonempty else None
        ),
        "median_any_region_dice_when_both_nonempty": (
            float(np.median([row["dice"] for row in both_nonempty])) if both_nonempty else None
        ),
        "diagnostic_labels_read": False,
    }
    return variant_rows, agreement_rows, agreement_summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = {variant: source_reference(variant) for variant in RUNS}
    source_validation = source_validation_rows()
    variant_rows, agreement_rows, agreement = cych_and_agreement()
    write_csv(OUT / "source_validation_dice.csv", source_validation)
    write_csv(OUT / "cych_variant_summary.csv", variant_rows)
    write_csv(OUT / "intervariant_agreement.csv", agreement_rows)
    (OUT / "source_region_reference.json").write_text(
        json.dumps(source, indent=2), encoding="utf-8"
    )

    nonempty_gate = 0.80
    agreement_gate = 0.60
    gates = {
        "technical_nonempty_rate_at_least": nonempty_gate,
        "cross_variant_median_any_region_dice_at_least": agreement_gate,
        "clinician_validation_required": True,
        "gate_timing": "Applied after the initial pilot but without reading diagnostic labels.",
        "variant_results": {
            row["variant"]: {
                "nonempty_rate": row["nonempty_rate"],
                "passes_nonempty_gate": row["nonempty_rate"] >= nonempty_gate,
            }
            for row in variant_rows
        },
        "cross_variant_result": {
            "median_any_region_dice_when_either_nonempty": agreement[
                "median_any_region_dice_when_either_nonempty"
            ],
            "passes_agreement_gate": (
                agreement["median_any_region_dice_when_either_nonempty"] is not None
                and agreement["median_any_region_dice_when_either_nonempty"] >= agreement_gate
            ),
        },
        "decision": "Do not use kidneyUS Task002 masks or derived features in the CYCH inflammation classifier.",
        "reconsideration_condition": "A clinician-labeled local internal-region set plus local adaptation must pass per-class validation before reconsideration.",
    }
    (OUT / "decision.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")

    a1, a2 = variant_rows
    readme = f"""# kidneyUS Task002 feasibility decision

## Decision

Do not use the downloaded kidneyUS Task002 masks or any features derived from them in the current CYCH inflammation classifier. The established YOLO/manual mask remains the outer-kidney ROI. This decision is based on a label-blind segmentation feasibility test, not on classification performance.

## CYCH evidence

- Fixed evaluation set: 32 unique images from the previously generated blinded internal-region annotation packet.
- Input: immutable CYCH native image and established outer ROI; diagnostic labels were never read.
- Scale: 2.0, derived from the official-to-CYCH median foreground bounding-box ratio.
- Five-fold softmax ensemble for each released annotator-specific Task002 model.
- `annotator_1`: {a1['nonempty_images']}/32 nonempty ({a1['nonempty_rate']:.1%}); mean outer-ROI coverage {a1['mean_coverage_all_images']:.2%}; all three classes present in {a1['three_class_images']}/32.
- `annotator_2`: {a2['nonempty_images']}/32 nonempty ({a2['nonempty_rate']:.1%}); mean outer-ROI coverage {a2['mean_coverage_all_images']:.2%}; all three classes present in {a2['three_class_images']}/32.
- Either model was nonempty in {agreement['either_variant_nonempty_images']}/32 images; both were nonempty in {agreement['both_variants_nonempty_images']}/32.
- Median cross-model Dice for any predicted region, counting every image where either model was nonempty: {agreement['median_any_region_dice_when_either_nonempty']:.3f}.

Both models fail the outcome-blind technical gates of at least 80% nonempty outputs and median cross-model region Dice of at least 0.60. Empty-empty pairs are excluded from the reported agreement so that common failure is not mistaken for concordance.

## Interpretation boundary

The released models show useful source-domain performance, especially for CEC, but that performance does not transfer reliably to the cropped CYCH images. A few overlays appear anatomically plausible; they cannot establish local validity without clinician masks. The output must not be described as inflammation, scar, swelling, renal sinus, or renal pelvis localization.

The model may be reconsidered only after local adaptation and evaluation against returned blinded clinician annotations. Until then, adding these regions would introduce missingness and unstable anatomy into the feature vote rather than strengthen the paper.

## Files

- `cych_variant_summary.csv`: completeness and coverage by released model.
- `intervariant_agreement.csv`: image-level and class-level agreement.
- `source_validation_dice.csv`: archive-reported source-domain Dice summaries.
- `source_region_reference.json`: source annotation region proportions relative to capsule.
- `decision.json`: technical gates and current exclusion decision.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"variants": variant_rows, "agreement": agreement, "decision": gates["decision"]}, indent=2))


if __name__ == "__main__":
    main()
