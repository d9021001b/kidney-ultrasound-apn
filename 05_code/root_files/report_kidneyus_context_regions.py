"""Compare black-canvas and context-preserving kidneyUS Task002 probes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CYCH_kidneyUS_context_region_comparison_20260914"
RUNS = {
    "black_annotator_1": ROOT / "CYCH_kidneyUS_internal_regions_20260914",
    "black_annotator_2": ROOT / "CYCH_kidneyUS_internal_regions_annotator_2_20260914",
    "context_annotator_1": ROOT / "CYCH_kidneyUS_context_regions_annotator_1_20260914",
    "context_annotator_2": ROOT / "CYCH_kidneyUS_context_regions_annotator_2_20260914",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def mask_path(run_name: str, image_id: str) -> Path:
    run = RUNS[run_name]
    subdir = "region_masks" if run_name.startswith("context") else "region_masks/s200"
    return run / subdir / f"{image_id}_labels.png"


def summarize(run_name: str) -> dict:
    rows = read_csv(RUNS[run_name] / "region_plausibility_metrics.csv")
    coverage = np.asarray([float(row["region_coverage_of_outer"]) for row in rows])
    nonempty = coverage > 0
    return {
        "run": run_name,
        "images": len(rows),
        "nonempty_images": int(nonempty.sum()),
        "nonempty_rate": float(nonempty.mean()),
        "three_class_images": int(sum(int(row["present_class_count"]) == 3 for row in rows)),
        "mean_coverage_all_images": float(coverage.mean()),
        "median_coverage_nonempty_images": float(np.median(coverage[nonempty])) if nonempty.any() else 0.0,
    }


def agreement(prefix: str) -> tuple[list[dict], dict]:
    first = f"{prefix}_annotator_1"
    second = f"{prefix}_annotator_2"
    ids = [row["image_id"] for row in read_csv(RUNS[first] / "region_plausibility_metrics.csv")]
    rows = []
    for image_id in ids:
        a = cv2.imread(str(mask_path(first, image_id)), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(mask_path(second, image_id)), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None or a.shape != b.shape:
            raise RuntimeError(f"Missing or mismatched masks for {prefix}/{image_id}")
        aa, bb = a > 0, b > 0
        denominator = int(aa.sum() + bb.sum())
        dice = 2 * int(np.logical_and(aa, bb).sum()) / denominator if denominator else 1.0
        rows.append(
            {
                "input": prefix,
                "image_id": image_id,
                "annotator_1_nonempty": bool(aa.any()),
                "annotator_2_nonempty": bool(bb.any()),
                "either_nonempty": bool(aa.any() or bb.any()),
                "both_nonempty": bool(aa.any() and bb.any()),
                "any_region_dice": dice,
            }
        )
    either = [row["any_region_dice"] for row in rows if row["either_nonempty"]]
    both = [row["any_region_dice"] for row in rows if row["both_nonempty"]]
    summary = {
        "input": prefix,
        "images": len(rows),
        "either_nonempty_images": sum(row["either_nonempty"] for row in rows),
        "both_nonempty_images": sum(row["both_nonempty"] for row in rows),
        "median_dice_when_either_nonempty": float(np.median(either)) if either else None,
        "median_dice_when_both_nonempty": float(np.median(both)) if both else None,
    }
    return rows, summary


def main() -> None:
    OUT.mkdir(exist_ok=True)
    run_rows = [summarize(name) for name in RUNS]
    black_rows, black = agreement("black")
    context_rows, context = agreement("context")
    write_csv(OUT / "input_variant_summary.csv", run_rows)
    write_csv(OUT / "intermodel_agreement.csv", black_rows + context_rows)

    nonempty_gate = 0.80
    dice_gate = 0.60
    decision = {
        "diagnostic_labels_read": False,
        "technical_gates": {
            "each_model_nonempty_rate_at_least": nonempty_gate,
            "median_intermodel_dice_when_either_nonempty_at_least": dice_gate,
        },
        "black_canvas_agreement": black,
        "context_preserving_agreement": context,
        "context_model_results": {
            row["run"]: row for row in run_rows if row["run"].startswith("context")
        },
        "passes": bool(
            all(
                row["nonempty_rate"] >= nonempty_gate
                for row in run_rows
                if row["run"].startswith("context")
            )
            and context["median_dice_when_either_nonempty"] >= dice_gate
        ),
        "classification_action": "Exclude Task002-derived anatomy features from the CYCH inflammation classifier.",
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    a1 = decision["context_model_results"]["context_annotator_1"]
    a2 = decision["context_model_results"]["context_annotator_2"]
    readme = f"""# kidneyUS Task002 context-preserving transfer audit

## Question

The first CYCH Task002 transfer placed a tight kidney crop on a black canvas, whereas the
released nnU-Net models were trained on complete kidney ultrasound frames. This outcome-blind
audit repeats the same fixed 32-image packet and five-fold model ensembles after retaining
surrounding tissue in a kidney-centred two-times zoom of each clean original frame.

## Results

| Input and released model | Nonempty | Three classes | Mean region coverage |
|---|---:|---:|---:|
| Black canvas, annotator 1 | 11/32 | 1/32 | 1.96% |
| Context zoom, annotator 1 | {a1['nonempty_images']}/32 | {a1['three_class_images']}/32 | {100*a1['mean_coverage_all_images']:.2f}% |
| Black canvas, annotator 2 | 11/32 | 5/32 | 4.53% |
| Context zoom, annotator 2 | {a2['nonempty_images']}/32 | {a2['three_class_images']}/32 | {100*a2['mean_coverage_all_images']:.2f}% |

When either released model predicted an internal region, median intermodel Dice increased
from {black['median_dice_when_either_nonempty']:.3f} with the black canvas to
{context['median_dice_when_either_nonempty']:.3f} with retained context. When both were nonempty,
the corresponding values were {black['median_dice_when_both_nonempty']:.3f} and
{context['median_dice_when_both_nonempty']:.3f}.

## Decision

Retaining surrounding tissue improves transfer modestly, but both released models remain far
below the prespecified 80% nonempty-output gate and their median agreement remains below 0.60.
The official Task002 masks therefore remain excluded from inflammation modeling. They cannot be
used as cortex, medulla, central echogenic complex, scar, or inflammation truth in the manuscript.

The experiment did not load diagnosis labels. The result supports local adaptation only after
returned blinded clinician compartment masks become available.

## Method provenance

- Open Kidney Ultrasound Data Set and released weights: https://github.com/rsingla92/kidneyUS
- nnU-Net: https://doi.org/10.1038/s41592-020-01008-z
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
