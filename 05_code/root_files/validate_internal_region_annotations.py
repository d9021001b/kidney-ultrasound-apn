"""Validate blinded local kidney internal-region annotations.

This module deliberately consumes no diagnosis labels or patient identifiers.
It validates masks against the frozen whole-kidney ROI distributed in an
annotation packet and can summarize inter-rater and hidden-repeat agreement.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


CLASSES = ("cec", "medulla", "cortex", "uncertain")
SEMANTIC_CLASSES = CLASSES[:3]
CLASS_LABELS = {
    "cec": "Central Echo Complex",
    "medulla": "Medulla",
    "cortex": "Cortex",
    "uncertain": "Uncertain",
}
COLORS_BGR = {
    "cec": (0, 165, 255),
    "medulla": (255, 180, 0),
    "cortex": (0, 200, 0),
    "uncertain": (200, 0, 200),
}
VALID_STATUS = frozenset(("present", "absent", "not_identifiable", "not_annotated"))


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_tasks(packet_root: Path) -> pd.DataFrame:
    tasks = pd.read_csv(packet_root / "annotation_tasks_blinded.csv", dtype={"task_id": str})
    required = {"task_id", "display_order", "view"}
    missing = required - set(tasks.columns)
    if missing:
        raise ValueError(f"Blinded task registry lacks columns: {sorted(missing)}")
    if tasks.task_id.duplicated().any():
        raise ValueError("Blinded task registry contains duplicate task ids")
    return tasks.sort_values("display_order").reset_index(drop=True)


def read_mask(path: Path) -> tuple[np.ndarray | None, list[str]]:
    """Read a binary annotation mask while recording rather than hiding defects."""
    if not path.exists():
        return None, ["missing_mask"]
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, ["unreadable_mask"]
    if image.ndim == 3:
        image = image[:, :, 0]
    values = np.unique(image)
    warnings: list[str] = []
    if not set(values.tolist()).issubset({0, 1, 255}):
        warnings.append("nonbinary_values_thresholded")
    return (image > 0).astype(np.uint8), warnings


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float | None:
    """Dice for nonempty class pairs; empty/empty remains not estimable."""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    denominator = int(a.sum() + b.sum())
    if denominator == 0:
        return None
    return float(2 * np.logical_and(a, b).sum() / denominator)


def _read_statuses(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    statuses: dict[str, dict[str, str]] = {}
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        if not task_id:
            continue
        statuses[task_id] = {
            name: str(row.get(f"{name}_status", "")).strip().lower()
            for name in SEMANTIC_CLASSES
        }
    return statuses


def _overlay(gray: np.ndarray, outer: np.ndarray, masks: dict[str, np.ndarray], path: Path) -> None:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for class_name in CLASSES:
        mask = masks.get(class_name)
        if mask is None:
            continue
        color = np.full_like(canvas, COLORS_BGR[class_name])
        selected = mask.astype(bool)
        canvas[selected] = cv2.addWeighted(canvas, 0.55, color, 0.45, 0)[selected]
    contours, _ = cv2.findContours(outer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def validate_submission(packet_root: Path, annotations: Path, output: Path) -> pd.DataFrame:
    """Validate all expected masks and write a human-reviewable QC table/panel set."""
    packet_root = Path(packet_root)
    annotations = Path(annotations)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    panels = output / "qc_overlays"
    panels.mkdir(exist_ok=True)
    tasks = read_tasks(packet_root)
    statuses = _read_statuses(annotations / "annotation_status.csv")
    records: list[dict[str, object]] = []

    for task in tasks.itertuples(index=False):
        task_id = str(task.task_id)
        gray = cv2.imread(str(packet_root / "images" / f"{task_id}.png"), cv2.IMREAD_GRAYSCALE)
        outer, outer_errors = read_mask(packet_root / "outer_roi_masks" / f"{task_id}.png")
        if gray is None or outer is None:
            raise RuntimeError(f"Packet asset unavailable for {task_id}: {outer_errors}")
        if gray.shape != outer.shape:
            raise RuntimeError(f"Packet image/mask shape mismatch for {task_id}")

        masks: dict[str, np.ndarray] = {}
        all_masks: list[np.ndarray] = []
        base_errors: list[str] = []
        for class_name in CLASSES:
            mask, errors = read_mask(annotations / f"{task_id}_{class_name}.png")
            class_errors = list(errors)
            if mask is not None:
                if mask.shape != outer.shape:
                    class_errors.append("shape_mismatch")
                else:
                    leakage = int(np.logical_and(mask > 0, outer == 0).sum())
                    if leakage:
                        class_errors.append(f"outside_outer_roi:{leakage}")
                    masks[class_name] = mask
                    all_masks.append(mask)
            status = statuses.get(task_id, {}).get(class_name, "") if class_name in SEMANTIC_CLASSES else ""
            if class_name in SEMANTIC_CLASSES and status and status not in VALID_STATUS:
                class_errors.append(f"invalid_status:{status}")
            if class_name in SEMANTIC_CLASSES and mask is not None and mask.shape == outer.shape:
                if status == "present" and int(mask.sum()) == 0:
                    class_errors.append("present_but_empty")
                if status in {"absent", "not_identifiable"} and int(mask.sum()) > 0:
                    class_errors.append("nonempty_mask_conflicts_with_status")
            records.append(
                {
                    "task_id": task_id,
                    "view": task.view,
                    "class_name": class_name,
                    "class_label": CLASS_LABELS[class_name],
                    "mask_pixels": None if mask is None else int(mask.sum()),
                    "outer_roi_pixels": int(outer.sum()),
                    "area_fraction_of_outer_roi": None
                    if mask is None
                    else float(mask.sum() / max(1, outer.sum())),
                    "status": status,
                    "errors": ";".join(class_errors),
                }
            )
            base_errors.extend(class_errors)

        same_shape = [mask for mask in all_masks if mask.shape == outer.shape]
        if len(same_shape) > 1:
            stack = np.stack(same_shape, axis=0)
            overlap_pixels = int((stack.sum(axis=0) > 1).sum())
            if overlap_pixels:
                base_errors.append(f"class_overlap:{overlap_pixels}")
                for record in records[-len(CLASSES) :]:
                    record["errors"] = ";".join(filter(None, [str(record["errors"]), f"class_overlap:{overlap_pixels}"]))
        _overlay(gray, outer, masks, panels / f"{task_id}.png")

    table = pd.DataFrame(records)
    table.to_csv(output / "annotation_qc.csv", index=False)
    errors = table.errors.fillna("").str.len() > 0
    write_json(
        output / "annotation_qc_summary.json",
        {
            "tasks": int(tasks.shape[0]),
            "mask_rows": int(table.shape[0]),
            "rows_with_errors": int(errors.sum()),
            "tasks_with_errors": int(table.loc[errors, "task_id"].nunique()),
            "status_file_present": (annotations / "annotation_status.csv").exists(),
            "scope": "Structural QC only. Passing QC does not establish anatomical validity.",
        },
    )
    return table


def _class_mask(annotations: Path, task_id: str, class_name: str) -> np.ndarray | None:
    mask, errors = read_mask(annotations / f"{task_id}_{class_name}.png")
    if mask is None or errors and errors[0] in {"missing_mask", "unreadable_mask"}:
        return None
    return mask


def compare_submissions(packet_root: Path, annotation_a: Path, annotation_b: Path, output: Path) -> pd.DataFrame:
    """Calculate class Dice only after both separate submissions pass basic availability checks."""
    tasks = read_tasks(Path(packet_root))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for task in tasks.itertuples(index=False):
        for class_name in SEMANTIC_CLASSES:
            mask_a = _class_mask(Path(annotation_a), str(task.task_id), class_name)
            mask_b = _class_mask(Path(annotation_b), str(task.task_id), class_name)
            issue = ""
            value: float | None = None
            if mask_a is None or mask_b is None:
                issue = "missing_or_unreadable_mask"
            elif mask_a.shape != mask_b.shape:
                issue = "shape_mismatch"
            else:
                value = dice(mask_a, mask_b)
                if value is None:
                    issue = "both_empty_not_estimable"
            rows.append(
                {
                    "task_id": str(task.task_id),
                    "view": task.view,
                    "class_name": class_name,
                    "dice": value,
                    "issue": issue,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output / "interrater_dice.csv", index=False)
    summary = (
        table.dropna(subset=["dice"])
        .groupby("class_name", as_index=False)["dice"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    summary.to_csv(output / "interrater_dice_summary.csv", index=False)
    return table


def compare_hidden_repeats(
    packet_root: Path,
    annotations: Path,
    private_task_key: Path,
    output: Path,
) -> pd.DataFrame:
    """Use the internal mapping to compare a rater's hidden repeated task pairs."""
    del packet_root  # Mapping is intentionally kept outside the blinded archive.
    key = pd.read_csv(private_task_key, dtype={"task_id": str, "duplicate_of_task_id": str})
    repeats = key.loc[key.duplicate_of_task_id.notna() & key.duplicate_of_task_id.ne("")].copy()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for row in repeats.itertuples(index=False):
        for class_name in SEMANTIC_CLASSES:
            original = _class_mask(Path(annotations), str(row.duplicate_of_task_id), class_name)
            repeated = _class_mask(Path(annotations), str(row.task_id), class_name)
            issue = ""
            value: float | None = None
            if original is None or repeated is None:
                issue = "missing_or_unreadable_mask"
            elif original.shape != repeated.shape:
                issue = "shape_mismatch"
            else:
                value = dice(original, repeated)
                if value is None:
                    issue = "both_empty_not_estimable"
            rows.append(
                {
                    "task_id": str(row.task_id),
                    "duplicate_of_task_id": str(row.duplicate_of_task_id),
                    "class_name": class_name,
                    "dice": value,
                    "issue": issue,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output / "intrarater_repeat_dice.csv", index=False)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, default=Path(__file__).resolve().parent)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Perform structural QC on one returned annotation folder.")
    validate.add_argument("--annotations", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare", help="Calculate per-task inter-rater Dice.")
    compare.add_argument("--annotations-a", type=Path, required=True)
    compare.add_argument("--annotations-b", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    repeats = subparsers.add_parser("repeats", help="Calculate hidden-repeat Dice with the internal mapping.")
    repeats.add_argument("--annotations", type=Path, required=True)
    repeats.add_argument("--private-task-key", type=Path, required=True)
    repeats.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate":
        validate_submission(args.packet_root, args.annotations, args.output)
    elif args.command == "compare":
        compare_submissions(args.packet_root, args.annotations_a, args.annotations_b, args.output)
    elif args.command == "repeats":
        compare_hidden_repeats(args.packet_root, args.annotations, args.private_task_key, args.output)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
