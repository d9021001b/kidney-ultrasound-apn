"""Create a deidentified provenance audit for the reused kidney segmenter."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> tuple[list[dict[str, str]], set[str], set[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    hashes = {row["source_sha256"].lower() for row in rows if row.get("source_sha256")}
    basenames = {Path(row["filename"]).name.lower() for row in rows if row.get("filename")}
    return rows, hashes, basenames


def read_split(split_root: Path) -> tuple[list[Path], list[Path]]:
    def load(name: str) -> list[Path]:
        return [Path(line.strip()) for line in (split_root / name).read_text(encoding="utf-8").splitlines() if line.strip()]

    return load("train.txt"), load("val.txt")


def checkpoint_metadata(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("model")
    args = checkpoint.get("train_args") or {}
    metrics = checkpoint.get("train_metrics") or {}
    yaml_data = getattr(model, "yaml", {}) or {}
    return {
        "sha256": sha256(path),
        "date": checkpoint.get("date"),
        "ultralytics_version": checkpoint.get("version"),
        "license": checkpoint.get("license"),
        "task": args.get("task"),
        "architecture": yaml_data.get("yaml_file"),
        "classes": yaml_data.get("nc"),
        "class_names": getattr(model, "names", None),
        "configured_epochs": args.get("epochs"),
        "configured_image_size": args.get("imgsz"),
        "configured_batch_size": args.get("batch"),
        "seed": args.get("seed"),
        "deterministic": args.get("deterministic"),
        "stored_training_data": args.get("data"),
        "mask_precision": metrics.get("metrics/precision(M)"),
        "mask_recall": metrics.get("metrics/recall(M)"),
        "mask_map50": metrics.get("metrics/mAP50(M)"),
        "mask_map50_95": metrics.get("metrics/mAP50-95(M)"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(r"C:\Users\udoo_w2\Desktop\work-kidney-tumor\kidney-tumor\best.pt"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "CYCH_random_validation_20260913" / "image_manifest.csv")
    parser.add_argument("--split-root", type=Path, default=Path(r"C:\Users\udoo_w2\Desktop\work-kidney-tumor\kidney-tumor\yolo_split"))
    parser.add_argument("--sibling-readme", type=Path, default=Path(r"C:\Users\udoo_w2\Desktop\work-kidney-tumor\datasets\README.md"))
    parser.add_argument("--sibling-data-yaml", type=Path, default=Path(r"C:\Users\udoo_w2\Desktop\work-kidney-tumor\kidney-tumor\yolo_split\data.yaml"))
    parser.add_argument("--output", type=Path, default=ROOT / "CYCH_segmentation_provenance_audit_20260914")
    args = parser.parse_args()

    rows, study_hashes, study_basenames = read_manifest(args.manifest)
    train, val = read_split(args.split_root)
    split_paths = train + val
    missing = [path for path in split_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} split-referenced files are unavailable")

    split_basenames = {path.name.lower() for path in split_paths}
    split_hashes = {sha256(path) for path in split_paths}
    train_names = {path.name.lower() for path in train}
    val_names = {path.name.lower() for path in val}
    patient_count = len({row.get("patient_id", "") for row in rows if row.get("patient_id")})
    sibling_readme = args.sibling_readme.read_text(encoding="utf-8")
    sibling_yaml = args.sibling_data_yaml.read_text(encoding="utf-8")
    sibling_calls_tumor = "`kidney-tumor/best.pt` (tumor segmenter)" in sibling_readme
    sibling_class_tumor = "0: tumor" in sibling_yaml

    audit = {
        "audit_date": date.today().isoformat(),
        "scope": "Deidentified aggregate provenance audit; no patient identifiers or filenames are emitted.",
        "checkpoint": checkpoint_metadata(args.checkpoint),
        "current_study": {
            "patients": patient_count,
            "first_examination_images": len(rows),
            "unique_source_hashes": len(study_hashes),
        },
        "locally_available_segmentation_split": {
            "train_entries": len(train),
            "validation_entries": len(val),
            "unique_file_hashes": len(split_hashes),
            "train_validation_basename_overlap": len(train_names & val_names),
            "exact_basename_overlap_with_study": len(split_basenames & study_basenames),
            "exact_sha256_overlap_with_study": len(split_hashes & study_hashes),
            "sibling_project_readme_calls_checkpoint_tumor_segmenter": sibling_calls_tumor,
            "sibling_split_class_is_tumor": sibling_class_tumor,
        },
        "interpretation": {
            "exact_duplicate_found": bool((split_basenames & study_basenames) or (split_hashes & study_hashes)),
            "local_split_proven_as_checkpoint_training_source": False,
            "task_provenance_conflict": True,
            "reason": "The checkpoint records kidneyseg.v1i.yolov12 on another host, but its class is unnamed ('0'); the adjacent project README calls the same path a tumor segmenter and the available local split names its class tumor. No record links the local split to the archived checkpoint path.",
            "conclusion": "The checkpoint is verified only as a single-class YOLOv12x-seg model. Kidney specificity, original task, and source-population overlap remain unresolved. It must be described as a candidate-mask model rather than an independently validated kidney segmenter.",
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "segmentation_checkpoint_provenance_audit.json"
    md_path = args.output / "README.md"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    cp = audit["checkpoint"]
    sp = audit["locally_available_segmentation_split"]
    md_path.write_text(
        "\n".join(
            [
                "# Segmentation checkpoint provenance audit",
                "",
                "This deidentified audit separates evidence stored inside the reused checkpoint from evidence available for subject-overlap review.",
                "",
                f"- Checkpoint SHA-256: `{cp['sha256']}`",
                f"- Architecture/task: `{cp['architecture']}` / `{cp['task']}`; one unnamed class (`0`); input {cp['configured_image_size']} px",
                f"- Embedded mask metrics: precision {cp['mask_precision']:.5f}, recall {cp['mask_recall']:.5f}, mAP50 {cp['mask_map50']:.5f}, mAP50-95 {cp['mask_map50_95']:.5f}",
                f"- Study cohort audited: {patient_count} patients and {len(rows)} first-examination images",
                f"- Available local split audited: {len(split_paths)} files ({len(train)} train, {len(val)} validation)",
                f"- Exact basename overlap: {sp['exact_basename_overlap_with_study']}",
                f"- Exact SHA-256 overlap: {sp['exact_sha256_overlap_with_study']}",
                f"- Adjacent README calls the checkpoint a tumor segmenter: {sp['sibling_project_readme_calls_checkpoint_tumor_segmenter']}",
                f"- Available local split names class 0 as tumor: {sp['sibling_split_class_is_tumor']}",
                "",
                "The saved run metrics describe the checkpoint's archived run and are not a current-cohort segmentation estimate. Its embedded `kidneyseg.v1i.yolov12` path conflicts with the adjacent project's tumor-segmentation description, while the checkpoint class itself is unnamed. The available local split is not proven to be the archived source. Therefore, the checkpoint is treated only as a single-class candidate-mask model; kidney specificity and patient-level independence require the original training manifest and task record.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
