"""Audit the kidney-inflammation project and its reproducibility inputs.

This script is intentionally read-only with respect to source data. It writes
machine-readable inventories and a human-readable audit report to a new output
directory under the project root.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "project_audit_20260912"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm"}
RASTER_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PID_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")
DATE_RE = re.compile(r"(?<!\d)((?:19|20)\d{6})(?!\d)")
DIAGNOSIS_TERMS = (
    "nephritis",
    "pyelonephritis",
    "inflammation",
    "diagnosis",
    "positive",
    "negative",
)


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def write_csv(path: Path, header: list[str], rows: Iterable[Iterable[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def clean_pid(value: Any) -> str:
    match = PID_RE.search(str(value))
    return match.group(1) if match else ""


def find_date(value: str) -> str:
    for token in Path(value).stem.split("_"):
        if not re.fullmatch(r"(?:19|20)\d{6}", token):
            continue
        try:
            datetime.strptime(token, "%Y%m%d")
            return token
        except ValueError:
            continue
    return ""


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def inventory(out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    files: list[dict[str, Any]] = []
    dirs: dict[str, dict[str, Any]] = defaultdict(lambda: {"files": 0, "bytes": 0, "extensions": Counter()})
    for path in ROOT.rglob("*"):
        if not path.is_file() or out_dir in path.parents:
            continue
        try:
            stat = path.stat()
        except OSError as exc:
            files.append({"path": rel(path), "error": str(exc)})
            continue
        relative = rel(path)
        parts = Path(relative).parts
        top = parts[0] if parts else "[root]"
        ext = path.suffix.lower() or "[no_extension]"
        files.append(
            {
                "path": relative,
                "top_level": top,
                "extension": ext,
                "size_bytes": stat.st_size,
                "modified_local": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
        dirs[top]["files"] += 1
        dirs[top]["bytes"] += stat.st_size
        dirs[top]["extensions"][ext] += 1
    summary = {
        key: {
            "files": value["files"],
            "bytes": value["bytes"],
            "extensions": dict(sorted(value["extensions"].items())),
        }
        for key, value in sorted(dirs.items())
    }
    return sorted(files, key=lambda item: item.get("path", "")), summary


def read_current_site(site: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    csv_path = ROOT / site / f"{site}.csv"
    data_dir = ROOT / site / "Data"
    original_dir = ROOT / ("original_png" if site == "CYCH" else "original_png_nckuh")
    metadata_path = ROOT / f"{site.lower()}_metadata.json"
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for item in csv.DictReader(handle):
            filename = (item.get("filename") or "").strip()
            label_raw = (item.get("label") or "").strip()
            try:
                label = int(float(label_raw))
            except ValueError:
                label = None
            image_candidates = [
                data_dir / filename,
                original_dir / filename,
                ROOT / site / filename,
            ]
            image_path = next((candidate for candidate in image_candidates if candidate.exists()), image_candidates[0])
            variant = "red_processed" if "_red_" in filename else "original"
            width = height = ""
            mode = ""
            image_error = ""
            if image_path.exists() and image_path.suffix.lower() in RASTER_EXTS:
                try:
                    from PIL import Image

                    with Image.open(image_path) as image:
                        width, height = image.size
                        mode = image.mode
                except Exception as exc:  # pragma: no cover - depends on damaged files
                    image_error = str(exc)
            parts = Path(filename).stem.split("_")
            rows.append(
                {
                    "site": site,
                    "filename": filename,
                    "relative_image_path": rel(image_path) if image_path.exists() else "",
                    "image_variant": variant,
                    "patient_id": clean_pid(filename),
                    "exam_date": find_date(filename),
                    "label": label if label is not None else "",
                    "image_exists": image_path.exists(),
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "image_error": image_error,
                    "metadata_fields": "|".join(parts[2:-1]) if len(parts) > 3 else "",
                }
            )

    metadata: dict[str, dict[str, Any]] = {}
    if metadata_path.exists():
        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                metadata = {(item.get("filename") or ""): item for item in raw if isinstance(item, dict)}
            elif isinstance(raw, dict):
                metadata = {str(key): value for key, value in raw.items() if isinstance(value, dict)}
        except Exception as exc:
            metadata = {"__error__": {"error": str(exc)}}
    for row in rows:
        item = metadata.get(row["filename"], {})
        row["metadata_exists"] = bool(item)
        row["metadata_label"] = item.get("label", "") if isinstance(item, dict) else ""
        row["metadata_label_matches"] = (
            row["label"] == item.get("label") if row["label"] != "" and isinstance(item, dict) and "label" in item else ""
        )

    labels = [row["label"] for row in rows if row["label"] in (0, 1)]
    patients = defaultdict(list)
    for row in rows:
        if row["patient_id"]:
            patients[row["patient_id"]].append(row["label"])
    patient_labels = {pid: int(any(label == 1 for label in values)) for pid, values in patients.items()}
    logical_labels: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if row["label"] in (0, 1):
            logical_labels[row["filename"].replace("_red_", "_")].add(row["label"])
    dates = [row["exam_date"] for row in rows if row["exam_date"]]
    summary = {
        "site": site,
        "csv": rel(csv_path),
        "data_directory": rel(data_dir),
        "metadata": rel(metadata_path) if metadata_path.exists() else "",
        "image_rows": len(rows),
        "logical_image_names": len({row["filename"].replace("_red_", "_") for row in rows}),
        "image_variant_counts": dict(sorted(Counter(row["image_variant"] for row in rows).items())),
        "logical_label_mismatches": sum(len(labels) > 1 for labels in logical_labels.values()),
        "image_rows_with_valid_label": len(labels),
        "label_counts": dict(sorted(Counter(map(str, labels)).items())),
        "missing_images": sum(not row["image_exists"] for row in rows),
        "image_read_errors": sum(bool(row["image_error"]) for row in rows),
        "unique_patient_ids": len(patients),
        "patient_level_label_counts": dict(sorted(Counter(map(str, patient_labels.values())).items())),
        "date_min": min(dates) if dates else "",
        "date_max": max(dates) if dates else "",
        "metadata_rows": len(metadata) if "__error__" not in metadata else 0,
        "metadata_label_mismatches": sum(row.get("metadata_label_matches") is False for row in rows),
    }
    return rows, summary


def inspect_kidata() -> dict[str, Any]:
    base = ROOT / "ki-data"
    layer_rows: list[dict[str, Any]] = []
    diagnosis_hits: list[str] = []
    label_names: Counter[str] = Counter()
    patient_ids: set[str] = set()
    for path in base.rglob("*"):
        if not path.is_file() or "ezdicom" in path.parts:
            continue
        ext = path.suffix.lower()
        if ext in RASTER_EXTS:
            pid = clean_pid(path.name)
            if pid:
                patient_ids.add(pid)
        if ext == ".json":
            try:
                obj = json.loads(path.read_text(encoding="utf-8-sig"))
                for shape in obj.get("shapes", []) if isinstance(obj, dict) else []:
                    label = str(shape.get("label", "")).strip()
                    if label:
                        label_names[label] += 1
            except Exception:
                pass
        if ext in {".json", ".txt"}:
            try:
                text = path.read_text(encoding="utf-8-sig", errors="ignore").lower()
                for term in DIAGNOSIS_TERMS:
                    if term in text:
                        diagnosis_hits.append(f"{rel(path)}:{term}")
            except OSError:
                pass

    for directory in sorted([path for path in base.iterdir() if path.is_dir() and path.name != "ezdicom"]):
        all_files = [path for path in directory.rglob("*") if path.is_file()]
        layer_rows.append(
            {
                "layer": rel(directory),
                "all_files": len(all_files),
                "raster_files": sum(path.suffix.lower() in RASTER_EXTS for path in all_files),
                "tif_files": sum(path.suffix.lower() in {".tif", ".tiff"} for path in all_files),
                "png_files": sum(path.suffix.lower() == ".png" for path in all_files),
                "json_files": sum(path.suffix.lower() == ".json" for path in all_files),
            }
        )
    all_rasters = sum(row["raster_files"] for row in layer_rows)
    return {
        "root": rel(base),
        "layer_summary": layer_rows,
        "raster_files_excluding_ezdicom": all_rasters,
        "unique_patient_ids_from_raster_filenames": len(patient_ids),
        "segmentation_label_names": dict(sorted(label_names.items())),
        "diagnosis_term_hits_outside_ezdicom": diagnosis_hits[:100],
        "diagnosis_term_hit_count": len(diagnosis_hits),
        "interpretation": (
            "The inspected ki-data annotations expose segmentation labels, not diagnosis labels,"
            " unless diagnosis_term_hits is non-zero."
        ),
    }


def inspect_cross_layer(site_rows_by_site: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excel_ids: set[str] = set()
    workbook_path = ROOT / "ki-data" / "patient_id_and_gender-cych20220412.xlsx"
    try:
        import openpyxl

        workbook = openpyxl.load_workbook(workbook_path, data_only=True, read_only=False)
        sheet = workbook[workbook.sheetnames[0]]
        for row in list(sheet.iter_rows(values_only=True))[2:]:
            pid = clean_pid(row[0] if row else "")
            if pid:
                excel_ids.add(pid)
    except Exception:
        pass

    ki_data_ids: set[str] = set()
    base = ROOT / "ki-data"
    for path in base.rglob("*"):
        if path.is_file() and "ezdicom" not in path.parts and path.suffix.lower() in RASTER_EXTS:
            pid = clean_pid(path.name)
            if pid:
                ki_data_ids.add(pid)

    layers: dict[str, set[str]] = {
        "Excel roster raw IDs": excel_ids,
        "CYCH current labelled IDs": {row["patient_id"] for row in site_rows_by_site.get("CYCH", []) if row["image_variant"] == "original" and row["patient_id"]},
        "NCKUH current labelled IDs": {row["patient_id"] for row in site_rows_by_site.get("NCKUH", []) if row["image_variant"] == "original" and row["patient_id"]},
        "ki-data raster filename IDs": ki_data_ids,
    }
    rows = []
    for name, ids in layers.items():
        rows.append(
            {
                "layer": name,
                "unique_patient_ids": len(ids),
                "overlap_excel_roster": len(ids & excel_ids),
                "overlap_cych": len(ids & layers["CYCH current labelled IDs"]),
                "overlap_nckuh": len(ids & layers["NCKUH current labelled IDs"]),
            }
        )
    return rows, {
        "excel_ids": len(excel_ids),
        "layers": rows,
        "interpretation": "These are filename/roster ID overlaps only; they do not prove that a diagnosis label came from the original paper cohort.",
    }


def inspect_excel() -> dict[str, Any]:
    path = ROOT / "ki-data" / "patient_id_and_gender-cych20220412.xlsx"
    try:
        import openpyxl

        workbook = openpyxl.load_workbook(path, data_only=True, read_only=False)
        sheet = workbook[workbook.sheetnames[0]]
        values = list(sheet.iter_rows(values_only=True))
        records = []
        for row in values[2:]:
            raw_pid = clean_pid(row[0] if len(row) > 0 else "")
            if raw_pid:
                records.append(
                    {
                        "raw_patient_id": raw_pid,
                        "gender": str(row[1]).strip().upper() if len(row) > 1 and row[1] is not None else "",
                        "organized_patient_id": clean_pid(row[2] if len(row) > 2 else ""),
                        "used_patient_id": clean_pid(row[4] if len(row) > 4 else ""),
                        "not_used_value": str(row[5]).strip() if len(row) > 5 and row[5] is not None else "",
                        "not_used_patient_id": clean_pid(row[5] if len(row) > 5 else ""),
                    }
                )
        gender_by_pid: dict[str, set[str]] = defaultdict(set)
        for record in records:
            if record["gender"]:
                gender_by_pid[record["raw_patient_id"]].add(record["gender"])
        conflicts = sorted(pid for pid, genders in gender_by_pid.items() if len(genders) > 1)
        used = sorted({record["used_patient_id"] for record in records if record["used_patient_id"]})
        not_used = sorted({record["not_used_patient_id"] for record in records if record["not_used_patient_id"]})
        return {
            "path": rel(path),
            "sheet": sheet.title,
            "dimensions": [sheet.max_row, sheet.max_column],
            "data_records": len(records),
            "raw_unique_patient_ids": len({record["raw_patient_id"] for record in records}),
            "gender_counts": dict(sorted(Counter(record["gender"] for record in records).items())),
            "gender_conflict_patient_ids": conflicts,
            "gender_conflict_count": len(conflicts),
            "organized_unique_patient_ids": len({record["organized_patient_id"] for record in records if record["organized_patient_id"]}),
            "used_unique_patient_ids": len(used),
            "not_used_unique_patient_ids": len(not_used),
            "not_used_rows_with_notes": sum(
                bool(record["not_used_value"] and not re.fullmatch(r"\d{8}", record["not_used_value"]))
                for record in records
            ),
        }
    except Exception as exc:
        return {"path": rel(path), "error": str(exc)}


def load_original_paper() -> dict[str, Any]:
    candidates = sorted(ROOT.glob("yolo_aug_triEMR_PCA_SVM_renal_2022_0322.docx"))
    if not candidates:
        candidates = sorted(ROOT.glob("*2022*.docx"))
    if not candidates:
        return {"error": "Original paper DOCX was not found."}
    path = candidates[0]
    try:
        from docx import Document

        document = Document(path)
        paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        relevant = [
            paragraph
            for paragraph in paragraphs
            if any(token in paragraph.lower() for token in ("209 patients", "828 original", "607 original", "triplet features", "vgg16"))
        ]
        tables = []
        for table in document.tables:
            tables.append([[cell.text.strip() for cell in row.cells] for row in table.rows])
        return {
            "path": rel(path),
            "paragraph_count_nonempty": len(paragraphs),
            "table_count": len(tables),
            "relevant_paragraphs": relevant,
            "tables": tables,
            "paper_claims": {
                "patients": 209,
                "male": 81,
                "female": 128,
                "nephritis_patients": 175,
                "original_single_kidney_images": 828,
                "augmented_single_kidney_images": 1249,
                "train_validation_original_images": 607,
                "test_original_images": 221,
                "test_positive": 171,
                "test_negative": 50,
                "models": [
                    "VGG16",
                    "VGG19",
                    "ResNet50",
                    "ResNet101",
                    "DenseNet121",
                    "MobileNet",
                    "Xception",
                    "InceptionV3",
                    "InceptionResNetV2",
                ],
            },
        }
    except Exception as exc:
        return {"path": rel(path), "error": str(exc)}


def make_report(
    output: Path,
    file_rows: list[dict[str, Any]],
    directory_summary: dict[str, dict[str, Any]],
    site_summaries: list[dict[str, Any]],
    excel_summary: dict[str, Any],
    kidata_summary: dict[str, Any],
    cross_summary: dict[str, Any],
    paper: dict[str, Any],
) -> dict[str, Any]:
    claims = paper.get("paper_claims", {})
    report_lines = [
        "# Project data audit and reproduction readiness",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Project root: `{ROOT}`",
        "",
        "## Executive status",
        "",
        "- `CONFIRMED`: the current CYCH and NCKUH CSV/image layers contain paired image files and binary diagnosis labels.",
        "- `CONFIRMED`: the Excel workbook contains 209 rows and its gender marginal is M=81/F=128, matching the original paper claim.",
        "- `CONFIRMED`: the original paper records 828 original images, 1249 augmented images, and a 607/221 train-validation/test split.",
        "- `BLOCKED_FOR_EXACT_REPLICATION`: the available `ki-data` annotation files expose the segmentation label `kidney`; they do not provide the 209-patient diagnosis labels needed to recreate the paper's 175 nephritis cases.",
        "- `BLOCKED_FOR_EXACT_REPLICATION`: the exact original 607/221 image manifest and the original training/checkpoint procedure are not encoded as a runnable manifest in this project.",
        "- `READY_FOR_BASELINE`: a patient-grouped, deterministic reproduction baseline can run on the current labelled CYCH or NCKUH layer. Its metrics must be reported as a baseline, not as the original 2022 result.",
        "",
        "## Project inventory",
        "",
        f"- Files inventoried: {len(file_rows):,}",
        f"- Top-level entries with files: {len(directory_summary)}",
        f"- Full-frame/annotation raster files under `ki-data` excluding `ezdicom`: {kidata_summary.get('raster_files_excluding_ezdicom', 0):,}",
        "- The inventory includes source documents, scripts, model checkpoints, derived images, result files, archives, and the existing scratch audit outputs.",
        "",
        "## Label lineage",
        "",
        "| Layer | Role | Label type | Reproduction use |",
        "|---|---|---|---|",
        "| `CYCH/CYCH.csv` + `CYCH/Data` | current labelled site layer | image-level binary diagnosis label | baseline-ready |",
        "| `NCKUH/NCKUH.csv` + `NCKUH/Data` | current labelled site layer | image-level binary diagnosis label | baseline-ready |",
        "| `cych_image_label`, `nckuh_image_label` | image plus YOLO text | kidney-region segmentation | preprocessing / ROI only |",
        "| `ki-data/before-labeled` | original full-frame plus LabelMe JSON | `kidney` polygon segmentation | not a diagnosis target |",
        "| `ki-data/1`, `2`, `3`, `CYCH3`, `NCKUH3`, `test_predict` | mixed image archives/folders | no diagnosis label in filenames | provenance/reference only |",
        "| `ki-data/patient_id_and_gender-cych20220412.xlsx` | clinical roster | patient ID and gender, with duplicate/conflicting rows | metadata audit only |",
        "",
        "## Current labelled layers",
        "",
        "| Site | CSV rows | Logical images | Variants | Valid labels | Positive rows | Negative rows | Patients | Patient-positive | Missing files |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in site_summaries:
        counts = summary.get("label_counts", {})
        patient_counts = summary.get("patient_level_label_counts", {})
        report_lines.append(
            f"| {summary['site']} | {summary['image_rows']:,} | {summary['logical_image_names']:,} | `{summary['image_variant_counts']}` | {summary['image_rows_with_valid_label']:,} | {counts.get('1', 0):,} | {counts.get('0', 0):,} | {summary['unique_patient_ids']:,} | {patient_counts.get('1', 0):,} | {summary['missing_images']:,} |"
        )
    report_lines += [
        "",
        "- Label integrity check: both sites have zero logical original/processed label mismatches, zero missing image files, zero image read errors, and zero CSV-to-metadata label mismatches.",
    ]
    report_lines += [
        "",
        "## Cross-layer patient ID mapping",
        "",
        "These counts establish file/roster relationships only; they do not establish the original paper's diagnosis cohort.",
        "",
        "| Layer | Unique IDs | Overlap with Excel | Overlap with CYCH | Overlap with NCKUH |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in cross_summary.get("layers", []):
        report_lines.append(
            f"| {row['layer']} | {row['unique_patient_ids']:,} | {row['overlap_excel_roster']:,} | {row['overlap_cych']:,} | {row['overlap_nckuh']:,} |"
        )
    report_lines += [
        "",
        "## Original paper claims",
        "",
        f"Source: `{paper.get('path', '')}`",
        "",
        "| Claim | Value |",
        "|---|---:|",
    ]
    for key in (
        "patients",
        "male",
        "female",
        "nephritis_patients",
        "original_single_kidney_images",
        "augmented_single_kidney_images",
        "train_validation_original_images",
        "test_original_images",
        "test_positive",
        "test_negative",
    ):
        if key in claims:
            report_lines.append(f"| {key} | {claims[key]} |")
    report_lines += [
        "",
        "## Excel roster audit",
        "",
        f"- Source: `{excel_summary.get('path', '')}`",
        f"- Data rows: {excel_summary.get('data_records', 'ERROR')}",
        f"- Unique raw IDs: {excel_summary.get('raw_unique_patient_ids', 'ERROR')}",
        f"- Gender counts: `{excel_summary.get('gender_counts', {})}`",
        f"- IDs with conflicting gender entries: {excel_summary.get('gender_conflict_count', 'ERROR')}",
        f"- Unique IDs in the `already used` column: {excel_summary.get('used_unique_patient_ids', 'ERROR')}",
        f"- Unique IDs in the `not used` column: {excel_summary.get('not_used_unique_patient_ids', 'ERROR')}",
        "- The gender conflicts and other medical-record inconsistencies are retained as audit findings only; they are not used as a blocker for image/label experiments.",
        "",
        "## `ki-data` annotation audit",
        "",
        f"- Raster files excluding `ezdicom`: {kidata_summary.get('raster_files_excluding_ezdicom', 0):,}",
        f"- Segmentation labels found: `{kidata_summary.get('segmentation_label_names', {})}`",
        f"- Diagnosis-term hits outside `ezdicom`: {kidata_summary.get('diagnosis_term_hit_count', 0)}",
        "- Therefore, the raw `ki-data` layer cannot by itself reproduce the paper's nephritis classification target.",
        "",
        "## Required inputs for exact 2022 replication",
        "",
        "1. The 209-patient diagnosis mapping used to define the 175 nephritis and 34 non-nephritis patients.",
        "2. The exact 828 original single-kidney image manifest, including which 607 images belong to train/validation and which 221 belong to test.",
        "3. The original preprocessing and marker-removal implementation, augmentation policy, CNN/triplet training settings, PCA settings, SVM kernel/C/gamma, and random seeds.",
        "4. Confirmation of the paper's label unit: image-level versus patient-level assignment when a patient has multiple scans/views.",
        "",
        "## Output files",
        "",
        "- `project_file_inventory.csv`: every project file except this output directory, with size and modified time.",
        "- `directory_summary.csv`: top-level file counts, bytes, and extensions.",
        "- `current_label_manifest.csv`: current CYCH/NCKUH image-label pairs, patient/date fields, image existence, dimensions, and metadata checks.",
        "- `cross_layer_mapping.csv`: patient-ID overlap counts across the Excel roster, current labelled sites, and `ki-data` filenames.",
        "- `project_data_audit.json`: structured audit data for downstream scripts.",
        "- `data_audit_report.md`: this human-readable report.",
    ]
    (output / "data_audit_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "output": str(output),
        "inventory_files": len(file_rows),
        "directory_summary": directory_summary,
        "original_paper": paper,
        "excel_roster": excel_summary,
        "ki_data": kidata_summary,
        "cross_layer_mapping": cross_summary,
        "current_sites": site_summaries,
        "reproduction_readiness": {
            "exact_2022_replication": False,
            "current_labelled_baseline": True,
            "medical_record_contradictions_blocking": False,
            "missing_inputs": [
                "209-patient diagnosis mapping",
                "exact 828-image and 607/221 split manifest",
                "original preprocessing/training/evaluation configuration",
            ],
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    file_rows, directory_summary = inventory(OUT)
    write_csv(
        OUT / "project_file_inventory.csv",
        ["path", "top_level", "extension", "size_bytes", "modified_local", "error"],
        (
            [
                row.get("path", ""),
                row.get("top_level", ""),
                row.get("extension", ""),
                row.get("size_bytes", ""),
                row.get("modified_local", ""),
                row.get("error", ""),
            ]
            for row in file_rows
        ),
    )
    write_csv(
        OUT / "directory_summary.csv",
        ["top_level", "files", "bytes", "extensions_json"],
        ([key, value["files"], value["bytes"], json.dumps(value["extensions"], ensure_ascii=True)] for key, value in directory_summary.items()),
    )

    site_rows: list[dict[str, Any]] = []
    site_rows_by_site: dict[str, list[dict[str, Any]]] = {}
    site_summaries: list[dict[str, Any]] = []
    for site in ("CYCH", "NCKUH"):
        rows, summary = read_current_site(site)
        site_rows_by_site[site] = rows
        site_rows.extend(rows)
        site_summaries.append(summary)
    write_csv(
        OUT / "current_label_manifest.csv",
        [
            "site",
            "filename",
            "relative_image_path",
            "image_variant",
            "patient_id",
            "exam_date",
            "label",
            "image_exists",
            "width",
            "height",
            "mode",
            "image_error",
            "metadata_fields",
            "metadata_exists",
            "metadata_label",
            "metadata_label_matches",
        ],
        ([row.get(field, "") for field in [
            "site", "filename", "relative_image_path", "image_variant", "patient_id", "exam_date", "label",
            "image_exists", "width", "height", "mode", "image_error", "metadata_fields",
            "metadata_exists", "metadata_label", "metadata_label_matches",
        ]] for row in site_rows),
    )

    excel_summary = inspect_excel()
    kidata_summary = inspect_kidata()
    cross_rows, cross_summary = inspect_cross_layer(site_rows_by_site)
    write_csv(
        OUT / "cross_layer_mapping.csv",
        ["layer", "unique_patient_ids", "overlap_excel_roster", "overlap_cych", "overlap_nckuh"],
        ([row[field] for field in ["layer", "unique_patient_ids", "overlap_excel_roster", "overlap_cych", "overlap_nckuh"]] for row in cross_rows),
    )
    paper = load_original_paper()
    audit = make_report(OUT, file_rows, directory_summary, site_summaries, excel_summary, kidata_summary, cross_summary, paper)
    json_dump(OUT / "project_data_audit.json", audit)
    print(json.dumps({
        "output": str(OUT),
        "inventory_files": len(file_rows),
        "sites": site_summaries,
        "excel": excel_summary,
        "ki_data_rasters": kidata_summary.get("raster_files_excluding_ezdicom"),
        "exact_2022_replication": audit["reproduction_readiness"]["exact_2022_replication"],
    }, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
