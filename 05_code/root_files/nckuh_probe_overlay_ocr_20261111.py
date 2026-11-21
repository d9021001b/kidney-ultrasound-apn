"""Classify NCKUH probe overlays without persisting OCR text or identifiers."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

import cv2
import easyocr
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "2004-2019-NCKUH-亞大.xlsx"
IMAGE_ROOT = ROOT / "03_data_models/ki-data"
OUT = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
FOLDERS = ("1", "2", "3", "NCKUH3", "test_predict")
EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
PROBE_RE = re.compile(r"\b(convex|linear)\b", re.IGNORECASE)


def source_images():
    ws = load_workbook(SOURCE, read_only=True, data_only=True)["完整資料彙整"]
    patients = {}
    for row in list(ws.iter_rows(values_only=True))[1:]:
        if row[0] is not None:
            patients[str(row[0]).strip()] = {x.strip() for x in str(row[6]).split(",")}
    assert len(patients) == 383
    paths = defaultdict(list)
    for folder in FOLDERS:
        for path in sorted((IMAGE_ROOT / folder).iterdir()):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                patient_id = path.name.split("_", 1)[0]
                if folder in patients.get(patient_id, ()):
                    paths[folder].append(path)
    return paths


def classify(reader, path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return "unreadable", 0.0
    h, w = image.shape[:2]
    header = image[:min(95, max(45, int(.15 * h))), :min(w, 1100)]
    found = []
    for _, text, confidence in reader.readtext(header, detail=1):
        for match in PROBE_RE.finditer(text):
            if confidence >= .5:
                found.append((match.group(1).lower(), float(confidence)))
    kinds = {kind for kind, _ in found}
    if len(kinds) == 1:
        return next(iter(kinds)), max(confidence for _, confidence in found)
    return ("ambiguous" if kinds else "unknown"), 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-folder", type=int, default=12)
    parser.add_argument("--folders", nargs="+", choices=FOLDERS, default=list(FOLDERS))
    args = parser.parse_args()
    reader = easyocr.Reader(["en"], gpu=True, download_enabled=False, verbose=False)
    paths = source_images()
    counts = Counter()
    by_folder = defaultdict(Counter)
    records = []
    for folder in args.folders:
        candidates = paths[folder]
        if args.limit_per_folder == 0 or len(candidates) <= args.limit_per_folder:
            selected = candidates
        elif args.limit_per_folder == 1:
            selected = [candidates[len(candidates) // 2]]
        else:
            selected = [candidates[round(i * (len(candidates) - 1) / (args.limit_per_folder - 1))]
                        for i in range(args.limit_per_folder)]
        for path in selected:
            label, confidence = classify(reader, path)
            counts[label] += 1
            by_folder[folder][label] += 1
            records.append({"folder": folder, "file_ordinal": candidates.index(path),
                            "probe": label, "confidence": round(confidence, 4)})
            if label in ("convex", "linear"):
                assert confidence >= .5
    result = {
        "source_sha256": sha256(SOURCE.read_bytes()).hexdigest(),
        "limit_per_folder": args.limit_per_folder,
        "total_examined": sum(counts.values()),
        "probe_counts": dict(sorted(counts.items())),
        "by_folder": {folder: dict(sorted(by_folder[folder].items())) for folder in args.folders},
        "deidentified_records": records,
        "method": "EasyOCR on top 45-95 pixels, explicit convex/linear token only, confidence >= 0.5; unrecognized images remain unknown",
        "limitations": "OCR token is unverified unless manually reviewed; no performance comparison yet",
    }
    suffix = "full" if args.limit_per_folder == 0 else f"sample{args.limit_per_folder}"
    if args.folders != list(FOLDERS):
        suffix += "_" + "-".join(args.folders)
    target = OUT / f"probe_overlay_ocr_{suffix}.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "deidentified_records"}, indent=2))


if __name__ == "__main__":
    main()
