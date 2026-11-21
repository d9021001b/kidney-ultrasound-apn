"""Verify and package the blinded NCKUH full-cohort kidney review workbook."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(r"C:\Users\udoo_w2\Desktop\work_kidney_inflammation")
EXPERIMENT = ROOT / "02_experiments/NCKUH_fullcohort_visibility_annotation_20261103"
PACKET = EXPERIMENT / "blinded_packet"
WORKBOOK = PACKET / "annotation_form_clinician_fullcohort_20261107.xlsx"
OUTPUT_ZIP = EXPERIMENT / "NCKUH_fullcohort_visibility_BLINDED_CLINICIAN_20261107.zip"
AUDIT = EXPERIMENT / "clinician_workbook_audit_20261107.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    workbook = load_workbook(WORKBOOK, data_only=False)
    assert workbook.sheetnames == ["腎臟標註", "填寫說明"]
    sheet = workbook["腎臟標註"]
    assert sheet.max_row == 424 and sheet.max_column == 16
    rows = [
        (sheet.cell(row, 1).value, sheet.cell(row, 2).value, sheet.cell(row, 3).value)
        for row in range(5, 425)
    ]
    expected = [(f"V{case:03d}", side) for case in range(1, 211) for side in ("left", "right")]
    assert [(case, side) for case, side, _ in rows] == expected
    assert all(value in {"L", "T", "L+T", "none"} for _, _, value in rows)
    assert sum(value == "none" for _, _, value in rows) == 6
    assert all(
        sheet.cell(row, column).value in (None, "")
        for row in range(5, 425)
        for column in range(4, 17)
    )
    assert list(sheet.tables.values())[0].ref == "A4:P424"

    validation_ranges = {str(validation.sqref) for validation in sheet.data_validations.dataValidation}
    expected_ranges = {
        "D5:D424", "E5:E424", "F5:F424", "G5:G424", "H5:H424",
        "I5:I424", "J5:J424", "K5:K424", "L5:L424", "M5:M424",
        "N5:N424", "O5:O424",
    }
    assert validation_ranges == expected_ranges

    pngs = sorted(PACKET.glob("V[0-9][0-9][0-9].png"))
    assert [path.stem for path in pngs] == [f"V{case:03d}" for case in range(1, 211)]
    assert not any("private" in path.name.lower() or "dmsa" in path.name.lower() for path in pngs)

    readme = PACKET / "README.md"
    members = [WORKBOOK, readme, *pngs]
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in members:
            archive.write(path, arcname=path.name)

    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        names = archive.namelist()
        assert len(names) == 212
        assert not any("private" in name.lower() or "dmsa" in name.lower() for name in names)
        assert WORKBOOK.name in names and readme.name in names

    audit = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "blinded_cases": 210,
        "kidney_rows": 420,
        "imaged_kidneys": 414,
        "unimaged_kidney_rows": 6,
        "png_files": len(pngs),
        "workbook_sheets": workbook.sheetnames,
        "blank_annotation_cells": 420 * 13,
        "data_validation_ranges": sorted(validation_ranges),
        "private_key_files_packaged": 0,
        "zip_members": len(names),
        "workbook_sha256": sha256(WORKBOOK),
        "zip_sha256": sha256(OUTPUT_ZIP),
    }
    AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
