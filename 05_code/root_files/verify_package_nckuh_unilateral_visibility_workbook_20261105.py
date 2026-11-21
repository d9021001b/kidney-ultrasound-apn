"""Verify and package the blinded NCKUH unilateral visibility workbook."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(r"C:\Users\udoo_w2\Desktop\work_kidney_inflammation")
EXPERIMENT = ROOT / "02_experiments/NCKUH_unilateral_visibility_annotation_20261103"
PACKET = EXPERIMENT / "blinded_packet"
WORKBOOK = PACKET / "annotation_form_clinician_20261105.xlsx"
INSTRUCTIONS = PACKET / "回傳說明_繁體中文.md"
OUTPUT_ZIP = EXPERIMENT / "NCKUH_unilateral_visibility_BLINDED_CLINICIAN_20261105.zip"
AUDIT = EXPERIMENT / "clinician_workbook_audit_20261105.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    workbook = load_workbook(WORKBOOK, data_only=False)
    assert workbook.sheetnames == ["標註表", "填寫說明"]
    sheet = workbook["標註表"]
    assert sheet.max_row == 87 and sheet.max_column == 13
    case_ids = [sheet.cell(row, 1).value for row in range(5, 88)]
    assert case_ids == [f"U{i:03d}" for i in range(1, 84)]
    assert all(
        sheet.cell(row, column).value in (None, "")
        for row in range(5, 88)
        for column in range(2, 14)
    )
    assert list(sheet.tables.values())[0].ref == "A4:M87"

    validation_ranges = {str(validation.sqref) for validation in sheet.data_validations.dataValidation}
    expected_validation_ranges = {"B5:E87", "F5:G87", "H5:I87", "J5:K87", "L5:L87"}
    assert validation_ranges == expected_validation_ranges

    pngs = sorted(PACKET.glob("U[0-9][0-9][0-9].png"))
    assert [path.stem for path in pngs] == case_ids
    assert not any("private" in path.name.lower() for path in pngs)

    members = [WORKBOOK, INSTRUCTIONS, *pngs]
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in members:
            archive.write(path, arcname=path.name)

    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        names = archive.namelist()
        assert len(names) == 85
        assert not any("private" in name.lower() or "dmsa_key" in name.lower() for name in names)
        assert "annotation_form_clinician_20261105.xlsx" in names
        assert "回傳說明_繁體中文.md" in names

    audit = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "blinded_cases": len(case_ids),
        "png_files": len(pngs),
        "workbook_sheets": workbook.sheetnames,
        "blank_annotation_cells": 83 * 12,
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
