"""Build a blinded side-specific ultrasound visibility review packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import nckuh_dmsa_fullcohort_clinical_features_20261103 as C
import nckuh_dmsa_fullcohort_fullframe_20261103 as F
from nckuh_only_study_20261102 import write_json


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_unilateral_visibility_annotation_20261103"
BLIND = OUT / "blinded_packet"
PRIVATE = OUT / "private_key"
SEED = 20261103
SLOTS = (("left", "L", "Left kidney - longitudinal"),
         ("left", "T", "Left kidney - transverse"),
         ("right", "L", "Right kidney - longitudinal"),
         ("right", "T", "Right kidney - transverse"))


def font(size: int):
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def masked_image(identifier: str, normalized: bool) -> Image.Image:
    mask = cv2.imread(str(F.OUT / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE) > 0
    if normalized:
        gray = cv2.imread(str(F.OUT / "percentile" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        bgr = cv2.imread(str(F.OUT / "clean_rgb" / f"{identifier}.png"), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb[~mask] = 0
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (255, 64, 64), 2)
    return Image.fromarray(rgb)


def fit_tile(image: Image.Image, size=(720, 460)) -> Image.Image:
    output = Image.new("RGB", size, "black")
    current = image.copy()
    current.thumbnail((size[0] - 20, size[1] - 20), Image.Resampling.LANCZOS)
    output.paste(current, ((size[0] - current.width) // 2, (size[1] - current.height) // 2))
    return output


def case_page(case_id: str, selected: dict[tuple[str, str], str]) -> Image.Image:
    page = Image.new("RGB", (1500, 2170), "white")
    draw = ImageDraw.Draw(page)
    draw.text((40, 25), f"NCKUH unilateral review - {case_id}", fill="black", font=font(34))
    draw.text((40, 75), "Red line: supplied kidney contour. Left: original color ROI. Right: within-mask P1-P99 image.",
              fill="#333333", font=font(20))
    top = 125
    for index, (side, suffix, label) in enumerate(SLOTS):
        y = top + index * 500
        draw.text((40, y), label, fill="black", font=font(24))
        identifier = selected.get((side, suffix))
        if identifier is None:
            draw.rectangle((40, y + 40, 1460, y + 490), fill="#eeeeee")
            draw.text((600, y + 245), "View unavailable", fill="#666666", font=font(24))
            continue
        original = fit_tile(masked_image(identifier, False))
        normalized = fit_tile(masked_image(identifier, True))
        page.paste(original, (40, y + 40)); page.paste(normalized, (740, y + 40))
    return page


def selected_views(images: pd.DataFrame, alias: str) -> dict[tuple[str, str], str]:
    output = {}
    patient = images[images.patient_alias.eq(alias)]
    for side, suffix, _ in SLOTS:
        rows = patient[patient.side.eq(side) & patient.view.str.endswith(suffix)]
        if len(rows):
            output[(side, suffix)] = rows.image_id.iloc[0]
    return output


def build() -> None:
    if OUT.exists():
        raise FileExistsError("Preserve completed visibility packet")
    BLIND.mkdir(parents=True); PRIVATE.mkdir(parents=True)
    patients, _, images = C.load_reference()
    unilateral = patients[patients.DMSA_APN.isin(["L", "R"])].copy()
    rng = np.random.default_rng(SEED)
    unilateral = unilateral.iloc[rng.permutation(len(unilateral))].reset_index(drop=True)
    public_rows, private_rows, manifest_rows = [], [], []
    for index, patient in enumerate(unilateral.itertuples(index=False), start=1):
        case_id = f"U{index:03d}"
        selected = selected_views(images, patient.patient_alias)
        page = case_page(case_id, selected)
        page_path = BLIND / f"{case_id}.png"
        page.save(page_path, dpi=(300, 300), optimize=True)
        public_rows.append({
            "case_id": case_id,
            "left_laterality_confirmed": "",
            "right_laterality_confirmed": "",
            "left_visible_abnormality": "",
            "right_visible_abnormality": "",
            "left_finding_type": "",
            "right_finding_type": "",
            "left_region": "",
            "right_region": "",
            "left_confidence_1_to_5": "",
            "right_confidence_1_to_5": "",
            "image_quality_exclusion": "",
            "reviewer_notes": "",
        })
        private_rows.append({"case_id": case_id, "patient_alias": patient.patient_alias,
                             "DMSA_APN": patient.DMSA_APN})
        for (side, suffix), identifier in selected.items():
            manifest_rows.append({"case_id": case_id, "side": side, "view_type": suffix,
                                  "page_file": page_path.name, "image_slot": f"{side}_{suffix}"})
    pd.DataFrame(public_rows).to_csv(BLIND / "annotation_form.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest_rows).to_csv(BLIND / "page_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(private_rows).to_csv(PRIVATE / "case_to_DMSA_key_PRIVATE.csv", index=False, encoding="utf-8-sig")
    instructions = """# Blinded unilateral APN ultrasound review

Review every case without opening the separate private key directory.

For each kidney, record whether a visible abnormality is present. Suggested finding types are focal hyperechoic, focal hypoechoic, swelling, reduced corticomedullary differentiation, perfusion abnormality, collecting-system abnormality, other, or none. Region may be upper pole, middle, lower pole, diffuse, or uncertain. Laterality confirmation should be yes, no, or uncertain. Confidence is 1 to 5.

The red boundary is the supplied kidney contour. The left panel preserves color; the right panel is within-mask P1-P99 grayscale normalization. These images are for visibility and data-linkage review, not a replacement for the DMSA reference diagnosis.
"""
    (BLIND / "README.md").write_text(instructions, encoding="utf-8")
    archive = shutil.make_archive(str(OUT / "NCKUH_unilateral_visibility_BLINDED_20261103"), "zip", BLIND)
    hashes = {}
    for path in sorted(BLIND.iterdir()):
        if path.is_file():
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(OUT / "packet_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "unilateral_cases": len(unilateral),
        "left_DMSA_cases_private": int(unilateral.DMSA_APN.eq("L").sum()),
        "right_DMSA_cases_private": int(unilateral.DMSA_APN.eq("R").sum()),
        "blinded_pages": len(list(BLIND.glob("U*.png"))),
        "public_files_contain_DMSA_column": False,
        "public_files_contain_patient_alias_column": False,
        "private_key_separate": True,
        "archive": Path(archive).name,
        "public_sha256": hashes,
    })
    print(f"Built {len(unilateral)} blinded unilateral review pages", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build",))
    args = parser.parse_args(); build()
