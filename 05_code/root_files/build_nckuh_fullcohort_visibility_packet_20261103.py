"""Build a blinded all-cohort renal-ultrasound lesion-visibility review packet."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

import build_nckuh_unilateral_visibility_packet_20261103 as U
import nckuh_dmsa_fullcohort_clinical_features_20261103 as C
import nckuh_dmsa_fullcohort_fullframe_20261103 as F


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_fullcohort_visibility_annotation_20261103"
BLIND, PRIVATE = OUT / "blinded_packet", OUT / "private_key"
SEED = 20261103


def case_page(case_id: str, selected: dict[tuple[str, str], str]) -> Image.Image:
    page = Image.new("RGB", (1500, 2170), "white")
    draw = ImageDraw.Draw(page)
    draw.text((40, 25), f"NCKUH renal ultrasound review - {case_id}", fill="black", font=U.font(34))
    draw.text((40, 75), "Red line: supplied kidney contour. Left: original color ROI. Right: within-mask P1-P99 image.", fill="#333333", font=U.font(20))
    for index, (side, suffix, label) in enumerate(U.SLOTS):
        y = 125 + index * 500
        draw.text((40, y), label, fill="black", font=U.font(24))
        identifier = selected.get((side, suffix))
        if identifier is None:
            draw.rectangle((40, y + 40, 1460, y + 490), fill="#eeeeee")
            draw.text((600, y + 245), "View unavailable", fill="#666666", font=U.font(24))
            continue
        page.paste(U.fit_tile(U.masked_image(identifier, False)), (40, y + 40))
        page.paste(U.fit_tile(U.masked_image(identifier, True)), (740, y + 40))
    return page


def build() -> None:
    transient_logs = {"build.stdout.log", "build.stderr.log"}
    if OUT.exists() and any(path.name not in transient_logs for path in OUT.iterdir()):
        raise FileExistsError("Preserve completed full-cohort visibility packet")
    BLIND.mkdir(parents=True); PRIVATE.mkdir(parents=True)
    patients, _, images = C.load_reference()
    patients = patients.iloc[__import__("numpy").random.default_rng(SEED).permutation(len(patients))].reset_index(drop=True)
    public, private, manifest = [], [], []
    for number, patient in enumerate(patients.itertuples(index=False), start=1):
        case_id = f"V{number:03d}"
        selected = U.selected_views(images, patient.patient_alias)
        page_path = BLIND / f"{case_id}.png"
        case_page(case_id, selected).save(page_path, dpi=(300, 300), optimize=True)
        public.append({"case_id": case_id, "left_laterality_confirmed": "", "right_laterality_confirmed": "",
                       "left_visible_abnormality": "", "right_visible_abnormality": "",
                       "left_finding_type": "", "right_finding_type": "", "left_region": "", "right_region": "",
                       "left_confidence_1_to_5": "", "right_confidence_1_to_5": "", "image_quality_exclusion": "", "reviewer_notes": ""})
        private.append({"case_id": case_id, "patient_alias": patient.patient_alias, "DMSA_APN": patient.DMSA_APN})
        manifest.extend({"case_id": case_id, "side": side, "view_type": suffix, "page_file": page_path.name, "image_slot": f"{side}_{suffix}"}
                        for (side, suffix) in selected)
    pd.DataFrame(public).to_csv(BLIND / "annotation_form.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest).to_csv(BLIND / "page_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(private).to_csv(PRIVATE / "case_to_DMSA_key_PRIVATE.csv", index=False, encoding="utf-8-sig")
    (BLIND / "README.md").write_text("""# Blinded all-cohort renal ultrasound review

Review without opening `private_key`. For each side record laterality confidence, whether a visible abnormality is present, its type (focal hyperechoic/hypoechoic, swelling, reduced corticomedullary differentiation, perfusion, collecting-system, other, or none), region, confidence (1–5), and any quality exclusion. Do not infer or record DMSA status.

The red boundary is the supplied kidney contour. The left panel retains the image colour information; the right panel is within-mask P1–P99 grayscale normalization. These images support visibility and linkage review, not a replacement for DMSA diagnosis.
""", encoding="utf-8")
    archive = shutil.make_archive(str(OUT / "NCKUH_fullcohort_visibility_BLINDED_20261103"), "zip", BLIND)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(BLIND.iterdir()) if path.is_file()}
    (OUT / "packet_audit.json").write_text(json.dumps({
        "completed_utc": datetime.now(timezone.utc).isoformat(), "cases": len(patients), "blinded_pages": len(list(BLIND.glob("V*.png"))),
        "public_files_contain_DMSA_column": False, "public_files_contain_patient_alias_column": False, "private_key_separate": True,
        "archive": Path(archive).name, "public_sha256": hashes,
    }, indent=2), encoding="utf-8")
    print(f"Built {len(patients)} blinded full-cohort review pages", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build",))
    globals()[parser.parse_args().stage]()
