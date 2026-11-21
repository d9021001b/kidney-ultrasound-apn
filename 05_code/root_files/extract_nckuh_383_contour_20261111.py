"""Extract label-blind kidney ROI caches for the revised NCKUH roster."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nckuh_dmsa_fullcohort_fullframe_20261103 import (
    crop_masked, load_polygon, percentile_gray, split_panel,
)
from prepare_nckuh_383_contour_cohort_20261111 import ROOT, OUT as REGISTRY, SOURCE, digest


OUT = ROOT / "02_experiments/NCKUH_revised_383_contour_cache_20261111"


def main():
    lock = json.loads((REGISTRY / "cohort_lock.json").read_text())
    assert digest(SOURCE) == lock["source_workbook_sha256"]
    for name, expected in lock["files_sha256"].items():
        assert digest(REGISTRY / name) == expected, name
    public = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    private = pd.read_csv(REGISTRY / "source_linkage_private.csv", keep_default_na=False)
    assert len(public) == len(private) == 1362
    assert public.image_id.tolist() == private.image_id.tolist()
    if (OUT / "extraction_audit.json").exists():
        raise FileExistsError("Preserve completed contour extraction")
    for name in ("clean_rgb", "masks", "percentile"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    rows = []
    for public_row, private_row in zip(public.itertuples(index=False), private.itertuples(index=False)):
        source = Path(private_row.source_path)
        if public_row.source_type == "panel_with_mask":
            image, mask = split_panel(source)
        elif public_row.source_type == "labelme_polygon":
            image, mask = load_polygon(source, Path(private_row.annotation_path))
        else:
            raise ValueError(f"Unregistered source type: {public_row.source_type}")
        roi, roi_mask = crop_masked(image, mask)
        gray = percentile_gray(roi, roi_mask)
        image_id = public_row.image_id
        if not cv2.imwrite(str(OUT / "clean_rgb" / f"{image_id}.png"), roi):
            raise OSError(f"Could not write {image_id} RGB")
        if not cv2.imwrite(str(OUT / "masks" / f"{image_id}.png"), roi_mask * 255):
            raise OSError(f"Could not write {image_id} mask")
        if not cv2.imwrite(str(OUT / "percentile" / f"{image_id}.png"), gray):
            raise OSError(f"Could not write {image_id} percentile image")
        rows.append({"image_id": image_id, "patient_alias": public_row.patient_alias,
                     "source_type": public_row.source_type, "height": image.shape[0],
                     "width": image.shape[1], "roi_height": roi.shape[0], "roi_width": roi.shape[1],
                     "mask_pixels": int(mask.sum()),
                     "mask_fraction_in_source_frame": float(mask.mean()),
                     "outside_mask_nonzero": int(np.count_nonzero(gray[roi_mask == 0]))})
    qc = pd.DataFrame(rows)
    qc.to_csv(OUT / "extraction_qc.csv", index=False)
    assert len(qc) == 1362 and qc.image_id.is_unique
    assert qc.outside_mask_nonzero.sum() == 0
    audit = {
        "registered_manifest_sha256": lock["files_sha256"]["contour_image_manifest.csv"],
        "images": len(qc), "patients": int(qc.patient_alias.nunique()),
        "source_counts": qc.source_type.value_counts().to_dict(),
        "mask_fraction_min_median_max": [float(qc.mask_fraction_in_source_frame.min()),
                                         float(qc.mask_fraction_in_source_frame.median()),
                                         float(qc.mask_fraction_in_source_frame.max())],
        "mask_fraction_below_0_01": int((qc.mask_fraction_in_source_frame < .01).sum()),
        "mask_fraction_above_0_40": int((qc.mask_fraction_in_source_frame > .4).sum()),
        "outside_mask_nonzero": int(qc.outside_mask_nonzero.sum()),
        "labels_used_in_extraction": [],
        "caveat": "Structural mask QC only; not contour IoU or anatomical correctness",
    }
    (OUT / "extraction_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
