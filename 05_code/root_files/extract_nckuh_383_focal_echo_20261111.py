"""Label-blind polar, radial, and local echo features in supplied kidney masks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nckuh_dmsa_fullcohort_clinical_features_20261103 import focal_features
from nckuh_dmsa_fullcohort_fullframe_20261103 import split_panel
from prepare_nckuh_383_contour_cohort_20261111 import digest


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
OUT = ROOT / "02_experiments/NCKUH_revised_383_focal_echo_20261111"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "focal_echo.npy").exists():
        raise FileExistsError("Preserve completed focal feature extraction")
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    sources = pd.read_csv(REGISTRY / "source_linkage_private.csv", dtype=str)
    sources = sources.set_index("image_id").reindex(images.image_id)
    assert len(images) == 1362 and sources.source_path.notna().all()
    rows = []
    for i, path in enumerate(sources.source_path):
        frame, mask = split_panel(Path(path))
        rows.append(focal_features(frame, mask))
        if i % 100 == 0:
            print(f"Focal echo {i + 1}/{len(images)}", flush=True)
    values = np.asarray(rows, dtype=np.float32)
    assert values.shape[0] == len(images) and np.isfinite(values).all()
    np.save(OUT / "focal_echo.npy", values)
    audit = {"images": len(images), "features_per_image": values.shape[1],
             "image_manifest_sha256": digest(REGISTRY / "contour_image_manifest.csv"),
             "features_sha256": digest(OUT / "focal_echo.npy"),
             "labels_used_in_extraction": [],
             "features": "published pediatric APN motivated polar/radial echo, local dispersion and multiscale Laplacian summaries; source ROI masks",
             "source": "nckuh_dmsa_fullcohort_clinical_features_20261103.focal_features"}
    (OUT / "extraction_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
