"""Split predicted dual-view masks into auditable, non-independent instances."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from finetune_nckuh_383_linear_segmentation_20261112 import OUT as FINETUNE, digest


ROOT = FINETUNE / "missing_linear_candidates"
OUT = ROOT / "component_review"


def main():
    if (OUT / "audit.json").exists():
        raise FileExistsError("Preserve completed component audit")
    source = json.loads((ROOT / "audit.json").read_text(encoding="utf-8"))
    assert digest(ROOT / "candidate_qc_deidentified.csv") == source["candidate_qc_sha256"]
    frame = pd.read_csv(ROOT / "candidate_qc_deidentified.csv")
    assert len(frame) == 138 and frame.candidate_id.is_unique
    (OUT / "masks").mkdir(parents=True, exist_ok=True)
    records, source_records = [], []
    for row in frame.itertuples(index=False):
        mask = cv2.imread(str(ROOT / "masks" / f"{row.candidate_id}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        assert mask is not None and mask.shape == (row.height, row.width), (
            row.candidate_id, None if mask is None else mask.shape, row.height, row.width
        )
        count, labels, stats, centers = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        components = sorted(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
        major = components[:2]
        dual = (len(major) == 2 and
                stats[major[1], cv2.CC_STAT_AREA] >= .5 * stats[major[0], cv2.CC_STAT_AREA] and
                (centers[major[0], 0] < row.width / 2) != (centers[major[1], 0] < row.width / 2))
        source_records.append({"candidate_id": row.candidate_id, "patient_alias": row.patient_alias,
                               "role": row.role, "side": row.side, "connected_components": len(components),
                               "dual_view_candidate": bool(dual),
                               "layout_review_status": "pending_manual_review"})
        for rank, component in enumerate(components, 1):
            instance_id = f"{row.candidate_id}_C{rank}"
            binary = (labels == component).astype(np.uint8)
            assert cv2.imwrite(str(OUT / "masks" / f"{instance_id}.png"), binary * 255)
            records.append({"instance_id": instance_id, "candidate_id": row.candidate_id,
                            "patient_alias": row.patient_alias, "role": row.role, "side": row.side,
                            "component_rank_by_area": rank,
                            "panel_half_by_centroid": "left" if centers[component, 0] < row.width / 2 else "right",
                            "mask_pixels": int(stats[component, cv2.CC_STAT_AREA]),
                            "relative_area_to_largest": float(stats[component, cv2.CC_STAT_AREA] /
                                                              stats[major[0], cv2.CC_STAT_AREA]),
                            "bbox_xywh": json.dumps([int(v) for v in stats[component, :4]]),
                            "review_status": "pending_manual_review"})
    sources = pd.DataFrame(source_records)
    instances = pd.DataFrame(records)
    assert len(sources) == 138 and len(instances) >= 250
    sources.to_csv(OUT / "source_layout_deidentified.csv", index=False)
    instances.to_csv(OUT / "component_manifest_deidentified.csv", index=False)
    audit = {"source_candidate_sha256": source["candidate_qc_sha256"],
             "source_images": len(sources), "candidate_components": len(instances),
             "source_component_counts": sources.connected_components.value_counts().sort_index().to_dict(),
             "dual_view_candidates": int(sources.dual_view_candidate.sum()),
             "dual_view_patients": int(sources.loc[sources.dual_view_candidate, "patient_alias"].nunique()),
             "zero_mask_sources": int(sources.connected_components.eq(0).sum()),
             "source_layout_sha256": digest(OUT / "source_layout_deidentified.csv"),
             "component_manifest_sha256": digest(OUT / "component_manifest_deidentified.csv"),
             "apn_test_scored": False,
             "policy": "Components are two views of a source frame, not independent patients or kidneys. These inferred masks need anatomical manual QC before APN feature extraction."}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
