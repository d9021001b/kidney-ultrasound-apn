"""Check whether existing image-mask panels correspond to the same raw TIFF pane."""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nckuh_dmsa_fullcohort_fullframe_20261103 import parse_source_key, split_panel
from validate_nckuh_tiff_red_contour_20261111 import red_contour_mask


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
OUT = ROOT / "02_experiments/NCKUH_revised_383_clinical_audit_20261111"


def correlation(a, b):
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    y = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    if x.std() < 1 or y.std() < 1:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main():
    raw = defaultdict(list)
    for folder in ("1", "2", "3"):
        for path in (ROOT / "03_data_models/ki-data" / folder).glob("*.tif"):
            key = parse_source_key(path)
            if key is not None:
                raw[key].append(path)
    sources = pd.read_csv(REGISTRY / "source_linkage_private.csv", dtype=str)
    records = []
    for row in sources.itertuples(index=False):
        panel_path = Path(row.source_path)
        key = parse_source_key(panel_path)
        if key not in raw:
            continue
        frame, mask = split_panel(panel_path)
        best = None
        for raw_path in raw[key]:
            image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[1] % 2:
                continue
            width = image.shape[1] // 2
            for pane, raw_frame in (("left", image[:, :width]), ("right", image[:, width:])):
                corr = correlation(frame, raw_frame)
                if best is None or corr > best[0]:
                    best = (corr, pane, raw_frame)
        if best is None:
            continue
        corr, pane, raw_frame = best
        red_iou = None
        red_pixels = 0
        if corr >= 0.9 and pane == "right" and frame.shape[:2] == raw_frame.shape[:2]:
            red_mask, _ = red_contour_mask(raw_frame)
            red_pixels = int(red_mask.sum())
            if red_pixels:
                intersection = np.count_nonzero(mask & red_mask)
                union = np.count_nonzero(mask | red_mask)
                red_iou = intersection / union if union else 0.0
        records.append({"image_id": row.image_id, "best_pane": pane,
                        "frame_correlation": corr, "red_pixels": red_pixels,
                        "panel_vs_red_iou": red_iou})
    frame = pd.DataFrame(records)
    summary = {
        "panels_with_raw_tiff_same_key": len(frame),
        "best_pane_counts": frame.best_pane.value_counts().to_dict(),
        "correlation_ge_0_9": int(frame.frame_correlation.ge(.9).sum()),
        "correlation_median": float(frame.frame_correlation.median()),
        "same_right_pane_red_contour_comparisons": int(frame.panel_vs_red_iou.notna().sum()),
        "panel_vs_red_median_iou": float(frame.panel_vs_red_iou.median()) if frame.panel_vs_red_iou.notna().any() else None,
        "caveat": "Same-key/frame correlation and supplied red-outline transcription are not independent expert segmentation truth."
    }
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "panel_raw_alignment.csv", index=False)
    (OUT / "panel_raw_alignment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
