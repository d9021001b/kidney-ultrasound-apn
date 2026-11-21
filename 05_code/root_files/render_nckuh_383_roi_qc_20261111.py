"""Render a label-blind, train-only ROI contact sheet for contour QA."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
CACHE = ROOT / "02_experiments/NCKUH_revised_383_contour_cache_20261111"


def main():
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv")
    train = images.merge(roles[["patient_alias", "role"]], on="patient_alias")
    train = train[train.role.eq("train")].reset_index(drop=True)
    assert len(train) == 796
    indices = np.linspace(0, len(train) - 1, 20).round().astype(int)
    sheet = np.full((5 * 220, 4 * 220, 3), 25, np.uint8)
    for n, index in enumerate(indices):
        image_id = train.image_id.iloc[index]
        gray = cv2.imread(str(CACHE / "percentile" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(CACHE / "masks" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        assert gray is not None and mask is not None and gray.shape == mask.shape
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (20, 180, 255), 2)
        scale = min(190 / canvas.shape[0], 200 / canvas.shape[1])
        new = cv2.resize(canvas, (max(1, round(canvas.shape[1] * scale)),
                                  max(1, round(canvas.shape[0] * scale))))
        tile = np.full((220, 220, 3), 25, np.uint8)
        y, x = 25 + (190 - new.shape[0]) // 2, (220 - new.shape[1]) // 2
        tile[y:y + new.shape[0], x:x + new.shape[1]] = new
        cv2.putText(tile, image_id, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .45,
                    (230, 230, 230), 1, cv2.LINE_AA)
        sheet[(n // 4) * 220:(n // 4 + 1) * 220, (n % 4) * 220:(n % 4 + 1) * 220] = tile
    target = CACHE / "roi_qc_train_contact.png"
    cv2.imwrite(str(target), sheet)
    print(target)


if __name__ == "__main__":
    main()
