"""Flag near-identical left/right kidney ROIs within training patients only."""

from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nckuh_dmsa_fullcohort_fullframe_20261103 import tight_canvas


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
CACHE = ROOT / "02_experiments/NCKUH_revised_383_contour_cache_20261111"


def image_hash(image):
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].ravel()[1:]
    return low > np.median(low)


def similarity(left, right):
    a = left.ravel().astype(np.float32)
    candidates = []
    for flipped in (False, True):
        b_image = np.fliplr(right) if flipped else right
        b = b_image.ravel().astype(np.float32)
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else 0.0
        distance = int(np.count_nonzero(image_hash(left) != image_hash(b_image)))
        candidates.append((corr, distance, flipped))
    return max(candidates, key=lambda item: (item[0], -item[1]))


def main():
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv")
    train = images.merge(roles[["patient_alias", "role"]], on="patient_alias")
    train = train[train.role.eq("train")].reset_index(drop=True)
    assert len(train) == 796
    canvases = {}
    for image_id in train.image_id:
        gray = cv2.imread(str(CACHE / "percentile" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(CACHE / "masks" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE) > 0
        canvas, _ = tight_canvas(gray, mask, size=224)
        canvases[image_id] = canvas[..., 0]
    flagged = []
    compared = 0
    for alias, patient in train.groupby("patient_alias"):
        left = patient[patient.side.eq("left")]
        right = patient[patient.side.eq("right")]
        for a in left.itertuples(index=False):
            for b in right.itertuples(index=False):
                compared += 1
                corr, distance, flipped = similarity(canvases[a.image_id], canvases[b.image_id])
                if corr >= .9 or distance <= 6:
                    flagged.append({"patient_alias": alias, "left_image_id": a.image_id,
                                    "right_image_id": b.image_id, "left_view": a.view,
                                    "right_view": b.view, "correlation": corr,
                                    "phash_hamming": distance, "reflection": flipped})
    pd.DataFrame(flagged, columns=["patient_alias", "left_image_id", "right_image_id",
                                   "left_view", "right_view", "correlation", "phash_hamming",
                                   "reflection"]).to_csv(CACHE / "opposite_side_similarity_flags_train.csv", index=False)
    summary = {"train_patients": train.patient_alias.nunique(), "compared_left_right_pairs": compared,
               "flagged_pairs": len(flagged),
               "flagged_patients": len({row["patient_alias"] for row in flagged}),
               "flagged_view_combinations": dict(Counter(
                   f"{row['left_view']}-{row['right_view']}" for row in flagged)),
               "threshold": "Pearson r >= 0.9 or 63-bit perceptual-hash Hamming <= 6, direct or reflected",
               "interpretation": "Screening only; flags require visual review, and no development/test labels were used"}
    (CACHE / "opposite_side_similarity_audit_train.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
