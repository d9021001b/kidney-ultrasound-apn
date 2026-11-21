"""Screen same-patient candidate panels for perceptually repeated kidney views."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
from audit_nckuh_additional_same_exam_views import OUT as CANDIDATES

OUT = base.ROOT / "02_experiments/NCKUH_candidate_view_similarity_20261103"


def roi(gray, mask):
    gray = np.where(mask, gray, 0).astype(np.uint8)
    rgb, _ = base.tight_canvas(gray, mask)
    return rgb[..., 0]


def phash(image):
    coefficients = cv2.dct(cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))
    low = coefficients[:8, :8].reshape(-1)[1:]
    return low > np.median(low)


def compare(left, right):
    left_hash = phash(left)
    a = left.ravel().astype(float)
    values = []
    for mirrored, image in ((False, right), (True, np.fliplr(right))):
        b = image.ravel().astype(float)
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else 0.
        hamming = int(np.count_nonzero(left_hash != phash(image)))
        values.append((corr, -hamming, mirrored, image))
    return max(values, key=lambda t: t[:2])


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    catalog = pd.read_csv(CANDIDATES / "candidate_qc_internal.csv")
    private = pd.read_csv(CANDIDATES / "source_linkage_PRIVATE.csv")
    candidates = catalog.merge(private[["candidate", "source_path"]], on="candidate", validate="one_to_one")
    assert len(candidates) == 14 and candidates.panel_readable.all()
    existing = pd.read_csv(base.OUT / "deidentified_source_manifest.csv")
    existing = existing[existing.patient_alias.isin(candidates.patient_alias)]
    bank = {}
    for row in existing.itertuples(index=False):
        gray = cv2.imread(str(base.OUT / "percentile" / f"{row.image_id}.png"), 0)
        mask = cv2.imread(str(base.OUT / "masks" / f"{row.image_id}.png"), 0) > 0
        bank[row.image_id] = (row.patient_alias, row.view, roi(gray, mask))
    for row in candidates.itertuples(index=False):
        image, mask = base.split_panel(Path(row.source_path))
        gray = base.percentile_gray(image, mask)
        bank[row.candidate] = (row.patient_alias, row.view, roi(gray, mask))
    pairs, sheet = [], []
    for row in candidates.itertuples(index=False):
        _, view, frame = bank[row.candidate]
        nearest = []
        for identifier, (patient, other_view, other) in bank.items():
            if patient != row.patient_alias or identifier == row.candidate:
                continue
            corr, neg_hamming, flipped, compared = compare(frame, other)
            pairs.append(dict(candidate=row.candidate, comparison=identifier, view=view,
                              comparison_view=other_view, correlation=corr, hamming=-neg_hamming,
                              comparison_flipped=flipped,
                              manual_trigger=bool(corr >= .90 or neg_hamming >= -6)))
            nearest.append((corr, identifier, compared, flipped))
        corr, identifier, comparison, flipped = max(nearest, key=lambda t: t[0])
        sheet.append((row.candidate, view, frame, comparison, corr, flipped))
    result = pd.DataFrame(pairs)
    page = Image.new("RGB", (960, 1360), "white")
    draw = ImageDraw.Draw(page)
    draw.text((15, 12), "Internal QA: candidate (left) and closest same-patient view (right)", fill="black")
    for index, (identifier, view, image, match, corr, flipped) in enumerate(sheet):
        x, y = 12 + (index % 3) * 320, 45 + (index // 3) * 262
        draw.text((x, y), f"{identifier} {view}  r={corr:.3f}  {'flip' if flipped else 'native'}", fill="black")
        for offset, array in ((0, image), (151, match)):
            page.paste(Image.fromarray(array).resize((145, 145), Image.Resampling.BILINEAR), (x + offset, y + 24))
    review = candidates[["candidate", "view"]].assign(
        actual_side_verified="", actual_view_verified="", same_exam_verified="",
        duplicate_disposition="", contour_usable="", notes="")
    OUT.mkdir()
    result.to_csv(OUT / "pair_similarity_INTERNAL.csv", index=False)
    review.to_csv(OUT / "candidate_review_INTERNAL.csv", index=False, encoding="utf-8-sig")
    page.save(OUT / "comparison_sheet_INTERNAL.png")
    audit = dict(candidate_count=len(candidates), compared_pairs=len(result),
                 flagged_pairs=int(result.manual_trigger.sum()),
                 flagged_candidates=result.loc[result.manual_trigger, "candidate"].nunique(),
                 max_correlation=float(result.correlation.max()),
                 minimum_hamming=int(result.hamming.min()), review_confirmed=0)
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    run()
