"""
Rebuild the kidney crops from the CLEAN frames using the existing YOLO segmenter.

Why this exists.  The crops every result so far was measured on (`*_0_1_inner/`) were
produced from the copy of each frame that carries the physician's hand-drawn red kidney
outline: the red curve was thresholded in HSV, filled into a mask, and the largest
axis-aligned rectangle inside that mask became the crop.  Three consequences follow.
The region of interest is a manual expert segmentation, so the pipeline is not
end-to-end automated.  Roughly 15 annotation pixels survive into about 98 percent of the
crops.  And the inscribed rectangle is around 210 by 100, which every downstream model
then stretches to 224 by 224, a 2:1 distortion.

This script removes all three at once:
  * pixels come from `original_png/` and `original_png_nckuh/` -- the clean twins, which
    have no red curve on them and have never been used by any experiment;
  * the kidney is located by the already-trained YOLOv12x segmenter, so nothing depends
    on the annotation at inference time;
  * the crop is padded to square (letterbox) instead of stretched, so anatomy keeps its
    aspect ratio.

Measured agreement of that segmenter with the physician's contour, scored on the clean
twins so the model cannot read the curve it is being compared against: median
intersection-over-union 0.84 at CYCH and 0.82 at NCKUH, detection in 35/40 and 33/40
sampled frames.

Frames where the segmenter finds nothing fall back to the physician polygon, and the
manifest records which source was used per image (`roi_source`), so the fallback subset
can be excluded or ablated later.

Usage
    python recrop_with_yolo.py --out-root yolo_crops --size 224
    python single_center_cv.py --site NCKUH --manifest yolo_crops/manifest.csv --out results_yolo/NCKUH
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = Path("C:/Users/udoo_w2/Desktop/work-kidney-tumor/kidney-tumor/best.pt")
SITES = {
    "CYCH": {"clean": "original_png", "labels": "cych_image_label"},
    "NCKUH": {"clean": "original_png_nckuh", "labels": "nckuh_image_label"},
}


def red_stem(clean_stem: str) -> str:
    """original_png names drop the '_red_' token that the label folders keep."""
    parts = clean_stem.split("_")
    return "_".join(parts[:2] + ["red"] + parts[2:])


def polygon_mask(txt: Path, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for line in txt.read_text().strip().splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        pts = (np.array(v[1:], dtype=float).reshape(-1, 2) * [w, h]).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m


def bbox_of(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def letterbox(img: np.ndarray, size: int, grayscale: bool = True) -> np.ndarray:
    """Scale the long side to `size` and pad the short side, so the kidney keeps its
    proportions.  Padding is black, matching the existing background convention.

    Grayscale first, then resize.  B-mode ultrasound carries no colour information here:
    a saturation survey over all 1384 clean frames found a median of 0.3 percent of
    pixels with any saturation and a maximum of 5 percent, i.e. compression fringing and
    burnt-in text, not colour Doppler.  Collapsing to one channel removes that fringing,
    removes any trace of the physician's annotation that could survive a crop, and makes
    it impossible for a model to key on a colour artefact.  The channel is replicated to
    three so ImageNet-pretrained backbones load unchanged.

    Interpolation is INTER_AREA when shrinking and INTER_LINEAR when enlarging.  Cubic
    overshoots at the sharp caliper edges in these frames, which was measurably turning
    grey fringing into saturated pixels.
    """
    if grayscale:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((size, size, 3), np.uint8)
    out[(size - nh) // 2:(size - nh) // 2 + nh, (size - nw) // 2:(size - nw) // 2 + nw] = r
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default="yolo_crops")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--margin", type=float, default=0.08,
                   help="Fraction of the box's own width/height added on each side, so a "
                        "slightly tight mask does not clip the capsule.")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--source-manifest",
                   default=str(ROOT / "patient_characteristics" / "used_ultrasound_image_filenames.csv"),
                   help="Supplies labels and clinical covariates; joined on the red-form filename.")
    p.add_argument("--grayscale", action=argparse.BooleanOptionalAction, default=True,
                   help="Collapse to one channel (replicated to three) before resizing.")
    p.add_argument("--limit", type=int, default=0, help="Debug: cap images per site.")
    args = p.parse_args(argv)

    from ultralytics import YOLO  # imported late so --help stays instant

    t0 = time.time()
    out_root = ROOT / args.out_root
    src = pd.read_csv(args.source_manifest, encoding="utf-8-sig")
    src["patient_id"] = src["patient_id"].astype(str)
    by_name = {r["filename"]: r for _, r in src.iterrows()}

    model = YOLO(args.weights)
    rows, stats = [], {}

    for site, cfg in SITES.items():
        clean_dir = ROOT / cfg["clean"]
        lab_dir = ROOT / cfg["labels"]
        dst = out_root / site
        for c in ("0", "1"):
            (dst / c).mkdir(parents=True, exist_ok=True)

        files = sorted(clean_dir.glob("*.png"))
        if args.limit:
            files = files[:args.limit]
        n_yolo = n_poly = n_skip = 0

        for f in tqdm(files, desc=f"{site}", ncols=78, ascii=True):
            rname = f"{red_stem(f.stem)}.png"
            meta = by_name.get(rname)
            if meta is None:
                n_skip += 1
                continue  # not part of the analysed cohort

            img = cv2.imread(str(f))
            if img is None:
                n_skip += 1
                continue
            h, w = img.shape[:2]

            box, source = None, None
            res = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            if res.masks is not None and len(res.masks) > 0:
                pr = np.zeros((h, w), np.uint8)
                for seg in res.masks.xy:
                    cv2.fillPoly(pr, [seg.astype(np.int32)], 1)
                box, source = bbox_of(pr), "yolo"
            if box is None:
                txt = lab_dir / f"{red_stem(f.stem)}.txt"
                if txt.exists():
                    box, source = bbox_of(polygon_mask(txt, h, w)), "physician_polygon"
            if box is None:
                n_skip += 1
                continue

            x0, y0, x1, y1 = box
            mx, my = int((x1 - x0) * args.margin), int((y1 - y0) * args.margin)
            x0, y0 = max(0, x0 - mx), max(0, y0 - my)
            x1, y1 = min(w, x1 + mx), min(h, y1 + my)
            if x1 - x0 < 8 or y1 - y0 < 8:
                n_skip += 1
                continue

            crop = letterbox(img[y0:y1, x0:x1], args.size, args.grayscale)
            label = int(meta["label"])
            rel = f"{args.out_root}/{site}/{label}/{rname}"
            cv2.imwrite(str(ROOT / rel), crop)

            n_yolo += source == "yolo"
            n_poly += source == "physician_polygon"
            rows.append({
                "site": site, "label": label,
                "diagnosis_group": meta["diagnosis_group"],
                "filename": rname, "relative_path": rel,
                "patient_id": meta["patient_id"],
                "exam_date_raw": meta["exam_date_raw"], "exam_date_iso": meta["exam_date_iso"],
                "gender": meta["gender"], "gender_code": meta["gender_code"],
                "age_years": meta["age_years"], "renal_measure_cm": meta["renal_measure_cm"],
                "view_raw": meta["view_raw"], "view": meta["view"],
                "roi_source": source,
                "roi_w": x1 - x0, "roi_h": y1 - y0,
                "roi_aspect": round((x1 - x0) / (y1 - y0), 3),
            })

        stats[site] = {"clean_frames": len(files), "written": n_yolo + n_poly,
                       "roi_from_yolo": n_yolo, "roi_from_polygon": n_poly, "skipped": n_skip}
        print(f"  {site}: {n_yolo + n_poly} written "
              f"({n_yolo} YOLO, {n_poly} polygon fallback), {n_skip} skipped")

    man = pd.DataFrame(rows).sort_values(["site", "patient_id", "filename"])
    man_path = out_root / "manifest.csv"
    man.to_csv(man_path, index=False, encoding="utf-8")

    summary = {
        "weights": args.weights, "size": args.size, "margin": args.margin,
        "conf": args.conf, "imgsz": args.imgsz, "grayscale": args.grayscale,
        "per_site": stats,
        "counts": man.groupby(["site", "label"]).size().to_dict().__repr__(),
        "patients": man.groupby("site")["patient_id"].nunique().to_dict(),
        "roi_source": man["roi_source"].value_counts().to_dict(),
        "aspect_median": float(man["roi_aspect"].median()),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    (out_root / "recrop_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\nmanifest -> {man_path}  ({len(man)} images)")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
