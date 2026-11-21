"""Extract registered image-only radiomics and frozen RadImageNet features."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from prepare_nckuh_383_contour_cohort_20261111 import OUT as REGISTRY, digest


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "02_experiments/NCKUH_revised_383_contour_cache_20261111"
sys.path.insert(0, str(ROOT / "02_experiments/CYCH_honest_reevaluation_20261003"))
from extract_site import _radiomics_worker, dense_tight, run_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("radiomics", "dense"))
    args = parser.parse_args()
    lock = json.loads((REGISTRY / "cohort_lock.json").read_text())
    assert digest(REGISTRY / "contour_image_manifest.csv") == lock["files_sha256"]["contour_image_manifest.csv"]
    audit = json.loads((CACHE / "extraction_audit.json").read_text())
    assert audit["images"] == 1362 and audit["outside_mask_nonzero"] == 0
    manifest = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    ids = manifest.image_id.tolist()
    assert len(ids) == len(set(ids)) == 1362
    if args.stage == "radiomics":
        target = CACHE / "radiomics.csv"
        if target.exists():
            raise FileExistsError(target)
        jobs = [(ident, CACHE / "percentile" / f"{ident}.png", CACHE / "masks" / f"{ident}.png")
                for ident in ids]
        frame = run_pool(_radiomics_worker, jobs, processes=8)
        assert frame.index.tolist() == ids and frame.shape[0] == 1362
        frame.to_csv(target)
        result = {"stage": "radiomics", "images": len(frame), "features": len(frame.columns),
                  "sha256": digest(target), "labels_used": []}
    else:
        target = CACHE / "radimagenet_dense_tight.npy"
        if target.exists():
            raise FileExistsError(target)
        matrix = dense_tight(CACHE / "percentile", CACHE / "masks", ids)
        assert matrix.shape[0] == 1362 and np.isfinite(matrix).all()
        np.save(target, matrix)
        result = {"stage": "dense", "images": len(matrix), "features": matrix.shape[1],
                  "sha256": digest(target), "labels_used": [], "frozen": True}
    (CACHE / f"{args.stage}_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
