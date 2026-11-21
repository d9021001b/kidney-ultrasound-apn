"""Frozen official URI-CADS five-fold features on revised NCKUH kidney ROIs."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from prepare_nckuh_383_contour_cohort_20261111 import OUT as REGISTRY, digest


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "02_experiments/NCKUH_revised_383_contour_cache_20261111"


def main():
    target = CACHE / "uricads_fold_embeddings.npy"
    if target.exists():
        raise FileExistsError("Preserve completed revised-cohort encoding")
    lock = json.loads((REGISTRY / "cohort_lock.json").read_text())
    assert digest(REGISTRY / "contour_image_manifest.csv") == lock["files_sha256"]["contour_image_manifest.csv"]
    manifest = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    assert len(manifest) == 1362
    mean, std = uri.public_normalization()
    canvases = []
    for image_id in manifest.image_id:
        image = cv2.imread(str(CACHE / "clean_rgb" / f"{image_id}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(CACHE / "masks" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE) > 0
        assert image is not None and mask.any() and image.shape[:2] == mask.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        canvas, _ = base.tight_canvas(rgb, mask, size=800)
        canvases.append(canvas)
    canvases = np.stack(canvases)
    fold_embeddings, fold_scores, weights = [], [], []
    for fold, checkpoint in enumerate(uri.MODEL_PATHS, 1):
        assert checkpoint.exists()
        model = uri.OfficialGlobalEncoder(checkpoint)
        embedding, score = uri.encode_batches(model, canvases, mean, std, batch_size=8)
        assert embedding.shape == (1362, 2048) and score.shape == (1362, 7)
        assert np.isfinite(embedding).all() and np.isfinite(score).all()
        fold_embeddings.append(embedding.astype(np.float32))
        fold_scores.append(score.astype(np.float32))
        weights.append({"fold": fold, "name": checkpoint.name, "sha256": digest(checkpoint)})
        print(f"Fold {fold}/5 complete", flush=True)
        del model
        torch.cuda.empty_cache()
    embeddings = np.stack(fold_embeddings, axis=1)
    scores = np.stack(fold_scores, axis=1)
    np.save(target, embeddings)
    np.save(CACHE / "uricads_fold_pathology_scores.npy", scores)
    audit = {"images": 1362, "embedding_shape": list(embeddings.shape),
             "score_shape": list(scores.shape), "input_size": 800,
             "normalization": {"mean": mean.tolist(), "std": std.tolist()},
             "official_checkpoints": weights, "labels_used_in_encoding": [],
             "embedding_sha256": digest(target),
             "scores_sha256": digest(CACHE / "uricads_fold_pathology_scores.npy")}
    (CACHE / "uricads_encoder_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "official_checkpoints"}, indent=2))


if __name__ == "__main__":
    main()
