"""Frozen official URFM ViT-H representations for contour-first NCKUH APN."""
from __future__ import annotations

from contextlib import contextmanager

import nckuh_dmsa_urfm_frozen_20261106 as study


WEIGHT = (
    study.MODEL_DIR
    / "weights/mae_vit_huge_patch16_dec768d8b_all_biomedclip_1199.pth"
)
OUT = study.ROOT / "02_experiments/NCKUH_DMSA_urfm_huge_frozen_20261108"
EXPECTED_WEIGHT_BYTES = 11_027_164_461
WEIGHT_SHA256 = (
    "0693b0ac0b335a095904362af9f2625e473a1040e2c1b2a0e5fed9ef9066dca6"
)
MODEL_FACTORY_NAME = "vit_huge_patch16"
ARCHITECTURE_NAME = "URFM ViT-H/16 with official global pooling"
PATCH_SIZE = 16
PATCH_GRID = 14
EMBED_DIM = 1280
DROP_PATH_RATE = 0.0

CONFIG = {
    "WEIGHT": WEIGHT,
    "OUT": OUT,
    "EXPECTED_WEIGHT_BYTES": EXPECTED_WEIGHT_BYTES,
    "WEIGHT_SHA256": WEIGHT_SHA256,
    "MODEL_FACTORY_NAME": MODEL_FACTORY_NAME,
    "ARCHITECTURE_NAME": ARCHITECTURE_NAME,
    "PATCH_SIZE": PATCH_SIZE,
    "PATCH_GRID": PATCH_GRID,
    "EMBED_DIM": EMBED_DIM,
    "DROP_PATH_RATE": DROP_PATH_RATE,
}


@contextmanager
def configured():
    original = {name: getattr(study, name) for name in CONFIG}
    try:
        for name, value in CONFIG.items():
            setattr(study, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(study, name, value)


def urfm_features(model, images):
    with configured():
        return study.urfm_features(model, images)


def run(stage: str) -> None:
    with configured():
        {"encode": study.encode, "evaluate": study.evaluate}[stage]()


if __name__ == "__main__":
    import argparse

    from threadpoolctl import threadpool_limits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("encode", "evaluate"))
    arguments = parser.parse_args()
    with threadpool_limits(limits=4):
        run(arguments.stage)
