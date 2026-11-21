"""One-draw patient-held-out URFM ViT-H/16 supervised LoRA pilot."""
from __future__ import annotations

from contextlib import contextmanager

import nckuh_dmsa_urfm_huge_frozen_20261108 as huge
import nckuh_dmsa_urfm_lora_patient_20261106 as study


OUT = study.ROOT / "02_experiments/NCKUH_DMSA_urfm_huge_lora_pilot_20261108"
CACHE = OUT / "cache"


@contextmanager
def configured():
    with huge.configured():
        names = {
            "OUT": OUT,
            "CACHE": CACHE,
            "SEEDS": (20261102,),
            "ARMS": ("lora_rank",),
            "MAX_EPOCHS": 30,
            "MIN_EPOCHS": 6,
            "PATIENCE": 7,
            "BATCH_PATIENTS": 4,
            "EMBED_DIM": 1280,
            "CACHE_BLOCKS": 28,
            "MODEL_BLOCKS": 32,
        }
        original = {name: getattr(study, name) for name in names}
        original_transfer = study.urfm.TRANSFER_NORM_TO_FC_NORM
        try:
            for name, value in names.items():
                setattr(study, name, value)
            study.urfm.TRANSFER_NORM_TO_FC_NORM = True
            yield
        finally:
            study.urfm.TRANSFER_NORM_TO_FC_NORM = original_transfer
            for name, value in original.items():
                setattr(study, name, value)


def run(stage: str) -> None:
    with configured():
        {"cache": study.cache_tokens, "evaluate": study.evaluate}[stage]()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cache", "evaluate"))
    arguments = parser.parse_args()
    run(arguments.stage)
