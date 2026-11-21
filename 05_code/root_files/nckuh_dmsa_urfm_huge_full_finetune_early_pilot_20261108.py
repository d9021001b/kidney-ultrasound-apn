"""Inner-curve justified early-selection control for full URFM ViT-H/16."""
from __future__ import annotations

from contextlib import contextmanager

import nckuh_dmsa_urfm_huge_full_finetune_pilot_20261108 as study


OUT = study.ROOT / "02_experiments/NCKUH_DMSA_urfm_huge_full_finetune_early_pilot_20261108"


@contextmanager
def configured():
    values = {"OUT": OUT, "MIN_EPOCHS": 1, "PATIENCE": 3, "MAX_EPOCHS": 8}
    original = {name: getattr(study, name) for name in values}
    try:
        for name, value in values.items():
            setattr(study, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(study, name, value)


def run() -> None:
    with configured():
        study.evaluate()


if __name__ == "__main__":
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run()
