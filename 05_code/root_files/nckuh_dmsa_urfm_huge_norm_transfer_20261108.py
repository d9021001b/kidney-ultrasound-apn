"""URFM ViT-H frozen features with explicit pretrained norm transfer."""
from __future__ import annotations

from contextlib import contextmanager

import nckuh_dmsa_urfm_huge_frozen_20261108 as huge


OUT = huge.study.ROOT / "02_experiments/NCKUH_DMSA_urfm_huge_norm_transfer_20261108"


@contextmanager
def configured():
    with huge.configured():
        original_out = huge.study.OUT
        original_transfer = huge.study.TRANSFER_NORM_TO_FC_NORM
        try:
            huge.study.OUT = OUT
            huge.study.TRANSFER_NORM_TO_FC_NORM = True
            yield
        finally:
            huge.study.OUT = original_out
            huge.study.TRANSFER_NORM_TO_FC_NORM = original_transfer


def run(stage: str) -> None:
    with configured():
        {"encode": huge.study.encode, "evaluate": huge.study.evaluate}[stage]()


if __name__ == "__main__":
    import argparse

    from threadpoolctl import threadpool_limits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("encode", "evaluate"))
    arguments = parser.parse_args()
    with threadpool_limits(limits=4):
        run(arguments.stage)
