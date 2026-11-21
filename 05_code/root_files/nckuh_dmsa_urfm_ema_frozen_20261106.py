"""Frozen URFM EMA-teacher ablation for contour-first NCKUH APN."""
from __future__ import annotations

import json
import sys

import torch

import nckuh_dmsa_urfm_frozen_20261106 as online


OUT = online.ROOT / "02_experiments/NCKUH_DMSA_urfm_ema_frozen_20261106"
MOTIVATION = (
    "The released checkpoint contains the decay-0.999 momentum teacher maintained "
    "by the official pretraining code; temporal parameter averaging is tested as a "
    "single stability ablation."
)


def load_ema_encoder() -> tuple[torch.nn.Module, dict]:
    if online.WEIGHT.stat().st_size != online.EXPECTED_WEIGHT_BYTES:
        raise ValueError("URFM official weight byte count mismatch")
    if online.digest(online.WEIGHT) != online.WEIGHT_SHA256:
        raise ValueError("URFM official weight hash mismatch")
    sys.path.insert(0, str(online.MODEL_DIR))
    import models_vit
    from util.pos_embed import interpolate_pos_embed

    checkpoint = torch.load(online.WEIGHT, map_location="cpu", weights_only=True)
    if "ema_state_dict" not in checkpoint:
        raise ValueError("URFM checkpoint has no EMA state dictionary")
    state = {
        name.removeprefix("module."): value
        for name, value in checkpoint["ema_state_dict"].items()
    }
    model = models_vit.vit_base_patch16(
        num_classes=2, drop_path_rate=0.0, global_pool=True
    )
    current = model.state_dict()
    removed: dict[str, list[int]] = {}
    for key in ("head.weight", "head.bias"):
        if key in state and state[key].shape != current[key].shape:
            removed[key] = list(state[key].shape)
            del state[key]
    interpolate_pos_embed(model, state)
    message = model.load_state_dict(state, strict=False)
    audit = {
        "selected_checkpoint_branch": "ema_state_dict",
        "checkpoint_top_level_keys": sorted(checkpoint),
        "checkpoint_model_tensors": len(state),
        "removed_incompatible_heads": removed,
        "missing_keys": list(message.missing_keys),
        "unexpected_keys": list(message.unexpected_keys),
        "motivation": MOTIVATION,
    }
    return model.eval().requires_grad_(False), audit


def main(stage: str) -> None:
    online.OUT = OUT
    online.load_encoder = load_ema_encoder
    {"encode": online.encode, "evaluate": online.evaluate}[stage]()
    if stage == "encode":
        audit_path = OUT / "encoder_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["source_variant"] = "official URFM ema_state_dict"
        audit["motivation"] = MOTIVATION
        audit_path.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    import argparse
    from threadpoolctl import threadpool_limits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("encode", "evaluate"))
    arguments = parser.parse_args()
    with threadpool_limits(limits=4):
        main(arguments.stage)
