"""Matched frozen/LoRA ultrasound experiment with patient-only APN supervision."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd

from nckuh_only_study_20261102 import ROOT, FEATURES, OUT as BASE, SEEDS, digest, write_json, metric
from nckuh_representation_screen_20261102 import load_metadata, tight_canvas
from nckuh_ultrasound_probe_20261102 import WEIGHT

OUT = ROOT / "02_experiments/NCKUH_lora_bilateral_pilot_20261102"
PILOT_SEEDS = SEEDS[:3]
GROUPS = ["frozen_original", "frozen_augmented", "lora_original", "lora_augmented"]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    purpose="Test target-specific representation adaptation and augmentation, controlling the bilateral pooling head",
                    source_methods=["https://arxiv.org/abs/2510.22990", "https://arxiv.org/abs/2106.09685", "https://huggingface.co/docs/peft/developer_guides/low_level_api"],
                    checkpoint_sha256=digest(WEIGHT), patient_assignments_sha256=digest(BASE / "random_patient_assignments.csv"),
                    analysis_unit="Patient only; 1=APN, 0=non-APN confirmed by user. No independently verified diseased-side output",
                    pilot_seeds=PILOT_SEEDS, groups=GROUPS,
                    encoder="Original USF-MAE ViT-B/16, all 150 encoder tensors strict-loaded. First 8 transformer blocks frozen and cached in float32. Last4 blocks frozen or adapted by LoRA in qkv projections only",
                    adapter="PEFT inject_adapter_in_model, rank8 alpha16 dropout.05, bias none; base pretrained tensors remain frozen",
                    cache="3 fixed label-blind variants per image: original, -2 degree rotation/gamma.9, +2 degree rotation/gamma1.1. All held-out evaluation uses original only. Training bags select one common variant for all their views",
                    head="Image CLS and mask-covered patch mean concatenated1536 -> LayerNorm -> linear32 ReLU; average projected views within side; concatenate sidewise maximum and absolute left-right difference; 64->16 ReLU dropout.25->1. Missing-side differences zeroed. No side labels",
                    fitting="30 fixed epochs, patient minibatch16, balanced patient BCE, AdamW weight_decay.001; head lr.001, adapters lr.0001, gradient norm5; no epoch selection on development",
                    selection="Development-only four-arm pilot. Expand a LoRA arm only if development mean AUROC>=.65 and AP>=.78 and improvement over its matched frozen arm >=.03 AUROC and >=.02 AP, jointly in >=2/3 pilot draws",
                    expansion="Only eligible selected arm and matched frozen control may extend to all ten existing draws. No held-out exploratory-test scores in this pilot",
                    goal_completion="Neither pilot thresholds nor selected development scores establish AUROC and AP>.8 on independently evaluated data",
                    software="Existing timm model execution and PEFT LoRA injection; computational configurations are not claimed as novel named architectures")
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve existing protocol")
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", dict(source_sha256=digest(__file__), protocol_sha256=digest(OUT / "protocol.json")))
    print("Three-draw matched frozen/LoRA augmentation pilot fixed", flush=True)


def require_lock():
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"] and digest(OUT / "protocol.json") == lock["protocol_sha256"]


def original_encoder():
    import timm
    import torch
    checkpoint = torch.load(WEIGHT, map_location="cpu", weights_only=True)
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    keys = set(model.state_dict())
    encoder = {key: value for key, value in checkpoint.items() if key in keys}
    assert set(encoder) == keys and len(keys) == 150
    model.load_state_dict(encoder, strict=True)
    return model.cuda().eval().requires_grad_(False)


def transform_image(image, mask, variant):
    if variant == 0:
        return image, mask
    angle, gamma = (-2., .9) if variant == 1 else (2., 1.1)
    matrix = cv2.getRotationMatrix2D((111.5, 111.5), angle, 1.)
    image = cv2.warpAffine(image, matrix, (224, 224), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(mask, matrix, (224, 224), flags=cv2.INTER_NEAREST, borderValue=0)
    image = np.rint(255 * (image / 255.) ** gamma).clip(0, 255).astype(np.uint8)
    image *= mask[..., None]
    return image, mask


def cache():
    require_lock()
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, _ = load_metadata()
    target = OUT / "lower8_float32.npy"
    if target.exists():
        raise FileExistsError("Inspect existing cache before rebuilding")
    model = original_encoder()
    lower = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=(3, len(manifest), 197, 768))
    embedding = np.zeros((3, len(manifest), 1536), dtype=np.float32)
    selections = np.zeros((3, len(manifest), 196), dtype=bool)
    mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1)
    error = 0.
    with torch.inference_mode():
        for variant in range(3):
            for start in range(0, len(manifest), 32):
                arrays, masks = [], []
                for ident in manifest.image_id.iloc[start:start+32]:
                    gray = cv2.imread(str(FEATURES / "percentile" / f"{ident}.png"), 0)
                    mask = cv2.imread(str(FEATURES / "masks" / f"{ident}.png"), 0) > 0
                    image, binary = tight_canvas(gray, mask)
                    image, binary = transform_image(image, binary, variant)
                    arrays.append(image)
                    coverage = binary.reshape(14, 16, 14, 16).mean((1, 3)).ravel()
                    chosen = coverage >= .8
                    if not chosen.any():
                        chosen[np.argmax(coverage)] = True
                    masks.append(chosen)
                tensor = torch.from_numpy(np.stack(arrays).transpose(0, 3, 1, 2).copy()).cuda().float() / 255.
                tensor = (tensor-mean)/std
                token = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(tensor))))
                for block in model.blocks[:8]:
                    token = block(token)
                lower[variant, start:start+len(arrays)] = token.cpu().numpy()
                tail = token
                for block in model.blocks[8:]:
                    tail = block(tail)
                tail = model.norm(tail)
                if variant == 0 and start == 0:
                    expected = model.forward_features(tensor)
                    error = float((expected-tail).abs().max())
                    assert error <= 1e-6, error
                selected = torch.from_numpy(np.stack(masks)).cuda()
                patch_mean = (tail[:, 1:] * selected[..., None]).sum(1) / selected.sum(1)[:, None]
                embedding[variant, start:start+len(arrays)] = torch.cat([tail[:, 0], patch_mean], dim=1).cpu().numpy()
                selections[variant, start:start+len(arrays)] = np.stack(masks)
            lower.flush()
            print(f"Frozen lower8 cache variant {variant} completed: {len(manifest)} images", flush=True)
    del lower
    np.save(OUT / "frozen_embedding.npy", embedding)
    np.save(OUT / "patch_selection.npy", selections)
    write_json(OUT / "cache_provenance.json", dict(dtype="float32", shape=[3, len(manifest), 197, 768],
                                                    checkpoint_sha256=digest(WEIGHT), original_forward_maxabs=error,
                                                    base_unchanged=True, cache_sha256=digest(target)))


def fit(seed, group):
    import torch
    from torch import nn
    from peft import LoraConfig, inject_adapter_in_model
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    manifest, patients = load_metadata()
    assignments = pd.read_csv(BASE / "random_patient_assignments.csv")
    roles = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role.reindex(patients.patient_alias)
    train, dev = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
    groups = [np.flatnonzero(manifest.patient_id.eq(pid)) for pid in patients.patient_id]
    width = max(map(len, groups))
    image_index = np.full((len(patients), width), -1, dtype=int)
    sides = np.full_like(image_index, -1)
    for i, rows in enumerate(groups):
        image_index[i, :len(rows)] = rows
        sides[i, :len(rows)] = manifest.side.iloc[rows].map({"left": 0, "right": 1}).to_numpy()
    adapt = group.startswith("lora")
    augmented = group.endswith("augmented")
    lower = np.load(OUT / "lower8_float32.npy", mmap_mode="r") if adapt else None
    frozen = np.load(OUT / "frozen_embedding.npy", mmap_mode="r")
    selections = np.load(OUT / "patch_selection.npy")
    encoder = original_encoder() if adapt else None
    if adapt:
        targets = [f"blocks.{i}.attn.qkv" for i in range(8, 12)]
        encoder = inject_adapter_in_model(LoraConfig(r=8, lora_alpha=16, lora_dropout=.05, bias="none", target_modules=targets), encoder)
        trainable_base = [name for name, p in encoder.named_parameters() if p.requires_grad]
        assert trainable_base and all("lora_" in name and any(f"blocks.{i}." in name for i in range(8, 12)) for name in trainable_base)
    torch.manual_seed(seed + 1)

    class BilateralHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 32), nn.ReLU())
            self.classifier = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Dropout(.25), nn.Linear(16, 1))

        def forward(self, values, side):
            node = self.project(values)
            left_mask, right_mask = side.eq(0), side.eq(1)
            left = (node * left_mask[..., None]).sum(1) / left_mask.sum(1).clamp_min(1)[:, None]
            right = (node * right_mask[..., None]).sum(1) / right_mask.sum(1).clamp_min(1)[:, None]
            both = left_mask.any(1) & right_mask.any(1)
            difference = (left-right).abs() * both[:, None]
            return self.classifier(torch.cat([torch.maximum(left, right), difference], dim=1)).squeeze(-1)

    head = BilateralHead().cuda()
    parameters = [{"params": list(head.parameters()), "lr": .001}]
    if adapt:
        parameters.append({"params": [p for p in encoder.parameters() if p.requires_grad], "lr": .0001})
    optimizer = torch.optim.AdamW(parameters, weight_decay=.001)
    y = torch.tensor(patients.label.to_numpy(), dtype=torch.float32, device="cuda")
    positives = float(y[train].sum()); negatives = len(train)-positives
    rng = np.random.default_rng(seed + 93)

    def forward(index, variants):
        indices = image_index[index]
        valid = indices >= 0
        variants_per_image = np.broadcast_to(variants[:, None], indices.shape)[valid]
        flattened = indices[valid]
        if adapt:
            values = torch.from_numpy(np.array(lower[variants_per_image, flattened])).cuda()
            for block in encoder.blocks[8:]:
                values = block(values)
            values = encoder.norm(values)
            selected = torch.from_numpy(selections[variants_per_image, flattened]).cuda()
            patch_mean = (values[:, 1:] * selected[..., None]).sum(1) / selected.sum(1)[:, None]
            embedding = torch.cat([values[:, 0], patch_mean], dim=1)
        else:
            embedding = torch.from_numpy(np.array(frozen[variants_per_image, flattened])).cuda()
        padded = torch.zeros(len(index)*width, 1536, device="cuda")
        padded = padded.index_copy(0, torch.from_numpy(np.flatnonzero(valid.ravel())).cuda(), embedding)
        return head(padded.reshape(len(index), width, 1536), torch.from_numpy(sides[index]).cuda())

    start = time.monotonic()
    loss_final = None
    for epoch in range(30):
        head.train()
        if encoder is not None:
            encoder.train()
        order = rng.permutation(train)
        loss_sum = 0.
        for offset in range(0, len(order), 16):
            index = order[offset:offset+16]
            variant = rng.integers(0, 3, size=len(index)) if augmented else np.zeros(len(index), dtype=int)
            optimizer.zero_grad()
            logits = forward(index, variant)
            target = y[index]
            weight = torch.where(target > 0, len(train)/(2*positives), len(train)/(2*negatives))
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).mean()
            loss.backward()
            nn.utils.clip_grad_norm_([p for param_group in optimizer.param_groups for p in param_group["params"]], 5.)
            optimizer.step()
            loss_sum += float(loss.detach())*len(index)
        loss_final = loss_sum/len(train)
        if (epoch+1) % 10 == 0:
            print(f"{group} seed {seed} epoch {epoch+1}/30 loss {loss_final:.4f} elapsed {time.monotonic()-start:.0f}s", flush=True)
    head.eval()
    if encoder is not None:
        encoder.eval()
    scores = []
    with torch.inference_mode():
        for offset in range(0, len(dev), 16):
            index = dev[offset:offset+16]
            scores.extend(torch.sigmoid(forward(index, np.zeros(len(index), dtype=int))).cpu().tolist())
    rows = [dict(seed=seed, group=group, role="development", patient_alias=patients.patient_alias.iloc[i], label=int(patients.label.iloc[i]), score=float(score)) for i, score in zip(dev, scores)]
    result = dict(seed=seed, group=group, role="development", **metric(patients.label.to_numpy()[dev], np.asarray(scores)),
                  final_training_loss=loss_final, fit_seconds=time.monotonic()-start,
                  trainable_parameters=sum(p.numel() for pg in optimizer.param_groups for p in pg["params"]))
    saved = {"head": head.state_dict(), "checkpoint_sha256": digest(WEIGHT), "seed": seed, "group": group}
    if encoder is not None:
        names = {name for name, p in encoder.named_parameters() if p.requires_grad}
        saved["adapters"] = {name: value.detach().cpu() for name, value in encoder.state_dict().items() if name in names}
    (OUT / "models").mkdir(exist_ok=True)
    torch.save(saved, OUT / "models" / f"{group}_{seed}.pth")
    write_json(OUT / f"result_{group}_{seed}.json", dict(metrics=result, predictions=rows))
    print(json.dumps(result), flush=True)
    del encoder, head
    torch.cuda.empty_cache()
    return result, rows


def pilot():
    require_lock()
    metrics, predictions = [], []
    for group in GROUPS:
        for seed in PILOT_SEEDS:
            result_path = OUT / f"result_{group}_{seed}.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                result, rows = saved["metrics"], saved["predictions"]
            else:
                result, rows = fit(seed, group)
            metrics.append(result); predictions.extend(rows)
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / "pilot_development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "pilot_development_predictions.csv", index=False)
    summary = frame.groupby("group")[["auroc", "ap"]].mean()
    summary["mean_score"] = summary.mean(axis=1)
    summary = summary.sort_values("mean_score", ascending=False)
    summary.to_csv(OUT / "pilot_summary.csv")
    eligible = []
    for group in summary.index:
        if not group.startswith("lora"):
            continue
        control = group.replace("lora", "frozen")
        paired = frame[frame.group.eq(group)].set_index("seed")[["auroc", "ap"]] - frame[frame.group.eq(control)].set_index("seed")[["auroc", "ap"]]
        if summary.loc[group, "auroc"] >= .65 and summary.loc[group, "ap"] >= .78 and paired.auroc.mean() >= .03 and paired.ap.mean() >= .02 and paired.gt(0).all(axis=1).sum() >= 2:
            eligible.append(group)
    write_json(OUT / "pilot_decision.json", dict(selected=eligible[0] if eligible else None, eligible=eligible,
                                                 no_test_scoring=True, needs_ten_draw_expansion=bool(eligible),
                                                 goal_complete=False))
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "cache", "pilot"])
    a = p.parse_args()
    globals()[a.stage]()
