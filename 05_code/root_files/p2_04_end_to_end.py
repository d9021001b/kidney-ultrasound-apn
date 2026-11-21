"""Paper 2, step 4: probe-specific end-to-end fine-tuning on the development pool.

Arms (all hyper-parameters fixed in advance; no early stopping on held-out folds):
  specific_context : ConvNeXt-Tiny fine-tuned only on the probe group's kidney crops (with surrounding context)
  specific_masked  : same, pixels outside the kidney contour set to zero
  pooled_init      : trained on both probes, then fine-tuned on the probe group (context crops)
Five patient-grouped folds are drawn once over the whole pool and shared by every arm, so a held-out
patient is never seen by any model that scores it. Kidney score = mean logit over its routed images
with horizontal-flip test-time augmentation. Exploratory-test role is never loaded for scoring.
"""

import json
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.model_selection import StratifiedKFold

from p2_common import CACHE, FOLD_IDS, P2, P2_OUT, SEED, assign_folds, cluster_interval, metric, pool_mask
from p2_features import routed_images

OUT = P2_OUT / "end_to_end"
H, W = 224, 288
EPOCHS, FT_EPOCHS, BATCH, LR = 14, 8, 32, 1e-4
SEEDS = (0, 1)
DEVICE = "cuda"


def load_pixels(images):
    gray = np.zeros((len(images), H, W), np.uint8)
    mask = np.zeros((len(images), H, W), np.uint8)
    for i, image_id in enumerate(images.image_id):
        g = cv2.imread(str(CACHE / "clean_rgb" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(str(CACHE / "masks" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        gray[i] = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
        mask[i] = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return gray, mask


def augment(x, rng):
    n = len(x)
    angle = torch.from_numpy(rng.uniform(-10, 10, n) * np.pi / 180).float()
    scale = torch.from_numpy(rng.uniform(0.85, 1.15, n)).float()
    shift = torch.from_numpy(rng.uniform(-0.08, 0.08, (n, 2))).float()
    flip = torch.from_numpy(rng.choice([-1.0, 1.0], n)).float()
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = torch.cos(angle) / scale * flip
    theta[:, 0, 1] = -torch.sin(angle) / scale
    theta[:, 1, 0] = torch.sin(angle) / scale * flip
    theta[:, 1, 1] = torch.cos(angle) / scale
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta.to(x.device), x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    gamma = torch.from_numpy(rng.uniform(0.75, 1.3, (n, 1, 1, 1))).float().to(x.device)
    gain = torch.from_numpy(rng.uniform(0.85, 1.15, (n, 1, 1, 1))).float().to(x.device)
    return (x.clamp(1e-4, 1) ** gamma * gain).clamp(0, 1)


def to_input(x):
    return ((x - 0.25) / 0.2).repeat(1, 3, 1, 1)


def new_model():
    model = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1", stochastic_depth_prob=0.1)
    model.classifier[2] = nn.Linear(768, 1)
    return model.to(DEVICE)


def train(model, pixels, y, idx, epochs, seed, lr=LR):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * int(np.ceil(len(idx) / BATCH))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=0.15)
    positive = y[idx].mean()
    weight = torch.tensor((1 - positive) / positive, device=DEVICE)
    scaler = torch.amp.GradScaler()
    for _ in range(epochs):
        model.train()
        order = rng.permutation(idx)
        for start in range(0, len(order), BATCH):
            b = order[start:start + BATCH]
            if len(b) < 2:
                continue
            x = augment(torch.from_numpy(pixels[b]).float().div(255).unsqueeze(1).to(DEVICE), rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(to_input(x)).squeeze(1).float()
            loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(y[b]).float().to(DEVICE), pos_weight=weight)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
    return model


@torch.no_grad()
def predict(model, pixels, idx):
    model.eval()
    out = []
    for start in range(0, len(idx), 128):
        x = torch.from_numpy(pixels[idx[start:start + 128]]).float().div(255).unsqueeze(1).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = (model(to_input(x)) + model(to_input(torch.flip(x, dims=[3])))).squeeze(1).float() / 2
        out.append(logit.cpu().numpy())
    return np.concatenate(out)


def main():
    arms = sys.argv[1].split(",") if len(sys.argv) > 1 else ["specific_context", "specific_masked", "pooled_init"]
    OUT.mkdir(parents=True, exist_ok=True)
    patients, _, images = routed_images()
    images = images[pool_mask(images)].reset_index(drop=True)
    gray, mask = load_pixels(images)
    masked = gray * mask
    y = images.kidney_APN.to_numpy(np.float32)

    pool = patients[patients.patient_alias.isin(images.patient_alias)].reset_index(drop=True)
    fold_of = assign_folds(pool)
    fold = images.patient_alias.map(fold_of).to_numpy()
    pd.DataFrame({"patient_alias": list(fold_of), "fold": list(fold_of.values())}).to_csv(OUT / "patient_folds.csv", index=False)

    route = images.route.to_numpy()
    logits = {arm: np.zeros((len(SEEDS), len(images))) for arm in arms}
    for s, seed in enumerate(SEEDS):
        for k in FOLD_IDS:
            fit_all = np.flatnonzero((fold != k) & (fold >= 0))
            pooled = None
            if "pooled_init" in arms:
                pooled = train(new_model(), gray, y, fit_all, EPOCHS, 1000 * seed + k)
                pooled_state = {name: value.clone() for name, value in pooled.state_dict().items()}
            for probe in ("convex", "linear"):
                fit = np.flatnonzero((fold != k) & (fold >= 0) & (route == probe))
                hold = np.flatnonzero((fold == k) & (route == probe))
                for arm in arms:
                    pixels = masked if arm == "specific_masked" else gray
                    if arm == "pooled_init":
                        pooled.load_state_dict(pooled_state)
                        model = train(pooled, pixels, y, fit, FT_EPOCHS, 1000 * seed + k + 7, lr=LR / 4)
                    else:
                        model = train(new_model(), pixels, y, fit, EPOCHS, 1000 * seed + k + 3)
                    logits[arm][s, hold] = predict(model, pixels, hold)
                    part = metric(y[hold], logits[arm][s, hold])
                    print(f"seed {seed} fold {k} {probe:6s} {arm:17s} image-level AUROC {part['auroc']:.3f} AP {part['ap']:.3f}", flush=True)

    report = {"protocol": __doc__, "arms": {}, "test_APN_scored": False}
    keep = np.isin(fold, FOLD_IDS)
    frame = images.loc[keep, ["image_id", "patient_alias", "kidney_alias", "side", "route", "kidney_APN"]].copy()
    for arm in arms:
        frame[arm] = logits[arm].mean(0)[keep]
    frame.to_csv(OUT / f"image_oof_logits_{'_'.join(arms)}.csv", index=False)
    for arm in arms:
        report["arms"][arm] = {}
        for probe in ("convex", "linear"):
            kid = frame[frame.route.eq(probe)].groupby("kidney_alias").agg(
                patient_alias=("patient_alias", "first"), label=("kidney_APN", "first"), score=(arm, "mean"))
            result = metric(kid.label, kid.score) | cluster_interval(kid.patient_alias, kid.label, kid.score)
            result |= {"patients": int(kid.patient_alias.nunique()), "kidneys": len(kid), "prevalence": float(kid.label.mean())}
            report["arms"][arm][probe] = result
            print(arm, probe, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in result.items()}, flush=True)
    (OUT / f"results_{'_'.join(arms)}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
