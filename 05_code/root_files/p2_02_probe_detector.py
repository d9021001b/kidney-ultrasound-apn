"""Paper 2 router: image-level convex/linear probe detector (fine-tuned ResNet-18).

Labels are the (2) workbook Probe entries of single-probe patients. APN labels are never
read. Detector development uses patient-grouped five-fold CV on train+development roles;
exploratory-test patients are a held-out probe-only check. The five fold models are then
averaged to route every image, including those of `both` patients, with abstention.
"""

import json

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from nckuh_probe_blind_sector import image_frame
from p2_common import OLD_ROUTER, P2, REGISTRY, SEED, load_cohort

OUT = P2 / "router"
SIZE, EPOCHS, LOWER, UPPER = 224, 6, 0.10, 0.90
DEVICE = "cuda"


def load_frames(images):
    private = pd.read_csv(REGISTRY / "source_linkage_private.csv", dtype=str).set_index("image_id").source_path
    frames = np.zeros((len(images), SIZE, SIZE), np.uint8)
    for i, row in enumerate(images.itertuples(index=False)):
        frame = image_frame(private[row.image_id], panel=row.source_type == "panel_with_mask")
        frames[i] = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return frames


def batch(frames, idx, augment, rng):
    x = torch.from_numpy(frames[idx]).float().div(255.0).unsqueeze(1)
    if augment:
        gain = torch.from_numpy(rng.uniform(0.8, 1.2, (len(idx), 1, 1, 1))).float()
        shift = rng.integers(-8, 9, (len(idx), 2))
        x = torch.stack([torch.roll(img, (int(dy), int(dx)), (1, 2)) for img, (dy, dx) in zip(x, shift)])
        x = (x * gain).clamp(0, 1)
    return ((x - 0.5) / 0.25).repeat(1, 3, 1, 1).to(DEVICE)


def fit(frames, y, train_idx, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(512, 1)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    positive = y[train_idx].mean()
    weight = torch.tensor((1 - positive) / positive, device=DEVICE)
    for _ in range(EPOCHS):
        model.train()
        order = rng.permutation(train_idx)
        for start in range(0, len(order), 32):
            idx = order[start:start + 32]
            target = torch.from_numpy(y[idx]).float().to(DEVICE)
            loss = nn.functional.binary_cross_entropy_with_logits(model(batch(frames, idx, True, rng)).squeeze(1),
                                                                  target, pos_weight=weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict(model, frames, idx):
    model.eval()
    out = [torch.sigmoid(model(batch(frames, idx[s:s + 128], False, None)).squeeze(1)).cpu().numpy()
           for s in range(0, len(idx), 128)]
    return np.concatenate(out)


def summarize(y, p):
    routed = (p <= LOWER) | (p >= UPPER)
    return {"images": int(len(y)), "auroc": float(roc_auc_score(y, p)),
            "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, p >= .5)),
            "errors_at_0.5": int(((p >= .5) != y).sum()),
            "selective_coverage": float(routed.mean()),
            "selective_accuracy": float(((p[routed] >= .5) == y[routed]).mean()),
            "selective_errors": int(((p[routed] >= .5) != y[routed]).sum())}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    patients, _, images = load_cohort()
    images = images.merge(patients[["patient_alias", "role", "workbook_probe"]], on="patient_alias", validate="many_to_one")
    frames = load_frames(images)
    single = images.workbook_probe.isin(("convex", "linear")).to_numpy()
    y = images.workbook_probe.eq("convex").to_numpy().astype(np.float32)
    held = images.role.eq("exploratory_test").to_numpy()
    pool = np.flatnonzero(single & ~held)
    test = np.flatnonzero(single & held)
    everything = np.arange(len(images))

    oof = np.full(len(images), np.nan)
    ensemble = np.zeros(len(images))
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for k, (fit_idx, hold_idx) in enumerate(folds.split(pool, y[pool], images.patient_alias.iloc[pool])):
        model = fit(frames, y, pool[fit_idx], SEED + k)
        oof[pool[hold_idx]] = predict(model, frames, pool[hold_idx])
        ensemble += predict(model, frames, everything) / 5
        print("fold", k, summarize(y[pool[hold_idx]], oof[pool[hold_idx]]), flush=True)
    probability = np.where(np.isfinite(oof), oof, ensemble)  # never score a pool image with a model that saw it

    route = np.where(probability >= UPPER, "convex", np.where(probability <= LOWER, "linear", "unknown"))
    images["detector_convex_probability"] = probability
    images["detector_route"] = route
    geometry = pd.read_csv(OLD_ROUTER / "image_routes.csv").set_index("image_id").convex_probability
    images["geometry_convex_probability"] = images.image_id.map(geometry)
    # Final route used downstream: the detector alone. Workbook labels are only its training target.
    images["route"] = route

    report = {"model": "ResNet-18 (ImageNet init), 224x224 grayscale frame, 6 epochs AdamW 3e-4, intensity gain and "
                       "8-pixel shift augmentation",
              "labels": "workbook (2) Probe for single-probe patients; no APN label used",
              "abstention": f"linear if P(convex)<={LOWER}, convex if >={UPPER}, otherwise unknown",
              "cv_pool_patient_grouped_oof": summarize(y[pool], probability[pool]),
              "held_out_exploratory_test_probe_only": summarize(y[test], probability[test]),
              "geometry_random_forest_same_images": {
                  "cv_pool": summarize(y[pool], images.geometry_convex_probability.to_numpy()[pool]),
                  "held_out": summarize(y[test], images.geometry_convex_probability.to_numpy()[test])},
              "routes_all_images": images.route.value_counts().to_dict(),
              "routes_of_both_patients": images.loc[images.workbook_probe.eq("both"), "route"].value_counts().to_dict(),
              "pool_patients": int(images.patient_alias.iloc[pool].nunique()),
              "held_out_patients": int(images.patient_alias.iloc[test].nunique()),
              "APN_labels_used": False}
    OUT.mkdir(parents=True)
    images[["image_id", "patient_alias", "kidney_alias", "role", "workbook_probe", "detector_convex_probability",
            "geometry_convex_probability", "route"]].to_csv(OUT / "image_routes.csv", index=False)
    (OUT / "detector_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
