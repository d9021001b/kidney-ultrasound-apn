"""Codex-protocol test: add recovered-label patients to training only.

Same three 126/42 patient roles, same development patients, same fixed-chain reference,
same hierarchical tile+USFM model and seeds as
`nckuh_dmsa_hierarchical_tile_usfm_20261104.py`. The only change: the 107 recovered
patients (`NCKUH_DMSA_recovered_labels_20261104`, none in the 210 cohort) are appended to
every training role. Exploratory-test patients are neither trained on nor scored.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import torch

import nckuh_dmsa_hierarchical_tile_usfm_20261104 as ref
import nckuh_dmsa_usfm_frozen as usfm_mod

base, hierarchy, cohort = ref.base, ref.hierarchy, ref.cohort
REC = base.ROOT / "02_experiments/NCKUH_DMSA_recovered_labels_20261104"
OUT = base.ROOT / "02_experiments/NCKUH_DMSA_recovered_augmented_hierarchy_20261105"


def usfm_gap(root, ids):
    model, _, _ = usfm_mod.construct_model()
    model = model.cuda().eval().requires_grad_(False)
    mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1)
    rows = []
    with torch.inference_mode():
        for start in range(0, len(ids), 24):
            canvases = []
            for ident in ids[start:start + 24]:
                bgr = cv2.imread(str(root / "clean_rgb" / f"{ident}.png"), cv2.IMREAD_COLOR)
                mask = cv2.imread(str(root / "masks" / f"{ident}.png"), cv2.IMREAD_GRAYSCALE) > 0
                canvases.append(base.tight_canvas(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask)[0])
            x = torch.from_numpy(np.stack(canvases).transpose(0, 3, 1, 2).copy()).cuda().float() / 255
            rows.append(usfm_mod.all_tokens(model, (x - mean) / std)[:, 1:].mean(1).cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def recovered():
    plans = [pd.read_csv(REC / name, dtype=str) for name in
             ("recovered_images_private.csv", "recovered_images_stage2_private.csv")]
    plan = pd.concat(plans, ignore_index=True)
    feats = [dict(np.load(REC / name)) for name in ("recovered_features.npz", "recovered_features_stage2.npz")]
    tile = np.r_[feats[0]["tile_mean"], feats[1]["tile_mean"]]
    return plan, tile


def run() -> None:
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(20261105)
    torch.backends.cuda.matmul.allow_tf32 = False
    patients, kidneys, images = cohort.load_reference()
    old = ref.image_representations(images)
    plan, new_tile = recovered()

    # USFM for recovered images, plus reproduction check against the stored cache.
    new_usfm = usfm_gap(REC, plan.image_id.tolist())
    check = usfm_gap(base.OUT, images.image_id.iloc[:24].tolist())
    usfm_repro = float(np.abs(check - old["usfm"][:24]).max())
    np.save(OUT / "recovered_usfm_gap.npy", new_usfm)

    new_patients = plan.drop_duplicates("patient_alias")[["patient_alias", "DMSA_APN"]]
    all_patients = pd.concat([patients[["patient_alias", "DMSA_APN"]], new_patients], ignore_index=True)
    all_images = pd.concat([images[["patient_alias", "side", "view"]],
                            plan[["patient_alias", "side", "view"]]], ignore_index=True)
    reps = {
        "tile_usfm": np.r_[old["tile_usfm"], np.c_[new_tile, new_usfm]],
        "tile": np.r_[old["tile"], new_tile],
    }
    structured = {name: hierarchy.structured_views(base.patient_pair_features(v, all_images, all_patients))
                  for name, v in reps.items()}
    states = all_patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    n_old = len(patients)
    recovered_idx = np.arange(n_old, len(all_patients))

    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    fixed = pd.read_csv(ref.FIXED / "development_predictions.csv")
    fixed = fixed[fixed.group.eq("replace_with_tile_two_heads")].set_index(["seed", "kidney_alias"])
    saved = pd.read_csv(ref.OUT / "development_predictions.csv")

    metrics, predictions = [], []
    for seed in base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train_old = np.flatnonzero(patients.patient_alias.map(mapping).eq("train"))
        dev = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        target = kidneys.iloc[dev]
        reference = fixed.loc[[(seed, a) for a in target.kidney_alias]].score.to_numpy()
        scores = {"fixed_chain_reference": reference}
        for name, (sym, anti) in structured.items():
            for tag, train in (("original126", train_old),
                               ("augmented", np.r_[train_old, recovered_idx])):
                prob, _ = hierarchy.state_probabilities(sym, anti, states, train, seed)
                score = hierarchy.kidney_scores(prob, all_patients, target)
                scores[f"{name}_{tag}"] = score
                scores[f"{name}_{tag}_quarter_plus_fixed"] = 0.25 * score + 0.75 * reference
        # reproduction of the saved Codex result
        prior = saved[(saved.seed == seed) & (saved.group == "hierarchical_tile_usfm_quarter_plus_fixed")]
        repro = float(np.abs(prior.score.to_numpy() - scores["tile_usfm_original126_quarter_plus_fixed"]).max())
        for group, score in scores.items():
            metrics.append({"seed": seed, "group": group, **hierarchy.metric(labels[dev], score),
                            "repro_maxabs_vs_saved": repro})
            for local, i in enumerate(dev):
                predictions.append({"seed": seed, "group": group, "patient_alias": kidneys.patient_alias.iloc[i],
                                    "kidney_alias": kidneys.kidney_alias.iloc[i], "label": int(labels[i]),
                                    "score": float(score[local])})
        print(f"seed {seed} done, repro {repro:.2e}", flush=True)

    m = pd.DataFrame(metrics)
    m.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    summary = m.groupby("group")[["auroc", "ap"]].agg(["mean", "std"]).round(4)
    summary.to_csv(OUT / "development_summary.csv")
    (OUT / "audit.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recovered_patients_added_to_training": int(len(recovered_idx)),
        "recovered_images": int(len(plan)),
        "usfm_repro_maxabs": usfm_repro,
        "codex_result_repro_maxabs": float(m.repro_maxabs_vs_saved.max()),
        "exploratory_test_scored": False,
    }, indent=2), encoding="utf-8")
    print(summary.to_string(), flush=True)
    print("usfm repro", usfm_repro, flush=True)


if __name__ == "__main__":
    run()
