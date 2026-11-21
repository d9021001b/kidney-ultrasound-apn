"""Select separate image-only APN heads after probe-based image routing."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from evaluate_nckuh_383_acquisition_strata_20261112 import (
    CANDIDATES, CACHE, DINO, LEGACY, SEED, bootstrap, fit_predict, metric,
    threshold_for_sensitivity,
)
from prepare_nckuh_383_contour_cohort_20261111 import OUT as REGISTRY, digest


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
ROUTER = EXP / "NCKUH_revised_383_probe_router_20261112"
OUT = ROUTER / "apn_method_screen"


def main():
    routing = json.loads((ROUTER / "router_validation.json").read_text(encoding="utf-8"))
    assert digest(REGISTRY / "cohort_lock.json") == routing["source_lock_sha256"]
    full_images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    full_kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    full_people = pd.read_csv(REGISTRY / "patient_assignments.csv")
    routes = pd.read_csv(ROUTER / "image_routes.csv")
    assert routes.image_id.tolist() == full_images.image_id.tolist()
    index = {key: i for i, key in enumerate(full_images.image_id)}
    raw = pd.read_csv(CACHE / "radiomics.csv").set_index("image_id").reindex(full_images.image_id)
    columns = [col for col in raw if not col.endswith(("_Energy", "_TotalEnergy"))]
    radiomics = raw[columns].to_numpy(float)
    radiomics[~np.isfinite(radiomics)] = np.nan
    sources = {
        "rad_color": np.load(LEGACY / "rad_color.npy", mmap_mode="r"),
        "tile": np.load(LEGACY / "tile_mean.npy", mmap_mode="r"),
        "dinov3_class": np.load(DINO / "dinov3_class.npy", mmap_mode="r"),
        "dinov3_patch": np.load(DINO / "dinov3_patch.npy", mmap_mode="r"),
        "uri_mean": np.load(CACHE / "uricads_fold_embeddings.npy", mmap_mode="r").mean(1),
        "radiomics": radiomics,
    }
    assert all(len(v) == len(full_images) for v in sources.values())
    report = {"router_sha256": digest(ROUTER / "router_validation.json"),
              "image_routes_sha256": digest(ROUTER / "image_routes.csv"),
              "candidates": list(CANDIDATES),
              "selection": "Within each probe group, four-fold patient-OOF maximize min(AUROC, AP), AUROC tie-break; no development selection.",
              "threshold": "Within each probe group, training OOF highest specificity with sensitivity >=0.85.",
              "unit": "Routed kidney-probe bag; a both-probe patient may contribute a kidney to each group, but patient roles remain locked.",
              "test_APN_scored": False, "groups": {}}
    OUT.mkdir(parents=True, exist_ok=True)

    for probe in ("convex", "linear"):
        images = full_images[full_images.image_id.isin(routes.loc[routes.route.eq(probe), "image_id"])].copy().reset_index(drop=True)
        people = full_people[full_people.patient_alias.isin(images.patient_alias) &
                             full_people.role.isin(("train", "development"))].copy().reset_index(drop=True)
        images = images[images.patient_alias.isin(people.patient_alias)].reset_index(drop=True)
        kidneys = full_kidneys[full_kidneys.kidney_alias.isin(images.kidney_alias)].copy().reset_index(drop=True)
        rows = np.asarray([index[key] for key in images.image_id])
        reps = {name: np.asarray(values[rows], dtype=np.float64) for name, values in sources.items()}
        train_people = people[people.role.eq("train")].reset_index(drop=True)
        train_kidneys = np.flatnonzero(kidneys.patient_alias.isin(train_people.patient_alias))
        dev_aliases = set(people.loc[people.role.eq("development"), "patient_alias"])
        dev = np.flatnonzero(kidneys.patient_alias.isin(dev_aliases))
        target = kidneys.iloc[dev].reset_index(drop=True)
        splitter = StratifiedKFold(4, shuffle=True, random_state=SEED)
        candidates = {}
        for name in CANDIDATES:
            oof = np.full(len(kidneys), np.nan)
            for fit, hold in splitter.split(train_people, train_people.DMSA_APN):
                fit_aliases = set(train_people.patient_alias.iloc[fit])
                hold_aliases = set(train_people.patient_alias.iloc[hold])
                query = np.flatnonzero(kidneys.patient_alias.isin(hold_aliases))
                oof[query] = fit_predict(name, reps[name], images, kidneys, fit_aliases, query)
            assert np.isfinite(oof[train_kidneys]).all()
            candidates[name] = metric(kidneys.kidney_APN.iloc[train_kidneys], oof[train_kidneys])
            candidates[name]["scores"] = oof[train_kidneys]
        winner = max(CANDIDATES, key=lambda name: (min(candidates[name]["auroc"], candidates[name]["ap"]),
                                                   candidates[name]["auroc"]))
        labels = kidneys.kidney_APN.iloc[train_kidneys].to_numpy(int)
        threshold, sensitivity = threshold_for_sensitivity(labels, candidates[winner]["scores"])
        for name in CANDIDATES:
            candidates[name].pop("scores")
        score = fit_predict(winner, reps[winner], images, kidneys, set(train_people.patient_alias), dev)
        dev_labels = target.kidney_APN.to_numpy(int)
        predicted = score >= threshold
        result = {"train_patients": len(train_people), "development_patients": len(people) - len(train_people),
                  "train_kidneys": len(train_kidneys), "development_kidneys": len(dev),
                  "development_positive_fraction": float(dev_labels.mean()),
                  "training_oof": candidates, "selected_method": winner,
                  "training_oof_threshold": threshold, "training_oof_sensitivity": sensitivity,
                  "development": metric(dev_labels, score) | bootstrap(target, score) | {
                      "sensitivity_at_training_threshold": float(predicted[dev_labels == 1].mean()),
                      "specificity_at_training_threshold": float((~predicted[dev_labels == 0]).mean()),
                  }}
        report["groups"][probe] = result
        target[["patient_alias", "kidney_alias", "side", "kidney_APN"]].assign(
            probe_route=probe, selected_method=winner, score=score, prediction=predicted.astype(int)
        ).to_csv(OUT / f"{probe}_development_predictions.csv", index=False)
        print(probe, winner, result["development"], flush=True)
    (OUT / "separate_probe_development_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
