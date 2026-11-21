"""Fixed probe-specific frozen USFM single-encoder APN controls."""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from evaluate_nckuh_383_acquisition_strata_20261112 import bootstrap, fit_predict, metric
from prepare_nckuh_383_contour_cohort_20261111 import OUT as REGISTRY, ROOT, digest


EXP = ROOT / "02_experiments"
ROUTER = EXP / "NCKUH_revised_383_probe_router_20261112"
LEGACY = EXP / "NCKUH_revised_383_legacy_features_20261112"
OUT = ROUTER / "usfm_single_encoder_probe"
SEED = 20261112


def main():
    if (OUT / "results.json").exists():
        raise FileExistsError("Preserve USFM probe result")
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    people = pd.read_csv(REGISTRY / "patient_assignments.csv")
    routes = pd.read_csv(ROUTER / "image_routes.csv")
    assert routes.image_id.tolist() == images.image_id.tolist()
    source = np.load(LEGACY / "usfm_gap.npy", mmap_mode="r")
    assert source.shape == (len(images), 768) and np.isfinite(source).all()
    audit = json.loads((LEGACY / "usfm_audit.json").read_text(encoding="utf-8"))
    assert audit["images"] == len(images)
    assert digest(LEGACY / "usfm_gap.npy") == audit["sha256"]
    assert digest(REGISTRY / "contour_image_manifest.csv") == audit["manifest_sha256"]
    index = {image_id: i for i, image_id in enumerate(images.image_id)}
    report = {"image_routes_sha256": digest(ROUTER / "image_routes.csv"),
              "usfm_features_sha256": digest(LEGACY / "usfm_gap.npy"),
              "model": "Frozen USFM global average pooled image representation; training-image scaler/PCA32; routed kidney mean/std; balanced LR C=0.1.",
              "selection": "None: single prespecified encoder and head. Four-fold patient OOF is diagnostic only; no development-driven tuning.",
              "test_APN_scored": False, "groups": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    for probe in ("convex", "linear"):
        subset = images[images.image_id.isin(routes.loc[routes.route.eq(probe), "image_id"])].copy()
        subset = subset.merge(people[["patient_alias", "role", "DMSA_APN"]], on="patient_alias", validate="many_to_one")
        subset = subset[subset.role.isin(("train", "development"))].reset_index(drop=True)
        body = kidneys[kidneys.kidney_alias.isin(subset.kidney_alias)].reset_index(drop=True)
        roles = people[people.patient_alias.isin(subset.patient_alias) & people.role.isin(("train", "development"))]
        train_people = roles[roles.role.eq("train")].reset_index(drop=True)
        train_ids = set(train_people.patient_alias)
        dev_ids = set(roles.loc[roles.role.eq("development"), "patient_alias"])
        assert train_ids.isdisjoint(dev_ids)
        assert len(train_people) >= 50 and len(dev_ids) >= 20
        matrix = np.asarray(source[[index[image_id] for image_id in subset.image_id]], dtype=np.float64)
        train_rows = np.flatnonzero(body.patient_alias.isin(train_ids))
        dev_rows = np.flatnonzero(body.patient_alias.isin(dev_ids))
        oof = np.full(len(body), np.nan)
        cv = StratifiedKFold(4, shuffle=True, random_state=SEED)
        for fit, hold in cv.split(train_people, train_people.DMSA_APN):
            fit_ids = set(train_people.patient_alias.iloc[fit])
            hold_ids = set(train_people.patient_alias.iloc[hold])
            query = np.flatnonzero(body.patient_alias.isin(hold_ids))
            oof[query] = fit_predict("usfm_gap", matrix, subset, body, fit_ids, query)
        assert np.isfinite(oof[train_rows]).all()
        scores = fit_predict("usfm_gap", matrix, subset, body, train_ids, dev_rows)
        target = body.iloc[dev_rows].reset_index(drop=True)
        result = {"train_patients": len(train_ids), "train_kidneys": len(train_rows),
                  "development_patients": len(dev_ids), "development_kidneys": len(dev_rows),
                  "training_oof": metric(body.kidney_APN.iloc[train_rows], oof[train_rows]),
                  "development": metric(target.kidney_APN, scores) | bootstrap(target, scores)}
        report["groups"][probe] = result
        target[["patient_alias", "kidney_alias", "side", "kidney_APN"]].assign(
            score=scores
        ).to_csv(OUT / f"{probe}_development_predictions.csv", index=False)
        print(probe, result, flush=True)
    (OUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
