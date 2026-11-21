"""Paper 2, step 11: Green-learning (single-stage native Saab) probe-specific model, and fusion.

Reuses the paper-1 Saab machinery: `green_learning_cv.fit_saab_unit` for the basis and
`renal_local_features` for native 5x5/9x9 patches, training-patch sampling and the five response
statistics. Per shared fold k (the same five patient-grouped folds as the other P2 models):

1. The Saab basis is fitted only on the training-fold patients (512 patches per patient per kernel,
   balanced across their images). No held-out-fold patch is touched.
2. Every routed development-pool image gets 530 Saab statistics (106 channels x mean, SD, P10, P50, P90).
3. Each kidney, within its probe group, gets the mean and SD over its routed images (1,060 values); the
   `own` view uses these alone, the `paired` view adds the contralateral kidney and the difference.
4. A PCA32 whitened, class-balanced logistic head (C = 0.1, fixed before scoring) is trained on the
   training-fold kidneys and predicts the held-out fold.

Fusion `fusion_saab_frozen` is a fixed equal-weight mean of within-group ranks of this score and the
nested frozen-feature score (no fitted weights). The exploratory-test role is never read.
"""

import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "02_experiments/CYCH_feature_research_results_20260913/source"))
import green_learning_cv as G  # noqa: E402
import renal_local_features as L  # noqa: E402

from p2_03_landscape import bag, head, kidney_table, paired  # noqa: E402
from p2_common import CACHE, FOLD_IDS, P2_OUT, SEED, cluster_interval, metric, pool_mask  # noqa: E402
from p2_05_nested_frozen import shared_folds  # noqa: E402
from p2_features import routed_images  # noqa: E402

OUT = P2_OUT / "green_saab"
PROBES = ("convex", "linear")
VIEWS = ("own", "paired")
COMPONENTS, C = 32, 0.1
FROZEN = P2_OUT / "nested_frozen/kidney_oof_scores.csv"


def build_patch_cache(pool):
    cache = {k: {} for k in L.KERNELS}
    for row in pool.itertuples():
        gray = cv2.imread(str(CACHE / "percentile" / f"{row.image_id}.png"), 0)
        mask = (cv2.imread(str(CACHE / "masks" / f"{row.image_id}.png"), 0) > 0).astype(np.uint8)
        assert gray is not None and gray.shape == mask.shape and mask.any()
        assert not gray[mask == 0].any(), row.image_id
        for k in L.KERNELS:
            cache[k][row.image_id] = L.get_patches(gray, mask, k, row.image_id)[0]
    return cache


def image_saab_features(cache, units, image_ids):
    return np.stack([np.r_[L.local_statistics(cache[5][i], units[5], 5),
                           L.local_statistics(cache[9][i], units[9], 9)] for i in image_ids])


def kidney_saab_features(feat, members):
    return np.stack([np.r_[feat[m].mean(0), feat[m].std(0)] for m in members])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    patients, _, images = routed_images()
    pool = images[pool_mask(images)].reset_index(drop=True)
    fold_of = shared_folds(patients, pool)
    pool["fold"] = pool.patient_alias.map(fold_of).astype(int)
    cache = build_patch_cache(pool)
    sampling = pool[["image_id", "patient_alias"]].rename(columns={"patient_alias": "patient_id"})
    tables = {probe: kidney_table(pool, probe) for probe in PROBES}
    oof = {(probe, view): [] for probe in PROBES for view in VIEWS}
    basis_audit = []
    for k in FOLD_IDS:
        train_patients = sorted(pool.loc[(pool.fold >= 0) & (pool.fold != k), "patient_alias"].unique())
        units = {}
        for kernel in L.KERNELS:
            samples, audit = L.training_patches(cache, sampling, train_patients, SEED + k, kernel)
            units[kernel] = G.fit_saab_unit(samples)
            basis_audit.append(dict(fold=int(k), kernel=kernel, patients=len(train_patients), patches=int(len(samples))))
        feat = image_saab_features(cache, units, pool.image_id)
        assert feat.shape[1] == 530 and np.isfinite(feat).all()
        for probe in PROBES:
            table, members = tables[probe]
            own = kidney_saab_features(feat, members)
            fold = table.patient_alias.map(fold_of).to_numpy()
            y = table.label.to_numpy(int)
            fit, hold = (fold >= 0) & (fold != k), fold == k
            for view in VIEWS:
                x = own if view == "own" else paired(own, table)
                model = head(COMPONENTS, C).fit(x[fit], y[fit])
                score = model.predict_proba(x[hold])[:, 1]
                oof[(probe, view)].append(table[hold][["kidney_alias", "patient_alias", "side", "label"]]
                                          .assign(route=probe, view=view, fold=k, score=score))
        print("fold", k, "done", flush=True)

    frames, report = [], {"protocol": __doc__, "test_APN_scored": False, "basis_fits": basis_audit, "groups": {}}
    frozen = pd.read_csv(FROZEN)
    for probe in PROBES:
        for view in VIEWS:
            frame = pd.concat(oof[(probe, view)], ignore_index=True)
            frames.append(frame)
            y, groups, s = frame.label.to_numpy(int), frame.patient_alias.to_numpy(), frame.score.to_numpy(float)
            report["groups"][f"{probe}/saab_{view}"] = metric(y, s) | cluster_interval(groups, y, s) | {
                "kidneys": len(y), "patients": int(np.unique(groups).size), "prevalence": float(y.mean())}
        saab = pd.concat([f for f in frames if f.route.iloc[0] == probe and f.view.iloc[0] == "own"], ignore_index=True)
        merged = saab.merge(frozen[frozen.route.eq(probe)][["kidney_alias", "score"]].rename(columns={"score": "frozen"}),
                            on="kidney_alias", validate="one_to_one")
        n = len(merged)
        merged["fused"] = (rankdata(merged.score) / n + rankdata(merged.frozen) / n) / 2
        y, groups, s = merged.label.to_numpy(int), merged.patient_alias.to_numpy(), merged.fused.to_numpy(float)
        report["groups"][f"{probe}/fusion_saab_frozen"] = metric(y, s) | cluster_interval(groups, y, s) | {
            "kidneys": len(y), "patients": int(np.unique(groups).size), "prevalence": float(y.mean())}
        frames.append(merged.assign(view="fusion_saab_frozen")[["kidney_alias", "patient_alias", "side", "label",
                                                              "route", "view", "fused"]].rename(columns={"fused": "score"}))
    pd.concat(frames, ignore_index=True).to_csv(OUT / "kidney_oof_scores.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for key, value in report["groups"].items():
        print(key, round(value["auroc"], 3), round(value["ap"], 3), value["auroc_95pct"], value["ap_95pct"], flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    with threadpool_limits(limits=6):
        main()
