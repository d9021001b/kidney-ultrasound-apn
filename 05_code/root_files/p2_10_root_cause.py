"""Paper 2 root-cause analysis: where does kidney-level discrimination come from, and what caps it?

1. Dedicated tasks on the development pool (5 shared patient folds, repeated 3x with different fold seeds):
   - patient-level any-APN (N vs L/R/B) from side-symmetric features (mean of both kidneys, |difference|);
   - laterality among unilateral patients from the signed left-minus-right feature difference;
   - bilateral vs unilateral among APN patients.
   Each uses the same fixed head (train-only standardise, PCA16, balanced LR C=0.1) for several frozen sources.
2. Structural ceiling by simulation: kidney-level AUROC/AP that would result if within-patient laterality
   were perfect while between-patient information were absent, and for intermediate between-patient skill.
3. Learning curves: kidney-level and any-APN performance versus fraction of training patients.
No exploratory-test patient is used.
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from p2_03_landscape import head
from p2_common import P2, SEED
from p2_features import image_sources, routed_images

OUT = P2 / "root_cause"
SOURCES = ("rad_tile", "rad_color", "rad_dense", "usfm", "uricads", "dinov3_class", "radiomics", "geometry")


def patient_tables(images, sources, probe):
    sub = images[images.role.isin(("train", "development")) & (images.route.eq(probe) if probe != "all" else True)]
    rows = {k: g.index.to_numpy() for k, g in sub.groupby(["patient_alias", "side"])}
    people = sub.drop_duplicates("patient_alias")[["patient_alias", "DMSA_APN", "age"]].reset_index(drop=True)
    both = np.asarray([(a, "left") in rows and (a, "right") in rows for a in people.patient_alias])
    people = people[both].reset_index(drop=True)
    feats = {}
    for name in SOURCES:
        values = sources[name]
        left = np.stack([np.nanmean(values[rows[(a, "left")]], 0) for a in people.patient_alias])
        right = np.stack([np.nanmean(values[rows[(a, "right")]], 0) for a in people.patient_alias])
        feats[name] = (left, right)
    return people, feats


def cv_auc(x, y, repeats=3, components=16, fraction=1.0):
    y = np.asarray(y, int)
    aucs, aps = [], []
    for r in range(repeats):
        score = np.zeros(len(y))
        rng = np.random.default_rng(SEED + r)
        for fit, hold in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(x, y):
            if fraction < 1:
                keep = np.concatenate([rng.choice(fit[y[fit] == c], max(4, int(round(fraction * (y[fit] == c).sum()))),
                                                  replace=False) for c in (0, 1)])
                fit = keep
            k = min(components, len(fit) - 1, x.shape[1])
            score[hold] = head(k, 0.1).fit(x[fit], y[fit]).predict_proba(x[hold])[:, 1]
        aucs.append(roc_auc_score(y, score))
        aps.append(average_precision_score(y, score))
    return float(np.mean(aucs)), float(np.mean(aps))


def ceiling(people, between_auc_target, trials=400, seed=SEED):
    """Kidney-level AUROC/AP when the affected kidney always outranks its normal partner and a
    patient-level score separates APN from N patients with the requested AUROC (0.5 = none)."""
    rng = np.random.default_rng(seed)
    from scipy.stats import norm
    shift = np.sqrt(2) * norm.ppf(between_auc_target)
    aucs, aps = [], []
    for _ in range(trials):
        labels, scores = [], []
        for state in people.DMSA_APN:
            base = rng.normal(shift if state != "N" else 0.0, 1.0)
            left, right = {"N": (0, 0), "L": (1, 0), "R": (0, 1), "B": (1, 1)}[state]
            jitter = np.sort(rng.normal(0, 0.35, 2))
            if left == right:
                pair = rng.permutation(jitter)
            else:
                pair = (jitter[1], jitter[0]) if left else (jitter[0], jitter[1])
            labels += [left, right]
            scores += [base + pair[0], base + pair[1]]
        aucs.append(roc_auc_score(labels, scores))
        aps.append(average_precision_score(labels, scores))
    return float(np.mean(aucs)), float(np.mean(aps))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, images = routed_images()
    images = images.reset_index(drop=True)
    sources = image_sources(images)
    report = {"protocol": __doc__, "tasks": {}, "ceiling": {}, "learning_curve": {}}
    for probe in ("convex", "linear", "all"):
        people, feats = patient_tables(images, sources, probe)
        state = people.DMSA_APN.to_numpy()
        any_apn, uni, apn = state != "N", np.isin(state, ["L", "R"]), state != "N"
        block = {"patients": len(people), "states": pd.Series(state).value_counts().to_dict(), "sources": {}}
        for name, (left, right) in feats.items():
            symmetric = np.c_[(left + right) / 2, np.abs(left - right)]
            signed = left - right
            block["sources"][name] = {
                "any_apn_auroc": cv_auc(symmetric, any_apn)[0],
                "laterality_auroc": cv_auc(signed[uni], state[uni] == "L")[0],
                "bilateral_vs_unilateral_auroc": cv_auc(symmetric[apn], state[apn] == "B")[0]}
        block["age_only_any_apn_auroc"] = cv_auc(people[["age"]].fillna(people.age.median()).to_numpy(), any_apn, components=0)[0]
        for key in ("any_apn_auroc", "laterality_auroc", "bilateral_vs_unilateral_auroc"):
            values = [v[key] for v in block["sources"].values()]
            block[f"{key}_range"] = [float(min(values)), float(max(values)), float(np.median(values))]
        report["tasks"][probe] = block
        print(probe, block["states"], {k: [round(x, 3) for x in block[f"{k}_range"]] for k in
                                       ("any_apn_auroc", "laterality_auroc", "bilateral_vs_unilateral_auroc")},
              "age-only any-APN", round(block["age_only_any_apn_auroc"], 3), flush=True)
        report["ceiling"][probe] = {f"between_patient_auroc_{b}": dict(zip(("kidney_auroc", "kidney_ap"), ceiling(people, b)))
                                    for b in (0.5, 0.6, 0.7, 0.8, 0.9)}
        print(probe, "ceiling with PERFECT laterality:", {k: (round(v["kidney_auroc"], 3), round(v["kidney_ap"], 3))
                                                           for k, v in report["ceiling"][probe].items()}, flush=True)
        left, right = feats["rad_tile"]
        symmetric = np.c_[(left + right) / 2, np.abs(left - right)]
        report["learning_curve"][probe] = {str(f): {"any_apn_auroc": cv_auc(symmetric, any_apn, fraction=f)[0],
                                                    "laterality_auroc": cv_auc((left - right)[uni], state[uni] == "L", fraction=f)[0]}
                                           for f in (0.25, 0.5, 0.75, 1.0)}
        print(probe, "learning curve (rad_tile):", {f: {k: round(v, 3) for k, v in d.items()}
                                                     for f, d in report["learning_curve"][probe].items()}, flush=True)
    (OUT / "root_cause.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    with threadpool_limits(limits=8):
        main()
