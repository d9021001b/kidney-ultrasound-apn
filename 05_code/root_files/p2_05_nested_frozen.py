"""Paper 2, step 5: probe-specific frozen-feature model with nested selection.

Outer: the five shared patient folds (identical to the end-to-end step). Inside each outer training
part, every (source, view, PCA size, C) candidate gets 4-fold patient-grouped OOF scores; each probe
group keeps, per source, its best configuration and then the three best sources by min(AUROC, AP).
Those three are refitted on the whole outer training part and their held-out scores are averaged
after rank-normalisation against the training scores. All selection therefore happens without the
held-out outer fold, and each probe group ends up with its own feature set.
"""

import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from threadpoolctl import threadpool_limits

from p2_03_landscape import bag, head, kidney_table, paired
from p2_common import FOLD_IDS, P2, P2_OUT, SEED, assign_folds, cluster_interval, metric, pool_mask
from p2_features import image_sources, routed_images

OUT = P2_OUT / "nested_frozen"
TOP = 3


def shared_folds(patients, images):
    pool = patients[patients.patient_alias.isin(images.patient_alias)].reset_index(drop=True)
    return assign_folds(pool)


def fit_predict(x, y, fit, query, components, c):
    k = min(components, len(fit) - 1, x.shape[1]) if components else 0
    return head(k, c).fit(x[fit], y[fit]).predict_proba(x[query])[:, 1]


def to_rank(reference, values):
    return np.searchsorted(np.sort(reference), values, side="right") / len(reference)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    patients, _, images = routed_images()
    pool_images = images[pool_mask(images)]
    fold_of = shared_folds(patients, pool_images)
    sources = image_sources(images)
    report, frames = {"protocol": __doc__, "test_APN_scored": False, "groups": {}}, []
    for probe in ("convex", "linear"):
        table, members = kidney_table(images, probe)
        y, groups = table.label.to_numpy(int), table.patient_alias.to_numpy()
        fold = table.patient_alias.map(fold_of).to_numpy()
        designs = {}
        for name, values in sources.items():
            own = bag(values, members)
            for view, x in (("own", own), ("paired", paired(own, table))):
                for components in ((16, 32) if x.shape[1] > 40 else (0,)):
                    for c in (0.01, 0.1):
                        designs[(name, view, components, c)] = x
        designs[("age", "own", 0, 0.1)] = table[["age"]].to_numpy(float)
        score = np.full(len(y), np.nan)
        chosen = []
        for k in FOLD_IDS:
            fit, hold = np.flatnonzero((fold != k) & (fold >= 0)), np.flatnonzero(fold == k)
            inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=SEED + k).split(fit, y[fit], groups[fit]))
            best = {}
            for key, x in designs.items():
                oof = np.zeros(len(fit))
                for a, b in inner:
                    oof[b] = fit_predict(x, y, fit[a], fit[b], key[2], key[3])
                m = metric(y[fit], oof)
                value = (min(m["auroc"], m["ap"] - y[fit].mean() + 0.5), m["auroc"])  # AP judged against prevalence
                if key[0] not in best or value > best[key[0]][0]:
                    best[key[0]] = (value, key, oof)
            top = sorted(best.values(), key=lambda item: item[0], reverse=True)[:TOP]
            parts = []
            for _, key, oof in top:
                parts.append(to_rank(oof, fit_predict(designs[key], y, fit, hold, key[2], key[3])))
                chosen.append(key)
            score[hold] = np.mean(parts, axis=0)
            print(probe, "fold", k, [f"{key[0]}/{key[1]}" for _, key, _ in top],
                  {n: round(v, 3) for n, v in metric(y[hold], score[hold]).items()}, flush=True)
        keep = np.isin(fold, FOLD_IDS)
        y, score, groups = y[keep], score[keep], groups[keep]
        result = metric(y, score) | cluster_interval(groups, y, score)
        result |= {"patients": int(np.unique(groups).size), "kidneys": len(y), "prevalence": float(y.mean()),
                   "source_selection_counts": dict(Counter(key[0] for key in chosen).most_common()),
                   "view_selection_counts": dict(Counter(key[1] for key in chosen))}
        report["groups"][probe] = result
        frames.append(table[keep][["kidney_alias", "patient_alias", "side", "label"]].assign(route=probe, fold=fold[keep], score=score))
        print(probe, json.dumps(result), flush=True)
    pd.concat(frames, ignore_index=True).to_csv(OUT / "kidney_oof_scores.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    with threadpool_limits(limits=6):
        main()
