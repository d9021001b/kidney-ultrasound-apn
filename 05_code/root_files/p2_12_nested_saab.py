"""Paper 2, step 12: nested selection with Green-learning (native Saab) features added to the candidate pool.

Same outer folds, inner folds, selection rule and fixed head as `p2_05_nested_frozen.py`. The only change
is one extra candidate source, `green_saab`, with own and paired views. Its basis is fitted on the outer
training part of each fold (label-free, training patients only), so the held-out fold never enters
selection or fitting. Every choice is made inside the outer training part; the held-out fold is scored
once per probe group. The exploratory-test role is never read.
"""

import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from p2_03_landscape import bag, kidney_table, paired
from p2_05_nested_frozen import TOP, fit_predict, shared_folds, to_rank
from p2_common import FOLD_IDS, P2_OUT, SEED, cluster_interval, metric, pool_mask
from p2_features import image_sources, routed_images
import p2_11_green_saab as S

OUT = P2_OUT / "nested_saab"


def saab_designs_for_fold(cache, pool, tables, k):
    """Saab kidney designs for one outer fold: basis fitted on training-part patients only."""
    train_patients = sorted(pool.loc[(pool.fold >= 0) & (pool.fold != k), "patient_alias"].unique())
    sampling = pool[["image_id", "patient_alias"]].rename(columns={"patient_alias": "patient_id"})
    units = {kernel: S.G.fit_saab_unit(S.L.training_patches(cache, sampling, train_patients, SEED + k, kernel)[0])
             for kernel in S.L.KERNELS}
    feat = S.image_saab_features(cache, units, pool.image_id)
    out = {}
    for probe, (table, members) in tables.items():
        own = S.kidney_saab_features(feat, members)
        out[probe] = {"own": own, "paired": paired(own, table)}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    patients, _, images = routed_images()
    pool_images = images[pool_mask(images)].reset_index(drop=True)
    fold_of = shared_folds(patients, pool_images)
    pool = pool_images.assign(fold=pool_images.patient_alias.map(fold_of).astype(int))
    cache = S.build_patch_cache(pool)
    sources = image_sources(images)
    report = {"protocol": __doc__, "test_APN_scored": False, "groups": {}}
    frames = []
    for probe in ("convex", "linear"):
        table, members = kidney_table(images, probe)
        pool_table, pool_members = kidney_table(pool, probe)
        assert table.kidney_alias.tolist() == pool_table.kidney_alias.tolist()
        y, groups = table.label.to_numpy(int), table.patient_alias.to_numpy()
        fold = table.patient_alias.map(fold_of).to_numpy()
        frozen_designs = {}
        for name, values in sources.items():
            own = bag(values, members)
            for view, x in (("own", own), ("paired", paired(own, table))):
                for components in ((16, 32) if x.shape[1] > 40 else (0,)):
                    for c in (0.01, 0.1):
                        frozen_designs[(name, view, components, c)] = x
        frozen_designs[("age", "own", 0, 0.1)] = table[["age"]].to_numpy(float)
        score = np.full(len(y), np.nan)
        chosen = []
        for k in FOLD_IDS:
            saab = saab_designs_for_fold(cache, pool, {probe: (pool_table, pool_members)}, k)[probe]
            designs = dict(frozen_designs)
            for view, x in saab.items():
                for components in (16, 32):
                    for c in (0.01, 0.1):
                        designs[("green_saab", view, components, c)] = x
            fit, hold = np.flatnonzero((fold != k) & (fold >= 0)), np.flatnonzero(fold == k)
            inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=SEED + k).split(fit, y[fit], groups[fit]))
            best = {}
            for key, x in designs.items():
                oof = np.zeros(len(fit))
                for a, b in inner:
                    oof[b] = fit_predict(x, y, fit[a], fit[b], key[2], key[3])
                m = metric(y[fit], oof)
                value = (min(m["auroc"], m["ap"] - y[fit].mean() + 0.5), m["auroc"])
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
        yk, sk, gk = y[keep], score[keep], groups[keep]
        result = metric(yk, sk) | cluster_interval(gk, yk, sk)
        result |= {"patients": int(np.unique(gk).size), "kidneys": len(yk), "prevalence": float(yk.mean()),
                   "source_selection_counts": dict(Counter(key[0] for key in chosen).most_common()),
                   "view_selection_counts": dict(Counter(key[1] for key in chosen))}
        report["groups"][probe] = result
        frames.append(table[keep][["kidney_alias", "patient_alias", "side", "label"]].assign(route=probe, fold=fold[keep], score=sk))
        print(probe, json.dumps(result), flush=True)
    pd.concat(frames, ignore_index=True).to_csv(OUT / "kidney_oof_scores.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    with threadpool_limits(limits=6):
        main()
