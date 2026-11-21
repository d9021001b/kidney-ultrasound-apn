"""Paper 2, step 3: per-probe landscape of frozen representations (development pool only).

Unit: routed kidney-probe bag (mean of that kidney's routed image vectors). Two views of each
kidney are compared: `own` (the kidney alone) and `paired` (own, contralateral, own-minus-other).
Repeated patient-grouped 5-fold CV on train+development roles. This is a descriptive screen;
the honest estimate comes from the nested selection in a later step. No test-role scoring.
"""

import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from p2_common import P2, SEED, metric, pool_mask
from p2_features import image_sources, routed_images

OUT = P2 / "landscape"
REPEATS = 3


def kidney_table(images, probe):
    """Kidneys with at least one image routed to `probe`, in the development pool."""
    sub = images[images.route.eq(probe) & pool_mask(images)]
    table = sub.groupby("kidney_alias", sort=False).agg(
        patient_alias=("patient_alias", "first"), side=("side", "first"),
        label=("kidney_APN", "first"), state=("DMSA_APN", "first"), age=("age", "first")).reset_index()
    members = [np.flatnonzero((images.kidney_alias == k).to_numpy() & images.route.eq(probe).to_numpy())
               for k in table.kidney_alias]
    return table, members


def bag(values, members):
    return np.stack([np.nanmean(values[m], axis=0) for m in members])


def paired(own, table):
    """Concatenate own, contralateral (zeros with indicator if absent) and difference."""
    lookup = {(p, s): i for i, (p, s) in enumerate(zip(table.patient_alias, table.side))}
    other_idx = [lookup.get((p, "right" if s == "left" else "left"), -1) for p, s in zip(table.patient_alias, table.side)]
    has = np.asarray([j >= 0 for j in other_idx])
    other = np.where(has[:, None], own[np.maximum(other_idx, 0)], own)
    return np.c_[own, other, own - other]


def head(components, c):
    steps = [SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()]
    if components:
        steps += [PCA(components, whiten=True, random_state=SEED)]
    return make_pipeline(*steps, LogisticRegression(C=c, class_weight="balanced", max_iter=3000))


def cv_scores(x, table, components, c, repeats=REPEATS):
    y, groups = table.label.to_numpy(int), table.patient_alias.to_numpy()
    out = np.zeros((repeats, len(y)))
    for r in range(repeats):
        for fit, hold in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r).split(x, y, groups):
            k = min(components, len(fit) - 1, x.shape[1]) if components else 0
            out[r, hold] = head(k, c).fit(x[fit], y[fit]).predict_proba(x[hold])[:, 1]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, images = routed_images()
    sources = image_sources(images)
    rows = []
    for probe in ("convex", "linear"):
        table, members = kidney_table(images, probe)
        y = table.label.to_numpy(int)
        print(probe, "patients", table.patient_alias.nunique(), "kidneys", len(table), "prevalence", round(y.mean(), 3), flush=True)
        for name, values in sources.items():
            own = bag(values, members)
            for view, x in (("own", own), ("paired", paired(own, table))):
                for components in ((16, 32) if x.shape[1] > 40 else (0,)):
                    for c in (0.01, 0.1):
                        scores = cv_scores(x, table, components, c)
                        per = [metric(y, s) for s in scores]
                        rows.append({"probe": probe, "source": name, "view": view, "pca": components, "C": c,
                                     "auroc": np.mean([m["auroc"] for m in per]), "ap": np.mean([m["ap"] for m in per])})
            best = max((r for r in rows if r["probe"] == probe and r["source"] == name), key=lambda r: r["auroc"])
            print(f"  {name:18s} best {best['view']:6s} pca{best['pca']:<3d} C{best['C']:<5} AUROC {best['auroc']:.3f} AP {best['ap']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "frozen_feature_landscape.csv", index=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    with threadpool_limits(limits=8):
        main()
