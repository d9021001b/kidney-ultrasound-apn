"""Test regional texture after real contour-axis alignment."""
import cv2
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_directional as d
from nckuh_dmsa_cluster_ci import cluster_intervals

OUT = d.base.ROOT / "02_experiments/NCKUH_DMSA_regional_texture_20261103"


def regions(mask):
    _, x = np.nonzero(mask)
    edges = np.linspace(x.min(), x.max() + 1, 4)
    xx = np.arange(mask.shape[1])[None, :]
    return [mask * ((xx >= lo) & (xx < hi)) for lo, hi in zip(edges, edges[1:])]


def worker(job):
    ident, gray_path, mask_path, names = job
    gray = cv2.imread(str(gray_path), 0)
    mask = (cv2.imread(str(mask_path), 0) > 0).astype(np.uint8)
    _, (image, roi), _ = d.matched_canvases(gray, mask)
    features = []
    for region in regions(roi):
        y, x = np.nonzero(region)
        if len(x) < 64 or np.ptp(x) < 1 or np.ptp(y) < 1:
            features.append(np.full(len(names), np.nan))
        else:
            extracted = d.extract_texture(image, region)
            features.append(np.asarray([extracted[k] for k in names]))
    a, mid, b = features
    row = {"image_id": ident, "unavailable_regions": sum(np.isnan(v).all() for v in features)}
    for key, values in (("middle", mid), ("end_low", np.minimum(a, b)),
                        ("end_high", np.maximum(a, b)), ("end_difference", np.abs(a - b))):
        row.update({f"{key}_{name}": value for name, value in zip(names, values)})
    return row


def run():
    from extract_site import run_pool

    if OUT.exists():
        raise FileExistsError(OUT)
    patients, kidneys, images = d.load_reference()
    whole = pd.read_csv(d.OUT / "directional_features.csv").set_index("image_id").loc[images.image_id]
    whole = whole.filter(regex="^aligned_")
    names = [c.removeprefix("aligned_") for c in whole.columns]
    jobs = [(i, d.base.OUT / "percentile" / f"{i}.png", d.base.OUT / "masks" / f"{i}.png", names)
            for i in images.image_id]
    regional = run_pool(worker, jobs, processes=8)
    assert regional.shape == (820, 101) and regional.index.equals(whole.index)
    fields = regional.drop(columns="unavailable_regions").to_numpy()
    combined = np.c_[whole.to_numpy(), fields]
    vectors = {a: d.base.orient_kidneys(d.base.patient_pair_features(
        values, images, patients), patients, kidneys)
        for a, values in (("regional", fields), ("whole_and_regional", combined))}
    previous = pd.read_csv(d.OUT / "development_predictions.csv")
    previous = previous[previous.group.isin(["fixed_reference", "aligned", "aligned_fusion"])]
    assignments = pd.read_csv(d.base.OUT / "patient_assignments.csv")
    y = kidneys.kidney_APN.to_numpy(int)
    rows = [previous]
    for seed in d.base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        roles = kidneys.patient_alias.map(mapping)
        train, dev = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
        target = kidneys.iloc[dev]
        assert target.patient_alias.nunique() == 42 and kidneys.iloc[train].patient_alias.nunique() == 126
        ref = previous[previous.seed.eq(seed) & previous.group.eq("fixed_reference")].set_index(
            "kidney_alias").loc[target.kidney_alias]
        np.testing.assert_array_equal(ref.label, y[dev])
        np.testing.assert_array_equal(ref.patient_alias, target.patient_alias)
        for arm, x in vectors.items():
            score, _, _ = d.fit_anova(x, train, dev, y, seed, 64)
            for group, s in ((arm, score), (arm + "_fusion", .75 * ref.score.to_numpy() + .25 * score)):
                rows.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                    seed=seed, group=group, label=y[dev], score=s))
    pred = pd.concat(rows, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), d.base.SEEDS, "fixed_reference")
    _, paired, _ = cluster_intervals(pred, ["aligned_fusion", "regional_fusion", "whole_and_regional_fusion"],
                                     d.base.SEEDS, "aligned_fusion")
    OUT.mkdir()
    regional.to_csv(OUT / "regional_features.csv")
    pred.to_csv(OUT / "development_predictions.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    pd.concat([delta, paired]).to_csv(OUT / "paired_delta_ci.csv", index=False)
    print("Unavailable regions:", int(regional.unavailable_regions.sum()), flush=True)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
