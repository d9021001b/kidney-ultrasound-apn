"""Use within-patient angle reassignment to control directional resampling."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_directional as d
from nckuh_dmsa_cluster_ci import cluster_intervals

OUT = d.base.ROOT / "02_experiments/NCKUH_DMSA_rotation_control_20261103"


def run():
    from extract_site import run_pool

    if OUT.exists():
        raise FileExistsError(OUT)
    patients, kidneys, images = d.load_reference()
    old = pd.read_csv(d.OUT / "directional_features.csv").set_index("image_id")
    angles = old.loc[images.image_id].alignment_degrees.to_numpy()
    sham = pd.Series(angles).groupby(images.patient_alias).transform(lambda a: np.roll(a.to_numpy(), 1))
    np.testing.assert_array_equal(np.sort(sham), np.sort(angles))
    jobs = [(i, d.base.OUT / "percentile" / f"{i}.png", d.base.OUT / "masks" / f"{i}.png", a)
            for i, a in zip(images.image_id, sham)]
    frame = run_pool(d.worker, jobs, processes=8)
    assert frame.shape == (820, 26)
    values = d.base.orient_kidneys(d.base.patient_pair_features(
        frame.filter(regex="^aligned_").to_numpy(), images, patients), patients, kidneys)
    previous = pd.read_csv(d.OUT / "development_predictions.csv")
    assignments = pd.read_csv(d.base.OUT / "patient_assignments.csv")
    y = kidneys.kidney_APN.to_numpy(int)
    rows = [previous]
    for seed in d.base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        roles = kidneys.patient_alias.map(mapping)
        train, dev = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
        target = kidneys.iloc[dev]
        assert target.patient_alias.nunique() == 42 and kidneys.iloc[train].patient_alias.nunique() == 126
        score, _, _ = d.fit_anova(values, train, dev, y, seed, 64)
        ref = previous[previous.seed.eq(seed) & previous.group.eq("fixed_reference")].set_index(
            "kidney_alias").loc[target.kidney_alias]
        np.testing.assert_array_equal(ref.label, y[dev])
        for group, s in (("reassigned_angle", score),
                         ("reassigned_angle_fusion", .75 * ref.score.to_numpy() + .25 * score)):
            rows.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=group, label=y[dev], score=s))
    pred = pd.concat(rows, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), d.base.SEEDS, "reassigned_angle")
    _, paired, _ = cluster_intervals(pred, ["aligned_fusion", "reassigned_angle_fusion"],
                                     d.base.SEEDS, "reassigned_angle_fusion")
    OUT.mkdir()
    frame.to_csv(OUT / "reassigned_angle_features.csv")
    pred.to_csv(OUT / "development_predictions.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    pd.concat([delta, paired]).to_csv(OUT / "paired_delta_ci.csv", index=False)
    delta_angle = (angles - sham.to_numpy() + 90) % 180 - 90
    print("Median absolute angular change:", np.median(np.abs(delta_angle)), flush=True)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
