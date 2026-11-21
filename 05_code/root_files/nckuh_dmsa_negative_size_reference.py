"""Compare all-training and DMSA-N-only age-conditioned size references."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_clinical_features_20261103 as clinical
import nckuh_dmsa_fullcohort_directional as d
import nckuh_dmsa_fullcohort_tick_geometry_20261103 as ticks
from nckuh_dmsa_cluster_ci import cluster_intervals

OUT = d.base.ROOT / "02_experiments/NCKUH_DMSA_negative_size_reference_20261103"


def size_residual(values, age, view, training, eligible):
    good = training & eligible & np.isfinite(age) & np.isfinite(values).all(1)
    if good.sum() < 20:
        raise ValueError("Insufficient age and tick coverage for reference")
    log_age = np.log1p(np.where(np.isfinite(age), age, np.median(age[good])))
    context = np.c_[log_age, log_age ** 2, np.eye(4)[view]]
    model = make_pipeline(StandardScaler(), Ridge(alpha=15.))
    model.fit(context[good], values[good])
    residual = values - model.predict(context)
    scale = np.maximum(residual[good].std(axis=0), .05)
    residual /= scale
    residual[~np.isfinite(age)] = np.nan
    return residual, int(good.sum())


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    patients, kidneys, images = d.load_reference()
    frame = pd.read_csv(ticks.OUT / "tick_geometry.csv").set_index("image_id").loc[images.image_id]
    dimensions = np.c_[frame.tick_00, frame.tick_01, np.sqrt(frame.tick_05)]
    sizes = np.log(np.where(dimensions > 0, dimensions, np.nan))
    demographics = pd.read_csv(clinical.OUT / "deidentified_demographics.csv").set_index("patient_alias")
    age = images.patient_alias.map(demographics.age_years).to_numpy()
    assert (age[np.isfinite(age)] >= 0).all()
    view = images.view.map({"LL": 0, "LT": 1, "RL": 2, "RT": 3}).to_numpy(int)
    state = images.patient_alias.map(patients.set_index("patient_alias").DMSA_APN).to_numpy()
    assignments = pd.read_csv(d.base.OUT / "patient_assignments.csv")
    reference = pd.read_csv(d.OUT / "development_predictions.csv")
    reference = reference[reference.group.eq("aligned_fusion")]
    records, coverage = [reference], []
    y = kidneys.kidney_APN.to_numpy(int)
    for seed in d.base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        roles = kidneys.patient_alias.map(mapping)
        train, dev = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
        fit_images = images.patient_alias.map(mapping).eq("train").to_numpy()
        target = kidneys.iloc[dev]
        assert target.patient_alias.nunique() == 42 and kidneys.iloc[train].patient_alias.nunique() == 126
        ref = reference[reference.seed.eq(seed)].set_index("kidney_alias").loc[target.kidney_alias]
        np.testing.assert_array_equal(ref.label, y[dev])
        np.testing.assert_array_equal(ref.patient_alias, target.patient_alias)
        all_adjusted, all_n = size_residual(sizes, age, view, fit_images, np.ones(len(images), bool))
        negative_adjusted, n = size_residual(sizes, age, view, fit_images, state == "N")
        coverage.append(dict(seed=seed, all_reference_images=all_n, negative_reference_images=n))
        for arm, values in (("raw_size", sizes), ("all_reference", all_adjusted), ("N_reference", negative_adjusted)):
            x = d.base.orient_kidneys(d.base.patient_pair_features(values, images, patients), patients, kidneys)
            model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                  LogisticRegression(C=.1, class_weight="balanced", solver="liblinear", random_state=seed))
            score = model.fit(x[train], y[train]).predict_proba(x[dev])[:, 1]
            for group, s in ((arm, score), (arm + "_fusion", .75 * ref.score.to_numpy() + .25 * score)):
                records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                    seed=seed, group=group, label=y[dev], score=s))
    pred = pd.concat(records, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), d.base.SEEDS, "aligned_fusion")
    _, control, _ = cluster_intervals(pred, ["all_reference", "N_reference"], d.base.SEEDS, "all_reference")
    OUT.mkdir()
    pred.to_csv(OUT / "development_predictions.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    pd.concat([delta, control]).to_csv(OUT / "paired_delta_ci.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "fit_coverage.csv", index=False)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
