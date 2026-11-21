"""Fixed dual-kernel logistic comparison for deep and aligned texture features."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_directional as d
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_reflection_probe import OUT as REFLECTION
from nckuh_dmsa_triplet_rank_20261103 import fit_transformer

OUT = d.base.ROOT / "02_experiments/NCKUH_DMSA_dual_kernel_20261103"


def kernel(train, evaluation):
    square = cdist(train, train, "sqeuclidean")
    nonzero = square[square > 0]
    if not len(nonzero):
        raise ValueError("Constant training representation")
    scale = np.median(nonzero)
    return np.exp(-square / scale), np.exp(-cdist(evaluation, train, "sqeuclidean") / scale)


def score_kernel(gram, cross, labels, seed):
    lam, basis = np.linalg.eigh(gram)
    retained = lam > 1e-8
    width = np.sqrt(lam[retained])
    basis = basis[:, retained]
    fit = basis * width
    evaluation = (cross @ basis) / width
    model = LogisticRegression(C=1., class_weight="balanced", solver="lbfgs", random_state=seed)
    return model.fit(fit, labels).predict_proba(evaluation)[:, 1]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    patients, kidneys, images = d.load_reference()
    texture = pd.read_csv(d.OUT / "directional_features.csv").set_index("image_id")
    np.testing.assert_array_equal(texture.index, images.image_id)
    deep = (np.load(d.base.OUT / "radimagenet_color.npy") +
            np.load(REFLECTION / "radimagenet_color_horizontal_reflection.npy")) / 2
    raw_deep, raw_texture = [
        d.base.orient_kidneys(d.base.patient_pair_features(a, images, patients), patients, kidneys)
        for a in (deep, texture.filter(regex="^aligned_").to_numpy())]
    assignments = pd.read_csv(d.base.OUT / "patient_assignments.csv")
    reference = pd.read_csv(d.OUT / "development_predictions.csv")
    reference = reference[reference.group.eq("aligned_fusion")]
    records = [reference]
    y = kidneys.kidney_APN.to_numpy(int)
    for seed in d.base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        roles = kidneys.patient_alias.map(mapping)
        train, dev = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
        target = kidneys.iloc[dev]
        assert target.patient_alias.nunique() == 42 and kidneys.iloc[train].patient_alias.nunique() == 126
        _, deep_x = fit_transformer(raw_deep, train, seed)
        _, texture_transform, _ = d.fit_anova(raw_texture, train, dev, y, seed, 64)
        texture_train = texture_transform.transform(raw_texture[train])
        texture_dev = texture_transform.transform(raw_texture[dev])
        kd, vd = kernel(deep_x[train], deep_x[dev])
        kt, vt = kernel(texture_train, texture_dev)
        ref = reference[reference.seed.eq(seed)].set_index("kidney_alias").loc[target.kidney_alias]
        np.testing.assert_array_equal(ref.label, y[dev])
        np.testing.assert_array_equal(ref.patient_alias, target.patient_alias)
        scores = {
            "deep_kernel": score_kernel(kd, vd, y[train], seed),
            "texture_kernel": score_kernel(kt, vt, y[train], seed),
            "dual_kernel": score_kernel((kd + kt) / 2, (vd + vt) / 2, y[train], seed),
        }
        scores["dual_kernel_fusion"] = .25 * scores["dual_kernel"] + .75 * ref.score.to_numpy()
        for group, score in scores.items():
            records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=group, label=y[dev], score=score))
    pred = pd.concat(records, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), d.base.SEEDS, "aligned_fusion")
    _, control, _ = cluster_intervals(pred, ["deep_kernel", "dual_kernel", "texture_kernel"],
                                      d.base.SEEDS, "deep_kernel")
    OUT.mkdir()
    pred.to_csv(OUT / "development_predictions.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    pd.concat([delta, control]).to_csv(OUT / "paired_delta_ci.csv", index=False)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
