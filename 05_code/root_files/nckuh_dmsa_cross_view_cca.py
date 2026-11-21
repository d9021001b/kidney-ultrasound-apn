"""Compare view-specific and canonical shared kidney features on fixed draws."""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_fullcohort_radimagenet_finetune_20261103 import load_reference
from nckuh_dmsa_inductive_graph import OUT as GRAPH
from nckuh_dmsa_reflection_probe import OUT as REFLECTION

OUT = base.ROOT / "02_experiments/NCKUH_DMSA_cross_view_cca_20261103"
ARMS = ("independent_pca", "cca_shared", "cca_shared_and_difference")


def view_means(values, images, kidney_ids):
    features = pd.DataFrame(values)
    assert images.view.isin(["LL", "LT", "RL", "RT"]).all()
    return [features[images.view.str.endswith(s)].groupby(images.kidney_alias).mean()
            .reindex(kidney_ids).to_numpy() for s in ("L", "T")]


def prepare_features(view_arrays, train):
    available = np.stack([np.isfinite(x).all(1) for x in view_arrays], axis=1)
    assert available.any(1).all()
    complete = available.all(1)
    fitting = train[complete[train]]
    compressed = []
    for col, raw in enumerate(view_arrays):
        valid = available[:, col]
        pca = PCA(16, whiten=True, svd_solver="full").fit(raw[fitting])
        transformed = np.zeros((len(raw), 16))
        transformed[valid] = pca.transform(raw[valid])
        compressed.append(transformed)
    left, right = compressed
    cca = CCA(8, scale=False, max_iter=2000, tol=1e-7).fit(left[fitting], right[fitting])
    a, b = cca.transform(left, right)
    a[~available[:, 0]] = 0
    b[~available[:, 1]] = 0
    average = (a + b) / available.sum(1, keepdims=True)
    difference = np.abs(a - b)
    difference[~complete] = 0
    return {
        ARMS[0]: np.c_[left[:, :8], right[:, :8], available],
        ARMS[1]: np.c_[average, available],
        ARMS[2]: np.c_[average, difference, available],
    }, complete


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    _, kidneys, images = load_reference()
    manifest = pd.read_csv(base.OUT / "deidentified_source_manifest.csv")
    assert images.image_id.tolist() == manifest.image_id.tolist()
    features = np.load(base.OUT / "radimagenet_color.npy")
    reflected = np.load(REFLECTION / "radimagenet_color_horizontal_reflection.npy")
    views = view_means((features + reflected) / 2, images, kidneys.kidney_alias)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    reference = pd.read_csv(GRAPH / "development_predictions.csv")
    reference = reference[reference.group.eq("knn_fusion")].set_index(["seed", "kidney_alias"])
    records, coverage = [], []
    y = kidneys.kidney_APN.to_numpy(int)
    for seed in base.SEEDS:
        roles = kidneys.patient_alias.map(assignments[assignments.seed.eq(seed)].set_index("patient_alias").role)
        train = np.flatnonzero(roles.eq("train"))
        dev = np.flatnonzero(roles.eq("development"))
        assert kidneys.iloc[train].patient_alias.nunique() == 126
        assert kidneys.iloc[dev].patient_alias.nunique() == 42
        vectors, complete = prepare_features(views, train)
        target = kidneys.iloc[dev]
        ref = reference.loc[[(seed, k) for k in target.kidney_alias]]
        np.testing.assert_array_equal(ref.label, y[dev])
        np.testing.assert_array_equal(ref.patient_alias, target.patient_alias)
        scores = {"fixed_reference": ref.score.to_numpy()}
        for arm, x in vectors.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=.1, class_weight="balanced", solver="liblinear", random_state=seed))
            model.fit(x[train], y[train])
            scores[arm] = model.predict_proba(x[dev])[:, 1]
            scores[arm + "_fusion"] = .75 * scores["fixed_reference"] + .25 * scores[arm]
        coverage.append(dict(seed=seed, fit_complete=int(complete[train].sum()),
                             dev_complete=int(complete[dev].sum()), dev_total=len(dev)))
        for group, score in scores.items():
            records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=group, label=y[dev], score=score))
    pred = pd.concat(records, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), base.SEEDS, "fixed_reference")
    OUT.mkdir()
    for name, table in (("development_predictions", pred), ("development_ci", ci),
                        ("paired_delta_ci", delta), ("view_coverage", pd.DataFrame(coverage))):
        table.to_csv(OUT / f"{name}.csv", index=False)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
