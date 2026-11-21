"""Compare supervised compression with fixed PCA and uncompressed tile heads."""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import nckuh_dmsa_tile_replacement as parent
from nckuh_dmsa_cluster_ci import cluster_intervals

BASE = parent.BASE
OUT = BASE.ROOT / "02_experiments/NCKUH_DMSA_tile_supervised_projection_20261103"


def fit_predict(values, train, dev, y, seed, mode, weights=None):
    transform = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    train_x = transform.fit_transform(values[train])
    dev_x = transform.transform(values[dev])
    if mode == "pls8":
        projection = PLSRegression(n_components=8, scale=False).fit(train_x, y[train])
        train_x, dev_x = projection.transform(train_x), projection.transform(dev_x)
        scaler = StandardScaler()
        train_x, dev_x = scaler.fit_transform(train_x), scaler.transform(dev_x)
    head = LogisticRegression(C=.1, class_weight="balanced" if weights is None else None,
                              solver="liblinear", max_iter=3000, random_state=seed)
    return head.fit(train_x, y[train], sample_weight=weights).predict_proba(dev_x)[:, 1]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    patients, kidneys, images = parent.tile.load_reference()
    manifest = pd.read_csv(parent.tile.OUT / "encoding_coverage.csv")
    np.testing.assert_array_equal(manifest.image_id, images.image_id)
    with np.load(parent.tile.OUT / "tile_features.npz") as f:
        mean = f["mean"]
    oriented = BASE.orient_kidneys(BASE.patient_pair_features(mean, images, patients), patients, kidneys)
    yi, yk = images.kidney_APN.to_numpy(int), kidneys.kidney_APN.to_numpy(int)
    roles = pd.read_csv(BASE.OUT / "patient_assignments.csv")
    paths = {"cross_attention_q4": parent.audit.MIL, "knn": parent.graph.OUT,
             "aligned": parent.directional.OUT, "replace_with_tile_two_heads": parent.OUT}
    reference = {}
    for name, folder in paths.items():
        saved = pd.read_csv(folder / "development_predictions.csv")
        reference[name] = saved[saved.group.eq(name)].set_index(["seed", "kidney_alias"], verify_integrity=True)
    records = []
    for seed in BASE.SEEDS:
        mapping = roles[roles.seed.eq(seed)].set_index("patient_alias").role
        kr, ir = kidneys.patient_alias.map(mapping), images.patient_alias.map(mapping)
        kt, kd = np.flatnonzero(kr.eq("train")), np.flatnonzero(kr.eq("development"))
        it, iv = np.flatnonzero(ir.eq("train")), np.flatnonzero(ir.eq("development"))
        target = kidneys.iloc[kd]
        assert target.patient_alias.nunique() == 42 and kidneys.iloc[kt].patient_alias.nunique() == 126
        idx = [(seed, k) for k in target.kidney_alias]
        fixed = {}
        for name, saved in reference.items():
            selected = saved.loc[idx]
            np.testing.assert_array_equal(selected.label, yk[kd])
            np.testing.assert_array_equal(selected.patient_alias, target.patient_alias)
            fixed[name] = selected.score.to_numpy()
        q, k, t = (fixed[name] for name in ("cross_attention_q4", "knn", "aligned"))
        weights = BASE.image_weights(images, it, yi)
        for mode in ("pca32", "pls8", "regularized_linear"):
            if mode == "pca32":
                model = BASE.fit_image_head(mean, it, yi, weights, seed)
                image_scores = model.predict_proba(mean[iv])[:, 1]
                kidney_score, _ = BASE.fit_deep(oriented, kt, kd, yk, seed, 32)
            else:
                image_scores = fit_predict(mean, it, iv, yi, seed, mode, weights)
                kidney_score = fit_predict(oriented, kt, kd, yk, seed, mode)
            image_score = BASE.aggregate_image_logits(image_scores, images.iloc[iv], target.kidney_alias)
            combined = parent.fixed_chain((image_score + kidney_score) / 2, q, k, t)
            if mode == "pca32":
                np.testing.assert_allclose(combined, fixed["replace_with_tile_two_heads"], atol=1e-5, rtol=0)
            records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=mode, label=yk[kd], score=combined))
    pred = pd.concat(records, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), BASE.SEEDS, "pca32")
    OUT.mkdir()
    pred.to_csv(OUT / "development_predictions.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
