"""Fixed age and view-distance controls for the kidney nearest-neighbor branch."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_clinical_features_20261103 as clinical
import nckuh_dmsa_tile_replacement as parent
from nckuh_dmsa_cluster_ci import cluster_intervals

BASE = parent.BASE
OUT = BASE.ROOT / "02_experiments/NCKUH_DMSA_context_neighbors_20261104"


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    _, kidneys, images = parent.tile.load_reference()
    manifest = pd.read_csv(BASE.OUT / "deidentified_source_manifest.csv")
    np.testing.assert_array_equal(manifest.image_id, images.image_id)
    features = (np.load(BASE.OUT / "radimagenet_color.npy") +
                np.load(parent.reflection.OUT / "radimagenet_color_horizontal_reflection.npy")) / 2
    raw = parent.kidney_summary_features(features, images, kidneys)
    demo = pd.read_csv(clinical.OUT / "deidentified_demographics.csv").set_index("patient_alias")
    age = np.log1p(kidneys.patient_alias.map(demo.age_years).to_numpy(float))
    views = pd.crosstab(images.kidney_alias, images.view.str[-1]).reindex(kidneys.kidney_alias)[["L", "T"]]
    planes = views.gt(0).to_numpy(float)
    assignments = pd.read_csv(BASE.OUT / "patient_assignments.csv")
    paths = {"knn": parent.graph.OUT, "cross_attention_q4": parent.audit.MIL,
             "aligned": parent.directional.OUT, "replace_with_tile_two_heads": parent.OUT}
    sources = {}
    for name, folder in paths.items():
        rows = pd.read_csv(folder / "development_predictions.csv")
        sources[name] = rows[rows.group.eq(name)].set_index(["seed", "kidney_alias"], verify_integrity=True)
    y = kidneys.kidney_APN.to_numpy(int)
    records = []
    for seed in BASE.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        roles = kidneys.patient_alias.map(mapping)
        train, dev = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
        target = kidneys.iloc[dev]
        assert target.patient_alias.nunique() == 42 and kidneys.iloc[train].patient_alias.nunique() == 126
        selected = {}
        for name, frame in sources.items():
            rows = frame.loc[[(seed, k) for k in target.kidney_alias]]
            np.testing.assert_array_equal(rows.patient_alias, target.patient_alias)
            np.testing.assert_array_equal(rows.label, y[dev])
            selected[name] = rows.score.to_numpy()
        _, x = parent.fit_transformer(raw, train, seed)
        scale = np.median(pdist(x[train]))
        age_iqr = np.nanpercentile(age[train], 75) - np.nanpercentile(age[train], 25)
        centered = (age - np.nanmedian(age[train])) / max(age_iqr, 1e-6)
        context = np.nan_to_num(centered, nan=0.)[:, None] * scale * .5
        context_view = planes * scale * .5
        query_ids, bank_ids = target.patient_alias.to_numpy(), kidneys.iloc[train].patient_alias.to_numpy()

        def predict(values):
            return parent.graph.edges(values[dev], values[train], query_ids, bank_ids) @ y[train]

        original = predict(x)
        np.testing.assert_allclose(original, selected["knn"], atol=1e-10, rtol=0)
        arms = {"original_fusion": selected["replace_with_tile_two_heads"]}
        for name, vectors in (("age", np.c_[x, context]), ("age_and_view", np.c_[x, context, context_view])):
            near = predict(vectors)
            arms[name + "_knn"] = near
            # The neighbor probability enters the two outer 75/25 layers.
            arms[name + "_fusion"] = arms["original_fusion"] + (.75 * .25) * (near - original)
        arms["original_knn"] = original
        for name, scores in arms.items():
            records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=name, label=y[dev], score=scores))
    pred = pd.concat(records, ignore_index=True)
    groups = list(pred.group.unique())
    ci, delta, _ = cluster_intervals(pred, groups, BASE.SEEDS, "original_fusion")
    _, direct, _ = cluster_intervals(pred, ["original_knn", "age_knn", "age_and_view_knn"],
                                     BASE.SEEDS, "original_knn")
    OUT.mkdir()
    pred.to_csv(OUT / "development_predictions.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    pd.concat([delta, direct]).to_csv(OUT / "paired_delta_ci.csv", index=False)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
