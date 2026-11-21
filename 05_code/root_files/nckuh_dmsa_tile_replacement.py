"""Replace representations in a fixed fusion without adding or tuning weights."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import audit_nckuh_fullcohort_ranking_20261103 as audit
import nckuh_dmsa_fullcohort_directional as directional
import nckuh_dmsa_inductive_graph as graph
import nckuh_dmsa_native_tile_probe as tile
import nckuh_dmsa_reflection_probe as reflection
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_triplet_rank_20261103 import fit_transformer, kidney_summary_features

BASE = tile.base
OUT = BASE.ROOT / "02_experiments/NCKUH_DMSA_tile_replacement_20261103"


def fixed_chain(deep, mil, nearest, texture):
    return .75 * (.75 * (.75 * deep + .25 * mil) + .25 * nearest) + .25 * texture


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    patients, kidneys, images = tile.load_reference()
    coverage = pd.read_csv(tile.OUT / "encoding_coverage.csv")
    np.testing.assert_array_equal(coverage.image_id, images.image_id)
    with np.load(tile.OUT / "tile_features.npz") as features:
        mean = features["mean"]
    oriented = BASE.orient_kidneys(BASE.patient_pair_features(mean, images, patients), patients, kidneys)
    local_kidney = kidney_summary_features(mean, images, kidneys)
    assignments = pd.read_csv(BASE.OUT / "patient_assignments.csv")
    y = kidneys.kidney_APN.to_numpy(int)
    paths = {
        "reflection_average": reflection.OUT,
        "cross_attention_q4": audit.MIL,
        "knn": graph.OUT,
        "aligned": directional.OUT,
        "aligned_fusion": directional.OUT,
        "tile_mean": tile.OUT,
        "tile_mean_fusion": tile.OUT,
    }
    sources = {}
    for name, folder in paths.items():
        path = folder / "development_predictions.csv"
        frame = pd.read_csv(path)
        sources[name] = frame[frame.group.eq(name)].set_index(["seed", "kidney_alias"], verify_integrity=True)
    records = []
    for seed in BASE.SEEDS:
        role = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        kidney_role = kidneys.patient_alias.map(role)
        train, dev = np.flatnonzero(kidney_role.eq("train")), np.flatnonzero(kidney_role.eq("development"))
        target = kidneys.iloc[dev]
        assert target.patient_alias.nunique() == 42 and kidneys.iloc[train].patient_alias.nunique() == 126
        index = [(seed, k) for k in target.kidney_alias]
        scores = {}
        for name, source in sources.items():
            selected = source.loc[index]
            np.testing.assert_array_equal(selected.patient_alias, target.patient_alias)
            np.testing.assert_array_equal(selected.side, target.side)
            np.testing.assert_array_equal(selected.label, y[dev])
            scores[name] = selected.score.to_numpy()
        q, k, t = (scores[name] for name in ("cross_attention_q4", "knn", "aligned"))
        np.testing.assert_allclose(fixed_chain(scores["reflection_average"], q, k, t),
                                   scores["aligned_fusion"], atol=1e-12, rtol=0)
        direct, _ = BASE.fit_deep(oriented, train, dev, y, seed, 32)
        tile_two_heads = (scores["tile_mean"] + direct) / 2
        _, x = fit_transformer(local_kidney, train, seed)
        tile_neighbors = graph.edges(x[dev], x[train], target.patient_alias.to_numpy(),
                                     kidneys.iloc[train].patient_alias.to_numpy()) @ y[train]
        results = {
            "aligned_fusion": scores["aligned_fusion"],
            "tile_mean_fusion": scores["tile_mean_fusion"],
            "replace_with_tile_image": fixed_chain(scores["tile_mean"], q, k, t),
            "replace_with_tile_two_heads": fixed_chain(tile_two_heads, q, k, t),
            "replace_with_tile_image_and_neighbors": fixed_chain(scores["tile_mean"], q, tile_neighbors, t),
        }
        for group, score in results.items():
            records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=group, label=y[dev], score=score))
    pred = pd.concat(records, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), BASE.SEEDS, "tile_mean_fusion")
    metrics = pd.DataFrame([dict(seed=seed, group=group, **BASE.metric(part.label, part.score))
                            for (seed, group), part in pred.groupby(["seed", "group"])])
    OUT.mkdir()
    pred.to_csv(OUT / "development_predictions.csv", index=False)
    metrics.to_csv(OUT / "development_metrics.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
