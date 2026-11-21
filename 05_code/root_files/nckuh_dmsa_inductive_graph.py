"""Train-only kidney graph with independent inductive development queries."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_fullcohort_radimagenet_finetune_20261103 import load_reference
from nckuh_dmsa_reflection_mil_fusion import NEW, OUT as FUSION
from nckuh_dmsa_reflection_probe import OUT as REFLECTION
from nckuh_dmsa_triplet_rank_20261103 import fit_transformer, kidney_summary_features

OUT = base.ROOT / "02_experiments/NCKUH_DMSA_inductive_graph_20261103"


def edges(query, bank, query_patients, bank_patients, k=8):
    distance = cdist(query, bank, metric="euclidean")
    weights = np.zeros_like(distance)
    for i, row in enumerate(distance):
        seen = {query_patients[i]}
        chosen = []
        for j in np.argsort(row, kind="stable"):
            patient = bank_patients[j]
            if patient not in seen:
                seen.add(patient)
                chosen.append(j)
                if len(chosen) == k:
                    break
        if len(chosen) != k:
            raise ValueError("Not enough distinct eligible training patients")
        local = row[chosen]
        raw = np.exp(-(local / max(local[-1], 1e-12)) ** 2)
        weights[i, chosen] = raw / raw.sum()
    return weights


def diffuse(weights, labels, alpha=0.5):
    return np.linalg.solve(np.eye(len(weights)) - alpha * weights, (1 - alpha) * labels)


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    _, kidneys, images = load_reference()
    manifest = pd.read_csv(base.OUT / "deidentified_source_manifest.csv")
    assert images.image_id.tolist() == manifest.image_id.tolist()
    original = np.load(base.OUT / "radimagenet_color.npy")
    mirrored = np.load(REFLECTION / "radimagenet_color_horizontal_reflection.npy")
    raw = kidney_summary_features((original + mirrored) / 2, images, kidneys)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    reference = pd.read_csv(FUSION / "development_predictions.csv")
    reference = reference[reference.group.eq(NEW)].set_index(["seed", "kidney_alias"])
    records, metrics = [], []
    for seed in base.SEEDS:
        roles = kidneys.patient_alias.map(assignments[assignments.seed.eq(seed)].set_index("patient_alias").role)
        train = np.flatnonzero(roles.eq("train"))
        dev = np.flatnonzero(roles.eq("development"))
        bank, target = kidneys.iloc[train], kidneys.iloc[dev]
        assert bank.patient_alias.nunique() == 126 and target.patient_alias.nunique() == 42
        assert set(bank.patient_alias).isdisjoint(target.patient_alias)
        _, values = fit_transformer(raw, train, seed)
        labels = bank.kidney_APN.to_numpy(float)
        bank_ids, query_ids = bank.patient_alias.to_numpy(), target.patient_alias.to_numpy()
        graph = edges(values[train], values[train], bank_ids, bank_ids)
        query = edges(values[dev], values[train], query_ids, bank_ids)
        ref = reference.loc[[(seed, alias) for alias in target.kidney_alias]]
        np.testing.assert_array_equal(ref.label, target.kidney_APN)
        np.testing.assert_array_equal(ref.patient_alias, target.patient_alias)
        direct = query @ labels
        propagated = query @ diffuse(graph, labels)
        scores = {"fixed_reference": ref.score.to_numpy(), "knn": direct, "diffusion": propagated}
        for arm in ("knn", "diffusion"):
            scores[arm + "_fusion"] = .75 * scores["fixed_reference"] + .25 * scores[arm]
        for group, score in scores.items():
            assert np.isfinite(score).all() and ((score >= -1e-12) & (score <= 1 + 1e-12)).all()
            score = np.clip(score, 0, 1)
            records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=group, label=target.kidney_APN.astype(int), score=score))
            metrics.append(dict(seed=seed, group=group, **base.metric(target.kidney_APN, score)))
    pred = pd.concat(records, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), base.SEEDS, "fixed_reference")
    _, diff, _ = cluster_intervals(pred, ["knn", "diffusion"], base.SEEDS, "knn")
    OUT.mkdir()
    for name, frame in (("development_predictions", pred), ("development_metrics", pd.DataFrame(metrics)),
                        ("development_ci", ci), ("paired_delta_ci", pd.concat([delta, diff]))):
        frame.to_csv(OUT / f"{name}.csv", index=False)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
