"""Compare fixed kidney-level pooling rules for focal APN image evidence."""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_tile_replacement as replacement
from nckuh_dmsa_cluster_ci import cluster_intervals


BASE = replacement.BASE
TILE = replacement.tile
OUT = BASE.ROOT / "02_experiments/NCKUH_DMSA_image_pooling_20261104"


def pool(scores: np.ndarray, images: pd.DataFrame, aliases: pd.Series, mode: str) -> np.ndarray:
    logits = np.log(np.clip(scores, 1e-6, 1 - 1e-6) / np.clip(1 - scores, 1e-6, 1))
    frame = pd.DataFrame({"kidney_alias": images.kidney_alias.to_numpy(), "logit": logits})
    values = []
    for alias in aliases:
        current = frame.loc[frame.kidney_alias.eq(alias), "logit"].to_numpy(float)
        if not len(current):
            raise ValueError(f"Missing image score for {alias}")
        if mode == "mean_logit":
            combined = current.mean()
        elif mode == "max_logit":
            combined = current.max()
        elif mode == "softmax2_logit":
            weight = np.exp(2 * (current - current.max()))
            combined = np.average(current, weights=weight)
        else:
            raise ValueError(mode)
        values.append(1 / (1 + np.exp(-np.clip(combined, -30, 30))))
    return np.asarray(values)


def run() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    patients, kidneys, images = TILE.load_reference()
    features = np.load(TILE.OUT / "tile_features.npz")["mean"]
    assignments = pd.read_csv(BASE.OUT / "patient_assignments.csv")
    y_kidney = kidneys.kidney_APN.to_numpy(int)
    y_image = images.kidney_APN.to_numpy(int)
    sources = {}
    for name, folder in {
        "cross_attention_q4": replacement.audit.MIL,
        "knn": replacement.graph.OUT,
        "aligned": replacement.directional.OUT,
    }.items():
        frame = pd.read_csv(folder / "development_predictions.csv")
        sources[name] = frame[frame.group.eq(name)].set_index(["seed", "kidney_alias"], verify_integrity=True)
    expected = pd.read_csv(replacement.OUT / "development_predictions.csv")
    expected = expected[expected.group.eq("replace_with_tile_two_heads")].set_index(
        ["seed", "kidney_alias"], verify_integrity=True)
    oriented = BASE.orient_kidneys(BASE.patient_pair_features(features, images, patients), patients, kidneys)

    records = []
    modes = ("mean_logit", "max_logit", "softmax2_logit")
    for seed in BASE.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        kidney_role = kidneys.patient_alias.map(mapping)
        image_role = images.patient_alias.map(mapping)
        kt = np.flatnonzero(kidney_role.eq("train"))
        kd = np.flatnonzero(kidney_role.eq("development"))
        it = np.flatnonzero(image_role.eq("train"))
        iv = np.flatnonzero(image_role.eq("development"))
        target = kidneys.iloc[kd]
        direct, _ = BASE.fit_deep(oriented, kt, kd, y_kidney, seed, 32)
        model = BASE.fit_image_head(features, it, y_image, BASE.image_weights(images, it, y_image), seed)
        image_scores = model.predict_proba(features[iv])[:, 1]
        index = [(seed, alias) for alias in target.kidney_alias]
        selected = {name: frame.loc[index] for name, frame in sources.items()}
        q, k, t = (selected[name].score.to_numpy() for name in ("cross_attention_q4", "knn", "aligned"))
        for frame in selected.values():
            np.testing.assert_array_equal(frame.label, y_kidney[kd])
            np.testing.assert_array_equal(frame.patient_alias, target.patient_alias)
        for mode in modes:
            image = pool(image_scores, images.iloc[iv], target.kidney_alias, mode)
            score = replacement.fixed_chain((image + direct) / 2, q, k, t)
            if mode == "mean_logit":
                reference = expected.loc[index]
                np.testing.assert_allclose(score, reference.score, atol=1e-10, rtol=0)
            records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=mode, label=y_kidney[kd], score=score))
    prediction = pd.concat(records, ignore_index=True)
    ci, delta, _ = cluster_intervals(prediction, list(modes), BASE.SEEDS, "mean_logit")
    metrics = pd.DataFrame([
        dict(seed=seed, group=group, **BASE.metric(part.label, part.score))
        for (seed, group), part in prediction.groupby(["seed", "group"])
    ])
    OUT.mkdir()
    prediction.to_csv(OUT / "development_predictions.csv", index=False)
    metrics.to_csv(OUT / "development_metrics.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
