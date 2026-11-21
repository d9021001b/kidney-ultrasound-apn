"""Measure acquisition-only discrimination and a train-only residualization control."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_fullcohort_radimagenet_finetune_20261103 import load_reference
from nckuh_dmsa_inductive_graph import OUT as GRAPH
from nckuh_dmsa_reflection_probe import OUT as REFLECTION
from nckuh_dmsa_triplet_rank_20261103 import fit_transformer, kidney_summary_features

OUT = base.ROOT / "02_experiments/NCKUH_DMSA_acquisition_audit_20261103"


def pair_summary(draw):
    p, n = draw[draw.label.eq(1)], draw[draw.label.eq(0)]
    difference = p.score.to_numpy()[:, None] - n.score.to_numpy()
    same = p.signature.to_numpy()[:, None] == n.signature.to_numpy()
    across = p.patient_alias.to_numpy()[:, None] != n.patient_alias.to_numpy()
    concordance = (difference > 0).astype(float) + .5 * (difference == 0)
    return [{"stratum": kind, "pairs": int(select.sum()), "concordant": float(concordance[select].sum())}
            for kind, select in (("matched_signature", same & across), ("different_signature", ~same & across))]


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    _, kidneys, images = load_reference()
    manifest = pd.read_csv(base.OUT / "deidentified_source_manifest.csv")
    np.testing.assert_array_equal(manifest.image_id, images.image_id)
    qc = pd.read_csv(base.OUT / "deidentified_extraction_qc.csv").set_index("image_id").loc[images.image_id]
    np.testing.assert_array_equal(qc.patient_alias, images.patient_alias)
    context = pd.DataFrame(dict(view=images.view, source=qc.source_type.to_numpy(),
                                dimensions=(qc.frame_width.astype(str) + "x" + qc.frame_height.astype(str)).to_numpy()))
    tokens = context.view.str[-1] + ":" + context.source + ":" + context.dimensions
    signature = tokens.groupby(images.kidney_alias).agg(lambda s: "|".join(sorted(s))).reindex(kidneys.kidney_alias)
    assert signature.notna().all()
    x = (np.load(base.OUT / "radimagenet_color.npy") +
         np.load(REFLECTION / "radimagenet_color_horizontal_reflection.npy")) / 2
    raw = kidney_summary_features(x, images, kidneys)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    reference = pd.read_csv(GRAPH / "development_predictions.csv")
    reference = reference[reference.group.eq("knn_fusion")].set_index(["seed", "kidney_alias"])
    y = kidneys.kidney_APN.to_numpy(int)
    records, ranks = [], []
    for seed in base.SEEDS:
        role = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        image_fit = images.patient_alias.map(role).eq("train").to_numpy()
        train = np.flatnonzero(kidneys.patient_alias.map(role).eq("train"))
        dev = np.flatnonzero(kidneys.patient_alias.map(role).eq("development"))
        target = kidneys.iloc[dev]
        assert target.patient_alias.nunique() == 42 and kidneys.iloc[train].patient_alias.nunique() == 126
        design = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(context[image_fit]).transform(context)
        ridge = Ridge(alpha=20).fit(design[image_fit], x[image_fit])
        corrected = x - ridge.predict(design)
        residual = kidney_summary_features(corrected, images, kidneys)
        rows = pd.DataFrame(design).groupby(images.kidney_alias).mean().reindex(kidneys.kidney_alias)
        counts = images.kidney_alias.value_counts().reindex(kidneys.kidney_alias).to_numpy()
        acquisition = np.c_[rows.to_numpy(), counts]
        ref = reference.loc[[(seed, k) for k in target.kidney_alias]]
        np.testing.assert_array_equal(ref.label, y[dev])
        np.testing.assert_array_equal(ref.patient_alias, target.patient_alias)
        scores = {"fixed_reference": ref.score.to_numpy()}
        for group, features in (("acquisition_only", acquisition), ("raw_image", raw), ("residual_image", residual)):
            if group != "acquisition_only":
                _, features = fit_transformer(features, train, seed)
            head = make_pipeline(StandardScaler(), LogisticRegression(
                class_weight="balanced", solver="liblinear", C=.1, random_state=seed))
            head.fit(features[train], y[train])
            scores[group] = head.predict_proba(features[dev])[:, 1]
        for group, score in scores.items():
            draw = target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=group, label=y[dev], score=score, signature=signature.iloc[dev].to_numpy())
            records.append(draw.drop(columns="signature"))
            ranks.extend(dict(seed=seed, group=group, **row) for row in pair_summary(draw))
    pred = pd.concat(records, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), base.SEEDS, "raw_image")
    counts = pd.DataFrame(ranks)
    OUT.mkdir()
    for name, table in (("development_predictions", pred), ("development_ci", ci),
                        ("paired_delta_ci", delta), ("development_pair_counts", counts)):
        table.to_csv(OUT / f"{name}.csv", index=False)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())
    print(counts.groupby(["group", "stratum"])[["pairs", "concordant"]].sum().to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
