"""Rebuild non-MIL components of the old fixed chain on revised roles."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_inductive_graph as graph
import nckuh_dmsa_multiquery_mil_20261103 as old_cohort
from nckuh_dmsa_fullcohort_clinical_features_20261103 import fit_anova
from nckuh_dmsa_triplet_rank_20261103 import fit_transformer, kidney_summary_features
from prepare_nckuh_383_contour_cohort_20261111 import OUT as REGISTRY, digest


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
FEATURES = EXP / "NCKUH_revised_383_legacy_features_20261112"
OUT = EXP / "NCKUH_revised_383_root_cause_20261112"


def fit_components(patients, kidneys, images, tile, original, mirrored, texture, train, dev, seed):
    labels = kidneys.kidney_APN.to_numpy(int)
    image_labels = images.kidney_APN.to_numpy(int)
    roles = set(patients.patient_alias.iloc[train])
    image_train = np.flatnonzero(images.patient_alias.isin(roles))
    image_dev = np.flatnonzero(images.patient_alias.isin(kidneys.patient_alias.iloc[dev]))
    target = kidneys.iloc[dev]
    weights = base.image_weights(images, image_train, image_labels)

    def image_and_context(values):
        image_model = base.fit_image_head(values, image_train, image_labels, weights, seed)
        image_probability = image_model.predict_proba(values[image_dev])[:, 1]
        image_score = base.aggregate_image_logits(
            image_probability, images.iloc[image_dev], target.kidney_alias
        )
        paired = base.patient_pair_features(values, images, patients)
        oriented = base.orient_kidneys(paired, patients, kidneys)
        direct_score, _ = base.fit_deep(oriented, train_kidneys, dev, labels, seed, 32)
        return image_score, (image_score + direct_score) / 2

    train_kidneys = np.flatnonzero(kidneys.patient_alias.isin(roles))
    tile_image, tile_two_heads = image_and_context(tile)
    reflected = (original + mirrored) / 2
    _, reflection_average = image_and_context(reflected)

    raw = kidney_summary_features(reflected, images, kidneys)
    _, transformed = fit_transformer(raw, train_kidneys, seed)
    neighbors = graph.edges(
        transformed[dev], transformed[train_kidneys],
        target.patient_alias.to_numpy(), kidneys.patient_alias.iloc[train_kidneys].to_numpy(),
    ) @ labels[train_kidneys]

    aligned_columns = [column for column in texture if column.startswith("aligned_")]
    aligned = base.orient_kidneys(
        base.patient_pair_features(texture[aligned_columns].to_numpy(float), images, patients),
        patients, kidneys,
    )
    aligned_score, _, _ = fit_anova(aligned, train_kidneys, dev, labels, seed, 64)
    return {
        "tile_mean": tile_image,
        "tile_two_heads": tile_two_heads,
        "reflection_average": reflection_average,
        "knn": neighbors,
        "aligned": aligned_score,
    }


def saved_group(path, group, seed, target):
    rows = pd.read_csv(path)
    selected = rows[rows.seed.eq(seed) & rows.group.eq(group)].set_index("kidney_alias")
    selected = selected.loc[target.kidney_alias]
    assert selected.label.to_numpy(int).tolist() == target.kidney_APN.to_numpy(int).tolist()
    return selected.score.to_numpy(float)


def reproduce_old():
    patients, kidneys, images = old_cohort.load_reference()
    seed = 20261102
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    roles = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
    train = np.flatnonzero(patients.patient_alias.map(roles).eq("train"))
    dev = np.flatnonzero(kidneys.patient_alias.map(roles).eq("development"))
    with np.load(EXP / "NCKUH_DMSA_native_tile_probe_20261103/tile_features.npz") as bundle:
        tile = bundle["mean"]
    original = np.load(base.OUT / "radimagenet_color.npy")
    mirrored = np.load(
        EXP / "NCKUH_DMSA_reflection_probe_20261103/radimagenet_color_horizontal_reflection.npy"
    )
    texture = pd.read_csv(
        EXP / "NCKUH_DMSA_fullcohort_directional_20261103/directional_features.csv"
    ).set_index("image_id").loc[images.image_id].reset_index(drop=True)
    scores = fit_components(patients, kidneys, images, tile, original, mirrored,
                            texture, train, dev, seed)
    target = kidneys.iloc[dev]
    expected = {
        "tile_mean": ("NCKUH_DMSA_native_tile_probe_20261103", "tile_mean"),
        "reflection_average": ("NCKUH_DMSA_reflection_probe_20261103", "reflection_average"),
        "knn": ("NCKUH_DMSA_inductive_graph_20261103", "knn"),
        "aligned": ("NCKUH_DMSA_fullcohort_directional_20261103", "aligned"),
    }
    errors = {}
    for name, (folder, group) in expected.items():
        reference = saved_group(EXP / folder / "development_predictions.csv", group, seed, target)
        errors[name] = float(np.max(np.abs(scores[name] - reference)))
    old_mil = saved_group(
        EXP / "NCKUH_DMSA_multiquery_mil_20261103/development_predictions.csv",
        "cross_attention_q4", seed, target,
    )
    fixed = .75 * (.75 * (.75 * scores["tile_two_heads"] + .25 * old_mil)
                   + .25 * scores["knn"]) + .25 * scores["aligned"]
    reference = saved_group(
        EXP / "NCKUH_DMSA_tile_replacement_20261103/development_predictions.csv",
        "replace_with_tile_two_heads", seed, target,
    )
    errors["fixed_chain_with_saved_mil"] = float(np.max(np.abs(fixed - reference)))
    assert max(errors.values()) < 1e-5, errors
    return errors


def main():
    errors = reproduce_old()
    lock = json.loads((REGISTRY / "cohort_lock.json").read_text(encoding="utf-8"))
    assert digest(REGISTRY / "contour_image_manifest.csv") == lock["files_sha256"]["contour_image_manifest.csv"]
    image_manifest = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidney_manifest = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    assignments = pd.read_csv(REGISTRY / "patient_assignments.csv")
    patients = assignments[
        assignments.role.isin(("train", "development"))
        & assignments.patient_alias.isin(image_manifest.patient_alias)
    ].reset_index(drop=True)
    assert patients.role.value_counts().to_dict() == {"train": 204, "development": 70}
    selected = np.flatnonzero(image_manifest.patient_alias.isin(patients.patient_alias))
    images = image_manifest.iloc[selected].reset_index(drop=True)
    kidneys = kidney_manifest[kidney_manifest.patient_alias.isin(patients.patient_alias)].reset_index(drop=True)
    role = patients.set_index("patient_alias").role
    train = np.flatnonzero(patients.role.eq("train"))
    dev = np.flatnonzero(kidneys.patient_alias.map(role).eq("development"))
    texture = pd.read_csv(FEATURES / "directional_features.csv").set_index("image_id")
    texture = texture.loc[images.image_id].reset_index(drop=True)
    scores = fit_components(
        patients, kidneys, images,
        np.load(FEATURES / "tile_mean.npy")[selected],
        np.load(FEATURES / "rad_color.npy")[selected],
        np.load(FEATURES / "rad_color_horizontal_reflection.npy")[selected],
        texture, train, dev, 20261111,
    )
    target = kidneys.iloc[dev].reset_index(drop=True)
    records = []
    metrics = {}
    for group, score in scores.items():
        result = base.metric(target.kidney_APN.to_numpy(int), score)
        metrics[group] = result
        records.extend(target[["patient_alias", "kidney_alias", "side", "kidney_APN"]].assign(
            group=group, score=score
        ).to_dict(orient="records"))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUT / "legacy_fixed_component_predictions.csv", index=False)
    report = {"old_draw_reproduction_max_abs": errors,
              "new_development_metrics": metrics,
              "train_patients": 204, "development_patients": 70,
              "test_scored": False,
              "missing_component": "cross_attention_q4; final fixed chain not yet assembled"}
    (OUT / "legacy_fixed_components.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
