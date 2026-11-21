"""Patient-stratified bagging for the strongest contour-first tile branch."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_tile_replacement as replacement
from nckuh_dmsa_cluster_ci import cluster_intervals


BASE = replacement.BASE
TILE = replacement.tile
OUT = BASE.ROOT / "02_experiments/NCKUH_DMSA_patient_bagging_20261104"
MEMBERS = 25


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def stratified_patient_bootstrap(patients: pd.DataFrame, rng: np.random.Generator) -> list[str]:
    sampled = []
    for state in ("N", "L", "R", "B"):
        aliases = patients.loc[patients.DMSA_APN.eq(state), "patient_alias"].to_numpy()
        if not len(aliases):
            raise ValueError(f"Missing state {state}")
        sampled.extend(rng.choice(aliases, len(aliases), replace=True).tolist())
    rng.shuffle(sampled)
    return sampled


def repeated_indices(frame: pd.DataFrame, sampled_aliases: list[str]) -> np.ndarray:
    groups = {alias: np.flatnonzero(frame.patient_alias.to_numpy() == alias)
              for alias in set(sampled_aliases)}
    return np.concatenate([groups[alias] for alias in sampled_aliases])


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir()
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "The uniformly weak development draw reflects small-sample training variance; patient-level bagging may stabilize the two trainable tile heads",
        "reference": "Breiman, Bagging Predictors, Machine Learning 24, 123-140 (1996), https://doi.org/10.1007/BF00058655",
        "cohort": "Unchanged 210 patients, 414 kidneys and 820 contour-first images",
        "resampling": "25 bootstrap members; sample training patients with replacement within each N/L/R/B source state; duplicate all images and kidneys of a sampled patient together",
        "heads": "Unchanged tile-mean image PCA32 logistic head and ordered-patient-context PCA32 logistic head",
        "aggregation": "Arithmetic mean of member probabilities for each head, then unchanged equal two-head average and unchanged fixed fusion chain",
        "selection": "No member, epoch, feature, weight or ensemble size selected on development data",
        "evaluation": "Same three patient-random development roles; exploratory-test role remains unscored",
        "target": "Mean development AUROC and AP both above 0.8 with stability across draws before any locked evaluation",
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(OUT / "protocol.json"),
    })
    print("Registered patient-stratified bagging experiment", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(Path(__file__)) == lock["script_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed predictions")
    patients, kidneys, images = TILE.load_reference()
    features = np.load(TILE.OUT / "tile_features.npz")["mean"]
    assignments = pd.read_csv(BASE.OUT / "patient_assignments.csv")
    y_kidney = kidneys.kidney_APN.to_numpy(int)
    y_image = images.kidney_APN.to_numpy(int)
    oriented = BASE.orient_kidneys(BASE.patient_pair_features(features, images, patients), patients, kidneys)
    sources = {}
    for name, folder in {
        "cross_attention_q4": replacement.audit.MIL,
        "knn": replacement.graph.OUT,
        "aligned": replacement.directional.OUT,
    }.items():
        frame = pd.read_csv(folder / "development_predictions.csv")
        sources[name] = frame[frame.group.eq(name)].set_index(["seed", "kidney_alias"], verify_integrity=True)
    reference = pd.read_csv(replacement.OUT / "development_predictions.csv")
    reference = reference[reference.group.eq("replace_with_tile_two_heads")]
    records = [reference]
    audits = []

    for seed in BASE.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        training_patients = patients[patients.patient_alias.map(mapping).eq("train")].reset_index(drop=True)
        kd = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        iv = np.flatnonzero(images.patient_alias.map(mapping).eq("development"))
        target = kidneys.iloc[kd]
        index = [(seed, alias) for alias in target.kidney_alias]
        selected = {name: frame.loc[index] for name, frame in sources.items()}
        q, k, t = (selected[name].score.to_numpy() for name in ("cross_attention_q4", "knn", "aligned"))
        image_members, context_members = [], []
        for member in range(MEMBERS):
            rng = np.random.default_rng(seed + 17000 + member)
            aliases = stratified_patient_bootstrap(training_patients, rng)
            kt = repeated_indices(kidneys, aliases)
            it = repeated_indices(images, aliases)
            context, _ = BASE.fit_deep(oriented, kt, kd, y_kidney, seed + member + 1, 32)
            model = BASE.fit_image_head(
                features, it, y_image, BASE.image_weights(images, it, y_image), seed + member + 1)
            image = BASE.aggregate_image_logits(
                model.predict_proba(features[iv])[:, 1], images.iloc[iv], target.kidney_alias)
            context_members.append(context)
            image_members.append(image)
            state_counts = pd.Series(aliases).map(training_patients.set_index("patient_alias").DMSA_APN).value_counts()
            audits.append({
                "seed": seed, "member": member,
                "sampled_patient_slots": len(aliases),
                "distinct_training_patients": len(set(aliases)),
                **{f"sampled_{state}": int(state_counts.get(state, 0)) for state in ("N", "L", "R", "B")},
            })
        two_heads = (np.mean(image_members, axis=0) + np.mean(context_members, axis=0)) / 2
        score = replacement.fixed_chain(two_heads, q, k, t)
        records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
            seed=seed, group="patient_stratified_bagging25", label=y_kidney[kd], score=score))
        print(f"Completed bagging draw {seed}", flush=True)

    prediction = pd.concat(records, ignore_index=True)
    groups = ["replace_with_tile_two_heads", "patient_stratified_bagging25"]
    prediction = prediction[prediction.group.isin(groups)]
    ci, delta, _ = cluster_intervals(prediction, groups, BASE.SEEDS, groups[0])
    metrics = pd.DataFrame([
        dict(seed=seed, group=group, **BASE.metric(part.label, part.score))
        for (seed, group), part in prediction.groupby(["seed", "group"])
    ])
    prediction.to_csv(OUT / "development_predictions.csv", index=False)
    metrics.to_csv(OUT / "development_metrics.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    pd.DataFrame(audits).to_csv(OUT / "member_audit.csv", index=False)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    with threadpool_limits(limits=4):
        globals()[parser.parse_args().stage]()
