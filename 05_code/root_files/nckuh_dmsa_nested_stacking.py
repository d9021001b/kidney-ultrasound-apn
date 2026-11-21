"""Nested patient-level stacking of complementary contour-first branches."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_directional as directional
import nckuh_dmsa_inductive_graph as graph
import nckuh_dmsa_multiquery_mil_20261103 as multi
import nckuh_dmsa_tile_replacement as replacement
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_triplet_rank_20261103 import fit_transformer, kidney_summary_features


BASE = replacement.BASE
TILE = replacement.tile
OUT = BASE.ROOT / "02_experiments/NCKUH_DMSA_nested_stacking_20261104"
BRANCHES = ("tile_two_heads", "cross_attention_q4", "knn", "aligned_texture")


def digest(path: Path) -> str:
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir()
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Fixed 75/25 fusion weights may underuse complementary branches; low-dimensional nested stacking can learn fusion without exposing outer development outcomes",
        "cohort": "Unchanged 210 patients, 414 kidneys and 820 contour-first images",
        "outer_roles": "Three fixed 126-patient training and 42-patient development roles; exploratory-test roles remain unscored",
        "inner_roles": "Five shuffled stratified folds of the 126 outer-training patients using N/L/R/B state; all kidneys and images from a patient remain together",
        "branches": list(BRANCHES),
        "branch_definitions": "Tile-mean image/context average; four-query local RadImageNet MIL; train-only eight-patient KNN; contour-axis aligned Coiflet/GLRLM head",
        "meta_model": "Standardized four-score balanced logistic regression C=0.1 trained only on inner out-of-fold kidney scores",
        "arms": ["nested_stack_lr", "nested_stack_fixed_equal"],
        "shrinkage_control": "Second arm is an a-priori equal average of nested stack and unchanged fixed-chain score",
        "selection": "No development-selected fold, branch, coefficient, C, weight or epoch; q4 epoch is the existing outer-training inner selection",
        "target": "Mean development AUROC and AP both above 0.8, followed by locked evaluation",
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(OUT / "protocol.json"),
    })
    print("Registered nested stacking experiment", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(Path(__file__)) == lock["script_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]


def image_and_context_score(features, oriented, patients, kidneys, images, labels_kidney,
                            labels_image, train, evaluation, seed):
    train_alias = set(kidneys.patient_alias.iloc[train])
    evaluation_alias = set(kidneys.patient_alias.iloc[evaluation])
    train_image = np.flatnonzero(images.patient_alias.isin(train_alias))
    evaluation_image = np.flatnonzero(images.patient_alias.isin(evaluation_alias))
    target = kidneys.iloc[evaluation]
    model = BASE.fit_image_head(
        features, train_image, labels_image,
        BASE.image_weights(images, train_image, labels_image), seed)
    image_score = BASE.aggregate_image_logits(
        model.predict_proba(features[evaluation_image])[:, 1],
        images.iloc[evaluation_image], target.kidney_alias)
    context_score, _ = BASE.fit_deep(oriented, train, evaluation, labels_kidney, seed, 32)
    return (image_score + context_score) / 2


def branch_scores(train, evaluation, train_aliases, seed, q4_epoch, data):
    patients, kidneys, images = data["reference"]
    y_kidney, y_image = data["y_kidney"], data["y_image"]
    tile_score = image_and_context_score(
        data["tile"], data["tile_oriented"], patients, kidneys, images,
        y_kidney, y_image, train, evaluation, seed)

    _, transformed = fit_transformer(data["graph_raw"], train, seed)
    knn = graph.edges(
        transformed[evaluation], transformed[train],
        kidneys.patient_alias.iloc[evaluation].to_numpy(),
        kidneys.patient_alias.iloc[train].to_numpy()) @ y_kidney[train]
    texture, _, _ = directional.fit_anova(
        data["aligned"], train, evaluation, y_kidney, seed, 64)

    bags, mask, views, _ = multi.prepare_bags(
        data["patch_values"], data["patch_patients"], data["patch_views"],
        data["patch_groups"], set(train_aliases), seed)
    _, history = multi.train_model(
        "cross_attention_q4", seed + 10000, bags, mask, views,
        data["torch_labels"], train, (q4_epoch,), evaluation)
    q4 = history[q4_epoch]
    del bags, mask, views
    torch.cuda.empty_cache()
    return np.column_stack([tile_score, q4, knn, texture])


def run() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed predictions")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    patients, kidneys, images = TILE.load_reference()
    y_kidney = kidneys.kidney_APN.to_numpy(int)
    y_image = images.kidney_APN.to_numpy(int)
    tile = np.load(TILE.OUT / "tile_features.npz")["mean"]
    tile_oriented = BASE.orient_kidneys(BASE.patient_pair_features(tile, images, patients), patients, kidneys)
    reflection = (np.load(BASE.OUT / "radimagenet_color.npy") +
                  np.load(replacement.reflection.OUT / "radimagenet_color_horizontal_reflection.npy")) / 2
    graph_raw = kidney_summary_features(reflection, images, kidneys)
    aligned_frame = pd.read_csv(directional.OUT / "directional_features.csv").set_index("image_id").loc[images.image_id]
    aligned_image = aligned_frame.filter(regex="^aligned_").to_numpy(float)
    aligned = BASE.orient_kidneys(BASE.patient_pair_features(aligned_image, images, patients), patients, kidneys)
    patch_values, patch_patients, patch_views, patch_groups = multi.patch_data(images, kidneys)
    selections = pd.read_csv(multi.OUT / "training_inner_epoch_selections.csv")
    q4_epochs = selections[selections.arm.eq("cross_attention_q4")].set_index("seed").selected_epoch.astype(int)
    assignments = pd.read_csv(BASE.OUT / "patient_assignments.csv")
    fixed = pd.read_csv(replacement.OUT / "development_predictions.csv")
    fixed = fixed[fixed.group.eq("replace_with_tile_two_heads")].set_index(["seed", "kidney_alias"])
    data = {
        "reference": (patients, kidneys, images), "y_kidney": y_kidney, "y_image": y_image,
        "tile": tile, "tile_oriented": tile_oriented, "graph_raw": graph_raw, "aligned": aligned,
        "patch_values": patch_values, "patch_patients": patch_patients,
        "patch_views": patch_views, "patch_groups": patch_groups,
        "torch_labels": torch.from_numpy(y_kidney.astype(np.float32)).cuda(),
    }
    records, audits = [], []
    for seed in BASE.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        outer_patients = patients[patients.patient_alias.map(mapping).eq("train")].reset_index(drop=True)
        outer_train = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("train"))
        development = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        oof = np.full((len(kidneys), len(BRANCHES)), np.nan)
        folds = StratifiedKFold(5, shuffle=True, random_state=seed + 9000)
        for fold, (fit_local, validation_local) in enumerate(
                folds.split(outer_patients.patient_alias, outer_patients.DMSA_APN)):
            fit_alias = outer_patients.patient_alias.iloc[fit_local]
            validation_alias = outer_patients.patient_alias.iloc[validation_local]
            fit = np.flatnonzero(kidneys.patient_alias.isin(fit_alias))
            validation = np.flatnonzero(kidneys.patient_alias.isin(validation_alias))
            oof[validation] = branch_scores(
                fit, validation, fit_alias, seed + 2000 + fold,
                int(q4_epochs.loc[seed]), data)
            audits.append({
                "seed": seed, "fold": fold, "fit_patients": len(fit_alias),
                "validation_patients": len(validation_alias),
                "fit_kidneys": len(fit), "validation_kidneys": len(validation),
            })
            print(f"Completed nested scores seed {seed} fold {fold + 1}/5", flush=True)
        if not np.isfinite(oof[outer_train]).all():
            raise ValueError("Incomplete outer-training OOF predictions")
        meta = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=.1, class_weight="balanced", solver="liblinear",
                               max_iter=3000, random_state=seed))
        meta.fit(oof[outer_train], y_kidney[outer_train])
        outer = branch_scores(
            outer_train, development, outer_patients.patient_alias,
            seed + 100, int(q4_epochs.loc[seed]), data)
        stack = meta.predict_proba(outer)[:, 1]
        target = kidneys.iloc[development]
        fixed_score = fixed.loc[[(seed, alias) for alias in target.kidney_alias]].score.to_numpy()
        scores = {
            "nested_stack_lr": stack,
            "nested_stack_fixed_equal": (stack + fixed_score) / 2,
            "fixed_chain_reference": fixed_score,
        }
        coefficients = meta.named_steps["logisticregression"].coef_[0]
        for name, coefficient in zip(BRANCHES, coefficients):
            audits.append({"seed": seed, "fold": "meta", "branch": name,
                           "coefficient": float(coefficient)})
        for group, score in scores.items():
            records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                seed=seed, group=group, label=y_kidney[development], score=score))
        print(f"Completed nested stack draw {seed}", flush=True)
    prediction = pd.concat(records, ignore_index=True)
    groups = ["fixed_chain_reference", "nested_stack_lr", "nested_stack_fixed_equal"]
    ci, delta, _ = cluster_intervals(prediction, groups, BASE.SEEDS, groups[0])
    metrics = pd.DataFrame([
        dict(seed=seed, group=group, **BASE.metric(part.label, part.score))
        for (seed, group), part in prediction.groupby(["seed", "group"])
    ])
    prediction.to_csv(OUT / "development_predictions.csv", index=False)
    metrics.to_csv(OUT / "development_metrics.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    pd.DataFrame(audits).to_csv(OUT / "nested_audit.csv", index=False)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    with threadpool_limits(limits=4):
        globals()[parser.parse_args().stage]()
