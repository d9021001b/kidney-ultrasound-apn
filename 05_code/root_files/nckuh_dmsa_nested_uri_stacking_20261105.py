"""Nested patient-level stacking with official URI-CADS renal features."""

from __future__ import annotations

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
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_inductive_graph as graph
import nckuh_dmsa_multiquery_mil_20261103 as multi
import nckuh_dmsa_nested_stacking as nested
import nckuh_dmsa_tile_replacement as replacement
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_triplet_rank_20261103 import kidney_summary_features


BASE = replacement.BASE
TILE = replacement.tile
OUT = BASE.ROOT / "02_experiments/NCKUH_DMSA_nested_uri_stacking_20261105"
BRANCHES = (*nested.BRANCHES, "official_uri_state")
CURRENT_GROUP = "hierarchical_tile_usfm_quarter_plus_fixed"
URI_FIXED_GROUP = "official_uri_embedding_folds_quarter_plus_current"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    inputs = [
        Path(__file__),
        uri.OUT / "official_fold_embeddings.npy",
        uri.CURRENT,
        uri.OUT / "development_predictions.csv",
        BASE.OUT / "patient_assignments.csv",
    ]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "The weak but complementary official URI-CADS renal pathology representation may add information to the existing leakage-resistant nested stack",
        "cohort": "Unchanged 210 patients, 414 kidneys, and 820 contour-first images",
        "outer_roles": "Three fixed 126-patient training and 42-patient development roles; exploratory-test roles remain unscored",
        "inner_roles": "Five N/L/R/B-stratified patient folds inside each outer training role",
        "branches": list(BRANCHES),
        "uri_branch": "Five official URI-CADS fold embeddings concatenated at image level, converted to symmetric/antisymmetric patient-pair features and three training-only hierarchical state heads",
        "meta_model": "Standardized five-score balanced logistic regression C=0.1 fitted only to inner out-of-fold kidney scores",
        "arms": [
            "nested_uri_stack_lr",
            "nested_uri_stack_quarter_plus_current",
            "nested_uri_stack_equal_plus_current",
            "current_tile_usfm_reference",
            "official_uri_fixed_quarter_reference",
        ],
        "shrinkage": "Two weights fixed before execution: 25% or 50% nested stack with the unchanged current reference",
        "selection": "No outer-development branch fitting, coefficient fitting, C selection, epoch selection, or fusion-weight selection",
        "target": "Mean development AUROC and AP both above 0.8, followed by locked evaluation",
        "inputs": {str(path.relative_to(BASE.ROOT)): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(
        OUT / "run_lock.json",
        {
            "script_sha256": digest(Path(__file__)),
            "protocol_sha256": digest(OUT / "protocol.json"),
        },
    )
    print("Registered nested URI-CADS stacking protocol", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(Path(__file__)) == lock["script_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]


def uri_score(
    train_aliases,
    evaluation: np.ndarray,
    seed: int,
    patients: pd.DataFrame,
    kidneys: pd.DataFrame,
    states: np.ndarray,
    symmetric: np.ndarray,
    antisymmetric: np.ndarray,
) -> tuple[np.ndarray, dict]:
    train_alias_set = set(train_aliases)
    train_patients = np.flatnonzero(patients.patient_alias.isin(train_alias_set))
    probability, models = hierarchy.state_probabilities(
        symmetric, antisymmetric, states, train_patients, seed
    )
    all_kidney_scores = hierarchy.kidney_scores(probability, patients, kidneys)
    return all_kidney_scores[evaluation], models


def branch_scores(
    train: np.ndarray,
    evaluation: np.ndarray,
    train_aliases,
    seed: int,
    q4_epoch: int,
    data: dict,
) -> tuple[np.ndarray, dict]:
    original = nested.branch_scores(
        train, evaluation, train_aliases, seed, q4_epoch, data
    )
    uri_values, uri_models = uri_score(
        train_aliases,
        evaluation,
        seed + 30000,
        data["reference"][0],
        data["reference"][1],
        data["states"],
        data["uri_symmetric"],
        data["uri_antisymmetric"],
    )
    return np.column_stack([original, uri_values]), uri_models


def run() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed nested URI predictions")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    patients, kidneys, images = TILE.load_reference()
    y_kidney = kidneys.kidney_APN.to_numpy(int)
    y_image = images.kidney_APN.to_numpy(int)
    tile = np.load(TILE.OUT / "tile_features.npz")["mean"]
    tile_oriented = BASE.orient_kidneys(
        BASE.patient_pair_features(tile, images, patients), patients, kidneys
    )
    reflection = (
        np.load(BASE.OUT / "radimagenet_color.npy")
        + np.load(replacement.reflection.OUT / "radimagenet_color_horizontal_reflection.npy")
    ) / 2
    graph_raw = kidney_summary_features(reflection, images, kidneys)
    aligned_frame = (
        pd.read_csv(directional.OUT / "directional_features.csv")
        .set_index("image_id")
        .loc[images.image_id]
    )
    aligned_image = aligned_frame.filter(regex="^aligned_").to_numpy(float)
    aligned = BASE.orient_kidneys(
        BASE.patient_pair_features(aligned_image, images, patients), patients, kidneys
    )
    patch_values, patch_patients, patch_views, patch_groups = multi.patch_data(images, kidneys)
    selections = pd.read_csv(multi.OUT / "training_inner_epoch_selections.csv")
    q4_epochs = (
        selections[selections.arm.eq("cross_attention_q4")]
        .set_index("seed")
        .selected_epoch.astype(int)
    )

    uri_embedding = np.load(uri.OUT / "official_fold_embeddings.npy").reshape(len(images), -1)
    uri_symmetric, uri_antisymmetric = hierarchy.structured_views(
        BASE.patient_pair_features(uri_embedding, images, patients)
    )
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    assignments = pd.read_csv(BASE.OUT / "patient_assignments.csv")

    current = pd.read_csv(uri.CURRENT)
    current = current[current.group.eq(CURRENT_GROUP)].set_index(
        ["seed", "kidney_alias"], verify_integrity=True
    )
    uri_fixed = pd.read_csv(uri.OUT / "development_predictions.csv")
    uri_fixed = uri_fixed[uri_fixed.group.eq(URI_FIXED_GROUP)].set_index(
        ["seed", "kidney_alias"], verify_integrity=True
    )
    data = {
        "reference": (patients, kidneys, images),
        "y_kidney": y_kidney,
        "y_image": y_image,
        "tile": tile,
        "tile_oriented": tile_oriented,
        "graph_raw": graph_raw,
        "aligned": aligned,
        "patch_values": patch_values,
        "patch_patients": patch_patients,
        "patch_views": patch_views,
        "patch_groups": patch_groups,
        "torch_labels": torch.from_numpy(y_kidney.astype(np.float32)).cuda(),
        "states": states,
        "uri_symmetric": uri_symmetric,
        "uri_antisymmetric": uri_antisymmetric,
    }

    records, audits = [], []
    model_dir = OUT / "uri_inner_models"
    model_dir.mkdir(exist_ok=True)
    for seed in BASE.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        outer_patients = patients[patients.patient_alias.map(mapping).eq("train")].reset_index(drop=True)
        outer_train = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("train"))
        development = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        oof = np.full((len(kidneys), len(BRANCHES)), np.nan)
        folds = StratifiedKFold(5, shuffle=True, random_state=seed + 9000)
        for fold, (fit_local, validation_local) in enumerate(
            folds.split(outer_patients.patient_alias, outer_patients.DMSA_APN)
        ):
            fit_alias = outer_patients.patient_alias.iloc[fit_local]
            validation_alias = outer_patients.patient_alias.iloc[validation_local]
            fit = np.flatnonzero(kidneys.patient_alias.isin(fit_alias))
            validation = np.flatnonzero(kidneys.patient_alias.isin(validation_alias))
            oof[validation], uri_models = branch_scores(
                fit,
                validation,
                fit_alias,
                seed + 2000 + fold,
                int(q4_epochs.loc[seed]),
                data,
            )
            for head, model in uri_models.items():
                audits.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "branch": f"official_uri_state_{head}",
                        "fit_patients": len(fit_alias),
                        "validation_patients": len(validation_alias),
                        "inner_cv_auroc": float(model.best_score_),
                        **model.best_params_,
                    }
                )
            print(f"Nested URI scores seed {seed} fold {fold + 1}/5", flush=True)
        if not np.isfinite(oof[outer_train]).all():
            raise ValueError("Incomplete outer-training OOF predictions")

        meta = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                solver="liblinear",
                max_iter=3000,
                random_state=seed,
            ),
        )
        meta.fit(oof[outer_train], y_kidney[outer_train])
        outer, uri_models = branch_scores(
            outer_train,
            development,
            outer_patients.patient_alias,
            seed + 100,
            int(q4_epochs.loc[seed]),
            data,
        )
        stack = meta.predict_proba(outer)[:, 1]
        target = kidneys.iloc[development]
        keys = [(seed, alias) for alias in target.kidney_alias]
        current_score = current.loc[keys].score.to_numpy()
        uri_fixed_score = uri_fixed.loc[keys].score.to_numpy()
        np.testing.assert_array_equal(current.loc[keys].label, y_kidney[development])
        np.testing.assert_array_equal(uri_fixed.loc[keys].label, y_kidney[development])
        scores = {
            "nested_uri_stack_lr": stack,
            "nested_uri_stack_quarter_plus_current": 0.25 * stack + 0.75 * current_score,
            "nested_uri_stack_equal_plus_current": 0.50 * stack + 0.50 * current_score,
            "current_tile_usfm_reference": current_score,
            "official_uri_fixed_quarter_reference": uri_fixed_score,
        }
        coefficients = meta.named_steps["logisticregression"].coef_[0]
        for name, coefficient in zip(BRANCHES, coefficients):
            audits.append(
                {
                    "seed": seed,
                    "fold": "meta",
                    "branch": name,
                    "coefficient": float(coefficient),
                }
            )
        for group, score in scores.items():
            records.append(
                target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                    seed=seed,
                    group=group,
                    label=y_kidney[development],
                    score=score,
                )
            )
        print(f"Completed nested URI stack draw {seed}", flush=True)

    prediction = pd.concat(records, ignore_index=True)
    groups = list(prediction.group.drop_duplicates())
    reference = "current_tile_usfm_reference"
    ci, delta, samples = cluster_intervals(prediction, groups, BASE.SEEDS, reference)
    metrics = pd.DataFrame(
        [
            {"seed": seed, "group": group, **BASE.metric(part.label, part.score)}
            for (seed, group), part in prediction.groupby(["seed", "group"])
        ]
    )
    summary = (
        metrics.groupby("group")[["auroc", "ap", "brier"]]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    prediction.to_csv(OUT / "development_predictions.csv", index=False)
    metrics.to_csv(OUT / "development_metrics.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv")
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    pd.DataFrame(audits).to_csv(OUT / "nested_audit.csv", index=False)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(
        OUT / "decision.json",
        {
            "groups_above_both_development_targets": achieved,
            "exploratory_test_rows_scored": 0,
            "promotion": bool(achieved),
            "reference": reference,
        },
    )
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    with threadpool_limits(limits=4):
        globals()[parser.parse_args().stage]()
