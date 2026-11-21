"""Training-only rank calibration of hierarchical bilateral NCKUH APN evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_uri_contralateral_local_20261105 as local_study
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_rank_calibrated_state_20261106"
REFERENCE = uri.OUT / "development_predictions.csv"
CURRENT_GROUP = "current_tile_usfm_reference"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"
SEEDS = (20261102, 20261103, 20261104)
WEIGHT_VALUES = (0.0, 0.25, 0.5, 1.0, 2.0)
GROUP = "uri_hierarchical_rank_calibrated"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def clipped_logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, float), 1e-5, 1 - 1e-5)
    return np.log(probability / (1 - probability))


def state_components(probability: np.ndarray) -> np.ndarray:
    """Recover any-APN, bilateral-given-APN, and left-given-unilateral logits."""
    p_n, p_l, p_r, p_b = probability.T
    p_any = 1 - p_n
    p_bilateral = p_b / np.maximum(p_any, 1e-8)
    p_left = p_l / np.maximum(p_l + p_r, 1e-8)
    return np.c_[clipped_logit(p_any), clipped_logit(p_bilateral), clipped_logit(p_left)]


def rank_scores(
    components: np.ndarray,
    patients: pd.DataFrame,
    kidneys: pd.DataFrame,
    weights: tuple[float, float],
) -> np.ndarray:
    """Compose disease, bilateral, and signed-side evidence into kidney rankings."""
    bilateral_weight, side_weight = weights
    raw_left = components[:, 0] + bilateral_weight * components[:, 1] + side_weight * components[:, 2]
    raw_right = components[:, 0] + bilateral_weight * components[:, 1] - side_weight * components[:, 2]
    lookup = {alias: index for index, alias in enumerate(patients.patient_alias)}
    raw = np.asarray([
        raw_left[lookup[row.patient_alias]] if row.side == "left" else raw_right[lookup[row.patient_alias]]
        for row in kidneys.itertuples(index=False)
    ])
    return 1 / (1 + np.exp(-np.clip(raw, -30, 30)))


def oof_components(
    symmetric: np.ndarray,
    antisymmetric: np.ndarray,
    states: np.ndarray,
    train: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Create outer-training-only patient component predictions."""
    outputs = np.full((len(states), 3), np.nan, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 401)
    for fold, (fit_local, validation_local) in enumerate(splitter.split(train, states[train])):
        fit = train[fit_local]
        validation = train[validation_local]
        probability, _ = hierarchy.state_probabilities(
            symmetric, antisymmetric, states, fit, seed + 1000 * (fold + 1)
        )
        outputs[validation] = state_components(probability)[validation]
    if not np.isfinite(outputs[train]).all():
        raise ValueError("Incomplete OOF state components")
    return outputs


def select_weights(
    components: np.ndarray,
    patients: pd.DataFrame,
    kidneys: pd.DataFrame,
    train: np.ndarray,
) -> tuple[tuple[float, float], pd.DataFrame]:
    train_aliases = set(patients.patient_alias.iloc[train])
    selected_kidneys = kidneys.patient_alias.isin(train_aliases).to_numpy()
    labels = kidneys.kidney_APN.to_numpy(int)[selected_kidneys]
    rows = []
    for bilateral_weight, side_weight in product(WEIGHT_VALUES, repeat=2):
        score = rank_scores(
            components, patients, kidneys, (bilateral_weight, side_weight)
        )[selected_kidneys]
        rows.append(
            {
                "bilateral_weight": bilateral_weight,
                "side_weight": side_weight,
                "oof_auroc": float(roc_auc_score(labels, score)),
                "oof_ap": float(average_precision_score(labels, score)),
                "weight_norm": bilateral_weight**2 + side_weight**2,
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["oof_auroc", "oof_ap", "weight_norm", "bilateral_weight", "side_weight"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    best = frame.iloc[0]
    return (float(best.bilateral_weight), float(best.side_weight)), frame


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve registered experiment: {OUT}")
    OUT.mkdir(parents=True)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "The kidney-level AUROC deficit is driven by ranking an unaffected contralateral kidney within unilateral APN patients; probability marginals may suppress otherwise useful laterality evidence.",
        "representation": "Official five-fold URI-CADS kidney embeddings, paired into symmetric and signed left-right patient features",
        "model": "Nested hierarchical heads for any APN, bilateral given APN, and left given unilateral APN",
        "rank_composition": "kidney logit = any-APN logit + w_bilateral * bilateral logit +/- w_side * laterality logit",
        "weight_selection": "Twenty-five nonnegative weight pairs selected by AUROC then AP on four-fold outer-training OOF predictions only",
        "outer_design": "Three fixed patient-level draws with 126 train, 42 development, and 42 untouched exploratory-test patients",
        "fixed_comparisons": [CURRENT_GROUP, URI_GROUP, GROUP, f"{GROUP}_quarter_plus_current"],
        "development_target": {"mean_auroc": ">0.8", "mean_ap": ">0.8"},
        "source_sha256": digest(Path(__file__)),
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(
        OUT / "run_lock.json",
        {"protocol_sha256": digest(OUT / "protocol.json"), "source_sha256": digest(Path(__file__))},
    )
    print("Registered hierarchical rank-calibration experiment", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    if digest(OUT / "protocol.json") != lock["protocol_sha256"]:
        raise ValueError("Protocol changed after registration")
    if digest(Path(__file__)) != lock["source_sha256"]:
        raise ValueError("Source changed after registration")


def stratum_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, group), frame in predictions.groupby(["seed", "group"], sort=False):
        positive_type = np.where(
            frame.DMSA_APN.eq("B"), "bilateral", np.where(frame.label.eq(1), "unilateral_affected", "negative")
        )
        negative_type = np.where(
            frame.DMSA_APN.eq("N"), "normal_patient", np.where(frame.label.eq(0), "unilateral_unaffected", "positive")
        )
        work = frame.assign(positive_type=positive_type, negative_type=negative_type)
        for positive, negative in product(
            ("bilateral", "unilateral_affected"),
            ("normal_patient", "unilateral_unaffected"),
        ):
            subset = work[
                ((work.label.eq(1)) & work.positive_type.eq(positive))
                | ((work.label.eq(0)) & work.negative_type.eq(negative))
            ]
            rows.append(
                {
                    "seed": seed,
                    "group": group,
                    "contrast": f"{positive}_vs_{negative}",
                    "positives": int(subset.label.sum()),
                    "negatives": int((1 - subset.label).sum()),
                    "auroc": float(roc_auc_score(subset.label, subset.score)),
                }
            )
        orders = []
        for _, pair in work[work.DMSA_APN.isin(["L", "R"])].groupby("patient_alias"):
            if len(pair) == 2 and pair.label.nunique() == 2:
                orders.append(
                    float(pair.loc[pair.label.eq(1), "score"].iloc[0] > pair.loc[pair.label.eq(0), "score"].iloc[0])
                )
        rows.append(
            {
                "seed": seed,
                "group": group,
                "contrast": "unilateral_affected_above_contralateral",
                "positives": len(orders),
                "negatives": 0,
                "auroc": float(np.mean(orders)),
            }
        )
    return pd.DataFrame(rows)


def run() -> None:
    require_lock()
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve completed rank-calibration evaluation")
    (OUT / "models").mkdir()
    patients, kidneys, images = local_study.load_reference()
    representation = uri.representations()["official_uri_embedding_folds"]
    symmetric, antisymmetric = hierarchy.structured_views(
        base.patient_pair_features(representation, images, patients)
    )
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    references = pd.read_csv(REFERENCE)
    references = references[references.group.isin([CURRENT_GROUP, URI_GROUP])].set_index(
        ["seed", "group", "kidney_alias"], verify_integrity=True
    )

    metrics, predictions, selections, split_rows = [], [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        patient_roles = patients.patient_alias.map(mapping)
        kidney_roles = kidneys.patient_alias.map(mapping)
        train = np.flatnonzero(patient_roles.eq("train"))
        development = np.flatnonzero(kidney_roles.eq("development"))
        target = kidneys.iloc[development]
        if len(train) != 126 or target.patient_alias.nunique() != 42:
            raise ValueError("Unexpected fixed patient split")
        if set(patients.patient_alias.iloc[train]).intersection(target.patient_alias):
            raise ValueError("Patient overlap")

        oof = oof_components(symmetric, antisymmetric, states, train, seed)
        weights, grid = select_weights(oof, patients, kidneys, train)
        grid.insert(0, "seed", seed)
        grid.to_csv(OUT / f"training_weight_grid_{seed}.csv", index=False)
        selections.append(
            {
                "seed": seed,
                "bilateral_weight": weights[0],
                "side_weight": weights[1],
                "oof_auroc": float(grid.iloc[0].oof_auroc),
                "oof_ap": float(grid.iloc[0].oof_ap),
            }
        )

        probability, models = hierarchy.state_probabilities(
            symmetric, antisymmetric, states, train, seed
        )
        candidate = rank_scores(state_components(probability), patients, kidneys, weights)[development]
        current = references.loc[
            [(seed, CURRENT_GROUP, alias) for alias in target.kidney_alias]
        ].score.to_numpy(float)
        fixed_uri = references.loc[
            [(seed, URI_GROUP, alias) for alias in target.kidney_alias]
        ].score.to_numpy(float)
        score_sets = {
            CURRENT_GROUP: current,
            URI_GROUP: fixed_uri,
            GROUP: candidate,
            f"{GROUP}_quarter_plus_current": 0.25 * candidate + 0.75 * current,
        }
        joblib.dump(
            {"heads": models, "weights": weights},
            OUT / "models" / f"{seed}_rank_calibrated.joblib",
            compress=3,
        )
        for group, score in score_sets.items():
            metrics.append(
                {
                    "seed": seed,
                    "role": "development",
                    "group": group,
                    "patients": target.patient_alias.nunique(),
                    "kidneys": len(target),
                    **hierarchy.metric(labels[development], score),
                }
            )
            for local, index in enumerate(development):
                predictions.append(
                    {
                        "seed": seed,
                        "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[index],
                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                        "side": kidneys.side.iloc[index],
                        "label": int(labels[index]),
                        "score": float(score[local]),
                    }
                )
        split_rows.append(
            {
                "seed": seed,
                "training_patients": len(train),
                "development_patients": target.patient_alias.nunique(),
                "exploratory_test_scored": 0,
                "patient_overlap": 0,
            }
        )
        print(f"Completed rank-calibrated draw {seed}: weights={weights}", flush=True)

    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv")
    pd.DataFrame(selections).to_csv(OUT / "training_weight_selections.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUT / "split_audit.csv", index=False)
    stratum_audit(prediction_frame).to_csv(OUT / "error_stratum_audit.csv", index=False)
    groups = prediction_frame.group.drop_duplicates().tolist()
    intervals, differences, samples = cluster_intervals(
        prediction_frame, groups, SEEDS, URI_GROUP
    )
    intervals.to_csv(OUT / "development_ci.csv", index=False)
    differences.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(
        OUT / "integrity_and_decision.json",
        {
            "status": "pass",
            "patients": len(patients),
            "kidneys": len(kidneys),
            "images": len(images),
            "patient_overlap": 0,
            "exploratory_test_rows_scored": 0,
            "weights_selected_from_outer_training_oof_only": True,
            "groups_above_both_development_targets": achieved,
            "labels_used_in_frozen_feature_encoding": [],
        },
    )
    print(summary.to_string(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        globals()[args.stage]()


if __name__ == "__main__":
    main()
