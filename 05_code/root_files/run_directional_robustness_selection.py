"""Training-only robustness and redundancy screen for CYCH I3 features.

The experiment reuses the locked perturbation matrices from the completed r2
stability study. It does not inspect exploratory-test outcomes while selecting a
candidate. A separate candidate lock is required before test evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
COMMON_SOURCE = ROOT / "CYCH_feature_research_results_20260913" / "source"
STABILITY_SOURCE = ROOT / "CYCH_stability_region_probe_20260913_r2" / "source"
for source_dir in (COMMON_SOURCE, STABILITY_SOURCE):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import renal_random_validation as R
import renal_stability_region_probe as S


PRIOR = ROOT / "CYCH_interpretable_features_20260913"
STABILITY = ROOT / "CYCH_stability_region_probe_20260913_r2"
OUT = ROOT / "CYCH_directional_robustness_selection_20260914"
VARIANTS = ("baseline", "contrast_low", "contrast_high", "erode1", "dilate1")
CANDIDATES = ("I3-ROBUST", "I3-REPRESENTATIVES", "I3-COMPONENT-POOL")
ICC_THRESHOLD = 0.90
RANK_THRESHOLD = 0.90
REDUNDANCY_THRESHOLD = 0.90
PAIR_GATE = {"delta_auroc": 0.03, "delta_ap": 0.02, "joint_wins": 6}
I3_VOTE_NONINFERIOR_MARGIN = 0.005


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name)


def metric(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
    }


def indices_for(patients: pd.DataFrame, assignments: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    current = assignments.loc[assignments.seed.eq(seed)]
    if len(current) != 98 or current.patient_alias.nunique() != 98:
        raise RuntimeError(f"Invalid assignment table for seed {seed}")
    result = {
        role: np.flatnonzero(patients.patient_alias.isin(current.loc[current.role.eq(role), "patient_alias"]))
        for role in ("train", "validation", "test")
    }
    if [len(result[name]) for name in result] != [58, 20, 20]:
        raise RuntimeError(f"Unexpected split sizes for seed {seed}")
    if len(np.unique(np.concatenate(list(result.values())))) != 98:
        raise RuntimeError(f"Overlapping split for seed {seed}")
    return result


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    value = pd.Series(left).corr(pd.Series(right), method="spearman")
    return float(value) if np.isfinite(value) else -1.0


def stability_table(
    arrays: dict[str, np.ndarray], feature_names: list[str], train: np.ndarray, seed: int
) -> pd.DataFrame:
    rows = []
    for column, name in enumerate(feature_names):
        contrast = S.icc_2_1(
            np.column_stack([arrays[variant][train, column] for variant in S.CONTRAST_VARIANTS])
        )
        contour = S.icc_2_1(
            np.column_stack([arrays[variant][train, column] for variant in S.CONTOUR_VARIANTS])
        )
        ranks = [
            rank_correlation(arrays["baseline"][train, column], arrays[variant][train, column])
            for variant in VARIANTS
            if variant != "baseline"
        ]
        minimum_icc = float(min(contrast, contour)) if np.isfinite(contrast) and np.isfinite(contour) else -1.0
        minimum_rank = float(min(ranks))
        rows.append(
            {
                "seed": seed,
                "feature_index": column,
                "feature": name,
                "contrast_icc_2_1": float(contrast),
                "contour_icc_2_1": float(contour),
                "minimum_rank_correlation": minimum_rank,
                "robustness_score": float(min(minimum_icc, minimum_rank)),
                "robust": bool(minimum_icc >= ICC_THRESHOLD and minimum_rank >= RANK_THRESHOLD),
            }
        )
    return pd.DataFrame(rows)


def correlation_components(values: np.ndarray, eligible: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    selected = np.flatnonzero(eligible)
    if len(selected) == 0:
        raise RuntimeError("No robust features")
    corr = pd.DataFrame(values[:, selected]).corr(method="spearman").to_numpy(float)
    corr = np.nan_to_num(corr, nan=0.0)
    adjacency = np.abs(corr) >= REDUNDANCY_THRESHOLD
    np.fill_diagonal(adjacency, True)
    unseen = set(range(len(selected)))
    components: list[list[int]] = []
    while unseen:
        start = min(unseen)
        stack = [start]
        local = []
        unseen.remove(start)
        while stack:
            node = stack.pop()
            local.append(node)
            neighbors = [item for item in unseen if adjacency[node, item]]
            for item in neighbors:
                unseen.remove(item)
                stack.append(item)
        components.append([int(selected[item]) for item in sorted(local)])
    return components, corr


def choose_representatives(
    components: list[list[int]], robustness: np.ndarray, feature_names: list[str]
) -> list[int]:
    representatives = []
    for component in components:
        representatives.append(
            sorted(component, key=lambda index: (-float(robustness[index]), feature_names[index]))[0]
        )
    return representatives


def pooled_components(
    all_values: np.ndarray,
    train: np.ndarray,
    components: list[list[int]],
    representatives: list[int],
) -> tuple[np.ndarray, list[dict]]:
    train_values = all_values[train]
    center = np.median(train_values, axis=0)
    scale = np.std(train_values, axis=0)
    scale[scale <= 1e-12] = 1.0
    standardized = (all_values - center) / scale
    columns = []
    audit = []
    for component, representative in zip(components, representatives):
        signs = []
        for feature_index in component:
            association = rank_correlation(
                train_values[:, representative], train_values[:, feature_index]
            )
            signs.append(1.0 if association >= 0 else -1.0)
        columns.append((standardized[:, component] * np.asarray(signs)).mean(axis=1))
        audit.append(
            {
                "representative_index": int(representative),
                "member_indices": [int(value) for value in component],
                "alignment_signs": signs,
            }
        )
    matrix = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite pooled component matrix")
    return matrix, audit


def fit_head(seed: int, values: np.ndarray, y: np.ndarray, train: np.ndarray):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(values[train], y[train])
    return model


def prior_scores(seed: int, role: str, group: str, aliases: pd.Series) -> np.ndarray:
    prior = pd.read_csv(PRIOR / "patient_predictions_by_role.csv")
    frame = prior.loc[
        prior.seed.eq(seed) & prior.role.eq(role) & prior.group.eq(group),
        ["patient_alias", "score"],
    ]
    mapping = frame.set_index("patient_alias").score
    if len(frame) != len(aliases) or not set(aliases).issubset(mapping.index):
        raise RuntimeError(f"Missing prior {group} scores for {seed}/{role}")
    return mapping.loc[aliases].to_numpy(float)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (role, group), frame in metrics.groupby(["role", "group"], sort=False):
        rows.append(
            {
                "role": role,
                "group": group,
                "draws": int(len(frame)),
                "joint_target_draws": int(((frame.auroc > 0.8) & (frame.average_precision > 0.8)).sum()),
                "auroc_mean": float(frame.auroc.mean()),
                "auroc_std": float(frame.auroc.std()),
                "average_precision_mean": float(frame.average_precision.mean()),
                "average_precision_std": float(frame.average_precision.std()),
            }
        )
    return pd.DataFrame(rows)


def prepare() -> None:
    OUT.mkdir(exist_ok=True)
    for name in ("models", "features", "source"):
        (OUT / name).mkdir(exist_ok=True)

    source_patients = pd.read_csv(STABILITY / "patients.csv", dtype={"patient_id": str})
    patients = source_patients[["patient_alias", "label", "images"]].copy()
    source_split = pd.read_csv(STABILITY / "random_patient_assignments.csv", dtype={"patient_id": str})
    assignments = source_split[["seed", "role", "patient_alias", "label", "images"]].copy()
    if len(patients) != 98 or int(patients.label.sum()) != 52:
        raise RuntimeError("Unexpected cohort")
    patients.to_csv(OUT / "patients_deidentified.csv", index=False)
    assignments.to_csv(OUT / "random_patient_assignments_deidentified.csv", index=False)
    for seed in R.SEEDS:
        indices_for(patients, assignments, seed)

    protocol = {
        "design": "Further retrospective exploratory screen on the same previously explored 98-patient cohort. Development-only selection precedes any exploratory-test access.",
        "cohort": {"patients": 98, "images": 349, "positive": 52, "negative": 46},
        "roles": {"train": 58, "development": 20, "exploratory_test": 20},
        "source": {
            "stability_protocol_sha256": R.sha256(STABILITY / "protocol.json"),
            "stability_features_sha256": {
                variant: R.sha256(STABILITY / "features" / f"whole_kidney_{variant}.npy")
                for variant in VARIANTS
            },
            "i3_feature_names_sha256": R.sha256(STABILITY / "features" / "whole_kidney_feature_names.json"),
        },
        "selection": {
            "scope": "Each random draw uses only its 58 training patients for robustness and redundancy calculations.",
            "robustness": f"Both contrast and contour ICC(2,1) >= {ICC_THRESHOLD:.2f}, and every baseline-to-perturbation Spearman rank correlation >= {RANK_THRESHOLD:.2f}.",
            "redundancy": f"Connected components of absolute training-only Spearman correlation >= {REDUNDANCY_THRESHOLD:.2f}; representative is the member with highest minimum robustness score, then lexical feature name.",
            "component_pool": "Training-median centered and training-SD scaled component members, sign-aligned to the representative by training-only Spearman correlation, then equally averaged.",
            "labels": "Outcome labels enter only the fixed logistic-regression head; robustness and redundancy screens are label-free.",
        },
        "groups": {
            "I3-ALL": "All 50 inherited I3 patient features.",
            "I3-ROBUST": "Features passing the fixed ICC and rank-stability criteria.",
            "I3-REPRESENTATIVES": "One unsupervised representative per robust correlation component.",
            "I3-COMPONENT-POOL": "One sign-aligned average per robust correlation component.",
        },
        "head": "Fixed training-only standardizer and L2 logistic regression C=0.1/liblinear/balanced. No parameter search.",
        "vote": "Existing fixed RAD+RadImageNet DenseNet pair receives one candidate as (RAD + DenseNet + candidate)/3, computed as (2*PAIR + candidate)/3.",
        "development_gate": {
            "versus_pair": PAIR_GATE,
            "versus_existing_i3_vote": f"Mean AUROC and AP may each decline by at most {I3_VOTE_NONINFERIOR_MARGIN:.3f}.",
            "tie_break": "Highest mean of development AUROC and AP, then group name.",
        },
        "test_access": "No candidate or vote is evaluated on exploratory test unless a development candidate is selected and candidate_lock.json is written.",
        "limitations": [
            "Existing perturbations are fixed contrast and one-pixel contour changes; they are not clinician artifact masks or scanner-vendor reproducibility data.",
            "All 98 patients were previously explored, so any later test result remains exploratory.",
            "This analysis cannot identify scar, swelling, cortex, sinus, or focal inflammation.",
        ],
    }
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError("Protocol changed; use a new output directory")
    write_json(protocol_path, protocol)
    source_copy = OUT / "source" / Path(__file__).name
    shutil.copy2(__file__, source_copy)
    lock_path = OUT / "protocol_locked.json"
    if not lock_path.exists():
        write_json(
            lock_path,
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": R.sha256(protocol_path),
                "source_sha256": R.sha256(source_copy),
            },
        )
    print("Protocol locked; de-identified cohort and ten fixed assignments verified", flush=True)


def require_protocol_lock() -> None:
    lock = json.loads((OUT / "protocol_locked.json").read_text(encoding="utf-8"))
    if lock["protocol_sha256"] != R.sha256(OUT / "protocol.json"):
        raise RuntimeError("Protocol lock mismatch")
    if lock["source_sha256"] != R.sha256(OUT / "source" / Path(__file__).name):
        raise RuntimeError("Locked source mismatch")


def evaluate_development() -> None:
    require_protocol_lock()
    patients = read("patients_deidentified.csv")
    assignments = read("random_patient_assignments_deidentified.csv")
    y = patients.label.to_numpy(int)
    arrays = {
        variant: np.load(STABILITY / "features" / f"whole_kidney_{variant}.npy")
        for variant in VARIANTS
    }
    feature_names = json.loads(
        (STABILITY / "features" / "whole_kidney_feature_names.json").read_text(encoding="utf-8")
    )
    if any(array.shape != (98, 50) or not np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError("Invalid inherited feature arrays")

    metrics = []
    predictions = []
    stability_rows = []
    selection_audit = []
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            index = indices_for(patients, assignments, seed)
            stability = stability_table(arrays, feature_names, index["train"], seed)
            stability_rows.append(stability)
            eligible = stability.robust.to_numpy(bool)
            if not eligible.any():
                raise RuntimeError(f"No robust feature for seed {seed}")
            components, _ = correlation_components(arrays["baseline"][index["train"]], eligible)
            representatives = choose_representatives(
                components, stability.robustness_score.to_numpy(float), feature_names
            )
            pooled, pool_audit = pooled_components(
                arrays["baseline"], index["train"], components, representatives
            )
            transformed = {
                "I3-ALL": arrays["baseline"],
                "I3-ROBUST": arrays["baseline"][:, eligible],
                "I3-REPRESENTATIVES": arrays["baseline"][:, representatives],
                "I3-COMPONENT-POOL": pooled,
            }
            selection_audit.append(
                {
                    "seed": seed,
                    "robust_feature_count": int(eligible.sum()),
                    "component_count": len(components),
                    "representative_indices": representatives,
                    "representative_features": [feature_names[value] for value in representatives],
                    "components": pool_audit,
                }
            )
            development = index["validation"]
            aliases = patients.patient_alias.iloc[development]
            pair = prior_scores(seed, "development_validation", "PAIR", aliases)
            prior_vote = prior_scores(seed, "development_validation", "VOTE", aliases)
            metrics.append({"seed": seed, "role": "development_validation", "group": "PAIR", **metric(y[development], pair)})
            metrics.append({"seed": seed, "role": "development_validation", "group": "PRIOR-I3-VOTE", **metric(y[development], prior_vote)})
            for group, values in transformed.items():
                model = fit_head(seed, values, y, index["train"])
                score = model.predict_proba(values)[:, 1]
                path = OUT / "models" / f"{group}_{seed}.joblib"
                joblib.dump(model, path, compress=3)
                np.save(OUT / "features" / f"{group}_{seed}.npy", values.astype(np.float32))
                current = score[development]
                metrics.append({"seed": seed, "role": "development_validation", "group": group, **metric(y[development], current)})
                vote = (2.0 * pair + current) / 3.0
                vote_group = f"PAIR+{group}"
                metrics.append({"seed": seed, "role": "development_validation", "group": vote_group, **metric(y[development], vote)})
                for alias, label, single_score, vote_score in zip(aliases, y[development], current, vote):
                    predictions.append({"seed": seed, "role": "development_validation", "group": group, "patient_alias": alias, "label": int(label), "score": float(single_score)})
                    predictions.append({"seed": seed, "role": "development_validation", "group": vote_group, "patient_alias": alias, "label": int(label), "score": float(vote_score)})

    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / "development_metrics_per_draw.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.concat(stability_rows, ignore_index=True).to_csv(OUT / "training_stability_by_feature_seed.csv", index=False)
    write_json(OUT / "training_selection_audit.json", selection_audit)
    summary = summarize(metric_frame)
    summary.to_csv(OUT / "development_summary.csv", index=False)

    indexed = metric_frame.set_index(["group", "seed"])[["auroc", "average_precision"]]
    pair = indexed.loc["PAIR"]
    prior_vote = indexed.loc["PRIOR-I3-VOTE"]
    decisions = []
    for candidate in CANDIDATES:
        vote_group = f"PAIR+{candidate}"
        current = indexed.loc[vote_group]
        delta_pair = current - pair
        delta_prior = current - prior_vote
        joint_wins = int(delta_pair.gt(0).all(axis=1).sum())
        eligible = bool(
            delta_pair.auroc.mean() >= PAIR_GATE["delta_auroc"] - 1e-12
            and delta_pair.average_precision.mean() >= PAIR_GATE["delta_ap"] - 1e-12
            and joint_wins >= PAIR_GATE["joint_wins"]
            and delta_prior.auroc.mean() >= -I3_VOTE_NONINFERIOR_MARGIN - 1e-12
            and delta_prior.average_precision.mean() >= -I3_VOTE_NONINFERIOR_MARGIN - 1e-12
        )
        decisions.append(
            {
                "candidate": candidate,
                "vote_group": vote_group,
                "delta_pair_auroc": float(delta_pair.auroc.mean()),
                "delta_pair_ap": float(delta_pair.average_precision.mean()),
                "joint_pair_improvement_draws": joint_wins,
                "delta_prior_i3_vote_auroc": float(delta_prior.auroc.mean()),
                "delta_prior_i3_vote_ap": float(delta_prior.average_precision.mean()),
                "vote_auroc": float(current.auroc.mean()),
                "vote_ap": float(current.average_precision.mean()),
                "eligible": eligible,
            }
        )
    eligible = sorted(
        (item for item in decisions if item["eligible"]),
        key=lambda item: (-(item["vote_auroc"] + item["vote_ap"]), item["candidate"]),
    )
    decision = {
        "selection_role": "development_validation",
        "candidates": decisions,
        "selected": eligible[0]["candidate"] if eligible else None,
        "exploratory_test_access_allowed": bool(eligible),
    }
    write_json(OUT / "development_decision.json", decision)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)


def lock_candidate() -> None:
    require_protocol_lock()
    decision_path = OUT / "development_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not decision.get("selected"):
        print("No development candidate passed; exploratory test remains sealed", flush=True)
        return
    write_json(
        OUT / "candidate_lock.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "selected": decision["selected"],
            "development_decision_sha256": R.sha256(decision_path),
            "protocol_sha256": R.sha256(OUT / "protocol.json"),
        },
    )
    print(f"Locked development-selected candidate: {decision['selected']}", flush=True)


def evaluate_test() -> None:
    require_protocol_lock()
    lock_path = OUT / "candidate_lock.json"
    if not lock_path.exists():
        raise RuntimeError("No candidate lock; exploratory test access is not allowed")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["development_decision_sha256"] != R.sha256(OUT / "development_decision.json"):
        raise RuntimeError("Development decision changed after candidate lock")
    selected = lock["selected"]
    patients = read("patients_deidentified.csv")
    assignments = read("random_patient_assignments_deidentified.csv")
    y = patients.label.to_numpy(int)
    rows = []
    predictions = []
    for seed in R.SEEDS:
        index = indices_for(patients, assignments, seed)
        current = index["test"]
        aliases = patients.patient_alias.iloc[current]
        pair = prior_scores(seed, "exploratory_test", "PAIR", aliases)
        values = np.load(OUT / "features" / f"{selected}_{seed}.npy")
        model = joblib.load(OUT / "models" / f"{selected}_{seed}.joblib")
        single = model.predict_proba(values[current])[:, 1]
        vote = (2.0 * pair + single) / 3.0
        for group, scores in ((selected, single), (f"PAIR+{selected}", vote), ("PAIR", pair)):
            rows.append({"seed": seed, "role": "exploratory_test", "group": group, **metric(y[current], scores)})
            for alias, label, score in zip(aliases, y[current], scores):
                predictions.append({"seed": seed, "role": "exploratory_test", "group": group, "patient_alias": alias, "label": int(label), "score": float(score)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "exploratory_test_metrics_per_draw.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "exploratory_test_predictions.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(OUT / "exploratory_test_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def self_test() -> None:
    base = np.arange(12, dtype=float)[:, None]
    arrays = {variant: base.copy() for variant in VARIANTS}
    table = stability_table(arrays, ["x"], np.arange(8), 1)
    assert bool(table.robust.iloc[0])
    values = np.column_stack([np.arange(20), np.arange(20) * 2, np.arange(20)[::-1], np.sin(np.arange(20))])
    components, _ = correlation_components(values, np.ones(4, dtype=bool))
    assert sorted(map(len, components)) == [1, 3]
    representatives = choose_representatives(components, np.array([.9, .8, .7, .95]), ["a", "b", "c", "d"])
    pooled, audit = pooled_components(values, np.arange(12), components, representatives)
    assert pooled.shape == (20, 2) and len(audit) == 2 and np.isfinite(pooled).all()
    print("Self-test passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("self-test", "prepare", "development", "lock-candidate", "test"),
    )
    args = parser.parse_args()
    {
        "self-test": self_test,
        "prepare": prepare,
        "development": evaluate_development,
        "lock-candidate": lock_candidate,
        "test": evaluate_test,
    }[args.stage]()


if __name__ == "__main__":
    main()
