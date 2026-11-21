from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
BASE_RUN = ROOT / "CYCH_dinov3_native_probe_20260913"
I3_RUN = ROOT / "CYCH_stability_region_probe_20260913_r2"
SPATIAL_RUN = ROOT / "CYCH_literature_spatial_focality_development_20260914"
SPLIT_RUN = ROOT / "CYCH_directional_robustness_selection_20260914"
OUT = ROOT / "CYCH_oof_convex_stacking_development_20260914"

PATIENTS_PATH = BASE_RUN / "patients.csv"
ASSIGNMENTS_PATH = SPLIT_RUN / "random_patient_assignments_deidentified.csv"
RAD_PATH = BASE_RUN / "features" / "radiomics_percentile.npy"
DENSE_PATH = BASE_RUN / "features" / "radimagenet_percentile.npy"
I3_PATH = I3_RUN / "features" / "whole_kidney_baseline.npy"
SPATIAL_PATH = SPATIAL_RUN / "patient_features_deidentified.csv"
PAIR_PREDICTIONS_PATH = BASE_RUN / "patient_predictions_by_role.csv"
I3_PREDICTIONS_PATH = SPLIT_RUN / "development_predictions.csv"
SCRIPT_COPY = OUT / "source" / Path(__file__).name
PROTOCOL_PATH = OUT / "protocol.json"
REFERENCES_PATH = OUT / "references.json"
LOCK_PATH = OUT / "protocol_lock.json"

SEEDS = list(range(20260913, 20260923))
BASE_GROUPS = ("RAD", "DENSE", "I3", "SPATIAL-NATIVE")
META_GROUPS = {
    "OOF-CONVEX-RDI": ("RAD", "DENSE", "I3"),
    "OOF-CONVEX-RDIS": ("RAD", "DENSE", "I3", "SPATIAL-NATIVE"),
}
META_SHRINKAGE = 0.05


REFERENCES = [
    {
        "method": "Stacked generalization",
        "citation": "Wolpert, Neural Networks, 1992",
        "doi": "10.1016/S0893-6080(05)80023-1",
        "url": "https://doi.org/10.1016/S0893-6080(05)80023-1",
        "use_here": (
            "Supports fitting a second-level combiner on predictions generated for "
            "patients excluded from each first-level training fold."
        ),
    },
    {
        "method": "Super Learner",
        "citation": "van der Laan, Polley, and Hubbard, 2007",
        "doi": "10.2202/1544-6115.1309",
        "url": "https://pubmed.ncbi.nlm.nih.gov/17910531/",
        "use_here": (
            "Supports cross-validated risk-based combination of a prespecified learner library."
        ),
    },
    {
        "method": "Local texture and deep feature fusion in renal ultrasound",
        "citation": "Deep learning radiomics based on ultrasound images for CKD, 2024",
        "doi": "10.1111/nep.14376",
        "url": "https://onlinelibrary.wiley.com/doi/10.1111/nep.14376",
        "use_here": (
            "Motivates testing whether local kidney-mask texture adds information to a "
            "medical-image pretrained representation. The disease endpoint differs."
        ),
    },
]


PROTOCOL = {
    "title": "Development-only out-of-fold convex stacking experiment",
    "status": "locked before model fitting",
    "cohort": {
        "patients": 98,
        "images": 349,
        "unit": "patient",
        "fixed_seeds": SEEDS,
        "per_seed": {"train": 58, "development_validation": 20, "unused_test": 20},
    },
    "base_learners": {
        "RAD": "fixed kidney-mask radiomics head: median imputation, scaling, top-32 ANOVA, balanced L2 LR C=0.1",
        "DENSE": "fixed frozen RadImageNet DenseNet head: median imputation, scaling, top-32 ANOVA, balanced L2 LR C=0.1",
        "I3": "fixed original directional feature head: scaling and balanced L2 LR C=0.1",
        "SPATIAL-NATIVE": (
            "fixed native spatial feature head chosen from the prior training-only tuning mode: "
            "median imputation, scaling, top-20 ANOVA, balanced RBF SVC C=1"
        ),
    },
    "stacking": {
        "base_oof": (
            "Four-fold stratified out-of-fold predictions are generated only among the 58 "
            "outer-training patients. Each patient is excluded from the fitted base learner "
            "that generates that patient's meta-feature."
        ),
        "meta_model": (
            "Nonnegative weights constrained to sum to one, selected by binary log loss on "
            "outer-training OOF predictions with fixed L2 shrinkage 0.05 toward equal weights."
        ),
        "registered_candidates": META_GROUPS,
        "development": (
            "After meta weights are frozen, each base learner is refit on all 58 outer-training "
            "patients and scored once on the 20 development patients."
        ),
    },
    "hard_constraints": [
        "Kidney segmentation is inherited and mandatory for every image-derived feature.",
        "No development or test patient label is used to fit base or meta learners.",
        "Rows assigned role=test and their prior predictions are never loaded.",
        "The two stacking libraries are fixed before development evaluation.",
        "A candidate must pass the fixed development gate before a separate test lock is allowed.",
    ],
    "candidate_gate_vs_fixed_pair": {
        "mean_auroc_gain_at_least": 0.03,
        "mean_average_precision_gain_at_least": 0.02,
        "seeds_with_both_metrics_improved_at_least": 6,
    },
    "interpretation": (
        "This is exploratory repeated development evaluation on 98 previously explored "
        "patients, not k-fold performance reporting and not independent confirmation."
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def lock_protocol() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SCRIPT_COPY.parent.mkdir(parents=True, exist_ok=True)
    write_json(PROTOCOL_PATH, PROTOCOL)
    write_json(REFERENCES_PATH, REFERENCES)
    shutil.copy2(Path(__file__), SCRIPT_COPY)
    inputs = {
        "protocol": PROTOCOL_PATH,
        "references": REFERENCES_PATH,
        "script": Path(__file__),
        "patients": PATIENTS_PATH,
        "assignments": ASSIGNMENTS_PATH,
        "radiomics": RAD_PATH,
        "dense": DENSE_PATH,
        "i3": I3_PATH,
        "spatial": SPATIAL_PATH,
        "pair_development_predictions": PAIR_PREDICTIONS_PATH,
        "i3_development_predictions": I3_PREDICTIONS_PATH,
    }
    write_json(LOCK_PATH, {name: sha256(path) for name, path in inputs.items()})
    print(f"Locked protocol: {LOCK_PATH}", flush=True)


def verify_lock() -> None:
    if not LOCK_PATH.exists():
        raise RuntimeError("Run --lock-only before model fitting")
    expected = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    paths = {
        "protocol": PROTOCOL_PATH,
        "references": REFERENCES_PATH,
        "script": Path(__file__),
        "patients": PATIENTS_PATH,
        "assignments": ASSIGNMENTS_PATH,
        "radiomics": RAD_PATH,
        "dense": DENSE_PATH,
        "i3": I3_PATH,
        "spatial": SPATIAL_PATH,
        "pair_development_predictions": PAIR_PREDICTIONS_PATH,
        "i3_development_predictions": I3_PREDICTIONS_PATH,
    }
    changed = [name for name, path in paths.items() if sha256(path) != expected[name]]
    if changed:
        raise RuntimeError(f"Locked input changed: {changed}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    patients = pd.read_csv(PATIENTS_PATH, dtype={"patient_id": str, "patient_alias": str})
    assignments = pd.read_csv(
        ASSIGNMENTS_PATH,
        dtype={"patient_alias": str},
        usecols=["seed", "role", "patient_alias", "label"],
    )
    spatial = pd.read_csv(SPATIAL_PATH, dtype={"patient_alias": str})
    if patients.patient_alias.tolist() != spatial.patient_alias.tolist():
        raise RuntimeError("Spatial feature patient order mismatch")
    spatial_columns = [
        column
        for column in spatial
        if column.startswith("view_present__")
        or "__native_" in column
        or "__shape_" in column
    ]
    arrays = {
        "RAD": np.load(RAD_PATH).astype(np.float64),
        "DENSE": np.load(DENSE_PATH).astype(np.float64),
        "I3": np.load(I3_PATH).astype(np.float64),
        "SPATIAL-NATIVE": spatial[spatial_columns].to_numpy(dtype=np.float64),
    }
    if any(values.shape[0] != 98 for values in arrays.values()):
        raise RuntimeError({name: values.shape for name, values in arrays.items()})
    if not all(np.isfinite(arrays[name]).all() for name in ("RAD", "DENSE", "I3")):
        raise RuntimeError("Unexpected non-finite inherited feature matrix")
    return patients, assignments, arrays


def build_base(group: str, seed: int):
    if group in {"RAD", "DENSE"}:
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SelectKBest(f_classif, k=32),
            LogisticRegression(
                C=0.1,
                solver="liblinear",
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            ),
        )
    if group == "I3":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                solver="liblinear",
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            ),
        )
    if group == "SPATIAL-NATIVE":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SelectKBest(f_classif, k=20),
            SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=seed,
            ),
        )
    raise ValueError(group)


def fit_predict(model, x: np.ndarray, y: np.ndarray, train: np.ndarray, score: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x[train], y[train])
    return model.predict_proba(x[score])[:, list(model.classes_).index(1)]


def oof_and_development_scores(
    group: str,
    x: np.ndarray,
    y: np.ndarray,
    outer_train: np.ndarray,
    development: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    oof = np.full(len(outer_train), np.nan, dtype=float)
    train_y = y[outer_train]
    for fold, (inner_train_local, inner_valid_local) in enumerate(splitter.split(outer_train, train_y)):
        inner_train = outer_train[inner_train_local]
        inner_valid = outer_train[inner_valid_local]
        oof[inner_valid_local] = fit_predict(
            build_base(group, seed + fold + 1), x, y, inner_train, inner_valid
        )
    if not np.isfinite(oof).all():
        raise RuntimeError(f"Incomplete OOF prediction for {group}/{seed}")
    development_score = fit_predict(build_base(group, seed), x, y, outer_train, development)
    return oof, development_score


def fit_convex_weights(predictions: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    member_count = predictions.shape[1]
    equal = np.full(member_count, 1.0 / member_count)

    def objective(weights: np.ndarray) -> float:
        score = np.clip(predictions @ weights, 1e-6, 1 - 1e-6)
        penalty = META_SHRINKAGE * float(np.sum((weights - equal) ** 2))
        return float(log_loss(y, score, labels=[0, 1]) + penalty)

    result = minimize(
        objective,
        equal,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * member_count,
        constraints=[{"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Convex optimization failed: {result.message}")
    weights = np.maximum(result.x, 0.0)
    weights /= weights.sum()
    return weights, objective(weights)


def load_reference(path: Path, group: str, score_name: str) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["role"] == "development_validation" and row["group"] == group:
                rows.append(
                    {
                        "seed": int(row["seed"]),
                        "patient_alias": row["patient_alias"],
                        "label": int(row["label"]),
                        score_name: float(row["score"]),
                    }
                )
    result = pd.DataFrame(rows)
    if result.shape[0] != 200:
        raise RuntimeError(f"Unexpected reference predictions for {group}: {result.shape}")
    return result


def metric(seed: int, group: str, y: np.ndarray, score: np.ndarray) -> dict[str, object]:
    return {
        "seed": seed,
        "role": "development_validation",
        "group": group,
        "n": len(y),
        "auroc": roc_auc_score(y, score),
        "average_precision": average_precision_score(y, score),
    }


def run_development() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients, assignments, arrays = load_inputs()
    y = patients.label.to_numpy(dtype=int)
    pair_reference = load_reference(PAIR_PREDICTIONS_PATH, "PAIR", "pair_score")
    i3_reference = load_reference(I3_PREDICTIONS_PATH, "I3-ALL", "i3_score")
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        current = assignments.loc[assignments.seed.eq(seed)]
        outer_train = np.flatnonzero(
            patients.patient_alias.isin(current.loc[current.role.eq("train"), "patient_alias"])
        )
        development = np.flatnonzero(
            patients.patient_alias.isin(current.loc[current.role.eq("validation"), "patient_alias"])
        )
        test = np.flatnonzero(
            patients.patient_alias.isin(current.loc[current.role.eq("test"), "patient_alias"])
        )
        if [len(outer_train), len(development), len(test)] != [58, 20, 20]:
            raise RuntimeError(f"Invalid split for seed {seed}")
        base_oof: dict[str, np.ndarray] = {}
        base_development: dict[str, np.ndarray] = {}
        for group in BASE_GROUPS:
            base_oof[group], base_development[group] = oof_and_development_scores(
                group, arrays[group], y, outer_train, development, seed
            )
            for patient_index, score in zip(outer_train, base_oof[group]):
                oof_rows.append(
                    {
                        "seed": seed,
                        "group": group,
                        "patient_alias": patients.patient_alias.iloc[patient_index],
                        "label": int(y[patient_index]),
                        "oof_score": float(score),
                    }
                )
        reference = pd.DataFrame(
            {
                "patient_alias": patients.patient_alias.iloc[development].to_numpy(),
                "label": y[development],
            }
        ).merge(
            pair_reference.loc[pair_reference.seed.eq(seed), ["patient_alias", "label", "pair_score"]],
            on=["patient_alias", "label"],
            validate="one_to_one",
        ).merge(
            i3_reference.loc[i3_reference.seed.eq(seed), ["patient_alias", "label", "i3_score"]],
            on=["patient_alias", "label"],
            validate="one_to_one",
        )
        pair_score = reference.pair_score.to_numpy(float)
        prior_i3_score = reference.i3_score.to_numpy(float)
        computed_pair = (base_development["RAD"] + base_development["DENSE"]) / 2.0
        if not np.allclose(computed_pair, pair_score, atol=1e-6, rtol=0):
            raise RuntimeError(f"Fixed pair reproduction failed for seed {seed}")
        if not np.allclose(base_development["I3"], prior_i3_score, atol=1e-6, rtol=0):
            raise RuntimeError(f"I3 reproduction failed for seed {seed}")
        metric_rows.append(metric(seed, "PAIR", y[development], pair_score))
        metric_rows.append(
            metric(seed, "PAIR+I3-REFERENCE", y[development], (2 * pair_score + prior_i3_score) / 3)
        )
        for meta_group, members in META_GROUPS.items():
            oof_matrix = np.column_stack([base_oof[member] for member in members])
            development_matrix = np.column_stack([base_development[member] for member in members])
            weights, objective = fit_convex_weights(oof_matrix, y[outer_train])
            score = development_matrix @ weights
            metric_rows.append(metric(seed, meta_group, y[development], score))
            for member, weight in zip(members, weights):
                weight_rows.append(
                    {
                        "seed": seed,
                        "group": meta_group,
                        "member": member,
                        "weight": float(weight),
                        "training_oof_objective": objective,
                        "development_labels_used_for_weight": False,
                        "test_accessed": False,
                    }
                )
            for patient_index, value in zip(development, score):
                prediction_rows.append(
                    {
                        "seed": seed,
                        "role": "development_validation",
                        "group": meta_group,
                        "patient_alias": patients.patient_alias.iloc[patient_index],
                        "label": int(y[patient_index]),
                        "score": float(value),
                    }
                )
        print(f"Completed OOF stacking seed {seed}", flush=True)
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(weight_rows),
        pd.DataFrame(oof_rows),
    )


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        metrics.groupby("group", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_auroc=("auroc", "mean"),
            sd_auroc=("auroc", "std"),
            min_auroc=("auroc", "min"),
            max_auroc=("auroc", "max"),
            mean_average_precision=("average_precision", "mean"),
            sd_average_precision=("average_precision", "std"),
            min_average_precision=("average_precision", "min"),
            max_average_precision=("average_precision", "max"),
        )
        .sort_values(["mean_auroc", "mean_average_precision"], ascending=False)
    )
    baseline = metrics.loc[metrics.group.eq("PAIR"), ["seed", "auroc", "average_precision"]].rename(
        columns={"auroc": "pair_auroc", "average_precision": "pair_average_precision"}
    )
    gate_rows = []
    for group in META_GROUPS:
        comparison = metrics.loc[metrics.group.eq(group)].merge(baseline, on="seed", validate="one_to_one")
        delta_auc = comparison.auroc - comparison.pair_auroc
        delta_ap = comparison.average_precision - comparison.pair_average_precision
        joint = int(((delta_auc > 0) & (delta_ap > 0)).sum())
        gain_auc = float(delta_auc.mean())
        gain_ap = float(delta_ap.mean())
        gate_rows.append(
            {
                "group": group,
                "mean_auroc_gain": gain_auc,
                "mean_average_precision_gain": gain_ap,
                "seeds_both_improved": joint,
                "passes_registered_gate": bool(gain_auc >= 0.03 and gain_ap >= 0.02 and joint >= 6),
                "exploratory_test_authorized": False,
            }
        )
    return summary, pd.DataFrame(gate_rows).sort_values("mean_auroc_gain", ascending=False)


def write_readme(summary: pd.DataFrame, gate: pd.DataFrame, weights: pd.DataFrame) -> None:
    baseline = summary.loc[summary.group.eq("PAIR")].iloc[0]
    i3 = summary.loc[summary.group.eq("PAIR+I3-REFERENCE")].iloc[0]
    best = summary.loc[summary.group.isin(META_GROUPS)].iloc[0]
    passed = gate.loc[gate.passes_registered_gate]
    mean_weights = (
        weights.groupby(["group", "member"], as_index=False).weight.mean().sort_values(["group", "weight"], ascending=[True, False])
    )
    weight_text = "\n".join(
        f"- {row.group} / {row.member}: mean weight {row.weight:.3f}"
        for row in mean_weights.itertuples(index=False)
    )
    decision = (
        f"Passing candidate(s): {', '.join(passed.group)}. A separate lock is required before test evaluation."
        if not passed.empty
        else "No candidate passed; the exploratory internal test remains unopened."
    )
    text = f"""# Out-of-fold convex stacking development experiment

## Scope

This experiment replaces hand-adjusted voting weights with a prespecified convex combiner.
Every meta-model is trained on out-of-fold predictions generated solely within the 58
outer-training patients. Development patients never determine the stacking weights, and
the `test` role is not loaded or scored.

## Development results

- Fixed pair: AUROC {baseline.mean_auroc:.3f}, AP {baseline.mean_average_precision:.3f}.
- Existing fixed pair plus I3 reference: AUROC {i3.mean_auroc:.3f}, AP {i3.mean_average_precision:.3f}.
- Best registered OOF stack: {best.group}, AUROC {best.mean_auroc:.3f}, AP {best.mean_average_precision:.3f}.

Registered gate versus pair: mean AUROC gain >=0.03, mean AP gain >=0.02, and
both improve in at least 6/10 draws. {decision}

## Mean learned weights

{weight_text}

## Boundary

These are repeated fixed development results on 98 previously explored patients. They are
not k-fold performance estimates and not independent confirmation. The spatial member uses
reproducible geometry-derived image statistics, not anatomical or lesion ground truth.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-only", action="store_true")
    args = parser.parse_args()
    if args.lock_only:
        lock_protocol()
        return
    verify_lock()
    metrics, predictions, weights, oof = run_development()
    summary, gate = summarize(metrics)
    paths = {
        "metrics_per_seed.csv": metrics,
        "development_predictions.csv": predictions,
        "learned_weights.csv": weights,
        "training_oof_predictions.csv": oof,
        "summary_development.csv": summary,
        "candidate_gate.csv": gate,
    }
    for name, frame in paths.items():
        frame.to_csv(OUT / name, index=False)
    write_json(
        OUT / "run_audit.json",
        {
            "patients": 98,
            "outer_training_patients_per_seed": 58,
            "development_patients_per_seed": 20,
            "roles_evaluated": ["train_oof", "development_validation"],
            "test_role_evaluated": False,
            "exploratory_test_predictions_loaded": False,
            "fixed_pair_reproduced": True,
            "fixed_i3_reproduced": True,
            "passing_candidates": gate.loc[gate.passes_registered_gate, "group"].tolist(),
            "output_hashes": {name: sha256(OUT / name) for name in paths},
        },
    )
    write_readme(summary, gate, weights)
    print(summary.to_string(index=False))
    print("\nCandidate gate:")
    print(gate.to_string(index=False))
    print("\nMean weights:")
    print(weights.groupby(["group", "member"]).weight.mean().to_string())


if __name__ == "__main__":
    main()
