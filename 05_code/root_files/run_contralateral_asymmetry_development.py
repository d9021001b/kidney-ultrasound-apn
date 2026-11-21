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


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "CYCH_random_validation_20260913"
OUT = ROOT / "CYCH_contralateral_asymmetry_development_20260914"
MANIFEST_PATH = SOURCE / "image_manifest.csv"
RADIOMICS_PATH = SOURCE / "features" / "radiomics_model_features.csv"
ASSIGNMENTS_PATH = (
    ROOT
    / "CYCH_directional_robustness_selection_20260914"
    / "random_patient_assignments_deidentified.csv"
)
PAIR_PREDICTIONS_PATH = (
    ROOT / "CYCH_dinov3_native_probe_20260913" / "patient_predictions_by_role.csv"
)
PROTOCOL_PATH = OUT / "protocol.json"
REFERENCES_PATH = OUT / "references.json"
LOCK_PATH = OUT / "protocol_lock.json"
SCRIPT_COPY = OUT / "source" / Path(__file__).name

SEEDS = list(range(20260913, 20260923))
K_VALUES = [5, 10, 20]
C_VALUES = [0.1, 1.0, 10.0]


REFERENCES = [
    {
        "finding": "Renal enlargement and left-right size asymmetry in upper urinary-tract infection",
        "citation": "Dinkel et al., AJR, 1986",
        "doi": "10.2214/ajr.146.4.775",
        "url": "https://pubmed.ncbi.nlm.nih.gov/3513490/",
        "relevance": (
            "In 175 children, renal volume increase and size asymmetry were useful for "
            "distinguishing upper from lower urinary-tract infection. This study supplies "
            "the clinical rationale, not a directly transferable classifier."
        ),
    },
    {
        "finding": "Acute renal swelling in pyelonephritis",
        "citation": "Johnson et al., Clinical Infectious Diseases, 1992",
        "doi": "10.1093/clinids/14.1.15",
        "url": "https://pubmed.ncbi.nlm.nih.gov/1571421/",
        "relevance": (
            "Twenty-one evaluable women had mean acute swelling of 20%; one or both kidneys "
            "were enlarged by at least 15% in 81%. Pixel spacing is unavailable in the current "
            "dataset, so only within-exam paired image features are tested here."
        ),
    },
    {
        "finding": "Renal parenchymal-volume enlargement during acute pyelonephritis",
        "citation": "Renal parenchymal volume during and after acute pyelonephritis",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1779153/",
        "relevance": "Supports a renal-size/asymmetry hypothesis and the need for paired comparison.",
    },
    {
        "finding": "Focal loss of corticomedullary differentiation in acute focal bacterial nephritis",
        "citation": "Hosokawa et al., Emergency Radiology, 2020",
        "url": "https://pubmed.ncbi.nlm.nih.gov/32162084/",
        "relevance": (
            "Supports paired texture differences as a possible patient-internal control, while "
            "not providing pixel-level labels for the current cohort."
        ),
    },
    {
        "finding": "B-mode findings of acute pyelonephritis",
        "citation": "Sim, Ultrasonography, 2018",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6177687/",
        "relevance": (
            "Describes enlargement, heterogeneous cortical echogenicity, loss of the "
            "corticomedullary junction, and focal lesions."
        ),
    },
    {
        "method": "PyRadiomics feature definitions",
        "citation": "van Griethuysen et al., Cancer Research, 2017",
        "doi": "10.1158/0008-5472.CAN-17-0339",
        "url": "https://pyradiomics.readthedocs.io/",
        "relevance": "The experiment reuses the locked kidney-mask PyRadiomics feature table.",
    },
]


PROTOCOL = {
    "title": "Development-only contralateral and cross-plane renal asymmetry experiment",
    "status": "locked before model fitting",
    "cohort": {
        "patients": 98,
        "images": 349,
        "views": ["LL", "LT", "RL", "RT"],
        "unit_of_split_and_evaluation": "patient",
        "fixed_seeds": SEEDS,
        "per_seed": {"train": 58, "development_validation": 20, "unused_test": 20},
    },
    "hard_constraints": [
        "Use only features previously extracted inside the fixed kidney segmentation masks.",
        "Build paired features without reading diagnosis labels.",
        "Fit feature selection and classifiers on the 58 outer-training patients only.",
        "Use four-fold stratified inner CV only inside each outer-training set.",
        "Do not evaluate any row assigned role=test.",
        "Do not load exploratory-test baseline predictions.",
        "Do not interpret a high asymmetry score as a localized lesion or affected-side label.",
    ],
    "feature_definitions": {
        "ASYMMETRY": [
            "absolute difference and scale-normalized absolute difference between LL and RL",
            "absolute difference and scale-normalized absolute difference between LT and RT",
            "scale-normalized absolute difference between LL and LT",
            "scale-normalized absolute difference between RL and RT",
        ],
        "VIEW+ASYMMETRY": (
            "All ASYMMETRY features plus the observed LL, LT, RL, and RT feature values. "
            "Missing views are median-imputed from outer-training patients."
        ),
        "relative_difference": "abs(a-b)/(abs(a)+abs(b)+1e-8)",
        "interpretation": (
            "A literature-guided adaptation using patient-internal paired views. It is not a "
            "replication of a published pyelonephritis radiomics classifier."
        ),
    },
    "model": {
        "pipeline": "median imputation, standardization, ANOVA F selection, RBF SVC",
        "inner_selection": "four-fold stratified CV within outer training patients",
        "inner_objective": "mean of AUROC and average precision",
        "feature_counts": K_VALUES,
        "C": C_VALUES,
        "gamma": "scale",
        "class_weight": "balanced",
    },
    "registered_candidates": [
        "ASYMMETRY",
        "VIEW+ASYMMETRY",
        "PAIR75+25%ASYMMETRY",
        "PAIR50+50%ASYMMETRY",
        "PAIR75+25%VIEW+ASYMMETRY",
        "PAIR50+50%VIEW+ASYMMETRY",
    ],
    "candidate_gate_vs_fixed_pair": {
        "mean_auroc_gain_at_least": 0.03,
        "mean_average_precision_gain_at_least": 0.02,
        "seeds_with_both_metrics_improved_at_least": 6,
        "test_action": "No test evaluation unless a registered ensemble passes all criteria.",
    },
    "known_limitations": [
        "No DICOM pixel spacing is available; absolute organ size is not inferred.",
        "The affected kidney side is not available as a trusted label.",
        "A normal contralateral kidney may coexist with a positive patient label.",
        "All 98 patients have already been used in exploratory work.",
    ],
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
    write_json(
        LOCK_PATH,
        {
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "references_sha256": sha256(REFERENCES_PATH),
            "script_sha256": sha256(Path(__file__)),
            "manifest_sha256": sha256(MANIFEST_PATH),
            "radiomics_sha256": sha256(RADIOMICS_PATH),
            "assignments_sha256": sha256(ASSIGNMENTS_PATH),
            "fixed_pair_predictions_sha256": sha256(PAIR_PREDICTIONS_PATH),
        },
    )
    print(f"Locked protocol: {LOCK_PATH}", flush=True)


def verify_lock() -> None:
    if not LOCK_PATH.exists():
        raise RuntimeError("Run --lock-only before model fitting")
    expected = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    paths = {
        "protocol_sha256": PROTOCOL_PATH,
        "references_sha256": REFERENCES_PATH,
        "script_sha256": Path(__file__),
        "manifest_sha256": MANIFEST_PATH,
        "radiomics_sha256": RADIOMICS_PATH,
        "assignments_sha256": ASSIGNMENTS_PATH,
        "fixed_pair_predictions_sha256": PAIR_PREDICTIONS_PATH,
    }
    changed = [key for key, path in paths.items() if sha256(path) != expected[key]]
    if changed:
        raise RuntimeError(f"Locked inputs changed: {changed}")


def relative_difference(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).abs() / (a.abs() + b.abs() + 1e-8)


def build_patient_features() -> pd.DataFrame:
    manifest = pd.read_csv(
        MANIFEST_PATH,
        usecols=["image_id", "patient_alias", "view"],
        dtype=str,
    )
    radiomics = pd.read_csv(RADIOMICS_PATH)
    table = manifest.merge(radiomics, on="image_id", validate="one_to_one")
    feature_columns = [column for column in radiomics.columns if column != "image_id"]
    view_tables = {}
    for view in ("LL", "LT", "RL", "RT"):
        subset = table.loc[table["view"] == view].groupby("patient_alias")[feature_columns].mean()
        view_tables[view] = subset
    patients = sorted(manifest["patient_alias"].unique())
    index = pd.Index(patients, name="patient_alias")
    aligned = {view: frame.reindex(index) for view, frame in view_tables.items()}
    output = pd.DataFrame(index=index)
    pairs = {
        "long_lr": ("LL", "RL"),
        "trans_lr": ("LT", "RT"),
        "left_plane": ("LL", "LT"),
        "right_plane": ("RL", "RT"),
    }
    for pair_name, (first_view, second_view) in pairs.items():
        first = aligned[first_view]
        second = aligned[second_view]
        absolute = (first - second).abs()
        relative = relative_difference(first, second)
        absolute.columns = [f"asym_abs__{pair_name}__{column}" for column in feature_columns]
        relative.columns = [f"asym_rel__{pair_name}__{column}" for column in feature_columns]
        output = output.join(absolute).join(relative)
    for view, frame in aligned.items():
        values = frame.copy()
        values.columns = [f"view_value__{view}__{column}" for column in feature_columns]
        output = output.join(values)
        output[f"view_present__{view}"] = frame.notna().any(axis=1).astype(int)
    result = output.reset_index()
    if result.shape[0] != 98:
        raise RuntimeError(f"Expected 98 patients, found {result.shape[0]}")
    return result


def load_development_pair_predictions() -> pd.DataFrame:
    rows = []
    with PAIR_PREDICTIONS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["role"] != "development_validation" or row["group"] != "PAIR":
                continue
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "patient_alias": row["patient_alias"],
                    "label": int(row["label"]),
                    "pair_score": float(row["score"]),
                }
            )
    result = pd.DataFrame(rows)
    if result.shape != (200, 4):
        raise RuntimeError(f"Unexpected pair-development shape: {result.shape}")
    return result


def load_modeling_dependencies() -> None:
    global SelectKBest, f_classif, SimpleImputer, average_precision_score
    global roc_auc_score, StratifiedKFold, Pipeline, StandardScaler, SVC
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC


def build_model(k: int, c_value: float, seed: int) -> object:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=c_value,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def tune_inside_training(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[int, float, float]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    records = []
    for k in K_VALUES:
        for c_value in C_VALUES:
            fold_scores = []
            for train_index, validation_index in splitter.split(x, y):
                model = build_model(k, c_value, seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x[train_index], y[train_index])
                score = model.predict_proba(x[validation_index])[:, 1]
                fold_scores.append(
                    (
                        roc_auc_score(y[validation_index], score)
                        + average_precision_score(y[validation_index], score)
                    )
                    / 2.0
                )
            records.append((float(np.mean(fold_scores)), -k, -c_value, k, c_value))
    best = max(records)
    return int(best[3]), float(best[4]), float(best[0])


def metric(seed: int, group: str, y: np.ndarray, score: np.ndarray) -> dict[str, object]:
    return {
        "seed": seed,
        "group": group,
        "role": "development_validation",
        "n": len(y),
        "auroc": roc_auc_score(y, score),
        "average_precision": average_precision_score(y, score),
    }


def run_development(patient_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments = pd.read_csv(
        ASSIGNMENTS_PATH,
        usecols=["seed", "role", "patient_alias", "label"],
        dtype={"patient_alias": str},
    )
    assignments = assignments.loc[assignments["role"].isin(["train", "validation"])].copy()
    if set(assignments["role"]) != {"train", "validation"}:
        raise RuntimeError("Unexpected evaluation role")
    pair = load_development_pair_predictions()
    asymmetry_columns = [column for column in patient_features if column.startswith("asym_")]
    view_columns = [column for column in patient_features if column.startswith("view_value__")]
    indicator_columns = [column for column in patient_features if column.startswith("view_present__")]
    candidate_columns = {
        "ASYMMETRY": asymmetry_columns + indicator_columns,
        "VIEW+ASYMMETRY": asymmetry_columns + view_columns + indicator_columns,
    }
    metrics = []
    predictions = []
    tuning = []
    for seed in SEEDS:
        split = assignments.loc[assignments["seed"] == seed]
        train = split.loc[split["role"] == "train", ["patient_alias", "label"]]
        development = split.loc[split["role"] == "validation", ["patient_alias", "label"]]
        train_frame = train.merge(patient_features, on="patient_alias", validate="one_to_one")
        development_frame = development.merge(patient_features, on="patient_alias", validate="one_to_one")
        pair_frame = development.merge(
            pair.loc[pair["seed"] == seed, ["patient_alias", "label", "pair_score"]],
            on=["patient_alias", "label"],
            validate="one_to_one",
        )
        if list(development_frame["patient_alias"]) != list(pair_frame["patient_alias"]):
            raise RuntimeError(f"Patient order mismatch at seed {seed}")
        y_train = train_frame["label"].to_numpy(dtype=int)
        y_development = development_frame["label"].to_numpy(dtype=int)
        pair_score = pair_frame["pair_score"].to_numpy(dtype=float)
        metrics.append(metric(seed, "PAIR", y_development, pair_score))
        for group, columns in candidate_columns.items():
            x_train = train_frame[columns].to_numpy(dtype=float)
            x_development = development_frame[columns].to_numpy(dtype=float)
            k, c_value, inner_score = tune_inside_training(x_train, y_train, seed)
            model = build_model(k, c_value, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_train, y_train)
            score = model.predict_proba(x_development)[:, 1]
            metrics.append(metric(seed, group, y_development, score))
            tuning.append(
                {
                    "seed": seed,
                    "group": group,
                    "selected_k": k,
                    "selected_C": c_value,
                    "inner_cv_objective": inner_score,
                    "test_role_accessed": False,
                }
            )
            for patient_alias, label, value in zip(
                development_frame["patient_alias"], y_development, score
            ):
                predictions.append(
                    {
                        "seed": seed,
                        "role": "development_validation",
                        "group": group,
                        "patient_alias": patient_alias,
                        "label": int(label),
                        "score": float(value),
                    }
                )
            for candidate_weight in (0.25, 0.50):
                ensemble_name = f"PAIR{int((1-candidate_weight)*100)}+{int(candidate_weight*100)}%{group}"
                ensemble = (1.0 - candidate_weight) * pair_score + candidate_weight * score
                metrics.append(metric(seed, ensemble_name, y_development, ensemble))
                for patient_alias, label, value in zip(
                    development_frame["patient_alias"], y_development, ensemble
                ):
                    predictions.append(
                        {
                            "seed": seed,
                            "role": "development_validation",
                            "group": ensemble_name,
                            "patient_alias": patient_alias,
                            "label": int(label),
                            "score": float(value),
                        }
                    )
        print(f"Completed development seed {seed}", flush=True)
    return pd.DataFrame(metrics), pd.DataFrame(predictions), pd.DataFrame(tuning)


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
    baseline = metrics.loc[metrics["group"] == "PAIR", ["seed", "auroc", "average_precision"]].rename(
        columns={"auroc": "pair_auroc", "average_precision": "pair_average_precision"}
    )
    gate_rows = []
    ensemble_metrics = metrics.loc[metrics["group"].str.startswith("PAIR") & (metrics["group"] != "PAIR")]
    for group, frame in ensemble_metrics.groupby("group"):
        comparison = frame.merge(baseline, on="seed", validate="one_to_one")
        auc_gain = comparison["auroc"] - comparison["pair_auroc"]
        ap_gain = comparison["average_precision"] - comparison["pair_average_precision"]
        both = int(((auc_gain > 0) & (ap_gain > 0)).sum())
        gate_rows.append(
            {
                "group": group,
                "mean_auroc_gain": float(auc_gain.mean()),
                "mean_average_precision_gain": float(ap_gain.mean()),
                "seeds_both_improved": both,
                "passes_registered_gate": bool(auc_gain.mean() >= 0.03 and ap_gain.mean() >= 0.02 and both >= 6),
                "exploratory_test_authorized": False,
            }
        )
    gate = pd.DataFrame(gate_rows).sort_values(
        ["passes_registered_gate", "mean_auroc_gain", "mean_average_precision_gain"],
        ascending=False,
    )
    return summary, gate


def write_readme(summary: pd.DataFrame, gate: pd.DataFrame, feature_count: int) -> None:
    baseline = summary.loc[summary["group"] == "PAIR"].iloc[0]
    standalone = summary.loc[summary["group"].isin(["ASYMMETRY", "VIEW+ASYMMETRY"])].iloc[0]
    ensembles = summary.loc[summary["group"].str.startswith("PAIR") & (summary["group"] != "PAIR")]
    best_ensemble = ensembles.iloc[0]
    passed = gate.loc[gate["passes_registered_gate"]]
    decision = (
        f"Passing registered ensemble(s): {', '.join(passed['group'])}. A separate pipeline lock is required before test evaluation."
        if not passed.empty
        else "No registered ensemble passed; exploratory internal test remains unopened."
    )
    text = f"""# Contralateral renal asymmetry development experiment

## Scope

This experiment tests a literature-supported but newly adapted hypothesis: renal inflammation
may create patient-internal differences in kidney size, echogenicity, heterogeneity, or
corticomedullary texture. Existing kidney-mask PyRadiomics features were explicitly paired
between LL/RL and LT/RT views. This is not a published classifier replication.

Only the ten fixed 20-patient development sets were scored. Hyperparameters were selected
inside each corresponding 58-patient training set. The `test` role and exploratory-test
baseline predictions were not evaluated.

## Results

- De-identified patient feature columns: {feature_count:,}.
- Fixed pair: AUROC {baseline.mean_auroc:.3f}, AP {baseline.mean_average_precision:.3f}.
- Best standalone candidate: {standalone.group}, AUROC {standalone.mean_auroc:.3f}, AP {standalone.mean_average_precision:.3f}.
- Best registered ensemble: {best_ensemble.group}, AUROC {best_ensemble.mean_auroc:.3f}, AP {best_ensemble.mean_average_precision:.3f}.
- Decision: {decision}

## Interpretation boundary

No physical pixel spacing is available, so this experiment does not claim absolute renal
length or volume. No trusted affected-side label is available, so an asymmetry feature cannot
be interpreted as lesion localization. Results are repeated development estimates from an
already explored 98-patient cohort, not k-fold performance or independent confirmation.

See `protocol.json`, `references.json`, `protocol_lock.json`, `metrics_per_seed.csv`,
`summary_development.csv`, `candidate_gate.csv`, and `selected_hyperparameters.csv`.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    if args.lock_only:
        lock_protocol()
        return
    verify_lock()
    load_modeling_dependencies()
    feature_path = OUT / "patient_features_deidentified.csv"
    if args.reuse_features and feature_path.exists():
        patient_features = pd.read_csv(feature_path)
    else:
        patient_features = build_patient_features()
        patient_features.to_csv(feature_path, index=False)
    metrics, predictions, tuning = run_development(patient_features)
    summary, gate = summarize(metrics)
    metrics.to_csv(OUT / "metrics_per_seed.csv", index=False)
    predictions.to_csv(OUT / "development_predictions.csv", index=False)
    tuning.to_csv(OUT / "selected_hyperparameters.csv", index=False)
    summary.to_csv(OUT / "summary_development.csv", index=False)
    gate.to_csv(OUT / "candidate_gate.csv", index=False)
    write_json(
        OUT / "run_audit.json",
        {
            "patients": int(patient_features["patient_alias"].nunique()),
            "features": int(patient_features.shape[1] - 1),
            "roles_evaluated": ["train", "development_validation"],
            "test_role_evaluated": False,
            "exploratory_test_predictions_loaded": False,
            "passing_candidates": gate.loc[gate["passes_registered_gate"], "group"].tolist(),
            "output_hashes": {
                path.name: sha256(path)
                for path in [
                    feature_path,
                    OUT / "metrics_per_seed.csv",
                    OUT / "development_predictions.csv",
                    OUT / "selected_hyperparameters.csv",
                    OUT / "summary_development.csv",
                    OUT / "candidate_gate.csv",
                ]
            },
        },
    )
    write_readme(summary, gate, patient_features.shape[1] - 1)
    print(summary.to_string(index=False))
    print("\nCandidate gate:")
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
