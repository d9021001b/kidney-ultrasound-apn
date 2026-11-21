"""Locked exploratory-test evaluation of the selected attention-weighted I3 model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

import renal_random_validation as R
import renal_voting_validation as V
from renal_dinov3_native_probe import indices_for


ROOT = Path(__file__).resolve().parent
DEVELOPMENT = ROOT / "CYCH_attention_weighted_directional_development_20260914"
OUT = ROOT / "CYCH_attention_weighted_directional_exploratory_test_20260914"
PAIR_PREDICTIONS = ROOT / "CYCH_dinov3_native_probe_20260913" / "patient_predictions_by_role.csv"
HISTORICAL_PREDICTIONS = ROOT / "CYCH_interpretable_features_20260913" / "patient_predictions_by_role.csv"
SELECTED = "PAIR+ATTN-I3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"patient_id": str})


def require_lock() -> None:
    lock = json.loads((OUT / "test_protocol_locked.json").read_text(encoding="utf-8"))
    assert lock["protocol_sha256"] == sha256(OUT / "test_protocol.json")
    assert lock["source_sha256"] == sha256(Path(__file__))
    candidate = json.loads((OUT / "candidate_lock.json").read_text(encoding="utf-8"))
    assert candidate["selected"] == SELECTED
    assert candidate["development_decision_sha256"] == sha256(DEVELOPMENT / "development_decision.json")


def prepare() -> None:
    decision_path = DEVELOPMENT / "development_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["selected"] == SELECTED
    OUT.mkdir(exist_ok=True)
    (OUT / "source").mkdir(exist_ok=True)
    for name in ("patients.csv", "random_patient_assignments.csv"):
        source = DEVELOPMENT / name
        destination = OUT / name
        if destination.exists():
            assert sha256(destination) == sha256(source)
        else:
            shutil.copy2(source, destination)
    model_hashes = {}
    for seed in R.SEEDS:
        for prefix, suffix in (("attention_i3", ".joblib"), ("attention_i3", ".npy")):
            directory = "models" if suffix == ".joblib" else "features"
            path = DEVELOPMENT / directory / f"{prefix}_{seed}{suffix}"
            model_hashes[str(path)] = sha256(path)
    candidate_lock = {
        "selected": SELECTED,
        "development_decision_sha256": sha256(decision_path),
        "development_predictions_sha256": sha256(DEVELOPMENT / "development_predictions.csv"),
        "artifacts": model_hashes,
    }
    candidate_path = OUT / "candidate_lock.json"
    if candidate_path.exists():
        assert json.loads(candidate_path.read_text(encoding="utf-8")) == candidate_lock
    else:
        write_json(candidate_path, candidate_lock)
    protocol = {
        "scope": "Exploratory internal-test evaluation after development-only candidate selection",
        "selected_candidate": SELECTED,
        "score": "Equal probabilities from patient-level PyRadiomics, fixed DenseNet, and attention-weighted I3; implemented as (2*fixed_pair + attention_I3)/3",
        "test_role": "Ten fixed 20-patient exploratory internal-test subsets from the previously explored 98-patient cohort; not independent confirmation and not k-fold validation",
        "no_selection": "No model, feature, hyperparameter, threshold, or ensemble weight may change after this lock",
        "reproduction_gate": "All selected-candidate development predictions must reproduce at absolute tolerance 1e-12 before any test metric is written",
        "inputs": {
            "pair_predictions_sha256": sha256(PAIR_PREDICTIONS),
            "historical_predictions_sha256": sha256(HISTORICAL_PREDICTIONS),
            "candidate_lock_sha256": sha256(candidate_path),
        },
    }
    protocol_path = OUT / "test_protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding="utf-8")) == protocol
    else:
        write_json(protocol_path, protocol)
    lock_path = OUT / "test_protocol_locked.json"
    if not lock_path.exists():
        write_json(
            lock_path,
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": sha256(protocol_path),
                "source_sha256": sha256(Path(__file__)),
            },
        )
    shutil.copy2(__file__, OUT / "source" / Path(__file__).name)
    print("Selected candidate and exploratory-test protocol locked", flush=True)


def reference_scores(path: Path, group: str, role: str, aliases: list[str], seed: int) -> np.ndarray:
    frame = pd.read_csv(path)
    frame = frame.loc[
        frame.seed.eq(seed) & frame.group.eq(group) & frame.role.eq(role),
        ["patient_alias", "label", "score"],
    ]
    mapping = frame.set_index("patient_alias")
    assert len(mapping) == 20 and mapping.index.is_unique
    return mapping.reindex(aliases).score.to_numpy(float)


def evaluate() -> None:
    require_lock()
    patients = read_csv(OUT / "patients.csv")
    assignments = read_csv(OUT / "random_patient_assignments.csv")
    labels = patients.label.to_numpy(int)
    aliases = patients.patient_alias.tolist()
    development_predictions = pd.read_csv(DEVELOPMENT / "development_predictions.csv")
    rows = []
    predictions = []
    reproduction_maxabs = []

    for seed in R.SEEDS:
        indices = indices_for(patients, assignments, seed)
        features = np.load(DEVELOPMENT / "features" / f"attention_i3_{seed}.npy")
        head = joblib.load(DEVELOPMENT / "models" / f"attention_i3_{seed}.joblib")
        directional_scores = head.predict_proba(features)[:, 1]
        pair_dev = reference_scores(PAIR_PREDICTIONS, "PAIR", "development_validation", aliases, seed)
        selected_dev = (2.0 * pair_dev + directional_scores) / 3.0
        expected = development_predictions.loc[
            development_predictions.seed.eq(seed) & development_predictions.group.eq(SELECTED),
            ["patient_alias", "score"],
        ].set_index("patient_alias").reindex(aliases).score.to_numpy(float)
        development = indices["validation"]
        difference = float(np.max(np.abs(selected_dev[development] - expected[development])))
        reproduction_maxabs.append(difference)
        assert difference <= 1e-12

        test = indices["test"]
        pair_test = reference_scores(PAIR_PREDICTIONS, "PAIR", "exploratory_test", aliases, seed)
        selected_test = (2.0 * pair_test + directional_scores) / 3.0
        incumbent_test = reference_scores(HISTORICAL_PREDICTIONS, "VOTE", "exploratory_test", aliases, seed)
        rows.append({"seed": seed, "group": SELECTED, **V.score_metrics(labels[test], selected_test[test])})
        rows.append({"seed": seed, "group": "PAIR+I3-REFERENCE", **V.score_metrics(labels[test], incumbent_test[test])})
        rows.append({"seed": seed, "group": "PAIR", **V.score_metrics(labels[test], pair_test[test])})
        for group, scores in ((SELECTED, selected_test), ("PAIR+I3-REFERENCE", incumbent_test), ("PAIR", pair_test)):
            for index in test:
                predictions.append(
                    {
                        "seed": seed,
                        "group": group,
                        "role": "exploratory_test",
                        "patient_alias": aliases[index],
                        "label": int(labels[index]),
                        "score": float(scores[index]),
                    }
                )
        print(f"Locked exploratory test complete: {seed}", flush=True)

    frame = pd.DataFrame(rows)
    summary = frame.groupby("group", as_index=False).agg(
        mean_auroc=("auroc", "mean"),
        std_auroc=("auroc", "std"),
        mean_average_precision=("average_precision", "mean"),
        std_average_precision=("average_precision", "std"),
    )
    summary["joint_target_draws"] = [
        int(((frame.loc[frame.group.eq(group), "auroc"] > 0.8) &
             (frame.loc[frame.group.eq(group), "average_precision"] > 0.8)).sum())
        for group in summary.group
    ]
    reference = frame.loc[frame.group.eq("PAIR+I3-REFERENCE")].set_index("seed")
    candidate = frame.loc[frame.group.eq(SELECTED)].set_index("seed")
    delta_auc = candidate.auroc - reference.auroc
    delta_ap = candidate.average_precision - reference.average_precision
    decision = {
        "candidate": SELECTED,
        "development_reproduction_maxabs": max(reproduction_maxabs),
        "delta_auroc_vs_reference": float(delta_auc.mean()),
        "delta_ap_vs_reference": float(delta_ap.mean()),
        "joint_improvement_draws": int(((delta_auc > 0) & (delta_ap > 0)).sum()),
        "interpretation": "Exploratory internal-test result on a previously explored cohort; independent confirmation is still required.",
    }
    frame.to_csv(OUT / "exploratory_test_metrics_per_draw.csv", index=False)
    summary.to_csv(OUT / "exploratory_test_summary.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "exploratory_test_predictions.csv", index=False)
    write_json(OUT / "exploratory_test_decision.json", decision)
    lines = [
        "# Locked attention-weighted I3 exploratory test",
        "",
        "The development-selected candidate was locked before test evaluation, and all development predictions reproduced exactly.",
        "These subsets belong to the same previously explored 98-patient cohort and are not independent confirmation.",
        "",
        "| Group | Mean AUROC | Mean AP | Both >0.8 draws |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(f"| {row.group} | {row.mean_auroc:.3f} | {row.mean_average_precision:.3f} | {row.joint_target_draws}/10 |")
    lines.extend(["", "## Audit", "", f"Decision record: `{json.dumps(decision, separators=(',', ':'))}`"])
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(decision), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "evaluate", "all"))
    args = parser.parse_args()
    stages = ("prepare", "evaluate") if args.stage == "all" else (args.stage,)
    for stage in stages:
        globals()[stage]()


if __name__ == "__main__":
    main()
