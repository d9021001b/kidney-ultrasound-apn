"""Development-only equal vote of radiomics, DenseNet, graph MIL, and I3."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import renal_random_validation as R
from renal_dinov3_native_probe import indices_for
from run_gated_attention_mil_development import load_reference, metric_row, read_csv, sha256, write_json


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CYCH_graph_dense_vote_development_20260914"
FEATURE_ROOT = ROOT / "CYCH_dinov3_native_probe_20260913"
BASE_PREDICTIONS = FEATURE_ROOT / "patient_predictions_by_role.csv"
I3_PREDICTIONS = ROOT / "CYCH_directional_robustness_selection_20260914" / "development_predictions.csv"
GRAPH_PREDICTIONS = ROOT / "CYCH_graph_attention_mil_development_20260914" / "development_predictions.csv"


def require_lock() -> None:
    lock = json.loads((OUT / "protocol_locked.json").read_text(encoding="utf-8"))
    assert lock["protocol_sha256"] == sha256(OUT / "protocol.json")
    assert lock["source_sha256"] == sha256(Path(__file__))


def prepare() -> None:
    (OUT / "source").mkdir(parents=True, exist_ok=True)
    for name in ("patients.csv", "random_patient_assignments.csv"):
        source = FEATURE_ROOT / name
        destination = OUT / name
        if destination.exists():
            assert sha256(destination) == sha256(source)
        else:
            shutil.copy2(source, destination)
    patients = read_csv(OUT / "patients.csv")
    assignments = read_csv(OUT / "random_patient_assignments.csv")
    assert len(patients) == 98 and int(patients.label.sum()) == 52
    for seed in R.SEEDS:
        indices_for(patients, assignments, seed)
    protocol = {
        "scope": "development-only fixed equal-probability vote; no model fitting and no exploratory-test evaluation",
        "cohort": {"patients": 98, "positive": 52, "negative": 46},
        "draws": {"seeds": R.SEEDS, "development": 20},
        "components": [
            "existing patient-level PyRadiomics score (RAD)",
            "existing tight-ROI RadImageNet DenseNet-121 score (DN-C)",
            "new graph-attention patient-bag DenseNet score recovered exactly from PAIR-GRAPH-MIL = mean(RAD, graph)",
            "existing directional I3 score",
        ],
        "vote": "Arithmetic mean of the four registered component probabilities; no weight search",
        "rationale": "Retain the complementary fixed DenseNet patient summary while adding the graph-attention patient-bag representation; no weight search",
        "promotion_gate": "Compared with GRAPH-MIL+I3: both mean metrics improve and at least 6/10 draws improve jointly",
        "reference_hashes": {
            "base": sha256(BASE_PREDICTIONS),
            "i3": sha256(I3_PREDICTIONS),
            "graph": sha256(GRAPH_PREDICTIONS),
        },
    }
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding="utf-8")) == protocol
    else:
        write_json(protocol_path, protocol)
    lock_path = OUT / "protocol_locked.json"
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
    print("Four-component vote protocol locked", flush=True)


def recover_graph_score(rad: np.ndarray, pair_graph: np.ndarray) -> np.ndarray:
    graph = 2.0 * np.asarray(pair_graph, dtype=float) - np.asarray(rad, dtype=float)
    if not np.isfinite(graph).all() or np.any((graph < -1e-8) | (graph > 1.0 + 1e-8)):
        raise ValueError("Recovered graph probability is invalid")
    return np.clip(graph, 0.0, 1.0)


def evaluate() -> None:
    require_lock()
    patients = read_csv(OUT / "patients.csv")
    assignments = read_csv(OUT / "random_patient_assignments.csv")
    labels = patients.label.to_numpy(int)
    aliases = patients.patient_alias.tolist()
    rad = load_reference(BASE_PREDICTIONS, "RAD", aliases)
    dense = load_reference(BASE_PREDICTIONS, "DN-C", aliases)
    pair = load_reference(BASE_PREDICTIONS, "PAIR", aliases)
    i3 = load_reference(I3_PREDICTIONS, "I3-ALL", aliases)
    pair_graph = load_reference(GRAPH_PREDICTIONS, "PAIR-GRAPH-MIL", aliases)
    graph_reference = load_reference(GRAPH_PREDICTIONS, "GRAPH-MIL+I3", aliases)
    metrics = []
    predictions = []
    component_rows = []
    for seed in R.SEEDS:
        development = indices_for(patients, assignments, seed)["validation"]
        rad_dev = rad[seed][development]
        dense_dev = dense[seed][development]
        i3_dev = i3[seed][development]
        graph = recover_graph_score(rad_dev, pair_graph[seed][development])
        incumbent = (2.0 * pair[seed][development] + i3_dev) / 3.0
        graph_vote = graph_reference[seed][development]
        four_vote = (rad_dev + dense_dev + graph + i3_dev) / 4.0
        groups = {
            "PAIR+I3-REFERENCE": incumbent,
            "GRAPH-MIL+I3-REFERENCE": graph_vote,
            "RAD+DN+GRAPH+I3": four_vote,
        }
        for group, score in groups.items():
            metrics.append(metric_row(seed, group, labels[development], score))
            for offset, patient_index in enumerate(development):
                predictions.append(
                    {
                        "seed": seed,
                        "group": group,
                        "role": "development_validation",
                        "patient_alias": aliases[patient_index],
                        "label": int(labels[patient_index]),
                        "score": float(score[offset]),
                    }
                )
        for offset, patient_index in enumerate(development):
            component_rows.append(
                {
                    "seed": seed,
                    "patient_alias": aliases[patient_index],
                    "label": int(labels[patient_index]),
                    "rad": float(rad_dev[offset]),
                    "dense": float(dense_dev[offset]),
                    "graph": float(graph[offset]),
                    "i3": float(i3_dev[offset]),
                }
            )
    frame = pd.DataFrame(metrics)
    summary = frame.groupby("group", as_index=False).agg(
        mean_auroc=("auroc", "mean"),
        std_auroc=("auroc", "std"),
        mean_average_precision=("average_precision", "mean"),
        std_average_precision=("average_precision", "std"),
    )
    summary["joint_target_draws"] = [
        int(
            (
                (frame.loc[frame.group.eq(group), "auroc"] > 0.8)
                & (frame.loc[frame.group.eq(group), "average_precision"] > 0.8)
            ).sum()
        )
        for group in summary.group
    ]
    wide = frame.pivot(index="seed", columns="group", values=["auroc", "average_precision"])
    delta_auc = wide["auroc"]["RAD+DN+GRAPH+I3"] - wide["auroc"]["GRAPH-MIL+I3-REFERENCE"]
    delta_ap = (
        wide["average_precision"]["RAD+DN+GRAPH+I3"]
        - wide["average_precision"]["GRAPH-MIL+I3-REFERENCE"]
    )
    wins = int(((delta_auc > 0) & (delta_ap > 0)).sum())
    decision = {
        "selection_role": "development_validation",
        "delta_auroc": float(delta_auc.mean()),
        "delta_ap": float(delta_ap.mean()),
        "joint_wins": wins,
        "promoted": bool(delta_auc.mean() > 0 and delta_ap.mean() > 0 and wins >= 6),
        "mean_targets_both_exceeded": bool(
            summary.set_index("group").loc["RAD+DN+GRAPH+I3", "mean_auroc"] > 0.8
            and summary.set_index("group").loc["RAD+DN+GRAPH+I3", "mean_average_precision"] > 0.8
        ),
        "exploratory_test_access": "prohibited",
    }
    frame.to_csv(OUT / "development_metrics_per_draw.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(component_rows).to_csv(OUT / "development_components.csv", index=False)
    write_json(OUT / "development_decision.json", decision)
    lines = [
        "# Four-component development vote",
        "",
        "No exploratory-test result was computed and no ensemble weight was searched.",
        "",
        "| Group | Mean AUROC | Mean AP | Both >0.8 draws |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.group} | {row.mean_auroc:.3f} | {row.mean_average_precision:.3f} | {row.joint_target_draws}/10 |"
        )
    lines.extend(["", "## Decision", "", f"Decision record: `{json.dumps(decision, separators=(',', ':'))}`"])
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
