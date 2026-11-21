"""Development-only error decomposition and non-deployable laterality oracle."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "02_experiments/NCKUH_DMSA_uricads_official_frozen_20261105/development_predictions.csv"
OUT = ROOT / "02_experiments/NCKUH_DMSA_error_oracle_20261106"
GROUP = "official_uri_embedding_folds_quarter_plus_current"


def laterality_oracle(frame: pd.DataFrame) -> np.ndarray:
    """Swap only complete unilateral-pair scores so the affected kidney ranks first."""
    scores = frame.score.to_numpy(float).copy()
    positions = {index: position for position, index in enumerate(frame.index)}
    unilateral = frame[frame.DMSA_APN.isin(["L", "R"])]
    for _, pair in unilateral.groupby("patient_alias"):
        if pair.side.nunique() != 2 or len(pair) != 2:
            continue
        affected = (
            (pair.DMSA_APN.eq("L") & pair.side.eq("left"))
            | (pair.DMSA_APN.eq("R") & pair.side.eq("right"))
        )
        affected_position = positions[pair.index[affected][0]]
        contralateral_position = positions[pair.index[~affected][0]]
        high = max(scores[affected_position], scores[contralateral_position])
        low = min(scores[affected_position], scores[contralateral_position])
        scores[affected_position] = high
        scores[contralateral_position] = low
    return scores


def metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
    }


def subgroup(frame: pd.DataFrame) -> pd.Series:
    affected = (
        (frame.DMSA_APN.eq("L") & frame.side.eq("left"))
        | (frame.DMSA_APN.eq("R") & frame.side.eq("right"))
    )
    return pd.Series(
        np.select(
            [frame.DMSA_APN.eq("B"), frame.DMSA_APN.eq("N"), affected],
            ["B_positive", "N_negative", "U_affected"],
            default="U_contralateral",
        ),
        index=frame.index,
    )


def run() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Preserve completed error decomposition: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE)
    data = data[data.group.eq(GROUP)].copy()
    data["role"] = "development"
    if sorted(data.role.unique()) != ["development"]:
        raise ValueError("Oracle diagnostic must use development rows only")
    if sorted(data.seed.unique().tolist()) != [20261102, 20261103, 20261104]:
        raise ValueError("Unexpected development draws")
    data["subgroup"] = subgroup(data)
    task_rows: list[dict] = []
    pair_rows: list[dict] = []
    oracle_rows: list[dict] = []
    for seed, draw in data.groupby("seed", sort=True):
        tasks = {
            "all_kidneys": draw,
            "N_vs_B": draw[draw.DMSA_APN.isin(["N", "B"])],
            "unilateral_kidneys": draw[draw.DMSA_APN.isin(["L", "R"])],
            "N_vs_unilateral_affected": draw[
                draw.subgroup.isin(["N_negative", "U_affected"])
            ],
        }
        for name, current in tasks.items():
            if current.label.nunique() != 2:
                continue
            task_rows.append(
                {
                    "seed": seed,
                    "task": name,
                    "rows": len(current),
                    "patients": current.patient_alias.nunique(),
                    **metric(current.label.to_numpy(int), current.score.to_numpy(float)),
                }
            )
        patient = draw.groupby(["patient_alias", "DMSA_APN"], as_index=False).score.max()
        patient_label = patient.DMSA_APN.ne("N").to_numpy(int)
        task_rows.append(
            {
                "seed": seed,
                "task": "patient_any_apn_max",
                "rows": len(patient),
                "patients": len(patient),
                **metric(patient_label, patient.score.to_numpy(float)),
            }
        )
        oracle_score = laterality_oracle(draw)
        for name, score in (("fixed", draw.score.to_numpy(float)), ("perfect_unilateral_order", oracle_score)):
            oracle_rows.append(
                {
                    "seed": seed,
                    "group": name,
                    **metric(draw.label.to_numpy(int), score),
                }
            )
        for positive in ("B_positive", "U_affected"):
            positive_scores = draw.loc[draw.subgroup.eq(positive), "score"].to_numpy(float)
            for negative in ("N_negative", "U_contralateral"):
                negative_scores = draw.loc[draw.subgroup.eq(negative), "score"].to_numpy(float)
                comparisons = positive_scores[:, None] - negative_scores
                concordance = float(
                    (comparisons > 0).mean() + 0.5 * (comparisons == 0).mean()
                )
                pair_rows.append(
                    {
                        "seed": seed,
                        "positive": positive,
                        "negative": negative,
                        "positive_rows": len(positive_scores),
                        "negative_rows": len(negative_scores),
                        "pairwise_concordance": concordance,
                    }
                )

    tasks = pd.DataFrame(task_rows)
    pairs = pd.DataFrame(pair_rows)
    oracle = pd.DataFrame(oracle_rows)
    tasks.to_csv(OUT / "task_by_draw.csv", index=False)
    pairs.to_csv(OUT / "pairwise_concordance_by_draw.csv", index=False)
    oracle.to_csv(OUT / "laterality_oracle_by_draw.csv", index=False)
    tasks.groupby("task")[["auroc", "ap", "brier"]].agg(["mean", "std"]).to_csv(
        OUT / "task_summary.csv"
    )
    pairs.groupby(["positive", "negative"])["pairwise_concordance"].agg(
        ["mean", "std"]
    ).to_csv(OUT / "pairwise_concordance_summary.csv")
    oracle_summary = oracle.groupby("group")[["auroc", "ap", "brier"]].agg(
        ["mean", "std"]
    )
    oracle_summary.to_csv(OUT / "laterality_oracle_summary.csv")
    write_json = {
        "purpose": "Diagnostic upper-bound analysis only; not a trained model and not reportable predictive performance.",
        "source_group": GROUP,
        "roles": sorted(data.role.unique()),
        "exploratory_test_rows_used": 0,
        "laterality_oracle_rule": "Within each complete unilateral patient, preserve the two score values but assign the higher value to the DMSA-affected kidney.",
        "oracle_above_both_targets": bool(
            oracle_summary.loc["perfect_unilateral_order", ("auroc", "mean")] > 0.8
            and oracle_summary.loc["perfect_unilateral_order", ("ap", "mean")] > 0.8
        ),
    }
    (OUT / "audit.json").write_text(
        json.dumps(write_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    run()
