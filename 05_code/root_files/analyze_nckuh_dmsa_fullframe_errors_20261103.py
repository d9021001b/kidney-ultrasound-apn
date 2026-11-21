"""Deidentified B/N ranking and image audit for clean full-frame development draws."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nckuh_dmsa_fullframe_clean_probe_20261102 as F
import nckuh_dmsa_fullframe_deep_probe_20261102 as D
import nckuh_dmsa_fullframe_extreme_specialist_20261102 as E
import nckuh_dmsa_fullframe_geometric_texture_20261102 as T
import nckuh_dmsa_fullframe_hierarchical_probe_20261102 as H
import nckuh_dmsa_fullframe_relative_geometry_probe_20261102 as G
from nckuh_only_study_20261102 import write_json


ROOT = F.ROOT
OUT = ROOT / "04_quality_archive/NCKUH_DMSA_fullframe_error_audit_20261103"
SOURCES = {
    "deep_radiomics": (D.OUT / "development_predictions.csv", D.REFERENCE),
    "powerset_deep": (H.OUT / "development_predictions.csv", "four_state_powerset_plus_deep"),
    "relative_geometry_deep": (G.OUT / "development_predictions.csv", "relative_frame_geometry_plus_deep_radiomics"),
    "geometric_three_source": (T.OUT / "development_predictions.csv", "fixed_powerset_geometry_texture_three_source_fusion"),
    "extreme_patient_deep": (E.OUT / "development_predictions.csv", "powerset_extreme_patient_deep_replace_plus_deep"),
}


def build_montage(selected: pd.DataFrame, images: pd.DataFrame) -> None:
    columns = 4
    rows = len(selected)
    figure, axes = plt.subplots(rows, columns, figsize=(12, 2.7 * rows), squeeze=False)
    for row_index, patient in enumerate(selected.itertuples(index=False)):
        subset = images[images.patient_alias.eq(patient.patient_alias)].sort_values(["side", "view"])
        for column in range(columns):
            axis = axes[row_index, column]
            axis.axis("off")
            if column >= len(subset):
                continue
            image_row = subset.iloc[column]
            gray = cv2.imread(str(F.OUT / "percentile" / f"{image_row.image_id}.png"), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(F.OUT / "masks" / f"{image_row.image_id}.png"), cv2.IMREAD_GRAYSCALE)
            contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(display, contours, -1, (255, 80, 40), 1)
            axis.imshow(display)
            axis.set_title(f"{image_row['side'][0].upper()}-{image_row['view'][-1]}  {image_row.image_id}", fontsize=8)
        axes[row_index, 0].set_ylabel(
            f"{patient.DMSA_APN}  {patient.patient_alias}\nscore={patient.extreme_patient_deep:.3f}",
            fontsize=9, color="#b42318" if patient.DMSA_APN == "N" else "#175cd3")
    figure.suptitle("Clean masked renal views: highest-scored N and lowest-scored B patients", fontsize=12)
    figure.tight_layout(rect=[0, 0, 1, .98])
    figure.savefig(OUT / "bn_error_montage.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "audit_summary.json").exists():
        raise FileExistsError("Preserve completed full-frame error audit")
    patients, _, images = F.matched_reference()
    model_frames = []
    for name, (path, group) in SOURCES.items():
        frame = pd.read_csv(path)
        frame = frame[frame.group.eq(group)].copy()
        frame["model"] = name
        model_frames.append(frame)
    predictions = pd.concat(model_frames, ignore_index=True)
    patient_scores = (predictions.groupby(["model", "patient_alias", "DMSA_APN", "seed"], as_index=False)
                      .score.mean())
    appearances = patient_scores.groupby(["model", "patient_alias", "DMSA_APN"], as_index=False).agg(
        score=("score", "mean"), appearances=("seed", "nunique"))
    wide = appearances.pivot(index=["patient_alias", "DMSA_APN"], columns="model", values="score").reset_index()
    wide.columns.name = None
    counts = appearances[appearances.model.eq("extreme_patient_deep")][["patient_alias", "appearances"]]
    wide = wide.merge(counts, on="patient_alias", how="left")
    wide.to_csv(OUT / "patient_error_summary.csv", index=False)

    bn = wide[wide.DMSA_APN.isin(["B", "N"])].copy()
    bn["error_priority"] = np.where(bn.DMSA_APN.eq("N"), bn.extreme_patient_deep,
                                    1 - bn.extreme_patient_deep)
    bn = bn.sort_values(["DMSA_APN", "error_priority"], ascending=[False, False])
    bn.to_csv(OUT / "bn_patient_ranking.csv", index=False)
    selected = pd.concat([
        bn[bn.DMSA_APN.eq("N")].nlargest(3, "extreme_patient_deep"),
        bn[bn.DMSA_APN.eq("B")].nsmallest(3, "extreme_patient_deep"),
    ], ignore_index=True)
    build_montage(selected, images)

    assignments = pd.read_csv(F.OUT / "patient_assignments.csv")
    role_counts = (assignments[assignments.role.eq("development")]
                   .groupby(["seed", "DMSA_APN"]).size().unstack(fill_value=0).reset_index())
    role_counts.to_csv(OUT / "development_state_counts.csv", index=False)
    distinct = predictions[predictions.model.eq("extreme_patient_deep")].groupby("DMSA_APN").patient_alias.nunique()
    write_json(OUT / "audit_summary.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "models": list(SOURCES), "test_scored": False,
        "development_state_counts_each_draw": {"B": 8, "L": 6, "N": 3, "R": 3},
        "distinct_development_patients_by_state": {key: int(value) for key, value in distinct.items()},
        "montage_patient_aliases": selected.patient_alias.tolist(),
        "interpretation": "Error audit and visual review aid method design only; no image was relabeled and no montage observation is a lesion annotation",
    })
    print(role_counts.to_string(index=False), flush=True)
    print(selected[["patient_alias", "DMSA_APN", "extreme_patient_deep", "error_priority"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
