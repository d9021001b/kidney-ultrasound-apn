"""Describe present-cohort agreement between model masks and supplied contours.

This is a quality-control analysis, not an independent validation of the
segmentation checkpoint. The accepted-model subset is defined by the fixed
quality-control rule, so its IoU distribution must not be interpreted as an
unselected accuracy estimate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "CYCH_feature_research_results_20260913" / "image_filename_manifest.csv"
OUTPUT = ROOT / "CYCH_segmentation_qc_current_cohort_20260915"
SEED = 20260915
N_BOOTSTRAP = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def summarize(frame: pd.DataFrame, group_type: str, group: str) -> dict[str, object]:
    accepted = frame.loc[frame["accepted_model_mask"], "predicted_iou"]
    fallback = frame.loc[~frame["accepted_model_mask"]]
    quantiles = accepted.quantile([0.25, 0.50, 0.75]) if len(accepted) else pd.Series(dtype=float)
    return {
        "group_type": group_type,
        "group": group,
        "n_patients": int(frame["patient_id"].nunique()),
        "n_images": int(len(frame)),
        "accepted_model_masks": int(frame["accepted_model_mask"].sum()),
        "supplied_contour_fallbacks": int((~frame["accepted_model_mask"]).sum()),
        "accepted_model_mask_rate": float(frame["accepted_model_mask"].mean()),
        "accepted_iou_q1": float(quantiles.get(0.25, np.nan)),
        "accepted_iou_median": float(quantiles.get(0.50, np.nan)),
        "accepted_iou_q3": float(quantiles.get(0.75, np.nan)),
        "fallback_no_prediction": int((fallback["prediction_count"] == 0).sum()),
        "fallback_with_prediction_failed_qc": int((fallback["prediction_count"] > 0).sum()),
    }


def stratified_patient_bootstrap(frame: pd.DataFrame) -> dict[str, object]:
    patient = (
        frame.groupby(["patient_id", "label"], as_index=False)
        .agg(n_images=("image_id", "size"), accepted=("accepted_model_mask", "sum"))
    )
    strata = {
        int(label): group[["n_images", "accepted"]].to_numpy(dtype=float)
        for label, group in patient.groupby("label")
    }
    if set(strata) != {0, 1}:
        raise ValueError("Expected binary labels 0 and 1")

    rng = np.random.default_rng(SEED)
    overall_rates = np.empty(N_BOOTSTRAP, dtype=float)
    differences = np.empty(N_BOOTSTRAP, dtype=float)
    for index in range(N_BOOTSTRAP):
        sampled: dict[int, np.ndarray] = {}
        for label, values in strata.items():
            positions = rng.integers(0, len(values), size=len(values))
            sampled[label] = values[positions].sum(axis=0)
        rate0 = sampled[0][1] / sampled[0][0]
        rate1 = sampled[1][1] / sampled[1][0]
        overall_rates[index] = (sampled[0][1] + sampled[1][1]) / (
            sampled[0][0] + sampled[1][0]
        )
        differences[index] = rate1 - rate0

    observed = frame.groupby("label")["accepted_model_mask"].mean()
    return {
        "unit": "patient",
        "stratification": "supplied patient-level label",
        "replicates": N_BOOTSTRAP,
        "seed": SEED,
        "observed_overall_acceptance_rate": float(frame["accepted_model_mask"].mean()),
        "overall_acceptance_rate_95_percentile_interval": [
            float(value) for value in np.quantile(overall_rates, [0.025, 0.975])
        ],
        "observed_positive_minus_negative_acceptance_rate": float(observed[1] - observed[0]),
        "positive_minus_negative_acceptance_rate_95_percentile_interval": [
            float(value) for value in np.quantile(differences, [0.025, 0.975])
        ],
    }


def main() -> None:
    frame = pd.read_csv(INPUT)
    required = {
        "patient_id",
        "label",
        "image_id",
        "view",
        "mask_source",
        "prediction_count",
        "predicted_iou",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(frame) != 349 or frame["patient_id"].nunique() != 98:
        raise ValueError("Expected the locked 349-image, 98-patient analysis manifest")
    if frame["predicted_iou"].isna().any():
        raise ValueError("Every image must have a recorded predicted-contour IoU")

    frame["accepted_model_mask"] = frame["mask_source"].eq("existing_yolo")
    rows = [summarize(frame, "overall", "all")]
    rows.extend(
        summarize(group, "label", str(int(label)))
        for label, group in frame.groupby("label", sort=True)
    )
    rows.extend(
        summarize(group, "view", str(view))
        for view, group in frame.groupby("view", sort=True)
    )
    summary = pd.DataFrame(rows)
    bootstrap = stratified_patient_bootstrap(frame)

    total = rows[0]
    if total["accepted_model_masks"] != 228 or total["supplied_contour_fallbacks"] != 121:
        raise ValueError("Mask-source counts do not match the locked manuscript denominator")
    if total["fallback_no_prediction"] + total["fallback_with_prediction_failed_qc"] != 121:
        raise ValueError("Fallback reason counts do not reconcile")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT / "segmentation_qc_summary.csv"
    bootstrap_path = OUTPUT / "segmentation_qc_patient_bootstrap.json"
    summary.to_csv(summary_path, index=False, float_format="%.9f")
    bootstrap_path.write_text(json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8")

    readme = f"""# Present-cohort segmentation quality-control summary

This analysis compares the existing model prediction with the supplied contour for all 349 images from 98 patients. The fixed workflow retained the model mask only when a usable largest instance had at least 64 pixels and IoU at least 0.75; otherwise it used the supplied contour. Consequently, IoU among accepted model masks is selection-conditioned quality-control evidence, not an independent or unselected segmentation-accuracy estimate. Supplied-contour authorship and diagnostic blinding remain author-verification fields.

- Accepted model masks: {total['accepted_model_masks']}/349 ({100 * total['accepted_model_mask_rate']:.1f}%).
- Supplied-contour fallbacks: {total['supplied_contour_fallbacks']}/349 ({100 * (1 - total['accepted_model_mask_rate']):.1f}%).
- Fallbacks with no model prediction: {total['fallback_no_prediction']}.
- Fallbacks with a prediction that failed the fixed usability/IoU rule: {total['fallback_with_prediction_failed_qc']}.
- Accepted-model IoU median (IQR): {total['accepted_iou_median']:.3f} ({total['accepted_iou_q1']:.3f}-{total['accepted_iou_q3']:.3f}).
- Positive-minus-negative accepted-mask rate: {100 * bootstrap['observed_positive_minus_negative_acceptance_rate']:.1f} percentage points; stratified patient-bootstrap 95% percentile interval {100 * bootstrap['positive_minus_negative_acceptance_rate_95_percentile_interval'][0]:.1f} to {100 * bootstrap['positive_minus_negative_acceptance_rate_95_percentile_interval'][1]:.1f} percentage points.

Input SHA-256: `{sha256(INPUT)}`

Run with `python analyze_cych_segmentation_qc.py` from the project root.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    print(summary.to_string(index=False))
    print(json.dumps(bootstrap, indent=2))


if __name__ == "__main__":
    main()
