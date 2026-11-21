"""Training-scoped Doppler-perfusion probe on contour-first NCKUH RGB images.

This is intentionally a separate evidence branch from the grayscale/RadImageNet
models.  It quantifies saturated red/orange/blue pixels *inside an eroded kidney
mask*, excludes cyan/green UI contours, and learns its compact kidney-level head
only from the outer training patients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import nckuh_dmsa_fullcohort_fullframe_20261103 as F
from nckuh_only_study_20261102 import metric


ROOT = F.ROOT
BASE = F.OUT
OUT = ROOT / "02_experiments/NCKUH_DMSA_doppler_perfusion_20261103"
REFERENCE_GROUP = "fullcohort_rad_color_image_plus_patient_direct"
SEEDS = F.SEEDS
INNER_C = (0.01, 0.1, 1.0)
INNER_K = (12, 24)
GROUPS = (
    "doppler_perfusion_paired_lr",
    "doppler_perfusion_quarter_plus_instance_reference",
    "doppler_perfusion_half_plus_instance_reference",
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_reference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference as load

    patients, kidneys, images = load()
    images = images[images.kidney_APN.notna()].reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].reset_index(drop=True)
    assert (len(patients), len(kidneys), len(images)) == (210, 414, 820)
    return patients, kidneys, images


def doppler_masks(bgr: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Return mutually exclusive, conservative colour masks inside the kidney.

    Green/cyan is deliberately not called perfusion because it is predominantly
    a supplied contour/UI overlay in this source.  A 5x5 erosion protects
    against the contour itself even where it touches the kidney boundary.
    """
    core = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    hue, saturation, value = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    active = core & (saturation >= 80) & (value >= 80)
    red_orange = active & ((hue <= 40) | (hue >= 170))
    blue = active & (hue >= 100) & (hue <= 140)
    cyan_green = active & ~(red_orange | blue)
    return {
        "core": core,
        "red_orange": red_orange,
        "blue": blue,
        "flow": red_orange | blue,
        "cyan_green": cyan_green,
        "value": value,
    }


def image_features(bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    masks = doppler_masks(bgr, mask)
    core, flow = masks["core"], masks["flow"]
    pixels = max(int(core.sum()), 1)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(flow.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if components > 1 else np.empty(0, dtype=int)
    yy, xx = np.nonzero(flow)
    height, width = core.shape
    return {
        "flow_fraction": float(flow.sum() / pixels),
        "red_orange_fraction": float(masks["red_orange"].sum() / pixels),
        "blue_fraction": float(masks["blue"].sum() / pixels),
        "cyan_green_fraction": float(masks["cyan_green"].sum() / pixels),
        "flow_component_count": float(len(areas)),
        "flow_largest_fraction": float(areas.max() / pixels) if len(areas) else 0.0,
        "flow_mean_value": float(masks["value"][flow].mean() / 255.0) if flow.any() else 0.0,
        "flow_center_y": float(yy.mean() / max(height - 1, 1)) if len(yy) else 0.5,
        "flow_center_x": float(xx.mean() / max(width - 1, 1)) if len(xx) else 0.5,
        "flow_spread_y": float(yy.std() / max(height - 1, 1)) if len(yy) else 0.0,
        "flow_spread_x": float(xx.std() / max(width - 1, 1)) if len(xx) else 0.0,
        "flow_present": float(flow.any()),
    }


def image_feature_table(images: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in images.itertuples(index=False):
        bgr = cv2.imread(str(BASE / "clean_rgb" / f"{row.image_id}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(BASE / "masks" / f"{row.image_id}.png"), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            raise FileNotFoundError(row.image_id)
        item = image_features(bgr, mask > 0)
        rows.append({"image_id": row.image_id, "kidney_alias": row.kidney_alias, **item})
    frame = pd.DataFrame(rows)
    assert len(frame) == len(images) and frame.image_id.is_unique
    return frame


def kidney_matrix(kidneys: pd.DataFrame, images: pd.DataFrame, values: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    numeric = [column for column in values if column not in {"image_id", "kidney_alias"}]
    indexed = values.set_index("image_id").reindex(images.image_id)
    assert indexed[numeric].notna().all().all()
    summary, names = [], []
    for reducer in ("mean", "max", "std"):
        names.extend([f"{column}_{reducer}" for column in numeric])
    groups = {alias: np.flatnonzero(images.kidney_alias.eq(alias)) for alias in kidneys.kidney_alias}
    for alias in kidneys.kidney_alias:
        block = indexed.iloc[groups[alias]][numeric].to_numpy(float)
        summary.append(np.concatenate([block.mean(0), block.max(0), block.std(0)]))
    direct = np.vstack(summary)
    # The five differences below are interpretable bilateral perfusion contrasts;
    # no opposite-side diagnosis enters the predictor.
    bilateral = ["flow_fraction_mean", "red_orange_fraction_mean", "blue_fraction_mean",
                 "flow_largest_fraction_mean", "flow_present_mean"]
    selected = [names.index(column) for column in bilateral]
    paired = np.zeros((len(kidneys), len(selected) + 1), float)
    for _, rows in kidneys.groupby("patient_alias", sort=False):
        if len(rows) == 2:
            first, second = rows.index.to_numpy()
            paired[first, :-1] = direct[first, selected] - direct[second, selected]
            paired[second, :-1] = direct[second, selected] - direct[first, selected]
        else:
            paired[rows.index, -1] = 1.0
    return np.column_stack([direct, paired]), names + [f"contralateral_delta_{name}" for name in bilateral] + ["contralateral_missing"]


def make_head(k: int, c: float, n_features: int):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        SelectKBest(f_classif, k=min(k, n_features)),
        LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=4000),
    )


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered Doppler-perfusion protocol")
    inputs = [
        Path(__file__), BASE / "run_lock.json", BASE / "extraction_audit.json",
        BASE / "patient_assignments.csv", BASE / "instance_development_predictions.csv",
        ROOT / "02_experiments/NCKUH_DMSA_label_audit_20261102/image_labels_dmsa_private.csv",
    ]
    write_json(OUT / "protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "210 patients, 414 imaged kidneys, 820 contour-first RGB ultrasound images; unchanged three 126/42/42 patient draws",
        "motivation": "Direct review found intrarenal red/orange/blue Doppler pixels in the actual contour-first RGB inputs. These are distinct from cyan/green supplied contours and are not represented by grayscale normalization.",
        "colour_definition": "HSV saturation>=80 and value>=80 within a 5x5-eroded kidney mask; red/orange hue<=40 or>=170 and blue hue100-140. Green/cyan is excluded from flow and retained only as an audit feature.",
        "features": "Image-level flow occupancy, red/orange and blue occupancy, component extent, brightness and spatial distribution; kidney mean/max/SD plus five same-patient contralateral flow differences.",
        "model_selection": "A 100/26 R/L/B/N-stratified patient split inside each outer 126-patient training role selects C in [.01,.1,1] and ANOVA feature count in [12,24] by mean(inner AUROC, AP). The chosen head is reset and refit on all 126 training patients.",
        "evaluation": "One kidney score per image bag on the fixed 42-patient development role only; no exploratory-test scoring.",
        "fusion": "Prespecified 25% and 50% equal-scale probability contributions from the Doppler head with the unchanged RadImageNet image-instance plus patient-direct reference; no development-derived fusion weight.",
        "target": "Development AUROC>0.8 and AP>0.8 followed by separate locked confirmation.",
        "prohibited": "No diagnosis text, DMSA field, filename suffix, patient identifier, development label, or exploratory-test score enters image feature extraction or hyperparameter fitting.",
    })
    write_json(OUT / "run_lock.json", {
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs},
    })
    print("Registered Doppler-perfusion development-only protocol", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for path, expected in lock["inputs"].items():
        assert digest(Path(path)) == expected, path


def run() -> None:
    from sklearn.model_selection import train_test_split

    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed Doppler-perfusion predictions")
    patients, kidneys, images = load_reference()
    image_values = image_feature_table(images)
    image_values.to_csv(OUT / "deidentified_image_doppler_features.csv", index=False)
    values, names = kidney_matrix(kidneys, images, image_values)
    assert values.shape == (414, len(names)) and np.isfinite(values).all()
    assignments = pd.read_csv(BASE / "patient_assignments.csv")
    reference = pd.read_csv(BASE / "instance_development_predictions.csv")
    reference = reference[reference.group.eq(REFERENCE_GROUP)].set_index(["seed", "kidney_alias"]).score
    labels = kidneys.kidney_APN.to_numpy(int)
    metrics, predictions, selections = [], [], []
    for seed in SEEDS:
        roles = kidneys.patient_alias.map(assignments[assignments.seed.eq(seed)].set_index("patient_alias").role)
        train = np.flatnonzero(roles.eq("train"))
        development = np.flatnonzero(roles.eq("development"))
        train_patients = patients[patients.patient_alias.isin(kidneys.patient_alias.iloc[train])].reset_index(drop=True)
        fit_rows, inner_rows = train_test_split(
            np.arange(len(train_patients)), train_size=100, random_state=seed + 61103,
            stratify=train_patients.DMSA_APN,
        )
        fit_alias = set(train_patients.patient_alias.iloc[fit_rows])
        inner_alias = set(train_patients.patient_alias.iloc[inner_rows])
        fit = np.flatnonzero(kidneys.patient_alias.isin(fit_alias))
        inner = np.flatnonzero(kidneys.patient_alias.isin(inner_alias))
        candidates = []
        for c in INNER_C:
            for k in INNER_K:
                head = make_head(k, c, values.shape[1]).fit(values[fit], labels[fit])
                score = head.predict_proba(values[inner])[:, 1]
                result = metric(labels[inner], score)
                candidates.append((-(result["auroc"] + result["ap"]) / 2, k, c, result))
        _, chosen_k, chosen_c, inner_result = min(candidates, key=lambda row: (row[0], row[1], row[2]))
        head = make_head(chosen_k, chosen_c, values.shape[1]).fit(values[train], labels[train])
        score = head.predict_proba(values[development])[:, 1]
        ref = np.asarray([reference.loc[(seed, alias)] for alias in kidneys.kidney_alias.iloc[development]])
        scores = {
            GROUPS[0]: score,
            GROUPS[1]: score * .25 + ref * .75,
            GROUPS[2]: score * .50 + ref * .50,
        }
        selected = head.named_steps["selectkbest"].get_support()
        selections.append({
            "seed": seed, "chosen_c": chosen_c, "chosen_k": chosen_k,
            "inner_auroc": inner_result["auroc"], "inner_ap": inner_result["ap"],
            "train_patients": len(train_patients), "fit_patients": len(fit_alias), "inner_patients": len(inner_alias),
            "selected_features": json.dumps([name for name, keep in zip(names, selected) if keep]),
        })
        for group, candidate in scores.items():
            metrics.append({"seed": seed, "role": "development", "group": group,
                            "patients": 42, "kidneys": len(development), **metric(labels[development], candidate)})
            predictions.extend({
                "seed": seed, "group": group, "patient_alias": kidneys.patient_alias.iloc[index],
                "kidney_alias": kidneys.kidney_alias.iloc[index], "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                "side": kidneys.side.iloc[index], "label": int(labels[index]), "score": float(candidate[local]),
            } for local, index in enumerate(development))
        print(f"Completed Doppler-perfusion development draw {seed}: C={chosen_c}, k={chosen_k}", flush=True)
    pd.DataFrame(metrics).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "inner_selection.csv", index=False)
    summary = pd.DataFrame(metrics).groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    flow = image_values[["flow_fraction", "red_orange_fraction", "blue_fraction", "flow_present"]]
    write_json(OUT / "development_decision.json", {
        "groups_above_both_development_targets": summary[(summary.auroc_mean > .8) & (summary.ap_mean > .8)].index.tolist(),
        "exploratory_test_scored": False,
        "flow_images": int((flow.flow_present > 0).sum()),
        "flow_image_fraction": float(flow.flow_present.mean()),
        "median_intrarenal_flow_fraction": float(flow.flow_fraction.median()),
        "note": "Absence of Doppler pixels is a measured acquisition property, not an imputed negative disease label.",
    })
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    globals()[parser.parse_args().stage]()
