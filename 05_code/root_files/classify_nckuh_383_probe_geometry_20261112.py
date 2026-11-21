"""Patient-grouped validation of probe type from ultrasound-sector geometry."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from nckuh_probe_overlay_ocr_20261111 import source_images
from nckuh_dmsa_fullcohort_fullframe_20261103 import normalize_patient_id, split_panel
from nckuh_dmsa_fullframe_relative_geometry_probe_20261102 import ultrasound_sector


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
OUT = ROOT / "02_experiments/NCKUH_revised_383_probe_geometry_20261112"
SEED = 20261112


def sector_features(path):
    frame, kidney = split_panel(path)
    sector, fallback = ultrasound_sector(frame, kidney)
    yy, xx = np.nonzero(sector)
    h, w = sector.shape
    rows = np.linspace(0, h, 17, dtype=int)
    widths, centers = [], []
    for a, b in zip(rows[:-1], rows[1:]):
        selected = sector[a:b]
        sy, sx = np.nonzero(selected)
        if len(sx):
            widths.append((sx.max() - sx.min() + 1) / w)
            centers.append((sx.mean() / w) - .5)
        else:
            widths.append(0.0); centers.append(0.0)
    return np.asarray([
        h / w, sector.mean(), (xx.max() - xx.min() + 1) / w,
        (yy.max() - yy.min() + 1) / h,
        np.mean(widths[:4]), np.mean(widths[6:10]), np.mean(widths[-4:]),
        np.std(widths), np.max(widths) - np.min([x for x in widths if x > 0] or [0]),
        np.std(centers), float(fallback), *widths,
    ], dtype=np.float32)


def model():
    return RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=4,
                                  class_weight="balanced", random_state=SEED, n_jobs=4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "validation.json").exists():
        raise FileExistsError("Preserve probe geometry validation")
    ocr = json.loads((AUDIT / "probe_overlay_ocr_full_test_predict.json").read_text())
    paths = source_images()["test_predict"]
    records = []
    for row in ocr["deidentified_records"]:
        path = paths[row["file_ordinal"]]
        if row["probe"] not in ("convex", "linear"):
            continue
        records.append((path, row["probe"], normalize_patient_id(path.name.split("_", 1)[0]),
                        sector_features(path)))
    x = np.stack([row[3] for row in records])
    y = np.asarray([row[1] == "convex" for row in records], dtype=int)
    groups = np.asarray([row[2] for row in records])
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    folds = []
    for fold, (train, hold) in enumerate(splitter.split(x, y, groups)):
        fitted = model().fit(x[train], y[train])
        oof[hold] = fitted.predict_proba(x[hold])[:, 1]
        folds.append({"fold": fold, "patients": len(set(groups[hold])), "images": len(hold),
                      "auroc": roc_auc_score(y[hold], oof[hold]),
                      "balanced_accuracy": balanced_accuracy_score(y[hold], oof[hold] >= .5)})
    matrix = confusion_matrix(y, oof >= .5, labels=[0, 1])
    fitted = model().fit(x, y)
    summary = {"ocr_labeled_images": len(y), "ocr_labeled_patients": len(set(groups)),
               "convex_images": int(y.sum()), "linear_images": int((1-y).sum()),
               "patient_grouped_oof_auroc": roc_auc_score(y, oof),
               "patient_grouped_oof_balanced_accuracy": balanced_accuracy_score(y, oof >= .5),
               "confusion_linear_convex": matrix.tolist(), "folds": folds,
               "gate": "Do not use inferred probe labels unless grouped OOF AUROC and balanced accuracy are both >=0.90",
               "ocr_caveat": "Training labels are OCR overlay tokens, not manufacturer metadata."}
    (OUT / "validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame({"probe": np.where(y, "convex", "linear"), "oof_convex_probability": oof,
                  "patient_group": pd.factorize(groups)[0]}).to_csv(OUT / "validation_predictions_deidentified.csv", index=False)
    print(json.dumps(summary, indent=2))
    if summary["patient_grouped_oof_auroc"] < .9 or summary["patient_grouped_oof_balanced_accuracy"] < .9:
        return
    source = pd.read_csv(AUDIT / "source_linkage_private.csv", dtype=str)
    path_to_id = {str(Path(row.source_path)): row.image_id for row in source.itertuples(index=False)}
    known = {str(paths[row["file_ordinal"]]): row["probe"] for row in ocr["deidentified_records"]
             if row["probe"] in ("convex", "linear")}
    output = []
    for path in paths:
        image_id = path_to_id.get(str(path))
        if image_id is None:
            continue
        if str(path) in known:
            output.append({"image_id": image_id, "probe": known[str(path)],
                           "evidence": "OCR", "convex_probability": float(known[str(path)] == "convex")})
        else:
            probability = fitted.predict_proba(sector_features(path)[None])[0, 1]
            label = "convex" if probability >= .9 else "linear" if probability <= .1 else "unknown"
            output.append({"image_id": image_id, "probe": label, "evidence": "sector_geometry",
                           "convex_probability": probability})
    pd.DataFrame(output).to_csv(OUT / "registered_image_probe_labels.csv", index=False)


if __name__ == "__main__":
    main()
