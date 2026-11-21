"""Lock revised NCKUH patient roles and contour-first image references."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import pandas as pd
from openpyxl import load_workbook
from sklearn.model_selection import train_test_split

from nckuh_dmsa_fullcohort_fullframe_20261103 import (
    normalize_patient_id, parse_source_key, polygon_index, split_panel,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "2004-2019-NCKUH-亞大.xlsx"
IMAGE_ROOT = ROOT / "03_data_models/ki-data"
OUT = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
FOLDERS = ("1", "2", "3", "NCKUH3", "test_predict")
EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
PANEL_FOLDERS = {"NCKUH3", "test_predict"}
SEED = 20261111


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    targets = [OUT / name for name in (
        "patient_linkage_private.csv", "patient_assignments.csv",
        "source_linkage_private.csv", "contour_image_manifest.csv",
        "contour_kidney_manifest.csv", "cohort_lock.json",
    )]
    if any(path.exists() for path in targets):
        raise FileExistsError("Revised cohort registration already exists; preserve it")
    ws = load_workbook(SOURCE, read_only=True, data_only=True)["完整資料彙整"]
    patients = {}
    for row in list(ws.iter_rows(values_only=True))[1:]:
        if row[0] is None:
            continue
        patient_id = normalize_patient_id(str(row[0]).strip())
        assert patient_id not in patients
        code = str(row[5]).strip().upper()
        assert code in {"R", "L", "B", "N"}
        folders = {x.strip() for x in str(row[6]).split(",")}
        assert folders and folders <= set(FOLDERS)
        patients[patient_id] = (code, folders)
    assert len(patients) == 383
    ids = sorted(patients)
    aliases = {patient_id: f"P{i:03d}" for i, patient_id in enumerate(ids, 1)}
    states = [patients[patient_id][0] for patient_id in ids]
    train, remainder = train_test_split(list(range(len(ids))), train_size=230,
                                         stratify=states, random_state=SEED)
    dev, test = train_test_split(remainder, train_size=76,
                                 stratify=[states[i] for i in remainder], random_state=SEED)
    roles = {ids[i]: role for role, indices in (("train", train), ("development", dev),
                                               ("exploratory_test", test)) for i in indices}
    assert Counter(roles.values()) == {"train": 230, "development": 76, "exploratory_test": 77}
    pd.DataFrame([{"patient_id": patient_id, "patient_alias": aliases[patient_id]}
                  for patient_id in ids]).to_csv(OUT / "patient_linkage_private.csv", index=False)
    pd.DataFrame([{"patient_alias": aliases[patient_id], "DMSA_APN": patients[patient_id][0],
                   "role": roles[patient_id]} for patient_id in ids]).to_csv(
                       OUT / "patient_assignments.csv", index=False)

    raw = defaultdict(list)
    for folder in FOLDERS:
        for path in (IMAGE_ROOT / folder).iterdir():
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            patient_id = normalize_patient_id(path.name.split("_", 1)[0])
            if patient_id not in patients or folder not in patients[patient_id][1]:
                continue
            key = parse_source_key(path)
            if key is not None and key[1] is not None:
                raw[key].append((folder, path))
    polygons = polygon_index()
    private_rows, public_rows = [], []
    for key in sorted(raw):
        patient_id, date, view = key
        panels = sorted((path for folder, path in raw[key] if folder in PANEL_FOLDERS),
                        key=lambda path: ("test_predict" not in path.parts, str(path)))
        selected, source_type, annotation = None, None, None
        for path in panels:
            try:
                split_panel(path)
                selected, source_type = path, "panel_with_mask"
                break
            except (ValueError, cv2.error):
                continue
        if selected is None and key in polygons:
            selected, annotation = polygons[key]
            source_type = "labelme_polygon"
        if selected is None:
            continue
        image_id = f"I{len(public_rows) + 1:04d}"
        side = "left" if view.startswith("L") else "right"
        code = patients[patient_id][0]
        kidney_apn = int(code == "B" or side[0].upper() in code)
        public_rows.append({"image_id": image_id, "patient_alias": aliases[patient_id],
                            "kidney_alias": aliases[patient_id] + "_" + side[0].upper(),
                            "side": side, "view": view, "source_type": source_type,
                            "kidney_APN": kidney_apn})
        private_rows.append({"image_id": image_id, "patient_id": patient_id,
                             "exam_date_raw": date, "source_path": str(selected),
                             "annotation_path": "" if annotation is None else str(annotation)})
    public = pd.DataFrame(public_rows)
    assert len(public) == 1362 and public.patient_alias.nunique() == 347
    public.to_csv(OUT / "contour_image_manifest.csv", index=False)
    pd.DataFrame(private_rows).to_csv(OUT / "source_linkage_private.csv", index=False)
    kidneys = public.drop_duplicates("kidney_alias")[
        ["kidney_alias", "patient_alias", "side", "kidney_APN"]].reset_index(drop=True)
    assert len(kidneys) == 686 and kidneys.kidney_APN.sum() == 418
    kidneys.to_csv(OUT / "contour_kidney_manifest.csv", index=False)
    observed_roles = public.merge(pd.read_csv(OUT / "patient_assignments.csv"), on="patient_alias")
    lock = {
        "source_workbook_sha256": digest(SOURCE), "script_sha256": digest(Path(__file__)),
        "seed": SEED, "split": "230 train / 76 development / 77 exploratory test, stratified by R/L/B/N at patient level",
        "all_patients": 383, "contour_ready_patients": int(public.patient_alias.nunique()),
        "contour_ready_kidneys": len(kidneys), "contour_ready_images": len(public),
        "contour_ready_images_by_role": observed_roles.role.value_counts().to_dict(),
        "contour_ready_patients_by_role": observed_roles.groupby("role").patient_alias.nunique().to_dict(),
        "test_policy": "Do not use exploratory-test labels or scores for model/feature/probe selection",
        "files_sha256": {path.name: digest(path) for path in targets[:-1]},
    }
    (OUT / "cohort_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in lock.items() if k not in ("files_sha256", "script_sha256")}, indent=2))


if __name__ == "__main__":
    main()
