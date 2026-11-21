"""Recover DMSA labels for image-only NCKUH patients and extract frozen features.

The 2026-09 audit only used the '成大264位' flag column of the root workbook.
Unflagged rows of the 2007-2012 and 2014-2019 sheets carry DMSA_APN reads for
137 patients whose images sit in ki-data but who were never added to the cohort.

Inclusion: DMSA_APN present; ultrasound exam date within 14 days of the sheet's
admission date; four-view aligned image/mask panel available (same source rule as
the 210-patient full-cohort pipeline). Filename diagnosis suffixes are never read.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_native_tile_probe as tilemod

ROOT = base.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_recovered_labels_20261104"
WORKBOOK = ROOT / "2004-2019-NCKUH-亞大-20210723.xlsx"
NEW201 = ROOT / "04_quality_archive/scratch_kidata/new_patients_201.csv"
CODE = {"B": "B", "L": "L", "R": "R", "N": "N", "bil": "B", "l": "L", "r": "R", "n": "N"}
SIDE = {"LL": "left", "LT": "left", "RL": "right", "RT": "right"}


def recovered_patients() -> pd.DataFrame:
    new = pd.read_csv(NEW201, dtype=str).set_index("pid8")
    book = pd.ExcelFile(WORKBOOK)
    rows = []
    for sheet in ("2007-2012", "2014-2019"):
        frame = book.parse(sheet, dtype=str)
        adm_col = "admission_date" if "admission_date" in frame else frame.columns[4]
        frame["id"] = frame["id"].str.strip()
        for _, row in frame[frame["id"].isin(new.index)].iterrows():
            image_date = pd.to_datetime(new.loc[row["id"], "date_min"], format="%Y%m%d")
            admission = pd.to_datetime(row[adm_col], errors="coerce")
            rows.append({
                "patient_id": row["id"], "sheet": sheet,
                "DMSA_raw": row["DMSA_APN"], "DMSA_APN": CODE.get(str(row["DMSA_APN"]).strip()),
                "exam_date_raw": new.loc[row["id"], "date_min"],
                "gap_days": (image_date - admission).days if pd.notna(admission) else np.nan,
                "birth": row["birth"], "gender": row["gender"],
            })
    out = pd.DataFrame(rows)
    out["included_label_date"] = out.DMSA_APN.notna() & out.gap_days.abs().le(14)
    return out


def image_plan(patients: pd.DataFrame) -> pd.DataFrame:
    exact, _ = base.panel_index()
    records = []
    for row in patients[patients.included_label_date].itertuples(index=False):
        pid = base.normalize_patient_id(row.patient_id)
        found = {view: exact.get((pid, row.exam_date_raw, view), []) for view in SIDE}
        if not all(found.values()):
            continue
        for view, candidates in found.items():
            records.append({"patient_id": pid, "view": view, "side": SIDE[view],
                            "DMSA_APN": row.DMSA_APN, "exam_date_raw": row.exam_date_raw,
                            "source_path_private": str(candidates[0])})
    plan = pd.DataFrame(records)
    plan["image_id"] = [f"N{i:04d}" for i in range(1, len(plan) + 1)]
    aliases = {pid: f"Q{i:03d}" for i, pid in enumerate(plan.patient_id.drop_duplicates(), 1)}
    plan["patient_alias"] = plan.patient_id.map(aliases)
    plan["kidney_alias"] = plan.patient_alias + "_" + plan.side.str[0].str.upper()
    apn = plan.DMSA_APN
    plan["kidney_APN"] = ((apn == "B") | ((apn == "L") & (plan.side == "left")) | ((apn == "R") & (plan.side == "right"))).astype(int)
    return plan


def extract(plan: pd.DataFrame) -> None:
    for name in ("clean_rgb", "masks", "percentile"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    for row in plan.itertuples(index=False):
        image, mask = base.split_panel(Path(row.source_path_private))
        roi, roi_mask = base.crop_masked(image, mask)
        cv2.imwrite(str(OUT / "clean_rgb" / f"{row.image_id}.png"), roi)
        cv2.imwrite(str(OUT / "masks" / f"{row.image_id}.png"), roi_mask * 255)
        cv2.imwrite(str(OUT / "percentile" / f"{row.image_id}.png"), base.percentile_gray(roi, roi_mask))


def encode(root: Path, ids: list[str]) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(base.RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    rad = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    gray_rows, color_rows, tile_rows = [], [], []
    with torch.inference_mode():
        for ident in ids:
            color = cv2.cvtColor(cv2.imread(str(root / "clean_rgb" / f"{ident}.png")), cv2.COLOR_BGR2RGB)
            gray = cv2.imread(str(root / "percentile" / f"{ident}.png"), 0)
            mask = cv2.imread(str(root / "masks" / f"{ident}.png"), 0) > 0
            canvases = [base.tight_canvas(gray, mask)[0], base.tight_canvas(color, mask)[0]]
            tiles, _ = tilemod.tiles(color, mask)
            x = torch.from_numpy(np.stack(canvases + tiles).transpose(0, 3, 1, 2).copy()).cuda().float() / 127.5 - 1
            values = rad(x).cpu().numpy()
            gray_rows.append(values[0]); color_rows.append(values[1]); tile_rows.append(values[2:].mean(0))
    return {k: np.asarray(v, np.float32) for k, v in
            (("rad_gray", gray_rows), ("rad_color", color_rows), ("tile_mean", tile_rows))}


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261104)
    patients = recovered_patients()
    patients.drop(columns=["birth"]).to_csv(OUT / "recovered_patients_private.csv", index=False)
    plan = image_plan(patients)
    plan.to_csv(OUT / "recovered_images_private.csv", index=False)
    extract(plan)
    new = encode(OUT, plan.image_id.tolist())
    np.savez(OUT / "recovered_features.npz", **new)
    # Reproduction check on the first 24 existing images
    ref_ids = base.labeled_images().image_id.tolist()
    check = encode(base.OUT, ref_ids[:24])
    stored = {"rad_gray": np.load(base.OUT / "radimagenet_gray.npy")[:24],
              "rad_color": np.load(base.OUT / "radimagenet_color.npy")[:24],
              "tile_mean": np.load(tilemod.OUT / "tile_features.npz")["mean"][:24]}
    diff = {k: float(np.abs(check[k] - stored[k]).max()) for k in stored}
    (OUT / "audit.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workbook_matches": int(len(patients)),
        "label_and_date_included": int(patients.included_label_date.sum()),
        "four_view_panel_patients": int(plan.patient_alias.nunique()),
        "images": int(len(plan)),
        "state_counts": plan.drop_duplicates("patient_alias").DMSA_APN.value_counts().to_dict(),
        "reproduction_maxabs_vs_existing_cache": diff,
        "filename_suffix_used": False,
    }, indent=2), encoding="utf-8")
    print(json.dumps(diff), plan.patient_alias.nunique(), len(plan), flush=True)


if __name__ == "__main__":
    run()


# --- Stage 2: recovered patients with raw TIF only (red outline or LabelMe polygon) ---
INVENTORY = ROOT / "04_quality_archive/scratch_kidata/kidata_image_inventory.csv"


def red_outline(image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    b, g, r = (image[..., i].astype(int) for i in range(3))
    red = ((r > 150) & (g < 90) & (b < 90)).astype(np.uint8)
    if red.sum() < 500:
        return None
    closed = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(red.shape, np.uint8)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)
    if mask.sum() < 2000:
        return None
    clean = cv2.inpaint(image, cv2.dilate(red, np.ones((3, 3), np.uint8)), 3, cv2.INPAINT_TELEA)
    return clean, mask


def run_stage2() -> None:
    patients = recovered_patients()
    done = pd.read_csv(OUT / "recovered_images_private.csv", dtype=str)
    patients = patients[patients.included_label_date].copy()
    patients["pid"] = patients.patient_id.map(base.normalize_patient_id)
    patients = patients[~patients.pid.isin(done.patient_id)]
    inv = pd.read_csv(INVENTORY, dtype=str).dropna(subset=["pid"])
    inv["pid"] = inv.pid.map(base.normalize_patient_id)
    records = []
    for row in patients.itertuples(index=False):
        for item in inv[inv.pid.eq(row.pid) & inv.date.eq(row.exam_date_raw)].itertuples(index=False):
            path = ROOT / "03_data_models/ki-data" / item.folder / item.name
            json_path = path.with_suffix(".json")
            if json_path.exists():
                image, mask = base.load_polygon(path, json_path)
                found = red_outline(image)
                if found is not None:
                    image = found[0]
                source = "labelme_polygon"
            else:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                found = red_outline(image)
                if found is None:
                    continue
                image, mask = found
                source = "red_outline_inpainted"
            records.append({"patient_id": row.pid, "view": item.view, "side": SIDE[item.view],
                            "DMSA_APN": row.DMSA_APN, "exam_date_raw": row.exam_date_raw,
                            "source_type": source, "source_path_private": str(path), "img_arr": image, "mask_arr": mask})
    plan = pd.DataFrame(records)
    sides = plan.groupby("patient_id").side.nunique()
    plan = plan[plan.patient_id.isin(sides[sides.eq(2)].index)].reset_index(drop=True)
    plan["image_id"] = [f"M{i:04d}" for i in range(1, len(plan) + 1)]
    aliases = {pid: f"T{i:03d}" for i, pid in enumerate(plan.patient_id.drop_duplicates(), 1)}
    plan["patient_alias"] = plan.patient_id.map(aliases)
    plan["kidney_alias"] = plan.patient_alias + "_" + plan.side.str[0].str.upper()
    apn = plan.DMSA_APN
    plan["kidney_APN"] = ((apn == "B") | ((apn == "L") & (plan.side == "left")) | ((apn == "R") & (plan.side == "right"))).astype(int)
    for row in plan.itertuples(index=False):
        roi, roi_mask = base.crop_masked(row.img_arr, row.mask_arr)
        cv2.imwrite(str(OUT / "clean_rgb" / f"{row.image_id}.png"), roi)
        cv2.imwrite(str(OUT / "masks" / f"{row.image_id}.png"), roi_mask * 255)
        cv2.imwrite(str(OUT / "percentile" / f"{row.image_id}.png"), base.percentile_gray(roi, roi_mask))
    plan = plan.drop(columns=["img_arr", "mask_arr"])
    plan.to_csv(OUT / "recovered_images_stage2_private.csv", index=False)
    np.savez(OUT / "recovered_features_stage2.npz", **encode(OUT, plan.image_id.tolist()))
    print(plan.patient_alias.nunique(), len(plan), plan.source_type.value_counts().to_dict(),
          plan.drop_duplicates("patient_alias").DMSA_APN.value_counts().to_dict(), flush=True)
