"""
Validate the returned per-kidney DMSA table and turn it into kidney-level labels.

Input  kidney_label_request/kidney_label_request.xlsx (or its .csv) after the clinical
       team has filled it.  Built by build_kidney_label_template.py.

Output, in --out-dir (default kidney_labels/)
  kidney_labels_validated.csv   one row per exam x side: patient_id, us_exam_date, side, dmsa
  exam_labels.csv               one row per exam, with the patient label implied by the two kidneys
  label_conflicts.csv           analysed exams where the two kidneys disagree with the stored
                                patient label -- these need a clinician's second look
  kidney_manifest.csv           the image manifest with side, kidney_label and kidney_unit added
  new_patient_images.csv        source images of newly labelled patients, still to be cropped
  summary.json, summary.txt     (validation_errors.txt instead, if the table has bad values)

What the headline numbers mean
  unilateral rate     Among exams positive on DMSA with both kidneys read, the share with only
                      one positive kidney.  It replaces the 20-30 percent label-noise figure
                      that has so far only been an estimate.
  images relabelled   Images of a kidney DMSA read as normal that currently carry a positive
                      patient label.  This is the label noise the per-kidney table removes.

Splitting rule for anything built on kidney_manifest.csv: SPLIT BY PATIENT, SCORE BY KIDNEY.
Two kidneys of one child share the child, the scanner, the operator and the session, so one in
training and the other in test is leakage.  kidney_unit is an analysis unit, never a grouping
key.  single_center_cv.py still asserts patient-constant labels and needs a kidney-unit mode
before it can consume this manifest.

Usage
  python apply_kidney_labels.py --labels kidney_label_request/kidney_label_request.xlsx
  python apply_kidney_labels.py --selftest        # plumbing check on synthetic labels
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_REQUEST = ROOT / "kidney_label_request" / "kidney_label_request.xlsx"
DEFAULT_MANIFEST = ROOT / "yolo_crops" / "manifest.csv"
COHORT_CSV = ROOT / "patient_characteristics" / "used_ultrasound_image_filenames.csv"
KI_IMAGES_CSV = ROOT / "scratch_kidata" / "ki_images_parsed.csv"
KI_ROOT = ROOT / "ki-data"
MAX_DMSA_GAP_DAYS = 14
REQUIRED = {"patient_id", "us_exam_date", "site", "cohort_status",
            "dmsa_date", "left_dmsa", "right_dmsa"}


def pid8(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    m = re.match(r"\s*0*(\d+)", str(value))
    return m.group(1).zfill(8) if m else None


def norm_date(raw) -> str | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = re.sub(r"\D", "", str(raw))
    if len(s) == 9:
        s = s[:8]          # the one malformed cohort date, 201208087 -> 20120808
    if len(s) != 8 or pd.isna(pd.to_datetime(s, format="%Y%m%d", errors="coerce")):
        return None
    return s


def coerce_result(value) -> tuple[float, str]:
    """-> (0.0 | 1.0 | nan, state) with state in value / na / blank / invalid."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, "blank"
    s = str(value).strip().upper()
    if s == "":
        return np.nan, "blank"
    if s in ("0", "0.0"):
        return 0.0, "value"
    if s in ("1", "1.0"):
        return 1.0, "value"
    if s in ("NA", "N/A"):
        return np.nan, "na"
    return np.nan, "invalid"


def read_table(path: Path, sheet: str) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [str(c).strip().split(" ", 1)[0] for c in df.columns]
    missing = REQUIRED - set(df.columns)
    if missing:
        raise SystemExit(f"label table is missing columns: {sorted(missing)}")
    return df


def exam_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    recs, errors = [], []
    for i, r in df.reset_index(drop=True).iterrows():
        line = i + 2                                   # spreadsheet row, header is row 1
        p, d = pid8(r["patient_id"]), norm_date(r["us_exam_date"])
        if not p or not d:
            errors.append(f"row {line}: patient_id or us_exam_date unreadable")
            continue
        site = "" if pd.isna(r["site"]) else str(r["site"]).strip().upper()
        rec = {"patient_id": p, "us_exam_date": d, "site": site,
               "cohort_status": str(r["cohort_status"] or "").split(" ", 1)[0]}
        for side, col in (("L", "left_dmsa"), ("R", "right_dmsa")):
            val, state = coerce_result(r[col])
            if state == "invalid":
                errors.append(f"row {line} {col}: {r[col]!r} is not 0, 1 or NA")
            rec[f"dmsa_{side}"], rec[f"state_{side}"] = val, state
        dd = norm_date(r["dmsa_date"])
        rec["dmsa_date"] = dd or ""
        rec["dmsa_gap_days"] = ((pd.to_datetime(dd, format="%Y%m%d")
                                 - pd.to_datetime(d, format="%Y%m%d")).days if dd else np.nan)
        recs.append(rec)
    ex = pd.DataFrame(recs)
    dup = ex.duplicated(["patient_id", "us_exam_date"], keep=False)
    for _, r in ex[dup].drop_duplicates(["patient_id", "us_exam_date"]).iterrows():
        errors.append(f"exam {r['patient_id']} {r['us_exam_date']} appears more than once")
    return ex, errors


def stored_patient_labels() -> dict:
    coh = pd.read_csv(COHORT_CSV, dtype=str, encoding="utf-8-sig")
    coh["pid8"] = coh["patient_id"].map(pid8)
    return coh.groupby("pid8")["label"].first().astype(float).to_dict()


def run(df: pd.DataFrame, manifest_path: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ex, errors = exam_table(df)
    if errors:
        (out / "validation_errors.txt").write_text("\n".join(errors), encoding="utf-8")
        raise SystemExit(f"{len(errors)} validation error(s); see {out / 'validation_errors.txt'}\n"
                         + "\n".join(errors[:20]))

    # ---- exam level: what the two kidneys imply --------------------------------------
    ex["n_read"] = ex[["dmsa_L", "dmsa_R"]].notna().sum(axis=1)
    any_pos = (ex["dmsa_L"] == 1) | (ex["dmsa_R"] == 1)
    ex["exam_label"] = np.where(any_pos, 1.0, np.where(ex["n_read"] == 2, 0.0, np.nan))
    ex["filled"] = (ex["state_L"] != "blank") | (ex["state_R"] != "blank")

    stored = stored_patient_labels()
    analysed = ex["cohort_status"].str.startswith("analysed")
    ex["stored_patient_label"] = np.where(analysed, ex["patient_id"].map(stored), np.nan)
    conflict = (analysed & ex["exam_label"].notna() & ex["stored_patient_label"].notna()
                & (ex["exam_label"] != ex["stored_patient_label"]))
    ex["conflict"] = np.where(~conflict, "",
                              np.where(ex["stored_patient_label"] == 1,
                                       "stored positive, DMSA both kidneys normal",
                                       "stored negative, DMSA shows a positive kidney"))

    both_read_pos = (ex["exam_label"] == 1) & (ex["n_read"] == 2)
    unilateral = int((both_read_pos & ((ex["dmsa_L"] + ex["dmsa_R"]) == 1)).sum())
    bilateral = int((both_read_pos & ((ex["dmsa_L"] + ex["dmsa_R"]) == 2)).sum())
    gap = ex["dmsa_gap_days"].dropna()

    # ---- kidney level --------------------------------------------------------------
    long = pd.concat([
        ex.assign(side="L", dmsa=ex["dmsa_L"], state=ex["state_L"]),
        ex.assign(side="R", dmsa=ex["dmsa_R"], state=ex["state_R"]),
    ])[["patient_id", "us_exam_date", "site", "cohort_status", "side", "dmsa", "state"]]
    long = long.sort_values(["patient_id", "us_exam_date", "side"]).reset_index(drop=True)

    man = pd.read_csv(manifest_path, dtype=str, encoding="utf-8-sig")
    man["_pid8"] = man["patient_id"].map(pid8)
    man["_date"] = man["exam_date_raw"].map(norm_date)
    man["side"] = man["view"].astype(str).str.upper().str[0]
    key = long.rename(columns={"patient_id": "_pid8", "us_exam_date": "_date"})
    man = man.merge(key[["_pid8", "_date", "side", "dmsa"]], on=["_pid8", "_date", "side"],
                    how="left")
    man["kidney_label"] = man["dmsa"].astype("Int64")
    man["kidney_unit"] = man["_pid8"] + "|" + man["_date"].fillna("") + "|" + man["side"]
    man["patient_label_stored"] = man["label"]
    relabelled = int(((man["patient_label_stored"] == "1") & (man["kidney_label"] == 0)).sum())
    man.drop(columns=["dmsa", "_pid8", "_date"]).to_csv(out / "kidney_manifest.csv",
                                                        index=False, encoding="utf-8-sig")

    # ---- newly labelled patients: which source images to crop next ---------------------
    new_ids = set(ex.loc[(ex["cohort_status"] == "new_patient") & ex["exam_label"].notna(),
                         "patient_id"])
    n_new_images = 0
    if new_ids and KI_IMAGES_CSV.exists():
        ki = pd.read_csv(KI_IMAGES_CSV, dtype=str)
        ki["pid8"] = ki["pid8"].map(pid8)
        ki = ki[ki["pid8"].isin(new_ids)].drop_duplicates("name")
        ki["source_path"] = [str(KI_ROOT / r) for r in ki["relpath"]]
        ki.to_csv(out / "new_patient_images.csv", index=False, encoding="utf-8-sig")
        n_new_images = int(len(ki))

    long.to_csv(out / "kidney_labels_validated.csv", index=False, encoding="utf-8-sig")
    ex.to_csv(out / "exam_labels.csv", index=False, encoding="utf-8-sig")
    ex[conflict].to_csv(out / "label_conflicts.csv", index=False, encoding="utf-8-sig")

    summary = {
        "exams_in_table": int(len(ex)),
        "exams_filled": int(ex["filled"].sum()),
        "exams_both_kidneys_read": int((ex["n_read"] == 2).sum()),
        "kidneys_positive": int((long["dmsa"] == 1).sum()),
        "kidneys_negative": int((long["dmsa"] == 0).sum()),
        "kidneys_na": int((long["state"] == "na").sum()),
        "kidneys_blank": int((long["state"] == "blank").sum()),
        "positive_exams_both_read": unilateral + bilateral,
        "unilateral": unilateral,
        "bilateral": bilateral,
        "unilateral_rate": (unilateral / (unilateral + bilateral)) if (unilateral + bilateral) else None,
        "analysed_exams_checked": int((analysed & ex["exam_label"].notna()).sum()),
        "conflicts_with_stored_label": int(conflict.sum()),
        "images_in_manifest": int(len(man)),
        "images_with_kidney_label": int(man["kidney_label"].notna().sum()),
        "positive_label_images_now_negative": relabelled,
        "dmsa_gap_days_median": float(gap.median()) if len(gap) else None,
        "dmsa_gap_over_limit": int((gap.abs() > MAX_DMSA_GAP_DAYS).sum()),
        "new_patients_labelled": len(new_ids),
        "new_patient_source_images": n_new_images,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"{k:<38} {v}" for k, v in summary.items()]
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out}")
    return summary


def synthetic_request(n_unilateral: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every analysed exam gets both kidneys = the stored patient label, then n_unilateral
    positive exams have one randomly chosen kidney set to 0."""
    coh = pd.read_csv(COHORT_CSV, dtype=str, encoding="utf-8-sig")
    coh["pid8"] = coh["patient_id"].map(pid8)
    coh["d"] = coh["exam_date_raw"].map(norm_date)
    ex = coh.groupby(["pid8", "d"]).agg(label=("label", "first"), site=("site", "first")).reset_index()
    ex["left_dmsa"] = ex["label"]
    ex["right_dmsa"] = ex["label"]
    rng = np.random.default_rng(seed)
    pick = rng.choice(ex.index[ex["label"] == "1"].to_numpy(), size=n_unilateral, replace=False)
    zeroed = []
    for idx in pick:
        side = "L" if rng.integers(0, 2) == 0 else "R"
        ex.loc[idx, "left_dmsa" if side == "L" else "right_dmsa"] = "0"
        zeroed.append((ex.loc[idx, "pid8"], ex.loc[idx, "d"], side))
    req = pd.DataFrame({"patient_id": ex["pid8"], "us_exam_date": ex["d"], "site": ex["site"],
                        "cohort_status": "analysed", "dmsa_date": ex["d"],
                        "left_dmsa": ex["left_dmsa"], "right_dmsa": ex["right_dmsa"]})
    return req, pd.DataFrame(zeroed, columns=["pid8", "date", "side"])


def selftest(manifest_path: Path, out: Path) -> int:
    n = 30
    req, zeroed = synthetic_request(n, seed=0)
    s = run(req, manifest_path, out)

    man = pd.read_csv(manifest_path, dtype=str, encoding="utf-8-sig")
    keys = set(zip(man["patient_id"].map(pid8), man["exam_date_raw"].map(norm_date),
                   man["view"].astype(str).str.upper().str[0]))
    expected_relabel = sum(int(((man["patient_id"].map(pid8) == p)
                                & (man["exam_date_raw"].map(norm_date) == d)
                                & (man["view"].astype(str).str.upper().str[0] == sd)).sum())
                           for p, d, sd in zeroed.itertuples(index=False))
    positives = int((req["left_dmsa"].eq("1") | req["right_dmsa"].eq("1")).sum())
    checks = {
        "no conflicts (a unilateral exam still implies a positive patient)":
            s["conflicts_with_stored_label"] == 0,
        f"unilateral count equals the {n} injected": s["unilateral"] == n,
        "bilateral count equals the remaining positive exams": s["bilateral"] == positives - n,
        "relabelled images equal the images of the zeroed kidneys":
            s["positive_label_images_now_negative"] == expected_relabel,
        "every manifest image received a kidney label": s["images_with_kidney_label"] == len(man),
        "manifest keys resolved": len(keys) > 0,
    }
    print("\nSELFTEST")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  expected relabelled images {expected_relabel}, "
          f"got {s['positive_label_images_now_negative']}")
    return 0 if all(checks.values()) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default=str(DEFAULT_REQUEST))
    p.add_argument("--sheet", default="待填表")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--out-dir", default="kidney_labels")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)

    if args.selftest:
        return selftest(Path(args.manifest), ROOT / args.out_dir / "_selftest")
    run(read_table(Path(args.labels), args.sheet), Path(args.manifest), ROOT / args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
