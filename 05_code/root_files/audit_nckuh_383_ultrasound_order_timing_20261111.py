"""Compare ultrasound filename dates with the 6000115 pediatric renal ultrasound order."""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from audit_nckuh_383_clinical_availability_20261111 import patient_key


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
OUT = ROOT / "02_experiments/NCKUH_revised_383_clinical_audit_20261111"


def parse_date(value):
    if value is None:
        return pd.NaT
    return pd.to_datetime(str(value).strip().split()[0][:8], format="%Y%m%d", errors="coerce")


def main():
    book = load_workbook(ROOT / "2004-2019-NCKUH-亞大.xlsx", read_only=True, data_only=True)
    description = list(book["說明"].values)
    assert any("6000115" in str(row[2]) and "超音波" in str(row[3]) for row in description)
    orders = defaultdict(list)
    for row in list(book["2007-2012"].values)[1:]:
        if row[0] is None:
            continue
        date = parse_date(row[6])
        if not pd.isna(date):
            orders[patient_key(row[0])].append(date)
    source = pd.read_csv(REGISTRY / "source_linkage_private.csv", dtype=str)
    public = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    link = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype=str)
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv")
    images = source[["image_id", "exam_date_raw"]].merge(
        public[["image_id", "patient_alias"]], on="image_id", validate="one_to_one").merge(
        link, on="patient_alias", validate="many_to_one").merge(
        roles[["patient_alias", "role"]], on="patient_alias", validate="many_to_one")
    images["key"] = images.patient_id.map(patient_key)
    images = images[images.key.isin(orders)].copy()
    images["image_date"] = images.exam_date_raw.map(parse_date)
    images["order_date"] = images.key.map(lambda key: min(orders[key]))
    images = images[images.image_date.notna()].copy()
    images["days_order_after_image"] = (images.order_date - images.image_date).dt.days
    per_patient = images.groupby("patient_alias").days_order_after_image.agg(["min", "max"])
    flagged = per_patient[(per_patient["min"] < -30) | (per_patient["max"] > 30)].index
    review = images[images.patient_alias.isin(flagged)].groupby(
        ["patient_alias", "role", "order_date"], as_index=False).agg(
        image_dates=("image_date", lambda x: ";".join(sorted(set(x.dt.strftime("%Y-%m-%d"))))),
        min_days_order_after_image=("days_order_after_image", "min"),
        max_days_order_after_image=("days_order_after_image", "max"),
        images=("image_id", "count"))
    gap = images.days_order_after_image
    summary = {
        "code_6000115": "pediatric renal ultrasound 2D, NOT DMSA",
        "patients_with_order_date_and_contour_images": int(images.patient_alias.nunique()),
        "images": len(images),
        "patient_roles": images.groupby("role").patient_alias.nunique().to_dict(),
        "order_minus_image_days": {
            "image_min": int(gap.min()), "image_median": float(gap.median()),
            "image_max": int(gap.max()), "images_same_day": int(gap.eq(0).sum()),
            "images_absolute_gap_over_30_days": int(gap.abs().gt(30).sum()),
            "patients_any_absolute_gap_over_30_days": int(
                ((per_patient["min"] < -30) | (per_patient["max"] > 30)).sum()),
        },
        "dmsa_scan_date_available_from_this_column": False,
        "caveat": "Only dated 2007-2012 rows. The 6000115 date may be an order/exam timestamp and is not a DMSA label timestamp."
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ultrasound_order_timing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    review.to_csv(OUT / "ultrasound_order_timing_review.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
