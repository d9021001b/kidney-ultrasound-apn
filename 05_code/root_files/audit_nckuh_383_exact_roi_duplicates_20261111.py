"""Audit exact ROI duplicates across revised-cohort roles without scoring test labels."""

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
CACHE = ROOT / "02_experiments/NCKUH_revised_383_contour_cache_20261111"


def main():
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv")
    rows = images.merge(roles[["patient_alias", "role"]], on="patient_alias")
    grouped = defaultdict(list)
    for row in rows.itertuples(index=False):
        path = CACHE / "percentile" / f"{row.image_id}.png"
        grouped[sha256(path.read_bytes()).hexdigest()].append(row)
    groups = [items for items in grouped.values() if len(items) > 1]
    report = []
    for items in groups:
        patients = {item.patient_alias for item in items}
        sides = {item.side for item in items}
        roles_seen = {item.role for item in items}
        train_items = [item for item in items if item.role == "train"]
        train_discordant = len({item.kidney_APN for item in train_items}) > 1
        report.append({"image_ids": ";".join(item.image_id for item in items),
                       "copies": len(items), "patients": len(patients),
                       "opposite_sides": len(sides) > 1,
                       "cross_role": len(roles_seen) > 1,
                       "train_label_discordance": train_discordant})
    pd.DataFrame(report, columns=["image_ids", "copies", "patients", "opposite_sides",
                                  "cross_role", "train_label_discordance"]).to_csv(
                                      CACHE / "exact_roi_duplicate_groups.csv", index=False)
    summary = {"images": len(rows), "duplicate_groups": len(groups),
               "images_in_duplicate_groups": sum(len(items) for items in groups),
               "cross_patient_groups": sum(item["patients"] > 1 for item in report),
               "cross_role_groups": sum(item["cross_role"] for item in report),
               "opposite_side_groups": sum(item["opposite_sides"] for item in report),
               "train_label_discordant_groups": sum(item["train_label_discordance"] for item in report),
               "scope": "Exact bytes of deidentified percentile ROI PNG. Test labels were not inspected.",
               "interpretation": "Duplicate groups require source review; no automatic relabeling or deletion."}
    (CACHE / "exact_roi_duplicate_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
