"""Audit NCKUH 383-to-388 workbook changes without exporting patient IDs."""

from collections import Counter
import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from prepare_nckuh_383_contour_cohort_20261111 import ROOT, digest


OLD = ROOT / "2004-2019-NCKUH-亞大(1).xlsx"
NEW = ROOT / "2004-2019-NCKUH-亞大(2).xlsx"
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
OUT = ROOT / "02_experiments/NCKUH_revised_388_audit_20261112"


def records(path):
    sheet = load_workbook(path, read_only=True, data_only=True)["完整資料彙整"]
    rows = sheet.iter_rows(values_only=True)
    columns = list(next(rows))
    assert columns[0] == "id" and columns[5] == "DMSA_APN" and columns[7] == "Probe"
    items = {}
    for row in rows:
        if row[0] is None:
            continue
        key = str(row[0]).strip().lstrip("0")
        assert key and key not in items
        items[key] = row
    return items


def main():
    target = OUT / "revision_audit.json"
    if target.exists():
        raise FileExistsError("Preserve previous workbook revision audit")
    old = records(OLD)
    new = records(NEW)
    assert len(old) == 383 and len(new) == 388
    common = set(old) & set(new)
    removed = set(old) - set(new)
    added = set(new) - set(old)
    assert (len(common), len(removed), len(added)) == (381, 2, 7)
    old_probe = Counter(str(row[7]).strip().lower() for row in old.values())
    new_probe = Counter(str(row[7]).strip().lower() for row in new.values())
    assert new_probe == {"convex": 245, "linear": 133, "both": 10}
    for row in new.values():
        assert str(row[5]).strip().upper() in {"R", "L", "B", "N"}
    changed = {index: sum(str(old[key][index]).strip() != str(new[key][index]).strip()
                          for key in common) for index in range(8)}
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv", dtype=str)
    linkage = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype=str)
    linkage["normalized_id"] = linkage.patient_id.str.lstrip("0")
    assert len(linkage) == 383 and set(linkage.normalized_id) == set(old)
    role_by_id = dict(zip(linkage.normalized_id, linkage.patient_alias.map(
        roles.set_index("patient_alias").role)))
    assert set(role_by_id) == set(old)
    changed_label_ids = {key for key in common if str(old[key][5]).strip() != str(new[key][5]).strip()}
    changed_probe_ids = {key for key in common if str(old[key][7]).strip() != str(new[key][7]).strip()}
    report = {
        "old_workbook_sha256": digest(OLD), "new_workbook_sha256": digest(NEW),
        "old_patients": len(old), "new_patients": len(new), "retained": len(common),
        "removed": len(removed), "added": len(added),
        "probe_exclusive_old": dict(old_probe), "probe_exclusive_new": dict(new_probe),
        "probe_including_both_new": {"convex": new_probe["convex"] + new_probe["both"],
                                     "linear": new_probe["linear"] + new_probe["both"]},
        "dmsa_new_counts": dict(Counter(str(row[5]).strip().upper() for row in new.values())),
        "retained_changed_columns_0_to_7": changed,
        "removed_by_locked_role": dict(Counter(role_by_id[key] for key in removed)),
        "retained_dmsa_label_changes_by_locked_role": dict(Counter(role_by_id[key] for key in changed_label_ids)),
        "retained_probe_changes_by_locked_role": dict(Counter(role_by_id[key] for key in changed_probe_ids)),
        "added_not_assigned_to_old_roles": len(added),
        "caveat": "Workbook revision cannot by itself prove ultrasound/DMSA temporal concordance; the source dates and exclusion rule require a separate patient-level QC. Existing 383-person role lock and historical results are not rewritten. No patient IDs exported.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
