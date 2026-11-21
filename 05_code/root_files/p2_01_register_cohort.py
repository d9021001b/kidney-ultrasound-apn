"""Paper 2 cohort: contour-ready images of the 383 lock, re-keyed to the (2) workbook.

Patient roles are inherited unchanged from the earlier lock so that the never-scored
exploratory-test role stays untouched. DMSA side codes and Probe entries come from the
(2) workbook; patients removed from that workbook are dropped. Only aliases are exported.
"""

import json

import pandas as pd
from openpyxl import load_workbook

from p2_common import CACHE, P2, REGISTRY, STATE_SIDES, WORKBOOK_V2
from prepare_nckuh_383_contour_cohort_20261111 import digest


def main():
    out = P2 / "cohort"
    if out.exists():
        raise FileExistsError(out)
    sheet = load_workbook(WORKBOOK_V2, read_only=True, data_only=True)["完整資料彙整"]
    rows = sheet.iter_rows(values_only=True)
    header = list(next(rows))
    i_id, i_age, i_dmsa, i_probe = header.index("id"), header.index("age"), header.index("DMSA_APN"), header.index("Probe")
    book = pd.DataFrame([(str(r[i_id]).strip().lstrip("0"), r[i_age], str(r[i_dmsa]).strip().upper(),
                          str(r[i_probe]).strip().lower()) for r in rows if r[i_id] is not None],
                        columns=["key", "age", "DMSA_APN", "workbook_probe"])
    assert len(book) == 388 and book.key.is_unique and set(book.DMSA_APN) == set("NLRB")

    link = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype=str)
    link["key"] = link.patient_id.str.lstrip("0")
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv").rename(columns={"DMSA_APN": "DMSA_APN_v1"})
    old_images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    qc = pd.read_csv(CACHE / "extraction_qc.csv")[["image_id", "height", "width", "roi_height", "roi_width", "mask_pixels"]]

    patients = roles.merge(link[["patient_alias", "key"]], on="patient_alias", validate="one_to_one")
    patients = patients.merge(book, on="key", how="left", validate="one_to_one")
    removed = patients.DMSA_APN.isna()
    contour_ready = patients.patient_alias.isin(old_images.patient_alias)
    summary = {"lock_patients": len(patients), "removed_in_v2": int(removed.sum()),
               "contour_ready_in_lock": int(contour_ready.sum()),
               "label_changed_v1_to_v2": int((~removed & patients.DMSA_APN.ne(patients.DMSA_APN_v1)).sum())}
    patients = patients[~removed & contour_ready].drop(columns=["key"]).reset_index(drop=True)
    patients["age"] = pd.to_numeric(patients.age, errors="coerce")

    images = old_images[old_images.patient_alias.isin(patients.patient_alias)].drop(columns=["kidney_APN"])
    images = images.merge(qc, on="image_id", validate="one_to_one")
    images["cache_row"] = images.image_id.map({k: i for i, k in enumerate(old_images.image_id)})
    side_index = {"left": 0, "right": 1}
    state = patients.set_index("patient_alias").DMSA_APN
    images["kidney_APN"] = [STATE_SIDES[state[p]][side_index[s]] for p, s in zip(images.patient_alias, images.side)]
    kidneys = images[["kidney_alias", "patient_alias", "side", "kidney_APN"]].drop_duplicates().reset_index(drop=True)
    assert kidneys.kidney_alias.is_unique

    out.mkdir(parents=True)
    patients.to_csv(out / "patients.csv", index=False)
    kidneys.to_csv(out / "kidneys.csv", index=False)
    images.to_csv(out / "images.csv", index=False)
    summary |= {"patients": len(patients), "kidneys": len(kidneys), "images": len(images),
                "patients_by_role": patients.role.value_counts().to_dict(),
                "patients_by_workbook_probe": patients.workbook_probe.value_counts().to_dict(),
                "patients_by_role_and_probe": {f"{r}|{p}": int(n) for (r, p), n in
                                               patients.groupby(["role", "workbook_probe"]).size().items()},
                "kidney_positive_fraction": float(kidneys.kidney_APN.mean()),
                "workbook_v2_sha256": digest(WORKBOOK_V2),
                "source_lock_sha256": digest(REGISTRY / "cohort_lock.json"),
                "note": "Seven patients added in the (2) workbook and 36 linear-only patients have no accepted "
                        "kidney contour and are not in this registered cohort. Exploratory-test role is never "
                        "scored for APN during development."}
    (out / "cohort_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
