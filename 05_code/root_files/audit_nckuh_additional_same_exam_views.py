"""Find dated panel views absent from the current fixed kidney cohort."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import nckuh_dmsa_fullcohort_fullframe_20261103 as base

OUT = base.ROOT / "02_experiments/NCKUH_additional_same_exam_views_20261103"


def fingerprint(image):
    digest = hashlib.sha256(str(image.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    images = base.labeled_images()
    exact, _ = base.panel_index()
    existing = {(r.normalized_patient_id, r.exam_date_raw, r.view): r for r in images.itertuples()}
    episodes = {}
    for r in existing.values():
        if pd.isna(r.exam_date_iso):
            continue
        episode = (r.normalized_patient_id, r.exam_date_raw)
        episodes.setdefault(episode, []).append(r)
    observed = set(images.kidney_alias)
    records, private = [], []
    for patient, date, view in sorted(exact.keys() - existing.keys()):
        if (patient, date) not in episodes:
            continue
        rows = episodes[(patient, date)]
        assert len({(r.patient_alias, r.DMSA_APN) for r in rows}) == 1
        alias = rows[0].patient_alias
        side = view[0]
        kidney = f"{alias}_{side}"
        label = int(rows[0].DMSA_APN in (side, "B"))
        seen = set()
        for r in rows:
            paths = exact.get((patient, date, r.view), [])
            if paths:
                try:
                    native, _ = base.split_panel(paths[0])
                    seen.add(fingerprint(native))
                except (ValueError, OSError):
                    pass
        for index, path in enumerate(exact[(patient, date, view)], 1):
            key = f"C{len(records) + 1:03d}"
            readable, duplicate, pixels, area_fraction = False, None, None, None
            try:
                image, mask = base.split_panel(path)
                readable, pixels = True, int(mask.sum())
                area_fraction = pixels / mask.size
                duplicate = fingerprint(image) in seen
            except (ValueError, OSError):
                pass
            records.append(dict(candidate=key, patient_alias=alias, kidney_alias=kidney, view=view,
                                ranked_source=index, kidney_already_imaged=kidney in observed,
                                panel_readable=readable, exact_same_pixels_as_existing_view=duplicate,
                                mask_pixels=pixels, mask_fraction=area_fraction))
            private.append(dict(candidate=key, source_path=str(path), exam_date=date,
                                workbook_state=rows[0].DMSA_APN, provisional_kidney_label=label))
    audit = pd.DataFrame(records)
    valid = audit[audit.panel_readable & audit.exact_same_pixels_as_existing_view.eq(False)]
    audit_summary = dict(candidate_panel_rows=len(audit), candidate_view_keys=len(audit.drop_duplicates(
        ["patient_alias", "view"])), readable_nonidentical_panel_rows=len(valid),
        patients_with_readable_nonidentical_candidates=valid.patient_alias.nunique(),
        new_view_keys_with_readable_nonidentical_candidates=len(valid.drop_duplicates(["patient_alias", "view"])),
        not_previously_imaged_kidneys=valid.loc[~valid.kidney_already_imaged, "kidney_alias"].nunique(),
        scope="Dated test_predict and NCKUH3 panels, matched by exact patient and valid existing exam date",
        exact_duplicate_scope="Decoded left frame versus existing preferred dated panels in same exam only",
        reviewed_or_added=0)
    OUT.mkdir()
    audit.to_csv(OUT / "candidate_qc_internal.csv", index=False)
    pd.DataFrame(private).to_csv(OUT / "source_linkage_PRIVATE.csv", index=False)
    (OUT / "audit.json").write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
    print(json.dumps(audit_summary, indent=2))


if __name__ == "__main__":
    run()
