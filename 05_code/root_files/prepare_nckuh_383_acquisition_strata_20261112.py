"""Define label-blind frame-size datasets on the locked revised NCKUH roles."""

import json
from pathlib import Path

import pandas as pd

from prepare_nckuh_383_contour_cohort_20261111 import OUT as REGISTRY, digest


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
CACHE = EXP / "NCKUH_revised_383_contour_cache_20261111"
OUT = EXP / "NCKUH_revised_383_acquisition_strata_20261112"
GROUPS = {"720x960": "frame_720x960", "450x560": "frame_450x560"}


def main():
    lock = json.loads((REGISTRY / "cohort_lock.json").read_text(encoding="utf-8"))
    for name in ("contour_image_manifest.csv", "contour_kidney_manifest.csv", "patient_assignments.csv"):
        assert digest(REGISTRY / name) == lock["files_sha256"][name]
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    assignments = pd.read_csv(REGISTRY / "patient_assignments.csv")
    qc = pd.read_csv(CACHE / "extraction_qc.csv")
    assert images.image_id.tolist() == qc.image_id.tolist()
    qc["frame"] = qc.height.astype(str) + "x" + qc.width.astype(str)
    unique = qc.groupby("patient_alias").frame.nunique()
    mode = qc.groupby("patient_alias").frame.agg(lambda values: values.mode().iloc[0])
    cohort = assignments[assignments.patient_alias.isin(images.patient_alias)].copy()
    cohort["frame"] = cohort.patient_alias.map(mode)
    cohort["frame_count"] = cohort.patient_alias.map(unique)
    cohort["dataset"] = cohort.frame.map(GROUPS).fillna("other_or_mixed")
    cohort.loc[cohort.frame_count.ne(1), "dataset"] = "other_or_mixed"
    assert len(cohort) == 347 and cohort.patient_alias.is_unique
    assert set(cohort.dataset) == {"frame_720x960", "frame_450x560", "other_or_mixed"}

    output = {"cohort_lock_sha256": digest(REGISTRY / "cohort_lock.json"),
              "source_qc_sha256": digest(CACHE / "extraction_qc.csv"),
              "rule": "Patient assigned only if all available contour-image frames have one exact size: 720x960 or 450x560; all others are held apart.",
              "probe_is_not_equivalent_to_frame": True,
              "test_scored": False,
              "datasets": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    probe = pd.read_csv(
        EXP / "NCKUH_revised_383_probe_geometry_20261112/registered_image_probe_labels.csv"
    )
    image_dataset = images.merge(cohort[["patient_alias", "dataset"]], on="patient_alias", validate="many_to_one")
    joined_probe = image_dataset[["image_id", "dataset"]].merge(probe, on="image_id", how="left")
    for name in ("frame_720x960", "frame_450x560", "other_or_mixed"):
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        people = cohort[cohort.dataset.eq(name)].copy()
        frame_images = image_dataset[image_dataset.dataset.eq(name)].copy()
        frame_kidneys = kidneys[kidneys.patient_alias.isin(people.patient_alias)].copy()
        people.to_csv(folder / "patient_manifest.csv", index=False)
        frame_images.to_csv(folder / "image_manifest.csv", index=False)
        frame_kidneys.to_csv(folder / "kidney_manifest.csv", index=False)
        known = joined_probe[joined_probe.dataset.eq(name)]
        output["datasets"][name] = {
            "patients": len(people), "kidneys": len(frame_kidneys), "images": len(frame_images),
            "roles": people.role.value_counts().to_dict(),
            "DMSA_states": people.DMSA_APN.value_counts().to_dict(),
            "known_probe_images": known.probe.value_counts(dropna=False).to_dict(),
            "patient_manifest_sha256": digest(folder / "patient_manifest.csv"),
            "image_manifest_sha256": digest(folder / "image_manifest.csv"),
            "kidney_manifest_sha256": digest(folder / "kidney_manifest.csv"),
        }
    (OUT / "dataset_protocol.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
