"""Cached frozen image representations for the paper-2 cohort, aligned to cohort/images.csv."""

import numpy as np
import pandas as pd

from p2_common import CACHE, EXP, P2, load_cohort

LEGACY = EXP / "NCKUH_revised_383_legacy_features_20261112"
DINO = EXP / "NCKUH_revised_383_dinov3_20261111"


def routed_images():
    patients, kidneys, images = load_cohort()
    routes = pd.read_csv(P2 / "router/image_routes.csv")[["image_id", "route"]]
    images = images.merge(routes, on="image_id", validate="one_to_one")
    images = images.merge(patients[["patient_alias", "role", "DMSA_APN", "age", "workbook_probe"]], on="patient_alias", validate="many_to_one")
    # The workbook Probe entry is the ground truth of a single-probe patient and is not routed; only patients with both probes use the detector.
    single = images.workbook_probe.isin(("convex", "linear"))
    images["detector_route"] = images.route
    images.loc[single, "route"] = images.loc[single, "workbook_probe"]
    return patients, kidneys, images


def image_sources(images):
    rows = images.cache_row.to_numpy()
    old_ids = pd.read_csv(EXP / "NCKUH_revised_383_audit_20261111/contour_image_manifest.csv").image_id
    old = rows < len(old_ids)  # rows 1362 onward are the patients added on 2026-11-17: image I(row+1)
    assert (old_ids.iloc[rows[old]].to_numpy() == images.image_id.to_numpy()[old]).all()
    assert all(f"I{r + 1:04d}" == i for r, i in zip(rows[~old], images.image_id.to_numpy()[~old]))
    raw = pd.read_csv(CACHE / "radiomics.csv").set_index("image_id").reindex(images.image_id)
    radiomics = raw[[c for c in raw if not c.endswith(("_Energy", "_TotalEnergy"))]].to_numpy(float)
    radiomics[~np.isfinite(radiomics)] = np.nan
    load = lambda path: np.asarray(np.load(path, mmap_mode="r")[rows], dtype=np.float64)
    uri = np.load(CACHE / "uricads_fold_embeddings.npy", mmap_mode="r")
    return {
        "rad_color": load(LEGACY / "rad_color.npy"),
        "rad_tile": load(LEGACY / "tile_mean.npy"),
        "rad_dense": load(CACHE / "radimagenet_dense_tight.npy"),
        "usfm": load(LEGACY / "usfm_gap.npy"),
        "dinov3_class": load(DINO / "dinov3_class.npy"),
        "dinov3_patch": load(DINO / "dinov3_patch.npy"),
        "uricads": np.asarray(uri[rows], dtype=np.float64).mean(1),
        "uricads_pathology": np.asarray(np.load(CACHE / "uricads_fold_pathology_scores.npy")[rows], float).mean(1),
        "radiomics": radiomics,
        "focal_echo": load(EXP / "NCKUH_revised_383_focal_echo_20261111/focal_echo.npy"),
        "geometry": load(EXP / "NCKUH_revised_383_relative_geometry_20261112/relative_geometry_image.npy"),
    }
