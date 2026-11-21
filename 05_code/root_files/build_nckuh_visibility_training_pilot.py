"""Package a per-view blinded pilot drawn only from common training patients."""
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "02_experiments"
SOURCE = ROOT / "NCKUH_fullcohort_visibility_annotation_20261103"
BASE = ROOT / "NCKUH_DMSA_fullcohort_fullframe_20261103"
OUT = ROOT / "NCKUH_visibility_training_pilot_20261103"
SEED = 20261103


def choose_cases(roles, key, count=30):
    assignments = roles.pivot(index="patient_alias", columns="seed", values="role")
    if assignments.shape[1] != 3 or assignments.isna().any().any():
        raise ValueError("Expected three complete split assignments")
    common = sorted(assignments.index[assignments.eq("train").all(axis=1)])
    patients = np.random.default_rng(SEED).choice(common, count, replace=False)
    chosen = key[key.patient_alias.isin(patients)].sort_values("case_id")
    if len(chosen) != count or not chosen.case_id.is_unique or not chosen.patient_alias.is_unique:
        raise ValueError("Incorrect private case linkage")
    return chosen, len(common)


def build():
    if OUT.exists():
        raise FileExistsError(OUT)
    roles = pd.read_csv(BASE / "patient_assignments.csv", usecols=["seed", "patient_alias", "role"])
    key = pd.read_csv(SOURCE / "private_key/case_to_DMSA_key_PRIVATE.csv", usecols=["case_id", "patient_alias"])
    chosen, eligible = choose_cases(roles, key)
    blind = SOURCE / "blinded_packet"
    slots = pd.read_csv(blind / "page_manifest.csv")
    slots = slots[slots.case_id.isin(chosen.case_id)].sort_values(["case_id", "side", "view_type"])
    assert slots.case_id.nunique() == len(chosen)
    assert not slots.duplicated(["case_id", "image_slot"]).any()
    images = pd.read_csv(BASE / "deidentified_source_manifest.csv")
    images["view_type"] = images.view.str[-1]
    first_views = images.drop_duplicates(["patient_alias", "side", "view_type"])
    private = slots.merge(chosen, on="case_id", validate="many_to_one").merge(
        first_views[["patient_alias", "side", "view_type", "image_id"]],
        on=["patient_alias", "side", "view_type"], validate="one_to_one", how="left")
    assert private.image_id.notna().all() and private.image_id.is_unique
    hashes = json.loads((SOURCE / "packet_audit.json").read_text(encoding="utf-8"))["public_sha256"]
    pages = sorted(slots.page_file.unique())
    for page in pages:
        assert hashlib.sha256((blind / page).read_bytes()).hexdigest() == hashes[page]

    public = OUT / "blinded_packet"
    public.mkdir(parents=True)
    for page in pages:
        shutil.copyfile(blind / page, public / page)
    form = slots.assign(readability="", visible_finding="", finding_type="",
                        region_on_display="", confidence_1_to_5="", notes="")
    form.to_csv(public / "view_review.csv", index=False, encoding="utf-8-sig")
    shutil.copyfile(Path(__file__).with_name("nckuh_visibility_training_pilot_guide.md"),
                    public / "review_guide_zh_TW.md")
    archive = Path(shutil.make_archive(str(OUT / "NCKUH_visibility_30_cases_BLINDED"), "zip", public))
    with ZipFile(archive) as opened:
        assert opened.testzip() is None
        assert set(opened.namelist()) == {p.name for p in public.iterdir()}
    private_dir = OUT / "private_key"
    private_dir.mkdir()
    private.to_csv(private_dir / "view_to_image_PRIVATE.csv", index=False)
    (OUT / "packet_audit.json").write_text(json.dumps({
        "common_training_patients": eligible, "selected_patients": len(chosen),
        "review_views": len(form), "sampling_seed": SEED,
        "sampled_using_dmsa_or_model_scores": False,
        "all_selected_patients_train_in_every_draw": True,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    print(f"{len(chosen)} patients, {len(form)} available views, ZIP verified")


if __name__ == "__main__":
    build()
