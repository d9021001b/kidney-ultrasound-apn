"""Prospective technical confirmation of model-predicted CEC after three-class failure."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from nckuh_internal_regions_20261102 import ROOT, BASE, FEATURES, K, locate_outer, full_frame_canvas, invert_full_frame

OUT = ROOT / "02_experiments/NCKUH_cec_reference_20261102"
PRIOR = ROOT / "02_experiments/NCKUH_internal_regions_20261102"
VARIANTS = ["annotator_1", "annotator_2"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def largest_component(mask):
    mask = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return np.zeros_like(mask)
    selected = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == selected).astype(np.uint8)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    previous = pd.read_csv(PRIOR / "selected_images_private.csv", dtype=str)
    pool = pd.read_csv(BASE / "image_manifest_private.csv", usecols=["image_id", "filename", "patient_alias", "view"], dtype=str)
    used = set(previous.patient_alias)
    rng = np.random.default_rng(20261103)
    rows = []
    for view in ["LL", "LT", "RL", "RT"]:
        available = pool[pool["view"].eq(view) & ~pool.patient_alias.isin(used)]
        available = available.iloc[rng.permutation(len(available))]
        count = 0
        for row in available.itertuples(index=False):
            if row.patient_alias in used:
                continue
            rows.append(dict(zip(pool.columns, row)))
            used.add(row.patient_alias); count += 1
            if count == 8:
                break
        assert count == 8
    validation = pd.DataFrame(rows)
    assert validation.patient_alias.nunique() == 32 and not set(validation.patient_alias).intersection(previous.patient_alias)
    previous["technical_role"] = "initial_probe"
    validation["technical_role"] = "new_confirmation"
    selected = pd.concat([previous, validation], ignore_index=True)
    selected.to_csv(OUT / "selected_images_private.csv", index=False)
    checkpoint = {variant: K.validate_model_family(variant)[0] for variant in VARIANTS}
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), diagnostic_labels_read=False,
                    rationale="Initial three-class NCKUH transfer failed. CEC alone had median intermodel Dice .799 on full frames; medulla and cortex were unreliable. This new class-specific hypothesis is checked on 32 additional patients, not declared a pass of the original three-class gate",
                    source="https://github.com/rsingla92/kidneyUS", class_id=1, class_name="model_predicted_CEC",
                    source_checkpoint_index=0, checkpoints=checkpoint,
                    source_weight_interpretation="Two independently annotated source models, no target training. A single released checkpoint is a computational approximation to the original five-checkpoint ensemble, not a validation fold on NCKUH",
                    roles=dict(initial_probe=32, new_confirmation=32), postprocessing="Inside unchanged kidney outline, retain largest connected predicted CEC component; never a brightness threshold",
                    prospective_technical_gates=dict(new_each_model_nonempty_rate_at_least=.85,
                        new_each_model_area_fraction_02_to_50_rate_at_least=.8, new_both_models_nonempty_rate_at_least=.8,
                        new_median_intermodel_CEC_dice_when_either_nonempty_at_least=.7,
                        initial_each_single_checkpoint_vs_ensemble_median_CEC_dice_at_least=.85),
                    interpretation="Technical reproducibility only, not independently adjudicated CEC anatomy. No cortex or medulla claim. APN classification prohibited until these gates and visual review are complete",
                    selection_sha256=K.sha256(OUT / "selected_images_private.csv"), helper_source_sha256=K.sha256(Path(__file__).with_name("nckuh_internal_regions_20261102.py")))
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve existing protocol")
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", dict(source_sha256=K.sha256(Path(__file__)), protocol_sha256=K.sha256(OUT / "protocol.json")))
    print("CEC-only follow-up fixed; 32 new patients, no APN label columns", flush=True)


def require_lock():
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert K.sha256(Path(__file__)) == lock["source_sha256"] and K.sha256(OUT / "protocol.json") == lock["protocol_sha256"]
    return json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))


def predict(variant):
    protocol = require_lock()
    from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
    torch.set_num_threads(4); cv2.setNumThreads(1)
    selected = pd.read_csv(OUT / "selected_images_private.csv", dtype=str)
    assert K.sha256(OUT / "selected_images_private.csv") == protocol["selection_sha256"]
    record = protocol["checkpoints"][variant]
    assert K.sha256(Path(record["path"])) == record["sha256"]
    run_out = OUT / variant
    if (run_out / "technical_metrics.csv").exists():
        raise FileExistsError("Preserve completed technical outputs")
    trainer = K.build_trainer(variant, run_out)
    trainer.load_checkpoint_ram(K.load_checkpoint(Path(record["path"])), train=False)
    trainer.network.eval()
    metrics = []
    for i, row in enumerate(selected.itertuples(index=False)):
        gray, outer, location = locate_outer(row)
        canvas, transform = full_frame_canvas(gray, outer)
        path = run_out / "inputs_nifti" / f"{row.image_id}_0000.nii.gz"
        K.array_to_nifti(canvas, path)
        data, _, properties = trainer.preprocess_patient([str(path)])
        with torch.inference_mode():
            softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(data, do_mirroring=False, mirror_axes=(),
                use_sliding_window=True, step_size=.5, use_gaussian=True, all_in_gpu=False, verbose=False, mixed_precision=True)[1]
        softmax = softmax.transpose([0]+[axis+1 for axis in trainer.transpose_backward])
        exported = run_out / "output_nifti" / f"{row.image_id}.nii.gz"
        exported.parent.mkdir(parents=True, exist_ok=True)
        export = trainer.plans.get("segmentation_export_params", {})
        save_segmentation_nifti_from_softmax(softmax, str(exported), properties, export.get("interpolation_order", 1),
            trainer.regions_class_order, None, None, None, None, force_separate_z=export.get("force_separate_z"), interpolation_order_z=export.get("interpolation_order_z", 0))
        labels = sitk.GetArrayFromImage(sitk.ReadImage(str(exported)))
        assert labels.shape == (1, 768, 1024)
        restored = invert_full_frame(labels[0], transform)
        x, y, w, h = [location[k] for k in ["x", "y", "width", "height"]]
        cec = largest_component((restored == 1) & (outer > 0))
        native = cec[y:y+h, x:x+w]
        folder = run_out / "cec_masks"; folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{row.image_id}.png"), native*255)
        K.write_overlay(gray[y:y+h, x:x+w], outer[y:y+h, x:x+w], native,
                        run_out / "overlays" / f"{row.image_id}.png", f"{row.image_id} {variant} predicted CEC only")
        pixels = int(cec.sum())
        result = dict(image_id=row.image_id, view=row.view, technical_role=row.technical_role, cec_pixels=pixels,
                      outer_pixels=int(outer.sum()), area_fraction=float(pixels/outer.sum()))
        if row.technical_role == "initial_probe":
            ensemble = cv2.imread(str(PRIOR / variant / "region_masks/full_frame" / f"{row.image_id}.png"), 0)
            result["dice_vs_five_checkpoint_CEC"] = K.dice(native > 0, largest_component(ensemble == 1) > 0)
        else:
            result["dice_vs_five_checkpoint_CEC"] = None
        metrics.append(result)
        if (i+1) % 16 == 0:
            print(f"CEC {variant} {i+1}/{len(selected)} source-checkpoint0", flush=True)
    pd.DataFrame(metrics).to_csv(run_out / "technical_metrics.csv", index=False)


def summarize():
    require_lock()
    rows = []
    model_results = []
    for variant in VARIANTS:
        data = pd.read_csv(OUT / variant / "technical_metrics.csv")
        new = data[data.technical_role.eq("new_confirmation")]
        original = data[data.technical_role.eq("initial_probe")]
        assert len(new) == len(original) == 32
        model_results.append(dict(variant=variant, new_nonempty_rate=float(new.cec_pixels.ge(64).mean()),
                                  new_area_fraction_in_range_rate=float(new.area_fraction.between(.02, .5).mean()),
                                  new_median_area_fraction=float(new.area_fraction.median()),
                                  median_single_vs_ensemble_dice=float(original.dice_vs_five_checkpoint_CEC.median())))
    selected = pd.read_csv(OUT / "selected_images_private.csv", dtype=str)
    for row in selected.itertuples(index=False):
        masks = [cv2.imread(str(OUT / v / "cec_masks" / f"{row.image_id}.png"), 0) > 0 for v in VARIANTS]
        rows.append(dict(image_id=row.image_id, technical_role=row.technical_role,
                         either_nonempty=bool(masks[0].any() or masks[1].any()),
                         both_nonempty=bool(masks[0].sum() >= 64 and masks[1].sum() >= 64),
                         dice=K.dice(*masks)))
    agreement = pd.DataFrame(rows)
    agreement.to_csv(OUT / "agreement.csv", index=False)
    new = agreement[agreement.technical_role.eq("new_confirmation")]
    dice = float(new.loc[new.either_nonempty, "dice"].median()) if new.either_nonempty.any() else 0.
    both = float(new.both_nonempty.mean())
    passed = bool(all(m["new_nonempty_rate"] >= .85 and m["new_area_fraction_in_range_rate"] >= .8 and m["median_single_vs_ensemble_dice"] >= .85 for m in model_results) and both >= .8 and dice >= .7)
    result = dict(diagnostic_labels_read=False, models=model_results, new_median_intermodel_dice=dice,
                  new_both_nonempty_rate=both, technical_gates_passed=passed, anatomical_truth_verified=False,
                  classification_performed=False, next_action="Visual review of source-predicted CEC reference before separately fixed classification experiment" if passed else "Do not promote this computational CEC reference")
    write_json(OUT / "decision.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "predict", "summarize"])
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    if args.stage == "predict":
        if args.variant is None:
            parser.error("predict requires --variant")
        predict(args.variant)
    else:
        globals()[args.stage]()
